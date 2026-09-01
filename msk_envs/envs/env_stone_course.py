from dataclasses import dataclass

import bolt
import torch
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, MIN_ROOT_HEIGHT, SIDE_IDX, UP_IDX, build_axis
from msk_envs.utils.reward_lib import velocity_reward_max
from .env_config import EnvConfig
from .env_lanes import LanesEnv


@dataclass(frozen=True)
class StoneCourseSpec:
    """Geometry and randomization rules for a batch of slab courses."""

    num_stones: int
    step_length_range: tuple[float, float]
    lateral_jitter: float
    slab_size: tuple[float, float, float]
    top_height: float
    lookahead: int

    # The first pair is deliberately easy so the initial pose is supported.
    launch_step_length_range: tuple[float, float] = (0.32, 0.38)
    alternating_lateral_offset: float = 0.12
    launch_jitter_scale: float = 0.2
    passed_margin: float = 0.10

    def __post_init__(self) -> None:
        if self.num_stones < self.lookahead + 1:
            raise ValueError("course_stones must provide one spare slab beyond course_lookahead")
        self._validate_range("course_step_length_range", self.step_length_range)
        self._validate_range("launch_step_length_range", self.launch_step_length_range)
        if self.lateral_jitter < 0.0:
            raise ValueError("course_lateral_jitter must be non-negative")
        if len(self.slab_size) != 3 or any(size <= 0.0 for size in self.slab_size):
            raise ValueError("course_slab_size must contain three positive dimensions")
        if self.top_height <= self.slab_size[UP_IDX] * 0.5:
            raise ValueError("course_top_height must leave the slab above the ground plane")
        if self.lookahead < 1:
            raise ValueError("course_lookahead must be at least 1")

    @staticmethod
    def _validate_range(name: str, values: tuple[float, float]) -> None:
        if len(values) != 2 or values[0] <= 0.0 or values[0] > values[1]:
            raise ValueError(f"{name} must be a positive (min, max) pair")

    @classmethod
    def from_env_config(cls, config: EnvConfig) -> "StoneCourseSpec":
        return cls(
            num_stones=config.course_stones,
            step_length_range=config.course_step_length_range,
            lateral_jitter=config.course_lateral_jitter,
            slab_size=config.course_slab_size,
            top_height=config.course_top_height,
            lookahead=config.course_lookahead,
        )

    @property
    def half_extents(self) -> tuple[float, float, float]:
        return tuple(size * 0.5 for size in self.slab_size)

    @property
    def center_height(self) -> float:
        return self.top_height - self.half_extents[UP_IDX]

    def default_positions(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Deterministic layout used while Bolt constructs shared geometry."""
        midpoint = sum(self.step_length_range) * 0.5
        step_lengths = torch.full((1, self.num_stones), midpoint, device=device)
        launch_count = min(2, self.num_stones)
        step_lengths[:, :launch_count] = sum(self.launch_step_length_range) * 0.5
        lateral_jitter = torch.zeros_like(step_lengths)
        return self._positions_from_steps(step_lengths, lateral_jitter).squeeze(0)

    def sample_positions(
        self,
        num_courses: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        step_length_max: float | None = None,
    ) -> torch.Tensor:
        """Sample one independent five-slab buffer per course/world."""
        if num_courses < 0:
            raise ValueError("num_courses must be non-negative")

        lo, final_hi = self.step_length_range
        hi = final_hi if step_length_max is None else step_length_max
        if hi < lo or hi > final_hi:
            raise ValueError("step_length_max must stay inside course_step_length_range")
        step_lengths = torch.rand(
            (num_courses, self.num_stones), device=device, generator=generator
        ) * (hi - lo) + lo

        launch_count = min(2, self.num_stones)
        launch_lo, launch_hi = self.launch_step_length_range
        step_lengths[:, :launch_count] = torch.rand(
            (num_courses, launch_count), device=device, generator=generator
        ) * (launch_hi - launch_lo) + launch_lo

        lateral_jitter = (
            torch.rand(
                (num_courses, self.num_stones), device=device, generator=generator
            ) * 2.0 - 1.0
        ) * self.lateral_jitter
        lateral_jitter[:, :launch_count] *= self.launch_jitter_scale
        return self._positions_from_steps(step_lengths, lateral_jitter)

    def _positions_from_steps(
        self,
        step_lengths: torch.Tensor,
        lateral_jitter: torch.Tensor,
    ) -> torch.Tensor:
        stone_x = torch.cumsum(step_lengths, dim=1)
        stone_ids = torch.arange(self.num_stones, device=step_lengths.device)
        alternating_side = torch.where(
            stone_ids % 2 == 0,
            self.alternating_lateral_offset,
            -self.alternating_lateral_offset,
        ).unsqueeze(0)
        stone_z = alternating_side + lateral_jitter
        stone_y = torch.full_like(stone_x, self.center_height)
        return torch.stack((stone_x, stone_y, stone_z), dim=-1)

    def root_relative_observation(
        self,
        stone_positions: torch.Tensor,
        root_positions: torch.Tensor,
        lookahead_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flatten the next N root-relative (forward, lateral) slab centers."""
        order = stone_positions[:, :, FWD_IDX].argsort(dim=1)
        ordered = torch.gather(
            stone_positions,
            1,
            order.unsqueeze(-1).expand(-1, -1, 3),
        )
        root_xz = root_positions[:, [FWD_IDX, SIDE_IDX]]
        passed = ordered[:, :, FWD_IDX] < (
            root_positions[:, FWD_IDX, None] - self.passed_margin
        )
        next_stone = passed.sum(dim=1)
        offsets = lookahead_offsets
        if offsets is None:
            offsets = torch.arange(self.lookahead, device=stone_positions.device)
        indices = (next_stone[:, None] + offsets[None, :]).clamp_max(self.num_stones - 1)
        targets = torch.gather(
            ordered,
            1,
            indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        return (targets[:, :, [FWD_IDX, SIDE_IDX]] - root_xz[:, None, :]).flatten(1)


@dataclass
class SpacingCurriculum:
    """Monotonically expand only the upper spacing bound with competence."""

    minimum: float
    maximum: float
    current_maximum: float
    increment: float
    success_threshold: float
    window: int
    episodes: int = 0
    successes: int = 0
    last_completion_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.minimum <= self.current_maximum <= self.maximum:
            raise ValueError(
                "course_initial_step_length_max must lie inside course_step_length_range"
            )
        if self.increment <= 0.0:
            raise ValueError("course_curriculum_increment must be positive")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("course_curriculum_success_threshold must be in [0, 1]")
        if self.window < 1:
            raise ValueError("course_curriculum_window must be at least 1")

    @classmethod
    def from_env_config(
        cls,
        config: EnvConfig,
        course: StoneCourseSpec,
    ) -> "SpacingCurriculum":
        return cls(
            minimum=course.step_length_range[0],
            maximum=course.step_length_range[1],
            current_maximum=config.course_initial_step_length_max,
            increment=config.course_curriculum_increment,
            success_threshold=config.course_curriculum_success_threshold,
            window=config.course_curriculum_window,
        )

    def observe(self, successful_episodes: torch.Tensor) -> bool:
        """Record completed episodes and promote once the window is competent."""
        count = successful_episodes.numel()
        if count == 0:
            return False
        self.episodes += count
        self.successes += int(successful_episodes.bool().sum().item())
        if self.episodes < self.window:
            return False

        self.last_completion_rate = self.successes / self.episodes
        promoted = (
            self.last_completion_rate >= self.success_threshold
            and self.current_maximum < self.maximum
        )
        if promoted:
            self.current_maximum = min(
                self.maximum,
                self.current_maximum + self.increment,
            )
        self.episodes = 0
        self.successes = 0
        return promoted

    def state_dict(self) -> dict:
        """Return the small, device-independent state needed for a resume."""
        return {
            "current_maximum": self.current_maximum,
            "episodes": self.episodes,
            "successes": self.successes,
            "last_completion_rate": self.last_completion_rate,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore progress while retaining bounds from the current config."""
        current_maximum = float(state["current_maximum"])
        if not self.minimum <= current_maximum <= self.maximum:
            raise ValueError("checkpoint curriculum maximum is outside the configured range")

        episodes = int(state.get("episodes", 0))
        successes = int(state.get("successes", 0))
        completion_rate = float(state.get("last_completion_rate", 0.0))
        if episodes < 0 or successes < 0 or successes > episodes:
            raise ValueError("checkpoint curriculum episode counts are invalid")
        if not 0.0 <= completion_rate <= 1.0:
            raise ValueError("checkpoint curriculum completion rate must be in [0, 1]")

        self.current_maximum = current_maximum
        self.episodes = episodes
        self.successes = successes
        self.last_completion_rate = completion_rate


class StoneCourseEnv(LanesEnv):
    """Endless randomized slab terrain backed by five recycled box colliders.

    Every world owns independent transforms for the same five collider IDs.
    Once a slab is safely behind the pelvis, that world's slab is moved beyond
    its furthest slab and receives a new random gap. The policy always observes
    the four closest upcoming slabs.

    The reward remains capped forward velocity plus staying upright. Falling,
    turning away, or catching a slab only by its edge is a terminal failure.
    The ordinary time limit is a neutral truncation; there is no final slab.
    """

    def __init__(self, num_envs, env_config, device, requires_visuals, cuda_graph):
        self.course = StoneCourseSpec.from_env_config(env_config)
        self.spacing_curriculum = SpacingCurriculum.from_env_config(env_config, self.course)
        self.require_interior_landing = env_config.course_require_interior_landing
        self.landing_check_delay = env_config.course_landing_check_delay
        self.recycle_distance_behind = env_config.course_recycle_distance_behind
        self.curriculum_min_progress = env_config.course_curriculum_min_progress
        if self.landing_check_delay < 0.0:
            raise ValueError("course_landing_check_delay must be non-negative")
        if self.recycle_distance_behind < 0.0:
            raise ValueError("course_recycle_distance_behind must be non-negative")
        if self.curriculum_min_progress < 0.0:
            raise ValueError("course_curriculum_min_progress must be non-negative")

        default_positions = self.course.default_positions()
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
            lane_width=1e9,
        )
        self.target_speed = env_config.walk_target_speed

        self.stone_ids = torch.tensor(
            [self.collider_id_lookup[f"stone_{index}"] for index in range(self.course.num_stones)],
            device=device,
            dtype=torch.long,
        )
        foot_entries = [
            (name, collider_id)
            for name, collider_id in self.collider_id_lookup.items()
            if name.startswith(("left_foot_", "right_foot_"))
        ]
        if not foot_entries:
            raise ValueError("StoneCourse requires named left_foot_/right_foot_ colliders")
        self.foot_collider_ids = torch.tensor(
            [collider_id for _, collider_id in foot_entries],
            device=device,
            dtype=torch.long,
        )
        self.foot_side_masks = (
            torch.tensor([name.startswith("left_foot_") for name, _ in foot_entries], device=device),
            torch.tensor([name.startswith("right_foot_") for name, _ in foot_entries], device=device),
        )
        self.foot_collider_radii = self.collider_sizes[self.foot_collider_ids, 0]
        self.slab_half_extents_xz = torch.tensor(
            [self.course.half_extents[FWD_IDX], self.course.half_extents[SIDE_IDX]],
            device=device,
            dtype=self.foot_collider_radii.dtype,
        )
        self.interior_half_extents_xz = (
            self.slab_half_extents_xz - self.foot_collider_radii.unsqueeze(1)
        )
        if (self.interior_half_extents_xz <= 0.0).any():
            raise ValueError("course_slab_size must exceed every foot contact diameter")

        self.lookahead_offsets = torch.arange(self.course.lookahead, device=device)
        self.stone_positions = default_positions.to(device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.next_lateral_sign = torch.full(
            (num_envs,),
            -1.0 if self.course.num_stones % 2 else 1.0,
            device=device,
        )
        self.previous_foot_contact = torch.zeros(
            (num_envs, 2), device=device, dtype=torch.bool
        )
        self._episode_started = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._episode_start_x = torch.zeros(num_envs, device=device)
        self._last_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._last_edge_violation = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.episode_slabs_recycled = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.last_progress_x = 0.0
        self.last_mean_slabs_recycled = 0.0

    # ---- course geometry and per-world recycling ---------------------------------

    def _add_colliders(self, env_config: EnvConfig) -> None:
        half_extents = self.course.half_extents
        for index, position in enumerate(self.course.default_positions().tolist()):
            slab = bolt.UserGeomData(
                name=f"stone_{index}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.BOX,
                transform=wp.transform(wp.vec3(*position), wp.quat_identity(dtype=float)),
                size=wp.vec3(*half_extents),
                priority=9,
            )
            self.load_result.colliders.append(bolt.convert_user_collider(slab))

    def _set_course_positions(
        self,
        world_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        expected_shape = (world_ids.numel(), self.course.num_stones, 3)
        if positions.shape != expected_shape:
            raise ValueError(f"positions must have shape {expected_shape}, got {positions.shape}")
        self.stone_positions[world_ids] = positions
        self.collider_local_transforms[
            world_ids[:, None], self.stone_ids[None, :], :3
        ] = positions

    def _record_finished_episodes(self, world_ids: torch.Tensor) -> None:
        finished = world_ids[self._episode_started[world_ids]]
        if finished.numel() > 0:
            self.spacing_curriculum.observe(self._last_success[finished])

    def _randomize_stones(self, reset_mask: torch.Tensor) -> None:
        """Install a new independent five-slab buffer in resetting worlds."""
        world_ids = torch.where(reset_mask.flatten().bool())[0]
        if world_ids.numel() == 0:
            return
        self._record_finished_episodes(world_ids)
        positions = self.course.sample_positions(
            world_ids.numel(),
            self.device,
            step_length_max=self.spacing_curriculum.current_maximum,
        )
        self._set_course_positions(world_ids, positions)
        self.next_lateral_sign[world_ids] = -1.0 if self.course.num_stones % 2 else 1.0
        self.previous_foot_contact[world_ids] = False
        self._last_success[world_ids] = False
        self._last_edge_violation[world_ids] = False
        self.episode_slabs_recycled[world_ids] = 0
        self._episode_started[world_ids] = True

    def _recycle_passed_stones(self) -> None:
        """Move safely passed slabs ahead, independently in every world."""
        safely_behind = self.stone_positions[:, :, FWD_IDX] < (
            self.root_pos[:, FWD_IDX, None] - self.recycle_distance_behind
        )
        inactive = self.collider_forces[:, self.stone_ids] <= 0.0
        eligible = safely_behind & inactive
        world_ids = torch.where(eligible.any(dim=1))[0]

        # At normal locomotion speeds no world can pass two 0.4 m-spaced slabs
        # in one 1/30-second policy step, so one move per world is sufficient.
        # Keeping this branch-free avoids repeated CPU/GPU synchronizations.
        masked_x = torch.where(
            eligible[world_ids],
            self.stone_positions[world_ids, :, FWD_IDX],
            torch.full_like(self.stone_positions[world_ids, :, FWD_IDX], torch.inf),
        )
        recycled_local_ids = masked_x.argmin(dim=1)
        furthest_x = self.stone_positions[world_ids, :, FWD_IDX].max(dim=1).values
        lo = self.spacing_curriculum.minimum
        hi = self.spacing_curriculum.current_maximum
        gaps = torch.rand(world_ids.numel(), device=self.device) * (hi - lo) + lo
        lateral_jitter = (
            torch.rand(world_ids.numel(), device=self.device) * 2.0 - 1.0
        ) * self.course.lateral_jitter
        new_positions = torch.stack(
            (
                furthest_x + gaps,
                torch.full_like(gaps, self.course.center_height),
                self.next_lateral_sign[world_ids]
                * self.course.alternating_lateral_offset
                + lateral_jitter,
            ),
            dim=1,
        )

        collider_ids = self.stone_ids[recycled_local_ids]
        self.stone_positions[world_ids, recycled_local_ids] = new_positions
        self.collider_local_transforms[world_ids, collider_ids, :3] = new_positions
        self.next_lateral_sign[world_ids] *= -1.0
        self.episode_slabs_recycled[world_ids] += 1

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self._randomize_stones(reset_mask)

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        pelvis_height = self.qpos_id_lookup["pelvis_ty"]
        self.joint_positions[reset_mask, pelvis_height] += self.course.top_height
        self.launch_sim_reset()
        self._episode_start_x[reset_mask] = self.root_pos[reset_mask, FWD_IDX]

    def _pre_step(self) -> None:
        self._recycle_passed_stones()

    # ---- observation, objective, and episode boundaries ---------------------------

    def _get_obs(self) -> torch.Tensor:
        course_obs = self.course.root_relative_observation(
            self.stone_positions,
            self.root_pos,
            self.lookahead_offsets,
        )
        return torch.cat((course_obs, super()._get_obs()), dim=1).detach().clone()

    def _compute_raw_reward_dict(self) -> None:
        forward_velocity = torch.nan_to_num(
            velocity_reward_max(
                self.body_velocities,
                self.root_id,
                FWD_IDX,
                linear=True,
                target_speed=self.target_speed,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.reward_dict = {
            "rew_vel": forward_velocity.detach(),
            "rew_alive": torch.ones(self.num_worlds, device=self.device),
        }

    def _interior_foot_support(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-side physical contact and whole-foot interior support."""
        foot_active = self.collider_forces[:, self.foot_collider_ids] > 0.0
        stone_active = self.collider_forces[:, self.stone_ids] > 0.0
        foot_xz = self.collider_positions[:, self.foot_collider_ids][
            :, :, [FWD_IDX, SIDE_IDX]
        ]
        stone_xz = self.stone_positions[:, :, [FWD_IDX, SIDE_IDX]]
        collider_inside = (
            (foot_xz[:, :, None, :] - stone_xz[:, None, :, :]).abs()
            <= self.interior_half_extents_xz[None, :, None, :]
        ).all(dim=3)

        contact_by_side = []
        interior_by_side = []
        for mask in self.foot_side_masks:
            side_active = foot_active[:, mask]
            side_inside = collider_inside[:, mask]
            whole_foot_inside = side_inside.all(dim=1)
            active_collider_inside = (side_inside & side_active[:, :, None]).any(dim=1)
            valid_slab = whole_foot_inside & active_collider_inside & stone_active
            contact_by_side.append(side_active.any(dim=1))
            interior_by_side.append(valid_slab.any(dim=1))
        return torch.stack(contact_by_side, dim=1), torch.stack(interior_by_side, dim=1)

    def _invalid_edge_touchdown(self) -> torch.Tensor:
        contact_by_side, interior_by_side = self._interior_foot_support()
        touchdown = contact_by_side & ~self.previous_foot_contact
        after_launch = self.time >= self.landing_check_delay
        invalid = (touchdown & ~interior_by_side).any(dim=1) & after_launch
        self.previous_foot_contact.copy_(contact_by_side)
        self._last_edge_violation.copy_(invalid)
        if not self.require_interior_landing:
            return torch.zeros_like(invalid)
        return invalid

    def _get_terminated(self) -> torch.Tensor:
        fallen = self.root_pos[:, UP_IDX] < (self.course.top_height + MIN_ROOT_HEIGHT)
        not_facing = ~self._is_body_facing_direction(self.root_id)
        invalid_edge_landing = self._invalid_edge_touchdown()
        return (fallen | not_facing | invalid_edge_landing).float().detach()

    def _get_truncated(self) -> torch.Tensor:
        timed_out = super()._get_truncated().bool()
        progress = self.root_pos[:, FWD_IDX] - self._episode_start_x
        competent = timed_out & (progress >= self.curriculum_min_progress)
        self._last_success.copy_(competent)
        return timed_out.float().detach()

    # ---- metrics -------------------------------------------------------------------

    def update_metrics(self) -> None:
        progress = self.root_pos[:, FWD_IDX] - self._episode_start_x
        self.last_progress_x = progress.mean().item()
        self.last_mean_slabs_recycled = self.episode_slabs_recycled.float().mean().item()

    def additional_metrics(self) -> dict:
        return {
            "mean_forward_progress": self.last_progress_x,
            "mean_slabs_recycled": self.last_mean_slabs_recycled,
            "curriculum_step_length_max": self.spacing_curriculum.current_maximum,
            "curriculum_completion_rate": self.spacing_curriculum.last_completion_rate,
        }

    def get_task_state(self) -> dict:
        """State shared with evaluation and persisted in TD3 checkpoints."""
        return {"spacing_curriculum": self.spacing_curriculum.state_dict()}

    def load_task_state(self, state: dict) -> None:
        curriculum_state = state.get("spacing_curriculum") if state else None
        if curriculum_state is not None:
            self.spacing_curriculum.load_state_dict(curriculum_state)
