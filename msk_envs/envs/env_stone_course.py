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
        if self.num_stones < 1:
            raise ValueError("course_stones must be at least 1")
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
        """Deterministic layout used while Bolt constructs the shared geometry."""
        midpoint = sum(self.step_length_range) * 0.5
        step_lengths = torch.full((1, self.num_stones), midpoint, device=device)
        launch_count = min(2, self.num_stones)
        launch_midpoint = sum(self.launch_step_length_range) * 0.5
        step_lengths[:, :launch_count] = launch_midpoint
        lateral_jitter = torch.zeros_like(step_lengths)
        return self._positions_from_steps(step_lengths, lateral_jitter).squeeze(0)

    def sample_positions(
        self,
        num_courses: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one independent layout per course/world."""
        if num_courses < 0:
            raise ValueError("num_courses must be non-negative")

        lo, hi = self.step_length_range
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
        root_xz = root_positions[:, [FWD_IDX, SIDE_IDX]]
        passed = stone_positions[:, :, FWD_IDX] < (
            root_positions[:, FWD_IDX, None] - self.passed_margin
        )
        next_stone = passed.sum(dim=1)
        offsets = lookahead_offsets
        if offsets is None:
            offsets = torch.arange(self.lookahead, device=stone_positions.device)
        indices = (next_stone[:, None] + offsets[None, :]).clamp_max(self.num_stones - 1)
        targets = torch.gather(
            stone_positions,
            1,
            indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        return (targets[:, :, [FWD_IDX, SIDE_IDX]] - root_xz[:, None, :]).flatten(1)


class StoneCourseEnv(LanesEnv):
    """Raised box slabs with a new independent layout in every world/reset.

    Bolt keeps one shared set of collider IDs, but collider transforms live in
    the per-world data tensor. Updating only a world's transform rows gives it
    its own course while preserving batched broadphase and collision isolation.

    The task intentionally has no foot-target or gait-shaping reward. It rewards
    capped forward velocity and staying upright. Falling is a terminal failure;
    contacting the final slab is a successful truncation.
    """

    def __init__(self, num_envs, env_config, device, requires_visuals, cuda_graph):
        self.course = StoneCourseSpec.from_env_config(env_config)
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
        self.foot_collider_ids = torch.tensor(
            [
                collider_id
                for name, collider_id in self.collider_id_lookup.items()
                if name.startswith(("left_foot_", "right_foot_"))
            ],
            device=device,
            dtype=torch.long,
        )
        if self.foot_collider_ids.numel() == 0:
            raise ValueError("StoneCourse requires named left_foot_/right_foot_ colliders")

        self.foot_collider_radii = self.collider_sizes[self.foot_collider_ids, 0]
        self.slab_half_extents_xz = torch.tensor(
            [self.course.half_extents[FWD_IDX], self.course.half_extents[SIDE_IDX]],
            device=device,
            dtype=self.foot_collider_radii.dtype,
        )
        self.lookahead_offsets = torch.arange(self.course.lookahead, device=device)
        self.stone_positions = default_positions.to(device).unsqueeze(0).repeat(num_envs, 1, 1)
        self._last_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_progress_x = 0.0

    # ---- course geometry and per-world randomization -------------------------------

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

    def _randomize_stones(self, reset_mask: torch.Tensor) -> None:
        """Sample and install a fresh physical course for each resetting world."""
        world_ids = torch.where(reset_mask.flatten().bool())[0]
        if world_ids.numel() == 0:
            return
        positions = self.course.sample_positions(world_ids.numel(), self.device)
        self._set_course_positions(world_ids, positions)
        self._last_success[world_ids] = False

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self._randomize_stones(reset_mask)

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        pelvis_height = self.qpos_id_lookup["pelvis_ty"]
        self.joint_positions[reset_mask, pelvis_height] += self.course.top_height
        self.launch_sim_reset()

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

    def _get_terminated(self) -> torch.Tensor:
        fallen = self.root_pos[:, UP_IDX] < (self.course.top_height + MIN_ROOT_HEIGHT)
        not_facing = ~self._is_body_facing_direction(self.root_id)
        return (fallen | not_facing).float().detach()

    def _reached_last_stone(self) -> torch.Tensor:
        """Return worlds with an active foot contact on the final physical slab."""
        final_slab_active = self.collider_forces[:, self.stone_ids[-1]] > 0.0
        foot_active = self.collider_forces[:, self.foot_collider_ids] > 0.0

        foot_xz = self.collider_positions[:, self.foot_collider_ids][
            :, :, [FWD_IDX, SIDE_IDX]
        ]
        final_slab_xz = self.stone_positions[:, -1, [FWD_IDX, SIDE_IDX]].unsqueeze(1)
        reach = self.slab_half_extents_xz + self.foot_collider_radii.unsqueeze(1)
        foot_over_final_slab = (
            (foot_xz - final_slab_xz).abs() <= reach.unsqueeze(0)
        ).all(dim=2)
        return (
            final_slab_active & (foot_active & foot_over_final_slab).any(dim=1)
        ).detach()

    def _get_truncated(self) -> torch.Tensor:
        timed_out = super()._get_truncated().bool()
        reached_final_slab = self._reached_last_stone()
        self._last_success.copy_(reached_final_slab)
        return (timed_out | reached_final_slab).float().detach()

    # ---- metrics -------------------------------------------------------------------

    def update_metrics(self) -> None:
        self.last_progress_x = self.root_pos[:, FWD_IDX].mean().item()

    def additional_metrics(self) -> dict:
        return {
            "mean_forward_progress": self.last_progress_x,
            "successful_completion_rate": self._last_success.float().mean().item(),
        }
