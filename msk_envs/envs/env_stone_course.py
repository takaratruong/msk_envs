import bolt
import torch
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, MIN_ROOT_HEIGHT, SIDE_IDX, UP_IDX, build_axis
from msk_envs.utils.reward_lib import velocity_reward_max
from .env_config import EnvConfig
from .env_lanes import LanesEnv


STONE_TOP_Y = 0.45
STONE_HALF_THICKNESS = 0.05


class StoneCourseEnv(LanesEnv):
    """Physical stepping stones with independent per-world layouts.

    Every simulated world owns the same number of box collider IDs, while the
    collider transforms are stored per world in Bolt data. Layouts are sampled
    independently whenever a world resets. The policy observes the next four
    slab centers relative to its root.
    """

    N_LOOKAHEAD = 4

    def __init__(self, num_envs, env_config, device, requires_visuals, cuda_graph):
        self.stones_per_course = env_config.course_stones
        self.course_step_len = env_config.course_step_len
        self.course_step_width = env_config.course_step_width
        self.stone_radius = env_config.course_stone_radius
        if self.stones_per_course < 1:
            raise ValueError("course_stones must be at least 1")
        self._default_stones = self._build_default_layout()

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
            [self.collider_id_lookup[f"stone_{stone_id}"]
             for stone_id in range(self.stones_per_course)],
            device=device,
            dtype=torch.long,
        )
        self.foot_collider_ids = torch.tensor(
            [collider_id for name, collider_id in self.collider_id_lookup.items()
             if name.startswith("left_foot_") or name.startswith("right_foot_")],
            device=device,
            dtype=torch.long,
        )
        if self.foot_collider_ids.numel() == 0:
            raise ValueError("StoneCourse requires named left_foot_/right_foot_ colliders")
        self.foot_collider_radii = self.collider_sizes[self.foot_collider_ids, 0]
        self.stone_positions = torch.zeros(
            (num_envs, self.stones_per_course, 3), device=device, dtype=torch.float32
        )
        self._last_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_progress_x = 0.0

    def _build_default_layout(self):
        """Return a deterministic layout used while the model is initialized."""
        lo, hi = self.course_step_len
        stones = []
        x = 0.0
        for stone_id in range(self.stones_per_course):
            x += 0.35 if stone_id < 2 else (lo + hi) * 0.5
            side = 0.12 if stone_id % 2 == 0 else -0.12
            stones.append((x, STONE_TOP_Y - STONE_HALF_THICKNESS, side))
        return stones

    def _randomize_stones(self, reset_mask: torch.Tensor) -> None:
        """Sample and install a fresh physical course for each resetting world."""
        world_ids = torch.where(reset_mask)[0]
        n_reset = world_ids.numel()
        if n_reset == 0:
            return

        lo, hi = self.course_step_len
        step_lengths = torch.rand(
            (n_reset, self.stones_per_course), device=self.device
        ) * (hi - lo) + lo

        launch_count = min(2, self.stones_per_course)
        step_lengths[:, :launch_count] = (
            torch.rand((n_reset, launch_count), device=self.device) * 0.06 + 0.32
        )

        stone_x = torch.cumsum(step_lengths, dim=1)
        stone_ids = torch.arange(self.stones_per_course, device=self.device)
        base_side = torch.where(
            stone_ids % 2 == 0,
            torch.tensor(0.12, device=self.device),
            torch.tensor(-0.12, device=self.device),
        ).unsqueeze(0)
        lateral_jitter = (
            torch.rand((n_reset, self.stones_per_course), device=self.device) * 2.0 - 1.0
        ) * self.course_step_width
        lateral_jitter[:, :launch_count] *= 0.2
        stone_z = base_side + lateral_jitter
        stone_y = torch.full_like(stone_x, STONE_TOP_Y - STONE_HALF_THICKNESS)
        positions = torch.stack((stone_x, stone_y, stone_z), dim=-1)

        self.stone_positions[world_ids] = positions
        self.collider_local_transforms[
            world_ids[:, None], self.stone_ids[None, :], :3
        ] = positions

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self._randomize_stones(reset_mask)

    def _add_colliders(self, env_config: EnvConfig) -> None:
        for stone_id, (sx, sy, sz) in enumerate(self._default_stones):
            stone = bolt.UserGeomData(
                name=f"stone_{stone_id}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.BOX,
                transform=wp.transform(wp.vec3(sx, sy, sz), wp.quat_identity(dtype=float)),
                size=wp.vec3(self.stone_radius, STONE_HALF_THICKNESS, self.stone_radius),
                priority=9,
            )
            self.load_result.colliders.append(bolt.convert_user_collider(stone))

    def _get_obs(self) -> torch.Tensor:
        base_obs = super()._get_obs()
        root_xz = self.root_pos[:, [FWD_IDX, SIDE_IDX]]
        passed = self.stone_positions[:, :, FWD_IDX] < (
            self.root_pos[:, FWD_IDX, None] - 0.10
        )
        next_idx = passed.sum(dim=1)
        lookahead = []
        for offset in range(self.N_LOOKAHEAD):
            idx = torch.clamp(next_idx + offset, max=self.stones_per_course - 1)
            target = torch.gather(
                self.stone_positions,
                1,
                idx.view(-1, 1, 1).expand(-1, 1, 3),
            ).squeeze(1)
            lookahead.append(target[:, [FWD_IDX, SIDE_IDX]] - root_xz)
        stone_obs = torch.cat(lookahead, dim=1)
        return torch.cat((stone_obs, base_obs), dim=1).detach().clone()

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        self.joint_positions[reset_mask, self.qpos_id_lookup["pelvis_ty"]] += STONE_TOP_Y
        self.launch_sim_reset()

    def _compute_raw_reward_dict(self):
        rew_vel = torch.nan_to_num(
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
            "rew_vel": rew_vel.detach(),
            "rew_alive": torch.ones(self.num_worlds, device=self.device).detach(),
        }

    def _get_terminated(self):
        fallen = self.root_pos[:, UP_IDX] < (STONE_TOP_Y + MIN_ROOT_HEIGHT)
        not_facing = ~self._is_body_facing_direction(self.root_id)
        return (fallen | not_facing).float().detach()

    def _reached_last_stone(self) -> torch.Tensor:
        """Return worlds with an active foot contact on the final slab."""
        last_stone_id = self.stone_ids[-1]
        last_stone_active = self.collider_forces[:, last_stone_id] > 0.0
        foot_positions = self.collider_positions[:, self.foot_collider_ids]
        foot_active = self.collider_forces[:, self.foot_collider_ids] > 0.0
        last_xz = self.stone_positions[:, -1, [FWD_IDX, SIDE_IDX]].unsqueeze(1)
        foot_xz = foot_positions[:, :, [FWD_IDX, SIDE_IDX]]
        reach = self.stone_radius + self.foot_collider_radii.unsqueeze(0)
        over_last_stone = ((foot_xz - last_xz).abs() <= reach.unsqueeze(-1)).all(dim=2)
        return (last_stone_active & (foot_active & over_last_stone).any(dim=1)).detach()

    def _get_truncated(self):
        timed_out = super()._get_truncated().bool()
        reached_last = self._reached_last_stone()
        self._last_success.copy_(reached_last)
        return (timed_out | reached_last).float().detach()

    def get_render_targets(self, world_id: int):
        positions = self.stone_positions[world_id].detach().cpu().tolist()
        return [
            ((sx, STONE_TOP_Y, sz), self.stone_radius, False)
            for sx, _, sz in positions
        ]

    def update_metrics(self) -> None:
        self.last_progress_x = self.root_pos[:, FWD_IDX].mean().item()

    def additional_metrics(self) -> dict:
        return {
            "mean_forward_progress": self.last_progress_x,
            "successful_completion_rate": self._last_success.float().mean().item(),
        }
