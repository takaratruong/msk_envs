import torch
import bolt
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, SIDE_IDX, build_axis, MIN_ROOT_HEIGHT
from msk_envs.utils.reward_lib import velocity_reward_max
from .env_config import EnvConfig
from .env_lanes import LanesEnv

# Stones are box slabs. Their tops are RAISED well above the ground plane so a missed step
# drops the walker down to the ground — a real fall that trips has_fallen.
# (The ground plane itself can't be lowered: env_base calls modify_ground_collider
# right after _add_colliders, which resets its transform to the origin.)
# Each world owns the same set of stone geometry IDs, but their local transforms
# live in Bolt Data and are independent for every world. A fresh layout is
# sampled whenever that world resets.
STONE_TOP_Y = 0.45         # stone tops raised 0.45m above ground → miss = drop = fall
STONE_HALF_THICKNESS = 0.05


class StoneCourseEnv(LanesEnv):
    """ Physical stepping-stone courses with a MINIMAL objective.

    The only reward is forward progress; the only failure is falling. Reaching
    the final slab is a successful episode boundary, not a failure. There is NO
    foot-target reward, no heel/toe term, no gait shaping of any kind. Real box
    slabs raised above the ground physically constrain where feet can land
    (miss = fall to the lower ground); how the foot contacts each stone is left
    entirely to physics. This is the emergence-first setup: change the terrain,
    not the reward.

    Every simulated world has the same number of slab geometry instances, with
    per-world local transforms. Layouts are regenerated independently at each
    episode reset. The policy observes root-relative offsets to the next few
    slabs so randomized placements are a learnable task rather than hidden state.
    """

    N_LOOKAHEAD = 4

    def __init__(self, num_envs, env_config, device, requires_visuals, cuda_graph):
        self.stones_per_course = env_config.course_stones
        self.course_step_len = env_config.course_step_len      # (min,max) forward gap
        self.course_step_width = env_config.course_step_width  # lateral alternation
        self.stone_radius = env_config.course_stone_radius
        if self.stones_per_course < 1:
            raise ValueError("course_stones must be at least 1")
        # A valid shared layout is needed while the model and CUDA graphs are
        # initialized. It is replaced independently for each world on reset.
        self._default_stones = self._build_default_layout()

        super().__init__(
            num_envs=num_envs, env_config=env_config, device=device,
            requires_visuals=requires_visuals, cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
            lane_width=1e9,  # disable lane termination — stones define the path
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
            (num_envs, self.stones_per_course, 3),
            device=device,
            dtype=torch.float32,
        )
        self._last_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_progress_x = 0.0

    # ---- per-world course layout + physical stones ----------------------------------

    def _build_default_layout(self):
        """Return a safe deterministic layout used only during initialization."""
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
        if n_reset == 0 or self.stones_per_course == 0:
            return

        lo, hi = self.course_step_len
        step_lengths = torch.rand(
            (n_reset, self.stones_per_course), device=self.device
        ) * (hi - lo) + lo

        # Keep the launch pair reachable, but randomize both their forward and
        # lateral placement so even the beginning is not identical per world.
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
        colliders = self.load_result.colliders
        # Add one shared set of geometry IDs. Data.geom_X_loc supplies a distinct
        # transform for every (world, stone) instance.
        for stone_id, (sx, sy, sz) in enumerate(self._default_stones):
            stone = bolt.UserGeomData(
                name=f"stone_{stone_id}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.BOX,
                transform=wp.transform(wp.vec3(sx, sy, sz), wp.quat_identity(dtype=float)),
                size=wp.vec3(self.stone_radius, STONE_HALF_THICKNESS, self.stone_radius),
                priority=9,
            )
            colliders.append(bolt.convert_user_collider(stone))
        return

    # ---- observation + spawn --------------------------------------------------------

    def _get_obs(self) -> torch.Tensor:
        base_obs = super()._get_obs()
        root_xz = self.root_pos[:, [FWD_IDX, SIDE_IDX]]

        # Treat a slab as passed only once the pelvis is slightly beyond its
        # center, then expose the next N_LOOKAHEAD centers relative to the root.
        passed = self.stone_positions[:, :, FWD_IDX] < (self.root_pos[:, FWD_IDX, None] - 0.10)
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
        # Raise spawn so the feet begin on the elevated launch slabs.
        ty = self.qpos_id_lookup["pelvis_ty"]
        self.joint_positions[reset_mask, ty] += STONE_TOP_Y
        # Re-realize the state after changing root height so the returned
        # observations and first policy step see consistent body transforms.
        self.launch_sim_reset()
        return

    # ---- MINIMAL reward: forward progress only --------------------------------------

    def _compute_raw_reward_dict(self):
        # Forward velocity, capped so it walks rather than sprints. This is the
        # ONLY reward. No foot targets, no gait terms.
        rew_vel = torch.nan_to_num(
            velocity_reward_max(self.body_velocities, self.root_id, FWD_IDX,
                                linear=True, target_speed=self.target_speed),
            nan=0.0, posinf=0.0, neginf=0.0)
        rew_alive = torch.ones(self.num_worlds, device=self.device)
        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        # Fall = root drops below (stone-top + MIN_ROOT_HEIGHT). Standing on a
        # raised stone the pelvis is ~STONE_TOP_Y + normal_root_height; a missed
        # step drops it toward the ground (~STONE_TOP_Y lower), tripping this.
        # Threshold is stone-relative so the drop actually registers as a fall.
        fallen = self.root_pos[:, UP_IDX] < (STONE_TOP_Y + MIN_ROOT_HEIGHT)
        not_facing = ~self._is_body_facing_direction(self.root_id)
        return (fallen | not_facing).float().detach()

    def _reached_last_stone(self) -> torch.Tensor:
        """Whether an active foot contact is on the final physical slab."""
        last_stone_id = self.stone_ids[-1]
        last_stone_active = self.collider_forces[:, last_stone_id] > 0.0

        foot_positions = self.collider_positions[:, self.foot_collider_ids]
        foot_active = self.collider_forces[:, self.foot_collider_ids] > 0.0
        last_xz = self.stone_positions[:, -1, [FWD_IDX, SIDE_IDX]].unsqueeze(1)
        foot_xz = foot_positions[:, :, [FWD_IDX, SIDE_IDX]]

        # Slabs are axis-aligned boxes. Include each sphere's radius so an edge
        # contact counts as reaching the slab, then require both the sphere and
        # the final slab to carry contact force in the same simulation state.
        reach = self.stone_radius + self.foot_collider_radii.unsqueeze(0)
        over_last_stone = ((foot_xz - last_xz).abs() <= reach.unsqueeze(-1)).all(dim=2)
        return (last_stone_active & (foot_active & over_last_stone).any(dim=1)).detach()

    def _get_truncated(self):
        # The training API resets on terminated OR truncated. Use truncation for
        # successful course completion so TD3 bootstraps this boundary instead
        # of assigning it the same zero-terminal value as a fall.
        timed_out = super()._get_truncated().bool()
        reached_last = self._reached_last_stone()
        self._last_success.copy_(reached_last)
        return (timed_out | reached_last).float().detach()

    def get_render_targets(self, world_id: int):
        positions = self.stone_positions[world_id].detach().cpu().tolist()
        return [((sx, STONE_TOP_Y, sz), self.stone_radius, False)
                for sx, _, sz in positions]

    def update_metrics(self) -> None:
        self.last_progress_x = self.root_pos[:, FWD_IDX].mean().item()
        return

    def additional_metrics(self) -> dict:
        return {
            "mean_forward_progress": self.last_progress_x,
            "successful_completion_rate": self._last_success.float().mean().item(),
        }
