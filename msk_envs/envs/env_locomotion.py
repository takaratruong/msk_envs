import math
import random

import bolt
import torch
import warp as wp

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, SIDE_IDX
from msk_envs.utils.reward_lib import joint_penalty, has_fallen, muscle_passive_penalty
from .env_base import MSKEnv
from .env_config import EnvConfig

# Radius of the buried spheres used as terrain bumps; only the cap above the
# ground plane (terrain_bump_height) is walkable, so bumps stay smooth.
TERRAIN_BUMP_RADIUS = 0.1
# Fixed seed so train and eval envs (constructed separately) share the layout.
TERRAIN_SEED = 1234


class LocomotionEnv(MSKEnv):
    """ General locomotion: track a per-episode commanded horizontal velocity. """

    def _add_colliders(self, env_config: EnvConfig) -> None:
        # Small unobserved ground bumps: buried spheres whose caps protrude by a
        # few mm-cm. Punishes zero-clearance gaits (scuffing, foot-dragging)
        # without meaningfully changing the task on the scale of a stride.
        if not env_config.apply_terrain_noise:
            return
        colliders = self.load_result.colliders
        rng = random.Random(TERRAIN_SEED)
        h_lo, h_hi = env_config.terrain_bump_height
        for i in range(env_config.terrain_bump_count):
            # Uniform over the disk, excluding the spawn clearing
            r = env_config.terrain_extent * math.sqrt(rng.random())
            r = max(r, env_config.terrain_clear_radius)
            theta = rng.random() * 2.0 * math.pi
            x, z = r * math.cos(theta), r * math.sin(theta)
            height = rng.uniform(h_lo, h_hi)
            pos = [0.0, 0.0, 0.0]
            pos[FWD_IDX] = x
            pos[SIDE_IDX] = z
            pos[UP_IDX] = height - TERRAIN_BUMP_RADIUS  # bury: only the cap protrudes
            bump = bolt.UserGeomData(
                name=f"terrain_bump_{i}",
                body_name=bolt.GROUND,
                geom_type=bolt.GeomType.SPHERE,
                transform=wp.transform(wp.vec3(*pos), wp.quat_identity(dtype=float)),
                size=wp.vec3(TERRAIN_BUMP_RADIUS, TERRAIN_BUMP_RADIUS, TERRAIN_BUMP_RADIUS),
                priority=9,
            )
            colliders.append(bolt.convert_user_collider(bump))
        return

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool,
            speed_min: float = 0.0,
            speed_max: float = 4.0,
            heading_range_deg: float = 120.0,
            stand_prob: float = 0.1,
            track_sigma: float = 0.25,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, requires_visuals=requires_visuals,
                         cuda_graph=cuda_graph)

        # Command sampling settings
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.heading_range = math.radians(heading_range_deg)  # half-cone about +forward
        self.stand_prob = stand_prob
        self.track_sigma = track_sigma

        # Per-env commanded horizontal velocity (FWD, SIDE), world frame.
        # Sampled on every reset (initialized here so _get_obs works pre-reset).
        self.command_vel = torch.zeros((num_envs, 2), device=device)

        # Mid-episode command resampling: forces gait transitions (start/stop/turn),
        # which filters out gaits that can only hold one steady state.
        self.resample_commands = env_config.resample_commands
        self.resample_interval = env_config.command_resample_interval
        self.command_timer = torch.zeros(num_envs, device=device)
        self.command_hold_duration = torch.zeros(num_envs, device=device)

        # Obs excludes absolute horizontal root translation so the policy is
        # position-invariant (it can move in any direction). Keep everything else
        # including root height and orientation.
        drop_names = ["pelvis_tx", "pelvis_tz"]
        drop_ids = {self.qpos_id_lookup[n] for n in drop_names if n in self.qpos_id_lookup}
        keep_ids = [i for i in range(self.num_qpos) if i not in drop_ids]
        self.obs_qpos_keep_ids = torch.tensor(keep_ids, device=device, dtype=torch.long)

        # Logging
        self.last_track_err = 0.0
        self.last_mean_activation = 0.0
        self.last_root_height = 0.0
        return

    def _sample_commands(self, reset_mask: torch.Tensor) -> None:
        """ Sample a new velocity command for the envs flagged in reset_mask. """
        n = int(reset_mask.sum())
        if n == 0:
            return
        speed = torch.rand(n, device=self.device) * (self.speed_max - self.speed_min) + self.speed_min
        angle = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.heading_range
        # A fraction of commands are "stand still" (zero velocity) for robustness.
        stand = torch.rand(n, device=self.device) < self.stand_prob
        speed = torch.where(stand, torch.zeros_like(speed), speed)
        cmd = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle)], dim=1)  # (FWD, SIDE)
        self.command_vel[reset_mask] = cmd
        # Restart the hold timer with a fresh duration for the resampled envs
        low, high = self.resample_interval
        hold = torch.rand(self.num_worlds, device=self.device) * (high - low) + low
        self.command_timer[reset_mask] = 0.0
        self.command_hold_duration[reset_mask] = hold[reset_mask]
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        self._sample_commands(reset_mask)
        return

    def _pre_step(self) -> None:
        # Resample commands mid-episode once their hold duration expires
        if not self.resample_commands:
            return
        self.command_timer += self.delta_t
        expired = self.command_timer >= self.command_hold_duration
        if expired.any():
            self._sample_commands(expired)
        return

    def _root_vel_xy(self) -> torch.Tensor:
        """ World-frame horizontal (FWD, SIDE) linear velocity of the root. """
        return self.body_velocities[:, self.root_id][:, [FWD_IDX + 3, SIDE_IDX + 3]]

    def _get_obs(self) -> torch.Tensor:
        """
        Observation space:
         0. Commanded horizontal velocity (FWD, SIDE)
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q), excluding absolute horizontal root translation
         4. Joint velocities (qv)
        """
        joint_positions = self.joint_positions[:, self.obs_qpos_keep_ids]
        obs = torch.cat([
            self.command_vel,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        # Velocity tracking: exponential of squared error to the command.
        # Guard against a rare diverging env (adaptive integrator hitting min step
        # can produce inf/nan velocities); a single bad env must not poison the batch.
        root_vel = torch.nan_to_num(self._root_vel_xy(), nan=0.0, posinf=1e3, neginf=-1e3)
        vel_err = root_vel - self.command_vel
        rew_vel_track = torch.exp(-self.track_sigma * vel_err.pow(2).sum(dim=1))

        # Alive bonus (kept positive while upright; falling terminates the episode).
        rew_alive = torch.ones(self.num_worlds, device=self.device)

        # Penalties (shared with the lane environments). Same NaN/inf guard.
        rew_limit = torch.nan_to_num(
            joint_penalty(self.ufrc_limit, squared=False), nan=0.0, posinf=0.0, neginf=0.0)
        rew_muscle_passive = torch.nan_to_num(
            muscle_passive_penalty(self.muscle_passive_length_multiplier, threshold=0.1, squared=False),
            nan=0.0, posinf=0.0, neginf=0.0)

        self.reward_dict = {
            "rew_vel_track": rew_vel_track.detach(),
            "rew_alive": rew_alive.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)
        return fallen.float().detach()

    def update_metrics(self) -> None:
        # Mean speed-tracking error this step (norm of velocity error).
        vel_err = self._root_vel_xy() - self.command_vel
        self.last_track_err = vel_err.norm(dim=1).mean().item()
        # Crouch/co-contraction watch: if mean activation climbs while root
        # height drops as push magnitude ramps, the policy is finding the
        # robust-but-unnatural crouch attractor.
        self.last_mean_activation = self.muscle_activations.mean().item()
        self.last_root_height = self.root_pos[:, UP_IDX].mean().item()
        return

    def additional_metrics(self) -> dict:
        return {
            "vel_track_err": self.last_track_err,
            "command_speed_mean": self.command_vel.norm(dim=1).mean().item(),
            "mean_muscle_activation": self.last_mean_activation,
            "mean_root_height": self.last_root_height,
        }
