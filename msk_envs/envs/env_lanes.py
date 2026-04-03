import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis
from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import velocity_reward, joint_limit_penalty, joint_damping_penalty, \
    actuator_sq_penalty, metabolic_penalty, fatigue_penalty, mid_lane_reward, has_fallen
from ..utils.scene_settings import SceneSettings


class LanesEnv(MSKEnv):
    """ Represents an environment where the agent must face a specific direction and stay within lanes """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
            target_dir: list[float],
            angle_tolerance: float = 30.0,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph
        )
        assert target_dir[UP_IDX] == 0.0, "Target direction should be horizontal only"

        # Precompute 2d target facing direction
        self.target_facing = torch.tensor(target_dir, device=self.device).unsqueeze(0)
        self.target_facing = self.target_facing[:, [FWD_IDX, SIDE_IDX]]
        self.target_facing = self.target_facing / torch.norm(self.target_facing, dim=1, keepdim=True)

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)
        self.toes_ids = [self.lookup_body_id("toes_l"), self.lookup_body_id("toes_r")]
        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(angle_tolerance, device=self.device)))
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Normalized time
         2. Muscle activations, fiber lengths
         3. Actuator activations
         4. Joint positions (q)
         5. Joint velocities (qv)
        """
        time_curr = self.time.view(self.num_worlds, 1) / self.max_episode_duration
        obs = torch.cat([
            time_curr,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_mid_lane = mid_lane_reward(self.root_pos)
        rew_limit = joint_limit_penalty(self.limit_torques, self.num_limits, squared=False)
        rew_damping = joint_damping_penalty(self.ufrc_damper, squared=False)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        rew_fatigue = fatigue_penalty(self.muscle_activations, self.num_muscles)
        # rew_metabolic = metabolic_penalty(self.muscle_powers, self.num_muscles)

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_damping": rew_damping.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_fatigue": rew_fatigue.detach(),
            # "rew_metabolic": rew_metabolic.detach(),
        }

    def _get_terminated(self):
        # Has fallen
        fallen = has_fallen(self.root_pos, self.head_pos, self.head_rot, self.head_offset)

        # # Any of the *toes* are out of the lanes
        toes_out = torch.zeros_like(fallen, dtype=torch.bool)
        for body_idx in self.toes_ids:
            body_pos = self.body_positions[:, body_idx]
            toes_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)

        # # Pelvis no longer facing forward (within N degrees)
        pelvis_rot = self.body_rotations[:, self.root_id]
        pelvis_fwd = rotate_vec(pelvis_rot, self.fwd_axis)
        pelvis_fwd = pelvis_fwd[:, [FWD_IDX, SIDE_IDX]]  # Only care about x/z components
        pelvis_fwd = pelvis_fwd / torch.norm(pelvis_fwd, dim=1, keepdim=True)

        pelvis_fwd_dot_target = torch.sum(pelvis_fwd * self.target_facing, dim=1)
        facing_direction = pelvis_fwd_dot_target >= self.cos_angle_threshold
        not_facing_direction = ~facing_direction

        terminated = (fallen | toes_out | not_facing_direction).float()
        return terminated.detach()

    def scene_settings(self) -> SceneSettings:
        return SceneSettings(
            lanes=True,
            lane_width=0.6,
            meter_markers=True,
            axes=False
        )
