import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis
from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import velocity_reward, joint_limit_penalty, \
    actuator_penalty


class SprintingEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render,
                         cuda_graph=cuda_graph)
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths, fiber velocities, actuations
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Body positions relative to root, rotations, velocities
        """
        root_positions = self.body_positions[:, self.root_id, :]
        rel_body_positions = self.body_positions - root_positions.unsqueeze(1)
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.muscle_fiber_velocities,
            self.actuator_activations,
            self.joint_positions[:, 1:],  # exclude x position
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
            self.body_rotations.view(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_limit = joint_limit_penalty(self.limit_torques)
        rew_actuator = actuator_penalty(self.actuator_activations, self.num_actuators)

        reached_finish = (self.root_pos[:, 0] >= 100.0).float()
        time_left = (self.max_episode_duration - self.time).clamp(min=0.0)
        rew_finish = reached_finish * time_left

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_finish": rew_finish.detach(),
        }

    def _get_terminated(self):
        # Reached finish line
        reached_finish = (self.root_pos[:, FWD_IDX] >= 100.0)

        # Root falls below threshold
        min_root_height = 0.6
        root_height = self.root_pos[:, UP_IDX]
        fallen = (root_height < min_root_height)

        # Head falls below threshold
        min_head_height = 1.0
        head_pos = self.torso_pos + rotate_vec(self.torso_rot, self.head_offset)
        head_fallen = (head_pos[:, UP_IDX] < min_head_height)

        # Any of the bodies are out of the lanes
        body_out = torch.zeros_like(head_fallen)
        for body_idx in range(self.body_positions.shape[1]):
            body_pos = self.body_positions[:, body_idx]
            body_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)

        # Pelvis no longer facing forward (within N degrees)
        pelvis_rot = self.body_rotations[:, self.root_id]
        x_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device)

        pelvis_fwd = rotate_vec(pelvis_rot, x_axis.unsqueeze(0))
        facing_forward = (pelvis_fwd[:, FWD_IDX] >= torch.cos(
            torch.deg2rad(torch.tensor(30.0))))
        not_facing_forward = ~facing_forward

        terminated = (reached_finish | fallen | head_fallen | body_out | not_facing_forward).float()
        return terminated.detach()
