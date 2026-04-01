import torch

from msk_envs.utils.global_params import UP_IDX
from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import max_vertical_reward, joint_limit_penalty, \
    actuator_sq_penalty


class VerticalEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 live_render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, live_render=live_render,
                         cuda_graph=cuda_graph)

        # Track maximum height achieved during episode for each environment
        self.max_height_achieved = torch.zeros(num_envs, device=self.body_positions.device)
        return

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        # Reset max height tracking
        # self.max_height_achieved[reset_mask.bool().squeeze(-1)] = 0.0  
        self.max_height_achieved[torch.ones_like(reset_mask).bool().squeeze(-1)] = 0.0  # This is to make the reward non-cumulative
        return

    def _compute_raw_reward_dict(self):
        current_max_height = max_vertical_reward(self.body_positions)
        self.max_height_achieved = torch.max(self.max_height_achieved, current_max_height)
        rew_max_vertical = self.max_height_achieved.clone()  # Cloned to avoid early reset issues

        rew_limit = joint_limit_penalty(self.limit_torques, self.num_limits, squared=False)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)

        self.reward_dict = {
            "rew_max_vertical": rew_max_vertical.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
        }

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
            self.joint_positions,
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
            self.body_rotations.view(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()
    
    def _get_terminated(self):
        # Root falls below threshold
        min_root_height = 0.6
        root_height = self.root_pos[:, UP_IDX]
        fallen = (root_height < min_root_height)

        # Head falls below threshold
        min_head_height = 1.0
        head_pos = self.torso_pos + rotate_vec(self.torso_rot, self.head_offset)
        head_fallen = (head_pos[:, UP_IDX] < min_head_height)

        terminated = (fallen | head_fallen).float()
        return terminated.detach()