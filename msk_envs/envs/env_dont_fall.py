import torch

from msk_envs.utils.reward_lib import joint_limit_penalty, \
    actuator_sq_penalty, metabolic_penalty, fatigue_penalty, has_fallen, alive_bonus, root_zero_reward, \
    match_start_pos_reward
from .env_base import MSKEnv
from .env_config import EnvConfig


class DontFallEnv(MSKEnv):
    """ Represents an env where the agent is rewarded for not falling """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph
        )
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
        """
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            self.body_positions.reshape(self.num_worlds, -1),
            self.body_rotations.reshape(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_alive = alive_bonus(self._get_terminated())
        rew_limit = joint_limit_penalty(self.limit_torques, self.num_limits, squared=False)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        rew_fatigue = fatigue_penalty(self.muscle_activations, self.num_muscles)
        rew_metabolic = metabolic_penalty(self.muscle_powers, self.num_muscles)
        rew_root_zero = root_zero_reward(self.root_pos, weight=5.0)
        rew_match_start = match_start_pos_reward(self.joint_positions, self.start_pose, weight=1.0, ignore_root=True)

        self.reward_dict = {
            "rew_alive": rew_alive,
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_fatigue": rew_fatigue.detach(),
            "rew_metabolic": rew_metabolic.detach(),
            "rew_root_zero": rew_root_zero.detach(),
            "rew_match_start": rew_match_start.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(self.root_pos, self.head_pos, self.head_rot, self.head_offset)
        terminated = fallen.float()
        return terminated.detach()
