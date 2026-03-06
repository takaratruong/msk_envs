import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from msk_envs.utils.reward_lib import joint_limit_penalty, actuator_sq_penalty, has_fallen, alive_bonus
from .env_base import MSKEnv
from .env_config import EnvConfig


class BroadJumpEnv(MSKEnv):
    """ Represents an environment where the agent perform a broad jump"""

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            render: bool,
            cuda_graph: bool,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render, cuda_graph=cuda_graph)
        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)
        self.root_pos = self.body_positions[:, self.root_id]
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

    def _new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        """ Override this method to compute new target positions for the environments indicated by reset_mask """
        raise NotImplementedError

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        super()._upon_reset_post_sim(reset_mask)
        # Do any post reset tasks here
        return

    def _compute_raw_reward_dict(self):
        # Add rewards here

        terminated = self._get_terminated()
        rew_alive = alive_bonus(terminated)

        self.reward_dict = {
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        # Fallen
        fallen = has_fallen(self.root_pos, self.torso_pos, self.torso_rot, self.head_offset)
        terminated = fallen.float()
        return terminated.detach()
