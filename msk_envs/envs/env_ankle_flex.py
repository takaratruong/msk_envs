import torch

from .env_base import MSKEnv
from .env_config import EnvConfig


class AnkleFlexEnv(MSKEnv):
    """ Represents an environment where the agent must face a specific direction and stay within lanes """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            cuda_graph: bool,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, live_render=live_render, cuda_graph=cuda_graph)
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
        rew_test = torch.zeros(self.num_worlds, device=self.device)

        self.reward_dict = {
            "rew_test": rew_test,
        }

    def _get_terminated(self):
        terminated = torch.zeros(self.num_worlds, device=self.device)
        return terminated.detach()
