import torch

from msk_envs.utils.reward_lib import has_fallen, single_body_pos_track_reward, update_dict
from .env_base import MSKEnv
from .env_config import EnvConfig


class StaticEnv(MSKEnv):
    """ Represents an environment where the agent match a pelvis position """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            cuda_graph: bool,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, live_render=live_render, cuda_graph=cuda_graph)
        self.target_position = torch.tensor(env_config.target_position, device=self.device).unsqueeze(0)
        self.imitation_weights = env_config.imitation_weights

        # update fall thresholds
        self.min_root_height = max(env_config.target_position[1] - 0.2, 0.0)
        self.min_head_height = max(env_config.target_position[1] + 0.2, 0.0)
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
        self.reward_dict = {}

        rew_track_root_pos = single_body_pos_track_reward(
            self.root_pos, self.target_position,
            weight=self.imitation_weights["imitation_weight_track"])
        update_dict(self.reward_dict, "rew_track_root_pos", rew_track_root_pos)

        self.reward_dict = {
            "rew_track": rew_track_root_pos,
        }

    def _get_terminated(self):
        fallen = has_fallen(self.root_pos, self.torso_pos, self.torso_rot, self.head_offset,
                            min_root=self.min_root_height, min_head=self.min_head_height)
        terminated = fallen.float()
        return terminated.detach()
