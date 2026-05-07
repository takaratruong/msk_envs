import torch

from msk_envs.utils.global_params import UP_IDX, build_axis
from msk_envs.utils.reward_lib import joint_penalty, has_fallen
from .env_base import MSKEnv
from .env_config import EnvConfig


class VerticalSparseEnv(MSKEnv):
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
        self.up_axis = torch.tensor(build_axis(UP_IDX, 1.0), device=self.device).unsqueeze(0)
        self.current_best_height = torch.zeros(self.num_worlds, device=self.device)
        self.initial_height = torch.zeros(self.num_worlds, device=self.device)
        # Logging
        self.max_height_reached = 0.0
        return

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.current_best_height.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _head_height(self) -> torch.Tensor:
        return self.body_positions[:, self.head_id, UP_IDX]

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        self.current_best_height[reset_mask] = self._head_height()[reset_mask]
        self.initial_height[reset_mask] = self._head_height()[reset_mask]
        return

    def _compute_raw_reward_dict(self):
        self.current_best_height = torch.maximum(
            self._head_height(),
            self.current_best_height,
        )

        # Only at the VERY last step do we reward
        rew_jump = self.current_best_height - self.initial_height
        episode_not_done_mask = self.get_time() < (self.max_episode_duration - self.delta_t)
        rew_jump[episode_not_done_mask] = 0.0

        rew_limit = joint_penalty(self.ufrc_limit, squared=False)
        rew_alive = torch.ones_like(rew_limit)

        self.reward_dict = {
            "rew_jump": rew_jump,
            "rew_limit": rew_limit.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation, min_root=0.3)
        terminated = fallen.float()
        return terminated.detach() * 0

    def update_metrics(self) -> None:
        max_current_head_height = self.body_positions[:, self.head_id, UP_IDX].max()
        self.max_height_reached = max(max_current_head_height, self.max_height_reached)
        return

    def additional_metrics(self) -> dict:
        return {
            "max_height_reached": self.max_height_reached,
        }
