import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_reach_target import ReachTargetEnv


class RandomTargetEnv(ReachTargetEnv):
    """ Env where the agent must run between random targets that are 3 meters apart"""

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

    @staticmethod
    def _sample_target_from_points(points: torch.Tensor) -> torch.Tensor:
        rand_dirs = torch.randn((points.shape[0], 3), device=points.device)
        rand_dirs[:, UP_IDX] = 0.0
        rand_dirs = rand_dirs / torch.norm(rand_dirs, dim=1, keepdim=True)
        new_targets = points + rand_dirs * 3.0
        return new_targets

    def _new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        if new_env:
            # If new env, current target is random around root, next target is random around current target
            self.curr_target_pos[reset_mask] = self._sample_target_from_points(self.root_pos[reset_mask])
            self.next_target_pos[reset_mask] = self._sample_target_from_points(self.curr_target_pos[reset_mask])
        else:
            # Otherwise, current target gets the next target, next target is random around current target
            self.curr_target_pos[reset_mask] = self.next_target_pos[reset_mask]
            self.next_target_pos[reset_mask] = self._sample_target_from_points(self.curr_target_pos[reset_mask])
        return
