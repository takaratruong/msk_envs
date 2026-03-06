import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_reach_target import ReachTargetEnv


class ShuttleRunEnv(ReachTargetEnv):
    """ Env where the agent must run between two targets repeatedly """
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            render: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            render=render,
            cuda_graph=cuda_graph,
        )
        self.target_1 = torch.tensor(build_axis(FWD_IDX, 2.5), device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self.target_2 = torch.tensor(build_axis(FWD_IDX, -2.5), device=self.device).unsqueeze(0).repeat(self.num_worlds, 1)
        self.target_1[:, UP_IDX] = 1.0
        self.target_2[:, UP_IDX] = 1.0

        self.curr_target_pos[:] = self.target_1
        self.next_target_pos[:] = self.target_2
        return

    def _new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        # New env, reset to original
        if new_env:
            self.curr_target_pos[reset_mask] = self.target_1[reset_mask]
            self.next_target_pos[reset_mask] = self.target_2[reset_mask]
            return
        # Otherwise, swap curr target and next target for envs indicated by reset_mask
        curr_targets_reset = self.curr_target_pos[reset_mask].clone()
        self.curr_target_pos[reset_mask] = self.next_target_pos[reset_mask]
        self.next_target_pos[reset_mask] = curr_targets_reset
        return


