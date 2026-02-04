import torch

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class SideShuffleEnv(LanesEnv):
    """ Env where the agent must face sideways and move forward without crossing legs """
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
            target_dir=build_axis(SIDE_IDX, 1.0)
        )
        return

    def _get_terminated(self):
        # Get normal termination conditions
        terminated_lanes = super()._get_terminated().bool()

        # Check if the right toe went past the left toe
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        right_toe_pos = self.body_positions[:, self.toes_ids[1], :]
        left_toe_x = left_toe_pos[:, FWD_IDX]
        right_toe_x = right_toe_pos[:, FWD_IDX]
        right_toe_crossed = (right_toe_x > left_toe_x)

        terminated = (terminated_lanes | right_toe_crossed).bool()
        return terminated.detach()
