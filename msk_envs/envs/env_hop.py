import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_max_effort_lanes import MaxEffortLanesEnv


class HopEnv(MaxEffortLanesEnv):
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
            target_dir=build_axis(FWD_IDX, 1.0),
        )
        return


    def _get_terminated(self):
        # Get normal termination conditions
        terminated_sprint = super()._get_terminated()

        # Check if left toe is on the ground
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        left_toe_height = left_toe_pos[:, UP_IDX]
        left_toe_on_ground = (left_toe_height < 0.05).float()

        terminated = torch.max(terminated_sprint, left_toe_on_ground)
        return terminated.detach()
