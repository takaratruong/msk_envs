import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class RaceWalkEnv(LanesEnv):
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
        terminated_lanes = super()._get_terminated().bool()

        # In race-walking, one foot must always be on the ground
        grf_mag = torch.norm(self.grf, dim=1)
        not_touching_ground = ~(grf_mag > 0.0)

        terminated = (terminated_lanes | not_touching_ground).bool()
        return terminated.detach()
