import torch

from msk_envs.utils.global_params import SIDE_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class SideShuffleEnv(LanesEnv):
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
