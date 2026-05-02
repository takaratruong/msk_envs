import torch

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class BackPedalEnv(LanesEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, -1.0)
        )
        return
    #
    # def _get_terminated(self):
    #     # Get normal termination conditions
    #     terminated_lanes = super()._get_terminated().bool()
    #
    #     # Check if head is past pelvis
    #     head_pos_fwd = self.body_positions[:, self.head_id, FWD_IDX]
    #     root_pos_fwd = self.body_positions[:, self.root_id, FWD_IDX]
    #     head_not_over_root = head_pos_fwd - root_pos_fwd > 0.1
    #
    #     terminated = (terminated_lanes | head_not_over_root).bool()
    #     return terminated.detach()
