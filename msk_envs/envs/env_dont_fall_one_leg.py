import torch

from .env_config import EnvConfig
from .env_dont_fall import DontFallEnv
from msk_envs.utils.global_params import UP_IDX


class DontFallEnvOneLeg(DontFallEnv):
    """ Represents an env where the agent is rewarded for not falling """

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
        self.toes_l_id = self.body_id_lookup["toes_l"]
        return

    def _get_terminated(self):
        terminated_super = super()._get_terminated().bool()
        toes_l_pos = self.body_positions[:, self.toes_l_id]
        toes_l_height = toes_l_pos[:, UP_IDX]
        toes_on_ground = toes_l_height < 0.1
        return (terminated_super | toes_on_ground).detach()
