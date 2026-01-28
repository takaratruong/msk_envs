import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class HopEnv(LanesEnv):
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
        self.left_knee_qpos_idx = self.dof_id_lookup["knee_angle_l"][0]
        self.left_knee_min_angle = -torch.deg2rad(torch.tensor(100.0, device=self.device))
        return

    def _get_terminated(self):
        # Get normal termination conditions
        terminated_lanes = super()._get_terminated().bool()

        # Check if left toe is too low
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        left_toe_height = left_toe_pos[:, UP_IDX]
        left_toe_on_ground = (left_toe_height < 0.3)

        # Check if left knee is extended too much
        left_knee_angle = self.joint_positions[:, self.left_knee_qpos_idx]
        left_knee_extended = (left_knee_angle > self.left_knee_min_angle)

        terminated = (terminated_lanes | left_toe_on_ground | left_knee_extended).bool()
        return terminated.detach()
