import torch

from msk_envs.utils.global_params import FWD_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class WaddleEnv(LanesEnv):
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
            angle_tolerance=45.0,
        )
        self.right_hip_add_idx = self.dof_id_lookup["hip_adduction_r"][0]
        self.left_hip_add_idx = self.dof_id_lookup["hip_adduction_l"][0]
        self.left_toe_id = self.lookup_body_id("toes_l")
        self.right_toe_id = self.lookup_body_id("toes_r")
        return

    def _get_terminated(self):
        # Leaving lanes/falling
        terminated_lanes = super()._get_terminated().bool()

        # Adduction must be less than 0.0
        right_hip_add = self.joint_positions[:, self.right_hip_add_idx]
        left_hip_add = self.joint_positions[:, self.left_hip_add_idx]
        right_leg_too_far_in = right_hip_add > 0.0
        left_leg_too_far_in = left_hip_add > 0.0
        legs_too_far_in = right_leg_too_far_in | left_leg_too_far_in

        # Toes too close
        l_toes_pos, r_toes_pos = self.body_positions[:, self.left_toe_id], self.body_positions[:, self.right_toe_id]
        toes_dist = torch.norm(l_toes_pos - r_toes_pos, dim=1)
        toes_too_close = toes_dist < 0.25

        terminated = (terminated_lanes | legs_too_far_in | toes_too_close)
        return terminated.detach()
