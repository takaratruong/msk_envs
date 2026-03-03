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
            angle_tolerance=90.0,  # we're more lenient since hip swaying is fine
        )
        self.weight = torch.abs(self.total_mass * self.gravity)
        collider_id_lookup = self.collider_id_lookup
        self.left_foot_contact_ids = [cid for name, cid in collider_id_lookup.items() if name.startswith("left_foot_")]
        self.right_foot_contact_ids = [cid for name, cid in collider_id_lookup.items() if name.startswith("right_foot_")]

        self.left_toes_id = self.lookup_body_id("toes_l")
        self.right_toes_id = self.lookup_body_id("toes_r")

        self.left_knee_qpos_idx = self.dof_id_lookup["knee_angle_l"][0]
        self.right_knee_qpos_idx = self.dof_id_lookup["knee_angle_r"][0]

        self.adv_foot_in_contact = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        super()._upon_reset_post_sim(reset_mask)
        self.adv_foot_in_contact[reset_mask] = True
        return

    # def _get_obs(self) -> torch.Tensor:
    #     # append whether the advancing foot is in contact with the ground to the observation
    #     obs = torch.cat([super()._get_obs(), self.adv_foot_in_contact.unsqueeze(1).float()], dim=1)
    #     return obs.detach().clone()

    def _get_terminated(self):
        # Race walking has two notable rules:
        #  1. One foot must always be on the ground
        #  2. The advancing leg must be straightened at the moment of first contact

        # Leaving lanes/falling
        terminated_lanes = super()._get_terminated().bool()

        # Foot on ground. we add a threshold to ensure the contact can be "judged by eye"
        grf_vert = self.grf[:, UP_IDX]
        not_touching_ground = ~(grf_vert > self.weight * 0.15)

        # Advancing leg, if touching ground, must be straight. first find the advanced foot
        left_toes_pos = self.body_positions[:, self.left_toes_id]
        right_toes_pos = self.body_positions[:, self.right_toes_id]
        left_forward = left_toes_pos[:, FWD_IDX] > right_toes_pos[:, FWD_IDX]
        right_forward = ~left_forward
        # now check if the advanced foot is making contact with the ground
        threshold = self.weight * 0.15
        left_foot_collider_forces = torch.sum(self.collider_forces[:, self.left_foot_contact_ids], dim=1)
        right_foot_collider_forces = torch.sum(self.collider_forces[:, self.right_foot_contact_ids], dim=1)
        adv_foot_in_contact = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        adv_foot_in_contact[left_forward] = left_foot_collider_forces[left_forward, UP_IDX] > threshold
        adv_foot_in_contact[right_forward] = right_foot_collider_forces[right_forward, UP_IDX] > threshold
        # first point of contact: check if advanced foot *previously* was not in contact but now is
        adv_foot_first_contact = adv_foot_in_contact & ~self.adv_foot_in_contact
        self.adv_foot_in_contact[:] = adv_foot_in_contact
        # if the advancing foot is not in contact with the ground, it's ok
        # if the advancing foot is in contact with the ground, check if the leg is straight enough
        left_knee_angle = self.joint_positions[:, self.left_knee_qpos_idx]
        right_knee_angle = self.joint_positions[:, self.right_knee_qpos_idx]
        left_knee_straight = torch.abs(left_knee_angle) < torch.deg2rad(torch.tensor(25.0, device=self.device))
        right_knee_straight = torch.abs(right_knee_angle) < torch.deg2rad(torch.tensor(25.0, device=self.device))
        advancing_leg_straight = (left_forward & left_knee_straight) | (right_forward & right_knee_straight)
        advancing_leg_ok = ~adv_foot_first_contact | (adv_foot_first_contact & advancing_leg_straight)

        terminated = (terminated_lanes | not_touching_ground | ~advancing_leg_ok).bool()
        return terminated.detach()
