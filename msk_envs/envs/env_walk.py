import torch

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import joint_limit_penalty, \
    actuator_sq_penalty
from msk_envs.utils.global_params import UP_IDX, FWD_IDX, SIDE_IDX

class WalkEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render,
                         cuda_graph=cuda_graph)

        self.prev_head_v = torch.zeros(
            self.num_worlds, 3, device=self.reset_tensor.device)
        self.prev_head_av = torch.zeros(
            self.num_worlds, 3, device=self.reset_tensor.device)
        return

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor):
        # Reset previous velocities
        self.prev_head_v[reset_mask.bool()] = (
            self.body_velocities)[reset_mask.bool(), self.torso_id, 3:]
        self.prev_head_av[reset_mask.bool()] = (
            self.body_velocities)[reset_mask.bool(), self.torso_id, :3]
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths, fiber velocities, actuations
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Body positions relative to root, rotations, velocities
        """
        root_positions = self.body_positions[:, 0, :]
        rel_body_positions = self.body_positions - root_positions.unsqueeze(1)
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.muscle_fiber_velocities,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
            self.body_rotations.view(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        # TODO: Rewards should be implemented in the reward library
        # TODO: Add bindings to muscle_powers then bring back the commented out reward term
        curr_lane_root_velocity = self.body_velocities[:, self.root_id, FWD_IDX + 3]  # +3 because self.body_velocities is {ang_vel, lin_vel} (6 in total)
        curr_lin_velocity = self.body_velocities[:, :, 3:]
        curr_ang_velocity = self.body_velocities[:, :, :3]

        # # Cost of transport: power / velocity
        # mean_muscle_powers = torch.mean(self.muscle_powers, dim=1)
        # mean_muscle_powers_sq = torch.mean(self.muscle_powers ** 2, dim=1)
        # # cost_of_transport = torch.abs(mean_muscle_powers / curr_velocity)
        # cost_of_transport = torch.abs(mean_muscle_powers_sq / curr_velocity)
        # # rew_cot = torch.exp(-cost_of_transport * 3E-2)
        # rew_cot = torch.exp(-cost_of_transport * 5E-4)

        # Head accelerations
        head_offset = torch.tensor([0.0, 0.0, 0.215], device=self.body_positions.device)
        head_offset = head_offset.unsqueeze(0).repeat(self.num_worlds, 1)
        head_global_offset = rotate_vec(self.body_rotations[:, self.torso_id], head_offset)
        head_ang_vel = curr_ang_velocity[:, self.torso_id, :]
        head_lin_vel = curr_lin_velocity[:, self.torso_id, :] + torch.cross(
            head_ang_vel, head_global_offset, dim=1)
        head_lin_acc = (head_lin_vel - self.prev_head_v) / self.delta_t
        head_ang_acc = (head_ang_vel - self.prev_head_av) / self.delta_t
        head_acc_mag_sq = torch.sum(head_lin_acc ** 2, dim=1) + torch.sum(head_ang_acc ** 2, dim=1)
        head_acc_arg = torch.abs(head_acc_mag_sq / curr_lane_root_velocity)
        rew_head = torch.exp(-head_acc_arg * 5E-3)

        # self.prev_head_v[:] = head_lin_vel
        # self.prev_head_av[:] = head_ang_vel

        # Muscle activations (squared)
        mean_activations_sq = torch.mean(self.muscle_activations ** 2, dim=1)
        mean_activations_arg = torch.abs(mean_activations_sq / curr_lane_root_velocity)
        rew_activation = torch.exp(-mean_activations_arg * 1e1)

        # Limits torques
        limit_arg = torch.abs(joint_limit_penalty(self.limit_torques) / curr_lane_root_velocity)
        rew_limit = torch.exp(-limit_arg)

        # Actuator costs
        actuator_arg = torch.abs(actuator_sq_penalty(self.actuator_activations, self.num_actuators) / curr_lane_root_velocity)
        rew_actuator = torch.exp(-actuator_arg)

        # Alive bonus
        terminated = self._get_terminated()
        rew_alive = 1.0 - terminated

        self.reward_dict = {
            # "rew_cot": rew_cot.detach(),
            "rew_head": rew_head.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_activation": rew_activation.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        # Root falls below threshold
        min_root_height = 0.6
        root_height = self.body_positions[:, self.root_id, UP_IDX]
        fallen = (root_height < min_root_height)

        # Head falls below threshold
        min_head_height = 1.0
        torso_pos = self.body_positions[:, self.torso_id]
        torso_rot = self.body_rotations[:, self.torso_id]
        head_offset = torch.tensor([0.0, 0.0, 0.215], device=torso_pos.device)
        head_offset = head_offset.unsqueeze(0).repeat(self.num_worlds, 1)
        head_pos = torso_pos + rotate_vec(torso_rot, head_offset)
        head_fallen = (head_pos[:, UP_IDX] < min_head_height)

        # Any of the bodies are out of the lanes
        body_out = torch.zeros_like(head_fallen)
        for body_idx in range(self.body_positions.shape[1]):
            body_pos = self.body_positions[:, body_idx]
            body_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)

        # Pelvis no longer facing forward (within N degrees)
        pelvis_rot = self.body_rotations[:, self.root_id]
        pelvis_fwd = rotate_vec(pelvis_rot, torch.tensor([1.0, 0.0, 0.0], device=pelvis_rot.device).unsqueeze(0))
        facing_forward = (pelvis_fwd[:, FWD_IDX] > torch.cos(torch.deg2rad(torch.tensor(30.0))))
        not_facing_forward = ~facing_forward

        terminated = (fallen | head_fallen | body_out | not_facing_forward).float()
        return terminated.detach()
