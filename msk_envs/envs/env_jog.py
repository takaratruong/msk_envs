import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis
from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import velocity_reward, joint_limit_penalty, \
    actuator_sq_penalty, metabolic_penalty, fatigue_penalty, acceleration_sq_penalty


class JoggingEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render, cuda_graph=cuda_graph)
        self.prev_joint_velocities = self.joint_velocities.clone()
        # Randomly chosen each episode: scales energy terms
        self.energy_factor = torch.zeros(self.num_worlds, device=self.reset_tensor.device)
        self._upon_reset_post_sim(torch.ones_like(self.energy_factor, dtype=torch.bool))
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor):
        # Reset previous joint velocities
        self.prev_joint_velocities[reset_mask.bool()] = self.joint_velocities[reset_mask.bool()]
        # Sample new energy factors, log-spaced from 1e-6 to 1.0
        n_samples = torch.sum(reset_mask).item()
        random_vals = torch.rand(n_samples, device=self.device)
        new_energy_factors = 10.0 ** (-6.0 + 6.0 * random_vals)
        self.energy_factor[reset_mask.bool()] = new_energy_factors
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths, fiber velocities, actuations
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Body positions relative to root, rotations, velocities
         6. Energy factor
        """
        root_positions = self.body_positions[:, self.root_id, :]
        rel_body_positions = self.body_positions - root_positions.unsqueeze(1)
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.muscle_fiber_velocities,
            self.actuator_activations,
            self.joint_positions[:, 1:],  # exclude x position
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
            self.body_rotations.view(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
            self.energy_factor.view(self.num_worlds, 1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_vel = torch.clamp(rew_vel, min=0.0)
        rew_vel = rew_vel * rew_vel

        joint_accelerations = (self.joint_velocities - self.prev_joint_velocities) / self.delta_t
        self.prev_joint_velocities[:] = self.joint_velocities
        # For the first 0.1 seconds, ignore accelerations to avoid large initial penalties
        init_ignore_mask = (self.time < 0.1).float().unsqueeze(1)
        joint_accelerations = joint_accelerations * (1.0 - init_ignore_mask)

        # "Marathon contribution"
        rew_metabolic = metabolic_penalty(self.muscle_powers, self.num_muscles, squared=True)
        rew_fatigue = fatigue_penalty(self.muscle_activations, self.num_muscles)
        rew_acc = acceleration_sq_penalty(joint_accelerations)
        rew_limit = joint_limit_penalty(self.limit_torques, squared=True)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        # Scale energy terms by energy factor
        rew_metabolic = rew_metabolic * self.energy_factor
        rew_fatigue = rew_fatigue * self.energy_factor
        rew_acc = rew_acc * self.energy_factor
        rew_limit = rew_limit * self.energy_factor
        rew_actuator = rew_actuator * self.energy_factor

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_metabolic": rew_metabolic.detach(),
            "rew_fatigue": rew_fatigue.detach(),
            "rew_acc": rew_acc.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
        }

    def _get_terminated(self):
        # Root falls below threshold
        min_root_height = 0.6
        root_height = self.root_pos[:, UP_IDX]
        fallen = (root_height < min_root_height)

        # Head falls below threshold
        min_head_height = 1.0
        head_pos = self.torso_pos + rotate_vec(self.torso_rot, self.head_offset)
        head_fallen = (head_pos[:, UP_IDX] < min_head_height)

        # Any of the bodies are out of the lanes
        body_out = torch.zeros_like(head_fallen)
        for body_idx in range(self.body_positions.shape[1]):
            body_pos = self.body_positions[:, body_idx]
            body_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)

        # Pelvis no longer facing forward (within N degrees)
        pelvis_rot = self.body_rotations[:, self.root_id]
        x_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device)

        pelvis_fwd = rotate_vec(pelvis_rot, x_axis.unsqueeze(0))
        facing_forward = (pelvis_fwd[:, FWD_IDX] >= torch.cos(torch.deg2rad(torch.tensor(30.0))))
        not_facing_forward = ~facing_forward

        terminated = (fallen | head_fallen | body_out | not_facing_forward).float()
        return terminated.detach()
