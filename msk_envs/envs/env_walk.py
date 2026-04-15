import torch

from msk_envs.utils.global_params import FWD_IDX, build_axis
from msk_envs.utils.reward_lib import joint_penalty, actuator_sq_penalty, velocity_reward, alive_bonus, \
    cost_of_transport, head_acceleration_penalty
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class WalkEnv(LanesEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            live_render=live_render,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0)
        )

        self.prev_head_v = torch.zeros(self.num_worlds, 3, device=self.reset_tensor.device)
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
        root_positions = self.body_positions[:, self.root_id, :]
        rel_body_positions = self.body_positions - root_positions.unsqueeze(1)
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions[:, 1:],  # exclude x position
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _head_velocities(self):
        torso_ang = self.body_velocities[:, self.torso_id, :3]
        torso_vel = self.body_velocities[:, self.torso_id, 3:]
        head_vel = torso_vel + torch.cross(torso_ang, self.head_offset, dim=1)
        return head_vel

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor):
        self.prev_head_v[reset_mask.bool()] = self._head_velocities()[reset_mask.bool()]
        return

    def _compute_raw_reward_dict(self):
        rew_cot = cost_of_transport(self.muscle_powers, self.body_velocities, self.root_id)
        rew_limit = joint_penalty(self.get_joint_passive_torques(), squared=True)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)

        head_vel = self._head_velocities()
        rew_head = head_acceleration_penalty(head_vel, self.prev_head_v, self.delta_t)
        self.prev_head_v[:] = head_vel[:]

        # Alive/not fallen bonus
        terminated = self._get_terminated()
        rew_alive = alive_bonus(terminated)

        self.reward_dict = {
            "rew_cot": rew_cot.detach(),
            "rew_head": rew_head.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_alive": rew_alive.detach(),
        }
