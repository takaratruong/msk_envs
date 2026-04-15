import torch

from msk_envs.utils.reward_lib import joint_penalty, \
    actuator_sq_penalty, metabolic_penalty, fatigue_penalty, has_fallen, alive_bonus, root_zero_reward, \
    match_start_pos_reward
from .env_base import MSKEnv
from .env_config import EnvConfig
from ..utils.quat import quat_mul, quat_conjugate


class DontFallEnv(MSKEnv):
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
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
        """
        # Grab bodies that aren't root or ground
        bodies_mask = torch.ones(self.num_bodies, dtype=torch.bool, device=self.device)
        bodies_mask[[self.ground_id, self.root_id]] = False
        body_positions = self.body_positions[:, bodies_mask, :]
        body_rotations = self.body_rotations[:, bodies_mask, :]

        # Relative to root
        relative_body_positions = body_positions - self.root_pos.unsqueeze(1)
        root_rotation_inv = quat_conjugate(self.root_rot)
        relative_body_rotations = quat_mul(root_rotation_inv.unsqueeze(1), body_rotations)

        time_curr = self.time.view(self.num_worlds, 1) / self.max_episode_duration
        obs = torch.cat([
            time_curr,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            relative_body_positions.reshape(self.num_worlds, -1),
            relative_body_rotations.reshape(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_alive = alive_bonus(self._get_terminated())
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        rew_fatigue = fatigue_penalty(self.muscle_activations, self.num_muscles)
        rew_metabolic = metabolic_penalty(self.muscle_powers, self.num_muscles)
        rew_root_zero = root_zero_reward(self.root_pos, weight=5.0)
        rew_match_start = match_start_pos_reward(self.joint_positions, self.start_pose, weight=1.0, ignore_root=True)

        # Joint passive penalty
        squared_penalties = False
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)
        rew_muscle_passive = joint_penalty(self.ufrc_muscle_passive, squared=squared_penalties)

        self.reward_dict = {
            "rew_alive": rew_alive,
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_fatigue": rew_fatigue.detach(),
            "rew_metabolic": rew_metabolic.detach(),
            "rew_root_zero": rew_root_zero.detach(),
            "rew_match_start": rew_match_start.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(self.root_pos, self.head_pos, self.head_rot, self.head_offset)
        terminated = fallen.float()
        return terminated.detach()
