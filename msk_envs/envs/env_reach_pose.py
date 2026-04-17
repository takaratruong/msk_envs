import os

import torch

from msk_envs.utils.pose import parse_starting_pose
from msk_envs.utils.quat import quat_conjugate, quat_mul
from msk_envs.utils.reward_lib import exp_distance, single_body_pos_track_reward
from .env_base import MSKEnv
from .env_config import EnvConfig


class ReachPoseEnv(MSKEnv):
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
            cuda_graph=cuda_graph
        )

        joint_positions_init = self.joint_positions.clone()

        # Load target pose
        target_pose_path = os.path.join(self.curr_path, env_config.target_pose_path)
        q, _ = parse_starting_pose(
            target_pose_path, self.qpos_id_lookup, self.dof_id_lookup, self.num_qpos, self.num_dofs
        )
        self.ref_joint_positions = torch.tensor(q, dtype=torch.float32, device=device)

        # Compute the target global body positions/rotations
        self.joint_positions[:] = self.ref_joint_positions
        self.fk()
        # body positions, visual reference
        self.ref_body_positions = self.body_positions[0, ...].clone()
        self.ref_vis_positions = self.visual_positions[0, ...].clone()
        self.ref_vis_rotations = self.visual_rotations[0, ...].clone()
        # Restore
        self.joint_positions[:] = joint_positions_init

        # Get joint ranges
        self.joint_ranges = torch.zeros(self.num_qpos, dtype=torch.float, device=self.device)
        for k, (low, up) in self.limit_id_lookup.items():
            qpos_id = self.qpos_id_lookup[k]
            if qpos_id > -1:
                self.joint_ranges[qpos_id] = up - low

        # Reshape
        self.ref_joint_positions = self.ref_joint_positions.repeat(self.num_worlds, 1)
        self.ref_body_positions = self.ref_body_positions.repeat(self.num_worlds, 1, 1)
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Normalized time
         2. Muscle activations, fiber lengths
         3. Actuator activations
         4. Joint positions (q)
         5. Joint velocities (qv)
         6. Relative body positions and rotations (wrst root), ignore ground
         7. Reference joint positions and body positions
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

            self.ref_joint_positions.reshape(self.num_worlds, -1),
            self.ref_body_positions.reshape(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        # Reward for matching the target position
        rew_target = exp_distance(
            self.joint_positions, self.ref_joint_positions, self.joint_ranges, weight=100.0).mean(dim=1)

        # Global positions
        rew_target_global = torch.zeros_like(rew_target)
        bodies_idx = [i for i in range(self.num_bodies) if i != self.ground_id]
        for k in bodies_idx:
            body_position = self.body_positions[:, k]
            target_body_position = self.ref_body_positions[:, k]
            rew_target_global += single_body_pos_track_reward(body_position, target_body_position, weight=10.0)

        self.reward_dict = {
            "rew_target": rew_target.detach(),
            "rew_target_global": rew_target_global.detach(),
        }

    def _get_terminated(self):
        return torch.zeros_like(self._get_truncated())

    def get_reference_visuals(self):
        return self.ref_vis_positions, self.ref_vis_rotations
