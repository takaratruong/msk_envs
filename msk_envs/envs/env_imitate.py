import torch
import os
import msk_warp

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec, quat_diff_angle, quat_diff, quat_to_angle_axis, quat_conjugate
from msk_envs.utils.global_params import UP_IDX
from msk_envs.utils.parse_mot import parse_mot


class ImitateEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render,
                         cuda_graph=cuda_graph)
        device = self.device

        # Load reference motion
        curr_path = os.path.abspath(os.path.dirname(__file__))
        motion_file = os.path.join(curr_path, f"{env_config.motion_name}.mot")
        data, col_names = parse_mot(motion_file, self.model_path)
        ref_motion = torch.tensor(data, device=device)
        ref_time, ref_frames = ref_motion[0, :], ref_motion[1:, :]

        # Now we can compute the reference body positions
        #  let's also process the visuals while we're at it
        print("Processing reference motion")
        body_positions, body_rotations = [], []
        vis_positions, vis_rotations = [], []
        for i in range(data.shape[1]):
            # Modify world 0, run FK
            self.joint_positions[0, :] = ref_frames[:, i]
            self.fk()
            body_positions.append(self.body_positions[0, :, :].clone())
            body_rotations.append(self.body_rotations[0, :, :].clone())
            vis_positions.append(self.visual_positions[0, :, :].clone())
            vis_rotations.append(self.visual_rotations[0, :, :].clone())
        # [n_frames, n_bodies, 3 or 4]
        self.ref_body_positions = torch.stack(body_positions, dim=0).to(device)
        self.ref_body_rotations = torch.stack(body_rotations, dim=0).to(device)
        # [n_frames, n_visuals, 3 or 4]
        self.ref_vis_positions = torch.stack(vis_positions, dim=0).to(device)
        self.ref_vis_rotations = torch.stack(vis_rotations, dim=0).to(device)

        # Extract time and frames, store
        n_joints, n_frames = ref_frames.shape
        self.ref_time = torch.tensor(ref_time, device=device)
        self.ref_frame_angles = torch.tensor(ref_frames, device=device)
        self.max_time = self.ref_time[-1].item()
        self.n_frames = n_frames

        assert (n_joints == self.joint_positions.shape[1])
        print(f"Loaded {n_frames} frames, duration {self.max_time:.2f}s")

        # Update max episode duration
        self.max_episode_duration = self.max_time

        # Target frame (useful if we want to interpolate between frames)
        self.curr_target_angles = torch.zeros((self.num_worlds, n_joints), device=device)
        self.curr_target_bp = torch.zeros_like(self.body_positions)
        self.curr_target_br = torch.zeros_like(self.body_rotations)

        self._set_curr_target_frame()
        return

    def _set_curr_target_frame(self):
        # Get the indices of the frames right after the current time
        curr_time = self.time
        frame_indices = torch.searchsorted(self.ref_time, curr_time)
        frame_indices = torch.clamp(frame_indices, 0, len(self.ref_time) - 1)
        self.curr_target_angles[:] = self.ref_frame_angles[:, frame_indices].T
        self.curr_target_bp[:] = self.ref_body_positions[frame_indices, :, :]
        self.curr_target_br[:] = self.ref_body_rotations[frame_indices, :, :]

        # Finite diff with prev frame to get target velocities:
        prev_frame_indices = torch.clamp(frame_indices - 1, 0, len(self.ref_time) - 1)
        zero_vel_mask = (frame_indices == 0)  # for frame 0, use *next* frame
        prev_frame_indices[zero_vel_mask] = frame_indices[zero_vel_mask] + 1
        dx = self.ref_frame_angles[:, frame_indices] - self.ref_frame_angles[:, prev_frame_indices]
        dt = self.ref_time[frame_indices] - self.ref_time[prev_frame_indices]
        target_velocities = (dx / dt).T

        # Special handling for root quaternion velocity
        root_rot_frame = self.ref_frame_angles[3:7, frame_indices]
        root_rot_prev_frame = self.ref_frame_angles[3:7, prev_frame_indices]
        root_rot_diff_quat = quat_diff(root_rot_frame.T, root_rot_prev_frame.T)
        root_rot_diff_aa = quat_to_angle_axis(root_rot_diff_quat).T
        root_rot_vel = (root_rot_diff_aa / dt).T
        # Rotate it by the root orientation inverse to get body-local angular velocity
        root_rot_conj = quat_conjugate(root_rot_frame.T)
        root_rot_vel = rotate_vec(root_rot_conj, root_rot_vel)

        # Update the starting position (in case we reset)
        self.start_pose[:] = self.curr_target_angles.detach().clone()
        self.start_velocity[:, 0:3] = target_velocities[:, 0:3]  # root lin v
        self.start_velocity[:, 3:6] = root_rot_vel  # root ang v
        self.start_velocity[:, 6:] = target_velocities[:, 7:]  # joint qv
        return

    def _upon_reset(self, reset_mask: torch.Tensor):
        self._set_curr_target_frame()
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths, fiber velocities, actuations
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Body positions relative to root, rotations, velocities
         6. Current time in reference motion
         7. Target joint positions
        """
        root_positions = self.body_positions[:, 0, :]
        rel_body_positions = self.body_positions - root_positions.unsqueeze(1)

        curr_time = self.time

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
            curr_time.unsqueeze(1),
            self.curr_target_angles,
            self.curr_target_bp.view(self.num_worlds, -1),
            # self.curr_target_br.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        self._set_curr_target_frame()

        # TODO: Rewards should be implemented in the reward library

        # Tracking reward: joints (all except root)
        curr_joints = self.joint_positions[:, 7:]
        target_joints = self.curr_target_angles[:, 7:]
        joint_diff = curr_joints - target_joints
        joint_diff_sq = joint_diff ** 2
        rew_track_joints = torch.exp(-0.15 * joint_diff_sq.sum(dim=1))

        # Root position
        curr_root_pos = self.joint_positions[:, 0:3]
        target_root_pos = self.curr_target_angles[:, 0:3]
        root_pos_diff = curr_root_pos - target_root_pos
        root_pos_diff_sq = root_pos_diff ** 2
        rew_track_root_pos = torch.exp(-10 * root_pos_diff_sq.sum(dim=1))

        # Root quaternion, use angle difference
        curr_root_quat = self.joint_positions[:, 3:7]
        target_root_quat = self.curr_target_angles[:, 3:7]
        root_quat_diff_angle = quat_diff_angle(curr_root_quat, target_root_quat)
        root_quat_diff_sq = root_quat_diff_angle ** 2
        rew_track_root_rot = torch.exp(-10 * root_quat_diff_sq)

        # # Global body positions
        curr_body_pos = self.body_positions
        target_body_pos = self.curr_target_bp
        body_pos_diff = curr_body_pos - target_body_pos
        body_pos_diff_sq = body_pos_diff ** 2
        # Sum across coordinates
        body_pos_diff_sq_sum = body_pos_diff_sq.sum(dim=2)
        # Scale by body weights, then sum across bodies
        body_pos_diff_sq_sum = (self.body_mass / self.total_mass) * body_pos_diff_sq_sum
        body_pos_diff_sq_sum = body_pos_diff_sq_sum.sum(dim=1)
        rew_track_body_pos = torch.exp(-30.0 * body_pos_diff_sq_sum)

        # Global body rotations
        curr_body_rot = self.body_rotations  # [num_worlds, n_bodies, 4]
        target_body_rot = self.curr_target_br  # [num_worlds, n_bodies, 4]
        body_rot_diff_angle = quat_diff_angle(curr_body_rot, target_body_rot)  # [num_worlds, n_bodies]
        body_rot_diff_sq = body_rot_diff_angle ** 2  # [num_worlds, n_bodies]
        # Scale by body weights, then sum across bodies
        body_rot_diff_sq = (self.body_mass / self.total_mass) * body_rot_diff_sq  # [num_worlds, n_bodies]
        body_rot_diff_sq = body_rot_diff_sq.sum(dim=1)
        rew_track_body_rot = torch.exp(-10.0 * body_rot_diff_sq)

        self.reward_dict = {
            "rew_track_joints": rew_track_joints.detach(),
            "rew_track_root_pos": rew_track_root_pos.detach(),
            "rew_track_root_rot": rew_track_root_rot.detach(),
            "rew_track_body_pos": rew_track_body_pos.detach(),
            "rew_track_body_rot": rew_track_body_rot.detach(),
        }

    def _get_terminated(self):
        # Root position difference too high
        curr_root_pos = self.body_positions[:, self.root_id]
        target_root_pos = self.curr_target_bp[:, self.root_id]
        root_diff_high = ((curr_root_pos - target_root_pos).norm(dim=1) > 0.15)

        # Root rotation difference too high
        curr_root_rot = self.body_rotations[:, self.root_id]
        target_root_rot = self.curr_target_br[:, self.root_id]
        root_rot_diff_angle = quat_diff_angle(curr_root_rot, target_root_rot)
        root_rot_diff_high = (root_rot_diff_angle > 0.5)

        terminated = (root_diff_high | root_rot_diff_high).float()
        return terminated.detach()

    def _get_truncated(self):
        # No truncation
        return self.time >= self.max_episode_duration

    def get_reference_visuals(self):
        return self.ref_vis_positions, self.ref_vis_rotations

    def get_reference_times(self):
        return self.ref_time

    def get_reference_joint_angles(self):
        return self.curr_target_angles
