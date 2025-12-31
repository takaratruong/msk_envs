import torch
import os
import msk_warp

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec, quat_diff_angle, quat_diff, quat_to_angle_axis, quat_conjugate
from msk_envs.utils.parse_mot import parse_mot
from msk_envs.utils.pose import get_base_name
from msk_envs.utils.reward_lib import joint_angle_track_reward, body_pos_track_reward, body_rot_track_reward, update_dict


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

        # Imitation weights
        self.imitation_weights = env_config.imitation_weights
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
        self.reward_dict = {}

        # Joint angle errors
        for joint, (qpos_adr, dof_adr) in self.dof_id_lookup.items():
            # First 7 are root, skip (handled in body reward)
            if qpos_adr < 7:
                continue
            weight_key = f"imitation_weight_{get_base_name(joint)}"
            rew_key = f"rew_track_{get_base_name(joint)}"
            track_angle_reward = joint_angle_track_reward(
                self.joint_positions, self.curr_target_angles, qpos_adr,
                weight=self.imitation_weights[weight_key])
            update_dict(self.reward_dict, rew_key, track_angle_reward)

        # Global body position and rotation errors
        for body, body_id in self.body_id_lookup.items():
            if body == "ground":
                continue
            weight_pos_key = f"imitation_weight_{get_base_name(body)}_pos"
            weight_rot_key = f"imitation_weight_{get_base_name(body)}_rot"
            rew_pos_key = f"rew_track_{get_base_name(body)}_pos"
            rew_rot_key = f"rew_track_{get_base_name(body)}_rot"

            track_pos_reward = body_pos_track_reward(
                self.body_positions, self.curr_target_bp, body_id,
                weight=self.imitation_weights[weight_pos_key])
            track_rot_reward = body_rot_track_reward(
                self.body_rotations, self.curr_target_br, body_id,
                weight=self.imitation_weights[weight_rot_key])
            update_dict(self.reward_dict, rew_pos_key, track_pos_reward)
            update_dict(self.reward_dict, rew_rot_key, track_rot_reward)

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
