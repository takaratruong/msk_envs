import torch
import os
import math

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec, quat_diff_angle, quat_diff, quat_to_angle_axis, quat_conjugate, quat_mul
from msk_envs.utils.parse_mot import parse_mot
from msk_envs.utils.reward_lib import joint_angle_track_reward, body_pos_track_reward, body_rot_track_reward, \
    update_dict


class ImitateEnv(MSKEnv):
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
        device = self.device

        # Load reference motion
        curr_path = os.path.abspath(os.path.dirname(__file__))
        motion_file = os.path.join(curr_path, env_config.motion_name)
        motion = parse_mot(
            motion_file=motion_file,
            load_result=self.load_result,
            device=self.device
        )
        ref_time, ref_angles = motion[:, 0], motion[:, 1:]
        n_frames = len(motion)

        # Now we can compute the reference body positions
        #  let's also process the visuals while we're at it
        print("Processing reference motion")
        body_positions, body_rotations = [], []
        vis_positions, vis_rotations = [], []
        for i in range(n_frames):
            # Modify world 0, run FK
            self.joint_positions[0, :] = ref_angles[i, :]
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
        self.ref_time = torch.tensor(ref_time, device=device)
        self.ref_angles = torch.tensor(ref_angles, device=device)
        self.max_time = self.ref_time[-1].item()
        self.n_frames = n_frames

        print(f"Loaded {n_frames} frames, duration {self.max_time:.2f}s")

        # Update max episode duration
        self.max_episode_duration = self.max_time

        # Target frame (useful if we want to interpolate between frames)
        self.curr_target_angles = torch.zeros((self.num_worlds, self.num_qpos), device=device)
        self.curr_target_bp = torch.zeros_like(self.body_positions)
        self.curr_target_br = torch.zeros_like(self.body_rotations)

        self._set_curr_target_frame()

        self.all_but_root_joint_ids = range(7, self.joint_positions.shape[1])
        self.all_but_root_body_ids = list(range(1, self.body_positions.shape[1]))
        self.all_but_root_body_ids.remove(self.root_id)

        self.imitation_weights = env_config.imitation_weights
        self.extra_rewarded_joints = env_config.extra_rewarded_joints
        self.extra_rewarded_dofs = env_config.extra_rewarded_dofs
        return

    def _set_curr_target_frame(self):
        # Get the indices of the frames right after the current time
        curr_time = self.time
        frame_indices = torch.searchsorted(self.ref_time, curr_time)
        frame_indices = torch.clamp(frame_indices, 0, len(self.ref_time) - 1)
        self.curr_target_angles[:] = self.ref_angles[frame_indices]
        self.curr_target_bp[:] = self.ref_body_positions[frame_indices, :, :]
        self.curr_target_br[:] = self.ref_body_rotations[frame_indices, :, :]

        # Finite diff with prev frame to get target velocities:
        prev_frame_indices = torch.clamp(frame_indices - 1, 0, len(self.ref_time) - 1)
        zero_vel_mask = (frame_indices == 0)  # for frame 0, use *next* frame
        prev_frame_indices[zero_vel_mask] = frame_indices[zero_vel_mask] + 1
        dx = self.ref_angles[frame_indices] - self.ref_angles[prev_frame_indices]
        dt = self.ref_time[frame_indices] - self.ref_time[prev_frame_indices]
        target_velocities = dx / dt.unsqueeze(1)

        # Special handling for root quaternion velocity
        # root_rot_frame = self.ref_angles[frame_indices, 0:4]
        # root_rot_prev_frame = self.ref_angles[prev_frame_indices, 0:4]
        # root_rot_diff_quat = quat_diff(root_rot_frame, root_rot_prev_frame)
        # root_rot_diff_aa = quat_to_angle_axis(root_rot_diff_quat)
        # root_rot_vel = (root_rot_diff_aa / dt)

        # Update the starting position (in case we reset)
        self.start_pose[:] = self.curr_target_angles.detach().clone()
        # self.start_velocity[:, 0:3] = root_rot_vel  # root ang v. TODO: fix me
        self.start_velocity[:, 3:6] = target_velocities[:, 4:7]  # root lin v
        self.start_velocity[:, 6:] = target_velocities[:, 7:]  # joint qv
        return

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor):
        # We use the reset-hook before sim reset since we need to modify start pose/vel
        self.time[reset_mask] = 0.0  # Requires time to be reset now
        self._set_curr_target_frame()
        return

    def _get_obs(self) -> torch.Tensor:
        curr_time = self.time
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            self.body_positions.reshape(self.num_worlds, -1),
            self.body_rotations.reshape(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
            curr_time.unsqueeze(1),
            self.curr_target_angles,
            self.curr_target_bp.view(self.num_worlds, -1),
            self.curr_target_br.view(self.num_worlds, -1),
        ], dim=1)

        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        self._set_curr_target_frame()
        self.reward_dict = {}

        # Body positions relative to root
        root_pos, ref_root_pos = self.body_positions[:, self.root_id, :], self.curr_target_bp[:, self.root_id, :]
        rel_body_positions = self.body_positions - root_pos.unsqueeze(1)
        rel_ref_body_positions = self.curr_target_bp - ref_root_pos.unsqueeze(1)
        # Body rotations relative to root
        root_rot, ref_root_rot = self.body_rotations[:, self.root_id, :], self.curr_target_br[:, self.root_id, :]
        rel_body_rotations = quat_mul(quat_conjugate(root_rot).unsqueeze(1), self.body_rotations)
        rel_ref_body_rotations = quat_mul(quat_conjugate(ref_root_rot).unsqueeze(1), self.curr_target_br)

        # Tracking reward: generalized joint angles (all except root joint values)
        track_angle_reward = joint_angle_track_reward(
            self.joint_positions, self.curr_target_angles, qpos_adr=self.all_but_root_joint_ids,
            weight=self.imitation_weights["imitation_weight_track_joints"])
        update_dict(self.reward_dict, "rew_track_joints", track_angle_reward)

        # Global root position
        rew_track_root_pos = body_pos_track_reward(
            self.body_positions, self.curr_target_bp, self.root_id,
            weight=self.imitation_weights["imitation_weight_track_root_pos"])
        update_dict(self.reward_dict, "rew_track_root_pos", rew_track_root_pos)

        # Global root rotation
        rew_track_root_rot = body_rot_track_reward(
            self.body_rotations, self.curr_target_br, self.root_id,
            weight=self.imitation_weights["imitation_weight_track_root_rot"])
        update_dict(self.reward_dict, "rew_track_root_rot", rew_track_root_rot)

        # Relative body positions
        rew_track_body_pos = body_pos_track_reward(
            rel_body_positions, rel_ref_body_positions, self.all_but_root_body_ids,
            weight=self.imitation_weights["imitation_weight_track_body_pos"])
        update_dict(self.reward_dict, "rew_track_body_pos", rew_track_body_pos)

        # Relative body rotations
        rew_track_body_rot = body_rot_track_reward(
            rel_body_rotations, rel_ref_body_rotations, self.all_but_root_body_ids,
            weight=self.imitation_weights["imitation_weight_track_body_rot"])
        update_dict(self.reward_dict, "rew_track_body_rot", rew_track_body_rot)

        # Extra rewarded DOFs - for debug!
        if len(self.extra_rewarded_dofs) > 0:
            assert self.reward_lambdas[
                       "lambda_extra_rewarded_dofs"] > 0.0, "lambda_extra_rewarded_dofs must be set when extra_rewarded_dofs are set"
            extra_rewarded_dofs_reward = 0.0
            for dof in self.extra_rewarded_dofs:
                dof_adr = self.dof_id_lookup[dof][0]  # qpos_adr
                track_angle_reward = joint_angle_track_reward(
                    self.joint_positions, self.curr_target_angles, dof_adr,
                    weight=self.imitation_weights["imitation_weight_track_joints"])
                extra_rewarded_dofs_reward += track_angle_reward
            update_dict(self.reward_dict, "rew_extra_rewarded_dofs", extra_rewarded_dofs_reward)

        # Extra rewarded joints - for debug!
        if len(self.extra_rewarded_joints) > 0:
            assert self.reward_lambdas[
                       "lambda_extra_rewarded_joints"] > 0.0, "lambda_extra_rewarded_joints must be set when extra_rewarded_joints are set"
            extra_rewarded_joints_reward = 0.0
            for body in self.extra_rewarded_joints:
                body_id = self.body_id_lookup[body]
                track_pos_reward = body_pos_track_reward(
                    rel_body_positions, rel_ref_body_positions, body_id,
                    weight=self.imitation_weights["imitation_weight_track_body_pos"])
                track_rot_reward = body_rot_track_reward(
                    rel_body_rotations, rel_ref_body_rotations, body_id,
                    weight=self.imitation_weights["imitation_weight_track_body_rot"])
                extra_rewarded_joints_reward += track_pos_reward + track_rot_reward
            update_dict(self.reward_dict, "rew_extra_rewarded_joints", extra_rewarded_joints_reward)

    def _get_terminated(self):
        # Root position difference too high
        curr_root_pos = self.body_positions[:, self.root_id]
        target_root_pos = self.curr_target_bp[:, self.root_id]
        root_diff_high = ((curr_root_pos - target_root_pos).abs().max(dim=1).values > 0.5)

        # Root rotation difference too high
        curr_root_rot = self.body_rotations[:, self.root_id]
        target_root_rot = self.curr_target_br[:, self.root_id]
        root_rot_diff_angle = quat_diff_angle(curr_root_rot, target_root_rot)
        root_rot_diff_high = (root_rot_diff_angle > math.radians(30.0))

        terminated = (root_diff_high | root_rot_diff_high).float()
        return terminated.detach() * 0

    def _get_truncated(self):
        return self.time >= self.max_episode_duration

    def get_reference_visuals(self):
        return self.ref_vis_positions, self.ref_vis_rotations

    def get_reference_times(self):
        return self.ref_time

    def get_reference_joint_angles(self):
        return self.curr_target_angles
