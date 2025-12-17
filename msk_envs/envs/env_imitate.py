import torch
import os

from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec, quat_diff_angle, quat_diff, quat_to_angle_axis
from msk_envs.utils.global_params import UP_IDX


class ImitateEnv(MSKEnv):
    def __init__(self,
                 num_envs: int,
                 env_config: EnvConfig,
                 device: torch.device,
                 render: bool,
                 cuda_graph: bool):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, render=render,
                         cuda_graph=cuda_graph)
        device = self.joint_positions.device

        # Load reference motion
        curr_path = os.path.abspath(os.path.dirname(__file__))
        motion_dir_path = os.path.join(curr_path, "..", "motions")
        motion_file = f"{env_config.motion_name}.pt"
        ref_motion = torch.load(
            os.path.join(motion_dir_path, motion_file),
            map_location=device,
        )

        # # pre-computed body positions
        # self.ref_body_positions = torch.load(
        #     os.path.join(motion_dir_path, "reference_stride_bp.pt"),
        #     map_location=device
        # )

        # Extract time and frames, store
        ref_time, ref_frames = ref_motion[0, :], ref_motion[1:, :]
        # increase z values a bit
        ref_frames[2, :] += 0.01
        n_joints, n_frames = ref_frames.shape
        self.ref_time = torch.tensor(ref_time, device=device)
        self.ref_frames = torch.tensor(ref_frames, device=device)
        self.max_time = self.ref_time[-1].item()
        self.n_frames = n_frames

        assert (n_joints == self.joint_positions.shape[1])
        print(f"Loaded {n_frames} frames, duration {self.max_time:.2f}s")

        # Current time (we'll track this ourselves)
        self.world_times = torch.zeros(self.num_worlds, device=device)

        # Each world will have a random time offset into the motion
        self.time_offset = torch.rand(
            self.num_worlds, device=device) * self.max_time
        self.time_offset[0] = 0.0   # for debugging, first env starts at beginning

        # Target frame (useful if we want to interpolate between frames)
        self.curr_target = torch.zeros(
            (self.num_worlds, n_joints), device=device)
        self.curr_bp_target = torch.zeros_like(self.body_positions)

        self._set_curr_target_frame()
        return

    def _set_curr_target_frame(self):
        # Get the indices of the frames right after the current time
        curr_time = (self.world_times + self.time_offset) % self.max_time
        frame_indices = torch.searchsorted(self.ref_time, curr_time)
        frame_indices = torch.clamp(frame_indices, 0, len(self.ref_time) - 1)
        self.curr_target[:] = self.ref_frames[:, frame_indices].T
        # self.curr_bp_target[:] = self.ref_body_positions[frame_indices, :, :]

        # Finite diff with prev frame to get target velocities:
        prev_frame_indices = torch.clamp(frame_indices - 1, 0,
                                         len(self.ref_time) - 1)
        zero_vel_mask = (frame_indices == 0)  # for frame 0, use *next* frame
        prev_frame_indices[zero_vel_mask] = frame_indices[zero_vel_mask] + 1
        dx = self.ref_frames[:, frame_indices] - self.ref_frames[
            :, prev_frame_indices]
        dt = self.ref_time[frame_indices] - self.ref_time[prev_frame_indices]
        target_velocities = (dx / dt).T

        # Special handling for root quaternion velocity
        root_rot_frame = self.ref_frames[3:7, frame_indices]
        root_rot_prev_frame = self.ref_frames[3:7, prev_frame_indices]
        root_rot_diff_quat = quat_diff(root_rot_frame.T, root_rot_prev_frame.T)
        root_rot_diff_aa = quat_to_angle_axis(root_rot_diff_quat).T
        root_rot_vel = (root_rot_diff_aa / dt).T

        # Update the starting position (in case we reset)
        self.start_pose[:] = self.curr_target.detach().clone()
        self.start_velocity[:, 0:3] = target_velocities[:, 0:3]  # root lin v
        self.start_velocity[:, 3:6] = root_rot_vel               # root ang v
        self.start_velocity[:, 6:] = target_velocities[:, 7:]    # joint qv
        return

    def _upon_reset(self, reset_mask: torch.Tensor):
        # Note: reset_mask currently only includes envs that fell (terminated)
        # A bit hacky, but mark queue sim reset for worlds past the max time
        #  to reset the motion (we don't want RL to see these resets)
        curr_time = self.world_times + self.time_offset
        over_time_mask = (curr_time >= self.max_time)
        self.reset_tensor[:] = (self.reset_tensor[:].bool() | over_time_mask.unsqueeze(1)).to(torch.float32)

        # Now we can reset the world times and time offsets
        reset_mask = reset_mask | over_time_mask
        self.world_times[reset_mask.bool()] = 0.0
        self.time_offset[reset_mask.bool()] = torch.rand(
            reset_mask.sum(), device=self.time_offset.device) * self.max_time
        self.time_offset[0] = 0.0   # for debugging, first env starts at beginning

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

        curr_time = (self.world_times + self.time_offset) % self.max_time

        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            # self.muscle_fiber_velocities,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            rel_body_positions.view(self.num_worlds, -1),
            self.body_rotations.view(self.num_worlds, -1),
            self.body_velocities.view(self.num_worlds, -1),
            curr_time.unsqueeze(1),
            self.curr_target,
            self.curr_bp_target.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        self.world_times += self.delta_t

        self._set_curr_target_frame()

        # TODO: Rewards should be implemented in the reward library

        # Tracking reward: joints (all except root)
        curr_joints = self.joint_positions[:, 7:]
        target_joints = self.curr_target[:, 7:]
        joint_diff = curr_joints - target_joints
        joint_diff_sq = joint_diff ** 2
        rew_track_joints = torch.exp(-0.05 * joint_diff_sq.sum(dim=1))

        # Root position
        curr_root_pos = self.joint_positions[:, 0:3]
        target_root_pos = self.curr_target[:, 0:3]
        root_pos_diff = curr_root_pos - target_root_pos
        root_pos_diff_sq = root_pos_diff ** 2
        rew_track_root_pos = torch.exp(-10 * root_pos_diff_sq.sum(dim=1))

        # Root quaternion, use angle difference
        curr_root_quat = self.joint_positions[:, 3:7]
        target_root_quat = self.curr_target[:, 3:7]
        root_quat_diff_angle = quat_diff_angle(curr_root_quat, target_root_quat)
        root_quat_diff_sq = root_quat_diff_angle ** 2
        rew_track_root_rot = torch.exp(-10 * root_quat_diff_sq)

        # # Global body positions
        # curr_body_pos = self.body_positions
        # target_body_pos = self.curr_bp_target
        # body_pos_diff = curr_body_pos - target_body_pos
        # body_pos_diff_sq = body_pos_diff ** 2
        # # Sum across coordinates, then bodies
        # body_pos_diff_sq_sum = body_pos_diff_sq.sum(dim=2)
        # body_pos_diff_sq_sum = body_pos_diff_sq_sum.sum(dim=1)
        # rew_track_body_pos = torch.exp(-0.1 * body_pos_diff_sq_sum)

        self.reward_dict = {
            "rew_track_joints": rew_track_joints.detach(),
            "rew_track_root_pos": rew_track_root_pos.detach(),
            "rew_track_root_rot": rew_track_root_rot.detach(),
            # "rew_track_body_pos": rew_track_body_pos.detach(),
        }

    def _get_terminated(self):
        # min_ref_root_height = torch.min(self.ref_frames[2, :]).item()
        # max_ref_root_height = torch.max(self.ref_frames[2, :]).item()

        # Root falls below/above threshold
        min_root_height, max_root_height = 0.8, 1.2
        root_idx = self.bodies.index("pelvis")
        root_height = self.body_positions[:, root_idx, UP_IDX]
        fallen = (root_height < min_root_height)
        fallen |= (root_height > max_root_height)

        # Head falls below threshold
        min_head_height = 1.4
        head_pos = self.torso_pos + rotate_vec(self.torso_rot, self.head_offset)
        head_fallen = (head_pos[:, UP_IDX] < min_head_height)

        # Root rot diff too large
        curr_root_quat = self.joint_positions[:, 3:7]
        target_root_quat = self.curr_target[:, 3:7]
        root_quat_diff_angle = quat_diff_angle(curr_root_quat, target_root_quat)
        angle_diff_high = (
                root_quat_diff_angle > torch.deg2rad(torch.tensor(30.0)))

        terminated = (fallen | head_fallen | angle_diff_high).float()
        return terminated.detach()

    def _get_truncated(self):
        # No truncation
        return torch.zeros(self.num_worlds, device=self.joint_positions.device)
