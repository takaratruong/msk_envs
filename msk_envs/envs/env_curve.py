import torch

from msk_envs.utils.global_params import SIDE_IDX, FWD_IDX, build_axis
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import joint_penalty, has_fallen
from .env_base import MSKEnv
from .env_config import EnvConfig
from ..utils.scene_settings import SceneSettings


class CurvedTrackEnv(MSKEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
            angle_tolerance: float = 30.0,
            lane_index: int = 0  # 0 for Lane 1
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph
        )

        # Olympic Track Specs
        self.lane_width = 1.22
        self.inner_radius = 36.5

        # Target radius is the center of the chosen lane
        self.target_radius = (self.inner_radius + (lane_index * self.lane_width) + (self.lane_width / 2))

        # Center of the curve circle in world space, agent starts at (x=0, z=0) facing +x
        #   so the circle center must be at z=+R
        # so that the initial tangent direction is +x
        self.curve_center = torch.zeros((1, 3), device=self.device)
        self.curve_center[:, SIDE_IDX] = self.target_radius

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device, dtype=torch.float32).unsqueeze(0)
        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(angle_tolerance, device=self.device)))
        self.max_distance_reached = 0.0
        return

    def _get_obs(self) -> torch.Tensor:
        obs = torch.cat([
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _get_curved_metrics(self):
        """Calculates relative position and target direction for a curve"""
        # Vector from track center to agent
        rel_pos = self.root_pos - self.curve_center
        pos_2d = rel_pos[:, [FWD_IDX, SIDE_IDX]]  # ignore height
        dist_from_center = torch.norm(pos_2d, dim=1, keepdim=True)
        radial_vec = pos_2d / (dist_from_center + 1e-6)  # Pointing from center to agent

        # Tangent of track:
        target_dir = torch.stack([-radial_vec[:, 1], radial_vec[:, 0]], dim=1)

        lane_deviation = (dist_from_center.squeeze(1) - self.target_radius)

        return target_dir, lane_deviation

    def _compute_raw_reward_dict(self):
        target_dir, lane_deviation = self._get_curved_metrics()

        # Velocity Reward: project actual 2D velocity onto the tangent target
        #  oops, +3 since needs to be linear, need to make that easier
        root_vel_2d = self.body_velocities[:, self.root_id, [FWD_IDX + 3, SIDE_IDX + 3]]
        vel_along_curve = torch.sum(root_vel_2d * target_dir, dim=1)
        rew_vel = vel_along_curve

        rew_mid_lane = torch.exp(-2.5 * lane_deviation.pow(2))
        rew_limit = joint_penalty(self.ufrc_limit, squared=False)
        self._get_curve_progress()

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_limit": rew_limit.detach(),
        }

    def _is_body_facing_direction(self, body_id):
        target_dir, _ = self._get_curved_metrics()
        body_rot = self.body_rotations[:, body_id]
        body_fwd = rotate_vec(body_rot, self.fwd_axis)
        body_fwd_2d = body_fwd[:, [FWD_IDX, SIDE_IDX]]
        body_fwd_2d = body_fwd_2d / (torch.norm(body_fwd_2d, dim=1, keepdim=True) + 1e-6)
        facing_dot = torch.sum(body_fwd_2d * target_dir, dim=1)
        return facing_dot >= self.cos_angle_threshold

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)
        # Termination if toes leave the lane width
        _, lane_deviation = self._get_curved_metrics()

        # Lane width is 1.22m, so 0.61m from center is the limit
        out_of_lane = (torch.abs(lane_deviation) > (self.lane_width / 2))

        not_facing = ~self._is_body_facing_direction(self.root_id)
        return (fallen | out_of_lane | not_facing).float().detach()

    def _get_curve_progress(self):
        rel_pos = self.root_pos - self.curve_center
        pos_2d = rel_pos[:, [FWD_IDX, SIDE_IDX]]

        # Angle in XZ plane
        angle = torch.atan2(pos_2d[:, 1], pos_2d[:, 0])  # z, x

        # Reference angle at start position:
        # start is (x=0, z=0), center is (0, +R)
        # so initial vector is (0, -R) -> angle = -pi/2
        start_angle = -torch.pi / 2
        angle_progress = angle - start_angle
        angle_progress = (angle_progress + torch.pi) % (2 * torch.pi) - torch.pi
        # Convert to distance along arc: s = R * theta
        arc_length = self.target_radius * angle_progress
        return arc_length

    def update_metrics(self) -> None:
        # Update max distance reached
        max_forward = self._get_curve_progress().max()
        self.max_distance_reached = max(max_forward, self.max_distance_reached)
        return

    def additional_metrics(self) -> dict:
        return {
            "max_distance_reached": self.max_distance_reached,
        }

    def scene_settings(self) -> SceneSettings:
        return SceneSettings(
            lanes=False,
            curve=True,
            axes=False,
        )
