import torch

from msk_envs.utils.global_params import UP_IDX, SIDE_IDX, FWD_IDX, build_axis
from .env_base import MSKEnv
from .env_config import EnvConfig
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import velocity_reward, joint_penalty, ignore_toe_joint_penalty, \
    actuator_sq_penalty, head_acc_penalty, activation_square_penalty, mid_lane_reward, has_fallen, \
    self_collision_penalty, muscle_activation_penalty
from ..utils.scene_settings import SceneSettings
from ..utils.quat import quat_mul, quat_conjugate


class LanesEnv(MSKEnv):
    """ Represents an environment where the agent must face a specific direction and stay within lanes """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
            target_dir: list[float],
            angle_tolerance: float = 30.0,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            live_render=live_render,
            cuda_graph=cuda_graph
        )
        assert target_dir[UP_IDX] == 0.0, "Target direction should be horizontal only"

        # Precompute 2d target facing direction
        self.target_facing = torch.tensor(target_dir, device=self.device).unsqueeze(0)
        self.target_facing = self.target_facing[:, [FWD_IDX, SIDE_IDX]]
        self.target_facing = self.target_facing / torch.norm(self.target_facing, dim=1, keepdim=True)

        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)
        self.toes_ids = [self.lookup_body_id("toes_l"), self.lookup_body_id("toes_r")]
        self.toes_dof_ids = [
            self.lookup_dof_id("mtp_angle_l"),
            self.lookup_dof_id("mtp_angle_r"),
            self.lookup_dof_id("pro_sup_l"),
            self.lookup_dof_id("pro_sup_r"),
        ]
        assert -1 not in self.toes_ids, "Toes body IDs not found in model"
        self.toes_dof_ids = [dof_id for dof_id in self.toes_dof_ids if dof_id != -1]

        self.cos_angle_threshold = torch.cos(torch.deg2rad(torch.tensor(angle_tolerance, device=self.device)))
        self.max_distance_reached = 0.0
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

        # Exclude the x position
        root_x_qpos_id = self.qpos_id_lookup["pelvis_tx"]
        joint_positions_without_x = torch.cat((
            self.joint_positions[:, :root_x_qpos_id],
            self.joint_positions[:, root_x_qpos_id + 1:]),
            dim=1)

        time_curr = self.time.view(self.num_worlds, 1) / self.max_episode_duration
        obs = torch.cat([
            time_curr,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            joint_positions_without_x,
            self.joint_velocities,
            relative_body_positions.reshape(self.num_worlds, -1),
            relative_body_rotations.reshape(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _compute_raw_reward_dict(self):
        rew_vel = velocity_reward(self.body_velocities, self.root_id, FWD_IDX, linear=True)
        rew_mid_lane = mid_lane_reward(self.root_pos)

        # Joint passive penalty
        squared_penalties = False
        # rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        # rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        # rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)
        # rew_muscle_passive = joint_penalty(self.ufrc_muscle_passive, squared=squared_penalties)
        rew_spring = ignore_toe_joint_penalty(self.ufrc_spring, self.toes_dof_ids, squared=squared_penalties)
        rew_damper = ignore_toe_joint_penalty(self.ufrc_damper, self.toes_dof_ids, squared=squared_penalties)
        rew_limit = ignore_toe_joint_penalty(self.ufrc_limit, self.toes_dof_ids, squared=squared_penalties)
        rew_muscle_passive = ignore_toe_joint_penalty(self.ufrc_muscle_passive, self.toes_dof_ids,
                                                      squared=squared_penalties)
        rew_muscle_activation = muscle_activation_penalty(self.muscle_activations)

        rew_actuator = actuator_sq_penalty(self.actuator_activations)
        rew_activation_squared = activation_square_penalty(self.muscle_activations)
        rew_self_collision = self_collision_penalty(self.body_self_collision_forces, 500.0)
        rew_head_acc_ang, rew_head_acc_lin = head_acc_penalty(self.body_accelerations[:, self.head_id])

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_fatigue": rew_activation_squared.detach(),
            "rew_muscle_activation": rew_muscle_activation.detach(),
            "rew_self_collision": rew_self_collision.detach(),
            "rew_head_acc_ang": rew_head_acc_ang.detach(),
            "rew_head_acc_lin": rew_head_acc_lin.detach(),
            # "rew_metabolic": rew_metabolic.detach(),
        }

    def _get_facing_direction(self, body_id):
        body_rot = self.body_rotations[:, body_id]
        body_fwd = rotate_vec(body_rot, self.fwd_axis)
        body_fwd = body_fwd[:, [FWD_IDX, SIDE_IDX]]  # Only care about x/z components
        body_fwd = body_fwd / torch.norm(body_fwd, dim=1, keepdim=True)
        body_fwd_dot_target = torch.sum(body_fwd * self.target_facing, dim=1)
        facing_direction = body_fwd_dot_target >= self.cos_angle_threshold
        return facing_direction.detach()

    def _get_terminated(self):
        # Has fallen
        fallen = has_fallen(self.root_pos, self.head_pos, self.head_rot, self.head_offset)

        # Any of the bodies are out of the lanes
        # body_ids = [i for i in range(self.num_bodies) if i != self.ground_id]
        # body_out = torch.zeros_like(fallen, dtype=torch.bool)
        # for body_idx in body_ids:
        #     body_pos = self.body_positions[:, body_idx]
        #     body_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)
        toes_out = torch.zeros_like(fallen, dtype=torch.bool)
        for body_idx in self.toes_ids:
            body_pos = self.body_positions[:, body_idx]
            toes_out |= (torch.abs(body_pos[:, SIDE_IDX]) > 0.6)

        # Pelvis or head no longer facing forward (within N degrees)
        pelvis_facing_direction = self._get_facing_direction(self.root_id)
        # head_facing_direction = self._get_facing_direction(self.head_id)
        # not_facing_direction = ~(pelvis_facing_direction & head_facing_direction)
        not_facing_direction = ~pelvis_facing_direction

        terminated = (fallen | toes_out | not_facing_direction).float()
        return terminated.detach()

    def scene_settings(self) -> SceneSettings:
        return SceneSettings(
            lanes=True,
            lane_width=0.6,
            meter_markers=True,
            axes=False
        )

    def update_metrics(self) -> None:
        # Update max distance reached
        max_forward = self.root_pos[:, FWD_IDX].max()
        self.max_distance_reached = max(max_forward, self.max_distance_reached)
        return

    def additional_metrics(self) -> dict:
        return {
            "max_distance_reached": self.max_distance_reached,
        }
