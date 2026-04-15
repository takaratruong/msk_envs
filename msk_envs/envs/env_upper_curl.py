import torch

from msk_envs.utils.global_params import UP_IDX, build_axis
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import joint_penalty, actuator_sq_penalty, alive_bonus, derivative_sq_penalty, \
    fatigue_penalty, exp_distance
from .env_base import MSKEnv
from .env_config import EnvConfig


class UpperCurlEnv(MSKEnv):
    """ Represents an environment where the agent must curl """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            requires_visuals: bool,
            cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            live_render=live_render,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph
        )
        self.up_axis = torch.tensor(build_axis(UP_IDX, 1.0), device=self.device).unsqueeze(0)
        self.thorax_id = self.lookup_body_id("thorax")

        dof_interest = "elbow_flex_r"
        # dof_interest = "shoulder_flexion_r"
        # dof_interest = "shoulder_rotation_r"
        # dof_interest = "scapula_elevation_r"
        # dof_interest = "thorax_rotation"
        self.dof_interest_id = self.dof_id_lookup[dof_interest]
        self.dof_interest_low, self.dof_interest_high = self.limit_id_lookup[dof_interest]

        # Whether the agent should raise or lower its arm
        self.should_raise = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self.curr_closest_dist = torch.zeros(self.num_worlds, device=self.device)

        # We should match the rest of the joint angles to this
        self.starting_qs = torch.zeros_like(self.joint_positions)

        self.joint_ranges = torch.zeros(self.num_qpos, dtype=torch.float, device=self.device)
        for k, (low, up) in self.limit_id_lookup.items():
            qpos_id = self.qpos_id_lookup[k]
            if qpos_id > -1:
                self.joint_ranges[qpos_id] = up - low

        self.tolerance = 0.1
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Normalized time
         2. Muscle activations, fiber lengths
         3. Actuator activations
         4. Joint positions (q)
         5. Joint velocities (qv)
         6. Current closest distance to target
        """
        time_curr = self.time.view(self.num_worlds, 1) / self.max_episode_duration
        obs = torch.cat([
            time_curr,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            self.curr_closest_dist.view(self.num_worlds, 1),
        ], dim=1)
        return obs.detach().clone()

    def _get_dof_value(self) -> torch.Tensor:
        dof_value = self.joint_positions[:, self.dof_interest_id]
        return dof_value

    def _get_targets(self) -> torch.Tensor:
        # For arms that should raise, try to move closer to upper limit, for lowering try to move closer to lower
        targets = torch.zeros_like(self.should_raise, dtype=torch.float, device=self.device)
        targets[self.should_raise] = self.dof_interest_high
        targets[~self.should_raise] = self.dof_interest_low
        return targets

    def _dist_to_targets(self, dof_value) -> torch.Tensor:
        targets = self._get_targets()
        dof_interest_range = self.dof_interest_high - self.dof_interest_low
        dist_to_target = torch.abs(dof_value - targets) / dof_interest_range
        return dist_to_target

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        super()._upon_reset_post_sim(reset_mask)
        dof_value = self._get_dof_value()
        # Determine which is closer: low or up
        dist_to_high = torch.abs(dof_value - self.dof_interest_high)
        dist_to_low = torch.abs(dof_value - self.dof_interest_low)
        closer_to_low = dist_to_low < dist_to_high
        self.should_raise[reset_mask] = closer_to_low[reset_mask]
        self.curr_closest_dist[reset_mask] = self._dist_to_targets(dof_value)[reset_mask]

        # Set starting positions
        self.starting_qs[reset_mask] = self.joint_positions[reset_mask]
        return

    def _compute_raw_reward_dict(self):
        # Reward for improving over best distance to target
        dof_value = self._get_dof_value()
        dist_to_target = self._dist_to_targets(dof_value)
        rew_target = torch.clamp(self.curr_closest_dist - dist_to_target, min=0.0)
        # Update current best distance
        self.curr_closest_dist = torch.min(self.curr_closest_dist, dist_to_target)
        # If within tolerance of limit, switch from flex to unflex or vice versa
        within_tolerance_mask = dist_to_target < self.tolerance
        self.should_raise[within_tolerance_mask] = ~self.should_raise[within_tolerance_mask]
        self.curr_closest_dist[within_tolerance_mask] = self._dist_to_targets(dof_value)[within_tolerance_mask]

        # Reward for matching the starting position (except dof of interest)
        targets_starting = self.starting_qs.clone()
        targets_starting[:, self.dof_interest_id] = self.joint_positions[:, self.dof_interest_id]
        rew_starting = exp_distance(self.joint_positions, targets_starting, self.joint_ranges, weight=100.0).mean(dim=1)

        # Joint passive penalty
        squared_penalties = False
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = joint_penalty(self.ufrc_limit, squared=squared_penalties)
        rew_muscle_passive = joint_penalty(self.ufrc_muscle_passive, squared=squared_penalties)

        # Actuator penalty, if any
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)
        rew_actuator_dot = derivative_sq_penalty(self.actuator_activations_dot, self.num_actuators)

        terminated = self._get_terminated()
        rew_alive = alive_bonus(terminated)

        self.reward_dict = {
            "rew_target": rew_target.detach(),
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_actuator_dot": rew_actuator_dot.detach(),
            "rew_starting": rew_starting.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        # Head is not upright
        head_rot = self.body_rotations[:, self.head_id]
        head_up = rotate_vec(head_rot, self.up_axis)
        head_up = head_up / torch.norm(head_up, dim=1, keepdim=True)

        head_up_dot_up = torch.sum(head_up * self.up_axis, dim=1)
        head_upright = head_up_dot_up > torch.cos(torch.deg2rad(torch.tensor(45.0, device=self.device)))
        not_head_upright = ~head_upright

        # Thorax is not upright
        thorax_rot = self.body_rotations[:, self.thorax_id]
        thorax_up = rotate_vec(thorax_rot, self.up_axis)
        thorax_up = thorax_up / torch.norm(thorax_up, dim=1, keepdim=True)

        thorax_up_dot_up = torch.sum(thorax_up * self.up_axis, dim=1)
        thorax_upright = thorax_up_dot_up > torch.cos(torch.deg2rad(torch.tensor(45.0, device=self.device)))
        not_thorax_upright = ~thorax_upright

        terminated = not_head_upright | not_thorax_upright
        # return terminated.detach()
        return terminated.detach() * 0
