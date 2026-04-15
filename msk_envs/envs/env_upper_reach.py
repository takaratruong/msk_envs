import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis, SIDE_IDX
from msk_envs.utils.quat import rotate_vec
from msk_envs.utils.reward_lib import joint_penalty, actuator_sq_penalty, has_fallen, alive_bonus
from .env_base import MSKEnv
from .env_config import EnvConfig


class UpperReachTargetEnv(MSKEnv):
    """ Represents an environment where the agent must reach several targets in sequence """

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

        self.right_hand_id = self.lookup_body_id("hand_r")
        self.right_hand_pos = self.body_positions[:, self.right_hand_id]

        self.curr_target_pos = torch.zeros((self.num_worlds, 3), device=self.device)
        self.next_target_pos = torch.zeros((self.num_worlds, 3), device=self.device)
        self.curr_closest_dist = torch.zeros(self.num_worlds, device=self.device)
        self.target_tolerance = 0.15
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Normalized time
         2. Muscle activations, fiber lengths
         3. Actuator activations
         4. Joint positions (q)
         5. Joint velocities (qv)
         6. Current target position
         7. Next target position
         8. Current closest distance to target reached
        """
        time_curr = self.time.view(self.num_worlds, 1) / self.max_episode_duration
        obs = torch.cat([
            time_curr,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions,
            self.joint_velocities,
            self.curr_target_pos.view(self.num_worlds, -1),
            self.next_target_pos.view(self.num_worlds, -1),
            self.curr_closest_dist.view(self.num_worlds, 1),
        ], dim=1)
        return obs.detach().clone()

    def _distance_to_target(self):
        to_target_vec = self.curr_target_pos - self.right_hand_pos
        dist_to_target = torch.norm(to_target_vec, dim=1)
        return dist_to_target

    def _sample_target_around(self, points: torch.Tensor) -> torch.Tensor:
        rand_dirs = torch.randn((points.shape[0], 3), device=points.device)
        rand_dirs = rand_dirs / torch.norm(rand_dirs, dim=1, keepdim=True)
        points_height_one = points.clone()
        points_height_one[:, UP_IDX] = 1.15
        new_targets = points_height_one + rand_dirs * 0.25
        return new_targets

    def _new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        if new_env:
            # If new env, sample two random targets around the root position
            self.curr_target_pos[reset_mask] = self._sample_target_around(self.root_pos[reset_mask])
            self.next_target_pos[reset_mask] = self._sample_target_around(self.root_pos[reset_mask])
        else:
            # Otherwise, current target gets the next target
            self.curr_target_pos[reset_mask] = self.next_target_pos[reset_mask]
            self.next_target_pos[reset_mask] = self._sample_target_around(self.root_pos[reset_mask])
        return

    def compute_new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        self._new_targets(reset_mask, new_env=new_env)
        self.curr_closest_dist[reset_mask] = self._distance_to_target()[reset_mask]
        return

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        super()._upon_reset_post_sim(reset_mask)
        self.compute_new_targets(reset_mask, new_env=True)
        return

    def _compute_raw_reward_dict(self):
        # Reward for getting closer to target than ever before
        dist_to_target = self._distance_to_target()
        rew_target_closer = torch.clamp(self.curr_closest_dist - dist_to_target, min=0.0)
        self.curr_closest_dist = torch.min(self.curr_closest_dist, dist_to_target)

        # Reset target positions for envs that have reached the target
        reached_target_mask = dist_to_target < self.target_tolerance
        if reached_target_mask.any():
            self.compute_new_targets(reached_target_mask, new_env=False)

        rew_limit = joint_penalty(self.get_joint_passive_torques(), squared=True)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)

        terminated = self._get_terminated()
        rew_alive = alive_bonus(terminated)

        self.reward_dict = {
            "rew_target": rew_target_closer.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
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
        return terminated.detach()
