import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis, MIN_ROOT_HEIGHT, MIN_HEAD_HEIGHT
from msk_envs.utils.reward_lib import joint_penalty, actuator_sq_penalty, has_fallen, alive_bonus
from .env_base import MSKEnv
from .env_config import EnvConfig


class ReachTargetEnv(MSKEnv):
    """ Represents an environment where the agent must reach several targets in sequence """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            live_render: bool,
            cuda_graph: bool,
            ignore_vertical: bool = True,
            target_tolerance: float = 0.25,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device, live_render=live_render, cuda_graph=cuda_graph)
        self.fwd_axis = torch.tensor(build_axis(FWD_IDX, 1.0), device=self.device).unsqueeze(0)

        self.right_hand_id = self.lookup_body_id("hand_r")
        self.right_hand_pos = self.body_positions[:, self.right_hand_id]

        self.curr_target_pos = torch.zeros((self.num_worlds, 3), device=self.device)
        self.next_target_pos = torch.zeros((self.num_worlds, 3), device=self.device)
        self.curr_closest_dist = torch.zeros(self.num_worlds, device=self.device)
        self.target_tolerance = target_tolerance
        self.ignore_vertical = ignore_vertical
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

    def _new_targets(self, reset_mask: torch.Tensor, new_env: bool) -> None:
        """ Override this method to compute new target positions for the environments indicated by reset_mask """
        raise NotImplementedError

    def _distance_to_target(self):
        to_target_vec = self.curr_target_pos - self.right_hand_pos
        if self.ignore_vertical:  # ignore vertical displacement to target
            to_target_vec[:, UP_IDX] = 0.0
        dist_to_target = torch.norm(to_target_vec, dim=1)
        return dist_to_target

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

        # Reward for facing the target
        # pelvis_rot = self.body_rotations[:, self.root_id]
        # pelvis_fwd = rotate_vec(pelvis_rot, self.fwd_axis)
        # pelvis_fwd = pelvis_fwd[:, [FWD_IDX, SIDE_IDX]]  # Only care about x/z components
        # pelvis_fwd = pelvis_fwd / torch.norm(pelvis_fwd, dim=1, keepdim=True)
        # to_target_vec = self.curr_target_pos - self.root_pos
        # to_target_vec = to_target_vec[:, [FWD_IDX, SIDE_IDX]]
        # to_target_vec = to_target_vec / torch.norm(to_target_vec, dim=1, keepdim=True)
        # facing_target = torch.sum(pelvis_fwd * to_target_vec, dim=1)
        # rew_facing_target = torch.clamp(facing_target, min=0.0)

        # Reset target positions for envs that have reached the target
        reached_target_mask = dist_to_target < self.target_tolerance
        if reached_target_mask.any():
            self.compute_new_targets(reached_target_mask, new_env=False)

        rew_limit = joint_penalty(self.get_joint_passive_torques(), squared=True)
        rew_actuator = actuator_sq_penalty(self.actuator_activations, self.num_actuators)

        terminated = self._get_terminated()
        rew_alive = alive_bonus(terminated)

        self.reward_dict = {
            "rew_target_closer": rew_target_closer.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
            "rew_alive": rew_alive.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(self.root_pos, self.torso_pos, self.torso_rot, self.head_offset,
                            min_root=min(MIN_ROOT_HEIGHT, 0.3), min_head=min(MIN_HEAD_HEIGHT, 0.5))
        terminated = fallen.float()
        return terminated.detach()
