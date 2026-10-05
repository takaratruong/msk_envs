import torch

from msk_envs.utils.global_params import FWD_IDX, build_axis
from msk_envs.utils.reward_lib import velocity_reward_max, mid_lane_reward, joint_penalty, muscle_passive_penalty
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class WalkEnv(LanesEnv):
    """ Sprint environment, but for walking.

    Structurally identical to SprintingEnv (same LanesEnv machinery: forward
    lane + facing termination + run start). The ONLY difference is the velocity
    reward: sprint uses an unbounded forward-velocity reward (optimum = max
    speed, a speed a hop physically can't reach, so alternating gait is forced),
    whereas walking caps the reward at a target speed and penalizes exceeding it
    (velocity_reward_max). The lane + facing constraints forbid lateral/turning
    degeneracy; the run start seeds the alternating-gait basin.
    """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
        )
        self.target_speed = env_config.walk_target_speed
        return

    def _compute_raw_reward_dict(self):
        # Forward velocity rewarded up to target_speed, penalized above it.
        rew_vel = torch.nan_to_num(
            velocity_reward_max(self.body_velocities, self.root_id, FWD_IDX,
                                linear=True, target_speed=self.target_speed),
            nan=0.0, posinf=0.0, neginf=0.0)
        rew_mid_lane = mid_lane_reward(self.root_pos)

        squared_penalties = False
        rew_muscle_passive = torch.nan_to_num(muscle_passive_penalty(
            self.muscle_passive_length_multiplier, threshold=0.1, squared=squared_penalties),
            nan=0.0, posinf=0.0, neginf=0.0)
        rew_spring = joint_penalty(self.ufrc_spring, squared=squared_penalties)
        rew_damper = joint_penalty(self.ufrc_damper, squared=squared_penalties)
        rew_limit = torch.nan_to_num(joint_penalty(self.ufrc_limit, squared=squared_penalties),
                                     nan=0.0, posinf=0.0, neginf=0.0)

        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_mid_lane": rew_mid_lane.detach(),
            "rew_spring": rew_spring.detach(),
            "rew_damper": rew_damper.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
        }
