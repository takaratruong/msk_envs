import torch

from msk_envs.utils.global_params import FWD_IDX, build_axis
from msk_envs.utils.reward_lib import joint_penalty, actuator_sq_penalty, activation_square_penalty
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class WalkEnv(LanesEnv):
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
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
        )

        self.target_velocity = torch.tensor([1.2, 0.0, 0.0], device=device)
        return

    def _compute_raw_reward_dict(self):
        # Match target velocity
        root_velocity = self.body_velocities[:, self.root_id]
        velocity_diff = root_velocity - self.target_velocity[torch.newaxis, :]
        velocity_diff_sq = torch.pow(velocity_diff, 2)
        rew_vel = torch.exp(-velocity_diff_sq.sum(dim=1))

        # Fatigue
        rew_fatigue = activation_square_penalty(self.muscle_activations)

        # Joint penalties
        rew_limit = joint_penalty(self.ufrc_limit, squared=False)
        rew_actuator = actuator_sq_penalty(self.actuator_activations)
        self.reward_dict = {
            "rew_vel": rew_vel.detach(),
            "rew_fatigue": rew_fatigue.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_actuator": rew_actuator.detach(),
        }
