import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class HopEnv(LanesEnv):
    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            render: bool,
            cuda_graph: bool
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            render=render,
            cuda_graph=cuda_graph,
            target_dir=build_axis(FWD_IDX, 1.0),
        )
        return

    def _get_obs(self) -> torch.Tensor:
        """
        Observations space:
         1. Muscle activations, fiber lengths, fiber velocities, actuations
         2. Actuator activations
         3. Joint positions (q)
         4. Joint velocities (qv)
         5. Body positions relative to root, rotations, velocities
        """
        obs = torch.cat([
            self.time.view(self.num_worlds, 1),
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions[:, 1:],  # exclude x position
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    def _get_terminated(self):
        # Get normal termination conditions
        terminated_lanes = super()._get_terminated()

        # Check if left toe is too low
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        left_toe_height = left_toe_pos[:, UP_IDX]
        left_toe_on_ground = (left_toe_height < 0.5).float()

        terminated = torch.max(terminated_lanes, left_toe_on_ground)
        return terminated.detach()
