import torch

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class CariocaEnv(LanesEnv):
    """
    Carioca / grapevine drill environment.

    The agent must:
    - Face sideways
    - Move laterally down the lane
    - Alternate crossing the trailing leg
      in FRONT and BEHIND the lead leg
    - Avoid staying in the same crossing state too long
    """

    # Crossing states
    CROSS_FRONT = 1
    CROSS_BEHIND = -1

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
            target_dir=build_axis(SIDE_IDX, 1.0),
            angle_tolerance=30.0,
        )

        # Current desired crossing mode
        # (+1, -1) -> right foot should cross in (front, behind)
        self.cross_state = torch.ones(self.num_worlds, dtype=torch.long, device=self.device)

        # Counts how long we've stayed in same phase
        self.cross_timer = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

        # Rules
        self.max_phase_steps = 30
        self.cross_threshold = 0.05
        return

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self.cross_state[reset_mask] = 1
        self.cross_timer[reset_mask] = 0

    def _get_cross_delta(self):
        """
        Positive delta: right foot is in FRONT of left foot relative to facing direction (z)
        Negative delta: right foot is BEHIND left foot
        """
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        right_toe_pos = self.body_positions[:, self.toes_ids[1], :]
        left_toe_side = left_toe_pos[:, SIDE_IDX]
        right_toe_side = right_toe_pos[:, SIDE_IDX]
        return right_toe_side - left_toe_side

    def _update_carioca_state(self):
        """ Detect successful crossing transitions and alternate desired crossing direction. """
        delta = self._get_cross_delta()

        front_crossed = delta > self.cross_threshold
        behind_crossed = delta < -self.cross_threshold

        completed_front = ((self.cross_state == self.CROSS_FRONT) & front_crossed)
        completed_behind = ((self.cross_state == self.CROSS_BEHIND) & behind_crossed)
        completed = completed_front | completed_behind

        # Flip desired crossing direction
        self.cross_state = torch.where(completed, -self.cross_state, self.cross_state)
        # Reset timer on successful transition
        self.cross_timer = torch.where(completed, torch.zeros_like(self.cross_timer), self.cross_timer + 1)
        return

    def _get_obs(self) -> torch.Tensor:
        lanes_obs = super()._get_obs()
        obs = torch.cat([
            lanes_obs,
            self.cross_state.view(self.num_worlds, -1),
            self.cross_timer.view(self.num_worlds, -1),
        ], dim=1)
        return obs.detach().clone()

    def _get_terminated(self):
        # Base lane terminations
        terminated_lanes = super()._get_terminated().bool()

        delta = self._get_cross_delta()
        front_crossed = delta > self.cross_threshold
        behind_crossed = delta < -self.cross_threshold

        expected_front = (self.cross_state == self.CROSS_FRONT)
        expected_behind = (self.cross_state == self.CROSS_BEHIND)

        # Wrong crossing direction
        wrong_cross = ((expected_front & behind_crossed) | (expected_behind & front_crossed))
        # Failed to alternate for too long
        stuck_phase = (self.cross_timer >= self.max_phase_steps)

        terminated = (terminated_lanes | wrong_cross | stuck_phase).bool()
        return terminated.detach()
