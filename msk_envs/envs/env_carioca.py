import torch

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX, build_axis
from .env_config import EnvConfig
from .env_lanes import LanesEnv


class CariocaEnv(LanesEnv):
    """
    Carioca / grapevine drill environment.
    The agent must move laterally and alternate crossing one foot
    in front of and behind the other, with a deliberate reset
    (feet side-by-side) between each crossing.
    Phase cycle (0 -> 1 -> 2 -> 3 -> 0 -> ...):
      0  CROSS_FRONT   – trailing foot crosses in front
      1  RESET_1       – feet return to side-by-side
      2  CROSS_BEHIND  – trailing foot crosses behind
      3  RESET_2       – feet return to side-by-side again
    """

    CROSS_FRONT = 0
    RESET_1 = 1
    CROSS_BEHIND = 2
    RESET_2 = 3
    NUM_PHASES = 4

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            live_render: bool,
            cuda_graph: bool,
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

        # Current phase index per world (0-3)
        self.cross_phase = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

        # Steps spent in the current phase without completing it
        self.cross_timer = torch.zeros(self.num_worlds, dtype=torch.long, device=self.device)

        # How far the trailing foot must cross past the lead foot (m)
        self.cross_threshold = 0.05
        # How close to side-by-side counts as a reset (m)
        self.reset_threshold = 0.03
        # Steps allowed per phase before termination
        self.max_phase_steps = 30

    def _upon_reset_pre_sim(self, reset_mask: torch.Tensor) -> None:
        self.cross_phase[reset_mask] = 0
        self.cross_timer[reset_mask] = 0

    def _get_cross_delta(self) -> torch.Tensor:
        """
        Signed offset along the direction of travel: right_toe − left_toe.
          > 0  → right foot is ahead of left (crossed in front)
          < 0  → right foot is behind left (crossed behind)
        """
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        right_toe_pos = self.body_positions[:, self.toes_ids[1], :]
        return right_toe_pos[:, FWD_IDX] - left_toe_pos[:, FWD_IDX]

    def _update_carioca_state(self) -> None:
        """
        Advance the phase when the completion condition for the current
        phase is satisfied. The timer resets on every successful transition
        and increments otherwise, driving termination if a phase stalls.
        """
        delta = self._get_cross_delta()

        # Per-phase completion conditions
        phase_done = torch.stack([
            delta > self.cross_threshold,  # 0: CROSS_FRONT  – left past right
            delta.abs() < self.reset_threshold,  # 1: RESET_1      – side by side
            delta < -self.cross_threshold,  # 2: CROSS_BEHIND – right past left
            delta.abs() < self.reset_threshold,  # 3: RESET_2      – side by side
        ], dim=1)  # (num_worlds, 4)

        # Look up whether the current phase's condition is met
        completed = phase_done.gather(
            dim=1,
            index=self.cross_phase.unsqueeze(1)
        ).squeeze(1)  # (num_worlds,)

        # Cycle to the next phase on completion
        self.cross_phase = torch.where(
            completed,
            (self.cross_phase + 1) % self.NUM_PHASES,
            self.cross_phase,
        )
        self.cross_timer = torch.where(
            completed,
            torch.zeros_like(self.cross_timer),
            self.cross_timer + 1,
        )

    def _get_obs(self) -> torch.Tensor:
        lanes_obs = super()._get_obs()

        # One-hot encode the current phase: shape (num_worlds, NUM_PHASES)
        phase_one_hot = torch.zeros(
            self.num_worlds, self.NUM_PHASES,
            dtype=torch.float32, device=self.device
        )
        phase_one_hot.scatter_(1, self.cross_phase.unsqueeze(1), 1.0)

        obs = torch.cat([
            lanes_obs,
            phase_one_hot,
            self.cross_timer.float().view(self.num_worlds, 1),
        ], dim=1)
        return obs.detach().clone()

    def _get_terminated(self) -> torch.Tensor:
        terminated_lanes = super()._get_terminated().bool()

        # Stalling in any single phase triggers termination
        stuck_phase = (self.cross_timer >= self.max_phase_steps)

        self._update_carioca_state()

        return (terminated_lanes | stuck_phase).bool().detach()
