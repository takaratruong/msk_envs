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

    def _get_cross_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        fwd_delta:  right_toe − left_toe along FWD_IDX  (>0 = right foot in front)
        side_delta: right_toe − left_toe along SIDE_IDX (>0 = right foot laterally ahead)
        """
        left_toe_pos = self.body_positions[:, self.toes_ids[0], :]
        right_toe_pos = self.body_positions[:, self.toes_ids[1], :]
        fwd_delta = right_toe_pos[:, FWD_IDX] - left_toe_pos[:, FWD_IDX]
        side_delta = right_toe_pos[:, SIDE_IDX] - left_toe_pos[:, SIDE_IDX]
        return fwd_delta, side_delta

    def _update_carioca_state(self) -> torch.Tensor:
        """
        Returns a wrong_cross mask: True for any world that achieved
        the opposite crossing condition to the one currently expected.
        """
        fwd_delta, side_delta = self._get_cross_deltas()

        in_front = fwd_delta > self.cross_threshold
        laterally_ahead = side_delta > self.cross_threshold
        laterally_behind = side_delta < -self.cross_threshold
        neutral = (fwd_delta.abs() < self.reset_threshold) & \
                  (side_delta.abs() < self.reset_threshold)

        cross_front = in_front & laterally_ahead
        cross_behind = in_front & laterally_behind

        # Completion condition per phase
        phase_done = torch.stack([
            cross_front,  # 0: CROSS_FRONT  expects right foot in front + laterally ahead
            neutral,  # 1: RESET_1      expects feet side-by-side
            cross_behind,  # 2: CROSS_BEHIND expects right foot in front + laterally behind
            neutral,  # 3: RESET_2      expects feet side-by-side
        ], dim=1)  # (num_worlds, 4)

        completed = phase_done.gather(1, self.cross_phase.unsqueeze(1)).squeeze(1)

        # Wrong cross: achieved the opposite crossing while in a crossing phase
        in_front_phase = (self.cross_phase == self.CROSS_FRONT)
        in_behind_phase = (self.cross_phase == self.CROSS_BEHIND)
        wrong_cross = (in_front_phase & cross_behind) | (in_behind_phase & cross_front)

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
        return wrong_cross

    def _get_terminated(self) -> torch.Tensor:
        terminated_lanes = super()._get_terminated().bool()
        stuck_phase = (self.cross_timer >= self.max_phase_steps)
        wrong_cross = self._update_carioca_state()

        return (terminated_lanes | stuck_phase | wrong_cross).bool().detach()
