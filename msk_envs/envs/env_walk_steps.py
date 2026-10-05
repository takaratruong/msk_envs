import math

import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, SIDE_IDX
from .env_config import EnvConfig
from .env_walk import WalkEnv


class WalkStepsEnv(WalkEnv):
    """ Stepping targets (ALLSTEPS, Xie et al. 2020) layered on the WALKING base.

    WalkEnv already produces stable left/right alternation (lane + facing
    termination + run start + capped-velocity reward). This adds per-env foot
    targets and the ALLSTEPS target-hitting + progress rewards ON TOP, so the
    walker learns to place its feet on a target sequence — the far easier
    learning problem the bare-env StepsEnv failed at (it had to learn balance
    and gait from scratch simultaneously).

    Targets remain per-env VIRTUAL points (tensors), not physical colliders,
    because the foot-target reward is the core ALLSTEPS signal. Bolt now also
    supports per-world physical collider transforms; StoneCourseEnv uses those
    for its slabs. Alternating L/R targets at symmetric lateral offsets pressure
    both legs to do the same job (fixes gait asymmetry).
    """

    N_LOOKAHEAD = 2

    def __init__(self, num_envs, env_config, device, requires_visuals, cuda_graph):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device,
                         requires_visuals=requires_visuals, cuda_graph=cuda_graph)

        self.n_targets = env_config.steps_n_targets
        self.target_radius = env_config.steps_target_radius
        self.hit_sigma = env_config.steps_hit_sigma
        self.stone_radius = env_config.steps_stone_radius
        self.gap_terminate = env_config.steps_gap_terminate
        self.steps_grace_time = env_config.steps_grace_time
        self.require_support = env_config.steps_require_support
        self.steps_max_flight_time = env_config.steps_max_flight_time
        # Per-env accumulator: how long (s) the walker has had NO foot on a stone.
        self._unsupported_time = torch.zeros(num_envs, device=device)

        self.curriculum = env_config.steps_curriculum_start
        # Adaptive curriculum state: EMA of targets-reached-at-reset; widen
        # difficulty by curriculum_step once it clears the threshold.
        self._progress_ema = 0.0
        self.curriculum_advance_threshold = env_config.steps_curriculum_threshold
        self.curriculum_step = env_config.steps_curriculum_step
        self.step_len_range = env_config.steps_len_range
        self.step_width_easy, self.step_width_hard = 0.15, env_config.steps_width_max
        self.turn_easy, self.turn_hard = 0.0, math.radians(env_config.steps_turn_max_deg)

        self.foot_ids = [self.body_id_lookup["calcn_l"], self.body_id_lookup["calcn_r"]]

        self.targets = torch.zeros((num_envs, self.n_targets, 3), device=device)
        self.target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.target_foot = torch.zeros((num_envs, self.n_targets), dtype=torch.long, device=device)

        self.last_dist = 0.0
        self.last_progress = 0.0
        return

    # ---- target generation ----------------------------------------------------------

    def _sample_step_params(self, n):
        c = self.curriculum
        length = torch.rand(n, device=self.device) * (self.step_len_range[1] - self.step_len_range[0]) + self.step_len_range[0]
        width_max = self.step_width_easy + c * (self.step_width_hard - self.step_width_easy)
        turn_max = self.turn_easy + c * (self.turn_hard - self.turn_easy)
        width = (torch.rand(n, device=self.device) * 2 - 1) * width_max
        turn = (torch.rand(n, device=self.device) * 2 - 1) * turn_max
        return length, width, turn

    def _generate_targets(self, reset_mask):
        idxs = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            return
        n = idxs.numel()
        heading = torch.zeros(n, device=self.device)
        pos = torch.zeros((n, 3), device=self.device)
        half_stance = 0.12
        for k in range(self.n_targets):
            length, width, turn = self._sample_step_params(n)
            if k < 2:
                length = torch.full_like(length, 0.35)  # first steps straight, forward
                turn = torch.zeros_like(turn)
                width = torch.zeros_like(width)
            heading = heading + turn
            foot = k % 2  # 0=left,1=right
            side = (half_stance if foot == 0 else -half_stance) + width
            dx = length * torch.cos(heading) - side * torch.sin(heading)
            dz = length * torch.sin(heading) + side * torch.cos(heading)
            pos = pos.clone()
            pos[:, FWD_IDX] = pos[:, FWD_IDX] + dx
            pos[:, SIDE_IDX] = pos[:, SIDE_IDX] + dz
            pos[:, UP_IDX] = 0.0
            self.targets[idxs, k, :] = pos
            self.target_foot[idxs, k] = foot
        self.target_idx[idxs] = 0
        return

    def _upon_reset_post_sim(self, reset_mask):
        # Adaptive curriculum (ALLSTEPS §4): before regenerating a resetting env's
        # course, record how far it got. Keep an EMA over resets and widen the
        # difficulty (step width + turn) once the agent reliably steps deep into
        # the sequence. Self-contained — no training-loop hook needed.
        idxs = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if idxs.numel() > 0:
            reached = self.target_idx[idxs].float().mean().item()
            self._progress_ema = 0.98 * self._progress_ema + 0.02 * reached
            if self._progress_ema > self.curriculum_advance_threshold and self.curriculum < 1.0:
                self.curriculum = min(1.0, self.curriculum + self.curriculum_step)
                self._progress_ema = 0.0  # reset the bar after each promotion
        self._generate_targets(reset_mask)
        self._unsupported_time[reset_mask] = 0.0
        return

    # ---- helpers --------------------------------------------------------------------

    def _current_target(self):
        return torch.gather(self.targets, 1, self.target_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    def _plant_foot_pos(self):
        cur_foot = torch.gather(self.target_foot, 1, self.target_idx.view(-1, 1)).squeeze(1)
        lpos = self.body_positions[:, self.foot_ids[0]]
        rpos = self.body_positions[:, self.foot_ids[1]]
        sel = cur_foot.view(-1, 1).expand(-1, 3)
        return torch.where(sel == 0, lpos, rpos)

    def _swing_foot_pos(self):
        """ The OTHER foot (should be airborne when the plant foot lands). """
        cur_foot = torch.gather(self.target_foot, 1, self.target_idx.view(-1, 1)).squeeze(1)
        lpos = self.body_positions[:, self.foot_ids[0]]
        rpos = self.body_positions[:, self.foot_ids[1]]
        sel = cur_foot.view(-1, 1).expand(-1, 3)
        return torch.where(sel == 0, rpos, lpos)

    def _dist_to_target(self):
        foot = self._plant_foot_pos()
        tgt = self._current_target()
        d = foot[:, [FWD_IDX, SIDE_IDX]] - tgt[:, [FWD_IDX, SIDE_IDX]]
        return d.norm(dim=1)

    def _get_obs(self):
        base = super()._get_obs()  # WalkEnv/LanesEnv obs (no root x)
        root_xz = self.root_pos[:, [FWD_IDX, SIDE_IDX]]
        offs = []
        for j in range(self.N_LOOKAHEAD):
            idx = torch.clamp(self.target_idx + j, max=self.n_targets - 1)
            tgt = torch.gather(self.targets, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            offs.append(tgt[:, [FWD_IDX, SIDE_IDX]] - root_xz)
            offs.append(torch.gather(self.target_foot, 1, idx.view(-1, 1)).float())
        target_obs = torch.cat(offs, dim=1)
        return torch.cat([target_obs, base], dim=1).detach().clone()

    # ---- reward: WalkEnv terms + ALLSTEPS target/progress ---------------------------

    def _compute_raw_reward_dict(self):
        super()._compute_raw_reward_dict()  # rew_vel, rew_mid_lane, springs, limit, ...

        dist = torch.nan_to_num(self._dist_to_target(), nan=10.0, posinf=10.0)
        rew_target = torch.exp(-(dist / self.hit_sigma).pow(2))

        # Progress: plant foot within radius AND low, AND the OTHER foot airborne
        # (enforces true single-support stepping — a hop can't satisfy this).
        plant_low = self._plant_foot_pos()[:, UP_IDX] < 0.12
        swing_up = self._swing_foot_pos()[:, UP_IDX] > 0.08
        hit = (dist < self.target_radius) & plant_low & swing_up & (self.target_idx < self.n_targets - 1)
        rew_progress = hit.float()
        if hit.any():
            self.target_idx[hit] = self.target_idx[hit] + 1

        self.reward_dict["rew_target"] = rew_target.detach()
        self.reward_dict["rew_progress"] = rew_progress.detach()

    # ---- termination: ALLSTEPS "miss = fall" (gap) ----------------------------------

    def _foot_min_dist_to_any_target(self, foot_id):
        """ Horizontal distance from a foot to its NEAREST target (over the whole
        sequence — a plant is legal if it lands on any stone, not only the current). """
        fp = self.body_positions[:, foot_id][:, [FWD_IDX, SIDE_IDX]]           # (N,2)
        tg = self.targets[:, :, [FWD_IDX, SIDE_IDX]]                            # (N,K,2)
        d = (tg - fp.unsqueeze(1)).norm(dim=2)                                  # (N,K)
        return d.min(dim=1).values                                             # (N,)

    def _get_terminated(self):
        base = super()._get_terminated().bool()  # fall + lane + facing (LanesEnv)
        if not self.gap_terminate:
            return base.float().detach()
        # A foot that is DOWN (planted) must be on a stone; else it stepped in a gap.
        # Grace period: don't gap-terminate for the first steps_grace_time seconds,
        # so the run-start pose (feet not exactly on stones 0/1) can settle onto the
        # course instead of dying on frame one.
        past_grace = self.time > self.steps_grace_time
        term = base
        # Rule A: a foot that is DOWN must be on a stone (planting in a gap = fall).
        for fid in self.foot_ids:
            low = self.body_positions[:, fid][:, UP_IDX] < 0.08
            off_stone = self._foot_min_dist_to_any_target(fid) > self.stone_radius
            term = term | (low & off_stone & past_grace)
        # Rule B (anti-leap): forbid a SUSTAINED flight phase. A normal step has a
        # brief airborne swing, so instead of demanding a foot always be grounded
        # (which taught the policy to freeze and never step), we allow being
        # unsupported for up to steps_max_flight_time seconds, then terminate.
        if self.require_support:
            supported = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
            for fid in self.foot_ids:
                low = self.body_positions[:, fid][:, UP_IDX] < 0.10
                on_stone = self._foot_min_dist_to_any_target(fid) <= self.stone_radius
                supported = supported | (low & on_stone)
            # accumulate unsupported time; reset it whenever a foot is on a stone
            self._unsupported_time = torch.where(
                supported,
                torch.zeros_like(self._unsupported_time),
                self._unsupported_time + self.delta_t,
            )
            leaping = self._unsupported_time > self.steps_max_flight_time
            term = term | (leaping & past_grace)
        return term.float().detach()

    # ---- curriculum + metrics -------------------------------------------------------

    def get_render_targets(self, world_id):
        cur = int(self.target_idx[world_id].item())
        return [(self.targets[world_id, k].tolist(), self.target_radius, k == cur)
                for k in range(self.n_targets)]

    def update_metrics(self):
        super().update_metrics()
        self.last_dist = self._dist_to_target().mean().item()
        self.last_progress = self.target_idx.float().mean().item()
        return

    def additional_metrics(self):
        m = super().additional_metrics()
        m.update({
            "dist_to_target": self.last_dist,
            "mean_targets_reached": self.last_progress,
            "curriculum": self.curriculum,
        })
        return m
