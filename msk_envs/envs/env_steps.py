import math

import torch

from msk_envs.utils.global_params import FWD_IDX, UP_IDX, SIDE_IDX
from msk_envs.utils.reward_lib import joint_penalty, has_fallen, muscle_passive_penalty
from .env_base import MSKEnv
from .env_config import EnvConfig


class StepsEnv(MSKEnv):
    """ ALLSTEPS-style stepping targets (Xie et al. 2020).

    Each env gets its own sequence of foot targets on the ground. The agent is
    rewarded for placing the swing foot on the *current* target (a Gaussian
    target-hitting bonus) and for advancing through the sequence (progress).
    A difficulty curriculum widens the sampling ranges of step length, width,
    and heading turn as training succeeds.

    Targets remain per-env virtual points (plain tensors) because ALLSTEPS uses
    the foot-target reward as its core learning signal. Bolt now supports
    per-world collider transforms, which StoneCourseEnv uses for physical slabs.
    """

    N_LOOKAHEAD = 2  # number of upcoming targets exposed in the observation

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device,
            requires_visuals: bool,
            cuda_graph: bool,
            n_targets: int = 32,
            target_radius: float = 0.10,
            hit_sigma: float = 0.25,
    ):
        super().__init__(num_envs=num_envs, env_config=env_config, device=device,
                         requires_visuals=requires_visuals, cuda_graph=cuda_graph)

        self.n_targets = n_targets
        self.target_radius = target_radius
        self.hit_sigma = hit_sigma

        # Curriculum-controlled sampling ranges (ALLSTEPS §4). Start easy: modest
        # forward steps, near-zero width/turn. curriculum in [0,1] scales toward hard.
        self.curriculum = env_config.steps_curriculum_start
        self.curriculum_max = 1.0
        self.step_len_range = env_config.steps_len_range          # forward spacing (m)
        self.step_width_easy, self.step_width_hard = 0.15, env_config.steps_width_max   # lateral (m)
        self.turn_easy, self.turn_hard = 0.0, math.radians(env_config.steps_turn_max_deg)

        # Foot bodies (target is "hit" by whichever foot is the swing/plant foot)
        self.foot_ids = [self.body_id_lookup["calcn_l"], self.body_id_lookup["calcn_r"]]

        # Per-env target buffers
        self.targets = torch.zeros((num_envs, n_targets, 3), device=device)   # world xyz
        self.target_idx = torch.zeros(num_envs, dtype=torch.long, device=device)  # current target
        # Which foot should hit the current target (0=left,1=right), alternating
        self.target_foot = torch.zeros((num_envs, n_targets), dtype=torch.long, device=device)

        # Logging
        self.last_track_err = 0.0
        self.last_progress = 0.0
        self.last_mean_activation = 0.0
        self.last_root_height = 0.0
        return

    # ---- target sequence generation -------------------------------------------------

    def _sample_step_params(self, n: int):
        c = self.curriculum
        length = torch.rand(n, device=self.device) * (self.step_len_range[1] - self.step_len_range[0]) + self.step_len_range[0]
        width_max = self.step_width_easy + c * (self.step_width_hard - self.step_width_easy)
        turn_max = self.turn_easy + c * (self.turn_hard - self.turn_easy)
        width = (torch.rand(n, device=self.device) * 2 - 1) * width_max
        turn = (torch.rand(n, device=self.device) * 2 - 1) * turn_max
        return length, width, turn

    def _generate_targets(self, reset_mask: torch.Tensor) -> None:
        """ Build a fresh alternating-foot target sequence for the reset envs. """
        idxs = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            return
        n = idxs.numel()
        heading = torch.zeros(n, device=self.device)
        pos = torch.zeros((n, 3), device=self.device)
        half_stance = 0.12  # foot lateral half-stance
        for k in range(self.n_targets):
            length, width, turn = self._sample_step_params(n)
            if k < 2:
                # seed the first two steps directly under the feet (small, straight)
                length = torch.full_like(length, 0.10)
                turn = torch.zeros_like(turn)
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

    def _upon_reset_post_sim(self, reset_mask: torch.Tensor) -> None:
        self._generate_targets(reset_mask)
        return

    # ---- helpers --------------------------------------------------------------------

    def _current_target(self) -> torch.Tensor:
        return torch.gather(self.targets, 1, self.target_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    def _foot_pos(self, foot: int) -> torch.Tensor:
        return self.body_positions[:, self.foot_ids[foot]]

    def _plant_foot_pos(self) -> torch.Tensor:
        """ Position of the foot that is supposed to hit the current target. """
        cur_foot = torch.gather(self.target_foot, 1, self.target_idx.view(-1, 1)).squeeze(1)  # (N,)
        lpos = self._foot_pos(0)
        rpos = self._foot_pos(1)
        sel = cur_foot.view(-1, 1).expand(-1, 3)
        return torch.where(sel == 0, lpos, rpos)

    def _dist_to_target(self) -> torch.Tensor:
        foot = self._plant_foot_pos()
        tgt = self._current_target()
        d = foot[:, [FWD_IDX, SIDE_IDX]] - tgt[:, [FWD_IDX, SIDE_IDX]]
        return d.norm(dim=1)

    def _get_obs(self) -> torch.Tensor:
        # Root-relative offsets to the next N_LOOKAHEAD targets (world axes; the
        # policy also sees root orientation via joint positions).
        root_xz = self.root_pos[:, [FWD_IDX, SIDE_IDX]]
        offs = []
        for j in range(self.N_LOOKAHEAD):
            idx = torch.clamp(self.target_idx + j, max=self.n_targets - 1)
            tgt = torch.gather(self.targets, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
            offs.append(tgt[:, [FWD_IDX, SIDE_IDX]] - root_xz)
            foot = torch.gather(self.target_foot, 1, idx.view(-1, 1)).float()
            offs.append(foot)
        target_obs = torch.cat(offs, dim=1)

        drop_names = ["pelvis_tx", "pelvis_tz"]
        drop_ids = {self.qpos_id_lookup[n] for n in drop_names if n in self.qpos_id_lookup}
        keep_ids = [i for i in range(self.num_qpos) if i not in drop_ids]
        keep = torch.tensor(keep_ids, device=self.device, dtype=torch.long)

        obs = torch.cat([
            target_obs,
            self.muscle_activations,
            self.muscle_fiber_lengths,
            self.actuator_activations,
            self.joint_positions[:, keep],
            self.joint_velocities,
        ], dim=1)
        return obs.detach().clone()

    # ---- reward (ALLSTEPS: target-hitting + progress) --------------------------------

    def _compute_raw_reward_dict(self):
        dist = torch.nan_to_num(self._dist_to_target(), nan=10.0, posinf=10.0)

        # Target-hitting: Gaussian bonus peaking when the plant foot is on target.
        rew_target = torch.exp(-(dist / self.hit_sigma).pow(2))

        # Progress: advance target_idx when the plant foot is within radius AND low.
        planted = (self._plant_foot_pos()[:, UP_IDX] < 0.12)
        hit = (dist < self.target_radius) & planted & (self.target_idx < self.n_targets - 1)
        rew_progress = hit.float()
        if hit.any():
            self.target_idx[hit] = self.target_idx[hit] + 1

        rew_alive = torch.ones(self.num_worlds, device=self.device)
        rew_limit = torch.nan_to_num(joint_penalty(self.ufrc_limit, squared=False), nan=0.0, posinf=0.0, neginf=0.0)
        rew_muscle_passive = torch.nan_to_num(
            muscle_passive_penalty(self.muscle_passive_length_multiplier, threshold=0.1, squared=False),
            nan=0.0, posinf=0.0, neginf=0.0)

        self.reward_dict = {
            "rew_target": rew_target.detach(),
            "rew_progress": rew_progress.detach(),
            "rew_alive": rew_alive.detach(),
            "rew_limit": rew_limit.detach(),
            "rew_muscle_passive": rew_muscle_passive.detach(),
        }

    def _get_terminated(self):
        fallen = has_fallen(root_pos=self.root_pos, ground_rotation=self.ground_rotation)
        return fallen.float().detach()

    # ---- curriculum + metrics --------------------------------------------------------

    @property
    def curriculum_advance_threshold(self) -> float:
        return 6.0  # mean targets reached before we make it harder

    def advance_curriculum(self, mean_progress: float) -> None:
        """ Widen sampling ranges once the agent reliably reaches deep into the
        sequence. Intended to be called periodically by the training loop. """
        if mean_progress > self.curriculum_advance_threshold:
            self.curriculum = min(self.curriculum_max, self.curriculum + 0.05)
        return

    def get_render_targets(self, world_id: int):
        out = []
        cur = int(self.target_idx[world_id].item())
        for k in range(self.n_targets):
            pos = self.targets[world_id, k].tolist()
            out.append((pos, self.target_radius, k == cur))
        return out

    def update_metrics(self) -> None:
        self.last_track_err = self._dist_to_target().mean().item()
        self.last_progress = self.target_idx.float().mean().item()
        self.last_mean_activation = self.muscle_activations.mean().item()
        self.last_root_height = self.root_pos[:, UP_IDX].mean().item()
        return

    def additional_metrics(self) -> dict:
        return {
            "dist_to_target": self.last_track_err,
            "mean_targets_reached": self.last_progress,
            "curriculum": self.curriculum,
            "mean_muscle_activation": self.last_mean_activation,
            "mean_root_height": self.last_root_height,
        }
