import math
from typing import Tuple

from msk_envs.train.nets.normalizers import BatchRenorm1d

import torch
import torch.nn as nn


def _bn_mlp(in_dim: int, hidden_sizes, out_dim: int) -> nn.Module:
    """Build ``BN -> [Linear, ReLU, BN] * n -> Linear`` (matches ``mlp_with_bn``)."""
    layers = [BatchRenorm1d(in_dim)]
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.ReLU(), BatchRenorm1d(h)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Scalar Q(obs, act) with BatchRenorm (``QNetBN``)."""

    def __init__(self, n_obs: int, n_act: int, hidden_sizes):
        super().__init__()
        self.net = _bn_mlp(n_obs + n_act, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class Critic(nn.Module):
    """Twin Q-networks, mirroring the ``q1``/``q2`` pair in QFlex."""

    def __init__(self, n_obs: int, n_act: int, hidden_sizes):
        super().__init__()
        self.qnet1 = QNetwork(n_obs, n_act, hidden_sizes)
        self.qnet2 = QNetwork(n_obs, n_act, hidden_sizes)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.qnet1(obs, act), self.qnet2(obs, act)

    def min_q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, act)
        return torch.minimum(q1, q2)


class ReferencePolicy(nn.Module):
    """BatchRenorm tanh-Gaussian reference policy (``PolicyNetBN``)."""

    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_sizes,
            min_log_std: float = -20.0,
            max_log_std: float = 2.0,
    ):
        super().__init__()
        self.n_act = n_act
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.net = _bn_mlp(n_obs, hidden_sizes, 2 * n_act)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, torch.exp(log_std)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Return ``tanh(mean + std * z)`` and the diagonal-Gaussian log-prob.

        The log-prob is of the pre-tanh sample (matching ``evaluate_reference``),
        and is only used for diagnostics, not in any loss.
        """
        mean, std = self.forward(obs)
        if deterministic:
            z = torch.zeros_like(mean)
        else:
            z = torch.randn_like(mean)
        pre_tanh = mean + std * z
        logp = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))
        return torch.tanh(pre_tanh), logp, mean, std


class VelocityField(nn.Module):
    """BatchRenorm velocity field v(obs, act, t) (``FlowVelocityFieldBN``)."""

    def __init__(self, n_obs: int, n_act: int, hidden_sizes):
        super().__init__()
        self.net = _bn_mlp(n_obs + n_act + 1, hidden_sizes, n_act)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act, t], dim=-1)
        return self.net(x)


class QFlexActor(nn.Module):
    """Reference policy + velocity field with the clipped-Euler flow sampler."""

    def __init__(self, reference: ReferencePolicy, velocity_field: VelocityField, num_timesteps: int):
        super().__init__()
        self.reference = reference
        self.velocity_field = velocity_field
        self.num_timesteps = num_timesteps

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Integrate the learned flow from a reference action.

        Returns ``(flow_action, init_action)`` where ``init_action`` is the
        tanh-squashed reference sample used as the flow's starting point.
        """
        was_training = self.training
        self.eval()  # BatchRenorm uses running stats at inference time.

        mean, std = self.reference(obs)
        if deterministic:
            x = torch.tanh(mean)
        else:
            x = torch.tanh(mean + std * torch.randn_like(mean))
        x0 = x

        # Clipped forward Euler over t in linspace(0, 1, num_timesteps + 1)[1:].
        ts = torch.linspace(0.0, 1.0, self.num_timesteps + 1, device=obs.device)
        dt = ts[1] - ts[0]
        for i in range(1, self.num_timesteps + 1):
            ti = ts[i].expand(obs.shape[0], 1)
            dx = self.velocity_field(obs, x, ti)
            dx = dx.clamp(-1.0 / dt, 1.0 / dt)
            x = x + dt * dx

        if was_training:
            self.train()
        return x, x0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action for environment interaction (``exp_prob = 1``)."""
        flow_action, _ = self.sample(obs, deterministic=deterministic)
        return flow_action
