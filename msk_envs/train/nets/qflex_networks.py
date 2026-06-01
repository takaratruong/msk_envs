import copy
from typing import Optional, Sequence, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .normalizers import BatchRenorm

Activation = Callable[[torch.Tensor], torch.Tensor]


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


class MLPWithBN(nn.Module):
    """ MLP with BatchRenorm applied before the first layer and after each hidden layer """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            activation: Activation = F.relu,
            output_activation: Activation = identity,
            eps: float = 1e-5,
            decay_rate: float = 0.99,
            squeeze_output: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.squeeze_output = squeeze_output
        # Input BatchRenorm
        self.input_bn = BatchRenorm(input_size, eps=eps, decay_rate=decay_rate)
        # Hidden layers + per-layer BatchRenorm
        self.hidden_linears: nn.ModuleList = nn.ModuleList()
        self.hidden_bns: nn.ModuleList = nn.ModuleList()
        in_size = input_size
        for h in hidden_sizes:
            linear = nn.Linear(in_size, h)
            self.hidden_linears.append(linear)
            self.hidden_bns.append(BatchRenorm(h, eps=eps, decay_rate=decay_rate))
            in_size = h
        # Output linear
        self.output_linear = nn.Linear(in_size, output_size)
        return

    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.input_bn(x, is_training)
        for linear, bn in zip(self.hidden_linears, self.hidden_bns):
            x = bn(self.activation(linear(x)), is_training)
        x = self.output_activation(self.output_linear(x))
        return x.squeeze(-1) if self.squeeze_output else x


class PolicyNetBN(nn.Module):
    """ Gaussian policy network with Batch Renorm """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: Sequence[int],
            activation: Activation = F.relu,
            output_activation: Activation = identity,
            min_log_std: float = -20.0,
            max_log_std: float = 2.0,
            log_std_mode: Union[str, float] = "shared",
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std_mode = log_std_mode

        make_network = lambda out: MLPWithBN(
            obs_dim, hidden_sizes, out, activation, output_activation, eps=eps
        )

        if log_std_mode == "shared":
            self.net = make_network(act_dim * 2)
        elif log_std_mode == "separate":
            self.mean_net = make_network(act_dim)
            self.logstd_net = make_network(act_dim)
        else:
            self.net = make_network(act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_mode)))
        return

    def forward(
            self,
            obs: torch.Tensor,
            is_training: bool = True,
            return_log_std: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if self.log_std_mode == "shared":
            mean, log_std = self.net(obs, is_training).chunk(2, dim=-1)
        elif self.log_std_mode == "separate":
            mean = self.mean_net(obs, is_training)
            log_std = self.logstd_net(obs, is_training)
        else:
            mean = self.net(obs, is_training)
            log_std = self.log_std.expand_as(mean)

        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return (mean, log_std) if return_log_std else (mean, log_std.exp())


class QNetBN(nn.Module):
    """ Action-value network with Batch Renorm """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: Sequence[int],
            activation: Activation,
            output_activation: Activation = identity,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.net = MLPWithBN(
            input_size=obs_dim + act_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            output_activation=output_activation,
            eps=eps,
            squeeze_output=True,
        )

    def forward(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            is_training: bool = True,
    ) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1), is_training)


def simple_euler(func, x0, t, obs):
    """Integrate a vector field with a clipped Euler solver. """
    dt = t[1] - t[0]
    is_tuple = isinstance(x0, tuple)
    x = x0
    flow = [x]
    for ti in t[1:]:
        dx = func(obs, x, ti)
        if is_tuple:
            x_state, vs = x
        else:
            x_state = x
        # Clip each Euler step to avoid unstable jumps during sampling.
        dx = torch.clamp(dx, -1.0 / dt, 1.0 / dt)
        x_state = x_state + dt * dx
        if is_tuple:
            x = (x_state, vs)
        else:
            x = x_state
        flow.append(x)
    return flow


def _sample_probe(
        num_probes: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        dist: str = "rademacher",
) -> torch.Tensor:
    """ Sample probe vectors for the Hutchinson trace estimator """
    if dist == "rademacher":  # random choice [-1, 1]
        return torch.randint(0, 2, (num_probes, dim), device=device, dtype=dtype) * 2.0 - 1.0
    elif dist == "gaussian":
        return torch.randn(num_probes, dim, device=device, dtype=dtype)
    else:
        raise ValueError(f"dist must be 'rademacher' or 'gaussian'; got '{dist}'")


class FlowMatching(nn.Module):
    """ Flow-matching sampler and likelihood helper """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: Sequence[int],
            velocity_activation: Callable,
            num_timesteps: int,
            sigma: float = 1e-4,
            reference_gn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.act_dim = act_dim
        self.num_timesteps = num_timesteps
        self.sigma = sigma
        self.velocity_field = FlowVelocityFieldBN(
            obs_dim, act_dim, hidden_sizes, velocity_activation
        )

        # Non-owned reference Gaussian - list prevents nn.Module auto-registration
        self._ref_gn_container: list = [reference_gn]

    @property
    def reference_gn(self) -> Optional[nn.Module]:
        """The reference Gaussian policy (``None`` if not used)."""
        return self._ref_gn_container[0]

    def _draw_initial_sample(
            self,
            obs: torch.Tensor,
            is_training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Draw an initial action sample from the reference distribution """
        B = obs.shape[0]
        noise = torch.randn(B, self.act_dim, device=obs.device, dtype=obs.dtype)
        if self.reference_gn is not None:
            ref_mean, ref_std = self.reference_gn(obs, is_training=is_training)
            x_pretanh = ref_mean + ref_std * noise
        else:
            ref_mean = torch.ones_like(noise)
            ref_std = torch.zeros_like(noise)
            x_pretanh = noise
        x0 = torch.tanh(x_pretanh)
        return x0, noise, ref_mean, ref_std

    def _hutchinson_trace(
            self,
            obs: torch.Tensor,
            x: torch.Tensor,
            t_batch: torch.Tensor,
            is_training: bool,
            vs: torch.Tensor,
    ) -> torch.Tensor:
        """ Per-sample Hutchinson estimate of div(v) = trace(∂v/∂x) """
        B = x.shape[0]
        traces = []

        for b in range(B):
            obs_b = obs[b: b + 1]  # (1, O)
            t_b = t_batch[b: b + 1]  # (1, 1)
            x_b = x[b]  # (A,)

            def vel_b(x_single: torch.Tensor) -> torch.Tensor:
                """Velocity for a single (obs, t); both input/output are (A,)."""
                return self.velocity_field(
                    obs_b, x_single.unsqueeze(0), t_b, is_training=is_training
                ).squeeze(0)

            # Hutchinson: E_v[v^T (J_v · v)] ≈ trace(J_v)
            probe_vals = []
            for v in vs:  # v: (A,)
                _, jvp_out = torch.autograd.functional.jvp(
                    vel_b,
                    (x_b,),
                    (v,),
                    create_graph=is_training,
                )
                probe_vals.append((v * jvp_out).sum())

            traces.append(torch.stack(probe_vals).mean())

        return torch.stack(traces)  # (B,)

    def sample(
            self,
            obs: torch.Tensor,
            is_training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample a batch of actions by integrating the learned flow """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x0, _, _, _ = self._draw_initial_sample(obs, is_training)
        action = simple_euler(
            self.velocity_field, x0, obs, self.num_timesteps
        )
        return action, x0

    def sample_with_log_prob(
            self,
            obs: torch.Tensor,
            is_training: bool = False,
            num_probes: int = 16,
    ) -> Tuple[
        torch.Tensor,  # action
        torch.Tensor,  # log_prob
        torch.Tensor,  # ref_mean
        torch.Tensor,  # ref_std
        torch.Tensor,  # action_init
    ]:
        """ Sample actions and estimate log probabilities via CNF likelihood. """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        x0, noise, ref_mean, ref_std = self._draw_initial_sample(obs, is_training)

        # ---- Initial log probability under N(0, I) -----------------------
        # Mirrors: p0 = norm_logpdf(sample).sum()
        p0 = Normal(
            torch.zeros_like(noise), torch.ones_like(noise)
        ).log_prob(noise).sum(dim=-1)  # (B,)

        # ---- Tanh-squashing correction (reproduces the original exactly) --
        # Mirrors: tanh_trans_prob = clip(1 - clip(x0, -1+eps, 1-eps), -5, 5).sum()
        x0_safe = x0.clamp(-1 + 1e-6, 1 - 1e-6)
        tanh_correction = (1.0 - x0_safe).clamp(-5.0, 5.0).sum(dim=-1)  # (B,)
        p0 = p0 - tanh_correction

        # ---- Augmented Euler ODE -----------------------------------------
        t_grid = torch.linspace(
            0, 1, self.num_timesteps + 1, device=obs.device, dtype=obs.dtype
        )
        dt = (t_grid[1] - t_grid[0]).item()

        x = x0
        log_det = torch.zeros(B, device=obs.device, dtype=obs.dtype)

        # Sample probe vectors ONCE; reuse at every Euler step.
        # The original threads them through the scan state unchanged.
        vs = _sample_probe(
            num_probes, self.act_dim, obs.device, obs.dtype, dist="rademacher"
        )  # (P, A)

        for i in range(self.num_timesteps):
            ti_val = t_grid[i + 1].item()
            t_batch = torch.full(
                (B, 1), ti_val, device=obs.device, dtype=obs.dtype
            )

            # Hutchinson divergence estimate: −trace(∂v/∂x) per sample
            trace_est = self._hutchinson_trace(
                obs, x, t_batch, is_training, vs
            )  # (B,)

            # Position Euler update
            dx = self.velocity_field(obs, x, t_batch, is_training=is_training)
            dx = dx.clamp(-1.0 / dt, 1.0 / dt)
            x = x + dt * dx

            # Accumulate log-determinant change: d(log_p)/dt = −trace(J_v)
            log_det = log_det - dt * trace_est

        log_prob = p0 + log_det  # (B,)
        return x, log_prob, ref_mean, ref_std, x0


class FlowVelocityFieldBN(nn.Module):
    """Velocity field network with Batch Renorm.

    Input: concatenation of (obs, act, t) where t ∈ ℝ¹ per sample.
    """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: Sequence[int],
            activation: Activation = F.relu,
            output_activation: Activation = identity,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # +1 for the scalar time dimension t
        self.net = MLPWithBN(
            obs_dim + act_dim + 1, hidden_sizes, act_dim,
            activation, output_activation, eps=eps,
        )

    def forward(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            t: torch.Tensor,
            is_training: bool = True,
    ) -> torch.Tensor:
        x = torch.cat([obs, act, t], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x, is_training)


class FlowNet(nn.Module):
    """Network bundle used by the Qflex algorithm """

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_sizes: Sequence[int],
            activation: type = nn.ReLU,
            velocity_activation: type = nn.ReLU,
            num_timesteps: int = 20,
            target_entropy: float = -1.0,
            learn_reference_gn: bool = True,
    ) -> None:
        super().__init__()

        self.act_dim = act_dim
        self.num_timesteps = num_timesteps
        self.target_entropy = target_entropy

        # Q networks
        self.q1 = QNetBN(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = QNetBN(obs_dim, act_dim, hidden_sizes, activation)

        # Target networks: deep copies with frozen parameters.
        # Mirrors FlowParams.target_q1 / target_q2 in the original.
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        # Reference Gaussian policy
        if learn_reference_gn:
            self.reference_gn: Optional[nn.Module] = PolicyNetBN(
                obs_dim=obs_dim, act_dim=act_dim,
                hidden_sizes=hidden_sizes, activation=activation
            )
        else:
            self.reference_gn = None

        # Flow-matching helper (owns the velocity field subnetwork)
        self.flow = FlowMatching(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            velocity_activation=velocity_activation,
            num_timesteps=num_timesteps,
            reference_gn=self.reference_gn,
        )

        # Entropy coefficient
        self.log_alpha = nn.Parameter(torch.zeros(act_dim))

        # exp_prob controls the mix between the flow policy and the reference
        # Gaussian at action-sampling time (1 = always use flow).
        self.register_buffer("exp_prob", torch.ones(1))

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ Sample a stochastic action for environment interaction """
        action, action_init = self.flow.sample(obs, is_training=False)

        # Bernoulli mask: shape (B, 1) to broadcast over action dimensions.
        exp_mode = torch.bernoulli(
            self.exp_prob.expand(obs.shape[0], 1)
        )
        return exp_mode * action + (1.0 - exp_mode) * action_init

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ Return the deterministic flow action used for evaluation """
        action, _ = self.flow.sample(obs, is_training=False)
        return action

    def evaluate_reference(
            self,
            obs: torch.Tensor,
            is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """ Sample from the learned reference Gaussian policy """
        if self.reference_gn is None:
            raise ValueError("reference_gn is None; cannot evaluate reference policy.")

        mean, std = self.reference_gn(obs, is_training=is_training)
        z = torch.randn_like(mean)
        act_pretanh = mean + std * z  # re-parameterized sample
        logp = Normal(mean, std).log_prob(act_pretanh)  # pre-squash log-prob
        return torch.tanh(act_pretanh), logp, None, mean, std

    def evaluate(
            self,
            obs: torch.Tensor,
            is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Sample actions and log-probabilities for algorithm updates """
        action, logp, reference_mean, reference_std, action_init = (
            self.flow.sample_with_log_prob(obs, is_training=is_training)
        )
        return action, logp, reference_mean, reference_std, action_init


def create_flow_net(
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        num_timesteps: int,
        learn_reference_gn,
        activation: type = nn.ReLU,
        velocity_activation: type = nn.ReLU,
) -> FlowNet:
    """Create and return an initialized FlowNet """
    net = FlowNet(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        velocity_activation=velocity_activation,
        num_timesteps=num_timesteps,
        target_entropy=-1.0,
        learn_reference_gn=learn_reference_gn,
    )
    return net
