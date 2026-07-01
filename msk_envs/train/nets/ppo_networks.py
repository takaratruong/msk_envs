""" Based on Holosoma """
import torch
import torch.nn as nn
from torch.distributions import Normal

from msk_envs.train.nets.normalizers import EmpiricalNormalization


def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "ELU",
        dropout_prob: float = 0.0,
        use_layer_norm: bool = False,
        device: torch.device = None,
) -> nn.Sequential:
    act_cls = getattr(nn, activation)
    layers: list[nn.Module] = []

    if len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim, device=device))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, hidden_dims[0], device=device))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dims[0], device=device))
    layers.append(act_cls())
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], device=device))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[i + 1], device=device))
        layers.append(act_cls())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(hidden_dims[-1], output_dim, device=device))
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """Gaussian policy with a raw, state-independent std parameter """

    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dims: list[int],
            init_noise_std: float,
            activation: str = "ELU",
            dropout_prob: float = 0.0,
            use_layer_norm: bool = False,
            min_noise_std: float | None = None,
            min_mean_noise_std: float | None = None,
            device: torch.device = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.device = device
        self.min_noise_std = min_noise_std
        self.min_mean_noise_std = min_mean_noise_std

        self.actor = build_mlp(
            n_obs, hidden_dims, n_act, activation, dropout_prob, use_layer_norm, device
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(n_act, device=device))

    def _clamped_std(self) -> torch.Tensor:
        if self.min_noise_std:
            return torch.clamp(self.std, min=self.min_noise_std)
        elif self.min_mean_noise_std:
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                return self.std * scale_up
            return self.std
        return self.std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (distribution mean)."""
        return self.actor(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self._clamped_std()
        return Normal(mean, mean * 0.0 + std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample an action; returns (action, log_prob, mean, std)."""
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist.mean, dist.stddev

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate stored actions; returns (log_prob, entropy, mean, std)."""
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, dist.mean, dist.stddev

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class PPOCritic(nn.Module):
    """State-value function V(s)."""

    def __init__(
            self,
            n_obs: int,
            hidden_dims: list[int],
            activation: str = "ELU",
            dropout_prob: float = 0.0,
            use_layer_norm: bool = False,
            device: torch.device = None,
    ):
        super().__init__()
        self.critic = build_mlp(
            n_obs, hidden_dims, 1, activation, dropout_prob, use_layer_norm, device
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


# ---------------------------------------------------------------- inference
class Policy(nn.Module):
    """Deterministic inference wrapper (normalizer + actor mean)."""

    def __init__(self, n_obs: int, n_act: int, args: dict):
        super().__init__()
        self.args = args
        self.actor = PPOActor(
            n_obs=n_obs,
            n_act=n_act,
            hidden_dims=list(args.get("hidden_dims", [512, 256, 128])),
            init_noise_std=args.get("init_noise_std", 0.8),
            activation=args.get("activation", "ELU"),
            dropout_prob=args.get("dropout_prob", 0.0),
            use_layer_norm=args.get("use_layer_norm", False),
            min_noise_std=args.get("min_noise_std", None),
            min_mean_noise_std=args.get("min_mean_noise_std", None),
            device="cpu",
        )
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")
        self.actor.eval()
        self.obs_normalizer.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        return self.actor(norm_obs)

    def act(self, obs: torch.Tensor) -> Normal:
        actions = self.forward(obs)
        return Normal(actions, torch.ones_like(actions) * 1e-8)


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(
        f"{checkpoint_path}", map_location="cpu", weights_only=False
    )
    args = torch_checkpoint["args"]
    actor_state = torch_checkpoint["actor_state_dict"]
    n_obs = actor_state["actor.0.weight"].shape[-1]
    n_act = actor_state["std"].shape[0]

    policy = Policy(n_obs=n_obs, n_act=n_act, args=args)
    policy.actor.load_state_dict(actor_state)

    obs_norm_state = torch_checkpoint["obs_normalizer_state"]
    if obs_norm_state is None or len(obs_norm_state) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(obs_norm_state)

    return policy
