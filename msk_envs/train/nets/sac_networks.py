from __future__ import annotations

import torch
from torch import nn

from msk_envs.train.nets.normalizers import EmpiricalNormalization


class Actor(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_envs: int,
            hidden_dim: int,
            log_std_max: float,
            log_std_min: float,
            use_tanh: bool = True,
            use_layer_norm: bool = True,
            device: torch.device | str | None = None,
            action_scale: torch.Tensor | None = None,
            action_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        # Setup the network - this will be overridden in subclasses if needed
        self.setup_network()

        # Register action scaling parameters as buffers
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        self._setup_network_with_input_dim(self.n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension."""
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
        )
        self.fc_logstd = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                log_std + 1
        )  # From SpinUp / Denis Yarats

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            # Apply tanh to get bounded actions in [-1, 1]
            tanh_action = torch.tanh(raw_action)
            # Scale and bias to get final actions
            action = tanh_action * self.action_scale + self.action_bias

            # Compute log probability with proper Jacobian correction
            log_prob = dist.log_prob(raw_action)
            # Jacobian correction for tanh transformation
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            # Jacobian correction for scaling transformation
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            # Non-tanh case
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob

    @torch.no_grad()
    def explore(
            self, obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = raw_action

        return action


class Policy(nn.Module):
    def __init__(self, n_obs: int, n_act: int, args: dict):
        super().__init__()

        self.args = args
        self.actor = Actor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=args["num_envs"],
            hidden_dim=args["actor_hidden_dim"],
            log_std_max=args["log_std_max"],
            log_std_min=args["log_std_min"],
            use_tanh=args["use_tanh"],
            use_layer_norm=args["use_layer_norm"],
            device="cpu",
            action_scale=args["action_scale"] if "action_scale" in args else None,
            action_bias=args["action_bias"] if "action_bias" in args else None,
        )

        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")
        self.actor.eval()
        self.obs_normalizer.eval()

    def forward(self, obs: torch.Tensor, use_mean: bool = False) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        action, mean, std = self.actor(norm_obs)
        return action if not use_mean else mean


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(f"{checkpoint_path}", map_location="cpu", weights_only=False)
    args = torch_checkpoint["args"]
    n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
    n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]
    policy = Policy(n_obs=n_obs, n_act=n_act, args=args)
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(
            torch_checkpoint["obs_normalizer_state"])
    return policy
