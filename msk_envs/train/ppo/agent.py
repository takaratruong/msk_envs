import torch
import torch.nn as nn

from .actions import NormalDistribution
from .agent_config import AgentConfig
from .normalize import RunningMeanStd

mlp_init = torch.nn.Identity()


def initialize_layers(layers):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            mlp_init(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    return


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "tanh":
        return nn.Tanh()
    return nn.Identity()


class BackboneMLP(nn.Module):
    def __init__(self, input_dim, backbone_dims, activation):
        super().__init__()
        layers = []
        for i in range(len(backbone_dims)):
            layer_in = input_dim if i == 0 else backbone_dims[i - 1]
            layer_out = backbone_dims[i]

            layers.append(nn.Linear(layer_in, layer_out))
            layers.append(get_activation(activation))
        initialize_layers(layers)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActorHead(nn.Module):
    def __init__(self, num_channels, output_dim, fixed_sigma):
        super().__init__()
        self.fixed_sigma = fixed_sigma

        self.mu = nn.Linear(num_channels, output_dim)
        self.mu_act = nn.Identity()
        initialize_layers([self.mu])

        if fixed_sigma:
            self.log_sigma = nn.Parameter(
                torch.zeros(output_dim, requires_grad=True)
            )
            torch.nn.init.zeros_(self.log_sigma)
        else:
            self.log_sigma = nn.Linear(num_channels, output_dim)
            initialize_layers([self.log_sigma])

        self.sigma_act = nn.Identity()

    def forward(self, x):
        mu = self.mu_act(self.mu(x))
        if self.fixed_sigma:
            log_sigma = self.sigma_act(self.log_sigma)
        else:
            log_sigma = self.sigma_act(self.log_sigma(x))
        return NormalDistribution(mu, log_sigma)


class CriticHead(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.value = nn.Linear(num_channels, 1)
        self.value_act = nn.Identity()
        initialize_layers([self.value])

    def forward(self, x):
        return self.value_act(self.value(x)).squeeze(-1)


class ContinuousAC(nn.Module):
    def __init__(self, cfg: AgentConfig, input_dim, output_dim):
        super().__init__()
        self.backbone = BackboneMLP(input_dim, cfg.backbone_dims,
                                    cfg.activation_fn)
        # actor head
        last_latent_dim = cfg.backbone_dims[-1]
        self.actor = ActorHead(last_latent_dim, output_dim, cfg.fixed_sigma)
        # critic head
        self.critic = CriticHead(cfg.backbone_dims[-1])
        # normalizers
        self.obs_norm = RunningMeanStd(input_dim, clamp=5.0)
        self.value_norm = RunningMeanStd(1, clamp=5.0)

    def reset_noise(self):
        self.actor.reset_noise()

    def act(self, obs, best=False):
        obs_norm = self.obs_norm(obs)
        latent = self.backbone(obs_norm)

        pi = self.actor(latent)
        a = pi.mean() if best else pi.sample()
        lp = pi.log_prob(a)

        v = self.critic(latent)
        return a, lp, v

    def unnorm_value(self, value):
        return self.value_norm(value, unnorm=True)

    def evaluate(self, obs):
        obs_norm = self.obs_norm(obs)
        latent = self.backbone(obs_norm)
        v = self.critic(latent)
        return v

    def forward(self, obs):
        obs_norm = self.obs_norm(obs)
        latent = self.backbone(obs_norm)

        pi = self.actor(latent)
        v = self.critic(latent)
        return pi, v

    def save(self, path):
        torch.save(self.state_dict(), path)
        return

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        return


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SoftQNetwork(nn.Module):
    def __init__(self, cfg: AgentConfig, obs_dim, act_dim):
        super().__init__()

        layer_dims = ([obs_dim + act_dim] + list(cfg.q_dims) + [1])
        layers = []
        for i, (in_d, out_d) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i != len(layer_dims[:-1]) - 1:
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                if cfg.layer_norm:
                    layers.append(nn.LayerNorm(out_d))
                layers.append(get_activation(cfg.sac_activation_fn))
        self.net = nn.Sequential(*layers)

        self.apply(weight_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


class SACActor(nn.Module):
    def __init__(self, cfg: AgentConfig, obs_dim, act_dim):
        super().__init__()

        self.obs_norm = RunningMeanStd(obs_dim, clamp=5.0)

        layer_dims = [obs_dim] + list(cfg.a_dims)
        layers = []
        for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(get_activation(cfg.sac_activation_fn))
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(layer_dims[-1], act_dim)
        self.log_std = nn.Linear(layer_dims[-1], act_dim)

        # action rescaling
        h, l = 1.0, -1.0
        self.register_buffer(
            "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
        )

        self._log_std_min = cfg.log_std_min
        self._log_std_max = cfg.log_std_max

        self.apply(weight_init)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (
                self._log_std_max - self._log_std_min) * (
                          log_std + 1
                  )
        return mean, log_std

    def get_action(self, x, norm_obs=True):
        if norm_obs:
            x = self.obs_norm(x)
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
