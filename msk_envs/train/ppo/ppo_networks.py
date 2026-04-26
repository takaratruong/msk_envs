from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from .ppo_config import ModuleConfig


def build_mlp_layer(input_dim, hidden_dims, output_dim, layer_config, ):
    """Builds a multi-layer perceptron (MLP) layer. """
    if hidden_dims is None:
        return None

    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob

    if len(hidden_dims) == 0:
        # No hidden layer, just one linear layer
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


class BaseModule(nn.Module):
    def __init__(self, input_dim, output_dim, module_config_dict: ModuleConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._build_network_layer(module_config_dict)

    def _build_network_layer(self, module_config: ModuleConfig):
        layer_config = module_config.layer_config
        self.module = build_mlp_layer(
            self.input_dim,
            layer_config.hidden_dims,
            self.output_dim,
            layer_config,
        )

    def forward(self, policy_input):
        # Only forward the MLP layer
        return self.module(policy_input)


class PPOActor(nn.Module):
    def __init__(
            self,
            n_obs,
            n_act,
            module_config_dict: ModuleConfig,
            init_noise_std,
    ):
        super().__init__()
        self.actor_module = BaseModule(n_obs, n_act, module_config_dict)

        self.std = nn.Parameter(init_noise_std * torch.ones(n_act))
        self.min_noise_std = module_config_dict.min_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        print(f"Actor Module: {self.actor_module.module}")

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        if self.min_noise_std:
            clamped_std = torch.clamp(self.std, min=self.min_noise_std)
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        elif self.min_mean_noise_std:
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                clamped_std = self.std * scale_up
            else:
                clamped_std = self.std
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        else:
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_state_dict):
        self.update_distribution(policy_state_dict["obs"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        return self.actor(policy_state_dict["obs"])

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, n_obs, module_config_dict):
        super().__init__()
        self.critic_module = BaseModule(n_obs, 1, module_config_dict)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        obs = policy_state_dict["obs"]
        return self.critic(obs)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass
