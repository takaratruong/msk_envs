import torch
import torch.nn as nn


class ReferencePolicy(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            layer_norm: bool,
            device: torch.device,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        return

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x_net = self.net(obs)
        x_head = self.fc_head(x_net)
        action = self.fc_mu(x_head)
        return action


class VelocityField(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            layer_norm: bool,
            device: torch.device,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act + 1, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_act, device=device),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act, t], dim=-1)
        return self.net(x)


class QFlexActor(nn.Module):
    """Reference policy + velocity field with the clipped-Euler flow sampler."""

    def __init__(
            self,
            reference: ReferencePolicy,
            velocity_field: VelocityField,
            num_timesteps: int,
            device: torch.device,
            n_act: int,
            num_envs: int,
            std_min: float,
            std_max: float,
            action_low: float,
            action_high: float,
    ):
        super().__init__()
        self.reference = reference
        self.velocity_field = velocity_field
        self.num_timesteps = num_timesteps
        self.action_low = action_low
        self.action_high = action_high

        self.n_envs = num_envs
        self.device = device
        noise_scales = (torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min)
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.register_buffer("noise", torch.zeros(num_envs, n_act, device=device))

    @torch.no_grad()
    def apply_flow(self, x0: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = x0.clone()

        # Clipped forward Euler over t in linspace(0, 1, num_timesteps + 1)[1:].
        ts = torch.linspace(0.0, 1.0, self.num_timesteps + 1, device=obs.device)
        dt = ts[1] - ts[0]
        for i in range(1, self.num_timesteps + 1):
            ti = ts[i].expand(obs.shape[0], 1)
            dx = self.velocity_field(obs, x, ti)
            dx = dx.clamp(-1.0 / dt, 1.0 / dt)
            x = x + dt * dx
        return x.clamp(self.action_low, self.action_high)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample an action for environment interaction (``exp_prob = 1``)."""
        x0 = self.reference(obs)
        act = self.apply_flow(x0=x0, obs=obs)
        return act

    @torch.no_grad()
    def _sample_new_noise(self, dones):
        """ Generate new exploration noise """
        # Generate new noise scales for done environments
        if dones is not None and dones.sum() > 0:
            new_scales = (torch.rand(self.n_envs, 1, device=self.device) *
                          (self.std_max - self.std_min) + self.std_min)
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        self.noise.copy_(torch.randn_like(self.noise) * self.noise_scales)
        return

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, dones) -> torch.Tensor:
        self._sample_new_noise(dones)
        # noise output of ref policy then apply flow
        x0 = self.reference(obs) + self.noise
        act = self.apply_flow(x0=x0, obs=obs)
        return act
