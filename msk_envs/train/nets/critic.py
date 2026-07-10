import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            use_layer_norm: bool,
            device: torch.device,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        x = self.fc_head(x)
        return x


class Critic(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            hidden_dim: int,
            use_layer_norm: bool,
            device: torch.device,
            num_q_networks: int = 2,
    ):
        super().__init__()

        self.n_obs = n_obs
        self.n_act = n_act
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.num_q_networks = num_q_networks
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        assert num_q_networks >= 1, "Number of Q networks must be at least 1"
        self.setup_qnetworks()
        return

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        self._setup_qnetworks_with_obs_dim(self.n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                QNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """ Returns Q(s,a) for each Q-network """
        x = obs
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)  # (num_q_networks, batch, 1)

    def q_value(self, obs: torch.Tensor, actions: torch.Tensor, use_cdq: bool) -> torch.Tensor:
        """ Returns Q(s,a) """
        q_values = self.forward(obs, actions).squeeze(-1)  # (num_q_networks, batch)
        return q_values.amin(dim=0) if use_cdq else q_values.mean(dim=0)  # (batch)
