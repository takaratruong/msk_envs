import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.simba import SimbaActor


class DistributionalQNetwork(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
            hidden_dim: int,
            use_layer_norm: bool = True,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x  # noqa: RET504

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: torch.Tensor,
            q_support: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
            hidden_dim: int,
            use_layer_norm: bool = True,
            num_q_networks: int = 2,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        self.setup_qnetworks()

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        self._setup_qnetworks_with_obs_dim(self.n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    num_atoms=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = obs
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        x = obs
        projections = [
            qnet.projection(
                x,
                actions,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)


class Actor(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_envs: int,
            hidden_dim: int,
            std_min: float,
            std_max: float,
            use_tanh: bool = True,
            use_layer_norm: bool = True,
            use_gsde: bool = False,
            gsde_steps: int = 10,
            device: torch.device = None,
            action_scale: torch.Tensor | None = None,
            action_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
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

        # Exploration noise scales per environment
        noise_scales = (torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min)
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.register_buffer("noise", torch.zeros(num_envs, n_act, device=device))
        self.register_buffer("gsde_step_count", torch.zeros(1, device=device, dtype=torch.int32))
        self.use_gsde = use_gsde
        self.gsde_steps = gsde_steps

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
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mu(x)
        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean
        return action

    def explore(self, obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        # Generate new noise scales for done environments
        if dones is not None and dones.sum() > 0:
            new_scales = (torch.rand(self.n_envs, 1, device=obs.device) *
                          (self.std_max - self.std_min) + self.std_min)
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        # add noise to mean
        act = self(obs)

        # Sample noise every gsde_steps or every step
        if self.use_gsde:
            resample_noise = (self.gsde_step_count % self.gsde_steps) == 0
            new_noise = torch.randn_like(act) * self.noise_scales
            self.noise.copy_(torch.where(resample_noise, new_noise, self.noise))
            self.gsde_step_count += 1
        else:
            self.noise.copy_(torch.randn_like(act) * self.noise_scales)

        return act + self.noise


# The following are used for inference
class Policy(nn.Module):
    def __init__(self, n_obs: int, n_act: int, args: dict, agent: str):
        super().__init__()

        self.args = args

        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=args["num_envs"],
            device="cpu",
            hidden_dim=args["actor_hidden_dim"],
            std_min=args["std_min"],
            std_max=args["std_max"],
            use_tanh=args["use_tanh"],
            use_layer_norm=args["use_layer_norm"],
            action_scale=args["action_scale"] if "action_scale" in args else None,
            action_bias=args["action_bias"] if "action_bias" in args else None,
        )
        actor_hidden_dim = args["actor_hidden_dim"]

        if agent == "fasttd3":
            actor_cls = Actor
        elif agent == "simbav2":
            actor_cls = SimbaActor
            actor_num_blocks = args["actor_num_blocks"]
            actor_kwargs.update(
                {
                    "scaler_init": math.sqrt(2.0 / actor_hidden_dim),
                    "scaler_scale": math.sqrt(2.0 / actor_hidden_dim),
                    "alpha_init": 1.0 / (actor_num_blocks + 1),
                    "alpha_scale": 1.0 / math.sqrt(actor_hidden_dim),
                    "expansion": 4,
                    "c_shift": 3.0,
                    "num_blocks": actor_num_blocks,
                }
            )
        else:
            raise ValueError(f"Agent {agent} not supported")

        self.actor = actor_cls(**actor_kwargs, )
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")

        self.actor.eval()
        self.obs_normalizer.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        actions = self.actor(norm_obs)
        return actions

    def act(self, obs: torch.Tensor) -> torch.distributions.Normal:
        actions = self.forward(obs)
        return torch.distributions.Normal(actions, torch.ones_like(actions) * 1e-8)


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(
        f"{checkpoint_path}", map_location="cpu", weights_only=False
    )
    args = torch_checkpoint["args"]
    agent = args.get("agent", "fasttd3")

    if agent == "fasttd3":
        n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
        n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]
    elif agent == "simbav2":
        n_obs = (
                torch_checkpoint["actor_state_dict"]["embedder.w.w.weight"].shape[-1] - 1
        )
        n_act = torch_checkpoint["actor_state_dict"]["predictor.mean_bias"].shape[0]
    else:
        raise ValueError(f"Agent {agent} not supported")

    policy = Policy(n_obs=n_obs, n_act=n_act, args=args, agent=agent)
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(
            torch_checkpoint["obs_normalizer_state"])

    return policy
