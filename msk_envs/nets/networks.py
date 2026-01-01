import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from msk_envs.nets.normalizers import EmpiricalNormalization
from msk_envs.nets.simba import SimbaActor


class DistributionalQNetwork(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_atoms: int,
            v_min: float,
            v_max: float,
            hidden_dim: int,
            device: torch.device,
    ):
        super().__init__()
        # (obs, action) -> logits over support
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: float,
            q_support: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        # see categorical algorithm in https://arxiv.org/pdf/1707.06887
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
                rewards.unsqueeze(1)
                + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        is_int = (l == u)
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
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
            device: torch.device = None,
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

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            bootstrap: torch.Tensor,
            discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_envs: int,
            init_scale: float,
            hidden_dim: int,
            std_min: float,
            std_max: float,
            use_gsde: bool = False,
            gsde_steps: int = 10,
            device: torch.device = None,
    ):
        super().__init__()
        # obs -> action mean
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        # Exploration noise scales per environment
        noise_scales = (torch.rand(num_envs, 1, device=device) *
                        (std_max - std_min) + std_min)
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.register_buffer("noise", torch.zeros(num_envs, n_act, device=device))
        self.register_buffer("gsde_step_count", torch.zeros(1, device=device, dtype=torch.int32))
        self.n_envs = num_envs
        self.use_gsde = use_gsde
        self.gsde_steps = gsde_steps
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        action = self.fc_mu(x)
        return action

    def explore(self, obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        # Generate new noise scales for done environments
        if dones is not None and dones.sum() > 0:
            new_scales = (
                    torch.rand(self.n_envs, 1, device=obs.device)
                    * (self.std_max - self.std_min)
                    + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales))

        # add noise to mean
        act = self(obs)

        if self.use_gsde:
            resample_noise = (self.gsde_step_count % self.gsde_steps) == 0
            self.gsde_step_count += 1
            new_noise = torch.randn_like(act) * self.noise_scales
            self.noise.copy_(torch.where(resample_noise, new_noise, self.noise))
        else:
            self.noise.copy_(torch.randn_like(act) * self.noise_scales)

        return act + self.noise


# The following are used for inference
class Policy(nn.Module):
    def __init__(self, n_obs: int, n_act: int, args: dict, agent: str):
        super().__init__()

        self.args = args

        num_envs = args["num_envs"]
        init_scale = args["init_scale"]
        actor_hidden_dim = args["actor_hidden_dim"]
        std_min = args["std_min"]
        std_max = args["std_max"]

        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device="cpu",
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            std_min=std_min,
            std_max=std_max,
        )

        if agent == "fasttd3":
            actor_cls = Actor
        elif agent == "simbav2":
            actor_cls = SimbaActor
            actor_num_blocks = args["actor_num_blocks"]
            actor_kwargs.pop("init_scale")
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
        return torch.distributions.Normal(actions,
                                          torch.ones_like(actions) * 1e-8)


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
