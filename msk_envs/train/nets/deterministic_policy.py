import math
from typing import Optional

import torch
import torch.nn as nn

from msk_envs.train.nets.normalizers import EmpiricalNormalization, SimNormLinear
from msk_envs.train.nets.simba import SimbaActor


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            n_obs: int,
            n_act: int,
            num_envs: int,
            hidden_dim: int,
            std_min: float,
            std_max: float,
            sim_type: str,
            sim_dimension: int,
            seq_len: int,
            use_layer_norm: bool = True,
            use_gsde: bool = False,
            gsde_steps: int = 10,
            device: torch.device = None,
    ):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.sim_type = sim_type
        self.sim_dimension = sim_dimension
        self.seq_len = seq_len

        # This will be overridden in subclasses if needed
        self.setup_network()

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
        )
        if self.sim_type in ["sim_both", "sim_actor"]:
            self.fc_head = SimNormLinear(
                self.hidden_dim // 2,
                seq_len=self.seq_len,
                simnorm_dim=self.sim_dimension,
                device=self.device,
            )
            self.fc_mu = nn.Sequential(
                nn.Linear(self.seq_len * self.sim_dimension, self.n_act, device=self.device),
                nn.Tanh(),
            )
        else:
            self.fc_head = nn.Sequential(
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
                nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
                nn.SiLU(),
            )
            self.fc_mu = nn.Sequential(
                nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
                nn.Tanh(),
            )
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x_net = self.net(obs)
        x_head = self.fc_head(x_net)
        action = self.fc_mu(x_head)
        return action

    def _sample_new_noise(self, dones: Optional[torch.Tensor]):
        """ Generate new exploration noise """
        # Generate new noise scales for done environments
        if dones is not None and dones.sum() > 0:
            new_scales = (torch.rand(self.n_envs, 1, device=self.device) *
                          (self.std_max - self.std_min) + self.std_min)
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(torch.where(dones_view, new_scales, self.noise_scales))

        # Sample noise every gsde_steps or every step
        if self.use_gsde:
            resample_noise = (self.gsde_step_count % self.gsde_steps) == 0
            new_noise = torch.randn_like(self.noise) * self.noise_scales
            self.noise.copy_(torch.where(resample_noise, new_noise, self.noise))
            self.gsde_step_count += 1
        else:
            self.noise.copy_(torch.randn_like(self.noise) * self.noise_scales)
        return

    def explore(self, obs: torch.Tensor, dones: Optional[torch.Tensor]) -> torch.Tensor:
        self._sample_new_noise(dones)
        act = self(obs)
        return act + self.noise

    def explore_synergistic(
            self,
            obs: torch.Tensor,
            dones: Optional[torch.Tensor],
            max_isometric_force: torch.Tensor,
            active_length_multiplier: torch.Tensor,
            active_velocity_multiplier: torch.Tensor,
            moment_arms: torch.Tensor
    ) -> torch.Tensor:
        act = self(obs)

        self._sample_new_noise(dones)

        # Grab "joint" noise: [envs, n_qpos]
        _, n_muscles, n_qpos = moment_arms.shape
        noise_joints = self.noise[:, :n_qpos]

        # Compute relative strength/sensitivity of muscles
        scaled_isometric_forces = max_isometric_force / torch.mean(max_isometric_force)
        fal, fav = active_length_multiplier, active_velocity_multiplier
        W = (scaled_isometric_forces.view(1, n_muscles) * fal * fav).unsqueeze(-1)  # [n_envs, n_muscles, 1]
        # Compute muscle torque capacity matrix
        R = moment_arms  # [n_envs, n_muscles, n_qpos]
        G = R * W  # [n_envs, n_muscles, n_qpos]
        # Map joint noise to muscle noise
        GtG = torch.bmm(G.transpose(1, 2), G)
        GtG_diag_mean = torch.mean(torch.diagonal(GtG, dim1=1, dim2=2), dim=1, keepdim=True)
        GtG_damped = GtG + (1e-4 * GtG_diag_mean[:, None] * torch.eye(n_qpos, device=GtG.device))
        y = torch.linalg.solve(GtG_damped, noise_joints.unsqueeze(-1))  # [n_envs, n_qpos, 1]
        muscle_noise = torch.bmm(G, y).squeeze(-1)
        # Match noise scales
        muscle_noise_normalized = muscle_noise / (torch.std(muscle_noise, dim=-1, keepdim=True) + 1e-8)
        muscle_noise = muscle_noise_normalized * self.noise_scales

        # Add noise
        noised_act = act.clone()
        noised_act[:, :n_muscles] += muscle_noise
        noised_act[:, n_muscles:] += self.noise[..., n_muscles:]
        return noised_act


# Used for inference only
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
            use_layer_norm=args["use_layer_norm"],
        )
        actor_hidden_dim = args["actor_hidden_dim"]

        if agent == "fasttd3":
            actor_cls = DeterministicPolicy
            actor_kwargs.update(
                {
                    "sim_type": args.get("sim_type", ""),
                    "sim_dimension": args.get("sim_dimension", 8),
                    "seq_len": args.get("actor_seq_len", 8),
                }
            )
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
        n_obs = torch_checkpoint["actor_state_dict"]["embedder.w.w.weight"].shape[-1] - 1
        n_act = torch_checkpoint["actor_state_dict"]["predictor.mean_bias"].shape[0]
    else:
        raise ValueError(f"Agent {agent} not supported")

    policy = Policy(n_obs=n_obs, n_act=n_act, args=args, agent=agent)
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])

    return policy
