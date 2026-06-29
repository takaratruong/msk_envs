from dataclasses import dataclass, field

from .agent_config import AgentConfig


@dataclass
class PPOConfig:
    num_envs: int = 2048
    num_iterations: int = 100_000
    num_rollout_steps: int = 8
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.97
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 5

    rewards_scale: float = 1.0

    num_eval_envs: int = 1
    eval_freq: int = 100

    # Losses
    clip_coef: float = 0.2
    c_coef: float = 4.0
    ent_coef: float = 0.001
    bounds_loss_coef: float = 0.0001
    max_grad_norm: float = 1.0
    target_kl: float = None

    agent_config: AgentConfig = field(default_factory=AgentConfig)

    @staticmethod
    def pretty_print(td3_config):
        print("PPO Configuration:")
        for fld in td3_config.__dataclass_fields__:
            value = getattr(td3_config, fld)
            print(f"  {fld}: {value}")
