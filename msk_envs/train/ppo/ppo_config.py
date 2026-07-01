from dataclasses import dataclass


@dataclass
class PPOConfig:
    num_envs: int = 2048
    """number of parallel environments"""

    num_learning_iterations: int = 20000
    """total number of policy-update iterations (each collects a full rollout)"""

    num_steps_per_env: int = 24
    """number of environment steps collected per env per rollout"""

    num_learning_epochs: int = 5
    """number of passes over the collected rollout per iteration"""

    num_mini_batches: int = 4
    """number of minibatches the rollout is split into per epoch"""

    # ------------------------------------------------------------------ optim
    actor_learning_rate: float = 3e-4
    """the learning rate of the actor"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    max_learning_rate: float = 1e-2
    """upper bound for the adaptive learning rate"""

    min_learning_rate: float = 1e-5
    """lower bound for the adaptive learning rate"""

    schedule: str = "adaptive"
    """learning-rate schedule: 'adaptive' (KL based) or 'fixed'"""

    desired_kl: float = 0.01
    """target KL divergence used by the adaptive schedule"""

    # ------------------------------------------------------------- ppo losses
    gamma: float = 0.99
    """the discount factor gamma"""

    lam: float = 0.95
    """the GAE lambda"""

    clip_param: float = 0.2
    """the PPO surrogate clipping coefficient"""

    entropy_coef: float = 0.0
    """coefficient of the entropy bonus"""

    value_loss_coef: float = 1.0
    """coefficient of the value loss"""

    use_clipped_value_loss: bool = True
    """whether to use the clipped value loss"""

    max_grad_norm: float = 1.0
    """maximum gradient norm for clipping (<= 0 disables)"""

    normalize_advantage: bool = True
    """whether to normalize advantages per rollout"""

    # ------------------------------------------------------------- networks
    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    critic_hidden_dim: int = 512
    """the hidden dimension of the critic network"""

    use_layer_norm: bool = True
    """whether to use layer normalization in the networks"""

    init_noise_std: float = 1.0
    """initial standard deviation of the (state-independent) action noise"""

    std_min: float = 0.01
    """minimum action standard deviation"""

    std_max: float = 4.0
    """maximum action standard deviation"""

    # ------------------------------------------------------------ misc/train
    obs_normalization: bool = True
    """whether to normalize observations"""

    compile: bool = False
    """whether to torch.compile the loss computation"""
    compile_mode: str | None = None
    compile_backend: str = "inductor"

    save_interval: int = 200
    """the interval (in iterations) to save the model"""

    logging_interval: int = 1
    """the interval (in iterations) to log the metrics"""

    checkpoint_path: str = ""

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 200

    @staticmethod
    def pretty_print(ppo_config):
        print("PPO Configuration:")
        for field in ppo_config.__dataclass_fields__:
            value = getattr(ppo_config, field)
            print(f"  {field}: {value}")
