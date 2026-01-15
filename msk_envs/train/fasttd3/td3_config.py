from dataclasses import dataclass


@dataclass
class TD3Config:
    agent: str = "simbav2"  # fasttd3, simbav2

    """ device/torch settings """
    checkpoint_path: str = ""
    amp: bool = True
    amp_dtype: str = "bf16"

    """ evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 1000

    """ learning rates """
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate_end: float = 3e-5
    actor_learning_rate_end: float = 3e-5
    weight_decay: float = 0.001

    use_grad_norm_clipping: bool = False
    max_grad_norm: float = 0.0

    """ TD3 hyperparameters """
    num_envs: int = 8192
    total_timesteps: int = 25000
    learning_starts: int = 10
    num_updates: int = 8  # number of updates per step
    policy_frequency: int = 4  # frequency of training policy (delayed)

    buffer_size: int = 256  # (per env)
    num_steps: int = 1  # n value of n-step returns
    gamma: float = 0.99
    tau: float = 0.125  # target smoothing coefficient
    batch_size: int = 8192

    """ Policy hyperparameters """
    actor_hidden_dim: int = 512
    init_scale: float = 0.01  # scale of initial weights
    policy_noise: float = 0.001  # scale of target action noise
    noise_clip: float = 0.5  # clip range for target action noise

    """ Exploration hyperparameters """
    std_min: float = 0.01  # minimum scale of exploration noise
    std_max: float = 0.05  # maximum scale of exploration noise
    use_gsde: bool = False  # whether to use generalized state-dependent exploration (gSDE)
    gsde_steps: int = 10  # number of steps to sample new noise for gSDE

    """ Q/Value function hyperparameters
    Distributional critic outputs logits over linspace(v_min, v_max, num_atoms)
    """
    critic_hidden_dim: int = 768
    num_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    use_cdq: bool = False  # whether to use Clipped Double Q-learning
    disable_bootstrap: bool = False  # whether to disable bootstrap in the critic learning

    """ Normalization """
    obs_normalization: bool = True
    reward_normalization: bool = True  # uses v_min, v_max

    """ Miscellaneous """
    save_interval: int = 1000

    compile: bool = True
    """whether to use torch.compile."""
    # compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    compile_mode: str = "max-autotune"
    compile_backend: str = "inductor"  # "eager" is slower but safer

    """ SimbaV2 """
    critic_num_blocks: int = 2
    actor_num_blocks: int = 1

    @staticmethod
    def pretty_print(td3_config):
        print("TD3 Configuration:")
        for field in td3_config.__dataclass_fields__:
            value = getattr(td3_config, field)
            print(f"  {field}: {value}")
