from dataclasses import dataclass


@dataclass
class QFlexConfig:
    num_envs: int = 1024
    """number of parallel environments"""

    num_learning_iterations: int = 150000
    """total timesteps of the experiments"""

    learning_rate: float = 3e-4
    """learning rate"""

    alpha_learning_rate: float = 3e-4
    """alpha learning rate"""

    buffer_size: int = 256 * 2
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.99
    """the discount factor"""

    tau: float = 1.0
    """the soft update coefficient"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    hidden_dim: int = 256
    hidden_num: int = 3

    diffusion_steps: int = 20
    """number of diffusion/flow steps"""

    learn_reference_gn: bool = True
    """whether to learn reference (Gaussian) guidance network"""

    grad_step_size: float = 1e-2
    grad_step_num: int = 20

    compile: bool = True
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    compile_backend: str = "inductor"  # "eager" is slower but safer

    logging_interval: int = 100
    """ the interval to log the metrics """

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 1000

    @staticmethod
    def pretty_print(qflex_config):
        print("TD3 Configuration:")
        for field in qflex_config.__dataclass_fields__:
            value = getattr(qflex_config, field)
            print(f"  {field}: {value}")
