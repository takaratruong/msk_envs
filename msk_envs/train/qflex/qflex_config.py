from dataclasses import dataclass


@dataclass
class QFlexConfig:
    num_envs: int = 4096
    """number of parallel environments"""

    num_learning_iterations: int = 150000
    """total timesteps of the experiments"""

    learning_rate: float = 3e-4
    """learning rate"""

    buffer_size: int = 256 * 20
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
    num_updates: int = 2
    """the number of updates to perform per step"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""

    num_atoms: int = 101
    """the number of atoms"""
    v_min: float = -5.0
    """the minimum value of the support"""
    v_max: float = 15.0
    """the maximum value of the support"""
    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""
    use_layer_norm: bool = False
    """whether to use layer normalization"""
    num_q_networks: int = 2
    """number of Q-networks to ensemble"""
    policy_noise: float = 0.001
    """the scale of target action noise"""
    noise_clip: float = 0.5
    """the clip range of target action noise"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    velocity_hidden_dim: int = 768
    """hidden dimension of velocity network"""

    obs_normalization: bool = True
    """ whether to normalize observations """

    num_flow_steps: int = 20
    """Euler steps for sampling from the flow (``diffusion_steps``)."""
    grad_step_size: float = 1e-2
    """step size of the Q-gradient ascent that builds the flow target."""
    grad_step_num: int = 20
    """number of Q-gradient ascent steps."""

    compile: bool = False
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    compile_backend: str = "inductor"  # "eager" is slower but safer

    save_interval: int = 1000
    """ the interval to save the model """
    logging_interval: int = 100
    """ the interval to log the metrics """

    """ Evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 1000

    @staticmethod
    def pretty_print(qflex_config):
        print("QFlex Configuration:")
        for field in qflex_config.__dataclass_fields__:
            value = getattr(qflex_config, field)
            print(f"  {field}: {value}")
