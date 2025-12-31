from dataclasses import dataclass, asdict, fields
from datetime import datetime

import tyro
from msk_envs.envs.env_variants import DerivedEnv


@dataclass
class BaseArgs:
    """wandb configuration"""
    disable_wandb: bool = False
    env_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project: str = "msk_sprinter"
    exp_prefix: str = ""

    agent: str = "simbav2"  # fasttd3, simbav2

    """ device/torch settings """
    seed: int = 1
    cuda: bool = True
    gpu_id: int = 0
    checkpoint_path: str = ""
    amp: bool = True
    amp_dtype: str = "bf16"

    """ evaluation """
    num_eval_envs: int = 1
    eval_freq: int = 2000

    """ learning rates """
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate_end: float = 3e-5
    actor_learning_rate_end: float = 3e-5
    weight_decay: float = 0.0

    use_grad_norm_clipping: bool = False
    max_grad_norm: float = 0.0

    """ TD3 hyperparameters """
    num_envs: int = 2048
    total_timesteps: int = 150000
    learning_starts: int = 0
    num_updates: int = 4  # number of updates per step
    policy_frequency: int = 2  # frequency of training policy (delayed)

    buffer_size: int = 256 * 4  # (per env)
    num_steps: int = 3  # n value of n-step returns
    gamma: float = 0.997
    tau: float = 0.1  # target smoothing coefficient
    batch_size: int = 8192

    """ Policy hyperparameters """
    actor_hidden_dim: int = 256
    init_scale: float = 0.01  # scale of initial weights
    policy_noise: float = 0.001  # scale of target action noise
    noise_clip: float = 0.5  # clip range for target action noise

    """ Exploration hyperparameters """
    std_min: float = 0.001  # minimum scale of exploration noise
    std_max: float = 0.4  # maximum scale of exploration noise
    use_gsde: bool = True  # whether to use generalized state-dependent exploration (gSDE)
    gsde_steps: int = 10  # number of steps to sample new noise for gSDE

    """ Q/Value function hyperparameters
    Distributional critic outputs logits over linspace(v_min, v_max, num_atoms)
    """
    critic_hidden_dim: int = 512
    num_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    use_cdq: bool = True  # whether to use Clipped Double Q-learning
    disable_bootstrap: bool = False  # whether to disable bootstrap in the critic learning

    """ Normalization """
    obs_normalization: bool = True
    reward_normalization: bool = True  # uses v_min, v_max

    """ Miscellaneous """
    save_interval: int = 1000

    compile: bool = True
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"  # "max-autotune" can fail on some GPU architectures
    # compile_mode: str = "max-autotune"
    compile_backend: str = "inductor"  # "eager" is slower but safer

    """ SimbaV2 """
    critic_num_blocks: int = 2
    actor_num_blocks: int = 1

    """ Environment configuration """
    env_variant: DerivedEnv = DerivedEnv.SPRINT
    render: bool = False  # Enable rendering (headless by default)

    exp_name: str = ""
    traj_out_folder: str = ""
    analytics_out_folder: str = ""

    def __post_init__(self):
        """Compute derived fields after exp_prefix is set from outside"""
        self.exp_name = f"{self.exp_prefix}_{self.env_name}" if self.exp_prefix else self.env_name
        self.traj_out_folder = f"dashboard/trajectories/{self.exp_name}"
        self.analytics_out_folder = f"models/frame_data/{self.exp_name}"

    def get_reward_lambdas(self):
        """Extract all lambda_* fields as a dictionary"""
        return {k: v for k, v in self.__dict__.items() if
                k.startswith("lambda_")}

    def get_imitation_weights(self):
        """Extract all lambda_track_* fields as a dictionary"""
        return {k: v for k, v in self.__dict__.items() if
                k.startswith("imitation_weight_")}


def pretty_print_base_args(args: BaseArgs):
    """Nicely print BaseArgs (and env‑specific overrides) at experiment start."""
    args_dict = asdict(args)

    base_field_names = [f.name for f in fields(BaseArgs)]
    base_items = {k: args_dict.get(k) for k in base_field_names}
    extra_items = {k: v for k, v in args_dict.items() if k not in base_items}

    line = "=" * 80
    print(line)
    print(f"Experiment config: {args_dict.get('exp_name', '')}  "
          f"(env_variant={args_dict.get('env_variant')})")
    print(line)

    print("BaseArgs:")
    for k in base_field_names:
        print(f"  {k:24} = {base_items[k]}")

    if extra_items:
        print("-" * 80)
        print(f"{type(args).__name__} extras:")
        for k in sorted(extra_items.keys()):
            print(f"  {k:24} = {extra_items[k]}")

    print(line)


@dataclass
class WalkConfig(BaseArgs):
    """Walk environment specific reward scales"""
    lambda_cot: float = 1.0
    lambda_head: float = 1.0
    lambda_limit: float = 0.2
    lambda_actuator: float = 1.0
    lambda_activation: float = 1.0
    lambda_alive: float = 1.0


@dataclass
class SprintConfig(BaseArgs):
    """Sprint environment specific reward scales"""
    lambda_vel: float = 1.0
    lambda_limit: float = -1.0
    lambda_actuator: float = -1.0
    lambda_finish: float = 20.0


@dataclass
class VerticalConfig(BaseArgs):
    """Vertical jump environment specific reward scales"""
    lambda_max_vertical: float = 1.0
    lambda_limit: float = -0.2
    lambda_actuator: float = -5.0


@dataclass
class ImitateConfig(BaseArgs):
    """Imitate environment specific reward scales"""
    # Weights are used for exp(-weight * error)
    # Lambdas are used to scale the reward terms

    # Joint angle weights
    imitation_weight_hip_flexion: float = 1.0
    imitation_weight_hip_adduction: float = 1.0
    imitation_weight_hip_rotation: float = 1.0
    imitation_weight_knee_angle: float = 1.0
    imitation_weight_ankle_angle: float = 1.0
    imitation_weight_subtalar_angle: float = 1.0
    imitation_weight_mtp_angle: float = 1.0
    imitation_weight_lumbar_extension: float = 1.0
    imitation_weight_lumbar_bending: float = 1.0
    imitation_weight_lumbar_rotation: float = 1.0
    imitation_weight_arm_flex: float = 1.0
    imitation_weight_arm_add: float = 1.0
    imitation_weight_arm_rot: float = 1.0
    imitation_weight_elbow_flex: float = 1.0
    imitation_weight_pro_sup: float = 1.0
    imitation_weight_wrist_flex: float = 1.0
    imitation_weight_wrist_dev: float = 1.0
    # Body position and rotation weights
    imitation_weight_pelvis_pos: float = 1.0
    imitation_weight_pelvis_rot: float = 1.0
    imitation_weight_femur_pos: float = 1.0
    imitation_weight_femur_rot: float = 1.0
    imitation_weight_tibia_pos: float = 1.0
    imitation_weight_tibia_rot: float = 1.0
    imitation_weight_talus_pos: float = 1.0
    imitation_weight_talus_rot: float = 1.0
    imitation_weight_calcn_pos: float = 1.0
    imitation_weight_calcn_rot: float = 1.0
    imitation_weight_toes_pos: float = 1.0
    imitation_weight_toes_rot: float = 1.0
    imitation_weight_torso_pos: float = 1.0
    imitation_weight_torso_rot: float = 1.0
    imitation_weight_humerus_pos: float = 1.0
    imitation_weight_humerus_rot: float = 1.0
    imitation_weight_ulna_pos: float = 1.0
    imitation_weight_ulna_rot: float = 1.0
    imitation_weight_radius_pos: float = 1.0
    imitation_weight_radius_rot: float = 1.0
    imitation_weight_hand_pos: float = 1.0
    imitation_weight_hand_rot: float = 1.0

    # Joint angle lambdas
    lambda_track_hip_flexion: float = 1.0
    lambda_track_hip_adduction: float = 1.0
    lambda_track_hip_rotation: float = 1.0
    lambda_track_knee_angle: float = 1.0
    lambda_track_ankle_angle: float = 1.0
    lambda_track_subtalar_angle: float = 1.0
    lambda_track_mtp_angle: float = 1.0
    lambda_track_lumbar_extension: float = 1.0
    lambda_track_lumbar_bending: float = 1.0
    lambda_track_lumbar_rotation: float = 1.0
    lambda_track_arm_flex: float = 1.0
    lambda_track_arm_add: float = 1.0
    lambda_track_arm_rot: float = 1.0
    lambda_track_elbow_flex: float = 1.0
    lambda_track_pro_sup: float = 1.0
    lambda_track_wrist_flex: float = 1.0
    lambda_track_wrist_dev: float = 1.0
    # Body position and rotation lambdas
    lambda_track_pelvis_pos: float = 1.0
    lambda_track_pelvis_rot: float = 1.0
    lambda_track_femur_pos: float = 1.0
    lambda_track_femur_rot: float = 1.0
    lambda_track_tibia_pos: float = 1.0
    lambda_track_tibia_rot: float = 1.0
    lambda_track_talus_pos: float = 1.0
    lambda_track_talus_rot: float = 1.0
    lambda_track_calcn_pos: float = 1.0
    lambda_track_calcn_rot: float = 1.0
    lambda_track_toes_pos: float = 1.0
    lambda_track_toes_rot: float = 1.0
    lambda_track_torso_pos: float = 1.0
    lambda_track_torso_rot: float = 1.0
    lambda_track_humerus_pos: float = 1.0
    lambda_track_humerus_rot: float = 1.0
    lambda_track_ulna_pos: float = 1.0
    lambda_track_ulna_rot: float = 1.0
    lambda_track_radius_pos: float = 1.0
    lambda_track_radius_rot: float = 1.0
    lambda_track_hand_pos: float = 1.0
    lambda_track_hand_rot: float = 1.0


def get_args():
    """Get configuration arguments based on env_variant."""
    import sys

    # Parse env_variant first to determine which config class to use
    env_variant = DerivedEnv.IMITATE
    for i, arg in enumerate(sys.argv):
        if arg == "--env-variant" and i + 1 < len(sys.argv):
            env_variant = DerivedEnv[sys.argv[i + 1]]
            break

    # Select config class based on env_variant
    config_class = {
        DerivedEnv.WALK: WalkConfig,
        DerivedEnv.SPRINT: SprintConfig,
        DerivedEnv.VERTICAL: VerticalConfig,
        DerivedEnv.IMITATE: ImitateConfig,
    }.get(env_variant, SprintConfig)

    args = tyro.cli(config_class)
    args.env_variant = env_variant
    args.use_wandb = not args.disable_wandb
    return args
