from dataclasses import dataclass, asdict, fields, field
from datetime import datetime

import tyro

from msk_envs.envs.env_config import EnvConfig
from msk_envs.envs.env_variants import DerivedEnv
from msk_envs.train.fasttd3.td3_config import TD3Config


@dataclass
class BaseArgs:
    project: str = "msk_sprinter"
    exp_prefix: str = ""
    exp_name: str = ""
    disable_wandb: bool = False

    seed: int = 1
    cuda: bool = True
    gpu_id: int = 0

    env_variant: DerivedEnv = DerivedEnv.SPRINT
    td3_config: TD3Config = field(default_factory=TD3Config)
    env_config: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        """Compute derived fields after exp_prefix is set from outside"""
        date_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.exp_name = f"{self.exp_prefix}_{date_name}" if self.exp_prefix else date_name
        self.traj_out_folder = f"dashboard/trajectories/{self.exp_name}"
        self.analytics_out_folder = f"models/frame_data/{self.exp_name}"

        reward_lambdas = {k: v for k, v in self.__dict__.items() if k.startswith("lambda_")}
        imitation_weights = {k: v for k, v in self.__dict__.items() if k.startswith("imitation_weight_")}

        self.env_config.reward_lambdas = reward_lambdas
        self.env_config.imitation_weights = imitation_weights
        self.env_config.env_variant = self.env_variant


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

    EnvConfig.pretty_print(args.env_config)
    TD3Config.pretty_print(args.td3_config)

    print("BaseArgs:")
    for k in base_field_names:
        if k in ["env_config", "td3_config"]:
            continue
        print(f"  {k:24} = {base_items[k]}")

    if extra_items:
        print("-" * 80)
        print(f"{type(args).__name__} extras:")
        for k in sorted(extra_items.keys()):
            print(f"  {k:24} = {extra_items[k]}")

    print(line)


@dataclass
class WalkConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        delta_t=1.0 / 100.0,
        max_episode_duration=5.0,
    ))

    """Walk environment specific reward scales"""
    lambda_cot: float = 1.0
    lambda_head: float = 1.0
    lambda_limit: float = 0.2
    lambda_actuator: float = 1.0
    lambda_activation: float = 1.0
    lambda_alive: float = 1.0


@dataclass
class SprintConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        delta_t=1.0 / 100.0,
        max_episode_duration=12.0,
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 1.0
    lambda_limit: float = -2.0
    lambda_actuator: float = -1.0
    lambda_finish: float = 20.0


@dataclass
class VerticalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        delta_t=1.0 / 100.0,
        max_episode_duration=2.0,
    ))

    """Vertical jump environment specific reward scales"""
    lambda_max_vertical: float = 1.0
    lambda_limit: float = -0.2
    lambda_actuator: float = -5.0


@dataclass
class ImitateConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        delta_t=1.0 / 500.0,
        use_prescribed_starting_activations=True,
        joint_limits_path="../msk_models/joint_limits_sprinting.yaml"
    ))

    """Imitate environment specific reward scales"""
    lambda_track_joints: float = 1.0
    lambda_track_root_pos: float = 1.0
    lambda_track_root_rot: float = 1.0
    lambda_track_body_pos: float = 1.0
    lambda_track_body_rot: float = 1.0

    """Imitation reward weights"""
    imitation_weight_track_joints: float = 10.0
    imitation_weight_track_root_pos: float = 100.0
    imitation_weight_track_root_rot: float = 10.0
    imitation_weight_track_body_pos: float = 100.0
    imitation_weight_track_body_rot: float = 10.0

    extra_rewarded_joints: str = ""  # comma-separated list of joints to reward, default is empty
    lambda_extra_rewarded_joints: float = 0.  # This feature is disabled by default

    extra_rewarded_dofs: str = ""  # comma-separated list of DOFs to reward, default is empty
    lambda_extra_rewarded_dofs: float = 0.0  # This feature is disabled by default

    def __post_init__(self):
        super().__post_init__()
        # Convert comma-separated string to list
        if isinstance(self.extra_rewarded_joints, str):
            self.env_config.extra_rewarded_joints = [s.strip() for s in self.extra_rewarded_joints.split(",") if
                                                     s.strip()]
        if isinstance(self.extra_rewarded_dofs, str):
            self.env_config.extra_rewarded_dofs = [s.strip() for s in self.extra_rewarded_dofs.split(",") if s.strip()]


def get_args():
    """Get configuration arguments based on env_variant."""
    import sys

    # Parse env_variant first to determine which config class to use
    env_variant = DerivedEnv.SPRINT
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

    # Parse full args
    args = tyro.cli(config_class)
    args.use_wandb = not args.disable_wandb
    return args
