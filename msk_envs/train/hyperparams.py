from dataclasses import dataclass, asdict, fields, field
from datetime import datetime
from typing import Union
from typing_extensions import Annotated

import tyro

from msk_envs.envs.env_config import EnvConfig, EnvConfigNoHands, EnvConfigNoHandsWalk, EnvConfigNoHandsExp, \
    EnvConfigStaticLegs
from msk_envs.envs.env_variants import DerivedEnv
from msk_envs.train.fastsac.sac_config import SACConfig
from msk_envs.train.fasttd3.td3_config import TD3Config


@dataclass
class BaseArgs:
    algo: str = "td3"

    project: str = "msk_sprinter"
    exp_prefix: str = ""
    exp_name: str = ""
    disable_wandb: bool = False

    seed: int = 1
    cuda: bool = True
    gpu_id: int = 0

    td3_config: TD3Config = field(default_factory=TD3Config)
    sac_config: SACConfig = field(default_factory=SACConfig)
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
    if args.algo.lower() == "sac":
        SACConfig.pretty_print(args.sac_config)
    elif args.algo.lower() == "td3":
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
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHandsWalk(
        env_variant=DerivedEnv.WALK,
        delta_t=1.0 / 30.0,
        max_episode_duration=5.0,

        muscle_multiplier=1.0,
        toes_stiffness=25.0,
        toes_damping=1.9,
    ))

    """Walk environment specific reward scales"""
    lambda_cot: float = -1e-5
    lambda_head: float = -3e-3
    lambda_limit: float = -1e-2
    lambda_actuator: float = -1e-1
    lambda_alive: float = 1e-1


@dataclass
class JogConfig(BaseArgs):
    env_config: EnvConfigNoHands = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.JOG,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_run.yaml",
    ))

    """Walk environment specific reward scales"""
    lambda_vel: float = 1.0
    lambda_metabolic: float = -0.01
    lambda_fatigue: float = -0.05
    lambda_acc: float = -0.001
    lambda_limit: float = -0.1
    lambda_actuator: float = -1.0


@dataclass
class SprintConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_run.yaml",
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -2e-3
    lambda_damping: float = -1e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class RaceWalkConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.RACE_WALK,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_run.yaml",
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -2e-3
    lambda_damping: float = -1e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class BackPedalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.BACKPEDAL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand_backward.yaml",
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -2e-3
    lambda_damping: float = -1e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class SideShuffleConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.SIDE_SHUFFLE,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand_side.yaml",
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -2e-3
    lambda_damping: float = -1e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class HopConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.HOP,
        delta_t=1.0 / 100.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_left_leg_up.yaml",
        swap_lr=False,
    ))

    """Sprint environment specific reward scales"""
    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -2e-3
    lambda_damping: float = -1e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class VerticalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.VERTICAL,
        delta_t=1.0 / 30.0,
        max_episode_duration=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand.yaml",
    ))

    """Vertical jump environment specific reward scales"""
    lambda_max_vertical: float = 1.0
    lambda_limit: float = -0.2
    lambda_actuator: float = -5.0


@dataclass
class PerturbConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.PERTURB,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand.yaml",
    ))
    lambda_alive: float = 1e-1
    lambda_limit: float = -2e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = -2e-2
    lambda_metabolic: float = -2e-3


@dataclass
class DontFallConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.DONT_FALL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        noise_start=False,
    ))
    lambda_alive: float = 0
    lambda_limit: float = 0
    lambda_actuator: float = 0
    lambda_fatigue: float = 0
    lambda_metabolic: float = 0
    lambda_root_zero: float = 0
    lambda_match_start: float = 1e-1


@dataclass
class ImitateConfig(BaseArgs):
    """ Default Imitate environment configuration. Tracks a two-step sprinting motion. """
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        env_variant=DerivedEnv.IMITATE,
        delta_t=1.0 / 500.0,
        use_prescribed_starting_activations=True,
        starting_activations_path="../msk_models/starting_activations.yaml",
        use_specified_joint_limits=True,
        joint_limits_path="../msk_models/joint_limits_sprinting.yaml",
        motion_name="../motions/pred_sprint_two_step.mot",
    ))

    td3_config: TD3Config = field(default_factory=lambda: TD3Config(
        reward_normalization=True,
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


@dataclass
class StaticSquatConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigNoHands(
        env_variant=DerivedEnv.STATIC,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand.yaml",
    ))

    lambda_track: float = 0.1
    imitation_weight_track: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        self.env_config.target_position = [0.0, 0.9, 0.0]


@dataclass
class AnkleFlexConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigStaticLegs(
        env_variant=DerivedEnv.ANKLE_FLEX,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
    ))

    lambda_test: float = 0.1


Config = Union[
    Annotated[WalkConfig, tyro.conf.subcommand(name="walk")],
    Annotated[JogConfig, tyro.conf.subcommand(name="jog")],
    Annotated[SprintConfig, tyro.conf.subcommand(name="sprint")],
    Annotated[RaceWalkConfig, tyro.conf.subcommand(name="racewalk")],
    Annotated[BackPedalConfig, tyro.conf.subcommand(name="backpedal")],
    Annotated[SideShuffleConfig, tyro.conf.subcommand(name="sideshuffle")],
    Annotated[HopConfig, tyro.conf.subcommand(name="hop")],
    Annotated[VerticalConfig, tyro.conf.subcommand(name="vertical")],
    Annotated[PerturbConfig, tyro.conf.subcommand(name="perturb")],
    Annotated[ImitateConfig, tyro.conf.subcommand(name="imitate")],
    Annotated[DontFallConfig, tyro.conf.subcommand(name="dontfall")],
    Annotated[StaticSquatConfig, tyro.conf.subcommand(name="static")],
    Annotated[AnkleFlexConfig, tyro.conf.subcommand(name="ankleflex")],
]


def get_args():
    """Parse command-line arguments into Config dataclass."""
    args = tyro.cli(Config)
    args.use_wandb = not args.disable_wandb
    return args
