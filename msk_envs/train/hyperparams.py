from dataclasses import dataclass, asdict, fields, field
from datetime import datetime
from typing import Union

import tyro
from typing_extensions import Annotated

from msk_envs.envs.env_config import EnvConfig, EnvConfigSprinter
from msk_envs.envs.env_variants import DerivedEnv
from msk_envs.train.dep.dep_config import DEPConfig
from msk_envs.train.fastsac.sac_config import SACConfig
from msk_envs.train.fasttd3.td3_config import TD3Config
from msk_envs.train.qflex.qflex_config import QFlexConfig
from msk_envs.utils.train_utils import find_latest_checkpoint


@dataclass
class BaseArgs:
    algo: str = "td3"

    project: str = "msk_sprinter"
    exp_prefix: str = ""
    exp_name: str = ""
    disable_wandb: bool = False
    resume: bool = False
    override_wandb_config: bool = False

    use_dep: bool = False

    seed: int = 1
    gpu_id: int = 0

    td3_config: TD3Config = field(default_factory=TD3Config)
    sac_config: SACConfig = field(default_factory=SACConfig)
    qflex_config: QFlexConfig = field(default_factory=QFlexConfig)
    dep_config: DEPConfig = field(default_factory=DEPConfig)
    env_config: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        """Compute derived fields after exp_prefix is set from outside"""
        # Handle resume logic
        if self.resume and self.exp_prefix:
            checkpoint_path, found_exp_name, global_step = find_latest_checkpoint(self.exp_prefix)
            if checkpoint_path:
                self.exp_name = found_exp_name
                if self.algo.lower() == "sac":
                    self.sac_config.checkpoint_path = checkpoint_path
                elif self.algo.lower() == "td3":
                    self.td3_config.checkpoint_path = checkpoint_path
                print(f"Resuming training from checkpoint: {checkpoint_path} at global_step={global_step}")
            else:
                print(f"No checkpoint found for exp_prefix '{self.exp_prefix}'. Starting new training.")

        # Control whether optimizer/scheduler state is included in checkpoints
        self.td3_config.save_optimizer_state = self.resume

        # Generate exp_name if not set by resume
        if not self.exp_name:
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
    return


@dataclass
class LaneConfig(BaseArgs):
    """ Reusable hyperparams for lane environments """
    lambda_vel: float = 1e-2
    lambda_mid_lane: float = 1e-2

    lambda_spring: float = 0.0
    lambda_damper: float = 0.0
    lambda_limit: float = -3e-4
    lambda_muscle_passive: float = 0.0

    lambda_actuator: float = 0.0
    lambda_fatigue: float = 0.0
    lambda_muscle_activation: float = 0.0
    lambda_metabolic: float = 0.0
    lambda_self_collision: float = 0.0
    lambda_head_acc_ang: float = 0.0
    lambda_head_acc_lin: float = 0.0


@dataclass
class SprintConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_run.yaml",
    ))


@dataclass
class SprintBlockStartConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.SPRINT_BLOCK_START,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_blockstart.yaml",
        noise_start=False,
        swap_lr=False,
        default_activation=0.01,
    ))


@dataclass
class BackpedalConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.BACKPEDAL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_backpedal.yaml",
    ))


@dataclass
class SideShuffleConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.SIDE_SHUFFLE,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_side.yaml",
    ))


@dataclass
class HurdlesConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.HURDLES,
        delta_t=1.0 / 30.0,
        max_episode_duration=12.0,
        starting_pose_path="../msk_models/poses/starting_pose_run.yaml",
    ))


@dataclass
class UphillSprintConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_run.yaml",
        ground_rotation=(0.0, 0.0, 0.2, 0.8)  # ~30 deg incline
    ))


@dataclass
class RunTheBendConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.RUN_THE_BEND,
        delta_t=1.0 / 30.0,
        max_episode_duration=12.0,
        starting_pose_path="../msk_models/poses/starting_pose_run.yaml",
    ))


@dataclass
class HopConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.HOP,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_hop.yaml",
        swap_lr=False,
    ))


@dataclass
class CariocaConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.CARIOCA,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/poses/starting_pose_carioca.yaml",
        noise_start=False,
        swap_lr=False,
    ))


@dataclass
class VerticalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigSprinter(
        env_variant=DerivedEnv.VERTICAL,
        delta_t=1.0 / 30.0,
        starting_pose_path="../msk_models/poses/starting_pose_vertical.yaml",
        default_activation=0.01,
        noise_start=False,
    ))

    """Vertical jump environment specific reward scales"""
    lambda_jump: float = 1e-1
    lambda_limit: float = -3e-4
    lambda_alive: float = 1e-2


Config = Union[
    Annotated[SprintConfig, tyro.conf.subcommand(name="sprint")],
    Annotated[SprintBlockStartConfig, tyro.conf.subcommand(name="blockstart")],
    Annotated[BackpedalConfig, tyro.conf.subcommand(name="backpedal")],
    Annotated[SideShuffleConfig, tyro.conf.subcommand(name="sideshuffle")],
    Annotated[HurdlesConfig, tyro.conf.subcommand(name="hurdles")],
    Annotated[UphillSprintConfig, tyro.conf.subcommand(name="uphillsprint")],
    Annotated[RunTheBendConfig, tyro.conf.subcommand(name="sprintcurve")],
    Annotated[HopConfig, tyro.conf.subcommand(name="hop")],
    Annotated[CariocaConfig, tyro.conf.subcommand(name="carioca")],
    Annotated[VerticalConfig, tyro.conf.subcommand(name="vertical")],
]


def get_args():
    """Parse command-line arguments into Config dataclass."""
    args = tyro.cli(Config)
    args.use_wandb = not args.disable_wandb
    return args
