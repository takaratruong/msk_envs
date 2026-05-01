from dataclasses import dataclass, asdict, fields, field
from datetime import datetime
from typing import Union
from typing_extensions import Annotated

import tyro

from msk_envs.envs.env_config import EnvConfig, EnvConfigGeneric, EnvConfigUpper, EnvConfigRegression, \
    EnvConfigUpperNoSpine, EnvConfigLower, EnvConfigTall, EnvConfigTallLower, EnvConfigTallMotorArms, \
    EnvConfigRegressionNoArms, EnvConfigHybrid, EnvConfigTallLowerOldJointLimits, EnvConfigHybridNoArms, \
    EnvConfigTallLowerOldJointLimitsOldUpper, EnvConfigHybridMotorArms
from msk_envs.envs.env_variants import DerivedEnv
from msk_envs.train.fastsac.sac_config import SACConfig
from msk_envs.train.fasttd3.td3_config import TD3Config
from msk_envs.train.dep.dep_config import DEPConfig
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


@dataclass
class JogConfig(BaseArgs):
    env_config: EnvConfigGeneric = field(default_factory=lambda: EnvConfigGeneric(
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
class LaneConfig(BaseArgs):
    """ Reusable hyperparams for lane environments """
    lambda_vel: float = 1e-2
    lambda_mid_lane: float = 0.0

    lambda_spring: float = 0.0
    lambda_damper: float = 0.0
    lambda_limit: float = -1e-3
    lambda_muscle_passive: float = -1e-4

    lambda_actuator: float = -1e-3
    lambda_fatigue: float = -1e-4
    lambda_muscle_activation: float = 0.0
    lambda_metabolic: float = 0.0
    lambda_self_collision: float = 0.0
    lambda_head_acc_ang: float = 0.0
    lambda_head_acc_lin: float = 0.0


@dataclass
class SprintConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
        noise_start=True,
    ))


@dataclass
class SprintConfigTall(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigTall(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
    ))


@dataclass
class SprintConfigTallMotor(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigTallMotorArms(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
        noise_start=False,
    ))


@dataclass
class SprintLowerConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigLower(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
    ))


@dataclass
class SprintConfigTallLower(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigTallLower(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
    ))


@dataclass
class SprintConfigTallLowerOldJointLimits(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigTallLowerOldJointLimits(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
    ))


@dataclass
class SprintConfigTallLowerOldJointLimitsOldUpper(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigTallLowerOldJointLimitsOldUpper(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
    ))


@dataclass
class SprintRegressionConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigRegression(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/regression/starting_pose_run.yaml",
        default_activation=0.01,
    ))


@dataclass
class SprintRegressionNoArmsConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigRegressionNoArms(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/regression/starting_pose_run.yaml",
        default_activation=0.01,
    ))


@dataclass
class SprintHybridConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybrid(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run_hybrid.yaml",
    ))


@dataclass
class BackpedalHybridConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybrid(
        env_variant=DerivedEnv.BACKPEDAL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_backpedal_hybrid.yaml",
    ))


@dataclass
class SideShuffleHybridConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybrid(
        env_variant=DerivedEnv.SIDE_SHUFFLE,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_sideshuffle_hybrid.yaml",
    ))


@dataclass
class SprintHybridMotorArmsConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybridMotorArms(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run_hybrid.yaml",
    ))


@dataclass
class SprintHybridNoArmsConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybridNoArms(
        env_variant=DerivedEnv.SPRINT,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run_hybrid.yaml",
    ))


@dataclass
class RaceWalkConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.RACE_WALK,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_run.yaml",
    ))

    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -5e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class BackPedalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigLower(
        env_variant=DerivedEnv.BACKPEDAL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_backward.yaml",
        default_activation=0.01,
    ))

    lambda_vel: float = 1e-2
    lambda_mid_lane: float = 3e-2

    lambda_spring: float = -1e-4
    lambda_damper: float = -3e-4
    lambda_limit: float = -1e-3
    # lambda_muscle_passive: float = -1e-4
    lambda_muscle_passive: float = 0.0

    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0
    lambda_self_collision: float = -1e-2
    lambda_head_acc_ang: float = 0.0
    lambda_head_acc_lin: float = 0.0


@dataclass
class SideShuffleConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigLower(
        env_variant=DerivedEnv.SIDE_SHUFFLE,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_side.yaml",
        default_activation=0.01,
    ))

    lambda_vel: float = 1e-2
    lambda_mid_lane: float = 3e-2  # acts as an alive reward

    lambda_spring: float = -3e-5
    lambda_damper: float = -3e-5
    lambda_limit: float = -3e-5
    lambda_muscle_passive: float = -3e-5

    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0
    lambda_self_collision: float = -1e-2
    lambda_head_acc_ang: float = 0.0
    lambda_head_acc_lin: float = 0.0


@dataclass
class HopConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.HOP,
        delta_t=1.0 / 60.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_left_leg_up.yaml",
        swap_lr=False,
    ))

    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -0.005
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class HopHybridConfig(LaneConfig):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybrid(
        env_variant=DerivedEnv.HOP,
        delta_t=1.0 / 60.0,
        max_episode_duration=10.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_hop.yaml",
        swap_lr=False,
    ))

    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -0.005
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class VerticalConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
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
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigLower(
        env_variant=DerivedEnv.PERTURB,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        starting_pose_path="../msk_models/starting_pose_stand.yaml"
    ))
    lambda_alive: float = 3e-1

    lambda_spring: float = -1e-4
    lambda_damper: float = -1e-4
    lambda_limit: float = -1e-4
    lambda_muscle_passive: float = -1e-4

    # lambda_fatigue: float = 0
    # lambda_metabolic: float = 0
    # lambda_actuator: float = 0
    # lambda_actuator_dot: float = 0.0


@dataclass
class DontFallConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigLower(
        env_variant=DerivedEnv.DONT_FALL,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        noise_start=False,
        starting_pose_path="../msk_models/starting_pose_stand.yaml",
        default_activation=0.01,
    ))
    lambda_alive: float = 1e-1
    lambda_spring: float = -1e-4
    lambda_damper: float = -3e-4
    lambda_limit: float = -1e-3
    # lambda_muscle_passive: float = -1e-4
    lambda_muscle_passive: float = 0

    lambda_actuator: float = 0
    lambda_fatigue: float = 0
    lambda_metabolic: float = 0
    lambda_root_zero: float = 1e-1
    lambda_match_start: float = 1e-1


@dataclass
class DontFallConfigOneLeg(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.DONT_FALL_ONE_LEG,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        noise_start=False,
        starting_pose_path="../msk_models/starting_pose_stand_one_leg.yaml",
        use_prescribed_starting_activations=False,
        default_activation=0.0,
        swap_lr=False,
    ))
    lambda_alive: float = 0
    lambda_limit: float = -1e-2
    lambda_actuator: float = 0
    lambda_fatigue: float = 0
    lambda_metabolic: float = 0
    lambda_root_zero: float = 1e-1
    lambda_match_start: float = 1e-1


@dataclass
class ImitateConfig(BaseArgs):
    """ Default Imitate environment configuration. Tracks a two-step sprinting motion. """
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.IMITATE,
        delta_t=1.0 / 500.0,
        motion_name="../motions/sprint.mot",
        noise_start=False,
        muscle_multiplier=2.0,
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
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.STATIC,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand.yaml",
    ))

    lambda_track: float = 0.1
    imitation_weight_track: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        self.env_config.target_position = [0.0, 0.9, 0.0]


# @dataclass
# class AnkleFlexConfig(BaseArgs):
#     env_config: EnvConfig = field(default_factory=lambda: EnvConfigStaticLegs(
#         env_variant=DerivedEnv.ANKLE_FLEX,
#         delta_t=1.0 / 30.0,
#         max_episode_duration=10.0,
#         # muscle_multiplier=2.0,
#     ))
#
#     lambda_test: float = 0.1
#

@dataclass
class ShuttleRunConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.SHUTTLE_RUN,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_run.yaml",
    ))

    lambda_target_closer: float = 1.0
    lambda_limit: float = -2e-3
    lambda_actuator: float = -1e-2
    lambda_alive: float = 3e-2


@dataclass
class WaddleConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.WADDLE,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand_waddle.yaml",
    ))

    lambda_vel: float = 0.01
    lambda_mid_lane: float = 0.01
    lambda_limit: float = -5e-3
    lambda_actuator: float = -1e-2
    lambda_fatigue: float = 0.0
    lambda_metabolic: float = 0.0


@dataclass
class RandomTargetConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigHybrid(
        env_variant=DerivedEnv.RANDOM_TARGET,
        delta_t=1.0 / 30.0,
        max_episode_duration=10.0,
        muscle_multiplier=2.0,
        starting_pose_path="../msk_models/starting_pose_run_hybrid.yaml",
    ))

    lambda_target_closer: float = 1.0
    lambda_target_facing: float = 5e-3

    lambda_spring: float = 0.0
    lambda_damper: float = 0.0
    lambda_limit: float = -3e-4
    lambda_muscle_passive: float = 0.0

    lambda_actuator: float = 0.0
    lambda_alive: float = 3e-2


@dataclass
class BroadJumpConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.BROAD_JUMP,
        delta_t=1.0 / 30.0,
        max_episode_duration=3.0,
        # muscle_multiplier=2.0,
        starting_pose_path="../msk_models/no_hands/starting_pose_stand.yaml",
    ))

    lambda_alive: float = 1e-1
    # TODO:


@dataclass
class UpperReachTarget(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigUpper(
        env_variant=DerivedEnv.UPPER_REACH,
        delta_t=1.0 / 30.0,
        max_episode_duration=5.0,
        muscle_multiplier=1.0,
        starting_pose_path="../msk_models/starting_pose_stand.yaml",
        swap_lr=False,
        use_linear_stop=True,
        default_activation=0.01,
        noise_start=False,
    ))

    lambda_target: float = 0.5
    lambda_limit: float = 0.0
    lambda_actuator: float = 0.0
    lambda_alive: float = 0.0


@dataclass
class UpperCurl(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigUpper(
        env_variant=DerivedEnv.UPPER_CURL,
        delta_t=1.0 / 30.0,
        max_episode_duration=5.0,
        muscle_multiplier=1.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
        swap_lr=False,
    ))

    lambda_target: float = 2e-1
    lambda_starting: float = 2e-2
    lambda_spring: float = -1e-4
    lambda_damper: float = -1e-4
    lambda_limit: float = -1e-4
    lambda_muscle_passive: float = -1e-4
    lambda_actuator: float = -3e-2
    lambda_actuator_dot: float = -1e-3
    lambda_alive: float = 3e-2
    # lambda_limit: float = 0
    # lambda_actuator: float = 0
    # lambda_actuator_dot: float = 0
    # lambda_alive: float = 0


@dataclass
class NoSpineReachTarget(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigUpperNoSpine(
        env_variant=DerivedEnv.UPPER_REACH,
        delta_t=1.0 / 30.0,
        max_episode_duration=5.0,
        muscle_multiplier=1.0,
        starting_pose_path="../msk_models/starting_pose_run.yaml",
        swap_lr=False,
    ))

    lambda_target: float = 0.5
    lambda_limit: float = -2e-3
    lambda_actuator: float = -1e-2
    lambda_alive: float = 3e-2


@dataclass
class ReachPoseConfig(BaseArgs):
    env_config: EnvConfig = field(default_factory=lambda: EnvConfigGeneric(
        env_variant=DerivedEnv.REACH_POSE,
        delta_t=1.0 / 30.0,
        max_episode_duration=2.5,
        muscle_multiplier=1.0,
        starting_pose_path="../msk_models/starting_pose_stand.yaml",
        target_pose_path="../msk_models/target_pose_stand_one_leg.yaml",
        default_activation=0.01,
    ))

    lambda_target: float = 1e-2
    lambda_target_global: float = 3e-3


Config = Union[
    # Annotated[WalkConfig, tyro.conf.subcommand(name="walk")],
    Annotated[JogConfig, tyro.conf.subcommand(name="jog")],
    Annotated[SprintConfig, tyro.conf.subcommand(name="sprint")],
    Annotated[SprintConfigTall, tyro.conf.subcommand(name="sprinttall")],
    Annotated[SprintConfigTallMotor, tyro.conf.subcommand(name="sprinttallmotor")],
    Annotated[SprintLowerConfig, tyro.conf.subcommand(name="sprintlower")],
    Annotated[SprintConfigTallLower, tyro.conf.subcommand(name="sprinttalllower")],
    Annotated[SprintConfigTallLowerOldJointLimits, tyro.conf.subcommand(name="sprinttallloweroldlimits")],
    Annotated[
        SprintConfigTallLowerOldJointLimitsOldUpper, tyro.conf.subcommand(name="sprinttalllowernewoldlimitsoldupper")],
    Annotated[SprintRegressionConfig, tyro.conf.subcommand(name="sprintregression")],
    Annotated[SprintRegressionNoArmsConfig, tyro.conf.subcommand(name="sprintregressionnoarms")],
    Annotated[SprintHybridConfig, tyro.conf.subcommand(name="sprinthybrid")],
    Annotated[SprintHybridMotorArmsConfig, tyro.conf.subcommand(name="sprinthybridmotorarms")],
    Annotated[SprintHybridNoArmsConfig, tyro.conf.subcommand(name="sprinthybridnoarms")],
    Annotated[RaceWalkConfig, tyro.conf.subcommand(name="racewalk")],
    Annotated[BackPedalConfig, tyro.conf.subcommand(name="backpedal")],
    Annotated[BackpedalHybridConfig, tyro.conf.subcommand(name="backpedalhybrid")],
    Annotated[SideShuffleConfig, tyro.conf.subcommand(name="sideshuffle")],
    Annotated[SideShuffleHybridConfig, tyro.conf.subcommand(name="sideshufflehybrid")],
    Annotated[HopConfig, tyro.conf.subcommand(name="hop")],
    Annotated[HopHybridConfig, tyro.conf.subcommand(name="hophybrid")],
    Annotated[VerticalConfig, tyro.conf.subcommand(name="vertical")],
    Annotated[PerturbConfig, tyro.conf.subcommand(name="perturb")],
    Annotated[ImitateConfig, tyro.conf.subcommand(name="imitate")],
    Annotated[DontFallConfig, tyro.conf.subcommand(name="dontfall")],
    Annotated[DontFallConfigOneLeg, tyro.conf.subcommand(name="dontfalloneleg")],
    Annotated[StaticSquatConfig, tyro.conf.subcommand(name="static")],
        # Annotated[AnkleFlexConfig, tyro.conf.subcommand(name="ankleflex")],
    Annotated[ShuttleRunConfig, tyro.conf.subcommand(name="shuttlerun")],
    Annotated[RandomTargetConfig, tyro.conf.subcommand(name="randomtarget")],
    Annotated[WaddleConfig, tyro.conf.subcommand(name="waddle")],
    Annotated[BroadJumpConfig, tyro.conf.subcommand(name="broadjump")],
    Annotated[UpperReachTarget, tyro.conf.subcommand(name="upperreach")],
    Annotated[UpperCurl, tyro.conf.subcommand(name="uppercurl")],
    Annotated[NoSpineReachTarget, tyro.conf.subcommand(name="nospinereach")],
    Annotated[ReachPoseConfig, tyro.conf.subcommand(name="reachpose")],
]


def get_args():
    """Parse command-line arguments into Config dataclass."""
    args = tyro.cli(Config)
    args.use_wandb = not args.disable_wandb
    return args
