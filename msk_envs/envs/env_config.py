import json
import msk_warp
from dataclasses import dataclass, field

from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    env_variant: DerivedEnv = DerivedEnv.SPRINT
    """ Environment type """

    # --- Control and Simulation Frequency ---
    delta_t: float = 1.0 / 100.0
    """ Control/policy step size """
    delta_t_sim: float = 1.0 / 10000.0
    """ Simulator/physics step size """
    max_episode_duration: float = 12.0
    """ Max episode duration in seconds """
    integrator: msk_warp.IntegratorType = msk_warp.IntegratorType.EULER_FIXED
    """ Integrator type (EULER_FIXED, RK4_FIXED) """

    # --- Model articulation properties ---
    model_path: str = "../msk_models/model_motor_arms_full_contact.osim"
    """ OpenSim model file path """
    joint_damping: float = 0.1
    """ Joint damping applied to all joints """
    joint_armature: float = 0.0
    """ Armature added to all joints (increases inertia but improves stability) """
    toe_armature: float = 0.0
    """ Armature specifically for toes joint """
    torso_damping: float = 1.0
    """ Damping specifically for torso joint """
    toes_stiffness: float = 65.0
    """ Toes joint stiffness """
    toes_damping: float = 0.4
    """ Toes joint damping """
    use_specified_joint_limits: bool = True
    """ Whether to use joint limits defined in joint_limits_path, otherwise use limits defined in model file """
    joint_limits_path: str = "../msk_models/joint_limits_hc.yaml"
    """ Joint limits file path (YAML). NOTE: this overrides limits defined in the model file """
    enable_drag: bool = False
    """ Whether to enable drag forces """
    use_specified_contact_params: bool = True
    """ Whether to use contact parameters defined in contact_params_path """
    contact_params_path: str = "../msk_models/contact_params_sprint.yaml"
    """ Contact parameters file path (YAML). NOTE: this overrides contact parameters defined in the model file """

    # --- Model muscle properties ---
    muscle_multiplier: float = 2.0
    """ Multiplier to max isometric force """
    muscle_activation_time_const: float = 0.015
    """ Muscle activation time constant """
    muscle_deactivation_time_const: float = 0.060
    """ Muscle deactivation time constant """
    muscle_activation_dynamics_smoothing: float = 0.1
    """ Muscle activation dynamics smoothing factor """
    muscle_fiber_damping: float = 0.01
    """ Fiber damping (0.0 = undamped) """
    muscle_min_activation: float = 0.0
    """ Minimum muscle activation. Use non-zero for undamped muscle """
    muscle_max_activation: float = 1.0
    """ Maximum muscle activation """
    muscle_v_max: float = 12.0
    """ Maximum contraction velocity (in optimal fiber lengths per second) """
    muscle_dynamics_substeps: int = 0
    """ Number of substeps for muscle dynamics integration (can improve stability) """
    use_function_based_path: bool = True
    """ Whether to use function-based path (or geometry path)"""
    muscle_function_path: str = "../msk_models/muscle_fn_path_info.json"
    """ Function-based path data file (JSON) """
    use_specified_metabolic_params: bool = True
    """ Whether to use metabolic parameters defined in metabolic_params_path """
    metabolic_params_path: str = "../msk_models/muscle_metabolic_params.yaml"
    """ Muscle metabolic parameters file path (YAML) """

    # --- Constraint properties ---
    contact_type: msk_warp.ContactType = msk_warp.ContactType.HUNT_CROSSLEY
    """ Contact model type (HUNT_CROSSLEY, HUNT_CROSSLEY_SMOOTH, MUJOCO) """
    limit_type: msk_warp.LimitType = msk_warp.LimitType.HUNT_CROSSLEY
    """ Joint limit model type (MUJOCO, EXPONENTIAL, HUNT_CROSSLEY) """
    limit_force_curves_path: str = "../msk_models/no_hands/limit_force_curves_hc.yaml"
    """ Limit force curves file path (if using EXPONENTIAL or HUNT_CROSSLEY limits) """
    solref: tuple[float, float] = (0.02, 1.0)
    """ MuJoCo limit/contact parameters (if using MuJoCo limits/contacts) """

    # Starting pose (starting_pose and noise is ignored for IMITATE variant)
    starting_pose_path: str = "../msk_models/no_hands/starting_pose_stand.yaml"
    """ Starting pose file path (YAML) """
    noise_start: bool = True
    """ Whether to add noise to starting state """
    q_noise: float = 0.0
    """ std of starting joint position noise"""
    qv_noise: float = 0.0
    """ std of starting joint velocity noise"""
    swap_lr: bool = True
    """ Whether to swap left/right sides when adding noise to starting state """
    motion_name: str = "../motions/pred_sprint_two_step.mot"
    """ motion file name for IMITATE environments (or variants) """
    use_prescribed_starting_activations: bool = False
    """ Whether to use prescribed starting activations from file """
    starting_activations_path: str = "../msk_models/starting_activations.yaml"
    """ Starting activations file path (YAML) """
    default_activation: float = -1.0
    """ Default activation value when prescribed activations are not used. -1.0 is random. """

    # Rewards for specific env variants: The following need to be set
    reward_lambdas: dict = field(default_factory=dict)
    """ Reward weights """
    imitation_weights: dict = field(default_factory=dict)
    """ Imitation reward weights """
    extra_rewarded_joints: list = field(default_factory=list)
    """ List of body names to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_joints: float = 0.0
    """ Lambda for extra rewarded joints """
    extra_rewarded_dofs: list = field(default_factory=list)
    """ List of DOFs to give extra reward for tracking (for debug) """
    lambda_extra_rewarded_dofs: float = 0.0
    """ Lambda for extra rewarded DOFs """

    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_json_file(cls, file_path):
        with open(file_path, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def pretty_print(env_config):
        print("Environment Configuration:")
        for key, value in env_config.to_dict().items():
            print(f"  {key}: {value}")


@dataclass
class EnvConfigNoHands(EnvConfig):
    """ Environment configuration for no-hands model"""
    model_path: str = "../msk_models/no_hands/model_motor_arms_no_hand_full_contact.osim"
    joint_limits_path: str = "../msk_models/no_hands/joint_limits_hc.yaml"
    limit_force_curves_path: str = "../msk_models/no_hands/limit_force_curves_hc.yaml"
    starting_pose_path: str = "../msk_models/no_hands/starting_pose_stand.yaml"
    contact_params_path: str = "../msk_models/contact_params_sprint.yaml"


@dataclass
class EnvConfigNoHandsWalk(EnvConfig):
    """ Environment configuration for no-hands model, with walking-specific contact params"""
    model_path: str = "../msk_models/no_hands/model_motor_arms_no_hand_walk_contact.osim"
    joint_limits_path: str = "../msk_models/no_hands/joint_limits_hc.yaml"
    limit_force_curves_path: str = "../msk_models/no_hands/limit_force_curves_hc.yaml"
    starting_pose_path: str = "../msk_models/no_hands/starting_pose_stand.yaml"
    contact_params_path: str = "../msk_models/contact_params_walking.yaml"
