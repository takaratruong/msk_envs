import json
from dataclasses import dataclass

from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    env_variant: DerivedEnv = DerivedEnv.SPRINT
    """ Environment type """

    delta_t: float = 1.0 / 720.0
    """ Control/policy step size """
    delta_t_sim: float = 1.0 / 7200.0
    """ Simulator/physics step size """

    max_episode_duration: float = 12.0
    """ Max episode duration in seconds """

    # Model properties
    model_path: str = "../msk_models/model_motor_arms_foot_contact.osim"
    """ OpenSim model file path """
    joint_damping: float = 0.1
    """ Joint damping applied to all joints """
    joint_armature: float = 0.0001
    """ Armature added to all joints (improves stability) """
    torso_damping: float = 1.0
    """ Damping specifically for torso joint """
    toes_stiffness: float = 65.0
    """ Toes joint stiffness """
    toes_damping: float = 0.1
    """ Toes joint damping """
    use_default_joint_limits: bool = False
    """ Whether to use joint limits defined in the model file """
    joint_limits_path: str = "../msk_models/joint_limits.yaml"
    """ Joint limits file path (YAML). NOTE: this overrides limits defined in the model file """

    # Constraint properties
    use_hunt_crossley: bool = True
    """ Whether to use Hunt-Crossley contact model (if not, then MuJoCo contact) """
    use_exponential_limit: bool = False
    """ Whether to use Exponential Spring limit model (if not, then MuJoCo joint limit) """
    limit_force_curves_path: str = "../msk_models/limit_force_curves.yaml"
    """ Exponential limit force curves file path (if using Exponential limits) """
    solref: tuple[float, float] = (0.005, 1.0)
    """ MuJoCo limit/contact parameters (if using MuJoCo limits/contacts) """

    # Muscle properties
    muscle_multiplier: float = 2.0
    """ Multiplier to max isometric force """
    muscle_fiber_damping: float = 0.01
    """ Fiber damping (0.0 = undamped) """
    muscle_min_activation: float = 0.0
    """ Minimum muscle activation. Use non-zero for undamped muscle """
    muscle_max_activation: float = 1.0
    """ Maximum muscle activation """
    muscle_v_max: float = 12.0
    """ Maximum contraction velocity (in optimal fiber lengths per second) """
    muscle_dynamics_substeps: int = 10
    """ Number of substeps for muscle dynamics integration (can improve stability) """

    # Starting pose (starting_pose and noise is ignored for IMITATE variant)
    starting_pose: str = "../msk_models/starting_pose_run.yaml"
    """ Starting pose file path (YAML) """
    noise_start: bool = True
    """ Whether to add noise to starting state """
    q_noise: float = 0.05
    """ std of starting joint position noise"""
    qv_noise: float = 0.1
    """ std of starting joint velocity noise"""
    swap_lr: bool = True
    """ Whether to swap left/right sides when adding noise to starting state """
    motion_name: str = "../motions/pred_sprint_two_step"
    """ motion file name (without .mot extension) for IMITATE variant """
    use_prescribed_starting_activations: bool = True
    """ Whether to use prescribed starting activations from file """
    starting_activations: str = "../msk_models/starting_activations.yaml"
    """ Starting activations file path (YAML) """
    default_activation: float = 0.05
    """ Default activation value when prescribed activations are not used """

    reward_lambdas: dict = None
    """ Reward weights """
    imitation_weights: dict = None
    """ Imitation reward weights """

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
