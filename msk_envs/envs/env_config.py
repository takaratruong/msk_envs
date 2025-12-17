import json
from dataclasses import dataclass

from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    # Environment type
    env_variant: DerivedEnv = DerivedEnv.SPRINT

    # Control frequency
    delta_t: float = 1.0 / 60.0
    # Simulator frequency (only relevant for fixed-step integrators)
    delta_t_sim: float = 1.0 / 1800.0

    # Environment parameters
    max_episode_duration: float = 15.0  # seconds
    model_path: str = "../msk_models/model.osim"  # located at data/
    motion_name: str = "reference_stride"  # motion file name (without .pt extension) for IMITATE variant

    # Model physics properties
    joint_damping: float = 0.1
    joint_armature: float = 0.01
    torso_damping: float = 1.0
    toes_stiffness: float = 65.0
    toes_damping: float = 0.4

    use_hunt_crossley: bool = True

    # Muscle properties
    muscle_multiplier: float = 2.0  # max isometric force scale
    muscle_fiber_damping: float = 0.01
    muscle_min_activation: float = 0.0
    muscle_max_activation: float = 1.0
    muscle_v_max: float = 12.0
    muscle_dynamics_substeps: int = 5

    # Starting state
    starting_pose: str = "../msk_models/starting_pose.yaml"
    noise_start: bool = True
    q_noise: float = 0.05
    qv_noise: float = 0.1
    swap_lr: bool = True

    # Reward scales
    reward_lambdas: dict = None

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
