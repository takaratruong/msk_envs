import json
from dataclasses import dataclass, field
from .env_variants import DerivedEnv


@dataclass
class EnvConfig:
    # Environment type
    env_variant: DerivedEnv = DerivedEnv.SPRINT

    # Control frequency
    delta_t: float = 1.0 / 60.0
    # Simulator frequency (only relevant for fixed-step integrators)
    delta_t_sim: float = 1.0 / 600.0

    # Environment parameters
    max_episode_duration: float = 7.5  # seconds
    model_path: str = "../models/model.osim"  # located at data/
    muscle_multiplier: float = 2.0  # max isometric force scale

    # Starting state
    starting_pose: str = "../models/starting_pose.yaml"
    noise_start: bool = True
    q_noise: float = 0.0
    qv_noise: float = 0.05
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
