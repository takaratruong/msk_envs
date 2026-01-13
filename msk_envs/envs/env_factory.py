from .env_variants import DerivedEnv
from .env_sprint import SprintingEnv
from .env_jog import JoggingEnv
from .env_back_pedal import BackPedalEnv
from .env_side_shuffle import SideShuffleEnv
from .env_hop import HopEnv
from .env_vertical import VerticalEnv
from .env_walk import WalkEnv
from .env_imitate import ImitateEnv


# factory
class EnvFactory:
    @staticmethod
    def create_env(**kwargs):
        env_config = kwargs["env_config"]
        env_variant = env_config.env_variant
        if env_variant == DerivedEnv.WALK:
            return WalkEnv(**kwargs)
        elif env_variant == DerivedEnv.JOG:
            return JoggingEnv(**kwargs)
        elif env_variant == DerivedEnv.SPRINT:
            return SprintingEnv(**kwargs)
        elif env_variant == DerivedEnv.BACKPEDAL:
            return BackPedalEnv(**kwargs)
        elif env_variant == DerivedEnv.SIDE_SHUFFLE:
            return SideShuffleEnv(**kwargs)
        elif env_variant == DerivedEnv.HOP:
            return HopEnv(**kwargs)
        elif env_variant == DerivedEnv.VERTICAL:
            return VerticalEnv(**kwargs)
        elif env_variant == DerivedEnv.IMITATE:
            return ImitateEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_variant}")