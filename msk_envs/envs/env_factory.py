from .env_variants import DerivedEnv
from .env_sprint import SprintingEnv
from .env_back_pedal import BackPedalEnv
from .env_side_shuffle import SideShuffleEnv
from .env_hop import HopEnv
from .env_vertical import VerticalEnv
from .env_curve import CurvedTrackEnv
from .env_hurdles import HurdlesEnv
from .env_carioca import CariocaEnv
from .env_sprint_blocks import BlockStartSprintingEnv
from .env_locomotion import LocomotionEnv


# factory
class EnvFactory:
    @staticmethod
    def create_env(**kwargs):
        env_config = kwargs["env_config"]
        env_variant = env_config.env_variant
        if env_variant == DerivedEnv.SPRINT:
            return SprintingEnv(**kwargs)
        elif env_variant == DerivedEnv.BACKPEDAL:
            return BackPedalEnv(**kwargs)
        elif env_variant == DerivedEnv.SIDE_SHUFFLE:
            return SideShuffleEnv(**kwargs)
        elif env_variant == DerivedEnv.HOP:
            return HopEnv(**kwargs)
        elif env_variant == DerivedEnv.VERTICAL:
            return VerticalEnv(**kwargs)
        elif env_variant == DerivedEnv.RUN_THE_BEND:
            return CurvedTrackEnv(**kwargs)
        elif env_variant == DerivedEnv.HURDLES:
            return HurdlesEnv(**kwargs)
        elif env_variant == DerivedEnv.CARIOCA:
            return CariocaEnv(**kwargs)
        elif env_variant == DerivedEnv.SPRINT_BLOCK_START:
            return BlockStartSprintingEnv(**kwargs)
        elif env_variant == DerivedEnv.LOCOMOTION:
            return LocomotionEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_variant}")
