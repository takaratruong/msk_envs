from .env_variants import DerivedEnv
from .env_sprint import SprintingEnv
from .env_jog import JoggingEnv
from .env_back_pedal import BackPedalEnv
from .env_side_shuffle import SideShuffleEnv
from .env_hop import HopEnv
from .env_vertical import VerticalEnv
from .env_walk import WalkEnv
from .env_perturb import PerturbEnv
from .env_dont_fall import DontFallEnv
from .env_imitate import ImitateEnv
from .env_static import StaticEnv
from .env_ankle_flex import AnkleFlexEnv
from .env_race_walk import RaceWalkEnv
from.env_shuttle_run import ShuttleRunEnv
from.env_random_target import RandomTargetEnv
from.env_waddle import WaddleEnv
from.env_broad_jump import BroadJumpEnv
from.env_upper_reach import UpperReachTargetEnv

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
        elif env_variant == DerivedEnv.PERTURB:
            return PerturbEnv(**kwargs)
        elif env_variant == DerivedEnv.DONT_FALL:
            return DontFallEnv(**kwargs)
        elif env_variant == DerivedEnv.IMITATE:
            return ImitateEnv(**kwargs)
        elif env_variant == DerivedEnv.STATIC:
            return StaticEnv(**kwargs)
        elif env_variant == DerivedEnv.ANKLE_FLEX:
            return AnkleFlexEnv(**kwargs)
        elif env_variant == DerivedEnv.RACE_WALK:
            return RaceWalkEnv(**kwargs)
        elif env_variant == DerivedEnv.SHUTTLE_RUN:
            return ShuttleRunEnv(**kwargs)
        elif env_variant == DerivedEnv.RANDOM_TARGET:
            return RandomTargetEnv(**kwargs)
        elif env_variant == DerivedEnv.WADDLE:
            return WaddleEnv(**kwargs)
        elif env_variant == DerivedEnv.BROAD_JUMP:
            return BroadJumpEnv(**kwargs)
        elif env_variant == DerivedEnv.UPPER_REACH:
            return UpperReachTargetEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_variant}")