from .env_variants import DerivedEnv
from .env_sprint import SprintingEnv
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
        if env_variant == DerivedEnv.SPRINT:
            return SprintingEnv(**kwargs)
        elif env_variant == DerivedEnv.VERTICAL:
            return VerticalEnv(**kwargs)
        elif env_variant == DerivedEnv.HURDLES:
            raise NotImplementedError
        elif env_variant == DerivedEnv.LONG_JUMP:
            raise NotImplementedError
        elif env_variant == DerivedEnv.HIGH_JUMP:
            raise NotImplementedError
        elif env_variant == DerivedEnv.SHOT_PUT:
            raise NotImplementedError
        elif env_variant == DerivedEnv.JAVELIN:
            raise NotImplementedError
        elif env_variant == DerivedEnv.IMITATE:
            return ImitateEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_variant}")