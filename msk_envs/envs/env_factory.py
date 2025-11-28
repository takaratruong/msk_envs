from .env_variants import DerivedEnv
from .env_sprint import SprintingEnv


class EnvFactory:
    @staticmethod
    def create_env(**kwargs):
        env_config = kwargs["env_config"]
        env_variant = env_config.env_variant

        if env_variant == DerivedEnv.SPRINT:
            return SprintingEnv(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_variant}")
