from msk_envs.envs import MSKEnv
from msk_envs.envs import EnvConfig


def main():
    env_config = EnvConfig()
    env = MSKEnv(num_envs=1, env_config=env_config)
    return


if __name__ == "__main__":
    main()
