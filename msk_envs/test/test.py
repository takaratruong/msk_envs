import torch
from msk_envs.envs import MSKEnv
from msk_envs.envs import EnvConfig


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    env = MSKEnv(num_envs=1, env_config=env_config, device=device)

    time = env.get_time()
    print(time)
    return


if __name__ == "__main__":
    main()
