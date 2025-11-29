import torch
from msk_envs.envs import EnvFactory, EnvConfig


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_lambdas = {"lambda_vel": 1.0}
    env_config = EnvConfig(reward_lambdas=reward_lambdas)
    num_envs = 8
    env = EnvFactory.create_env(
        env_config=env_config,
        num_envs=num_envs,
        device=device,
        render=False
    )

    time = env.get_time()
    steps = 10
    for _ in range(steps):
        env.step(torch.zeros(num_envs, env.num_actions(), device=device))
    return


if __name__ == "__main__":
    main()
