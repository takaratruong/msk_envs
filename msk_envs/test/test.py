import torch
from msk_envs.envs import EnvFactory, EnvConfig
from time import perf_counter


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_lambdas = {"lambda_vel": 1.0}
    env_config = EnvConfig(reward_lambdas=reward_lambdas)
    num_envs = 8
    env = EnvFactory.create_env(
        env_config=env_config,
        num_envs=num_envs,
        device=device,
        render=True,
        cuda_graph=torch.cuda.is_available(),
    )
    env.reset()

    steps = 100
    start = perf_counter()
    for _ in range(steps):
        env.step(torch.zeros(num_envs, env.num_actions(), device=device))
    end = perf_counter()

    steps_taken = steps * num_envs
    elapsed = end - start
    print(f"Steps/sec: {steps_taken / elapsed:.2f}")
    return


if __name__ == "__main__":
    main()
