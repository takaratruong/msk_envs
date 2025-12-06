import torch
from msk_envs.envs import EnvFactory, EnvConfig
from time import perf_counter
from tqdm import tqdm

import warp as wp


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_lambdas = {"lambda_vel": 1.0}
    env_config = EnvConfig(reward_lambdas=reward_lambdas)
    num_envs = 8192
    env = EnvFactory.create_env(
        env_config=env_config,
        num_envs=num_envs,
        device=device,
        render=False,
        cuda_graph=True,
    )

    steps = 100
    start = perf_counter()
    for _ in tqdm(range(steps)):
        env.step(torch.zeros(num_envs, env.num_actions(), device=device))
    end = perf_counter()

    steps_taken = steps * num_envs
    elapsed = end - start
    print(f"Steps/sec: {steps_taken / elapsed:.2f}")
    return


if __name__ == "__main__":
    main()
