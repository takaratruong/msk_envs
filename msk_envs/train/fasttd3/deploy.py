from .hyperparams import get_args
from msk_envs.envs.env_config import EnvConfig
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.utils.logged_sim import LoggedSim

import torch
from tqdm import tqdm


def main():
    args = get_args()
    env_config = EnvConfig(
        env_variant=args.env_variant,
        reward_lambdas=args.get_reward_lambdas(),
    )

    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False

    device = torch.device("cuda" if args.cuda else f"cpu")
    envs = EnvFactory.create_env(num_envs=1,
                                 env_config=env_config,
                                 render=False,
                                 cuda_graph=args.cuda,
                                 device=device)

    actions = envs.get_blank_actions()

    # Build a SimLogger to give us a whole pdf of stuff
    max_episode_length = int(env_config.max_episode_duration / env_config.delta_t)
    sim = LoggedSim(envs, max_episode_length, device)
    sim.reset()

    for _ in tqdm(range(max_episode_length)):
        finished, obs = sim.step(actions)
        if finished:
            break
    print("Mean rewards: ", sim.get_rewards_mean())

    # write to torch
    sim.save_animation("dashboard/trajectories/test", "999")


if __name__ == "__main__":
    main()
