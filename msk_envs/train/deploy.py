import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.nets.networks import load_policy
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.utils.logged_sim import LoggedSim


def main():
    args = get_args()

    # no noise
    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda_support else f"cpu")
    envs = EnvFactory.create_env(num_envs=1,
                                 env_config=env_config,
                                 render=False,
                                 cuda_graph=has_cuda_support,
                                 device=device)

    actions = envs.get_blank_actions()

    # policy = load_policy("/home/marth/Documents/msk_envs/models/sprint_gsde_lims_2026-01-07_17-10-24/sprint_gsde_lims_2026-01-07_17-10-24_10000.pt")
    # policy.to(device)


    # Build a SimLogger to give us a whole pdf of stuff
    max_episode_length = int(env_config.max_episode_duration / env_config.delta_t)
    sim = LoggedSim(envs, device)
    obs = sim.reset()

    for _ in tqdm(range(max_episode_length)):
        # actions = torch.randn_like(actions)
        # actions = envs.get_blank_actions()
        # with torch.no_grad():
        #     actions = policy(obs)
        finished, obs = sim.step(actions)
        if finished:
            break
    print("Mean rewards: ", sim.get_rewards_mean())

    # write to torch
    sim.save_animation("dashboard/trajectories/test", "999", use_gzip=True)
    sim.save_analytics(".", "deploy_analytics")
    sim.save_frame_data(".", "deploy_frame_data", use_gzip=True)


if __name__ == "__main__":
    main()
