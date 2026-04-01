import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.train.dep.dep import DEP
from msk_envs.utils.logged_sim import LoggedSim
# from msk_envs.train.nets.sac_networks import load_policy
from msk_envs.train.nets.td3_networks import load_policy


def main():
    args = get_args()
    pretty_print_base_args(args)

    # set seed
    torch.manual_seed(args.seed)

    # no noise
    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False
    env_config.integrator_accuracy = 0.1

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda_support else f"cpu")
    num_envs = 1
    envs = EnvFactory.create_env(num_envs=num_envs,
                                 env_config=env_config,
                                 live_render=False,
                                 requires_visuals=True,
                                 cuda_graph=has_cuda_support,
                                 device=device)

    actions = envs.get_blank_actions()
    #
    policy = load_policy("/home/marth/Documents/msk_envs/models/regression_2026-03-26_12-11/regression_2026-03-26_12-11_82000.pt")
    policy.to(device)

    dep = DEP(n_motors=envs.num_muscles,
              n_envs=num_envs,
              buffer_size=200,
              bias_rate=0.002,
              kappa=1000,
              tau=40,
              s4avg=2,
              regularization=32,
              time_dist=5,
              with_learning=True,
              device=device)

    # Build a SimLogger to give us a whole pdf of stuff
    max_episode_length = int(env_config.max_episode_duration / env_config.delta_t)
    sim = LoggedSim(envs, device)
    obs = sim.reset()

    for _ in tqdm(range(max_episode_length)):
        # actions = torch.randn_like(actions)
        # actions = torch.ones_like(actions) * -1
        # muscle_states = envs.muscle_fiber_lengths
        # actions[:, :envs.num_muscles] = dep.step(muscle_states)
        # actions[:, envs.num_muscles:] = torch.randn_like(actions[:, envs.num_muscles:])

        with torch.no_grad():
            actions = policy(obs)

        # try to step the sim, but if it takes too long, break out of the loop
        finished, obs = sim.step(actions)
        if finished.all():
            break

    print("Mean rewards: ", sim.get_rewards_mean())

    # write to torch
    sim.save_animation("dashboard/trajectories/test", "999", use_gzip=True)
    sim.save_analytics(".", "deploy_analytics")
    sim.save_frame_data(".", "deploy_frame_data", use_gzip=True)


if __name__ == "__main__":
    main()
