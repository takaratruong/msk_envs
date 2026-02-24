import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.train.dep.dep import DEP
from msk_envs.utils.logged_sim import LoggedSim
# from msk_envs.train.nets.sac_networks import load_policy
from msk_envs.train.nets.td3_networks import load_policy
from time import perf_counter


def main():
    args = get_args()
    pretty_print_base_args(args)

    # no noise
    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    env_config.swap_lr = False

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda_support else f"cpu")
    num_envs = 1024
    envs = EnvFactory.create_env(num_envs=num_envs,
                                 env_config=env_config,
                                 render=False,
                                 cuda_graph=has_cuda_support,
                                 device=device)

    actions = envs.get_blank_actions()

    import warp as wp
    all_steps_attempted = []
    steps_taken = wp.to_torch(envs.d.steps_attempted)

    # Build a SimLogger to give us a whole pdf of stuff
    max_steps = 20
    time_start = perf_counter()
    steps_attempted = wp.to_torch(envs.d.steps_attempted)
    for _ in tqdm(range(max_steps)):
        actions = torch.randn_like(actions)
        envs.step(actions)
        all_steps_attempted.append(steps_taken.clone())
        print(torch.min(steps_taken), torch.max(steps_taken))
    time_end = perf_counter()
    simulated_time = max_steps * env_config.delta_t
    print("Real time factor (1 world): ", simulated_time / (time_end - time_start))
    print("Real time factor (all worlds): ", num_envs * simulated_time / (time_end - time_start))


if __name__ == "__main__":
    main()
