# from msk_envs.train.nets.sac_networks import load_policy
from time import perf_counter

import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args


def main():
    args = get_args()
    pretty_print_base_args(args)

    env_config = args.env_config
    # no noise
    # env_config.q_noise = 0.0
    # env_config.qv_noise = 0.2
    # env_config.swap_lr = False

    has_cuda_support = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda_support else f"cpu")
    num_envs = 4096
    envs = EnvFactory.create_env(num_envs=num_envs,
                                 env_config=env_config,
                                 requires_visuals=False,
                                 live_render=False,
                                 cuda_graph=has_cuda_support,
                                 device=device)
    envs.reset()

    actions = envs.get_blank_actions()

    import warp as wp
    steps_taken = wp.to_torch(envs.d.steps_attempted)

    max_steps = 50
    time_start = perf_counter()
    for _ in tqdm(range(max_steps)):
        actions = torch.randn_like(actions)
        envs.step(actions)
        print("Steps taken:", torch.min(steps_taken).item(), "to", torch.max(steps_taken).item())
    time_end = perf_counter()
    simulated_time = max_steps * env_config.delta_t
    print("Real time factor (1 world): ", simulated_time / (time_end - time_start))
    print("Real time factor (all worlds): ", num_envs * simulated_time / (time_end - time_start))


if __name__ == "__main__":
    main()
