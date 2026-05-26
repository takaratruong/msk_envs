import bolt
import torch
from tqdm import tqdm

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args
from msk_envs.utils.logged_sim import LoggedSim


def main():
    args = get_args()
    pretty_print_base_args(args)

    # set seed
    torch.manual_seed(args.seed)

    # no noise
    env_config = args.env_config
    env_config.q_noise = 0.0
    env_config.qv_noise = 0.0
    # env_config.swap_lr = False
    # env_config.integrator_accuracy = 0.1
    # env_config.armature = 0.0
    # env_config.integrator_use_inf_norm = True

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

    # Build a SimLogger to give us a whole pdf of stuff
    max_episode_length = int(env_config.max_episode_duration / env_config.delta_t)
    recording_fps = 120.0
    sim = LoggedSim(envs, device, delta_t_log=1.0 / recording_fps)
    sim.reset()

    increasing = True
    for i in tqdm(range(max_episode_length)):
        actions = torch.ones_like(actions) * -1

        moment_arms = bolt.muscle_moment_arms(envs.d)
        dof_interest = "shoulder_flexion_l"
        dof_interest_id = envs.qpos_id_lookup[dof_interest]
        dof_low, dof_high = envs.limit_id_lookup[dof_interest]
        dof_med = (dof_low + dof_high) / 2
        dof_quart = (dof_high - dof_low) / 4

        dof_value = envs.joint_positions[0, dof_interest_id]
        muscle_idx_to_name = {v: k for k, v in envs.muscle_id_lookup.items()}

        # If we're currently increasing, but exceed 75%, switch
        if increasing and dof_value > (dof_med + dof_quart):
            increasing = False
        # If we're currently decreasing, but are below 25%, switch
        if not increasing and dof_value < (dof_med - dof_quart):
            increasing = True

        for j in range(envs.num_muscles):
            moment_arm = moment_arms[0, j, dof_interest_id]
            muscle_name = muscle_idx_to_name[j]
            if moment_arm > 1e-3:
                actions[:, j] = 1.0 if increasing else -1.0
            elif moment_arm < -1e-3:
                actions[:, j] = -1.0 if increasing else 1.0

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
