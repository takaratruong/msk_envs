import os

# Set Warp cache directory before importing warp to ensure it uses the correct location
if 'WARP_CACHE_DIR' not in os.environ:
    import tempfile

    os.environ['WARP_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'warp_cache_{os.getuid()}')
    os.makedirs(os.environ['WARP_CACHE_DIR'], exist_ok=True)

import torch
import warp as wp

import msk_envs.train.fasttd3.train as fasttd3
import msk_envs.train.fastsac.train as fastsac
from msk_envs.utils.train_utils import set_seed, init_wandb_run
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args


def _build_envs(num_envs, num_eval_envs, env_config, build_cuda_graph, device):
    envs = EnvFactory.create_env(
        num_envs=num_envs,
        env_config=env_config,
        live_render=False,
        requires_visuals=False,
        cuda_graph=build_cuda_graph,
        device=device,
    )
    eval_envs = EnvFactory.create_env(
        num_envs=num_eval_envs,
        env_config=env_config,
        live_render=False,
        requires_visuals=True,
        cuda_graph=build_cuda_graph,
        device=device,
    )
    return envs, eval_envs


def main():
    # wp.clear_kernel_cache()  # can't risk caching issues

    # Restore original HOME after Warp has initialized (for wandb and other tools)
    if 'ORIG_HOME' in os.environ:
        os.environ['HOME'] = os.environ['ORIG_HOME']

    args = get_args()
    set_seed(args.seed)
    pretty_print_base_args(args)

    env_config = args.env_config
    td3_config = args.td3_config
    sac_config = args.sac_config

    assert torch.cuda.is_available(), "CUDA device not available"
    device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    wp.set_device(device_str)

    if args.use_wandb:
        init_wandb_run(args)

    traj_out_folder, analytics_out_folder = args.traj_out_folder, args.analytics_out_folder
    if args.algo == "td3":
        envs, eval_envs = _build_envs(
            num_envs=td3_config.num_envs,
            num_eval_envs=td3_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        fasttd3.train(
            td3_config=td3_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )
    elif args.algo == "sac":
        envs, eval_envs = _build_envs(
            num_envs=sac_config.num_envs,
            num_eval_envs=sac_config.num_eval_envs,
            env_config=env_config,
            build_cuda_graph=True,
            device=device,
        )
        fastsac.train(
            sac_config=sac_config,
            envs=envs,
            eval_envs=eval_envs,
            traj_out_folder=traj_out_folder,
            analytics_out_folder=analytics_out_folder,
            exp_name=args.exp_name,
            device=device,
        )


if __name__ == "__main__":
    main()
