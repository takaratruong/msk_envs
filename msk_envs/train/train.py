import os

# Set Warp cache directory before importing warp to ensure it uses the correct location
if 'WARP_CACHE_DIR' not in os.environ:
    import tempfile

    os.environ['WARP_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'warp_cache_{os.getuid()}')
    os.makedirs(os.environ['WARP_CACHE_DIR'], exist_ok=True)

import torch
import wandb
import warp as wp

import msk_envs.train.fasttd3.train as fasttd3
from msk_envs.utils.train_utils import set_seed
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.train.hyperparams import get_args, pretty_print_base_args

wp.clear_kernel_cache()  # can't risk caching issues


def main():
    # Restore original HOME after Warp has initialized (for wandb and other tools)
    if 'ORIG_HOME' in os.environ:
        os.environ['HOME'] = os.environ['ORIG_HOME']

    args = get_args()
    set_seed(args.seed)
    pretty_print_base_args(args)

    td3_config, env_config = args.td3_config, args.env_config
    if args.cuda:
        assert torch.cuda.is_available(), "CUDA device not available"
    device = torch.device(f"cuda:{args.gpu_id}" if args.cuda else "cpu")

    # Build envs. currently only uses cuda if it's available
    envs = EnvFactory.create_env(
        num_envs=td3_config.num_envs,
        env_config=env_config,
        render=False,
        cuda_graph=args.cuda,
        device=device,
    )
    eval_envs = EnvFactory.create_env(
        num_envs=td3_config.num_eval_envs,
        env_config=env_config,
        render=False,
        cuda_graph=args.cuda,
        device=device,
    )

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

    traj_out_folder, analytics_out_folder = args.traj_out_folder, args.analytics_out_folder
    fasttd3.train(
        td3_config=td3_config,
        envs=envs, eval_envs=eval_envs,
        traj_out_folder=traj_out_folder,
        analytics_out_folder=analytics_out_folder,
        exp_name=args.exp_name,
        cuda=args.cuda,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()
