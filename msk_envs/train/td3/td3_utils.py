import os

import torch
import wandb

from msk_envs.utils.train_utils import cpu_state, get_ddp_state_dict


def save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        args,
        save_path,
        actor_optimizer=None,
        q_optimizer=None,
        actor_scheduler=None,
        q_scheduler=None,
        scaler=None,
        include_optim_state: bool = False,
):
    """Save model parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(get_ddp_state_dict(qnet_target)),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),
        "global_step": global_step,
    }
    if include_optim_state:
        if actor_optimizer is not None:
            save_dict["actor_optimizer_state_dict"] = actor_optimizer.state_dict()
        if q_optimizer is not None:
            save_dict["q_optimizer_state_dict"] = q_optimizer.state_dict()
        if actor_scheduler is not None:
            save_dict["actor_scheduler_state_dict"] = actor_scheduler.state_dict()
        if q_scheduler is not None:
            save_dict["q_scheduler_state_dict"] = q_scheduler.state_dict()
        if scaler is not None:
            save_dict["grad_scaler_state_dict"] = scaler.state_dict()
    if wandb.run is not None:
        save_dict["wandb_run_id"] = wandb.run.id
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")
