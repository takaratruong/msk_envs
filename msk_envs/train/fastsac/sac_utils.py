import os

import torch
import wandb

from msk_envs.utils.train_utils import cpu_state


def save_params(
        global_step, actor, qnet, qnet_target, log_alpha, obs_normalizer,
        actor_optimizer, q_optimizer, alpha_optimizer, scaler, args, save_path: str, save_fn=torch.save,
        metadata=None,
):
    """Save model parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "log_alpha": log_alpha.detach().cpu(),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict()) if hasattr(obs_normalizer, "state_dict") else None
        ),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    # Save wandb run ID if available
    if wandb.run is not None:
        save_dict["wandb_run_id"] = wandb.run.id
    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving FastSAC parameters.")
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    print(f"Saved parameters and configuration to {save_path}")
