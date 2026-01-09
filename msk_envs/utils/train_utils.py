import os
import random

import numpy as np
import torch


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def get_ddp_state_dict(model):
    """Get state dict from model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_ddp_state_dict(model, state_dict):
    """Load state dict into model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def save_params(global_step, actor, qnet, qnet_target, obs_normalizer,
                args, save_path):
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
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def mark_step():
    # call this once per iteration *before* any compiled function
    torch.compiler.cudagraph_mark_step_begin()
