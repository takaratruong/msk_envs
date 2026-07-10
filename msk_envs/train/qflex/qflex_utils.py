import os

import torch


def save_params(
        global_step: int,
        actor,
        critic,
        obs_normalizer,
        args,
        save_path: str,
):
    """Persist model weights, normalizer stats and the training config."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "obs_normalizer_state": (
                obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None
            ),
            "args": vars(args) if not isinstance(args, dict) else args,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")
