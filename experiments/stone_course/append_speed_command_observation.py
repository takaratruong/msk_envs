"""Append a commanded-speed observation column to a stone-course checkpoint.

The speed command is the LAST observation feature (see StoneCourseEnv._get_obs),
so actor and critic input layers gain one trailing zero column and the
normalizer gains a trailing entry with a fixed prior matching the command
range midpoint. Zero policy weights mean the migrated policy initially ignores
the command and behaves exactly like its parent.

Usage:
    python experiments/stone_course/append_speed_command_observation.py \
        models/<run>/<checkpoint>.pt models/<output>.pt --speed-range 0.9 1.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _append_input_column(weight: torch.Tensor, action_features: int = 0) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D input weight, got {tuple(weight.shape)}")
    observations = weight.shape[1] - action_features
    expanded = weight.new_zeros((weight.shape[0], weight.shape[1] + 1))
    expanded[:, :observations] = weight[:, :observations]
    if action_features:
        expanded[:, observations + 1:] = weight[:, observations:]
    return expanded


def append_speed_command(checkpoint: dict, speed_low: float, speed_high: float) -> dict:
    migrated = dict(checkpoint)
    actor = dict(checkpoint["actor_state_dict"])
    old_observations = actor["net.0.weight"].shape[1]
    actor["net.0.weight"] = _append_input_column(actor["net.0.weight"])
    migrated["actor_state_dict"] = actor

    for state_name in ("qnet_state_dict", "qnet_target_state_dict"):
        state = dict(checkpoint[state_name])
        for key, value in tuple(state.items()):
            if key.endswith(".net.0.weight"):
                action_features = value.shape[1] - old_observations
                if action_features <= 0:
                    raise ValueError(f"could not infer action width from {state_name}.{key}")
                state[key] = _append_input_column(value, action_features)
        migrated[state_name] = state

    normalizer = dict(checkpoint["obs_normalizer_state"])
    mid = (speed_low + speed_high) * 0.5
    span = max(speed_high - speed_low, 1e-3)
    # Uniform command in [low, high]: match its true mean and variance so the
    # normalized feature lands near a unit scale from the first step.
    var = span * span / 12.0
    for name, prior in (("_mean", mid), ("_var", var), ("_std", var ** 0.5)):
        value = normalizer[name]
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"unsupported normalizer tensor shape for {name}: {value.shape}")
        expanded = value.new_zeros((1, value.shape[1] + 1))
        expanded[:, :-1] = value
        expanded[:, -1] = prior
        normalizer[name] = expanded
    migrated["obs_normalizer_state"] = normalizer
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed-range", nargs=2, type=float, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    migrated = append_speed_command(checkpoint, *args.speed_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    print(
        f"Appended speed-command column {checkpoint['actor_state_dict']['net.0.weight'].shape[1]}"
        f" -> {migrated['actor_state_dict']['net.0.weight'].shape[1]}: {args.output}"
    )


if __name__ == "__main__":
    main()
