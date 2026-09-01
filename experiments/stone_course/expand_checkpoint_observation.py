"""Warm-start the 3D stone curriculum from an eight-feature flat checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OLD_COURSE_FEATURES = 8
NEW_COURSE_FEATURES = 20
INSERTED_FEATURES = NEW_COURSE_FEATURES - OLD_COURSE_FEATURES


def _expand_input_weight(
    weight: torch.Tensor,
    old_observations: int,
    action_features: int = 0,
) -> torch.Tensor:
    """Insert zero columns after the legacy four-target X/Z observation."""
    expected = old_observations + action_features
    if weight.ndim != 2 or weight.shape[1] != expected:
        raise ValueError(
            f"expected a 2D input weight with {expected} columns, got {tuple(weight.shape)}"
        )
    new_observations = old_observations + INSERTED_FEATURES
    expanded = weight.new_zeros((weight.shape[0], new_observations + action_features))
    expanded[:, :OLD_COURSE_FEATURES] = weight[:, :OLD_COURSE_FEATURES]
    expanded[:, NEW_COURSE_FEATURES:new_observations] = weight[
        :, OLD_COURSE_FEATURES:old_observations
    ]
    if action_features:
        expanded[:, new_observations:] = weight[:, old_observations:]
    return expanded


def _expand_normalizer_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError(f"unsupported normalizer tensor shape for {name}: {value.shape}")
    old_observations = value.shape[1]
    expanded = value.new_zeros((1, old_observations + INSERTED_FEATURES))
    expanded[:, :OLD_COURSE_FEATURES] = value[:, :OLD_COURSE_FEATURES]
    expanded[:, NEW_COURSE_FEATURES:] = value[:, OLD_COURSE_FEATURES:]

    # Four target top heights begin around one metre below the pelvis. The
    # orientation features begin at zero. These priors avoid a startup spike;
    # a deliberately small count lets all statistics adapt within a few steps.
    if name == "_mean":
        expanded[:, OLD_COURSE_FEATURES:12] = -1.0
    elif name == "_var":
        expanded[:, OLD_COURSE_FEATURES:12] = 0.04
        expanded[:, 12:NEW_COURSE_FEATURES] = 1.0
    elif name == "_std":
        expanded[:, OLD_COURSE_FEATURES:12] = 0.2
        expanded[:, 12:NEW_COURSE_FEATURES] = 1.0
    else:
        raise ValueError(f"unsupported normalizer tensor: {name}")
    return expanded


def expand_checkpoint(checkpoint: dict) -> dict:
    """Return a migrated copy suitable for the 20-feature terrain observer."""
    migrated = dict(checkpoint)
    actor = dict(checkpoint["actor_state_dict"])
    old_observations = actor["net.0.weight"].shape[1]
    actor["net.0.weight"] = _expand_input_weight(
        actor["net.0.weight"], old_observations
    )
    migrated["actor_state_dict"] = actor

    for state_name in ("qnet_state_dict", "qnet_target_state_dict"):
        state = dict(checkpoint[state_name])
        for key, value in tuple(state.items()):
            if key.endswith(".net.0.weight"):
                action_features = value.shape[1] - old_observations
                if action_features <= 0:
                    raise ValueError(f"could not infer action width from {state_name}.{key}")
                state[key] = _expand_input_weight(
                    value, old_observations, action_features
                )
        migrated[state_name] = state

    normalizer = dict(checkpoint["obs_normalizer_state"])
    for name in ("_mean", "_var", "_std"):
        normalizer[name] = _expand_normalizer_tensor(name, normalizer[name])
    normalizer["count"] = normalizer["count"].new_tensor(1024)
    migrated["obs_normalizer_state"] = normalizer
    migrated["environment_state"] = {
        "terrain_curriculum": {
            "current_maximum": 0.80,
            "current_elevation_maximum_degrees": 0.0,
            "current_yaw_maximum_degrees": 0.0,
            "current_surface_tilt_maximum_degrees": 0.0,
            "episodes": 0,
            "successes": 0,
            "last_completion_rate": 0.0,
        }
    }
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    migrated = expand_checkpoint(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    print(
        f"Expanded observation {checkpoint['actor_state_dict']['net.0.weight'].shape[1]}"
        f" -> {migrated['actor_state_dict']['net.0.weight'].shape[1]}: {args.output}"
    )


if __name__ == "__main__":
    main()
