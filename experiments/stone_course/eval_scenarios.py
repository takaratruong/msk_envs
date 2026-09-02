"""Fixed evaluation scenarios for stepping-stone policies.

Rolls a checkpoint through a set of deterministic courses so successive
checkpoints and runs can be compared on identical terrain:

- flat:    even spacing at walking stride on one level, like flat ground
- ascent:  continuously climbing at a fixed elevation angle
- descent: continuously dropping, steepening until the height floor
- rolling: alternating climb/drop waves
- random-<seed>: seeded random courses at the given difficulty

Each scenario writes a trajectory JSON to
dashboard/trajectories/<experiment>/scenario_<name>_0.json.gz plus a summary
line in scenarios_summary.json, so the ordinary renderer can video them.

Usage (from the repository root, in the Bolt conda environment):
    python experiments/stone_course/eval_scenarios.py \
        models/<run>/<checkpoint>.pt --out-tag stride_mix_149000
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from msk_envs.envs.env_stone_course import StoneCourseEnv, FWD_IDX, UP_IDX, SIDE_IDX
from msk_envs.train.hyperparams import StoneCourseConfig
from msk_envs.train.nets.deterministic_policy import load_policy
from msk_envs.utils.logged_sim import LoggedSim

import tyro


# (forward gap m, elevation rad) per fixed scenario
SCENARIO_GAPS = {
    "flat": (0.80, 0.0),
    "ascent": (0.90, math.radians(12.0)),
    "descent": (0.90, math.radians(-6.0)),
    "rolling": (0.90, math.radians(18.0)),
}


def scenario_positions(spec, kind: str, num_stones: int, device) -> torch.Tensor:
    """Deterministic slab centers for one named scenario."""
    forward_gap, elevation = SCENARIO_GAPS[kind]
    min_top, max_top = spec.top_height_range
    positions = torch.zeros((num_stones, 3))
    top = spec.top_height
    x = 0.0
    for index in range(num_stones):
        sign = 1.0 if index % 2 == 0 else -1.0
        if index < 2:
            x += sum(spec.launch_step_length_range) * 0.5
        else:
            x += forward_gap
            angle = elevation
            if kind == "rolling":
                # two up, two down waves, climbing first
                angle = elevation if ((index - 2) // 2) % 2 == 0 else -elevation
            if kind == "descent":
                # steepen with every slab until the floor
                angle = elevation * (1.0 + 0.5 * (index - 2))
            top = float(min(max(top + forward_gap * math.tan(angle), min_top), max_top))
        positions[index, FWD_IDX] = x
        positions[index, UP_IDX] = top - spec.half_extents[UP_IDX]
        positions[index, SIDE_IDX] = sign * spec.alternating_lateral_offset
    return positions.to(device)


class ScenarioStoneCourseEnv(StoneCourseEnv):
    """StoneCourseEnv whose layouts come from a fixed generator, not random."""

    scenario_kind: str | None = None
    scenario_generator: torch.Generator | None = None

    def _sample_reset_layouts(self, world_ids):
        if self.scenario_kind is None:  # seeded random scenario
            positions = self.course.sample_positions(
                world_ids.numel(),
                self.device,
                generator=self.scenario_generator,
                step_length_max=self.terrain_curriculum.current_maximum,
                elevation_angle_max_degrees=(
                    self.terrain_curriculum.current_elevation_maximum_degrees
                ),
                yaw_angle_max_degrees=self.terrain_curriculum.current_yaw_maximum_degrees,
                step_length_min=self.step_length_minimums[world_ids],
            )
            surface_tilts = self.course.sample_surface_tilts(
                world_ids.numel() * self.course.num_stones,
                self.device,
                self.terrain_curriculum.current_surface_tilt_maximum_degrees,
                generator=self.scenario_generator,
            ).reshape(world_ids.numel(), self.course.num_stones, 2)
            surface_tilts[:, : self.course.fixed_flat_stones] = 0.0
            return positions, surface_tilts
        positions = scenario_positions(
            self.course, self.scenario_kind, self.course.num_stones, self.device
        ).unsqueeze(0).repeat(world_ids.numel(), 1, 1)
        tilts = torch.zeros(
            (world_ids.numel(), self.course.num_stones, 2), device=self.device
        )
        return positions, tilts

    def _sample_recycled_slabs(self, world_ids, predecessors):
        if self.scenario_kind is None:
            positions = self.course.sample_next_position(
                predecessors,
                self.next_lateral_sign[world_ids],
                self.terrain_curriculum.current_maximum,
                self.terrain_curriculum.current_elevation_maximum_degrees,
                self.terrain_curriculum.current_yaw_maximum_degrees,
                generator=self.scenario_generator,
                step_length_min=self.step_length_minimums[world_ids],
            )
            tilts = self.course.sample_surface_tilts(
                world_ids.numel(),
                self.device,
                self.terrain_curriculum.current_surface_tilt_maximum_degrees,
                generator=self.scenario_generator,
            )
            return positions, tilts
        # Deterministic continuation of the pattern from the furthest slab.
        forward_gap, elevation = SCENARIO_GAPS[self.scenario_kind]
        min_top, max_top = self.course.top_height_range
        positions = predecessors.clone()
        positions[:, FWD_IDX] += forward_gap
        if self.scenario_kind == "rolling":
            wave = torch.sin(predecessors[:, FWD_IDX] * (math.pi / (2 * 0.9)))
            rise = forward_gap * math.tan(elevation) * torch.sign(wave)
        else:
            rise = torch.full_like(
                predecessors[:, UP_IDX], forward_gap * math.tan(elevation)
            )
        half_up = self.course.half_extents[UP_IDX]
        top = (predecessors[:, UP_IDX] + half_up + rise).clamp(min_top, max_top)
        positions[:, UP_IDX] = top - half_up
        positions[:, SIDE_IDX] = (
            self.next_lateral_sign[world_ids] * self.course.alternating_lateral_offset
        )
        tilts = torch.zeros((world_ids.numel(), 2), device=self.device)
        return positions, tilts


def run_scenario(env, policy, device) -> tuple[LoggedSim, dict]:
    sim = LoggedSim(env, device=device)
    obs = sim.reset()
    start_x = float(env.root_pos[0, FWD_IDX].item())
    peak_x = start_x
    for _ in range(sim.max_env_steps):
        with torch.no_grad():
            actions = policy(obs)
        finished, obs = sim.step(actions)
        if not bool(sim.finished[0]):
            peak_x = max(peak_x, float(env.root_pos[0, FWD_IDX].item()))
        if finished:
            break
    steps = int(sim.get_episode_length_mean().item())
    stats = {
        "duration_s": round(steps * env.delta_t, 2),
        "distance_m": round(peak_x - start_x, 2),
        "mean_reward": round(float(sim.get_rewards_mean().item()), 3),
        "episode_length": steps,
    }
    return sim, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-tag", default=None,
                        help="trajectory folder name; defaults to scenarios_<ckpt stem>")
    parser.add_argument("--random-seeds", type=int, nargs="*", default=[101, 202])
    parser.add_argument("--device", default="cuda:0")
    args, remaining = parser.parse_known_args()

    train_args = tyro.cli(
        StoneCourseConfig,
        args=["--disable-wandb", "env-config:sprinter"] + remaining,
    )
    cfg = train_args.env_config

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = load_policy(args.checkpoint).to(device=args.device)

    tag = args.out_tag or f"scenarios_{args.checkpoint.stem}"
    out_dir = REPO_ROOT / "dashboard" / "trajectories" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = ["flat", "ascent", "descent", "rolling"]
    scenarios = [(name, None) for name in fixed] + [
        (f"random-{seed}", seed) for seed in args.random_seeds
    ]

    # One environment reused for every scenario: building it is expensive
    # (Bolt model construction and, on first use, Warp kernel compilation).
    # cuda_graph=True matches the trainer's eval env, so the Warp kernel
    # cache from training runs is reused instead of recompiling from scratch.
    env = ScenarioStoneCourseEnv(
        num_envs=1, env_config=cfg, device=args.device,
        requires_visuals=True, cuda_graph=True,
    )
    state = checkpoint.get("environment_state")
    if state:
        env.load_task_state(state)

    summary = {}
    for name, seed in scenarios:
        if seed is None:
            env.scenario_kind = name
            env.scenario_generator = None
        else:
            env.scenario_kind = None
            env.scenario_generator = torch.Generator(device=args.device)
            env.scenario_generator.manual_seed(seed)
        sim, stats = run_scenario(env, policy, args.device)
        sim.save_animation(str(out_dir), f"scenario_{name}", use_gzip=True)
        summary[name] = stats
        print(f"{name}: {stats}", flush=True)

    summary_path = out_dir / "scenarios_summary.json"
    summary_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "tag": tag,
        "scenarios": summary,
    }, indent=2))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
