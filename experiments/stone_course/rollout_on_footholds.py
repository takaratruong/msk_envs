"""Roll a trained policy over the exact footholds of the terrain visualization.

terrain_from_curriculum.py samples foothold placements (slab tops) from the
curriculum distribution to render "equivalent natural terrain" images. This
script builds a fixed 14-slab course whose slab tops are those exact footholds
(mid stage, STAGES[2]) and rolls a checkpoint over it, so the resulting video
and the terrain image share one foothold set.

Adapted from eval_scenarios.py: same env subclass seam (_sample_reset_layouts
with fixed positions) and the same rollout/save loop via LoggedSim. Surface
tilts are zero; recycled slabs continue forward deterministically at the last
top height. Interior-landing termination is disabled for evaluation so an
imperfect landing does not cut the rollout.

Usage (bolt conda env, repo root):
    CUDA_VISIBLE_DEVICES=7 python experiments/stone_course/rollout_on_footholds.py \
        models/<run>/<checkpoint>.pt
"""

import argparse
import json
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

from experiments.stone_course.terrain_from_curriculum import (
    STAGES,
    curriculum_spec,
    sample_footholds,
)

CONTINUATION_GAP = 0.90  # walking-stride forward gap for recycled slabs


def foothold_positions(stage: dict, seed: int, device) -> torch.Tensor:
    """Slab centers (env axes) whose tops are the visualization footholds."""
    spec = curriculum_spec()
    fh = sample_footholds(spec, stage, seed=seed)  # (N,3): x, lateral, top
    positions = torch.zeros((fh.shape[0], 3))
    positions[:, FWD_IDX] = torch.from_numpy(fh[:, 0])
    positions[:, UP_IDX] = torch.from_numpy(fh[:, 2]) - spec.half_extents[1]
    positions[:, SIDE_IDX] = torch.from_numpy(fh[:, 1])
    return positions.to(device)


class FootholdStoneCourseEnv(StoneCourseEnv):
    """StoneCourseEnv whose reset layout is a fixed foothold course."""

    course_positions: torch.Tensor | None = None  # (num_stones, 3), env axes

    def _sample_reset_layouts(self, world_ids):
        positions = self.course_positions.unsqueeze(0).repeat(
            world_ids.numel(), 1, 1
        )
        tilts = torch.zeros(
            (world_ids.numel(), self.course.num_stones, 2), device=self.device
        )
        return positions, tilts

    def _sample_recycled_slabs(self, world_ids, predecessors):
        # Deterministic continuation past the fixed course: same top height,
        # walking-stride gap, alternating lateral offset, flat tops.
        min_top, max_top = self.course.top_height_range
        half_up = self.course.half_extents[UP_IDX]
        positions = predecessors.clone()
        positions[:, FWD_IDX] += CONTINUATION_GAP
        top = (predecessors[:, UP_IDX] + half_up).clamp(min_top, max_top)
        positions[:, UP_IDX] = top - half_up
        positions[:, SIDE_IDX] = (
            self.next_lateral_sign[world_ids] * self.course.alternating_lateral_offset
        )
        tilts = torch.zeros((world_ids.numel(), 2), device=self.device)
        return positions, tilts


def run_rollout(env, policy, device) -> tuple[LoggedSim, dict]:
    sim = LoggedSim(env, device=device)
    obs = sim.reset()
    start_x = float(env.root_pos[0, FWD_IDX].item())
    peak_x = start_x
    outcome = "time limit"
    for _ in range(sim.max_env_steps):
        with torch.no_grad():
            actions = policy(obs)
        was_finished = bool(sim.finished[0])
        finished, obs = sim.step(actions)
        if not was_finished and bool(sim.finished[0]) and not bool(
            env._last_timed_out[0]
        ):
            outcome = (
                "edge-landing termination"
                if bool(env._last_edge_violation[0])
                else "fall"
            )
        if not bool(sim.finished[0]):
            peak_x = max(peak_x, float(env.root_pos[0, FWD_IDX].item()))
        if finished:
            break
    steps = int(sim.get_episode_length_mean().item())
    stats = {
        "outcome": outcome,
        "duration_s": round(steps * env.delta_t, 2),
        "distance_m": round(peak_x - start_x, 2),
        "mean_reward": round(float(sim.get_rewards_mean().item()), 3),
        "episode_length": steps,
    }
    return sim, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*", default=[9, 29, 49],
                        help="foothold seeds; later ones only run if the "
                             "first rollout lasts under --min-duration")
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage-index", type=int, default=2,
                        help="terrain_from_curriculum.STAGES index (default: "
                             "2, the mid stage); ignored if stage params are "
                             "given explicitly")
    parser.add_argument("--step-length-max", type=float, default=None)
    parser.add_argument("--elevation-deg", type=float, default=None)
    parser.add_argument("--yaw-deg", type=float, default=None)
    parser.add_argument("--height-scale", type=float, default=1.0)
    parser.add_argument("--tag", default=None,
                        help="filename prefix for saved trajectories "
                             "(default: 'foothold' for the mid stage, "
                             "'foothold_<stagename>' otherwise)")
    args, remaining = parser.parse_known_args()

    if args.step_length_max is not None:
        stage_name = (f"custom_sl{args.step_length_max:g}"
                      f"_el{args.elevation_deg:g}_yaw{args.yaw_deg:g}")
        stage = dict(step_length_max=args.step_length_max,
                     elevation_deg=args.elevation_deg or 0.0,
                     yaw_deg=args.yaw_deg or 0.0,
                     height_scale=args.height_scale)
    else:
        stage_name, stage = STAGES[args.stage_index][0], STAGES[args.stage_index][1]
    tag = args.tag or ("foothold" if args.stage_index == 2
                       and args.step_length_max is None
                       else f"foothold_{stage_name}")

    # Match the stonecourse_symaug4 training environment (see
    # models/stonecourse_symaug4_launch.log), except interior-landing
    # termination is off so an imperfect landing does not end the video.
    train_args = tyro.cli(StoneCourseConfig, args=[
        "--disable-wandb", "env-config:sprinter",
        "--env-config.model-path", "../msk_models/sprinter/sprinter_model_sym.osim",
        "--env-config.course-stones", "14",
        "--env-config.course-step-length-range", "0.40", "1.50",
        "--env-config.course-lateral-jitter", "0.10",
        "--env-config.course-alternating-lateral-offset", "0.12",
        "--env-config.course-stone-gated-reward",
        "--env-config.no-course-require-interior-landing",
        "--env-config.course-landing-margin-inactive", "0.02",
        "--env-config.course-initial-landing-margin", "0.06",
        "--env-config.no-course-terminate-below-supports",
        "--env-config.no-course-terminate-on-ground-contact",
        "--env-config.course-continuation-probability", "0.95",
        "--env-config.course-initial-height-scale", "1.0",
        "--env-config.course-recycle-distance-behind", "2.0",
    ] + remaining)

    policy = load_policy(args.checkpoint).to(device=args.device)

    out_dir = REPO_ROOT / "dashboard" / "trajectories" / "foothold_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One environment reused for every seed: building it is expensive.
    env = FootholdStoneCourseEnv(
        num_envs=1, env_config=train_args.env_config, device=args.device,
        requires_visuals=True, cuda_graph=True,
    )

    results = {}
    for seed in args.seeds:
        env.course_positions = foothold_positions(stage, seed, args.device)
        # Starting-pose noise draws from the global RNG; seed it per rollout.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sim, stats = run_rollout(env, policy, args.device)
        sim.save_animation(str(out_dir), f"{tag}_seed{seed}", use_gzip=True)
        results[seed] = stats
        print(f"seed {seed}: {stats}", flush=True)
        if stats["duration_s"] >= args.min_duration:
            break

    kept = max(results, key=lambda s: results[s]["duration_s"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "stage": stage_name,
        "stage_params": stage,
        "kept_seed": kept,
        "kept_trajectory": str(out_dir / f"{tag}_seed{kept}_0.json.gz"),
        "rollouts": {str(s): results[s] for s in results},
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"kept seed {kept}: {results[kept]}")
    print(f"Summary: {out_dir / f'summary_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
