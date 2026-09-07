"""Left/right mirroring of StoneCourse observations, actions, and worlds.

The course runs along +X with Y up; the mirror plane is z = 0. A mirrored
transition is physically valid training data because the model, the reward,
and the (statistically symmetric) course distribution are all invariant
under this reflection, so every collected transition can be flipped and
added to the replay buffer as if it had been experienced.

Sign conventions verified by the sanity check in __main__:
- Root orientation is a quaternion stored in qpos as
  (pelvis_tilt, pelvis_list, pelvis_rotation, pelvis_quat_w) = (qx, qy, qz, qw).
  Reflecting across z = 0 negates qx and qy and keeps qz, qw.
- Root angular velocity (wx, wy, wz) negates wx and wy; pelvis_tz negates.
- Paired joint coordinates (hip_*, knee_*, ankle_*, shoulder_*, subtalar_*,
  elbow_*, mtp_*) swap sides with unchanged sign: OpenSim defines each
  side's positive direction mirror-symmetrically.
- Unpaired lateral coordinates (torso_bending, torso_rotation) negate;
  sagittal ones (torso_extension) are unchanged.
- Slab surface tilts are (roll about X, pitch about Z): roll negates,
  pitch is unchanged.

Run the sanity check (uses two worlds, one the mirror of the other):
    python experiments/stone_course/symmetry.py \
        models/<run>/<checkpoint>.pt
"""

import dataclasses
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX, UP_IDX  # noqa: E402


from msk_envs.utils.symmetry import (  # noqa: E402
    MirrorSpec,
    QPOS_NEGATE,
    QVEL_NEGATE,
    _permutation_and_signs,
    build_stone_course_mirror_spec as build_mirror_spec,
    flip_qpos,
    flip_qvel,
    mirror_world_layout,
)


def obs_block_slices(env) -> dict[str, slice]:
    """Column ranges of each observation block, for diagnostics."""
    lookahead = env.course.lookahead
    n_muscles = env.muscle_activations.shape[1]
    n_actuators = env.actuator_activations.shape[1]
    n_qpos = env.joint_positions.shape[1]
    n_qvel = env.joint_velocities.shape[1]
    blocks = {}
    offset = 0
    blocks["course"] = slice(offset, offset + 5 * lookahead); offset += 5 * lookahead
    blocks["muscle_act"] = slice(offset, offset + n_muscles); offset += n_muscles
    blocks["fiber_len"] = slice(offset, offset + n_muscles); offset += n_muscles
    blocks["actuator"] = slice(offset, offset + n_actuators); offset += n_actuators
    blocks["qpos"] = slice(offset, offset + n_qpos - 1); offset += n_qpos - 1
    blocks["qvel"] = slice(offset, offset + n_qvel); offset += n_qvel
    if env.command_speed_range != (0.0, 0.0):
        blocks["command"] = slice(offset, offset + 1)
    return blocks


def report_block_errors(spec, env, obs: torch.Tensor, label: str) -> None:
    flipped = spec.flip_obs(obs[1:2])
    for name, sl in obs_block_slices(env).items():
        err = (flipped[0, sl] - obs[0, sl]).abs()
        print(f"  {label} {name:11s} max_err={err.max().item():.3e} "
              f"argmax_col={sl.start + int(err.argmax())}")


def main() -> int:
    import argparse
    import tyro
    from msk_envs.train.hyperparams import StoneCourseConfig
    from msk_envs.train.nets.deterministic_policy import load_policy
    from msk_envs.envs.env_stone_course import StoneCourseEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=60)
    args, remaining = parser.parse_known_args()

    train_args = tyro.cli(StoneCourseConfig, args=[
        "--disable-wandb", "env-config:sprinter",
        # Match the heritage training terrain, deterministically.
        "--env-config.course-step-length-range", "0.40", "1.50",
        "--env-config.course-initial-step-length-max", "0.70",
        "--env-config.no-course-require-interior-landing",
        "--env-config.no-course-terminate-below-supports",
        "--env-config.no-course-terminate-on-ground-contact",
        "--env-config.no-apply-start-noise",
        "--env-config.no-apply-swap-lr",
        "--env-config.course-lateral-jitter", "0.10",
        "--env-config.course-alternating-lateral-offset", "0.12",
        # Recycling samples new random slabs per world independently, which
        # would silently break the mirror mid-rollout. Push it out of reach.
        "--env-config.course-recycle-distance-behind", "1000.0",
    ] + remaining)

    policy = load_policy(args.checkpoint).to(args.device)
    env = StoneCourseEnv(
        num_envs=2, env_config=train_args.env_config, device=args.device,
        requires_visuals=False, cuda_graph=False,
    )
    env.reset()
    spec = build_mirror_spec(env)

    # World 1 becomes the exact mirror of world 0: layout and body state.
    mirror_world_layout(env, 0, 1)
    muscle_perm = spec.act_perm[:env.num_muscles].to(env.device)
    env.joint_positions[1] = flip_qpos(env.joint_positions[0:1], env)[0]
    env.joint_velocities[1] = flip_qvel(env.joint_velocities[0:1], env)[0]
    env.muscle_activations[1] = env.muscle_activations[0, muscle_perm]
    env.actuator_activations[1] = env.actuator_activations[0]
    # Re-derive all pose-dependent state (fiber lengths, collider positions)
    # from the mirrored qpos before reading any observation. launch_sim_reset
    # only touches worlds flagged in reset_tensor, so flag them explicitly.
    env.reset_tensor.fill_(1.0)
    env.launch_sim_reset()
    env.reset_tensor.fill_(0.0)

    obs = env._get_obs()
    obs_err0 = (spec.flip_obs(obs[1:2]) - obs[0:1]).abs().max().item()
    print(f"initial mirrored-obs error: {obs_err0:.2e}")
    if obs_err0 > 1e-3:
        report_block_errors(spec, env, obs, "t=0")

    worst_obs = worst_pos = 0.0
    for step in range(args.steps):
        obs = env._get_obs()
        with torch.no_grad():
            action_0 = policy(obs[0:1])
            action_1 = spec.flip_action(policy(spec.flip_obs(obs[1:2])))
        actions = torch.cat((action_0, action_1), dim=0)
        env.pre_sim_step(actions)
        env.launch_sim_step()
        env.update_metrics()
        env._compute_raw_reward_dict()

        root = env.root_pos
        mirrored_root = root[1].clone()
        mirrored_root[SIDE_IDX] *= -1.0
        pos_err = (mirrored_root - root[0]).abs().max().item()
        obs_now = env._get_obs()
        obs_err = (spec.flip_obs(obs_now[1:2]) - obs_now[0:1]).abs().max().item()
        worst_pos = max(worst_pos, pos_err)
        worst_obs = max(worst_obs, obs_err)
        if step % 15 == 14:
            print(f"step {step + 1:3d}: root divergence {pos_err:.2e}, "
                  f"obs divergence {obs_err:.2e}")
            if obs_err > 0.5:
                report_block_errors(spec, env, obs_now, f"t={step + 1}")

    print(f"\nover {args.steps} steps ({args.steps / 30.0:.1f} s):")
    print(f"  max mirrored root divergence: {worst_pos:.3e} m")
    print(f"  max mirrored obs divergence:  {worst_obs:.3e}")
    verdict = "PASS" if worst_pos < 5e-2 else "FAIL"
    print(f"  sanity check: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
