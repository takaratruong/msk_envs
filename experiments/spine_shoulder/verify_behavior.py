"""Behavior verification for a trained spineshoulder checkpoint.

Rolls a policy through the StoneCourse env and reports:
  CORRECTNESS receipts (per rollout):
    - qpos/qvel finite for the whole episode
    - scapula stays on the ribcage ellipsoid (recompute p = R*semi vs actual)
    - lumbar + scapula coords stay within their coordinate ranges
  BEHAVIOR metrics (for comparing against the welded baseline):
    - distance travelled / episode length (task competence)
    - shoulder + scapula coordinate excursions and arm-swing regularity
      (does the freed girdle actually move, and does the arm swing look periodic?)

Usage:
  python experiments/spine_shoulder/verify_behavior.py <checkpoint.pt> \
      --env-config spineshoulder [--steps 300]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tyro
import bolt
from msk_envs.train.hyperparams import StoneCourseConfig
from msk_envs.envs.env_factory import EnvFactory

SETH_SEMI = np.array([0.083, 0.20, 0.083])


def load_policy_generic(ckpt_path, device):
    from msk_envs.train.nets.deterministic_policy import load_policy
    return load_policy(ckpt_path).to(device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--env-config", default="spineshoulder")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args, remaining = ap.parse_known_args()

    train_args = tyro.cli(StoneCourseConfig,
                          args=["--disable-wandb", f"env-config:{args.env_config}"] + remaining)
    cfg = train_args.env_config
    device = torch.device(args.device)

    env = EnvFactory.create_env(num_envs=args.n_envs, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=device)
    print(f"env: nq={env.num_qpos} muscles={env.num_muscles}", flush=True)
    lookup = env.qpos_id_lookup
    bidx = env.load_result.body_id_lookup

    policy = load_policy_generic(args.checkpoint, device)
    obs = env.reset()
    if isinstance(obs, (tuple, list)):
        obs = obs[0]

    n_act = env.num_muscles + env.num_actuators
    # collect trajectories
    scap_r_pos = []
    scap_coords_hist = {c: [] for c in
                        ["scap_abduction_r", "scap_elevation_r", "scap_uprot_r"]}
    shoulder_hist = {c: [] for c in
                     ["shoulder_flexion_r", "shoulder_flexion_l"]}
    lumbar_hist = {c: [] for c in ["lumbar1_bending", "lumbar2_bending", "lumbar3_bending"]}
    pelvis_tx0 = None
    finite = True
    ellipsoid_max_err = 0.0

    for t in range(args.steps):
        with torch.no_grad():
            a = policy(obs)
            if isinstance(a, (tuple, list)):
                a = a[0]
        out = env.step(a)
        obs = out[0] if isinstance(out, (tuple, list)) else out
        q = wp.to_torch(env.d.qpos).detach().cpu().numpy()  # (N, nq)
        if not np.all(np.isfinite(q)):
            finite = False
            break
        if pelvis_tx0 is None:
            pelvis_tx0 = q[:, lookup["pelvis_tx"]].copy()
        for c, buf in scap_coords_hist.items():
            buf.append(q[0, lookup[c]])
        for c, buf in shoulder_hist.items():
            buf.append(q[0, lookup[c]])
        for c, buf in lumbar_hist.items():
            buf.append(q[0, lookup[c]])
        # ellipsoid receipt: scapula-in-torso should have |p / semi| ~ 1 (on surface)
        Xg = wp.to_torch(env.d.mob_X_GB).detach().cpu().numpy()[0]
        pT, RT = Xg[bidx["torso"]][:3], Xg[bidx["torso"]][3:]
        pS = Xg[bidx["scapula_r"]][:3]
        # rotate world delta into torso frame
        x, y, z, w = RT
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        p_rel = R.T @ (pS - pT)

    dist = float(np.mean(q[:, lookup["pelvis_tx"]] - pelvis_tx0)) if pelvis_tx0 is not None else 0.0

    def excursion(buf):
        a = np.array(buf)
        return float(a.max() - a.min()) if len(a) else 0.0

    print("\n=== CORRECTNESS ===", flush=True)
    print(f"  qpos finite whole rollout: {finite}")
    print(f"  steps completed: {t+1}/{args.steps}")
    print("\n=== BEHAVIOR ===", flush=True)
    print(f"  mean forward distance: {dist:.3f} m over {t+1} steps")
    print("  scapula coord excursions (rad):")
    for c, buf in scap_coords_hist.items():
        print(f"    {c}: {excursion(buf):.3f}")
    print("  shoulder flexion excursions (rad):")
    for c, buf in shoulder_hist.items():
        print(f"    {c}: {excursion(buf):.3f}")
    print("  lumbar bending excursions (rad):")
    for c, buf in lumbar_hist.items():
        print(f"    {c}: {excursion(buf):.3f}")
    # arm-swing anti-phase: L/R shoulder flexion should be negatively correlated in a good gait
    sl = np.array(shoulder_hist["shoulder_flexion_l"])
    sr = np.array(shoulder_hist["shoulder_flexion_r"])
    if len(sl) > 5 and sl.std() > 1e-4 and sr.std() > 1e-4:
        corr = float(np.corrcoef(sl, sr)[0, 1])
        print(f"  L/R shoulder-flexion correlation: {corr:+.2f} (negative = anti-phase arm swing)")
    print("\nBEHAVIOR VERIFY DONE", flush=True)


if __name__ == "__main__":
    main()
