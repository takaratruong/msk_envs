"""Render shoulder-girdle motion under SYMMETRIC muscle activation, back view.

Drives every muscle with the SAME activation on left and right (mirror-symmetric
input), so any left/right difference in the resulting scapula/clavicle/humerus
motion is a model-setup asymmetry, not a control choice. Emits:
  - a matplotlib animation (back view: looking down +X, so lateral = horizontal,
    up = vertical) of the girdle bodies, saved as mp4
  - printed L/R trajectories + mirror residual over the rollout
"""
import argparse
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/ubuntu/msk_envs-stone-course")
import bolt

MODEL = "msk_envs/msk_models/sprinter/sprinter_model_shoulderonly.osim"
FN = "msk_envs/msk_models/sprinter/sprinter_model_shoulderonly_fn.xml"
GIRDLE = ["torso", "clavicle_r", "scapula_r", "humerus_r",
          "clavicle_l", "scapula_l", "humerus_l"]


def quat_R(qxyzw):
    x, y, z, w = qxyzw
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fn", default=FN)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--out", default="experiments/spine_shoulder/shoulders_back_view.mp4")
    ap.add_argument("--tag", default="shoulderonly")
    args = ap.parse_args()

    wp.init()
    res = bolt.load_model(model_path=args.model, n_worlds=1,
                          integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                          requires_visuals=False, muscle_fn_path=args.fn,
                          render_kinematic_tree=False)
    m, d = res.model, res.data
    B = res.body_id_lookup
    midx = res.muscle_id_lookup

    # Build a symmetric excitation vector: same value for _r and _l of each muscle.
    nmus = len(midx)
    exc = wp.to_torch(d.m_excitations)
    # group muscles by side-stripped base name
    base_to_ids = {}
    for name, i in midx.items():
        base = name[:-2] if name.endswith(("_r", "_l")) else name
        base_to_ids.setdefault(base, []).append(i)

    def set_symmetric(level):
        v = exc.clone() * 0
        for base, ids in base_to_ids.items():
            v[0, ids] = level  # identical activation on both sides
        exc[:] = v

    bolt.fk(m, d)
    traj = {b: [] for b in GIRDLE}
    scap_rel = {"r": [], "l": []}

    # slow symmetric activation ramp: shoulder muscles pulse together
    for t in range(args.steps):
        level = 0.05 + 0.5 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 60.0))  # 0.05..0.55 pulsing
        set_symmetric(float(level))
        bolt.increment_next_time(m, d, 1.0 / 60.0)
        bolt.step(m, d)
        X = wp.to_torch(d.mob_X_GB).detach().cpu().numpy()[0]
        for b in GIRDLE:
            traj[b].append(X[B[b]][:3].copy())
        # scapula in torso frame
        pT = X[B["torso"]][:3]; RT = quat_R(X[B["torso"]][3:])
        for s in ("r", "l"):
            scap_rel[s].append(RT.T @ (X[B[f"scapula_{s}"]][:3] - pT))

    for b in traj:
        traj[b] = np.array(traj[b])
    sr = np.array(scap_rel["r"]); sl = np.array(scap_rel["l"])
    # mirror residual: r.x-l.x, r.y-l.y, r.z+l.z should be ~0 for symmetric model
    resid = np.stack([sr[:, 0]-sl[:, 0], sr[:, 1]-sl[:, 1], sr[:, 2]+sl[:, 2]], 1)
    print(f"[{args.tag}] scapula mirror residual over rollout: "
          f"mean={np.linalg.norm(resid,axis=1).mean():.4f} max={np.linalg.norm(resid,axis=1).max():.4f}")
    print(f"  scap_r vertical(y) range: {sr[:,1].min():.3f}..{sr[:,1].max():.3f} (bob = {sr[:,1].ptp():.3f} m)")
    print(f"  scap_l vertical(y) range: {sl[:,1].min():.3f}..{sl[:,1].max():.3f} (bob = {sl[:,1].ptp():.3f} m)")

    # ---- back-view animation: X into page, so plot lateral(Z) vs up(Y) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Shoulder girdle, BACK VIEW, symmetric activation ({args.tag})")
    ax.set_xlabel("lateral  (+Z, right shoulder →)")
    ax.set_ylabel("up (+Y)")
    allz = np.concatenate([traj[b][:, 2] for b in GIRDLE])
    ally = np.concatenate([traj[b][:, 1] for b in GIRDLE])
    ax.set_xlim(allz.min()-0.05, allz.max()+0.05)
    ax.set_ylim(ally.min()-0.05, ally.max()+0.05)
    ax.set_aspect("equal")
    ax.axvline(0, color="0.8", lw=0.8)

    colors = {"torso": "k", "clavicle_r": "tab:blue", "scapula_r": "tab:red",
              "humerus_r": "tab:green", "clavicle_l": "tab:cyan",
              "scapula_l": "tab:orange", "humerus_l": "tab:olive"}
    pts = {b: ax.plot([], [], "o", color=colors[b], label=b, ms=9)[0] for b in GIRDLE}
    # bones: connect clavicle-scapula-humerus each side, and torso to clavicles
    links = [("torso", "clavicle_r"), ("clavicle_r", "scapula_r"), ("scapula_r", "humerus_r"),
             ("torso", "clavicle_l"), ("clavicle_l", "scapula_l"), ("scapula_l", "humerus_l")]
    lines = [ax.plot([], [], "-", color="0.5", lw=1.5)[0] for _ in links]
    trail_r = ax.plot([], [], "-", color="tab:red", lw=0.8, alpha=0.5)[0]
    trail_l = ax.plot([], [], "-", color="tab:orange", lw=0.8, alpha=0.5)[0]
    ax.legend(loc="upper right", fontsize=7)

    def frame(i):
        for b in GIRDLE:
            pts[b].set_data([traj[b][i, 2]], [traj[b][i, 1]])
        for ln, (a, c) in zip(lines, links):
            ln.set_data([traj[a][i, 2], traj[c][i, 2]], [traj[a][i, 1], traj[c][i, 1]])
        trail_r.set_data(traj["scapula_r"][:i+1, 2], traj["scapula_r"][:i+1, 1])
        trail_l.set_data(traj["scapula_l"][:i+1, 2], traj["scapula_l"][:i+1, 1])
        return list(pts.values()) + lines + [trail_r, trail_l]

    anim = animation.FuncAnimation(fig, frame, frames=len(traj["torso"]), interval=33, blit=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer="ffmpeg", fps=30, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
