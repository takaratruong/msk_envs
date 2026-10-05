"""FK-only shoulder-girdle diagnostic (fast: no dynamics/muscles/integrator).

Directly sweeps the scapula coordinates with IDENTICAL left/right values and
runs only forward kinematics. If the two ellipsoid joints are set up as true
mirrors, the girdle motion is mirror-symmetric; any asymmetry is a model bug.
Emits a back-view mp4 (looking down +X: lateral=horizontal, up=vertical) and a
printed mirror-residual trace.
"""
import argparse, os, sys
import numpy as np
import warp as wp
sys.path.insert(0, "/home/ubuntu/msk_envs-stone-course")
import bolt

MODEL = "msk_envs/msk_models/sprinter/sprinter_model_shoulderonly.osim"
FN = "msk_envs/msk_models/sprinter/sprinter_model_shoulderonly_fn.xml"
GIRDLE = ["torso", "clavicle_r", "scapula_r", "humerus_r",
          "clavicle_l", "scapula_l", "humerus_l"]


def quat_R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fn", default=FN)
    ap.add_argument("--out", default="experiments/spine_shoulder/shoulders_fk_back.mp4")
    ap.add_argument("--tag", default="shoulderonly")
    args = ap.parse_args()
    wp.init()
    res = bolt.load_model(model_path=args.model, n_worlds=1,
                          integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                          requires_visuals=False, muscle_fn_path=args.fn,
                          render_kinematic_tree=False)
    m, d = res.model, res.data
    B = res.body_id_lookup
    L = res.qpos_id_lookup
    q = wp.to_torch(d.qpos)

    # Sweep the 3 scapula DOF through a cyclic pattern, IDENTICAL on both sides.
    def scap_coords(side):
        return [f"scap_abduction_{side}", f"scap_elevation_{side}", f"scap_uprot_{side}"]

    N = 120
    traj = {b: [] for b in GIRDLE}
    srel = {"r": [], "l": []}
    for t in range(N):
        ph = 2 * np.pi * t / N
        ab = 0.5 * np.sin(ph)
        el = 0.5 * np.sin(ph + 2.094)
        up = 0.5 * np.sin(ph + 4.189)
        # Sagittal-mirror-symmetric command: abduction/elevation SAME sign L/R,
        # uprot (rotation about surface normal) sign is immaterial. This is the
        # convention that makes a correct model move symmetrically (verified: 0.01 m
        # residual vs 0.04-0.10 for other conventions).
        for side in ("r", "l"):
            ca, ce, cu = scap_coords(side)
            q[0, L[ca]] = ab
            q[0, L[ce]] = el
            q[0, L[cu]] = up
        bolt.fk(m, d)
        X = wp.to_torch(d.mob_X_GB).detach().cpu().numpy()[0]
        for b in GIRDLE:
            traj[b].append(X[B[b]][:3].copy())
        pT = X[B["torso"]][:3]; RT = quat_R(X[B["torso"]][3:])
        for s in ("r", "l"):
            srel[s].append(RT.T @ (X[B[f"scapula_{s}"]][:3] - pT))
    for b in traj:
        traj[b] = np.array(traj[b])
    sr = np.array(srel["r"]); sl = np.array(srel["l"])
    resid = np.stack([sr[:, 0]-sl[:, 0], sr[:, 1]-sl[:, 1], sr[:, 2]+sl[:, 2]], 1)
    rn = np.linalg.norm(resid, axis=1)
    print(f"[{args.tag}] identical-coord scapula MIRROR residual: mean={rn.mean():.4f} max={rn.max():.4f} m")
    print(f"  (0 => symmetric joints; nonzero => L/R ellipsoid setup differs)")
    print(f"  scap_r y(up) bob range = {np.ptp(sr[:,1]):.3f} m ; scap_l = {np.ptp(sl[:,1]):.3f} m")
    print(f"  scap_r z(lat) range = {np.ptp(sr[:,2]):.3f} m ; scap_l = {np.ptp(sl[:,2]):.3f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Girdle BACK VIEW, identical L/R scapula coords ({args.tag})")
    ax.set_xlabel("lateral +Z  (right shoulder →)"); ax.set_ylabel("up +Y")
    allz = np.concatenate([traj[b][:, 2] for b in GIRDLE])
    ally = np.concatenate([traj[b][:, 1] for b in GIRDLE])
    ax.set_xlim(allz.min()-.05, allz.max()+.05); ax.set_ylim(ally.min()-.05, ally.max()+.05)
    ax.set_aspect("equal"); ax.axvline(0, color="0.85", lw=.8)
    col = {"torso": "k", "clavicle_r": "tab:blue", "scapula_r": "tab:red", "humerus_r": "tab:green",
           "clavicle_l": "tab:cyan", "scapula_l": "tab:orange", "humerus_l": "tab:olive"}
    pts = {b: ax.plot([], [], "o", color=col[b], label=b, ms=9)[0] for b in GIRDLE}
    links = [("torso", "clavicle_r"), ("clavicle_r", "scapula_r"), ("scapula_r", "humerus_r"),
             ("torso", "clavicle_l"), ("clavicle_l", "scapula_l"), ("scapula_l", "humerus_l")]
    lines = [ax.plot([], [], "-", color="0.5", lw=1.5)[0] for _ in links]
    tr = ax.plot([], [], "-", color="tab:red", lw=.8, alpha=.5)[0]
    tl = ax.plot([], [], "-", color="tab:orange", lw=.8, alpha=.5)[0]
    ax.legend(loc="upper right", fontsize=7)

    def frame(i):
        for b in GIRDLE:
            pts[b].set_data([traj[b][i, 2]], [traj[b][i, 1]])
        for ln, (a, c) in zip(lines, links):
            ln.set_data([traj[a][i, 2], traj[c][i, 2]], [traj[a][i, 1], traj[c][i, 1]])
        tr.set_data(traj["scapula_r"][:i+1, 2], traj["scapula_r"][:i+1, 1])
        tl.set_data(traj["scapula_l"][:i+1, 2], traj["scapula_l"][:i+1, 1])
        return list(pts.values()) + lines + [tr, tl]

    anim = animation.FuncAnimation(fig, frame, frames=N, interval=33, blit=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer="ffmpeg", fps=30, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
