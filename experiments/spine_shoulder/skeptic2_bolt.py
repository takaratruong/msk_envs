"""Bolt-side independent check: load sprinter_shoulderonly.osim, run FK, and
measure L/R scapula mirror for BOTH origin and orientation, plus downstream
humerus origin. Mirror plane z=0, M=diag(1,1,-1). Mirrored command = identical
L/R coord values (the symmetry module swaps scap_*_r/_l without negation)."""
import sys
import numpy as np
import warp as wp
sys.path.insert(0, "/home/ubuntu/msk_envs-stone-course")
import bolt

MODEL = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])


def quat_R(q):  # bolt stores xyzw
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    wp.init()
    res = bolt.load_model(model_path=MODEL, n_worlds=1,
                          integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                          requires_visuals=False, muscle_fn_path=None,
                          render_kinematic_tree=False)
    m, d = res.model, res.data
    B, L = res.body_id_lookup, res.qpos_id_lookup
    q = wp.to_torch(d.qpos)
    print(f"BOLT LOADED OK: nbody={m.nbody if hasattr(m,'nbody') else '?'} "
          f"nq={q.shape[1]}")

    def setq(vr, vl):
        for c, v in zip(("abduction", "elevation", "uprot"), vr):
            q[0, L[f"scap_{c}_r"]] = v
        for c, v in zip(("abduction", "elevation", "uprot"), vl):
            q[0, L[f"scap_{c}_l"]] = v
        bolt.fk(m, d)

    def resid(vr, vl):
        setq(vr, vl)
        X = wp.to_torch(d.mob_X_GB).detach().cpu().numpy()[0]
        pr, pl = X[B["scapula_r"]][:3], X[B["scapula_l"]][:3]
        Rr, Rl = quat_R(X[B["scapula_r"]][3:]), quat_R(X[B["scapula_l"]][3:])
        rot = Rl.T @ (M @ Rr @ M)
        ang = np.rad2deg(np.arccos(np.clip((np.trace(rot)-1)/2, -1, 1)))
        hr, hl = X[B["humerus_r"]][:3], X[B["humerus_l"]][:3]
        return (np.linalg.norm(pl - M @ pr),
                ang,
                np.linalg.norm(hl - M @ hr))

    print(f"\n{'command':26s} {'scap_pos(m)':>12s} {'scap_rot(deg)':>14s} {'hum_pos(m)':>12s}")
    cases = {
        "neutral":            ([0,0,0],[0,0,0]),
        "abd 0.10 mirror":    ([0.1,0,0],[0.1,0,0]),
        "abd 0.20 mirror":    ([0.2,0,0],[0.2,0,0]),
        "abd 0.40 mirror":    ([0.4,0,0],[0.4,0,0]),
        "elev 0.10 mirror":   ([0,0.1,0],[0,0.1,0]),
        "elev 0.40 mirror":   ([0,0.4,0],[0,0.4,0]),
        "uprot 0.40 mirror":  ([0,0,0.4],[0,0,0.4]),
        "all 0.10 mirror":    ([0.1,0.1,0.1],[0.1,0.1,0.1]),
        "all 0.20 mirror":    ([0.2,0.2,0.2],[0.2,0.2,0.2]),
    }
    for label, (vr, vl) in cases.items():
        sp, sr_deg, hp = resid(vr, vl)
        print(f"{label:26s} {sp:12.3e} {sr_deg:14.4f} {hp:12.3e}")


if __name__ == "__main__":
    main()
