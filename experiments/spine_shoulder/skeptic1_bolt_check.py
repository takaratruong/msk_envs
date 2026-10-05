"""Skeptic #1 -- confirm the mirror residual in the ACTUAL Bolt engine (not just
OpenSim), and confirm the model loads.

I read mob_X_GB (body world transform) directly from Bolt Data after
realize_position, for scapula_r/_l and hand_r/_l, under an identical L/R scapula
command (pure swap => same q on both sides). I compare left vs the z=0
reflection of right.
"""
import numpy as np
import warp as wp
import torch

import bolt

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])


def xform_to_Rp(xf):
    # warp transform: (px,py,pz, qx,qy,qz,qw)
    px, py, pz = xf[0], xf[1], xf[2]
    qx, qy, qz, qw = xf[3], xf[4], xf[5], xf[6]
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1 - 2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),   1 - 2*(qx*qx+qy*qy)],
    ])
    return R, np.array([px, py, pz])


def main():
    wp.clear_kernel_cache()
    lr = bolt.load_model(
        model_path=PATH, n_worlds=1, integrator=bolt.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=False, muscle_fn_path=None, render_kinematic_tree=False)
    m, d = lr.model, lr.data
    print(f"BOLT LOADED OK: nbody={m.nbody} nq={m.nq}")

    qid = lr.qpos_id_lookup
    bid = lr.body_id_lookup

    def set_scap(side, a, e, u):
        q = wp.to_torch(d.qpos)
        q[:, qid[f"scap_abduction_{side}"]] = a
        q[:, qid[f"scap_elevation_{side}"]] = e
        q[:, qid[f"scap_uprot_{side}"]] = u

    def zero_scap():
        for side in ("r", "l"):
            set_scap(side, 0.0, 0.0, 0.0)

    def body_Rp(name):
        xg = wp.to_torch(d.mob_X_GB).cpu().numpy()[0, bid[name]]
        return xform_to_Rp(xg)

    PROBES = np.array([[0,0,0],[0.05,0,0],[0,0.04,0],[0,0,0.03]], float)

    def residual(a, e, u):
        zero_scap(); set_scap("r", a, e, u)
        bolt.fk(m, d, run_reset=False)
        Rr, pr = body_Rp("scapula_r"); hr = body_Rp("hand_r")[1]
        zero_scap(); set_scap("l", a, e, u)
        bolt.fk(m, d, run_reset=False)
        Rl, pl = body_Rp("scapula_l"); hl = body_Rp("hand_l")[1]
        origin = float(np.linalg.norm(pl - M @ pr))
        rot = float(np.linalg.norm(Rl - M @ Rr @ M))
        left_pts = (Rl @ PROBES.T).T + pl
        right_mir = (M @ ((Rr @ (PROBES @ M).T).T + pr).T).T
        frame = float(np.linalg.norm(left_pts - right_mir, axis=1).max())
        hand = float(np.linalg.norm(hl - M @ hr))
        return origin, rot, frame, hand

    print(f"\n{'a,e,u':>16} | {'origin':>10} {'rot':>10} {'frame':>10} {'hand':>10}")
    for label, q in [
        ("neutral", (0, 0, 0)),
        ("abd 0.1", (0.1, 0, 0)),
        ("elev 0.1", (0, 0.1, 0)),
        ("uprot 0.1", (0, 0, 0.1)),
        ("all 0.1", (0.1, 0.1, 0.1)),
        ("abd 0.2", (0.2, 0, 0)),
        ("elev 0.2", (0, 0.2, 0)),
        ("abd 0.4", (0.4, 0, 0)),
    ]:
        o, r, f, h = residual(*q)
        print(f"{label:>16} | {o:10.3e} {r:10.3e} {f:10.3e} {h:10.3e}")


if __name__ == "__main__":
    main()
