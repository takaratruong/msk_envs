"""Smoke-test: load smoke_model.osim into Bolt, step it, and verify
(1) the EllipsoidJoint keeps the scapula on the ellipsoid surface by construction,
(2) the 3-segment lumbar ball chain integrates stably under gravity.

Bolt's ELLIPSOID mobilizer places the child at p = semi * n (n = body z-axis in
joint frame), so we recompute that from qpos and compare against the mobilizer
transform Bolt reports (mob_X_FM).
"""
import numpy as np
import torch
import warp as wp

import bolt

MODEL = "experiments/spine_shoulder/smoke_model.osim"
SETH_RADII = np.array([0.083, 0.20, 0.083])


def quat_xyz(qx, qy, qz):
    """Match bolt math.quat_from_xyz: intrinsic X,Y,Z euler -> quaternion (x,y,z,w)."""
    def axis_angle(ax, a):
        s, c = np.sin(a / 2), np.cos(a / 2)
        return np.array([ax[0] * s, ax[1] * s, ax[2] * s, c])

    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    q = qmul(axis_angle((1, 0, 0), qx), axis_angle((0, 1, 0), qy))
    return qmul(q, axis_angle((0, 0, 1), qz))


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)


def main():
    wp.init()
    res = bolt.load_model(
        model_path=MODEL,
        n_worlds=1,
        integrator=bolt.IntegratorType.EULER_ADAPTIVE,
        requires_visuals=False,
        muscle_fn_path=None,
        render_kinematic_tree=False,
    )
    m, d = res.model, res.data
    nq = bolt.get_num_qpos(m)
    print(f"LOADED: nq={nq}, qpos_id_lookup={res.qpos_id_lookup}")

    lookup = res.qpos_id_lookup
    scap_adr = [lookup[f"scap_q{c}"] for c in range(3)]

    # --- Test 1: FK places scapula on the ellipsoid (deterministic, pre-dynamics) ---
    qpos = wp.to_torch(d.qpos)
    test_angles = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.1, -0.25, 0.2), (0.5, 0.4, -0.3)]
    max_err = 0.0
    for angles in test_angles:
        for adr, val in zip(scap_adr, angles):
            qpos[0, adr] = val
        bolt.fk(m, d)
        # mob_X_FM holds each mobilizer's frame transform; find scapula's mobilizer id
        names = [n for n in res.body_name_ordering] if hasattr(res, "body_name_ordering") else None
        X_FM = wp.to_torch(d.mob_X_FM).cpu().numpy()[0]
        # Expected from Seth/Bolt formulation
        q = quat_xyz(*angles)
        n = rotate(q, np.array([0.0, 0.0, 1.0]))
        p_expected = SETH_RADII * n
        # locate scapula row: the one whose translation matches p_expected best
        errs = [np.linalg.norm(X_FM[i][:3] - p_expected) for i in range(X_FM.shape[0])]
        err = min(errs)
        max_err = max(max_err, err)
        print(f"  angles={angles} -> p_expected={np.round(p_expected, 4)} min_row_err={err:.2e}")
    assert max_err < 1e-5, f"ellipsoid FK mismatch: {max_err}"
    print(f"TEST1 PASS: scapula rides the ellipsoid by construction (max err {max_err:.2e})")

    # --- Test 2: dynamics stability, chain under gravity, 2 sim-seconds ---
    for adr in scap_adr:
        qpos[0, adr] = 0.1
    bolt.fk(m, d)
    dt = 0.01
    steps = 200
    for i in range(steps):
        bolt.increment_next_time(m, d, dt)
        bolt.step(m, d)
    q_final = wp.to_torch(d.qpos).cpu().numpy()[0]
    assert np.all(np.isfinite(q_final)), "NaN/inf in qpos after 2s"
    print(f"TEST2 PASS: 200 steps x 10ms stable, qpos finite. final scap q = "
          f"{[round(float(q_final[a]), 4) for a in scap_adr]}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
