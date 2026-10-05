"""Independent (skeptic #2) L/R scapula mirror check for sprinter_shoulderonly.osim.

Structured differently from sweep_mirror_residual.py:

  * Mirror is defined as full-rigid-transform reflection across ground z=0:
    for a mirrored command (SAME coord value both sides, since the symmetry
    module swaps scap_*_r/_l WITHOUT negating), a true mirror requires
        p_l = M p_r        (origin position)  with M=diag(1,1,-1)
        R_l = M R_r M      (orientation)  -> makes EVERY material point mirror
  * We therefore probe not just the scapula ORIGIN (what the old script did) but
    also (a) the full orientation residual, and (b) DOWNSTREAM bodies offset from
    the scapula origin (humerus, hand) whose world position AMPLIFIES any
    orientation-mirror error into a measurable position error. A pure origin test
    can pass while the body is mis-oriented; the humerus test cannot.
  * We sweep single coords, combined coords, and deliberately try to break it.
"""
import itertools
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
COORDS = ["abduction", "elevation", "uprot"]


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def body_X(m, s, name):
    X = m.getBodySet().get(name).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def set_scap(m, s, vals_r, vals_l):
    cs = m.getCoordinateSet()
    for c, v in zip(COORDS, vals_r):
        cs.get(f"scap_{c}_r").setValue(s, v, False)
    for c, v in zip(COORDS, vals_l):
        cs.get(f"scap_{c}_l").setValue(s, v, False)
    m.assemble(s)
    m.realizePosition(s)


def mirror_residuals(m, s):
    """Return dict of residual norms for scapula origin, scapula orientation,
    and downstream humerus/hand origins (all in ground frame)."""
    out = {}
    Rr, pr = body_X(m, s, "scapula_r")
    Rl, pl = body_X(m, s, "scapula_l")
    out["scap_pos"] = np.linalg.norm(pl - M @ pr)
    # orientation: R_l should equal M R_r M
    out["scap_rot_fro"] = np.linalg.norm(Rl - M @ Rr @ M)
    # angle of the residual rotation R_l^T (M R_r M)
    Rres = Rl.T @ (M @ Rr @ M)
    ang = np.arccos(np.clip((np.trace(Rres) - 1.0) / 2.0, -1.0, 1.0))
    out["scap_rot_deg"] = np.rad2deg(ang)
    for b in ("humerus", "hand"):
        try:
            _, pr_b = body_X(m, s, f"{b}_r")
            _, pl_b = body_X(m, s, f"{b}_l")
            out[f"{b}_pos"] = np.linalg.norm(pl_b - M @ pr_b)
        except Exception:
            pass
    return out


def main():
    m, s = load_stripped(PATH)
    print(f"model bodies={m.getNumBodies()} coords={m.getNumCoordinates()}")
    bs = m.getBodySet()
    have = {bs.get(i).getName() for i in range(bs.getSize())}
    print("has humerus_l:", "humerus_l" in have, " hand_l:", "hand_l" in have)

    # ---- baseline: neutral (pre-existing model asymmetry floor) ----
    set_scap(m, s, [0, 0, 0], [0, 0, 0])
    base = mirror_residuals(m, s)
    print("\nNEUTRAL residuals (model's intrinsic asymmetry floor):")
    for k, v in base.items():
        print(f"   {k:14s} = {v:.3e}")

    def report(tag, r):
        line = " ".join(f"{k}={r[k]:.2e}" for k in r)
        print(f"  {tag:38s} {line}")

    worst = {"scap_pos": (0, ""), "humerus_pos": (0, ""), "scap_rot_deg": (0, "")}

    def track(tag, r):
        for key in worst:
            if key in r and r[key] > worst[key][0]:
                worst[key] = (r[key], tag)

    # ---- 1) single-coord mirrored command, amplitudes incl. beyond ROM ----
    print("\n[1] single coord, mirrored (same value L=R):")
    for c in COORDS:
        for A in (0.1, 0.2, 0.4):
            vr = [A if cc == c else 0.0 for cc in COORDS]
            set_scap(m, s, vr, vr)  # mirrored command = identical values
            r = mirror_residuals(m, s)
            track(f"{c}={A}", r)
            report(f"{c}={A:+.2f}", r)

    # ---- 2) combined 3-coord mirrored command ----
    print("\n[2] all three coords together, mirrored:")
    for A in (0.1, 0.2, 0.26):
        vr = [A, A, A]
        set_scap(m, s, vr, vr)
        r = mirror_residuals(m, s)
        track(f"all={A}", r)
        report(f"abd=elev=uprot={A:+.2f}", r)

    # ---- 3) mixed-sign combos at realistic 0.1 rad (try to find worst) ----
    print("\n[3] mixed-sign combos @0.1 rad, mirrored:")
    for signs in itertools.product((+1, -1), repeat=3):
        vr = [0.1 * sg for sg in signs]
        set_scap(m, s, vr, vr)
        r = mirror_residuals(m, s)
        tag = "".join("+" if sg > 0 else "-" for sg in signs)
        track(f"signs{tag}", r)
        report(f"signs {tag} @0.1", r)

    # ---- 4) the axis that drives orientation error hardest: uprot only ----
    #      (uprot rotates about the surface normal -> pure orientation, ~0 origin move)
    print("\n[4] uprot-dominant, mirrored (origin ~fixed, orientation swings):")
    for A in (0.1, 0.2, 0.26):
        vr = [0.0, 0.0, A]
        set_scap(m, s, vr, vr)
        r = mirror_residuals(m, s)
        track(f"uprot={A}", r)
        report(f"uprot={A:+.2f}", r)

    print("\n=== WORST OBSERVED (mirrored commands) ===")
    for k, (v, tag) in worst.items():
        print(f"  {k:14s} max = {v:.3e}   at {tag}")


if __name__ == "__main__":
    main()
