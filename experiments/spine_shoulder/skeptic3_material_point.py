"""Skeptic #3 decisive test: does the WHOLE scapula rigid body mirror, or only its
origin?  A z=0 mirror is only valid for RL data augmentation if EVERY material point
(muscle insertions, the glenohumeral joint, the whole arm) reflects, not just the
body origin.

Method (convention-light):
  1. From this model at NEUTRAL, measure the local mirror map between the two scapula
     frames.  We do NOT assume it; we derive M_local = R_r0^T @ M @ R_l0 and verify it
     is exactly diag(1,1,-1) (this is what "the source scapulae are perfect mirrors"
     means at neutral).
  2. Under an IDENTICAL command q on both sides (symmetry.py pure swap, no negate),
     pick material points a on the right scapula and their mirror partners M_local@a on
     the left.  A true mirror requires  world_l(M_local a) == M @ world_r(a).
  3. Report the residual in METERS for realistic offsets (muscle insertions live
     ~0.05-0.15 m from the scapula origin).

Also drives the glenohumeral child (humerus origin) as a concrete downstream point.
"""
import sys
import numpy as np
import opensim as osim

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
SUFFIX = ["abduction", "elevation", "uprot"]


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    return m, m.initSystem()


def body_RT(m, s, name):
    X = m.getBodySet().get(name).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def apply(m, s, q):
    for k, suf in enumerate(SUFFIX):
        m.getCoordinateSet().get(f"scap_{suf}_r").setValue(s, float(q[k]), False)
        m.getCoordinateSet().get(f"scap_{suf}_l").setValue(s, float(q[k]), False)
    m.assemble(s)
    m.realizePosition(s)


def main():
    m, s = load_stripped(PATH)
    print(f"MODEL: {PATH}  nbodies={m.getNumBodies()} ncoords={m.getNumCoordinates()}")

    # --- 1. neutral local mirror map ---
    apply(m, s, [0, 0, 0])
    Rr0, pr0 = body_RT(m, s, "scapula_r")
    Rl0, pl0 = body_RT(m, s, "scapula_l")
    M_local = Rr0.T @ M @ Rl0
    print("neutral local mirror map M_local = R_r0^T M R_l0 (expect diag(1,1,-1)):")
    print(np.round(M_local, 6))
    print(f"  ||M_local - diag(1,1,-1)|| = {np.linalg.norm(M_local - M):.3e}")

    # material offsets on the right scapula (body-local, meters). Include a realistic
    # muscle-insertion-scale set and the axis unit vectors * 0.1 m.
    offsets = {
        "+x0.10": np.array([0.10, 0.0, 0.0]),
        "+y0.10": np.array([0.0, 0.10, 0.0]),
        "+z0.10": np.array([0.0, 0.0, 0.10]),
        "diag0.08": np.array([0.08, 0.08, 0.08]),
    }

    print("\n=== material-point mirror residual (meters) under identical L/R command ===")
    print("(origin residual is the pure position check; offset residuals fold in orientation)")
    worst_origin = 0.0
    worst_offset = 0.0
    worst_offset_ctx = None
    for amp in (0.1, 0.2, 0.4):
        for k, suf in enumerate(SUFFIX):
            q = [0.0, 0.0, 0.0]; q[k] = amp
            apply(m, s, q)
            Rr, pr = body_RT(m, s, "scapula_r")
            Rl, pl = body_RT(m, s, "scapula_l")
            orig_res = np.linalg.norm(pl - M @ pr)
            worst_origin = max(worst_origin, orig_res)
            line = f"  {suf:9s}+{amp:<4} origin={orig_res:.2e}"
            for oname, a in offsets.items():
                w_r = pr + Rr @ a               # right material point in world
                w_l = pl + Rl @ (M_local @ a)   # its mirror partner on the left
                res = np.linalg.norm(w_l - M @ w_r)
                line += f"  {oname}={res:.2e}"
                if res > worst_offset:
                    worst_offset = res
                    worst_offset_ctx = (suf, amp, oname)
            print(line)

    print(f"\nWORST origin (position) residual : {worst_origin:.3e} m")
    print(f"WORST offset (rigid-body) residual: {worst_offset:.3e} m  at {worst_offset_ctx}")

    # --- 3. concrete downstream: humerus origin under mirrored arm command ---
    # glenohumeral coords also swap L<->R; flexion/rotation negate under z-mirror in
    # this model's convention, but to isolate the SCAPULA we hold arm coords at 0 and
    # just read where the humerus origin lands (it rides on the scapula).
    print("\n=== humerus origin (rides on scapula), arm coords = 0 ===")
    for amp in (0.1, 0.2, 0.4):
        q = [amp, amp, 0.0]  # abduction+elevation, the failing axes
        apply(m, s, q)
        try:
            _, ph_r = body_RT(m, s, "humerus_r")
            _, ph_l = body_RT(m, s, "humerus_l")
            res = np.linalg.norm(ph_l - M @ ph_r)
            print(f"  abd=elev={amp}: humerus origin mirror residual = {res:.3e} m")
        except Exception as e:
            print(f"  humerus read failed: {e}")


if __name__ == "__main__":
    main()
