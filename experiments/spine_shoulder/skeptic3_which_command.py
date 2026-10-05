"""Which LEFT command actually mirrors a given RIGHT command (full rigid body)?

symmetry.py applies a PURE SWAP (no negation): left gets the SAME q as right.
Test whether that is the command that makes scapula_l the exact z=0 reflection of
scapula_r as a RIGID BODY (origin + orientation), or whether some sign pattern is
required.  Search the 8 sign patterns of (abduction, elevation, uprot) applied to the
left and report the full-body residual (origin + a 0.1 m off-origin material point).
"""
import itertools
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
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


def apply(m, s, q_r, q_l):
    for k, suf in enumerate(SUFFIX):
        m.getCoordinateSet().get(f"scap_{suf}_r").setValue(s, float(q_r[k]), False)
        m.getCoordinateSet().get(f"scap_{suf}_l").setValue(s, float(q_l[k]), False)
    m.assemble(s)
    m.realizePosition(s)


def full_residual(m, s, q_r, q_l, M_local):
    apply(m, s, q_r, q_l)
    Rr, pr = body_RT(m, s, "scapula_r")
    Rl, pl = body_RT(m, s, "scapula_l")
    origin = np.linalg.norm(pl - M @ pr)
    rot = np.linalg.norm(Rl - M @ Rr @ M)
    # off-origin material point (diag 0.08 m)
    a = np.array([0.08, 0.08, 0.08])
    w_r = pr + Rr @ a
    w_l = pl + Rl @ (M_local @ a)
    matp = np.linalg.norm(w_l - M @ w_r)
    return origin, rot, matp


def main():
    m, s = load_stripped(PATH)
    apply(m, s, [0, 0, 0], [0, 0, 0])
    Rr0, _ = body_RT(m, s, "scapula_r")
    Rl0, _ = body_RT(m, s, "scapula_l")
    M_local = Rr0.T @ M @ Rl0

    q_r = [0.15, 0.15, 0.15]  # a generic right command inside the clamp
    print(f"RIGHT command q_r = {q_r}")
    print("Search LEFT sign patterns; residuals in meters (origin, matpoint) and rot:")
    best = (1e9, None)
    for signs in itertools.product([1, -1], repeat=3):
        q_l = [q_r[k] * signs[k] for k in range(3)]
        origin, rot, matp = full_residual(m, s, q_r, q_l, M_local)
        tag = "  <-- symmetry.py PURE SWAP" if signs == (1, 1, 1) else ""
        print(f"  left signs {signs}: origin={origin:.3e}  matpoint={matp:.3e}  rot={rot:.3e}{tag}")
        if matp < best[0]:
            best = (matp, signs)
    print(f"\nBEST full-body mirror: left signs {best[1]} with matpoint residual {best[0]:.3e} m")
    print("(pure swap = (1,1,1); if best differs, symmetry.py's no-negate swap is wrong "
          "for the scapula ORIENTATION regardless of the radius fix)")

    # If the best pattern is not a clean 0, no sign pattern mirrors -> deeper issue.
    print("\n=== best-pattern residual vs amplitude ===")
    for amp in (0.05, 0.1, 0.2):
        q_r = [amp, amp, amp]
        q_l = [q_r[k] * best[1][k] for k in range(3)]
        o, r, mp = full_residual(m, s, q_r, q_l, M_local)
        # also pure swap for comparison
        o2, r2, mp2 = full_residual(m, s, q_r, q_r, M_local)
        print(f"  amp={amp}: best-signs matpoint={mp:.3e}  |  pure-swap matpoint={mp2:.3e}")


if __name__ == "__main__":
    main()
