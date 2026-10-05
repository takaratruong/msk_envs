"""Skeptic #1 independent mirror-symmetry check for the mobile scapula (v2).

CORRECT metric: the left scapula body frame must be the exact z=0 mirror of the
right under an IDENTICAL command (symmetry.py mirrors scap_*_r/_l by pure swap,
no negation). For a proper reflection M=diag(1,1,-1):
      p_l(q) == M @ p_r(q)
      R_l(q) == M @ R_r(q) @ M
I measure BOTH the origin position residual |p_l - M p_r| and the full-frame
residual as the max displacement of material points, where a probe at local v on
the LEFT is compared to the reflection of the SAME probe at local Mv on the
RIGHT (the geometrically correct mirror of a rigid body). I sweep single coords,
equal/asymmetric combos, a random box search, and amplitudes beyond the clamp.
"""
import itertools
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
COORDS = ("scap_abduction", "scap_elevation", "scap_uprot")
PROBES = np.array([[0,0,0],[0.05,0,0],[0,0.04,0],[0,0,0.03],[0.05,0.04,0.03]], float)


def load():
    osim.Logger.setLevelString("Off")
    m = osim.Model(PATH)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def transform_of(m, s, body):
    X = m.getBodySet().get(body).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def set_side(m, s, side, a, e, u):
    cs = m.getCoordinateSet()
    cs.get(f"scap_abduction_{side}").setValue(s, float(a), False)
    cs.get(f"scap_elevation_{side}").setValue(s, float(e), False)
    cs.get(f"scap_uprot_{side}").setValue(s, float(u), False)


def residual(m, s, a, e, u):
    for side in ("r", "l"):
        set_side(m, s, side, 0, 0, 0)
    set_side(m, s, "r", a, e, u)
    m.realizePosition(s)
    Rr, pr = transform_of(m, s, "scapula_r")
    for side in ("r", "l"):
        set_side(m, s, side, 0, 0, 0)
    set_side(m, s, "l", a, e, u)
    m.realizePosition(s)
    Rl, pl = transform_of(m, s, "scapula_l")

    # origin residual
    origin_res = float(np.linalg.norm(pl - M @ pr))
    # full-frame material-point residual: left@v  vs  M @ (right@(M v))
    left_pts = (Rl @ PROBES.T).T + pl
    right_mirror_pts = (M @ ((Rr @ (PROBES @ M).T).T + pr).T).T
    frame_res = float(np.linalg.norm(left_pts - right_mirror_pts, axis=1).max())
    # rotation residual
    rot_res = float(np.linalg.norm(Rl - M @ Rr @ M))
    return frame_res, origin_res, rot_res


def main():
    m, s = load()
    print(f"loaded {PATH}; nCoords={m.getNumCoordinates()} nBodies={m.getNumBodies()}")
    worst = 0.0
    worst_desc = ""

    def rec(tag, a, e, u):
        nonlocal worst, worst_desc
        fr, org, rot = residual(m, s, a, e, u)
        if fr > worst:
            worst, worst_desc = fr, f"{tag} (a={a:.3f},e={e:.3f},u={u:.3f})"
        return fr, org, rot

    print("\n== single-coord sweeps  (frame | origin | rot) ==")
    for amp in (0.0, 0.05, 0.1, 0.2, 0.4):
        line = f"amp={amp:+.2f}:\n"
        for ci, cname in enumerate(COORDS):
            v = [0.0, 0.0, 0.0]; v[ci] = amp
            fr, org, rot = rec(f"single-{cname}", *v)
            line += f"    {cname:16s} frame={fr:.3e} origin={org:.3e} rot={rot:.3e}\n"
        print(line, end="")

    print("== equal combo (a=e=u) ==")
    for amp in (0.05, 0.1, 0.2, 0.4):
        fr, org, rot = rec("combo-equal", amp, amp, amp)
        print(f"  amp={amp:+.2f}: frame={fr:.3e} origin={org:.3e} rot={rot:.3e}")

    print("== asymmetric +/- sign combos at 0.1 and 0.2 ==")
    for base in (0.1, 0.2):
        for sgn in itertools.product((-1, 1), repeat=3):
            rec("asym", base*sgn[0], base*sgn[1], base*sgn[2])
    print(f"  worst so far: {worst:.3e} at {worst_desc}")

    print("== random search box [-0.3,0.3]^3, 3000 draws ==")
    rng = np.random.default_rng(1)
    rmax, rarg = 0.0, None
    for _ in range(3000):
        q = rng.uniform(-0.3, 0.3, size=3)
        fr, _, _ = residual(m, s, *q)
        if fr > rmax:
            rmax, rarg = fr, q.copy()
    print(f"  random worst frame-residual = {rmax:.3e} at {np.round(rarg,3)}")
    if rmax > worst:
        worst, worst_desc = rmax, f"random {np.round(rarg,3)}"

    print("== extreme amplitude (beyond 15deg clamp) ==")
    for amp in (0.6, 0.8, 1.0, 1.2):
        fr, org, rot = rec("extreme", amp, amp, amp)
        print(f"  amp={amp:+.2f}: frame={fr:.3e} origin={org:.3e} rot={rot:.3e}")

    print(f"\nWORST frame-residual overall = {worst:.6e} m  @ {worst_desc}")
    r01 = max(residual(m, s, *v)[0]
              for v in ([0.1,0,0],[0,0.1,0],[0,0,0.1],[0.1,0.1,0.1],[-0.1,0.1,-0.1]))
    print(f"residual at realistic ~0.1 rad ROM = {r01:.6e} m")


if __name__ == "__main__":
    main()
