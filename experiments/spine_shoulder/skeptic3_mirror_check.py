"""Skeptic #3 independent L/R scapula mirror check.

Structure is intentionally different from the provided recipe:
  - Iterate over the ACTUAL coordinate objects (not hard-coded names).
  - Drive BOTH scapulae simultaneously with the SAME command vector (this is what
    symmetry.py's pure-swap mirror actually does), read both body transforms in ONE
    realize, and compare scapula_l against the z=0 reflection of scapula_r.
  - Compare not just origin position but the full rotation matrix (a mirror must
    reflect orientation too: R_l should equal M @ R_r @ M).
  - Randomized fuzz search over the coordinate cube to hunt for the worst case.

Mirror convention: reflection across z=0 is M = diag(1,1,-1).
  position:  p_l  ?=  M @ p_r
  rotation:  R_l  ?=  M @ R_r @ M   (improper-conjugated proper rotation)
"""
import sys
import numpy as np
import opensim as osim

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/spine_shoulder/sprinter_shoulderonly.osim"

M = np.diag([1.0, 1.0, -1.0])


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def body_RT(m, s, name):
    X = m.getBodySet().get(name).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def set_coord(m, s, name, val):
    m.getCoordinateSet().get(name).setValue(s, float(val), False)


SUFFIX = ["abduction", "elevation", "uprot"]


def apply_command(m, s, q_r, q_l):
    """Set right coords to q_r and left coords to q_l, then realize once."""
    for k, suf in enumerate(SUFFIX):
        set_coord(m, s, f"scap_{suf}_r", q_r[k])
        set_coord(m, s, f"scap_{suf}_l", q_l[k])
    m.assemble(s)
    m.realizePosition(s)


def residual(m, s, q_r, q_l):
    apply_command(m, s, q_r, q_l)
    Rr, pr = body_RT(m, s, "scapula_r")
    Rl, pl = body_RT(m, s, "scapula_l")
    pos_res = np.linalg.norm(pl - M @ pr)
    rot_res = np.linalg.norm(Rl - M @ Rr @ M)
    return pos_res, rot_res, pr, pl


def main():
    m, s = load_stripped(PATH)
    print(f"MODEL: {PATH}")
    print(f"nbodies={m.getNumBodies()} ncoords={m.getNumCoordinates()}")

    # sanity: neutral symmetry of the two scapulae as-built
    Rr0, pr0 = body_RT(m, s, "scapula_r")
    Rl0, pl0 = body_RT(m, s, "scapula_l")
    print(f"\nNEUTRAL (q=0): scapula_r={np.round(pr0,5)}  scapula_l={np.round(pl0,5)}")
    print(f"  neutral pos residual ||p_l - M p_r|| = {np.linalg.norm(pl0 - M@pr0):.3e} m")
    print(f"  neutral rot residual = {np.linalg.norm(Rl0 - M@Rr0@M):.3e}")

    print("\n=== SAME command on both sides (pure-swap mirror), per-coord ===")
    worst = (0.0, None)
    for amp in (0.1, 0.2, 0.4):
        for k, suf in enumerate(SUFFIX):
            q = [0.0, 0.0, 0.0]; q[k] = amp
            pos_res, rot_res, pr, pl = residual(m, s, q, q)
            print(f"  {suf:9s} +{amp:>4}: pos_res={pos_res:.3e} m  rot_res={rot_res:.3e}  "
                  f"pr_z={pr[2]:+.5f} pl_z={pl[2]:+.5f}")
            if pos_res > worst[0]:
                worst = (pos_res, ("percoord", suf, amp, list(q)))
            # also negative amplitude
            qn = [0.0, 0.0, 0.0]; qn[k] = -amp
            pos_res, rot_res, pr, pl = residual(m, s, qn, qn)
            if pos_res > worst[0]:
                worst = (pos_res, ("percoord-neg", suf, -amp, list(qn)))

    print("\n=== SAME command, all three coords together ===")
    for amp in (0.1, 0.2, 0.4):
        for signs in ([1,1,1],[1,-1,1],[-1,1,-1],[1,1,-1],[-1,-1,-1]):
            q = [amp*sg for sg in signs]
            pos_res, rot_res, pr, pl = residual(m, s, q, q)
            print(f"  q={np.round(q,3)}: pos_res={pos_res:.3e} m  rot_res={rot_res:.3e}")
            if pos_res > worst[0]:
                worst = (pos_res, ("combo", amp, list(signs), list(q)))

    print("\n=== Randomized fuzz over full clamp cube [-0.2618, 0.2618]^3 (2000 draws) ===")
    rng = np.random.default_rng(20260910)
    lim = 0.2617993877991494
    fuzz_worst = (0.0, None)
    for _ in range(2000):
        q = rng.uniform(-lim, lim, size=3).tolist()
        pos_res, rot_res, pr, pl = residual(m, s, q, q)
        if pos_res > fuzz_worst[0]:
            fuzz_worst = (pos_res, list(q), rot_res)
    print(f"  worst pos_res in cube = {fuzz_worst[0]:.3e} m at q={np.round(fuzz_worst[1],4)} "
          f"(rot_res={fuzz_worst[2]:.3e})")
    if fuzz_worst[0] > worst[0]:
        worst = (fuzz_worst[0], ("fuzz", fuzz_worst[1]))

    print("\n=== Extreme amplitudes beyond clamp (0.6, 1.0 rad) ===")
    for amp in (0.6, 1.0):
        q = [amp, amp, amp]
        pos_res, rot_res, pr, pl = residual(m, s, q, q)
        print(f"  q={amp}: pos_res={pos_res:.3e} m  rot_res={rot_res:.3e}")

    print(f"\n>>> WORST pos residual within clamp range: {worst[0]:.3e} m  ctx={worst[1]}")


if __name__ == "__main__":
    main()
