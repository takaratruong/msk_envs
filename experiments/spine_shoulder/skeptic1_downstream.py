"""Physical consequence of the scapula orientation mirror error.

The glenohumeral joint center sits at a NONZERO offset inside the scapula body
frame (translation ~(-0.0095,-0.034,+/-0.0097) m). So the humerus origin in
ground = p_scap + R_scap @ t_gh. The fix made p_scap mirror exactly, but R_scap
does NOT mirror for abduction/elevation, so the arm attachment DOES NOT mirror.

Here I hold ALL arm coords at neutral and set only the scapula command
(identical L/R, as symmetry.py's pure swap dictates), then measure how far each
downstream body origin is from the z=0 reflection of its partner.
"""
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
BODIES = ["scapula", "humerus", "ulna", "radius", "hand"]


def load():
    osim.Logger.setLevelString("Off")
    m = osim.Model(PATH)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def pos(m, s, body):
    p = m.getBodySet().get(body).getTransformInGround(s).p()
    return np.array([p.get(i) for i in range(3)])


def set_scap(m, s, side, a, e, u):
    cs = m.getCoordinateSet()
    cs.get(f"scap_abduction_{side}").setValue(s, float(a), False)
    cs.get(f"scap_elevation_{side}").setValue(s, float(e), False)
    cs.get(f"scap_uprot_{side}").setValue(s, float(u), False)


def measure(m, s, a, e, u):
    # right side commanded, left neutral
    for sd in ("r", "l"):
        set_scap(m, s, sd, 0, 0, 0)
    set_scap(m, s, "r", a, e, u)
    m.realizePosition(s)
    right = {b: pos(m, s, f"{b}_r") for b in BODIES}
    # left side commanded (same values), right neutral
    for sd in ("r", "l"):
        set_scap(m, s, sd, 0, 0, 0)
    set_scap(m, s, "l", a, e, u)
    m.realizePosition(s)
    left = {b: pos(m, s, f"{b}_l") for b in BODIES}
    return {b: float(np.linalg.norm(left[b] - M @ right[b])) for b in BODIES}


def main():
    m, s = load()
    print(f"loaded {PATH}\narms held at NEUTRAL; only scapula commanded (identical L/R)")
    print("residual = ||p_left - M @ p_right||  (m)\n")
    print(f"{'a,e,u':>16} | " + " ".join(f"{b:>9}" for b in BODIES))
    for label, (a, e, u) in [
        ("neutral", (0, 0, 0)),
        ("abd 0.1", (0.1, 0, 0)),
        ("elev 0.1", (0, 0.1, 0)),
        ("uprot 0.1", (0, 0, 0.1)),
        ("all 0.1", (0.1, 0.1, 0.1)),
        ("abd 0.2", (0.2, 0, 0)),
        ("elev 0.2", (0, 0.2, 0)),
        ("all 0.2", (0.2, 0.2, 0.2)),
        ("abd 0.26(clamp)", (0.2618, 0, 0)),
        ("elev 0.26(clamp)", (0, 0.2618, 0)),
        ("mix -0.1,0.2,-0.1", (-0.1, 0.2, -0.1)),
    ]:
        r = measure(m, s, a, e, u)
        print(f"{label:>16} | " + " ".join(f"{r[b]:9.2e}" for b in BODIES))


if __name__ == "__main__":
    main()
