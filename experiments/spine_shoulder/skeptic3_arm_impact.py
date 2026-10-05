"""Downstream impact: how far off-mirror is the whole arm when the scapula is driven
by the pure-swap mirror command (symmetry.py) at realistic ~0.1 rad ROM?

Drives ONLY the scapula coords with identical L/R values (arm coords at 0), then reads
the world origins of humerus/ulna/radius/hand and compares scapula_l-chain against the
z=0 reflection of the scapula_r-chain.  These are the points muscles attach to and
that the reward/observation see, so their residual is the real symmetry error.
"""
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
SUFFIX = ["abduction", "elevation", "uprot"]
CHAIN = ["scapula", "humerus", "ulna", "radius", "hand"]


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    return m, m.initSystem()


def pos(m, s, name):
    p = m.getBodySet().get(name).getPositionInGround(s)
    return np.array([p.get(i) for i in range(3)])


def apply(m, s, q):
    for k, suf in enumerate(SUFFIX):
        m.getCoordinateSet().get(f"scap_{suf}_r").setValue(s, float(q[k]), False)
        m.getCoordinateSet().get(f"scap_{suf}_l").setValue(s, float(q[k]), False)
    m.assemble(s)
    m.realizePosition(s)


def main():
    m, s = load_stripped(PATH)
    print(f"MODEL {PATH}")
    print("Pure-swap mirror command (identical L/R scap coords), arm coords = 0.")
    print("Residual = || pos_l - M @ pos_r ||  (meters), per arm body:\n")
    for amp in (0.1, 0.2):
        for axis, suf in enumerate(SUFFIX):
            q = [0.0, 0.0, 0.0]; q[axis] = amp
            apply(m, s, q)
            line = f"  q[{suf}]=+{amp}: "
            for b in CHAIN:
                r = np.linalg.norm(pos(m, s, b + "_l") - M @ pos(m, s, b + "_r"))
                line += f"{b}={r*1000:.2f}mm "
            print(line)
        print()
    # realistic combined abduction+elevation at 0.1
    apply(m, s, [0.1, 0.1, 0.0])
    print("  combined abd=elev=0.1 (realistic):")
    for b in CHAIN:
        r = np.linalg.norm(pos(m, s, b + "_l") - M @ pos(m, s, b + "_r"))
        print(f"    {b}: {r*1000:.2f} mm")


if __name__ == "__main__":
    main()
