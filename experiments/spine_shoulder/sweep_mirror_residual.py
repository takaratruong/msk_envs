"""Per-coordinate scapula mirror-residual sweep (OpenSim, CPU only).

Sweeps EACH scapula coordinate (abduction, elevation, uprot) individually on the
right vs the left ellipsoid across amplitudes, and for each compares the LEFT
scapula world displacement to the z=0 mirror of the RIGHT scapula displacement.

Mirrored command convention (per confirmed bug report): SAME coordinate value on
both sides. A correct mirror => d_l == (d_r.x, d_r.y, -d_r.z), residual 0.
"""
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
COORDS = ["abduction", "elevation", "uprot"]
AMPS = [0.05, 0.10, 0.20, 0.30]


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def scap_pos(m, s, side):
    p = m.getBodySet().get(f"scapula_{side}").getTransformInGround(s).p()
    return np.array([p.get(0), p.get(1), p.get(2)])


def set_all_zero(m, s):
    cs = m.getCoordinateSet()
    for c in COORDS:
        for side in ("r", "l"):
            cs.get(f"scap_{c}_{side}").setValue(s, 0.0)


def main():
    m, s = load_stripped(PATH)
    cs = m.getCoordinateSet()

    set_all_zero(m, s)
    m.realizePosition(s)
    neutral_r = scap_pos(m, s, "r")
    neutral_l = scap_pos(m, s, "l")
    print(f"neutral scapula_r = {np.round(neutral_r,6)}")
    print(f"neutral scapula_l = {np.round(neutral_l,6)}")
    print(f"neutral z-mirror check |r.z + l.z| = {abs(neutral_r[2]+neutral_l[2]):.2e}, "
          f"|r.x-l.x|={abs(neutral_r[0]-neutral_l[0]):.2e}, |r.y-l.y|={abs(neutral_r[1]-neutral_l[1]):.2e}")
    print()

    hdr = (f"{'coord':10s} {'amp':>5s} | {'d_r (x,y,z)':>26s} | {'d_l (x,y,z)':>26s} | "
           f"{'|d_r|':>8s} {'|d_l|':>8s} {'|d_l|-|d_r|':>11s} | {'resid norm':>10s} | "
           f"{'resid (x,y,z)':>26s}")
    print(hdr)
    print("-" * len(hdr))

    for c in COORDS:
        for A in AMPS:
            # RIGHT only
            set_all_zero(m, s)
            cs.get(f"scap_{c}_r").setValue(s, A)
            m.realizePosition(s)
            d_r = scap_pos(m, s, "r") - neutral_r

            # LEFT only, SAME value (mirrored command convention)
            set_all_zero(m, s)
            cs.get(f"scap_{c}_l").setValue(s, A)
            m.realizePosition(s)
            d_l = scap_pos(m, s, "l") - neutral_l

            mir = np.array([d_r[0], d_r[1], -d_r[2]])
            resid = d_l - mir
            rn = np.linalg.norm(resid)
            nr, nl = np.linalg.norm(d_r), np.linalg.norm(d_l)
            print(f"{c:10s} {A:5.2f} | "
                  f"({d_r[0]:+.5f},{d_r[1]:+.5f},{d_r[2]:+.5f}) | "
                  f"({d_l[0]:+.5f},{d_l[1]:+.5f},{d_l[2]:+.5f}) | "
                  f"{nr:8.5f} {nl:8.5f} {nl-nr:+11.5f} | {rn:10.6f} | "
                  f"({resid[0]:+.5f},{resid[1]:+.5f},{resid[2]:+.5f})")
        print()

    # Also test the ALTERNATE convention: left = -A (negated command), per-coord,
    # to see if any single coord mirrors exactly under a sign flip.
    print("=== ALTERNATE: left command negated (scap_{c}_l = -A) ===")
    print(hdr)
    print("-" * len(hdr))
    for c in COORDS:
        for A in AMPS:
            set_all_zero(m, s)
            cs.get(f"scap_{c}_r").setValue(s, A)
            m.realizePosition(s)
            d_r = scap_pos(m, s, "r") - neutral_r

            set_all_zero(m, s)
            cs.get(f"scap_{c}_l").setValue(s, -A)
            m.realizePosition(s)
            d_l = scap_pos(m, s, "l") - neutral_l

            mir = np.array([d_r[0], d_r[1], -d_r[2]])
            resid = d_l - mir
            rn = np.linalg.norm(resid)
            nr, nl = np.linalg.norm(d_r), np.linalg.norm(d_l)
            print(f"{c:10s} {A:5.2f} | "
                  f"({d_r[0]:+.5f},{d_r[1]:+.5f},{d_r[2]:+.5f}) | "
                  f"({d_l[0]:+.5f},{d_l[1]:+.5f},{d_l[2]:+.5f}) | "
                  f"{nr:8.5f} {nl:8.5f} {nl-nr:+11.5f} | {rn:10.6f} | "
                  f"({resid[0]:+.5f},{resid[1]:+.5f},{resid[2]:+.5f})")
        print()


if __name__ == "__main__":
    main()
