"""Pure scapula-body mirror residual: pick material points FIXED in the scapula
frame (its COM, and a far corner ~ the ellipsoid semi-axes away) and compare the
left point's world position to the z=0 mirror of the right point's. This isolates
the scapula rigid body -- no downstream joints involved. If scapula ORIENTATION
were mirrored, every such point would mirror to 0; the origin already does."""
import numpy as np
import opensim as osim

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])
COORDS = ["abduction", "elevation", "uprot"]

osim.Logger.setLevelString("Off")
m = osim.Model(PATH)
fs = m.getForceSet()
for i in reversed(range(fs.getSize())):
    fs.remove(i)
s = m.initSystem()
cs = m.getCoordinateSet()

def setv(vr, vl):
    for c, v in zip(COORDS, vr): cs.get(f"scap_{c}_r").setValue(s, v, False)
    for c, v in zip(COORDS, vl): cs.get(f"scap_{c}_l").setValue(s, v, False)
    m.realizePosition(s)

def station_ground(body, off):
    b = m.getBodySet().get(body)
    v = osim.Vec3(float(off[0]), float(off[1]), float(off[2]))
    g = b.findStationLocationInGround(s, v)
    return np.array([g.get(0), g.get(1), g.get(2)])

# right-side local offsets. The MIRROR of a right station offset o=(x,y,z) is the
# SAME local offset on the left body IF the scapula frames mirror (because the
# child frame local axes reflect via M R M). But a material point is best compared
# directly in world: p_l(o_L) vs M p_r(o_R) with o_L = o_R (same body-local vec)
# only holds if the frame is mirrored. Instead compare the far ends measured as the
# scapula-frame axis tips, which are frame-attached; use identical local offsets and
# report worst over a set of offsets (covers the rigid body's extent).
OFFSETS = [
    (0.0, 0.0, 0.0),       # origin
    (0.05, 0.0, 0.0),
    (0.0, 0.10, 0.0),
    (0.0, 0.0, 0.05),
    (0.05, 0.10, 0.05),
]

def worst_scap_resid():
    # A right-local material point o_r mirrors to left-local offset M@o_r; a true
    # mirror requires  p_l(M o_r) == M @ p_r(o_r).
    w = 0.0
    for o in OFFSETS:
        pr = station_ground("scapula_r", o)
        o_l = (M @ np.array(o))
        pl = station_ground("scapula_l", o_l)
        w = max(w, np.linalg.norm(pl - M @ pr))
    return w

print(f"{'command':28s} {'scap-body worst pt resid (m)':>30s}")
for label, (vr, vl) in {
    "neutral":              ([0,0,0],[0,0,0]),
    "abd 0.10 (mirror)":    ([0.1,0,0],[0.1,0,0]),
    "abd 0.20 (mirror)":    ([0.2,0,0],[0.2,0,0]),
    "abd 0.26 ROM (mirror)":([0.2618,0,0],[0.2618,0,0]),
    "abd 0.40 (mirror)":    ([0.4,0,0],[0.4,0,0]),
    "elev 0.10 (mirror)":   ([0,0.1,0],[0,0.1,0]),
    "elev 0.26 ROM":        ([0,0.2618,0],[0,0.2618,0]),
    "all 0.10 (mirror)":    ([0.1,0.1,0.1],[0.1,0.1,0.1]),
    "all 0.26 ROM":         ([0.2618]*3,[0.2618]*3),
    "uprot 0.26 (mirror)":  ([0,0,0.2618],[0,0,0.2618]),
}.items():
    setv(vr, vl)
    print(f"{label:28s} {worst_scap_resid():30.6f}")
