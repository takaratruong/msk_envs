"""Concrete confirmation: print actual left transform vs mirror-of-right, and the
joint chain, at a realistic 0.1 rad abduction. No assemble(), no forces."""
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

# joint chain: who parents the humerus / hand?
js = m.getJointSet()
print("Joint parent->child (arm-relevant):")
for i in range(js.getSize()):
    j = js.get(i)
    nm = j.getName()
    if any(k in nm for k in ("scapul", "gleno", "elbow", "radio", "wrist", "hand", "humer")):
        pf = j.getParentFrame().findBaseFrame().getName()
        cf = j.getChildFrame().findBaseFrame().getName()
        print(f"  {nm:20s} {pf:14s} -> {cf}")

def X(name):
    T = m.getBodySet().get(name).getTransformInGround(s)
    p = np.array([T.p().get(i) for i in range(3)])
    R = np.array([[T.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p

def setv(vr, vl):
    for c, v in zip(COORDS, vr): cs.get(f"scap_{c}_r").setValue(s, v, False)
    for c, v in zip(COORDS, vl): cs.get(f"scap_{c}_l").setValue(s, v, False)
    m.realizePosition(s)

# realistic mirrored command: +0.1 abduction both sides
setv([0.1, 0, 0], [0.1, 0, 0])
Rr, pr = X("scapula_r")
Rl, pl = X("scapula_l")
print("\n+0.1 abduction, mirrored (L=R=0.1):")
print("  scapula_r pos:", np.round(pr, 6))
print("  scapula_l pos:", np.round(pl, 6))
print("  M @ pr       :", np.round(M @ pr, 6), " (should equal scapula_l pos)")
print("  origin resid :", np.linalg.norm(pl - M @ pr))
print("  R_l:\n", np.round(Rl, 4))
print("  M R_r M (target mirror orientation):\n", np.round(M @ Rr @ M, 4))
print("  orientation Frobenius resid:", np.linalg.norm(Rl - M @ Rr @ M))
Rres = Rl.T @ (M @ Rr @ M)
ang = np.rad2deg(np.arccos(np.clip((np.trace(Rres) - 1) / 2, -1, 1)))
print(f"  orientation angle resid: {ang:.3f} deg")

for b in ("humerus", "hand"):
    _, prb = X(f"{b}_r"); _, plb = X(f"{b}_l")
    print(f"  {b}_l world pos:", np.round(plb, 5),
          " mirror-of-{b}_r:", np.round(M @ prb, 5),
          f" resid={np.linalg.norm(plb - M @ prb):.4f} m")
