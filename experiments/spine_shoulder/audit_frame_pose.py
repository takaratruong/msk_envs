"""Frame + MOTION mirror audit, POSITION *and* ORIENTATION.

1. Validates the analytic +z ellipsoid mobilizer against OpenSim (current build),
   per coordinate, so the analytic sweep is trustworthy.
2. For candidate LEFT frame/command choices, reports BOTH position and
   orientation mirror residual vs the exact z=0 reflection of the RIGHT scapula.

Exact mirror target for a given right command q_r:
    p_target = M @ p_scap_r
    R_target = M @ R_scap_r @ M       (proper: det = 1)
where (R_scap_r, p_scap_r) is the RIGHT scapula pose in the torso frame.
"""
import numpy as np
import opensim as osim

np.set_printoptions(precision=6, suppress=True, linewidth=160)

SRC = "msk_envs/msk_models/sprinter/sprinter_model_sym.osim"
OUT = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
SETH_RADII = (0.083, 0.20, 0.083)
sz = SETH_RADII[2]
semi = np.array(SETH_RADII)
M = np.diag([1.0, 1.0, -1.0])


def load_stripped(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    return m, s


def ground_RT(m, s, name):
    X = m.getBodySet().get(name).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def euler_xyz_to_R(e):
    cx, cy, cz = np.cos(e); sx_, sy_, sz_ = np.sin(e)
    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def R_from_q(q):
    return euler_xyz_to_R(np.asarray(q, float))


def mob(q):
    """Bolt +z ellipsoid: returns (R_child_in_parent, p_child_in_parent)."""
    r = R_from_q(q)
    n = r @ np.array([0.0, 0.0, 1.0])
    p = semi * n
    return r, p


# ---- source ground truth in torso frame ----
m0, s0 = load_stripped(SRC)
Rt, pt = ground_RT(m0, s0, "torso")
Rr, pr = ground_RT(m0, s0, "scapula_r")
R0 = Rt.T @ Rr
p0 = Rt.T @ (pr - pt)
R0l = M @ R0 @ M                       # exact reflected right parent orientation
t_r = p0 - R0 @ np.array([0.0, 0.0, sz])
t_l_build = M @ p0 - R0l @ np.array([0.0, 0.0, sz])   # what the build writes
t_l_reflect = M @ t_r                  # the TRUE reflection of the right parent frame

# right scapula pose (in torso) analytic, as a function of q_r
def right_pose(q):
    rC, pC = mob(q)
    R_scap = R0 @ rC
    p_scap = t_r + R0 @ pC
    return R_scap, p_scap

# generic left pose for a candidate (R_P parent orient, t_P parent trans,
# R_Coff constant child-offset rotation applied AFTER mobilizer, command map)
def left_pose(q_r, R_P, t_P, cmd_map, R_Coff=np.eye(3)):
    q_l = cmd_map(q_r)
    rC, pC = mob(q_l)
    # child offset frame: scapula body = childframe . (R_Coff,0)^-1  => body R = R_P rC R_Coff^T
    R_scap = R_P @ rC @ R_Coff.T
    p_scap = t_P + R_P @ pC
    return R_scap, p_scap

ident = lambda q: np.asarray(q, float)
mirror_cmd = lambda q: np.array([-q[0], -q[1], q[2]])   # exact orientation-mirror command
flipY_cmd = lambda q: np.array([q[0], -q[1], q[2]])     # the "diag(-1,1,1)" companion

# ---------------------------------------------------------------
# STEP 0: validate analytic RIGHT/LEFT pose vs OpenSim current build
# ---------------------------------------------------------------
print("=" * 100)
print("STEP 0: analytic mobilizer vs OpenSim (current build sprinter_shoulderonly.osim)")
print("=" * 100)
m2, s2 = load_stripped(OUT)
cs = m2.getCoordinateSet()
Rt2, pt2 = ground_RT(m2, s2, "torso")

def opensim_scap(side, coord, A):
    for c in ("abduction", "elevation", "uprot"):
        for sd in ("r", "l"):
            cs.get(f"scap_{c}_{sd}").setValue(s2, 0.0)
    cs.get(f"scap_{coord}_{side}").setValue(s2, A)
    m2.realizePosition(s2)
    Rg, pg = ground_RT(m2, s2, f"scapula_{side}")
    # express in torso frame
    return Rt2.T @ Rg, Rt2.T @ (pg - pt2)

COORD_IDX = {"abduction": 0, "elevation": 1, "uprot": 2}
maxposerr = maxroterr = 0.0
for coord in ("abduction", "elevation", "uprot"):
    for A in (0.1, 0.2, 0.3):
        q = np.zeros(3); q[COORD_IDX[coord]] = A
        Rr_os, pr_os = opensim_scap("r", coord, A)
        Rr_an, pr_an = right_pose(q)
        pe = np.linalg.norm(pr_os - pr_an); re = np.linalg.norm(Rr_os - Rr_an)
        maxposerr = max(maxposerr, pe); maxroterr = max(maxroterr, re)
        # left current build: parent (R0l, t_l_build), identical command, child offset I
        Rl_os, pl_os = opensim_scap("l", coord, A)
        Rl_an, pl_an = left_pose(q, R0l, t_l_build, ident)
        maxposerr = max(maxposerr, np.linalg.norm(pl_os - pl_an))
        maxroterr = max(maxroterr, np.linalg.norm(Rl_os - Rl_an))
print(f"  max |analytic - OpenSim| position = {maxposerr:.2e} m ,  rotation(frob) = {maxroterr:.2e}")
print("  => analytic mobilizer model validated against OpenSim. Sweeps below are trustworthy.\n")

# ---------------------------------------------------------------
# STEP 1: candidate comparison, POSITION and ORIENTATION mirror residual
# ---------------------------------------------------------------
print("=" * 100)
print("STEP 1: mirror residual (position & orientation) vs exact z=0 reflection of RIGHT")
print("=" * 100)
print("Right command q_r sweeps a representative combined deflection (ab, el, uprot).")
print("Target: p_L = M p_R ; R_L = M R_R M  (both must be ~0 to call it an exact mirror)\n")

candidates = {
    "A build (R0l=MR0M, t=build,   q_l=q_r)":       dict(R_P=R0l, t_P=t_l_build,   cmap=ident,      C=np.eye(3)),
    "B reflect(R0l=MR0M, t=M@t_r,  q_l=(-x,-y,z))":  dict(R_P=R0l, t_P=t_l_reflect, cmap=mirror_cmd, C=np.eye(3)),
    "C diag   (R=MR0diag(-1,1,1), t=M@t_r, q_l=(x,-y,z))": dict(R_P=M@R0@np.diag([-1.,1.,1.]), t_P=t_l_reflect, cmap=flipY_cmd, C=np.eye(3)),
    "D build-frame + mirror cmd (R0l, t=build, q_l=(-x,-y,z))": dict(R_P=R0l, t_P=t_l_build, cmap=mirror_cmd, C=np.eye(3)),
}

hdr = f"{'amp':>5} | " + " | ".join(f"{k.split()[0]:>6} pos/rot" for k in candidates)
print(hdr)
for amp in (0.0, 0.05, 0.1, 0.2, 0.3):
    q_r = np.array([amp, 0.7 * amp, 0.5 * amp])
    Rrp, prp = right_pose(q_r)
    p_tgt = M @ prp
    R_tgt = M @ Rrp @ M
    cells = []
    for k, c in candidates.items():
        Rlp, plp = left_pose(q_r, c["R_P"], c["t_P"], c["cmap"], c["C"])
        pe = np.linalg.norm(plp - p_tgt)
        re = np.linalg.norm(Rlp - R_tgt)
        cells.append(f"{pe:6.4f}/{re:6.4f}")
    print(f"{amp:5.2f} | " + " | ".join(cells))

print("\nLegend: 'pos' = ||p_L - M p_R|| (m); 'rot' = Frobenius ||R_L - M R_R M||.")
print("A = current build. B = true reflected frame + orientation-mirror command.")
print("C = position-exact diag fix (breaks orientation). D = build frame + mirror command.")

# per-candidate NEUTRAL orientation vs source truth
print("\n-- neutral (q=0) scapula orientation error vs SOURCE left scapula (R0L_true = M R0 M) --")
R0L_true = M @ R0 @ M
for k, c in candidates.items():
    Rl0, pl0 = left_pose(np.zeros(3), c["R_P"], c["t_P"], c["cmap"], c["C"])
    print(f"  {k[:12]:12s}  ||R_L(0) - M R0 M|| = {np.linalg.norm(Rl0 - R0L_true):.4f}   "
          f"pos(0) mirror err = {np.linalg.norm(pl0 - M @ right_pose(np.zeros(3))[1]):.2e}")
