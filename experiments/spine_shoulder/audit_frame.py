"""Frame-level audit of the LEFT ellipsoid parent frame vs exact z=0 reflection.

No GPU. OpenSim + numpy only. Reproduces build math, compares against source
ground truth, and against the WRITTEN frames in the produced model. Then drives
the mobilizer analytically (replicating bolt ELLIPSOID) to show the motion
residual and test candidate fixes.
"""
import numpy as np
import opensim as osim

np.set_printoptions(precision=6, suppress=True, linewidth=140)

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


def frame_ground_RT(frame, s):
    X = frame.getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def euler_xyz_to_R(e):
    cx, cy, cz = np.cos(e); sx_, sy_, sz_ = np.sin(e)
    Rx = np.array([[1,0,0],[0,cx,-sx_],[0,sx_,cx]])
    Ry = np.array([[cy,0,sy_],[0,1,0],[-sy_,0,cy]])
    Rz = np.array([[cz,-sz_,0],[sz_,cz,0],[0,0,1]])
    return Rx @ Ry @ Rz


def R_to_bodyfixed_xyz(R):
    rot = osim.Rotation()
    for r in range(3):
        for c in range(3):
            rot.set(r, c, float(R[r, c]))
    e = rot.convertRotationToBodyFixedXYZ()
    return np.array([e.get(0), e.get(1), e.get(2)])


def quat_from_xyz(qx, qy, qz):
    # body-fixed XYZ -> rotation matrix (matches bolt math.quat_from_xyz composition)
    return euler_xyz_to_R(np.array([qx, qy, qz]))


def mob_n_p(q):
    r = quat_from_xyz(*q)
    n = r @ np.array([0.0, 0.0, 1.0])
    p = semi * n
    return n, p, r


print("=" * 90)
print("STEP 1: SOURCE GROUND TRUTH (sprinter_model_sym.osim welded scapulae)")
print("=" * 90)
m0, s0 = load_stripped(SRC)
Rt, pt = ground_RT(m0, s0, "torso")
Rr, pr = ground_RT(m0, s0, "scapula_r")
Rl, pl = ground_RT(m0, s0, "scapula_l")
# right & left scapula expressed in torso frame
R0 = Rt.T @ Rr
p0 = Rt.T @ (pr - pt)
R0L_true = Rt.T @ Rl
p0L_true = Rt.T @ (pl - pt)
print("R0 (torso->scapula_r):\n", R0)
print("p0 (torso->scapula_r):", p0)
print("R0L_true (torso->scapula_l):\n", R0L_true)
print("p0L_true (torso->scapula_l):", p0L_true)

print("\n-- Is the SOURCE left scapula an exact z=0 reflection of right, in torso frame? --")
print("   M p0 (target left pos)      :", M @ p0)
print("   p0L_true                    :", p0L_true)
print("   translation mirror residual :", np.linalg.norm(M @ p0 - p0L_true))
print("   M R0 M (pure reflect of R0) :\n", M @ R0 @ M)
print("   R0L_true                    :\n", R0L_true)
print("   ||M R0 M - R0L_true||       :", np.linalg.norm(M @ R0 @ M - R0L_true))
# what fixed axis-flip relates them?  R0L_true = M R0 F  => F = R0^T M R0L_true
F_true = R0.T @ M @ R0L_true
print("   F = R0^T M R0L_true (should be a signed diag if related by 1 axis flip):\n", F_true)


print("\n" + "=" * 90)
print("STEP 2: REPRODUCE BUILD MATH for LEFT parent frame")
print("=" * 90)
t_r = p0 - R0 @ np.array([0.0, 0.0, sz])
R0l = M @ R0 @ M
t_l = (M @ p0 - R0l @ np.array([0.0, 0.0, sz]))
o_r = R_to_bodyfixed_xyz(R0)
o_l = R_to_bodyfixed_xyz(R0l)
print("t_r (build right parent trans):", t_r)
print("t_l (build left  parent trans):", t_l)
print("R0l = M R0 M:\n", R0l)
print("euler o_r:", o_r, " round-trip err:", np.linalg.norm(euler_xyz_to_R(o_r) - R0))
print("euler o_l:", o_l, " round-trip err:", np.linalg.norm(euler_xyz_to_R(o_l) - R0l))

print("\n-- FRAME-LEVEL exact-reflection test of the LEFT parent frame vs RIGHT parent frame --")
print("   The exact z=0 reflection of right parent frame (R_Fr,t_r) is (M R_Fr M, M t_r).")
print("   R_Fr = R0, so reflected rotation = M R0 M = R0l  (build MATCHES this).")
print("   reflected translation = M t_r :", M @ t_r)
print("   build t_l                     :", t_l)
print("   TRANSLATION frame residual    :", np.linalg.norm(t_l - M @ t_r), "m")
print("   (t_l - M t_r) decomposed      :", t_l - M @ t_r)
print("   predicted M R0 (0,0,2 sz)     :", M @ R0 @ np.array([0.0, 0.0, 2 * sz]))


print("\n" + "=" * 90)
print("STEP 3: WRITTEN FRAMES in produced model (sprinter_shoulderonly.osim)")
print("=" * 90)
try:
    m2, s2 = load_stripped(OUT)
    Rt2, pt2 = ground_RT(m2, s2, "torso")
    jr = m2.getJointSet().get("scapulothoracic_r")
    jl = m2.getJointSet().get("scapulothoracic_l")
    pfr = osim.PhysicalOffsetFrame.safeDownCast(jr.getParentFrame())
    pfl = osim.PhysicalOffsetFrame.safeDownCast(jl.getParentFrame())
    # express these offset frames in torso frame
    def parent_in_torso(pf):
        Rg, pg = frame_ground_RT(pf, s2)
        return Rt2.T @ Rg, Rt2.T @ (pg - pt2)
    Rfr, tfr = parent_in_torso(pfr)
    Rfl, tfl = parent_in_torso(pfl)
    print("WRITTEN right parent-in-torso: R=\n", Rfr, "\n t=", tfr)
    print("WRITTEN left  parent-in-torso: R=\n", Rfl, "\n t=", tfl)
    print("\n-- 4x4 exact z=0 reflection residual (left vs M @ right @ M) --")
    print("   rotation residual ||Rfl - M Rfr M|| :", np.linalg.norm(Rfl - M @ Rfr @ M))
    print("   translation residual ||tfl - M tfr||:", np.linalg.norm(tfl - M @ tfr), "m")
    # neutral scapula mirror
    sr0, _ = ground_RT(m2, s2, "scapula_r"); pr0 = ground_RT(m2, s2, "scapula_r")[1]
    pl0 = ground_RT(m2, s2, "scapula_l")[1]
    pr0_t = Rt2.T @ (pr0 - pt2); pl0_t = Rt2.T @ (pl0 - pt2)
    print("\n   neutral scapula_r in torso:", pr0_t)
    print("   neutral scapula_l in torso:", pl0_t)
    print("   neutral MIRROR residual   :", np.linalg.norm(M @ pr0_t - pl0_t), "m (expect ~0)")
except Exception as ex:
    print("could not load produced model:", ex)


print("\n" + "=" * 90)
print("STEP 4: MOTION residual, analytic mobilizer (build frames), several commands")
print("=" * 90)
# analytic scapula-in-torso position for a coord vector q on a side
def scap_in_torso(R_Fr, t_frame, q):
    n, p, r = mob_n_p(q)
    return R_Fr @ p + t_frame

print("Using build frames: right (R0, t_r), left (R0l, t_l).")
print(f"{'amp':>6} {'identical q':>26} {'qy_l=-qy_r':>26} {'F=diag(-1,1,1)+qymap':>26}")
for amp in (0.0, 0.05, 0.1, 0.2, 0.3):
    qr = np.array([amp, amp * 0.7, amp * 0.5])  # abduction, elevation, uprot
    # convention A: identical command (fk script uses same-sign ab/el)
    ql_ident = qr.copy()
    posr = scap_in_torso(R0, t_r, qr)
    posl_ident = scap_in_torso(R0l, t_l, ql_ident)
    res_ident = np.linalg.norm(M @ posr - posl_ident)
    # convention B: flip qy (elevation) on left, build frames
    ql_flipy = np.array([qr[0], -qr[1], qr[2]])
    posl_flipy = scap_in_torso(R0l, t_l, ql_flipy)
    res_flipy = np.linalg.norm(M @ posr - posl_flipy)
    # candidate FIX: R_Fl = M R0 diag(-1,1,1), t_l = M t_r, command qy_l=-qy_r, qx_l=qx_r
    Ffix = np.diag([-1.0, 1.0, 1.0])
    R0l_fix = M @ R0 @ Ffix
    t_l_fix = M @ t_r
    ql_fix = np.array([qr[0], -qr[1], qr[2]])
    posl_fix = scap_in_torso(R0l_fix, t_l_fix, ql_fix)
    res_fix = np.linalg.norm(M @ posr - posl_fix)
    print(f"{amp:6.2f} {res_ident:26.6f} {res_flipy:26.6f} {res_fix:26.6f}")

print("\nNote: 'identical q' column reproduces the reported ~0.003-0.010 m growing residual.")
print("      Fix column should be ~0 for all amplitudes if the corrected frame+command mirror.")

# Also report the neutral position under the FIX frame vs build, to check we didn't break neutral
print("\n-- FIX neutral check --")
Ffix = np.diag([-1.0, 1.0, 1.0]); R0l_fix = M @ R0 @ Ffix; t_l_fix = M @ t_r
posl_fix0 = scap_in_torso(R0l_fix, t_l_fix, np.zeros(3))
posr0 = scap_in_torso(R0, t_r, np.zeros(3))
print("   right neutral in torso:", posr0)
print("   FIX left neutral      :", posl_fix0)
print("   FIX neutral mirror res:", np.linalg.norm(M @ posr0 - posl_fix0), "m")
o_l_fix = R_to_bodyfixed_xyz(R0l_fix)
print("   FIX left euler o_l    :", o_l_fix, " round-trip:", np.linalg.norm(euler_xyz_to_R(o_l_fix) - R0l_fix))

# orientation mirror under fix: compare scapula ROT in torso vs actual source left scapula rot
print("\n-- FIX orientation vs SOURCE left scapula (torso frame), at neutral --")
Rscap_l_fix0 = R0l_fix @ quat_from_xyz(0, 0, 0)
print("   FIX left scap R (neutral):\n", Rscap_l_fix0)
print("   SOURCE R0L_true          :\n", R0L_true)
print("   ||fix - source||         :", np.linalg.norm(Rscap_l_fix0 - R0L_true))
print("   build R0l (neutral)      :\n", R0l)
print("   ||build R0l - source||   :", np.linalg.norm(R0l - R0L_true))
