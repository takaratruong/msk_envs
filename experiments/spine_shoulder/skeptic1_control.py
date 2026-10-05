"""Control: is the ~0.06 m probe residual intrinsic to the source model's
scapula frames, or introduced by the ellipsoid fix?

Compare, at NEUTRAL, R_l vs M@R_r@M and p_l vs M@p_r for:
  (a) the SOURCE welded model msk_envs/msk_models/sprinter/sprinter_model_sym.osim
  (b) the fixed shoulderonly model
"""
import numpy as np
import opensim as osim

M = np.diag([1.0, 1.0, -1.0])
PROBES = np.array([[0,0,0],[0.05,0,0],[0,0.04,0],[0,0,0.03]], float)


def transform_of(m, s, body):
    X = m.getBodySet().get(body).getTransformInGround(s)
    p = np.array([X.p().get(i) for i in range(3)])
    R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
    return R, p


def report(path):
    osim.Logger.setLevelString("Off")
    m = osim.Model(path)
    fs = m.getForceSet()
    for i in reversed(range(fs.getSize())):
        fs.remove(i)
    s = m.initSystem()
    Rr, pr = transform_of(m, s, "scapula_r")
    Rl, pl = transform_of(m, s, "scapula_l")
    R_ref = M @ Rr @ M
    p_ref = M @ pr
    dR = Rl - R_ref
    dp = pl - p_ref
    wl = (Rl @ PROBES.T).T + pl
    wr_ref = (M @ ((Rr @ PROBES.T).T + pr).T).T
    probe = np.linalg.norm(wl - wr_ref, axis=1)
    print(f"\n--- {path}")
    print(f"  origin residual |p_l - M p_r| = {np.linalg.norm(dp):.3e} m")
    print(f"  rotation residual ||R_l - M R_r M||_F = {np.linalg.norm(dR):.3e}")
    print(f"  probe residuals (origin,+x,+y,+z) = {np.round(probe,4)} m")
    print(f"  R_r=\n{np.round(Rr,4)}")
    print(f"  M R_r M (expected R_l)=\n{np.round(R_ref,4)}")
    print(f"  R_l (actual)=\n{np.round(Rl,4)}")


report("msk_envs/msk_models/sprinter/sprinter_model_sym.osim")
report("experiments/spine_shoulder/sprinter_shoulderonly.osim")
