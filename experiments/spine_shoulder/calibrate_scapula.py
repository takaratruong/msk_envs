"""Empirically determine Bolt's EllipsoidJoint convention so we can place the
scapula's neutral pose exactly at the original welded location.

Bolt FK gives, per mobilizer, mob_X_FM (the mobilizer transform F->M) and
mob_X_GB (body frame in ground). We build a probe model with a KNOWN parent
offset (identity) and read what Bolt actually produces at q=0. From that we solve
for the parent-offset transform X_PF that lands the scapula at target X0.
"""
import numpy as np
import warp as wp
import opensim as osim

import bolt

PROBE = "experiments/spine_shoulder/_probe_scap.osim"
SRC = "msk_envs/msk_models/sprinter/sprinter_model_sym.osim"
SETH_RADII = (0.083, 0.20, 0.083)


def quat_to_R(q):
    # bolt/warp transform stores quat as (x,y,z,w)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    wp.init()
    # Original scapula pose relative to torso (target).
    osim.Logger.setLevelString("Off")
    m0 = osim.Model(SRC); s0 = m0.initSystem()
    torso = m0.getBodySet().get("torso")
    for side in ("r", "l"):
        scap = m0.getBodySet().get(f"scapula_{side}")
        X0 = torso.findTransformBetween(s0, scap)
        p0 = np.array([X0.p().get(i) for i in range(3)])
        R0 = np.array([[X0.R().get(r, c) for c in range(3)] for r in range(3)])
        print(f"\nside {side}: target scapula-in-torso p={np.round(p0,4)}")
        print(f"  target R=\n{np.round(R0,3)}")

    # Load the already-built model and read what Bolt produces for the scapula joints.
    res = bolt.load_model(model_path="experiments/spine_shoulder/sprinter_spineshoulder.osim",
                          n_worlds=1, integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                          requires_visuals=False,
                          muscle_fn_path="experiments/spine_shoulder/sprinter_spineshoulder_fn.xml",
                          render_kinematic_tree=False)
    m, d = res.model, res.data
    bolt.fk(m, d)
    Xg = wp.to_torch(d.mob_X_GB).cpu().numpy()[0]
    Xfm = wp.to_torch(d.mob_X_FM).cpu().numpy()[0]
    bidx = res.body_id_lookup
    def scap_in_torso(Xg, bidx, side):
        si, ti = bidx[f"scapula_{side}"], bidx["torso"]
        pT, RT = Xg[ti][:3], quat_to_R(Xg[ti][3:])
        pS = Xg[si][:3]
        return RT.T @ (pS - pT)

    for side in ("r", "l"):
        p_rel = scap_in_torso(Xg, bidx, side)
        print(f"\nBolt NEW side {side}: scapula-in-torso p={np.round(p_rel,4)}  X_FM p={np.round(Xfm[bidx[f'scapula_{side}']][:3],4)}")

    # Definitive apples-to-apples: base model in Bolt, scapula-in-torso.
    resb = bolt.load_model(model_path=SRC, n_worlds=1,
                           integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                           requires_visuals=False,
                           muscle_fn_path="msk_envs/msk_models/sprinter/sprinter_model_fn.xml",
                           render_kinematic_tree=False)
    bolt.fk(resb.model, resb.data)
    Xgb = wp.to_torch(resb.data.mob_X_GB).cpu().numpy()[0]
    print("\n=== NEW vs BASE scapula-in-torso (Bolt, torso frame) ===")
    for side in ("r", "l"):
        pn = scap_in_torso(Xg, bidx, side)
        pb = scap_in_torso(Xgb, resb.body_id_lookup, side)
        print(f"  side {side}: new={np.round(pn,4)} base={np.round(pb,4)} |Δ|={np.linalg.norm(pn-pb):.4f}")


if __name__ == "__main__":
    main()
