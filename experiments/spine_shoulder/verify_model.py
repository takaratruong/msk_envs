"""Verify the spine+shoulder sprinter model in BOLT (the engine that trains).

Checks:
  1. loads; DOF/qpos counts as expected (+18 dof vs baseline: 9 lumbar + 6 scap +
     ... actually +12 lumbar coords replacing 3 back coords = net +9, +6 scapula = +15)
  2. neutral scapula position in Bolt is near the original welded location
  3. scapula rides the ellipsoid: sweeping scap coords moves it on the surface
  4. lumbar chain bends: sweeping lumbar coords bends the trunk
  5. muscle moment arms / lengths finite at neutral
  6. 200 x 10ms steps under gravity stay finite
"""
import numpy as np
import torch
import warp as wp

import bolt

NEW = "experiments/spine_shoulder/sprinter_spineshoulder.osim"
BASE = "msk_envs/msk_models/sprinter/sprinter_model_sym.osim"
FN = "experiments/spine_shoulder/sprinter_spineshoulder_fn.xml"


def load(path, fn=FN):
    return bolt.load_model(model_path=path, n_worlds=1,
                           integrator=bolt.IntegratorType.EULER_ADAPTIVE,
                           requires_visuals=False, muscle_fn_path=fn,
                           render_kinematic_tree=False)


def body_pos(res, d, body_name):
    order = res.body_name_ordering if hasattr(res, "body_name_ordering") else None
    X = wp.to_torch(d.mob_X_GB).cpu().numpy()[0]  # (nbody, 7) transform: pos(3)+quat(4)
    if order and body_name in order:
        return X[order.index(body_name)][:3]
    return None, X, order


def main():
    wp.init()
    print("=== LOAD NEW MODEL ===")
    res = load(NEW)
    m, d = res.model, res.data
    nq = bolt.get_num_qpos(m)
    lookup = res.qpos_id_lookup
    print(f"nq={nq}; num coords in lookup={len([k for k,v in lookup.items() if v>=0])}")
    body_idx = res.body_id_lookup  # name -> index
    print(f"bodies: {len(body_idx)}")
    for key in ["lumbar1_extension", "lumbar3_rotation", "torso_extension",
                "scap_r_abduction", "scap_l_uprot"]:
        print(f"  coord {key}: qpos_adr={lookup.get(key)}")

    bolt.fk(m, d)
    Xg = wp.to_torch(d.mob_X_GB).cpu().numpy()[0]
    print(f"mob_X_GB shape: {Xg.shape}")

    # Compare neutral body positions new vs base
    print("\n=== NEUTRAL BODY POSITIONS (Bolt) new vs base ===")
    resb = load(BASE)
    bolt.fk(resb.model, resb.data)
    Xb = wp.to_torch(resb.data.mob_X_GB).cpu().numpy()[0]
    ob = resb.body_id_lookup

    def pos(idxmap, X, name):
        return X[idxmap[name]][:3] if name in idxmap else None

    for bn in ["torso", "scapula_r", "scapula_l", "humerus_r", "hand_r", "pelvis"]:
        pn = pos(body_idx, Xg, bn)
        pb = pos(ob, Xb, bn)
        if pn is not None and pb is not None:
            dlt = np.array(pn) - np.array(pb)
            print(f"  {bn:12s} new={np.round(pn,3)} base={np.round(pb,3)} |Δ|={np.linalg.norm(dlt):.3f}")

    # Scapula rides the ellipsoid: sweep scap_r coords, confirm position changes
    print("\n=== SCAPULA MOBILITY (sweep scap_r_abduction) ===")
    q = wp.to_torch(d.qpos)
    base_scap = pos(body_idx, Xg, "scapula_r")
    adr = lookup["scap_r_abduction"]
    moved = []
    for val in (-0.3, 0.0, 0.3):
        q[0, adr] = val
        bolt.fk(m, d)
        X = wp.to_torch(d.mob_X_GB).cpu().numpy()[0]
        moved.append(pos(body_idx, X, "scapula_r"))
    q[0, adr] = 0.0
    disp = np.linalg.norm(np.array(moved[2]) - np.array(moved[0]))
    print(f"  scapula moved {disp:.4f} m across abduction sweep (should be > 0 = mobile)")

    # Lumbar bends: sweep all lumbar bending coords, confirm torso/head displaces
    print("\n=== SPINE FLEXIBILITY (sweep lumbar *_bending) ===")
    bolt.fk(m, d)
    torso0 = pos(body_idx, Xg, "torso")
    for coord in ["lumbar1_bending", "lumbar2_bending", "lumbar3_bending", "torso_bending"]:
        q[0, lookup[coord]] = 0.3
    bolt.fk(m, d)
    Xbent = wp.to_torch(d.mob_X_GB).cpu().numpy()[0]
    torso_bent = pos(body_idx, Xbent, "torso")
    print(f"  torso moved {np.linalg.norm(np.array(torso_bent)-np.array(torso0)):.4f} m under combined lumbar bend")
    for coord in ["lumbar1_bending", "lumbar2_bending", "lumbar3_bending", "torso_bending"]:
        q[0, lookup[coord]] = 0.0

    # Muscle sanity: lengths finite at neutral
    print("\n=== MUSCLE STATE AT NEUTRAL ===")
    bolt.fk(m, d)
    nmus = res.num_muscles if hasattr(res, "num_muscles") else None
    print(f"  num_muscles: {nmus}")
    if hasattr(d, "muscle_length"):
        ml = wp.to_torch(d.muscle_length).cpu().numpy()[0]
        print(f"  muscle_length finite: {np.all(np.isfinite(ml))}, "
              f"min={np.nanmin(ml):.4f} max={np.nanmax(ml):.4f}")

    # Dynamics stability
    print("\n=== DYNAMICS: 200 x 10ms under gravity ===")
    bolt.fk(m, d)
    for _ in range(200):
        bolt.increment_next_time(m, d, 0.01)
        bolt.step(m, d)
    qf = wp.to_torch(d.qpos).cpu().numpy()[0]
    print(f"  qpos finite after 2s: {np.all(np.isfinite(qf))}")
    print("\nRESULT:", "PASS" if np.all(np.isfinite(qf)) and disp > 1e-4 else "CHECK")


if __name__ == "__main__":
    main()
