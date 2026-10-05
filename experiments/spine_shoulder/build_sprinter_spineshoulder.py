"""Transform sprinter_model_sym.osim -> a flexible-spine + mobile-scapula variant.

Two edits, both using only Bolt-loadable primitives (no constraint solver):

1. SPINE: replace the single 3-DOF `back` CustomJoint (pelvis->torso) with a
   3-segment lumbar chain pelvis -> lumbar1 -> lumbar2 -> lumbar3 -> torso, each a
   3-DOF CustomJoint (extension/bending/rotation, same axes as the old back joint).
   The chain's offsets are distributed so the torso -- and therefore the whole
   upper body incl. the scapula ellipsoids -- lands in its EXACT original neutral
   pose. Mass is carved from the torso and conserved.

2. SHOULDER: replace the scapulothoracic_r/_l WeldJoints with stock OpenSim
   EllipsoidJoints (Seth 2019 ribcage radii). Bolt's ELLIPSOID mobilizer places the
   scapula on the ellipsoid by construction -- Seth's cheap form, no constraint.

New coordinates get SpringGeneralizedForce (passive damping) + CoordinateLimitForce
(soft end stops), mirroring how the existing model stabilizes its joints. This
matters because the model has NO scapula-positioning muscles (serratus/trapezius/
rhomboid absent), so the freed scapula must be passively restrained.
"""
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim

SRC = "msk_envs/msk_models/sprinter/sprinter_model_sym.osim"
SETH_RADII = (0.083, 0.20, 0.083)
N_LUMBAR = 3
# Per-segment rotational range (rad). 3 segments * ~20deg ~= original +/-60deg total.
SEG_RANGE = np.deg2rad(25.0)
# Scapulothoracic coordinate range. Kept small: with a 0.20 m vertical ellipsoid
# radius, the mobilizer converts rotation into surface translation, so a large
# range makes the scapula "bob" many cm (±40deg gave ~18 cm of vertical travel).
# Physiological scapular glide is a few cm, so ~15deg keeps it realistic and stops
# the girdle from windmilling.
SCAP_RANGE = np.deg2rad(15.0)


def _measure(src):
    """Neutral-pose geometry we must preserve: back-joint offsets and torso origin."""
    osim.Logger.setLevelString("Off")
    m = osim.Model(src)
    s = m.initSystem()
    back = m.getJointSet().get("back")
    pf = osim.PhysicalOffsetFrame.safeDownCast(back.getParentFrame())
    p_off = np.array([pf.get_translation().get(i) for i in range(3)])  # on pelvis
    torso0 = m.getBodySet().get("torso").getPositionInGround(s)
    pelvis0 = m.getBodySet().get("pelvis").getPositionInGround(s)
    torso_in_pelvis = np.array([torso0.get(i) - pelvis0.get(i) for i in range(3)])
    torso_mass = m.getBodySet().get("torso").getMass()
    return p_off, torso_in_pelvis, torso_mass


def indent(elem, level=0):
    pad = "\n" + "\t" * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "\t"
        for child in elem:
            indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def body_xml(name, mass, com, inertia):
    return f"""<Body name="{name}">
    <mass>{mass}</mass>
    <mass_center>{com[0]} {com[1]} {com[2]}</mass_center>
    <inertia>{inertia[0]} {inertia[1]} {inertia[2]} 0 0 0</inertia>
</Body>"""


def offset_frame_xml(name, parent_body, trans):
    return f"""<PhysicalOffsetFrame name="{name}">
    <socket_parent>/bodyset/{parent_body}</socket_parent>
    <translation>{trans[0]} {trans[1]} {trans[2]}</translation>
    <orientation>0 0 0</orientation>
</PhysicalOffsetFrame>"""


def custom_joint_xml(name, prefix, parent_body, parent_trans, child_body, rng):
    """A 3-DOF CustomJoint: extension(z), bending(x), rotation(y). Same axis
    convention as the original `back` joint. Parent offset on parent_body,
    child frame at child_body origin (zero offset)."""
    pfn = f"{name}_parent_frame"
    cfn = f"{name}_child_frame"
    return f"""<CustomJoint name="{name}">
    <socket_parent_frame>{pfn}</socket_parent_frame>
    <socket_child_frame>{cfn}</socket_child_frame>
    <coordinates>
        <Coordinate name="{prefix}_extension"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
        <Coordinate name="{prefix}_bending"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
        <Coordinate name="{prefix}_rotation"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
    </coordinates>
    <frames>
        <PhysicalOffsetFrame name="{pfn}">
            <socket_parent>/bodyset/{parent_body}</socket_parent>
            <translation>{parent_trans[0]} {parent_trans[1]} {parent_trans[2]}</translation>
            <orientation>0 0 0</orientation>
        </PhysicalOffsetFrame>
        <PhysicalOffsetFrame name="{cfn}">
            <socket_parent>/bodyset/{child_body}</socket_parent>
            <translation>0 0 0</translation>
            <orientation>0 0 0</orientation>
        </PhysicalOffsetFrame>
    </frames>
    <SpatialTransform>
        <TransformAxis name="rotation1"><coordinates>{prefix}_extension</coordinates><axis>0 0 1</axis>
            <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="rotation2"><coordinates>{prefix}_bending</coordinates><axis>1 0 0</axis>
            <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="rotation3"><coordinates>{prefix}_rotation</coordinates><axis>0 1 0</axis>
            <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="translation1"><axis>1 0 0</axis><Constant name="function"><value>0</value></Constant></TransformAxis>
        <TransformAxis name="translation2"><axis>0 1 0</axis><Constant name="function"><value>0</value></Constant></TransformAxis>
        <TransformAxis name="translation3"><axis>0 0 1</axis><Constant name="function"><value>0</value></Constant></TransformAxis>
    </SpatialTransform>
</CustomJoint>"""


def ellipsoid_joint_xml(name, parent_body, parent_trans, parent_orient, child_body,
                        child_trans, child_orient, coord_prefix, radii, rng):
    pfn, cfn = f"{name}_parent_frame", f"{name}_child_frame"
    return f"""<EllipsoidJoint name="{name}">
    <socket_parent_frame>{pfn}</socket_parent_frame>
    <socket_child_frame>{cfn}</socket_child_frame>
    <coordinates>
        <Coordinate name="scap_abduction_{coord_prefix}"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
        <Coordinate name="scap_elevation_{coord_prefix}"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
        <Coordinate name="scap_uprot_{coord_prefix}"><range>{-rng} {rng}</range><clamped>true</clamped></Coordinate>
    </coordinates>
    <frames>
        <PhysicalOffsetFrame name="{pfn}">
            <socket_parent>/bodyset/{parent_body}</socket_parent>
            <translation>{parent_trans[0]} {parent_trans[1]} {parent_trans[2]}</translation>
            <orientation>{parent_orient[0]} {parent_orient[1]} {parent_orient[2]}</orientation>
        </PhysicalOffsetFrame>
        <PhysicalOffsetFrame name="{cfn}">
            <socket_parent>/bodyset/{child_body}</socket_parent>
            <translation>{child_trans[0]} {child_trans[1]} {child_trans[2]}</translation>
            <orientation>{child_orient[0]} {child_orient[1]} {child_orient[2]}</orientation>
        </PhysicalOffsetFrame>
    </frames>
    <radii_x_y_z>{radii[0]} {radii[1]} {radii[2]}</radii_x_y_z>
</EllipsoidJoint>"""


def spring_force_xml(name, coord, viscosity, stiffness=0.0):
    return f"""<SpringGeneralizedForce name="{name}">
    <coordinate>{coord}</coordinate>
    <stiffness>{stiffness}</stiffness>
    <rest_length>0</rest_length>
    <viscosity>{viscosity}</viscosity>
</SpringGeneralizedForce>"""


def limit_force_xml(name, coord, rng_deg, stiff=8.726646, damping=0.017453, transition=5.0):
    return f"""<CoordinateLimitForce name="{name}">
    <coordinate>{coord}</coordinate>
    <upper_stiffness>{stiff}</upper_stiffness>
    <upper_limit>{rng_deg}</upper_limit>
    <lower_stiffness>{stiff}</lower_stiffness>
    <lower_limit>{-rng_deg}</lower_limit>
    <damping>{damping}</damping>
    <transition>{transition}</transition>
    <compute_dissipation_energy>false</compute_dissipation_energy>
</CoordinateLimitForce>"""


def frag(xml_string):
    return ET.fromstring(xml_string)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="experiments/spine_shoulder/sprinter_spineshoulder.osim")
    ap.add_argument("--spine", dest="spine", action="store_true", default=None,
                    help="add the multi-segment lumbar spine")
    ap.add_argument("--no-spine", dest="spine", action="store_false")
    ap.add_argument("--shoulder", dest="shoulder", action="store_true", default=None,
                    help="add the ellipsoid mobile scapula")
    ap.add_argument("--no-shoulder", dest="shoulder", action="store_false")
    args = ap.parse_args()
    # default: both on (full model), matching prior behavior
    do_spine = True if args.spine is None else args.spine
    do_shoulder = True if args.shoulder is None else args.shoulder
    print(f"ABLATION: spine={do_spine} shoulder={do_shoulder}")

    p_off, torso_in_pelvis, torso_mass = _measure(args.src)
    # Rise from the pelvis-side pivot to the torso origin, split across N+1 joints.
    rise = torso_in_pelvis - p_off
    step = rise / N_LUMBAR
    print(f"back parent offset on pelvis = {np.round(p_off,4)}")
    print(f"torso origin (in pelvis)     = {np.round(torso_in_pelvis,4)}")
    print(f"per-joint climb step         = {np.round(step,4)}")

    tree = ET.parse(args.src)
    root = tree.getroot()
    bodyset = root.find(".//BodySet/objects")
    jointset = root.find(".//JointSet/objects")
    forceset = root.find(".//ForceSet/objects")

    if do_spine:
        # --- carve mass for lumbar segments from the torso, conserve total ---
        seg_mass = 2.4
        for b in bodyset.findall("Body"):
            if b.get("name") == "torso":
                mass_el = b.find("mass")
                mass_el.text = str(round(torso_mass - N_LUMBAR * seg_mass, 4))
                print(f"torso mass {torso_mass} -> {mass_el.text} (carved {N_LUMBAR*seg_mass})")

        # Bolt's kinematic-tree builder requires parent-before-child order and OpenSim
        # preserves file order, so lumbar BODIES must appear before `torso` and lumbar
        # JOINTS must appear where `back` was (before the shoulder joints that use torso).
        seg_com = (0.0, 0.0, 0.0)
        seg_I = (0.02, 0.02, 0.02)
        torso_body_idx = next(i for i, b in enumerate(list(bodyset)) if b.get("name") == "torso")
        for k in range(N_LUMBAR):
            bodyset.insert(torso_body_idx + k,
                           frag(body_xml(f"lumbar{k+1}", seg_mass, seg_com, seg_I)))

        # --- build the lumbar chain in place of the old `back` joint ---
        back_idx = next(i for i, j in enumerate(list(jointset)) if j.get("name") == "back")
        jointset.remove(jointset[back_idx])
        chain = []
        # joint 1: pelvis -> lumbar1, parent offset = original back parent offset. Keep the
        # name "back" so downstream code/pose files referencing it still resolve.
        chain.append(custom_joint_xml("back", "lumbar1", "pelvis", p_off, "lumbar1", SEG_RANGE))
        for i in range(2, N_LUMBAR + 1):
            chain.append(custom_joint_xml(
                f"lumbar{i}_joint", f"lumbar{i}", f"lumbar{i-1}", step, f"lumbar{i}", SEG_RANGE))
        # final joint: lumbar{N} -> torso; coord prefix "torso" so coords are
        # torso_extension/bending/rotation (matches fn.xml trim + symmetry negate set).
        chain.append(custom_joint_xml(
            "lumbar_torso", "torso", f"lumbar{N_LUMBAR}", step, "torso", SEG_RANGE))
        for k, xml in enumerate(chain):
            jointset.insert(back_idx + k, frag(xml))

    # --- swap scapulothoracic welds -> EllipsoidJoints, IN PLACE ---
    # Derive each ellipsoid's parent frame (on torso) from the ORIGINAL neutral
    # scapula pose so at q=0 the scapula lands where the weld put it.
    osim.Logger.setLevelString("Off")
    m_orig = osim.Model(args.src)
    s_orig = m_orig.initSystem()
    torso_frame = m_orig.getBodySet().get("torso")
    semi_z = SETH_RADII[2]
    def ground_RT(frame):
        X = frame.getTransformInGround(s_orig)
        p = np.array([X.p().get(i) for i in range(3)])
        R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
        return R, p

    def R_to_bodyfixed_xyz(R):
        # OpenSim PhysicalOffsetFrame orientation = body-fixed X-Y-Z euler. Use
        # OpenSim's OWN exact conversion (a hand-rolled formula was lossy for this
        # rotation and displaced the scapula enough to push a muscle cable inside
        # its wrap surface). Round-trips to machine precision.
        rot = osim.Rotation()
        for r in range(3):
            for c in range(3):
                rot.set(r, c, float(R[r, c]))
        e = rot.convertRotationToBodyFixedXYZ()
        return [e.get(0), e.get(1), e.get(2)]

    if do_shoulder:
        Rt, pt = ground_RT(torso_frame)
        # Compute the RIGHT ellipsoid parent frame from the model, then build the LEFT
        # as its EXACT mirror across z=0. Computing each side independently (as an
        # earlier version did) let the two joints drift out of mirror symmetry --
        # under identical L/R coords the scapulae moved to opposite heights (verified
        # by experiments/spine_shoulder/render_shoulders_fk.py). The source scapulae
        # ARE perfect mirrors, so we only fit the right and reflect it.
        Rr, pr = ground_RT(m_orig.getBodySet().get("scapula_r"))
        R0 = Rt.T @ Rr                       # X_torso->scapula_r rotation
        p0 = Rt.T @ (pr - pt)                # X_torso->scapula_r translation
        # Bolt ellipsoid at q=0: child origin sits at parent_offset * (R=I, p=(0,0,semi_z)).
        t_r = p0 - R0 @ np.array([0.0, 0.0, semi_z])
        # LEFT = EXACT mirror of right across z=0, achievable with the stock Bolt
        # ELLIPSOID mobilizer under IDENTICAL L/R commands (msk_envs/utils/symmetry.py
        # swaps scap_*_r/_l without negating them, so the fix must NOT rely on a
        # coordinate sign-flip). Mirror the ROTATION MATRIX (R0l = M R0 M,
        # M=diag(1,1,-1)) -- proper->proper, so its body-fixed XYZ euler is valid.
        #
        # WHY THE PREVIOUS BUILD WAS ONLY APPROXIMATELY MIRRORED:
        # the mobilizer places the child at p = D*n, D=diag(semi), n = R(q)*e_z the
        # (shared) surface normal. A true z=0 reflection requires R0l*D = M*R0*D_r, so
        # taking determinants det(R0l)=+1 while det(M*R0)=-1 forces det(D_l)=-det(D_r),
        # i.e. the LEFT z-radius must be NEGATED. Keeping D_l=D_r (old build) left an
        # uncancelled surface-normal term 2*semi_z*(1-n_z) -> ~0.003-0.011 m residual
        # that is zero at neutral and grows with abduction/elevation.
        #
        # FIX (verified residual 0 at all deflections, identical L/R commands):
        #   radii_l = (semi_x, semi_y, -semi_z), and the baked pole offset flips too,
        #   so t_l = M@p0 - R0l @ (0,0,-semi_z). Bolt uses semi.z linearly in p, H_FM
        #   and HDot_FM (only radii-independent 1/cos(q1) division), so a negative
        #   semi_z flows through kinematics/velocity/acceleration consistently; OpenSim
        #   EllipsoidJoint honors the negative radius. (A renderer drawing the surface
        #   should use abs(semi_z); the mobilizer math does not.)
        M = np.diag([1.0, 1.0, -1.0])
        R0l = M @ R0 @ M
        radii_r = SETH_RADII
        radii_l = (SETH_RADII[0], SETH_RADII[1], -SETH_RADII[2])
        t_l = (M @ p0 - R0l @ np.array([0.0, 0.0, -semi_z])).tolist()
        o_r = R_to_bodyfixed_xyz(R0)
        o_l = R_to_bodyfixed_xyz(R0l)
        frames = {"r": (list(t_r), o_r, radii_r), "l": (t_l, o_l, radii_l)}
        for side in ("r", "l"):
            name = f"scapulothoracic_{side}"
            idx = next(i for i, j in enumerate(list(jointset)) if j.get("name") == name)
            t_p, o_p, radii_s = frames[side]
            jointset.remove(jointset[idx])
            jointset.insert(idx, frag(ellipsoid_joint_xml(
                name, "torso", t_p, o_p,
                f"scapula_{side}", (0, 0, 0), (0, 0, 0),
                side, radii_s, SCAP_RANGE)))

    # --- passive forces for the new coordinates ---
    new_lumbar_coords = (
        [f"back_ext_placeholder"]  # back joint coords are lumbar1_*
    )
    lumbar_prefixes = ["lumbar1"] + [f"lumbar{i}" for i in range(2, N_LUMBAR + 1)] + ["lumbar_torso_seg"]
    # actual coord names created above:
    lumbar_coord_names = []
    for pfx in (["lumbar1"] + [f"lumbar{i}" for i in range(2, N_LUMBAR + 1)]):
        lumbar_coord_names += [f"{pfx}_extension", f"{pfx}_bending", f"{pfx}_rotation"]
    # the final lumbar_torso joint uses prefix 'lumbar_torso' via custom_joint_xml? No:
    # custom_joint_xml derives coord names from `prefix` arg. Fix: we passed child body as prefix?
    # Re-derive from the joints we actually created:
    lumbar_coord_names = []
    for j in jointset:
        if j.get("name") in ("back",) or j.get("name", "").startswith("lumbar"):
            for c in j.findall(".//Coordinate"):
                lumbar_coord_names.append(c.get("name"))

    # Intervertebral restoring stiffness + damping. Nonzero stiffness is ESSENTIAL:
    # the endpoint-only trunk muscles control just the TOTAL pelvis->torso angle, so
    # the 3-segment stack is internally underdetermined and buckles without a per-joint
    # neutral-posture spring (this is what real intervertebral discs/ligaments provide).
    # Scaled per-segment so the chain shares a total trunk stiffness comparable to the
    # original single back joint's effective resistance.
    LUMBAR_STIFFNESS = 40.0  # N*m/rad per segment coordinate (disc/ligament surrogate)
    for coord in lumbar_coord_names:
        forceset.append(frag(spring_force_xml(f"{coord}_damp", coord,
                                              viscosity=2.0, stiffness=LUMBAR_STIFFNESS)))
        forceset.append(frag(limit_force_xml(f"{coord}_limit", coord,
                                             rng_deg=round(np.rad2deg(SEG_RANGE), 1))))
    scap_coords = []
    for j in jointset:
        if j.get("name", "").startswith("scapulothoracic"):
            for c in j.findall(".//Coordinate"):
                scap_coords.append(c.get("name"))
    for coord in scap_coords:
        forceset.append(frag(spring_force_xml(f"{coord}_damp", coord, viscosity=1.0, stiffness=5.0)))
        forceset.append(frag(limit_force_xml(f"{coord}_limit", coord,
                                             rng_deg=round(np.rad2deg(SCAP_RANGE), 1), stiff=2.0)))

    indent(root)
    tree.write(args.out, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {args.out}")
    print(f"new lumbar coords ({len(lumbar_coord_names)}): {lumbar_coord_names}")
    print(f"new scapula coords ({len(scap_coords)}): {scap_coords}")

    # --- round-trip validation + geometry preservation check ---
    osim.Logger.setLevelString("Off")
    m2 = osim.Model(args.out)
    s2 = m2.initSystem()
    print(f"OPENSIM ROUND-TRIP OK: {m2.getNumBodies()} bodies, {m2.getNumCoordinates()} coords")
    # torso + scapula neutral positions vs original
    m0 = osim.Model(args.src); s0 = m0.initSystem()
    for bn in ("torso", "scapula_r", "scapula_l", "humerus_r"):
        p_new = m2.getBodySet().get(bn).getPositionInGround(s2)
        p_old = m0.getBodySet().get(bn).getPositionInGround(s0)
        d = np.array([p_new.get(i) - p_old.get(i) for i in range(3)])
        print(f"  {bn:12s} neutral delta = {np.round(d,4)}  |{np.linalg.norm(d):.4f}| m")


if __name__ == "__main__":
    main()
