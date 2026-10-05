"""Independent check of the ELLIPSOID mobilizer with pure numpy (no OpenSim solver),
replicating bolt/_src/mobilizers.py exactly, to rule out an OpenSim assemble artifact.

Bolt ELLIPSOID at coords q=(q0,q1,q2):
   r = quat_from_xyz(q0,q1,q2)         # intrinsic body-fixed XYZ
   n = r.rotate([0,0,1])
   p = (semi.x*n.x, semi.y*n.y, semi.z*n.z)
   child_in_parentframe = transform(p, r)

We parse both scapulothoracic parent frames (translation+orientation, radii) from the
osim, build the torso->parentframe transform, apply the SAME q to both, and compare
scapula_l against M @ scapula_r for BOTH origin and orientation.
"""
import numpy as np
import xml.etree.ElementTree as ET

PATH = "experiments/spine_shoulder/sprinter_shoulderonly.osim"
M = np.diag([1.0, 1.0, -1.0])


def rot_xyz_bodyfixed(a, b, c):
    """Body-fixed (intrinsic) X then Y then Z: R = Rx(a) Ry(b) Rz(c)."""
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def parse_scap(path, side):
    tree = ET.parse(path)
    root = tree.getroot()
    for j in root.iter("EllipsoidJoint"):
        if j.get("name") == f"scapulothoracic_{side}":
            pf = None
            for f in j.iter("PhysicalOffsetFrame"):
                if f.get("name") == f"scapulothoracic_{side}_parent_frame":
                    pf = f
            trans = np.array([float(x) for x in pf.find("translation").text.split()])
            orient = np.array([float(x) for x in pf.find("orientation").text.split()])
            radii = np.array([float(x) for x in j.find("radii_x_y_z").text.split()])
            return trans, orient, radii
    raise KeyError(side)


def ellipsoid_child(q, semi):
    # r = Rx(q0) Ry(q1) Rz(q2); n = r @ e_z; p = semi * n
    R = rot_xyz_bodyfixed(q[0], q[1], q[2])
    n = R @ np.array([0.0, 0.0, 1.0])
    p = np.array([semi[0] * n[0], semi[1] * n[1], semi[2] * n[2]])
    return p, R


def main():
    # torso frame is shared and on the midline; to compare the two scapulae relative to
    # torso we only need the parent-frame transforms (torso->parentframe) since a common
    # left-multiply by X_torso cancels in the mirror relation expressed in torso frame.
    tr, orr, rr = parse_scap(PATH, "r")
    tl, orl, rl = parse_scap(PATH, "l")
    print(f"RIGHT parent: t={np.round(tr,5)} orient={np.round(orr,5)} radii={rr}")
    print(f"LEFT  parent: t={np.round(tl,5)} orient={np.round(orl,5)} radii={rl}")

    Rp_r = rot_xyz_bodyfixed(*orr)
    Rp_l = rot_xyz_bodyfixed(*orl)
    print(f"\nparent-frame mirror check ||Rp_l - M Rp_r M|| = {np.linalg.norm(Rp_l - M@Rp_r@M):.3e}")
    print(f"parent transl mirror check ||t_l - M t_r|| = {np.linalg.norm(tl - M@tr):.3e}")

    print("\n=== identical q both sides; residual of scapula frame in TORSO coords ===")
    print("origin residual = ||p_l - M p_r||;  orient residual = ||R_l - M R_r M||")
    for amp in (0.1, 0.2, 0.4):
        for axis, nm in enumerate(["abduction", "elevation", "uprot"]):
            q = [0.0, 0.0, 0.0]; q[axis] = amp
            pr_, Rr_ = ellipsoid_child(q, rr)
            pl_, Rl_ = ellipsoid_child(q, rl)
            # full transform in torso frame: X = parentframe ∘ child
            Pr = tr + Rp_r @ pr_
            Rr_w = Rp_r @ Rr_
            Pl = tl + Rp_l @ pl_
            Rl_w = Rp_l @ Rl_
            o_res = np.linalg.norm(Pl - M @ Pr)
            r_res = np.linalg.norm(Rl_w - M @ Rr_w @ M)
            print(f"  {nm:9s}+{amp:<4}: origin={o_res:.3e}  orient={r_res:.3e}")


if __name__ == "__main__":
    main()
