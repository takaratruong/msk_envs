"""Build a minimal .osim exercising the two primitives the spine/shoulder redesign needs:

  ground --(WeldJoint)-- pelvis
      --(BallJoint x3: lumbar1..3, the flexible 'tube')-- thorax
      --(EllipsoidJoint: scapulothoracic, Seth radii)--   scapula_r

Bolt has never loaded an EllipsoidJoint or a chain like this; this model exists purely
to prove the loader + kinematics path before touching the sprinter.
"""
import argparse

import opensim as osim


SETH_RADII = (0.083, 0.20, 0.083)  # thorax ellipsoid, Seth et al. 2019 (m)


def add_body(model, name, mass, inertia=(0.01, 0.01, 0.01)):
    body = osim.Body(name, mass, osim.Vec3(0), osim.Inertia(*inertia))
    model.addBody(body)
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/spine_shoulder/smoke_model.osim")
    args = parser.parse_args()

    model = osim.Model()
    model.setName("spine_shoulder_smoke")
    ground = model.getGround()

    pelvis = add_body(model, "pelvis", 10.0)
    model.addJoint(osim.WeldJoint("ground_pelvis", ground, osim.Vec3(0, 1.0, 0), osim.Vec3(0),
                                  pelvis, osim.Vec3(0), osim.Vec3(0)))

    # Flexible lumbar tube: 3 stacked segments on BallJoints, 0.05 m apart.
    parent = pelvis
    for i in range(1, 4):
        seg = add_body(model, f"lumbar{i}", 2.0)
        joint = osim.BallJoint(f"lumbar{i}_joint", parent, osim.Vec3(0, 0.05, 0), osim.Vec3(0),
                               seg, osim.Vec3(0), osim.Vec3(0))
        for c in range(3):
            coord = joint.upd_coordinates(c)
            coord.setName(f"lumbar{i}_q{c}")
            coord.setRangeMin(-0.6)
            coord.setRangeMax(0.6)
        model.addJoint(joint)
        parent = seg

    thorax = add_body(model, "thorax", 15.0, inertia=(0.1, 0.1, 0.1))
    model.addJoint(osim.WeldJoint("lumbar_thorax", parent, osim.Vec3(0, 0.05, 0), osim.Vec3(0),
                                  thorax, osim.Vec3(0), osim.Vec3(0)))

    # Seth-style scapulothoracic: stock EllipsoidJoint, thorax -> scapula.
    scapula = add_body(model, "scapula_r", 0.7, inertia=(0.001, 0.001, 0.001))
    st = osim.EllipsoidJoint("scapulothoracic_r",
                             thorax, osim.Vec3(0, 0.1, 0), osim.Vec3(0),
                             scapula, osim.Vec3(0), osim.Vec3(0),
                             osim.Vec3(*SETH_RADII))
    for c in range(3):
        coord = st.upd_coordinates(c)
        coord.setName(f"scap_q{c}")
        coord.setRangeMin(-0.7)
        coord.setRangeMax(0.7)
    model.addJoint(st)

    model.finalizeConnections()
    model.printToXML(args.out)
    print(f"wrote {args.out}")

    # Sanity: OpenSim itself can round-trip it.
    check = osim.Model(args.out)
    check.initSystem()
    print(f"opensim round-trip OK: {check.getNumBodies()} bodies, "
          f"{check.getNumCoordinates()} coordinates")


if __name__ == "__main__":
    main()
