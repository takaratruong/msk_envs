#!/usr/bin/env python3
"""Read the frozen physical scene into explicit motion-planning geometry.

USD row-vector matrices are transposed once at this boundary. Geometry is
already in nominal metres and Z up; no inferred reconstruction scale is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangles(mesh):
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    result, offset = [], 0
    for count in counts:
        poly = indices[offset:offset + count]
        result.extend((int(poly[0]), int(poly[i]), int(poly[i + 1]))
                      for i in range(1, count - 1))
        offset += count
    return np.asarray(result, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--door', type=int, choices=[0, 1], default=1)
    parser.add_argument('--grip-height', type=float, default=1.15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(args.scene.resolve()))
    assert UsdGeom.GetStageUpAxis(stage) == 'Z'
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    cache = UsdGeom.XformCache()
    body_path = f'/World/Assets/refrigerator_01/Links/upper_door_{args.door}'
    joint_path = f'/World/Assets/refrigerator_01/Joints/upper_door_{args.door}_hinge'
    body = stage.GetPrimAtPath(body_path)
    body_world = np.asarray(cache.GetLocalToWorldTransform(body)).T
    body_parent = np.asarray(cache.GetLocalToWorldTransform(body.GetParent())).T
    body_local = np.linalg.inv(body_parent) @ body_world
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
    base_world = np.asarray(cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(str(joint.GetBody0Rel().GetTargets()[0])))).T
    pivot_world = base_world[:3, :3] @ np.asarray(joint.GetLocalPos0Attr().Get()) + base_world[:3, 3]
    pivot_parent = np.linalg.inv(body_parent) @ np.r_[pivot_world, 1.]
    handle = stage.GetPrimAtPath(body_path + f'/upper_door_{args.door}_long_pull_segment_09')
    handle_world = np.asarray(cache.GetLocalToWorldTransform(handle)).T
    points = np.asarray(UsdGeom.Mesh(handle).GetPointsAttr().Get())
    points_world = points @ handle_world[:3, :3].T + handle_world[:3, 3]
    grip_world = .5 * (points_world.min(0) + points_world.max(0))
    # Selected point along the straight vertical rail; changing the grasp height
    # does not alter scene geometry. Keep the source centre as separate evidence.
    source_rail_center = grip_world.copy()
    grip_world[2] = args.grip_height
    grip_body = np.linalg.inv(body_world) @ np.r_[grip_world, 1.]
    fridge = stage.GetPrimAtPath('/World/Assets/refrigerator_01')
    fridge_world = np.asarray(cache.GetLocalToWorldTransform(fridge)).T
    outward = -fridge_world[:3, 1]
    task = dict(
        schema='g1-fridge-motion-task/v1',
        scene=dict(path=str(args.scene.resolve()), sha256=sha(args.scene)),
        units='nominal_meters', up_axis='Z', physical_execution_verified=False,
        door=dict(body_path=body_path, joint_path=joint_path,
                  axis_parent=[0., 0., 1.], pivot_parent=pivot_parent[:3].tolist(),
                  reference_angle_radians=0., world_pivot=pivot_world.tolist(),
                  source_local_transform=body_local.tolist(),
                  source_world_transform=body_world.tolist(),
                  limits_degrees=[joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]),
        grasp=dict(rail_source_path=str(handle.GetPath()),
                   source_rail_center_world=source_rail_center.tolist(),
                   rail_point_world=grip_world.tolist(), rail_point_body=grip_body[:3].tolist(),
                   selected_height_m=args.grip_height, rail_radius_m=.009,
                   hand='right', frame_status='selected rail point; robot grasp offset solved separately'),
        fridge_outward_world=outward.tolist(),
        facing_yaw_radians=float(np.arctan2(-outward[1], -outward[0])),
        notes=['Source geometry and scale are modeled estimates.',
               'Door angle is a planned kinematic coordinate, not measured robot-driven opening.'])
    (args.output/'task.json').write_text(json.dumps(task, indent=2)+'\n')
    vs, fs, records = [], [], []
    v_count = f_count = 0
    assets = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        faces = triangles(mesh)
        path = str(prim.GetPath())
        asset = path.split('/')[3] if path.startswith('/World/Assets/') else path.split('/')[2]
        bb = [vertices.min(0).tolist(), vertices.max(0).tolist()]
        moving = path.startswith(body_path+'/')
        record = dict(path=path, asset=asset, vertex_start=v_count, vertex_count=len(vertices),
                      face_start=f_count, face_count=len(faces), bounds=bb, selected_door=moving,
                      approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                      if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else 'none')
        records.append(record)
        vs.append(vertices.astype(np.float32)); fs.append(faces+v_count)
        v_count += len(vertices); f_count += len(faces)
        if asset not in assets: assets[asset] = np.array(bb)
        else:
            assets[asset][0] = np.minimum(assets[asset][0], bb[0])
            assets[asset][1] = np.maximum(assets[asset][1], bb[1])
    np.savez_compressed(args.output/'collision.npz', vertices=np.concatenate(vs), faces=np.concatenate(fs))
    index = dict(schema='g1-source-collision/v1', scene_sha256=sha(args.scene),
                 geometry_sha256=sha(args.output/'collision.npz'), parts=records,
                 asset_bounds={k:v.tolist() for k,v in assets.items()})
    (args.output/'collision.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps(dict(task=str(args.output/'task.json'), parts=len(records),
                         vertices=v_count, triangles=f_count, facing=task['facing_yaw_radians'])))


if __name__ == '__main__':
    main()
