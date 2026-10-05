#!/usr/bin/env python3
"""Export compiled MuJoCo visual geometry/FK as a portable kinematic USD preview.

No MuJoCo integration or USD physics simulation occurs. qpos must already use
the scene's nominal world frame. Source layers/textures are copied byte-for-byte.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


SCENE_SHA = 'f78a48e358080b0079c9d176beea9286ef8e9eb2e4ff827eb90ca686f2647fbc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def matrix(position, rotation):
    """USD/Gf uses row vectors; MuJoCo xmat is a column-vector rotation."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation).reshape(3, 3).T
    out[3, :3] = position
    return out


def quat_matrix(quaternion):
    out = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(out, np.asarray(quaternion, dtype=np.float64))
    return out.reshape(3, 3)


def about_pivot(axis, pivot, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    out = matrix(np.zeros(3), rotation)
    out[3, :3] = np.asarray(pivot) - np.asarray(pivot) @ out[:3, :3]
    return out


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'size': path.stat().st_size}


def checked_input(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as inp:
        if b'version https://git-lfs.github.com/spec/v1' in inp.read(256):
            raise ValueError('Git LFS pointer is not an asset: ' + str(path))
    return file_record(path)


def model_inputs(path):
    """Resolve XML includes and file-backed mesh/texture assets, without writes."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    texturedir = compiler.get('texturedir', '') if compiler is not None else ''
    records = {}

    def read(document):
        document = document.resolve(strict=True)
        if str(document) in records:
            return
        records[str(document)] = checked_input(document)
        tree = ET.parse(document).getroot()
        for node in tree.iter('include'):
            read(document.parent / node.attrib['file'])
        for kind, directory in [('mesh', meshdir), ('texture', texturedir)]:
            for node in tree.iter(kind):
                if 'file' in node.attrib:
                    raw = Path(node.attrib['file'])
                    asset = raw if raw.is_absolute() else path.parent / directory / raw
                    records[str(asset.resolve())] = checked_input(asset)
    read(path)
    return list(records.values())


def environment_copy(source, output):
    """Preserve source relative asset paths by copying their common directory tree."""
    stage = Usd.Stage.Open(str(source))
    assert stage, source
    files = {source.resolve()}
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            files.add(Path(layer.realPath).resolve(strict=True))
    for prim in stage.Traverse():
        for attr in prim.GetAuthoredAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value and value.path:
                resolved = Path(value.resolvedPath) if value.resolvedPath else source.parent / value.path
                if not resolved.is_file():
                    raise ValueError('Unresolved source asset: ' + value.path)
                if Path(value.path).is_absolute():
                    raise ValueError('Cannot preserve absolute asset paths in an unchanged portable source layer: ' + value.path)
                files.add(resolved.resolve())
    common = Path(os.path.commonpath([str(p.parent) for p in files]))
    copied = []
    for original in sorted(files):
        relative = original.relative_to(common)
        destination = output / 'environment' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        assert sha(original) == sha(destination)
        copied.append({'source': file_record(original), 'copy': file_record(destination)})
    return stage, output / 'environment' / source.relative_to(common), copied


def trajectory_samples(model, path, fps):
    arrays = np.load(path, allow_pickle=False)
    qpos = np.asarray(arrays['qpos'], dtype=np.float64)
    times = np.asarray(arrays['time'], dtype=np.float64)
    door = np.asarray(arrays['door_angle'], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or times.shape != (len(qpos),) or door.shape != times.shape:
        raise ValueError(f'Expected qpos[T,{model.nq}], time[T], door_angle[T]')
    if len(times) < 2 or not np.isfinite(qpos).all() or not np.isfinite(times).all() or not np.isfinite(door).all():
        raise ValueError('At least two finite trajectory samples are required')
    if abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0):
        raise ValueError('Trajectory time must start at0 and increase strictly')
    # Preserve time duration and endpoint, with a native uniform30-Hz sample grid.
    count = int(round(times[-1] * fps))
    if abs(count / fps - times[-1]) > 1e-6:
        raise ValueError('End time must be aligned to the requested USD frame rate')
    target = np.arange(count + 1, dtype=np.float64) / fps
    if len(times) == len(target) and np.allclose(times, target, rtol=0, atol=1e-8):
        sampled = qpos.copy()
        sampled_door = door.copy()
        resampled = False
    else:
        sampled = np.empty((len(target), model.nq))
        velocity = np.zeros(model.nv)
        for i, t in enumerate(target):
            j = int(np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times) - 2))
            dt = times[j + 1] - times[j]
            mujoco.mj_differentiatePos(model, velocity, dt, qpos[j], qpos[j + 1])
            sampled[i] = qpos[j]
            mujoco.mj_integratePos(model, sampled[i], velocity, float(t - times[j]))
        sampled_door = np.interp(target, times, door)
        resampled = True
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        adr = int(model.jnt_qposadr[joint])
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            adr += 3
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            if np.max(np.abs(np.linalg.norm(sampled[:, adr:adr + 4], axis=1) - 1.)) > 1e-5:
                raise ValueError('Input has non-unit joint quaternions; exporter does not repair poses')
    return target, sampled, sampled_door, {'input_samples': len(times), 'output_samples': len(target), 'resampled': resampled,
        'method': 'MuJoCo configuration-space differentiate/integrate for qpos; linear signed door-angle interpolation'}


def appearance(model, data, geom_ids):
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + model.nsite + 100)
    mujoco.mjv_updateScene(model, data, option, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
    visual = {int(g.objid): g for g in scene.geoms[:scene.ngeom] if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
    result = {}
    for i in geom_ids:
        g = visual[i]
        material_id = int(model.geom_matid[i])
        if material_id >= 0 and np.any(model.mat_texid[material_id] >= 0):
            raise ValueError('Textured robot material needs an explicit UV/texture adapter; refusing an incomplete export')
        roughness = float(model.mat_roughness[material_id]) if material_id >= 0 else -1.
        metallic = float(model.mat_metallic[material_id]) if material_id >= 0 else -1.
        result[i] = {'rgba': np.asarray(g.rgba).tolist(), 'shininess': float(g.shininess),
            'specular': float(g.specular), 'reflectance': float(g.reflectance),
            'roughness': roughness if roughness >= 0 else float((2 / (128 * g.shininess + 2)) ** .25),
            'metallic': max(0., metallic), 'material_id': material_id,
            'conversion': 'Native MuJoCo display RGBA; legacy shininess mapped to USD roughness when no PBR value is authored'}
    return result


def add_material(stage, name, values):
    path = '/World/G1/Looks/' + name
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*values['rgba'][:3]))
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(values['rgba'][3])
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(values['roughness'])
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(values['metallic'])
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    material.GetPrim().SetCustomDataByKey('mujocoAppearanceJson', json.dumps(values, sort_keys=True))
    return material


def add_geometry(stage, path, model, geom_id):
    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    record = {'geom_id': geom_id, 'type': kind, 'group': int(model.geom_group[geom_id]),
              'body_id': int(model.geom_bodyid[geom_id]), 'path': path}
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        na, nn = int(model.mesh_normaladr[mesh_id]), int(model.mesh_normalnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32)
        faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int32)
        assert len(vertices) and len(faces) and faces.min() >= 0 and faces.max() < len(vertices)
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, dtype=np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
        shape.CreateSubdivisionSchemeAttr('none')
        shape.CreateOrientationAttr('rightHanded')
        shape.CreateDoubleSidedAttr(False)
        if nn:
            normal_ids = np.asarray(model.mesh_facenormal[fa:fa + fn], dtype=np.int32)
            assert normal_ids.min() >= 0 and normal_ids.max() < nn
            normals = np.asarray(model.mesh_normal[na:na + nn], dtype=np.float32)[normal_ids.ravel()]
            shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            shape.SetNormalsInterpolation('faceVarying')
        shape.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0), vertices.max(0)], dtype=np.float32)))
        record.update(mesh_id=mesh_id, vertices=vn, triangles=fn,
            compiled_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
            compiled_faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest())
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(float(size[0])); shape.CreateHeightAttr(float(2 * size[1])); shape.CreateAxisAttr('Z')
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = UsdGeom.Sphere.Define(stage, path); shape.CreateRadiusAttr(float(size[0]))
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape = UsdGeom.Cube.Define(stage, path); shape.CreateSizeAttr(2.)
        # Primitive scale precedes the compiled geom rotation/translation below.
    else:
        raise ValueError(f'Unsupported explicit visual primitive type{kind}, geom{geom_id}; no substitute geometry emitted')
    local = matrix(model.geom_pos[geom_id], quat_matrix(model.geom_quat[geom_id]))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.diag([*size, 1.]) @ local
    UsdGeom.Xformable(shape).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(local))
    shape.GetPrim().SetCustomDataByKey('mujocoGeomId', geom_id)
    record['compiled_local_matrix'] = local.tolist()
    return shape, record


def validate_task_inputs(task, model, model_path, source, source_stage):
    """Bind the authored trajectory's declared identities to these exact inputs.

    These checks concern configuration identity and source joint conventions;
    they are separate from kinematic/contact acceptance of the trajectory.
    """
    checks = []

    def require(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})
        if not passed:
            raise ValueError('Task input binding failed: ' + name)

    robot = task['robot']
    scene = task['scene']
    require('task_model_path_matches_cli', Path(robot['model']).resolve(strict=True) == model_path)
    require('task_model_sha256_matches_cli', robot['sha256'] == sha(model_path))
    actual_files = {r['path']: r['sha256'] for r in model_inputs(model_path)}
    declared_files = {}
    duplicate_conflicts = []
    for record in robot['asset_files']:
        path = str(Path(record['path']).resolve(strict=True))
        digest = record['sha256']
        if path in declared_files and declared_files[path] != digest:
            duplicate_conflicts.append(path)
        declared_files[path] = digest
    require('task_model_asset_paths_and_hashes_match', not duplicate_conflicts and declared_files == actual_files,
            resolved_unique_model_files=len(actual_files), declared_model_records=len(robot['asset_files']),
            duplicate_conflicts=duplicate_conflicts)
    require('task_model_nq_matches_compiled', robot['nq'] == model.nq)
    names = [model.joint(i).name for i in range(model.njnt)
             if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    require('task_joint_names_match_compiled_qpos_order', robot['joint_names'] == names,
            compiled_joint_names=names)
    require('task_scene_path_matches_cli', Path(scene['path']).resolve(strict=True) == source)
    require('task_scene_sha256_matches_cli', scene['sha256'] == sha(source) == SCENE_SHA)
    door = task['door']
    prim = source_stage.GetPrimAtPath(door['joint_path'])
    require('task_door_joint_is_source_revolute', bool(prim) and prim.IsA(UsdPhysics.RevoluteJoint))
    joint = UsdPhysics.RevoluteJoint(prim)
    targets0, targets1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    require('task_door_is_source_joint_body1', len(targets0) == 1 and [str(p) for p in targets1] == [door['body_path']])
    body0 = source_stage.GetPrimAtPath(targets0[0])
    body1 = source_stage.GetPrimAtPath(door['body_path'])
    parent_world = np.asarray(UsdGeom.Xformable(body1.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    body0_world = np.asarray(UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    local_rotation = joint.GetLocalRot0Attr().Get()
    local_quat = [local_rotation.GetReal(), *local_rotation.GetImaginary()]
    frame0_parent = matrix(joint.GetLocalPos0Attr().Get(), quat_matrix(local_quat)) @ body0_world @ np.linalg.inv(parent_world)
    axis_token = joint.GetAxisAttr().Get()
    require('source_joint_axis_supported', axis_token in ('X', 'Y', 'Z'))
    source_axis = np.eye(3)[('X', 'Y', 'Z').index(axis_token)] @ frame0_parent[:3, :3]
    source_axis /= np.linalg.norm(source_axis)
    supplied_axis = np.asarray(door['axis_parent'], dtype=float)
    supplied_pivot = np.asarray(door['pivot_parent'], dtype=float)
    require('task_door_axis_matches_source_joint', supplied_axis.shape == (3,) and
            np.allclose(supplied_axis, source_axis, rtol=0, atol=1e-7), source_axis_parent=source_axis.tolist())
    require('task_door_pivot_matches_source_joint', supplied_pivot.shape == (3,) and
            np.allclose(supplied_pivot, frame0_parent[3, :3], rtol=0, atol=1e-7), source_pivot_parent=frame0_parent[3, :3].tolist())
    reference_attr = prim.GetAttribute('state:angular:physics:position')
    reference_degrees = reference_attr.Get() if reference_attr else None
    require('source_joint_reference_state_is_authored', reference_degrees is not None and math.isfinite(reference_degrees))
    reference = math.radians(reference_degrees)
    require('task_reference_angle_matches_source_joint_state', math.isfinite(float(door['reference_angle_radians'])) and
            abs(float(door['reference_angle_radians']) - reference) < 1e-7, source_reference_angle_radians=reference)
    for field, actual in [('source_local_transform', UsdGeom.Xformable(body1).GetLocalTransformation()),
                          ('source_world_transform', UsdGeom.Xformable(body1).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))]:
        if field in door:
            supplied = np.asarray(door[field], dtype=float)
            require('task_' + field + '_matches_source_column_convention', supplied.shape == (4, 4) and
                    np.allclose(supplied.T, np.asarray(actual), rtol=0, atol=1e-7))
    if 'limits_degrees' in door:
        supplied = np.asarray(door['limits_degrees'], dtype=float)
        limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
        require('task_door_limits_match_source_joint', supplied.shape == (2,) and
                np.allclose(supplied, limits, rtol=0, atol=1e-7), source_limits_degrees=limits)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--task-json', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--geom-groups', type=int, nargs='+', help='Explicit alternative to non-colliding visual-geom selection')
    args = parser.parse_args()
    source = args.scene.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    trajectory_path = args.trajectory.resolve(strict=True)
    task_path = args.task_json.resolve(strict=True)
    output = args.output_dir.resolve()
    if args.fps <= 0 or sha(source) != SCENE_SHA:
        raise ValueError('Invalid FPS or source scene SHA does not match frozen mesh09')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema': 'real2sim-g1-motion-usd/v1', 'status': 'starting',
        'produced_at': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': sha(__file__),
        'source_scene_sha256': SCENE_SHA, 'source_scene': str(source),
        'kinematic_preview': True, 'physics_simulated': False, 'robot_contact_validated': False,
        'metric_accuracy_verified': False, 'inputs': [file_record(p) for p in (model_path, trajectory_path, task_path)],
        'model_files': model_inputs(model_path), 'units': 'nominal meters, same source USD world frame', 'fps': args.fps}
    save(output / 'status.json', report)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    times, qpos, door_angles, sampling = trajectory_samples(model, trajectory_path, args.fps)
    task = json.loads(task_path.read_text())
    source_stage = Usd.Stage.Open(str(source))
    report['input_identity_checks'] = validate_task_inputs(task, model, model_path, source, source_stage)
    save(output / 'status.json', report)
    door = task['door']
    axis = np.asarray(door['axis_parent'], dtype=np.float64)
    pivot = np.asarray(door['pivot_parent'], dtype=np.float64)
    reference_angle = float(door['reference_angle_radians'])
    assert axis.shape == pivot.shape == (3,) and np.isfinite(axis).all() and np.isfinite(pivot).all() and np.linalg.norm(axis) > 1e-8
    source_stage, copied_source, environment = environment_copy(source, output)
    report['environment_files'] = environment
    source_door = source_stage.GetPrimAtPath(door['body_path'])
    assert source_door and source_door.HasAPI(UsdPhysics.RigidBodyAPI)
    source_local = np.asarray(UsdGeom.Xformable(source_door).GetLocalTransformation())
    if np.linalg.norm(source_local[3, :3] - pivot) > 1e-5:
        raise ValueError('Selected source door body origin must match the supplied parent-local pivot')
    parent_world = np.asarray(UsdGeom.Xformable(source_door.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world = np.asarray(UsdGeom.Xformable(source_stage.GetPrimAtPath('/World')).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world_inverse = np.linalg.inv(world)
    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0 and (
        int(model.geom_group[i]) in args.geom_groups if args.geom_groups is not None else model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0)]
    if not geom_ids:
        raise ValueError('No explicit visual robot geoms found; select --geom-groups rather than guessing')
    data.qpos[:] = qpos[0]; mujoco.mj_forward(model, data)
    looks = appearance(model, data, geom_ids)
    geom_ids = [i for i in geom_ids if looks[i]['rgba'][3] > 0]
    robot_path = output / 'g1_motion.usdc'
    robot = Usd.Stage.CreateNew(str(robot_path))
    UsdGeom.SetStageUpAxis(robot, 'Z'); UsdGeom.SetStageMetersPerUnit(robot, 1.)
    robot.SetTimeCodesPerSecond(args.fps); robot.SetFramesPerSecond(args.fps)
    robot.SetStartTimeCode(0); robot.SetEndTimeCode(len(times) - 1)
    UsdGeom.Xform.Define(robot, '/World')
    robot_root = UsdGeom.Xform.Define(robot, '/World/G1')
    robot_root.GetPrim().SetCustomDataByKey('previewType', 'kinematic FK samples; no force/contact simulation')
    robot_root.GetPrim().SetCustomDataByKey('sourceModelSha256', sha(model_path))
    materials = {}
    bodies, body_ops = [], {}
    for body_id in range(1, model.nbody):
        name = f'b{body_id:03d}_' + (model.body(body_id).name or 'unnamed')
        name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        path = '/World/G1/Bodies/' + name
        prim = UsdGeom.Xform.Define(robot, path)
        prim.GetPrim().SetCustomDataByKey('mujocoBodyId', body_id)
        prim.GetPrim().SetCustomDataByKey('mujocoParentBodyId', int(model.body_parentid[body_id]))
        body_ops[body_id] = prim.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        bodies.append({'id': body_id, 'name': model.body(body_id).name, 'path': path, 'parent_id': int(model.body_parentid[body_id])})
    body_paths = {b['id']: b['path'] for b in bodies}
    geoms = []
    for geom_id in geom_ids:
        values = looks[geom_id]
        key = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]
        if key not in materials:
            materials[key] = add_material(robot, 'material_' + key, values)
        path = body_paths[int(model.geom_bodyid[geom_id])] + f'/geom_{geom_id:03d}'
        shape, record = add_geometry(robot, path, model, geom_id)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(materials[key])
        record['appearance'] = values
        geoms.append(record)
    body_world = np.empty((len(times), len(bodies), 4, 4), dtype=np.float64)
    geom_fk_error = 0.
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for index, body in enumerate(bodies):
            bw = matrix(data.xpos[body['id']], data.xmat[body['id']])
            body_world[frame, index] = bw
            local = bw @ world_inverse
            body_ops[body['id']].Set(Gf.Matrix4d(local), float(frame))
            if frame == 0:
                body_ops[body['id']].Set(Gf.Matrix4d(local))
        for geom in geoms:
            if geom['type'] == mujoco.mjtGeom.mjGEOM_BOX:
                continue
            actual = matrix(data.geom_xpos[geom['geom_id']], data.geom_xmat[geom['geom_id']])
            predicted = np.asarray(geom['compiled_local_matrix']) @ body_world[frame, geom['body_id'] - 1]
            geom_fk_error = max(geom_fk_error, float(np.max(np.abs(actual - predicted))))
    assert geom_fk_error < 1e-10, geom_fk_error
    robot.GetRootLayer().Save()
    overlay_path = output / 'preview_overlay.usda'
    overlay = Usd.Stage.CreateNew(str(overlay_path))
    # USD scales time samples across sublayers with different time-code rates.
    # The motion and robot layers must declare the same30-Hz rate as their root.
    overlay.SetTimeCodesPerSecond(args.fps); overlay.SetFramesPerSecond(args.fps)
    overlay.SetStartTimeCode(0); overlay.SetEndTimeCode(len(times) - 1)
    disabled = {'rigid_bodies': [], 'collisions': [], 'joints': []}
    for prim in source_stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(overlay.OverridePrim(path)).CreateRigidBodyEnabledAttr(False)
            disabled['rigid_bodies'].append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(overlay.OverridePrim(path)).CreateCollisionEnabledAttr(False)
            disabled['collisions'].append(path)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(overlay.OverridePrim(path)).CreateJointEnabledAttr(False)
            disabled['joints'].append(path)
    door_prim = overlay.OverridePrim(door['body_path'])
    xf = UsdGeom.Xformable(door_prim)
    xf.ClearXformOpOrder()
    door_op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'kinematicPreview')
    door_op.Set(Gf.Matrix4d(source_local))  # Source default pose is unchanged.
    door_world = []
    for frame, angle in enumerate(door_angles):
        local = source_local @ about_pivot(axis, pivot, float(angle - reference_angle))
        door_op.Set(Gf.Matrix4d(local), float(frame))
        door_world.append(local @ parent_world)
    # Cameras occupy observed source-camera positions but use task-facing optics.
    root_positions = body_world[:, 0, 3, :3]
    hinge_world = np.r_[pivot, 1.] @ parent_world
    route_center = (root_positions.min(0) + root_positions.max(0)) / 2
    overview_target = .55 * route_center + .45 * hinge_world[:3]
    overview_target[2] = .9
    task_target = .7 * hinge_world[:3] + .3 * root_positions[-1]
    task_target[2] = 1.05
    camera_specs = task.get('preview_cameras')
    if camera_specs is None:
        camera_specs = []
        for name, source_name, target, focal in [('Overview', 'frame_001499', overview_target, 20.), ('Task', 'frame_000906', task_target, 22.)]:
            source_camera = source_stage.GetPrimAtPath('/World/Cameras/' + source_name)
            eye = np.asarray(UsdGeom.Xformable(source_camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))[3, :3]
            camera_specs.append({'name': name, 'position': eye.tolist(), 'target': target.tolist(), 'focal_length': focal, 'position_evidence_source_camera': source_name})
    cameras = []
    for item in camera_specs:
        path = '/World/PreviewCameras/' + item['name']
        cam = UsdGeom.Camera.Define(overlay, path)
        cam.CreateFocalLengthAttr(float(item.get('focal_length', 22.)))
        cam.CreateHorizontalApertureAttr(36.)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.03, 100.))
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*item['position']), Gf.Vec3d(*item['target']), Gf.Vec3d(0, 0, 1)).GetInverse()
        cam.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(np.asarray(view) @ world_inverse))
        cam.GetPrim().SetCustomDataByKey('previewCamera', True)
        cameras.append(item | {'path': path, 'world_matrix': np.asarray(view).tolist()})
    overlay.GetRootLayer().customLayerData = {'previewType': 'Kinematic FK playback; all source physics disabled in overlay only'}
    overlay.GetRootLayer().Save()
    combined_path = output / 'scene.usda'
    combined = Usd.Stage.CreateNew(str(combined_path))
    combined.GetRootLayer().subLayerPaths = [overlay_path.name, robot_path.name, str(copied_source.relative_to(output))]
    combined.SetDefaultPrim(combined.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(combined, 'Z'); UsdGeom.SetStageMetersPerUnit(combined, 1.)
    combined.SetTimeCodesPerSecond(args.fps); combined.SetFramesPerSecond(args.fps)
    combined.SetStartTimeCode(0); combined.SetEndTimeCode(len(times) - 1)
    render_settings = dict(source_stage.GetRootLayer().customLayerData.get('renderSettings', {}))
    render_settings['rtx:rendermode'] = 'RaytracedLighting'
    combined.GetRootLayer().customLayerData = {'previewType': 'Kinematic G1 and refrigerator motion; no dynamics or contact claims',
        'sourceSceneSha256': SCENE_SHA, 'renderSettings': render_settings}
    combined.GetRootLayer().Save()
    np.savez_compressed(output / 'fk_samples.npz', time=times, qpos=qpos, door_angle=door_angles,
                        body_world=body_world, door_world=np.asarray(door_world), body_ids=np.array([b['id'] for b in bodies]))
    checks = []
    def check(name, value, **detail):
        checks.append({'name': name, 'passed': bool(value), **detail})
    reopened = Usd.Stage.Open(str(combined_path))
    max_body_error = max_door_error = 0.
    for frame in range(len(times)):
        cache = UsdGeom.XformCache(float(frame))
        for index, body in enumerate(bodies):
            value = np.asarray(cache.GetLocalToWorldTransform(reopened.GetPrimAtPath(body['path'])))
            max_body_error = max(max_body_error, float(np.max(np.abs(value - body_world[frame, index]))))
        value = np.asarray(cache.GetLocalToWorldTransform(reopened.GetPrimAtPath(door['body_path'])))
        max_door_error = max(max_door_error, float(np.max(np.abs(value - door_world[frame]))))
    check('all_authored_body_fk_samples_match', max_body_error < 1e-10, max_abs_error=max_body_error)
    check('all_authored_door_samples_match', max_door_error < 1e-10, max_abs_error=max_door_error)
    check('compiled_geom_transform_matches_mujoco', geom_fk_error < 1e-10, max_abs_error=geom_fk_error)
    for geom in geoms:
        if geom['type'] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = UsdGeom.Mesh(reopened.GetPrimAtPath(geom['path']))
        v = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        check('compiled_mesh_' + str(geom['geom_id']) + '_arrays_preserved', hashlib.sha256(v.tobytes()).hexdigest() == geom['compiled_vertices_sha256'] and hashlib.sha256(f.tobytes()).hexdigest() == geom['compiled_faces_sha256'])
    visual_error = 0.
    original_cache, new_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    for prim in source_stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            value = reopened.GetPrimAtPath(prim.GetPath())
            visual_error = max(visual_error, float(np.max(np.abs(np.asarray(original_cache.GetLocalToWorldTransform(prim)) - np.asarray(new_cache.GetLocalToWorldTransform(value))))))
    check('all_source_default_visual_transforms_preserved', visual_error < 1e-10, max_abs_error=visual_error)
    check('source_file_and_texture_copies_exact', all(sha(x['source']['path']) == x['source']['sha256'] == sha(x['copy']['path']) for x in environment))
    check('all_input_files_unchanged', all(sha(x['path']) == x['sha256'] for x in report['inputs'] + report['model_files']))
    check('all_source_physics_disabled_only_in_preview', all(not UsdPhysics.RigidBodyAPI(reopened.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Get() for p in disabled['rigid_bodies']) and all(not UsdPhysics.Joint(reopened.GetPrimAtPath(p)).GetJointEnabledAttr().Get() for p in disabled['joints']) and all(not UsdPhysics.CollisionAPI(reopened.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get() for p in disabled['collisions']))
    report.update(status='passed' if all(c['passed'] for c in checks) else 'failed', elapsed_seconds=time.monotonic() - started,
        model={'nq': model.nq, 'nv': model.nv, 'nbody': model.nbody, 'ngeom': model.ngeom, 'nmesh': model.nmesh, 'mujoco': mujoco.__version__},
        sampling=sampling, duration_seconds=float(times[-1]), sample_count=len(times),
        selection={'method': 'explicit groups' if args.geom_groups is not None else 'non-colliding visual geoms', 'groups_override': args.geom_groups, 'geom_ids': geom_ids, 'excluded_geom_ids': [i for i in range(model.ngeom) if i not in geom_ids], 'world_floor_exported': False},
        bodies=bodies, geoms=geoms, cameras=cameras, door=door | {'source_local_matrix': source_local.tolist(), 'parent_world_matrix': parent_world.tolist()},
        disabled_source_physics=disabled, checks=checks,
        limitations=['Kinematic configuration playback; no control, balance, force, grasp or collision-validity claim.',
            'Body FK is exact at authored samples; interpolation between samples is USD visual interpolation.',
            'Nominal environment scale is inherited; no rescale or independent metric validation.',
            'MuJoCo material display colors are preserved; legacy shading conversion is documented per material.',
            'MuJoCo world geometry and collision-only robot proxies are excluded from the default visual export.'])
    report['artifacts'] = [file_record(p) for p in (combined_path, robot_path, overlay_path, output / 'fk_samples.npz')]
    save(output / 'export.receipt.json', report); save(output / 'status.json', report)
    print(json.dumps({'status': report['status'], 'output': str(combined_path), 'receipt_sha256': sha(output / 'export.receipt.json'), 'bodies': len(bodies), 'visual_geoms': len(geoms), 'samples': len(times), 'checks': len(checks)}), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
