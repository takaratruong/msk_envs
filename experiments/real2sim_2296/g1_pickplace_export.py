#!/usr/bin/env python3
"""Portable USD pick/place reference using compiled robot and task-object meshes.

The existing exporter is imported unchanged. Its fixed-scene identity is set to
the explicitly receipt-bound scene revision for this new task; all of its model,
door, FK, material and source-copy checks still execute. No physics is run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import g1_motion_usd as inherited
from g1_motion_usd import checked_input, file_record, matrix, quat_matrix, save, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--adapter-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    task=json.loads(args.task.read_text());adapter=json.loads(args.adapter_receipt.read_text())
    adapter_task_path=Path(adapter['task']['path'])
    if adapter['task']['sha256']!=sha(adapter_task_path):raise ValueError('Adapter task identity changed')
    adapter_task=json.loads(adapter_task_path.read_text())
    # Scene adapters bind task geometry, while new references may have different
    # timelines/origin metadata. Require exact geometry-bearing fields and bind
    # both source task hashes rather than claiming their entire files match.
    for field in ['robot','door','grasp','object','place']:
        if adapter_task.get(field)!=task.get(field):raise ValueError('Adapter and reference task geometry differ: '+field)
    source=Path(adapter['scene']['path']);object_model=Path(adapter['output']['path'])
    if task['scene']['sha256']!=adapter['scene']['sha256'] or Path(task['scene']['path']).resolve()!=source.resolve():
        raise ValueError('Reference and native adapter select different source scenes')
    if sha(source)!=adapter['scene']['sha256'] or sha(object_model)!=adapter['output']['sha256']:
        raise ValueError('Adapter source/compiled-input identity differs')
    for row in adapter['scene_layers']:
        if sha(row['path'])!=row['sha256']:raise ValueError('A scene dependency changed')
    z=np.load(args.trajectory,allow_pickle=False)
    if str(z['object_pose_origin'])!='center' or task.get('object_pose_origin')!='center':
        raise ValueError('Object poses must use the explicit body-center convention')
    times=z['time'];poses=z['object_pose'];fps=float(z['fps'])
    if fps not in (30,50) or poses.shape!=(len(times),7) or not np.isfinite(poses).all():
        raise ValueError('Expected finite 30 or 50 Hz center-origin object poses')
    if np.max(abs(np.linalg.norm(poses[:,3:],axis=1)-1))>1e-6:raise ValueError('Non-unit object quaternion')
    sources=output/'inputs';sources.mkdir()
    for name,path in [('original_task.json',args.task),('trajectory.npz',args.trajectory),('adapter.receipt.json',args.adapter_receipt)]:
        shutil.copyfile(path,sources/name)
    helpers=output/'helpers';helpers.mkdir()
    for path in [Path(__file__),Path(inherited.__file__),Path(inherited.__file__).with_name('g1_motion_plan.py')]:
        shutil.copyfile(path,helpers/path.name)
    # A complete independent robot XML/STL tree makes later editing inexpensive.
    original_model=Path(task['robot']['model']);files=[Path(r['path']) for r in task['robot']['asset_files']]
    common=Path(os.path.commonpath([str(f.parent) for f in files]));copies=[]
    for file in files:
        checked_input(file);dst=output/'robot_model'/file.relative_to(common)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dst)
        copies.append(dict(source=file_record(file),copy=file_record(dst)))
    render_task=json.loads(json.dumps(task));render_task['reference_planning_scene']=task['scene']
    render_task['scene']=dict(path=str(source.resolve()),sha256=sha(source))
    render_task['preview_cameras']=[
        dict(name='Fridge',position=[.1032484162,-2.3641632471,1.6931715491],target=[-2.2361969979,.0643031626,1.05],focal_length=22.),
        dict(name='Route',position=[2.2,-3.6,2.8],target=[.35,-1.4,.75],focal_length=18.),
        dict(name='Table',position=[1.55,-3.3,1.85],target=[3.35,-1.86,.85],focal_length=25.),
    ]
    task_path=sources/'render_task.json';save(task_path,render_task)
    # Runtime specialization, explicitly bound above, rather than modifying the
    # frozen prior helper or any existing USD. Its remaining checks are intact.
    inherited.SCENE_SHA=sha(source)
    oldargv=sys.argv[:]
    try:
        sys.argv=[inherited.__file__,'--model',str(original_model),'--trajectory',str(args.trajectory.resolve()),
                  '--task-json',str(task_path),'--scene',str(source),'--output-dir',str(output/'motion'),'--fps',str(int(fps))]
        inherited.main()
    finally:sys.argv=oldargv
    motion=output/'motion';stage=Usd.Stage.Open(str(motion/'scene.usda'))
    if not stage:raise RuntimeError('Exported motion stage missing')
    model=mujoco.MjModel.from_xml_path(str(object_model));data=mujoco.MjData(model)
    body=model.body('task_bottle').id
    geom_ids=np.flatnonzero(model.geom_bodyid==body).tolist()
    expected_parts=1 if task['object'].get('shape')=='cylinder' else 3
    if len(geom_ids)!=expected_parts:raise ValueError('Task prop geometry count differs from declared shape')
    if expected_parts==1:
        g=geom_ids[0]
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_CYLINDER or not np.allclose(model.geom_size[g,:2],[task['object']['radius'],task['object']['height']/2],rtol=0,atol=1e-10):
            raise ValueError('Native cylinder geometry differs from task dimensions')
    initial=np.array(task['object']['initial_base_world'])+[0,0,task['object']['height']/2]
    if np.max(abs(model.body_pos[body]-initial))>1e-8:raise ValueError('Object initial pose differs from task')
    mujoco.mj_forward(model,data)
    object_layer=output/'task_object.usdc';bottle_stage=Usd.Stage.CreateNew(str(object_layer))
    UsdGeom.SetStageMetersPerUnit(bottle_stage,1.);UsdGeom.SetStageUpAxis(bottle_stage,'Z')
    bottle_stage.SetTimeCodesPerSecond(fps);bottle_stage.SetFramesPerSecond(fps)
    bottle_stage.SetStartTimeCode(0);bottle_stage.SetEndTimeCode(len(times)-1)
    root=UsdGeom.Xform.Define(bottle_stage,'/World/TaskBottle');op=root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    root.GetPrim().SetCustomDataByKey('provenance',task['object']['provenance'])
    root.GetPrim().SetCustomDataByKey('motionScope','Sampled body poses; no physics integration or physical-success certification by this exporter')
    root.GetPrim().SetCustomDataByKey('poseOrigin','geometric body center; world XYZ + WXYZ quaternion')
    mat=UsdShade.Material.Define(bottle_stage,'/World/TaskBottle/Looks/BlueBottle')
    shader=UsdShade.Shader.Define(bottle_stage,str(mat.GetPath())+'/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*model.geom_rgba[geom_ids[0],:3].astype(float)))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.32)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    geometries=[]
    for g in geom_ids:
        shape,record=inherited.add_geometry(bottle_stage,'/World/TaskBottle/'+model.geom(g).name,model,g)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat);geometries.append(record)
    for frame,pose in enumerate(poses):
        transform=Gf.Matrix4d(matrix(pose[:3],quat_matrix(pose[3:])))
        op.Set(transform,float(frame))
        if frame==0:op.Set(transform)
    bottle_stage.GetRootLayer().Save()
    combined=Usd.Stage.CreateNew(str(output/'scene.usda'))
    combined.GetRootLayer().subLayerPaths=[object_layer.name,'motion/scene.usda']
    combined.SetDefaultPrim(combined.GetPrimAtPath('/World'))
    UsdGeom.SetStageMetersPerUnit(combined,1.);UsdGeom.SetStageUpAxis(combined,'Z')
    combined.SetTimeCodesPerSecond(fps);combined.SetFramesPerSecond(fps)
    combined.SetStartTimeCode(0);combined.SetEndTimeCode(len(times)-1)
    combined.GetRootLayer().customLayerData=dict(stage.GetRootLayer().customLayerData)|{
        'previewType':'G1/Dex3 sampled-motion playback. No dynamic balance, grasp, object support or force success claim by this exporter.'}
    combined.GetRootLayer().Save()
    maximum=0.
    for frame,pose in enumerate(poses):
        actual=np.asarray(UsdGeom.XformCache(float(frame)).GetLocalToWorldTransform(combined.GetPrimAtPath('/World/TaskBottle')))
        maximum=max(maximum,float(np.max(abs(actual-matrix(pose[:3],quat_matrix(pose[3:]))))))
    if maximum>1e-10:raise AssertionError('Composed object transforms disagree')
    # Existing renderer reads these next to its scene path. Preserve the exact
    # unchanged FK evidence and exporter receipt; the additive receipt below
    # binds object geometry, source revision and the combined wrapper stage.
    for name in ['fk_samples.npz','export.receipt.json']:
        shutil.copyfile(motion/name,output/name)
    report=dict(schema='g1-pickplace-usd-reference/v1',status='passed',physics_executed=False,
        source_scene=adapter['scene'],reference_planning_scene=task['scene'],
        inputs=[file_record(p) for p in [args.task,args.trajectory,args.adapter_receipt,adapter_task_path,object_model]],
        helpers=[file_record(p) for p in sorted(helpers.iterdir())],robot_editable_files=copies,
        object_geometries=geometries,object_body_origin='center',samples=len(times),duration_seconds=float(times[-1]),
        object_all_frame_transform_max_error=maximum,
        checks=dict(adapter_task_file_identity=True,adapter_reference_geometry_fields_equal=True,adapter_scene_layers=True,object_initial_pose=True,
                    exact_object_compiled_shapes=True,all_object_time_samples_match=True),
        inherited_export_receipt=file_record(motion/'export.receipt.json'),
        artifacts=[file_record(p) for p in [output/'scene.usda',object_layer,output/'fk_samples.npz']],
        limitations=['Reference playback only; object attachment and released support do not establish physical success.',
                     'All task-prop geometry is compiled from the task adapter; visual roughness .32 is authored.',
                     'The opening prefix retains the shared world frame; the selected floor-plan revision and added motion require independent contact review.'])
    save(output/'pickplace_export.receipt.json',report)
    print(json.dumps(dict(status='passed',output=str(output/'scene.usda'),receipt_sha256=sha(output/'pickplace_export.receipt.json'))),flush=True)


if __name__=='__main__':main()
