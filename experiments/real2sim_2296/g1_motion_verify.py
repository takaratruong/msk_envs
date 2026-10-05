#!/usr/bin/env python3
"""Independent FK and triangle-collision measurements of a saved reference.

No trajectory is changed by verification. Collision checks use visible robot
meshes, source scene collider parts and the complete planned door transform.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(r,p):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t


def check_source_geometry(task,collision_dir):
    scene=Path(task['scene']['path'])
    assert sha(scene)==task['scene']['sha256']=='f78a48e358080b0079c9d176beea9286ef8e9eb2e4ff827eb90ca686f2647fbc'
    stage=Usd.Stage.Open(str(scene));assert UsdGeom.GetStageUpAxis(stage)=='Z'
    assert UsdGeom.GetStageMetersPerUnit(stage)==1.
    index=json.loads((collision_dir/'collision.json').read_text())
    with np.load(collision_dir/'collision.npz') as packed:
        gi={key:packed[key] for key in packed.files}
    assert index['scene_sha256']==task['scene']['sha256']
    assert index['geometry_sha256']==sha(collision_dir/'collision.npz')
    prims=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
           and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    assert len(prims)==len(index['parts'])==3675
    assert {str(p.GetPath()) for p in prims}=={e['path'] for e in index['parts']}
    entries={e['path']:e for e in index['parts']};cache=UsdGeom.XformCache();max_error=0.
    for prim in prims:
        entry=entries[str(prim.GetPath())];mesh=UsdGeom.Mesh(prim)
        path=str(prim.GetPath());expected_asset=path.split('/')[3] if path.startswith('/World/Assets/') else path.split('/')[2]
        assert entry['asset']==expected_asset
        assert entry['selected_door']==path.startswith(task['door']['body_path']+'/')
        expected_approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else 'none'
        assert entry['approximation']==expected_approx
        source_v=np.asarray(mesh.GetPointsAttr().Get());mat=np.asarray(cache.GetLocalToWorldTransform(prim))
        source_v=np.c_[source_v,np.ones(len(source_v))]@mat
        va=entry['vertex_start'];vn=entry['vertex_count'];fa=entry['face_start'];fn=entry['face_count']
        actual_v=gi['vertices'][va:va+vn];assert source_v[:,:3].shape==actual_v.shape
        error=float(np.max(np.abs(source_v[:,:3]-actual_v)));assert error<1e-6;max_error=max(max_error,error)
        ind=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=mesh.GetFaceVertexCountsAttr().Get();faces=[];off=0
        for n in counts:
            polygon=ind[off:off+n];faces.extend((polygon[0],polygon[k],polygon[k+1]) for k in range(1,n-1));off+=n
        assert np.array_equal(np.asarray(faces),gi['faces'][fa:fa+fn]-va)
        assert np.max(np.abs(np.asarray(entry['bounds'])-np.array([source_v[:,:3].min(0),source_v[:,:3].max(0)])))<1e-7
    joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(task['door']['joint_path']))
    assert str(joint.GetBody1Rel().GetTargets()[0])==task['door']['body_path']
    assert joint.GetAxisAttr().Get()=='Z'
    base=stage.GetPrimAtPath(str(joint.GetBody0Rel().GetTargets()[0]));mat=np.asarray(cache.GetLocalToWorldTransform(base)).T
    pivot=mat[:3,:3]@np.asarray(joint.GetLocalPos0Attr().Get())+mat[:3,3]
    assert np.max(np.abs(pivot-np.asarray(task['door']['world_pivot'])))<1e-6
    handle=stage.GetPrimAtPath(task['grasp']['rail_source_path']);hp=np.asarray(UsdGeom.Mesh(handle).GetPointsAttr().Get())
    hm=np.asarray(cache.GetLocalToWorldTransform(handle));hp=np.c_[hp,np.ones(len(hp))]@hm
    center=(hp[:,:3].min(0)+hp[:,:3].max(0))/2
    assert np.max(np.abs(center[:2]-np.asarray(task['grasp']['rail_point_world'])[:2]))<1e-6
    # The selected grip height must lie on the central straight rail section.
    assert .93<=task['grasp']['rail_point_world'][2]<=1.41
    return gi,index,dict(source_collision_parts=len(prims),max_source_vertex_error_m=max_error,
                         door_limits_rad=np.deg2rad([joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]).tolist(),
                         collision_index_sha256=sha(collision_dir/'collision.json'),collision_geometry_sha256=sha(collision_dir/'collision.npz'))


def visual_geometries(m):
    result={}
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g])==0 or m.geom_contype[g] or m.geom_conaffinity[g]:continue
        kind=int(m.geom_type[g])
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
            b=int(m.mesh_faceadr[mid]);k=int(m.mesh_facenum[mid])
            mesh=trimesh.Trimesh(vertices=m.mesh_vert[a:a+n].copy(),faces=m.mesh_face[b:b+k].copy(),process=False)
        elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):mesh=trimesh.creation.cylinder(radius=m.geom_size[g,0],height=2*m.geom_size[g,1])
        else:raise ValueError('Unsupported visible robot geometry '+str(kind))
        result[g]=mesh
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True);p.add_argument('--collision-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=1)
    p.add_argument('--skip-collisions',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    task=json.loads(args.task.read_text());model=Path(task['robot']['model'])
    if sha(model)!=task['robot']['sha256']:raise ValueError('Robot model changed')
    xml=ET.parse(model).getroot();compiler=xml.find('compiler');meshdir=compiler.get('meshdir','') if compiler is not None else ''
    assert not xml.findall('.//include'),'Included model assets need explicit dependency validation'
    required={model.resolve()}|{(model.parent/meshdir/n.attrib['file']).resolve() for n in xml.findall('.//asset/mesh') if 'file' in n.attrib}
    bound=task['robot']['asset_files'];assert {Path(x['path']).resolve() for x in bound}==required
    for x in bound:assert sha(x['path'])==x['sha256'],x['path']
    gi,index,source_check=check_source_geometry(task,args.collision_dir)
    m=mujoco.MjModel.from_xml_path(str(model));d=mujoco.MjData(m)
    z=np.load(args.trajectory,allow_pickle=False);q=z['qpos'];tt=z['time'];phase=z['phase']
    if q.shape[1]!=m.nq:raise ValueError('Wrong qpos width')
    if task['robot']['joint_names']!=[m.joint(j).name for j in range(1,m.njnt)]:raise ValueError('Joint order mismatch')
    geoms=visual_geometries(m);foot_ids=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    wrist=m.body('right_wrist_yaw_link').id
    lower=m.jnt_range[1:,0];upper=m.jnt_range[1:,1]
    violation=np.maximum(np.maximum(lower-q[:,7:],q[:,7:]-upper),0)
    foot_error=[];foot_height=[];wrist_error=[];rot_error=[];com_margin=[];bodies=[];bodyrot=[]
    actual_feet=[];task_grip_error=[];task_orientation_error=[]
    collision_events=[];sampled=[]
    environment=robot_manager=None;moving=[];names={}
    if not args.skip_collisions:
        environment=trimesh.collision.CollisionManager();robot_manager=trimesh.collision.CollisionManager()
        lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
        boxes={g:np.array([[x,y,z] for x in mesh.bounds[:,0] for y in mesh.bounds[:,1] for z in mesh.bounds[:,2]]) for g,mesh in geoms.items()}
        for row in q:
            d.qpos[:]=row;mujoco.mj_forward(m,d)
            for g,box in boxes.items():
                world=box@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                lo=np.minimum(lo,world.min(0));hi=np.maximum(hi,world.max(0))
        lo-=.02;hi+=.02
        for entry in index['parts']:
            bb=np.asarray(entry['bounds']);name=entry['path']
            # Ground contact is measured against the visual soles separately.
            if entry['asset']=='floor_surface_region_01':continue
            if not entry['selected_door'] and (np.any(bb[0]>hi) or np.any(bb[1]<lo)):continue
            va=entry['vertex_start'];vn=entry['vertex_count'];fa=entry['face_start'];fn=entry['face_count']
            mesh=trimesh.Trimesh(gi['vertices'][va:va+vn],gi['faces'][fa:fa+fn]-va,process=False)
            if entry['approximation']=='convexHull':mesh=mesh.convex_hull
            environment.add_object(name,mesh)
            if entry['selected_door']:moving.append(name)
        for g,mesh in geoms.items():
            name=f'{m.body(int(m.geom_bodyid[g])).name}/geom_{g}'
            robot_manager.add_object(name,mesh);names[g]=name
    for i,row in enumerate(q):
        d.qpos[:]=row;mujoco.mj_forward(m,d)
        bodies.append(d.xpos.copy());bodyrot.append(d.xquat.copy())
        actual_feet.append(d.xpos[foot_ids].copy())
        foot_error.append(np.linalg.norm(d.xpos[foot_ids]-z['foot_target'][i],axis=1))
        corners=[];heights=[]
        for foot in foot_ids:
            points=[]
            for g,mesh in geoms.items():
                if m.geom_bodyid[g]==foot:
                    points.append(mesh.vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            v=np.concatenate(points);heights.append(float(v[:,2].min()));corners.append(v[:,:2])
        foot_height.append(heights)
        active=np.flatnonzero(z['foot_contact'][i])
        if len(active):
            hull=ConvexHull(np.concatenate([corners[k] for k in active]));com=d.subtree_com[1,:2]
            margin=-(hull.equations[:,:2]@com+hull.equations[:,2]).max();com_margin.append(float(margin))
        else:com_margin.append(float('nan'))
        target=z['wrist_target'][i];targetr=z['wrist_rotation_target'][i]
        wrist_error.append(float(np.linalg.norm(d.xpos[wrist]-target)) if np.isfinite(target).all() else float('nan'))
        rot_error.append(float(np.linalg.norm(Rotation.from_matrix(targetr@d.xmat[wrist].reshape(3,3).T).as_rotvec())) if np.isfinite(targetr).all() else float('nan'))
        door_rotation=Rotation.from_euler('z',z['door_angle'][i]).as_matrix()
        world_pivot=np.asarray(task['door']['world_pivot']);rail0=np.asarray(task['grasp']['rail_point_world'])
        rail_expected=world_pivot+door_rotation@(rail0-world_pivot)
        wrist_r=d.xmat[wrist].reshape(3,3)
        grip_actual=d.xpos[wrist]+wrist_r@np.asarray(task['grasp']['grip_point_wrist'])
        task_grip_error.append(float(np.linalg.norm(grip_actual-rail_expected)))
        desired_r=door_rotation@np.asarray(task['grasp']['wrist_closed_rotation'])
        task_orientation_error.append(float(np.linalg.norm(Rotation.from_matrix(desired_r@wrist_r.T).as_rotvec())))
        if environment is not None and (i%args.stride==0 or i==len(q)-1):
            sampled.append(i)
            for g in geoms:robot_manager.set_transform(names[g],transform(d.geom_xmat[g].reshape(3,3),d.geom_xpos[g]))
            rr=Rotation.from_euler('z',z['door_angle'][i]).as_matrix();pivot=np.asarray(task['door']['world_pivot'])
            dt=transform(rr,pivot-rr@pivot)
            for name in moving:environment.set_transform(name,dt)
            hit,pairs,data=environment.in_collision_other(robot_manager,return_names=True,return_data=True)
            if hit:
                records=[]
                for x in data:
                    records.append(dict(names=sorted(x.names),depth_m=float(x.depth),point=np.asarray(x.point).tolist()))
                records.sort(key=lambda x:x['depth_m'],reverse=True)
                collision_events.append(dict(frame=i,phase=str(phase[i]),max_depth_m=max([x['depth_m'] for x in records] or [0.]),contacts=records[:12]))
            if len(sampled)%45==0:print(json.dumps(dict(event='collision_progress',frame=i,total=len(q),events=len(collision_events))),flush=True)
    foot_error=np.asarray(foot_error);foot_height=np.asarray(foot_height);wrist_error=np.asarray(wrist_error);rot_error=np.asarray(rot_error)
    active=z['foot_contact'].astype(bool)
    required_grasp=np.isin(phase,['pull','hold_open','finish'])
    pull=required_grasp | z['grasp_active'].astype(bool)
    actual_feet=np.asarray(actual_feet);plant_drift=[]
    for side in range(2):
        start=None
        for i,value in enumerate(np.r_[active[:,side],False]):
            if value and start is None:start=i
            if not value and start is not None:
                points=actual_feet[start:i,side];anchor=np.median(points,axis=0)
                plant_drift.extend(np.linalg.norm(points-anchor,axis=1).tolist());start=None
    root_r=Rotation.from_quat(q[:,[4,5,6,3]])
    root_rot_step=(root_r[1:]*root_r[:-1].inv()).magnitude()
    required_stand=np.isin(phase,['stand','reach','close_hand','pull','hold_open','finish'])
    support_schedule_ok=bool(active[required_stand].all())
    compressed=[str(phase[0])]+[str(phase[i]) for i in range(1,len(phase)) if phase[i]!=phase[i-1]]
    required_phases=['ready','walk','turn_and_settle','stand','reach','close_hand','pull','hold_open','finish']
    phase_order_ok=compressed==required_phases
    walk_ids=np.flatnonzero(phase=='walk')
    walk_distance=float(np.linalg.norm(q[walk_ids[-1],:2]-q[walk_ids[0],:2])) if len(walk_ids)>1 else 0.
    finger_error=0.
    finger_names={f'right_hand_{finger}_{j}_joint' for finger,js in [('thumb',range(3)),('index',range(2)),('middle',range(2))] for j in js}
    assert set(task['grasp']['finger_joint_targets'])==finger_names
    assert np.isfinite(list(task['grasp']['finger_joint_targets'].values())).all()
    for name,value in task['grasp']['finger_joint_targets'].items():
        adr=m.jnt_qposadr[m.joint(name).id];finger_error=max(finger_error,float(np.max(np.abs(q[pull,adr]-value))))
    boundaries=np.flatnonzero(phase[1:]!=phase[:-1])+1
    join=[]
    for i in boundaries:
        join.append(dict(frame=int(i),before=str(phase[i-1]),after=str(phase[i]),root_step_m=float(np.linalg.norm(q[i,:3]-q[i-1,:3])),joint_step_rad=float(np.max(np.abs(q[i,7:]-q[i-1,7:])))))
    max_depth=max([x['max_depth_m'] for x in collision_events] or [0.])
    metrics=dict(frames=len(q),duration_seconds=float(tt[-1]-tt[0]),max_joint_limit_violation_rad=float(violation.max()),
                 max_planted_foot_position_error_m=float(foot_error[active].max()),
                 max_fixed_plant_drift_m=float(max(plant_drift or [float('inf')])),
                 min_visual_sole_height_m=float(foot_height.min()),
                 max_planted_visual_sole_height_m=float(foot_height[active].max()),
                 max_pull_wrist_error_m=float(np.nanmax(wrist_error[pull])),
                 max_pull_wrist_orientation_error_deg=float(np.rad2deg(np.nanmax(rot_error[pull]))),
                 max_any_target_wrist_error_m=float(np.nanmax(wrist_error)),
                 max_task_grip_error_m=float(np.max(np.asarray(task_grip_error)[pull])),
                 max_task_grip_orientation_error_deg=float(np.rad2deg(np.max(np.asarray(task_orientation_error)[pull]))),
                 max_pull_finger_target_error_rad=finger_error,
                 max_root_rotation_step_rad=float(root_rot_step.max()),
                 max_door_step_rad=float(np.max(np.abs(np.diff(z['door_angle'])))),
                 max_root_step_m=float(np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1).max()),
                 max_joint_step_rad=float(np.abs(np.diff(q[:,7:],axis=0)).max()),
                 pull_min_static_com_support_margin_m=float(np.nanmin(np.asarray(com_margin)[pull])),
                 final_door_angle_degrees=float(np.rad2deg(z['door_angle'][-1])),
                 approach_displacement_m=walk_distance,
                 sampled_collision_frames=len(sampled),collision_event_frames=len(collision_events),max_scene_penetration_m=max_depth)
    checks=dict(finite_qpos=bool(np.isfinite(q).all()),unit_quaternions=bool(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1))<1e-6),
                increasing_time=bool(np.all(np.diff(tt)>0)),joint_limits=metrics['max_joint_limit_violation_rad']<=1e-5,
                uniform_authored_time=bool(np.max(np.abs(tt-np.arange(len(q))/task['plan']['fps']))<1e-8),
                source_and_model_bindings=True,support_schedule=support_schedule_ok,
                complete_task_phases=phase_order_ok,
                walking_approach=walk_distance>=1.5,
                intended_stance=bool(np.linalg.norm(q[-1,:2]-np.asarray(task['plan']['stance_xy']))<=.025),
                grasp_schedule=bool(z['grasp_active'][required_grasp].all() and not z['grasp_active'][~required_grasp].any()),
                door_closed_until_grasp=bool(np.max(np.abs(z['door_angle'][~required_grasp]))<=1e-6),
                door_monotonic_pull=bool(np.all(np.diff(z['door_angle'][required_grasp])>=-1e-6)),
                planted_feet=metrics['max_planted_foot_position_error_m']<=.005,
                fixed_plant_anchors=metrics['max_fixed_plant_drift_m']<=.005,
                sole_floor=metrics['min_visual_sole_height_m']>=-.003,
                planted_sole_height=metrics['max_planted_visual_sole_height_m']<=.003,
                pull_wrist_position=metrics['max_pull_wrist_error_m']<=.005,
                pull_wrist_orientation=metrics['max_pull_wrist_orientation_error_deg']<=2.,
                task_derived_grip_position=metrics['max_task_grip_error_m']<=.005,
                task_derived_grip_orientation=metrics['max_task_grip_orientation_error_deg']<=2.,
                planned_finger_closure=finger_error<=1e-5,
                nonempty_grip_posture=all(task['grasp']['finger_joint_targets'][f'right_hand_{s}_{j}_joint']>.5 for s in ['index','middle'] for j in [0,1]) and task['grasp']['finger_joint_targets']['right_hand_thumb_2_joint']<-.2,
                no_pose_jump=metrics['max_joint_step_rad']<=.20 and metrics['max_root_step_m']<=.05,
                no_root_rotation_jump=metrics['max_root_rotation_step_rad']<=.15,
                door_limits=bool(np.all(z['door_angle']>=source_check['door_limits_rad'][0]-1e-6) and np.all(z['door_angle']<=source_check['door_limits_rad'][1]+1e-6)),
                no_door_jump=metrics['max_door_step_rad']<=.08,
                door_open=abs(metrics['final_door_angle_degrees']-task['plan']['door_degrees'])<=.1 and abs(metrics['final_door_angle_degrees'])>=60.)
    if environment is not None:checks['scene_collision']=max_depth<=.002
    result=dict(schema='g1-kinematic-reference-verification/v1',claim='Measured geometric quality of a saved motion reference; not a controller rollout.',
                trajectory_sha256=sha(args.trajectory),task_sha256=sha(args.task),script_sha256=sha(__file__),
                robot_model_sha256=sha(model),scene_sha256=task['scene']['sha256'],checks=checks,metrics=metrics,
                source_geometry_validation=source_check,robot_asset_files=bound,
                joins=join,collision_events=collision_events,collision_sample_frames=sampled,
                full_frame_collision_check=not args.skip_collisions and args.stride==1,
                kinematic_checks_passed=all(checks.values()) and not args.skip_collisions and args.stride==1,
                physical_execution_verified=False,self_collision_verified=False,
                elapsed_seconds=time.monotonic()-started)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output/'measurements.npz',foot_position_error=foot_error,visual_sole_height=foot_height,
                        wrist_error=wrist_error,wrist_orientation_error=rot_error,com_support_margin=np.asarray(com_margin),
                        body_position=np.asarray(bodies),body_quaternion=np.asarray(bodyrot))
    print(json.dumps(dict(checks=checks,metrics=metrics,output=str(args.output)),indent=2),flush=True)


if __name__=='__main__':main()
