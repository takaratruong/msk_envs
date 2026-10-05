#!/usr/bin/env python3
"""Independent saved-pose checks; this is not a physical execution test."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotation(q):return Rotation.from_quat(np.roll(q,-1)).as_matrix()


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'receipt.json').exists():raise FileExistsError('Use a new verification output')
    task=json.loads(a.task.read_text());z=np.load(a.reference,allow_pickle=False)
    source=mujoco.MjModel.from_xml_path(task['robot']['model']);m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m)
    adapter=json.loads(a.scene.with_name('scene.receipt.json').read_text())
    assert sha(a.scene)==adapter['output']['sha256'],'Adapter XML differs'
    assert sha(task['robot']['model'])==task['robot']['sha256']==adapter['robot']['sha256'],'Robot differs'
    assert task['scene']['sha256']==adapter['scene']['sha256'],'Reference targets a different room revision'
    adapter_task_path=Path(adapter['task']['path'])
    assert sha(adapter_task_path)==adapter['task']['sha256'],'Adapter geometry task changed'
    adapter_task=json.loads(adapter_task_path.read_text())
    for key in ['object','place','door']:
        assert task[key]==adapter_task[key],'Reference/adapter task geometry differs: '+key
    dependencies={}
    for item in adapter['scene_layers']+task['robot']['asset_files']:
        path=Path(item['path']);assert sha(path)==item['sha256'],'Input asset changed: '+str(path)
        dependencies[str(path.resolve())]=sha(path)
    for item in adapter['room']:
        path=a.scene.parent/'room_meshes'/(item['name']+'.obj')
        assert sha(path)==item['sha256'],'Room collider changed: '+str(path)
        dependencies[str(path.resolve())]=sha(path)
    for item in adapter['robot']['assets']:
        path=a.scene.parent/item['copy'];assert sha(path)==item['sha256'],'Copied robot geometry changed'
        dependencies[str(path.resolve())]=sha(path)
    # The added shoulder collider is generated alongside the XML. Bind its
    # actual bytes even though older adapter receipts lack an explicit digest.
    if task['object'].get('shape','compound_bottle')=='compound_bottle':
        path=a.scene.parent/'bottle_shoulder.obj';dependencies[str(path.resolve())]=sha(path)
    source_paths={x['name']:x['usd_path'] for x in adapter['room']}
    ground={n for n,path in source_paths.items() if '/floor_surface_region_01/' in path}
    supports={n for n,path in source_paths.items() if path==task['object']['initial_support'] or path==task['place']['table_path']+'/Links/base/maple_round_top'}
    rail_geoms={n for n,path in source_paths.items() if path.startswith(task['door']['body_path']+'/upper_door_1_long_pull')}
    q=np.array(z['qpos']);assert q.shape[1]==source.nq and np.isfinite(q).all()
    assert np.all(np.diff(z['time'])>0)
    assert np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-5),'Root quaternions are not normalized'
    assert len(q)==len(z['object_pose'])==len(z['phase'])==len(z['time'])
    assert np.isfinite(z['object_pose']).all() and np.allclose(np.linalg.norm(z['object_pose'][:,3:7],axis=1),1,atol=1e-5)
    maps=[]
    for j in range(source.njnt):
        name=source.joint(j).name;k=m.joint(name).id;width=7 if source.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
        maps.append((int(source.jnt_qposadr[j]),int(m.jnt_qposadr[k]),width))
    hinge=m.joint('task_fridge_hinge');obj=m.body('task_bottle');objq=int(m.jnt_qposadr[m.joint('task_bottle_free').id])
    wrist=m.body('right_wrist_yaw_link').id;feet=[m.body(x+'_ankle_roll_link').id for x in ['left','right']]
    grip=np.asarray(task['grasp']['grip_point_wrist']);objgrip=np.array([0.,0.,task['object']['grasp_height']-task['object']['height']/2])
    object_hand_grip=np.asarray(task.get('object_grasp',task['grasp'])['grip_point_wrist'])
    object_grasp_geom='task_bottle_body' if task['object'].get('shape')=='cylinder' else 'task_bottle_neck'
    labels=z['phase'];bad=[];contacts=[];native_self=[];footerr=[];sole=[];griperr=[];railerr=[];wristerr=[];bases=[]
    origin=task.get('object_pose_origin',task['object'].get('pose_origin','center'))
    # Authoring versions may record this explicit convention in reference metadata.
    if 'object_pose_origin' in z.files:origin=str(z['object_pose_origin'].item())
    lower=source.jnt_range[1:,0];upper=source.jnt_range[1:,1]
    violation=np.maximum(np.maximum(lower-q[:,7:],q[:,7:]-upper),0)
    robotbody={m.body(source.body(i).name).id for i in range(1,source.nbody)}
    handbody={i for i in robotbody if m.body(i).name.startswith('right_hand_') and 'palm' not in m.body(i).name}
    footbody=set(feet);doorbody=m.body('task_fridge_door').id
    for i,row in enumerate(q):
        d.qpos[:]=m.qpos0
        for r,t,width in maps:d.qpos[t:t+width]=row[r:r+width]
        d.qpos[int(hinge.qposadr[0])]=float(z['door_angle'][i]);op=np.array(z['object_pose'][i],float)
        if origin=='base':op[:3]+=rotation(op[3:7])@np.array([0.,0.,task['object']['height']/2])
        d.qpos[objq:objq+7]=op;d.qvel[:]=0;mujoco.mj_forward(m,d)
        rr=d.xmat[wrist].reshape(3,3);hg=d.xpos[wrist]+rr@grip
        objectneck=d.xpos[obj.id]+d.xmat[obj.id].reshape(3,3)@objgrip
        bases.append((d.xpos[obj.id]-d.xmat[obj.id].reshape(3,3)@np.array([0,0,task['object']['height']/2])).copy())
        ag=bool(z['object_grasp_active'][i]) if 'object_grasp_active' in z.files else bool(z['grasp_active'][i] and i>588)
        dg=bool(z['door_grasp_active'][i]) if 'door_grasp_active' in z.files else bool(z['grasp_active'][i] and i<=588)
        object_hg=d.xpos[wrist]+rr@object_hand_grip
        griperr.append(float(np.linalg.norm(object_hg-objectneck)) if ag else np.nan)
        pivot=np.asarray(task['door']['world_pivot']);rail=np.asarray(task['grasp']['rail_point_world'])
        railat=pivot+Rotation.from_euler('z',float(z['door_angle'][i])).as_matrix()@(rail-pivot)
        railerr.append(float(np.linalg.norm(hg-railat)) if dg else np.nan)
        if 'wrist_target' in z.files and np.isfinite(z['wrist_target'][i]).all():wristerr.append(float(np.linalg.norm(d.xpos[wrist]-z['wrist_target'][i])))
        else:wristerr.append(np.nan)
        footerr.append(np.linalg.norm(d.xpos[feet]-z['foot_target'][i],axis=1) if 'foot_target' in z.files else [np.nan,np.nan])
        low=[]
        for foot in feet:
            vs=[]
            for gi in range(m.ngeom):
                if m.geom_bodyid[gi]!=foot or m.geom_contype[gi] or m.geom_conaffinity[gi]:continue
                if m.geom_type[gi]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mi=int(m.geom_dataid[gi]);v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]
                vs.extend((v@d.geom_xmat[gi].reshape(3,3).T+d.geom_xpos[gi])[:,2])
            low.append(min(vs) if vs else np.nan)
        sole.append(low)
        for c in d.contact:
            if c.dist>=-.0005:continue
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2]);bs={b1,b2}
            if not (bs&robotbody) and obj.id not in bs:continue
            names=[m.geom(g1).name or f'geom_{g1}',m.geom(g2).name or f'geom_{g2}']
            rec=dict(frame=i,time=float(z['time'][i]),phase=str(labels[i]),distance_m=float(c.dist),geoms=names,bodies=[m.body(b1).name,m.body(b2).name])
            allowed=False
            if bs<=robotbody:native_self.append(rec)
            elif obj.id in bs:
                grip_phase=ag or str(labels[i]) in ['close_dex3_on_bottle','release_bottle_on_table','establish_reference_bottle_grasp']
                allowed=(bool(bs&handbody and grip_phase) and object_grasp_geom in names and c.dist>=-.003) or (0 in bs and not ag and bool(set(names)&supports) and c.dist>=-.001)
                contacts.append(rec|{'intended':allowed})
            elif doorbody in bs and bs&handbody:
                allowed=(dg or str(labels[i]) in ['close_hand','release_door_handle','release_door_hand']) and bool(set(names)&rail_geoms) and c.dist>=-.003
            elif 0 in bs and bs&footbody:
                # Ground identification comes from imported USD mapping below;
                # foot contact with a furniture collider is never whitelisted.
                allowed=bool(set(names)&ground)
            if not allowed:bad.append(rec)
    def finite_max(x):
        x=np.asarray(x);return float(np.nanmax(x)) if np.isfinite(x).any() else None
    active=np.asarray(z['foot_contact'],bool);fe=np.asarray(footerr)
    bodybases=np.asarray(bases);target=np.asarray(task['place']['target_base_world']);result=dict(
        frames=len(q),duration=float(z['time'][-1]),joint_limit_max_rad=float(violation.max()),
        maximum_joint_step_rad=float(np.abs(np.diff(q[:,7:],axis=0)).max()),
        maximum_planted_foot_error_m=finite_max(fe[active]),minimum_visible_sole_z_m=finite_max(-np.asarray(sole)),
        maximum_wrist_target_error_m=finite_max(wristerr),maximum_active_object_grip_error_m=finite_max(griperr),
        maximum_active_door_grip_error_m=finite_max(railerr),final_object_base=bodybases[-1].tolist(),
        final_object_target_error_m=float(np.linalg.norm(bodybases[-1]-target)),
        unexpected_native_contacts=len(bad),native_self_contacts=len(native_self),object_pose_origin=origin)
    result['minimum_visible_sole_z_m']=-result['minimum_visible_sole_z_m']
    final_window=np.asarray(z['time'])>=float(z['time'][-1])-2.0-1e-8
    hold_time=np.asarray(z['time'])[final_window]
    final_released=('object_grasp_active' in z.files and not np.any(z['object_grasp_active'][final_window]))
    final_error=np.linalg.norm(bodybases[final_window]-target,axis=1)
    result['released_final_hold_seconds']=float(hold_time[-1]-hold_time[0]) if final_released else 0.0
    result['maximum_final_hold_target_error_m']=float(final_error.max())
    def bounded(value,limit):return value is not None and np.isfinite(value) and value<limit
    gates=dict(joint_limits=result['joint_limit_max_rad']<1e-6,foot_plants=bounded(result['maximum_planted_foot_error_m'],.003),
               object_grip=bounded(result['maximum_active_object_grip_error_m'],.003),door_grip=bounded(result['maximum_active_door_grip_error_m'],.003),
               visible_sole_clearance=result['minimum_visible_sole_z_m']>=-.001,
               joint_step=result['maximum_joint_step_rad']<.25,
               placement=result['final_object_target_error_m']<.003,native_contacts=len(bad)==0,
               released_final_hold=result['released_final_hold_seconds']>=2.0-1e-8 and result['maximum_final_hold_target_error_m']<.003)
    np.savez_compressed(a.output/'measurements.npz',foot_error=footerr,sole_z=sole,object_grip_error=griperr,door_grip_error=railerr,wrist_error=wristerr,object_base=bodybases)
    (a.output/'contacts.json').write_text(json.dumps(dict(unexpected=bad,object=contacts,native_self=native_self),indent=2)+'\n')
    (a.output/'dependencies.json').write_text(json.dumps(dependencies,indent=2)+'\n')
    receipt=dict(schema='g1-pickplace-saved-pose-verification/v1',inputs={str(p.resolve()):sha(p) for p in [a.reference,a.task,a.scene,a.scene.with_name('scene.receipt.json'),adapter_task_path,Path(__file__)]},
        metrics=result,gates=gates,saved_pose_gates_passed=all(gates.values()),physical_execution_verified=False,
        scope='Native collision proxies and visible foot soles at saved poses; no continuous sweep or physical grasp proof',
        artifacts={p.name:sha(p) for p in a.output.iterdir() if p.is_file()})
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
