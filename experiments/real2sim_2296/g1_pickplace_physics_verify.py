#!/usr/bin/env python3
"""Evaluate actual torque-rollout evidence for the complete fridge-to-table task.

No scene state is changed and no reference attachment is treated as execution.
Whole-body contacts use control samples. Task contact/support gates also inspect
every recorded 500 Hz physics substep when the bound trace is available.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def longest(mask,t):
    best=0.;begin=None
    for i,v in enumerate(mask):
        if v and begin is None:begin=i
        if begin is not None and (not v or i==len(mask)-1):
            end=i if v else i-1;best=max(best,float(t[end]-t[begin]));begin=None
    return best


def audit_task_substeps(path, time, table, robot_bodies):
    """Conservatively aggregate task contacts over every control interval."""
    n=len(time);hand=np.ones(n,bool);opposed=np.ones(n,bool);off=np.ones(n,bool)
    ts=[];tables=[];hands=[];oppositions=[];supports=[];bad=[]
    door_contact=False;normal_frames=True;min_distance=0.;max_force=0.
    with path.open() as stream:
        for k,text in enumerate(stream):
            line=json.loads(text);t=float(line['time']);i=k//10
            assert i<n and abs(t-(k+1)*.002)<1e-7,'Task substep time/count mismatch'
            touched=False;on_table=False;nonhand=False;families=set();normals=np.zeros((2,3))
            for c in line['contacts']:
                bs={c['body1'],c['body2']};gs={c['geom1_name'],c['geom2_name']}
                local_force=np.asarray(c['force_contact_frame'],float)
                assert local_force.shape==(6,) and np.isfinite(local_force).all() and np.isfinite(c['distance']),'Nonfinite/invalid task contact'
                force=float(local_force[0]);rh={b for b in bs if b.startswith('right_hand_')}
                if bs&robot_bodies and bs&{'task_bottle','task_fridge_door'} and c['distance']<-.003:
                    bad.append(dict(substep=k,time=t,**c))
                if rh and bs&{'task_bottle','task_fridge_door'}:
                    min_distance=min(min_distance,float(c['distance']));max_force=max(max_force,abs(force))
                if 'task_fridge_door' in bs and rh and force>.05:door_contact=True
                if 'task_bottle' not in bs or force<=.05:continue
                if rh:
                    touched=True
                    for b in rh:
                        side=0 if 'thumb' in b else 1 if 'index' in b or 'middle' in b else None
                        if side is None:continue
                        families.add(side)
                        if 'frame_world_to_contact' not in c:normal_frames=False;continue
                        frame=np.asarray(c['frame_world_to_contact'],float)
                        assert frame.shape==(3,3) and np.isfinite(frame).all()
                        assert np.allclose(frame@frame.T,np.eye(3),rtol=0,atol=1e-6)
                        sign=-1 if c['body1']=='task_bottle' else 1
                        normals[side]+=sign*frame[0]*force
                else:nonhand=True
                if gs&table:on_table=True
            den=float(np.linalg.norm(normals[0])*np.linalg.norm(normals[1]))
            opposing=bool(families=={0,1} and den>1e-10 and np.dot(normals[0],normals[1])/den<-.8)
            hand[i]&=touched;opposed[i]&=opposing;off[i]&=not nonhand
            ts.append(t);tables.append(on_table);hands.append(touched);oppositions.append(opposing);supports.append(nonhand)
    assert len(ts)==n*10 and np.allclose(np.asarray(ts)[9::10],time,rtol=0,atol=1e-7),'Incomplete task substep trace'
    return dict(control_hand=hand,control_opposition=opposed,control_off_support=off,
                time=np.asarray(ts),table=np.asarray(tables),hand=np.asarray(hands),opposition=np.asarray(oppositions),
                nonhand_support=np.asarray(supports),door_contact=door_contact,normal_frames=normal_frames,bad=bad,
                minimum_hand_distance_m=min_distance,maximum_hand_normal_force_N=max_force)


def main():
    p=argparse.ArgumentParser();p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh evidence directory')
    out.mkdir(parents=True);run=a.rollout.resolve();raw=json.loads((run/'receipt.json').read_text())
    inputs=json.loads((run/'inputs.json').read_text());args=inputs['arguments'];task_path=Path(args['task'])
    task=json.loads(task_path.read_text());ref=np.load(args['reference'],allow_pickle=False);z=np.load(run/'rollout.npz',allow_pickle=False)
    bindings={str((run/x).resolve()):sha(run/x) for x in ['receipt.json','inputs.json','rollout.npz','contacts.jsonl']}
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    errors=[]
    for path,digest in {**inputs['inputs'],**inputs['model_dependencies']}.items():
        f=Path(path)
        if not f.is_file() or sha(f)!=digest:errors.append(path)
    for name,digest in raw['artifacts'].items():
        f=run/name
        if not f.is_file() or sha(f)!=digest:errors.append(str(f))
    assert sha(run/'inputs.json')==raw['inputs_receipt']['sha256'],'Rollout input receipt differs'
    executed=Path(inputs['executed_scene']['path']);assert sha(executed)==inputs['executed_scene']['sha256']
    bindings[str(executed)]=sha(executed)
    m=mujoco.MjModel.from_xml_path(str(executed));robot=mujoco.MjModel.from_xml_path(args['robot'])
    contact_solver={k:int(getattr(m.opt,k)) for k in ['solver','cone','iterations','noslip_iterations']}
    contact_solver.update({k:float(getattr(m.opt,k)) for k in ['tolerance','impratio','noslip_tolerance']})
    if inputs['model'].get('solver_options') is not None:
        assert contact_solver==inputs['model']['solver_options'],'Declared numerical contact options differ from executed model'
    sensitivity=inputs.get('solver_sensitivity')
    if args.get('physics_options'):
        options_path=Path(args['physics_options']).resolve()
        assert sha(options_path)==inputs['inputs'][str(options_path)],'Contact-treatment configuration is not hash-bound'
        assert sensitivity==json.loads(options_path.read_text()),'Contact-treatment declaration differs from bound configuration'
        options=sensitivity.get('options',{})
        assert options and set(options)<=set(contact_solver),'Missing or unsupported declared contact options'
        assert all(contact_solver[k]==v for k,v in options.items()),'Declared contact-treatment values differ from executed model'
    else:
        assert sensitivity is None,'Contact-treatment declaration has no bound configuration'
    adapter_path=Path(args['scene']).with_name('scene.receipt.json');adapter=json.loads(adapter_path.read_text())
    assert sha(Path(args['scene']))==adapter['output']['sha256'];bindings[str(adapter_path)]=sha(adapter_path)
    assert task['object']==adapter['bottle'],'Task prop differs from executed scene adapter'
    mapping={x['name']:x['usd_path'] for x in adapter['room']}
    top_path=task['place']['table_path']+'/Links/base/maple_round_top'
    table={n for n,v in mapping.items() if v==top_path};floor={n for n,v in mapping.items() if '/floor_surface_region_01/' in v}
    assert table and floor
    time=z['time'];n=len(time);assert n>1 and np.isfinite(time).all() and np.all(np.diff(time)>0)
    assert all(len(z[k])==n and np.isfinite(z[k]).all() for k in ['qpos','qvel','object_pose','door_angle','max_applied_torque'])
    assert np.allclose(np.linalg.norm(z['qpos'][:,3:7],axis=1),1,atol=1e-5)
    assert np.allclose(np.linalg.norm(z['object_pose'][:,3:7],axis=1),1,atol=1e-5)
    object_address=int(m.joint('task_bottle_free').qposadr[0]);door_address=int(m.joint('task_fridge_hinge').qposadr[0])
    object_coordinates=z['qpos'][:,object_address:object_address+7]
    assert np.allclose(object_coordinates[:,:3],z['object_pose'][:,:3],rtol=0,atol=1e-8),'Logged bottle position differs from physical qpos'
    assert np.allclose(np.abs(np.sum(object_coordinates[:,3:]*z['object_pose'][:,3:],axis=1)),1,rtol=0,atol=1e-8),'Logged bottle rotation differs from physical qpos'
    assert np.allclose(z['qpos'][:,door_address],z['door_angle'],rtol=0,atol=1e-10),'Logged door angle differs from physical qpos'
    rotation=Rotation.from_quat(z['object_pose'][:,[4,5,6,3]])
    axis=rotation.apply(np.array([0.,0.,1.]));base=z['object_pose'][:,:3]-axis*task['object']['height']/2
    root_axis=Rotation.from_quat(z['qpos'][:,[4,5,6,3]]).apply([0.,0.,1.])
    tilt=np.degrees(np.arccos(np.clip(axis[:,2],-1,1)))
    initial=np.asarray(task['object']['initial_base_world']);target=np.asarray(task['place']['target_base_world'])
    if 'object_velocity_world_angular_linear' in z.files:
        linear=np.linalg.norm(z['object_velocity_world_angular_linear'][:,3:],axis=1)
        angular=np.linalg.norm(z['object_velocity_world_angular_linear'][:,:3],axis=1)
    else:
        linear=np.linalg.norm(np.gradient(z['object_pose'][:,:3],time,axis=0),axis=1)
        angular=np.r_[0,(rotation[1:]*rotation[:-1].inv()).magnitude()/np.diff(time)]
    hand=np.zeros(n,bool);opposition=np.zeros(n,bool);support=np.zeros(n,bool);off_support=np.ones(n,bool)
    door_force=np.zeros(n,bool);bad=[];all_count=0;min_dist=0.;max_force=0.;records=[]
    contact_lines=[json.loads(x) for x in (run/'contacts.jsonl').read_text().splitlines()]
    assert len(contact_lines)==n and np.allclose([x['time'] for x in contact_lines],time,atol=1e-8)
    robot_bodies={robot.body(i).name for i in range(1,robot.nbody)}
    for i,line in enumerate(contact_lines):
        families=set()
        for c in line['contacts']:
            local_force=np.asarray(c['force_contact_frame'],float)
            assert local_force.shape==(6,) and np.isfinite(local_force).all() and np.isfinite(c['distance']),'Nonfinite/invalid control contact'
            all_count+=1;bs={c['body1'],c['body2']};gs={c['geom1_name'],c['geom2_name']};force=float(local_force[0])
            min_dist=min(min_dist,float(c['distance']));max_force=max(max_force,abs(force))
            rh={b for b in bs if b.startswith('right_hand_')};feet=bs&{'left_ankle_roll_link','right_ankle_roll_link'}
            if 'task_bottle' in bs:
                if rh and force>.05:
                    hand[i]=True
                    for b in rh:
                        if 'thumb' in b:families.add('thumb')
                        elif 'index' in b:families.add('index')
                        elif 'middle' in b:families.add('middle')
                if gs&table and force>.05:support[i]=True
                if any(not b.startswith('right_hand_') for b in bs-{'task_bottle'}) and force>.05:off_support[i]=False
            if 'task_fridge_door' in bs and rh and force>.05:door_force[i]=True
            if not bs&robot_bodies:continue
            allowed=(bool(feet) and 'world' in bs and bool(gs&floor)) or (bool(rh) and bool(bs&{'task_bottle','task_fridge_door'}) and c['distance']>=-.003)
            if not allowed and c['distance']<-.003:
                bad.append(dict(sample=i,time=float(time[i]),phase=str(z['phase'][i]),**c))
        opposition[i]='thumb' in families and bool(families&{'index','middle'})
    substeps=None;subpath=run/'task_contacts_substeps.jsonl'
    if subpath.is_file():
        assert sha(subpath)==raw['artifacts']['task_contacts_substeps.jsonl']
        bindings[str(subpath)]=sha(subpath)
        substeps=audit_task_substeps(subpath,time,table,robot_bodies)
        hand=substeps['control_hand'];opposition=substeps['control_opposition'];off_support=substeps['control_off_support']
        bad.extend(substeps['bad'])
    # The accepted reference lifts 35 mm before extraction; 25 mm requires
    # substantial actual shelf clearance without exceeding the authored lift.
    # Positive non-hand contacts still disqualify support-free lifting.
    required_lift_clearance=.025
    lifted=(base[:,2]>initial[2]+required_lift_clearance)&off_support&hand
    carried=(np.linalg.norm(base[:,:2]-initial[:2],axis=1)>1.)&(base[:,2]>.65)&hand&off_support
    final=time>=time[-1]-2.-1e-8
    final_duration=float(time[-1]-time[np.flatnonzero(final)[0]])
    final_xy=np.linalg.norm(base[final,:2]-target[:2],axis=1)
    final_z=np.abs(base[final,2]-task['place']['support_height'])
    released=not np.any(hand[final]);table_fraction=float(np.mean(support[final]))
    if substeps is not None:
        subfinal=substeps['time']>=time[-1]-2.-1e-8
        released=not np.any(substeps['hand'][subfinal])
        table_fraction=float(np.mean(substeps['table'][subfinal]))
    if 'reference_object_grasp_active' in z.files:released=released and not np.any(z['reference_object_grasp_active'][final])
    source_limits={robot.actuator(i).name:robot.actuator_ctrlrange[i].copy() for i in range(robot.nu)}
    indices=inputs['model']['actuator_indices'];limits=np.asarray([source_limits[m.actuator(i).name] for i in indices])
    torque=z['max_applied_torque'];assert torque.shape==(n,43)
    motor_limits_match=(m.nu==robot.nu==43 and all(np.array_equal(m.actuator_ctrlrange[i],source_limits[m.actuator(i).name]) for i in range(m.nu)))
    hinge=m.joint('task_fridge_hinge').id;bottle=m.joint('task_bottle_free').id
    motor_jointids=set(m.actuator_trnid[:,0].astype(int));passive=hinge not in motor_jointids and bottle not in motor_jointids
    external_samples=[z[x] if x in z.files else np.array([np.inf]) for x in ['substep_max_abs_qfrc_applied','substep_max_abs_xfrc_applied']]
    full_reference=('object_pose' in ref.files and 'object_grasp_active' in ref.files and np.any(ref['object_grasp_active']))
    # The first recorded state follows the first 20 ms of physics; it is not
    # an extra reset sample at reference time zero.
    expected_reference_time=np.minimum(float(args.get('start',0))+time,float(ref['time'][-1]))
    schedule_matches=np.allclose(z['reference_time'],expected_reference_time,rtol=0,atol=1e-6)
    coverage=bool(full_reference and raw['completed_requested_duration'] and schedule_matches and z['reference_time'][0]<=.021 and z['reference_time'][-1]>=ref['time'][-1]-.021)
    no_diagnostic=bool(float(args.get('start',0))==0 and not args.get('hold') and not args.get('diagnostic_reset_task_from_reference')
                   and not inputs.get('diagnostic_task_reset') and np.linalg.norm(base[0]-initial)<.001 and abs(z['door_angle'][0])<.001)
    modeldecl=inputs['model'];applied=np.max([float(np.max(np.abs(x))) for x in external_samples])
    gates=dict(
        immutable_declared_inputs=not errors,
        complete_reference_executed=coverage,
        complete_task_substep_contacts=bool(substeps is not None and substeps['normal_frames']),
        unassisted_initial_task=no_diagnostic,
        torque_only_freebase=(m.neq==0 and m.nmocap==0 and np.allclose(m.opt.gravity,[0,0,-9.81]) and int(m.opt.disableflags)==0 and passive and motor_limits_match
                             and modeldecl.get('per_step_qpos_assignment') is False and modeldecl.get('task_object_attachment') is False),
        source_motor_limits=bool(np.all(torque<=np.max(np.abs(limits),axis=1)+1e-8)),
        no_applied_external_force=bool(applied==0. and all(len(x)==n*10 for x in external_samples)),
        upright=bool(raw['fall'] is None and raw['exception'] is None and np.min(z['qpos'][:,2])>.45 and np.min(root_axis[:,2])>np.cos(np.deg2rad(45))),
        physical_door_opened=bool(np.max(z['door_angle'])>=np.deg2rad(60) and (np.any(door_force) or substeps is not None and substeps['door_contact'])),
        opposing_fingers_acquired=bool(longest(opposition,time)>=.1),
        physically_lifted=bool(longest(lifted,time)>=.4),
        carried_with_hand_contact=bool(longest(carried,time)>=1.0),
        reached_target_table=bool(np.max(final_xy)<.08 and np.max(final_z)<.01),
        released_and_supported_two_seconds=bool(final_duration>=2.-1e-8 and released and table_fraction>=.95),
        final_object_stable=bool(np.max(tilt[final])<15 and np.max(linear[final])<.04 and np.max(angular[final])<.3),
        no_major_unintended_robot_contacts=len(bad)==0)
    metrics=dict(samples=n,simulated_seconds=float(time[-1]),contact_records=all_count,
        required_lift_clearance_m=required_lift_clearance,
        sampled_minimum_contact_distance_m=min_dist,sampled_maximum_normal_contact_force_N=max_force,
        door_max_degrees=float(np.degrees(np.max(z['door_angle']))),door_final_degrees=float(np.degrees(z['door_angle'][-1])),
        opposing_finger_contact_longest_seconds=longest(opposition,time),lift_with_hand_contact_longest_seconds=longest(lifted,time),
        carried_with_hand_contact_longest_seconds=longest(carried,time),object_max_displacement_xy_m=float(np.max(np.linalg.norm(base[:,:2]-initial[:2],axis=1))),
        final_object_base=base[-1].tolist(),final_window_seconds=final_duration,final_window_table_contact_fraction=table_fraction,
        final_window_max_xy_error_m=float(np.max(final_xy)),final_window_max_support_z_error_m=float(np.max(final_z)),
        final_window_max_tilt_degrees=float(np.max(tilt[final])),final_window_max_linear_speed_m_s=float(np.max(linear[final])),
        final_window_max_angular_speed_rad_s=float(np.max(angular[final])),unexpected_contact_records=len(bad),max_external_force=applied)
    if substeps is not None:
        metrics.update(task_contact_substeps=len(substeps['time']),task_contact_hz=500,
            task_substep_hand_minimum_contact_distance_m=substeps['minimum_hand_distance_m'],
            task_substep_hand_maximum_normal_force_N=substeps['maximum_hand_normal_force_N'],
            task_substep_opposing_normal_contact_longest_seconds=longest(substeps['opposition'],substeps['time']))
        np.savez_compressed(out/'task_contact_measurements.npz',**{k:substeps[k] for k in ['time','table','hand','opposition','nonhand_support']})
    (out/'unexpected_contacts.json').write_text(json.dumps(bad,indent=2)+'\n')
    np.savez_compressed(out/'measurements.npz',time=time,object_base=base,object_tilt=tilt,hand_contact=hand,opposing_fingers=opposition,
                        table_contact=support,off_world_support=off_support,linear_speed=linear,angular_speed=angular)
    result=dict(schema='g1-pickplace-physical-task-evaluation/v1',produced_at=datetime.now(timezone.utc).isoformat(),rollout=str(run),
        inputs=bindings,input_mismatches=errors,gates=gates,complete_physical_task_passed=all(gates.values()),metrics=metrics,
        contact_solver=contact_solver,declared_solver_sensitivity=inputs.get('solver_sensitivity'),
        contact_sampling=raw.get('contact_sampling'),
        scope='Actual recorded torque simulation. Pose/stability and general whole-body contacts use 50 Hz control samples. All bottle contacts and the door-contact subset declared in contact_sampling are checked at every bound 500 Hz substep, including penetration by any recorded source robot body. Historical door traces include hand links only; other door/body contacts then have 50 Hz coverage. Opposing normals indicate contact geometry, not a force-closure certificate. Source/controller inspection must separately verify no hidden state setters or undeclared assistance.',
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
