#!/usr/bin/env python3
"""Exercise an existing physical USD in Isaac; source edits stay in session.

Requires the adjacent scene.manifest.json and build_inputs.json completion
receipt. Drive targets, gravity probes, and test cameras are transient. No
source stage is saved. Use jobctl plus an external process-group timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


BACKEND_FAILURE = re.compile(r'Invalid PhysX transform|Illegal BroadPhaseUpdateData|PhysX error:', re.IGNORECASE)


def backend_failure_lines(text):
    """Finite USD transforms can be stale after a solver error; check its log."""
    return [line for line in text.splitlines() if BACKEND_FAILURE.search(line)]


def sequential_plan(driven):
    """Each drive is a test, with real appliance-door prerequisites for racks."""
    plan=[]
    for joint in driven:
        if joint.get('passive') or joint.get('retainer_locked'):
            raise ValueError('A passive or locked joint cannot be commanded: '+joint['path'])
        doors=[]
        if joint['type']=='prismatic' and 'rack' in joint['path'].lower():
            doors=[d for d in driven if d['asset']==joint['asset'] and d['type']=='revolute'
                   and 'door' in d['path'].lower()]
            if not doors:raise ValueError('Rack has no selected movable appliance door: '+joint['path'])
        plan.append(dict(joint=joint,prerequisite_doors=doors))
    return plan


def mechanical_verdict(report, probes_requested):
    """A missing probe or handle control must not shrink the pass denominator."""
    summary = report.get('summary', {})
    outcomes = report.get('joint_outcomes', [])
    denominators = report.get('joint_denominators', {})
    category_counts = {
        'driven': sum(j.get('status') in ('drive_cycle_measured', 'blocked_by_appliance_door') for j in outcomes),
        'locked': sum(j.get('status') == 'locked_checked' for j in outcomes),
        'excluded_passive_or_legacy_mobile_joints': sum(j.get('status') == 'excluded_passive_or_legacy_mobile_subsystem' for j in outcomes),
        'unselected_focused_diagnostic': sum(j.get('status') == 'not_selected_for_focused_diagnostic' for j in outcomes),
    }
    checked = [j for j in outcomes if 'passed' in j]
    paths = [j.get('path') for j in outcomes]
    complete_outcomes = bool(checked and len(outcomes) == len(set(paths))
        and None not in paths and len(outcomes) == denominators.get('manifest')
        and set(paths) == set(report.get('authored_initial_joint_state', {}))
        and sum(category_counts.values()) == len(outcomes)
        and all(denominators.get(k) == v for k, v in category_counts.items())
        and len(checked) == category_counts['driven'] + category_counts['locked']
        and all(('passed' in j) == (j.get('status') in ('drive_cycle_measured', 'blocked_by_appliance_door', 'locked_checked')) for j in outcomes)
        and all(j.get('passed') is True for j in checked)
        and summary.get('manifest_joint_count') == len(outcomes)
        and summary.get('checked_joint_count') == len(checked)
        and summary.get('passed_joint_count') == len(checked)
        and summary.get('locked_joint_count') == category_counts['locked']
        and summary.get('excluded_joint_count') == category_counts['excluded_passive_or_legacy_mobile_joints']
        and summary.get('unselected_joint_count') == category_counts['unselected_focused_diagnostic'])
    mechanisms = bool(report.get('status') == 'completed' and summary
        and complete_outcomes
        and report.get('source_files_unchanged')
        and report.get('physics_backend_checks', {}).get('valid')
        and summary.get('failed_joint_count') == 0
        and summary.get('initial_settle_stable'))
    if not probes_requested:
        return mechanisms
    probes = report.get('support_probes', [])
    supports = (len(probes) == 3
        and {p.get('label') for p in probes} == {'countertop', 'cabinet_shelf', 'sink_bowl'}
        and all(p.get('support_observed') for p in probes))
    handle = report.get('handle_gap_test', {})
    return bool(mechanisms and supports and handle.get('status') == 'queried'
        and handle.get('positive_controls_passed') and handle.get('clearance_for_12mm_sphere'))


def quaternion_matrix(q):
    import numpy as np
    w = float(q.GetReal())
    x, y, z = map(float, q.GetImaginary())
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def joint_state(stage, records):
    """Read solver-updated body transforms, not the commanded USD targets."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = {}
    for record in records:
        prim = stage.GetPrimAtPath(record['path'])
        joint = UsdPhysics.Joint(prim)
        frames = []
        for side in (0, 1):
            targets = getattr(joint, f'GetBody{side}Rel')().GetTargets()
            matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(targets[0])), dtype=float)
            rotation = matrix[:3,:3].T
            anchor = rotation @ np.asarray(getattr(joint, f'GetLocalPos{side}Attr')().Get(), dtype=float) + matrix[3,:3]
            rotation = rotation @ quaternion_matrix(getattr(joint, f'GetLocalRot{side}Attr')().Get())
            frames.append((anchor, rotation))
        axis = 'XYZ'.index(prim.GetAttribute('physics:axis').Get())
        p0, r0 = frames[0]; p1, r1 = frames[1]
        delta = p1-p0
        if record['type'] == 'revolute':
            rel = r0.T @ r1
            j, k = (axis+1)%3, (axis+2)%3
            value = math.degrees(math.atan2(rel[k,j], rel[j,j]))
            anchor_error = float(np.linalg.norm(delta))
        else:
            value = float(np.dot(delta, r0[:,axis]))
            anchor_error = float(np.linalg.norm(delta-r0[:,axis]*value))
        result[record['path']] = dict(value=value, off_axis_anchor_error_m=anchor_error,
                                    body1_origin_world=np.asarray(cache.GetLocalToWorldTransform(
                                        stage.GetPrimAtPath(record['body1'])),dtype=float)[3,:3].tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--camera', default='/World/Cameras/frame_001499')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--dt', type=float, default=1/120)
    parser.add_argument('--phase-seconds', type=float, default=1.75)
    parser.add_argument('--open-fraction', type=float, default=.8,
                        help='Fraction of the authored limit used by ordinary joints; 1 exercises their full range. Bottom-hinged appliances always use their full limit.')
    parser.add_argument('--capture-every', type=int, default=0,
                        help='Capture every N actual physics steps; zero disables the motion movie')
    parser.add_argument('--no-probes', action='store_true')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--sequential-joints', nargs='+', default=None,
                        help='Focused diagnostic: exact asset/Joint leaf fragments, one drive cycle at a time')
    selection.add_argument('--all-sequential', action='store_true',
                        help='Cycle every non-passive, unlocked joint; open appliance doors before each rack')
    selection.add_argument('--probes-only', action='store_true',
                        help='Preserve all source drive targets and check supports/handle plus retained joints; no driven-cycle claim')
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if args.dt <= 0 or args.phase_seconds <= 0 or args.gpu < 0:
        parser.error('dt/phase-seconds must be positive and gpu nonnegative')
    if not 0. < args.open_fraction <= 1.:
        parser.error('open-fraction must be greater than zero and at most one')
    if args.probes_only and args.no_probes:
        parser.error('probes-only cannot disable probes')
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise RuntimeError('Run with env -u CUDA_VISIBLE_DEVICES; GPU ordinal must remain physical')
    source = args.scene.resolve(strict=True)
    manifest_path = source.with_suffix('.manifest.json')
    build_path = source.parent/'build_inputs.json'
    # The completion receipt is deliberately required before opening the USD.
    manifest = json.loads(manifest_path.read_text())
    build = json.loads(build_path.read_text())
    scene_hash = sha(source)
    if scene_hash != manifest['scene_sha256'] or scene_hash != build['scene_sha256']:
        raise RuntimeError('Scene does not match both completion receipts')
    out = args.output_root.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out/'motion_frames').mkdir()
    trajectory_stream=(out/'trajectory.jsonl').open('w')
    input_files = {str(source):scene_hash, str(manifest_path):sha(manifest_path),
                   str(build_path):sha(build_path), str(Path(__file__).resolve()):sha(__file__)}
    # Keep small input snapshots portable; source assets stay immutable in place.
    (out/'scene.manifest.snapshot.json').write_bytes(manifest_path.read_bytes())
    (out/'build_inputs.snapshot.json').write_bytes(build_path.read_bytes())
    report = dict(schema='real2sim-physical-simulation/v1', created_at=now(),
        source_scene=str(source), source_scene_sha256=scene_hash, input_files_sha256=input_files,
        request=vars(args)|{'scene':str(source),'output_root':str(args.output_root.resolve())},
        status='starting', output=str(out), pid=os.getpid(), metric_scale_measured=False,
        source_scene_edits='session_layer_only; no save operation',
        drive_claim='Joint positions are derived from simulated rigid-body transforms, not authored animation.',
        physics_validation_scope='Modeled-asset mechanical checks; not real-world joint dynamics or G1 transfer.')
    started = time.monotonic(); app = sim = product = annotator = contact_sub = None
    states, frames, contacts = [], [], []
    probe_paths = []
    scene_contact_summary = {}
    backend_events = []
    backend_log_offset = 0
    backend_log_tail = ''
    source_layers, texture_files = {}, {}
    active_phase = 'initializing'
    step_index = 0

    def status(value, **data):
        report.update(status=value, **data, elapsed_seconds=time.monotonic()-started)
        save(out/'status.json', report)
        emit(value, elapsed_seconds=report['elapsed_seconds'], **data)

    def scan_backend_errors():
        nonlocal backend_log_offset, backend_log_tail
        logfile = out/'kit.log'
        if logfile.is_file():
            with logfile.open('rb') as handle:
                handle.seek(backend_log_offset)
                chunk = handle.read()
                backend_log_offset = handle.tell()
            decoded=backend_log_tail+chunk.decode('utf-8', errors='replace')
            backend_log_tail=decoded[-512:]
            for line in backend_failure_lines(decoded):
                if len(backend_events) < 40:
                    backend_events.append(dict(step=step_index, phase=active_phase, message=line))
        report['physics_backend_checks'] = dict(log_bytes_checked=backend_log_offset,
            observed_failure_events=backend_events, valid=not backend_events,
            scope='Explicit PhysX errors/invalid-transform messages plus contact overflow; finite USD alone is insufficient.')
        if backend_events:
            save(out/'physics_backend_errors.json', report['physics_backend_checks'])
            raise RuntimeError('PhysX backend failure; USD transforms may be stale: '+backend_events[0]['message'])

    status('starting', output=str(out))
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=args.gpu, physics_gpu=args.gpu,
            multi_gpu=False, width=args.width, height=args.height, renderer='RaytracedLighting',
            sync_loads=True, disable_viewport_updates=True, anti_aliasing=0, fast_shutdown=True,
            enable_crashreporter=False, extra_args=['--portable-root',str(out/'kit-portable'),
                '--/app/settings/persistent=false','--/app/telemetry/enabled=false',f'--/log/file={out}/kit.log']))
        import carb.settings
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from omni.physx import get_physx_interface, get_physx_simulation_interface, get_physx_scene_query_interface
        from isaacsim.core.api import SimulationContext
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools, Gf
        context = omni.usd.get_context()
        if not context.open_stage(str(source)):
            raise RuntimeError('Isaac could not open completed source stage')
        for _ in range(8): app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        settings = carb.settings.get_settings()
        settings.set('/physics/updateToUsd',True)
        settings.set('/physics/updateVelocitiesToUsd',True)
        settings.set('/physics/outputVelocitiesLocalSpace',False)
        source_layers = {}
        for layer in stage.GetUsedLayers():
            if layer.realPath and Path(layer.realPath).is_file(): source_layers[layer.realPath] = sha(layer.realPath)
        # Bind local texture dependencies as well as the stage that references them.
        texture_files = {}
        for prim in stage.Traverse():
            attr = prim.GetAttribute('inputs:file')
            if attr:
                value = attr.Get()
                if value:
                    path = Path(value.resolvedPath) if value.resolvedPath else source.parent/value.path
                    if path.is_file(): texture_files[str(path.resolve())] = sha(path)
        report['render_asset_files_sha256'] = texture_files
        report['usd_layers_sha256'] = source_layers
        report['gpu_settings'] = {p:settings.get(p) for p in
                                  ('/renderer/activeGpu','/renderer/multiGpu/enabled','/physics/cudaDevice')}
        render_settings = dict(stage.GetRootLayer().customLayerData.get('renderSettings',{}))
        for key,value in render_settings.items(): settings.set('/'+key.replace(':','/').lstrip('/'),value)
        report['render_settings'] = render_settings
        records = manifest['joints']
        if len({j['path'] for j in records})!=len(records):raise ValueError('Duplicate manifest joint paths')
        assets = {a['name']:a for a in manifest['assets']}
        # Cart steering and free-running wheel travel require a separate mobile-base
        # test; commanding hundreds of revolutions here would be meaningless.
        excluded = [j for j in records if j.get('passive',False)]
        # Old manifests lacked the passive field for carts. Their endless wheel
        # travel is retained as an explicit legacy exclusion, never silently run.
        excluded += [j for j in records if 'passive' not in j
                     and assets[j['asset']]['metadata'].get('asset_type') == 'service_cart']
        excluded_paths = {j['path'] for j in excluded}
        locked = [j for j in records if j['retainer_locked']]
        if excluded_paths & {j['path'] for j in locked}:raise ValueError('Joint is classified as both passive and locked')
        all_driven = [j for j in records if j['path'] not in excluded_paths and not j['retainer_locked']]
        driven = all_driven
        unselected_paths=set()
        if args.sequential_joints:
            driven=[j for j in all_driven if any(j['path'].endswith('/'+fragment) for fragment in args.sequential_joints)]
            if len(driven)!=len(args.sequential_joints):raise ValueError('Sequential joint selection missing, ambiguous, or locked')
            unselected_paths={j['path'] for j in all_driven if j not in driven}
        if args.probes_only:
            driven=[]
            unselected_paths={j['path'] for j in all_driven}
        report['joint_denominators'] = dict(manifest=len(records), driven=len(driven), locked=len(locked),
                                          excluded_passive_or_legacy_mobile_joints=len(excluded),unselected_focused_diagnostic=len(unselected_paths))
        report['excluded_joints'] = [dict(path=j['path'],reason='Authored passive joint' if j.get('passive',False)
            else 'Legacy manifest omitted cart/wheel passive field; subsystem excluded') for j in excluded]
        if args.all_sequential or args.sequential_joints or args.probes_only:
            plan=sequential_plan(driven)
            report['sequential_plan']=[dict(joint=p['joint']['path'],prerequisite_doors=[j['path'] for j in p['prerequisite_doors']]) for p in plan]
        active_targets={}
        for j in records:
            kind='angular' if j['type']=='revolute' else 'linear'
            value=UsdPhysics.DriveAPI(stage.GetPrimAtPath(j['path']),kind).GetTargetPositionAttr().Get()
            active_targets[j['path']]=None if value is None else float(value)
        report['initial_usd_drive_targets']=dict(active_targets)
        report['drive_commands']=[]
        for record in records:
            prim = stage.GetPrimAtPath(record['path'])
            if not prim: raise RuntimeError('Manifest joint missing: '+record['path'])
            limits = [prim.GetAttribute('physics:lowerLimit').Get(),prim.GetAttribute('physics:upperLimit').Get()]
            if not np.allclose(limits,[record['lower'],record['upper']],atol=1e-5):
                raise RuntimeError('Manifest and USD limits disagree: '+record['path'])
        initial = joint_state(stage,records)
        report['authored_initial_joint_state'] = initial
        report['initial_state_matches_manifest'] = all(abs(initial[j['path']]['value']-j['initial']) < .001 for j in records if j['path'] not in excluded_paths)
        if not report['initial_state_matches_manifest']:
            raise RuntimeError('Initial joint transform does not agree with scene manifest')
        rigid_paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        # A cart/cooler with an articulation can be free: only an actual joint
        # to the world establishes a fixed base, never the presence of joints.
        fixed_bases = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                targets0, targets1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if not targets0 and targets1: fixed_bases.append(str(targets1[0]))
                elif targets0 and not targets1: fixed_bases.append(str(targets0[0]))
        free_asset_bases = [a['path']+'/Links/base' for a in manifest['assets']
                           if a['path']+'/Links/base' in rigid_paths and a['path']+'/Links/base' not in fixed_bases]
        report['base_classification'] = dict(fixed_to_world=fixed_bases, free=free_asset_bases)

        def body_positions():
            cache=UsdGeom.XformCache(Usd.TimeCode.Default());values={}
            for p in rigid_paths:
                matrix=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)),dtype=float)
                if not np.isfinite(matrix).all(): raise RuntimeError('Non-finite rigid transform: '+p)
                values[p]=matrix[3,:3].tolist()
            return values

        body_initial = body_positions()
        sim = SimulationContext(physics_dt=args.dt,rendering_dt=0.0,physics_prim_path='/World/PhysicsScene',
                                set_defaults=False,device='cpu')
        sim.get_physics_context().enable_fabric(False)
        sim.get_physics_context().set_physx_update_transformations_settings(True,True,False)
        physx = get_physx_interface()
        query = get_physx_scene_query_interface()
        report['simulation_configuration'] = dict(physics_dt=args.dt, device='cpu', gpu_renderer=args.gpu,
            update_to_usd=True, fabric=False, gravity_magnitude=UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene')).GetGravityMagnitudeAttr().Get(),
            gravity_direction=list(UsdPhysics.Scene(stage.GetPrimAtPath('/World/PhysicsScene')).GetGravityDirectionAttr().Get()),
            scene_mass_drive_collision_parameters_modified=False)

        # All nonmobile moving links matter during initial settling, even when
        # a focused command pass selects only a subset of their drives.
        monitored_bodies={j['body1'] for j in all_driven+locked}
        for body in monitored_bodies:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(body)).CreateThresholdAttr(0.)

        def contact_callback(headers,data):
            for header in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(header.actor0)),str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                is_probe=any('/PhysicsVerification/' in p for p in actors)
                if not is_probe and not any(p in monitored_bodies for p in actors): continue
                colliders=[str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1))]
                samples=[]
                for k in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                    c=data[k]; samples.append(dict(position=list(c.position),normal=list(c.normal),impulse=list(c.impulse),separation=float(c.separation)))
                if is_probe and len(contacts)<10000:
                    contacts.append(dict(step=step_index,phase=active_phase,actors=actors,colliders=colliders,samples=samples))
                elif not is_probe:
                    key='|'.join([active_phase]+sorted(colliders))
                    rec=scene_contact_summary.setdefault(key,dict(phase=active_phase,actors=actors,colliders=colliders,
                        contact_headers=0,sample_count=0,max_impulse_norm=0.,min_separation=0.))
                    rec['contact_headers']+=1;rec['sample_count']+=len(samples)
                    for c in samples:
                        impulse=float(np.linalg.norm(c['impulse']))
                        rec['max_impulse_norm']=max(rec['max_impulse_norm'],impulse)
                        rec['min_separation']=min(rec['min_separation'],c['separation'])
                        if (not np.isfinite(impulse) or impulse > 1e8) and len(backend_events)<40:
                            backend_events.append(dict(step=step_index,phase=active_phase,
                                message='Nonfinite or explosive contact impulse exceeds 1e8 nominal N.s',
                                colliders=colliders,impulse_norm=impulse))
        contact_sub=get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
        # Author and subscribe before initializing simulation so the very first
        # settling contacts cannot be missed by a deferred schema registration.
        sim.initialize_physics()
        if args.capture_every:
            cam=stage.GetPrimAtPath(args.camera)
            if not cam or not cam.IsA(UsdGeom.Camera):raise RuntimeError('Capture camera missing')
            rep.orchestrator.set_capture_on_play(False)
            product=rep.create.render_product(args.camera,(args.width,args.height),name='physical_verification')
            annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach(product)
            report['camera'] = dict(path=args.camera,world_matrix=np.asarray(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist())

        def sync():
            physx.update_transformations(False,True,True,False)

        def capture(phase):
            before = joint_state(stage,records)
            for _ in range(2): sim.render()
            rep.orchestrator.step(rt_subframes=1,delta_time=0.0,pause_timeline=False)
            rgb=np.asarray(annotator.get_data()).copy()
            if rgb.dtype!=np.uint8 or rgb.shape[:2]!=(args.height,args.width):raise RuntimeError('Invalid RGB capture')
            rgb=rgb[:,:,:3]
            file=out/'motion_frames'/f'{len(frames):05d}.png';Image.fromarray(rgb).save(file)
            after=joint_state(stage,records)
            delta=max(abs(before[p]['value']-after[p]['value']) for p in before)
            frames.append(dict(path=str(file.relative_to(out)),sha256=sha(file),phase=phase,step=step_index,
                               simulated_time_seconds=step_index*args.dt,capture_joint_position_max_change=delta,
                               channel_mean=rgb.mean(axis=(0,1)).tolist()))
            if not sim.is_playing():sim.play()

        def sample(phase):
            scan_backend_errors()
            sync();value=joint_state(stage,records)
            if not all(np.isfinite(j['value']) and np.isfinite(j['off_axis_anchor_error_m']) for j in value.values()):
                raise RuntimeError('Non-finite joint state')
            body_positions()
            record=dict(step=step_index,time_seconds=step_index*args.dt,phase=phase,joints=value,drive_targets=dict(active_targets))
            if probe_paths:
                probe_states={}
                for path in probe_paths:
                    prim=stage.GetPrimAtPath(path)
                    matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
                    velocity=np.asarray(UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Get(),dtype=float)
                    if not np.isfinite(matrix).all() or velocity.shape!=(3,) or not np.isfinite(velocity).all():
                        raise RuntimeError('Non-finite or missing native probe state: '+path)
                    probe_states[path]=dict(centre_world=matrix[3,:3].tolist(),linear_velocity_m_s=velocity.tolist())
                record['probes']=probe_states
            states.append(record)
            trajectory_stream.write(json.dumps(record,separators=(',',':'))+'\n')
            return value

        def steps(phase,count):
            nonlocal step_index,active_phase
            active_phase=phase
            status(phase,physics_step=step_index)
            for _ in range(count):
                sim.step(render=False);step_index+=1
                if step_index%10==0:sample(phase)
                if args.capture_every and step_index%args.capture_every==0:capture(phase)
            value=sample(phase)
            trajectory_stream.flush()
            if not args.all_sequential:save(out/'trajectory.json',states)
            save(out/'scene_contact_summary.json',list(scene_contact_summary.values()))
            save(out/'contacts.json',contacts)
            save(out/'capture_frames.json',frames)
            report.setdefault('phase_end_states',{})[phase]=value
            save(out/'status.json',report)
            emit('phase_complete',phase=phase,step=step_index)
            return value

        def command(joints, targets):
            from pxr import Sdf
            with Sdf.ChangeBlock():
                for j in joints:
                    if j['retainer_locked']:raise RuntimeError('Refusing to change a source-locked joint')
                    kind='angular' if j['type']=='revolute' else 'linear'
                    UsdPhysics.DriveAPI(stage.GetPrimAtPath(j['path']),kind).GetTargetPositionAttr().Set(float(targets[j['path']]))
                    active_targets[j['path']]=float(targets[j['path']])
            report['drive_commands'].append(dict(step=step_index,after_phase=active_phase,
                targets={j['path']:float(targets[j['path']]) for j in joints}))
            sim.render()  # Process USD drive edits without advancing physics.

        count=max(1,round(args.phase_seconds/args.dt))
        settled=steps('initial_settle',count)
        settled_bodies=body_positions()
        report['initial_settle']=dict(joint_delta={j['path']:settled[j['path']]['value']-initial[j['path']]['value'] for j in records},
            fixed_base_displacement_m={p:float(np.linalg.norm(np.asarray(settled_bodies[p])-body_initial[p])) for p in fixed_bases},
            free_asset_base_displacement_m={p:float(np.linalg.norm(np.asarray(settled_bodies[p])-body_initial[p])) for p in free_asset_bases},
            rigid_body_count=len(rigid_paths),all_transforms_finite=True)
        report['initial_settle']['uncommanded_joint_displacement_failures'] = [j['path'] for j in all_driven+locked
            if max(abs(s['joints'][j['path']]['value']-initial[j['path']]['value']) for s in states)
                > (5. if j['type']=='revolute' else .025)]
        report['initial_settle']['maximum_sampled_joint_deviation'] = {j['path']:
            max(abs(s['joints'][j['path']]['value']-initial[j['path']]['value']) for s in states) for j in records}
        report['initial_settle']['stable'] = bool(max(report['initial_settle']['fixed_base_displacement_m'].values(),default=0.)<.005
            and not report['initial_settle']['uncommanded_joint_displacement_failures'])
        save(out/'initial_settle.json',report['initial_settle'])
        emit('initial_settle_measured',max_fixed_base_drift_m=max(report['initial_settle']['fixed_base_displacement_m'].values(),default=0.),
             maximum_joint_coordinate_change=max(abs(v) for v in report['initial_settle']['joint_delta'].values()))
        if not report['initial_settle']['stable']:
            raise RuntimeError('Unstable initial settle: fixed base >5 mm or undriven joint >5 deg/25 mm')
        close={j['path']:min(j['upper'],max(j['lower'],0.)) for j in driven}
        hinges=[j for j in driven if j['type']=='revolute'];slides=[j for j in driven if j['type']=='prismatic']
        opens={}
        for j in driven:
            extreme=j['lower'] if abs(j['lower'])>abs(j['upper']) else j['upper']
            appliance=assets[j['asset']]['metadata'].get('asset_type') in ('dishwasher','range_oven')
            opens[j['path']]=extreme*(1.0 if appliance and j['type']=='revolute' else args.open_fraction)
        command(driven,close);closed0=steps('closed_before_cycle',count)
        if args.sequential_joints or args.all_sequential or args.probes_only:
            opened_hinges={};opened_slides={};closed_slides={};closed1={};blocked=[]
            report['appliance_prerequisite_checks']=[]
            for index,item in enumerate(plan):
                joint=item['joint'];prerequisites=item['prerequisite_doors']
                label=f"sequential_{index:02d}_{joint['asset']}_{Path(joint['path']).name}"
                if prerequisites:
                    command(prerequisites,opens);value=steps(label+'_prerequisite_doors_open',count)
                    checks=[dict(door=d['path'],target=opens[d['path']],measured=value[d['path']]['value'],
                        passed=abs(value[d['path']]['value']-opens[d['path']])<=5.) for d in prerequisites]
                    permitted=all(c['passed'] for c in checks)
                    report['appliance_prerequisite_checks'].append(dict(joint=joint['path'],checks=checks,permitted=permitted))
                    if not permitted:
                        blocked.append(joint)
                        command(prerequisites,close);steps(label+'_blocked_prerequisite_doors_closed',count)
                        continue
                command([joint],opens);value=steps(label+'_open',count)
                (opened_hinges if joint['type']=='revolute' else opened_slides)[joint['path']]=value[joint['path']]
                command([joint],close);value=steps(label+'_closed',count)
                (closed1 if joint['type']=='revolute' else closed_slides)[joint['path']]=value[joint['path']]
                if prerequisites:
                    command(prerequisites,close);steps(label+'_prerequisite_doors_closed',count)
            report['blocked_slide_prerequisites']=[j['path'] for j in blocked]
        else:
            command(hinges,opens);opened_hinges=steps('doors_open_before_slides',count)
        # Only slide a dishwasher rack after its parent appliance door actually
        # reached an open position. A failed prerequisite stays in the denominator.
            blocked=[];allowed=[]
            for j in slides:
                doors=[d for d in hinges if d['asset']==j['asset'] and 'door' in d['path'].lower()]
                if 'rack' in j['path'].lower() and any(abs(opened_hinges[d['path']]['value']-opens[d['path']])>5 for d in doors):
                    blocked.append(j)
                else:allowed.append(j)
            report['blocked_slide_prerequisites']=[j['path'] for j in blocked]
            command(allowed,opens);opened_slides=steps('drawers_and_racks_open',count)
            command(allowed,close);closed_slides=steps('drawers_and_racks_closed',count)
            command(hinges,close);closed1=steps('doors_closed',count)
        # Probe support tests are added below, on the same composed collision scene.
        report['support_probes']=[]
        if not args.no_probes:
            # An actual PhysX overlap query checks a small clearance behind a
            # modeled pull. Positive controls intersect the leaf and grip so a
            # missing collider or dead broadphase cannot make empty queries pass.
            handle_body=stage.GetPrimAtPath('/World/Assets/main_left_pair/Links/door_1')
            leaf=stage.GetPrimAtPath('/World/Assets/main_left_pair/Links/door_1/door_1_leaf')
            grip_parts=[p for p in handle_body.GetChildren() if p.IsA(UsdGeom.Mesh) and 'door_1_pull_' in p.GetName()] if handle_body else []
            if leaf and grip_parts:
                cache=UsdGeom.XformCache(Usd.TimeCode.Default())
                body_matrix=np.asarray(cache.GetLocalToWorldTransform(handle_body),dtype=float)
                def in_body(prim):
                    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
                    relative=np.asarray(cache.GetLocalToWorldTransform(prim),dtype=float)@np.linalg.inv(body_matrix)
                    return points@relative[:3,:3]+relative[3,:3]
                leaf_pts=in_body(leaf);grip_pts=np.concatenate([in_body(p) for p in grip_parts])
                low,high=grip_pts.min(0),grip_pts.max(0)
                handle_axis=int(np.argmax((high-low)[[0,2]]))*2
                grip_radius=float(min((high-low)[[0,2]])/2)
                gap_back=float(low[1]+2*grip_radius);leaf_front=float(leaf_pts[:,1].min())
                centre=(low+high)/2;centre[1]=(gap_back+leaf_front)/2
                leaf_control=(leaf_pts.min(0)+leaf_pts.max(0))/2
                grip_control=(low+high)/2;grip_control[1]=low[1]+grip_radius
                requests=[]
                for offset in (-.015,0.,.015):
                    point=centre.copy();point[handle_axis]+=offset;requests.append(('gap',point))
                requests += [('leaf_positive_control',leaf_control),('grip_positive_control',grip_control)]
                checks=[]
                for label,point in requests:
                    world_point=point@body_matrix[:3,:3]+body_matrix[3,:3];hits=[]
                    def overlap_callback(hit):
                        hits.append(dict(collider=str(hit.collision),rigid_body=str(hit.rigid_body)));return True
                    count_hits=query.overlap_sphere(.006,tuple(world_point),overlap_callback,False)
                    checks.append(dict(label=label,centre_world=world_point.tolist(),radius_m=.006,
                        hit_count=int(count_hits),hits=hits))
                controls=all(c['hit_count']>0 for c in checks if c['label']!='gap')
                report['handle_gap_test']=dict(status='queried',body=str(handle_body.GetPath()),
                    nominal_grip_to_leaf_gap_m=leaf_front-gap_back,checks=checks,positive_controls_passed=controls,
                    clearance_for_12mm_sphere=controls and all(c['hit_count']==0 for c in checks if c['label']=='gap'),
                    scope='Modeled collider overlap at three points; not measured human/G1 finger fit or a grasp/contact test.')
            else:report['handle_gap_test']=dict(status='not_available',reason='Expected leaf/pull geometry missing')
            support_candidates=[]
            for target,label in [('/World/Assets/island_countertop/Links/base/top_right','countertop'),
                                 ('/World/Assets/main_left_pair/Links/base/interior_shelf_0','cabinet_shelf')]:
                prim=stage.GetPrimAtPath(target)
                if not prim:continue
                mesh=UsdGeom.Mesh(prim);pts=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
                matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
                xyz=pts@matrix[:3,:3]+matrix[3,:3]
                p=(xyz.min(0)+xyz.max(0))/2;p[2]=xyz[:,2].max()
                support_candidates.append(dict(label=label,expected_collider=target,point=p.tolist(),radius=.018,
                    acceptance_kind='planar_support'))
            sink=next((a for a in manifest['assets'] if a['name']=='island_sink_basin_01'),None)
            if sink:
                # Challenge curved-bowl containment with an off-centre drop.
                # Rolling is expected. The 25 mm sphere has a geometric support
                # margin over the 18 mm drain radius; earlier 18 mm spheres and
                # planar drift/height gates are retained as historical diagnostics.
                drain=stage.GetPrimAtPath(sink['path']+'/Links/base/drain_1_recess')
                if drain and drain.IsA(UsdGeom.Mesh):
                    pts=np.asarray(UsdGeom.Mesh(drain).GetPointsAttr().Get(),dtype=float)
                    mat=np.asarray(UsdGeom.Xformable(drain).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
                    low,high=pts.min(0),pts.max(0);radius=.025
                    drain_radius=float(max((high-low)[:2])/2)
                    drain_centre=(low+high)/2
                    p=drain_centre.copy();p[0]+=drain_radius+radius+.030;p[2]+=.04
                    p=p@mat[:3,:3]+mat[3,:3]
                    bowl=[];opening=[];bowl_colliders=[]
                    for part in drain.GetParent().GetChildren():
                        if part.IsA(UsdGeom.Mesh) and part.GetName().startswith('bowl_1_wall_'):
                            q=np.asarray(UsdGeom.Mesh(part).GetPointsAttr().Get(),dtype=float)
                            pm=np.asarray(UsdGeom.Xformable(part).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
                            bowl.append(q@pm[:3,:3]+pm[3,:3]);bowl_colliders.append(str(part.GetPath()))
                            if part.GetName().startswith('bowl_1_wall_0_'):
                                # First four vertices are the inner drain-neck
                                # face in the source-bound convex shell cells.
                                opening.extend(np.linalg.norm(q[:4,:2]-drain_centre[:2],axis=1).tolist())
                    if not bowl or not opening:raise RuntimeError('Expected right-bowl collision geometry missing')
                    bounds=np.concatenate(bowl)
                    support_candidates.append(dict(label='sink_bowl',expected_asset_prefix=sink['path'],
                        point=p.tolist(),radius=radius,reference_height_from_expected_raycast=True,
                        acceptance_kind='curved_basin_containment',bowl_colliders=bowl_colliders,
                        bowl_bounds_world=[bounds.min(0).tolist(),bounds.max(0).tolist()],
                        drain_opening_radius_m=max(opening),
                        drain_opening_provenance='Authored nominal collider geometry; not a measurement from the video.',
                        sample_definition=dict(drain_mesh=str(drain.GetPath()),drain_radius_m=drain_radius,
                            clearance_between_probe_and_drain_footprints_m=.030,
                            method='Actual drain bounds locate an off-centre curved-floor drop; expected native ray hit sets its height. Rolling is allowed; containment, support margin, contact and final settling are tested.')))
            for index,candidate in enumerate(support_candidates):
                p=np.asarray(candidate['point']);origin=p+np.array([0,0,.075])
                hit=query.raycast_closest(tuple(origin),(0.,0.,-1.),.18)
                candidate['ray_query_origin_world']=origin.tolist()
                candidate['pre_drop_raycast']=dict(hit=bool(hit.get('hit')))
                if hit.get('hit'):
                    candidate['pre_drop_raycast'].update(position=list(hit['position']),normal=list(hit['normal']),
                        distance=float(hit['distance']),collider=str(hit.get('collision',hit.get('collider',''))),
                        rigid_body=str(hit.get('rigidBody','')))
                    if candidate.get('reference_height_from_expected_raycast') and candidate['pre_drop_raycast']['collider'].startswith(candidate['expected_asset_prefix']+'/'):
                        p=np.asarray(hit['position'],dtype=float);candidate['point']=p.tolist();origin=p+np.array([0,0,.075])
                candidate['path']=f'/World/PhysicsVerification/probe_{index}_{candidate["label"]}'
                sphere=UsdGeom.Sphere.Define(stage,candidate['path']);sphere.CreateRadiusAttr(candidate['radius'])
                sphere.AddTranslateOp().Set(Gf.Vec3d(*origin));sphere.CreateDisplayColorAttr([Gf.Vec3f(1.,.06,.16)])
                UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim());UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(.03)
                PhysxSchema.PhysxContactReportAPI.Apply(sphere.GetPrim()).CreateThresholdAttr(0.)
                candidate['initial_center_world']=origin.tolist()
                probe_paths.append(candidate['path'])
            sim.render()
            steps('gravity_support_probes',count)
            for candidate in support_candidates:
                mat=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(candidate['path'])).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),dtype=float)
                end=mat[3,:3];candidate['final_center_world']=end.tolist()
                candidate['expected_height_error_m']=float(end[2]-candidate['radius']-candidate['point'][2])
                candidate['downward_motion_m']=float(candidate['initial_center_world'][2]-end[2])
                matching=[c for c in contacts if candidate['path'] in c['actors']]
                candidate['reported_contact_event_count']=len(matching)
                candidate['contact_colliders']=sorted({p for c in matching for p in c['colliders'] if p!=candidate['path']})
                expected=lambda p:p==candidate['expected_collider'] if 'expected_collider' in candidate else p.startswith(candidate['expected_asset_prefix']+'/')
                candidate['expected_support_contact']=any(expected(p) for p in candidate['contact_colliders'])
                candidate['unexpected_contact_colliders']=[p for p in candidate['contact_colliders'] if not expected(p)]
                candidate['ray_hit_expected_support']=expected(candidate['pre_drop_raycast'].get('collider',''))
                candidate['horizontal_drift_m']=float(np.linalg.norm(end[:2]-np.asarray(candidate['initial_center_world'])[:2]))
                candidate['planar_support_observed']=bool(abs(candidate['expected_height_error_m'])<.015 and candidate['downward_motion_m']>.025
                    and candidate['expected_support_contact'] and candidate['ray_hit_expected_support'] and not candidate['unexpected_contact_colliders']
                    and candidate['horizontal_drift_m']<.05)
                candidate['support_observed']=candidate['planar_support_observed']
                if candidate.get('acceptance_kind')=='curved_basin_containment':
                    terminal=[s['probes'][candidate['path']] for s in states
                        if s['phase']=='gravity_support_probes' and 'probes' in s
                        and s['time_seconds']>=step_index*args.dt-.25-1e-8]
                    positions=np.asarray([s['centre_world'] for s in terminal])
                    velocity=np.asarray(terminal[-1]['linear_velocity_m_s']) if terminal else np.full(3,np.inf)
                    span=float(np.max(np.linalg.norm(positions-end,axis=1))) if len(terminal) else float('inf')
                    low,high=np.asarray(candidate['bowl_bounds_world']);radius=candidate['radius']
                    contained=bool(np.all(end[:2]-radius>=low[:2]) and np.all(end[:2]+radius<=high[:2])
                        and end[2]-radius>=low[2]-.001 and end[2]+radius<high[2])
                    final_hits=[]
                    def final_overlap_callback(hit):
                        if str(hit.collision)!=candidate['path']:final_hits.append(str(hit.collision))
                        return True
                    query.overlap_sphere(radius+.001,tuple(end),final_overlap_callback,False)
                    expected_set=set(candidate['bowl_colliders'])
                    final_expected=sorted({p for p in final_hits if p in expected_set})
                    final_unexpected=sorted({p for p in final_hits if p not in expected_set})
                    latest_step=max((c['step'] for c in matching),default=-1)
                    latest=[c for c in matching if c['step']>=latest_step-10]
                    supporting_samples=[s for c in latest for s in c['samples']
                        if any(p in expected_set for p in c['colliders']) and abs(s['normal'][2])>.2
                        and -.005<=s['separation']<=.003 and s['position'][2]<=end[2]]
                    margin=radius-candidate['drain_opening_radius_m']
                    wrong_bowl_contacts=[p for p in candidate['contact_colliders'] if p not in expected_set]
                    basin_pass=bool(contained and margin>=.005 and len(terminal)>=3 and span<.001
                        and np.linalg.norm(velocity)<.01 and final_expected and not final_unexpected and supporting_samples
                        and candidate['downward_motion_m']>.025 and candidate['expected_support_contact']
                        and candidate['ray_hit_expected_support'] and not candidate['unexpected_contact_colliders'] and not wrong_bowl_contacts)
                    candidate['basin_containment']=dict(passed=basin_pass,sphere_contained_below_rim=contained,
                        radial_margin_over_drain_m=margin,required_radial_margin_m=.005,
                        terminal_window_seconds=.25,terminal_pose_sample_count=len(terminal),terminal_max_displacement_m=span,
                        final_linear_velocity_m_s=velocity.tolist(),final_speed_m_s=float(np.linalg.norm(velocity)),
                        final_overlap_radius_m=radius+.001,overlap_query_radius_expansion_m=.001,final_expected_overlap_colliders=final_expected,
                        final_unexpected_overlap_colliders=final_unexpected,last_reported_contact_step=latest_step,
                        unexpected_bowl_contact_colliders=wrong_bowl_contacts,
                        last_reported_contact_age_seconds=(step_index-latest_step)*args.dt,
                        supported_latest_contact_sample_count=len(supporting_samples),
                        contact_scope='Sleeping bodies need not emit new contact events; final native overlap and sampled settled pose verify persistent support.',
                        acceptance='Containment and settling in a curved basin; horizontal displacement from the drop point is not a stability criterion.')
                    candidate['support_observed']=basin_pass
                report['support_probes'].append(candidate)
        command(driven,{j['path']:j['initial'] for j in driven});restored=steps('restore_source_initial_targets',count)
        outcomes=[]
        for j in records:
            path=j['path'];hist=np.array([s['joints'][path]['value'] for s in states]);angular=j['type']=='revolute'
            tolerance=3. if angular else .018; limit_tol=2. if angular else .012
            base=dict(path=path,asset=j['asset'],type=j['type'],unit=j['unit'],locked=j['retainer_locked'],
                lower_limit=j['lower'],upper_limit=j['upper'],authored_initial=j['initial'],
                observed_min=float(hist.min()),observed_max=float(hist.max()),
                limits_respected=bool(hist.min()>=j['lower']-limit_tol and hist.max()<=j['upper']+limit_tol),
                max_off_axis_anchor_error_m=max(s['joints'][path]['off_axis_anchor_error_m'] for s in states))
            if path in excluded_paths:base.update(status='excluded_passive_or_legacy_mobile_subsystem')
            elif path in unselected_paths:base.update(status='not_selected_for_focused_diagnostic')
            elif j['retainer_locked']:
                error=float(np.max(abs(hist-j['initial'])));base.update(status='locked_checked',max_locked_drift=error,passed=error<(.5 if angular else .002))
            elif j in blocked:
                base.update(status='blocked_by_appliance_door',passed=False)
            else:
                qopen=(opened_hinges if angular else opened_slides)[path]['value']
                qclose=(closed1 if angular else closed_slides)[path]['value']
                movement=abs(qopen-closed0[path]['value']);error_open=abs(qopen-opens[path]);error_close=abs(qclose-close[path])
                error_restore=abs(restored[path]['value']-j['initial'])
                base.update(status='drive_cycle_measured',open_command=opens[path],observed_open=qopen,observed_close=qclose,
                    actual_motion_span=movement,open_target_error=error_open,close_target_error=error_close,
                    close_open_close_residual=abs(qclose-closed0[path]['value']),
                    restored_initial_error=error_restore,
                    passed=bool(movement>(5. if angular else .03) and error_open<tolerance and error_close<tolerance
                        and error_restore<tolerance and base['limits_respected'] and base['max_off_axis_anchor_error_m']<.005))
            outcomes.append(base)
        report['joint_outcomes']=outcomes
        checked=[j for j in outcomes if 'passed' in j]
        report['summary']=dict(manifest_joint_count=len(records),checked_joint_count=len(checked),
            passed_joint_count=sum(j['passed'] for j in checked),failed_joint_count=sum(not j['passed'] for j in checked),
            excluded_joint_count=len(excluded),unselected_joint_count=len(unselected_paths),locked_joint_count=len(locked),
            max_fixed_base_drift_m=max(report['initial_settle']['fixed_base_displacement_m'].values(),default=0.),
            support_probe_count=len(report['support_probes']),support_probe_passed=sum(p['support_observed'] for p in report['support_probes']),
            all_usd_rigid_transforms_finite=True,physics_backend_valid=True,initial_settle_stable=report['initial_settle']['stable'],
            physics_steps=step_index,physics_seconds=step_index*args.dt)
        report['frames']=frames
        save(out/'contacts.json',contacts);save(out/'scene_contact_summary.json',list(scene_contact_summary.values()));save(out/'trajectory.json',states)
        if frames:
            fps=1/(args.dt*args.capture_every)
            command_line=['ffmpeg','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(out/'motion_frames/%05d.png'),
                          '-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(out/'motion.mp4')]
            result=subprocess.run(command_line,capture_output=True,text=True,timeout=30)
            report['movie_encoding']=dict(command=command_line,returncode=result.returncode,stderr=result.stderr[-2000:])
        report['source_files_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in (input_files|source_layers|texture_files).items())
        if not report['source_files_unchanged']:raise RuntimeError('Bound input changed during verification')
        status('completed',summary=report['summary'])
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        emit('failed',error=repr(error),traceback=report['traceback'])
        raise
    finally:
        report['finished_at']=now();report['elapsed_seconds']=time.monotonic()-started
        report['frames']=frames
        trajectory_stream.flush();trajectory_stream.close()
        save(out/'trajectory.json',states)
        save(out/'scene_contact_summary.json',list(scene_contact_summary.values()))
        save(out/'contacts.json',contacts)
        save(out/'capture_frames.json',frames)
        report['source_files_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in (input_files|source_layers|texture_files).items())
        if (out/'kit.log').is_file():
            (out/'backend_log.snapshot.txt').write_bytes((out/'kit.log').read_bytes())
        report['mechanical_subset_passed']=mechanical_verdict(report,not args.no_probes)
        report['probe_acceptance_scope']=('not requested; focused mechanisms only' if args.no_probes else
            'All three named support probes and the controlled 12 mm handle overlap query are required.')
        report['artifacts']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()
                             and 'kit-portable' not in p.parts and p.name not in ('results.json','status.json','kit.log')}
        save(out/'results.json',report);save(out/'status.json',report)
        if app is not None:
            try:
                if annotator is not None and product is not None:annotator.detach(product)
                if product is not None:product.destroy()
            except Exception:pass
            if report['status']=='failed':
                # Kit fast shutdown otherwise exits 0 even while unwinding a
                # Python exception. Carry the verified failure into jobctl.
                app._app.post_uncancellable_quit(1)
            app.close()


if __name__=='__main__':
    main()
