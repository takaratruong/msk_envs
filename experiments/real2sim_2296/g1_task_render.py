#!/usr/bin/env python3
"""Native Isaac rendering of the additive G1/Dex3 task reference export.

The timeline is explicitly positioned for every image. No physics stepping or
simulation context is created. Use an external jobctl process-group timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path)
    p.add_argument('--task-export-receipt', type=Path)
    p.add_argument('--physics-replay',type=Path,help='Optional exact physical-recording replay receipt; changes caption only after all pose bindings pass')
    p.add_argument('--camera-schedule', type=Path, help='JSON array of {time,path}; cameras must already exist in the export')
    p.add_argument('--camera-views',type=Path,help='Optional diagnostic camera definitions, applied only to the render session layer')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--camera', default='/World/PreviewCameras/Fridge')
    p.add_argument('--cut-time', type=float, help='Use the second camera at and after this authored time in seconds')
    p.add_argument('--second-camera', help='One optional hard cut; source camera transforms remain unchanged')
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--times', type=float, nargs='+', help='Static diagnostic times in seconds; no movie is encoded')
    p.add_argument('--max-frames', type=int, default=5000)
    p.add_argument('--subframes', type=int, default=2)
    args = p.parse_args()
    sys.argv = [sys.argv[0]]
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        raise RuntimeError('Unset CUDA_VISIBLE_DEVICES so GPU ordinal remains physical')
    if min(args.width, args.height, args.fps, args.subframes, args.max_frames) <= 0 or args.gpu < 0:
        p.error('Dimensions, FPS, subframes and frame bound must be positive')
    if (args.cut_time is None) != (args.second_camera is None):
        p.error('--cut-time and --second-camera must be supplied together')
    if args.cut_time is not None and (not math.isfinite(args.cut_time) or args.cut_time <= 0 or args.second_camera == args.camera):
        p.error('The cut time must be positive and finite, with a distinct second camera')
    scene_path = args.scene.resolve(strict=True)
    export_path = (args.export_receipt or scene_path.parent / 'export.receipt.json').resolve(strict=True)
    exported = json.loads(export_path.read_text())
    assert exported['status'] == 'passed' and exported['kinematic_preview'] is True and exported['physics_simulated'] is False
    original_root = Path(exported['artifacts'][0]['path']).parent
    current_root = scene_path.parent
    bound_files = {str(export_path): sha(export_path)}
    task_export_path=(args.task_export_receipt or current_root/'pickplace_export.receipt.json').resolve(strict=True)
    task_export=json.loads(task_export_path.read_text())
    assert task_export['status']=='passed' and task_export['physics_executed'] is False
    assert exported['source_scene_sha256']==task_export['source_scene']['sha256']
    assert sha(current_root/'motion/export.receipt.json')==task_export['inherited_export_receipt']['sha256']==sha(export_path)
    assert task_export['samples']==exported['sample_count']
    bound_files[str(task_export_path)]=sha(task_export_path)
    for record in task_export['artifacts']:
        local=current_root/Path(record['path']).name
        assert sha(local)==record['sha256'], str(local)
        bound_files[str(local)]=record['sha256']
    for name,record in zip(['original_task.json','trajectory.npz','adapter.receipt.json'],task_export['inputs'][:3]):
        local=current_root/'inputs'/name;assert sha(local)==record['sha256'], str(local)
        bound_files[str(local)]=record['sha256']
    schedule=json.loads(args.camera_schedule.read_text()) if args.camera_schedule else None
    if schedule is not None:
        assert schedule and schedule[0]['time']==0 and all(isinstance(x['path'],str) for x in schedule)
        assert all(float(x['time'])<float(y['time']) for x,y in zip(schedule,schedule[1:]))
        assert args.cut_time is None and args.second_camera is None
        args.camera=schedule[0]['path'];bound_files[str(args.camera_schedule.resolve())]=sha(args.camera_schedule)
    camera_paths=list(dict.fromkeys(x['path'] for x in schedule)) if schedule else [args.camera]+([args.second_camera] if args.second_camera else [])
    camera_views=json.loads(args.camera_views.read_text()) if args.camera_views else []
    if args.camera_views:bound_files[str(args.camera_views.resolve())]=sha(args.camera_views)

    def remap(path):
        return current_root / 'motion' / Path(path).relative_to(original_root)

    for record in exported['artifacts']:
        q = remap(record['path'])
        assert sha(q) == record['sha256'], str(q)
        bound_files[str(q)] = record['sha256']
    for record in exported['environment_files']:
        q = remap(record['copy']['path'])
        assert sha(q) == record['copy']['sha256'] == record['source']['sha256'], str(q)
        bound_files[str(q)] = record['copy']['sha256']
    replay=None
    if args.physics_replay:
        import numpy as np
        replay_path=args.physics_replay.resolve(strict=True)
        replay=json.loads(replay_path.read_text())
        assert replay['schema']=='g1-physical-pose-replay-export/v1' and replay['status']=='passed'
        assert replay['task_export_receipt_sha256']==sha(task_export_path)
        assert replay['inherited_export_receipt_sha256']==sha(export_path)
        assert replay['source_scene_sha256']==exported['source_scene_sha256']
        assert replay['source_samples']==exported['sample_count'] and replay['source_sample_rate_hz']==exported['fps']==50
        bound_files[str(replay_path)]=sha(replay_path)
        portable={}
        for name,record in replay['portable_source'].items():
            path=(current_root/record['path']).resolve(strict=True)
            assert sha(path)==record['sha256']
            bound_files[str(path)]=record['sha256'];portable[name]=path
        raw_receipt=json.loads(portable['rollout_receipt.json'].read_text())
        recorded_inputs=json.loads(portable['rollout_inputs.json'].read_text())
        assert raw_receipt['artifacts']['rollout.npz']==sha(portable['rollout.npz'])==replay['source_rollout_sha256']
        assert raw_receipt['inputs_receipt']['sha256']==sha(portable['rollout_inputs.json'])
        recorded_reset=bool(recorded_inputs.get('diagnostic_task_reset'))
        assert recorded_reset==bool(recorded_inputs['arguments'].get('diagnostic_reset_task_from_reference'))
        assert replay['diagnostic_reset']==recorded_reset and replay['source_scope']==raw_receipt['scope'],'Replay labels differ from the recorded scope'
        actual=np.load(portable['rollout.npz'],allow_pickle=False)
        animation=np.load(current_root/'inputs/trajectory.npz',allow_pickle=False)
        assert sha(current_root/'inputs/trajectory.npz')==replay['export_trajectory_sha256']
        assert np.array_equal(actual['time'],animation['simulation_time'])
        recorded_model=recorded_inputs['model']
        scene_addresses=dict(zip(recorded_model['robot_body_joint_names'],recorded_model['robot_body_qpos_addresses']))
        scene_addresses.update(zip(recorded_model['robot_hand_joint_names'],recorded_model['robot_hand_qpos_addresses']))
        source_task=json.loads((current_root/'inputs/original_task.json').read_text())
        expected_map=[dict(name='floating_base_joint',robot_address=0,scene_address=recorded_model['root_qpos_address'],width=7)]
        expected_map.extend(dict(name=name,robot_address=7+i,scene_address=scene_addresses[name],width=1)
            for i,name in enumerate(source_task['robot']['joint_names']))
        assert replay['robot_qpos_map']==expected_map,'Replay requires the exact named robot-to-scene coordinate map'
        coverage=[]
        for mapping in replay['robot_qpos_map']:
            r,s,w=[mapping[k] for k in ['robot_address','scene_address','width']]
            assert w in (1,7) and r>=0 and s>=0 and s+w<=actual['qpos'].shape[1]
            coverage.extend(range(r,r+w))
            assert np.array_equal(actual['qpos'][:,s:s+w],animation['qpos'][:,r:r+w])
        assert sorted(coverage)==list(range(animation['qpos'].shape[1])),'Replay robot coordinates must be covered exactly once'
        oq=replay['bottle_qpos_address'];dq=replay['door_qpos_address']
        assert np.array_equal(actual['qpos'][:,oq:oq+7],animation['object_pose'])
        assert np.array_equal(actual['qpos'][:,dq],animation['door_angle'])
    if scene_path != current_root / 'scene.usda':
        raise ValueError('Render the combined scene.usda emitted by the exporter')
    source_fps = float(exported['fps'])
    if args.cut_time is not None and args.cut_time > float(exported['duration_seconds']):
        p.error('The cut time must occur within the authored trajectory')
    if args.times is None:
        stride = round(source_fps / args.fps)
        if stride < 1 or abs(stride * args.fps - source_fps) > 1e-8:
            raise ValueError('Render FPS must divide the authored FPS, e.g.30 or15 from30 Hz')
        indices = list(range(0, exported['sample_count'], stride))
    else:
        indices = [round(t * source_fps) for t in args.times]
        if any(abs(i / source_fps - t) > 1e-7 for i, t in zip(indices, args.times)):
            raise ValueError('Diagnostic times must coincide with authored FK sample times')
    if not indices or len(indices) > args.max_frames or min(indices) < 0 or max(indices) >= exported['sample_count']:
        raise ValueError('Requested times/count exceed the authored trajectory or render bound')
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError('Capture times must be unique and increasing')

    def requested_camera(seconds):
        if schedule is not None:
            return [x['path'] for x in schedule if float(x['time'])<=seconds+1e-9][-1]
        return args.second_camera if args.cut_time is not None and seconds >= args.cut_time - 1e-9 else args.camera

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    report = {'schema': 'real2sim-g1-task-reference-render/v1', 'status': 'starting',
        'started_at': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': sha(__file__),
        'source_scene_sha256': exported['source_scene_sha256'], 'export_receipt_sha256': sha(export_path),
        'task_export_receipt_sha256':sha(task_export_path),'camera_schedule':schedule,'session_camera_definitions':camera_views,
        'input_files_sha256': bound_files, 'request': vars(args) | {'scene': str(scene_path),
            'export_receipt': str(export_path), 'output_dir': str(output)},
        'kinematic_preview': True, 'physics_simulated': False, 'robot_contact_validated': False,
        'physical_recording_replay':replay is not None,
        'physical_recording_scope':replay['source_scope'] if replay else None,
        'metric_accuracy_verified': False, 'capture_mode': 'static_diagnostics' if args.times else 'uniform_timeline_movie',
        'authored_fps': source_fps, 'render_fps': args.fps, 'sample_indices': indices,
        'camera_cut': {'requested_seconds': args.cut_time, 'first_camera': args.camera,
            'second_camera': args.second_camera,
            'policy': 'One hard cut on the first captured authored sample at or after the requested time'},
        'simulation_policy': 'All source rigid bodies, joints and collisions disabled in preview overlay; no integration calls',
        'caption_policy': 'Documentary label added to the bottom48 image rows; source scene is untouched'}
    # Paths in the request must remain JSON serializable.
    report['request'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['request'].items()}
    app = product = annotator = None
    started = time.monotonic()
    samples = []

    def status(phase, **values):
        report.update(status=phase, elapsed_seconds=time.monotonic() - started, **values)
        save(output / 'status.json', report)
        print(json.dumps({'phase': phase, 'elapsed_seconds': report['elapsed_seconds'], **values}), flush=True)

    try:
        status('initializing_isaac')
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=args.gpu, physics_gpu=args.gpu,
            multi_gpu=False, width=args.width, height=args.height, renderer='RaytracedLighting',
            sync_loads=True, disable_viewport_updates=True, anti_aliasing=3, fast_shutdown=True,
            enable_crashreporter=False, extra_args=['--portable-root', str(output / 'kit-portable'),
                '--/app/settings/persistent=false', '--/app/telemetry/enabled=false',
                '--/log/file=' + str(output / 'kit.log')]))
        import carb.settings
        import numpy as np
        from scipy.spatial.transform import Rotation
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
        context = omni.usd.get_context()
        assert context.open_stage(str(scene_path))
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        for item in camera_views:
            assert item['name'].isalnum() and len(item['position'])==len(item['target'])==3
            path='/World/PreviewCameras/'+item['name'];assert not stage.GetPrimAtPath(path),'Diagnostic camera may not replace an existing camera'
            camera=UsdGeom.Camera.Define(stage,path);camera.CreateFocalLengthAttr(float(item.get('focal_length',35)))
            camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(20.25);camera.CreateClippingRangeAttr(Gf.Vec2f(.03,100.))
            view=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*item['position']),Gf.Vec3d(*item['target']),Gf.Vec3d(0,0,1)).GetInverse()
            parent=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/PreviewCameras'))
            camera.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(view*parent.GetInverse())
        settings = carb.settings.get_settings()
        settings.set('/app/player/playSimulations', False)
        requested_settings = dict(stage.GetRootLayer().customLayerData.get('renderSettings', {}))
        for key, value in requested_settings.items():
            settings.set('/' + key.replace(':', '/').lstrip('/'), value)
        settings.set('/rtx/rendermode', 'RaytracedLighting')
        report['render_settings'] = requested_settings | {'rtx:rendermode': 'RaytracedLighting'}
        report['gpu_settings'] = {key: settings.get(key) for key in ('/renderer/activeGpu', '/renderer/multiGpu/enabled', '/physics/cudaDevice')}
        report['cameras'] = {}
        for camera_path in camera_paths:
            camera_prim = stage.GetPrimAtPath(camera_path)
            assert camera_prim and camera_prim.IsA(UsdGeom.Camera), camera_path
            camera = UsdGeom.Camera(camera_prim)
            report['cameras'][camera_path] = {'path': camera_path,
                'world_matrix': np.asarray(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist(),
                'focal_length': camera.GetFocalLengthAttr().Get(),
                'horizontal_aperture': camera.GetHorizontalApertureAttr().Get(),
                'vertical_aperture': camera.GetVerticalApertureAttr().Get()}
        report['camera'] = report['cameras'][args.camera]
        report['camera_switches'] = []
        enabled_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()]
        enabled_joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(p).GetJointEnabledAttr().Get()]
        enabled_colliders = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        assert not enabled_bodies and not enabled_joints and not enabled_colliders
        report['physics_disabled_readback'] = {'enabled_bodies': enabled_bodies, 'enabled_joints': enabled_joints, 'enabled_colliders': enabled_colliders}
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_auto_update(False)
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(args.camera, (args.width, args.height), name='g1_kinematic_preview')
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
        camera_relation = stage.GetPrimAtPath(product.path).GetRelationship('camera')
        assert [str(x) for x in camera_relation.GetTargets()] == [args.camera]
        active_camera = args.camera
        for _ in range(12):
            app.update()
        fk = np.load(current_root / 'fk_samples.npz', allow_pickle=False)
        task_poses=np.load(current_root/'inputs/trajectory.npz',allow_pickle=False)
        assert len(task_poses['object_pose'])==exported['sample_count']
        font_size = max(12, min(20, args.width // 60))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
        stream = (output / 'samples.jsonl').open('w')
        for frame, index in enumerate(indices):
            target_time = float(fk['time'][index])
            camera_path = requested_camera(target_time)
            switched = camera_path != active_camera
            if switched:
                prior_camera = active_camera
                camera_relation.SetTargets([Sdf.Path(camera_path)])
                active_camera = camera_path
                report['camera_switches'].append({'capture_frame': frame, 'authored_sample_index': index,
                    'seconds': target_time, 'from_camera': prior_camera, 'to_camera': camera_path})
            timeline.pause()
            timeline.set_auto_update(False)
            timeline.set_current_time(target_time)
            settings.set('/app/player/playSimulations', False)
            for _ in range(2):
                app.update()
            before = float(timeline.get_current_time())
            camera_before = [str(x) for x in camera_relation.GetTargets()]
            if camera_before != [camera_path]:
                raise RuntimeError(f'Render product camera did not switch: {camera_before} versus {camera_path}')
            if abs(before - target_time) > 1e-6:
                raise RuntimeError(f'Timeline did not seek to requested time: {before} versus {target_time}')
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0., pause_timeline=True)
            after = float(timeline.get_current_time())
            camera_after = [str(x) for x in camera_relation.GetTargets()]
            if camera_after != [camera_path]:
                raise RuntimeError(f'Capture changed the requested camera: {camera_after} versus {camera_path}')
            if abs(after - target_time) > 1e-6:
                raise RuntimeError(f'Capture changed the requested timeline time: {after} versus {target_time}')
            data = np.asarray(annotator.get_data()).copy()
            if data.dtype != np.uint8 or data.shape[:2] != (args.height, args.width) or data.shape[2] not in (3, 4):
                raise RuntimeError(f'Invalid RGB capture {data.shape}/{data.dtype}')
            time_code = after * stage.GetTimeCodesPerSecond()
            cache = UsdGeom.XformCache(time_code)
            body_matrices = []
            for body in exported['bodies']:
                body_matrices.append(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body['path']))))
            body_matrices = np.asarray(body_matrices)
            body_error = float(np.max(np.abs(body_matrices - fk['body_world'][index])))
            door_matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(exported['door']['body_path'])))
            door_error = float(np.max(np.abs(door_matrix - fk['door_world'][index])))
            if max(body_error, door_error) > 1e-7:
                raise RuntimeError(f'USD samples do not match exported FK: {body_error}/{door_error}')
            object_pose=task_poses['object_pose'][index];object_expected=np.eye(4)
            object_expected[:3,:3]=Rotation.from_quat(object_pose[[4,5,6,3]]).as_matrix().T
            object_expected[3,:3]=object_pose[:3]
            object_matrix=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/TaskBottle')))
            object_error=float(np.max(np.abs(object_matrix-object_expected)))
            if object_error>1e-7:raise RuntimeError('Rendered bottle pose differs from exported reference')
            image = Image.fromarray(data[:, :, :3])
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, args.height - 48, args.width, args.height), fill=(13, 20, 29))
            label='G1 / Dex3 actual MuJoCo physics replay | native RTX' if replay else 'G1 / Dex3 kinematic reference | prescribed bottle pose | native RTX'
            if replay and replay['diagnostic_reset']:label+=' | isolated reset diagnostic'
            draw.text((12, args.height - 43),label,font=font,fill='white')
            clock_label=f'physics time {float(task_poses["simulation_time"][index]):.3f} s' if replay else f'USD time {after:.3f} s'
            draw.text((12, args.height - 22), f'{camera_path.rsplit("/", 1)[-1]} | {clock_label} | frame {index} at {source_fps:g} Hz | door {math.degrees(float(fk["door_angle"][index])):.1f} deg | nominal scale', font=font, fill=(199, 219, 235))
            path = output / 'frames' / f'{frame:05d}.png'
            image.save(path)
            row = {'frame': frame, 'authored_sample_index': index, 'target_seconds': target_time,
                'timeline_seconds_before_capture': before, 'timeline_seconds_after_capture': after,
                'camera_path': camera_path, 'camera_relation_before_capture': camera_before,
                'camera_relation_after_capture': camera_after, 'camera_switched_before_capture': switched,
                'measured_usd_time_code': time_code, 'max_body_fk_matrix_error': body_error,
                'door_matrix_error': door_error, 'door_angle_radians': float(fk['door_angle'][index]),
                'object_matrix_error':object_error,'object_center_world':object_matrix[3,:3].tolist(),
                'body_origins_world': body_matrices[:, 3, :3].tolist(),
                'image': {'path': str(path), 'sha256': sha(path), 'rgb_mean': data[:, :, :3].mean(axis=(0, 1)).tolist()}}
            samples.append(row)
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            if frame % 15 == 0 or frame == len(indices) - 1:
                stream.flush()
                status('capturing', captured_frames=len(samples), target_frames=len(indices), timeline_seconds=after)
        stream.close()
        save(output / 'samples.json', samples)
        if args.times is None:
            movie = output / 'motion.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', str(args.fps),
                '-i', str(output / 'frames/%05d.png'), '-c:v', 'libx264', '-crf', '19',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)], check=True, capture_output=True, timeout=90)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format',
                '-of', 'json', str(movie)], check=True, capture_output=True, text=True, timeout=30)
            decoded = json.loads(probe.stdout)
            save(output / 'ffprobe.json', decoded)
            video = next(s for s in decoded['streams'] if s['codec_type'] == 'video')
            assert int(video['nb_read_frames']) == len(indices) and video['width'] == args.width and video['height'] == args.height
            assert video['avg_frame_rate'] == str(args.fps) + '/1'
            report['movie'] = {'path': str(movie), 'sha256': sha(movie), 'decoded_frames': int(video['nb_read_frames']),
                'duration_seconds': float(video['duration']), 'fps': args.fps,
                'timing_note': 'Each captured pose is displayed for one movie frame; the final pose therefore adds one display interval.'}
        report['samples'] = samples
        report['samples_jsonl_sha256'] = sha(output / 'samples.jsonl')
        report['checks'] = [
            {'name': 'all_requested_frames_captured', 'passed': len(samples) == len(indices)},
            {'name': 'timeline_positions_match_requests', 'passed': all(abs(s['target_seconds'] - s['timeline_seconds_after_capture']) < 1e-6 for s in samples)},
            {'name': 'all_captured_usd_poses_match_fk', 'passed': all(max(s['max_body_fk_matrix_error'], s['door_matrix_error']) < 1e-7 for s in samples)},
            {'name': 'all_captured_object_poses_match_reference','passed':all(s['object_matrix_error']<1e-7 for s in samples)},
            {'name': 'preview_physics_disabled', 'passed': not enabled_bodies and not enabled_colliders and not enabled_joints},
            {'name': 'all_bound_inputs_unchanged', 'passed': all(sha(p) == h for p, h in bound_files.items())},
            {'name': 'native_realtime_render_mode', 'passed': settings.get('/rtx/rendermode') == 'RaytracedLighting'},
            {'name': 'all_render_product_cameras_match_schedule', 'passed': all(
                s['camera_path'] == requested_camera(s['target_seconds']) and
                s['camera_relation_before_capture'] == s['camera_relation_after_capture'] == [s['camera_path']]
                for s in samples)},
        ]
        report['limitations'] = [('Displays recorded torque-simulation poses; this renderer does not re-execute physics or certify task completion.' if replay else
            'Kinematic FK/timeline preview; no dynamic balance, force, collision response or physical grasp validation.'),
            'The source environment retains inferred nominal dimensions.',
            'Robot material display colors are exported from MuJoCo, with documented legacy-to-USD shading conversion.',
            'Documentary captions cover the bottom48 image rows.']
        assert all(c['passed'] for c in report['checks'])
        status('passed', captured_frames=len(samples))
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = time.monotonic() - started
        report['samples'] = samples
        save(output / 'render.receipt.json', report)
        save(output / 'status.json', report)
        if app is not None:
            try:
                if annotator is not None and product is not None:
                    annotator.detach(product)
                if product is not None:
                    product.destroy()
            finally:
                if report['status'] != 'passed':
                    app._app.post_uncancellable_quit(1)
                app.close()


if __name__ == '__main__':
    main()
