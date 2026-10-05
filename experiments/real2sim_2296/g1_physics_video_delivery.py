#!/usr/bin/env python3
"""Bind native replay media to recorded states and copy an explicitly scoped video."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def review_state_bindings(review_path,source_name,expected):
    """Require one reviewed snapshot to contain the selected raw state triple."""
    review_path=Path(review_path).resolve();review=read(review_path)
    assert review.get('schema')=='redteam-review/v1','Expected an independent review receipt'
    files=review.get('files',{});parents=set()
    for relative in files:
        path=Path(relative)
        if path.name=='rollout.npz' and path.parent.name==source_name:parents.add(path.parent)
    for parent in sorted(parents):
        selected={name:parent/name for name in expected}
        if not all(files.get(str(path))==expected[name] for name,path in selected.items()):continue
        paths={name:(review_path.parent/path).resolve() for name,path in selected.items()}
        if not all(path.is_relative_to(review_path.parent) and path.is_file() and sha(path)==expected[name] for name,path in paths.items()):continue
        return {'review_sha256':sha(review_path),'snapshot_files':{name:{'path':str(path),'sha256':expected[name]} for name,path in paths.items()}}
    raise ValueError('Review does not bind the selected rollout, input receipt and raw receipt: '+str(review_path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--downloads',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--pose-review',type=Path,required=True)
    p.add_argument('--physics-review',type=Path,required=True)
    a=p.parse_args();render=a.render.resolve();out=a.output.resolve()
    if Path(a.name).name!=a.name or not a.name.endswith('.mp4'):raise ValueError('Use a plain MP4 filename')
    if out.exists():raise FileExistsError(out)
    receipt_path=render/'render.receipt.json';r=read(receipt_path)
    assert r['status']=='passed' and r['physical_recording_replay'] and not r['physics_simulated']
    assert all(x['passed'] for x in r['checks'])
    bound={}
    for path,h in r['input_files_sha256'].items():
        assert sha(path)==h,path
        bound[path]=h
    source=Path(r['request']['scene']).parent
    replay_path=Path(r['request']['physics_replay']);replay=read(replay_path)
    assert sha(replay_path)==bound[str(replay_path)]
    copied={name:source/row['path'] for name,row in replay['portable_source'].items()}
    for name,path in copied.items():assert sha(path)==replay['portable_source'][name]['sha256']
    raw_receipt=read(copied['rollout_receipt.json']);inputs=read(copied['rollout_inputs.json'])
    assert raw_receipt['artifacts']['rollout.npz']==sha(copied['rollout.npz'])==replay['source_rollout_sha256']
    assert raw_receipt['inputs_receipt']['sha256']==sha(copied['rollout_inputs.json'])
    expected_review_states={'rollout.npz':sha(copied['rollout.npz']),
        'inputs.json':sha(copied['rollout_inputs.json']),'receipt.json':sha(copied['rollout_receipt.json'])}
    review_joins={name:review_state_bindings(path,Path(replay['source_rollout']).name,expected_review_states)
        for name,path in [('pose',a.pose_review),('physics',a.physics_review)]}
    reset=bool(inputs.get('diagnostic_task_reset'))
    assert reset==bool(inputs['arguments'].get('diagnostic_reset_task_from_reference'))==replay['diagnostic_reset']
    assert raw_receipt['scope']==replay['source_scope']==r['physical_recording_scope']
    # NpzFile decompresses an array on every indexing operation. Cache the
    # arrays used by the complete per-frame audit once, preserving comparisons.
    with np.load(copied['rollout.npz'],allow_pickle=False) as z:
        raw={k:z[k] for k in ['qpos','time']}
    with np.load(source/'inputs/trajectory.npz',allow_pickle=False) as z:
        motion={k:z[k] for k in ['qpos','simulation_time','door_angle','object_pose']}
    with np.load(source/'fk_samples.npz',allow_pickle=False) as z:
        fk={k:z[k] for k in ['qpos','door_angle','body_world']}
    assert sha(source/'inputs/trajectory.npz')==replay['export_trajectory_sha256']
    model=inputs['model'];task=read(source/'inputs/original_task.json')
    addresses=dict(zip(model['robot_body_joint_names'],model['robot_body_qpos_addresses']))
    addresses.update(zip(model['robot_hand_joint_names'],model['robot_hand_qpos_addresses']))
    expected=[dict(name='floating_base_joint',robot_address=0,scene_address=model['root_qpos_address'],width=7)]
    expected.extend(dict(name=name,robot_address=7+i,scene_address=addresses[name],width=1) for i,name in enumerate(task['robot']['joint_names']))
    assert replay['robot_qpos_map']==expected
    coverage=[]
    for row in expected:
        start=row['robot_address'];s=row['scene_address'];width=row['width']
        coverage.extend(range(start,start+width))
        assert np.array_equal(raw['qpos'][:,s:s+width],motion['qpos'][:,start:start+width])
    assert sorted(coverage)==list(range(motion['qpos'].shape[1]))
    assert np.array_equal(raw['time'],motion['simulation_time'])
    assert np.array_equal(fk['qpos'],motion['qpos'])
    assert np.array_equal(fk['door_angle'],motion['door_angle'])
    oq=replay['bottle_qpos_address'];dq=replay['door_qpos_address']
    assert np.array_equal(raw['qpos'][:,oq:oq+7],motion['object_pose'])
    assert np.array_equal(raw['qpos'][:,dq],motion['door_angle'])
    movie=Path(r['movie']['path']);assert sha(movie)==r['movie']['sha256']
    assert sha(render/'samples.jsonl')==r['samples_jsonl_sha256']
    samples=r['samples'];fps=r['render_fps'];stride=round(50/fps)
    assert stride*fps==50 and len(samples)==r['captured_frames']==r['target_frames']
    indices=list(range(0,len(raw['time']),stride));assert r['sample_indices']==indices
    assert len(indices)==len(samples)
    for n,(sample,i) in enumerate(zip(samples,indices)):
        assert sample['frame']==n and sample['authored_sample_index']==i
        assert abs(sample['target_seconds']-i/50)<1e-9
        assert abs(sample['timeline_seconds_after_capture']-i/50)<1e-9
        assert max(sample['max_body_fk_matrix_error'],sample['door_matrix_error'],sample['object_matrix_error'])<1e-7
        assert np.allclose(sample['body_origins_world'],fk['body_world'][i,:,3,:3],rtol=0,atol=1e-7)
        assert np.allclose(sample['object_center_world'],motion['object_pose'][i,:3],rtol=0,atol=1e-7)
        assert abs(sample['door_angle_radians']-motion['door_angle'][i])<1e-7
        assert sha(sample['image']['path'])==sample['image']['sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)]))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert int(video['nb_read_frames'])==len(samples)
    assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p'
    assert abs(float(probe['format']['duration'])-len(samples)/fps)<1e-5
    # Keep portable evidence copies and three decoded frames for visual inspection.
    out.mkdir(parents=True);(out/'evidence').mkdir();(out/'inspection').mkdir()
    copy_sources={'render.receipt.json':receipt_path,'physics_replay.receipt.json':replay_path,
        'pose_review.receipt.json':a.pose_review,'physics_review.receipt.json':a.physics_review,
        'rollout.receipt.json':copied['rollout_receipt.json'],'rollout.inputs.json':copied['rollout_inputs.json'],
        'delivery_helper.py':Path(__file__)}
    for name,path in copy_sources.items():shutil.copy2(path,out/'evidence'/name)
    save(out/'evidence/ffprobe.json',probe)
    shutil.copy2(movie,out/a.name)
    for i in [0,len(samples)//2,len(samples)-1]:
        subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-vf',f'select=eq(n\\,{i})','-frames:v','1','-q:v','2',str(out/'inspection'/f'frame_{i:05d}.jpg')],check=True)
    text=('G1 / Dex3 actual physics recording, rendered with the textured USD scene.\n\n'
          +replay['source_scope']+'\n\n'
          'Robot, door and bottle poses come from the recorded MuJoCo free-base, passive-hinge and free-object states. Rendering does not re-execute physics. '
          'This media check validates the replay and its provenance; complete pickup/carry/placement success is a separate physical evaluation.\n')
    note=a.name[:-4]+'.README.txt';(out/note).write_text(text)
    transfers=[]
    for name in [a.name,note]:
        target=a.downloads/name
        if target.exists() and sha(target)!=sha(out/name):raise FileExistsError('Preserve existing different delivery: '+str(target))
        if not target.exists():shutil.copy2(out/name,target)
        assert sha(target)==sha(out/name)
        transfers.append(dict(path=str(target),sha256=sha(target),size=target.stat().st_size))
    manifest=[dict(path=str(f.relative_to(out)),sha256=sha(f),size=f.stat().st_size) for f in sorted(out.rglob('*')) if f.is_file()]
    save(out/'manifest.json',{'schema':'g1-physics-media-manifest/v1','files':manifest})
    evidence={'schema':'agentic-evidence/v1','run_id':out.name,'produced_at':datetime.now(timezone.utc).isoformat(),
        'claim':'Native video replay of recorded G1/Dex3 physics. '+replay['source_scope'],
        'inputs_sha256':sha(receipt_path),'path':str(a.downloads/a.name),'sha256':sha(movie),
        'status':'passed','diagnostic_reset':reset,'physical_task_success_claim':False,
        'raw_rollout_sha256':sha(copied['rollout.npz']),'pose_review_sha256':sha(a.pose_review),
        'physics_review_sha256':sha(a.physics_review),'all_exported_samples_checked':len(raw['time']),
        'review_state_joins':review_joins,
        'all_rendered_frames_checked':len(samples),'duration_seconds':float(probe['format']['duration']),
        'manifest_sha256':sha(out/'manifest.json'),'downloads':transfers,
        'visual_inspection':'Decoded frames saved; human/agent viewing recorded separately.'}
    save(out/'delivery.receipt.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
