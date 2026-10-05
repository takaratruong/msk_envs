#!/usr/bin/env python3
"""Assemble complete, nonoverlapping native RTX chunks of one physical replay."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,shutil,subprocess
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renders',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    records=[];bound={str(Path(__file__).resolve()):sha(__file__)}
    for directory in a.renders:
        directory=directory.resolve();path=directory/'render.receipt.json';r=read(path)
        assert r['status']=='passed' and r['physical_recording_replay'] and not r['physics_simulated']
        assert r['checks'] and all(c['passed'] for c in r['checks'])
        request=r['request'];scene=Path(request['scene']).resolve()
        critical={'scene':scene,'physics_replay':Path(request['physics_replay']).resolve(),
            'export_receipt':Path(request['export_receipt'] or scene.parent/'export.receipt.json').resolve(),
            'task_export_receipt':Path(request['task_export_receipt'] or scene.parent/'pickplace_export.receipt.json').resolve()}
        for role,source in critical.items():
            assert str(source) in r['input_files_sha256'],role+' missing from this chunk input bindings'
            assert r['input_files_sha256'][str(source)]==sha(source),role
        for source,digest in r['input_files_sha256'].items():
            assert sha(source)==digest,source
            assert source not in bound or bound[source]==digest
            bound[source]=digest
        assert sha(directory/'samples.jsonl')==r['samples_jsonl_sha256']
        lines=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
        assert lines==r['samples'] and len(lines)==r['captured_frames']==r['target_frames']
        assert [s['authored_sample_index'] for s in lines]==r['sample_indices']
        for n,s in enumerate(lines):
            assert s['frame']==n and sha(s['image']['path'])==s['image']['sha256']
        bound[str(path)]=sha(path);bound[str(directory/'samples.jsonl')]=sha(directory/'samples.jsonl')
        records.append((directory,r))
    first=records[0][1];source=Path(first['request']['scene']).parent
    replay_path=Path(first['request']['physics_replay']);replay=read(replay_path)
    assert replay['status']=='passed' and first['authored_fps']==50
    raw_source=source/replay['portable_source']['rollout.npz']['path']
    assert sha(raw_source)==replay['source_rollout_sha256']==replay['portable_source']['rollout.npz']['sha256']
    with np.load(raw_source,allow_pickle=False) as z:native_count=len(z['time'])
    fps=first['render_fps'];stride=round(50/fps);assert stride>0 and stride*fps==50
    equal_fields=['authored_fps','render_fps','source_scene_sha256','export_receipt_sha256',
        'task_export_receipt_sha256','camera_schedule','session_camera_definitions',
        'render_settings','caption_policy','physical_recording_scope','implementation_sha256']
    samples=[]
    for directory,r in records:
        for k in equal_fields:assert r[k]==first[k],k
        for k in ['scene','physics_replay','width','height','subframes','camera','cut_time','second_camera']:
            assert r['request'][k]==first['request'][k],k
        for s in r['samples']:
            item=copy.deepcopy(s);item['source_capture']={
                'receipt':str(directory/'render.receipt.json'),'receipt_sha256':sha(directory/'render.receipt.json'),
                'frame':s['frame'],'image_path':s['image']['path']}
            samples.append(item)
    samples.sort(key=lambda s:s['authored_sample_index'])
    expected=list(range(0,native_count,stride))
    assert [s['authored_sample_index'] for s in samples]==expected,'Incomplete, duplicate, or off-grid render coverage'
    out.mkdir(parents=True);(out/'frames').mkdir()
    for n,(s,i) in enumerate(zip(samples,expected)):
        assert abs(s['target_seconds']-i/50)<1e-9 and abs(s['timeline_seconds_after_capture']-i/50)<1e-9
        assert max(s['max_body_fk_matrix_error'],s['door_matrix_error'],s['object_matrix_error'])<1e-7
        path=out/'frames'/f'{n:05d}.png';shutil.copyfile(s['image']['path'],path)
        assert sha(path)==s['image']['sha256'];s['frame']=n;s['image']['path']=str(path)
    with (out/'samples.jsonl').open('w') as f:
        for s in samples:f.write(json.dumps(s,separators=(',',':'))+'\n')
    save(out/'samples.json',samples)
    movie=out/'motion.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate',str(fps),
        '-i',str(out/'frames/%05d.png'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(movie)],check=True,timeout=180)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
        '-show_format','-of','json',str(movie)],timeout=30));save(out/'ffprobe.json',probe)
    v=next(x for x in probe['streams'] if x['codec_type']=='video')
    assert int(v['nb_read_frames'])==len(samples) and v['avg_frame_rate']==str(fps)+'/1'
    assert v['width']==first['request']['width'] and v['height']==first['request']['height']
    assert abs(float(v['duration'])-len(samples)/fps)<1e-5
    r=copy.deepcopy(first);r.update(schema='g1-physical-native-render-assembly/v1',status='passed',
        input_files_sha256=bound,implementation_sha256=sha(__file__),sample_indices=expected,
        captured_frames=len(samples),target_frames=len(samples),samples=samples,
        samples_jsonl_sha256=sha(out/'samples.jsonl'),
        capture_mode='Assembly of bound native RTX frame chunks; no pose interpolation or physics integration.',
        source_render_receipts=[{'path':str(d/'render.receipt.json'),'sha256':sha(d/'render.receipt.json')} for d,_ in records],
        movie={'path':str(movie),'sha256':sha(movie),'decoded_frames':len(samples),
            'duration_seconds':float(v['duration']),'fps':fps},
        finished_at=datetime.now(timezone.utc).isoformat())
    r['request'].update(output_dir=str(out),times=None)
    r['checks']=[{'name':'all_source_render_checks_pass','passed':True},
        {'name':'complete_unique_on_grid_native_pose_coverage','passed':True},
        {'name':'all_frame_bytes_and_source_input_hashes_match','passed':True},
        {'name':'movie_decode_timing_resolution_and_count_match','passed':True}]
    save(out/'render.receipt.json',r)
    print(json.dumps({'frames':len(samples),'fps':fps,'movie':str(movie),'sha256':sha(movie),
        'receipt_sha256':sha(out/'render.receipt.json')}))


if __name__=='__main__':main()
