#!/usr/bin/env python3
"""Post-run provenance audit for the existing dense_stable workspace.

Read-only except for the requested receipt. No matching, fusion, reconstruction,
camera adjustment, or input rewriting is performed. A post-run receipt verifies
current consistency and observed timestamps; it does not invent a preflight
capture or establish metric geometry accuracy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:raise AuditError(message)


def digest(value):
    # Exact appearance.py digest convention (including default JSON separators).
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def timestamp_ns(value):
    return round(datetime.fromisoformat(value).timestamp()*1_000_000_000)


def stable_stat(path):
    s=path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def record(path):
    path=Path(path).resolve();before=stable_stat(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    require(before==stable_stat(path),f'File changed while hashing: {path}')
    return dict(path=str(path),sha256=h.hexdigest(),bytes=before[2],mtime_ns=before[3],ctime_ns=before[4])


def load_json(path):
    rec=record(path)
    data=json.loads(Path(path).read_bytes())
    require(record(path)['sha256']==rec['sha256'],f'JSON changed while reading: {path}')
    return data,rec


def verify_manifest(manifest):
    unsigned={k:v for k,v in manifest.items() if k!='inputs_sha256'}
    require(digest(unsigned)==manifest.get('inputs_sha256'),'Appearance input manifest digest mismatch')


def input_predates_job(rec,started_ns):
    require(rec['mtime_ns']<=started_ns and rec['ctime_ns']<=started_ns,
            f'Prepared input was modified after dense job started: {rec["path"]}')


def output_within_job(rec,started_ns,finished_ns):
    require(started_ns<=rec['mtime_ns']<=finished_ns and started_ns<=rec['ctime_ns']<=finished_ns,
            f'Dense output timestamps are outside owned job interval: {rec["path"]}')


def resolve_command_path(path,cwd):
    p=Path(path)
    return (p if p.is_absolute() else cwd/p).resolve()


def flag(argv,name):
    require(argv.count(name)==1,f'Expected exactly one {name} in owned command')
    i=argv.index(name);require(i+1<len(argv),f'Missing command value after {name}')
    return argv[i+1]


def read_job(run,attempt_arg):
    pointer,pointer_rec=load_json(run/'jobs'/'dense_stable'/'current.json')
    attempt=Path(pointer['attempt']).resolve() if attempt_arg is None else attempt_arg.resolve()
    require(attempt.parent==(run/'jobs'/'dense_stable').resolve(),'Attempt does not belong to this dense_stable job')
    require(Path(pointer['attempt']).resolve()==attempt,'Current owned job points to a different attempt')
    config,config_rec=load_json(attempt/'config.json');state,state_rec=load_json(attempt/'status.json')
    require(Path(config['attempt']).resolve()==attempt and Path(state['attempt']).resolve()==attempt,'Attempt record mismatch')
    require(state['command']==config['command'] and state['original_command']==config['original_command'],'Owned command changed between config/status')
    require(pointer['worker_identity']==state['worker_identity'],'Worker identity does not match owned job pointer')
    command=config['command'];cwd=Path(config['cwd']).resolve()
    require('dense' in command,'Owned command is not dense reconstruction')
    require(resolve_command_path(flag(command,'--run'),cwd)==run,'Owned dense command targets another run')
    script_entries=[(name,sha) for name,sha in config['script_sha256'].items() if Path(name).name=='pipeline.py']
    require(len(script_entries)==1,'Exactly one archived pipeline hash is required')
    archived=attempt/'pipeline.py';archived_rec=record(archived)
    require(archived_rec['sha256']==script_entries[0][1],'Archived pipeline does not match launch hash')
    require(str(archived.resolve()) in [str(resolve_command_path(x,cwd)) for x in command if x.endswith('.py')],
            'Dense command does not execute the hashed pipeline archive')
    require('started_at' in state,'Owned job never started')
    started_ns=timestamp_ns(state['started_at'])
    input_predates_job(archived_rec,started_ns)
    return dict(attempt=str(attempt),pointer=pointer_rec,config=config_rec,status=state_rec,pipeline=archived_rec,
                command=command,original_command=config['original_command'],cwd=str(cwd),
                physical_gpus=config.get('gpu'),worker_identity=state['worker_identity'],started_at=state['started_at'],
                started_ns=started_ns,finished_at=state.get('finished_at'),finished_ns=timestamp_ns(state['finished_at']) if state.get('finished_at') else None,
                state=state['status'],returncode=state.get('returncode'),model_name=flag(command,'--model-name'),max_size=int(flag(command,'--max-size')))


def config_names(path,training,patch_match=False):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    require(not patch_match or len(lines)%2==0,'Incomplete patch-match configuration pair')
    refs=lines[::2] if patch_match else lines
    require(len(refs)==len(set(refs)) and set(refs)==training,f'Configuration reference views differ from train: {path}')
    if patch_match:
        for sources in lines[1::2]:
            names=[x.strip() for x in sources.split(',')]
            if names[0]=='__auto__':
                require(len(names)==2 and int(names[1])>0,'Invalid automatic source-view specification')
            else:require(set(names)<=training,'Patch-match source includes a nontraining view')
    return refs


def audit_inputs(run,manifest_path,job,expected_count):
    import pycolmap
    data,dataset_rec=load_json(run/'dataset.json');manifest,manifest_rec=load_json(manifest_path)
    frames,frames_rec=load_json(run/'frames.json');verify_manifest(manifest)
    require(data['schema']=='real2sim-dataset/v1','Unknown dataset schema')
    require(dataset_rec['sha256']==manifest['dataset_sha256'],'Current dataset differs from frozen appearance input manifest')
    require(Path(manifest['dataset_path']).resolve()==run/'dataset.json','Appearance manifest targets another dataset')
    require(data['source_sha256']==manifest['source_sha256']==frames['source_sha256'],'Source video SHA differs between inputs')
    source_rec=record(frames['source_path']);require(source_rec['sha256']==data['source_sha256'],'Source video bytes differ from declared SHA')
    train=data['train'];names=[v['name'] for v in train];training=set(names)
    require(len(names)==len(training)==expected_count,'Unexpected training view count or duplicate training names')
    require(training.isdisjoint({v['name'] for v in data['val']}),'Training/validation overlap')
    original={v['name']:v for v in frames['frames']}
    require(all(original[n]['split']=='train' for n in names),'Dense view is held out or in buffer')
    manifest_train={v['name']:v for v in manifest['images'] if v['split']=='train'}
    require(len(manifest_train)==sum(v['split']=='train' for v in manifest['images']) and set(manifest_train)==training,
            'Frozen manifest training names differ or repeat')
    require({p.name for p in (run/'dense'/'images').iterdir() if p.is_file()}==training,'Dense image directory includes missing or extra views')
    require({p.name for p in (run/'dense'/'masks').iterdir() if p.is_file()}=={n+'.png' for n in names},'Dense mask directory includes missing or extra views')
    image_records=[]
    for view in train:
        name=view['name'];frozen=manifest_train[name];source=original[name]
        image_path=(run/'dense'/'images'/name).resolve();mask_path=(run/'dense'/'masks'/(name+'.png')).resolve()
        require(Path(view['image_path']).resolve()==Path(frozen['image_path']).resolve()==image_path,'Dense image path mismatch')
        require(Path(view['mask_path']).resolve()==Path(frozen['mask_path']).resolve()==mask_path,'Dense mask path mismatch')
        image_rec=record(image_path);mask_rec=record(mask_path);source_image_rec=record(source['image_path'])
        require(image_rec['sha256']==frozen['sha256'] and mask_rec['sha256']==frozen['mask_sha256'],'Dense input bytes differ from frozen appearance manifest')
        require(source_image_rec['sha256']==source['sha256'],'Original source frame bytes changed')
        for rec in (image_rec,mask_rec,source_image_rec):input_predates_job(rec,job['started_ns'])
        with Image.open(image_path) as im:require(im.size==(view['width'],view['height']),'Prepared image dimensions differ from dataset')
        with Image.open(mask_path) as im:
            a=np.asarray(im.convert('L'));require(im.size==(view['width'],view['height']) and np.isin(a,[0,255]).all() and (a==255).any(),'Invalid dense static mask')
        require(abs(view['time_seconds']-source['time_seconds'])<1e-6,'Dense source timestamp differs')
        image_records.append(dict(name=name,source_time_seconds=view['time_seconds'],image=image_rec,mask=mask_rec,source_image=source_image_rec))
    points_rec=record(data['points_path']);require(points_rec['sha256']==manifest['points_sha256'],'Initialization points differ from appearance manifest')
    require(Path(manifest['points_path']).resolve()==Path(data['points_path']).resolve(),'Initialization point path mismatch')
    input_predates_job(points_rec,job['started_ns'])
    mask_receipt=record(run/'appearance_masks'/'detections.json');require(mask_receipt['sha256']==data['mask_receipt_sha256'],'Reviewed mask receipt changed')
    model=(run/job['model_name']).resolve();require(Path(data['model_path']).resolve()==model,'Dataset model differs from owned dense command')
    model_records={p.name:record(p) for p in model.glob('*.bin')}
    require({k:v['sha256'] for k,v in model_records.items()}==data['model_artifacts'],'Current source model differs from dataset hashes')
    require(model_records['images.bin']['sha256']==data['model_images_sha256'],'Source model image hash mismatch')
    dense_model_records={p.name:record(p) for p in (run/'dense'/'sparse').glob('*.bin')}
    for rec in list(model_records.values())+list(dense_model_records.values()):input_predates_job(rec,job['started_ns'])
    source_model=pycolmap.Reconstruction(str(model));dense_model=pycolmap.Reconstruction(str(run/'dense'/'sparse'))
    source_images={im.name:im for im in source_model.images.values() if im.has_pose}
    dense_images={im.name:im for im in dense_model.images.values() if im.has_pose}
    require(set(source_images)==set(dense_images)==training,'Dense/source model registered names differ from training set')
    cameras=[];pose_delta=0.;K_delta=0.;source_pose_delta=0.
    for view in train:
        im=dense_images[view['name']];original_im=source_images[view['name']];cam=dense_model.cameras[im.camera_id]
        require(cam.model_name in ('PINHOLE','SIMPLE_PINHOLE'),'Dense camera is not an undistorted pinhole')
        w2c=np.eye(4);w2c[:3]=im.cam_from_world().matrix();K=cam.calibration_matrix()
        pd=float(np.max(abs(w2c-np.array(view['w2c']))));kd=float(np.max(abs(K-np.array(view['K']))))
        sd=float(np.max(abs(w2c[:3]-original_im.cam_from_world().matrix())))
        require(pd<1e-8 and kd<1e-7 and sd<1e-8,'Dense pose/intrinsics disagree with known cameras')
        require((cam.width,cam.height)==(view['width'],view['height']),'Dense camera dimensions mismatch')
        require(im.image_id==original_im.image_id,'Undistortion changed image identity')
        pose_delta=max(pose_delta,pd);K_delta=max(K_delta,kd);source_pose_delta=max(source_pose_delta,sd)
        cameras.append(dict(name=im.name,image_id=im.image_id,camera_id=im.camera_id,width=cam.width,height=cam.height,
                            model=cam.model_name,K=K.tolist(),w2c=w2c.tolist()))
    dense_ids=set(dense_model.points3D);require(dense_ids<=set(source_model.points3D),'Undistorted model introduced unknown point IDs')
    xyz_delta=max((float(np.max(abs(dense_model.points3D[i].xyz-source_model.points3D[i].xyz))) for i in dense_ids),default=0.)
    require(xyz_delta<1e-10,'Undistorted sparse model changed reconstruction frame')
    cfgs={}
    for name,patch in [('patch-match.cfg',True),('fusion.cfg',False)]:
        path=run/'dense'/'stereo'/name;rec=record(path);input_predates_job(rec,job['started_ns']);config_names(path,training,patch);cfgs[name]=rec
    return dict(dataset=dataset_rec,frames=frames_rec,source=source_rec,appearance_manifest=manifest_rec,
                appearance_inputs_sha256=manifest['inputs_sha256'],manifest_self_digest_verified=True,
                training_count=len(names),training_names=sorted(training),training_names_sha256=digest(sorted(training)),
                prepared_images=image_records,initial_points=points_rec,reviewed_mask_receipt=mask_receipt,
                source_model=model_records,dense_sparse_model=dense_model_records,dense_cameras=cameras,
                dense_cameras_sha256=digest(cameras),configuration=cfgs,pycolmap_version=pycolmap.__version__,
                comparisons=dict(max_dataset_pose_abs_error=pose_delta,max_dataset_K_abs_error=K_delta,
                                 max_source_model_pose_abs_error=source_pose_delta,max_shared_point_xyz_abs_error=xyz_delta,
                                 common_point_ids=len(dense_ids)),
                timing_note='Prepared training images/masks, source/dense model binaries and stereo configs predate job start. Dataset and frozen appearance manifest are audited after job start; their contents are not claimed to have been preflight-captured.'),data


def map_header(path):
    with path.open('rb') as f:
        header=bytearray()
        while header.count(b'&')<3 and len(header)<128:
            c=f.read(1);require(bool(c),f'Truncated map header: {path}');header.extend(c)
    require(header.count(b'&')==3,'Invalid COLMAP dense map header')
    try:w,h,c=map(int,header[:-1].decode('ascii').split('&'))
    except (ValueError,UnicodeError) as exc:raise AuditError(f'Invalid map dimensions: {path}') from exc
    require(w>0 and h>0 and c in (1,3),'Invalid map dimensions/channel count')
    require(path.stat().st_size==len(header)+w*h*c*4,f'Truncated or extra dense map payload: {path}')
    return w,h,c


def ply_header(path):
    properties=[];vertices=None;binary=False;header_bytes=0
    types={'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1','int':'<i4','uint':'<u4'}
    with path.open('rb') as f:
        require(f.readline()==b'ply\n','Fused output is not a PLY');header_bytes=4
        for _ in range(100):
            line=f.readline();header_bytes+=len(line);s=line.decode('ascii').strip()
            if s.startswith('format '):binary=s=='format binary_little_endian 1.0'
            elif s.startswith('element vertex '):vertices=int(s.split()[-1])
            elif s.startswith('element '):raise AuditError('Unexpected non-vertex element in fused PLY')
            elif s.startswith('property '):
                _,typ,name=s.split();require(typ in types,'Unsupported fused PLY scalar type');properties.append((name,types[typ]))
            elif s=='end_header':break
        else:raise AuditError('Unterminated PLY header')
    require(binary and vertices is not None and vertices>0,'Fused PLY must be nonempty binary little-endian points')
    dtype=np.dtype(properties);require({'x','y','z'}<=set(dtype.names),'Fused PLY lacks XYZ')
    require(path.stat().st_size==header_bytes+vertices*dtype.itemsize,'Fused PLY payload length mismatch')
    points=np.memmap(path,dtype=dtype,mode='r',offset=header_bytes,shape=(vertices,))
    step=max(1,vertices//100000);sample=points[::step]
    require(all(np.isfinite(sample[k]).all() for k in ('x','y','z')),'Sampled fused XYZ contain nonfinite values')
    return dict(vertex_count=vertices,properties=list(dtype.names),sampled_finite_xyz=len(sample),full_geometry_accuracy_verified=False)


def audit_outputs(run,inputs,job):
    training=set(inputs['training_names']);dims={v['name']:(v['width'],v['height']) for v in inputs['dense_cameras']}
    started=job['started_ns'];finished=job['finished_ns'];require(finished is not None and finished>=started,'Invalid owned job time interval')
    maps=[];counts=Counter()
    for directory,channels in [('depth_maps',1),('normal_maps',3)]:
        seen=set()
        for path in sorted((run/'dense'/'stereo'/directory).iterdir()):
            match=re.fullmatch(r'(.+)\.(photometric|geometric)\.bin',path.name)
            require(path.is_file() and match is not None,f'Unexpected dense map file: {path}')
            name,kind=match.groups();require(name in training,'Dense output includes nontraining view')
            key=(name,kind);require(key not in seen,'Duplicate dense map');seen.add(key)
            w,h,c=map_header(path);require(c==channels,'Dense map channel mismatch')
            require((w,h)==dims[name],f'Dense map dimensions differ from known input camera: {path}')
            rec=record(path);output_within_job(rec,started,finished);rec.update(name=name,kind=kind,map_type=directory,width=w,height=h,channels=c)
            maps.append(rec);counts[directory+'/'+kind]+=1
        require(seen=={(n,k) for n in training for k in ('photometric','geometric')},'Missing photometric or geometric maps')
    graphs=[]
    for path in sorted((run/'dense'/'stereo'/'consistency_graphs').iterdir()):
        match=re.fullmatch(r'(.+)\.(photometric|geometric)\.bin',path.name)
        require(path.is_file() and match is not None and match[1] in training,'Unexpected consistency graph')
        rec=record(path);output_within_job(rec,started,finished);graphs.append(rec)
    fused_path=run/'dense'/'fused.ply';fused=record(fused_path);output_within_job(fused,started,finished);fused.update(ply_header(fused_path))
    visibility=record(run/'dense'/'fused.ply.vis');output_within_job(visibility,started,finished)
    log_path=Path(job['attempt'])/'log.txt';log=record(log_path);text=log_path.read_text(errors='replace')
    log_names=re.findall(r'=== Processing view \d+ / \d+ for (.+?) ===',text)
    require(set(log_names)==training,'Dense log processed names differ from training set')
    require(all(count>=2 for count in Counter(log_names).values()),'Dense log lacks both matching passes for a training view')
    require(f'Configuration has {len(training)} problems' in text,'Dense log configuration count differs')
    return dict(fused=fused,fused_visibility=visibility,maps=maps,map_counts=dict(counts),maps_sha256=digest(maps),consistency_graphs=graphs,
                fusion_support_interpretation='min_num_pixels=5 is a contributing-pixel threshold, not a guarantee of five distinct supporting camera views.',
                log=log,log_processed_view_occurrences=len(log_names),timestamp_window_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--attempt',type=Path)
    p.add_argument('--inputs-only',action='store_true',help='Read-only input consistency check; never writes a dense receipt')
    p.add_argument('--expected-train-count',type=int,default=192)
    a=p.parse_args();run=a.run.resolve();out=(a.output or run/'dense_provenance.json').resolve()
    if not a.inputs_only:
        require(not out.exists(),'Receipt already exists and is immutable; choose a fresh --output path')
    manifest=a.manifest or run/'appearance_v2'/'input_manifest.json';audit_started=datetime.now(timezone.utc).isoformat()
    job=read_job(run,a.attempt)
    if not a.inputs_only:
        require(job['state']=='succeeded' and job['returncode']==0 and job['finished_ns'] is not None,
                'Owned dense_stable job has not succeeded; no dense provenance receipt written')
    inputs,data=audit_inputs(run,manifest,job,a.expected_train_count)
    if a.inputs_only:
        print(json.dumps(dict(status='input_consistency_only',receipt_written=False,training_count=inputs['training_count'],
                              dataset_sha256=inputs['dataset']['sha256'],comparisons=inputs['comparisons'],timing_note=inputs['timing_note']),indent=2));return
    outputs=audit_outputs(run,inputs,job)
    # Check immutable inputs and job state again after potentially large map reads.
    for rec in [inputs['dataset'],inputs['appearance_manifest'],job['config'],job['status'],job['pipeline'],outputs['log']]+list(inputs['source_model'].values())+list(inputs['dense_sparse_model'].values()):
        require(record(rec['path'])['sha256']==rec['sha256'],f'Bound input changed during audit: {rec["path"]}')
    result=dict(schema='real2sim-dense-provenance/v1',status='verified_post_run',run=str(run),
                source_sha256=data['source_sha256'],dataset_sha256=inputs['dataset']['sha256'],
                fused_sha256=outputs['fused']['sha256'],fused_path=outputs['fused']['path'],
                audit_started_at=audit_started,audit_finished_at=datetime.now(timezone.utc).isoformat(),
                implementation_sha256=record(__file__)['sha256'],capture_kind='post_run_audit',preflight_capture_verified=False,
                units=data['units'],metric_geometry_verified=False,training_count=inputs['training_count'],
                job=job,inputs=inputs,outputs=outputs,
                limitations=['Current hashes, preserved file timestamps and owned successful-job records establish post-run consistency, not a missing preflight receipt.',
                             'Filesystem timestamps are supporting local evidence, not cryptographic proof of historical immutability.',
                             'Images and masks match the frozen appearance manifest; unobserved geometry, metric scale, mesh quality and physical sensor performance remain unverified.'])
    out=out.resolve();out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:
        # Atomic publication without replacing a receipt another stage may use.
        os.link(tmp,out)
    except FileExistsError as exc:
        raise AuditError('Receipt path appeared during audit; refusing to replace it') from exc
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(dict(status=result['status'],path=str(out),sha256=record(out)['sha256'],
                                         dataset_sha256=result['dataset_sha256'],fused_sha256=result['fused_sha256'],
                                         training_count=result['training_count'],map_counts=outputs['map_counts'],
                                         fused_vertices=outputs['fused']['vertex_count']),indent=2))


if __name__=='__main__':
    try:main()
    except AuditError as exc:
        print(json.dumps(dict(status='rejected',error=str(exc),receipt_written=False)),file=sys.stderr);sys.exit(1)
