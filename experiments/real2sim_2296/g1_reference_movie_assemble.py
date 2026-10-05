#!/usr/bin/env python3
"""Assemble the exact take04b native kinematic reference camera correction."""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,struct,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

FPS=15;COUNT=2347;START=1008;REPLACEMENT_COUNT=642
SCENE_SHA='5d49ab65eaaa682d45c6d2e82b0dc0326631118dec432c549fcf0ce7d87f54f5'
TRAJECTORY_SHA='795b3a740cc22c0156a5f6230121c67ff681ec93288621e456fc95cc4ab9ce28'
TASK_SHA='74381f1f6deaf0f84b6b2f409843d69b1fa539ae80bffec34f9ccb94d4fab028'
INSPECT=[0,99,749,798,1007,1008,1245,1470,1649,1650,2152,2208,2250,2346]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def timestamp():return datetime.now(timezone.utc).isoformat()
def passed(value,message):
    if not value:raise ValueError(message)
def receipt(root):
    p=root/'render.receipt.json';r=read(p)
    passed(r['schema']=='real2sim-g1-task-reference-render/v1' and r['status']=='passed','Render has not passed: '+str(root))
    passed(r['kinematic_preview'] and not r['physics_simulated'] and not r.get('physical_recording_replay',False),'Expected kinematic reference only')
    required={'all_requested_frames_captured','timeline_positions_match_requests','all_captured_usd_poses_match_fk',
              'all_captured_object_poses_match_reference','preview_physics_disabled','all_bound_inputs_unchanged',
              'native_realtime_render_mode','all_render_product_cameras_match_schedule'}
    checks={x['name']:x['passed'] for x in r['checks']}
    passed(required.issubset(checks) and all(checks.values()),'Missing or failed native checks')
    passed(r['source_scene_sha256']==SCENE_SHA,'Wrong source scene')
    passed(r['authored_fps']==30 and r['render_fps']==FPS,'Wrong source/render rate')
    passed(r['request']['width']==1280 and r['request']['height']==720,'Wrong image dimensions')
    for path,h in r['input_files_sha256'].items():passed(sha(path)==h,'Changed bound input: '+path)
    passed(sha(root/'samples.jsonl')==r['samples_jsonl_sha256'],'Changed sample stream')
    rows=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()]
    passed(rows==r['samples'],'Sample stream/receipt disagreement')
    passed(len(rows)==r['captured_frames']==len(r['sample_indices']),'Source sample count disagreement')
    return r,rows
def source_job(root,r):
    matches=[]
    for p in (root.parents[1]/'jobs'/root.name).glob('*/config.json'):
        c=read(p)
        if str(root) in c['command']:
            helpers=[Path(x) for x in c['command'] if x.endswith('.py') and Path(x).is_file()]
            helpers=[x for x in helpers if sha(x)==r['implementation_sha256']]
            if helpers:
                s=read(p.parent/'status.json')
                passed(s['status']=='succeeded','Renderer job has not succeeded')
                matches.append((p,helpers[0]))
    passed(len(matches)==1,'Cannot uniquely bind renderer job and executed snapshot')
    return matches[0]
def expected_camera(r,seconds):
    schedule=r.get('camera_schedule')
    if schedule:
        return [x['path'] for x in schedule if seconds>=x['time']-1e-9][-1]
    return r['request']['camera']
def atom_order(path):
    atoms=[]
    with path.open('rb') as f:
        offset=0;size=path.stat().st_size
        while offset+8<=size:
            f.seek(offset);raw=f.read(8);n,kind=struct.unpack('>I4s',raw)
            if n==1:n=struct.unpack('>Q',f.read(8))[0]
            elif n==0:n=size-offset
            passed(n>=8,'Invalid MP4 atom')
            atoms.append({'type':kind.decode('ascii','replace'),'offset':offset,'size':n});offset+=n
    names=[x['type'] for x in atoms]
    passed('moov' in names and 'mdat' in names and names.index('moov')<names.index('mdat'),'MP4 is not faststart')
    return atoms
def assemble(a):
    start=time.monotonic();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    passed(not (out/'assembly.receipt.json').exists() and not (out/'motion.mp4').exists(),'Refusing to overwrite assembly')
    (out/'frames').mkdir();(out/'inspection').mkdir();(out/'sources').mkdir()
    def status(state,**kw):save(out/'status.json',{'schema':'g1-reference-movie-assembly-status/v1','status':state,'elapsed_seconds':time.monotonic()-start,**kw})
    status('validating_sources')
    base=a.base.resolve();replacement=a.replacement.resolve();br,brows=receipt(base);rr,rrows=receipt(replacement)
    passed(len(brows)==COUNT and len(rrows)==REPLACEMENT_COUNT,'Unexpected capture lengths')
    for k in ['source_scene_sha256','export_receipt_sha256','task_export_receipt_sha256']:
        passed(br[k]==rr[k],'Different source identity: '+k)
    for k in set(br['input_files_sha256'])&set(rr['input_files_sha256']):
        passed(br['input_files_sha256'][k]==rr['input_files_sha256'][k],'Different shared input: '+k)
    export=Path(br['request']['scene']).parent
    trajectory=export/'inputs/trajectory.npz';task=export/'inputs/original_task.json';fk_path=export/'fk_samples.npz'
    passed(sha(trajectory)==TRAJECTORY_SHA and sha(task)==TASK_SHA,'Wrong take04b task or trajectory')
    z=np.load(trajectory,allow_pickle=False);fk=np.load(fk_path,allow_pickle=False)
    passed(len(z['time'])==4693 and np.array_equal(z['qpos'],fk['qpos']),'Export FK reference differs')
    source_records={};source_start_hashes={}
    for label,root,r in [('base',base,br),('replacement',replacement,rr)]:
        config,helper=source_job(root,r);dest=out/'sources'/label;dest.mkdir()
        for p in [root/'render.receipt.json',root/'samples.jsonl',config,config.parent/'status.json',helper]:
            shutil.copy2(p,dest/p.name);source_start_hashes[str(p)]=sha(p)
        source_records[label]={'root':str(root),'render_receipt':binding(root/'render.receipt.json'),
             'samples_jsonl':binding(root/'samples.jsonl'),'job_config':binding(config),'executed_helper':binding(helper)}
    selected=[];maxima={k:0. for k in ['reported_body_matrix_error','reported_door_matrix_error','reported_object_matrix_error',
                                      'body_origin_error','object_center_error','door_angle_error','timeline_error_seconds']}
    manifest=out/'frame_sources.jsonl'
    with manifest.open('w') as stream:
        for i in range(COUNT):
            use=START<=i<START+REPLACEMENT_COUNT;label='replacement' if use else 'base'
            root=replacement if use else base;r=rr if use else br;j=i-START if use else i;row=(rrows if use else brows)[j]
            index=2*i;t=i/FPS
            passed(row['frame']==j and row['authored_sample_index']==index,'Incorrect frame/time mapping')
            passed(r['sample_indices'][j]==index,'Receipt sample-index mapping differs')
            terr=max(abs(row[k]-t) for k in ['target_seconds','timeline_seconds_before_capture','timeline_seconds_after_capture'])
            passed(terr<1e-6 and abs(row['measured_usd_time_code']-index)<1e-5,'Capture timeline mismatch')
            cam=expected_camera(r,t)
            passed(row['camera_path']==cam and row['camera_relation_before_capture']==row['camera_relation_after_capture']==[cam],'Camera readback mismatch')
            if use:passed(cam=='/World/PreviewCameras/Route','Replacement is not Route camera')
            errors={'reported_body_matrix_error':row['max_body_fk_matrix_error'],
                    'reported_door_matrix_error':row['door_matrix_error'],'reported_object_matrix_error':row['object_matrix_error'],
                    'body_origin_error':float(np.max(np.abs(np.asarray(row['body_origins_world'])-fk['body_world'][index,:,3,:3]))),
                    'object_center_error':float(np.max(np.abs(np.asarray(row['object_center_world'])-z['object_pose'][index,:3]))),
                    'door_angle_error':abs(row['door_angle_radians']-float(z['door_angle'][index])),
                    'timeline_error_seconds':terr}
            passed(all(math.isfinite(x) and x<1e-7 for k,x in errors.items() if k!='timeline_error_seconds'),'Selected pose check failed')
            for k,v in errors.items():maxima[k]=max(maxima[k],v)
            image=Path(row['image']['path']).resolve()
            passed(image==root/'frames'/f'{j:05d}.png' and sha(image)==row['image']['sha256'],'Selected image path/hash mismatch')
            with Image.open(image) as im:passed(im.size==(1280,720),'Selected image dimensions differ')
            dest=out/'frames'/f'{i:05d}.png'
            try:os.link(image,dest)
            except OSError:shutil.copy2(image,dest)
            item={'output_frame':i,'presentation_seconds':t,'source':label,'source_capture_frame':j,
                  'authored_sample_index':index,'camera':cam,'source_png':str(image),'source_png_sha256':row['image']['sha256'],
                  'source_render_receipt_sha256':source_records[label]['render_receipt']['sha256'],
                  'recorded_native_checks':errors,'reference_phase':str(z['phase'][index])}
            stream.write(json.dumps(item,separators=(',',':'))+'\n');selected.append(item)
            if i%200==0:status('validating_frames',validated_frames=i+1)
    switches=[{'output_frame':x['output_frame'],'seconds':x['presentation_seconds'],'camera':x['camera']} for i,x in enumerate(selected) if i==0 or x['camera']!=selected[i-1]['camera']]
    status('encoding',frame_count=COUNT)
    movie=out/'motion.mp4'
    cmd=['ffmpeg','-hide_banner','-nostdin','-v','warning','-framerate',str(FPS),'-i',str(out/'frames/%05d.png'),
         '-frames:v',str(COUNT),'-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
         '-movflags','+faststart','-threads','4','-metadata','title=G1 / Dex3 fridge-to-table kinematic reference',
         '-metadata','comment=Prescribed robot and bottle motion. Native Isaac RTX images; no physical pick-and-place success claim.',str(movie)]
    save(out/'encoding_command.json',{'argv':cmd,'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]})
    with (out/'encoding.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=log)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)],text=True))
    save(out/'ffprobe.json',probe);video=[x for x in probe['streams'] if x['codec_type']=='video']
    passed(len(video)==1 and len(probe['streams'])==1,'Unexpected output streams');v=video[0]
    passed(int(v['nb_read_frames'])==COUNT and v['avg_frame_rate']==v['r_frame_rate']=='15/1','Encoded count/rate mismatch')
    passed(v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and (v['width'],v['height'])==(1280,720),'Encoded video format mismatch')
    passed(abs(float(v['duration'])-COUNT/FPS)<.001,'Encoded duration mismatch')
    atoms=atom_order(movie)
    # Inspect frames decoded from the final lossy movie, preserving the original image hashes separately.
    filter_expr="select='"+'+'.join(f'eq(n,{i})' for i in INSPECT)+"'"
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-nostdin','-i',str(movie),'-vf',filter_expr,
                    '-vsync','0','-q:v','2',str(out/'inspection/decoded_%02d.jpg')],check=True)
    inspection=[];sheet=Image.new('RGB',(1280,6*268),(18,25,32));draw=ImageDraw.Draw(sheet)
    for k,i in enumerate(INSPECT):
        p=out/'inspection'/f'decoded_{k+1:02d}.jpg'
        decoded=np.asarray(Image.open(p).convert('RGB'),dtype=float);original=np.asarray(Image.open(out/'frames'/f'{i:05d}.png').convert('RGB'),dtype=float)
        mse=float(np.mean((decoded-original)**2));psnr=float(10*np.log10(255**2/max(mse,1e-12)))
        passed(psnr>28,'Decoded inspection frame differs substantially from selected source')
        row={'output_frame':i,'seconds':i/FPS,'camera':selected[i]['camera'],'decoded_image':binding(p),'psnr_against_input_png_db':psnr}
        inspection.append(row);im=Image.open(p).convert('RGB');im.thumbnail((420,236));x=(k%3)*426;y=(k//3)*268
        sheet.paste(im,(x,y));draw.text((x+5,y+239),f'{i} / {i/FPS:.3f}s / {selected[i]["camera"].rsplit("/",1)[-1]}',fill='white')
    sheet.save(out/'inspection_sheet.jpg',quality=93)
    # Close every selected PNG and input identity after encoding, including hard-linked source files.
    for item in selected:
        i=item['output_frame'];passed(sha(out/'frames'/f'{i:05d}.png')==item['source_png_sha256'],'Selected frame changed during encode')
    for p,h in source_start_hashes.items():passed(sha(p)==h,'Source receipt/helper changed during assembly')
    for path,h in {**br['input_files_sha256'],**rr['input_files_sha256']}.items():passed(sha(path)==h,'Source input changed during assembly')
    result={'schema':'g1-reference-movie-assembly/v1','status':'passed','produced_at':timestamp(),
         'implementation':binding(Path(__file__)),'source_scene_sha256':SCENE_SHA,'trajectory_sha256':TRAJECTORY_SHA,'task_sha256':TASK_SHA,
         'sources':source_records,'kinematic_reference':True,'prescribed_object_motion':True,'physics_simulated':False,
         'physical_pickplace_success':False,'metric_accuracy_verified':False,'frame_count':COUNT,'fps':FPS,
         'duration_seconds':COUNT/FPS,'last_reference_time_seconds':(COUNT-1)/FPS,
         'replacement':{'first_output_frame':START,'last_output_frame':START+REPLACEMENT_COUNT-1,'frames':REPLACEMENT_COUNT,
                        'first_reference_seconds':START/FPS,'last_reference_seconds':(START+REPLACEMENT_COUNT-1)/FPS,'camera':'/World/PreviewCameras/Route'},
         'unchanged_base_frames':COUNT-REPLACEMENT_COUNT,'frame_sources':binding(manifest),'maximum_pose_time_errors':maxima,
         'camera_switches':switches,'movie':binding(movie),'encoding_command':binding(out/'encoding_command.json'),
         'ffprobe':binding(out/'ffprobe.json'),'mp4_top_level_atoms':atoms,'decoded_inspection_frames':inspection,
         'inspection_sheet':binding(out/'inspection_sheet.jpg'),'elapsed_seconds':time.monotonic()-start,
         'verification_scope':'Every selected source PNG hash, recorded native FK/object/timeline/camera check, saved body origin, bottle center, door angle, reference phase and shared input identity. Final MP4 decoded count/FPS/format/faststart and selected decoded images checked.',
         'limitations':['Kinematic reference with prescribed robot, door and bottle motion; no dynamic balance or force/physical grasp proof.',
                        'Current full physical pickup trials are separate and failing; the successful isolated lift does not validate this full reference task.',
                        'Scene dimensions and task prop are nominal; visual waist-support intersections remain unresolved.',
                        'H.264 compression is lossy. Existing documentary captions cover the bottom 48 rows.']}
    save(out/'assembly.receipt.json',result);status('passed',frame_count=COUNT,movie_sha256=sha(movie))
    print(json.dumps({k:result[k] for k in ['status','frame_count','duration_seconds','movie','frame_sources','maximum_pose_time_errors']},indent=2))
def unique_copy(source,wanted):
    wanted.parent.mkdir(parents=True,exist_ok=True);h=sha(source)
    if wanted.exists() and sha(wanted)!=h:wanted=wanted.with_name(wanted.stem+'_'+h[:10]+wanted.suffix)
    if wanted.exists():passed(sha(wanted)==h,'Conflicting suffix file')
    else:
        # Exclusive creation preserves an existing file even if another process races this copy.
        with source.open('rb') as src,wanted.open('xb') as dst:shutil.copyfileobj(src,dst,1048576)
    passed(sha(wanted)==h,'Delivery copy mismatch');return wanted
def deliver(a):
    source=a.output.resolve();r=read(source/'assembly.receipt.json')
    passed(r['status']=='passed' and r['kinematic_reference'] and not r['physical_pickplace_success'],'Not an accepted reference assembly')
    movie=source/'motion.mp4';passed(sha(movie)==r['movie']['sha256'],'Movie changed')
    dest=a.delivery_dir.resolve();dest.mkdir(parents=True,exist_ok=False)
    for name in ['motion.mp4','assembly.receipt.json','frame_sources.jsonl','encoding_command.json','ffprobe.json','inspection_sheet.jpg']:
        shutil.copy2(source/name,dest/name)
    shutil.copytree(source/'sources',dest/'sources');shutil.copytree(source/'inspection',dest/'inspection')
    shutil.copy2(Path(__file__),dest/'g1_reference_movie_assemble.py')
    note=('G1 / Dex3 fridge-to-table KINEMATIC REFERENCE\n\n'
          'This 156.467 s native Isaac RTX movie shows prescribed robot, door and bottle motion. It is an editable motion reference, not a successful physical pick-and-place trial. The route camera was substituted for frames 1008–1649 (67.2–109.933 s); all other selected base images are unchanged before H.264 encoding.\n\n'
          'The final movie has 2347 frames at 15 fps, 1280x720, H.264 CRF18/yuv420p/faststart. Each source frame, native pose/time/camera record and shared input was hash-checked. The scene is nominal scale, not a metrically verified map. Physical full-task attempts and the separate isolated lift have their own evidence.\n\n'
          f'Movie SHA256: {sha(movie)}\nAssembly receipt SHA256: {sha(source/"assembly.receipt.json")}\n')
    (dest/'README.txt').write_text(note)
    if a.review_receipt:
        shutil.copy2(a.review_receipt,dest/'media_review.receipt.json')
    downloaded=unique_copy(movie,a.downloads_dir/'G1_fridge_to_table_reference.mp4')
    readme=unique_copy(dest/'README.txt',downloaded.with_suffix('.README.txt'))
    records=[{'path':str(p.relative_to(dest)),'sha256':sha(p),'size':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file()]
    rec={'schema':'g1-reference-video-delivery/v1','status':'copied_and_hash_verified','produced_at':timestamp(),
         'assembly_receipt_sha256':sha(source/'assembly.receipt.json'),'movie_sha256':sha(movie),
         'kinematic_reference':True,'physical_pickplace_success':False,'downloads_movie':binding(downloaded),'downloads_readme':binding(readme),
         'existing_different_files_preserved':True,'files':records}
    save(dest/'delivery.receipt.json',rec);print(json.dumps({k:v for k,v in rec.items() if k!='files'},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['assemble','deliver'])
    p.add_argument('--base',type=Path);p.add_argument('--replacement',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--delivery-dir',type=Path);p.add_argument('--downloads-dir',type=Path,default=Path.home()/'Downloads');p.add_argument('--review-receipt',type=Path)
    a=p.parse_args()
    if a.command=='assemble':
        if not a.base or not a.replacement:p.error('assemble needs --base and --replacement')
        assemble(a)
    else:
        if not a.delivery_dir:p.error('deliver needs --delivery-dir')
        deliver(a)
if __name__=='__main__':main()
