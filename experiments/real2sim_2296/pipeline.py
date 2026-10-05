#!/usr/bin/env python3
"""Evidence-preserving RGB reconstruction; no guessed metric scale."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def extract(a):
    import cv2
    from PIL import Image, ImageDraw
    root = a.run
    inspection = root / 'inspection'
    inspection.mkdir(parents=True, exist_ok=True)
    if (root/'frames.json').exists():
        previous=json.loads((root/'frames.json').read_text())
        if previous['source_sha256']!=sha(a.source) or previous['selection_fps']!=a.fps:
            raise RuntimeError('Source or sampling changed. Use a new run directory; never overwrite this dataset.')
        for frame in previous['frames']:
            if not Path(frame['image_path']).is_file() or sha(frame['image_path'])!=frame['sha256']:
                raise RuntimeError('Existing image missing/changed. Use a new run directory for extraction.')
        print('Existing extraction source/config/images verified; no files rewritten.',flush=True)
        return
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
              '-show_streams', '-of', 'json', str(a.source)]))
    write(inspection / 'ffprobe.json', probe)
    times = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time',
        '-of', 'json', str(a.source)]))['frames']
    pts = [float(x['best_effort_timestamp_time']) for x in times]
    cap = cv2.VideoCapture(str(a.source))
    for folder in ('images', 'holdout'):
        (root / folder).mkdir(exist_ok=True)
    selected, scores = [], []
    best, lastbin = None, -1

    def flush(item):
        if item is None:
            return
        frame, index, t, b, sharpness = item
        split = 'val' if b % 10 == 5 else 'train'
        folder = 'holdout' if split == 'val' else 'images'
        name = f'frame_{index:06d}.jpg'
        path = root / folder / name
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise RuntimeError(f'Could not write {path}')
        selected.append(dict(name=name, source_index=index, time_seconds=t, bin=b,
                             sharpness=sharpness, split=split, sha256=sha(path),
                             image_path=str(path), width=frame.shape[1], height=frame.shape[0]))

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = pts[index]
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
        b = int(t * a.fps)
        scores.append(dict(source_index=index, time_seconds=t, sharpness=sharpness))
        if b != lastbin:
            flush(best); best = None; lastbin = b
        if best is None or sharpness > best[-1]:
            best = (frame.copy(), index, t, b, sharpness)
        index += 1
    flush(best); cap.release()
    assert index == len(pts), (index, len(pts))
    manifest = dict(schema='real2sim-frames/v1', source_path=str(a.source),
        source_sha256=sha(a.source), decoded_frames=index, selection_fps=a.fps,
        selection='maximum 640px Laplacian variance within each time bin',
        holdout_rule='bin modulo 10 equals 5, excluded before SfM', frames=selected)
    write(root / 'frames.json', manifest)
    write(inspection / 'frame_quality.json', scores)
    # PTS-stamped contact sheets, replacing the initial approximate fps sheet.
    samples = selected[::max(1, round(2 * a.fps))]
    for page in range((len(samples) + 23) // 24):
        subset = samples[page*24:(page+1)*24]
        sheet = Image.new('RGB', (4*480, ((len(subset)+3)//4)*298), '#161820')
        d = ImageDraw.Draw(sheet)
        for i,item in enumerate(subset):
            im = Image.open(item['image_path']).resize((480,270))
            x,y = i%4*480,i//4*298
            sheet.paste(im,(x,y))
            d.text((x+7,y+277),f"{item['time_seconds']:.3f}s | frame {item['source_index']} | {item['split']}",fill='white')
        sheet.save(inspection/f'pts_sheet_{page+1}.jpg',quality=93)
    print(json.dumps({k:v for k,v in manifest.items() if k!='frames'}),flush=True)
    print('selected',len(selected),'train',sum(x['split']=='train' for x in selected),flush=True)


def sfm(a):
    import pycolmap as p
    root = a.run
    manifest = json.loads((root/'frames.json').read_text())
    names = [f['name'] for f in manifest['frames'] if f['split']=='train']
    database = root/a.database_name
    if a.map_only:
        return solve(a,p,root,manifest,names,database)
    eo = p.FeatureExtractionOptions()
    eo.max_image_size=1920; eo.num_threads=8; eo.gpu_index='0'
    eo.sift.max_num_features=8192
    reader = p.ImageReaderOptions()
    reader.camera_model='SIMPLE_RADIAL'
    # Approximate focal prior only. Optimize against multi-view evidence.
    reader.default_focal_length_factor=.85
    p.extract_features(database,root/'images',image_names=names,camera_mode=p.CameraMode.SINGLE,
                       reader_options=reader,extraction_options=eo,device=p.Device.cuda)
    mo=p.FeatureMatchingOptions(); mo.gpu_index='0'; mo.num_threads=8; mo.guided_matching=True
    so=p.SequentialPairingOptions(); so.overlap=15; so.quadratic_overlap=True
    p.match_sequential(database,matching_options=mo,pairing_options=so,device=p.Device.cuda)
    # Explicit revisit coverage, without downloading a vocabulary or treating order as geometry.
    pairs=[]
    for i,x in enumerate(names):
        for j in range(i+16,len(names)):
            if i%4==0 or j%4==0:
                pairs.append(f'{x} {names[j]}')
    (root/'revisit_pairs.txt').write_text('\n'.join(pairs)+'\n')
    po=p.ImportedPairingOptions(); po.match_list_path=root/'revisit_pairs.txt'
    p.match_image_pairs(database,matching_options=mo,pairing_options=po,device=p.Device.cuda)
    if a.exhaustive:
        p.match_exhaustive(database,matching_options=mo,device=p.Device.cuda)
    return solve(a,p,root,manifest,names,database)


def solve(a,p,root,manifest,names,database):
    import cv2
    options=p.IncrementalPipelineOptions()
    options.num_threads=8; options.random_seed=2296; options.image_names=names
    options.structure_less_registration_fallback=False
    options.mapper.init_min_tri_angle=a.init_min_angle
    options.fix_existing_frames=a.fixed_seed
    options.mapper.fix_existing_frames=a.fixed_seed
    options.mapper.abs_pose_min_inlier_ratio=a.abs_pose_ratio
    if a.init_pair:
        with sqlite3.connect(database) as connection:
            lookup={n:i for i,n in connection.execute('select image_id,name from images')}
        options.init_image_id1=lookup[a.init_pair[0]];options.init_image_id2=lookup[a.init_pair[1]]
    options.min_model_size=15; options.max_num_models=8
    options.mapper.filter_max_reproj_error=3.0
    options.mapper.abs_pose_max_error=6.0
    options.ba_global_max_num_iterations=75; options.ba_local_max_num_iterations=35
    options.ba_global_max_refinements=3
    sparse=root/('sparse' if a.model_name=='model' else 'sparse_'+a.model_name); sparse.mkdir(exist_ok=True)
    recs=p.incremental_mapping(database,root/'images',sparse,options=options,
        input_path=str(root/a.seed_model) if a.seed_model else '')
    if not recs:
        raise RuntimeError('No reconstructable component. Inspect camera matching before proceeding.')
    best_id=max(recs,key=lambda k:recs[k].num_reg_images())
    rec=recs[best_id]
    if not {im.name for im in rec.images.values() if im.has_pose}.issubset(set(names)):
        raise RuntimeError('A non-training image entered the camera solution.')
    target=root/a.model_name; target.mkdir(exist_ok=True); rec.write(target)
    rec.write_text(target)
    rec.export_PLY(root/(a.model_name+'_points.ply'))
    residuals=[];view_evidence=[]
    for im in rec.images.values():
        if not im.has_pose:
            continue
        camera=rec.cameras[im.camera_id]
        pts2d=[]; pts3d=[]
        for pt in im.points2D:
            if pt.has_point3D():
                pts2d.append(pt.xy);pts3d.append(rec.points3D[pt.point3D_id].xyz)
        if pts3d:
            c=(im.cam_from_world()*np.array(pts3d))
            pred=camera.img_from_cam(c)
            errors=np.linalg.norm(pred-np.array(pts2d),axis=1)
            residuals.extend(errors.tolist())
            hull=float(cv2.contourArea(cv2.convexHull(np.array(pts2d,dtype=np.float32))))/(camera.width*camera.height) if len(pts2d)>=3 else 0.
            view_evidence.append(dict(name=im.name,point_observations=len(pts3d),hull_fraction=hull,
                median_reprojection_px=float(np.median(errors)),p95_reprojection_px=float(np.percentile(errors,95))))
        else:
            view_evidence.append(dict(name=im.name,point_observations=0,hull_fraction=0.,
                median_reprojection_px=None,p95_reprojection_px=None))
    metrics=dict(schema='real2sim-sfm/v1',source_sha256=manifest['source_sha256'],
        components=[dict(id=k,registered=v.num_reg_images(),points=v.num_points3D()) for k,v in recs.items()],
        selected_component=best_id,selected_train=len(names),registered=rec.num_reg_images(),
        registered_fraction=rec.num_reg_images()/len(names),points=rec.num_points3D(),
        reprojection_median_px=float(np.median(residuals)),reprojection_p95_px=float(np.percentile(residuals,95)),
        mean_track_length=rec.compute_mean_track_length(),units='arbitrary',metric_scale_verified=False,
        pose_initialization=a.seed_model or 'incremental_two_view',seed_poses_fixed=a.fixed_seed,
        views=view_evidence,
        cameras_with_broad_track_support=sum(x['point_observations']>=30 and x['hull_fraction']>=.04 for x in view_evidence),
        cameras={k:v.todict() for k,v in rec.cameras.items()},
        unregistered=[n for n in names if n not in {im.name for im in rec.images.values() if im.has_pose}])
    centers=np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    central=np.median(centers,axis=0)
    radii=np.linalg.norm(centers-central,axis=1)
    typical=float(np.percentile(radii,90))
    metrics['camera_radius_quantiles']=np.percentile(radii,[50,90,95,99,100]).tolist()
    metrics['camera_extent_outlier_ratio']=float(radii.max()/max(typical,1e-9))
    metrics['degenerate_extent_rejected']=metrics['camera_extent_outlier_ratio']>20
    # Camera enum values need plain text for durable JSON.
    for cam in metrics['cameras'].values():
        for k,v in list(cam.items()):
            if isinstance(v,np.ndarray): cam[k]=v.tolist()
            elif not isinstance(v,(int,float,str,bool,list,dict,type(None))): cam[k]=str(v)
    write(root/('sfm_metrics.json' if a.model_name=='model' else a.model_name+'_metrics.json'),metrics)
    print(json.dumps(metrics,indent=2),flush=True)


def camera_sanity(rec,frames):
    times={x['name']:x['time_seconds'] for x in frames['frames']}
    images=sorted([x for x in rec.images.values() if x.has_pose],key=lambda x:times[x.name])
    centers=np.array([x.projection_center() for x in images])
    radii=np.linalg.norm(centers-np.median(centers,axis=0),axis=1)
    extent=float(np.percentile(radii,90));issues=[]
    if not np.isfinite(centers).all() or radii.max()>20*max(extent,1e-9):
        issues.append(dict(kind='degenerate_camera_extent'))
    for first,second in zip(images,images[1:]):
        dt=times[second.name]-times[first.name]
        relative=second.cam_from_world().rotation.matrix()@first.cam_from_world().rotation.matrix().T
        angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        distance=float(np.linalg.norm(second.projection_center()-first.projection_center()))
        # These conservative flags are specific to this continuously walked
        # video. Passing them is necessary, not proof of geometric correctness.
        if dt<.8 and (angle>75 or distance>extent):
            issues.append(dict(kind='implausible_temporal_jump',first=first.name,second=second.name,
                dt_seconds=dt,rotation_degrees=angle,translation_units=distance,scene_radius90=extent))
    return dict(status='rejected' if issues else 'no_gross_jump_detected',
        issues=issues,metric_accuracy_verified=False,rule='dt<0.8s and (rotation>75deg or displacement>R90)')


def prepare(a):
    import pycolmap as p
    root=a.run
    verify_masks(root)
    candidate=p.Reconstruction(root/a.model_name)
    sanity=camera_sanity(candidate,json.loads((root/'frames.json').read_text()))
    write(root/(a.model_name+'_sanity.json'),sanity)
    if sanity['issues']:
        raise RuntimeError('Rejected camera trajectory; do not start appearance or dense reconstruction.')
    p.undistort_images(root/'dense',root/a.model_name,root/'images',
        num_patch_match_src_images=12,num_threads=8,jpeg_quality=98)
    rec=p.Reconstruction(root/'dense'/'sparse')
    np.savez_compressed(root/'initial_points.npz',xyz=np.array([x.xyz for x in rec.points3D.values()]),
                        rgb=np.array([x.color for x in rec.points3D.values()],dtype=np.uint8))
    frames=json.loads((root/'frames.json').read_text())
    byname={x['name']:x for x in frames['frames']}
    train=[]
    for im in sorted(rec.images.values(),key=lambda x:x.name):
        if not im.has_pose: continue
        cam=rec.cameras[im.camera_id]; w2c=np.eye(4);w2c[:3]=im.cam_from_world().matrix()
        train.append(dict(name=im.name,image_path=str(root/'dense'/'images'/im.name),
            w2c=w2c.tolist(),K=cam.calibration_matrix().tolist(),width=cam.width,height=cam.height,
            time_seconds=byname[im.name]['time_seconds']))
        maskpath=root/'dense'/'masks'/(im.name+'.png')
        maskpath.parent.mkdir(exist_ok=True)
        original=candidate.cameras[candidate.images[im.image_id].camera_id]
        remap_mask(root,im.name,original,cam.calibration_matrix(),cam.width,cam.height,maskpath)
        train[-1]['mask_path']=str(maskpath)
    write(root/'dataset.json',dict(schema='real2sim-dataset/v1',source_sha256=frames['source_sha256'],
       units='arbitrary',points_path=str(root/'initial_points.npz'),train=train,val=[],
       holdout_policy=frames['holdout_rule'],model_path=str(root/a.model_name),
       model_images_sha256=sha(root/a.model_name/'images.bin'),
       model_artifacts={x.name:sha(x) for x in (root/a.model_name).glob('*.bin')},
       mask_receipt_sha256=sha(root/'appearance_masks'/'detections.json'),
       database_path=str(root/a.database_name),
       masks='Source-reviewed TV/person exclusion plus full bilinear support; conservative, imperfect segmentation.'))
    print('Prepared',len(train),'undistorted training views',flush=True)


def dense(a):
    import pycolmap as p
    opts=p.PatchMatchOptions();opts.gpu_index=a.stereo_gpus;opts.max_image_size=a.max_size
    opts.num_threads=8;opts.cache_size=16;opts.num_iterations=5
    opts.filter_min_num_consistent=3;opts.filter_min_triangulation_angle=2.
    p.patch_match_stereo(a.run/'dense',options=opts)
    fusion=p.StereoFusionOptions();fusion.num_threads=8;fusion.max_image_size=a.max_size
    fusion.min_num_pixels=5;fusion.cache_size=16
    fusion.mask_path=a.run/'dense'/'masks'
    p.stereo_fusion(a.run/'dense'/'fused.ply',a.run/'dense',options=fusion,output_type='ply')


def maps(original,K,width,height):
    # COLMAP pixel centers are half-integers; OpenCV remap indexes are integers.
    yy,xx=np.mgrid[:height,:width].astype(np.float32)
    x=(xx+.5-K[0,2])/K[0,0];y=(yy+.5-K[1,2])/K[1,1]
    assert original.model_name=='SIMPLE_RADIAL'
    f,cx,cy,k=original.params
    factor=1+k*(x*x+y*y)
    return (f*x*factor+cx-.5).astype(np.float32),(f*y*factor+cy-.5).astype(np.float32)


def verify_masks(root):
    frames=json.loads((root/'frames.json').read_text())
    report=json.loads((root/'appearance_masks'/'detections.json').read_text())
    if report['source_sha256']!=frames['source_sha256']:
        raise RuntimeError('Masks are from a different capture.')
    indexed={x['name']:x for x in report['views']}
    for item in frames['frames']:
        path=root/'appearance_masks'/(item['name']+'.png')
        if item['name'] not in indexed or not path.is_file() or sha(path)!=indexed[item['name']]['mask_sha256']:
            raise RuntimeError(f"Missing, incomplete or changed source mask: {item['name']}")
        camera_mask=root/'camera_masks'/(item['name']+'.png')
        if not camera_mask.is_file() or sha(camera_mask)!=indexed[item['name']]['camera_mask_sha256']:
            raise RuntimeError(f"Missing or changed camera mask: {item['name']}")


def remap_mask(root,name,original,K,width,height,path):
    import cv2
    maskpath=root/'appearance_masks'/(name+'.png')
    if not maskpath.is_file():raise RuntimeError(f'Missing source mask: {maskpath}')
    source=cv2.imread(str(maskpath),cv2.IMREAD_GRAYSCALE)
    x,y=maps(original,K,width,height)
    # RGB uses bilinear interpolation: every contributing source pixel must be
    # valid. A nearest mask would admit mixed dynamic/static boundary colors.
    support=cv2.remap(source.astype(np.float32)/255.,x,y,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    mask=np.where(support>=1.-1e-7,255,0).astype(np.uint8)
    # Exclude bilinear source support that extends beyond the input image.
    mask[(x<0)|(y<0)|(x>original.width-1)|(y>original.height-1)]=0
    cv2.imwrite(str(path),mask)


def localize(a):
    import pycolmap as p
    import cv2
    root=a.run
    verify_masks(root)
    # Copy the database so validation features cannot enter the mapping database.
    database=root/'validation.db'
    # Always reconstruct this disposable validation database from the current
    # frozen mapping database; stale keypoint/point-index associations are unsafe.
    if database.exists(): database.unlink()
    with sqlite3.connect(root/a.database_name) as src,sqlite3.connect(database) as dst:
        src.backup(dst)
    frames=json.loads((root/'frames.json').read_text())
    vals=[x for x in frames['frames'] if x['split']=='val']
    train=[x for x in frames['frames'] if x['split']=='train']
    rec=p.Reconstruction(root/a.model_name)
    camera_id=next(iter(rec.cameras))
    reader=p.ImageReaderOptions();reader.existing_camera_id=camera_id
    eo=p.FeatureExtractionOptions();eo.max_image_size=1920;eo.num_threads=8;eo.gpu_index='0'
    p.extract_features(database,root/'holdout',image_names=[x['name'] for x in vals],
         reader_options=reader,extraction_options=eo,device=p.Device.cuda)
    pairs=[]
    for val in vals:
        near=sorted(train,key=lambda x:abs(x['time_seconds']-val['time_seconds']))[:10]
        for item in near: pairs.append(f"{val['name']} {item['name']}")
    (root/'validation_pairs.txt').write_text('\n'.join(pairs)+'\n')
    mo=p.FeatureMatchingOptions();mo.gpu_index='0';mo.num_threads=8;mo.guided_matching=True
    po=p.ImportedPairingOptions();po.match_list_path=root/'validation_pairs.txt'
    p.match_image_pairs(database,matching_options=mo,pairing_options=po,device=p.Device.cuda)
    conn=sqlite3.connect(database)
    image_ids={name:i for i,name in conn.execute('select image_id,name from images')}
    reports=[]; valitems=[]
    cam=rec.cameras[camera_id]
    assert str(cam.model_name)=='SIMPLE_RADIAL',cam.model_name
    K=cam.calibration_matrix();dist=np.array([cam.params[3],0,0,0,0])
    output=root/'evaluation';(output/'images').mkdir(parents=True,exist_ok=True)
    train_records={x['name']:x for x in train}
    train_cameras=sorted([(train_records[im.name]['time_seconds'],im.projection_center())
                          for im in rec.images.values() if im.has_pose],key=lambda x:x[0])
    centers=np.array([x[1] for x in train_cameras]);center=np.median(centers,axis=0)
    extent=float(np.percentile(np.linalg.norm(centers-center,axis=1),90))
    speeds=[np.linalg.norm(b[1]-a[1])/max(b[0]-a[0],1e-6) for a,b in zip(train_cameras,train_cameras[1:])]
    typical_speed=float(np.median(speeds))
    valid_kps={}
    for name,iid in image_ids.items():
        maskfile=root/'camera_masks'/(name+'.png')
        if not maskfile.exists():continue
        mask=cv2.imread(str(maskfile),cv2.IMREAD_GRAYSCALE)
        kp=conn.execute('select rows,cols,data from keypoints where image_id=?',(iid,)).fetchone()
        if not kp:continue
        xy=np.frombuffer(kp[2],dtype=np.float32).reshape(kp[0],kp[1])[:,:2]
        pixels=np.floor(xy).astype(int)
        valid=(pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]<mask.shape[0])
        valid[valid]=mask[pixels[valid,1],pixels[valid,0]]>0
        valid_kps[iid]=valid
    for val in vals:
        iid=image_ids[val['name']]
        kp=conn.execute('select rows,cols,data from keypoints where image_id=?',(iid,)).fetchone()
        kps=np.frombuffer(kp[2],dtype=np.float32).reshape(kp[0],kp[1])[:,:2]
        candidates={}
        for im in rec.images.values():
            if not im.has_pose:continue
            lo,hi=sorted([iid,im.image_id]);pair_id=lo*2147483647+hi
            row=conn.execute('select rows,cols,data from two_view_geometries where pair_id=?',(pair_id,)).fetchone()
            if not row or row[0]==0:continue
            matches=np.frombuffer(row[2],dtype=np.uint32).reshape(row[0],row[1])
            if iid!=lo:matches=matches[:,::-1]
            for vi,ti in matches:
                if not valid_kps[iid][int(vi)] or not valid_kps[im.image_id][int(ti)]:continue
                pt=im.points2D[int(ti)]
                if pt.has_point3D():
                    candidates.setdefault(int(vi),[]).append(pt.point3D_id)
        correspond=[];used=set()
        for vi,ids in candidates.items():
            counts={x:ids.count(x) for x in set(ids)}
            pid=max(counts,key=counts.get)
            if pid not in used:
                correspond.append((vi,pid));used.add(pid)
        if len(correspond)<15:
            reports.append(dict(name=val['name'],status='insufficient_correspondences',n=len(correspond),
                validation_group=val.get('validation_group','interleaved')));continue
        xy=np.array([kps[i] for i,_ in correspond],dtype=float)
        xyz=np.array([rec.points3D[j].xyz for _,j in correspond])
        est=p.AbsolutePoseEstimationOptions();est.ransac.max_error=3.;est.ransac.random_seed=2296
        result=p.estimate_and_refine_absolute_pose(xy,xyz,cam,est)
        if result is None or result['num_inliers']<15:
            reports.append(dict(name=val['name'],status='pose_failed',n=len(correspond),
                validation_group=val.get('validation_group','interleaved')));continue
        pose=result['cam_from_world'];w2c=np.eye(4);w2c[:3]=pose.matrix()
        err=np.linalg.norm(cam.img_from_cam(pose*xyz)-xy,axis=1)
        inliers=np.array(result['inlier_mask'],dtype=bool)
        supported=xy[inliers].astype(np.float32)
        hull_area=float(cv2.contourArea(cv2.convexHull(supported)))/(cam.width*cam.height)
        cells=np.floor(supported/np.array([cam.width,cam.height])*4).astype(int).clip(0,3)
        cell_count=len(set(map(tuple,cells)))
        estimated_center=pose.inverse().translation
        near_time,near_center=min(train_cameras,key=lambda x:abs(x[0]-val['time_seconds']))
        gap=abs(near_time-val['time_seconds'])
        neighbor_distance=float(np.linalg.norm(estimated_center-near_center))
        allowed_distance=max(.15*extent,5*typical_speed*max(gap,.25))
        if hull_area<.025 or cell_count<4 or neighbor_distance>allowed_distance:
            reports.append(dict(name=val['name'],status='rejected_weak_or_aliased_pose',
                inliers=int(result['num_inliers']),hull_fraction=hull_area,grid_cells=cell_count,
                camera_neighbor_distance=neighbor_distance,allowed_neighbor_distance=allowed_distance,
                validation_group=val.get('validation_group','interleaved')))
            continue
        report=dict(name=val['name'],status='localized_fixed_map',correspondences=len(correspond),
                    inliers=int(result['num_inliers']),reprojection_median_px=float(np.median(err[inliers])),
                    reprojection_p95_px=float(np.percentile(err[inliers],95)),
                    hull_fraction=hull_area,grid_cells=cell_count,camera_neighbor_distance=neighbor_distance,
                    validation_group=val.get('validation_group','interleaved'))
        reports.append(report)
        image=cv2.imread(val['image_path'])
        mapx,mapy=maps(cam,K,cam.width,cam.height)
        cv2.imwrite(str(output/'images'/val['name']),cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR),[cv2.IMWRITE_JPEG_QUALITY,98])
        maskpath=output/'masks'/(val['name']+'.png');maskpath.parent.mkdir(exist_ok=True)
        remap_mask(root,val['name'],cam,K,cam.width,cam.height,maskpath)
        valitems.append(dict(name=val['name'],image_path=str(output/'images'/val['name']),
            w2c=w2c.tolist(),K=K.tolist(),width=cam.width,height=cam.height,time_seconds=val['time_seconds'],
            mask_path=str(maskpath),validation_group=val.get('validation_group','interleaved')))
    conn.close()
    dataset=json.loads((root/'dataset.json').read_text());dataset['val']=valitems
    grouped={}
    for report in reports:
        group=grouped.setdefault(report['validation_group'],dict(attempted=0,localized=0,rejected=0))
        group['attempted']+=1
        group['localized' if report['status']=='localized_fixed_map' else 'rejected']+=1
    dataset['validation_localization']=dict(attempted=len(vals),localized=len(valitems),
        rejected=len(vals)-len(valitems),by_group=grouped)
    write(root/'dataset.json',dataset)
    write(output/'localization.json',dict(attempted=len(vals),localized=len(valitems),
        rejected=len(vals)-len(valitems),by_group=grouped,
        source_sha256=frames['source_sha256'],frames_sha256=sha(root/'frames.json'),
        model_path=str(root/a.model_name),
        model_artifacts={x.name:sha(x) for x in (root/a.model_name).glob('*.bin')},
        camera_mask_receipt_sha256=sha(root/'camera_masks'/'receipt.json'),
        method='SIFT matches to frozen train-map 3D points; robust PnP; no bundle/map updates',views=reports))
    print('Localized',len(valitems),'/',len(vals),'held-out frames',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['extract','sfm','prepare','dense','localize'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('/home/ubuntu/Downloads/IMG_2296.MOV'))
    parser.add_argument('--fps',type=float,default=4)
    parser.add_argument('--max-size',type=int,default=1280)
    parser.add_argument('--stereo-gpus',default='0',help='Logical CUDA devices exposed to the owned stereo job.')
    parser.add_argument('--model-name',default='model')
    parser.add_argument('--seed-model',default='')
    parser.add_argument('--exhaustive',action='store_true')
    parser.add_argument('--map-only',action='store_true')
    parser.add_argument('--database-name',default='database.db')
    parser.add_argument('--init-pair',nargs=2)
    parser.add_argument('--init-min-angle',type=float,default=16.)
    parser.add_argument('--fixed-seed',action='store_true')
    parser.add_argument('--abs-pose-ratio',type=float,default=.25)
    a=parser.parse_args();a.run=a.run.resolve();a.source=a.source.resolve()
    if a.stage!='extract':
        manifest=json.loads((a.run/'frames.json').read_text())
        if manifest['source_sha256']!=sha(a.source):
            raise RuntimeError('Original source hash changed.')
        expected={x['name'] for x in manifest['frames'] if x['split']=='train'}
        actual={x.name for x in (a.run/'images').glob('*.jpg')}
        if expected!=actual:
            raise RuntimeError('Training image directory does not match the frame manifest.')
        for frame in manifest['frames']:
            if not Path(frame['image_path']).is_file() or sha(frame['image_path'])!=frame['sha256']:
                raise RuntimeError(f"Frame hash mismatch: {frame['name']}")
    globals()[a.stage](a)


if __name__=='__main__':main()
