#!/usr/bin/env python3
"""Robustly refine learned initialization with image tracks, auditing every view."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pycolmap as p


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(rec):
    rows=[];all_errors=[]
    for im in rec.images.values():
        if not im.has_pose:continue
        cam=rec.cameras[im.camera_id]
        observations=[x for x in im.points2D if x.has_point3D()]
        xy=np.array([x.xy for x in observations],dtype=np.float32)
        hull=float(cv2.contourArea(cv2.convexHull(xy)))/(cam.width*cam.height) if len(xy)>=3 else 0.
        errs=[]
        if observations:
            xyz=np.array([rec.points3D[x.point3D_id].xyz for x in observations])
            errs=np.linalg.norm(cam.img_from_cam(im.cam_from_world()*xyz)-xy,axis=1)
            all_errors.extend(errs.tolist())
        rows.append(dict(name=im.name,image_id=im.image_id,frame_id=im.frame_id,
            point_observations=len(observations),hull_fraction=hull,
            median_reprojection_px=float(np.median(errs)) if len(errs) else None,
            p95_reprojection_px=float(np.percentile(errs,95)) if len(errs) else None))
    return rows,all_errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--input-model',required=True)
    parser.add_argument('--output-model',required=True)
    parser.add_argument('--prune-only',action='store_true')
    parser.add_argument('--reject-point-ids',nargs='*',type=int,default=[])
    parser.add_argument('--floor-evidence',type=Path)
    parser.add_argument('--max-below-floor',type=float,default=.1)
    a=parser.parse_args();root=a.run.resolve();source=root/a.input_model;out=root/a.output_model
    if out.exists():raise RuntimeError('Output exists: use a fresh refinement name.')
    rec=p.Reconstruction(source);history=[]
    source_hashes={x.name:sha(x) for x in source.glob('*.bin')}
    rejected_points=[];floor_filter=None;reject_ids=set(a.reject_point_ids)
    if a.floor_evidence:
        if not a.prune_only:raise RuntimeError('Floor constraint requires frozen coordinates (--prune-only).')
        floor=json.loads(a.floor_evidence.read_text())
        if floor['source_sha256']!=json.loads((root/'frames.json').read_text())['source_sha256']:
            raise RuntimeError('Floor evidence belongs to another capture.')
        reference=root/floor['source_model']
        if {x.name:sha(x) for x in reference.glob('*.bin')}!=floor['source_model_artifacts']:
            raise RuntimeError('Floor evidence source model changed.')
        ref=p.Reconstruction(reference)
        common=set(ref.points3D)&set(rec.points3D)
        if len(common)<100 or any(not np.allclose(ref.points3D[k].xyz,rec.points3D[k].xyz,rtol=0,atol=1e-10) for k in common):
            raise RuntimeError('Floor evidence coordinate frame differs from the input model.')
        normal=np.array(floor['plane_normal']);offset=float(floor['plane_offset'])
        if abs(np.linalg.norm(normal)-1)>1e-6 or a.max_below_floor<=0:
            raise RuntimeError('Invalid signed floor constraint.')
        floor_ids={k for k,v in rec.points3D.items() if v.xyz@normal+offset>a.max_below_floor}
        reject_ids|=floor_ids
        floor_filter=dict(evidence_path=str(a.floor_evidence.resolve()),evidence_sha256=sha(a.floor_evidence),
            plane_normal=normal.tolist(),plane_offset=offset,max_below_floor=a.max_below_floor,
            units='arbitrary',positive_side='below visible floor',rejected_count=len(floor_ids),
            coordinate_checked_common_points=len(common),metric_validation=False)
    for point_id in sorted(reject_ids):
        point=rec.points3D[point_id]
        rejected_points.append(dict(id=point_id,xyz=point.xyz.tolist(),
            tracks=[dict(image_id=x.image_id,point2D_idx=x.point2D_idx) for x in point.track.elements]))
        rec.delete_point3D(point_id)
    for threshold in ([] if a.prune_only else [8.,5.,3.]):
        rows,before=evidence(rec);config=p.BundleAdjustmentConfig();fixed=[]
        for row in rows:
            if row['point_observations']==0:continue
            config.add_image(row['image_id'])
            if row['point_observations']<30 or row['hull_fraction']<.025:
                config.set_constant_rig_from_world_pose(row['frame_id']);fixed.append(row['name'])
        config.fix_gauge(p.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        options=p.BundleAdjustmentOptions();options.refine_principal_point=False
        options.ceres.loss_function_type=p.LossFunctionType.SOFT_L1
        options.ceres.loss_function_scale=2.
        options.ceres.solver_options.num_threads=8
        options.ceres.solver_options.max_num_iterations=100
        options.print_summary=True
        solver=p.create_default_bundle_adjuster(options,config,rec)
        solver.solve()
        removed=p.ObservationManager(rec).filter_all_points3D(threshold,2.)
        after_rows,after=evidence(rec)
        record=dict(max_error_px=threshold,fixed_weak_pose_count=len(fixed),fixed_weak_poses=fixed,
            before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),
            removed_observations=removed,points=rec.num_points3D())
        history.append(record);print(json.dumps(record),flush=True)
    # Poses without useful track support are preserved in the full diagnostic
    # model. The render/MVS model excludes them instead of labeling them solved.
    diagnostic=root/(a.output_model+'_all_poses');diagnostic.mkdir();rec.write(diagnostic)
    rows,errors=evidence(rec);original_pose_count=len(rows);excluded=[];prune_round=0
    while True:
        rows,_=evidence(rec)
        weak=[row for row in rows if row['point_observations']<30 or row['hull_fraction']<.025]
        if not weak:break
        prune_round+=1
        for row in weak:
            excluded.append(dict(**row,prune_round=prune_round))
            if rec.images[row['image_id']].has_pose:rec.deregister_frame(row['frame_id'])
    out.mkdir();rec.write(out);rec.write_text(out);rec.export_PLY(root/(a.output_model+'_points.ply'))
    remaining,errors=evidence(rec)
    frames=json.loads((root/'frames.json').read_text())
    metrics=dict(schema='real2sim-sfm/v1',source_sha256=frames['source_sha256'],
        selected_train=sum(x['split']=='train' for x in frames['frames']),registered=rec.num_reg_images(),
        points=rec.num_points3D(),reprojection_median_px=float(np.median(errors)),
        reprojection_p95_px=float(np.percentile(errors,95)),mean_track_length=rec.compute_mean_track_length(),
        units='arbitrary',metric_scale_verified=False,pose_initialization=a.input_model,
        learned_camera_count=original_pose_count,excluded_weakly_constrained_views=excluded,views=remaining,
        explicitly_rejected_points=rejected_points,support_pruning_rounds=prune_round,floor_filter=floor_filter,
        unregistered=[x['name'] for x in frames['frames'] if x['split']=='train' and x['name'] not in {v['name'] for v in remaining}],
        cameras={str(k):dict(model=c.model_name,params=c.params.tolist(),width=c.width,height=c.height) for k,c in rec.cameras.items()},
        source_model_hashes=source_hashes,history=history,
        model_artifacts={x.name:sha(x) for x in out.glob('*.bin')})
    metrics['registered_fraction']=metrics['registered']/metrics['selected_train']
    metrics['cameras_with_broad_track_support']=sum(x['point_observations']>=30 and x['hull_fraction']>=.04 for x in remaining)
    (root/(a.output_model+'_metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({k:metrics[k] for k in ['registered','points','reprojection_median_px','reprojection_p95_px']}),flush=True)


if __name__=='__main__':main()
