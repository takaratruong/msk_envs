#!/usr/bin/env python3
"""Initialize COLMAP in a learned camera frame, retaining exact SIFT indices.

Learned cameras are an initialization hypothesis, never measured poses/depth.
The original matching database and all failed maps remain unchanged.
"""
import argparse,hashlib,json,sqlite3
from pathlib import Path
import numpy as np
import pycolmap as p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--poses',type=Path,required=True)
    parser.add_argument('--source-db',type=Path,required=True)
    parser.add_argument('--name',default='vggt_seed')
    a=parser.parse_args();root=a.run.resolve();source=a.source_db.resolve()
    frames=json.loads((root/'frames.json').read_text());by={x['name']:x for x in frames['frames']}
    poses=json.loads(a.poses.read_text());views=poses['views']
    prediction_result=json.loads((a.poses.parent/'result.json').read_text())
    if prediction_result['artifacts']['cameras.json']['sha256']!=sha(a.poses):
        raise RuntimeError('Learned camera receipt does not match current camera bytes.')
    if prediction_result['provenance']['frames_sha256']!=sha(root/'frames.json'):
        raise RuntimeError('Learned inference uses a different frame split.')
    if any(by[x['name']]['split']!='train' for x in views):
        raise RuntimeError('Learned seed contains a non-training image.')
    if prediction_result['provenance']['source_sha256']!=frames['source_sha256']:
        raise RuntimeError('Learned camera source mismatch.')
    Ks=np.array([x['K'] for x in views],dtype=float)
    focal=float(np.median(Ks[:,[0,1],[0,1]]))
    database=root/('database_'+a.name+'.db');out=root/('model_'+a.name)
    if database.exists() or out.exists():raise RuntimeError('Use a new seed name; output exists.')
    with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as src,sqlite3.connect(database) as dst:src.backup(dst)
    conn=sqlite3.connect(database)
    cameras=list(conn.execute('select camera_id,model,width,height from cameras'))
    if len(cameras)!=1:raise RuntimeError('Expected one original shared camera.')
    cid,model,width,height=cameras[0]
    params=np.array([focal,float(np.median(Ks[:,0,2])),float(np.median(Ks[:,1,2])),0.],dtype=np.float64)
    conn.execute('update cameras set params=?,prior_focal_length=1 where camera_id=?',(params.tobytes(),cid));conn.commit()
    cam=p.Camera(camera_id=cid,model='SIMPLE_RADIAL',width=width,height=height,params=params,has_prior_focal_length=True)
    rec=p.Reconstruction();rec.add_camera_with_trivial_rig(cam)
    image_ids={n:i for i,n in conn.execute('select image_id,name from images')}
    for view in views:
        name=view['name'];iid=image_ids[name];w2c=np.array(view['w2c'],dtype=float)
        if w2c.shape!=(4,4) or not np.allclose(w2c[3],[0,0,0,1]) or not np.allclose(w2c[:3,:3]@w2c[:3,:3].T,np.eye(3),atol=1e-4):
            raise RuntimeError('Invalid learned camera transform: '+name)
        rows,cols,blob=conn.execute('select rows,cols,data from keypoints where image_id=?',(iid,)).fetchone()
        xy=np.frombuffer(blob,np.float32).reshape(rows,cols)[:,:2].astype(float)
        im=p.Image(name=name,keypoints=xy,camera_id=cid,image_id=iid)
        rec.add_image_with_trivial_frame(im,p.Rigid3d(w2c[:3]))
    conn.close()
    raw=root/('model_'+a.name+'_raw');raw.mkdir();rec.write(raw)
    options=p.IncrementalPipelineOptions();options.num_threads=8;options.random_seed=2296
    options.image_names=[x['name'] for x in views];options.structure_less_registration_fallback=False
    options.triangulation.ignore_two_view_tracks=False
    options.triangulation.create_max_angle_error=2.
    options.triangulation.min_angle=2.
    options.mapper.filter_max_reproj_error=8.
    options.ba_global_max_num_iterations=100
    out.mkdir()
    rec=p.triangulate_points(rec,database,root/'images',out,options=options,refine_intrinsics=False)
    # At this stage poses remain the learned hypothesis. Registration and
    # photometric/trajectory checks determine whether it can be promoted.
    rec.write(out);rec.write_text(out);rec.export_PLY(root/(a.name+'_points.ply'))
    result=dict(schema='real2sim-learned-seed/v1',source_sha256=frames['source_sha256'],
        source_poses_sha256=sha(a.poses),source_db_sha256=sha(source),database_sha256=sha(database),
        registered=rec.num_reg_images(),points=rec.num_points3D(),units='arbitrary',
        metric_scale_verified=False,learned_depth_is_measurement=False,
        shared_focal_prior=focal,individual_focal_range=[float(Ks[:,[0,1],[0,1]].min()),float(Ks[:,[0,1],[0,1]].max())],
        model_artifacts={x.name:sha(x) for x in out.glob('*.bin')})
    (root/(a.name+'_receipt.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
