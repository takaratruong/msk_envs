#!/usr/bin/env python3
"""Raycast a candidate triangle mesh at source cameras for visual audit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def rays(view,width):
    scale=min(1.,width/view['width']);w=round(view['width']*scale);h=round(view['height']*scale)
    K=np.array(view['K'],dtype=np.float64);K[0]*=w/view['width'];K[1]*=h/view['height']
    yy,xx=np.mgrid[:h,:w]
    directions=np.stack([(xx+.5-K[0,2])/K[0,0],(yy+.5-K[1,2])/K[1,1],np.ones_like(xx)],axis=-1)
    c2w=np.linalg.inv(np.array(view['w2c']))
    directions=directions@c2w[:3,:3].T
    origins=np.broadcast_to(c2w[:3,3],directions.shape)
    return np.concatenate([origins,directions],axis=-1).astype(np.float32),w,h


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--geometry-dir',type=Path)
    a=p.parse_args();root=a.run.resolve();geometry=(a.geometry_dir or root/'geometry').resolve()
    out=geometry/'review';out.mkdir(exist_ok=True)
    data=json.loads((root/'dataset.json').read_text())
    metrics=json.loads((geometry/'metrics.json').read_text())
    arrays=geometry/'surface_arrays.npz'
    if metrics['source_sha256']!=data['source_sha256'] or metrics['dataset_sha256']!=sha(root/'dataset.json') or metrics['artifacts']['surface_arrays.npz']!=sha(arrays):
        raise RuntimeError('Mesh evidence does not match the current inputs.')
    arr=np.load(arrays);vertices=arr['vertices'];triangles=arr['triangles'];colors=arr['colors']
    if not len(triangles):raise RuntimeError('Empty observed mesh.')
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(vertices,o3d.core.Dtype.Float32),
                        o3d.core.Tensor(triangles.astype(np.uint32),o3d.core.Dtype.UInt32))
    selected={}
    for t in [0,8.3,10.,13.,21.,30.,38.,50.,60.,68.,72.,75.]:
        v=min(data['train'],key=lambda x:abs(x['time_seconds']-t));selected[v['name']]=dict(v,split='train')
    for v in data['val']:
        if v.get('validation_group')=='buffered_block':selected[v['name']]=dict(v,split='val/buffered_block')
    rows=[]
    basis='LEARNED DEPTH PRIOR' if metrics.get('inferred_prior') else 'OBSERVED MVS SURFACE'
    for index,view in enumerate(sorted(selected.values(),key=lambda v:v['time_seconds'])):
        ray,w,h=rays(view,a.width)
        cast=scene.cast_rays(o3d.core.Tensor(ray),nthreads=8)
        depth=cast['t_hit'].numpy();hit=np.isfinite(depth)
        ids=cast['primitive_ids'].numpy()[hit]
        uv=cast['primitive_uvs'].numpy()[hit]
        weights=np.stack([1-uv.sum(axis=1),uv[:,0],uv[:,1]],axis=1)
        rgb=np.full((h,w,3),[35,22,42],dtype=np.uint8)
        if len(colors)==len(vertices):
            rgb[hit]=np.rint(np.clip((colors[triangles[ids]]*weights[:,:,None]).sum(axis=1),0,1)*255).astype(np.uint8)
        else:rgb[hit]=150
        normals=cast['primitive_normals'].numpy()
        normal_img=np.full((h,w,3),[35,22,42],dtype=np.uint8)
        normal_img[hit]=np.rint((np.clip(normals[hit],-1,1)*.5+.5)*255).astype(np.uint8)
        reference=Image.open(view['image_path']).convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(w*3,h+58),(19,23,30));draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f"{view['name']} | source {view['time_seconds']:.3f}s | {view['split']} | {basis}; unknown scale",fill='white')
        for col,(label,img) in enumerate([('SOURCE',reference),('MESH VERTEX COLOR | PURPLE = NO SURFACE',Image.fromarray(rgb)),('MESH NORMALS IN RAW FRAME',Image.fromarray(normal_img))]):
            draw.text((col*w+8,31),label,fill='white');canvas.paste(img,(col*w,58))
        path=out/f'{index:02d}_{Path(view["name"]).stem}.jpg';canvas.save(path,quality=95,subsampling=0)
        row=dict(name=view['name'],source_time_seconds=view['time_seconds'],split=view['split'],
            image_sha256=sha(view['image_path']),width=w,height=h,ray_hit_fraction=float(hit.mean()),
            depth_arbitrary_quantiles=np.percentile(depth[hit],[1,50,99]).tolist() if hit.any() else [],
            comparison=str(path),comparison_sha256=sha(path))
        if view.get('mask_path'):
            mask=np.array(Image.open(view['mask_path']).convert('L').resize((w,h),Image.Resampling.NEAREST))==255
            row['static_ray_hit_fraction']=float(hit[mask].mean())
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(schema='real2sim-mesh-ray-review/v1',produced_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=data['source_sha256'],dataset_sha256=sha(root/'dataset.json'),
        geometry_metrics_sha256=sha(geometry/'metrics.json'),mesh_arrays_sha256=sha(arrays),geometry_basis=basis,
        implementation_sha256=sha(__file__),views=rows,metric_accuracy_verified=False,
        interpretation='Ray hits measure candidate mesh image coverage only, not geometric correctness or independent survey completeness. Geometry basis is explicitly identified; learned priors are not range measurements.')
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    body=''.join(f'<figure><img src="{Path(v["comparison"]).name}" style="width:100%"><figcaption>{v["name"]}, {v["source_time_seconds"]:.3f}s; mesh ray hits {v["ray_hit_fraction"]:.1%}</figcaption></figure>' for v in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Candidate mesh audit</title><style>body{background:#13171e;color:#eee;font:16px system-ui;max-width:1800px;margin:30px auto}figure{margin:24px 0}figcaption{padding:8px}</style><h1>'+basis+' against source cameras</h1><p>Purple marks missing mesh. Colors and normals expose surface placement and holes. Units remain arbitrary; ray-hit coverage is not a metric accuracy test. Learned priors, when identified above, are not measured depth.</p>'+body)


if __name__=='__main__':main()
