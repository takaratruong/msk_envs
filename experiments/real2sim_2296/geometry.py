#!/usr/bin/env python3
"""Preserve observed MVS surfaces; export a unitless, non-certified mesh candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import open3d as o3d


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--max-points',type=int,default=2000000)
    a=p.parse_args();root=a.run.resolve();out=root/'geometry';out.mkdir(exist_ok=True)
    source=root/'dense'/'fused.ply';cloud=o3d.io.read_point_cloud(str(source))
    xyz=np.asarray(cloud.points)
    if len(xyz)<100 or not np.isfinite(xyz).all():raise RuntimeError('No valid dense point cloud.')
    raw_count=len(xyz)
    data=json.loads((root/'dataset.json').read_text())
    dense_receipt_path=root/'dense_provenance.json'
    dense_receipt=json.loads(dense_receipt_path.read_text())
    if (dense_receipt.get('status')!='verified_post_run' or
        dense_receipt['source_sha256']!=data['source_sha256'] or
        dense_receipt['dataset_sha256']!=sha(root/'dataset.json') or
        dense_receipt['fused_sha256']!=sha(source)):
        raise RuntimeError('Dense audit/source/dataset/fused points do not agree.')
    model=Path(data['model_path'])
    model_metrics=json.loads((root/(model.name+'_metrics.json')).read_text())
    if model_metrics['model_artifacts']!={x.name:sha(x) for x in model.glob('*.bin')}:
        raise RuntimeError('Geometry camera model differs from refinement evidence.')
    floor=model_metrics.get('floor_filter')
    floor_removed=0
    if floor:
        evidence=Path(floor['evidence_path'])
        if sha(evidence)!=floor['evidence_sha256']:
            raise RuntimeError('Floor evidence changed after camera refinement.')
        signed=xyz@np.array(floor['plane_normal'])+floor['plane_offset']
        keep=np.flatnonzero(signed<=floor['max_below_floor'])
        floor_removed=len(xyz)-len(keep)
        cloud=cloud.select_by_index(keep);xyz=np.asarray(cloud.points)
    centers=np.array([np.linalg.inv(np.array(x['w2c']))[:3,3] for x in data['train']])
    center=np.median(centers,axis=0)
    extent=float(np.percentile(np.linalg.norm(centers-center,axis=1),90))
    if extent<=1e-9:raise RuntimeError('Degenerate camera frame.')
    # This guard rejects numerical runaway points, not unseen room surfaces.
    finite_ids=np.where(np.linalg.norm(xyz-center,axis=1)<20*extent)[0]
    cloud=cloud.select_by_index(finite_ids)
    before_filter=len(cloud.points)
    cloud,_=cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=3)
    after_filter=len(cloud.points)
    voxel=extent/1500
    cloud=cloud.voxel_down_sample(voxel)
    while len(cloud.points)>a.max_points:
        voxel*=1.2;cloud=cloud.voxel_down_sample(voxel)
    if not cloud.has_normals():
        raise RuntimeError('Fusion must supply measured multiview normals before meshing.')
    cloud.normalize_normals()
    o3d.io.write_point_cloud(str(out/'observed_points.ply'),cloud)
    distances=np.asarray(cloud.compute_nearest_neighbor_distance())
    spacing=float(np.median(distances[distances>0]))
    radii=[1.5*spacing,3*spacing,6*spacing]
    print('Meshing',len(cloud.points),'points; radii in arbitrary units',radii,flush=True)
    mesh=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud,o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out/'observed_surface.ply'),mesh)
    # No explicit global hole completion is performed. Local ball pivots can
    # still bridge unsampled gaps; this is an unverified collision candidate.
    np.savez_compressed(out/'surface_arrays.npz',vertices=np.asarray(mesh.vertices).astype(np.float32),
        triangles=np.asarray(mesh.triangles).astype(np.int32),colors=np.asarray(mesh.vertex_colors).astype(np.float32),
        normals=np.asarray(mesh.vertex_normals).astype(np.float32))
    triangles=np.asarray(mesh.triangles)
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    result=dict(schema='real2sim-observed-geometry/v1',source_sha256=data['source_sha256'],
        dataset_sha256=sha(root/'dataset.json'),
        dense_provenance_sha256=sha(dense_receipt_path),
        fused_sha256=sha(source),units='arbitrary',metric_geometry_verified=False,
        raw_points=raw_count,floor_filter=floor,below_floor_points_removed=floor_removed,
        runaway_points_removed=raw_count-floor_removed-before_filter,
        statistical_outliers_removed=before_filter-after_filter,meshing_points=len(cloud.points),
        voxel_size=voxel,median_neighbor_spacing=spacing,ball_pivot_radii=radii,
        vertices=len(mesh.vertices),triangles=len(mesh.triangles),boundary_edges=int(sum(counts==1)),
        nonmanifold_edges=int(sum(counts>2)),watertight=bool(mesh.is_watertight()),
        collision_certified=False,articulation_separated=False,
        caveats=['MVS-derived surface, not sensor depth or independent survey.',
          'No global hole completion; local ball pivoting can bridge unsampled gaps and must be checked.',
          'Glass/reflective surfaces may contain optical ghosts or missing returns.',
          'Furniture and handles are not yet segmented into dynamic rigid bodies.',
          'Vertex color is a diagnostic appearance; use the linked Gaussian layer for visual review.'],
        artifacts={x.name:sha(x) for x in out.iterdir() if x.is_file() and x.suffix in ('.ply','.npz')})
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
