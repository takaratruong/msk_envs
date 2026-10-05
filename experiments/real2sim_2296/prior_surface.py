#!/usr/bin/env python3
"""Bounded inferred-depth TSDF pilot; never measured depth or certified collision."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sample(array, xy, interpolation=cv2.INTER_LINEAR):
    xy = np.asarray(xy, np.float32)
    if len(xy) == 0:
        return np.empty(0, dtype=np.float32)
    return cv2.remap(array.astype(np.float32), xy[:, 0:1], xy[:, 1:2],
                     interpolation, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0).reshape(-1)


def fit_scale(predicted, reference, xy, width, height, min_points=20):
    """Positive multiplicative camera-Z fit; robust rejects never change cameras."""
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    good = np.isfinite(predicted) & np.isfinite(reference) & (predicted > 0) & (reference > 0)
    ids = np.flatnonzero(good)
    result = {'candidate_tracks': int(len(predicted)), 'finite_positive_tracks': int(len(ids)),
              'accepted': False, 'reasons': []}
    if len(ids) < min_points:
        result['reasons'].append('fewer_than_minimum_positive_tracks')
        return result
    logs = np.log(reference[ids] / predicted[ids])
    median = float(np.median(logs))
    mad = float(np.median(np.abs(logs - median)))
    # Floor allows modest depth noise, cap prevents an incoherent view accepting all tracks.
    cutoff = float(np.clip(3 * 1.4826 * mad, 0.10, 0.35))
    keep = np.abs(logs - median) <= cutoff
    inliers = ids[keep]
    scale = float(np.exp(np.median(logs[keep])))
    rel = np.abs(predicted[ids] * scale - reference[ids]) / reference[ids]
    inlier_rel = np.abs(predicted[inliers] * scale - reference[inliers]) / reference[inliers]
    hull = cv2.convexHull(np.asarray(xy)[inliers].astype(np.float32))
    hull_fraction = float(cv2.contourArea(hull) / (width * height)) if len(hull) >= 3 else 0.0
    result.update(scale=scale, log_ratio_mad=mad, log_inlier_cutoff=cutoff,
                  inlier_tracks=int(len(inliers)), inlier_fraction=float(np.mean(keep)),
                  inlier_hull_fraction=hull_fraction,
                  median_relative_error_all=float(np.median(rel)),
                  p90_relative_error_all=float(np.percentile(rel, 90)),
                  median_relative_error_inliers=float(np.median(inlier_rel)),
                  p90_relative_error_inliers=float(np.percentile(inlier_rel, 90)),
                  median_absolute_error_inliers=float(np.median(np.abs(predicted[inliers] * scale - reference[inliers]))),
                  median_reference_depth=float(np.median(reference[inliers])))
    if len(inliers) < min_points:
        result['reasons'].append('too_few_inlier_tracks')
    if result['inlier_fraction'] < .5:
        result['reasons'].append('less_than_half_tracks_fit')
    if result['median_relative_error_all'] > .20:
        result['reasons'].append('median_all_track_relative_error_above_20_percent')
    if result['p90_relative_error_inliers'] > .30:
        result['reasons'].append('p90_inlier_relative_error_above_30_percent')
    if hull_fraction < .02:
        result['reasons'].append('inlier_spatial_support_below_2_percent')
    result['accepted'] = not result['reasons']
    return result


def target_rays(view, camera, width, prediction_width, prediction_height):
    height = round(view['height'] * width / view['width'])
    k = np.array(view['K'], dtype=float)
    k[0] *= width / view['width']
    k[1] *= height / view['height']
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    # Dataset K follows COLMAP corner coordinates. Open3D uses integer centers.
    x, y = (xx + .5 - k[0, 2]) / k[0, 0], (yy + .5 - k[1, 2]) / k[1, 1]
    if camera.model_name != 'SIMPLE_RADIAL':
        raise ValueError('This pilot explicitly supports the current SIMPLE_RADIAL source camera only.')
    f, cx, cy, radial = np.array(camera.params)
    factor = 1 + radial * (x * x + y * y)
    u = (f * x * factor + cx) * prediction_width / camera.width - .5
    v = (f * y * factor + cy) * prediction_height / camera.height - .5
    k_o3d = k.copy()
    k_o3d[:2, 2] -= .5
    return u.astype(np.float32), v.astype(np.float32), k_o3d


def cross_view_check(views, depths, ks):
    """Diagnostic predicted-surface agreement; occlusion makes this conservative."""
    pairs = []
    for i in range(len(views) - 1):
        j = i + 1
        if views[j]['time_seconds'] - views[i]['time_seconds'] > 1.5:
            continue
        d = depths[i]
        yy, xx = np.mgrid[4:d.shape[0]:12, 4:d.shape[1]:12]
        z = d[yy, xx].reshape(-1)
        pixels = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(len(z))], axis=1)
        xyz = (pixels @ np.linalg.inv(ks[i]).T) * z[:, None]
        transform = np.array(views[j]['w2c']) @ np.linalg.inv(np.array(views[i]['w2c']))
        other = xyz @ transform[:3, :3].T + transform[:3, 3]
        proj = other @ ks[j].T
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
        predicted_other = sample(depths[j], uv)
        good = (z > 0) & (other[:, 2] > 0) & (predicted_other > 0)
        good &= (uv[:, 0] >= 1) & (uv[:, 0] < d.shape[1] - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < d.shape[0] - 2)
        if good.sum() < 20:
            continue
        error = np.abs(predicted_other[good] - other[good, 2]) / other[good, 2]
        pairs.append({'a': views[i]['name'], 'b': views[j]['name'], 'samples': int(good.sum()),
                      'median_relative_error': float(np.median(error)),
                      'p90_relative_error': float(np.percentile(error, 90))})
    return {'method': 'Adjacent accepted cameras up to 1.5 seconds; projected depth samples, including occlusion errors.',
            'independent_validation': False, 'pairs': pairs,
            'median_pair_median_relative_error': float(np.median([p['median_relative_error'] for p in pairs])) if pairs else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--voxel-divisor', type=float, default=500)
    p.add_argument('--confidence-percentile', type=float, default=15)
    p.add_argument('--fit-only', action='store_true')
    a = p.parse_args()
    started = time.time()
    root, predpath = a.run.resolve(), a.predictions.resolve()
    out = (a.output or root / 'geometry_prior').resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists() or (out / 'surface_arrays.npz').exists():
        raise FileExistsError('Refusing to replace completed pilot output; select a fresh --output.')
    dataset_path = root / 'dataset.json'
    data = json.loads(dataset_path.read_text())
    dataset_sha = sha(dataset_path)
    modelpath = Path(data['model_path'])
    model_hashes = {x.name: sha(x) for x in modelpath.glob('*.bin')}
    if model_hashes != data['model_artifacts']:
        raise ValueError('Stable model does not match dataset hashes.')
    rec = pycolmap.Reconstruction(str(modelpath))
    model_images = {im.name: im for im in rec.images.values()}
    selection_path = predpath.parent / 'selected_frames.json'
    selected = json.loads(selection_path.read_text())
    if selected['provenance']['source_sha256'] != data['source_sha256']:
        raise ValueError('Predictions and dataset sources differ.')
    frames = selected['frames']
    if any(f['split'] != 'train' for f in frames) or len({f['name'] for f in frames}) != len(frames):
        raise ValueError('Prediction bundle must contain unique training-only frames.')
    selected_by_name = {f['name']: (i, f) for i, f in enumerate(frames)}
    train = sorted(data['train'], key=lambda v: v['time_seconds'])
    if len({v['name'] for v in train}) != len(train) or set(v['name'] for v in train) != set(model_images):
        raise ValueError('Training/model names differ or repeat.')
    manifest_path = root / 'appearance_v2' / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    unsigned = dict(manifest)
    digest = unsigned.pop('inputs_sha256')
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != digest:
        raise ValueError('Frozen prepared-image manifest digest mismatch.')
    if manifest['dataset_sha256'] != dataset_sha:
        raise ValueError('Current dataset differs from frozen prepared-image manifest.')
    frozen = {v['name']: v for v in manifest['images'] if v['split'] == 'train'}
    source_bindings = {}
    for view in train:
        name = view['name']
        idx, frame = selected_by_name[name]
        if frame['split'] != 'train' or frozen[name]['split'] != 'train':
            raise ValueError('Non-training view selected.')
        source_image_sha = sha(frame['image_path'])
        image_sha, mask_sha = sha(view['image_path']), sha(view['mask_path'])
        if source_image_sha != frame['sha256'] or image_sha != frozen[name]['sha256'] or mask_sha != frozen[name]['mask_sha256']:
            raise ValueError('Source/prepared image or mask hash mismatch: ' + name)
        w2c = np.eye(4)
        w2c[:3] = model_images[name].cam_from_world().matrix()
        if not np.allclose(w2c, view['w2c'], atol=1e-12, rtol=0):
            raise ValueError('Dataset/model poses differ: ' + name)
        source_bindings[name] = {'source_sha256': source_image_sha, 'prepared_image_sha256': image_sha, 'mask_sha256': mask_sha}
    with np.load(predpath) as z:
        depth, confidence, predicted_mask = z['depth'], z['confidence'], z['static_mask']
    if depth.shape != confidence.shape or depth.shape != predicted_mask.shape or len(depth) != len(frames):
        raise ValueError('Prediction array shape mismatch.')
    ph, pw = depth.shape[1:]
    if (pw, ph) != (518, 294):
        raise ValueError('Only audited 1920x1080 -> 518x294 no-crop preprocessing is supported.')
    records = []
    for ordinal, view in enumerate(train):
        name = view['name']
        idx, frame = selected_by_name[name]
        image = model_images[name]
        camera = rec.cameras[image.camera_id]
        if (camera.width, camera.height) != (1920, 1080):
            raise ValueError('Source camera size does not match audited preprocessing.')
        xy, points = [], []
        for observation in image.points2D:
            if not observation.has_point3D():
                continue
            point = rec.points3D[observation.point3D_id]
            if point.error <= 2.0 and point.track.length() >= 3:
                xy.append(observation.xy)
                points.append(point.xyz)
        xy, points = np.array(xy).reshape(-1, 2), np.array(points).reshape(-1, 3)
        uv = xy * [pw / camera.width, ph / camera.height] - .5
        predicted = sample(depth[idx], uv)
        conf = sample(confidence[idx], uv)
        static = sample(predicted_mask[idx], uv) >= .99999
        w2c = np.array(view['w2c'])
        xyz_camera = points @ w2c[:3, :3].T + w2c[:3, 3]
        # Validate each original track against the *current* undistorted static mask.
        k = np.array(view['K'])
        pix = xyz_camera @ k.T
        pix = pix[:, :2] / np.maximum(pix[:, 2:3], 1e-12) - .5
        current_mask = cv2.imread(view['mask_path'], cv2.IMREAD_GRAYSCALE)
        static &= sample(current_mask, pix) >= 254.999
        conf_valid = confidence[idx][predicted_mask[idx] & np.isfinite(confidence[idx])]
        threshold = float(np.percentile(conf_valid, a.confidence_percentile)) if len(conf_valid) else 1e30
        valid = static & (conf >= threshold) & (xyz_camera[:, 2] > 0)
        fit = fit_scale(predicted[valid], xyz_camera[valid, 2], xy[valid], camera.width, camera.height)
        fit.update(name=name, time_seconds=view['time_seconds'], prediction_index=idx,
                   source_track_candidates=int(len(points)), static_confident_tracks=int(valid.sum()),
                   confidence_threshold=threshold)
        records.append(fit)
        if ordinal % 32 == 0:
            print(json.dumps({'phase': 'fit', 'views_done': ordinal + 1, 'views_total': len(train)}), flush=True)
    initial = [r for r in records if r['accepted']]
    if not initial:
        raise RuntimeError('No views passed source-depth alignment.')
    logs = np.log([r['scale'] for r in initial])
    global_log = float(np.median(logs))
    global_mad = float(np.median(np.abs(logs - global_log)))
    global_cut = max(.40, min(.75, 4 * 1.4826 * global_mad))
    for record in records:
        if record['accepted'] and abs(np.log(record['scale']) - global_log) > global_cut:
            record['accepted'] = False
            record['reasons'].append('per_view_scale_outlier_against_global_median')
    accepted = {r['name']: r for r in records if r['accepted']}
    centers = np.array([np.linalg.inv(np.array(v['w2c']))[:3, 3] for v in train])
    center = np.median(centers, axis=0)
    extent = float(np.percentile(np.linalg.norm(centers - center, axis=1), 90))
    result = {'schema': 'real2sim-inferred-geometry/v1', 'status': 'alignment_complete',
              'inferred_prior': True, 'source_sha256': data['source_sha256'], 'dataset_sha256': dataset_sha,
              'model_artifacts': model_hashes, 'predictions_sha256': sha(predpath),
              'selected_frames_sha256': sha(selection_path), 'input_manifest_sha256': sha(manifest_path),
              'implementation_sha256': sha(__file__), 'source_bindings': source_bindings,
              'units': 'arbitrary', 'metric_geometry_verified': False, 'collision_certified': False,
              'articulation_separated': False, 'model_camera_count': len(train), 'accepted_views': len(accepted),
              'coordinate_frame': 'Unchanged refined model_stable; current dataset OpenCV world-to-camera poses.',
              'alignment': {'method': 'Per-view positive scale from median log refined-Z / predicted-Z; MAD rejection; no shift.',
                            'independent_validation': False, 'max_track_reprojection_error_pixels': 2.0,
                            'minimum_track_length': 3, 'minimum_inlier_tracks': 20,
                            'minimum_inlier_hull_fraction': .02, 'global_median_scale': float(np.exp(global_log)),
                            'per_view_log_scale_mad': global_mad, 'global_log_scale_outlier_cutoff': global_cut,
                            'views': records},
              'preprocessing': {'source_size': [1920, 1080], 'prediction_size': [pw, ph],
                                'source_to_prediction': 'pixel_corner * [518/1920,294/1080] - 0.5',
                                'fusion_mapping': 'Undistorted current rays -> original SIMPLE_RADIAL pixels -> predicted depth; bilinear full static support.',
                                'current_mask': 'Frozen current undistorted static mask plus stored inference-input mask; valid only with full interpolation support.',
                                'confidence': 'Uncalibrated score; per-view percentile threshold, not probability.',
                                'confidence_percentile': a.confidence_percentile},
              'camera_center_r90': extent,
              'prior_license': selected['provenance']['weights_license'],
              'caveats': ['Learned depth is inferred, not captured LiDAR/RGB-D or independently surveyed geometry.',
                          'Alignment uses training SfM tracks from this video; residuals cannot certify metric accuracy.',
                          'Per-view scale can fit tracks while retaining learned shape and focal-length bias elsewhere.',
                          'Glass, reflected scenes, thin geometry and articulation remain unreliable.',
                          'Local TSDF integration may bridge nearby surfaces; no global hidden-room completion is performed.',
                          'Original VGGT checkpoint is CC-BY-NC-4.0; this is a research pilot, not a deployment asset.']}
    write_json(out / 'alignment.json', result)
    print(json.dumps({'phase': 'aligned', 'accepted': len(accepted), 'total': len(train),
                      'scale': float(np.exp(global_log)), 'log_scale_mad': global_mad}), flush=True)
    if a.fit_only:
        return
    if len(accepted) < max(32, len(train) // 2):
        result['status'] = 'rejected_insufficient_aligned_views'
        write_json(out / 'metrics.json', result)
        raise RuntimeError(result['status'])
    fuse_views, depths, colors, ks, coverages = [], [], [], [], []
    for view in train:
        if view['name'] not in accepted:
            continue
        record = accepted[view['name']]
        idx = record['prediction_index']
        camera = rec.cameras[model_images[view['name']].camera_id]
        u, v, k = target_rays(view, camera, a.width, pw, ph)
        def remap(arr):
            return cv2.remap(arr.astype(np.float32), u, v, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        d = remap(depth[idx]) * record['scale']
        c = remap(confidence[idx])
        m = remap(predicted_mask[idx]) >= .99999
        valid = m & (u >= 0) & (v >= 0) & (u <= pw - 1) & (v <= ph - 1)
        mask = cv2.imread(view['mask_path'], cv2.IMREAD_GRAYSCALE)
        valid &= cv2.resize(mask.astype(np.float32), (u.shape[1], u.shape[0]), interpolation=cv2.INTER_AREA) >= 254.999
        valid &= (c >= record['confidence_threshold']) & np.isfinite(d) & (d > .02 * extent) & (d < 6 * extent)
        d[~valid] = 0
        image = cv2.cvtColor(cv2.imread(view['image_path']), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (u.shape[1], u.shape[0]), interpolation=cv2.INTER_AREA)
        fuse_views.append(view)
        depths.append(np.ascontiguousarray(d, np.float32))
        colors.append(np.ascontiguousarray(image))
        ks.append(k)
        coverages.append({'name': view['name'], 'time_seconds': view['time_seconds'],
                          'valid_depth_fraction': float(np.mean(valid)), 'valid_pixels': int(valid.sum())})
    consistency = cross_view_check(fuse_views, depths, ks)
    result['cross_view_depth_consistency'] = consistency
    result['depth_coverage'] = coverages
    # Gross disagreement is rejected before spending memory on TSDF integration.
    if consistency['median_pair_median_relative_error'] is None or consistency['median_pair_median_relative_error'] > .25:
        result['status'] = 'rejected_cross_view_depth_incoherence'
        write_json(out / 'metrics.json', result)
        raise RuntimeError(result['status'])
    voxel = extent / a.voxel_divisor
    trunc = 6 * voxel
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=16, depth_sampling_stride=4)
    for i, (view, d, c, k) in enumerate(zip(fuse_views, depths, colors, ks)):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(c), o3d.geometry.Image(d),
                  depth_scale=1.0, depth_trunc=6 * extent, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(d.shape[1], d.shape[0], k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        volume.integrate(rgbd, intrinsic, np.array(view['w2c'], dtype=np.float64))
        if i % 8 == 0 or i + 1 == len(fuse_views):
            progress = {'phase': 'fuse', 'views_done': i + 1, 'views_total': len(fuse_views), 'elapsed_seconds': time.time() - started}
            write_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(triangles) < 100 or not np.isfinite(vertices).all():
        raise RuntimeError('TSDF produced no valid surface.')
    if sha(dataset_path) != dataset_sha or {x.name: sha(x) for x in modelpath.glob('*.bin')} != model_hashes:
        raise RuntimeError('Dataset/model changed while pilot ran.')
    o3d.io.write_triangle_mesh(str(out / 'inferred_surface.ply'), mesh)
    np.savez_compressed(out / 'surface_arrays.npz', vertices=vertices.astype(np.float32),
                        triangles=triangles.astype(np.int32), colors=np.asarray(mesh.vertex_colors).astype(np.float32),
                        normals=np.asarray(mesh.vertex_normals).astype(np.float32))
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    result.update(status='produced_inferred_pilot', voxel_size=voxel, sdf_trunc=trunc,
                  fusion_width=a.width, vertices=len(vertices), triangles=len(triangles),
                  boundary_edges=int(np.sum(counts == 1)), nonmanifold_edges=int(np.sum(counts > 2)),
                  watertight=bool(mesh.is_watertight()), elapsed_seconds=time.time() - started,
                  artifacts={x.name: sha(x) for x in out.iterdir() if x.suffix in ('.ply', '.npz')})
    write_json(out / 'metrics.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'vertices', 'triangles', 'accepted_views', 'elapsed_seconds', 'artifacts')}), flush=True)


if __name__ == '__main__':
    main()
