#!/usr/bin/env python3
"""Bounded training-only VGGT camera/depth pilot; arbitrary scale, no scene writes.

Use --frames --repo --weights --output and optionally --database. The official
VGGT checkout and pretrained safetensors must already exist; this script performs
no network requests, uploads, package installs, or production database writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def project_support(i, j, world, depth, confidence, masks, colors, extrinsic, K, threshold):
    xyz = world[i, ::8, ::8].reshape(-1, 3)
    valid = (masks[i, ::8, ::8] & (confidence[i, ::8, ::8] >= threshold)).ravel()
    xyz = xyz[valid]
    rgb = colors[i, ::8, ::8].reshape(-1, 3)[valid]
    if not len(xyz):
        return dict(source_points=0, projected_points=0, depth_agreement_fraction=None, median_relative_depth_error=None)
    camera = xyz @ extrinsic[j, :3, :3].T + extrinsic[j, :3, 3]
    uvw = camera @ K[j].T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    h, w = depth.shape[1:3]
    inside = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = np.where(inside)[0]
    pixels = np.floor(uv[indices]).astype(int)
    keep = masks[j, pixels[:, 1], pixels[:, 0]] & (confidence[j, pixels[:, 1], pixels[:, 0]] >= threshold)
    indices, pixels = indices[keep], pixels[keep]
    if not len(indices):
        return dict(source_points=len(xyz), projected_points=0, depth_agreement_fraction=None, median_relative_depth_error=None)
    other_depth = depth[j, pixels[:, 1], pixels[:, 0]]
    own_depth = camera[indices, 2]
    relative = np.abs(own_depth - other_depth) / np.maximum(np.maximum(own_depth, other_depth), 1e-6)
    consistent = relative < .1
    color_error = np.abs(rgb[indices].astype(float) - colors[j, pixels[:, 1], pixels[:, 0]]) / 255
    return dict(source_points=len(xyz), projected_points=len(indices),
                depth_agreement_fraction=float(consistent.mean()), median_relative_depth_error=float(np.median(relative)),
                median_rgb_l1_on_depth_agreement=float(np.median(color_error[consistent].mean(axis=1))) if consistent.any() else None)


def epipolar_audit(database, selected, K, extrinsic, hw):
    if not database:
        return {'status': 'not_requested'}
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    ids = {name: iid for iid, name in db.execute('SELECT image_id,name FROM images')}
    keypoints = {}
    for view in selected:
        iid = ids.get(view['name'])
        if iid is None:
            continue
        rows, cols, blob = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (iid,)).fetchone()
        xy = np.frombuffer(blob, np.float32).reshape(rows, cols)[:, :2].copy()
        xy *= [hw[1] / view['width'], hw[0] / view['height']]
        xy -= .5
        keypoints[iid] = xy
    pairs = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = ids.get(selected[i]['name']), ids.get(selected[j]['name'])
            if a not in keypoints or b not in keypoints:
                continue
            low, high = sorted((a, b))
            row = db.execute('SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?', (low * 2147483647 + high,)).fetchone()
            if not row or row[0] < 15:
                continue
            m = np.frombuffer(row[2], np.uint32).reshape(row[0], row[1])
            if a > b:
                m = m[:, ::-1]
            xy1, xy2 = keypoints[a][m[:, 0]], keypoints[b][m[:, 1]]
            R = extrinsic[j, :3, :3] @ extrinsic[i, :3, :3].T
            t = extrinsic[j, :3, 3] - R @ extrinsic[i, :3, 3]
            skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            F = np.linalg.inv(K[j]).T @ skew @ R @ np.linalg.inv(K[i])
            p1, p2 = np.c_[xy1, np.ones(len(m))], np.c_[xy2, np.ones(len(m))]
            l2, l1 = p1 @ F.T, p2 @ F
            error = np.abs((p2 * l2).sum(axis=1)) / np.maximum(np.sqrt((l1[:, :2]**2).sum(axis=1) + (l2[:, :2]**2).sum(axis=1)), 1e-12)
            pairs.append(dict(names=[selected[i]['name'], selected[j]['name']], verified_matches=len(m),
                              median_sampson_distance_px=float(np.median(error)), p90_sampson_distance_px=float(np.quantile(error, .9)),
                              fraction_below_2px=float((error < 2).mean())))
    db.close()
    return dict(status='available', pairs=pairs,
                median_pair_sampson_distance_px=float(np.median([p['median_sampson_distance_px'] for p in pairs])) if pairs else None,
                note='Independent cached SIFT verified correspondences; this cache includes known repeated-pattern errors. Coordinates scaled from COLMAP pixel corners to VGGT integer pixel centers. A low epipolar residual alone does not verify depth or scale.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--database', type=Path)
    p.add_argument('--masks-dir', type=Path, required=True,
                   help='Reviewed static-valid masks, named IMAGE.jpg.png; every selected image requires a mask')
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--all-train', action='store_true', help='Authorized bounded full training split (maximum 259 views)')
    p.add_argument('--seed', type=int, default=2296)
    a = p.parse_args()
    if not a.all_train and not 2 <= a.count <= 64:
        p.error('The pilot accepts 2 to 64 frames')
    out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError('Use a fresh output directory')
    started = time.monotonic()
    frame_bytes = a.frames.read_bytes();manifest = json.loads(frame_bytes)
    train = sorted([r for r in manifest['frames'] if r['split'] == 'train'], key=lambda r: r['time_seconds'])
    if a.all_train:
        if not 2 <= len(train) <= 259:
            raise ValueError('This full-split experiment is bounded at 259 training views')
        a.count = len(train)
    targets = np.linspace(train[0]['time_seconds'], train[-1]['time_seconds'], a.count)
    available = set(range(len(train)))
    indices = []
    for target in targets:
        index = min(available, key=lambda i: (abs(train[i]['time_seconds'] - target), i))
        indices.append(index);available.remove(index)
    indices.sort()
    selected = [dict(train[i]) for i in (range(len(train)) if a.all_train else indices)]
    assert len(selected) == a.count and all(v['split'] == 'train' for v in selected)
    if any((v['width'], v['height']) != (1920, 1080) for v in selected):
        raise ValueError('This bounded pilot explicitly implements the provided 1920x1080 image preprocessing')
    provenance = dict(schema='real2sim-vggt-pilot-inputs/v1', source_sha256=manifest['source_sha256'],
                      frames_sha256=hashlib.sha256(frame_bytes).hexdigest(), implementation_sha256=sha(__file__),
                      repo_commit=subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip(),
                      code_license_path=str(a.repo / 'LICENSE.txt'), code_license_sha256=sha(a.repo / 'LICENSE.txt'),
                      weights_path=str(a.weights.resolve()), weights_sha256=sha(a.weights),
                      weights_repository='facebook/VGGT-1B', weights_revision='860abec7937da0a4c03c41d3c269c366e82abdf9',
                      weights_license='CC-BY-NC-4.0; public original checkpoint, research pilot only',
                      physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=str(torch.__version__),
                      selection='all training frames; no validation or buffer' if a.all_train else f'nearest training frame to {a.count} equally spaced source timestamps', units='arbitrary',
                      coordinate_frame='VGGT inferred first-view frame; not aligned to production SfM', source_uploads=False)
    for view in selected:
        view['actual_image_sha256'] = sha(view['image_path'])
        if view.get('sha256') != view['actual_image_sha256']:
            raise ValueError('Source image hash changed')
    write(out / 'selected_frames.json', dict(provenance=provenance, frames=selected))
    sys.path.insert(0, str(a.repo.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from safetensors.torch import load_file
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    write(out / 'progress.json', dict(status='loading_model', updated_at=datetime.now(timezone.utc).isoformat()))
    model = VGGT(enable_point=False, enable_track=False)
    weights = load_file(str(a.weights), device='cpu')
    excluded = {k: v for k, v in weights.items() if not k.startswith(('point_head.', 'track_head.'))}
    model.load_state_dict(excluded, strict=True);del weights, excluded
    model = model.eval().to('cuda')
    images = load_and_preprocess_images([v['image_path'] for v in selected], mode='crop')
    colors = (images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    h, w = images.shape[-2:]
    masks = np.ones((len(selected), h, w), bool)
    for i, view in enumerate(selected):
        maskpath = a.masks_dir / (view['name'] + '.png')
        view['mask_path'] = str(maskpath.resolve())
        view['mask_sha256'] = sha(maskpath)
        with Image.open(maskpath) as im:
            masks[i] = np.asarray(im.convert('L').resize((w, h), Image.Resampling.NEAREST)) == 255
        images[i, :, ~torch.from_numpy(masks[i])] = 0
    preprocessing = dict(source_width=1920, source_height=1080, width=w, height=h,
                         scale_xy=[w / 1920, h / 1080], crop_xy=[0, 0], pad_ltrb=[0, 0, 0, 0],
                         mode='official crop; no content crop for landscape input',
                         source_pixels='COLMAP pixel corners; top-left pixel center at (0.5, 0.5)',
                         model_pixels='integer pixel centers; source_to_model: u_model = u_source * scale - 0.5',
                         masks_dir=str(a.masks_dir.resolve()), mask='reviewed static-valid masks nearest-resampled; invalid RGB set to zero')
    write(out / 'selected_frames.json', dict(provenance=provenance, frames=selected, preprocessing=preprocessing))
    images = images.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    write(out / 'progress.json', dict(status='inferring', views=len(selected), width=w, height=h,
                                    updated_at=datetime.now(timezone.utc).isoformat()))
    inference_started = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started
    gpu_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    ext, K = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
    depth = predictions['depth'][0, ..., 0].float().cpu().numpy()
    confidence = predictions['depth_conf'][0].float().cpu().numpy()
    pose_enc = predictions['pose_enc'][0].float().cpu().numpy()
    del predictions, images, model;torch.cuda.empty_cache()
    assert all(np.isfinite(x).all() for x in (ext, K, depth, confidence, pose_enc))
    rotation_error = float(np.max(np.abs(ext[:, :3, :3] @ ext[:, :3, :3].transpose(0, 2, 1) - np.eye(3))))
    assert rotation_error < 1e-4 and np.allclose(np.linalg.det(ext[:, :3, :3]), 1, atol=1e-4)
    # Publish raw camera hypotheses promptly; dense products and audits follow.
    valid = masks & (depth > 0)
    centers = -np.einsum('nij,nj->ni', ext[:, :3, :3].transpose(0, 2, 1), ext[:, :3, 3])
    extent = float(np.quantile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), .9))
    cameras = []
    for i, view in enumerate(selected):
        w2c = np.eye(4);w2c[:3] = ext[i]
        original_K = K[i].copy();original_K[:2, 2] += .5
        original_K[0] *= 1920 / w;original_K[1] *= 1080 / h
        coverage = float(masks[i].mean())
        cameras.append(dict(name=view['name'], time_seconds=view['time_seconds'], source_time_seconds=view['time_seconds'],
                            split=view['split'], source_width=1920, source_height=1080,
                            w2c=w2c.tolist(), K_preprocessed=K[i].tolist(), K=original_K.tolist(),
                            center=centers[i].tolist(), median_depth=float(np.median(depth[i][valid[i]])),
                            median_confidence=float(np.median(confidence[i][valid[i]])), static_valid_fraction=coverage,
                            uncertainty_flags=['less_than_half_image_static_valid'] if coverage < .5 else [],
                            positive_static_depth_fraction=float(valid[i].mean())))
    write(out / 'cameras.json', dict(schema='real2sim-vggt-cameras/v1', source_sha256=manifest['source_sha256'],
                                   units='arbitrary', convention='OpenCV w2c, +Z forward',
                                   preprocessing=preprocessing, views=cameras))
    write(out / 'progress.json', dict(status='cameras_saved_dense_and_audits_pending', views=len(selected),
                                    inference_seconds=inference_seconds, gpu_peak_allocated_gb=gpu_peak_gb,
                                    updated_at=datetime.now(timezone.utc).isoformat()))
    np.savez_compressed(out / 'predictions.npz', extrinsics=ext, intrinsics=K, depth=depth,
                        confidence=confidence, pose_encoding=pose_enc, source_rgb=colors, static_mask=masks)
    world = unproject_depth_map_to_point_map(depth[..., None], ext, K)
    valid = masks & np.isfinite(world).all(axis=-1) & (depth > 0)
    threshold = float(np.quantile(confidence[valid], .5))
    cloud_valid = valid & (confidence >= threshold)
    xyz, rgb = world[cloud_valid], colors[cloud_valid]
    rng = np.random.default_rng(a.seed)
    if len(xyz) > 600000:
        ids = np.sort(rng.choice(len(xyz), 600000, replace=False));xyz, rgb = xyz[ids], rgb[ids]
    rows = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, name in enumerate(('x', 'y', 'z')):rows[name] = xyz[:, k]
    for k, name in enumerate(('red', 'green', 'blue')):rows[name] = rgb[:, k]
    with open(out / 'point_cloud.ply', 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\ncomment arbitrary_scale_VGGT_pilot_not_metric_map\n'
                 f'element vertex {len(rows)}\nproperty float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode());f.write(rows.tobytes())
    continuity = []
    for i in range(len(selected) - 1):
        relative_R = ext[i + 1, :3, :3] @ ext[i, :3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1))))
        gap = selected[i + 1]['time_seconds'] - selected[i]['time_seconds']
        distance = float(np.linalg.norm(centers[i + 1] - centers[i]))
        continuity.append(dict(names=[selected[i]['name'], selected[i + 1]['name']], gap_seconds=gap,
                               translation=distance, translation_over_extent=distance / max(extent, 1e-8), rotation_degrees=angle))
    overlaps = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            overlaps.append(dict(names=[selected[i]['name'], selected[j]['name']],
                                 forward=project_support(i, j, world, depth, confidence, masks, colors, ext, K, threshold),
                                 backward=project_support(j, i, world, depth, confidence, masks, colors, ext, K, threshold)))
    epi = epipolar_audit(a.database, selected, K, ext, (h, w))
    write(out / 'epipolar_audit.json', epi)
    write(out / 'depth_overlap.json', overlaps)
    write(out / 'pose_continuity.json', continuity)
    dlo, dhi = np.quantile(depth[valid], [.02, .98])
    sheet = Image.new('RGB', (2072, ((len(selected) + 3) // 4) * 175), (20, 20, 20));draw = ImageDraw.Draw(sheet)
    for i, view in enumerate(selected):
        x, y = (i % 4) * 518, (i // 4) * 175
        source = Image.fromarray(colors[i]).resize((259, 147))
        scaled = np.clip((depth[i] - dlo) / max(dhi - dlo, 1e-8), 0, 1)
        depth_rgb = cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype('uint8'), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        depth_rgb[~masks[i]] = 0
        sheet.paste(source, (x, y + 22));sheet.paste(Image.fromarray(depth_rgb).resize((259, 147)), (x + 259, y + 22))
        draw.text((x + 4, y + 4), f'{view["time_seconds"]:.3f}s | {view["name"]} | source / relative depth', fill='white')
    sheet.save(out / 'source_depth_contact.jpg', quality=94)
    artifacts = {p.name: dict(path=str(p), sha256=sha(p)) for p in out.iterdir()
                 if p.is_file() and p.name not in ('progress.json', 'result.json')}
    result = dict(schema='real2sim-vggt-pilot/v1', status='complete', produced_at=datetime.now(timezone.utc).isoformat(),
                  provenance=provenance, views=len(selected), training_only=True, metric_geometry_verified=False,
                  inference_seconds=inference_seconds, elapsed_seconds=time.monotonic() - started, gpu_peak_allocated_gb=gpu_peak_gb,
                  rotation_orthogonality_max_error=rotation_error, camera_extent_arbitrary=extent,
                  point_count=len(xyz), confidence_threshold=threshold, confidence_is_probability=False,
                  median_pair_sampson_distance_px=epi.get('median_pair_sampson_distance_px'),
                  max_adjacent_translation_over_extent=max(r['translation_over_extent'] for r in continuity),
                  max_adjacent_rotation_degrees=max(r['rotation_degrees'] for r in continuity), artifacts=artifacts,
                  limitations=['Learned cameras/depth are hypotheses in a separate arbitrary-scale frame.',
                               'Depth overlap is internal consistency, not ground-truth geometry.',
                               'Reviewed masks exclude configured dynamic regions; repeated labels and reflections may remain.',
                               'The original public checkpoint is noncommercial; this pilot is not a deployment artifact.'])
    write(out / 'result.json', result);write(out / 'progress.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
