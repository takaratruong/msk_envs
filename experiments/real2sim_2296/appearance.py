#!/usr/bin/env python3
"""Local, fixed-camera 3D Gaussian appearance reconstruction with gsplat 1.5.3.

Input JSON: {schema, source_sha256, units, points_path, train: [...], val: [...]}.
Each view has name, absolute image_path, OpenCV w2c (4x4), K (3x3), width,
height, and optionally source_timestamp_s / timestamp_s / time_seconds / time_s.
Optional absolute mask_path PNGs use 255 for valid static pixels and 0 for
invalid/dynamic pixels. Masks resize with nearest-neighbor sampling. Masked
training excludes invalid pixels and SSIM windows containing invalid pixels;
evaluation reports full-image and masked-static scores with coverage. Images must
already be undistorted and oriented consistently with their camera matrices.
The NPZ contains xyz (N,3) and rgb (N,3), with rgb encoded as uint8.

Example:
  python appearance.py --dataset /abs/dataset.json --output /abs/appearance \
      --steps 30000 --max-width 1280 --seed 2296
  python appearance.py --dataset /abs/dataset.json --output /abs/appearance \
      --steps 30000 --max-width 1280 --seed 2296 --resume
  python appearance.py --dataset /abs/dataset.json --output /abs/appearance \
      --resume --eval-only

The output contains atomic model/optimizer/RNG checkpoints, progress.json,
train_stats.jsonl, camera_manifest.json, input_manifest.json, evaluation JSON,
source/render JPG comparisons, an all-view comparison video, standard 3DGS
appearance.ply, and a colored appearance_points.ply. SIGTERM/SIGINT requests a
safe stop after the current completed optimizer/strategy step. An unexpected
failure preserves the last fully written model checkpoint.

Neither cameras nor world coordinates are normalized or optimized. Scene
extent affects initialization, learning rates and clipping only. Held-out
images never supply a training loss, initialization colors or camera updates;
upstream SfM may nevertheless have used them, which must be disclosed in the
dataset. Reproducible seeds and full stochastic state are saved, but CUDA
atomic reductions do not promise bitwise repeatability. Pixel metrics are
appearance evidence, not metric geometry, collision, or sim-to-real evidence.

API references are pinned to the official release, not the moving main branch:
https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py
https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/default.py
https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F


C0 = 0.28209479177387814
GIB = 1024**3
STOP_REQUESTED = False


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    return value


def capture_rng(rng):
    legacy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]],
        "generator": json.dumps(rng.bit_generator.state),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, rng):
    random.setstate(state["python"])
    legacy = state["numpy"]
    np.random.set_state((legacy[0], np.asarray(legacy[1], dtype=np.uint32), *legacy[2:]))
    rng.bit_generator.state = json.loads(state["generator"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def source_time(view):
    for key in ("source_timestamp_s", "timestamp_s", "time_seconds", "time_s", "pts_time", "timestamp"):
        if key in view:
            try:
                value = float(view[key])
                if math.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
    return None


def load_dataset(path, max_width):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    for key in ("schema", "source_sha256", "units", "points_path", "train", "val"):
        if key not in data:
            raise ValueError(f"Missing dataset field: {key}")
    if not data["train"] or not data["val"]:
        raise ValueError("At least one training view and one reserved validation view are required")
    if not isinstance(data["units"], str) or not data["units"]:
        raise ValueError("The reconstruction coordinate units must be explicitly recorded")
    source_hash = data["source_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(
        c not in "0123456789abcdefABCDEF" for c in source_hash
    ):
        raise ValueError("source_sha256 must be a SHA-256 hex digest")
    points_path = Path(data["points_path"])
    if not points_path.is_absolute() or not points_path.is_file():
        raise ValueError("points_path must be an existing absolute NPZ path")
    manifest = {"dataset_path": str(path), "dataset_sha256": sha256(path),
                "source_sha256": source_hash, "points_path": str(points_path),
                "points_sha256": sha256(points_path), "images": []}
    names, paths, image_hash_splits = set(), set(), {}
    views = {"train": [], "val": []}
    for split in views:
        for original in data[split]:
            view = dict(original)
            name = view["name"]
            image_path = Path(view["image_path"])
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"Duplicate or invalid view name: {name!r}")
            if not image_path.is_absolute() or not image_path.is_file():
                raise ValueError(f"Missing absolute image path for {name}")
            image_path = image_path.resolve()
            if str(image_path) in paths:
                raise ValueError(f"An image path appears in multiple views: {image_path}")
            names.add(name)
            paths.add(str(image_path))
            w, h = int(view["width"]), int(view["height"])
            with Image.open(image_path) as im:
                if im.size != (w, h):
                    raise ValueError(f"Image size disagrees with calibration for {name}: {im.size} != {(w, h)}")
                if im.getexif().get(274, 1) != 1:
                    raise ValueError(f"Unapplied EXIF image orientation for {name}; orient upstream with cameras")
            if min(w, h) < 16:
                raise ValueError(f"Image too small for SSIM: {name}")
            K = np.asarray(view["K"], dtype=np.float64)
            w2c = np.asarray(view["w2c"], dtype=np.float64)
            if K.shape != (3, 3) or w2c.shape != (4, 4) or not (
                np.isfinite(K).all() and np.isfinite(w2c).all()
            ):
                raise ValueError(f"Invalid camera matrices for {name}")
            R = w2c[:3, :3]
            if not (np.allclose(R @ R.T, np.eye(3), atol=1e-4)
                    and abs(np.linalg.det(R) - 1) < 1e-4
                    and np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-6)):
                raise ValueError(f"w2c is not a rigid OpenCV transform for {name}")
            if not (K[0, 0] > 0 and K[1, 1] > 0 and abs(K[0, 1]) < 1e-6
                    and abs(K[1, 0]) < 1e-6 and np.allclose(K[2], [0, 0, 1])):
                raise ValueError(f"Expected a zero-skew pinhole K for {name}")
            for distortion_key in ("distortion", "distortion_coeffs", "distortion_params"):
                if distortion_key in view and np.any(np.asarray(view[distortion_key], dtype=float) != 0):
                    raise ValueError(f"Distorted pixels must be undistorted upstream: {name}")
            scale = min(1.0, max_width / w)
            tw, th = max(16, round(w * scale)), max(16, round(h * scale))
            scaled_K = K.copy()
            scaled_K[0] *= tw / w
            scaled_K[1] *= th / h
            ihash = sha256(image_path)
            if ihash in image_hash_splits and image_hash_splits[ihash] != split:
                raise ValueError(f"Identical image bytes occur in training and validation: {name}")
            image_hash_splits[ihash] = split
            mask_path = view.get("mask_path")
            mask_hash = None
            if mask_path is not None:
                mask_path = Path(mask_path)
                if not mask_path.is_absolute() or not mask_path.is_file():
                    raise ValueError(f"Missing absolute mask path for {name}")
                mask_path = mask_path.resolve()
                with Image.open(mask_path) as mask_image:
                    mask_pixels = np.asarray(mask_image.convert("L"))
                    if mask_image.size != (w, h) or not np.isin(mask_pixels, [0, 255]).all():
                        raise ValueError(f"Mask must match original image size and contain only 0/255: {name}")
                    if not np.any(mask_pixels == 255):
                        raise ValueError(f"Mask has no valid static pixels: {name}")
                mask_hash = sha256(mask_path)
                mask_path = str(mask_path)
            manifest["images"].append({"name": name, "split": split,
                                       "image_path": str(image_path), "sha256": ihash,
                                       "mask_path": mask_path, "mask_sha256": mask_hash})
            view.update(image_path=str(image_path), split=split, original_K=K.tolist(),
                        original_width=w, original_height=h, K=scaled_K.tolist(),
                        width=tw, height=th, source_timestamp_s=source_time(view), mask_path=mask_path)
            views[split].append(view)
    manifest["inputs_sha256"] = digest(manifest)
    return data, views, manifest


class ImageCache:
    def __init__(self, max_gb):
        self.limit = int(max_gb * GIB)
        self.bytes = 0
        self.images = OrderedDict()

    def get(self, view):
        key = (view["image_path"], view["width"], view["height"])
        if key in self.images:
            self.images.move_to_end(key)
            return self.images[key]
        with Image.open(view["image_path"]) as im:
            im = im.convert("RGB")
            if im.size != (view["width"], view["height"]):
                im = im.resize((view["width"], view["height"]), Image.Resampling.LANCZOS)
            pixels = np.asarray(im).copy()
        self.store(key, pixels)
        return pixels

    def store(self, key, pixels):
        if pixels.nbytes <= self.limit:
            while self.images and self.bytes + pixels.nbytes > self.limit:
                _, old = self.images.popitem(last=False)
                self.bytes -= old.nbytes
            self.images[key] = pixels
            self.bytes += pixels.nbytes

    def mask(self, view):
        key = ("mask", view.get("mask_path"), view["width"], view["height"])
        if key in self.images:
            self.images.move_to_end(key)
            return self.images[key]
        if view.get("mask_path"):
            with Image.open(view["mask_path"]) as im:
                mask = np.asarray(im.convert("L").resize(
                    (view["width"], view["height"]), Image.Resampling.NEAREST
                )) == 255
        else:
            mask = np.ones((view["height"], view["width"]), dtype=bool)
        if not mask.any():
            raise ValueError(f"Mask has no valid pixels after resizing: {view['name']}")
        self.store(key, mask)
        return mask


def ssim(x, y, mask=None):
    """RGB SSIM, data_range=1, Gaussian 11x11/sigma=1.5, valid crop."""
    coords = torch.arange(11, dtype=x.dtype, device=x.device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    horizontal = g.view(1, 1, 1, 11).expand(3, 1, 1, 11)
    vertical = g.view(1, 1, 11, 1).expand(3, 1, 11, 1)
    moments = torch.cat((x, y, x * x, y * y, x * y), dim=0)
    # TF32 convolution corrupts near-constant moment differences at full image
    # size. The actual rejected run returned SSIM -2.55 versus 0.567 in FP32.
    # Preserve full precision here even when called outside the training main.
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=False):
        moments = F.conv2d(F.conv2d(moments, horizontal, groups=3), vertical, groups=3)
    mx, my, xx, yy, xy = moments.chunk(5, dim=0)
    vx, vy, cov = (xx - mx.square()).clamp_min(0), (yy - my.square()).clamp_min(0), xy - mx * my
    result = ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    )
    if mask is None:
        return result.mean()
    valid_windows = F.max_pool2d((~mask).float(), 11, stride=1) == 0
    count = valid_windows.sum()
    if count == 0:
        return None
    return (result * valid_windows).sum() / (count * result.shape[1])


def bounded_strategy_class(DefaultStrategy):
    class BoundedDefaultStrategy(DefaultStrategy):
        """Keep upstream split/prune behavior; bound growth before allocation."""
        def __init__(self, max_gaussians, reserve_gb, **kwargs):
            super().__init__(**kwargs)
            self.max_gaussians = max_gaussians
            self.reserve_bytes = int(reserve_gb * GIB)
            if self.refine_scale2d_stop_iter != 0:
                raise ValueError("The growth bound assumes the default disabled 2D-scale growth")

        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            super().step_post_backward(params, optimizers, state, step, info, packed=packed)
            # The pinned official v1.5.3 release uses `== 0 & step > 0`, a
            # chained comparison that never resets. Restore its intended
            # explicit-and schedule, including the upstream refinement stop.
            if step < self.refine_stop_iter and step > 0 and step % self.reset_every == 0:
                from gsplat.strategy.ops import reset_opa
                reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2)

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            n = len(params["means"])
            free, _ = torch.cuda.mem_get_info(params["means"].device)
            # Allow temporary copies of existing parameters/moments and extra
            # storage for newborn splats; this is a guard, not an OOM guarantee.
            memory_budget = max(0, (free - self.reserve_bytes - n * 2048) // 4096)
            budget = min(max(0, self.max_gaussians - n), int(memory_budget))
            scores = state["grad2d"] / state["count"].clamp_min(1)
            candidates = torch.where(scores > self.grow_grad2d)[0]
            if len(candidates) > budget:
                state["growth_limited_events"] = state.get("growth_limited_events", 0) + 1
                if budget == 0:
                    return 0, 0
                selected = candidates[torch.topk(scores[candidates], k=budget, sorted=False).indices]
                masked = torch.zeros_like(state["grad2d"])
                masked[selected] = state["grad2d"][selected]
                state["grad2d"] = masked
            result = super()._grow_gs(params, optimizers, state, step)
            if len(params["means"]) > self.max_gaussians:
                raise RuntimeError("Densification exceeded the primitive cap")
            return result

    return BoundedDefaultStrategy


def initialize(data, views, args, rng, device):
    from scipy.spatial import cKDTree

    with np.load(data["points_path"], allow_pickle=False) as points:
        xyz = np.asarray(points["xyz"], dtype=np.float32)
        rgb = np.asarray(points["rgb"])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or rgb.dtype != np.uint8:
        raise ValueError("NPZ must contain xyz (N,3) and uint8 rgb (N,3)")
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError("Need at least four finite initial 3D points")
    original_count = len(xyz)
    if len(xyz) > args.init_max_points:
        chosen = np.sort(rng.choice(len(xyz), args.init_max_points, replace=False))
        xyz, rgb = xyz[chosen], rgb[chosen]
    centers = []
    for view in views["train"]:
        w2c = np.asarray(view["w2c"])
        centers.append(-w2c[:3, :3].T @ w2c[:3, 3])
    centers = np.asarray(centers)
    extent = float(np.quantile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), .9))
    if extent < 1e-8:
        extent = float(np.quantile(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), .9))
    if not math.isfinite(extent) or extent <= 1e-8:
        raise ValueError("Degenerate reconstruction extent")
    distances = cKDTree(xyz).query(xyz, k=4, workers=1)[0][:, 1:]
    sizes = np.sqrt(np.mean(distances**2, axis=1)).clip(extent * 1e-6, extent * .1)
    n = len(xyz)
    sh = torch.zeros((n, 16, 3), dtype=torch.float32, device=device)
    sh[:, 0] = (torch.from_numpy(rgb.copy()).to(device).float() / 255 - .5) / C0
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz.copy()).to(device)),
        "scales": torch.nn.Parameter(torch.from_numpy(np.log(sizes).astype(np.float32)).to(device)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=device), dim=-1)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(.1 / .9), device=device)),
        "sh0": torch.nn.Parameter(sh[:, :1].contiguous()),
        "shN": torch.nn.Parameter(sh[:, 1:].contiguous()),
    })
    return splats, extent, {"input_points": original_count, "initial_gaussians": n,
                            "scene_extent": extent, "world_transform": np.eye(4).tolist()}


def optimizers_for(splats, extent):
    rates = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3,
             "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {name: torch.optim.Adam([{"params": [splats[name]], "lr": lr, "name": name}],
                                   eps=1e-15) for name, lr in rates.items()}


def render(splats, view, extent, degree, rasterization):
    device = splats["means"].device
    return rasterization(
        means=splats["means"], quats=splats["quats"], scales=splats["scales"].exp(),
        opacities=splats["opacities"].sigmoid(), colors=torch.cat((splats["sh0"], splats["shN"]), 1),
        viewmats=torch.tensor(view["w2c"], dtype=torch.float32, device=device)[None],
        Ks=torch.tensor(view["K"], dtype=torch.float32, device=device)[None],
        width=view["width"], height=view["height"], sh_degree=degree,
        near_plane=max(extent * 1e-4, 1e-8), far_plane=extent * 1e4,
        packed=True, sparse_grad=False, absgrad=False, render_mode="RGB",
        rasterize_mode="classic", camera_model="pinhole",
    )


def save_model(output, step, splats, optimizers, scheduler, strategy_state,
               rng, order, cursor, config, inputs_sha256, initialization, implementation_sha256,
               capacity_parent=None):
    checkpoint = {
        "schema": "real2sim-appearance-checkpoint/v1", "completed_steps": step,
        "capacity_parent": capacity_parent,
        "splats": cpu_tree(splats.state_dict()),
        "optimizers": {k: cpu_tree(v.state_dict()) for k, v in optimizers.items()},
        "scheduler": scheduler.state_dict(), "strategy_state": cpu_tree(strategy_state),
        "rng": capture_rng(rng), "order": list(map(int, order)), "cursor": int(cursor),
        "model_config": config, "inputs_sha256": inputs_sha256,
        "initialization": initialization, "implementation_sha256": implementation_sha256,
        "gsplat_version": importlib.metadata.version("gsplat"),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda, "produced_at": utcnow(),
    }
    path = output / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    receipt = {"path": str(path), "sha256": sha256(path), "completed_steps": step,
               "inputs_sha256": inputs_sha256, "implementation_sha256": implementation_sha256,
               "produced_at": utcnow()}
    atomic_json(path.with_suffix(".json"), receipt)
    atomic_json(output / "latest_checkpoint.json", receipt)
    return receipt


def export_ply(output, splats, units):
    """Write standard binary 3DGS properties, channel-major SH, wxyz rotation."""
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    n = len(splats["means"])
    outfiles = []
    for colored in (False, True):
        path = output / ("appearance_points.ply" if colored else "appearance.ply")
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        dtype = [(p, "<f4") for p in (["x", "y", "z"] if colored else names)]
        if colored:
            dtype += [(p, "u1") for p in ["red", "green", "blue"]]
        header = ["ply", "format binary_little_endian 1.0", "comment coordinate_frame unchanged_from_dataset",
                  "comment units " + units.replace("\n", " ").replace("\r", " "),
                  "comment appearance_model_not_collision_geometry", f"element vertex {n}"]
        header += [f"property {'uchar' if t == 'u1' else 'float'} {p}" for p, t in dtype]
        header.append("end_header")
        with open(tmp, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii", errors="replace"))
            for start in range(0, n, 65536):
                chunk = {k: v[start:start + 65536].detach().cpu().numpy() for k, v in splats.items()}
                if any(not np.isfinite(v).all() for v in chunk.values()):
                    raise ValueError("Nonfinite Gaussian parameters; refusing to export")
                xyz = chunk["means"]
                rows = np.zeros(len(xyz), dtype=dtype)
                for i, name in enumerate(["x", "y", "z"]):
                    rows[name] = xyz[:, i]
                if colored:
                    colors = np.rint(np.clip(chunk["sh0"][:, 0] * C0 + .5, 0, 1) * 255).astype(np.uint8)
                    for i, name in enumerate(["red", "green", "blue"]):
                        rows[name] = colors[:, i]
                else:
                    for i in range(3):
                        rows[f"f_dc_{i}"] = chunk["sh0"][:, 0, i]
                        rows[f"scale_{i}"] = chunk["scales"][:, i]
                    shn = chunk["shN"].transpose(0, 2, 1).reshape(len(xyz), 45)
                    for i in range(45):
                        rows[f"f_rest_{i}"] = shn[:, i]
                    rows["opacity"] = chunk["opacities"]
                    q = chunk["quats"]
                    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                    for i in range(4):
                        rows[f"rot_{i}"] = q[:, i]
                f.write(rows.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        outfiles.append({"path": str(path), "sha256": sha256(path), "gaussians": n})
    return outfiles


def summarize_evaluation(rows):
    """Keep overall fit and each supplied validation group independently visible."""
    def aggregate(subset):
        result = {"views": len(subset)}
        for key in ("psnr_db", "ssim", "lpips", "mean_alpha"):
            values = [r[key] for r in subset if r[key] is not None]
            result[key] = float(np.mean(values)) if values else None
        for region in ("full", "static"):
            result[region] = {}
            for key in ("psnr_db", "ssim", "lpips"):
                values = [r[region][key] for r in subset if r[region][key] is not None]
                result[region][key] = float(np.mean(values)) if values else None
                result[region][key + "_view_count"] = len(values)
        for key in ("valid_pixel_fraction", "valid_ssim_window_fraction"):
            result[key] = float(np.mean([r[key] for r in subset]))
        return result

    summary = {}
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            summary[split] = aggregate(subset)
            if split == "val":
                groups = sorted({r.get("validation_group") or "unspecified" for r in subset})
                summary[split]["by_group"] = {
                    group: aggregate([r for r in subset if (r.get("validation_group") or "unspecified") == group])
                    for group in groups
                }
    return summary


class Evaluator:
    def __init__(self, lpips_mode, device):
        self.lpips_mode = lpips_mode
        self.lpips_status = "not_initialized" if lpips_mode == "auto" else "disabled"
        self.lpips = None
        self.device = device

    def init_lpips(self):
        if self.lpips_status != "not_initialized":
            return
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            self.lpips.requires_grad_(False)
            self.lpips_status = "available: alex, official pretrained weights, full evaluation resolution"
        except Exception as exc:
            self.lpips_status = f"unavailable: {type(exc).__name__}: {exc}"
            self.lpips = None

    @torch.no_grad()
    def evaluate(self, output, step, splats, views, cache, extent, degree, rasterization,
                 manifest, rng, all_views=False, video_fps=6):
        saved_rng = capture_rng(rng)
        video = None
        video_log = None
        directory = output / "eval" / f"step_{step:06d}{'_all' if all_views else ''}"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        video_status = {"status": "not_requested"}
        try:
            self.init_lpips()
            selected = (views["train"] + views["val"]) if all_views else list(views["val"])
            if all(v["source_timestamp_s"] is not None for v in selected):
                selected.sort(key=lambda v: v["source_timestamp_s"])
            video_tmp = directory / "comparison.tmp.mp4"
            if all_views:
                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg:
                    video_log = open(directory / "video_ffmpeg.log", "wb")
                    video = subprocess.Popen([
                        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(video_fps), "-i", "-",
                        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_tmp),
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=video_log)
                    video_status = {"status": "writing"}
                else:
                    video_status = {"status": "unavailable", "reason": "ffmpeg is not on PATH"}
            for index, view in enumerate(selected):
                pixels = cache.get(view)
                reference = torch.from_numpy(pixels).to(self.device).float()[None] / 255
                mask = torch.from_numpy(cache.mask(view)).to(self.device)[None, None]
                prediction, alpha, _ = render(splats, view, extent, degree, rasterization)
                prediction = prediction.clamp(0, 1)
                ref_nchw, pred_nchw = reference.permute(0, 3, 1, 2), prediction.permute(0, 3, 1, 2)
                mse = F.mse_loss(prediction, reference).item()
                row = {"name": view["name"], "split": view["split"],
                       "validation_group": view.get("validation_group"),
                       "source_timestamp_s": view["source_timestamp_s"],
                       "width": view["width"], "height": view["height"],
                       "psnr_db": -10 * math.log10(max(mse, 1e-12)),
                       "ssim": float(ssim(pred_nchw, ref_nchw)), "mse": mse,
                       "mean_alpha": float(alpha.mean()), "lpips": None}
                if self.lpips is not None:
                    row["lpips"] = float(self.lpips(2 * pred_nchw - 1, 2 * ref_nchw - 1).mean())
                static_mse = float(((pred_nchw - ref_nchw).square() * mask).sum() / (3 * mask.sum()))
                static_ssim = ssim(pred_nchw, ref_nchw, mask)
                row["full"] = {key: row[key] for key in ("psnr_db", "ssim", "mse", "lpips")}
                row["static"] = {"psnr_db": -10 * math.log10(max(static_mse, 1e-12)),
                                 "ssim": float(static_ssim) if static_ssim is not None else None,
                                 "mse": static_mse, "lpips": None}
                row["valid_pixel_fraction"] = float(mask.float().mean())
                row["valid_ssim_window_fraction"] = float((F.max_pool2d((~mask).float(), 11, stride=1) == 0).float().mean())
                row["mask_path"] = view.get("mask_path")
                pred_u8 = (prediction[0].cpu().numpy() * 255).round().astype(np.uint8)
                canvas = Image.new("RGB", (view["width"] * 2, view["height"] + 46), (20, 20, 20))
                canvas.paste(Image.fromarray(pixels), (0, 46))
                canvas.paste(Image.fromarray(pred_u8), (view["width"], 46))
                draw = ImageDraw.Draw(canvas)
                t = view["source_timestamp_s"]
                stamp = f" source {t:.3f}s" if t is not None else " source time unavailable"
                group_label = f"/{view['validation_group']}" if view.get("validation_group") else ""
                label = f"{view['split'].upper()}{group_label} | {view['name']}{stamp}"
                draw.text((8, 5), label, fill="white")
                draw.text((8, 25), "SOURCE", fill="white")
                draw.text((view["width"] + 8, 25), f"RENDER | PSNR {row['psnr_db']:.2f} dB | SSIM {row['ssim']:.4f}", fill="white")
                image_path = directory / f"{index:05d}_{view['split']}.jpg"
                canvas.save(image_path, quality=94, subsampling=0)
                row["comparison"] = str(image_path)
                row["comparison_sha256"] = sha256(image_path)
                rows.append(row)
                if video is not None:
                    frame = ImageOps.pad(canvas, (1920, 1080), method=Image.Resampling.LANCZOS, color=(20, 20, 20))
                    try:
                        video.stdin.write(np.asarray(frame).tobytes())
                    except BrokenPipeError:
                        video.stdin.close()
                        code = video.wait()
                        video = None
                        video_status = {"status": "failed", "returncode": code,
                                        "log": str(directory / "video_ffmpeg.log")}
                if index % 10 == 0 or index == len(selected) - 1:
                    atomic_json(output / "progress.json", {"status": "evaluating", "updated_at": utcnow(),
                                "completed_steps": step, "evaluated_views": index + 1, "total_views": len(selected),
                                "inputs_sha256": manifest["inputs_sha256"]})
                    print(f"evaluate step={step} views={index + 1}/{len(selected)}", flush=True)
            if video is not None:
                video.stdin.close()
                code = video.wait()
                video = None
                if code == 0:
                    video_path = directory / "comparison.mp4"
                    os.replace(video_tmp, video_path)
                    video_status = {"status": "available", "path": str(video_path), "sha256": sha256(video_path),
                                    "fps": video_fps, "frames": len(rows),
                                    "timing": "one sampled source camera per video frame; playback is not source real time"}
                else:
                    video_status = {"status": "failed", "returncode": code,
                                    "log": str(directory / "video_ffmpeg.log")}
            summary = summarize_evaluation(rows)
            result = {"schema": "real2sim-appearance-evaluation/v1", "produced_at": utcnow(),
                      "completed_steps": step, "inputs_sha256": manifest["inputs_sha256"],
                      "source_sha256": manifest["source_sha256"], "active_sh_degree": degree,
                      "camera_optimization": False, "validation_used_in_appearance_training": False,
                      "metric_geometry_verified": False,
                      "metric_protocol": "RGB [0,1], full uncropped image PSNR (120dB numerical cap); SSIM Gaussian 11/sigma1.5 valid crop; per-view arithmetic means",
                      "mask_protocol": "255=static valid; nearest resize. Static PSNR uses valid pixels; static SSIM uses only wholly valid 11x11 windows. Full scores always include masked regions. Static LPIPS is not computed because feature receptive fields span masked regions.",
                      "lpips_status": self.lpips_status, "summary": summary, "views": rows, "video": video_status}
            atomic_json(directory / "metrics.json", result)
            atomic_json(output / "latest_evaluation.json", {"path": str(directory / "metrics.json"),
                        "sha256": sha256(directory / "metrics.json"), "summary": summary,
                        "completed_steps": step, "inputs_sha256": manifest["inputs_sha256"]})
            return result
        finally:
            if video is not None:
                if video.stdin:
                    video.stdin.close()
                video.terminate()
                video.wait()
            if video_log is not None:
                video_log.close()
            restore_rng(saved_rng, rng)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=2296)
    parser.add_argument("--resume", nargs="?", const="latest", help="Checkpoint path, or latest model in output")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate/hash inputs on CPU without changing output")
    parser.add_argument("--init-max-points", type=int, default=200000)
    parser.add_argument("--max-gaussians", type=int, default=1000000)
    parser.add_argument("--allow-growth-cap-increase", action="store_true",
                        help="Resume into a fresh output with only the primitive cap increased; record the parent checkpoint.")
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--video-fps", type=float, default=6)
    parser.add_argument("--lpips", choices=("auto", "off"), default="auto")
    args = parser.parse_args()
    for key in ("steps", "max_width", "init_max_points", "max_gaussians", "checkpoint_every", "eval_every", "log_every", "video_fps"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_width < 16 or args.init_max_points < 4 or args.init_max_points > args.max_gaussians:
        parser.error("Need max-width >= 16 and 4 <= init-max-points <= max-gaussians")
    if args.reserve_gb < 0 or args.cache_gb < 0:
        parser.error("Memory budgets cannot be negative")
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")
    return args


def main():
    global STOP_REQUESTED
    args = arguments()
    implementation_sha256 = sha256(__file__)
    if args.validate_only:
        _, views, manifest = load_dataset(args.dataset, args.max_width)
        print(json.dumps({"status": "inputs_validated", "train": len(views["train"]),
                          "val": len(views["val"]), **manifest}, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat appearance training/evaluation requires a CUDA GPU")
    gsplat_version = importlib.metadata.version("gsplat")
    if gsplat_version.split("+")[0] != "1.5.3":
        raise RuntimeError("This implementation requires gsplat==1.5.3; its strategy internals are version-pinned")
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / ".appearance.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "metadata.json").exists() and not args.resume:
        raise FileExistsError("Output already contains a run; use --resume or a new output directory")
    checkpoint = None
    if args.resume:
        if args.resume == "latest":
            receipt = json.loads((output / "latest_checkpoint.json").read_text())
            checkpoint_path = Path(receipt["path"])
            if sha256(checkpoint_path) != receipt["sha256"]:
                raise ValueError("Latest model checkpoint hash mismatch")
        else:
            checkpoint_path = Path(args.resume).resolve()
            receipt_path = checkpoint_path.with_suffix(".json")
            if not receipt_path.is_file():
                raise ValueError("Checkpoint sidecar receipt is required for safe resume")
            receipt = json.loads(receipt_path.read_text())
            if sha256(checkpoint_path) != receipt["sha256"]:
                raise ValueError("Model checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "real2sim-appearance-checkpoint/v1":
            raise ValueError("Unsupported model checkpoint schema")
        if args.eval_only:
            for key, value in checkpoint["model_config"].items():
                if hasattr(args, key):
                    setattr(args, key, value)
    data, views, manifest = load_dataset(args.dataset, args.max_width)
    model_config = {k: getattr(args, k) for k in ("steps", "max_width", "seed", "init_max_points", "max_gaussians", "reserve_gb")}
    model_config["training_semantics"] = "fixed-camera-masked-gsplat153-fp32ssim-v2"
    parent_continuation = checkpoint.get("capacity_parent") if checkpoint else None
    # Earlier v2 checkpoints predate the lineage field. Preserve a matching
    # owned output's metadata when resuming those files in place.
    if checkpoint and parent_continuation is None and (output / "metadata.json").exists():
        previous_metadata = json.loads((output / "metadata.json").read_text())
        if (previous_metadata.get("inputs_sha256") == checkpoint["inputs_sha256"]
                and previous_metadata.get("config") == checkpoint["model_config"]):
            parent_continuation = previous_metadata.get("parent_capacity_continuation")
    capacity_change_allowed = False
    if args.allow_growth_cap_increase:
        if not checkpoint or args.eval_only or any(p.name != ".appearance.lock" for p in output.iterdir()):
            raise ValueError("A capacity continuation requires a checkpoint and a fresh output directory")
        previous = checkpoint["model_config"]
        changed = {k for k in set(previous) | set(model_config) if previous.get(k) != model_config.get(k)}
        if changed != {"max_gaussians"} or model_config["max_gaussians"] <= previous["max_gaussians"]:
            raise ValueError("Only an explicit increase of max-gaussians is permitted in a capacity continuation")
        parent_continuation = {"checkpoint": receipt, "previous_config": previous,
                               "new_max_gaussians": model_config["max_gaussians"],
                               "optimizer_rng_and_strategy_preserved": True,
                               "previous_capacity_continuation": parent_continuation}
        capacity_change_allowed = True
    if checkpoint and (checkpoint["inputs_sha256"] != manifest["inputs_sha256"]
                       or (checkpoint["model_config"] != model_config and not capacity_change_allowed)):
        raise ValueError("Resume inputs or training configuration differ from the saved model")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if checkpoint:
        splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in checkpoint["splats"].items()})
        initialization = checkpoint["initialization"]
        extent = initialization["scene_extent"]
    else:
        splats, extent, initialization = initialize(data, views, args, rng, device)
    optimizers = optimizers_for(splats, extent)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.steps))
    scale = args.steps / 30000
    strategy = bounded_strategy_class(DefaultStrategy)(
        args.max_gaussians, args.reserve_gb,
        refine_start_iter=max(1, round(500 * scale)), refine_stop_iter=max(2, round(15000 * scale)),
        refine_every=max(1, round(100 * scale)), reset_every=max(2, round(3000 * scale)),
        pause_refine_after_reset=min(len(views["train"]), max(0, round(3000 * scale) // 2)),
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=extent)
    step, order, cursor = 0, [], 0
    if checkpoint:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
        scheduler.load_state_dict(checkpoint["scheduler"])
        strategy_state = device_tree(checkpoint["strategy_state"], device)
        step, order, cursor = checkpoint["completed_steps"], checkpoint["order"], checkpoint["cursor"]
        restore_rng(checkpoint["rng"], rng)
        if step > args.steps:
            raise ValueError("Saved model is beyond the requested number of steps")
    del checkpoint
    torch.cuda.empty_cache()
    metadata = {"schema": "real2sim-appearance-run/v1", "started_at": utcnow(),
                "implementation_sha256": implementation_sha256, "gsplat_version": gsplat_version,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device), "config": model_config,
                "operational_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_sha256": data["source_sha256"], "inputs_sha256": manifest["inputs_sha256"],
                "units": data["units"], "initialization": initialization,
                "camera_optimization": False, "world_frame_normalization": False,
                "mask_protocol": "Optional per-view binary PNG; nearest resize; training L1 uses valid pixels, SSIM uses wholly valid windows; no valid SSIM window falls back to L1 only. Missing mask means all pixels valid.",
                "masked_train_views": sum(v.get("mask_path") is not None for v in views["train"]),
                "masked_val_views": sum(v.get("mask_path") is not None for v in views["val"]),
                "strategy": {"name": "BoundedDefaultStrategy", "opacity_reset_fix": "explicit-and v1.5.3 schedule",
                             **{key: getattr(strategy, key) for key in ("refine_start_iter", "refine_stop_iter", "refine_every", "reset_every", "pause_refine_after_reset")}},
                "validation_used_in_appearance_training": False,
                "parent_capacity_continuation": parent_continuation,
                "upstream_holdout_policy": data.get("holdout_policy", "unknown; inspect SfM provenance"),
                "metric_geometry_verified": False,
                "limitations": ["Appearance metrics do not establish metric geometry or physical validity.",
                                "Unseen surfaces and articulation cannot be validated by this model.",
                                "Camera calibration, temporal separation, and upstream SfM leakage affect held-out interpretation.",
                                "Full RNG state is captured; CUDA atomics may prevent bitwise reproducibility."]}
    atomic_json(output / "metadata.json", metadata)
    atomic_json(output / "input_manifest.json", manifest)
    atomic_json(output / "camera_manifest.json", {"units": data["units"], "coordinate_convention": "OpenCV w2c, +Z forward",
                "world_transform": np.eye(4).tolist(), "train": views["train"], "val": views["val"]})
    cache = ImageCache(args.cache_gb)
    evaluator = Evaluator(args.lpips, device)
    started = time.monotonic()
    initial_step = step
    latest = receipt if args.resume else None

    def save():
        return save_model(output, step, splats, optimizers, scheduler, strategy_state, rng,
                          order, cursor, model_config, manifest["inputs_sha256"], initialization, implementation_sha256,
                          capacity_parent=parent_continuation)

    def stopped(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"signal {signum}: stopping after a complete training step", flush=True)

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        if not args.eval_only:
            if step == 0 and latest is None:
                latest = save()
            with open(output / "train_stats.jsonl", "a", buffering=1) as stats_file:
                while step < args.steps and not STOP_REQUESTED:
                    if cursor >= len(order):
                        order = rng.permutation(len(views["train"])).tolist()
                        cursor = 0
                    view = views["train"][order[cursor]]
                    cursor += 1
                    pixels = torch.from_numpy(cache.get(view)).to(device).float()[None] / 255
                    mask = torch.from_numpy(cache.mask(view)).to(device)[None, None]
                    degree = min(3, step // max(1, round(1000 * scale)))
                    prediction, alpha, info = render(splats, view, extent, degree, rasterization)
                    strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
                    pred_nchw, pixels_nchw = prediction.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                    l1 = ((pred_nchw - pixels_nchw).abs() * mask).sum() / (3 * mask.sum())
                    similarity = ssim(pred_nchw, pixels_nchw, mask)
                    if similarity is not None and (not torch.isfinite(similarity) or not -1.002 <= float(similarity) <= 1.002):
                        raise FloatingPointError(f"SSIM outside numerical tolerance at step {step}: {float(similarity)}")
                    loss = .8 * l1 + .2 * (1 - similarity) if similarity is not None else l1
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss at step {step}")
                    loss.backward()
                    for optimizer in optimizers.values():
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    strategy.step_post_backward(splats, optimizers, strategy_state, step, info, packed=True)
                    step += 1
                    if len(splats["means"]) < 4:
                        raise RuntimeError("Densification/pruning left fewer than four Gaussians")
                    if step % args.log_every == 0 or step == 1 or step == args.steps:
                        row = {"status": "training", "updated_at": utcnow(), "completed_steps": step,
                               "target_steps": args.steps, "view": view["name"], "loss": float(loss),
                               "l1": float(l1), "train_sample_ssim": float(similarity) if similarity is not None else None,
                               "valid_pixel_fraction": float(mask.float().mean()), "active_sh_degree": degree,
                               "gaussians": len(splats["means"]), "elapsed_this_session_s": time.monotonic() - started,
                               "steps_this_session": step - initial_step,
                               "cuda_allocated_gb": torch.cuda.memory_allocated(device) / GIB,
                               "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / GIB,
                               "growth_limited_events": strategy_state.get("growth_limited_events", 0),
                               "inputs_sha256": manifest["inputs_sha256"]}
                        stats_file.write(json.dumps(row, allow_nan=False) + "\n")
                        atomic_json(output / "progress.json", row)
                        print(f"step={step}/{args.steps} loss={row['loss']:.5f} splats={row['gaussians']} VRAM={row['cuda_allocated_gb']:.2f}GiB", flush=True)
                    del loss, l1, similarity, prediction, alpha, info, pixels, mask, pred_nchw, pixels_nchw
                    if step % args.checkpoint_every == 0 or step == args.steps or STOP_REQUESTED:
                        latest = save()
                    if step % args.eval_every == 0 and step < args.steps and not STOP_REQUESTED:
                        evaluator.evaluate(output, step, splats, views, cache, extent, degree, rasterization, manifest, rng)
        if STOP_REQUESTED:
            if latest is None or latest["completed_steps"] != step:
                latest = save()
            atomic_json(output / "progress.json", {"status": "stopped", "updated_at": utcnow(),
                        "completed_steps": step, "target_steps": args.steps, "latest_checkpoint": latest,
                        "inputs_sha256": manifest["inputs_sha256"]})
            return 0
        degree = min(3, max(0, step - 1) // max(1, round(1000 * scale)))
        artifacts = export_ply(output, splats, data["units"])
        evaluation = evaluator.evaluate(output, step, splats, views, cache, extent, degree,
                                         rasterization, manifest, rng, all_views=True, video_fps=args.video_fps)
        status = "complete" if step == args.steps else "evaluated_partial"
        result = {"schema": "real2sim-appearance-result/v1", "status": status, "produced_at": utcnow(),
                  "completed_steps": step, "target_steps": args.steps, "units": data["units"],
                  "inputs_sha256": manifest["inputs_sha256"], "implementation_sha256": implementation_sha256,
                  "gaussians": len(splats["means"]), "artifacts": artifacts, "latest_checkpoint": latest,
                  "appearance_metrics": evaluation["summary"], "video": evaluation["video"],
                  "metric_geometry_verified": False, "camera_optimization": False}
        atomic_json(output / "result.json", result)
        atomic_json(output / "progress.json", {**result, "updated_at": utcnow()})
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:
        atomic_json(output / "progress.json", {"status": "failed", "updated_at": utcnow(),
                    "completed_steps": step, "target_steps": args.steps,
                    "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                    "latest_checkpoint": latest, "inputs_sha256": manifest["inputs_sha256"]})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
