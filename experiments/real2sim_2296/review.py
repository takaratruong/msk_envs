#!/usr/bin/env python3
"""Build a static, read-only real2sim review from explicitly selected inputs.

Requires numpy, matplotlib, Pillow and pycolmap (unless dataset poses suffice).
No video decoding, training, normalization, scene edits, uploads or external CDN.
Default output: RUN/review/index.html. Use --output for an isolated review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class Audit:
    def __init__(self):
        self.files, self.checks, self.issues = {}, [], []

    def track(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            if not path.is_file():
                return None
            self.files[key] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        return self.files[key]["sha256"]

    def match(self, path, expected, claim):
        actual = self.track(path)
        status = "matched" if expected and actual == expected else "missing" if actual is None else "unverified" if not expected else "mismatch"
        self.checks.append({"claim": claim, "path": str(Path(path).resolve()),
                            "expected_sha256": expected, "actual_sha256": actual, "status": status})
        return status == "matched"

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        before = self.track(path)
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != before:
                raise ValueError("changed while reading")
            return json.loads(raw)
        except (ValueError, OSError) as error:
            self.issues.append(f"Cannot use {path}: {error}")
            return None

    def changed(self):
        return [path for path, item in self.files.items()
                if not Path(path).is_file() or file_hash(path) != item["sha256"]]


def resolve(path, base):
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (Path(base) / candidate).resolve()


def group_of(frame):
    return frame.get("validation_group") or "unspecified"


def finite_time(view, frames):
    original = frames.get(view.get("name"), {})
    value = original.get("time_seconds", view.get("source_timestamp_s", view.get("time_seconds")))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def load_cameras(run, model_name, frames, dataset, audit):
    model = run / model_name
    files = [model / name for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin")]
    if not files[0].is_file():
        files = [model / name for name in ("cameras.txt", "images.txt", "points3D.txt", "rigs.txt", "frames.txt")]
    for path in files:
        audit.track(path)
    cameras, xyz, rgb = [], np.empty((0, 3)), np.empty((0, 3))
    origin = "missing"
    if files[0].is_file():
        try:
            import pycolmap
            reconstruction = pycolmap.Reconstruction(str(model))
            for im in reconstruction.images.values():
                if not im.has_pose:
                    continue
                w2c = np.eye(4)
                w2c[:3] = im.cam_from_world().matrix()
                cameras.append({"name": im.name, "center": np.asarray(im.projection_center()),
                                "forward": w2c[:3, :3].T @ np.array([0., 0., 1.]),
                                "time": finite_time({"name": im.name}, frames), "split": "train"})
            xyz = np.array([p.xyz for p in reconstruction.points3D.values()]).reshape(-1, 3)
            rgb = np.array([p.color for p in reconstruction.points3D.values()]).reshape(-1, 3) / 255.
            origin = "selected sparse model; coordinates read without transformation"
        except (ImportError, RuntimeError, ValueError) as error:
            audit.issues.append(f"Cannot read selected sparse model: {error}")
    if not cameras and dataset and dataset.get("model_path") and resolve(dataset["model_path"], run) == model:
        for view in dataset.get("train", []):
            transform = np.asarray(view.get("w2c"), dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                audit.issues.append(f"Invalid dataset camera: {view.get('name')}")
                continue
            inverse = np.linalg.inv(transform)
            cameras.append({"name": view["name"], "center": inverse[:3, 3], "forward": inverse[:3, 2],
                            "time": finite_time(view, frames), "split": "train"})
        points = resolve(dataset["points_path"], run) if dataset.get("points_path") else None
        if points and audit.track(points):
            with np.load(points, allow_pickle=False) as arrays:
                xyz = np.asarray(arrays["xyz"])
                rgb = np.asarray(arrays["rgb"]) / 255.
        origin = "dataset camera fallback; underlying model binding remains unverified"
    bad = [c["name"] for c in cameras if not np.isfinite(c["center"]).all()]
    if bad:
        audit.issues.append(f"Nonfinite camera centers excluded from plots: {bad}")
    cameras = [c for c in cameras if np.isfinite(c["center"]).all()]
    valid = np.isfinite(xyz).all(axis=1)
    if not valid.all():
        audit.issues.append(f"{int((~valid).sum())} nonfinite sparse points excluded from diagnostic plots")
    return cameras, xyz[valid], rgb[valid], origin


def model_binding(dataset, run, model_name, audit):
    if not dataset:
        return False
    model = (run / model_name).resolve()
    if resolve(dataset.get("model_path", "missing"), run) != model:
        audit.issues.append("dataset.json belongs to a different model; its poses/evaluation cannot certify this selection")
        return False
    artifacts = dataset.get("model_artifacts", {})
    checks = [audit.match(model / name, expected, "dataset camera-model binding") for name, expected in artifacts.items()]
    core_sets = ({"cameras.bin", "images.bin", "points3D.bin"}, {"cameras.txt", "images.txt", "points3D.txt"})
    complete = any(core.issubset(artifacts) for core in core_sets)
    if not artifacts and dataset.get("model_images_sha256"):
        checks.append(audit.match(model / "images.bin", dataset["model_images_sha256"], "dataset camera-image binding only"))
    if not complete:
        audit.issues.append("Dataset lacks hashes for all three core camera-model artifacts; model binding is incomplete")
    return complete and bool(checks) and all(checks)


def localization_binding(report, run, model_name, source_hash, source_matched, audit):
    """Bind a fixed-map report to current source, frame, model and mask bytes."""
    if not report:
        return {"verified": False, "status": "Unbound localization report; no report supplied", "checks": {}}
    model = (run / model_name).resolve()
    artifacts = report.get("model_artifacts", {})
    current_names = {path.name for path in model.glob("*.bin") if path.is_file()}
    core = {"cameras.bin", "images.bin", "points3D.bin"}
    checks = {
        "source_identity": bool(source_hash and source_matched and report.get("source_sha256") == source_hash),
        "frames_manifest": audit.match(run / "frames.json", report.get("frames_sha256"), "localization frame manifest binding"),
        "selected_model_path": bool(report.get("model_path") and resolve(report["model_path"], run) == model),
        "complete_model_manifest": core.issubset(artifacts) and set(artifacts) == current_names,
        "camera_mask_receipt": audit.match(run / "camera_masks" / "receipt.json", report.get("camera_mask_receipt_sha256"), "localization camera-mask receipt binding"),
    }
    for name, expected in artifacts.items():
        # A model artifact must name a file within the selected binary model.
        valid_name = isinstance(name, str) and Path(name).name == name and name.endswith(".bin")
        checks["model_artifact:" + str(name)] = valid_name and audit.match(model / name, expected, "localization selected-model artifact")
    verified = all(checks.values())
    failed = [name for name, matched in checks.items() if not matched]
    status = "Bound to selected fixed map; source, frame, model and mask-receipt hashes matched" if verified else \
             "Stale or unbound localization report; unmatched checks: " + ", ".join(failed)
    return {"verified": verified, "status": status, "checks": checks,
            "evidence_limit": "Byte linkage only; reported pose estimates do not establish metric geometry or real LiDAR localization accuracy"}


def localization_summary(frames, report):
    """Keep every selected validation frame, including absent/rejected attempts."""
    views = (report or {}).get("views", [])
    frequencies = Counter(item.get("name") for item in views)
    by_name = {item["name"]: item for item in views if item.get("name")}
    duplicate_names = sorted(name for name, count in frequencies.items() if name and count > 1)
    rows, groups = [], {}
    for frame in frames:
        if frame["split"] != "val":
            continue
        group = group_of(frame)
        item = by_name.get(frame["name"])
        status = item.get("status", "unknown_report_status") if item else "not_reported"
        if frame["name"] in duplicate_names:
            status = "duplicate_report_ambiguous"
        ambiguous = status in {"unknown_report_status", "duplicate_report_ambiguous"}
        row = {"name": frame["name"], "time_seconds": frame["time_seconds"], "group": group,
               "status": status, "inliers": item.get("inliers") if item else None}
        rows.append(row)
        counts = groups.setdefault(group, {"selected": 0, "attempted": 0, "localized": 0, "rejected": 0, "ambiguous": 0, "not_reported": 0})
        counts["selected"] += 1
        counts["attempted"] += int(item is not None)
        counts["localized"] += int(status == "localized_fixed_map")
        counts["rejected"] += int(item is not None and status != "localized_fixed_map" and not ambiguous)
        counts["ambiguous"] += int(ambiguous)
        counts["not_reported"] += int(item is None)
    known_names = {row["name"] for row in rows}
    total_mismatches = []
    if report and "views" in report:
        for key in ("attempted", "localized", "rejected"):
            if key in report and report[key] != sum(group[key] for group in groups.values()):
                total_mismatches.append(key)
        for name, counts in report.get("by_group", {}).items():
            for key in ("attempted", "localized", "rejected"):
                if key in counts and counts[key] != groups.get(name, {}).get(key):
                    total_mismatches.append(f"by_group.{name}.{key}")
    return {"groups": groups, "views": rows, "duplicate_reports": bool(duplicate_names),
            "per_view_report_available": bool(report and "views" in report),
            "duplicate_report_names": duplicate_names, "unnamed_report_count": frequencies.get(None, 0) + frequencies.get("", 0),
            "reported_total_mismatches": total_mismatches,
            "extra_report_names": sorted(set(by_name) - known_names),
            "reported_totals": {key: report.get(key) for key in ("attempted", "localized", "rejected", "by_group")} if report else None}


def appearance_data(path, dataset, dataset_bound, run, source_hash, audit, source_matched=True):
    result = {"status": "missing", "verified_input_linkage": False, "metrics": None, "comparisons": []}
    if path is None:
        return result
    manifest = audit.read(path / "input_manifest.json")
    pointer = audit.read(path / "latest_evaluation.json")
    if not pointer or not pointer.get("path"):
        result["status"] = "supplied; no completed evaluation pointer"
        return result
    metrics_path = resolve(pointer["path"], path)
    pointer_match = audit.match(metrics_path, pointer.get("sha256"), "appearance evaluation pointer")
    metrics = audit.read(metrics_path)
    if not metrics:
        result["status"] = "unreadable evaluation"
        return result
    checks = [pointer_match, dataset_bound, bool(manifest), source_matched]
    if manifest:
        body = {k: v for k, v in manifest.items() if k != "inputs_sha256"}
        try:
            internal = digest(body) == manifest.get("inputs_sha256")
        except ValueError:
            internal = False
        checks += [internal, metrics.get("inputs_sha256") == manifest.get("inputs_sha256"),
                   pointer.get("inputs_sha256") == manifest.get("inputs_sha256"),
                   manifest.get("source_sha256") == source_hash, metrics.get("source_sha256") == source_hash,
                   audit.match(run / "dataset.json", manifest.get("dataset_sha256"), "appearance training dataset")]
        expected_views = [(item["name"], split) for split in ("train", "val") for item in (dataset or {}).get(split, [])]
        manifest_views = [(item.get("name"), item.get("split")) for item in manifest.get("images", [])]
        checks.append(bool(expected_views) and Counter(manifest_views) == Counter(expected_views))
        metric_views = [(item.get("name"), item.get("split")) for item in metrics.get("views", [])]
        checks.append(bool(metric_views) and len(set(metric_views)) == len(metric_views) and set(metric_views).issubset(expected_views))
        if manifest.get("points_path"):
            checks.append(audit.match(resolve(manifest["points_path"], path), manifest.get("points_sha256"), "appearance initial points"))
        for item in manifest.get("images", []):
            checks.append(audit.match(resolve(item["image_path"], path), item.get("sha256"), "appearance input image"))
            if item.get("mask_path"):
                checks.append(audit.match(resolve(item["mask_path"], path), item.get("mask_sha256"), "appearance input mask"))
    linked = all(checks)
    result.update(status="input and evaluation hashes matched" if linked else "unverified or mismatched input linkage",
                  verified_input_linkage=linked, metrics=metrics, metrics_path=str(metrics_path))
    for view in metrics.get("views", []):
        if not view.get("comparison"):
            continue
        comparison = resolve(view["comparison"], metrics_path.parent)
        matched = audit.match(comparison, view.get("comparison_sha256"), "source/Gaussian comparison image")
        result["comparisons"].append({"view": view, "path": comparison, "hash_matched": matched,
                                      "current_input_linkage": linked})
    return result


def scene_data(path, source_hash, audit, source_matched=True):
    if path is None:
        return {"status": "missing", "manifest": {}}
    directory = path if path.is_dir() else path.parent
    manifest = audit.read(directory / "manifest.json") or {}
    entry = path if path.is_file() else directory / manifest.get("entrypoint", "scene.usda")
    checks = [source_matched, manifest.get("source_sha256") == source_hash, bool(manifest.get("artifacts"))]
    for name, expected in manifest.get("artifacts", {}).items():
        checks.append(audit.match(directory / name, expected, "scene package artifact"))
    checks.append(bool(audit.track(entry)))
    checks.append(any(resolve(name, directory) == entry for name in manifest.get("artifacts", {})))
    return {"status": "artifact hashes matched; selected-model binding and physical assertions unverified" if all(checks)
            else "missing or unverified scene receipt", "manifest": manifest, "entrypoint": str(entry)}


def surface_reviews(run, source_hash, audit):
    rows = []
    for name in ("geometry", "geometry_prior"):
        directory = run / name
        metrics = audit.read(directory / "metrics.json")
        review = audit.read(directory / "review" / "metrics.json")
        if not metrics or not review:
            continue
        checks = [metrics.get("source_sha256") == source_hash, review.get("source_sha256") == source_hash,
                  audit.match(run / "dataset.json", review.get("dataset_sha256"), "mesh review camera dataset"),
                  audit.match(directory / "metrics.json", review.get("geometry_metrics_sha256"), "mesh review geometry metrics"),
                  audit.match(directory / "surface_arrays.npz", review.get("mesh_arrays_sha256"), "raycast mesh arrays")]
        for view in review.get("views", []):
            checks.append(audit.match(resolve(view["comparison"], directory / "review"),
                                      view.get("comparison_sha256"), "mesh source/color/normal comparison"))
        audit.track(directory / "review" / "index.html")
        fractions = [v["ray_hit_fraction"] for v in review.get("views", [])]
        rows.append({"directory": name, "basis": "Learned-depth hypothesis" if metrics.get("inferred_prior") else "RGB multiview stereo",
                     "inferred_prior": bool(metrics.get("inferred_prior")), "triangles": metrics.get("triangles"),
                     "boundary_edges": metrics.get("boundary_edges"), "watertight": metrics.get("watertight"),
                     "views": len(fractions), "ray_hit_min": min(fractions) if fractions else None,
                     "ray_hit_max": max(fractions) if fractions else None,
                     "input_artifact_binding": "matched" if all(checks) else "unverified / mismatch"})
    return rows


def style():
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#111923", "axes.facecolor": "#111923", "axes.edgecolor": "#526376",
                         "grid.color": "#334154", "grid.alpha": .45, "font.size": 10,
                         "savefig.facecolor": "#111923", "axes.titleweight": "bold"})


def trajectory_plot(path, cameras, frames):
    style()
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, .9]})
    timed = sorted([c for c in cameras if c["time"] is not None], key=lambda c: c["time"])
    times = np.array([c["time"] for c in timed])
    centers = np.array([c["center"] for c in timed]).reshape(-1, 3)
    for axis, dimension in enumerate("XYZ"):
        axes[axis].scatter(times, centers[:, axis], s=15, color="#57d6bd", zorder=3)
        # A display line may only connect estimates less than one second apart.
        for i in range(max(0, len(times) - 1)):
            if times[i + 1] - times[i] <= 1.:
                axes[axis].plot(times[i:i + 2], centers[i:i + 2, axis], color="#57d6bd", linewidth=.8, alpha=.55)
        axes[axis].set_ylabel(dimension + " (raw units)")
        axes[axis].grid(True)
    registered = {c["name"] for c in cameras}
    categories = [("train_registered", 4, "#57d6bd"), ("train_unregistered", 3, "#ff7979"),
                  ("val_interleaved", 2, "#78a9ff"), ("val_buffered", 1, "#edb75c"), ("buffer", 0, "#9c8aa9")]
    for name, y, color in categories:
        selected = []
        for frame in frames:
            category = ("train_registered" if frame["name"] in registered else "train_unregistered") if frame["split"] == "train" else \
                       "buffer" if frame["split"] == "buffer" else "val_buffered" if "buffer" in group_of(frame) else "val_interleaved"
            if category == name:
                selected.append(frame["time_seconds"])
        axes[3].scatter(selected, [y] * len(selected), marker="|", s=75, color=color)
    axes[3].set_yticks([r[1] for r in categories], [r[0].replace("_", " ") for r in categories], fontsize=8)
    axes[3].set_ylim(-.7, 4.7)
    axes[3].set_xlabel("Source video PTS (seconds), from the frame manifest")
    if frames:
        axes[3].set_xlim(min(f["time_seconds"] for f in frames), max(f["time_seconds"] for f in frames))
    fig.suptitle("Camera position over source time — original reconstruction coordinates\nNo metric scale or gravity alignment; gaps are not interpolated", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def cloud_plot(path, cameras, xyz, rgb):
    style()
    fig = plt.figure(figsize=(14, 7))
    stride = max(1, math.ceil(len(xyz) / 25000))
    displayed, colors = xyz[::stride], np.clip(rgb[::stride], 0, 1)
    centers = np.array([c["center"] for c in cameras]).reshape(-1, 3)
    all_xyz = np.concatenate([xyz, centers]) if len(xyz) + len(centers) else np.array([[0., 0., 0.]])
    full_bounds = (all_xyz.min(axis=0), all_xyz.max(axis=0))
    crop_bounds = np.quantile(xyz, [.005, .995], axis=0) if len(xyz) else np.array(full_bounds)
    if len(centers):
        crop_bounds[0] = np.minimum(crop_bounds[0], centers.min(axis=0))
        crop_bounds[1] = np.maximum(crop_bounds[1], centers.max(axis=0))
    for index, (bounds, azimuth) in enumerate(zip((full_bounds, crop_bounds), (-65, 35)), 1):
        lo, hi = bounds
        mid, radius = (lo + hi) / 2, max(float(np.max(hi - lo)) / 2, 1e-3)
        visible = np.all(np.abs(displayed - mid) <= radius * 1.000001, axis=1)
        outside = int(np.sum(np.any(np.abs(xyz - mid) > radius * 1.000001, axis=1)))
        ax = fig.add_subplot(1, 2, index, projection="3d")
        if visible.any():
            ax.scatter(*displayed[visible].T, c=colors[visible], s=.7 if index == 1 else 1.3, alpha=.65, rasterized=True)
        if len(centers):
            ax.scatter(*centers.T, c="#ffce73", s=14, depthshade=False)
            for camera in cameras[::max(1, math.ceil(len(cameras) / 50))]:
                ax.quiver(*camera["center"], *camera["forward"], length=radius * .08, color="#ffce73", alpha=.7)
        ax.set(xlim=(mid[0] - radius, mid[0] + radius), ylim=(mid[1] - radius, mid[1] + radius),
               zlim=(mid[2] - radius, mid[2] + radius), xlabel="Raw X", ylabel="Raw Y", zlabel="Raw Z")
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((.08, .12, .17, 1.))
        ax.view_init(elev=22, azim=azimuth)
        ax.set_title("Full extent, including outliers" if index == 1 else f"Expanded display crop · {outside:,} points outside view", fontsize=11)
    sampling = "All points displayed" if stride == 1 else f"Every {stride}th point displayed"
    fig.suptitle(f"Sparse points and camera viewing directions | {len(xyz):,} points, {len(cameras)} cameras\n"
                 f"{sampling}; coordinates unchanged. Right view clips display bounds only. Arrows use a display length.", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def contact_plot(path, frames, registered):
    ordered = sorted(frames, key=lambda item: item["time_seconds"])
    selected = set(np.linspace(0, len(ordered) - 1, min(20, len(ordered))).round().astype(int).tolist())
    for group in sorted({group_of(f) for f in ordered if f["split"] == "val"}):
        indexes = [i for i, f in enumerate(ordered) if f["split"] == "val" and group_of(f) == group]
        selected.update(indexes[::max(1, len(indexes) // 2)][:2])
    sample = [ordered[i] for i in sorted(selected)]
    width, height = 320, 218
    sheet = Image.new("RGB", (width * 4, height * math.ceil(len(sample) / 4)), "#111923")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(sample):
        x, y = i % 4 * width, i // 4 * height
        p = Path(frame["image_path"])
        if p.is_file():
            with Image.open(p) as source:
                thumbnail = ImageOps.pad(source.convert("RGB"), (width, 180), color="#111923")
            sheet.paste(thumbnail, (x, y))
        state = "registered" if frame["name"] in registered else "unregistered" if frame["split"] == "train" else group_of(frame)
        color = "#57d6bd" if frame["hash_matched"] else "#ff7979"
        draw.text((x + 5, y + 183), f"PTS {frame['time_seconds']:.3f}s · source #{frame['source_index']}", font=font, fill="white")
        draw.text((x + 5, y + 199), f"{frame['split']} · {state} · {'hash matched' if frame['hash_matched'] else 'UNVERIFIED'}", font=font, fill=color)
    sheet.save(path)


def text(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return "<div class='table-scroll'><table><thead><tr>" + "".join(f"<th>{text(v)}</th>" for v in headers) + \
           "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{text(v)}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--appearance", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--output", type=Path, help="Default RUN/review; use a new /tmp path for isolated testing")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_name):
        parser.error("model-name must be one simple directory name")
    run = args.run.resolve(strict=True)
    output = (args.output or run / "review").resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    audit.track(__file__)
    manifest_path = run / "frames.json"
    manifest = audit.read(manifest_path)
    if not manifest or not manifest.get("frames"):
        raise ValueError("A nonempty frames.json with original source PTS is required")
    frames = [dict(item) for item in manifest["frames"]]
    names = [frame["name"] for frame in frames]
    if len(set(names)) != len(names) or any(finite_time(f, {}) is None for f in frames):
        raise ValueError("Frame names must be unique and all original time_seconds values finite")
    source_hash = manifest.get("source_sha256")
    source_path = resolve(manifest.get("source_path", "missing_source"), run)
    source_matched = audit.match(source_path, source_hash, "source MOV byte identity; no new decode")
    for frame in frames:
        frame["image_path"] = str(resolve(frame["image_path"], run))
        frame["hash_matched"] = audit.match(frame["image_path"], frame.get("sha256"), "extracted source frame")
    by_name = {f["name"]: f for f in frames}
    dataset = audit.read(run / "dataset.json")
    bound = model_binding(dataset, run, args.model_name, audit)
    if dataset and dataset.get("source_sha256") != source_hash:
        bound = False
        audit.issues.append("Dataset source SHA differs from the selected frame manifest")
    cameras, xyz, rgb, origin = load_cameras(run, args.model_name, by_name, dataset, audit)
    registered = {camera["name"] for camera in cameras}
    train_names = {f["name"] for f in frames if f["split"] == "train"}
    foreign = sorted(registered - train_names)
    if foreign:
        audit.issues.append(f"Selected camera model contains {len(foreign)} names outside the training split")
    metrics_path = run / ("sfm_metrics.json" if args.model_name == "model" else args.model_name + "_metrics.json")
    sfm = audit.read(metrics_path) or {}
    fraction = len(registered & train_names) / len(train_names) if train_names else 0.
    components = sfm.get("components", [])
    invalid_cameras = bool(sfm.get("degenerate_extent_rejected")) or bool(foreign) or not registered
    rejected = invalid_cameras or (fraction < .95 and len(components) > 1)
    status = "Rejected by current camera gate; not promoted" if invalid_cameras else \
             "Rejected — fragmented candidate; not promoted" if rejected else \
             "Reconstruction candidate — partial camera coverage; not accepted as an exact replica" if fraction < .95 else \
             "Reconstruction candidate — promotion unverified"
    if sfm and (sfm.get("source_sha256") != source_hash or sfm.get("registered") != len(cameras) or sfm.get("points") != len(xyz)):
        audit.issues.append("SfM metric report disagrees with the current source/model counts; residuals are not current verified measurements")
    localization_report = audit.read(run / "evaluation" / "localization.json")
    localization_origin = "evaluation/localization.json" if localization_report else "missing"
    if not localization_report and (dataset or {}).get("validation_localization"):
        localization_report = dataset["validation_localization"]
        localization_origin = "dataset.validation_localization aggregate only; no per-frame report"
    localization = localization_summary(frames, localization_report)
    localization["report_origin"] = localization_origin
    localization["binding"] = localization_binding(localization_report, run, args.model_name, source_hash, source_matched, audit)
    localization["model_binding"] = localization["binding"]["status"]
    for key in ("duplicate_report_names", "extra_report_names", "reported_total_mismatches"):
        if localization[key]:
            audit.issues.append(f"Localization {key}: {localization[key]}")
    if localization["unnamed_report_count"]:
        audit.issues.append(f"Localization report contains {localization['unnamed_report_count']} unnamed records")
    appearance = appearance_data(args.appearance.resolve() if args.appearance else None, dataset, bound, run, source_hash, audit, source_matched)
    mismatched_eval_frames = []
    for view in (appearance["metrics"] or {}).get("views", []):
        frame = by_name.get(view.get("name"))
        timestamp = view.get("source_timestamp_s")
        if not frame or view.get("split") != frame["split"] or \
           (frame["split"] == "val" and group_of(view) != group_of(frame)) or \
           (timestamp is not None and (not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or abs(timestamp - frame["time_seconds"]) > 1e-6)):
            mismatched_eval_frames.append(view.get("name"))
    if mismatched_eval_frames:
        audit.issues.append(f"Appearance names, split, group or source PTS disagree with current frame manifest: {mismatched_eval_frames}")
        appearance["verified_input_linkage"] = False
        appearance["status"] = "unverified; evaluation frame identity mismatch"
        for item in appearance["comparisons"]:
            item["current_input_linkage"] = False
    scene = scene_data(args.scene.resolve() if args.scene else None, source_hash, audit, source_matched)
    geometry = audit.read(run / "geometry" / "metrics.json")
    geometry_matches = []
    if geometry:
        geometry_matches += [source_matched, bound, geometry.get("source_sha256") == source_hash,
                             audit.match(run / "dataset.json", geometry.get("dataset_sha256"), "observed geometry dataset binding"),
                             bool(geometry.get("artifacts"))]
        for name, expected in geometry.get("artifacts", {}).items():
            geometry_matches.append(audit.match(run / "geometry" / name, expected, "observed surface artifact"))
    surfaces = surface_reviews(run, source_hash, audit)
    static_scores = (appearance["metrics"] or {}).get("summary", {}).get("val", {}).get("static", {})
    below_target = any(isinstance(static_scores.get(key), (float, int)) and
                       (static_scores[key] < threshold if key != "lpips" else static_scores[key] > threshold)
                       for key, threshold in [("psnr_db", 28.), ("ssim", .90), ("lpips", .15)])
    appearance_gate = ("below target / rejected; " if below_target else "acceptance unverified; ") + appearance["status"]
    artifact_links = []
    for target, label in [(run / "semantic_inventory" / "index.html", "Source object inventory"),
                          (run / "final_native" / "index.html", "Actual canonical USD rendering"),
                          (run / "scene" / "scene.usda", "Canonical scene.usda"),
                          ((args.appearance.resolve() if args.appearance else run / "no_appearance_supplied") / "eval" / "step_030000_all" / "comparison.mp4", "Evaluated source/render camera comparison video")]:
        if target.is_file():
            audit.track(target)
            artifact_links.append(f"<a href='{text(os.path.relpath(target, output))}'>{text(label)}</a>")
    gates = [
        ("Source identity", "matched" if source_matched and all(f["hash_matched"] for f in frames) else "unverified / mismatch", "MOV and extracted frame bytes checked; timestamps retained from manifest, no re-decoding"),
        ("Camera coverage", "rejected" if rejected else "partial coverage; below target" if fraction < .95 else "candidate; not promoted", f"{len(registered & train_names)}/{len(train_names)} training selections registered; target ≥95%; {len(components) if components else 'unreported'} mapping components"),
        ("Photorealistic appearance", appearance_gate, "Static targets: PSNR ≥28 dB, SSIM ≥0.90, LPIPS ≤0.15. Missing static LPIPS remains unavailable; full-image LPIPS is not substituted. Failed localization stays in the denominator."),
        ("Metric scale / gravity", "unverified", "No independent metric anchors or gravity calibration accepted by this review; raw coordinate units remain unchanged"),
        ("Observed geometry", "input/artifact hashes matched; geometry unverified" if geometry_matches and all(geometry_matches) else "missing / unverified input linkage", "Surface distance, contact accuracy and unseen coverage require independent measurements"),
        ("Collision and physical parameters", "unverified", f"Scene collision_enabled={scene['manifest'].get('collision_enabled', 'not supplied')}; no validated mass, friction or contact evidence"),
        ("Objects and articulations", "unverified", f"articulations_authored={scene['manifest'].get('articulations_authored', 'not supplied')}; instance geometry and measured joint motion not certified"),
        ("Real LiDAR localization", "unverified", "No accepted real scan, sensor/extrinsic or ground-truth localization trial receipt"),
        ("Unitree G1 manipulation", "unverified", "Exact hand/configuration, stable-contact checks and matched real pick/place trials are not certified"),
        ("Real/sim performance correlation", "unverified", "No accepted matched real/sim interventions and repeated task outcomes"),
        ("Canonical OpenUSD", scene["status"], "A composed asset alone does not pass deployment or physical gates"),
    ]
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    media_name = "media_" + generation
    media = output / media_name
    media.mkdir()
    trajectory_plot(media / "trajectory.png", cameras, frames)
    cloud_plot(media / "camera_cloud.png", cameras, xyz, rgb)
    contact_plot(media / "source_contacts.png", frames, registered)
    comparison_cards = []
    ordered = sorted(appearance["comparisons"], key=lambda item: item["view"].get("psnr_db") if isinstance(item["view"].get("psnr_db"), (float, int)) else float("inf"))
    chosen = ordered[:6] + ordered[::max(1, len(ordered) // 6)][:6]
    seen = set()
    for item in chosen:
        source = item["path"]
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        dest = media / f"comparison_{len(seen):02d}.jpg"
        try:
            with Image.open(source) as im:
                preview = im.convert("RGB")
                preview.thumbnail((1600, 950), Image.Resampling.LANCZOS)
                preview.save(dest, quality=92)
        except (OSError, ValueError) as error:
            audit.issues.append(f"Cannot preview appearance comparison {source}: {error}")
            continue
        view = item["view"]
        label = "input/metric/image hashes matched" if item["hash_matched"] and item["current_input_linkage"] else "UNVERIFIED INPUT LINKAGE"
        comparison_cards.append(f"<figure><img loading='lazy' src='{text(media_name + '/' + dest.name)}' alt='Source and Gaussian comparison'>"
                                f"<figcaption>{text(view.get('name'))} · source PTS {text(number(finite_time(view, by_name)))}s · {text(view.get('split'))}/{text(view.get('validation_group') or 'none')} · "
                                f"PSNR {text(number(view.get('psnr_db')))} dB · {text(label)}</figcaption></figure>")
    changed = audit.changed()
    if changed:
        status = "STALE INPUT SNAPSHOT — files changed during review"
        audit.issues += ["Changed during review: " + path for path in changed]
    summary = {"schema": "real2sim-review/v1", "produced_at": datetime.now(timezone.utc).isoformat(),
               "run": str(run), "model_name": args.model_name, "status": status,
               "normalization_applied": False, "camera_origin": origin,
               "camera_count": len(cameras), "point_count": len(xyz), "registered_training": len(registered & train_names),
               "selected_training": len(train_names), "selected_validation": sum(f["split"] == "val" for f in frames),
               "excluded_buffer": sum(f["split"] == "buffer" for f in frames), "mapping_components_reported": len(components),
               "source_hash_matched": source_matched, "frame_hash_matches": sum(f["hash_matched"] for f in frames),
               "frame_count": len(frames), "timestamp_source": "original time_seconds in hashed frame manifest; no fps approximation or re-decoding",
               "sfm_metrics_status": "reported; counts compared, no residual-to-model hash binding in this report format",
               "sfm_metrics": sfm, "localization": localization, "appearance_status": appearance["status"],
               "appearance_inputs_verified": appearance["verified_input_linkage"], "scene_status": scene["status"],
               "surface_reviews": surfaces,
               "gates": [{"gate": g, "status": s, "evidence_limit": e} for g, s, e in gates],
               "issues": audit.issues, "inputs_changed_during_generation": changed}
    inputs = {"schema": "real2sim-review-inputs/v1", "run": str(run), "model_name": args.model_name,
              "appearance": str(args.appearance) if args.appearance else None, "scene": str(args.scene) if args.scene else None,
              "files": audit.files, "checks": audit.checks}
    inputs_hash = digest(inputs)
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    group_rows = []
    evaluated = defaultdict(int)
    for item in (appearance["metrics"] or {}).get("views", []):
        if item.get("split") == "val":
            evaluated[item.get("validation_group") or "unspecified"] += 1
    for group, counts in localization["groups"].items():
        group_rows.append([group, counts["selected"], counts["attempted"], counts["localized"], counts["rejected"], counts["ambiguous"], counts["not_reported"], evaluated[group]])
    metrics_rows = []
    for split, values in (appearance["metrics"] or {}).get("summary", {}).items():
        sets = [(split, values)] + [(split + "/" + group, v) for group, v in values.get("by_group", {}).items()]
        for label, group in sets:
            for region in ("full", "static"):
                scores = group.get(region, {})
                metrics_rows.append([label, region, group.get("views", group.get("count", "reported separately")),
                                     number(scores.get("psnr_db")), number(scores.get("ssim")), number(scores.get("lpips"))])
    localization_rows = [[r["name"], number(r["time_seconds"]), r["group"], r["status"], r["inliers"] if r["inliers"] is not None else "—"] for r in localization["views"]]
    sfm_rows = [["Registered training frames", f"{len(registered & train_names)} / {len(train_names)} ({fraction:.1%})", "measured from selected sparse model"],
                ["Sparse points", len(xyz), "read from selected model"],
                ["Mapping components", len(components) if components else "unavailable", "reported by mapping log/metrics"],
                ["Median reprojection error", number(sfm.get("reprojection_median_px")) + " px", "reported; not an independent geometric error"],
                ["P95 reprojection error", number(sfm.get("reprojection_p95_px")) + " px", "reported; target ≤2 px"],
                ["Coordinate frame", "raw reconstruction units", "no normalization; metric scale and gravity unverified"]]
    surface_rows = [[r["basis"], r["triangles"], r["boundary_edges"], r["watertight"], r["views"],
                     f"{r['ray_hit_min']:.1%}–{r['ray_hit_max']:.1%}" if r["views"] else "unavailable",
                     r["input_artifact_binding"]] for r in surfaces]
    surface_links = " · ".join(f"<a href='{text(os.path.relpath(run / r['directory'] / 'review' / 'index.html', output))}'>{text(r['basis'])} source/color/normal views</a>" for r in surfaces)
    warnings = "".join(f"<li>{text(issue)}</li>" for issue in audit.issues)
    css = """*{box-sizing:border-box}body{margin:0;background:#0a1018;color:#e7eef7;font:15px/1.5 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:35px 24px}h1{font-size:32px;margin:4px 0}h2{margin:30px 0 12px;font-size:21px}p{color:#adbed0}a{color:#7ab6ff}code{font-size:12px;overflow-wrap:anywhere}.eyebrow{color:#78a9ff;text-transform:uppercase;letter-spacing:.15em;font-size:12px}.banner{background:#3a2226;border-left:5px solid #ff7979;padding:18px;margin:18px 0;border-radius:7px;font-size:20px;font-weight:650}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#151f2c;border:1px solid #2b3b50;padding:17px;border-radius:9px}.value{display:block;font-size:26px;font-weight:700}.label{font-size:12px;color:#b2c1d3}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:#121c28;font-size:13px}th,td{padding:11px;text-align:left;border-bottom:1px solid #2c3b4e;vertical-align:top}th{color:#8cbaff}figure{margin:15px 0;border:1px solid #2b3b50;background:#111923;border-radius:8px;overflow:hidden}img{display:block;width:100%;height:auto}figcaption{padding:11px;color:#b7c5d6;font-size:12px}.comparisons{display:grid;grid-template-columns:1fr 1fr;gap:15px}details{margin:18px 0;padding:16px;background:#121c28;border-radius:8px}summary{cursor:pointer;font-weight:600}.muted{font-size:12px;color:#91a4bc}.footer{border-top:1px solid #2b3b50;margin-top:35px;padding-top:16px}@media(max-width:800px){main{padding:20px 12px}.cards,.comparisons{grid-template-columns:1fr 1fr}.banner{font-size:17px}}@media(max-width:500px){.cards,.comparisons{grid-template-columns:1fr}}"""
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{text(args.model_name)} · real2sim review</title><style>{css}</style></head><body><main>
<div class='eyebrow'>IMG_2296 · read-only reconstruction review</div><h1>{text(args.model_name)}</h1><p>{text(run)}<br>Generated {text(summary['produced_at'])}</p><div class='banner'>{text(status)}</div>
<div class='cards'><div class='card'><span class='value'>{len(registered & train_names)}/{len(train_names)}</span><span class='label'>training cameras registered</span></div><div class='card'><span class='value'>{summary['selected_validation']}</span><span class='label'>validation selections retained in denominator</span></div><div class='card'><span class='value'>{len(xyz):,}</span><span class='label'>sparse points · visual consistency only</span></div><div class='card'><span class='value'>Raw units</span><span class='label'>metric scale / physics / deployment unverified</span></div></div>
<p>This review does not promote a map or certify an exact replica. Recorded timestamps come from the current hashed frame manifest; no frame-rate approximation, camera normalization or video re-decoding is performed. Current frame hashes: {summary['frame_hash_matches']}/{len(frames)} matched. Source MOV identity: {'matched' if source_matched else 'unverified'}.</p>
<p>{' · '.join(artifact_links)}</p>
<h2>Source video contact previews</h2><figure><img src='{media_name}/source_contacts.png' alt='Source video frames annotated with original PTS, split and camera registration'><figcaption>Time-spanning previews are generated from the current frame manifest, not the initial approximate contact sheet. Hash labels refer to extracted image bytes; the timestamps are preserved manifest values.</figcaption></figure>
<h2>Camera trajectory and sparse geometry</h2>{table(['Measure','Value','Evidence limit'],sfm_rows)}
<figure><img src='{media_name}/trajectory.png' alt='Camera XYZ coordinates over original video timestamps and complete selection coverage'><figcaption>{text(origin)}. Missing camera intervals remain visible. Validation and excluded buffer groups are retained.</figcaption></figure>
<figure><img src='{media_name}/camera_cloud.png' alt='Full-extent and magnified 3D views of cameras and sparse points'><figcaption>Original raw axes; no metric or gravity claim. Left: full bounds including outliers. Right: a display crop based on point quantiles and all camera centers, with the number of points outside that view stated. Source data is unchanged. These points are neither a closed collision mesh nor a LiDAR map.</figcaption></figure>
<h2>Validation coverage — failures stay in the denominator</h2><p>{text(manifest.get('holdout_rule','Holdout policy unavailable'))}<br>{text(localization['model_binding'])}. “Not reported” is distinct from a failed localization; neither disappears from the selected count. Excluded temporal buffer frames: {summary['excluded_buffer']}.</p>
{table(['Group','Selected','Frames with records','Localized','Rejected','Ambiguous','Not reported','Image metric rows'],group_rows)}
<p class='muted'>Report source: {text(localization_origin)}. Reported aggregate counts ({'input hashes bound' if localization['binding']['verified'] else 'unbound'}): {text(json.dumps(localization['reported_totals'], sort_keys=True))}. Table counts are rebuilt from per-frame records where available. Matching input hashes do not establish metric geometry or real LiDAR localization accuracy.</p>
<details><summary>Every selected validation frame ({summary['selected_validation']})</summary>{table(['Frame','Source PTS (s)','Group','Localization status','Inliers'],localization_rows)}</details>
<h2>Source versus Gaussian appearance</h2><p>{text(appearance['status'])}. Reported metrics apply only to evaluated images; they do not cover unlocalized validation selections or establish surface accuracy. Full-image and static-mask scores remain separate. Missing LPIPS is unavailable, never zero.</p>
{table(['Split / group','Region','Evaluated views','PSNR dB','SSIM','LPIPS'],metrics_rows) if metrics_rows else '<p>No completed appearance evaluation supplied for this review.</p>'}
<div class='comparisons'>{''.join(comparison_cards)}</div>
<h2>Surfaces against source cameras</h2><p>Ray-hit fractions below measure image coverage, not correct geometry or surveyed completeness. The stereo surface has major holes; the learned-depth hypothesis fills more surfaces but retains floor artifacts and distorted thin parts. Both are open and unsuitable as validated collision. The canonical scene keeps these as hidden inspection layers with collision disabled.</p>
{table(['Surface basis','Triangles','Boundary edges','Watertight','Reviewed views','Image ray-hit range','Input / artifact binding'],surface_rows) if surface_rows else '<p>Surface raycast review unavailable.</p>'}<p>{surface_links}</p>
<h2>Environment acceptance gates</h2>{table(['Gate','Current evidence status','What remains unverified'],gates)}
<details><summary>Input issues and receipt checks</summary><ul>{warnings or '<li>No input inconsistency detected by the implemented checks. Missing task evidence remains unverified above.</li>'}</ul><p><a href='inputs.json'>Full input hashes and per-artifact receipt checks</a> · <a href='summary.json'>Machine-readable review summary</a> · <a href='receipt.json'>Generated review artifact receipt</a></p><p class='muted'>A matching file hash establishes byte identity, not physical accuracy or the truth of every statement inside a report. SfM residuals and scene flags are labeled according to the binding evidence available.</p></details>
<div class='footer muted'>Input snapshot SHA256 <code>{inputs_hash}</code><br>Source MOV SHA256 <code>{text(source_hash)}</code><br>Static local HTML and PNGs; no external services, scripts, controls or approval actions.</div></main></body></html>"""
    temporary = output / f".index.{os.getpid()}.tmp"
    temporary.write_text(page)
    os.replace(temporary, output / "index.html")
    artifacts = {str(p.relative_to(output)): file_hash(p) for p in [output / "index.html", output / "summary.json", output / "inputs.json", *sorted(media.iterdir())]}
    receipt = {"schema": "agentic-evidence/v1", "run_id": run.name + "-review-" + generation,
               "claim": "Static diagnostic review generated from the named input snapshot; no scene acceptance or deployment claim",
               "produced_at": summary["produced_at"], "inputs_sha256": inputs_hash,
               "path": str(output / "index.html"), "sha256": artifacts["index.html"],
               "status": "stale_inputs" if changed else "generated", "artifacts": artifacts}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"review": str(output / "index.html"), "receipt": str(output / "receipt.json"),
                      "status": status, "registered": len(registered & train_names), "selected_training": len(train_names),
                      "selected_validation": summary["selected_validation"], "issues": audit.issues}, indent=2))


if __name__ == "__main__":
    main()
