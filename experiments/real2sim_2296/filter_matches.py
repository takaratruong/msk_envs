#!/usr/bin/env python3
"""Filter an isolated COLMAP match cache without changing feature indices.

Example (run matchers BEFORE this command, then use incremental mapping only):
  python filter_matches.py --source-db /abs/database.db \
      --frames /abs/frames.json --masks /abs/source_masks \
      --output-db /abs/database_filtered.db

The source is opened read-only and copied with SQLite's online backup API,
including committed WAL content. Only matches/two_view_geometries are changed
in the copy. Original indexed keypoints, descriptors, cameras, and all other
tables are hashed before/after and must remain byte-for-byte logically equal.
Rejected pairs disappear from BOTH match tables. Retained raw matches and
verified inliers receive the same static-pixel mask; no geometry is refitted.

Nonlocal pairs (default >2 seconds apart) require verified-inlier convex hull
coverage >=4% and >=4 occupied cells of a 4x4 image grid in BOTH images. Local
pairs still require valid static inliers but are exempt from spatial coverage.
This is a conservative correspondence heuristic, not proof of correctness.
Pairwise one-to-many matches are reported; optional removal never selects an
arbitrary winner. Transitive track conflicts are audited separately, not used
to delete entire components by default. --drop-transitive-conflicts is an
explicit conservative experiment: it removes every correspondence touching any
feature in a conflicting component, including potentially correct observations
merged by one bad link. Its lost observations and graph changes are reported.

Outputs next to output-db: .pairs.jsonl (every cached edge), .report.json,
.masks.json, .track_conflicts.json, and .seed_candidates.json. Existing output
files are never overwritten. A failed assertion leaves only a named partial
copy; the requested output-db is published after all checks pass. Do not run
feature extraction, matching, or geometric verification on the filtered DB:
those steps would reintroduce removed edges.

Format reference: https://colmap.github.io/database.html
Pinned API reference:
https://github.com/colmap/colmap/blob/4.2.0/src/colmap/scene/database.h
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import time

import cv2
import numpy as np


PAIR_BASE = 2147483647
KNOWN_BAD = [("frame_002118.jpg", "frame_001582.jpg"), ("frame_002137.jpg", "frame_000664.jpg")]
WATCHED = KNOWN_BAD + [("frame_002145.jpg", "frame_002160.jpg"), ("frame_000906.jpg", "frame_001507.jpg")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def table_hash(db, table):
    # Table names originate in sqlite_master, not command-line SQL fragments.
    quoted = '"' + table.replace('"', '""') + '"'
    h = hashlib.sha256()
    columns = db.execute(f"PRAGMA table_info({quoted})").fetchall()
    primary = [r[1] for r in sorted(columns, key=lambda r: r[5]) if r[5]]
    order = ",".join('"' + n.replace('"', '""') + '"' for n in primary) or "rowid"
    h.update(json.dumps(columns).encode())
    for row in db.execute(f"SELECT * FROM {quoted} ORDER BY {order}"):
        for value in row:
            if isinstance(value, bytes):
                encoded = b"B" + value
            else:
                encoded = b"J" + json.dumps(value, allow_nan=False).encode()
            h.update(struct.pack("<Q", len(encoded)))
            h.update(encoded)
    return h.hexdigest()


def matrix(rows, cols, blob, dtype, label):
    if rows == 0:
        return np.empty((0, cols), dtype=dtype)
    data = np.frombuffer(blob, dtype=dtype)
    if data.size != rows * cols:
        raise ValueError(f"Malformed {label} blob")
    return data.reshape(rows, cols)


def correspondences(record, label):
    if record is None:
        return np.empty((0, 2), dtype=np.uint32)
    rows, cols, blob = record
    if cols != 2:
        raise ValueError(f"{label} must have exactly two feature-index columns")
    return matrix(rows, cols, blob, np.uint32, label)


def filter_indices(matches, left, right, ambiguity_policy):
    if not len(matches):
        return matches, {"masked_or_outside": 0, "duplicate_pairs": 0, "ambiguous_left_features": 0,
                         "ambiguous_right_features": 0, "ambiguous_correspondences": 0}
    if int(matches[:, 0].max()) >= len(left) or int(matches[:, 1].max()) >= len(right):
        raise ValueError("Match references a nonexistent keypoint; refusing a silently corrupted cache")
    keep = left[matches[:, 0]] & right[matches[:, 1]]
    valid = matches[keep]
    _, first = np.unique(valid, axis=0, return_index=True)
    unique = valid[np.sort(first)]
    ac, bc = Counter(unique[:, 0].tolist()), Counter(unique[:, 1].tolist())
    ambiguous = np.asarray([ac[int(a)] > 1 or bc[int(b)] > 1 for a, b in unique], dtype=bool)
    diagnostic = {"masked_or_outside": int((~keep).sum()), "duplicate_pairs": len(valid) - len(unique),
                  "ambiguous_left_features": sum(v > 1 for v in ac.values()),
                  "ambiguous_right_features": sum(v > 1 for v in bc.values()),
                  "ambiguous_correspondences": int(ambiguous.sum())}
    return unique[~ambiguous] if ambiguity_policy == "drop" else unique, diagnostic


def support(xy, width, height, grid_size, cell_min_points):
    if not len(xy):
        return {"hull_fraction": 0.0, "occupied_cells": 0, "supported_cells": 0,
                "cell_counts": [0] * grid_size**2, "extent_fraction_xy": [0.0, 0.0],
                "bbox_fraction_xyxy": None, "bbox_area_fraction": 0.0}
    normalized = xy.astype(np.float32) / np.array([width, height], np.float32)
    low, high = normalized.min(axis=0), normalized.max(axis=0)
    hull = float(cv2.contourArea(cv2.convexHull(normalized))) if len(xy) >= 3 else 0.0
    cells = np.clip((normalized * grid_size).astype(int), 0, grid_size - 1)
    counts = np.bincount(cells[:, 1] * grid_size + cells[:, 0], minlength=grid_size**2)
    return {"hull_fraction": hull, "occupied_cells": int((counts > 0).sum()),
            "supported_cells": int((counts >= cell_min_points).sum()), "cell_counts": counts.tolist(),
            "extent_fraction_xy": (high - low).tolist(), "bbox_fraction_xyxy": [*low.tolist(), *high.tolist()],
            "bbox_area_fraction": float(np.prod(high - low))}


def components(ids, edges, names):
    adjacent = {i: set() for i in ids}
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    unseen, result = set(ids), []
    while unseen:
        stack, group = [min(unseen)], []
        unseen.remove(stack[0])
        while stack:
            node = stack.pop()
            group.append(node)
            for other in adjacent[node] & unseen:
                unseen.remove(other)
                stack.append(other)
        result.append(sorted(group))
    result.sort(key=lambda g: (-len(g), g[0]))
    return {"component_count": len(result), "component_sizes": list(map(len, result)),
            "components": [[names[i] for i in group] for group in result],
            "isolated_names": [names[i] for i in ids if not adjacent[i]],
            "degrees": {names[i]: len(adjacent[i]) for i in sorted(ids)}}


def track_audit(retained, keypoints, images, distance, max_matches, return_bad_features=False):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    total = sum(len(m) for _, _, m in retained)
    if total > max_matches:
        if return_bad_features:
            raise RuntimeError("Transitive deletion requires a completed audit; raise --track-audit-max-matches")
        return {"status": "skipped_resource_guard", "correspondences": total, "maximum": max_matches}
    image_ids = sorted(keypoints)
    offsets, n = {}, 0
    for iid in image_ids:
        offsets[iid] = n
        n += len(keypoints[iid])
    if total == 0:
        summary = {"status": "available", "correspondences": 0, "conflicting_component_count": 0, "examples": []}
        return (summary, {i: np.zeros(len(k), dtype=bool) for i, k in keypoints.items()}) if return_bad_features else summary
    a = np.concatenate([m[:, 0].astype(np.int64) + offsets[i] for i, _, m in retained])
    b = np.concatenate([m[:, 1].astype(np.int64) + offsets[j] for _, j, m in retained])
    used, inverse = np.unique(np.concatenate((a, b)), return_inverse=True)
    graph = coo_matrix((np.ones(total, dtype=np.uint8), (inverse[:total], inverse[total:])),
                       shape=(len(used), len(used))).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    conflicts, affected = [], set()
    for iid in image_ids:
        lo, hi = np.searchsorted(used, [offsets[iid], offsets[iid] + len(keypoints[iid])])
        local_labels, local_indices = labels[lo:hi], used[lo:hi] - offsets[iid]
        order = np.argsort(local_labels, kind="stable")
        local_labels, local_indices = local_labels[order], local_indices[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(local_labels)) + 1, len(local_labels)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop - start < 2:
                continue
            indices = local_indices[start:stop]
            xy = keypoints[iid][indices]
            diagonal = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
            if diagonal <= distance:
                continue  # Multiple SIFT orientations at the same location are not repeated-object evidence.
            hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
            separation = float(np.linalg.norm(hull[:, None] - hull[None, :], axis=-1).max())
            if separation <= distance:
                continue
            component = int(local_labels[start])
            affected.add(component)
            conflicts.append({"component": component, "image": images[iid]["name"],
                              "keypoint_indices": indices.tolist(), "bbox_diagonal_px": diagonal,
                              "max_pairwise_separation_px": separation,
                              "xy": xy.tolist()})
    conflicts.sort(key=lambda r: -r["max_pairwise_separation_px"])
    summary = {"status": "available", "correspondences": total, "used_feature_nodes": len(used),
            "conflicting_component_count": len(affected), "conflicting_image_component_groups": len(conflicts),
            "strict_minimum_pairwise_separation_px": distance, "examples": conflicts[:100],
            "interpretation": "A transitive component contains separated feature indices in the same image. This reveals ambiguity but does not identify which edge is wrong."}
    if not return_bad_features:
        return summary
    flagged = np.isin(labels, np.fromiter(affected, dtype=labels.dtype))
    bad_features = {}
    for iid in image_ids:
        lo, hi = np.searchsorted(used, [offsets[iid], offsets[iid] + len(keypoints[iid])])
        bad_features[iid] = np.zeros(len(keypoints[iid]), dtype=bool)
        bad_features[iid][used[lo:hi][flagged[lo:hi]] - offsets[iid]] = True
    summary["features_in_conflicting_components"] = int(flagged.sum())
    return summary, bad_features


def accumulate(counts, row):
    counts["pairs_before"] += 1
    counts["pairs_retained" if row["kept"] else "pairs_rejected"] += 1
    counts["verified_before"] += row["verified_before"]
    counts["verified_retained"] += row["verified_retained"]
    counts["raw_ambiguous_correspondences"] += row["raw_filter"]["ambiguous_correspondences"]
    counts["verified_ambiguous_correspondences"] += row["inlier_filter"]["ambiguous_correspondences"]
    for reason in row["reasons"]:
        counts["rejected_" + reason] += 1


def drop_conflicting_components(db, pairs_path, retained, keypoints, images, args):
    audit, bad = track_audit(retained, keypoints, images, args.ambiguity_distance_px,
                            args.track_audit_max_matches, return_bad_features=True)
    staged_pairs = pairs_path.with_name(pairs_path.name + f".partial.{os.getpid()}")
    counts, final_edges, final_retained, final_seeds, watched = Counter(), [], [], [], {}
    watched_keys = {tuple(sorted(p)) for p in WATCHED + args.assert_drop + args.assert_retain}
    raw_removed, inliers_removed, lost_pairs, raw_before, inliers_before = 0, 0, 0, 0, 0
    with open(pairs_path) as source_rows, open(staged_pairs, "x") as destination:
        for line in source_rows:
            row = json.loads(line)
            row["kept_before_transitive_experiment"] = row["kept"]
            if row["kept"]:
                a, b = row["image_ids"]
                pair_id = row["pair_id"]
                raw = correspondences(db.execute("SELECT rows,cols,data FROM matches WHERE pair_id=?", (pair_id,)).fetchone(), "matches")
                verified = correspondences(db.execute("SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone(), "two_view_geometries")
                raw_before += len(raw)
                inliers_before += len(verified)
                clean_raw = raw[~bad[a][raw[:, 0]] & ~bad[b][raw[:, 1]]]
                clean = verified[~bad[a][verified[:, 0]] & ~bad[b][verified[:, 1]]]
                row["raw_removed_by_transitive_component"] = len(raw) - len(clean_raw)
                row["inliers_removed_by_transitive_component"] = len(verified) - len(clean)
                raw_removed += len(raw) - len(clean_raw)
                inliers_removed += len(verified) - len(clean)
                row["support_before_transitive_experiment"] = row["support"]
                row["support"] = [support(keypoints[i][clean[:, col]], images[i]["width"], images[i]["height"],
                                          args.grid_size, args.cell_min_points) for col, i in enumerate((a, b))]
                row["min_hull_fraction"] = min(s["hull_fraction"] for s in row["support"])
                row["min_supported_grid_cells"] = min(s["supported_cells"] for s in row["support"])
                row["spatial_policy_would_reject_nonlocal"] = (
                    row["min_hull_fraction"] < args.min_hull_fraction or row["min_supported_grid_cells"] < args.min_grid_cells)
                if len(clean) < args.min_inliers:
                    row["reasons"].append("insufficient_inliers_after_transitive_experiment")
                if not row["local"] and row["spatial_policy_would_reject_nonlocal"]:
                    row["reasons"].append("nonlocal_concentrated_support_after_transitive_experiment")
                row["kept"] = not row["reasons"]
                row["verified_after_transitive_component_removal"] = len(clean)
                row["verified_retained"] = len(clean) if row["kept"] else 0
                row["raw_retained"] = len(clean_raw) if row["kept"] else 0
                if row["kept"]:
                    db.execute("UPDATE matches SET rows=?,data=? WHERE pair_id=?", (len(clean_raw), clean_raw.tobytes(), pair_id))
                    db.execute("UPDATE two_view_geometries SET rows=?,data=? WHERE pair_id=?", (len(clean), clean.tobytes(), pair_id))
                    final_edges.append((a, b))
                    final_retained.append((a, b, clean))
                    if len(clean) >= args.seed_min_inliers and row["gap_seconds"] >= args.seed_min_seconds:
                        final_seeds.append(row)
                else:
                    lost_pairs += 1
                    db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
                    db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
            if tuple(sorted(row["names"])) in watched_keys:
                watched[tuple(sorted(row["names"]))] = row
            accumulate(counts, row)
            destination.write(json.dumps(row, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(staged_pairs, pairs_path)
    experiment = {"enabled": True, "classification": "conservative experiment, not proof of correctness",
                  "audit_before": audit, "raw_matches_before": raw_before, "inliers_before": inliers_before,
                  "raw_matches_removed_by_component": raw_removed, "inliers_removed_by_component": inliers_removed,
                  "inlier_fraction_removed_by_component": inliers_removed / max(1, inliers_before),
                  "additional_pairs_rejected_after_removal": lost_pairs,
                  "inliers_retained_after_regating": counts["verified_retained"],
                  "caveat": "All features in every conflicting component are removed, including correct observations potentially merged by one wrong correspondence. Remaining edges are rechecked for static support."}
    return counts, final_edges, final_retained, final_seeds, watched, experiment


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-db", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output-db", type=Path, required=True)
    p.add_argument("--local-seconds", type=float, default=2.0)
    p.add_argument("--min-hull-fraction", type=float, default=.04)
    p.add_argument("--min-grid-cells", type=int, default=4)
    p.add_argument("--grid-size", type=int, default=4)
    p.add_argument("--cell-min-points", type=int, default=1)
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--ambiguity-policy", choices=("report", "drop"), default="report")
    p.add_argument("--allow-missing-masks", action="store_true")
    p.add_argument("--skip-track-audit", action="store_true")
    p.add_argument("--drop-transitive-conflicts", action="store_true", help="Conservative experiment: remove every feature in conflicting track components, then recheck edges")
    p.add_argument("--track-audit-max-matches", type=int, default=3000000)
    p.add_argument("--ambiguity-distance-px", type=float, default=20.0)
    p.add_argument("--seed-min-inliers", type=int, default=200)
    p.add_argument("--seed-min-seconds", type=float, default=5.0)
    p.add_argument("--assert-drop", nargs=2, action="append", default=[])
    p.add_argument("--assert-retain", nargs=2, action="append", default=[])
    args = p.parse_args()
    if not all(math.isfinite(v) for v in (args.local_seconds, args.min_hull_fraction,
                                         args.seed_min_seconds, args.ambiguity_distance_px)) or args.ambiguity_distance_px <= 0:
        p.error("Floating-point thresholds must be finite and ambiguity distance strictly positive")
    if not 0 <= args.min_hull_fraction <= 1 or args.local_seconds < 0 or args.seed_min_seconds < 0:
        p.error("Hull fraction must be in [0,1] and time thresholds nonnegative")
    if min(args.grid_size, args.cell_min_points, args.min_inliers, args.seed_min_inliers,
           args.track_audit_max_matches) < 1 or not 1 <= args.min_grid_cells <= args.grid_size**2:
        p.error("Invalid positive integer threshold or grid cell count")
    if args.drop_transitive_conflicts and args.skip_track_audit:
        p.error("--drop-transitive-conflicts requires the track audit")
    return args


def main():
    args = arguments()
    implementation_sha256 = sha256(__file__)
    source, output, frames_path = args.source_db.resolve(), args.output_db.resolve(), args.frames.resolve()
    if source == output or not source.is_file():
        raise ValueError("Source must exist and output must be a different, fresh path")
    paths = {k: output.with_suffix(output.suffix + suffix) for k, suffix in {
        "pairs": ".pairs.jsonl", "report": ".report.json", "masks": ".masks.json",
        "tracks": ".track_conflicts.json", "seeds": ".seed_candidates.json"}.items()}
    if output.exists() or any(p.exists() for p in paths.values()):
        raise FileExistsError("Output DB or diagnostics already exist; choose a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".partial.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    started = time.monotonic()
    frames_bytes = frames_path.read_bytes()
    frames_sha256 = hashlib.sha256(frames_bytes).hexdigest()
    frames = json.loads(frames_bytes)
    byname = {r["name"]: r for r in frames["frames"]}
    if len(byname) != len(frames["frames"]):
        raise ValueError("Duplicate frame names")
    source_files = {}
    for suffix in ("", "-wal"):
        path = Path(str(source) + suffix)
        if path.exists():
            source_files[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=60)
    db = sqlite3.connect(temporary, timeout=60)
    try:
        src.backup(db)
    finally:
        src.close()
    db.execute("PRAGMA journal_mode=DELETE")
    db.commit()
    snapshot_hash = sha256(temporary)
    untouched_tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                        if r[0] not in ("matches", "two_view_geometries")]
    original_hashes = {name: table_hash(db, name) for name in untouched_tables}
    cameras = {i: (w, h) for i, w, h in db.execute("SELECT camera_id,width,height FROM cameras")}
    images = {i: {"name": name, "camera": camera, "width": cameras[camera][0], "height": cameras[camera][1]}
              for i, name, camera in db.execute("SELECT image_id,name,camera_id FROM images")}
    names = {i: v["name"] for i, v in images.items()}
    train_ids = {i for i, im in images.items() if byname.get(im["name"], {}).get("split") == "train"}
    missing_train_names = sorted(name for name, f in byname.items() if f["split"] == "train" and name not in set(names.values()))
    keypoints, valid, mask_reports = {}, {}, []
    for iid, rows, cols, blob in db.execute("SELECT image_id,rows,cols,data FROM keypoints"):
        xy = matrix(rows, cols, blob, np.float32, "keypoints")[:, :2].copy()
        if xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError(f"Malformed keypoints in image {iid}")
        im = images[iid]
        w, h = im["width"], im["height"]
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        mask_path = (args.masks / (im["name"] + ".png")).resolve()
        report = {"image": im["name"], "mask_path": str(mask_path), "features": rows}
        if mask_path.is_file():
            mask_bytes = mask_path.read_bytes()
            mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (h, w) or not np.isin(mask, [0, 255]).all():
                raise ValueError(f"Mask must be a matching {w}x{h} binary 0/255 image: {mask_path}")
            indices = np.where(keep)[0]
            pixels = np.floor(xy[indices]).astype(int)  # COLMAP pixel centers use .5 offsets.
            keep[indices] &= mask[pixels[:, 1], pixels[:, 0]] == 255
            report.update(sha256=hashlib.sha256(mask_bytes).hexdigest(), valid_pixel_fraction=float((mask == 255).mean()), status="applied")
        elif not args.allow_missing_masks:
            raise FileNotFoundError(mask_path)
        else:
            report.update(sha256=None, status="missing_all_in_bounds_features_assumed_static")
        report["valid_features"] = int(keep.sum())
        mask_reports.append(report)
        keypoints[iid], valid[iid] = xy, keep
    pair_ids = [r[0] for r in db.execute("SELECT pair_id FROM matches UNION SELECT pair_id FROM two_view_geometries ORDER BY pair_id")]
    counts, before_edges, after_edges, retained, seeds, watched = Counter(), [], [], [], [], {}
    watched_keys = {tuple(sorted(p)) for p in WATCHED + args.assert_drop + args.assert_retain}
    db.execute("BEGIN IMMEDIATE")
    with open(paths["pairs"], "x") as diagnostics:
        for number, pair_id in enumerate(pair_ids):
            a, b = divmod(pair_id, PAIR_BASE)
            if a not in images or b not in images or a not in keypoints or b not in keypoints:
                raise ValueError(f"Pair {pair_id} references a missing image or keypoint record")
            raw_record = db.execute("SELECT rows,cols,data FROM matches WHERE pair_id=?", (pair_id,)).fetchone()
            geometry = db.execute("SELECT rows,cols,data,config FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
            raw = correspondences(raw_record, "matches")
            inliers = correspondences(geometry[:3] if geometry else None, "two_view_geometries")
            filtered_raw, raw_diag = filter_indices(raw, valid[a], valid[b], args.ambiguity_policy)
            filtered, inlier_diag = filter_indices(inliers, valid[a], valid[b], args.ambiguity_policy)
            spatial = [support(keypoints[i][filtered[:, col]], images[i]["width"], images[i]["height"],
                               args.grid_size, args.cell_min_points) for col, i in enumerate((a, b))]
            fa, fb = byname.get(names[a]), byname.get(names[b])
            ta, tb = (fa.get("time_seconds") if fa else None), (fb.get("time_seconds") if fb else None)
            gap = abs(float(ta) - float(tb)) if ta is not None and tb is not None else None
            if gap is not None and not math.isfinite(gap):
                raise ValueError("Nonfinite source timestamp")
            local = gap is not None and gap <= args.local_seconds
            minimum_hull = min(r["hull_fraction"] for r in spatial)
            minimum_cells = min(r["supported_cells"] for r in spatial)
            spatial_reject = minimum_hull < args.min_hull_fraction or minimum_cells < args.min_grid_cells
            train_only = a in train_ids and b in train_ids
            reasons = []
            if not train_only:
                reasons.append("not_both_training")
            if gap is None:
                reasons.append("missing_source_timestamp")
            if len(filtered) < args.min_inliers:
                reasons.append("insufficient_static_inliers")
            if not local and spatial_reject:
                reasons.append("nonlocal_concentrated_support")
            if geometry is None or geometry[3] == 0:
                reasons.append("no_verified_geometry")
            if raw_record is None:
                reasons.append("no_raw_match_record")
            keep = not reasons
            if train_only and len(inliers) >= args.min_inliers:
                before_edges.append((a, b))
            if keep:
                db.execute("UPDATE matches SET rows=?,cols=2,data=? WHERE pair_id=?",
                           (len(filtered_raw), filtered_raw.astype(np.uint32).tobytes(), pair_id))
                db.execute("UPDATE two_view_geometries SET rows=?,cols=2,data=? WHERE pair_id=?",
                           (len(filtered), filtered.astype(np.uint32).tobytes(), pair_id))
                after_edges.append((a, b))
                retained.append((a, b, filtered))
            else:
                db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
                db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
            row = {"pair_id": pair_id, "image_ids": [a, b], "names": [names[a], names[b]],
                   "time_seconds": [ta, tb], "gap_seconds": gap, "local": local, "train_only": train_only,
                   "raw_before": len(raw), "raw_after_mask": len(filtered_raw),
                   "verified_before": len(inliers), "verified_after_mask": len(filtered),
                   "verified_retained": len(filtered) if keep else 0, "kept": keep, "reasons": reasons,
                   "raw_filter": raw_diag, "inlier_filter": inlier_diag, "support": spatial,
                   "min_hull_fraction": minimum_hull, "min_supported_grid_cells": minimum_cells,
                   "spatial_policy_would_reject_nonlocal": spatial_reject,
                   "source_geometry_config": int(geometry[3]) if geometry else None}
            diagnostics.write(json.dumps(row, allow_nan=False) + "\n")
            accumulate(counts, row)
            if tuple(sorted(row["names"])) in watched_keys:
                watched[tuple(sorted(row["names"]))] = row
            if keep and len(filtered) >= args.seed_min_inliers and gap >= args.seed_min_seconds:
                seeds.append(row)
            if number % 5000 == 0:
                print(f"filtered {number + 1}/{len(pair_ids)} pairs; retained {counts['pairs_retained']}", flush=True)
        diagnostics.flush()
        os.fsync(diagnostics.fileno())
    pre_experiment_graph = components(train_ids, after_edges, names)
    pre_experiment_counts = dict(counts)
    experiment = {"enabled": False}
    if args.drop_transitive_conflicts:
        counts, after_edges, retained, seeds, watched, experiment = drop_conflicting_components(
            db, paths["pairs"], retained, keypoints, images, args)
    assertions = []
    for pair in KNOWN_BAD + args.assert_drop:
        row = watched.get(tuple(sorted(pair)))
        if row and row["kept"]:
            raise AssertionError(f"Known bad pair survived: {pair}")
        assertions.append({"kind": "drop", "names": pair,
                           "status": "passed_removed" if row else "already_absent_in_source",
                           "before_verified": row["verified_before"] if row else 0,
                           "reasons": row["reasons"] if row else [],
                           "spatial_policy_would_reject_nonlocal": row["spatial_policy_would_reject_nonlocal"] if row else None})
    for pair in args.assert_retain:
        row = watched.get(tuple(sorted(pair)))
        if not row or not row["kept"]:
            raise AssertionError(f"Requested positive-control pair did not survive: {pair}")
        assertions.append({"kind": "retain", "names": pair, "status": "passed"})
    db.commit()
    unchanged_hashes = {name: table_hash(db, name) for name in untouched_tables}
    if unchanged_hashes != original_hashes:
        raise AssertionError("A non-match table changed, violating keypoint/descriptor index preservation")
    integrity = db.execute("PRAGMA integrity_check").fetchall()
    if integrity != [("ok",)]:
        raise AssertionError(f"SQLite integrity check failed: {integrity}")
    db.close()
    seeds.sort(key=lambda r: (r["min_hull_fraction"] >= .10, r["min_hull_fraction"], r["verified_retained"]), reverse=True)
    audit = {"status": "disabled"} if args.skip_track_audit else track_audit(
        retained, keypoints, images, args.ambiguity_distance_px, args.track_audit_max_matches)
    write_json(paths["tracks"], audit)
    write_json(paths["masks"], {"masks": mask_reports, "missing_masks_allowed": args.allow_missing_masks})
    write_json(paths["seeds"], {"ranking": "prefer min hull >=0.10, then descending min hull, then inlier count",
               "min_inliers": args.seed_min_inliers, "min_gap_seconds": args.seed_min_seconds,
               "candidate_count": len(seeds), "candidates": seeds[:20],
               "caveat": "Broad support proposes candidates; inspect the two source images before choosing a seed."})
    report = {"schema": "real2sim-colmap-match-filter/v1", "produced_at": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": implementation_sha256, "source_db": str(source), "source_files_before_backup": source_files,
              "source_database_snapshot_sha256": snapshot_hash,
              "source_snapshot_note": "The standalone SQLite backup hash is authoritative for the exact copied logical source, including committed WAL content. Filesystem hashes describe files observed before backup and may differ if the source writer was active.",
              "frames_path": str(frames_path), "frames_sha256": frames_sha256,
              "source_video_sha256": frames.get("source_sha256"), "output_db": str(output),
              "output_db_sha256": sha256(temporary), "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "counts": dict(counts), "unchanged_table_sha256": unchanged_hashes,
              "transitive_conflict_experiment": experiment,
              "counts_before_transitive_experiment": pre_experiment_counts,
              "graph_before_transitive_experiment": pre_experiment_graph,
              "keypoint_descriptor_indices_preserved": True, "geometry_refitted": False,
              "graph_before": components(train_ids, before_edges, names),
              "graph_after": components(train_ids, after_edges, names), "missing_train_names_in_cache": missing_train_names,
              "assertions": assertions, "watched_pairs": list(watched.values()),
              "track_audit_summary": {k: v for k, v in audit.items() if k != "examples"},
              "seed_candidate_count": len(seeds), "elapsed_seconds": time.monotonic() - started,
              "artifacts": {k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items() if k != "report"},
              "next_stage": "incremental_mapping_only; never rerun matching on this filtered database",
              "limitations": ["Spatial spread and temporal locality cannot prove correspondence correctness.",
                              "Local repeated patterns, reflections, and undetected dynamic pixels can remain.",
                              "A connected image graph does not imply a geometrically well-constrained map.",
                              "Pair geometries remain the source estimates; only their observation subsets are reduced."]}
    write_json(paths["report"], report)
    os.replace(temporary, output)
    print(json.dumps({"output_db": str(output), "report": str(paths["report"]), "counts": dict(counts),
                      "components_before": report["graph_before"]["component_sizes"],
                      "components_after": report["graph_after"]["component_sizes"],
                      "seed_candidates": len(seeds), "known_bad_assertions": assertions,
                      "track_conflicts": audit.get("conflicting_component_count")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
