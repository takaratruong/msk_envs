"""Render a foothold-rollout trajectory over its fitted natural terrain.

A variant of /home/ubuntu/bolt/render_trajectory.py (headless, under Xvfb)
that hides the stepping-stone boxes and instead draws the natural heightfield
fitted through those exact slab tops (terrain_for_trajectory.py). Physics is
untouched - the walker stepped on invisible slabs - but because the surface
passes through every slab top (max residual < 1 mm), each footfall lands on
the rendered terrain.

The slab boxes are simply skipped, not re-simulated. warp's OpenGL renderer
supports only one color per mesh instance, so a single terrain mesh reads as
a flat green sheet with no depth cues. Instead the triangles are split into
elevation bands (by mean vertex height) and each band is rendered as its own
mesh with a hypsometric tint (valley green -> grass -> tan -> ochre ->
grey-brown), alternate bands slightly darkened. The band boundaries act as
contour lines, making ridges/valleys and relative height readable. Backface
culling is disabled so the ribbon is visible from every camera elevation.

Usage:
    DISPLAY=:99 CUDA_VISIBLE_DEVICES=2 python \
        render_trajectory_natural_terrain.py <traj.json[.gz]> <out_frame_dir> \
        <terrain.npz> [--max-frames N] [--bands N] [--faceted-terrain]
"""
import argparse
import gzip
import json
import os

import numpy as np
import warp as wp
import warp.render
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("traj_path")
ap.add_argument("out_dir")
ap.add_argument("terrain_path")
ap.add_argument("--max-frames", type=int, default=0,
                help="render only the first N frames (0 = all)")
ap.add_argument("--bands", type=int, default=12,
                help="number of hypsometric elevation bands")
ap.add_argument("--faceted-terrain", action="store_true",
                help="flat-shade the terrain bands (default: smooth; faceted "
                     "was compared and reads as broken streaky ribbons on "
                     "this low-poly grid)")
args = ap.parse_args()
traj_path, out_dir, terrain_path = args.traj_path, args.out_dir, args.terrain_path
os.makedirs(out_dir, exist_ok=True)

opener = gzip.open if traj_path.endswith(".gz") else open
with opener(traj_path, "rt") as f:
    frames = json.load(f)
print(f"loaded {len(frames)} frames from {traj_path}", flush=True)

terrain = np.load(terrain_path)
gx, gy, Z = terrain["gx"], terrain["gy"], terrain["Z"]
ny, nx = Z.shape
# viz coords (x fwd, y lateral, z up) -> render coords (x fwd, y up, z lateral)
X, Ylat = np.meshgrid(gx, gy)
terrain_points = np.stack([X, Z, Ylat], -1).reshape(-1, 3).astype(np.float32)
iy, ix = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
v = (iy * nx + ix).ravel()
# Winding chosen so face normals point up (+Y): (A, A+nx, A+1), (A+1, A+nx, A+nx+1)
tris = np.stack(
    [v, v + nx, v + 1, v + 1, v + nx, v + nx + 1], -1
).reshape(-1, 3).astype(np.int32)
print(f"terrain mesh: {len(terrain_points)} vertices, {len(tris)} triangles", flush=True)

# --- Elevation banding: split triangles into height bands, one mesh each ---
# Hypsometric control points, dark valley-green -> grass -> tan -> ochre ->
# grey-brown summit.
_PALETTE = np.array([
    (0.25, 0.42, 0.22),
    (0.45, 0.58, 0.30),
    (0.72, 0.66, 0.40),
    (0.62, 0.48, 0.30),
    (0.55, 0.50, 0.45),
], dtype=np.float64)


def band_color(k, t):
    """t in [0, 1]: normalized elevation of the band's midpoint."""
    x = t * (len(_PALETTE) - 1)
    i = min(int(x), len(_PALETTE) - 2)
    c = _PALETTE[i] + (x - i) * (_PALETTE[i + 1] - _PALETTE[i])
    if k % 2 == 1:  # darken alternate bands so boundaries read as contours
        c = c * 0.88
    return tuple(c)


n_bands = max(args.bands, 1)
tri_h = terrain_points[tris, 1].mean(axis=1)  # mean vertex height (Y is up)
# Quantile edges spread the contour lines evenly over the surface actually
# seen (linear edges bunch most of a flat course into one or two bands), but
# the COLOR still follows true elevation so valleys stay green and only real
# summits go ochre/grey.
edges = np.quantile(tri_h, np.linspace(0.0, 1.0, n_bands + 1))
edges[-1] += 1e-6
tri_band = np.clip(np.digitize(tri_h, edges) - 1, 0, n_bands - 1)
h_lo, h_span = tri_h.min(), max(tri_h.max() - tri_h.min(), 1e-6)
terrain_bands = []  # (name, verts, flat indices, color)
for k in range(n_bands):
    band_tris = tris[tri_band == k]
    if len(band_tris) == 0:
        continue
    used, local = np.unique(band_tris, return_inverse=True)
    t_mid = (0.5 * (edges[k] + edges[k + 1]) - h_lo) / h_span
    terrain_bands.append((
        f"terrain_band_{k:02d}",
        terrain_points[used],
        local.astype(np.int32).reshape(-1),
        band_color(k, float(np.clip(t_mid, 0.0, 1.0))),
    ))
print(f"split into {len(terrain_bands)} elevation bands "
      f"(z {edges[0]:.2f}..{edges[-1]:.2f} m)", flush=True)

W, H = 1000, 800
renderer = warp.render.OpenGLRenderer(
    title="traj", vsync=False, up_axis="Y",
    screen_width=W, screen_height=H,
    camera_fov=45.0, draw_grid=False, draw_sky=True, draw_axis=False,
    enable_mouse_interaction=False, enable_keyboard_interaction=False,
    enable_backface_culling=False,
)

import pyvista as pv

GEOM_DIR = "/home/ubuntu/bolt/data/geometry"
_bone_cache = {}


def load_bone_mesh(mesh_file):
    if mesh_file in _bone_cache:
        return _bone_cache[mesh_file]
    path = os.path.join(GEOM_DIR, mesh_file)
    if not os.path.exists(path):
        _bone_cache[mesh_file] = None
        return None
    try:
        m = pv.read(path).triangulate()
        pts = np.asarray(m.points, dtype=np.float32)
        faces = m.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
        _bone_cache[mesh_file] = (pts, faces.reshape(-1))
    except Exception:
        _bone_cache[mesh_file] = None
    return _bone_cache[mesh_file]


def act_color(a):
    a = max(0.0, min(1.0, float(a)))
    return (0.15 + 0.85 * a, 0.15, 0.7 - 0.55 * a)  # blue->red with activation


if args.max_frames > 0:
    frames = frames[:args.max_frames]

saved = 0
for fi, fr in enumerate(frames):
    cam = fr.get("cam_pos", [3, 1.2, 3])
    tgt = fr["visuals"][0]["pos"] if fr.get("visuals") else [0, 1, 0]
    cp = (cam[0] + 2.5, cam[1] + 0.4, cam[2] + 2.5)
    renderer.update_view_matrix(
        cam_pos=cp, cam_front=(tgt[0] - cp[0], tgt[1] - cp[1], tgt[2] - cp[2])
    )
    renderer.begin_frame(fr.get("time", fi / 30.0))

    # The natural terrain replaces both the ground plane and the slab boxes.
    # Static geometry: registering it on the first frame is enough, the
    # instances persist across frames. One mesh per elevation band because
    # render_mesh takes a single flat color per instance.
    if fi == 0:
        for name, verts, idx, col in terrain_bands:
            renderer.render_mesh(
                name, verts, idx,
                colors=col, smooth_shading=not args.faceted_terrain,
            )

    for vi, vis in enumerate(fr.get("visuals", [])):
        mesh = load_bone_mesh(vis.get("mesh_file"))
        if mesh is None:
            continue
        pts, idx = mesh
        # Explicit bone color: warp assigns tab10 colors by shape index when
        # none is given, and replacing the slab boxes with the terrain mesh
        # shifts those indices (observed as a pink skeleton).
        renderer.render_mesh(
            f"bone_{vi}", pts, idx,
            pos=tuple(vis["pos"]), rot=tuple(vis["rot"]),
            scale=tuple(vis.get("scale", (1, 1, 1))),
            colors=(0.78, 0.76, 0.71), smooth_shading=True,
        )

    for mi, mus in enumerate(fr.get("muscles", [])):
        pts = mus.get("points", [])
        if len(pts) < 2:
            continue
        renderer.render_line_strip(
            f"mus_{mi}", np.array(pts, dtype=np.float32),
            color=act_color(mus.get("activation", 0.0)), radius=0.004,
        )

    renderer.end_frame()
    try:
        rh = getattr(renderer, "screen_height", H)
        rw = getattr(renderer, "screen_width", W)
        buf = wp.zeros((rh, rw, 3), dtype=wp.float32)
        renderer.get_pixels(buf, split_up_tiles=False, mode="rgb", use_uint8=False)
        arr = (buf.numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, f"frame_{fi:04d}.png"))
        saved += 1
    except Exception as e:
        if fi == 0:
            print("get_pixels failed:", repr(e), flush=True)
        break

print(f"SAVED {saved} frames to {out_dir}", flush=True)
