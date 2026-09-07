"""Render a foothold-rollout trajectory over its fitted natural terrain.

A variant of /home/ubuntu/bolt/render_trajectory.py (headless, under Xvfb)
that hides the stepping-stone boxes and instead draws the natural heightfield
fitted through those exact slab tops (terrain_for_trajectory.py). Physics is
untouched - the walker stepped on invisible slabs - but because the surface
passes through every slab top (max residual < 1 mm), each footfall lands on
the rendered terrain.

The slab boxes are simply skipped, not re-simulated: warp's OpenGL renderer
supports only one color per mesh instance, so the terrain is a single
earth-green with smooth shading. Backface culling is disabled so the ribbon
is visible from every camera elevation.

Usage:
    DISPLAY=:99 CUDA_VISIBLE_DEVICES=2 python \
        render_trajectory_natural_terrain.py <traj.json[.gz]> <out_frame_dir> \
        <terrain.npz>
"""
import gzip
import json
import os
import sys

import numpy as np
import warp as wp
import warp.render
from PIL import Image

traj_path, out_dir, terrain_path = sys.argv[1], sys.argv[2], sys.argv[3]
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
terrain_indices = tris.reshape(-1)
print(f"terrain mesh: {len(terrain_points)} vertices, {len(tris)} triangles", flush=True)

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
    # instance persists across frames.
    if fi == 0:
        renderer.render_mesh(
            "terrain", terrain_points, terrain_indices,
            colors=(0.42, 0.54, 0.32), smooth_shading=True,
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
