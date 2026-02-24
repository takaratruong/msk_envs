"""
Open this in blender scripting and edit the path to your JSON file below.
"""
import bpy
import gzip
import json
import os
import sys
from pathlib import Path

cdir = os.path.dirname(bpy.data.filepath)
if not cdir in sys.path:
    sys.path.append(cdir)

from camera import setup_camera, update_camera
from muscle import update_muscle
from options import setup_renderer
from scene import reset_scene, hide_objects, setup_floor, setup_meter_markers, create_wr_line, update_wr_line
from visuals import update_visual, clear_visual_cache
from colliders import update_collider, clear_collider_cache
from targets import update_target, clear_target_cache

# Locate dashboard
script_dir = Path(__file__).resolve().parent.parent.parent
dashboard_src = os.path.join(script_dir, "dashboard")
json_path = os.path.join(dashboard_src, "trajectories/test/render.json.gz")  # Trajectory to render
plane_texture_path = os.path.join(dashboard_src, "assets/textures/plane.png")  # Additional assets


def main():
    # Load trajectory
    if json_path.endswith(".gz"):
        with gzip.open(json_path, 'rt') as f:
            stacked_frames = json.load(f)
    else:
        with open(json_path, 'r') as f:
            stacked_frames = json.load(f)

    if len(stacked_frames) > 1:
        fps = 1.0 / (stacked_frames[1]["time"] - stacked_frames[0]["time"])
    else:
        fps = 1.0 / stacked_frames[0]["time"]

    reset_scene()
    clear_visual_cache()
    clear_collider_cache()
    clear_target_cache()
    setup_renderer(fps=int(round(fps)))
    camera = setup_camera()
    hide_objects()

    setup_floor(plane_texture_path, size=100, location=(-50, 0, 0))
    setup_floor(plane_texture_path, size=100, location=(50, 0, 0))

    frame0 = stacked_frames[0]
    if "scene_settings" in frame0:
        scene_settings = frame0["scene_settings"]
        if scene_settings["meter_markers"]:
            setup_meter_markers()
        wr_line = None
    else:
        setup_meter_markers()
        wr_line = create_wr_line()

    # Create frames
    for frame_index, frame_data in enumerate(stacked_frames):
        time = frame_data["time"]
        frame_num = frame_index + 1  # (Blender frames start at 1)
        bpy.context.scene.frame_set(frame_num)

        cam_pos = frame_data["cam_pos"]
        update_camera(camera, cam_pos, frame_num)

        update_wr_line(wr_line, time, frame_num)

        for visual in frame_data["visuals"]:
            update_visual(visual, frame_num, dashboard_src)

        for collider in frame_data["colliders"]:
            collider_name = collider["name"]
            # Only draw the shoes
            if collider_name.startswith("left_") or collider_name.startswith("right_"):
                update_collider(collider, frame_num, dashboard_src)

        for muscle in frame_data["muscles"]:
            update_muscle(muscle, frame_num)

        if "targets" in frame_data:
            for i, target in enumerate(frame_data["targets"]):
                update_target(f"target_{i}", target, frame_num)

        print(f"Built frame: {frame_num} of {len(stacked_frames)}")

    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(stacked_frames)
    print(f"Animation created with {len(stacked_frames)} frames")
    return


if __name__ == "__main__":
    main()
