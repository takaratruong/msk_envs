import bpy
from mathutils import Vector
from util import vec_yup_zup


def setup_camera():
    bpy.ops.object.camera_add(location=(0, -10, 5))
    camera = bpy.context.active_object
    camera.name = "AnimationCamera"

    bpy.context.scene.camera = camera
    camera.data.lens = 50  # 50mm
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000
    return camera


def point_camera_at_target(camera, target_location):
    direction = Vector(target_location) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def update_camera(camera, camera_pos, frame_idx):
    camera_pos = vec_yup_zup(camera_pos)

    camera_pos[2] = 1

    # side cam
    # camera.location = [camera_pos[0] + 1.0, camera_pos[1] - 7.5, camera_pos[2] + 1.0]

    # olympic cam
    # camera.location = [0.8 * (camera_pos[0] + 10.0), camera_pos[1] - 10.0, camera_pos[2] + 3.0]

    # front view
    camera.location = [camera_pos[0] + 7.5, camera_pos[1], camera_pos[2] + 1.0]

    look_at_target = [camera_pos[0], camera_pos[1], camera_pos[2]]
    point_camera_at_target(camera, look_at_target)

    # keyframes
    camera.keyframe_insert(data_path="location", frame=frame_idx)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    return
