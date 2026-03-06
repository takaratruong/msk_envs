import math

import bpy
import mathutils

from material import create_target_material
from util import vec_yup_zup

active_target_material = create_target_material(active=True)
inactive_target_material = create_target_material(active=False)


def create_target(name, active):
    print(f"Creating target {name}")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=(0, 0, 0)
    )
    target = bpy.context.active_object
    target.name = name
    target.hide_render = False
    target.hide_viewport = False
    target.data.materials.append(active_target_material if active else inactive_target_material)
    bpy.ops.object.shade_smooth()
    return target


def update_target(name, target, frame_num):
    # Fetch object data
    pos = vec_yup_zup(target["pos"])
    radius = target["radius"]
    # Rotation y-up to z-up
    quat_data = target["rot"]
    rotation = mathutils.Quaternion([quat_data[0], quat_data[1], quat_data[2], quat_data[3]])
    q_conv = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
    rotation = q_conv @ rotation

    # Update or create object
    existing_collider = bpy.data.objects.get(name)
    if existing_collider is None:
        target = create_target(name, active=target["active"])
    else:
        target = existing_collider

    if target is not None:
        target.location = pos
        target.rotation_mode = 'QUATERNION'
        target.rotation_quaternion = rotation
        target.scale = (radius, radius, radius)

        target.keyframe_insert(data_path="location", frame=frame_num)
        target.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        target.keyframe_insert(data_path="scale", frame=frame_num)
    return


def clear_target_cache():
    global active_target_material, inactive_target_material
    active_target_material = create_target_material(active=True)
    inactive_target_material = create_target_material(active=False)
