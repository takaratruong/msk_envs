import math

import bpy
import mathutils

from material import create_collider_material
from util import vec_yup_zup, quat_xyzw_to_wxyz

collider_material = create_collider_material()


def create_sphere(name):
    print(f"Creating sphere {name}")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1.0,
        location=(0, 0, 0)
    )
    sphere = bpy.context.active_object
    sphere.name = name
    sphere.hide_render = False
    sphere.hide_viewport = False
    sphere.data.materials.append(collider_material)
    return sphere


def create_collider(name, geom_type, scale):
    if geom_type == 0:  # plane
        collider = None  # todo: plane collider
    elif geom_type == 2:  # sphere
        collider = create_sphere(name)
    elif geom_type == 3:  # capsule
        collider = None  # todo: capsule collider
    else:
        collider = None  # todo: other collider types
    return collider


def update_collider(collider, frame_num, dashboard_src):
    # Fetch object data
    name = collider["name"]
    geom_type = collider["geom_type"]
    pos = vec_yup_zup(collider["pos"])
    scale = collider["scale"]
    # Rotation y-up to z-up
    quat_data = collider["rot"]
    rotation = mathutils.Quaternion(quat_xyzw_to_wxyz(quat_data))
    q_conv = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
    rotation = q_conv @ rotation

    # Update or create object
    existing_collider = bpy.data.objects.get(name)
    if existing_collider is None:
        collider = create_collider(name, geom_type, scale)
    else:
        collider = existing_collider

    if collider is not None:
        collider.location = pos
        collider.rotation_mode = 'QUATERNION'
        collider.rotation_quaternion = rotation
        collider.scale = scale

        collider.keyframe_insert(data_path="location", frame=frame_num)
        collider.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        collider.keyframe_insert(data_path="scale", frame=frame_num)

    return


def clear_collider_cache():
    global collider_material
    collider_material = create_collider_material()
