import bpy
import mathutils
from mathutils import Vector
import math

from util import vec_yup_zup
from material import create_muscle_material


def setup_activation_driver(obj, material):
    node_name = material.get("activation_value_node_name")
    if not node_name:
        return

    value_node = material.node_tree.nodes.get(node_name)
    if not value_node:
        return

    # Add driver
    driver = value_node.outputs[0].driver_add("default_value").driver
    driver.type = "AVERAGE"

    var = driver.variables.new()
    var.name = "activation"
    var.targets[0].id = obj
    var.targets[0].data_path = '["activation"]'

    driver.expression = "activation"


def update_capsule(capsule, point1, point2, activation):
    direction = Vector(point2) - Vector(point1)
    distance = direction.length
    capsule.location = (Vector(point1) + Vector(point2)) / 2
    capsule.scale[2] = distance

    # Rotate capsule to align with direction vector
    if abs(direction.z) < 0.999:  # Avoid gimbal lock
        z_axis = Vector((0, 0, 1))
        direction_normalized = direction.normalized()
        rotation_axis = z_axis.cross(direction_normalized)
        rotation_angle = math.acos(max(-1, min(1, z_axis.dot(direction_normalized))))
        if rotation_axis.length > 0.001:
            rotation_axis.normalize()
            rotation_quat = mathutils.Quaternion(rotation_axis, rotation_angle)
            capsule.rotation_mode = "QUATERNION"
            capsule.rotation_quaternion = rotation_quat
    else:
        capsule.rotation_mode = "QUATERNION"
        if direction.z < 0:
            capsule.rotation_quaternion = mathutils.Quaternion((0, 1, 0), math.pi)

    capsule["activation"] = activation
    return capsule


def setup_capsule_with_material(capsule):
    """
    Setup a muscle object with the animated material
    Call this once for each muscle object
    """
    # Create and assign material
    material = create_muscle_material()
    capsule.data.materials.clear()
    capsule.data.materials.append(material)

    # Setup the driver connection
    setup_activation_driver(capsule, material)

    # Initialize custom property
    capsule["activation"] = 0.0
    return material


def create_capsule_between_points(radius, name):
    print(f"Creating segment {name}")
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=1,  # we'll change this later
        location=(0, 0, 0)
    )
    capsule = bpy.context.active_object
    capsule.name = name
    capsule.hide_render = False
    capsule.hide_viewport = False

    setup_capsule_with_material(capsule)
    return capsule


def update_muscle(muscle, frame_num):
    name = muscle["name"]
    points = muscle.get("points", [])
    max_isometric_force = muscle["max_isometric_force"]
    activation = muscle["activation"]
    # radius = math.sqrt(max_isometric_force) / 8000
    radius = 0.005

    # build cylinder between points
    for i in range(len(points) - 1):
        point1 = vec_yup_zup(points[i])
        point2 = vec_yup_zup(points[i + 1])
        segment_name = f"{name}_segment_{i}"

        # Check if this segment already exists
        existing_segment = bpy.data.objects.get(segment_name)
        if existing_segment is None:  # Create new capsule segment
            capsule = create_capsule_between_points(
                radius=radius,
                name=segment_name
            )
        else:
            capsule = existing_segment

        update_capsule(capsule, point1, point2, activation)

        # keyframes
        capsule.keyframe_insert(data_path="location", frame=frame_num)
        capsule.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
        capsule.keyframe_insert(data_path="scale", frame=frame_num)
        capsule.keyframe_insert(data_path='["activation"]', frame=frame_num)
    return
