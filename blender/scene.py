import bpy

from material import create_plane_material, create_wr_line_material


def clear_scene():
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Clear all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Clear all mesh data
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Clear all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # Clear all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Clear all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Clear all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Clear all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Clear all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)

    # Clear all actions
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)

    return


def setup_lighting():
    """ Creates a simple sun light """
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun_light = bpy.context.active_object
    sun_light.name = "SunLight"
    sun_light.data.energy = 5.0
    sun_light.data.angle = 0.0873  # 5 deg
    sun_light.rotation_euler = (0.785, 0, 0.785)
    return


def setup_sky():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    # Clear existing nodes
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    # Create nodes
    sky_tex = nodes.new("ShaderNodeTexSky")
    sky_tex.sun_disc = False
    sky_tex.sun_elevation = 0.785398  # 45 degrees in radians
    sky_tex.sun_rotation = 0.785398  # 45 degrees in radians
    sky_tex.altitude = 0.0

    if 'NISHITA' in sky_tex.bl_rna.properties['sky_type'].enum_items:
        sky_tex.sky_type = 'NISHITA'
        sky_tex.air_density = 0.6
        sky_tex.dust_density = 0.0
        sky_tex.ozone_density = 2.0
    else:
        sky_tex.sky_type = 'SINGLE_SCATTERING'
        sky_tex.air_density = 0.6
        sky_tex.aerosol_density = 0.0
        sky_tex.ozone_density = 2.0

    background = nodes.new("ShaderNodeBackground")
    background.inputs['Strength'].default_value = 0.025
    # Position nodes
    output = nodes.new("ShaderNodeOutputWorld")
    sky_tex.location = (-400, 0)
    background.location = (-200, 0)
    output.location = (0, 0)
    # Link nodes
    links.new(sky_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])


def setup_meter_markers():
    """
    Create 3d text markers every 10m
    """
    for i in range(0, 101, 10):
        bpy.ops.object.text_add(location=(i, 1.5, 0.05))
        text_obj = bpy.context.active_object
        text_obj.name = f"MeterMarker_{i}m"
        text_obj.data.body = f"{i}"
        text_obj.data.size = 1.0
        text_obj.data.extrude = 0.05
        text_obj.rotation_euler = (1.5708, 0, 0)
    return


def create_wr_line():
    """ create a world-record line object (yellow line) """
    bpy.ops.mesh.primitive_plane_add(size=0.1, location=(0, 0, 0.01))
    wr_line = bpy.context.active_object
    wr_line.name = "WorldRecordLine"
    wr_line.scale[0] = 0.5
    wr_line.scale[1] = 1000.0

    wr_material = create_wr_line_material()
    if wr_line.data.materials:
        wr_line.data.materials[0] = wr_material
    else:
        wr_line.data.materials.append(wr_material)
    return wr_line


def update_wr_line(wr_line, time, frame_num):
    # determine position of line based on 2.5m splits
    split_length = 2.5
    splits = [0.87, 0.41, 0.33, 0.29, 0.27, 0.25, 0.24, 0.23, 0.23, 0.23, 0.22, 0.22, 0.22, 0.22, 0.21, 0.21, 0.21,
              0.21, 0.21, 0.21, 0.21, 0.21, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21,
              0.21, 0.21, 0.21, 0.21]

    time = max(0.0, min(time, sum(splits)))

    distance, elapsed = 0.0, 0.0
    for split in splits:
        if time <= elapsed + split:  # within split
            frac = (time - elapsed) / split
            distance += frac * split_length
            break
        else:
            distance += split_length
            elapsed += split

    distance = min(distance, 100.0)
    wr_line.location.x = distance
    wr_line.keyframe_insert(data_path="location", frame=frame_num)
    return


def setup_floor(plane_texture_path, size, location):
    # Create the plane mesh
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    plane = bpy.context.active_object
    plane.name = "TexturedPlane"

    # Create and assign the material
    plane_material = create_plane_material(plane_texture_path)

    # Assign material to the plane
    if plane.data.materials:
        plane.data.materials[0] = plane_material
    else:
        plane.data.materials.append(plane_material)

    return plane


def hide_objects():
    for obj in bpy.data.objects:
        if obj.type not in ["CAMERA", "LIGHT"]:
            obj.hide_render = True
            obj.hide_viewport = True
    return


def reset_scene():
    clear_scene()
    setup_lighting()
    setup_sky()
    return
