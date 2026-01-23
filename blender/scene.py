import bpy

from material import create_plane_material


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
    sun_light.data.energy = 4.0
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
    sky_tex.sky_type = 'NISHITA'
    sky_tex.sun_disc = False
    sky_tex.sun_elevation = 0.785398  # 45 degrees in radians
    sky_tex.sun_rotation = 0.785398  # 45 degrees in radians
    sky_tex.altitude = 0.0
    sky_tex.air_density = 1.0
    sky_tex.dust_density = 0.0
    sky_tex.ozone_density = 6.0
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
