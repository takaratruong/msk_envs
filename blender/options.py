import bpy


def setup_renderer(fps):
    bpy.context.scene.render.fps = fps

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.motion_blur_shutter = 0.15
    bpy.context.scene.cycles.motion_blur_position = "CENTER"
    # quality settings
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.preview_samples = 32
    return
