import bpy


def setup_renderer(fps):
    scene = bpy.context.scene

    scene.render.fps = fps
    scene.render.engine = "CYCLES"
    scene.render.motion_blur_shutter = 0.2

    if hasattr(scene.render, "motion_blur_position"):
        scene.render.motion_blur_position = 'CENTER'
    elif hasattr(scene.cycles, "motion_blur_position"):
        scene.cycles.motion_blur_position = 'CENTER'

    # quality settings
    scene.cycles.samples = 128
    scene.cycles.preview_samples = 32
    return
