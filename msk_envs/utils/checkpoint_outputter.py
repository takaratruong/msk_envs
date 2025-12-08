import json

from msk_envs.utils.checkpoint_parser import FrameData


def track_com(frame_data: list[FrameData]):
    """
    Smoothly track the center of mass.
    For now instant tracking is probably fine
    """
    com_positions = []
    for frame in frame_data:
        com_positions.append(frame.kinetic_data.com)
    return com_positions


def create_animation_json(frame_data: list[FrameData], out_file: str):
    """ Dump all relevant animation data to json """
    n_frames = len(frame_data)

    # Where the camera(s) should look
    cam_positions = track_com(frame_data)

    # Get all visuals, colliders, muscles
    stacked_frames = []
    for i in range(n_frames):
        visuals = [obj.to_dict() for obj in frame_data[i].visuals]
        colliders = [obj.to_dict() for obj in frame_data[i].colliders]
        muscles = [muscle.to_dict() for muscle in frame_data[i].muscles]
        time = frame_data[i].time
        frame = {
            "time": time,
            "visuals": visuals,
            "colliders": colliders,
            "muscles": muscles,
            "cam_pos": list(cam_positions[i])
        }
        stacked_frames.append(frame)

    with open(out_file, 'w') as f:
        json.dump(stacked_frames, f, indent=2)
    return
