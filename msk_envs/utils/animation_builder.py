import gzip
import json
import numpy as np

from msk_envs.utils.frame_parser import FrameData


def track_com(frame_data: list[FrameData]):
    """
    Smoothly track the center of mass.
    For now instant tracking is probably fine
    """
    com_positions = []
    for frame in frame_data:
        com_positions.append(frame.kinetic_data.com)
    return com_positions


def track_com_smooth_window(frame_data: list['FrameData'], window_size: int = 10):
    com_positions = []
    num_frames = len(frame_data)
    half_window = window_size // 2
    for i in range(num_frames):
        start_idx = max(0, i - half_window)
        end_idx = min(num_frames, i + half_window + 1)
        window_coms = [
            np.array(frame_data[j].kinetic_data.com)
            for j in range(start_idx, end_idx)
        ]
        avg_pos = sum(window_coms) / len(window_coms)

        com_positions.append(avg_pos.tolist())
    return com_positions


def track_feet(frame_data: list[FrameData]):
    """ Smoothly track the left and right foot positions. """
    foot_l_positions = []
    foot_r_positions = []
    for frame in frame_data:
        foot_l_positions.append(frame.kinetic_data.foot_pos_l)
        foot_r_positions.append(frame.kinetic_data.foot_pos_r)
    return foot_l_positions, foot_r_positions


def create_animation_json(frame_data: list[FrameData], out_file: str, use_gzip: bool):
    """ Dump all relevant animation data to json """
    n_frames = len(frame_data)

    # Where the camera(s) should look
    # cam_positions = track_com(frame_data)
    cam_positions = track_com_smooth_window(frame_data)
    foot_l_pos, foot_r_pos = track_feet(frame_data)

    # Get all visuals, colliders, muscles
    stacked_frames = []
    for i in range(n_frames):
        visuals = [obj.to_anim_dict() for obj in frame_data[i].visuals]
        beam_points = [point.to_dict() for point in frame_data[i].beam_points]
        colliders = [obj.to_anim_dict() for obj in frame_data[i].colliders]
        muscles = [muscle.to_anim_dict() for muscle in frame_data[i].muscles]
        arrows = [arrow.to_dict() for arrow in frame_data[i].arrows]
        targets = [target.to_dict() for target in frame_data[i].targets]
        time = frame_data[i].time
        scene_settings = frame_data[i].scene_settings.to_dict()
        frame = {
            "scene_settings": scene_settings,
            "time": time,
            "visuals": visuals,
            "beam_points": beam_points,
            "colliders": colliders,
            "muscles": muscles,
            "arrows": arrows,
            "targets": targets,
            "cam_pos": list(cam_positions[i]),
            "foot_l_pos": list(foot_l_pos[i]),
            "foot_r_pos": list(foot_r_pos[i]),
        }
        stacked_frames.append(frame)

    if use_gzip:
        with gzip.open(out_file, 'wt') as f:
            json.dump(stacked_frames, f, indent=2)
    else:
        with open(out_file, 'w') as f:
            json.dump(stacked_frames, f, indent=2)
    return


def append_reference_motion(animation_file: str, reference_motion_file: str, out_file: str):
    """ Append reference motion data to an existing animation json file """
    with open(animation_file, 'r') as f:
        animation_data = json.load(f)

    return
