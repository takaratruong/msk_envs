import json

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.utils.checkpoint_parser import FrameData
from .plot_helper import SequencePlot, PlotConfig


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


def create_pdf_output(frame_data: list[FrameData], out_file: str):
    """ Create a pdf with all the relevant plots """
    n_frames = len(frame_data)
    times = np.array([frame.time for frame in frame_data])
    frame_ind = np.arange(n_frames)
    with (PdfPages(out_file) as pdf):
        # Rewards plot
        reward_keys = list(frame_data[0].reward_data.keys())
        reward_data = []
        for frame in frame_data:
            reward_data.append([frame.reward_data[k] for k in reward_keys])
        reward_data = np.array(reward_data)

        rewards_plot = SequencePlot(
            PlotConfig(
                num_vertical=1,
                num_horizontal=1,
                fig_size=(8.5, 6),
                title="Rewards",
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label="Reward",
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".1f",
                x_sub_fmt=".0f",
                y_fmt=".1f",
            )
        )
        for i, key in enumerate(reward_keys):
            rewards_plot.add(0, reward_data[:, i], label=key)
        rewards_plot.add(0, np.sum(reward_data, axis=1), label="Total")
        rewards_plot.add_hline(0, 0.0)
        rewards_plot.finish(pdf)

        # Ground reaction force
        grf_plot = SequencePlot(
            PlotConfig(
                num_vertical=1,
                num_horizontal=1,
                fig_size=(8.5, 6),
                title="Ground Reaction Forces",
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label="Force (N)",
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".1f",
                x_sub_fmt=".0f",
                y_fmt=".0f",
            )
        )

        grf_plot.add_hline(0, 0.0)
        grf_data = np.array([frame.kinetic_data.grf for frame in frame_data])
        grf_plot.add(0, grf_data[:, 0], label="X")
        grf_plot.add(0, grf_data[:, 1], label="Y")
        grf_plot.add(0, grf_data[:, 2], label="Z")
        # grf_plot.add(0, np.linalg.norm(grf_data, axis=1), label="Norm")
        # Line for body weight, 2x body weight, 6x body weight
        kinetic_data = frame_data[0].kinetic_data
        mass = kinetic_data.total_mass
        weight = abs(float(mass * kinetic_data.gravity))
        grf_plot.add_hline(0, weight, f"Weight\n({weight:.1f} N)")
        grf_plot.add_hline(0, 2 * weight, f"2x Weight\n({2 * weight:.1f} N)")
        grf_plot.add_hline(0, 6 * weight, f"6x Weight\n({6 * weight:.1f} N)")
        grf_plot.finish(pdf)
    return
