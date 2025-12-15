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


def create_muscle_plot(
        muscle_names: list[str],
        times: np.ndarray,
        frame_ind: np.ndarray,
        plot_data: np.ndarray,
        fig_title: str,
        y_label: str,
        y_fmt: str,
        pdf: PdfPages,
        enforced_range: tuple[float, float] = None,
        sublabels: list[str] = None,
        subset_ind: list[list[int]] = None,
        add_zero_line: bool = True,
):
    num_muscles = plot_data.shape[1]
    num_muscles_per_fig = 1
    n_vertical, n_horizontal = 3, 1
    figs_per_page = n_vertical * n_horizontal
    n_figs = (num_muscles + num_muscles_per_fig - 1) // num_muscles_per_fig
    num_pages = (n_figs + figs_per_page - 1) // figs_per_page

    for p in range(num_pages):
        muscles_plot = SequencePlot(
            PlotConfig(
                num_vertical=n_vertical,
                num_horizontal=n_horizontal,
                fig_size=(8.5, 11),
                title=fig_title,
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label=y_label,
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".1f",
                x_sub_fmt=".0f",
                y_fmt=y_fmt
            )
        )

        for f in range(figs_per_page):
            start_muscle = (p * figs_per_page + f) * num_muscles_per_fig
            end_muscle = min(start_muscle + num_muscles_per_fig,
                             num_muscles)
            if start_muscle >= num_muscles:
                continue
            muscle_subset = plot_data[:, start_muscle:end_muscle]
            muscle_subset_names = muscle_names[start_muscle:end_muscle]
            title = ", ".join(muscle_subset_names)
            for m in range(muscle_subset.shape[1]):
                muscle_name = muscle_subset_names[m]
                muscle_sequence = muscle_subset[:, m]

                # Add a zero line if needed
                if add_zero_line:
                    muscles_plot.add_hline(f, 0.0)

                # simple 1d plot
                if len(muscle_sequence.shape) == 1:
                    muscles_plot.add(f, muscle_sequence, label=muscle_name,
                                     title=title)

                # multiple plots
                else:
                    assert sublabels is not None
                    # Check if we want a subset of the plots for this muscle
                    if subset_ind is not None:
                        muscle_idx = start_muscle + m
                        muscle_subset = muscle_subset[
                            :, m, subset_ind[muscle_idx]]
                        label = [sublabels[i] for i in subset_ind[muscle_idx]]
                    else:
                        label = sublabels

                    for part in range(muscle_subset.shape[-1]):
                        muscles_plot.add(f, muscle_subset[..., part],
                                         label=label[part],
                                         title=title)

                if enforced_range is not None:
                    muscles_plot.enforce_y_range(f,
                                                 enforced_range[0],
                                                 enforced_range[1])

        muscles_plot.finish(pdf)


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

        # --- Ground reaction forces ---
        grf_data = np.array([frame.kinetic_data.grf for frame in frame_data])

        # Express in terms of body weight
        kinetic_data = frame_data[0].kinetic_data
        mass = kinetic_data.total_mass
        weight = abs(float(mass * kinetic_data.gravity))
        grf_data /= weight

        def create_grf_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            grf_selected = grf_data[time_mask, :]

            grf_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Ground Reaction Forces",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="GRF (BW)",
                    x_data=time_selected,
                    x_data_sub=frame_ind_selected,
                    x_fmt=".1f",
                    x_sub_fmt=".0f",
                    y_fmt=".0f",
                )
            )

            grf_plot.add_hline(0, 0.0)
            grf_plot.add(0, grf_selected[:, 0], label="X")
            grf_plot.add(0, grf_selected[:, 1], label="Y")
            grf_plot.add(0, grf_selected[:, 2], label="Z")
            # # Line for body weight, 2x body weight, 6x body weight
            # grf_plot.add_hline(0, weight, f"Weight\n({weight:.1f} N)")
            # grf_plot.add_hline(0, 2 * weight, f"2x Weight\n({2 * weight:.1f} N)")
            # grf_plot.add_hline(0, 6 * weight, f"6x Weight\n({6 * weight:.1f} N)")
            grf_plot.finish(pdf)

        # GRF plot for entire duration
        create_grf_plot()

        # Create plots for 1 second intervals
        interval = 1.0
        time_current, final_time = 0.0, times[-1]
        while time_current < final_time:
            create_grf_plot(time_current, min(time_current + interval, final_time))
            time_current += interval

        # --- Muscle plots ---
        muscle_names = [m.name for m in frame_data[0].muscles]

        # Muscle activations, fiber/tendon lengths
        muscle_ae = []
        muscle_ftl = []
        for frame in frame_data:
            muscle_ae.append([(m.activation, m.excitation) for m in frame.muscles])
            muscle_ftl.append([(m.fiber_length, m.tendon_length) for m in frame.muscles])

        muscle_ftl = np.array(muscle_ftl)
        create_muscle_plot(muscle_names, times, frame_ind, np.array(muscle_ae),
                           "Muscle Activations/Excitations", "Activation/Excitation", ".2f",
                           pdf, enforced_range=(0.0, 1.0),
                           sublabels=["Activation", "Excitation"])
        create_muscle_plot(muscle_names, times, frame_ind, np.array(muscle_ftl),
                           "Muscle Fiber/Tendon Length", "Length (m)", ".3f",
                           pdf, sublabels=["Fiber", "Tendon"])

    return
