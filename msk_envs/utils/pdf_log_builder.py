import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.utils.checkpoint_parser import FrameData
from .plot_helper import SequencePlot, PlotConfig


def create_generic_plot(
        names: list[str],
        times: np.ndarray,
        frame_ind: np.ndarray,
        plot_data: np.ndarray,
        fig_title: str,
        y_label: str,
        y_fmt: str,
        pdf: PdfPages,
        enforced_range: tuple[float, float] = None,
        sublabels: list[str] = None,
        alphas: list[float] = None,
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
                x_fmt=".2f",
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
            muscle_subset_names = names[start_muscle:end_muscle]
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
                        alpha = 1.0 if alphas is None else alphas[part]
                        muscles_plot.add(f, muscle_subset[..., part],
                                         label=label[part],
                                         alpha=alpha,
                                         title=title)

                if enforced_range is not None:
                    muscles_plot.enforce_y_range(f,
                                                 enforced_range[0],
                                                 enforced_range[1])

        muscles_plot.finish(pdf)


def create_interval_plots(interval_duration: float, times: np.ndarray, fn):
    time_current, final_time = 0.0, times[-1]
    if final_time - time_current > interval_duration:  # only if longer than interval
        while time_current < final_time:
            fn(time_current, min(time_current + interval_duration, final_time))
            time_current += interval_duration
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
                x_fmt=".2f",
                x_sub_fmt=".0f",
                y_fmt=".1f",
            )
        )
        for i, key in enumerate(reward_keys):
            rewards_plot.add(0, reward_data[:, i], label=key)
        rewards_plot.add(0, np.sum(reward_data, axis=1), label="Total")
        rewards_plot.add_hline(0, 0.0)
        rewards_plot.finish(pdf)

        # --- GROUND REACTION FORCE ---
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
            time_mask = time_mask.flatten()
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            grf_selected = grf_data[time_mask, :]

            title = f"Ground Reaction Forces ({time_start:.1f}s to {time_end:.1f}s)"
            grf_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title=title,
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="GRF (BW)",
                    x_data=time_selected,
                    x_data_sub=frame_ind_selected,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".0f",
                )
            )

            grf_plot.add_hline(0, 0.0)
            grf_plot.add(0, grf_selected[:, 0], label="X")
            grf_plot.add(0, grf_selected[:, 1], label="Y")
            grf_plot.add(0, grf_selected[:, 2], label="Z")
            grf_plot.finish(pdf)

        # GRF plot for entire duration, and 1 second intervals
        create_grf_plot()
        create_interval_plots(1.0, times, create_grf_plot)

        # Find the intervals in which there is contact
        contact_intervals = []
        contact_threshold = 0.01  # 5% of body weight
        in_contact = False
        contact_start = 0.0
        for i in range(n_frames):
            grf_magnitude = np.linalg.norm(grf_data[i, :])
            if not in_contact and grf_magnitude >= contact_threshold:
                in_contact = True
                contact_start = times[i]
            elif in_contact and grf_magnitude < contact_threshold:
                in_contact = False
                contact_end = times[i]
                contact_intervals.append((contact_start, contact_end))
        if contact_intervals:
            contact_intervals = np.array(contact_intervals)
            contact_durations = contact_intervals[:, 1] - contact_intervals[:, 0]
            contact_mid_times = 0.5 * (contact_intervals[:, 0] + contact_intervals[:, 1])
            contact_time_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Contact Durations",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Contact Duration",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".2f",
                )
            )
            contact_time_plot.add_scatter(0, contact_mid_times, contact_durations, label="Contact Duration",
                                          connect_line=True, labeled=True)
            contact_time_plot.finish(pdf)

        # --- JOINT ANGLES ---
        has_reference = frame_data[0].joint_angles[0].has_reference()
        joint_names = [j.name for j in frame_data[0].joint_angles]
        joint_angles = []
        for frame in frame_data:
            if has_reference:
                joint_angles.append([(j.value, j.reference) for j in frame.joint_angles])
            else:
                joint_angles.append([j.value for j in frame.joint_angles])
        joint_angles = np.array(joint_angles)

        def create_joint_angles_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Angles ({time_start:.1f}s to {time_end:.1f}s)"
            sublabels = ["Value", "Reference"] if has_reference else None
            alpha = [1.0, 0.5] if has_reference else None
            create_generic_plot(joint_names, time_selected, frame_ind_selected, joint_angles[time_mask, :],
                                title, "Value (rad)", ".3f", pdf, add_zero_line=False,
                                sublabels=sublabels, alphas=alpha)

        # Joint angles plot for entire duration, and 1 second intervals
        create_joint_angles_plot()
        create_interval_plots(1.0, times, create_joint_angles_plot)

        # --- JOINT MOMENTS ---
        joint_names = [j.name for j in frame_data[0].joint_moments]
        joint_moments = []
        for frame in frame_data:
            joint_moments.append([j.value for j in frame.joint_moments])
        joint_moments = np.array(joint_moments)

        def create_joint_moments_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Moments ({time_start:.1f}s to {time_end:.1f}s)"
            create_generic_plot(joint_names, time_selected, frame_ind_selected, joint_moments[time_mask, :],
                                title, "Value (N m)", ".3f", pdf, add_zero_line=False)

        # Joint moments plot for entire duration, and 1 second intervals
        create_joint_moments_plot()
        create_interval_plots(1.0, times, create_joint_moments_plot)

        # --- MUSCLE PLOTS ---
        muscle_names = [m.name for m in frame_data[0].muscles]
        # Muscle activations, fiber/tendon lengths
        muscle_ae = []
        muscle_ftl = []
        for frame in frame_data:
            muscle_ae.append([(m.activation, m.excitation) for m in frame.muscles])
            muscle_ftl.append([(m.fiber_length, m.tendon_length) for m in frame.muscles])

        muscle_ftl = np.array(muscle_ftl)
        create_generic_plot(muscle_names, times, frame_ind, np.array(muscle_ae),
                            "Muscle Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_range=(0.0, 1.0),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5])
        create_generic_plot(muscle_names, times, frame_ind, np.array(muscle_ftl),
                            "Muscle Fiber/Tendon Length", "Length (m)", ".3f",
                            pdf, sublabels=["Fiber", "Tendon"])

        # --- ACTUATOR PLOTS ---
        actuator_names = [a.name for a in frame_data[0].actuators]
        actuator_ae = []
        for frame in frame_data:
            actuator_ae.append([(a.activation, a.excitation) for a in frame.actuators])
        actuator_ae = np.array(actuator_ae)
        create_generic_plot(actuator_names, times, frame_ind, np.array(actuator_ae),
                            "Actuator Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_range=(0.0, 1.0),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5], add_zero_line=False)

    return
