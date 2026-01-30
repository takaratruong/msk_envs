import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.utils.frame_parser import FrameData
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
        enforced_y_range: list[tuple[float, float]] = None,
        sublabels: list[str] = None,
        alphas: list[float] = None,
        linestyles: list[str] = None,
        horizontal_lines: list[list[float]] = None,
        omit_zeros: bool = False,
):
    n_vertical, n_horizontal = 3, 1
    n_plots = plot_data.shape[1]
    figs_per_page = n_vertical * n_horizontal
    num_pages = (n_plots + figs_per_page - 1) // figs_per_page

    for idx_page in range(num_pages):
        seq_plot = SequencePlot(
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

        # Add plots for this page
        for idx_fig in range(figs_per_page):
            # Retrieve data subset for this figure
            start_idx = (idx_page * figs_per_page + idx_fig)
            end_idx = start_idx + 1
            if start_idx >= n_plots:
                continue
            data_subset = plot_data[:, start_idx:end_idx]
            data_subset_names = names[start_idx:end_idx]
            title = ", ".join(data_subset_names)

            # Add each entry in the subset
            for i in range(data_subset.shape[1]):
                entry_name = data_subset_names[i]
                data_sequence = data_subset[:, i]

                if len(data_sequence.shape) == 1:  # simple 1d plot
                    seq_plot.add(idx_fig, data_sequence, label=entry_name, title=title)
                else:  # multiple values per entry (e.g., value and reference)
                    assert sublabels is not None
                    label = sublabels

                    for part in range(data_subset.shape[-1]):
                        alpha = 1.0 if alphas is None else alphas[part]
                        linestyle = 'solid' if linestyles is None else linestyles[part]
                        if omit_zeros and np.all(data_subset[..., part] == 0.0):
                            continue
                        seq_plot.add(idx_fig, data_subset[..., part],
                                     label=label[part],
                                     alpha=alpha,
                                     linestyle=linestyle,
                                     title=title)

                # Add horizontal lines if specified. can be used for zero lines or joint limits
                idx_entry = start_idx + i
                if horizontal_lines and horizontal_lines[idx_entry] is not None:
                    for hline in horizontal_lines[idx_entry]:
                        seq_plot.add_hline(idx_fig, hline)

                if enforced_y_range is not None and enforced_y_range[start_idx] is not None:
                    seq_plot.enforce_y_range(idx_fig, enforced_y_range[start_idx][0], enforced_y_range[start_idx][1])

        seq_plot.finish(pdf)


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
    frame0 = frame_data[0]
    with (PdfPages(out_file) as pdf):
        # Rewards plot
        reward_keys = list(frame0.reward_data.keys())
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
                y_fmt=".2f",
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
        kinetic_data = frame0.kinetic_data
        mass = kinetic_data.total_mass
        weight = abs(float(mass * kinetic_data.gravity))
        grf_data /= weight

        def create_grf_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_start = max(time_start, times[0])
            time_end = min(time_end, times[-1])
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
                    y_fmt=".1f",
                )
            )
            grf_plot.add_hline(0, 0.0)
            grf_plot.add(0, grf_selected[:, 0], label="X")
            grf_plot.add(0, grf_selected[:, 1], label="Y")
            grf_plot.add(0, grf_selected[:, 2], label="Z")
            grf_plot.finish(pdf)

        # GRF plot for entire duration
        create_grf_plot()

        # Find the intervals in which there is contact
        contact_intervals = []
        contact_threshold = 1e-3
        in_contact = False
        contact_start = 0.0
        for i in range(n_frames):
            grf_magnitude = np.linalg.norm(grf_data[i, :])
            if not in_contact and grf_magnitude >= contact_threshold:  # start of contact
                in_contact = True
                contact_start = times[i]
            elif in_contact and grf_magnitude < contact_threshold:  # end of contact
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
                    y_fmt=".3f",
                )
            )
            contact_time_plot.add_scatter(0, contact_mid_times, contact_durations, label="Contact Duration",
                                          connect_line=True, labeled=True)
            contact_time_plot.finish(pdf)

        # Create interval plots for each contact interval
        dt = times[1] - times[0]
        for (start_time, end_time) in contact_intervals:
            create_grf_plot(start_time - dt, end_time + dt)

        # --- JOINT ANGLES ---
        has_reference = frame0.joint_angles[0].has_reference()
        joint_qpos_names = [j.name for j in frame0.joint_angles]
        joint_angles = []
        joint_angle_limits = [j.limits for j in frame0.joint_angles]
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
            create_generic_plot(joint_qpos_names, time_selected, frame_ind_selected, joint_angles[time_mask, :],
                                title, "Value (m or rad)", ".3f", pdf, sublabels=sublabels, alphas=alpha,
                                horizontal_lines=joint_angle_limits)

        # Joint angles plot for entire duration, and 1 second intervals
        create_joint_angles_plot()
        create_interval_plots(1.0, times, create_joint_angles_plot)

        # --- JOINT MOMENTS ---
        joint_dof_names = [j.name for j in frame0.joint_moments]
        joint_moments = []
        for frame in frame_data:
            joint_moments.append([
                (m.spring, m.drag, m.muscle, m.actuator, m.limit, m.contact)
                for m in frame.joint_moments])
        joint_moments = np.array(joint_moments)
        sublabels = ["Spring", "Drag", "Muscle", "Actuator", "Limit", "Contact"]

        def create_joint_moments_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Moments ({time_start:.1f}s to {time_end:.1f}s)"
            lss, lsd = "solid", "dashed"
            create_generic_plot(
                names=joint_dof_names,
                times=time_selected,
                frame_ind=frame_ind_selected,
                plot_data=joint_moments[time_mask, :],
                fig_title=title,
                y_label="Value (N m)",
                y_fmt=".1f",
                pdf=pdf,
                sublabels=sublabels,
                alphas=[0.5] * (len(sublabels)),
                linestyles=[lss] * (len(sublabels)),
                horizontal_lines=[[0.0]] * len(joint_dof_names),
                omit_zeros=True
            )

        # Joint moments plot for entire duration, and 1 second intervals
        create_joint_moments_plot()
        create_interval_plots(1.0, times, create_joint_moments_plot)

        # --- MUSCLE PLOTS ---
        muscle_names = [m.name for m in frame0.muscles]
        # Muscle activations, fiber/tendon lengths, moment arms
        muscle_ae = []
        muscle_ftl = []
        muscle_ma = []
        muscle_frc = []
        for frame in frame_data:
            muscle_ae.append([(m.activation, m.excitation) for m in frame.muscles])
            muscle_ftl.append([(m.fiber_length, m.tendon_length, m.optimal_fiber_length, m.tendon_slack_length) for m in
                               frame.muscles])
            muscle_ma.append([m.moment_arm for m in frame.muscles])
            muscle_frc.append([(m.actuation, m.max_isometric_force) for m in frame.muscles])

        muscle_ae = np.array(muscle_ae)
        muscle_ftl = np.array(muscle_ftl)
        muscle_ma = np.array(muscle_ma)
        muscle_frc = np.array(muscle_frc)
        zero_lines = [[0.0, 0.0]] * len(muscle_names)

        # Activation/excitation
        create_generic_plot(muscle_names, times, frame_ind, muscle_ae,
                            "Muscle Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_y_range=[(0.0, 1.0)] * len(muscle_names),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5], horizontal_lines=zero_lines)

        # Fiber/tendon lengths
        enforced_range = []
        for m in frame0.muscles:
            min_range = min(0.75 * m.optimal_fiber_length, 0.75 * m.tendon_slack_length)
            max_range = max(1.25 * m.optimal_fiber_length, 1.25 * m.tendon_slack_length)
            enforced_range.append((min_range, max_range))
        create_generic_plot(
            muscle_names, times, frame_ind, muscle_ftl,
            "Muscle Fiber/Tendon Length", "Length (m)", ".3f",
            pdf,
            sublabels=["Fiber", "Tendon", "Optimal Fiber", "Tendon Slack"],
            alphas=[1.0, 1.0, 0.5, 0.5],
            linestyles=["solid", "solid", "dashed", "dashed"],
            enforced_y_range=enforced_range)

        # Moment arms
        enforced_range = [(-0.1, 0.1)] * len(muscle_names)
        create_generic_plot(muscle_names, times, frame_ind, muscle_ma,
                            "Muscle Moment Arms", "Moment Arm (m)", ".2f",
                            pdf, sublabels=joint_dof_names, omit_zeros=True, horizontal_lines=zero_lines,
                            enforced_y_range=enforced_range)

        # Muscle actuation
        enforced_range = [(0.0, 2.0 * m.max_isometric_force) for m in frame0.muscles]
        create_generic_plot(muscle_names, times, frame_ind, muscle_frc,
                            "Muscle Actuation", "Actuation (N)", ".3f",
                            pdf, sublabels=["Actuation", "Max Isometric Force"],
                            alphas=[1.0, 0.5],
                            enforced_y_range=enforced_range)

        # --- ACTUATOR PLOTS ---
        actuator_names = [a.name for a in frame0.actuators]
        actuator_ae = []
        for frame in frame_data:
            actuator_ae.append([(a.activation, a.excitation) for a in frame.actuators])
        actuator_ae = np.array(actuator_ae)
        create_generic_plot(actuator_names, times, frame_ind, np.array(actuator_ae),
                            "Actuator Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_y_range=[(0.0, 1.0)] * len(actuator_names),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5])

    return
