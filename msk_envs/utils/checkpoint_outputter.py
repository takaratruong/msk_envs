import json
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.utils.checkpoint_parser import FrameData
from msk_envs.utils.plot_helper import SequencePlot, PlotConfig
from msk_envs.utils.sim_objects import joint_id_to_name


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
        muscles = [muscle.to_anim_dict() for muscle in frame_data[i].muscles]
        time = frame_data[i].time
        frame = {
            "time": time,
            "visuals": visuals,
            "colliders": colliders,
            "muscles": muscles,
            "cam_pos": cam_positions[i]
        }
        stacked_frames.append(frame)

    with open(out_file, 'w') as f:
        json.dump(stacked_frames, f, indent=2)
    return


def create_force_plot(force_names,
                      times, frame_ind, plot_data,
                      fig_title, y_label, y_fmt, pdf,
                      enforced_range=None, sublabels=None,
                      subset_ind=None, add_zero_line=False):
    num_total_plots = plot_data.shape[1]
    num_muscles_per_fig = 1
    n_vertical, n_horizontal = 3, 1
    figs_per_page = n_vertical * n_horizontal
    n_figs = (num_total_plots + num_muscles_per_fig - 1) // num_muscles_per_fig
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
                             num_total_plots)
            if start_muscle >= num_total_plots:
                continue
            muscle_subset = plot_data[:, start_muscle:end_muscle]
            muscle_subset_names = force_names[start_muscle:end_muscle]
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
        weight = mass * kinetic_data.gravity
        grf_plot.add_hline(0, weight, f"Weight\n({weight:.1f} N)")
        grf_plot.add_hline(0, 2 * weight, f"2x Weight\n({2 * weight:.1f} N)")
        grf_plot.add_hline(0, 6 * weight, f"6x Weight\n({6 * weight:.1f} N)")
        grf_plot.finish(pdf)

        # Total metabolic power per kg
        muscle_power = []
        for frame in frame_data:
            muscle_power.append([m.metabolic_power for m in frame.muscles])
        muscle_power = np.array(muscle_power)
        total_power = np.sum(muscle_power, axis=1)
        total_power_per_kg = total_power / mass
        power_plot = SequencePlot(
            PlotConfig(
                num_vertical=1,
                num_horizontal=1,
                fig_size=(8.5, 6),
                title="Total Muscle Metabolic Power",
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label="Power (W/kg)",
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".1f",
                x_sub_fmt=".0f",
                y_fmt=".1f",
            )
        )
        power_plot.add(0, total_power_per_kg, label="Total Power per kg")
        power_plot.finish(pdf)


        # Joint angles
        joint_angles = []
        for frame in frame_data:
            joint_angles.append([ja.value for ja in frame.joint_angles])
        joint_angles = np.array(joint_angles)
        joint_names = [ja.name for ja in frame_data[0].joint_angles]
        limited = [ja.limited for ja in frame_data[0].joint_angles]
        joint_ranges = [ja.range for ja in frame_data[0].joint_angles]

        # plot will probably require multiple pages
        n_joints = joint_angles.shape[1]
        n_vertical, n_horizontal = 3, 1
        figs_per_page = n_vertical * n_horizontal
        num_pages = (n_joints + figs_per_page - 1) // figs_per_page
        for p in range(num_pages):
            ja_plot = SequencePlot(
                PlotConfig(
                    num_vertical=n_vertical,
                    num_horizontal=n_horizontal,
                    fig_size=(8.5, 11),
                    title="Joint Angles",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Angle (rad)",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".1f",
                    x_sub_fmt=".0f",
                    y_fmt=".2f",
                )
            )
            joint_idx_start = p * figs_per_page
            for i in range(figs_per_page):
                joint_idx = joint_idx_start + i
                if joint_idx >= n_joints:
                    continue
                # add joint angles
                ja_plot.add(i, joint_angles[:, joint_idx],
                            label="", title=joint_names[joint_idx])

                # joint limits
                if limited[joint_idx]:
                    lower, upper = joint_ranges[joint_idx]
                    joint_range = upper - lower
                    delta = 0.05 * abs(joint_range)
                    ja_plot.add_hline(i, lower - delta)
                    ja_plot.add_hline(i, upper + delta)
            ja_plot.finish(pdf)

        print(f"Created {n_joints} joint angle plots")

        # Actuator forces
        actuator_names = [a.name for a in frame_data[0].actuators]

        actuator_forces = []
        for frame in frame_data:
            actuator_forces.append(
                [(a.force, -a.optimal_force, a.optimal_force) for a in
                 frame.actuators])
        actuator_forces = np.array(actuator_forces)
        create_force_plot(actuator_names,
                          times, frame_ind, actuator_forces,
                          "Actuator/Motor Forces", "Force (N)", ".1f", pdf,
                          None, ["Force", "Min Optimal", "Max Optimal"])
        print(f"Created {len(actuator_names)} actuator force plots")

        # Muscle activations
        muscle_names = [m.name for m in frame_data[0].muscles]

        muscle_ae = []
        for frame in frame_data:
            muscle_ae.append(
                [(m.activation, m.excitation) for m in frame.muscles])
        muscle_ae = np.array(muscle_ae)
        create_force_plot(muscle_names, times, frame_ind, muscle_ae,
                          "Muscle Activations/Excitations",
                          "Activation/Excitation", ".2f", pdf,
                          enforced_range=(0.0, 1.0),
                          sublabels=["Activation", "Excitation"])
        print(f"Created {len(muscle_names)} muscle activation/excitation plots")

        # Muscle metabolic power
        create_force_plot(muscle_names, times, frame_ind,
                          muscle_power, "Muscle Metabolic Power",
                          "Power (W)", ".1f", pdf, None, ["Power"])
        print(f"Created {len(muscle_names)} muscle metabolic power plots")

        # Muscle actuation
        muscle_actuations = []
        for frame in frame_data:
            muscle_actuations.append([(m.actuation, m.max_isometric_force)
                                      for m in frame.muscles])
        muscle_actuations = np.array(muscle_actuations)
        create_force_plot(muscle_names,
                          times, frame_ind, muscle_actuations,
                          "Muscle Actuations", "Actuation (N)", ".1f", pdf,
                          None, ["Actuation", "Max Isometric Force"])
        print(f"Created {len(muscle_names)} muscle actuation plots")

        # Fiber/tendon lengths
        fiber_tendon_lengths = []
        for frame in frame_data:
            fiber_tendon_lengths.append([(m.fiber_length, m.tendon_length)
                                         for m in frame.muscles])
        fiber_tendon_lengths = np.array(fiber_tendon_lengths)
        create_force_plot(muscle_names, times, frame_ind,
                          fiber_tendon_lengths, "Muscle Fiber/Tendon Length",
                          "Length (m)", ".3f", pdf, None, ["Fiber", "Tendon"])
        print(f"Created {len(muscle_names)} muscle fiber/tendon length plots")

        # Muscle moment arms
        moment_arms = []
        for frame in frame_data:
            moment_arms.append([m.moment_arms for m in frame.muscles])
        moment_arms = np.array(moment_arms) # (n_frames, n_muscles, n_dofs)
        # Get the dofs we're interested in
        dof_interest = [np.array(m.dof_interest) for m in frame_data[0].muscles]

        # Disable joints with non-constant moment arms, update dof_interest
        # n_dofs = moment_arms.shape[2]
        # for muscle_idx in range(len(muscle_names)):
        #     for dof_idx in range(n_dofs):
        #         # take std over time
        #         std = np.std(moment_arms[:, muscle_idx, dof_idx])
        #         if std < 1e-5:
        #             dof_interest[muscle_idx][dof_idx] = False

        ind_interest = [np.where(dof_interest[m])[0] for m in
                        range(len(dof_interest))]

        # Joints of interest are just the dofs (velocity)
        # if len(ind_interest) == 0:
        #     print("No muscle moment arms to plot?")
        #     return
        max_dof_idx = max([np.max(ind) for ind in ind_interest]) + 1
        moment_names = [joint_id_to_name(i, False) for i in range(max_dof_idx)]
        create_force_plot(muscle_names, times, frame_ind, moment_arms,
                          "Muscle Moment Arms", "Moment Arm", ".3f", pdf,
                          None, moment_names, ind_interest, True)

        print(f"Created {len(muscle_names)} muscle moment arm plots")

    return
