import json
from .frame_data import FrameData
import gzip
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def main():
    # with open("deploy_frame_data_0.json", "r") as f:
    #     frame_data_json = json.load(f)
    with gzip.open("/home/marth/Documents/msk_envs/deploy_frame_data_0.json.gz", "rt", encoding="utf-8") as f:
        frame_data_json = json.load(f)

    time_start = 1.0
    time_end = 3.0

    frame_data = []
    for frame_json in frame_data_json:
        frame = FrameData.from_dict(frame_json)
        if time_start < frame.time < time_end:
            frame_data.append(frame)

    frame0 = frame_data[0]
    joint_qpos_names = [j.name for j in frame0.joint_angles]
    muscle_names = [m.name for m in frame0.muscles]

    joint_passive_muscle, joint_active_muscle = [], []
    for frame in frame_data:
        joint_passive_muscle.append([m.passive_breakdown for m in frame.joint_muscle_breakdown])
        joint_active_muscle.append([m.active_breakdown for m in frame.joint_muscle_breakdown])
    joint_passive_muscle = np.array(joint_passive_muscle)  # [n_frames, n_qpos, n_muscles]
    joint_active_muscle = np.array(joint_active_muscle)

    qpos_values = np.array([[j.value for j in frame.joint_angles] for frame in frame_data])

    n_frames, n_qpos, n_muscles = joint_active_muscle.shape
    print(n_frames, n_qpos, n_muscles)
    times = np.array([frame.time for frame in frame_data])

    for iq in range(n_qpos):
        qpos_name = joint_qpos_names[iq]
        if qpos_name != "shoulder_flexion_l":
            continue

        fig, ax = plt.subplots()
        # Color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', ':', '-.']
        combined_cycle = cycler(linestyle=linestyles) * cycler(color=colors)
        ax.set_prop_cycle(combined_cycle)

        for im in range(n_muscles):
            muscle_name = muscle_names[im]
            muscle_breakdown = joint_active_muscle[:, iq, im]
            if np.all(np.abs(muscle_breakdown) < 1e-2):
                continue

            ax.plot(times, joint_active_muscle[:, iq, im], label=muscle_name, alpha=0.8)

        ax2 = ax.twinx()
        ax2.plot(times, qpos_values[:, iq], label=qpos_name, linestyle="dashed", color="gray", linewidth=2)

        ax.set_title(f"Joint {qpos_name} Active Muscle Breakdown")
        ax.legend(loc="upper right", frameon=False, prop={'weight': 'bold', 'size': 10}, ncol=2)
    plt.show()


if __name__ == "__main__":
    main()
