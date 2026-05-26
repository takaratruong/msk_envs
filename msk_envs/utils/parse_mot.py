import re

import bolt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def swap_cols(data, i, j):
    data[[i, j], :] = data[[j, i], :]


def get_col_names(lines):
    """ column names: [time, joint1, joint2, ..., jointN] """
    col_names = lines[0].strip().split()
    for i in range(len(col_names)):
        # "/jointset/[joint_name]/[coordinate]/value", extrract [coordinate]
        match = re.match(r"/jointset/[^/]+/([^/]+)/value", col_names[i])
        if match:
            col_names[i] = match.group(1)
    return col_names


def get_raw_data(lines, col_names):
    data = [[] for _ in col_names]
    for line in lines[1:]:
        values = line.strip().split()
        for i in range(len(data)):
            data[i].append(float(values[i]))
    return data


def rot_to_quat(data, col_names, in_degrees):
    """ convert [pelvis_tilt, pelvis_list, pelvis_rotation] to quaternion """
    pelvis_tilt_idx = col_names.index("pelvis_tilt")
    pelvis_list_idx = col_names.index("pelvis_list")
    pelvis_rotation_idx = col_names.index("pelvis_rotation")
    pelvis_rot_idxs = [pelvis_tilt_idx, pelvis_list_idx, pelvis_rotation_idx]

    pelvis_quats = []
    for i in range(data.shape[1]):
        r = R.from_euler("ZXY",
                         [
                             data[pelvis_tilt_idx, i],
                             data[pelvis_list_idx, i],
                             data[pelvis_rotation_idx, i]
                         ],
                         degrees=in_degrees)
        q = r.as_quat(canonical=True)
        q = q / np.linalg.norm(q)
        pelvis_quats.append(q)
    pelvis_quats = np.array(pelvis_quats)

    # Remove old columns
    data = np.delete(data, pelvis_rot_idxs, axis=0)
    col_names = [col for j, col in enumerate(col_names) if j not in pelvis_rot_idxs]

    # Insert new columns for quaternion
    for i in range(4):
        data = np.insert(data, pelvis_tilt_idx + i, pelvis_quats[:, i], axis=0)
    # add new col names
    col_names.insert(pelvis_tilt_idx, "pelvis_tilt")
    col_names.insert(pelvis_tilt_idx + 1, "pelvis_list")
    col_names.insert(pelvis_tilt_idx + 2, "pelvis_rotation")
    col_names.insert(pelvis_tilt_idx + 3, "pelvis_quat_w")
    return data, col_names


def reorder_joint_values(data, col_names, qpos_id_lookup):
    """ based on the model's joint order, reorder the motion joint values """
    new_data = np.zeros_like(data)
    new_col_names = []

    to_replace = {}
    for i, col in enumerate(col_names):
        if col in qpos_id_lookup:
            qpos_id = qpos_id_lookup[col]
            j = qpos_id + 1  # + 1 for time col
            to_replace[j] = i

    # copy in columns of data
    for j in range(data.shape[0]):
        if j in to_replace:
            i = to_replace[j]
            new_data[j, :] = data[i, :]
            new_col_names.append(col_names[i])
        else:
            new_data[j, :] = data[j, :]
            new_col_names.append(col_names[j])

    return new_data, new_col_names


def joints_to_radians(data):
    data[8:] = np.deg2rad(data[8:])
    return data


def build_motion(data, col_names, qpos_id_lookup: dict[str, int]):
    num_qpos = max(qpos_id_lookup.values()) + 1
    num_frames = data.shape[1]

    motion = np.zeros((num_frames, num_qpos + 1))
    # Copy time
    motion[:, 0] = data[0, :]

    # Copy dof data
    for i, dof in enumerate(col_names):
        if dof not in qpos_id_lookup:
            print(f"{dof} not in qpos_id_lookup")
            continue
        if "lumbar" in dof:
            continue

        qpos_id = qpos_id_lookup[dof]
        motion[:, qpos_id + 1] = data[i, :]
    return motion


def correct_mot(
        motion,
        m: bolt.Model,
        d: bolt.Data,
        device: torch.device,
):
    motion = torch.tensor(motion, device=device)
    ref_time, ref_frames = motion[:, 0], motion[:, 1:]
    num_frames = len(motion)

    # Corrected motion handles joint limits
    corrected_motion = torch.zeros_like(motion)
    corrected_motion[:, 0] = ref_time

    joint_positions = bolt.joint_positions(d)
    for i in range(num_frames):
        # Set the joint positions
        joint_positions[0, :] = ref_frames[i, :]

        # Reset to fix limits
        d.world_reset.fill_(1.0)
        bolt.fix_limits(m, d)
        d.world_reset.fill_(0.0)
        bolt.fk(m, d)

        # Copy new joint positions
        corrected_motion[i, 1:] = joint_positions[0, :]

    return corrected_motion


def parse_mot(
        motion_file: str,
        load_result: bolt.ModelLoadResult,
        device: torch.device,
        in_degrees: bool = True
):
    with open(motion_file, "r") as f:
        lines = f.readlines()

    col_names = get_col_names(lines)
    data = get_raw_data(lines, col_names)
    data = np.array(data)

    # time should be the first column
    assert col_names.index("time") == 0

    # convert euler to quaternion
    data, col_names = rot_to_quat(data, col_names, in_degrees)

    # Deg to rad
    if in_degrees:
        data = joints_to_radians(data)

    # Make time start at 0
    data[0, :] = data[0, :] - data[0, 0]

    # Now we build the motion
    motion = build_motion(data, col_names, load_result.qpos_id_lookup)

    # Correct the motion to fit in joint limits
    print("Correcting motion...")
    # corrected_motion = correct_mot(motion, load_result.model, load_result.data, device)
    corrected_motion = torch.tensor(motion, device=device)

    return corrected_motion
