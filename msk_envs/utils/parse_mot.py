import torch
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import msk_warp


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


def reorder_translation(data, col_names):
    """ Move pelvis_tx, pelvis_ty, pelvis_tz to be directly time col """
    pelvis_tx_idx = col_names.index("pelvis_tx")
    pelvis_ty_idx = col_names.index("pelvis_ty")
    pelvis_tz_idx = col_names.index("pelvis_tz")
    # Swap data
    swap_cols(data, pelvis_tx_idx, 1)
    swap_cols(data, pelvis_ty_idx, 2)
    swap_cols(data, pelvis_tz_idx, 3)
    # Swap col_names
    col_names[1], col_names[pelvis_tx_idx] = (
        col_names[pelvis_tx_idx], col_names[1])
    col_names[2], col_names[pelvis_ty_idx] = (
        col_names[pelvis_ty_idx], col_names[2])
    col_names[3], col_names[pelvis_tz_idx] = (
        col_names[pelvis_tz_idx], col_names[3])
    return data, col_names


def rot_to_quat(data, col_names):
    """ convert [pelvis_tilt, pelvis_list, pelvis_rotation] to quaternion """
    pelvis_tilt_idx = col_names.index("pelvis_tilt")
    pelvis_list_idx = col_names.index("pelvis_list")
    pelvis_rotation_idx = col_names.index("pelvis_rotation")
    pelvis_rot_idxs = [pelvis_tilt_idx, pelvis_list_idx, pelvis_rotation_idx]

    pelvis_quats = []
    for i in range(data.shape[1]):
        # in the motion, it is ZXY (but we swap from y-up to z-up)
        r = R.from_euler("ZXY",
                         [
                             data[pelvis_tilt_idx, i],
                             data[pelvis_list_idx, i],
                             data[pelvis_rotation_idx, i]
                         ],
                         degrees=True)
        q = r.as_quat()
        pelvis_quats.append(q)
    pelvis_quats = np.array(pelvis_quats)
    pelvis_quats = pelvis_quats[:, [3, 0, 1, 2]]  # xyzw to wxyz

    # Remove old columns
    data = np.delete(data, pelvis_rot_idxs, axis=0)
    col_names = [col for j, col in enumerate(col_names)
                 if j not in pelvis_rot_idxs]

    # Insert new columns
    for i in range(4):
        data = np.insert(data, pelvis_tilt_idx + i,
                         pelvis_quats[:, i], axis=0)
    # add new col names
    col_names.insert(pelvis_tilt_idx, "pelvis_rot_w")
    col_names.insert(pelvis_tilt_idx + 1, "pelvis_rot_x")
    col_names.insert(pelvis_tilt_idx + 2, "pelvis_rot_y")
    col_names.insert(pelvis_tilt_idx + 3, "pelvis_rot_z")
    return data, col_names


def reorder_joint_values(data, col_names, dof_id_lookup):
    """ based on the model's joint order, reorder the motion joint values """
    new_data = np.zeros_like(data)
    new_col_names = []

    to_replace = {}
    for i, col in enumerate(col_names):
        if col in dof_id_lookup:
            qpos_idx, dof_idx = dof_id_lookup[col]
            j = qpos_idx + 1  # + 1 for time col
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
    data[7:] = np.deg2rad(data[7:])
    return data


def swap_yz(data):
    """ Swap from right-handed y-up to right-handed z-up """
    rot_convert = R.from_euler("X", 90, degrees=True)
    for i in range(data.shape[1]):
        # Pelvis position
        data[1:4, i] = rot_convert.apply(data[1:4, i])
    return data


def parse_mot(motion_file: str, model_file: str):
    with open(motion_file, "r") as f:
        lines = f.readlines()

    col_names = get_col_names(lines)
    data = get_raw_data(lines, col_names)
    data = np.array(data)

    # time should be the first column
    assert col_names.index("time") == 0

    # reorganize to be [time, pelvis_translation, pelvis_rot, ...]
    data, col_names = reorder_translation(data, col_names)
    data, col_names = rot_to_quat(data, col_names)

    # # Y up to Z up
    # data = swap_yz(data)

    # Grab the joint columns that correspond to our model, reorder them
    load_result = msk_warp.load_model(model_file, 1)
    dof_id_lookup = load_result.dof_id_lookup

    data, col_names = reorder_joint_values(data, col_names, dof_id_lookup)

    # Deg to rad
    data = joints_to_radians(data)

    # Make time start at 0
    data[0, :] = data[0, :] - data[0, 0]

    return data, col_names


def main(motion_file: str = "msk_envs/motions/reference_stride.mot",
         model_file: str = "msk_envs/msk_models/model.osim"):
    data, col_names = parse_mot(motion_file, model_file)
    torch.save(torch.tensor(data), motion_file.replace(".mot", ".pt"))
    print(col_names)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="msk_envs/motions/reference_stride.mot")
    parser.add_argument("--model", default="msk_envs/msk_models/model.osim")
    args = parser.parse_args()
    main(args.motion, args.model)
