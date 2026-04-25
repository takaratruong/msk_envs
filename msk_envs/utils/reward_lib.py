import torch
from msk_envs.utils.global_params import FWD_IDX, UP_IDX, SIDE_IDX, MIN_ROOT_HEIGHT, MIN_HEAD_HEIGHT
from msk_envs.utils.quat import quat_diff_angle, rotate_vec


def _get_velocity(body_velocities, body_id: int, linear: bool, idx: int):
    """ Returns the [idx]-velocity of body [body_id] """
    return body_velocities[:, body_id, idx + 3] if linear else body_velocities[:, body_id, idx]


def velocity_reward(body_velocities, body_id: int, coordinate: int, linear: bool):
    """Forward velocity reward (x-velocity of root body)"""
    root_velocity = _get_velocity(body_velocities, body_id, linear, coordinate)
    return root_velocity


def mid_lane_reward(root_position: torch.Tensor, weight: float = 2.5):
    """Reward for being in the center of the lane"""
    dist_from_center = torch.abs(root_position[:, SIDE_IDX])
    return torch.exp(-weight * dist_from_center.pow(2))


def muscle_activation_penalty(muscle_activations: torch.Tensor, threshold: float = 0.15):
    above_thresh = muscle_activations > threshold
    return above_thresh.float().mean(dim=1)


def exp_distance(values: torch.Tensor, targets: torch.Tensor, ranges: torch.Tensor, weight: float):
    diff = values - targets
    scaled_diff = diff / ranges
    return torch.exp(-weight * scaled_diff.pow(2))


def joint_penalty(limit_torques: torch.Tensor, squared: bool = False):
    """Joint limit penalty based on sum of absolute limit torques"""
    if squared:
        sq_limit_torque = torch.pow(limit_torques, 2)
        limit_torque_sum = torch.sum(sq_limit_torque, dim=1)
    else:
        abs_limit_torque = torch.abs(limit_torques)
        limit_torque_sum = torch.sum(abs_limit_torque, dim=1)

    return limit_torque_sum


def ignore_toe_joint_penalty(limit_torques: torch.Tensor, toe_dof_ids: list[int], squared: bool = False):
    """Joint limit penalty that ignores specified toe joints"""
    mask = torch.ones(limit_torques.shape[1], dtype=torch.bool, device=limit_torques.device)
    mask[toe_dof_ids] = False
    filtered_limit_torques = limit_torques[:, mask]

    if squared:
        sq_limit_torque = torch.pow(filtered_limit_torques, 2)
        limit_torque_sum = torch.sum(sq_limit_torque, dim=1)
    else:
        abs_limit_torque = torch.abs(filtered_limit_torques)
        limit_torque_sum = torch.sum(abs_limit_torque, dim=1)

    return limit_torque_sum


def alive_bonus(fallen: torch.Tensor):
    """Alive bonus based on whether the agent has fallen"""
    return 1.0 - fallen.float()


def actuator_sq_penalty(actuator_activations):
    """
    Actuator penalty based on squared activation. Note that 0.5 = no activation, so we shift and scale to [-1, 1] first.
    """
    actuator_act = (actuator_activations - 0.5) * 2.0
    squared_act = torch.pow(actuator_act, 2)
    mean_squared_act = torch.sum(squared_act, dim=1)
    return mean_squared_act


def derivative_sq_penalty(derivative, num_values):
    """
    Penalty based on the squared derivative.
    """
    squared_der = torch.pow(derivative, 2)
    mean_squared_der = torch.sum(squared_der, dim=1) / num_values
    if num_values == 0:
        return torch.zeros_like(mean_squared_der)
    return mean_squared_der


def metabolic_penalty(muscle_powers, num_muscles, squared: bool = False):
    """Metabolic penalty based on total muscle power"""
    if squared:
        muscle_powers_sq = torch.pow(muscle_powers, 2)
        total_power = torch.sum(muscle_powers_sq, dim=1)
    else:
        total_power = torch.sum(muscle_powers, dim=1)

    if num_muscles == 0:
        return torch.zeros_like(total_power)
    return total_power / num_muscles


def root_zero_reward(root_positions, weight: float):
    """Reward for root being close to zero position in all axes"""
    zero_pos = torch.zeros_like(root_positions)
    zero_pos[:, UP_IDX] = root_positions[:, UP_IDX]  # Ignore vertical axis
    dist_from_zero = torch.norm(root_positions - zero_pos, dim=1)
    reward = torch.exp(-weight * dist_from_zero.pow(2))
    return reward


def match_start_pos_reward(joint_positions, start_positions, weight: float, ignore_root: bool):
    """Reward for root being close to start position in all axes"""
    if ignore_root:  # first 7 values are root
        joint_positions = joint_positions[..., 7:]
        start_positions = start_positions[..., 7:]
    dist_from_start = torch.norm(joint_positions - start_positions, dim=1)
    reward = torch.exp(-weight * dist_from_start.pow(2))
    return reward


def cost_of_transport(muscle_powers, body_velocities, root_id):
    """Cost of transport based on total muscle power divided by forward velocity"""
    total_power = torch.sum(muscle_powers, dim=1)
    forward_velocity = _get_velocity(body_velocities, root_id, linear=True, idx=FWD_IDX)

    # Avoid division by zero, ensure non-negative forward velocity
    forward_velocity = torch.clamp(forward_velocity, min=1e-2)

    cot = total_power / forward_velocity
    return cot


def head_acceleration_penalty(head_velocities, prev_head_velocities, delta_t):
    """Penalty based on squared head accelerations"""
    head_accelerations = (head_velocities - prev_head_velocities) / delta_t
    head_acc_mag = torch.norm(head_accelerations, dim=1)
    return head_acc_mag


def activation_square_penalty(muscle_activations):
    """Fatigue penalty based on squared muscle activations"""
    activations_sq = torch.pow(muscle_activations, 2)
    total_fatigue = torch.sum(activations_sq, dim=1)
    return total_fatigue


def self_collision_penalty(collision_forces, force_bound):
    clamped_collision_forces = torch.clamp(collision_forces, min=-force_bound, max=force_bound)
    norm_collision_forces = torch.abs(clamped_collision_forces) / force_bound
    return norm_collision_forces.sum(dim=1)


def head_acc_penalty(head_accelerations):
    """Penalty based on squared head accelerations"""
    # acceleration is (angular, linear)
    acc_ang = head_accelerations[:, :3]
    acc_lin = head_accelerations[:, 3:]
    return torch.norm(acc_ang, dim=1), torch.norm(acc_lin, dim=1)


def acceleration_sq_penalty(joint_accelerations):
    """Penalty based on squared joint accelerations"""
    acc_sq = torch.pow(joint_accelerations, 2)
    return torch.mean(acc_sq, dim=1)


def max_vertical_reward(body_positions):
    """Maximum vertical height achieved (current max across all bodies)"""
    current_max_height = torch.max(body_positions[:, :, UP_IDX], dim=1)[0]
    return current_max_height


def joint_angle_track_reward(joint_positions, target_joint_positions, qpos_adr, weight):
    """Imitation reward for joint tracking"""
    # If qpos_adr can be either a range or an integer
    if isinstance(qpos_adr, range):
        _range = qpos_adr
    else:
        _range = range(qpos_adr, qpos_adr + 1)

    joint_values = joint_positions[:, _range]
    target_values = target_joint_positions[:, _range]

    mse = (joint_values - target_values).pow(2).mean(dim=1)
    reward = torch.exp(-weight * mse)
    return reward


def single_body_pos_track_reward(body_positions, target_body_positions, weight):
    """Imitation reward for body position tracking (for a single body)"""
    mse = (body_positions - target_body_positions).pow(2).mean(dim=1)
    reward = torch.exp(-weight * mse)
    return reward


def body_pos_track_reward(body_positions, target_body_positions, body_id, weight):
    """Imitation reward for body position tracking"""
    # If body_id can be either a range or an integer
    if isinstance(body_id, range) or isinstance(body_id, list):
        _range = body_id
    else:
        _range = range(body_id, body_id + 1)

    body_pos_values = body_positions[:, _range, :]
    target_pos_values = target_body_positions[:, _range, :]
    mse = (body_pos_values - target_pos_values).pow(2).mean(dim=(1, 2))
    reward = torch.exp(-weight * mse)
    return reward


def body_rot_track_reward(body_rotations, target_body_rotations, body_id, weight):
    # If body_id can be either a range or an integer
    if isinstance(body_id, range) or isinstance(body_id, list):
        _range = body_id
    else:
        _range = range(body_id, body_id + 1)

    body_rot_values = body_rotations[:, _range, :]
    target_rot_values = target_body_rotations[:, _range, :]
    rot_diff_angle = quat_diff_angle(body_rot_values, target_rot_values)
    mse = rot_diff_angle.pow(2).mean(dim=1)
    reward = torch.exp(-weight * mse)
    return reward


def has_fallen(root_pos, head_pos, head_rot, head_offset, min_root=MIN_ROOT_HEIGHT, min_head=MIN_HEAD_HEIGHT):
    # Root falls below threshold
    root_height = root_pos[:, UP_IDX]
    pelvis_fallen = (root_height < min_root)

    # Head falls below threshold
    actual_head_pos = head_pos + rotate_vec(head_rot, head_offset)
    head_fallen = (actual_head_pos[:, UP_IDX] < min_head)

    fallen = pelvis_fallen | head_fallen
    return fallen.detach()


def update_dict(reward_dict, key, value):
    if key in reward_dict:
        reward_dict[key] += value.detach()
    else:
        reward_dict[key] = value.detach()
