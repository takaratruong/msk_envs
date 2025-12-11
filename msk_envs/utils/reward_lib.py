import torch


def velocity_reward(body_velocities, body_id: int, coordinate: int, linear: bool):
    """Forward velocity reward (x-velocity of root body)"""
    root_velocity = body_velocities[:, body_id, :]
    return root_velocity[:, coordinate + 3] if linear else root_velocity[:, coordinate]


def joint_limit_penalty(limit_torques):
    """Joint limit penalty based on sum of absolute limit torques"""
    num_limits = limit_torques.shape[1]
    abs_limit_torque = torch.abs(limit_torques)
    abs_limit_torque_sum = torch.sum(abs_limit_torque, dim=1)
    if num_limits == 0:
        return torch.zeros_like(abs_limit_torque_sum)
    return abs_limit_torque_sum / num_limits


def actuator_penalty(actuator_activations, num_actuators):
    """Actuator penalty based on squared activation deviation from 0.5"""
    actuator_act = (actuator_activations - 0.5) * 2.0
    squared_act = torch.pow(actuator_act, 2)
    mean_squared_act = torch.sum(squared_act, dim=1) / num_actuators
    if num_actuators == 0:
        mean_squared_act = torch.zeros_like(mean_squared_act)
    return mean_squared_act


def max_vertical_reward(body_positions):
    """Maximum vertical height achieved (current max across all bodies)"""
    current_max_height = torch.max(body_positions[:, :, 2], dim=1)[0]
    return current_max_height
