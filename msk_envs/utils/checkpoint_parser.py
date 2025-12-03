from dataclasses import dataclass

import torch

from msk_envs.utils.sim_objects import (
    obj_id_to_file, muscle_id_to_name, joint_id_to_name, actuator_id_to_name)


@dataclass
class ObjectData:
    identifier: int  # unique ID, should be same across frames
    obj_file: str
    pos: tuple
    rot: tuple
    scale: tuple

    def to_dict(self):
        return {
            "identifier": self.identifier,
            "obj_file": self.obj_file,
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
        }


@dataclass
class MuscleData:
    name: str
    num_points: int
    points: list

    max_isometric_force: float
    activation: float
    excitation: float
    actuation: float

    path_length: float
    path_velocity: float
    fiber_length: float
    fiber_velocity: float
    tendon_length: float
    tendon_velocity: float
    pennation_angle: float
    metabolic_power: float

    num_dofs: int
    moment_arms: list
    dof_interest: list

    def to_dict(self):
        return {
            "name": self.name,
            "num_points": self.num_points,
            "points": [list(point) for point in self.points],
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "excitation": self.excitation,
            "actuation": self.actuation,
            "path_length": self.path_length,
            "path_velocity": self.path_velocity,
            "fiber_length": self.fiber_length,
            "fiber_velocity": self.fiber_velocity,
            "tendon_length": self.tendon_length,
            "tendon_velocity": self.tendon_velocity,
            "pennation_angle": self.pennation_angle,
            "metabolic_power": self.metabolic_power,
            "num_dofs": self.num_dofs,
            "moment_arms": list(self.moment_arms),
            "dof_interest": list(self.dof_interest),
        }

    def to_anim_dict(self):
        return {
            "name": self.name,
            "num_points": self.num_points,
            "points": [list(point) for point in self.points],
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "path_length": self.path_length,
            "fiber_length": self.fiber_length,
            "tendon_length": self.tendon_length,
        }

    def to_data_dict(self):
        return {
            "name": self.name,
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "excitation": self.excitation,
            "actuation": self.actuation,
            "path_length": self.path_length,
            "path_velocity": self.path_velocity,
            "fiber_length": self.fiber_length,
            "fiber_velocity": self.fiber_velocity,
            "tendon_length": self.tendon_length,
            "tendon_velocity": self.tendon_velocity,
            "pennation_angle": self.pennation_angle,
            "metabolic_power": self.metabolic_power,
            "num_dofs": self.num_dofs,
            "moment_arms": list(self.moment_arms),
            "dof_interest": list(self.dof_interest),
        }


@dataclass
class ActuatorData:
    name: str
    force: float
    optimal_force: float
    activation: float
    excitation: float

    def to_dict(self):
        return {
            "name": self.name,
            "force": self.force,
            "optimal_force": self.optimal_force,
            "activation": self.activation,
            "excitation": self.excitation,
        }


@dataclass
class JointAngle:
    name: str
    value: float
    limited: bool
    range: tuple

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "limited": self.limited,
            "range": list(self.range),
        }


@dataclass
class JointVelocity:
    name: str
    value: float

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
        }

@dataclass
class KineticData:
    total_mass: float
    gravity: float
    grf: tuple
    com: tuple

    def to_dict(self):
        return {
            "total_mass": self.total_mass,
            "gravity": self.gravity,
            "grf": list(self.grf),
            "com": list(self.com),
        }

@dataclass
class FrameData:
    time: float
    visuals: list[ObjectData]
    colliders: list[ObjectData]
    muscles: list[MuscleData]
    actuators: list[ActuatorData]
    joint_angles: list[JointAngle]
    joint_velocities: list[JointVelocity]
    kinetic_data: KineticData
    reward_data: dict

    def to_dict(self):
        return {
            "time": self.time,
            "visuals": [obj.to_dict() for obj in self.visuals],
            "colliders": [obj.to_dict() for obj in self.colliders],
            "muscles": [muscle.to_dict() for muscle in self.muscles],
            "actuators": [actuator.to_dict() for actuator in self.actuators],
            "joint_angles": [j.to_dict() for j in self.joint_angles],
            "joint_velocities": [j.to_dict() for j in self.joint_velocities],
            "kinetic_data": self.kinetic_data.to_dict(),
            "reward_data": self.reward_data,
        }

    def to_data_dict(self):
        return {
            "time": self.time,
            "muscles": [muscle.to_data_dict() for muscle in self.muscles],
            "actuators": [actuator.to_dict() for actuator in self.actuators],
            "joint_angles": [j.to_dict() for j in self.joint_angles],
            "joint_velocities": [j.to_dict() for j in self.joint_velocities],
            "kinetic_data": self.kinetic_data.to_dict(),
            "reward_data": self.reward_data,
        }


def fetch_items(tensor, idx, count, dtype):
    """
    Gets `count` items with type `dtype` from a byte (uint8) tensor starting at
    index `idx`. returns the items and the updated index
    """
    n_bytes = dtype.itemsize
    if count > 1:
        vals = tensor[idx: idx + n_bytes * count].view(dtype=dtype).tolist()
    else:
        vals = tensor[idx: idx + n_bytes].view(dtype=dtype).item()
    idx += n_bytes * count
    return vals, idx


def parse_object_data(obj_tensor, identifier) -> ObjectData:
    idx = 0
    obj_id, idx = fetch_items(obj_tensor, idx, 1, torch.int32)
    pos, idx = fetch_items(obj_tensor, idx, 3, torch.float32)
    rot, idx = fetch_items(obj_tensor, idx, 4, torch.float32)
    scale, idx = fetch_items(obj_tensor, idx, 3, torch.float32)
    obj_data = ObjectData(
        identifier=identifier,
        obj_file=obj_id_to_file(obj_id),
        pos=tuple(pos),
        rot=tuple(rot),
        scale=tuple(scale))
    return obj_data

def parse_muscle_data(muscle_log) -> MuscleData:
    max_points = 8  # don't hardcode this

    idx = 0
    muscle_id, idx = fetch_items(muscle_log, idx, 1, torch.int32)
    num_points, idx = fetch_items(muscle_log, idx, 1, torch.int32)

    # Parse all muscle points
    points = []
    for i in range(num_points):
        point, idx = fetch_items(muscle_log, idx, 3, torch.float32)
        point = tuple(point)
        points.append(point)

    # Continue
    idx = 8 + max_points * 12
    max_isometric_force, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    activation, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    excitation, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    actuation, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    path_length, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    path_velocity, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    fiber_length, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    fiber_velocity, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    tendon_length, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    tendon_velocity, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    pennation_angle, idx = fetch_items(muscle_log, idx, 1, torch.float32)
    metabolic_power, idx = fetch_items(muscle_log, idx, 1, torch.float32)

    num_dofs, idx = fetch_items(muscle_log, idx, 1, torch.int32)
    moment_arms, idx = fetch_items(muscle_log, idx, num_dofs, torch.float32)
    moment_arms = [0.0 if (isinstance(ma, float) and (ma != ma)) else ma for ma in moment_arms]
    dof_interest, idx = fetch_items(muscle_log, idx, num_dofs, torch.bool)

    return MuscleData(name=muscle_id_to_name(muscle_id),
                      num_points=num_points,
                      points=points,
                      max_isometric_force=max_isometric_force,
                      activation=activation,
                      excitation=excitation,
                      actuation=actuation,
                      path_length=path_length,
                      path_velocity=path_velocity,
                      fiber_length=fiber_length,
                      fiber_velocity=fiber_velocity,
                      tendon_length=tendon_length,
                      tendon_velocity=tendon_velocity,
                      pennation_angle=pennation_angle,
                      metabolic_power=metabolic_power,
                      num_dofs=num_dofs,
                      moment_arms=moment_arms,
                      dof_interest=dof_interest)


def parse_actuator_data(actuator_tensor) -> ActuatorData:
    idx = 0
    actuator_id, idx = fetch_items(actuator_tensor, idx, 1, torch.int32)
    actuator_name = actuator_id_to_name(actuator_id)
    force, idx = fetch_items(actuator_tensor, idx, 1, torch.float32)
    optimal_force, idx = fetch_items(actuator_tensor, idx, 1, torch.float32)
    activation, idx = fetch_items(actuator_tensor, idx, 1, torch.float32)
    excitation, idx = fetch_items(actuator_tensor, idx, 1, torch.float32)
    return ActuatorData(
        name=actuator_name,
        force=force,
        optimal_force=optimal_force,
        activation=activation,
        excitation=excitation
    )


def parse_joint_angle(joint_tensor) -> list[JointAngle]:
    joints = []
    for i in range(joint_tensor.shape[0]):
        name = joint_id_to_name(i, True)

        joint_tensor_i = joint_tensor[i]
        idx = 0
        value, idx = fetch_items(joint_tensor_i, idx, 1, torch.float32)
        range_vals, idx = fetch_items(joint_tensor_i, idx, 2, torch.float32)
        limited, idx = fetch_items(joint_tensor_i, idx, 1, torch.bool)
        joints.append(JointAngle(name=name, value=value,
                                 limited=limited, range=tuple(range_vals)))
    return joints


def parse_joint_velocity(joint_tensor) -> list[JointVelocity]:
    joints = []
    for i in range(joint_tensor.shape[0]):
        name = joint_id_to_name(i, False)
        value = float(joint_tensor[i])
        joints.append(JointVelocity(name=name, value=value))
    return joints

def parse_kinetic_data(kinetic_tensor) -> KineticData:
    idx = 0
    total_mass, idx = fetch_items(kinetic_tensor, idx, 1, torch.float32)
    gravity, idx = fetch_items(kinetic_tensor, idx, 1, torch.float32)
    grf, idx = fetch_items(kinetic_tensor, idx, 3, torch.float32)
    com, idx = fetch_items(kinetic_tensor, idx, 3, torch.float32)
    return KineticData(
        total_mass=total_mass,
        gravity=gravity,
        grf=tuple(grf),
        com=tuple(com)
    )


def parse_frame(
        frame_time,
        visual_log,
        collider_log,
        muscle_log,
        actuator_log,
        joint_angle_log,
        joint_velocities,
        kinetic_log,
        reward_data
) -> FrameData:
    num_visuals = visual_log.shape[0]
    num_colliders = collider_log.shape[0]
    num_muscles = muscle_log.shape[0]
    num_actuators = actuator_log.shape[0]

    # Parse object visuals
    visuals = []
    for i in range(num_visuals):
        obj_data = parse_object_data(visual_log[i, :], i)
        visuals.append(obj_data)

    # Parse object colliders
    colliders = []
    for i in range(num_colliders):
        obj_data = parse_object_data(collider_log[i, :], i)
        colliders.append(obj_data)

    # Parse muscles
    muscles = []
    for i in range(num_muscles):
        muscle_data = parse_muscle_data(muscle_log[i, :])
        muscles.append(muscle_data)

    # Parse actuators
    actuators = []
    for i in range(num_actuators):
        actuator_data = parse_actuator_data(actuator_log[i, :])
        actuators.append(actuator_data)

    # Parse joint angles and velocities
    joint_angles = parse_joint_angle(joint_angle_log)
    joint_velocities = parse_joint_velocity(joint_velocities)

    # Parse kinetic data
    kinetic_data = parse_kinetic_data(kinetic_log)

    frame_visuals = FrameData(
        time=frame_time,
        visuals=visuals,
        colliders=colliders,
        muscles=muscles,
        actuators=actuators,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        kinetic_data=kinetic_data,
        reward_data=reward_data
    )

    return frame_visuals
