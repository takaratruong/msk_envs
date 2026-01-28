import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ColliderData:
    name: str
    geom_type: int
    pos: list[float]
    rot: list[float]
    scale: list[float]

    def to_dict(self):
        return {
            "name": self.name,
            "geom_type": int(self.geom_type),
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
        }

    def to_anim_dict(self):
        return self.to_dict()

    @staticmethod
    def from_dict(data: dict) -> 'ColliderData':
        return ColliderData(
            name=data["name"],
            geom_type=data["geom_type"],
            pos=data["pos"],
            rot=data["rot"],
            scale=data["scale"],
        )


@dataclass
class VisualData:
    mesh_file: str
    pos: list[float]
    rot: list[float]
    scale: list[float]
    opacity: float = 1.0

    def to_dict(self):
        # Remove .vtp if it exists and replace with .obj
        mesh_obj_file = self.mesh_file
        if mesh_obj_file.endswith('.vtp'):
            mesh_obj_file = self.mesh_file[:-4] + '.obj'
        mesh_obj_file = os.path.join("assets", "geometry", "obj", mesh_obj_file)

        return {
            "mesh_file": mesh_obj_file,
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
            "opacity": self.opacity,
        }

    def to_anim_dict(self):
        return self.to_dict()

    @staticmethod
    def from_dict(data: dict) -> 'VisualData':
        return VisualData(
            mesh_file=data["mesh_file"],
            pos=data["pos"],
            rot=data["rot"],
            scale=data["scale"],
            opacity=data.get("opacity", 1.0),
        )


@dataclass
class MuscleData:
    name: str
    points: list

    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float

    activation: float
    excitation: float
    actuation: float

    path_length: float
    path_velocity: float
    fiber_length: float
    fiber_velocity: float
    tendon_length: float
    pennation_angle: float

    moment_arm: list[float]

    def to_dict(self):
        return {
            "name": self.name,
            "points": self.points,
            "max_isometric_force": self.max_isometric_force,
            "optimal_fiber_length": self.optimal_fiber_length,
            "tendon_slack_length": self.tendon_slack_length,
            "activation": self.activation,
            "excitation": self.excitation,
            "actuation": self.actuation,
            "path_length": self.path_length,
            "path_velocity": self.path_velocity,
            "fiber_length": self.fiber_length,
            "fiber_velocity": self.fiber_velocity,
            "tendon_length": self.tendon_length,
            "pennation_angle": self.pennation_angle,
            "moment_arm": self.moment_arm,
        }

    def to_anim_dict(self):
        return {
            "name": self.name,
            "points": self.points,
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "excitation": self.excitation,
            "path_length": self.path_length,
            "fiber_length": self.fiber_length,
            "tendon_length": self.tendon_length,
        }

    @staticmethod
    def from_dict(data: dict) -> 'MuscleData':
        return MuscleData(
            name=data["name"],
            points=data["points"],
            max_isometric_force=data["max_isometric_force"],
            optimal_fiber_length=data["optimal_fiber_length"],
            tendon_slack_length=data["tendon_slack_length"],
            activation=data["activation"],
            excitation=data["excitation"],
            actuation=data["actuation"],
            path_length=data["path_length"],
            path_velocity=data["path_velocity"],
            fiber_length=data["fiber_length"],
            fiber_velocity=data["fiber_velocity"],
            tendon_length=data["tendon_length"],
            pennation_angle=data["pennation_angle"],
            moment_arm=data["moment_arm"],
        )


@dataclass
class ActuatorData:
    name: str

    optimal_force: float
    activation: float
    excitation: float

    def to_dict(self):
        return {
            "name": self.name,
            "optimal_force": self.optimal_force,
            "activation": self.activation,
            "excitation": self.excitation,
        }

    @staticmethod
    def from_dict(data: dict) -> 'ActuatorData':
        return ActuatorData(
            name=data["name"],
            optimal_force=data["optimal_force"],
            activation=data["activation"],
            excitation=data["excitation"],
        )


@dataclass
class KineticData:
    com: tuple
    grf: tuple
    total_mass: float
    gravity: float

    def to_dict(self):
        return {
            "com": list(self.com),
            "grf": list(self.grf),
            "total_mass": self.total_mass,
            "gravity": self.gravity,
        }

    @staticmethod
    def from_dict(data: dict) -> 'KineticData':
        return KineticData(
            com=tuple(data["com"]),
            grf=tuple(data["grf"]),
            total_mass=data["total_mass"],
            gravity=data["gravity"],
        )


@dataclass
class NamedValue:
    name: str
    value: float
    reference: Optional[float] = None
    limits: Optional[tuple[float, float]] = None

    def has_reference(self) -> bool:
        return self.reference is not None

    def has_limits(self) -> bool:
        return self.limits is not None

    def to_dict(self):
        ret = {
            "name": self.name,
            "value": self.value,
        }
        if self.reference is not None:
            ret["reference"] = self.reference
        if self.limits is not None:
            ret["limits"] = list(self.limits)
        return ret

    @staticmethod
    def from_dict(data: dict) -> 'NamedValue':
        return NamedValue(
            name=data["name"],
            value=data["value"],
            reference=data.get("reference"),
            limits=tuple(data["limits"]) if "limits" in data else None,
        )


@dataclass
class JointMoment:
    name: str
    value: float
    spring: float
    damping: float
    bias: float
    drag: float
    muscle: float
    actuator: float
    limit: float
    contact: float

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "spring": self.spring,
            "damping": self.damping,
            "drag": self.drag,
            "bias": self.bias,
            "muscle": self.muscle,
            "actuator": self.actuator,
            "limit": self.limit,
            "contact": self.contact,
        }

    @staticmethod
    def from_dict(data: dict) -> 'JointMoment':
        return JointMoment(
            name=data["name"],
            value=data["value"],
            spring=data["spring"],
            damping=data["damping"],
            drag=data["drag"],
            bias=data["bias"],
            muscle=data["muscle"],
            limit=data["limit"],
            actuator=data["actuator"],
            contact=data["contact"],
        )


@dataclass
class Arrow:
    start: list[float]
    direction: list[float]

    def to_dict(self):
        return {
            "start": list(self.start),
            "direction": list(self.direction),
        }

    @staticmethod
    def from_dict(data: dict) -> 'Arrow':
        return Arrow(
            start=data["start"],
            direction=data["direction"],
        )


@dataclass
class FrameData:
    time: float
    visuals: list[VisualData]
    colliders: list[ColliderData]
    joint_angles: list[NamedValue]
    joint_velocities: list[NamedValue]
    joint_moments: list[JointMoment]
    muscles: list[MuscleData]
    actuators: list[ActuatorData]
    kinetic_data: KineticData
    arrows: list[Arrow]
    reward_data: dict

    def to_dict(self):
        return {
            "time": self.time,
            "visuals": [obj.to_dict() for obj in self.visuals],
            "colliders": [obj.to_dict() for obj in self.colliders],
            "joint_angles": [angle.to_dict() for angle in self.joint_angles],
            "joint_velocities": [vel.to_dict() for vel in self.joint_velocities],
            "joint_moments": [moment.to_dict() for moment in self.joint_moments],
            "muscles": [muscle.to_dict() for muscle in self.muscles],
            "actuators": [actuator.to_dict() for actuator in self.actuators],
            "kinetic_data": self.kinetic_data.to_dict(),
            "arrows": [arrow.to_dict() for arrow in self.arrows],
            "reward_data": self.reward_data,
        }

    @staticmethod
    def from_dict(data: dict) -> 'FrameData':
        return FrameData(
            time=data["time"],
            visuals=[VisualData.from_dict(obj) for obj in data["visuals"]],
            colliders=[ColliderData.from_dict(obj) for obj in data["colliders"]],
            joint_angles=[NamedValue.from_dict(angle) for angle in data["joint_angles"]],
            joint_velocities=[NamedValue.from_dict(vel) for vel in data["joint_velocities"]],
            joint_moments=[JointMoment.from_dict(moment) for moment in data["joint_moments"]],
            muscles=[MuscleData.from_dict(muscle) for muscle in data["muscles"]],
            actuators=[ActuatorData.from_dict(actuator) for actuator in data["actuators"]],
            kinetic_data=KineticData.from_dict(data["kinetic_data"]),
            arrows=[Arrow.from_dict(arrow) for arrow in data["arrows"]],
            reward_data=data["reward_data"],
        )
