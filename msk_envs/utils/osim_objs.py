from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum


@dataclass
class Vector2:
    x: float
    y: float


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Vector6:
    v0: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float


@dataclass
class Inertia:
    xx: float
    yy: float
    zz: float
    xy: float
    xz: float
    yz: float


@dataclass
class BodyMass:
    mass: float
    mass_center: Vector3
    inertia: Inertia


@dataclass
class DisplayGeometry:
    geometry_file: str
    color: Vector3
    texture_file: Optional[str]
    transform: Vector6
    scale_factors: Vector3


@dataclass
class VisibleObject:
    geometry_set: list[DisplayGeometry]
    scale_factors: Vector3
    transform: Vector6


class MotionType(Enum):
    ROTATIONAL = "rotational"
    TRANSLATIONAL = "translational"


@dataclass
class Joint:
    name: str


@dataclass
class Coordinate:
    name: str
    motion_type: MotionType
    default_value: float
    default_speed_value: float
    range: Vector2
    clamped: bool
    locked: bool


@dataclass
class CoordinateSet:
    coordinates: OrderedDict[str, Coordinate]


@dataclass
class Function:
    def scale(self, factor: float):
        pass


@dataclass
class LinearFunction(Function):
    coefficients: Vector2

    def scale(self, factor: float):
        self.coefficients.x *= factor
        self.coefficients.y *= factor


@dataclass
class ConstantFunction(Function):
    value: float

    def scale(self, factor: float):
        self.value *= factor


@dataclass
class TransformAxis:
    name: str
    coordinates: str
    axis: Vector3
    function: Function


@dataclass
class SpatialTransform:
    transform_axes: list[TransformAxis]


@dataclass
class CustomJoint(Joint):
    name: str
    parent_body: str
    location_in_parent: Vector3
    orientation_in_parent: Vector3
    location: Vector3
    orientation: Vector3
    coordinate_set: CoordinateSet
    spatial_transform: SpatialTransform


@dataclass
class Body:
    name: str
    body_mass: BodyMass
    visible_object: VisibleObject
    joint: Optional[Joint] = None


@dataclass
class BodySet:
    bodies: OrderedDict[str, Body]


@dataclass
class Marker:
    name: str
    body: str
    location: Vector3
    fixed: bool


@dataclass
class MarkerSet:
    markers: OrderedDict[str, Marker]


@dataclass
class ContactHalfSpace:
    name: str
    body_name: str
    location: Vector3
    orientation: Vector3


@dataclass
class ContactSphere:
    name: str
    body_name: str
    location: Vector3
    orientation: Vector3
    radius: float


@dataclass
class ContactGeometrySet:
    contact_half_spaces: OrderedDict[str, ContactHalfSpace]
    contact_spheres: OrderedDict[str, ContactSphere]


@dataclass
class PathPoint:
    name: str
    body: str
    location: Vector3


@dataclass
class ConditionalPathPoint(PathPoint):
    range: Vector2
    coordinate: str


@dataclass
class PathPointSet:
    path_points: OrderedDict[str, PathPoint]


@dataclass
class GeometryPath:
    path_point_set: PathPointSet
    # path_wrap_set
    # visible_object


@dataclass
class Muscle:
    name: str
    geometry_path: GeometryPath
    max_isometric_force: float
    optimal_fiber_length: float
    tendon_slack_length: float
    pennation_angle_at_optimal: float

@dataclass
class Actuator:
    name: str
    optimal_force: float
    coordinate: str

@dataclass
class ForceSet:
    muscles: OrderedDict[str, Muscle]
    actuators: OrderedDict[str, Actuator]


@dataclass
class Model:
    body_set: BodySet
    force_set: ForceSet
    marker_set: MarkerSet
    contact_geometry_set: ContactGeometrySet
