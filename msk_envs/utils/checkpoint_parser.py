import msk_warp
import torch
import os

from dataclasses import dataclass


@dataclass
class ColliderData:
    geom_type: int
    pos: list[float]
    rot: list[float]
    scale: list[float]

    def to_dict(self):
        return {
            "geom_type": int(self.geom_type),
            "pos": list(self.pos),
            "rot": list(self.rot),
            "scale": list(self.scale),
        }


@dataclass
class VisualData:
    mesh_file: str
    pos: list[float]
    rot: list[float]
    scale: list[float]

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
        }


@dataclass
class MuscleData:
    name: str
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
    pennation_angle: float

    def to_dict(self):
        return {
            "name": self.name,
            "points": self.points,
            "max_isometric_force": self.max_isometric_force,
            "activation": self.activation,
            "excitation": self.excitation,
            "actuation": self.actuation,
            "path_length": self.path_length,
            "path_velocity": self.path_velocity,
            "fiber_length": self.fiber_length,
            "fiber_velocity": self.fiber_velocity,
            "tendon_length": self.tendon_length,
            "pennation_angle": self.pennation_angle,
        }


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


@dataclass
class FrameData:
    time: float
    visuals: list[VisualData]
    colliders: list[ColliderData]
    muscles: list[MuscleData]
    kinetic_data: KineticData
    reward_data: dict

    def to_dict(self):
        return {
            "time": self.time,
            "visuals": [obj.to_dict() for obj in self.visuals],
            "colliders": [obj.to_dict() for obj in self.colliders],
            "muscles": [muscle.to_dict() for muscle in self.muscles],
            "kinetic_data": self.kinetic_data.to_dict(),
            "reward_data": self.reward_data,
        }


def parse_visual_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        visual_load_results: list[msk_warp.types.MeshLoadResult],
        world_id: int
) -> list[VisualData]:
    visual_positions = msk_warp.get_visual_positions(d)
    visual_rotations = msk_warp.get_visual_rotations(d)

    visuals = []
    for i in range(msk_warp.get_num_visuals(m)):
        visual_load = visual_load_results[i]
        visual_data = VisualData(
            mesh_file=visual_load.file,
            pos=visual_positions[world_id][i].tolist(),
            rot=visual_rotations[world_id][i].tolist(),
            scale=visual_load.scale,
        )
        visuals.append(visual_data)
    return visuals


def parse_collider_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> list[ColliderData]:
    collider_types = msk_warp.get_collider_types(m)
    collider_scales = msk_warp.get_collider_sizes(m)
    collider_positions = msk_warp.get_collider_positions(d)
    collider_rotations = msk_warp.get_collider_rotations(d)

    colliders = []
    for i in range(msk_warp.get_num_colliders(m)):
        collider_data = ColliderData(
            geom_type=int(collider_types[i]),
            pos=collider_positions[world_id][i].tolist(),
            rot=collider_rotations[world_id][i].tolist(),
            scale=collider_scales[i].tolist(),
        )
        colliders.append(collider_data)
    return colliders


def parse_muscle_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        muscle_id_lookup: dict[str, int],
        world_id: int
) -> list[MuscleData]:
    muscle_activations = msk_warp.muscle_activations(d)
    muscle_excitations = msk_warp.muscle_excitations(d)
    muscle_actuations = msk_warp.muscle_actuations(d)
    muscle_path_lengths = msk_warp.muscle_path_lengths(d)
    muscle_path_velocities = msk_warp.muscle_path_velocities(d)
    muscle_fiber_lengths = msk_warp.muscle_fiber_lengths(d)
    muscle_fiber_velocities = msk_warp.muscle_fiber_velocities(d)

    muscle_metadata = msk_warp.muscle_metadata_np(m)
    muscle_length_info = msk_warp.muscle_length_info_np(d)

    site_positions = msk_warp.site_positions(d)
    muscle_site_adr = msk_warp.muscle_site_adr(m)
    muscle_site_num = msk_warp.muscle_site_num(m)

    muscles = []
    id_to_muscle = {v: k for k, v in muscle_id_lookup.items()}
    for i in range(msk_warp.get_num_muscles(m)):
        pt_adr = muscle_site_adr[i]
        n_pts = muscle_site_num[i]
        muscle_data = MuscleData(
            name=id_to_muscle[i],
            points=site_positions[world_id][pt_adr:pt_adr + n_pts].tolist(),
            max_isometric_force=float(muscle_metadata["max_isometric_force"][i]),
            activation=float(muscle_activations[world_id][i].item()),
            excitation=float(muscle_excitations[world_id][i].item()),
            actuation=float(muscle_actuations[world_id][i].item()),
            path_length=float(muscle_path_lengths[world_id][i].item()),
            path_velocity=float(muscle_path_velocities[world_id][i].item()),
            fiber_length=float(muscle_fiber_lengths[world_id][i].item()),
            fiber_velocity=float(muscle_fiber_velocities[world_id][i].item()),
            tendon_length=float(muscle_length_info["tendon_length"][world_id][i].item()),
            pennation_angle=float(muscle_length_info["pennation_angle"][world_id][i].item()),
        )
        muscles.append(muscle_data)
    return muscles


def parse_kinetic_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> KineticData:
    com = msk_warp.subtree_com_positions(d)[world_id][1].tolist()  # why am I hardcoded!
    grf = msk_warp.grf(d)[world_id].tolist()
    mass = msk_warp.subtree_mass(m)[1]
    gravity = msk_warp.gravity(m)
    kinetic_data = KineticData(
        com=tuple(com),
        grf=tuple(grf),
        total_mass=mass,
        gravity=gravity,
    )
    return kinetic_data


def parse_frame(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        muscle_id_lookup: dict[str, int],
        visual_load_results: list[msk_warp.types.MeshLoadResult],
        world_id: int,
        frame_time,
        reward_data
) -> FrameData:
    visuals = parse_visual_data(m, d, visual_load_results, world_id)
    colliders = parse_collider_data(m, d, world_id)
    muscles = parse_muscle_data(m, d, muscle_id_lookup, world_id)
    kinetic_data = parse_kinetic_data(m, d, world_id)

    frame_visuals = FrameData(
        time=frame_time,
        visuals=visuals,
        colliders=colliders,
        muscles=muscles,
        kinetic_data=kinetic_data,
        reward_data=reward_data
    )

    return frame_visuals
