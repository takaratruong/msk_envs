import msk_warp
import torch

from msk_envs.utils.frame_data import *


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
        muscle_idx_to_name: dict[int, str],
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
    for i in range(msk_warp.get_num_muscles(m)):
        pt_adr = muscle_site_adr[i]
        n_pts = muscle_site_num[i]
        muscle_data = MuscleData(
            name=muscle_idx_to_name[i],
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


def parse_actuator_data(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        actuation_idx_to_name: dict[int, str],
        world_id: int
) -> list[ActuatorData]:
    actuator_activations = msk_warp.actuator_activations(d)
    actuator_excitations = msk_warp.actuator_excitations(d)
    actuator_metadata = msk_warp.actuator_metadata_np(m)

    actuators = []
    for i in range(msk_warp.get_num_actuators(m)):
        actuator_data = ActuatorData(
            name=actuation_idx_to_name[i],
            optimal_force=float(actuator_metadata["optimal_force"][i]),
            activation=float(actuator_activations[world_id][i].item()),
            excitation=float(actuator_excitations[world_id][i].item()),
        )
        actuators.append(actuator_data)
    return actuators


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
        total_mass=float(mass),
        gravity=gravity,
    )
    return kinetic_data


def find_index_1d(tensor, x):
    idx = torch.where(tensor == x)[0]
    return idx[0].item() if idx.numel() > 0 else None


def parse_joint_angles(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        qpos_idx_to_name: dict[int, str],
        world_id: int,
        ref_joint_angles: torch.Tensor | None
) -> list[NamedValue]:
    joint_angles = msk_warp.joint_positions(d)
    joint_limit_ranges = msk_warp.joint_limit_ranges(m)
    joint_limit_qadr = list(msk_warp.joint_limit_qadr(m))
    angles = []

    for i in range(msk_warp.get_num_qpos(m)):
        reference = None if ref_joint_angles is None else float(ref_joint_angles[world_id][i].item())
        limits = None
        limit_id = find_index_1d(torch.tensor(joint_limit_qadr), i)
        if limit_id is not None:
            limits = (
                float(joint_limit_ranges[limit_id, 0]),
                float(joint_limit_ranges[limit_id, 1]),
            )

        angle = NamedValue(
            name=qpos_idx_to_name[i],
            value=float(joint_angles[world_id][i].item()),
            reference=reference,
            limits=limits,
        )
        angles.append(angle)
    return angles


def parse_joint_velocities(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        world_id: int
) -> list[NamedValue]:
    joint_velocities = msk_warp.joint_velocities(d)
    velocities = []  # todo
    return velocities


def parse_joint_moments(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        dof_idx_to_name: dict[int, str],
        world_id: int
) -> list[JointMoment]:
    joint_moments = msk_warp.joint_moments(d)
    qfrc_spring = msk_warp.qfrc_spring(d)
    qfrc_damper = msk_warp.qfrc_damper(d)
    qfrc_bias = msk_warp.qfrc_bias(d)
    qfrc_muscle = msk_warp.qfrc_muscle(d)
    qfrc_actuator = msk_warp.qfrc_actuator(d)
    qfrc_limit = msk_warp.qfrc_limit(d)

    qv = msk_warp.joint_velocities(d)
    moments = []
    for i in range(msk_warp.get_num_dofs(m)):
        angle = JointMoment(
            name=dof_idx_to_name[i],
            value=float(joint_moments[world_id][i].item()),
            spring=float(qfrc_spring[world_id][i].item()),
            damping=float(qfrc_damper[world_id][i].item()),
            bias=float(qfrc_bias[world_id][i].item()),
            muscle=float(qfrc_muscle[world_id][i].item()),
            actuator=float(qfrc_actuator[world_id][i].item()),
            limit=float(qfrc_limit[world_id][i].item()),
        )
        moments.append(angle)
    return moments


def parse_frame(
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        qpos_idx_to_name: dict[int, str],
        dof_idx_to_name: dict[int, str],
        muscle_idx_to_name: dict[int, str],
        actuation_idx_to_name: dict[int, str],
        visual_load_results: list[msk_warp.types.MeshLoadResult],
        world_id: int,
        frame_time: float,
        reward_data: dict,
        ref_joint_angles=None,
) -> FrameData:
    visuals = parse_visual_data(m, d, visual_load_results, world_id)
    colliders = parse_collider_data(m, d, world_id)
    muscles = parse_muscle_data(m, d, muscle_idx_to_name, world_id)
    actuators = parse_actuator_data(m, d, actuation_idx_to_name, world_id)
    kinetic_data = parse_kinetic_data(m, d, world_id)
    joint_angles = parse_joint_angles(m, d, qpos_idx_to_name, world_id, ref_joint_angles)
    joint_velocities = parse_joint_velocities(m, d, world_id)
    joint_moments = parse_joint_moments(m, d, dof_idx_to_name, world_id)

    frame_visuals = FrameData(
        time=frame_time,
        visuals=visuals,
        colliders=colliders,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        joint_moments=joint_moments,
        muscles=muscles,
        actuators=actuators,
        kinetic_data=kinetic_data,
        arrows=[],
        reward_data=reward_data
    )

    return frame_visuals


def add_reference_visuals(
        frame: FrameData,
        ref_visuals_pos: list,
        ref_visuals_rot: list,
):
    for i in range(len(frame.visuals)):
        ref_pos = ref_visuals_pos[i].tolist()
        ref_rot = ref_visuals_rot[i].tolist()
        ref_visual = VisualData(
            mesh_file=frame.visuals[i].mesh_file,
            pos=ref_pos,
            rot=ref_rot,
            scale=frame.visuals[i].scale,
            opacity=0.3
        )
        frame.visuals.append(ref_visual)


def add_ext_forces_to_frame(
        frame: FrameData,
        m: msk_warp.types.Model,
        d: msk_warp.types.Data,
        idx_world: int
):
    num_bodies = msk_warp.get_num_bodies(m)
    body_positions = msk_warp.body_positions(d)
    ext_forces = msk_warp.body_user_forces(d)
    for i in range(1, num_bodies):
        force = ext_forces[idx_world][i][0:3].tolist()
        point = body_positions[idx_world][i].tolist()
        arrow = Arrow(
            start=point,
            direction=force
        )
        frame.arrows.append(arrow)
