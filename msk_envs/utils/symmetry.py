"""Left/right mirroring of StoneCourse observations and actions.

The course runs along +X with Y up; the mirror plane is z = 0. On the
symmetrized sprinter model (`sprinter_model_sym.osim`) a mirrored transition
is physically valid training data: model, reward, and course distribution are
invariant under this reflection, so every collected transition can be flipped
and stored as if it had been experienced.

Verified by the mirrored-rollout check in experiments/stone_course/symmetry.py:
course terrain, muscle activations, actuator activations, qpos, qvel, and
actions all round-trip at exactly 0.0 error. Known residual: Bolt's iterative
Scholz geodesic solver leaves ~3% fiber-length differences on the 18 curved-
path arm muscles (see experiments/stone_course/SYMMETRY_NOTES.md).
"""

import dataclasses

import torch

from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX


def _other_side(name: str) -> str | None:
    if name.endswith("_r"):
        return name[:-2] + "_l"
    if name.endswith("_l"):
        return name[:-2] + "_r"
    return None


def _permutation_and_signs(
    id_lookup: dict[str, int],
    width: int,
    negate_names: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column permutation (L<->R swap) and per-column signs for one block."""
    perm = torch.arange(width)
    signs = torch.ones(width)
    for name, idx in id_lookup.items():
        if idx >= width or idx < 0:
            continue  # sentinel entries such as __NO_DOF
        partner = _other_side(name)
        if partner is not None and partner in id_lookup:
            partner_idx = id_lookup[partner]
            if 0 <= partner_idx < width:
                perm[idx] = partner_idx
        if name in negate_names:
            signs[idx] = -1.0
    return perm, signs


# Unpaired coordinates whose positive direction is lateral or about a
# lateral-flipping axis. Root quaternion components qx, qy negate under a
# z = 0 reflection (stored in qpos as pelvis_tilt/list/rotation/quat_w).
QPOS_NEGATE = {"pelvis_tilt", "pelvis_list", "pelvis_tz",
               "torso_bending", "torso_rotation"}
QVEL_NEGATE = {"pelvis_tilt", "pelvis_list", "pelvis_tz",
               "torso_bending", "torso_rotation"}


@dataclasses.dataclass
class MirrorSpec:
    """Precomputed index/sign tensors that mirror obs and action columns."""

    obs_perm: torch.Tensor
    obs_signs: torch.Tensor
    act_perm: torch.Tensor
    act_signs: torch.Tensor

    def flip_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.obs_perm] * self.obs_signs

    def flip_action(self, action: torch.Tensor) -> torch.Tensor:
        return action[:, self.act_perm] * self.act_signs

    def to(self, device) -> "MirrorSpec":
        return MirrorSpec(
            self.obs_perm.to(device), self.obs_signs.to(device),
            self.act_perm.to(device), self.act_signs.to(device),
        )


def build_stone_course_mirror_spec(env) -> MirrorSpec:
    """Derive the mirror permutation for StoneCourseEnv's observation layout.

    Layout (StoneCourseEnv._get_obs):
      [course (5 * lookahead)] [muscle_activations] [muscle_fiber_lengths]
      [actuator_activations] [qpos minus pelvis_tx] [qvel]
      [command_speed (optional, 1)]
    """
    lookahead = env.course.lookahead
    n_muscles = env.muscle_activations.shape[1]
    n_actuators = env.actuator_activations.shape[1]
    n_qpos = env.joint_positions.shape[1]
    n_qvel = env.joint_velocities.shape[1]
    has_command = env.command_speed_range != (0.0, 0.0)

    course_n = 5 * lookahead
    total = course_n + 2 * n_muscles + n_actuators + (n_qpos - 1) + n_qvel
    total += 1 if has_command else 0

    perm = torch.arange(total)
    signs = torch.ones(total)
    offset = 0

    # Course block: [x0 z0 ... x(L-1) z(L-1)] [h0..h(L-1)] [r0 p0 ... r p].
    for i in range(lookahead):
        signs[offset + 2 * i + 1] = -1.0        # relative z negates
    tilt_base = offset + 3 * lookahead
    for i in range(lookahead):
        signs[tilt_base + 2 * i] = -1.0         # roll about X negates
    offset += course_n

    muscle_perm, _ = _permutation_and_signs(env.muscle_id_lookup, n_muscles, set())
    for block in range(2):                       # activations, fiber lengths
        base = offset + block * n_muscles
        perm[base:base + n_muscles] = muscle_perm + base
    offset += 2 * n_muscles

    act_perm, _ = _permutation_and_signs(env.actuator_id_lookup, n_actuators, set())
    perm[offset:offset + n_actuators] = act_perm + offset
    offset += n_actuators

    # qpos without pelvis_tx: build on the full qpos then delete the tx row.
    qpos_perm, qpos_signs = _permutation_and_signs(
        env.qpos_id_lookup, n_qpos, QPOS_NEGATE)
    tx = env.qpos_id_lookup["pelvis_tx"]
    keep = [i for i in range(n_qpos) if i != tx]
    remap = {old: new for new, old in enumerate(keep)}
    for new_idx, old_idx in enumerate(keep):
        perm[offset + new_idx] = offset + remap[int(qpos_perm[old_idx])]
        signs[offset + new_idx] = qpos_signs[old_idx]
    offset += n_qpos - 1

    qvel_perm, qvel_signs = _permutation_and_signs(
        env.dof_id_lookup, n_qvel, QVEL_NEGATE)
    perm[offset:offset + n_qvel] = qvel_perm + offset
    signs[offset:offset + n_qvel] = qvel_signs
    offset += n_qvel

    if has_command:
        offset += 1                              # command speed is invariant

    obs_width = env._get_obs().shape[1]
    if offset != obs_width:
        raise ValueError(f"mirror spec covers {offset} columns, obs has {obs_width}")

    # Actions: [muscle excitations][actuator excitations], pure permutation.
    n_act = n_muscles + n_actuators
    action_perm = torch.arange(n_act)
    action_perm[:n_muscles] = muscle_perm
    action_perm[n_muscles:] = act_perm + n_muscles
    return MirrorSpec(perm, signs, action_perm, torch.ones(n_act)).to(env.device)


def flip_qpos(qpos: torch.Tensor, env) -> torch.Tensor:
    perm, signs = _permutation_and_signs(
        env.qpos_id_lookup, qpos.shape[1], QPOS_NEGATE)
    return qpos[:, perm.to(qpos.device)] * signs.to(qpos.device)


def flip_qvel(qvel: torch.Tensor, env) -> torch.Tensor:
    perm, signs = _permutation_and_signs(
        env.dof_id_lookup, qvel.shape[1], QVEL_NEGATE)
    return qvel[:, perm.to(qvel.device)] * signs.to(qvel.device)


def mirror_world_layout(env, source_world: int, target_world: int) -> None:
    """Make target_world's slabs the z-mirror of source_world's."""
    positions = env.stone_positions[source_world].clone()
    positions[:, SIDE_IDX] *= -1.0
    tilts = env.stone_surface_tilts[source_world].clone()
    tilts[:, 0] *= -1.0                          # roll about X negates
    world_ids = torch.tensor([target_world], device=env.device)
    env._set_course_layout(world_ids, positions.unsqueeze(0), tilts.unsqueeze(0))
    env.next_lateral_sign[target_world] = -env.next_lateral_sign[source_world]
