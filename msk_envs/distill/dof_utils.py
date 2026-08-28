import torch

ROOT_DOFS = ("pelvis_tilt","pelvis_list","pelvis_rotation","pelvis_tx","pelvis_ty","pelvis_tz")
EXCLUDED_DOFS = ROOT_DOFS + ("__NO_DOF",)  # Exclude root and sentinel

def joint_dof_indices(env):
    """Indices into the nv-dim ufrc/dof arrays for the 25 non-root joint DoFs, in a stable order."""
    lu = env.dof_id_lookup                      # name -> index
    names = [n for n in lu if n not in EXCLUDED_DOFS]
    names.sort(key=lambda n: lu[n])             # stable: sim dof order
    idx = [lu[n] for n in names]
    return names, idx


def build_actuator_perm(env, names, device):
    """Permutation routing a per-DoF vector in `names` order into the env's actuator-slice order.

    Labels (Teacher.torque_label), optimal_force and the Student's output are ALL kept in `names`
    order (joint_dof_indices, sorted by sim DoF index). The env's coordinate-actuator slice, however,
    is ordered by `actuator_id_lookup`, which is a DIFFERENT order (for SprinterTorque only 4 of 25
    columns coincide). Writing a `names`-ordered vector straight into the actuator slice would send
    each torque to the wrong joint. Use this permutation only at the final env write:
        a[:, env.num_muscles:] = raw.index_select(1, perm)
    so that for actuator slot p, perm[p] is the `names`-column whose torque belongs on that actuator.

    Do NOT use this to permute labels or optimal_force — those stay in `names` order everywhere.
    Returns a LongTensor of length len(names). Asserts the mapping is a bijection.
    """
    slot_names = [None] * env.num_actuators
    for k, v in env.actuator_id_lookup.items():
        slot_names[v] = k[len("act_"):]         # strip the "act_" prefix -> joint-DoF name
    name_to_col = {n: i for i, n in enumerate(names)}
    perm = [name_to_col[sn] for sn in slot_names]
    assert sorted(perm) == list(range(len(names))), "actuator<->name permutation is not a bijection"
    return torch.tensor(perm, device=device, dtype=torch.long)


def build_teacher_obs_cols(env, device):
    """Column indices mapping the 359-dim SprinterTorque obs -> 336-dim base-Sprinter obs.

    The teacher checkpoint is a *base* Sprinter policy (obs dim 336, action dim 138 = 136 muscle
    excitations + 2 mtp-motor excitations). The SprinterTorque twin emits obs dim 359, because its
    obs layout carries a larger actuator-activation block:
        [ cmd(2) | muscle_act(136) | muscle_fiber(136) | actuator_act(N) | qpos(29) | qvel(31) ]
    where N = 2 in the base env (mtp_angle_r_motor, mtp_angle_l_motor) but N = 25 in the twin
    (the 25 coordinate actuators). Every other block is identical between the two models.

    So to feed the teacher, we keep everything EXCEPT the 25-actuator block, and re-insert just the
    two mtp-motor columns (in the base env's order: act_mtp_angle_r, act_mtp_angle_l). This drops the
    23 non-mtp actuator-activation columns that the base teacher never saw. The kept columns are:
        - cmd(2) + muscle_act(136) + muscle_fiber(136)  (everything before the actuator block)
        - act_mtp_angle_r, act_mtp_angle_l              (the 2 columns the teacher expects)
        - qpos(29) + qvel(31)                           (everything after the actuator block)
    Total = 274 + 2 + 60 = 336.

    Returns a LongTensor of length 336. Asserts the result length == 336.
    """
    cmd_dim = 2
    act_start = cmd_dim + 2 * env.num_muscles
    tail_start = act_start + env.num_actuators
    al = env.actuator_id_lookup
    mtp_cols = [act_start + al["act_mtp_angle_r"], act_start + al["act_mtp_angle_l"]]
    obs_dim = env._get_obs().shape[1]
    cols = list(range(act_start)) + mtp_cols + list(range(tail_start, obs_dim))
    assert len(cols) == 336, f"teacher obs adapter produced {len(cols)} cols, expected 336"
    return torch.tensor(cols, device=device, dtype=torch.long)
