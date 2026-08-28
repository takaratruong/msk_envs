ROOT_DOFS = ("pelvis_tilt","pelvis_list","pelvis_rotation","pelvis_tx","pelvis_ty","pelvis_tz")
EXCLUDED_DOFS = ROOT_DOFS + ("__NO_DOF",)  # Exclude root and sentinel

def joint_dof_indices(env):
    """Indices into the nv-dim ufrc/dof arrays for the 25 non-root joint DoFs, in a stable order."""
    lu = env.dof_id_lookup                      # name -> index
    names = [n for n in lu if n not in EXCLUDED_DOFS]
    names.sort(key=lambda n: lu[n])             # stable: sim dof order
    idx = [lu[n] for n in names]
    return names, idx
