import yaml
import msk_warp
from dataclasses import dataclass


@dataclass
class SwapPair:
    """ Represents a pair of left/right body parts to swap """
    start_qpos_r: int
    start_qpos_l: int
    num_qpos: int

    start_dof_r: int
    start_dof_l: int
    num_dof: int


def parse_starting_pose(
        file_path,
        dof_id_lookup: dict[str, tuple[int, int]],
        num_qpos: int,
        num_dofs: int,
):
    """ Parse starting pose from YAML file"""
    start_q, start_qv = [0.0] * num_qpos, [0.0] * num_dofs

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    for coord_name, values in data.items():
        if coord_name not in dof_id_lookup:
            raise ValueError(f"Coordinate '{coord_name}' not found")
        qpos_id, qvel_id = dof_id_lookup[coord_name]
        qpos, qvel = values["q"], values["v"]
        if qpos_id != -1:
            start_q[qpos_id] = qpos
        if qvel_id != -1:
            start_qv[qvel_id] = qvel

    return start_q, start_qv


def get_swap_left_right_data(
        m: msk_warp.types.Model,
        body_id_lookup: dict[str, int]
) -> list[SwapPair]:
    all_bodies = list(body_id_lookup.keys())
    # remove all "_r" and "_l" suffixes to get unique body names
    swappable_bodies = [body_name[:-2] for body_name in all_bodies if body_name.endswith(("_r", "_l"))]
    swappable_bodies = list(set(swappable_bodies))

    swap_data = []
    for body in swappable_bodies:
        body_r, body_l = f"{body}_r", f"{body}_l"
        body_r_id, body_l_id = body_id_lookup[body_r], body_id_lookup[body_l]

        qpos_num_r, qpos_num_l = msk_warp.get_qpos_num(m, body_r_id), msk_warp.get_qpos_num(m, body_l_id)
        qpos_adr_r, qpos_adr_l = msk_warp.get_qpos_adr(m, body_r_id), msk_warp.get_qpos_adr(m, body_l_id)
        dof_num_r, dof_num_l = msk_warp.get_dof_num(m, body_r_id), msk_warp.get_dof_num(m, body_l_id)
        dof_adr_r, dof_adr_l = msk_warp.get_dof_adr(m, body_r_id), msk_warp.get_dof_adr(m, body_l_id)

        # sanity checks
        assert qpos_num_r == qpos_num_l, "Mismatched numbers of qpos for left/right body parts"
        assert dof_num_r == dof_num_l, "Mismatched number of dofs for left/right body parts"
        assert not ((qpos_adr_l < qpos_adr_r + qpos_num_r) and (qpos_adr_r < qpos_adr_l + qpos_num_l)), \
            "Overlapping qpos addresses for left/right body parts"
        assert not ((dof_adr_l < dof_adr_r + dof_num_r) and (dof_adr_r < dof_adr_l + dof_num_l)), \
            "Overlapping dof addresses for left/right body parts"

        swap_data.append(
            SwapPair(
                start_qpos_l=int(qpos_adr_l),
                start_qpos_r=int(qpos_adr_r),
                num_qpos=int(qpos_num_l),
                start_dof_l=int(dof_adr_l),
                start_dof_r=int(dof_adr_r),
                num_dof=int(dof_num_l),
            )
        )
    return swap_data


def get_base_name(name: str) -> str:
    """ Get base name by removing _r/_l suffix """
    if name.endswith("_r") or name.endswith("_l"):
        return name[:-2]
    return name
