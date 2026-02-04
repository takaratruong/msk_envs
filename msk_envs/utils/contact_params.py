from dataclasses import dataclass

import yaml
import math


@dataclass
class ContactParam:
    geom_id: int
    stiffness: float
    dissipation: float
    priority: int

    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity: float


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def lookup_geom_skip(
        geom_name: str,
        geom_id_lookup: dict[str, int]
):
    """ Lookup geometry ID, skip if not found """
    if geom_name not in geom_id_lookup:
        print(f"Warning: geometry '{geom_name}' not found in geom_id_lookup, skipping.")
        return None
    return geom_id_lookup[geom_name]


def parse_contact_params(file_path: str, geom_id_lookup: dict[str, int]) -> list[ContactParam]:
    """ Load Hunt-Crossley contact params from YAML """
    data = load_yaml(file_path)
    params = []
    for geom_name, geom_data in data.items():
        geom_id = lookup_geom_skip(geom_name, geom_id_lookup)
        if geom_id is None:
            continue

        param = ContactParam(
            geom_id=geom_id,
            stiffness=math.pow(float(geom_data["stiffness"]), 2.0 / 3.0),
            dissipation=float(geom_data["dissipation"]),
            priority=int(geom_data["priority"]),
            static_friction=float(geom_data["static_friction"]),
            dynamic_friction=float(geom_data["dynamic_friction"]),
            viscous_friction=float(geom_data["viscous_friction"]),
            transition_velocity=float(geom_data["transition_velocity"]),
        )
        params.append(param)
    return params
