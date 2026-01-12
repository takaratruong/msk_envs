from dataclasses import dataclass

import yaml
import math


@dataclass
class ContactParam:
    geom_id: int
    stiffness: float
    dissipation: float
    priority: int


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def parse_contact_params(file_path: str, geom_id_lookup: dict[str, int]) -> list[ContactParam]:
    """ Load Hunt-Crossley contact params from YAML """
    data = load_yaml(file_path)
    params = []
    for geom_name, geom_data in data.items():
        param = ContactParam(
            geom_id=geom_id_lookup[geom_name],
            stiffness=math.pow(float(geom_data["stiffness"]), 2.0 / 3.0),
            dissipation=float(geom_data["dissipation"]),
            priority=int(geom_data["priority"])
        )
        params.append(param)
    return params
