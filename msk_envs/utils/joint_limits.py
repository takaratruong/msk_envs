from dataclasses import dataclass

import yaml


@dataclass
class JointLimit:
    limit_id: int
    lower: float
    upper: float


@dataclass
class LimitForceCurve:
    limit_id: int
    limit_force: list[float]
    shape_param: list[float]

    def fix_zeros(self):
        # if all are zero, use some reasonable defaults
        if all(f == 0.0 for f in self.limit_force):
            self.limit_force = [2.0, 2.0]
        if all(s == 0.0 for s in self.shape_param):
            self.shape_param = [20.0, 20.0]


def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def get_joint_limits(
        file_path: str,
        limit_id_lookup: dict[str, int]
) -> list[JointLimit]:
    """ Load joint limits from YAML """
    data = load_yaml(file_path)
    limits = []
    for limit_name, limit_data in data.items():
        limit = JointLimit(
            limit_id=limit_id_lookup[limit_name],
            lower=limit_data["lower"],
            upper=limit_data["upper"]
        )
        limits.append(limit)
    return limits


def get_limit_force_curves(
        file_path: str,
        limit_id_lookup: dict[str, int]
) -> list[LimitForceCurve]:
    """ Load exponential limit force curves from YAML """
    data = load_yaml(file_path)
    limit_curves = []
    for limit_name, curve_data in data.items():
        limit_id = limit_id_lookup[limit_name]
        limit_force = curve_data["limit_force"]
        shape_param = curve_data["shape_param"]
        limit_curve = LimitForceCurve(
            limit_id=limit_id,
            limit_force=limit_force,
            shape_param=shape_param
        )
        limit_curve.fix_zeros()
        limit_curves.append(limit_curve)

    return limit_curves
