import yaml
from dataclasses import dataclass


@dataclass
class MuscleMetabolic:
    muscle_id: int
    specific_tension: float
    slow_twitch_ratio: float
    density: float


def parse_starting_activations(
        file_path: str,
        muscle_id_lookup: dict[str, int],
        default_activation: float = 0.0,
) -> list[float]:
    """ Parse starting muscle activations from YAML file"""
    num_muscles = len(muscle_id_lookup)

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    activations = [default_activation] * num_muscles
    for muscle_name, activation in data.items():
        muscle_id = muscle_id_lookup[muscle_name]
        activations[muscle_id] = activation
    return activations


def parse_muscle_metabolic_params(
        file_path: str,
        muscle_id_lookup: dict[str, int],
) -> list[MuscleMetabolic]:
    """ Parse muscle metabolic parameters from YAML file"""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    metabolic_params = []
    for muscle_name, params in data.items():
        muscle_id = muscle_id_lookup[muscle_name]
        metabolic_param = MuscleMetabolic(
            muscle_id=muscle_id,
            specific_tension=1e6 * params["specific_tension"],
            slow_twitch_ratio=params["slow_twitch_ratio"],
            density=params["density"],
        )
        metabolic_params.append(metabolic_param)
    return metabolic_params
