import yaml


def parse_starting_pose(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data["joint_positions"], data["joint_velocities"]
