import torch


def get_position_from_transform(transform: torch.Tensor) -> torch.Tensor:
    position = transform[..., :3]
    return position


def get_rotation_from_transform(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., 3:]
    return rotation
