"""
Quaternions are in (x, y, z, w) format
"""
import torch


def rotate_vec(rot: torch.Tensor, v: torch.Tensor):
    pure = rot[..., :3]
    scalar = rot[..., 3:]

    pure_x_v = torch.cross(pure, v, dim=-1)
    pure_x_pure_x_v = torch.cross(pure, pure_x_v, dim=-1)
    return v + 2.0 * (pure_x_v * scalar + pure_x_pure_x_v)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / norm


def quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q2_conj = quat_conjugate(q2)
    return quat_mul(q1, q2_conj)


def quat_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(1.0 - w * w)

    axis = torch.zeros_like(q[..., 1:])
    axis[..., 2] = 1.0  # default z-axis

    mask = s > 1e-6
    axis = torch.where(
        mask.unsqueeze(-1),
        q[..., 1:] / s.unsqueeze(-1),
        axis,
    )

    return axis * angle.unsqueeze(-1)


def quat_diff_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    qd = quat_diff(q1, q2)
    angle_axis = quat_to_angle_axis(qd)
    angle = torch.norm(angle_axis, dim=-1)
    return angle
