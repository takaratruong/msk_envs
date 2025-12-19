FWD_IDX = 0  # x axis is the lane direction
UP_IDX = 1  # y axis is the up direction
SIDE_IDX = 2  # z axis is orthogonal to the lane direction


def build_axis(axis: int, scale: float) -> list[float]:
    a = [0.0, 0.0, 0.0]
    a[axis] = scale
    return a
