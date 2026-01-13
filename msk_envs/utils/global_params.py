FWD_IDX = 0  # x-axis is the lane direction
UP_IDX = 1  # y-axis is the up direction
SIDE_IDX = 2  # z-axis is orthogonal to the lane direction

# Height thresholds for terminating an episode
MIN_ROOT_HEIGHT = 0.6
MIN_HEAD_HEIGHT = 1.0


def build_axis(axis: int, scale: float) -> list[float]:
    a = [0.0, 0.0, 0.0]
    a[axis] = scale
    return a
