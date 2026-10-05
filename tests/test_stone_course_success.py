import torch

from msk_envs.envs.env_stone_course import StoneCourseEnv


def _success_test_env():
    env = object.__new__(StoneCourseEnv)
    env.stone_radius = 0.18
    env.stone_ids = torch.tensor([3, 4], dtype=torch.long)
    env.foot_collider_ids = torch.tensor([0, 1], dtype=torch.long)
    env.foot_collider_radii = torch.tensor([0.05, 0.02])
    env.stone_positions = torch.tensor([
        [[0.5, 0.4, 0.0], [1.0, 0.4, 0.1]],
        [[0.5, 0.4, 0.0], [2.0, 0.4, -0.1]],
        [[0.5, 0.4, 0.0], [3.0, 0.4, 0.0]],
    ])
    env.collider_positions = torch.zeros((3, 5, 3))
    env.collider_forces = torch.zeros((3, 5))

    # World 0: an active heel sphere overlaps the final slab and the slab is active.
    env.collider_positions[0, 0] = torch.tensor([1.0, 0.50, 0.1])
    env.collider_forces[0, 0] = 100.0
    env.collider_forces[0, 4] = 100.0

    # World 1: both colliders are active, but the foot is on an earlier slab.
    env.collider_positions[1, 0] = torch.tensor([0.5, 0.50, 0.0])
    env.collider_forces[1, 0] = 100.0
    env.collider_forces[1, 4] = 100.0

    # World 2: the foot is spatially over the final slab but has no contact force.
    env.collider_positions[2, 1] = torch.tensor([3.0, 0.47, 0.0])
    env.collider_forces[2, 4] = 100.0

    env.time = torch.tensor([1.0, 1.0, 10.0])
    env.max_episode_duration = 10.0
    env._last_success = torch.zeros(3, dtype=torch.bool)
    return env


def test_reached_last_stone_requires_active_foot_on_final_slab():
    env = _success_test_env()
    assert env._reached_last_stone().tolist() == [True, False, False]


def test_success_is_a_truncation_and_timeout_still_truncates():
    env = _success_test_env()
    assert env._get_truncated().tolist() == [1.0, 0.0, 1.0]
    assert env._last_success.tolist() == [True, False, False]
