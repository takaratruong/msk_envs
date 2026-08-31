import unittest

import torch

from msk_envs.envs.env_stone_course import StoneCourseEnv


def make_success_test_env():
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

    env.collider_positions[0, 0] = torch.tensor([1.0, 0.50, 0.1])
    env.collider_forces[0, 0] = 100.0
    env.collider_forces[0, 4] = 100.0

    env.collider_positions[1, 0] = torch.tensor([0.5, 0.50, 0.0])
    env.collider_forces[1, 0] = 100.0
    env.collider_forces[1, 4] = 100.0

    env.collider_positions[2, 1] = torch.tensor([3.0, 0.47, 0.0])
    env.collider_forces[2, 4] = 100.0

    env.time = torch.tensor([1.0, 1.0, 10.0])
    env.max_episode_duration = 10.0
    env._last_success = torch.zeros(3, dtype=torch.bool)
    return env


class StoneCourseSuccessTest(unittest.TestCase):
    def test_success_requires_active_foot_on_final_slab(self):
        env = make_success_test_env()
        self.assertEqual(env._reached_last_stone().tolist(), [True, False, False])

    def test_success_and_timeout_are_truncations(self):
        env = make_success_test_env()
        self.assertEqual(env._get_truncated().tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(env._last_success.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
