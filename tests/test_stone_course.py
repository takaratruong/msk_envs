import unittest

import torch

from msk_envs.envs.env_stone_course import StoneCourseEnv, StoneCourseSpec


def make_spec(**overrides) -> StoneCourseSpec:
    values = {
        "num_stones": 6,
        "step_length_range": (0.4, 0.7),
        "lateral_jitter": 0.10,
        "slab_size": (0.36, 0.10, 0.36),
        "top_height": 0.45,
        "lookahead": 4,
    }
    values.update(overrides)
    return StoneCourseSpec(**values)


def make_success_test_env() -> StoneCourseEnv:
    env = object.__new__(StoneCourseEnv)
    env.course = make_spec(num_stones=2)
    env.device = torch.device("cpu")
    env.stone_ids = torch.tensor([3, 4], dtype=torch.long)
    env.foot_collider_ids = torch.tensor([0, 1], dtype=torch.long)
    env.foot_collider_radii = torch.tensor([0.05, 0.02])
    env.slab_half_extents_xz = torch.tensor([0.18, 0.18])
    env.stone_positions = torch.tensor([
        [[0.5, 0.4, 0.0], [1.0, 0.4, 0.1]],
        [[0.5, 0.4, 0.0], [2.0, 0.4, -0.1]],
        [[0.5, 0.4, 0.0], [3.0, 0.4, 0.0]],
    ])
    env.collider_positions = torch.zeros((3, 5, 3))
    env.collider_forces = torch.zeros((3, 5))

    # Active heel and final slab contact in world 0.
    env.collider_positions[0, 0] = torch.tensor([1.0, 0.50, 0.1])
    env.collider_forces[0, 0] = 100.0
    env.collider_forces[0, 4] = 100.0

    # Active contacts in world 1, but the foot is on the first slab.
    env.collider_positions[1, 0] = torch.tensor([0.5, 0.50, 0.0])
    env.collider_forces[1, 0] = 100.0
    env.collider_forces[1, 4] = 100.0

    # Foot is over the final slab in world 2 but has no contact force.
    env.collider_positions[2, 1] = torch.tensor([3.0, 0.47, 0.0])
    env.collider_forces[2, 4] = 100.0

    env.time = torch.tensor([1.0, 1.0, 10.0])
    env.max_episode_duration = 10.0
    env._last_success = torch.zeros(3, dtype=torch.bool)
    return env


class StoneCourseSpecTest(unittest.TestCase):
    def test_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "course_stones"):
            make_spec(num_stones=0)
        with self.assertRaisesRegex(ValueError, "course_step_length_range"):
            make_spec(step_length_range=(0.7, 0.4))
        with self.assertRaisesRegex(ValueError, "course_slab_size"):
            make_spec(slab_size=(0.36, 0.0, 0.36))

    def test_samples_independent_layout_per_world(self):
        spec = make_spec()
        generator = torch.Generator().manual_seed(7)
        positions = spec.sample_positions(3, "cpu", generator)

        self.assertEqual(positions.shape, (3, 6, 3))
        self.assertFalse(torch.equal(positions[0], positions[1]))
        self.assertTrue(torch.allclose(
            positions[:, :, 1],
            torch.full((3, 6), spec.center_height),
        ))

        step_lengths = torch.diff(
            positions[:, :, 0],
            dim=1,
            prepend=torch.zeros((3, 1)),
        )
        self.assertTrue(((step_lengths[:, :2] >= 0.32) & (step_lengths[:, :2] <= 0.38)).all())
        self.assertTrue(((step_lengths[:, 2:] >= 0.4) & (step_lengths[:, 2:] <= 0.7)).all())

    def test_observation_is_root_relative_and_clamps_at_final_slab(self):
        spec = make_spec(num_stones=3, lookahead=2)
        stones = torch.tensor([[
            [0.5, 0.4, 0.1],
            [1.0, 0.4, -0.1],
            [1.5, 0.4, 0.2],
        ]])
        root = torch.tensor([[1.2, 1.4, 0.05]])

        observation = spec.root_relative_observation(stones, root)

        expected = torch.tensor([[0.3, 0.15, 0.3, 0.15]])
        self.assertTrue(torch.allclose(observation, expected))


class StoneCourseEnvironmentTest(unittest.TestCase):
    def test_reset_updates_only_selected_worlds_and_physical_transforms(self):
        env = object.__new__(StoneCourseEnv)
        env.course = make_spec(num_stones=3)
        env.device = torch.device("cpu")
        env.stone_ids = torch.tensor([2, 3, 4], dtype=torch.long)
        env.stone_positions = torch.full((3, 3, 3), -99.0)
        env.collider_local_transforms = torch.full((3, 5, 7), -99.0)
        env._last_success = torch.ones(3, dtype=torch.bool)

        env._randomize_stones(torch.tensor([False, True, False]))

        self.assertTrue((env.stone_positions[0] == -99.0).all())
        self.assertTrue((env.stone_positions[2] == -99.0).all())
        self.assertFalse((env.stone_positions[1] == -99.0).any())
        self.assertTrue(torch.equal(
            env.collider_local_transforms[1, env.stone_ids, :3],
            env.stone_positions[1],
        ))
        self.assertEqual(env._last_success.tolist(), [True, False, True])

    def test_success_requires_active_foot_on_final_slab(self):
        env = make_success_test_env()
        self.assertEqual(env._reached_last_stone().tolist(), [True, False, False])

    def test_success_and_timeout_are_truncations(self):
        env = make_success_test_env()
        self.assertEqual(env._get_truncated().tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(env._last_success.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
