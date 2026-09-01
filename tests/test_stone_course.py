import unittest

import torch

from msk_envs.envs.env_stone_course import (
    SpacingCurriculum,
    StoneCourseEnv,
    StoneCourseSpec,
)


def make_spec(**overrides) -> StoneCourseSpec:
    values = {
        "num_stones": 5,
        "step_length_range": (0.4, 0.85),
        "lateral_jitter": 0.10,
        "slab_size": (0.36, 0.10, 0.36),
        "top_height": 0.45,
        "lookahead": 4,
    }
    values.update(overrides)
    return StoneCourseSpec(**values)


def make_curriculum(**overrides) -> SpacingCurriculum:
    values = {
        "minimum": 0.4,
        "maximum": 0.85,
        "current_maximum": 0.55,
        "increment": 0.05,
        "success_threshold": 0.60,
        "window": 5,
    }
    values.update(overrides)
    return SpacingCurriculum(**values)


class StoneCourseSpecTest(unittest.TestCase):
    def test_requires_one_recycled_spare_beyond_lookahead(self):
        with self.assertRaisesRegex(ValueError, "one spare slab"):
            make_spec(num_stones=4, lookahead=4)

    def test_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "course_step_length_range"):
            make_spec(step_length_range=(0.7, 0.4))
        with self.assertRaisesRegex(ValueError, "course_slab_size"):
            make_spec(slab_size=(0.36, 0.0, 0.36))

    def test_samples_independent_layouts_at_current_curriculum_maximum(self):
        spec = make_spec()
        generator = torch.Generator().manual_seed(7)
        positions = spec.sample_positions(
            3,
            "cpu",
            generator,
            step_length_max=0.55,
        )

        self.assertEqual(positions.shape, (3, 5, 3))
        self.assertFalse(torch.equal(positions[0], positions[1]))
        self.assertTrue(torch.allclose(
            positions[:, :, 1],
            torch.full((3, 5), spec.center_height),
        ))

        step_lengths = torch.diff(
            positions[:, :, 0],
            dim=1,
            prepend=torch.zeros((3, 1)),
        )
        self.assertTrue(((step_lengths[:, :2] >= 0.32) & (step_lengths[:, :2] <= 0.38)).all())
        self.assertTrue(((step_lengths[:, 2:] >= 0.4) & (step_lengths[:, 2:] <= 0.55)).all())

    def test_observation_sorts_recycled_ids_and_returns_four_upcoming_slabs(self):
        spec = make_spec()
        stones = torch.tensor([[
            [1.5, 0.4, 0.0],
            [0.5, 0.4, 0.0],
            [2.0, 0.4, 0.0],
            [0.9, 0.4, 0.0],
            [1.2, 0.4, 0.0],
        ]])
        root = torch.tensor([[1.0, 1.4, 0.0]])

        observation = spec.root_relative_observation(stones, root).reshape(1, 4, 2)

        self.assertTrue(torch.allclose(
            observation[0, :, 0],
            torch.tensor([-0.1, 0.2, 0.5, 1.0]),
        ))


class SpacingCurriculumTest(unittest.TestCase):
    def test_promotes_only_the_upper_bound_after_a_competent_window(self):
        curriculum = make_curriculum()

        promoted = curriculum.observe(torch.tensor([True, True, True, False, False]))

        self.assertTrue(promoted)
        self.assertEqual(curriculum.minimum, 0.4)
        self.assertAlmostEqual(curriculum.current_maximum, 0.60)
        self.assertAlmostEqual(curriculum.last_completion_rate, 0.60)

    def test_holds_difficulty_after_an_incompetent_window(self):
        curriculum = make_curriculum()

        promoted = curriculum.observe(torch.tensor([True, False, False, False, False]))

        self.assertFalse(promoted)
        self.assertEqual(curriculum.current_maximum, 0.55)

    def test_caps_upper_bound_at_final_maximum(self):
        curriculum = make_curriculum(current_maximum=0.83, window=1)

        curriculum.observe(torch.tensor([True]))

        self.assertEqual(curriculum.current_maximum, 0.85)

    def test_checkpoint_state_round_trip_preserves_partial_window(self):
        source = make_curriculum()
        source.observe(torch.tensor([True, False, True]))
        source.last_completion_rate = 0.75
        target = make_curriculum()

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.current_maximum, 0.55)
        self.assertEqual(target.episodes, 3)
        self.assertEqual(target.successes, 2)
        self.assertEqual(target.last_completion_rate, 0.75)


class StoneCourseEnvironmentTest(unittest.TestCase):
    def make_recycling_env(self) -> StoneCourseEnv:
        env = object.__new__(StoneCourseEnv)
        env.course = make_spec()
        env.spacing_curriculum = make_curriculum()
        env.device = torch.device("cpu")
        env.recycle_distance_behind = 0.15
        env.root_pos = torch.tensor([
            [0.8, 1.4, 0.0],
            [0.0, 1.4, 0.0],
        ])
        env.stone_ids = torch.arange(5)
        env.stone_positions = torch.tensor([
            [
                [0.30, 0.40, 0.12],
                [0.65, 0.40, -0.12],
                [1.10, 0.40, 0.12],
                [1.50, 0.40, -0.12],
                [1.90, 0.40, 0.12],
            ],
            [
                [0.30, 0.40, 0.12],
                [0.65, 0.40, -0.12],
                [1.10, 0.40, 0.12],
                [1.50, 0.40, -0.12],
                [1.90, 0.40, 0.12],
            ],
        ])
        env.collider_local_transforms = torch.zeros((2, 5, 7))
        env.collider_forces = torch.zeros((2, 5))
        env.next_lateral_sign = torch.full((2,), -1.0)
        env.episode_slabs_recycled = torch.zeros(2, dtype=torch.long)
        return env

    def test_recycles_only_passed_inactive_slab_in_affected_world(self):
        env = self.make_recycling_env()
        old_world_one = env.stone_positions[1].clone()

        env._recycle_passed_stones()

        recycled = env.stone_positions[0, 0]
        self.assertGreaterEqual(recycled[0].item(), 1.90 + 0.4)
        self.assertLessEqual(recycled[0].item(), 1.90 + 0.55)
        self.assertTrue(torch.equal(env.collider_local_transforms[0, 0, :3], recycled))
        self.assertTrue(torch.equal(env.stone_positions[1], old_world_one))
        self.assertEqual(env.episode_slabs_recycled.tolist(), [1, 0])
        self.assertEqual(env.next_lateral_sign.tolist(), [1.0, -1.0])

    def test_does_not_recycle_a_supporting_slab(self):
        env = self.make_recycling_env()
        env.collider_forces[0, 0] = 100.0
        before = env.stone_positions.clone()

        env._recycle_passed_stones()

        self.assertTrue(torch.equal(env.stone_positions, before))
        self.assertEqual(env.episode_slabs_recycled.tolist(), [0, 0])

    def make_contact_env(self) -> StoneCourseEnv:
        env = object.__new__(StoneCourseEnv)
        env.course = make_spec()
        env.device = torch.device("cpu")
        env.require_interior_landing = True
        env.landing_check_delay = 0.25
        env.stone_ids = torch.tensor([3, 4, 5, 6, 7])
        env.foot_collider_ids = torch.tensor([0, 1, 2])
        env.foot_side_masks = (
            torch.tensor([True, True, False]),
            torch.tensor([False, False, True]),
        )
        env.interior_half_extents_xz = torch.tensor([
            [0.13, 0.13],
            [0.16, 0.16],
            [0.16, 0.16],
        ])
        env.stone_positions = torch.tensor([
            [[1.0, 0.4, 0.0]] * 5,
            [[2.0, 0.4, 0.0]] * 5,
            [[3.0, 0.4, 0.0]] * 5,
        ])
        env.collider_positions = torch.zeros((3, 8, 3))
        env.collider_forces = torch.zeros((3, 8))
        env.previous_foot_contact = torch.zeros((3, 2), dtype=torch.bool)
        env._last_edge_violation = torch.zeros(3, dtype=torch.bool)
        env.time = torch.ones(3)

        # World 0: both the active heel and inactive toe project inside.
        env.collider_positions[0, 0] = torch.tensor([1.0, 0.50, 0.0])
        env.collider_positions[0, 1] = torch.tensor([1.10, 0.55, 0.0])
        env.collider_forces[0, 0] = 100.0
        env.collider_forces[0, 3] = 100.0

        # World 1: active heel is centered, but the projected toe crosses the edge.
        env.collider_positions[1, 0] = torch.tensor([2.0, 0.50, 0.0])
        env.collider_positions[1, 1] = torch.tensor([2.17, 0.55, 0.0])
        env.collider_forces[1, 0] = 100.0
        env.collider_forces[1, 3] = 100.0
        return env

    def test_edge_first_touchdown_is_invalid_but_interior_touchdown_is_valid(self):
        env = self.make_contact_env()

        invalid = env._invalid_edge_touchdown()

        self.assertEqual(invalid.tolist(), [False, True, False])

    def test_time_limit_is_neutral_and_curriculum_success_requires_progress(self):
        env = object.__new__(StoneCourseEnv)
        env.time = torch.tensor([12.0, 12.0, 1.0])
        env.max_episode_duration = 12.0
        env.root_pos = torch.tensor([
            [13.0, 1.4, 0.0],
            [5.0, 1.4, 0.0],
            [2.0, 1.4, 0.0],
        ])
        env._episode_start_x = torch.zeros(3)
        env.curriculum_min_progress = 12.0
        env._last_success = torch.zeros(3, dtype=torch.bool)

        truncated = env._get_truncated()

        self.assertEqual(truncated.tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(env._last_success.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
