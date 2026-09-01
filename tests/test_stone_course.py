import unittest

import torch

from msk_envs.envs.env_stone_course import (
    TerrainCurriculum,
    StoneCourseEnv,
    StoneCourseSpec,
)
from msk_envs.utils.quat import rotate_vec


def make_spec(**overrides) -> StoneCourseSpec:
    values = {
        "num_stones": 5,
        "step_length_range": (0.65, 1.50),
        "lateral_jitter": 0.10,
        "slab_size": (0.36, 0.10, 0.36),
        "top_height": 0.45,
        "top_height_range": (0.20, 1.05),
        "elevation_angle_max_degrees": 50.0,
        "yaw_angle_max_degrees": 20.0,
        "surface_tilt_max_degrees": 20.0,
        "lookahead": 4,
    }
    values.update(overrides)
    return StoneCourseSpec(**values)


def make_curriculum(**overrides) -> TerrainCurriculum:
    values = {
        "minimum": 0.65,
        "maximum": 1.50,
        "current_maximum": 0.80,
        "increment": 0.14,
        "elevation_maximum_degrees": 50.0,
        "current_elevation_maximum_degrees": 0.0,
        "elevation_increment_degrees": 10.0,
        "yaw_maximum_degrees": 20.0,
        "current_yaw_maximum_degrees": 0.0,
        "yaw_increment_degrees": 4.0,
        "surface_tilt_maximum_degrees": 20.0,
        "current_surface_tilt_maximum_degrees": 0.0,
        "surface_tilt_increment_degrees": 4.0,
        "success_threshold": 0.60,
        "window": 5,
    }
    values.update(overrides)
    return TerrainCurriculum(**values)


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
            step_length_max=0.80,
        )

        self.assertEqual(positions.shape, (3, 5, 3))
        self.assertFalse(torch.equal(positions[0], positions[1]))
        self.assertTrue(torch.allclose(
            positions[:, :, 1],
            torch.full((3, 5), spec.center_height),
        ))

        deltas = torch.diff(
            positions,
            dim=1,
            prepend=torch.tensor([[[0.0, spec.center_height, 0.0]]] * 3),
        )
        self.assertTrue(((deltas[:, :2, 0] >= 0.32) & (deltas[:, :2, 0] <= 0.38)).all())
        radial_distances = torch.linalg.vector_norm(deltas[:, 2:], dim=2)
        self.assertTrue(((radial_distances >= 0.65) & (radial_distances <= 0.80)).all())

    def test_samples_bounded_3d_targets_and_surface_tilts(self):
        spec = make_spec()
        generator = torch.Generator().manual_seed(11)
        positions = spec.sample_positions(
            128,
            "cpu",
            generator,
            step_length_max=1.50,
            elevation_angle_max_degrees=50.0,
            yaw_angle_max_degrees=20.0,
        )
        top_heights = positions[:, :, 1] + spec.half_extents[1]
        self.assertTrue((top_heights >= 0.20 - 1e-6).all())
        self.assertTrue((top_heights <= 1.05 + 1e-6).all())
        self.assertTrue((torch.diff(positions[:, :, 0], dim=1) > 0.0).all())

        tilts = spec.sample_surface_tilts(1024, "cpu", 20.0, generator)
        self.assertLessEqual(tilts.abs().max().item(), torch.deg2rad(torch.tensor(20.0)).item())
        rotations = spec.surface_tilts_to_quaternions(tilts)
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(rotations, dim=1), torch.ones(1024), atol=1e-6
        ))

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
        tilts = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]])
        rotations = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4).repeat(1, 5, 1)

        observation = spec.root_relative_observation(stones, rotations, tilts, root)

        self.assertTrue(torch.allclose(
            observation[0, :8].reshape(4, 2)[:, 0],
            torch.tensor([-0.1, 0.2, 0.5, 1.0]),
        ))
        self.assertEqual(observation.shape, (1, 20))
        self.assertTrue(torch.allclose(
            observation[0, 12:].reshape(4, 2),
            torch.tensor([[0.7, 0.8], [0.9, 1.0], [0.1, 0.2], [0.5, 0.6]]),
        ))


class TerrainCurriculumTest(unittest.TestCase):
    def test_promotes_all_bounds_after_a_competent_window(self):
        curriculum = make_curriculum()

        promoted = curriculum.observe(torch.tensor([True, True, True, False, False]))

        self.assertTrue(promoted)
        self.assertEqual(curriculum.minimum, 0.65)
        self.assertAlmostEqual(curriculum.current_maximum, 0.94)
        self.assertAlmostEqual(curriculum.current_elevation_maximum_degrees, 10.0)
        self.assertAlmostEqual(curriculum.current_yaw_maximum_degrees, 4.0)
        self.assertAlmostEqual(curriculum.current_surface_tilt_maximum_degrees, 4.0)
        self.assertAlmostEqual(curriculum.last_completion_rate, 0.60)

    def test_holds_difficulty_after_an_incompetent_window(self):
        curriculum = make_curriculum()

        promoted = curriculum.observe(torch.tensor([True, False, False, False, False]))

        self.assertFalse(promoted)
        self.assertEqual(curriculum.current_maximum, 0.80)

    def test_caps_upper_bound_at_final_maximum(self):
        curriculum = make_curriculum(
            current_maximum=1.49,
            current_elevation_maximum_degrees=49.0,
            current_yaw_maximum_degrees=19.0,
            current_surface_tilt_maximum_degrees=19.0,
            window=1,
        )

        curriculum.observe(torch.tensor([True]))

        self.assertEqual(curriculum.current_maximum, 1.50)
        self.assertEqual(curriculum.current_elevation_maximum_degrees, 50.0)
        self.assertEqual(curriculum.current_yaw_maximum_degrees, 20.0)
        self.assertEqual(curriculum.current_surface_tilt_maximum_degrees, 20.0)

    def test_checkpoint_state_round_trip_preserves_partial_window(self):
        source = make_curriculum()
        source.observe(torch.tensor([True, False, True]))
        source.last_completion_rate = 0.75
        target = make_curriculum()

        target.load_state_dict(source.state_dict())

        self.assertEqual(target.current_maximum, 0.80)
        self.assertEqual(target.episodes, 3)
        self.assertEqual(target.successes, 2)
        self.assertEqual(target.last_completion_rate, 0.75)


class StoneCourseEnvironmentTest(unittest.TestCase):
    def make_recycling_env(self) -> StoneCourseEnv:
        env = object.__new__(StoneCourseEnv)
        env.course = make_spec()
        env.terrain_curriculum = make_curriculum()
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
        env.stone_surface_tilts = torch.zeros((2, 5, 2))
        env.stone_rotations = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(
            1, 1, 4
        ).repeat(2, 5, 1)
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
        radial_distance = torch.linalg.vector_norm(
            recycled - torch.tensor([1.90, 0.40, 0.12])
        )
        self.assertGreaterEqual(radial_distance.item(), 0.65 - 1e-6)
        self.assertLessEqual(radial_distance.item(), 0.80 + 1e-6)
        self.assertGreater(recycled[0].item(), 1.90)
        self.assertTrue(torch.equal(env.collider_local_transforms[0, 0, :3], recycled))
        self.assertTrue(torch.equal(
            env.collider_local_transforms[0, 0, 3:7],
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ))
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
        env.stone_rotations = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(
            1, 1, 4
        ).repeat(3, 5, 1)
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

    def test_interior_footprint_is_measured_in_a_tilted_slabs_local_frame(self):
        env = self.make_contact_env()
        env.stone_positions[0, 1:, 0] = 10.0
        tilt = torch.tensor([[0.0, torch.deg2rad(torch.tensor(30.0))]])
        rotation = env.course.surface_tilts_to_quaternions(tilt)[0]
        env.stone_rotations[0, 0] = rotation
        local_foot_positions = torch.tensor([
            [0.0, 0.30, 0.0],
            [0.10, 0.30, 0.0],
        ])
        env.collider_positions[0, :2] = env.stone_positions[0, 0] + rotate_vec(
            rotation.expand(2, 4), local_foot_positions
        )

        _, interior_by_side = env._interior_foot_support()

        self.assertTrue(interior_by_side[0, 0])

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
