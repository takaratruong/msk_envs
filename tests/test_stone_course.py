import unittest

import torch

from msk_envs.envs.env_stone_course import (
    TerrainCurriculum,
    StoneCourseEnv,
    StoneCourseSpec,
)
from msk_envs.utils.quat import rotate_vec

UP_IDX_TEST = 1


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

        # The launch pair sits side by side beneath the starting pose.
        self.assertTrue(torch.allclose(
            positions[:, :2, 0],
            torch.full((3, 2), spec.launch_forward_offset),
        ))
        self.assertTrue((positions[:, 0, 2] > 0.0).all())
        self.assertTrue((positions[:, 1, 2] < 0.0).all())

        deltas = torch.diff(
            positions,
            dim=1,
            prepend=torch.tensor([[[0.0, spec.center_height, 0.0]]] * 3),
        )
        radial_distances = torch.linalg.vector_norm(deltas[:, 2:], dim=2)
        self.assertTrue(((radial_distances >= 0.65) & (radial_distances <= 0.80)).all())

    def test_per_course_stride_minimum_raises_the_sampling_floor(self):
        spec = make_spec()
        generator = torch.Generator().manual_seed(13)
        minimums = torch.tensor([0.65, 1.10, 1.10])
        positions = spec.sample_positions(
            3,
            "cpu",
            generator,
            step_length_max=1.20,
            step_length_min=minimums,
        )

        deltas = torch.diff(
            positions,
            dim=1,
            prepend=torch.tensor([[[0.0, spec.center_height, 0.0]]] * 3),
        )
        radial_distances = torch.linalg.vector_norm(deltas[:, 2:], dim=2)
        self.assertTrue((radial_distances[1:] >= 1.10 - 1e-6).all())
        self.assertTrue((radial_distances <= 1.20 + 1e-6).all())

        recycled = spec.sample_next_position(
            positions[:, -1],
            torch.tensor([1.0, -1.0, 1.0]),
            1.20,
            0.0,
            0.0,
            generator,
            step_length_min=minimums,
        )
        recycled_distances = torch.linalg.vector_norm(
            recycled - positions[:, -1], dim=1
        )
        self.assertTrue((recycled_distances[1:] >= 1.10 - 1e-6).all())
        self.assertTrue((recycled_distances <= 1.20 + 1e-6).all())

    def test_stride_minimum_clamps_to_the_current_curriculum_maximum(self):
        spec = make_spec()
        generator = torch.Generator().manual_seed(17)
        minimums = torch.tensor([1.10, 1.10])
        positions = spec.sample_positions(
            2,
            "cpu",
            generator,
            step_length_max=0.80,
            step_length_min=minimums,
        )
        deltas = torch.diff(
            positions,
            dim=1,
            prepend=torch.tensor([[[0.0, spec.center_height, 0.0]]] * 2),
        )
        radial_distances = torch.linalg.vector_norm(deltas[:, 2:], dim=2)
        self.assertTrue(
            torch.allclose(radial_distances, torch.full_like(radial_distances, 0.80))
        )
        with self.assertRaisesRegex(ValueError, "step_length_min"):
            spec.sample_positions(
                2,
                "cpu",
                generator,
                step_length_max=0.80,
                step_length_min=torch.tensor([0.10, 0.10]),
            )

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
        # The launch pair shares one x; every later step advances forward.
        self.assertTrue((torch.diff(positions[:, 1:, 0], dim=1) > 0.0).all())

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
    def test_step_length_minimums_split_stride_and_base_worlds(self):
        minimums = StoneCourseEnv.build_step_length_minimums(
            10, 0.7, 1.10, 0.65, "cpu"
        )
        self.assertEqual(minimums.shape, (10,))
        self.assertTrue((minimums[:7] == 1.10).all())
        self.assertTrue((minimums[7:] == 0.65).all())

        disabled = StoneCourseEnv.build_step_length_minimums(
            10, 0.7, 0.0, 0.65, "cpu"
        )
        self.assertTrue((disabled == 0.65).all())

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
        env.step_length_minimums = torch.full((2,), 0.65)
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
        env.landing_margin_inactive = 0.0
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

    def test_inactive_sphere_margin_forgives_small_overhang(self):
        # World 1's violating toe sphere is inactive and only 0.03 m past the
        # shrunk interior bound; the margin forgives it. A loaded sphere in
        # the same position would remain a strict violation.
        env = self.make_contact_env()
        env.landing_margin_inactive = 0.05

        invalid = env._invalid_edge_touchdown()

        self.assertEqual(invalid.tolist(), [False, False, False])

        strict = self.make_contact_env()
        strict.landing_margin_inactive = 0.05
        strict.collider_forces[1, 1] = 100.0  # load the overhanging toe
        invalid = strict._invalid_edge_touchdown()
        self.assertEqual(invalid.tolist(), [False, True, False])

    def make_below_support_env(self) -> StoneCourseEnv:
        env = object.__new__(StoneCourseEnv)
        env.course = make_spec()
        env.foot_collider_ids = torch.tensor([0, 1, 2])
        env.foot_collider_radii = torch.full((3,), 0.02)
        env.foot_side_masks = (
            torch.tensor([True, True, False]),
            torch.tensor([False, False, True]),
        )
        env.below_support_margin = 0.15
        # One slab at top height 0.45 near the origin, the rest far ahead at
        # a lower level (top 0.25) so nearest-slab attribution matters.
        env.stone_positions = torch.tensor(
            [[[0.0, 0.40, 0.0]] + [[5.0, 0.20, 0.0]] * 4]
        ).repeat(3, 1, 1)
        env.collider_positions = torch.zeros((3, 3, 3))

        # World 0: all spheres of both feet at the near slab's top. Supported.
        env.collider_positions[0, :, UP_IDX_TEST] = 0.47
        # World 1: both feet 0.2 m below the near slab top. Fallen.
        env.collider_positions[1, :, UP_IDX_TEST] = 0.27
        # World 2: left foot dropped but the right foot stands on the distant
        # lower slab; stepping down a descent must not terminate.
        env.collider_positions[2, 0, UP_IDX_TEST] = 0.10
        env.collider_positions[2, 1, UP_IDX_TEST] = 0.10
        env.collider_positions[2, 2] = torch.tensor([5.0, 0.27, 0.0])
        return env

    def test_both_feet_below_their_nearest_supports_is_a_fall(self):
        env = self.make_below_support_env()

        below = env._feet_below_supports()

        self.assertEqual(below.tolist(), [False, True, False])

    def test_zero_command_episodes_do_not_feed_the_terrain_curriculum(self):
        env = object.__new__(StoneCourseEnv)
        env.terrain_curriculum = make_curriculum(window=2)
        env.command_speed_range = (0.0, 2.5)
        env._episode_started = torch.tensor([True, True, True])
        # Worlds 0 and 1 stood still successfully; world 2 walked and failed.
        env.command_speeds = torch.tensor([0.0, 0.0, 1.2])
        env._last_success = torch.tensor([True, True, False])

        env._record_finished_episodes(torch.tensor([0, 1, 2]))

        # Only the walking episode counts: one episode, zero successes.
        self.assertEqual(env.terrain_curriculum.episodes, 1)
        self.assertEqual(env.terrain_curriculum.successes, 0)

        # With commands disabled every episode still counts as before.
        env.command_speed_range = (0.0, 0.0)
        env._record_finished_episodes(torch.tensor([0, 1, 2]))
        self.assertEqual(env.terrain_curriculum.episodes, 0)  # window of 2 rolled over
        self.assertEqual(env.terrain_curriculum.last_completion_rate, 0.5)

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

    def make_truncation_env(self) -> StoneCourseEnv:
        env = object.__new__(StoneCourseEnv)
        env.time = torch.tensor([12.0, 12.0, 1.0])
        env.max_episode_duration = 12.0
        env.root_pos = torch.tensor([
            [13.0, 1.4, 0.0],
            [5.0, 1.4, 0.0],
            [2.0, 1.4, 0.0],
        ])
        env._episode_start_x = torch.zeros(3)
        env._episode_start_time = torch.zeros(3)
        env.curriculum_min_progress = 12.0
        env.curriculum_command_fraction = 0.0
        env._last_success = torch.zeros(3, dtype=torch.bool)
        env._last_timed_out = torch.zeros(3, dtype=torch.bool)
        return env

    def test_time_limit_is_neutral_and_curriculum_success_requires_progress(self):
        env = self.make_truncation_env()

        truncated = env._get_truncated()

        self.assertEqual(truncated.tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(env._last_success.tolist(), [True, False, False])
        self.assertEqual(env._last_timed_out.tolist(), [True, True, False])

    def test_truncation_is_relative_to_each_worlds_episode_start(self):
        env = self.make_truncation_env()
        # World 0 continued at t=6 s, so at t=12 s only 6 s have elapsed.
        env._episode_start_time = torch.tensor([6.0, 0.0, 0.0])

        truncated = env._get_truncated()

        self.assertEqual(truncated.tolist(), [0.0, 1.0, 0.0])

    def test_curriculum_success_is_relative_to_each_worlds_command(self):
        env = self.make_truncation_env()
        env.curriculum_command_fraction = 0.7
        # World 0: commanded 1.35 m/s, needs 0.7*1.35*12 = 11.34 m; made 13 m.
        # World 1: commanded 1.35 m/s, needs 11.34 m; made only 5 m.
        # World 2: not timed out yet, never a success.
        env.command_speeds = torch.tensor([1.35, 1.35, 1.35])

        truncated = env._get_truncated()

        self.assertEqual(truncated.tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(env._last_success.tolist(), [True, False, False])

    def test_zero_command_success_is_surviving_to_the_time_limit(self):
        env = self.make_truncation_env()
        env.curriculum_command_fraction = 0.7
        env.command_speeds = torch.zeros(3)
        # No world moved anywhere.
        env.root_pos = torch.zeros((3, 3))

        env._get_truncated()

        self.assertEqual(env._last_success.tolist(), [True, True, False])

    def test_continued_worlds_keep_walking_and_failed_worlds_reset(self):
        env = object.__new__(StoneCourseEnv)
        env.device = torch.device("cpu")
        env.num_worlds = 4
        env.continuation_probability = 1.0
        env.command_speed_range = (0.0, 0.0)
        env.root_pos = torch.tensor([
            [13.0, 1.4, 0.0],
            [14.0, 1.4, 0.0],
            [3.0, 1.4, 0.0],
            [5.0, 1.4, 0.0],
        ])
        env.time = torch.full((4,), 12.0)
        env._episode_start_x = torch.zeros(4)
        env._episode_start_time = torch.zeros(4)
        env._episode_started = torch.ones(4, dtype=torch.bool)
        env._last_timed_out = torch.tensor([True, True, False, False])
        env._last_terminated = torch.tensor([False, False, True, False])
        env._last_success = torch.tensor([True, True, False, False])
        env.episode_slabs_recycled = torch.full((4,), 9, dtype=torch.long)
        env.terrain_curriculum = make_curriculum()

        performed = {}
        env_reset = lambda _self, resets: performed.setdefault(
            "mask", resets.squeeze(-1).bool().clone()
        )
        import unittest.mock as mock
        with mock.patch.object(
            StoneCourseEnv.__bases__[0], "_perform_reset", env_reset
        ):
            # Worlds 0/1 timed out healthy, world 2 fell, world 3 keeps going.
            env._perform_reset(torch.tensor([[1.0], [1.0], [1.0], [0.0]]))

        self.assertEqual(performed["mask"].tolist(), [False, False, True, False])
        self.assertEqual(env._episode_start_x[:2].tolist(), [13.0, 14.0])
        self.assertEqual(env._episode_start_time[:2].tolist(), [12.0, 12.0])
        self.assertEqual(env.episode_slabs_recycled[:2].tolist(), [0, 0])
        self.assertEqual(env._last_success[:2].tolist(), [False, False])
        # Both continued episodes were still recorded for the curriculum.
        self.assertEqual(env.terrain_curriculum.episodes, 2)
        self.assertEqual(env.terrain_curriculum.successes, 2)

    def test_uprightness_scales_the_alive_bonus(self):
        env = object.__new__(StoneCourseEnv)
        env.device = torch.device("cpu")
        env.num_worlds = 3
        env.command_speed_range = (0.0, 0.0)
        env.upright_pelvis_range = (0.6, 1.0)
        env.target_speed = 1.35
        env.root_id = 0
        env.reward_lambdas = {"lambda_vel": 0.1, "lambda_alive": 0.01}
        env.body_velocities = torch.zeros((3, 1, 6))
        env.body_velocities[:, 0, 3] = 1.0
        env.foot_collider_ids = torch.tensor([0, 1])
        env.foot_collider_radii = torch.full((2,), 0.02)
        env.collider_positions = torch.zeros((3, 2, 3))
        env.collider_positions[:, :, 1] = 0.42
        # Pelvis heights: fully upright, mid-band, fully crouched.
        env.root_pos = torch.tensor([
            [0.0, 1.5, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.9, 0.0],
        ])

        env._compute_raw_reward_dict()

        alive = env.reward_dict["rew_alive"]
        self.assertEqual(alive[0].item(), 1.0)
        self.assertAlmostEqual(alive[1].item(), 0.5, places=5)
        self.assertEqual(alive[2].item(), 0.0)

    def test_command_sampler_places_a_point_mass_at_zero(self):
        env = object.__new__(StoneCourseEnv)
        env.device = torch.device("cpu")
        env.command_speed_range = (0.0, 2.5)
        env.command_zero_probability = 0.2
        env.command_speeds = torch.full((4096,), -1.0)

        torch.manual_seed(0)
        env._resample_command_speeds(torch.arange(4096))

        zero_fraction = (env.command_speeds == 0.0).float().mean().item()
        self.assertGreater(zero_fraction, 0.15)
        self.assertLess(zero_fraction, 0.25)
        nonzero = env.command_speeds[env.command_speeds > 0.0]
        self.assertLessEqual(nonzero.max().item(), 2.5)

    def test_command_speed_caps_velocity_reward_per_world(self):
        env = object.__new__(StoneCourseEnv)
        env.device = torch.device("cpu")
        env.num_worlds = 2
        env.command_speed_range = (0.9, 1.8)
        env.upright_pelvis_range = (0.0, 0.0)
        env.root_id = 0
        env.reward_lambdas = {"lambda_vel": 0.1, "lambda_alive": 0.01}
        env.command_speeds = torch.tensor([1.0, 1.6])
        env.body_velocities = torch.zeros((2, 1, 6))
        env.body_velocities[:, 0, 3] = 1.3  # both run at 1.3 m/s

        env._compute_raw_reward_dict()

        vel = env.reward_dict["rew_vel"]
        # World 0 commanded 1.0: overspeed by 0.3 -> 1.0 - 0.3 = 0.7.
        self.assertAlmostEqual(vel[0].item(), 0.7, places=5)
        # World 1 commanded 1.6: below command -> raw velocity.
        self.assertAlmostEqual(vel[1].item(), 1.3, places=5)

    def test_external_reset_never_continues_a_timed_out_world(self):
        env = object.__new__(StoneCourseEnv)
        env.device = torch.device("cpu")
        env.num_worlds = 2
        env.continuation_probability = 1.0
        # The previous rollout ended in a healthy timeout for both worlds.
        env._last_timed_out = torch.tensor([True, True])
        env._last_terminated = torch.tensor([False, False])

        performed = {}
        import unittest.mock as mock
        base = StoneCourseEnv.__bases__[0]
        with mock.patch.object(
            base, "_perform_reset",
            lambda _self, resets: performed.setdefault(
                "mask", resets.squeeze(-1).bool().clone()
            ),
        ), mock.patch.object(
            base, "reset",
            lambda _self: _self._perform_reset(torch.ones((2, 1))),
        ):
            env.reset()

        # Every world must be genuinely reset, none continued.
        self.assertEqual(performed["mask"].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
