import math
import unittest
from types import SimpleNamespace

import torch

from msk_envs.envs.env_stone_course import StoneCourseEnv, StoneCourseSpec
from msk_envs.envs.env_config import EnvConfig


def heading_env(count=1024, backward_probability=0.5):
    env = object.__new__(StoneCourseEnv)
    env.device = torch.device("cpu")
    env.heading_commands_enabled = True
    env.heading_fixed = False
    env.heading_fixed_degrees = 90.0
    env.heading_switch_fraction = 0.5
    env.heading_switch_grace = 1.5
    env.heading_switch_backward_probability = backward_probability
    env.heading_gap_scale = 0.4
    env.time = torch.zeros(count)
    env._last_heading_change_time = torch.zeros(count)
    env.command_headings = torch.zeros(count)
    env._pending_headings = torch.zeros(count)
    env._heading_switched = torch.zeros(count, dtype=torch.bool)
    env._episode_started = torch.ones(count, dtype=torch.bool)
    env._episode_start_time = torch.zeros(count)
    env.max_episode_duration = 12.0
    env.course = SimpleNamespace(step_length_range=(0.25, 1.5))
    env.terrain_curriculum = SimpleNamespace(current_maximum=0.7)
    return env


def direction(degrees):
    value = int(round(degrees))
    return 180 if abs(value) == 180 else value


class HeadingSwitchTest(unittest.TestCase):
    def test_invalid_backward_modes_fail_before_building_simulator(self):
        cases = [
            (-0.1, 0.5, 90, "must be in"),
            (1.1, 0.5, 90, "must be in"),
            (float("nan"), 0.5, 90, "must be in"),
            (0.5, 0.0, 90, "requires"),
            (0.5, 0.5, 180, "requires"),
        ]
        for probability, fraction, heading, message in cases:
            config = EnvConfig(
                course_heading_switch_backward_probability=probability,
                course_heading_switch_fraction=fraction,
                course_heading_max_degrees=heading,
            )
            with self.subTest(probability=probability, fraction=fraction, heading=heading):
                with self.assertRaisesRegex(ValueError, message):
                    StoneCourseEnv(2, config, torch.device("cpu"), False, False)

    def test_mixed_sampler_covers_all_eight_directed_pairs(self):
        env = heading_env(4096)
        torch.manual_seed(17)
        env._resample_command_headings(torch.arange(4096))
        pairs = list(zip(
            map(direction, torch.rad2deg(env.command_headings).tolist()),
            map(direction, torch.rad2deg(env._pending_headings).tolist()),
        ))
        expected = {
            (0, 90), (90, 0), (0, -90), (-90, 0),
            (180, 90), (90, 180), (180, -90), (-90, 180),
        }
        self.assertEqual(set(pairs), expected)
        for pair in expected:
            self.assertGreater(pairs.count(pair), 350)
            self.assertLess(pairs.count(pair), 700)

    def test_probability_endpoints_and_legacy_random_stream(self):
        env = heading_env(backward_probability=0.0)
        torch.manual_seed(23)
        signs = torch.where(torch.rand(1024) < 0.5, -1.0, 1.0)
        start_sideways = torch.rand(1024) < 0.5
        expected_start = torch.where(start_sideways, signs * math.pi / 2, 0.0)
        expected_end = torch.where(start_sideways, 0.0, signs * math.pi / 2)
        expected_next_random = torch.rand(8)
        torch.manual_seed(23)
        env._resample_command_headings(torch.arange(1024))
        torch.testing.assert_close(env.command_headings, expected_start)
        torch.testing.assert_close(env._pending_headings, expected_end)
        torch.testing.assert_close(torch.rand(8), expected_next_random)

        env.heading_switch_backward_probability = 1.0
        env._resample_command_headings(torch.arange(1024))
        endpoints = torch.cat((env.command_headings, env._pending_headings))
        self.assertFalse((endpoints.abs() < 0.1).any())
        self.assertTrue((endpoints.abs() > 3.0).any())

    def test_continuation_keeps_current_heading_and_only_plans_quarter_turns(self):
        env = heading_env(64)
        current = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi]).repeat(16)
        env.command_headings.copy_(current)
        env._heading_switched[:] = True
        env.time[:] = 12.0
        for _ in range(20):
            env._resample_command_headings(torch.arange(64), continued=True)
            torch.testing.assert_close(env.command_headings, current)
            delta = env._pending_headings - current
            turn = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
            torch.testing.assert_close(turn, torch.full((64,), math.pi / 2))
            self.assertFalse(env._heading_switched.any())
            torch.testing.assert_close(env._last_heading_change_time, env.time)

    def test_only_selected_worlds_are_resampled(self):
        env = heading_env(8)
        env.command_headings[:] = 0.123
        env._pending_headings[:] = 0.456
        env._heading_switched[:] = True
        env.time[:] = 5
        env._resample_command_headings(torch.tensor([1, 6]))
        keep = torch.tensor([0, 2, 3, 4, 5, 7])
        torch.testing.assert_close(env.command_headings[keep], torch.full((6,), 0.123))
        torch.testing.assert_close(env._pending_headings[keep], torch.full((6,), 0.456))
        self.assertTrue(env._heading_switched[keep].all())
        self.assertTrue((env._last_heading_change_time[keep] == 0).all())

    def test_switch_uses_each_episode_clock_and_runs_once(self):
        env = heading_env(4)
        env._episode_start_time = torch.tensor([0.0, 10.0, 20.0, 0.0])
        env.time = torch.tensor([5.99, 16.0, 26.1, 7.0])
        env._heading_switched[2] = True
        env._episode_started[3] = False
        env._pending_headings[:] = math.pi / 2
        calls = []
        env._relay_course_ahead = lambda ids: calls.append(ids.tolist())
        env._switch_pending_headings()
        self.assertEqual(calls, [[1]])
        self.assertAlmostEqual(env.command_headings[1].item(), math.pi / 2)
        self.assertTrue((env.command_headings[[0, 2, 3]] == 0).all())
        env.time[0] = 6.0
        env._switch_pending_headings()
        env._switch_pending_headings()
        self.assertEqual(calls, [[1], [0]])

    def test_backward_facing_and_turn_grace(self):
        env = heading_env(3)
        env.command_headings[:] = math.pi
        env.fwd_axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
        env.cos_angle_threshold = math.cos(math.pi / 4)
        env.body_rotations = torch.tensor([
            [[0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ])
        env._last_heading_change_time[:] = 6.0
        env.time[:] = torch.tensor([7.5, 7.49, 7.5])
        self.assertEqual(env._is_body_facing_direction(0).tolist(), [True, True, False])

    def test_heading_spacing_keeps_backward_longitudinal_and_sides_short(self):
        env = heading_env(4)
        env.command_headings[:] = torch.tensor([0, math.pi, math.pi / 2, -math.pi / 2])
        torch.testing.assert_close(
            env._heading_scaled_maximums(torch.arange(4)),
            torch.tensor([0.7, 0.7, 0.28, 0.28]),
        )

    def test_relayout_preserves_support_and_next_target(self):
        env = heading_env(2)
        env.course = StoneCourseSpec(
            num_stones=12, step_length_range=(0.25, 1.5), lateral_jitter=0.08,
            slab_size=(0.36, 0.10, 0.36), top_height=0.45,
            top_height_range=(0.2, 1.05), elevation_angle_max_degrees=50,
            yaw_angle_max_degrees=20, surface_tilt_max_degrees=20, lookahead=4,
        )
        env.terrain_curriculum.current_height_scale = 0.1
        env.terrain_curriculum.current_elevation_maximum_degrees = 0.0
        env.terrain_curriculum.current_yaw_maximum_degrees = 0.0
        env.terrain_curriculum.current_surface_tilt_maximum_degrees = 0.0
        env.root_pos = torch.zeros((2, 3))
        env.stone_ids = torch.arange(12)
        env.stone_positions = torch.zeros((2, 12, 3))
        env.stone_positions[:, :, 0] = torch.arange(12) * 0.5
        env.stone_positions[:, :, 1] = -0.005
        env.stone_rotations = torch.zeros((2, 12, 4))
        env.stone_surface_tilts = torch.zeros((2, 12, 2))
        env.collider_local_transforms = torch.zeros((2, 12, 7))
        env.collider_forces = torch.zeros((2, 12))
        env.collider_forces[:, 0] = 100.0
        env.next_lateral_sign = torch.ones(2)
        env.step_length_minimums = torch.full((2,), 0.25)
        env.command_headings[0] = math.pi / 2
        before = env.stone_positions.clone()
        env._relay_course_ahead(torch.tensor([0]))
        torch.testing.assert_close(env.stone_positions[0, :2], before[0, :2])
        torch.testing.assert_close(env.stone_positions[1], before[1])
        distances = torch.linalg.vector_norm(torch.diff(env.stone_positions[0, 1:], dim=0), dim=1)
        self.assertTrue((distances >= 0.25 - 1e-6).all())
        self.assertTrue((distances <= 0.28 + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
