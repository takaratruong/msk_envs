import unittest

import torch

from experiments.stone_course.expand_checkpoint_observation import expand_checkpoint


class StoneCourseCheckpointMigrationTest(unittest.TestCase):
    def test_inserts_zero_policy_and_critic_features_without_moving_actions(self):
        actor_weight = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12)
        critic_weight = torch.arange(3 * 17, dtype=torch.float32).reshape(3, 17)
        checkpoint = {
            "actor_state_dict": {"net.0.weight": actor_weight},
            "qnet_state_dict": {"qnets.0.net.0.weight": critic_weight},
            "qnet_target_state_dict": {"qnets.0.net.0.weight": critic_weight.clone()},
            "obs_normalizer_state": {
                "_mean": torch.arange(12, dtype=torch.float32).reshape(1, 12),
                "_var": torch.ones(1, 12),
                "_std": torch.ones(1, 12),
                "count": torch.tensor(999999),
            },
        }

        migrated = expand_checkpoint(checkpoint)
        actor = migrated["actor_state_dict"]["net.0.weight"]
        critic = migrated["qnet_state_dict"]["qnets.0.net.0.weight"]

        self.assertEqual(actor.shape, (2, 24))
        self.assertTrue(torch.equal(actor[:, :8], actor_weight[:, :8]))
        self.assertTrue(torch.equal(actor[:, 8:20], torch.zeros(2, 12)))
        self.assertTrue(torch.equal(actor[:, 20:], actor_weight[:, 8:]))
        self.assertEqual(critic.shape, (3, 29))
        self.assertTrue(torch.equal(critic[:, 24:], critic_weight[:, 12:]))
        self.assertEqual(migrated["obs_normalizer_state"]["count"].item(), 1024)
        self.assertEqual(
            migrated["environment_state"]["terrain_curriculum"]["current_maximum"],
            0.80,
        )


if __name__ == "__main__":
    unittest.main()
