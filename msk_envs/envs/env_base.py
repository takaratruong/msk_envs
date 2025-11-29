import torch
import msk_warp

from .env_config import EnvConfig


class MSKEnv:
    """ Superclass for MSK environments """

    def __init__(
            self,
            num_envs: int,
            env_config: EnvConfig,
            device: torch.device
    ):
        self.num_worlds = num_envs
        self.device = device
        load_result = msk_warp.load_model(env_config.model_path, num_envs)
        self.m, self.d = load_result.model, load_result.data
        self.body_id_lookup = load_result.body_id_lookup

        self.num_qpos = msk_warp.get_num_qpos(self.m)
        self.num_dofs = msk_warp.get_num_dofs(self.m)
        self.num_muscles = msk_warp.get_num_muscles(self.m)

        # [num_envs, num_muscles]
        self.muscle_excitations = msk_warp.muscle_excitations(self.d)
        self.muscle_activations = msk_warp.muscle_activations(self.d)
        self.muscle_fiber_lengths = msk_warp.muscle_fiber_lengths(self.d)
        self.muscle_fiber_velocities = msk_warp.muscle_fiber_velocities(self.d)

        # [num_envs, num_bodies, 3]
        self.body_positions = msk_warp.body_positions(self.d)
        # [num_envs, num_bodies, 4] (w, x, y, z)
        self.body_rotations = msk_warp.body_rotations(self.d)
        # [num_envs, num_bodies, 6] (ang, lin)
        self.body_velocities = msk_warp.body_velocities(self.d)
        # [num_envs, num_dofs]
        self.joint_positions = msk_warp.joint_positions(self.d)
        # [num_envs, num_dofs]
        self.joint_velocities = msk_warp.joint_velocities(self.d)

        self.action_range = (-1.0, 1.0)
        self.max_episode_duration = env_config.max_episode_duration
        self.delta_t = env_config.delta_t

        self.reset_tensor = torch.zeros(
            (num_envs, 1), dtype=torch.float32, device=device)
        self.start_pose = torch.zeros(
            (num_envs, self.num_qpos), dtype=torch.float32, device=device)
        self.start_velocity = torch.zeros(
            (num_envs, self.num_dofs), dtype=torch.float32, device=device)

        self.reward_dict = {}
        self.reward_lambdas = env_config.reward_lambdas

    def num_obs(self) -> int:
        return self._get_obs().shape[1]

    def num_actions(self) -> int:
        return self._get_actions().shape[1]

    def get_time(self) -> torch.Tensor:
        return msk_warp.get_time(self.d)

    def _upon_reset(self, reset_mask: torch.Tensor) -> None:
        """ Hook for additional reset behavior in subclasses """
        return

    def _handle_reset(self):
        # TODO
        self.reset_tensor.fill_(0.0)
        return

    # The following are environment-specific and need to be implemented
    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_actions(self) -> torch.Tensor:
        raise NotImplementedError

    def _compute_raw_reward_dict(self):
        """ Guarantee to only run once per step """
        raise NotImplementedError

    def _get_terminated(self):
        raise NotImplementedError

    # Rest is standard
    def get_blank_actions(self) -> torch.Tensor:
        """ Returns actions of all 0 in correct shape """
        return torch.zeros_like(self._get_actions())

    def get_random_actions(self) -> torch.Tensor:
        """ Returns randomly uniform actions in correct shape """
        return (torch.rand_like(self._get_actions()) *
                self.action_range[1] * 2 - self.action_range[1])

    def get_scaled_reward_dict(self) -> dict:
        reward_dict = self.reward_dict
        scaled_reward_dict = {}
        for key, raw_value in reward_dict.items():
            lambda_key = key.replace("rew_", "lambda_")
            lambda_value = self.reward_lambdas[lambda_key]
            scaled_reward_dict[key] = lambda_value * raw_value
        return scaled_reward_dict

    def get_rewards(self) -> torch.Tensor:
        scaled_rew_dict = self.get_scaled_reward_dict()
        total_rewards = torch.zeros(self.num_worlds, device=self.device)
        for key, value in scaled_rew_dict.items():
            total_rewards += value
        return total_rewards.detach()

    def _get_truncated(self):
        truncated = (self.get_time() >= self.max_episode_duration).float()
        return truncated.detach()

    def _set_muscle_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitation = (clamped_action + 1.0) / 2.0
        self.muscle_excitations.copy_(excitation)
        return

    def _set_actuator_excitations(self, raw_action) -> None:
        # Clamp to [-1, 1], then map to [0, 1]
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        excitations = (clamped_action + 1.0) / 2.0
        # self.actuator_excitations.copy_(excitations)
        return

    def _set_actions(self, raw_action) -> None:
        self._set_muscle_excitations(raw_action[:, :self.num_muscles])
        self._set_actuator_excitations(raw_action[:, self.num_muscles:])
        return

    def step(self, actions):
        # Sim step
        self._set_actions(actions)
        msk_warp.step_to(self.m, self.d, self.delta_t)

        # Only compute reward dict once per step
        self._compute_raw_reward_dict()

        final_obs = self._get_obs()
        rew = self.get_rewards()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Reset any worlds that are done
        done = torch.clamp(terminated + truncated, 0.0, 1.0).unsqueeze(-1)
        self.reset_tensor.copy_(done)
        self._upon_reset(done.squeeze(-1).bool())
        self._handle_reset()

        # Gym api requires the observation after the reset
        obs = self._get_obs()

        # Return raw reward terms for logging
        raw_rewards = self.reward_dict
        scaled_rewards = self.get_scaled_reward_dict()

        info = {
            "final_observation": final_obs,
            "raw_rewards": raw_rewards,
            "scaled_rewards": scaled_rewards,
        }

        # Assert actions aren't nan
        assert not torch.isnan(actions).any(), "Actions contain NaN!"
        assert not torch.isnan(obs).any(), "Observations contain NaN!"

        return obs, rew, terminated, truncated, info

    def reset(self):
        self.reset_tensor.fill_(1.0)
        self._handle_reset()
        obs = self._get_obs()
        return obs

    def lookup_body_id(self, body_name: str) -> int:
        return self.body_id_lookup[body_name]
