import torch
from collections import deque


class RolloutBuffer:
    """ Used for on-policy rollouts. Re-use this buffer across rollouts """

    def __init__(self, n_steps, n_envs, obs_dim, act_dim, device):
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs, act_dim), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.terminated = torch.zeros((n_steps, n_envs), device=device)

        # Computed after rollout
        self.next_value = torch.zeros((n_envs,), device=device)
        self.next_dones = torch.zeros((n_envs,), device=device)
        self.advantages = torch.zeros((n_steps, n_envs), device=device)
        self.returns = torch.zeros((n_steps, n_envs), device=device)

        self.horizon = n_steps
        self.n_envs = n_envs
        return

    def get_total_steps(self):
        return self.horizon * self.n_envs

    def get_minibatch(self, indices):
        o = self.obs.view(-1, *self.obs.shape[2:])[indices]
        a = self.actions.view(-1, *self.actions.shape[2:])[indices]
        lp = self.log_probs.view(-1)[indices]
        v = self.values.view(-1)[indices]
        adv = self.advantages.view(-1)[indices]
        ret = self.returns.view(-1)[indices]

        return o, a, lp, v, adv, ret


class ReplayBuffer:
    """ Off-policy replay buffer"""

    def __init__(self, buffer_size, n_envs, obs_dim, act_dim, device):
        self.n_envs = n_envs
        per_env = buffer_size // n_envs

        self.obs = torch.zeros((per_env, n_envs, obs_dim), device=device)
        self.next_obs = torch.zeros((per_env, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((per_env, n_envs, act_dim), device=device)
        self.rewards = torch.zeros((per_env, n_envs), device=device)
        self.dones = torch.zeros((per_env, n_envs), device=device)
        self.per_env = per_env

        self.pos = 0
        self.full = False
        return

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env:
            self.full = True
            self.pos = 0
        return

    def sample(self, batch_size):
        if self.full:
            batch_inds = torch.randint(0, self.per_env, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.n_envs, size=(batch_size,))
        return (
            self.obs[batch_inds, env_inds],
            self.next_obs[batch_inds, env_inds],
            self.actions[batch_inds, env_inds],
            self.rewards[batch_inds, env_inds],
            self.dones[batch_inds, env_inds],
        )


class NStepReplayBuffer:
    """
    Replay buffer used for computing n-step returns in off-policy algorithms
    """

    def __init__(self, buffer_size, n_envs, obs_dim, act_dim, n_steps, gamma,
                 device):
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        self.gamma = gamma
        per_env = buffer_size // n_envs

        self.obs = torch.zeros((per_env, n_envs, obs_dim), device=device)
        self.next_obs = torch.zeros((per_env, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((per_env, n_envs, act_dim), device=device)
        self.rewards = torch.zeros((per_env, n_envs), device=device)
        self.dones = torch.zeros((per_env, n_envs), device=device)
        self.per_env = per_env

        self.pos = 0
        self.full = False
        self.n_step_buffers = [deque(maxlen=n_steps) for _ in range(n_envs)]
        return

    def _get_n_step_info(self, env_idx):
        """Calculate n-step return and corresponding next obs"""
        transitions = list(self.n_step_buffers[env_idx])
        obs, action = transitions[0][:2]
        n_reward = 0.0
        done = False
        for i, (_, _, reward, next_obs, d) in enumerate(transitions):
            n_reward += (self.gamma ** i) * reward
            if d:
                done = True
                break
        final_next_obs = transitions[-1][3]
        return obs, action, n_reward, final_next_obs, done

    def add(self, obs, next_obs, action, reward, done):
        for env_idx in range(self.n_envs):
            transition = (
                obs[env_idx],
                action[env_idx],
                reward[env_idx].item(),
                next_obs[env_idx],
                done[env_idx].item(),
            )
            self.n_step_buffers[env_idx].append(transition)

            if len(self.n_step_buffers[env_idx]) == self.n_steps or done[env_idx]:
                o, a, r, no, d = self._get_n_step_info(env_idx)

                # Store into main buffer
                self.obs[self.pos, env_idx] = o
                self.actions[self.pos, env_idx] = a
                self.rewards[self.pos, env_idx] = r
                self.next_obs[self.pos, env_idx] = no
                self.dones[self.pos, env_idx] = d

        self.pos += 1
        if self.pos == self.per_env:
            self.full = True
            self.pos = 0

        # Clear finished trajectories from n-step buffer
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self.n_step_buffers[env_idx].clear()

        return

    def sample(self, batch_size):
        max_index = self.per_env if self.full else self.pos
        batch_inds = torch.randint(0, max_index, size=(batch_size,))
        env_inds = torch.randint(0, self.n_envs, size=(batch_size,))
        return (
            self.obs[batch_inds, env_inds],
            self.next_obs[batch_inds, env_inds],
            self.actions[batch_inds, env_inds],
            self.rewards[batch_inds, env_inds],
            self.dones[batch_inds, env_inds],
        )