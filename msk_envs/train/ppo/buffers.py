import torch


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
