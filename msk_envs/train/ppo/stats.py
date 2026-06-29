import torch
import torch.nn as nn
from dataclasses import dataclass
from time import perf_counter


@dataclass
class PPOStatsTracker:
    def __init__(self, n_envs: int, device: torch.device):
        self.curr_rewards = torch.zeros(n_envs, device=device)
        self.episode_lengths = torch.zeros(n_envs, device=device)

        self.mean_reward = AverageMeter(1, 100).to(device)
        self.mean_episode_length = AverageMeter(1, 100).to(device)

        self.a_loss = 0.0
        self.c_loss = 0.0
        self.e_loss = 0.0
        self.b_loss = 0.0

    def update(self, rew: torch.Tensor, dones: torch.Tensor):
        self.curr_rewards += rew
        self.episode_lengths += 1

        # Check for any worlds that finished
        if dones.any():
            ind_done = dones.nonzero(as_tuple=False).squeeze(-1)
            rewards_done = self.curr_rewards[ind_done]
            lengths_done = self.episode_lengths[ind_done]
            # Update trackers
            self.mean_reward.update(rewards_done.unsqueeze(-1))
            self.mean_episode_length.update(lengths_done.unsqueeze(-1))

        # Reset where done
        self.curr_rewards = self.curr_rewards * (1 - dones)
        self.episode_lengths = self.episode_lengths * (1 - dones)

    def set_losses(self, a_loss, c_loss, e_loss, b_loss):
        self.a_loss = a_loss
        self.c_loss = c_loss
        self.e_loss = e_loss
        self.b_loss = b_loss

    def reset(self):
        pass

    def print(self):
        print(f"Mean reward: {self.mean_reward.get_mean():.2f}. "
              f"Mean episode length: {self.mean_episode_length.get_mean():.2f}")
        pass


class PerfTimer:
    def __init__(self):
        self.t_iter = 0.0
        self.t_sim = 0.0
        self.t_inference = 0.0
        self.t_rollout = 0.0
        self.t_update = 0.0
        self.iter_step = 0

        self.global_step = 0

        # Temporary
        self.iter_start = None
        self.rollout_start = None
        self.sim_start = None
        self.inference_start = None
        self.update_start = None

    def start_iter(self):
        self.iter_start = perf_counter()

    def end_iter(self):
        assert (self.iter_start is not None), "Iteration start not set"
        elapsed = perf_counter() - self.iter_start
        self.t_iter += elapsed
        self.iter_start = None

    def reset(self):
        self.t_iter = 0.0
        self.t_sim = 0.0
        self.t_inference = 0.0
        self.t_rollout = 0.0
        self.t_update = 0.0
        self.iter_step = 0

    def start_rollout(self):
        self.rollout_start = perf_counter()

    def end_rollout(self):
        assert (self.rollout_start is not None), "Rollout start not set"
        elapsed = perf_counter() - self.rollout_start
        self.t_rollout += elapsed
        self.rollout_start = None

    def start_sim(self):
        self.sim_start = perf_counter()

    def end_sim(self):
        assert (self.sim_start is not None), "Simulation start not set"
        elapsed = perf_counter() - self.sim_start
        self.t_sim += elapsed
        self.sim_start = None

    def start_inference(self):
        self.inference_start = perf_counter()

    def end_inference(self):
        assert (self.inference_start is not None), "Inference start not set"
        elapsed = perf_counter() - self.inference_start
        self.t_inference += elapsed
        self.inference_start = None

    def start_update(self):
        self.update_start = perf_counter()

    def end_update(self):
        assert (self.update_start is not None), "Update start not set"
        elapsed = perf_counter() - self.update_start
        self.t_update += elapsed
        self.update_start = None

    def get_iter_fps(self):
        return int(self.iter_step / self.t_iter) if self.t_iter > 0 else 0

    def get_rollout_fps(self):
        return int(self.iter_step / self.t_rollout) if self.t_rollout > 0 else 0

    def get_sim_fps(self):
        return int(self.iter_step / self.t_sim) if self.t_sim > 0 else 0

    def get_inference_fps(self):
        return int(
            self.iter_step / self.t_inference) if self.t_inference > 0 else 0

    def get_update_fps(self):
        return int(self.iter_step / self.t_update) if self.t_update > 0 else 0

    def add_steps(self, steps: int):
        self.iter_step += steps
        self.global_step += steps

    def print(self):
        fps = self.get_iter_fps()
        print(f"Took {self.t_iter:.2f} seconds. "
              f"FPS: {fps}. Global {self.global_step:_}")
        print(f"Sim only: {self.t_sim:.2f}s, "
              f"Inference: {self.t_inference:.2f}s, "
              f"Update: {self.t_update:.2f}s")


class AverageMeter(nn.Module):
    def __init__(self, in_shape: int, max_size: int) -> None:
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        size = values.size()[0]
        new_mean = torch.mean(values.float(), dim=0)
        size = min(size, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def get_mean(self):
        if self.current_size == 0:
            return 0.0
        return self.mean.squeeze(0).cpu().numpy()

    def empty(self):
        return self.current_size == 0
