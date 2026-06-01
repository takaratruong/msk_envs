import torch
import torch.distributed as dist
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
            self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Calculate global batch size arithmetically
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            # Calculate the stats
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            # Sync the stats across all processes
            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            # Calculate the mean and variance of the global batch
            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = (
                    global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            )
            batch_mean = batch_mean_shifted + self._mean

        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
            self,
            gamma: float,
            device: torch.device,
            g_max: float = 10.0,
            epsilon: float = 1e-8,
    ):
        super().__init__()
        # running estimate of the discounted return
        self.register_buffer("G", torch.zeros(1, device=device))
        # running-max
        self.register_buffer("G_r_max", torch.zeros(1, device=device))
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
            self,
            rewards: torch.Tensor,
            dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))

        local_max = torch.max(torch.abs(self.G))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        self.G_r_max = max(self.G_r_max, local_max)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


class BatchRenorm(nn.Module):
    def __init__(
            self,
            num_features: int,
            decay_rate: float,
            eps: float = 1e-3,
            r_max: float = 3.0,
            d_max: float = 5.0,
            warmup_steps: int = 10,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = 1.0 - decay_rate  # fraction of incoming batch used in EMA
        self.r_max = r_max
        self.d_max = d_max
        self.warmup_steps = warmup_steps

        # Learnable affine: scale (weight) and offset (bias)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Non-trainable running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """ Normalize *x* of shape (B, C) """
        if is_training:
            # batch statistics
            mean = x.mean(dim=0)  # (C,)
            var = x.var(dim=0, unbiased=False)  # (C,)
            std = torch.sqrt(var + self.eps)
            # renorm corrections (stop-gradient)
            ra_std = torch.sqrt(self.running_var + self.eps)
            r = (std / ra_std).detach().clamp(1.0 / self.r_max, self.r_max)
            d = ((mean - self.running_mean) / ra_std).detach().clamp(-self.d_max, self.d_max)
            # corrected statistics
            custom_var = var / (r ** 2)
            custom_mean = mean - d * std / r
            # during warmup, use raw batch stats (warmed_up = counter >= warmup_th)
            warmed_up = float(self.num_batches_tracked.item() >= self.warmup_steps)
            custom_var = warmed_up * custom_var + (1.0 - warmed_up) * var
            custom_mean = warmed_up * custom_mean + (1.0 - warmed_up) * mean
            # update running statistics
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(
                    mean.detach(), alpha=self.momentum
                )
                self.running_var.mul_(1.0 - self.momentum).add_(
                    var.detach(), alpha=self.momentum
                )
                self.num_batches_tracked += 1
        else:
            custom_mean = self.running_mean
            custom_var = self.running_var

        x_hat = (x - custom_mean) / torch.sqrt(custom_var + self.eps)
        return self.weight * x_hat + self.bias
