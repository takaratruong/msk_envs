import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, dim: int, clamp: float):
        super(RunningMeanStd, self).__init__()
        self.dim = dim
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.ones((), dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x, unnorm=False):
        shape = x.shape
        x = x.view(-1, self.dim)

        mean = self.mean
        var = self.var + self.epsilon
        if unnorm:
            x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return (mean + torch.sqrt(var) * x).view(shape)
        else:
            x = (x - mean) * torch.rsqrt(var)
            return torch.clamp(x, min=-self.clamp, max=self.clamp).view(shape)

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, self.dim)
        var, mean = torch.var_mean(x, dim=0)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta ** 2 * self.count * count / count_
        self.mean.copy_(self.mean + delta * count / count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)

    def reset_counter(self):
        self.count.fill_(1)


class NoNormalize(nn.Module):
    def __init__(self, dim: int):
        super(NoNormalize, self).__init__()
        self.dim = dim

    def forward(self, x, unnorm=False):
        return x

    def update(self, x):
        pass


class RewardsShaper(nn.Module):
    def __init__(self, scale_value: float):
        super().__init__()
        self.scale_value = scale_value

    def forward(self, x):
        return x * self.scale_value
