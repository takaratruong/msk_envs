import torch
from torch.distributions import Normal


class ActionDistribution:
    def sample(self):
        raise NotImplementedError

    def log_prob(self, actions):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def update_mean_std(self, mu, log_sigma):
        raise NotImplementedError

    def reset_noise(self):
        """ For distributions which require resampling noise every n steps """
        pass


class NormalDistribution(ActionDistribution):
    """ Gaussian distribution with mean `mu` and std `exp(log_sigma)` a_t = mu + e_t where e_t ~ N(0, sigma^2) """

    def __init__(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        self.dist = Normal(mu, sigma)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, actions):
        return self.dist.log_prob(actions).sum(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1)

    def mean(self):
        return self.dist.mean

    def update_mean_std(self, mu, log_sigma):
        # probably easier to just create a new distribution
        sigma = torch.exp(log_sigma)
        self.dist = Normal(mu, sigma)

    def reset_noise(self):
        pass
