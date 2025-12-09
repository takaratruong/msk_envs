import torch
from collections import deque


class DEP:
    def __init__(self, n_motors, n_envs, buffer_size, bias_rate,
                 kappa, tau, s4avg, regularization, time_dist, with_learning,
                 device):
        self.num_motors = n_motors
        self.n_env = n_envs
        self.buffer_size = buffer_size
        self.bias_rate = bias_rate
        self.tau = tau
        self.kappa = kappa
        self.regularization = regularization
        self.time_dist = time_dist
        self.s4avg = s4avg
        self.with_learning = with_learning
        self.device = device

        # Identity model matrix
        self.M = torch.broadcast_to(
            -torch.eye(n_motors, n_motors),
            (n_envs, n_motors, n_motors),
        ).to(device=device)
        # Unnormalized controller matrix
        self.C = torch.zeros((n_envs, n_motors, n_motors), device=device)
        # Normalized controller matrix
        self.C_norm = torch.zeros((n_envs, n_motors, n_motors), device=device)
        # Controller biases
        self.Cb = torch.zeros((n_envs, n_motors), device=device)
        # Observation and action buffer
        self.buffer = deque(maxlen=buffer_size)
        # smoothed_observation
        self.obs_smoothed = torch.zeros((n_envs, n_motors), device=device)
        # time
        self.t = 0

    def to(self, device):
        self.device = device
        self.M = self.M.to(device)
        self.C = self.C.to(device)
        self.C_norm = self.C_norm.to(device)
        self.Cb = self.Cb.to(device)
        self.obs_smoothed = self.obs_smoothed.to(device)
        return self

    def step(self, obs):
        """
        Takes in an observation consisting of
        muscle_lengths + alpha * muscle_forces.
        """
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed += (obs - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs

        self.buffer.append([self.obs_smoothed.detach().clone(), None])
        # learning step
        if self.with_learning and len(self.buffer) > (2 + self.time_dist):
            self.learn_controller()
        # new action
        y = self.compute_action()
        self.buffer[-1][1] = y.detach().clone()
        self.t += 1
        return y

    def q_norm(self, q):
        """
        Normalization function for intermediate action
        obtained by applying the controller matrix to the
        input q = C @ x.
        """
        reg = 10.0 ** (-self.regularization)
        q_norm = 1.0 / (torch.linalg.norm(q, axis=-1) + reg)
        return q_norm

    def compute_action(self):
        """
        Compute a DEP action from the current C matrix
        """
        q = torch.einsum("ijk,ik -> ij", self.C_norm, self.obs_smoothed)
        q = torch.einsum("ij,i -> ij", q, self.q_norm(q))
        y = torch.clamp(torch.tanh(q * self.kappa + self.Cb), -1.0, 1.0)
        return y

    def learn_controller(self):
        """
        Update DEP by one learning step.
        """
        self.C = self.compute_C()
        # linear response in motor space (action -> action)
        R = torch.einsum("ijk, imk->ijm", self.C, self.M)
        reg = 10.0 ** (-self.regularization)
        # controller normalization c.f. Der et al (2015).
        factor = self.kappa / (torch.linalg.norm(R, axis=-1) + reg)
        self.C_norm = torch.einsum("ijk,ik->ijk", self.C, factor)

        if self.bias_rate >= 0:
            yy = self.buffer[-2][1]
            self.Cb -= (torch.clip(yy * self.bias_rate, -0.05, 0.05)
                        + self.Cb * 0.001)
        else:
            self.Cb *= 0

    def compute_C(self):
        """
        Recompute the controller matrix C from the
        buffer of recent transitions. This is similar
        to the rolling average shown in the publication, but without
        recency weighting.
        """
        C = torch.zeros_like(self.C, device=self.device)
        for s in range(2, min(self.t - self.time_dist, self.tau)):
            x = self.buffer[-s][0][:, : self.num_motors]
            xx = self.buffer[-s - 1][0][:, : self.num_motors]
            xx_t = (
                x
                if self.time_dist == 0
                else self.buffer[-s - self.time_dist][0][:, : self.num_motors]
            )
            xxx_t = self.buffer[-s - 1 - self.time_dist][0][:, : self.num_motors]

            chi = x - xx
            v = xx_t - xxx_t
            mu = torch.einsum("ijk, ik->ij", self.M, chi)

            C += torch.einsum("ij, ik->ijk", mu, v)
        return C