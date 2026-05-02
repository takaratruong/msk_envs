from typing import TypedDict

import torch


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, obs_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """
