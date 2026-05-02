import math
import os
import signal
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from tensordict import TensorDict
from torch.amp import autocast, GradScaler
from loguru import logger
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.train.ppo.ppo_config import PPOConfig
from msk_envs.train.nets.buffer import SimpleReplayBuffer
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.ppo.ppo_networks import PPOActor, PPOCritic
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper, save_params_sac

torch.set_float32_matmul_precision("high")

save_requested = False


def on_sigusr1(signum, frame):
    global save_requested
    save_requested = True


def train(
        ppo_config: PPOConfig,
        envs,
        eval_envs,
        dep_explorer,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
    global save_requested
    writer = TensorboardSummaryWriter(
        log_dir=f"models/{exp_name}",
        flush_secs=10
    )
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=ppo_config.num_envs,
        num_steps_per_env=ppo_config.logging_interval,
        num_learning_iterations=ppo_config.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    if ppo_config.obs_normalization:
        actor_obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    actor = PPOActor(
        n_obs=n_obs,
        n_act=n_act,
        module_config_dict=ppo_config.module_dict.actor,
        init_noise_std=ppo_config.init_noise_std,
    )
    critic = PPOCritic(
        n_obs=n_obs,
        module_config_dict=ppo_config.module_dict.critic,
    )

    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=ppo_config.actor_learning_rate,
        weight_decay=0.001,
    )
    critic_optimizer = optim.AdamW(
        list(critic.parameters()),
        lr=ppo_config.critic_learning_rate,
        weight_decay=0.001,
    )

    def _normalize_actor_obs(self, actor_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return actor_obs_normalizer(actor_obs, update=update)
        return actor_obs

    def _normalize_critic_obs(self, critic_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return critic_obs_normalizer(critic_obs, update=update)
        return critic_obs

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(key, shape=shape, dtype=dtype)
