import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import TensorAverageMeterDict, LoggingHelper
from .agent import ContinuousAC
from .buffers import RolloutBuffer
from .normalize import RewardsShaper
from .ppo_config import PPOConfig


@torch.no_grad()
def rollout(agent, envs, buffer, rewards_shaper, logging_helper):
    obs = envs._get_obs()
    dones = torch.zeros_like(buffer.next_dones)
    for step in range(0, buffer.horizon):
        actions, log_probs, values = agent.act(obs)

        obs_, rewards, terminated, truncations, info = envs.step(actions)
        dones_ = (terminated + truncations)
        logging_helper.update_episode_stats(rewards, dones)

        rewards = rewards_shaper(rewards)

        # store
        buffer.obs[step] = obs
        buffer.dones[step] = dones
        buffer.actions[step] = actions
        buffer.values[step] = values
        buffer.log_probs[step] = log_probs
        buffer.rewards[step] = rewards
        buffer.terminated[step] = terminated

        # last step, prepare boostrap
        if step == buffer.horizon - 1:
            buffer.next_value[:] = agent.evaluate(obs_)
            buffer.next_dones[:] = dones_

        obs = obs_
        dones = dones_
    return


@torch.no_grad()
def compute_advantages(buffer, agent, gamma, gae_lambda):
    advantages = buffer.advantages
    rewards = buffer.rewards
    returns = buffer.returns
    values = agent.unnorm_value(buffer.values)
    next_value = agent.unnorm_value(buffer.next_value)

    # Bootstrap
    last_gae_lam = 0.0
    for t in reversed(range(buffer.horizon)):
        if t == buffer.horizon - 1:
            next_non_terminal = 1.0 - buffer.next_dones
            next_values = next_value
        else:
            next_non_terminal = 1.0 - buffer.dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    returns[:] = advantages + values

    # Normalize advantages
    mu, sigma = advantages.mean(), advantages.std()
    advantages[:] = (advantages - mu) / (sigma + 1e-8)

    # Normalize values and returns (updates normalizers)
    agent.obs_norm.update(buffer.obs)

    agent.value_norm.update(values)
    values[:] = agent.value_norm(values)
    agent.value_norm.update(returns)
    returns[:] = agent.value_norm(returns)
    return


def update_policy(buffer, agent, optimizer, cfg, device, training_metrics):
    total_steps = buffer.get_total_steps()
    minibatch_size = total_steps // cfg.num_minibatches

    # Mini epochs
    for epoch in range(cfg.update_epochs):
        b_inds = torch.randperm(total_steps, device=device)
        # Sample minibatches
        for start in range(0, total_steps, minibatch_size):
            mb_inds = b_inds[start:start + minibatch_size]
            o, a, lp, v, adv, ret = buffer.get_minibatch(mb_inds)
            # stats of batch
            pi_, v_ = agent(o)
            mu = pi_.mean()
            lp_ = pi_.log_prob(a)
            e = pi_.entropy()

            # actor loss
            ratio = torch.exp(lp_ - lp)
            surr1 = -adv * ratio
            surr2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = torch.max(surr1, surr2).mean()
            # critic loss
            vf_loss = (v_ - ret) ** 2
            v_clip = v + (v_ - v).clamp(-cfg.clip_coef, cfg.clip_coef)
            vf_loss_clip = (v_clip - ret) ** 2
            c_loss = 0.5 * torch.max(vf_loss, vf_loss_clip).mean()
            # entropy loss
            entropy_loss = -e.mean()
            # bounds loss
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            bounds_loss = (mu_loss_high + mu_loss_low).sum(dim=-1).mean()

            c_loss *= cfg.c_coef
            entropy_loss *= cfg.ent_coef
            bounds_loss *= cfg.bounds_loss_coef
            loss = pg_loss + c_loss + entropy_loss + bounds_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            current_metrics = {
                "actor_loss": pg_loss,
                "c_loss": c_loss,
                "e_loss": entropy_loss,
            }
            training_metrics.add(current_metrics)
    return


@torch.no_grad()
def evaluate(agent, eval_envs, device, traj_out_folder, analytics_out_folder, global_step):
    # Build logged sim wrapper
    sim = LoggedSim(eval_envs, device=device)
    eval_obs = sim.reset()
    for _ in range(sim.max_env_steps):
        with torch.no_grad():
            pi_, v_ = agent(eval_obs)
            eval_actions = pi_.mean()
            finished, eval_obs = sim.step(eval_actions)

        if finished:
            break

    rewards_mean = sim.get_rewards_mean()
    episode_length_mean = sim.get_episode_length_mean()

    # Save analytics
    os.makedirs(traj_out_folder, exist_ok=True)
    sim.save_animation(traj_out_folder, str(global_step), use_gzip=True)

    os.makedirs(analytics_out_folder, exist_ok=True)
    sim.save_frame_data(analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
    sim.save_analytics(analytics_out_folder, f"analytics_{global_step}")
    return rewards_mean.item(), episode_length_mean.item()


def train(
        cfg: PPOConfig,
        envs,
        eval_envs,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
    # --- Logging ---
    writer = TensorboardSummaryWriter(
        log_dir=f"models/{exp_name}",
        flush_secs=10
    )
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=cfg.num_envs,
        num_steps_per_env=cfg.logging_interval,
        num_learning_iterations=cfg.num_iterations,
        is_main_process=True,
        num_gpus=1,
    )
    training_metrics = TensorAverageMeterDict()

    # --- Agent ---
    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    agent = ContinuousAC(
        cfg=cfg.agent_config,
        input_dim=n_obs,
        output_dim=n_act,
    )
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-8)
    rewards_shaper = RewardsShaper(scale_value=cfg.rewards_scale)

    # Allocate storage
    buffer = RolloutBuffer(
        n_steps=cfg.num_rollout_steps,
        n_envs=cfg.num_envs,
        obs_dim=n_obs,
        act_dim=n_act,
        device=device
    )

    # --- Training ---
    for iteration in range(1, cfg.num_iterations + 1):
        # Annealing the rate
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lr_now = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # Collect rollouts
        with logging_helper.record_collection_time():
            agent.eval()
            rollout(agent, envs, buffer, rewards_shaper, logging_helper)

        # Policy update
        with logging_helper.record_learn_time():
            compute_advantages(buffer, agent, cfg.gamma, cfg.gae_lambda)

            # Optimization steps
            agent.train()
            update_policy(buffer, agent, optimizer, cfg, device, training_metrics)

        # Logging
        if iteration % cfg.logging_interval == 0:
            with torch.no_grad():
                accumulated_metrics = training_metrics.mean_and_clear()
                # Convert tensor values to float for logging
                loss_dict = {}
                for key, value in accumulated_metrics.items():
                    if isinstance(value, torch.Tensor):
                        loss_dict[key] = value.item()
                    else:
                        loss_dict[key] = float(value)
                extra_log_dicts = {
                    "additional_metrics": envs.additional_metrics(),
                }
                logging_helper.post_epoch_logging(it=iteration, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

        # Evaluation
        if iteration % cfg.eval_freq == 0:
            evaluate(agent, eval_envs, device, traj_out_folder, analytics_out_folder, iteration)

    return
