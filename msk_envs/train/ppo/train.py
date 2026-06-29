import torch
import torch.nn as nn
import torch.optim as optim
import os

from .ppo_config import PPOConfig
from .agent import ContinuousAC
from .buffers import RolloutBuffer
from .normalize import RewardsShaper
from .stats import PPOStatsTracker, PerfTimer

from msk_envs.utils.logged_sim import LoggedSim


@torch.no_grad()
def rollout(agent, envs, buffer, rewards_shaper, stats, timer):
    obs = envs._get_obs()
    dones = torch.zeros_like(buffer.next_dones)
    for step in range(0, buffer.horizon):
        timer.start_inference()
        actions, log_probs, values = agent.act(obs)
        timer.end_inference()

        timer.start_sim()
        obs_, rews, terminated, truncations, info = envs.step(actions)
        dones_ = (terminated + truncations)
        stats.update(rews, dones_)

        rews = rewards_shaper(rews)
        timer.end_sim()

        # store
        buffer.obs[step] = obs
        buffer.dones[step] = dones
        buffer.actions[step] = actions
        buffer.values[step] = values
        buffer.log_probs[step] = log_probs
        buffer.rewards[step] = rews
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


def update_policy(buffer, agent, optimizer, cfg, stats, device):
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

            stats.set_losses(pg_loss, c_loss, entropy_loss, bounds_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()
    return


def ppo(env, agent, device, cfg: PPOConfig, callback=None):
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-8)

    rewards_shaper = RewardsShaper(scale_value=cfg.rewards_scale)

    # Allocate storage
    buffer = RolloutBuffer(
        n_steps=cfg.num_rollout_steps,
        n_envs=env.num_worlds,
        obs_dim=env.num_obs(),
        act_dim=env.num_actions(),
        device=device
    )

    stats = PPOStatsTracker(n_envs=env.num_worlds, device=device)
    timer = PerfTimer()
    for iteration in range(1, cfg.num_iterations + 1):
        timer.start_iter()
        timer.add_steps(env.num_worlds * cfg.num_rollout_steps)

        # Annealing the rate
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lr_now = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # Collect rollouts
        agent.eval()
        timer.start_rollout()
        rollout(agent, env, buffer, rewards_shaper, stats, timer)
        timer.end_rollout()

        # Advantages
        compute_advantages(buffer, agent, cfg.gamma, cfg.gae_lambda)

        # Optimization steps
        agent.train()
        timer.start_update()
        update_policy(buffer, agent, optimizer, cfg, stats, device)
        timer.end_update()

        timer.end_iter()
        if callback is not None:
            callback(iteration, stats, timer)


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
    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    agent = ContinuousAC(
        cfg=cfg.agent_config,
        input_dim=n_obs,
        output_dim=n_act,
    )

    def train_callback(iteration: int, ppo_stats: PPOStatsTracker,
                       ppo_timer: PerfTimer):
        if iteration % 10 == 0:
            print(f"\nUpdate: {iteration}", end=' ')
            ppo_timer.print()
            ppo_stats.print()

        # if iteration % 100 == 0:
        #     agent.save(os.path.join(checkpoint_dir, f"{iteration}.pt"))
        if iteration % cfg.eval_freq == 0:
            evaluate(agent, eval_envs, device, traj_out_folder, analytics_out_folder, iteration)

        ppo_timer.reset()
        ppo_stats.reset()

    ppo(envs, agent, device, cfg, train_callback)
    return
