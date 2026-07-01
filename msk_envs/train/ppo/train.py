import os

import torch
import torch.nn as nn
import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.optimizer import make_optimizer
from msk_envs.train.nets.ppo_networks import PPOActor, PPOCritic, load_policy
from msk_envs.train.ppo.ppo_config import PPOConfig
from msk_envs.train.ppo.ppo_utils import save_params
from msk_envs.train.ppo.rollout_storage import RolloutStorage
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper

torch.set_float32_matmul_precision("high")


def train(
        cfg: PPOConfig,
        envs,
        eval_envs,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
    # ------------------------------------------------------------------ logging
    writer = TensorboardSummaryWriter(log_dir=f"models/{exp_name}", flush_secs=10)
    logging_helper = LoggingHelper(
        writer,
        log_dir=f"models/{exp_name}",
        device=device,
        num_envs=cfg.num_envs,
        # samples collected between two logging calls
        num_steps_per_env=cfg.num_steps_per_env * cfg.logging_interval,
        num_learning_iterations=cfg.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )
    training_metrics = TensorAverageMeterDict()

    # ------------------------------------------------------------------ envs
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    n_act = envs.num_actions()
    action_low, action_high = envs.action_range

    # --------------------------------------------------------------- networks
    actor = PPOActor(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dims=cfg.hidden_dims,
        init_noise_std=cfg.init_noise_std,
        activation=cfg.activation,
        dropout_prob=cfg.dropout_prob,
        use_layer_norm=cfg.use_layer_norm,
        min_noise_std=cfg.min_noise_std,
        min_mean_noise_std=cfg.min_mean_noise_std,
        device=device,
    )
    critic = PPOCritic(
        n_obs=n_obs,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        dropout_prob=cfg.dropout_prob,
        use_layer_norm=cfg.use_layer_norm,
        device=device,
    )

    actor_optimizer = make_optimizer(
        model=actor, lr=cfg.actor_learning_rate, betas=(0.9, 0.95), weight_decay=0.0, use_soap=False,
    )
    critic_optimizer = make_optimizer(
        model=critic, lr=cfg.critic_learning_rate, betas=(0.9, 0.95), weight_decay=0.0, use_soap=False,
    )
    actor_lr = cfg.actor_learning_rate
    critic_lr = cfg.critic_learning_rate

    # Adaptive-LR bounds (None -> holosoma-style defaults)
    min_actor_lr = cfg.min_actor_learning_rate if cfg.min_actor_learning_rate is not None else 1e-5
    max_actor_lr = cfg.max_actor_learning_rate if cfg.max_actor_learning_rate is not None else 1e-2
    min_critic_lr = cfg.min_critic_learning_rate if cfg.min_critic_learning_rate is not None else 1e-5
    max_critic_lr = cfg.max_critic_learning_rate if cfg.max_critic_learning_rate is not None else 1e-2

    obs_normalizer = (
        EmpiricalNormalization(shape=n_obs, device=device)
        if cfg.empirical_normalization
        else nn.Identity()
    )

    storage = RolloutStorage(
        num_envs=cfg.num_envs,
        num_steps=cfg.num_steps_per_env,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
    )

    def normalize(x, update):
        if isinstance(obs_normalizer, nn.Identity):
            return x
        return obs_normalizer(x, update=update)

    # ---------------------------------------------------------- loss computation
    def compute_losses(data):
        obs = data["observations"]
        actions = data["actions"]
        advantages = data["advantages"]
        returns = data["returns"]
        old_values = data["values"]
        old_log_probs = data["old_log_probs"]
        old_mu = data["old_mu"]
        old_sigma = data["old_sigma"]

        log_probs, entropy, mu, sigma = actor.evaluate_actions(obs, actions)
        values = critic(obs)

        # Clipped surrogate objective
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Clipped value loss (holosoma always clips)
        value_clipped = old_values + (values - old_values).clamp(
            -cfg.clip_param, cfg.clip_param
        )
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        entropy_loss = entropy.mean()

        # Approximate KL for logging / adaptive schedule (Gaussian, closed form)
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5,
                dim=-1,
            )
            kl_mean = kl.mean()

        actor_loss = surrogate_loss - cfg.entropy_coef * entropy_loss
        critic_loss = cfg.value_loss_coef * value_loss
        return actor_loss, critic_loss, surrogate_loss, value_loss, entropy_loss, kl_mean

    if cfg.compile:
        compute_losses = torch.compile(
            compute_losses, mode=cfg.compile_mode, backend=cfg.compile_backend
        )

    def adapt_learning_rate(kl_mean, actor_lr, critic_lr):
        """rsl_rl-style KL adaptive learning-rate schedule."""
        if cfg.schedule != "adaptive":
            return actor_lr, critic_lr
        if kl_mean > cfg.desired_kl * 2.0:
            actor_lr = max(min_actor_lr, actor_lr / 1.5)
            critic_lr = max(min_critic_lr, critic_lr / 1.5)
        elif 0.0 < kl_mean < cfg.desired_kl / 2.0:
            actor_lr = min(max_actor_lr, actor_lr * 1.5)
            critic_lr = min(max_critic_lr, critic_lr * 1.5)
        for group in actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for group in critic_optimizer.param_groups:
            group["lr"] = critic_lr
        return actor_lr, critic_lr

    def update(data):
        actor_loss, critic_loss, surrogate_loss, value_loss, entropy_loss, kl_mean = compute_losses(data)

        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        actor_optimizer.step()
        critic_optimizer.step()

        return {
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy_loss.detach(),
            "kl": kl_mean.detach(),
        }

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    @torch.compiler.disable
    def evaluate() -> tuple[float, float]:
        actor.eval()
        sim = LoggedSim(eval_envs, device=device)
        eval_obs = sim.reset()
        for _ in range(sim.max_env_steps):
            norm_eval_obs = normalize(eval_obs, update=False)
            eval_actions = actor.act_inference(norm_eval_obs)
            finished, eval_obs = sim.step(eval_actions)
            if finished:
                break
        rewards_mean = sim.get_rewards_mean()
        episode_length_mean = sim.get_episode_length_mean()

        os.makedirs(traj_out_folder, exist_ok=True)
        os.makedirs(analytics_out_folder, exist_ok=True)
        sim.save_animation(traj_out_folder, str(global_step), use_gzip=True)
        sim.save_frame_data(analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
        sim.save_analytics(analytics_out_folder, f"analytics_{global_step}")
        actor.train()
        return rewards_mean.item(), episode_length_mean.item()

    # ------------------------------------------------------------- checkpoint
    global_step = 0
    if cfg.checkpoint_path:
        ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        if ckpt.get("obs_normalizer_state") is not None and hasattr(obs_normalizer, "load_state_dict"):
            obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        global_step = ckpt.get("global_step", 0)

    # -------------------------------------------------------------- training
    obs = envs.reset()
    raw_rewards_dict = {}
    pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=global_step)

    while global_step < cfg.num_learning_iterations:
        mark_step()

        # ---------------------------------------------------- rollout collection
        storage.clear()
        with logging_helper.record_collection_time():
            actor.eval()
            for _ in range(cfg.num_steps_per_env):
                with torch.no_grad():
                    norm_obs = normalize(obs, update=True)  # updates running stats while training
                    actions, log_probs, mu, sigma = actor.act(norm_obs)
                    values = critic(norm_obs)

                next_obs, rewards, terminated, truncations, info = envs.step(actions)
                dones = (terminated + truncations).bool()

                storage.add(
                    obs=norm_obs, actions=actions, rewards=rewards, dones=dones,
                    values=values, log_probs=log_probs, mus=mu, sigmas=sigma,
                )
                logging_helper.update_episode_stats(rewards, dones)
                obs = next_obs

            # Bootstrap value for the final state, then GAE.
            with torch.no_grad():
                last_norm_obs = normalize(obs, update=False)
                last_values = critic(last_norm_obs)
            storage.compute_returns(last_values, gamma=cfg.gamma, lam=cfg.lam)

            raw_rewards_dict = {
                f"{name}_raw": t.mean() for name, t in info["raw_rewards"].items()
            }

        # ------------------------------------------------------------- learning
        actor.train()
        with logging_helper.record_learn_time():
            generator = storage.mini_batch_generator(
                num_mini_batches=cfg.num_mini_batches,
                num_epochs=cfg.num_learning_epochs,
            )
            for data in generator:
                metrics = update(data)
                actor_lr, critic_lr = adapt_learning_rate(metrics["kl"], actor_lr, critic_lr)
                training_metrics.add(metrics)

        # ------------------------------------------------------------- logging
        if global_step % cfg.logging_interval == 0 and global_step > 0:
            with torch.no_grad():
                loss_metrics = training_metrics.get_metrics_and_clear()
                loss_metrics["env_rewards"] = storage.rewards.mean().item()
                loss_metrics["actor_lr"] = actor_lr
                loss_metrics["critic_lr"] = critic_lr
                extra_log_dicts = {
                    "raw_rewards": raw_rewards_dict,
                    "additional_metrics": envs.additional_metrics(),
                }
                logging_helper.post_epoch_logging(
                    it=global_step, loss_dict=loss_metrics, extra_log_dicts=extra_log_dicts
                )

        if cfg.save_interval > 0 and global_step > 0 and global_step % cfg.save_interval == 0:
            logger.info(f"Saving model at global step {global_step}")
            latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
            save_params(global_step, actor, critic, obs_normalizer, cfg, latest_model_path)

        if global_step % cfg.eval_freq == 0 and global_step > 0:
            logger.info(f"Evaluating at global step {global_step}")
            eval_avg_return, eval_avg_length = evaluate()
            logger.info(f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}")

        global_step += 1
        pbar.update(1)
