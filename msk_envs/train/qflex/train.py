import copy
import math
import os

import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.nets.buffer import SimpleReplayBuffer
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.qflex.qflex import Critic, QFlexActor, ReferencePolicy, VelocityField
from msk_envs.train.qflex.qflex_config import QFlexConfig
from msk_envs.train.qflex.qflex_utils import save_params
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper


def train(
        cfg: QFlexConfig,
        envs,
        eval_envs,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        device: torch.device,
):
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
        num_learning_iterations=cfg.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )
    training_metrics = TensorAverageMeterDict()

    # ------------------------------------------------------------------ envs
    n_obs, n_act = envs.num_obs(), envs.num_actions()
    hidden_sizes = [cfg.hidden_dim] * cfg.hidden_num

    # --------------------------------------------------------------- networks
    critic = Critic(n_obs, n_act, hidden_sizes).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    reference = ReferencePolicy(n_obs, n_act, hidden_sizes).to(device)
    velocity_field = VelocityField(n_obs, n_act, hidden_sizes).to(device)
    actor = QFlexActor(reference, velocity_field, cfg.num_flow_steps).to(device)

    betas = (0.5, 0.999)
    q_optim = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate, betas=betas)
    ref_optim = torch.optim.Adam(reference.parameters(), lr=cfg.learning_rate, betas=betas)
    vel_optim = torch.optim.Adam(velocity_field.parameters(), lr=cfg.learning_rate, betas=betas)

    obs_normalizer = (
        EmpiricalNormalization(shape=n_obs, device=device)
        if cfg.obs_normalization
        else torch.nn.Identity()
    )

    rb = SimpleReplayBuffer(
        n_env=cfg.num_envs,
        buffer_size=cfg.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=cfg.num_steps,
        gamma=cfg.gamma,
        device=device,
    )

    max_update = 2.0 * math.sqrt(n_act)  # bound on the Q-gradient-ascent step

    # ---------------------------------------------------------- update steps
    def update_critic(data):
        critic.train()
        observations = data["observations"]
        actions = data["actions"]
        next_observations = data["next"]["observations"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"]

        with torch.no_grad():
            reference.eval()
            next_action, _, _, _ = reference.sample(next_observations)

        b = observations.shape[0]
        cat_obs = torch.cat([observations, next_observations], dim=0)
        cat_act = torch.cat([actions, next_action], dim=0)
        q1_all, q2_all = critic(cat_obs, cat_act)
        cur_q1, next_q1 = q1_all[:b], q1_all[b:]
        cur_q2, next_q2 = q2_all[:b], q2_all[b:]

        q_target = torch.minimum(next_q1, next_q2).detach()
        q_backup = rewards + (1.0 - dones) * cfg.gamma * q_target
        q_loss = F.mse_loss(cur_q1, q_backup) + F.mse_loss(cur_q2, q_backup)

        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optim.step()
        return {"q_loss": q_loss.detach(), "q_mean": cur_q1.mean().detach()}

    def update_reference(data):
        reference.train()
        critic_target.eval()
        observations = data["observations"]
        new_action, new_logp, _, _ = reference.sample(observations)
        q = critic_target.min_q(observations, new_action)
        ref_loss = -q.mean()

        ref_optim.zero_grad(set_to_none=True)
        ref_loss.backward()
        ref_optim.step()
        return {"reference_loss": ref_loss.detach(), "reference_entropy": -new_logp.sum(-1).mean().detach()}

    def update_velocity(data):
        reference.eval()
        velocity_field.train()
        critic_target.eval()
        next_obs = data["next"]["observations"]

        with torch.no_grad():
            action_init, _, _, _ = reference.sample(next_obs)
        q_init = critic_target.min_q(next_obs, action_init).detach()

        # Bounded Q-gradient ascent in action space builds the flow target.
        y = action_init.clone()
        for _ in range(cfg.grad_step_num):
            y = y.detach().requires_grad_(True)
            q = critic_target.min_q(next_obs, y)
            grad_y = torch.autograd.grad(q.sum(), y)[0]
            grad_norm = grad_y.norm(dim=1, keepdim=True)
            step = torch.minimum(
                torch.full_like(grad_norm, cfg.grad_step_size),
                max_update / (grad_norm + 1e-6),
            )
            y = (y + step * grad_y).detach()
        action_flow_update = y

        # Conditional flow matching toward the straight line init -> updated.
        t = torch.rand(next_obs.shape[0], 1, device=next_obs.device)
        temp_action = (1.0 - t) * action_init + t * action_flow_update
        temp_velocity = velocity_field(next_obs, temp_action, t)
        target_velocity = action_flow_update - action_init
        vel_loss = ((temp_velocity - target_velocity) ** 2).mean()

        vel_optim.zero_grad(set_to_none=True)
        vel_loss.backward()
        vel_optim.step()

        q_flow = critic_target.min_q(next_obs, action_flow_update).detach()
        return {
            "velocity_loss": vel_loss.detach(),
            "q_init": q_init.mean(),
            "q_flow_update": q_flow.mean(),
            "target_velocity_norm": target_velocity.norm(dim=1).mean().detach(),
        }

    @torch.no_grad()
    def soft_update(tau: float):
        # Polyak-average params; hard-copy BatchRenorm buffers so the target's
        # normalization tracks the online statistics (matches the JAX version).
        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.mul_(1.0 - tau).add_(tau * p)
        for b, tb in zip(critic.buffers(), critic_target.buffers()):
            tb.copy_(b)

    def sample_and_prepare_batches(batch_size_per_env: int) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size_per_env * cfg.num_updates
        large_data = rb.sample(large_batch_size)
        samples_per_update = batch_size_per_env * envs.num_worlds

        # Normalize all data once
        large_data["observations"] = obs_normalizer(large_data["observations"])
        large_data["next"]["observations"] = obs_normalizer(large_data["next"]["observations"])

        # Split into smaller batches
        prepared_batches = []

        for i in range(cfg.num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][start_idx:end_idx],
                        "observations": large_data["next"]["observations"][start_idx:end_idx],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][start_idx:end_idx],
                    },
                },
                batch_size=samples_per_update,
            )
            prepared_batches.append(batch_data)
        return prepared_batches

    if cfg.compile:
        update_critic = torch.compile(update_critic)
        update_reference = torch.compile(update_reference)
        # update_velocity uses autograd.grad in a python loop; left uncompiled.

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    @torch.compiler.disable
    def evaluate() -> tuple[float, float]:
        actor.eval()

        # Build logged sim wrapper
        sim = LoggedSim(eval_envs, device=device)
        eval_obs = sim.reset()
        for _ in range(sim.max_env_steps):
            with torch.no_grad():
                norm_eval_obs = obs_normalizer(eval_obs)
                eval_actions = actor.act(norm_eval_obs, deterministic=True)
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

    # -------------------------------------------------------------- training
    obs = envs.reset()
    global_step = 0

    pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=global_step)

    while global_step < cfg.num_learning_iterations:
        mark_step()
        with logging_helper.record_collection_time():
            with torch.no_grad():
                norm_obs = obs_normalizer.forward(obs)
                if global_step < cfg.learning_starts:
                    actions = torch.rand((cfg.num_envs, n_act), device=device) * 2.0 - 1.0
                else:
                    actor.eval()
                    actions = actor.act(norm_obs)

            next_obs, rewards, terminated, truncations, info = envs.step(actions)
            dones = (terminated + truncations).bool()
            # Update episode stats using logging helper
            logging_helper.update_episode_stats(rewards, dones)

            # Compute 'true' next_obs for saving
            true_next_obs = torch.where(dones[:, None] > 0, info["final_observation"], next_obs)

            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(envs.num_worlds,),
                device=device,
            )
            rb.extend(transition)
            obs = next_obs

        batch_size = max(cfg.batch_size // cfg.num_envs, 1)
        if rb.ptr >= cfg.learning_starts:
            with logging_helper.record_learn_time():
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = sample_and_prepare_batches(batch_size)
                for i, data in enumerate(prepared_batches):
                    c_logs = update_critic(data)
                    r_logs = update_reference(data)
                    v_logs = update_velocity(data)
                    soft_update(cfg.tau)

                    # Logging metrics
                    current_metrics = {
                        **c_logs, **r_logs, **v_logs
                    }
                    raw_rewards_dict = {}
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        raw_rewards_dict[f"{reward_name}_raw"] = reward_tensor.mean()
                    training_metrics.add(current_metrics)

            if global_step % cfg.logging_interval == 0:
                with torch.no_grad():
                    # Use accumulated training metrics for smoother logging (reduces noise)
                    accumulated_metrics = training_metrics.mean_and_clear()

                    # Convert tensor values to float for logging
                    loss_dict = {}
                    for key, value in accumulated_metrics.items():
                        if isinstance(value, torch.Tensor):
                            loss_dict[key] = value.item()
                        else:
                            loss_dict[key] = float(value)

                    # Add current env rewards (not part of training loop accumulation)
                    loss_dict["env_rewards"] = rewards.mean().item()

                # Use logging helper
                extra_log_dicts = {
                    "raw_rewards": raw_rewards_dict,
                    "additional_metrics": envs.additional_metrics(),
                }
                logging_helper.post_epoch_logging(it=global_step, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

            if cfg.save_interval > 0 and global_step > 0 and global_step % cfg.save_interval == 0:
                logger.info(f"Saving model at global step {global_step}")
                latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
                save_params(
                    global_step,
                    actor,
                    critic,
                    critic_target,
                    obs_normalizer,
                    cfg,
                    latest_model_path,
                )

            if global_step % cfg.eval_freq == 0:
                logger.info(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate()
                logger.info(f"Eval Average Return: {eval_avg_return}, Eval Average Length: {eval_avg_length}")

        global_step += 1
        pbar.update(1)
