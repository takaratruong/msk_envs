import copy
import math
import os

import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.nets.buffer import SimpleReplayBuffer, collect_experience, sample_and_prepare_batches
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.nets.qflex_networks import QFlexActor, ReferencePolicy, VelocityField
from msk_envs.train.nets.distributional_critic import Critic
from msk_envs.train.nets.optimizer import make_optimizer
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
    # ------------------------------------------------------------------ logging
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
    c_logs, r_logs, v_logs = {}, {}, {}

    # ------------------------------------------------------------------ envs
    n_obs, n_act = envs.num_obs(), envs.num_actions()
    action_low, action_high = envs.action_range

    # --------------------------------------------------------------- networks
    critic = Critic(
        n_obs=n_obs,
        n_act=n_act,
        num_atoms=cfg.num_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
        hidden_dim=cfg.critic_hidden_dim,
        use_layer_norm=cfg.use_layer_norm,
        num_q_networks=cfg.num_q_networks,
        device=device,
    )
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    reference = ReferencePolicy(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.actor_hidden_dim,
        layer_norm=cfg.use_layer_norm,
        device=device
    )
    velocity_field = VelocityField(
        n_obs=n_obs,
        n_act=n_act,
        hidden_dim=cfg.velocity_hidden_dim,
        layer_norm=cfg.use_layer_norm,
        device=device
    )
    actor = QFlexActor(
        reference=reference,
        velocity_field=velocity_field,
        num_timesteps=cfg.num_flow_steps,
        device=device,
        n_act=n_act,
        num_envs=cfg.num_envs,
        std_min=0.001,
        std_max=0.4,
        clamp_velocity=cfg.clamp_velocity,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    q_optimizer = make_optimizer(
        model=critic,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
        use_soap=False,
    )
    ref_optimizer = make_optimizer(
        model=reference,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
        use_soap=False,
    )
    vel_optimizer = make_optimizer(
        model=velocity_field,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
        use_soap=False,
    )

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
        observations = data["observations"]
        next_observations = data["next"]["observations"]
        critic_observations = observations
        next_critic_observations = next_observations
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_actions = reference(next_observations)
            next_actions = (next_actions + clipped_noise).clamp(action_low, action_high)

            discount = cfg.gamma ** data["next"]["effective_n_steps"]
            qf1_next_target_projected, qf2_next_target_projected = (
                critic_target.projection(
                    next_critic_observations,
                    next_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
            )
            qf1_next_target_value = critic_target.get_value(qf1_next_target_projected)
            qf2_next_target_value = critic_target.get_value(qf2_next_target_projected)
            if cfg.use_cdq:
                qf_next_target_dist = torch.where(
                    qf1_next_target_value.unsqueeze(1)
                    < qf2_next_target_value.unsqueeze(1),
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )
                qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
            else:
                qf1_next_target_dist, qf2_next_target_dist = (
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )
        qf1, qf2 = critic(critic_observations, actions)
        qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        q_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optimizer.step()

        return {
            "q_loss": q_loss.detach(),
            "q_min": qf1_next_target_value.min().detach(),
            "q_max": qf1_next_target_value.max().detach(),
        }

    def update_reference(data):
        ref_actions = reference(data["observations"])
        critic_observations = data["observations"]
        q_value = critic.q_value(critic_observations, ref_actions, use_cdq=cfg.use_cdq)
        ref_loss = -q_value.mean()

        ref_optimizer.zero_grad(set_to_none=True)
        ref_loss.backward()
        ref_optimizer.step()
        return {
            "reference_loss": ref_loss.detach()
        }

    def update_velocity(data):
        critic_target.eval()

        next_obs = data["next"]["observations"]
        with torch.no_grad():
            action_init = reference(next_obs)
        q_init = critic_target.q_value(next_obs, action_init, cfg.use_cdq).detach()

        # Bounded Q-gradient ascent in action space builds the flow target.
        y = action_init.clone()
        for _ in range(cfg.grad_step_num):
            y = y.detach().requires_grad_(True)
            q = critic_target.q_value(next_obs, y, cfg.use_cdq)
            grad_y = torch.autograd.grad(q.sum(), y)[0]
            grad_norm = grad_y.norm(dim=1, keepdim=True)
            step = torch.minimum(
                torch.full_like(grad_norm, cfg.grad_step_size),
                max_update / (grad_norm + 1e-6),
            )
            y = (y + step * grad_y).detach()
            if cfg.clamp_velocity:
                y = torch.clamp(y, action_low, action_high)

        action_flow_update = y

        # Conditional flow matching toward the straight line init -> updated
        t = torch.rand(next_obs.shape[0], 1, device=next_obs.device)
        temp_action = (1.0 - t) * action_init + t * action_flow_update
        temp_velocity = velocity_field(next_obs, temp_action, t)
        target_velocity = action_flow_update - action_init
        vel_loss = ((temp_velocity - target_velocity) ** 2).mean()

        vel_optimizer.zero_grad(set_to_none=True)
        vel_loss.backward()
        vel_optimizer.step()

        q_flow = critic_target.q_value(next_obs, action_flow_update, cfg.use_cdq).detach()
        return {
            "velocity_loss": vel_loss.detach(),
            "q_init": q_init.mean(),
            "q_flow_update": q_flow.mean(),
            "target_velocity_norm": target_velocity.norm(dim=1).mean().detach(),
        }

    @torch.no_grad()
    def update_target():
        src_ps = [p.data for p in critic.parameters()]
        tgt_ps = [p.data for p in critic_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - cfg.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=cfg.tau)
        return

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
                norm_eval_obs = obs_normalizer(eval_obs, update=False)
                eval_actions = actor.act(norm_eval_obs)
                finished, eval_obs = sim.step(eval_actions)
            if finished:
                break
        rewards_mean = sim.get_rewards_mean()
        episode_length_mean = sim.get_episode_length_mean()
        # Save analytics
        os.makedirs(traj_out_folder, exist_ok=True)
        os.makedirs(analytics_out_folder, exist_ok=True)
        sim.save_animation(traj_out_folder, str(global_step), use_gzip=True)
        sim.save_frame_data(analytics_out_folder, f"frame_data_{global_step}", use_gzip=True)
        sim.save_analytics(analytics_out_folder, f"analytics_{global_step}")
        return rewards_mean.item(), episode_length_mean.item()

    # -------------------------------------------------------------- compilation
    if cfg.compile:
        compile_mode = cfg.compile_mode
        compile_backend = cfg.compile_backend
        update_critic = torch.compile(
            update_critic,
            mode=compile_mode,
            backend=compile_backend,
        )
        update_reference = torch.compile(
            update_reference,
            mode=compile_mode,
            backend=compile_backend,
        )

    # -------------------------------------------------------------- training
    obs = envs.reset()
    dones = None
    global_step = 0
    pbar = tqdm.tqdm(total=cfg.num_learning_iterations, initial=global_step)
    while global_step < cfg.num_learning_iterations:
        mark_step()

        with logging_helper.record_collection_time():
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                if global_step < cfg.learning_starts:
                    actions = torch.rand((cfg.num_envs, n_act), device=device) * 2.0 - 1.0
                else:
                    actor.eval()
                    actions = actor.explore(norm_obs, dones)

            next_obs, rewards, terminated, truncations, info = envs.step(actions)
            collect_experience(
                rb=rb, obs=obs, actions=actions, next_obs=next_obs, rewards=rewards,
                terminated=terminated, truncations=truncations, info=info,
            )

            # Update episode stats using logging helper
            dones = (terminated + truncations).bool()
            logging_helper.update_episode_stats(rewards, dones)

            obs = next_obs

        batch_size = max(cfg.batch_size // cfg.num_envs, 1)
        if rb.ptr >= cfg.learning_starts:
            with logging_helper.record_learn_time():
                prepared_batches = sample_and_prepare_batches(
                    rb=rb, obs_normalizer=obs_normalizer,
                    num_updates=cfg.num_updates, target_batch_size=batch_size
                )
                for i, data in enumerate(prepared_batches):
                    c_logs = update_critic(data)
                    if cfg.num_updates > 1:
                        if i % cfg.policy_frequency == 1:
                            r_logs = update_reference(data)
                            v_logs = update_velocity(data)
                    elif global_step % cfg.policy_frequency == 0:
                        r_logs = update_reference(data)
                        v_logs = update_velocity(data)

                    update_target()

                    # Logging metrics
                    current_metrics = {**c_logs, **r_logs, **v_logs}
                    raw_rewards_dict = {}
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        raw_rewards_dict[f"{reward_name}_raw"] = reward_tensor.mean()
                    training_metrics.add(current_metrics)

            if global_step % cfg.logging_interval == 0:
                with torch.no_grad():
                    loss_metrics = training_metrics.get_metrics_and_clear()
                    loss_metrics["env_rewards"] = rewards.mean().item()
                    extra_log_dicts = {
                        "raw_rewards": raw_rewards_dict,
                        "additional_metrics": envs.additional_metrics(),
                    }
                    logging_helper.post_epoch_logging(it=global_step, loss_dict=loss_metrics,
                                                      extra_log_dicts=extra_log_dicts)

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
