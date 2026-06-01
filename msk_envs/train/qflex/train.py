import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from loguru import logger
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from msk_envs.train.nets.buffer import SimpleReplayBuffer
from msk_envs.train.nets.qflex_networks import create_flow_net
from msk_envs.train.qflex.qflex_config import QFlexConfig
from msk_envs.utils.train_utils import mark_step, TensorAverageMeterDict, LoggingHelper, save_params_td3

torch.set_float32_matmul_precision("high")


def train(
        qflex_config: QFlexConfig,
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
        num_envs=qflex_config.num_envs,
        num_steps_per_env=qflex_config.logging_interval,
        num_learning_iterations=qflex_config.num_learning_iterations,
        is_main_process=True,
        num_gpus=1,
    )

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]

    agent = create_flow_net(
        obs_dim=n_obs,
        act_dim=n_act,
        hidden_sizes=[qflex_config.hidden_dim] * qflex_config.hidden_num,
        num_timesteps=qflex_config.diffusion_steps,
        learn_reference_gn=qflex_config.learn_reference_gn,
    )
    agent = agent.to(device)

    q1_optimizer = optim.AdamW(
        list(agent.q1.parameters()),
        lr=qflex_config.learning_rate,
        betas=(0.5, 0.999),
    )
    q2_optimizer = optim.AdamW(
        list(agent.q2.parameters()),
        lr=qflex_config.learning_rate,
        betas=(0.5, 0.999),
    )
    velocity_field_optimizer = optim.AdamW(
        list(agent.flow.velocity_field.parameters()),
        lr=qflex_config.learning_rate,
        betas=(0.5, 0.999),
    )
    reference_gn_optimizer = optim.AdamW(
        list(agent.reference_gn.parameters()),
        lr=qflex_config.learning_rate,
        betas=(0.5, 0.999),
    ) if qflex_config.learn_reference_gn else None
    log_alpha_optimizer = optim.AdamW([agent.log_alpha], lr=qflex_config.alpha_learning_rate)

    rb = SimpleReplayBuffer(
        n_env=qflex_config.num_envs,
        buffer_size=qflex_config.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=qflex_config.num_steps,
        gamma=qflex_config.gamma,
        device=device,
    )

    def update_main(data):
        observations, next_observations = data["observations"], data["next"]["observations"]
        critic_observations, next_critic_observations = observations, next_observations
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            next_state_actions = actor(next_observations)
            discount = qflex_config.gamma ** data["next"]["effective_n_steps"]

            qf1_next_target_projected, qf2_next_target_projected = (
                qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
            )
            qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
            qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
            if qflex_config.use_cdq:
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

        qf1, qf2 = qnet(critic_observations, actions)
        qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
        qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if qflex_config.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=qflex_config.max_grad_norm if qflex_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.mean(),
            qf_loss.mean(),
            qf1_next_target_value.mean(),
            qf1_next_target_value.min(),
        )

    def update_pol(data):
        critic_observations = data["observations"]
        qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
        qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
        qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
        if qflex_config.use_cdq:
            qf_value = torch.minimum(qf1_value, qf2_value)
        else:
            qf_value = (qf1_value + qf2_value) / 2.0
        actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if qflex_config.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=qflex_config.max_grad_norm if qflex_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach().item(),
            actor_loss.detach().item(),
        )

    def _sample_and_prepare_batches() -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * qflex_config.num_updates
        large_data = rb.sample(large_batch_size)
        samples_per_update = batch_size * envs.num_worlds

        # Split into smaller batches
        prepared_batches = []

        for i in range(qflex_config.num_updates):
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

    if qflex_config.compile:
        # Default settings are kept the same, but can now be overridden via train_config.
        compile_mode = qflex_config.compile_mode
        compile_backend = qflex_config.compile_backend

        update_main = torch.compile(
            update_main,
            mode=compile_mode,
            backend=compile_backend,
        )
        update_pol = torch.compile(
            update_pol,
            mode=compile_mode,
            backend=compile_backend,
        )
        policy = torch.compile(
            policy,
            mode=None,
            backend=compile_backend,
        )

    global_step = 0
    obs = envs.reset()
    dones = None
    training_metrics = TensorAverageMeterDict()
    latest_model_path = None

    actor_loss = torch.tensor(0.0, device=device)
    actor_grad_norm = torch.tensor(0.0, device=device)
    pbar = tqdm.tqdm(total=qflex_config.num_learning_iterations, initial=global_step)

    while global_step < qflex_config.num_learning_iterations:
        mark_step()
        with logging_helper.record_collection_time():
            with torch.no_grad():
                actions = policy(obs=obs, dones=dones)

            next_obs, rewards, terminated, truncations, info = envs.step(actions)
            dones = (terminated + truncations).bool()

            # Update episode stats using logging helper
            logging_helper.update_episode_stats(rewards, dones)

            # Compute 'true' next_obs for saving
            true_next_obs = torch.where(dones[:, None] > 0, info["final_observation"], next_obs)
            # true_next_obs = torch.where(truncations[:, None] > 0, info["final_observation"], next_obs)

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

        batch_size = max(qflex_config.batch_size // qflex_config.num_envs, 1)
        # Wait until the replay buffer has collected enough transitions before learning.
        if rb.ptr >= qflex_config.learning_starts:
            with logging_helper.record_learn_time():
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = _sample_and_prepare_batches()
                for i, data in enumerate(prepared_batches):
                    buffer_rewards, critic_grad_norm, qf_loss, qf_max, qf_min = update_main(data)
                    if qflex_config.num_updates > 1:
                        if i % qflex_config.policy_frequency == 1:
                            actor_grad_norm, actor_loss = update_pol(data)
                    elif global_step % qflex_config.policy_frequency == 0:
                        actor_grad_norm, actor_loss = update_pol(data)

                    # Accumulate training metrics for smoother logging
                    current_metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "buffer_rewards": buffer_rewards,
                    }

                    # Log raw reward terms before lambda multiplication
                    raw_rewards_dict = {}
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        raw_rewards_dict[f"{reward_name}_raw"] = reward_tensor.mean()

                    training_metrics.add(current_metrics)

                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - qflex_config.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=qflex_config.tau)

            if global_step % qflex_config.logging_interval == 0:
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

            if qflex_config.save_interval > 0 and global_step > 0 and global_step % qflex_config.save_interval == 0:
                logger.info(f"Saving model at global step {global_step}")
                latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"
                save_params_td3(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    qflex_config,
                    latest_model_path,
                    actor_optimizer=actor_optimizer,
                    q_optimizer=q_optimizer,
                    actor_scheduler=actor_scheduler,
                    q_scheduler=q_scheduler,
                    scaler=scaler,
                    include_optim_state=qflex_config.save_optimizer_state,
                )

            if global_step % qflex_config.eval_freq == 0 and latest_model_path is not None:
                print(f"Evaluating at global step {global_step}")

        if global_step >= qflex_config.num_learning_iterations:
            break
        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)
