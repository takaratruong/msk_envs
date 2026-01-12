import os

# Set Warp cache directory before importing warp to ensure it uses the correct location
if 'WARP_CACHE_DIR' not in os.environ:
    import tempfile

    os.environ['WARP_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'warp_cache_{os.getuid()}')
    os.makedirs(os.environ['WARP_CACHE_DIR'], exist_ok=True)

from msk_envs.utils.logged_sim import LoggedSim

from .buffer import SimpleReplayBuffer
from msk_envs.nets.normalizers import EmpiricalNormalization, RewardNormalizer
from msk_envs.nets.networks import Actor, Critic, load_policy
from msk_envs.nets.simba import SimbaActor, SimbaCritic
from msk_envs.utils.train_utils import mark_step, save_params
from msk_envs.train.fasttd3.td3_config import TD3Config

import math
import time
import torch
import tqdm
import wandb

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import autocast, GradScaler
from tensordict import TensorDict, from_module

torch.set_float32_matmul_precision("high")


def train(
        td3_config: TD3Config,
        envs,
        eval_envs,
        traj_out_folder: str,
        analytics_out_folder: str,
        exp_name: str,
        cuda: bool,
        use_wandb: bool,
):
    amp_enabled = td3_config.amp and cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda" if cuda and torch.cuda.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if td3_config.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    device = torch.device("cuda:0" if cuda else "cpu")

    n_act = envs.num_actions()
    n_obs = envs.num_obs() if type(envs.num_obs()) == int else envs.num_obs()[0]
    n_critic_obs = n_obs
    action_low, action_high = envs.action_range

    if td3_config.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    if td3_config.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=td3_config.gamma,
            device=device,
            g_max=min(abs(td3_config.v_min), abs(td3_config.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": td3_config.num_envs,
        "device": device,
        "init_scale": td3_config.init_scale,
        "hidden_dim": td3_config.actor_hidden_dim,
        "std_min": td3_config.std_min,
        "std_max": td3_config.std_max,
        "use_gsde": td3_config.use_gsde,
        "gsde_steps": td3_config.gsde_steps,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": td3_config.num_atoms,
        "v_min": td3_config.v_min,
        "v_max": td3_config.v_max,
        "hidden_dim": td3_config.critic_hidden_dim,
        "device": device,
    }

    if td3_config.agent == "simbav2":
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / td3_config.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / td3_config.actor_hidden_dim),
                "alpha_init": 1.0 / (td3_config.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(td3_config.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": td3_config.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / td3_config.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / td3_config.critic_hidden_dim),
                "alpha_init": 1.0 / (td3_config.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(td3_config.critic_hidden_dim),
                "num_blocks": td3_config.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
        actor_cls = SimbaActor
        critic_cls = SimbaCritic
    elif td3_config.agent == "fasttd3":
        actor_cls = Actor
        critic_cls = Critic
    else:
        raise ValueError(f"Agent {td3_config.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    actor_detach = actor_cls(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(td3_config.critic_learning_rate, device=device),
        weight_decay=td3_config.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(td3_config.actor_learning_rate, device=device),
        weight_decay=td3_config.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=td3_config.total_timesteps,
        eta_min=td3_config.critic_learning_rate_end,
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=td3_config.total_timesteps,
        eta_min=td3_config.actor_learning_rate_end,
    )

    rb = SimpleReplayBuffer(
        n_env=td3_config.num_envs,
        buffer_size=td3_config.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_steps=td3_config.num_steps,
        gamma=td3_config.gamma,
        device=device,
    )

    policy_noise = td3_config.policy_noise
    noise_clip = td3_config.noise_clip

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(model_path: str):
        policy_eval = load_policy(model_path).to(device=device)

        # Build logged sim wrapper
        sim = LoggedSim(eval_envs, device=device)
        eval_obs = sim.reset()
        for _ in range(sim.max_episode_length):
            with torch.no_grad():
                eval_actions = policy_eval(eval_obs)
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

        # Restore back to training device
        actor.to(device=device)
        obs_normalizer.to(device=device)
        return rewards_mean.item(), episode_length_mean.item()

    def update_main(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if td3_config.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = td3_config.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
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
                if td3_config.use_cdq:
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
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if td3_config.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=td3_config.max_grad_norm if td3_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = data["observations"]
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if td3_config.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if td3_config.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=td3_config.max_grad_norm if td3_config.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if td3_config.compile:
        # Default settings are kept the same, but can now be overridden via train_config.
        compile_mode = td3_config.compile_mode
        compile_backend = td3_config.compile_backend

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

        # Don't compile normalize_obs to avoid Triton compilation issues
        @torch._dynamo.disable
        def normalize_obs(x):
            return obs_normalizer.forward(x)

        if td3_config.reward_normalization:
            update_stats = torch.compile(
                reward_normalizer.update_stats,
                mode=None,
                backend=compile_backend,
            )
        normalize_reward = torch.compile(
            reward_normalizer.forward,
            mode=None,
            backend=compile_backend,
        )
    else:
        normalize_obs = obs_normalizer.forward
        if td3_config.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    obs = envs.reset()
    if td3_config.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{td3_config.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        # global_step = torch_checkpoint["global_step"]
        global_step = 0
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=td3_config.total_timesteps, initial=global_step)
    start_time = None
    latest_model_path = None

    while global_step < td3_config.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if start_time is None and global_step >= td3_config.learning_starts:
            start_time = time.time()

        with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, terminated, truncations, info = envs.step(actions)
        dones = (terminated + truncations).bool()

        if td3_config.reward_normalization:
            update_stats(rewards, dones.float())

        final_obs = info["final_observation"]
        true_next_obs = torch.where(
            dones[:, None] > 0, final_obs, next_obs
        )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_worlds,),
            device=device,
        )
        rb.extend(transition)

        obs = next_obs

        if global_step > td3_config.learning_starts:
            for i in range(td3_config.num_updates):
                data = rb.sample(max(1, td3_config.batch_size // td3_config.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if td3_config.num_updates > 1:
                    if i % td3_config.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % td3_config.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, td3_config.tau)

            if global_step % td3_config.eval_freq == 0 and latest_model_path is not None:
                print(f"Evaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate(latest_model_path)
                logs["eval_avg_return"] = eval_avg_return
                logs["eval_avg_length"] = eval_avg_length

            if global_step % 100 == 0 and start_time is not None:
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "rewards/total": rewards.mean(),
                    }

                    # Log raw reward terms before lambda multiplication
                    for reward_name, reward_tensor in info["raw_rewards"].items():
                        logs[f"rewards/{reward_name}_raw"] = reward_tensor.mean()

                if use_wandb:
                    wandb.log(
                        {
                            "frame": global_step * td3_config.num_envs,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=global_step,
                    )

            if global_step > 0 and global_step % td3_config.save_interval == 0:
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    td3_config,
                    f"models/{exp_name}/{exp_name}_{global_step}.pt",
                )
                latest_model_path = f"models/{exp_name}/{exp_name}_{global_step}.pt"

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)
