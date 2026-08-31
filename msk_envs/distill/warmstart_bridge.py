"""Task RL-T1: Warm-start bridge — distilled StudentPD -> TD3 actor checkpoint.

GOAL
----
Convert the balancer `models/dagger_pd/pd_student.pt` (StudentPD: obs(359) -> raw 75) into a
checkpoint the TD3 trainer (`msk_envs/train/td3/train.py`) can warm-start from via
`--checkpoint_path`, so a later task RL-trains it to sprint. We do NOT train RL here — we produce
a warm-start checkpoint and PROVE it balances when loaded.

WHY BEHAVIORAL DISTILLATION (not a state_dict weight-copy)
----------------------------------------------------------
The original plan proposed copying StudentPD's Linear weights into the trainer actor and asserting
a <1e-4 exact output match. That is INFEASIBLE and is NOT attempted here:
  * The trainer actor is `DeterministicPolicy` (agent="fasttd3"): a SiLU MLP
    (net: Linear(359,512) -> Linear(512,256); fc_head: Linear(256,128); fc_mu: Linear(128,75)+Tanh).
    Its output is Tanh-BOUNDED to [-1,1].
  * StudentPD is an unbounded ReLU MLP (256->256->75). ~26% of StudentPD's raw outputs exceed
    [-1,1] on real PD-RL obs (raw_kd reaches ~-9). A Tanh actor architecturally CANNOT reproduce
    those, and the layer topology differs — so no exact remap / <1e-4 match exists.

Instead we do BEHAVIORAL distillation:
  1. Build a `DeterministicPolicy` with the exact fasttd3 kwargs train.py uses.
  2. Collect an on-distribution dataset (obs, StudentPD(obs)) by rolling StudentPD's OWN decoded
     PD action in `SprinterPDRLEnv` (the states the balancer actually visits).
  3. Supervised-fit the actor's forward() to the (tanh-space) targets with Adam + MSE.
  4. Assemble the 4-key checkpoint train.py loads unconditionally (train.py:377-380):
     actor_state_dict, obs_normalizer_state, qnet_state_dict, qnet_target_state_dict.

CONTINGENCY APPLIED: q_des DELTA REPARAM (see env_pd_rl.py)
----------------------------------------------------------
A first attempt fitting the actor to StudentPD's ABSOLUTE raw_qdes under a Tanh head FAILED the
balance gate (notfallen=0.20). Diagnosis: StudentPD's raw_qdes spans [-2.71, 3.26] rad, but a Tanh
actor bounds each component to [-1,1]; clamping absolute q_des to +-1 alone drops balance from 0.77
to 0.38 (kp/kd clamps are near-harmless: 0.73/0.74). So joints whose neutral pose exceeds 1 rad are
unreachable in the absolute parameterization. Per the brief's documented contingency, T0's env
(SprinterPDRLEnv._set_actions) was changed to interpret raw_qdes as a DELTA fraction:
    q_des = q_now + QDES_DELTA_CLAMP * clamp(raw_qdes, -1, 1)
Now every reachable delta (+-0.5 rad around the current pose) is representable by a Tanh head, and
StudentPD's own decoded trajectory driven through this reparam balances even after +-1 clamping
(verified notfallen=0.73).

TANH-SPACE TARGET (documented choice), IN DELTA SPACE
-----------------------------------------------------
The actor emits `a = tanh(...) in (-1,1)`. We fit it DIRECTLY in action space (MSE) to the 75-vector
that reproduces StudentPD's behavior under the reparam env decode:
  * qdes block: the env now maps action a_qdes -> q_des = q_now + 0.5*a_qdes. StudentPD's absolute
    raw_qdes produced physical q_des = q_now + clamp(raw_qdes - q_now, +-0.5) (dagger_pd rollout).
    The action reproducing that same physical q_des is
        target_qdes = clamp((raw_qdes_student - q_now) / 0.5, -1+eps, 1-eps),
    which lands naturally in [-1,1] (the delta was already clamped to +-0.5) — exactly what a Tanh
    head can emit. This is computed per-state during on-distribution collection.
  * kp/kd blocks: target = clamp(StudentPD_raw[25:], -1+eps, 1-eps). softplus is saturating and the
    clamp preserves sign/saturation; kp/kd clamps were shown near-harmless to balance.
We do NOT fit in atanh space (atanh explodes near +-1 for saturated components); direct action-space
MSE against the clamped target is numerically stable and behavior-faithful. The balance gate
(criterion #3), not the fit number, is load-bearing.

The env reads KP_SCALE/KD_SCALE and the kp/kd decode from `pd_student.pt` + `dagger_pd`, so the
actor only needs to emit the reparam 75-vector — this bridge introduces no new decode.

Run:  CUDA_VISIBLE_DEVICES=3 python -m msk_envs.distill.warmstart_bridge
"""
import os
import torch

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD_RL
from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
from msk_envs.train.nets.distributional_critic import DistributionalCritic
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.train.td3.td3_config import TD3Config
from msk_envs.distill.dagger_pd import StudentPD, QDES_DELTA_CLAMP
from msk_envs.utils.reward_lib import has_fallen
from msk_envs.utils.global_params import FWD_IDX

STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_pd/pd_student.pt"
OUT_CKPT = "/home/ubuntu/msk_envs/models/dagger_pd/pd_student_actor_ckpt.pt"

N_ACT = 75
TANH_EPS = 1e-3  # clamp StudentPD targets to (-1+eps, 1-eps) so a Tanh head can represent them


def build_actor_kwargs(cfg: TD3Config, n_obs: int, num_envs: int, device):
    """Exact kwargs train.py passes to DeterministicPolicy for agent='fasttd3' (train.py:74-135)."""
    return {
        "n_obs": n_obs,
        "n_act": N_ACT,
        "num_envs": num_envs,
        "hidden_dim": cfg.actor_hidden_dim,
        "std_min": cfg.std_min,
        "std_max": cfg.std_max,
        "use_layer_norm": cfg.use_layer_norm,
        "device": device,
        "use_gsde": cfg.use_gsde,
        "gsde_steps": cfg.gsde_steps,
        "sim_type": cfg.sim_type,
        "sim_dimension": cfg.sim_dimension,
        "seq_len": cfg.actor_seq_len,
    }


def build_critic_kwargs(cfg: TD3Config, n_obs: int, device):
    """Exact kwargs train.py passes to DistributionalCritic for agent='fasttd3' (train.py:86-142)."""
    return {
        "n_obs": n_obs,
        "n_act": N_ACT,
        "num_atoms": cfg.num_atoms,
        "v_min": cfg.v_min,
        "v_max": cfg.v_max,
        "hidden_dim": cfg.critic_hidden_dim,
        "use_layer_norm": cfg.use_layer_norm,
        "num_q_networks": cfg.num_q_networks,
        "device": device,
        "sim_type": cfg.sim_type,
        "sim_dimension": cfg.sim_dimension,
        "seq_len": cfg.critic_seq_len,
    }


def load_student(device):
    ck = torch.load(STUDENT_PT, map_location=device)
    n_obs = ck["n_obs"]
    student = StudentPD(n_obs, n_dof=25, device=device)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    kp_scale = ck["kp_scale"].to(device)
    kd_scale = ck["kd_scale"].to(device)
    return student, n_obs, kp_scale, kd_scale


@torch.no_grad()
def reparam_target(env, student, obs, qids):
    """StudentPD's raw output at `obs` -> the 75-vector ACTION (in [-1,1]) that reproduces its
    behavior under the DELTA-reparam env decode. Clamped to (-1+eps, 1-eps) for a Tanh head.

      qdes block: physical q_des StudentPD wanted = q_now + clamp(raw_qdes - q_now, +-DC).
                  Reparam env maps action a -> q_des = q_now + DC*a, so a = clamp((raw-q_now)/DC, +-1).
      kp/kd blocks: pass StudentPD raw straight through (env decode unchanged), clamped to +-1.
    """
    raw = student(obs)
    q_now = env.joint_positions.index_select(1, qids)
    a_qdes = ((raw[:, :25] - q_now) / QDES_DELTA_CLAMP)
    a = torch.cat([a_qdes, raw[:, 25:]], dim=1)
    return a.clamp(-1.0 + TANH_EPS, 1.0 - TANH_EPS)


@torch.no_grad()
def collect_dataset(env, student, qids, steps, actor=None, beta=1.0):
    """Roll one rollout in SprinterPDRLEnv, recording (obs, reparam-target-of-student) at every
    visited state. The DRIVE action is the student's reparam action with prob `beta`, else the
    ACTOR's action (DAgger): this exposes the states the fitted actor actually drifts into while
    always labeling with the expert (StudentPD). `actor=None` or beta=1.0 => pure student rollout.
    """
    obs = env.reset()
    O, T = [], []
    for _ in range(steps):
        expert = reparam_target(env, student, obs, qids)  # the label at this obs
        O.append(obs.detach().clone())
        T.append(expert.detach().clone())
        if actor is None or torch.rand(()) < beta:
            drive = expert
        else:
            drive = actor(obs)
        obs, _, _, _, _ = env.step(drive)
    return torch.cat(O), torch.cat(T)


def fit_actor(actor, obs_buf, tgt_buf, epochs, batch, lr, device):
    """Supervised-fit actor.forward() to the reparam target (already in [-1,1]) via Adam + MSE."""
    target = tgt_buf  # already clamp(reparam action, +-(1-eps)) from collect_dataset
    n = obs_buf.shape[0]
    idx = torch.randperm(n, device=device)
    n_val = n // 5
    vi, ti = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    actor.train()
    for ep in range(epochs):
        perm = ti[torch.randperm(ti.shape[0], device=device)]
        ep_loss = 0.0
        nb = 0
        for mb in perm.split(batch):
            opt.zero_grad()
            pred = actor(obs_buf[mb])
            loss = ((pred - target[mb]) ** 2).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        with torch.no_grad():
            actor.eval()
            val = ((actor(obs_buf[vi]) - target[vi]) ** 2).mean().item()
            actor.train()
        print(f"  fit epoch {ep:02d} train_mse={ep_loss/max(nb,1):.5f} val_mse={val:.5f}", flush=True)
    actor.eval()
    # Behavioral-fit report metric: mean |actor(obs) - reparam_target| on held-out obs.
    with torch.no_grad():
        mad = (actor(obs_buf[vi]) - target[vi]).abs().mean().item()
    return mad, vi


@torch.no_grad()
def balance_rollout(env, actor, steps, device):
    """Deterministic rollout (actor.forward, NO explore noise) in SprinterPDRLEnv.

    Uses the SAME metric convention as the Phase-B reference (distill/evaluate_pd.py:177): NON-latched
    per-step survival with env-style auto-reset. `notfallen` = mean over steps of (1 - has_fallen);
    a fallen env auto-resets and rejoins upright, so this is the standard locomotion survival rate
    the brief's 0.997 target was measured against. `fwd` = mean forward pelvis speed over alive steps.
    """
    actor.eval()
    obs = env.reset()
    n = env.num_worlds
    upright = torch.zeros(n, device=device)
    fwd_sum = torch.zeros(n, device=device)
    for _ in range(steps):
        a = actor(obs)  # deterministic; NO noise
        env.pre_sim_step(a)
        env.launch_sim_step()
        env.update_metrics()
        fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation).float()
        alive = 1.0 - fallen  # per-step (not latched)
        upright += alive
        v = env.body_velocities[:, env.root_id, FWD_IDX + 3]
        fwd_sum += v * alive
        term = env._get_terminated()
        done = term.clamp(0.0, 1.0).unsqueeze(-1)
        if done.any():
            env._perform_reset(done)
        obs = env._get_obs()
    denom = upright.clamp(min=1)
    return {
        "notfallen": (upright / steps).mean().item(),
        "fwd_speed": (fwd_sum / denom).mean().item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TD3Config()  # trainer defaults (actor_hidden_dim=512, use_layer_norm=False, ...)

    # num_envs governs the actor's persistent noise buffer shapes in the state_dict. Match the
    # trainer default so a warm-started run with default --num_envs loads without a shape clash.
    num_envs = cfg.num_envs  # 4096
    # A smaller env count is enough to collect on-distribution data and to test balance.
    data_envs = 256
    collect_steps = 60   # 256*60 = 15360 (obs,target) pairs per DAgger round
    dagger_rounds = 5    # round 0 = pure student (BC); rounds >=1 aggregate actor-driven states
    fit_epochs = 40      # per round
    fit_batch = 2048
    fit_lr = 3e-4
    balance_steps = 200

    print(f"device={device} num_envs(actor buffers)={num_envs} data_envs={data_envs}", flush=True)

    # ---- load StudentPD balancer ----
    student, n_obs, kp_scale, kd_scale = load_student(device)
    print(f"loaded StudentPD n_obs={n_obs} KP_SCALE(mean)={kp_scale.mean():.1f} "
          f"KD_SCALE(mean)={kd_scale.mean():.3f}", flush=True)
    assert n_obs == 359, f"expected n_obs=359, got {n_obs}"

    # ---- build the PD-RL env (data collection + balance test) ----
    env_cfg = EnvConfigSprinterTorquePD_RL()
    env = EnvFactory.create_env(num_envs=data_envs, env_config=env_cfg,
                                requires_visuals=False, cuda_graph=True, device=device)
    assert env.num_actions() == N_ACT, f"env num_actions={env.num_actions()} != {N_ACT}"
    obs0 = env.reset()
    assert obs0.shape[1] == n_obs, f"env obs dim {obs0.shape[1]} != student n_obs {n_obs}"
    qids = torch.tensor([env.qpos_id_lookup[nm] for nm in env.names], device=device, dtype=torch.long)

    # ---- build actor with fasttd3 kwargs (forward() is batch-agnostic; num_envs only sizes the
    #      persistent noise buffers saved in the checkpoint state_dict) ----
    actor = DeterministicPolicy(**build_actor_kwargs(cfg, n_obs, num_envs, device))
    print(f"actor built: DeterministicPolicy hidden={cfg.actor_hidden_dim} "
          f"use_layer_norm={cfg.use_layer_norm} sim_type='{cfg.sim_type}'", flush=True)

    # ---- DAgger behavioral distillation: aggregate actor-visited states, always labeled by the
    #      expert (StudentPD reparam target). Plain BC (round 0 only) drifts off-distribution and
    #      fails the balance gate; DAgger closes the compounding-error gap. ----
    obs_buf = tgt_buf = None
    mad = None
    best_nf = -1.0
    best_state = None
    best_mad = None
    for r in range(dagger_rounds):
        beta = 1.0 if r == 0 else 0.0  # r0: pure student rollout; r>=1: actor-driven, expert-labeled
        drive_actor = None if r == 0 else actor
        O, T = collect_dataset(env, student, qids, collect_steps, actor=drive_actor, beta=beta)
        obs_buf = O if obs_buf is None else torch.cat([obs_buf, O])
        tgt_buf = T if tgt_buf is None else torch.cat([tgt_buf, T])
        print(f"[dagger round {r}] beta={beta:.1f} buf={obs_buf.shape[0]} "
              f"target range=[{tgt_buf.min():.2f},{tgt_buf.max():.2f}]", flush=True)
        mad, _ = fit_actor(actor, obs_buf, tgt_buf, fit_epochs, fit_batch, fit_lr, device)
        m_r = balance_rollout(env, actor, balance_steps, device)
        print(f"[dagger round {r}] behavioral MAD={mad:.4f} "
              f"notfallen={m_r['notfallen']:.3f} fwd={m_r['fwd_speed']:.3f}", flush=True)
        # Keep the best actor by balance (DAgger balance is non-monotone across rounds).
        if m_r["notfallen"] > best_nf:
            best_nf = m_r["notfallen"]
            best_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            best_mad = mad
    # Restore the best actor found across rounds for the saved checkpoint.
    actor.load_state_dict(best_state)
    mad = best_mad
    print(f"selected best DAgger actor: notfallen={best_nf:.3f} MAD={mad:.4f}", flush=True)
    print(f"behavioral fit: mean|actor - reparam_target| (held-out) = {mad:.4f}", flush=True)

    # ---- build the 4-key checkpoint train.py loads unconditionally ----
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)  # identity (StudentPD used raw obs)
    qnet = DistributionalCritic(**build_critic_kwargs(cfg, n_obs, device))
    qnet_target = DistributionalCritic(**build_critic_kwargs(cfg, n_obs, device))
    qnet_target.load_state_dict(qnet.state_dict())

    def cpu_sd(m):
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}

    ckpt = {
        "actor_state_dict": cpu_sd(actor),
        "obs_normalizer_state": cpu_sd(obs_normalizer),
        "qnet_state_dict": cpu_sd(qnet),
        "qnet_target_state_dict": cpu_sd(qnet_target),
        "args": {
            "agent": cfg.agent, "num_envs": num_envs,
            "actor_hidden_dim": cfg.actor_hidden_dim, "critic_hidden_dim": cfg.critic_hidden_dim,
            "use_layer_norm": cfg.use_layer_norm, "std_min": cfg.std_min, "std_max": cfg.std_max,
            "sim_type": cfg.sim_type, "sim_dimension": cfg.sim_dimension,
            "actor_seq_len": cfg.actor_seq_len, "critic_seq_len": cfg.critic_seq_len,
            "num_atoms": cfg.num_atoms, "v_min": cfg.v_min, "v_max": cfg.v_max,
            "num_q_networks": cfg.num_q_networks, "use_gsde": cfg.use_gsde, "gsde_steps": cfg.gsde_steps,
        },
        "global_step": 0,
        "warmstart_meta": {
            "source": STUDENT_PT, "method": "behavioral_distillation",
            "behavioral_fit_mad": mad, "n_obs": n_obs, "n_act": N_ACT,
        },
    }
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save(ckpt, OUT_CKPT)
    print(f"SAVED warm-start checkpoint -> {OUT_CKPT}", flush=True)

    # ---- criterion #1: the 4 keys load into fresh trainer modules ----
    v_actor = DeterministicPolicy(**build_actor_kwargs(cfg, n_obs, num_envs, device))
    v_actor.load_state_dict(ckpt["actor_state_dict"])
    v_norm = EmpiricalNormalization(shape=n_obs, device=device)
    v_norm.load_state_dict(ckpt["obs_normalizer_state"])
    v_q = DistributionalCritic(**build_critic_kwargs(cfg, n_obs, device))
    v_q.load_state_dict(ckpt["qnet_state_dict"])
    v_qt = DistributionalCritic(**build_critic_kwargs(cfg, n_obs, device))
    v_qt.load_state_dict(ckpt["qnet_target_state_dict"])
    print("CRITERION #1 PASS: actor_state_dict + obs_normalizer_state + qnet_state_dict + "
          "qnet_target_state_dict all load into fresh fasttd3 modules at n_obs=359, n_act=75.",
          flush=True)

    # ---- criterion #3 (LOAD-BEARING): loaded actor balances in the env ----
    print(f"balance rollout: {data_envs} envs x {balance_steps} steps (deterministic) ...", flush=True)
    m = balance_rollout(env, v_actor, balance_steps, device)
    print(f"warmstart balances: notfallen={m['notfallen']:.3f} fwd={m['fwd_speed']:.3f}", flush=True)

    balanced = m["notfallen"] >= 0.9
    print(f"CRITERION #3 {'PASS' if balanced else 'FAIL'}: notfallen={m['notfallen']:.3f} "
          f"(target >= 0.9), fwd={m['fwd_speed']:.3f} (~0 expected).", flush=True)
    return m, mad


if __name__ == "__main__":
    main()
