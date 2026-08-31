"""Task RL-T4 (PRIMARY USER GATE) — RL PD-impedance actor evaluation: sprint + compliance.

The decisive experiment for the RL phase: does the TD3-trained PD-gain policy (warm-started from
the balancer, regularized toward the teacher's impedance) (a) SPRINT (forward speed approaching the
muscle teacher) and (b) stay MUSCLE-COMPLIANT (per-joint kp/kd near the teacher's extracted
impedance oracle, in log space)?

This file REUSES the Task-4 (Phase-B) harness in `evaluate_pd.py` verbatim:
    - sprint_rollout        : manual-step rollout with a per-step has_fallen survival metric and
                              matched-seed starts (identical for teacher and student).
    - _reset_matched, read_optimal_force, load_student, constants (CKPT, N_ENVS, SEED, MAX_STEPS,
      BLANK_REWARD_LAMBDAS).
    - the TEACHER-SANITY GATE (Phase A/B P0 lesson): the teacher MUST sprint in THIS harness
      (fwd>1.0 & not_fallen>0.7) before we report any student number — otherwise it is a harness
      bug and we abort instead of scoring the student against a broken teacher.

WHAT THIS FILE CHANGES vs evaluate_pd
-------------------------------------
1. STUDENT = the RL actor (`DeterministicPolicy`, fasttd3) loaded from the RL checkpoint's
   `actor_state_dict`, NOT the supervised StudentPD net.
     * The checkpoint's per-env noise buffers are sized (num_envs=4096); we build the actor at
       num_envs=256 and load with the (4096-sized) `noise`/`noise_scales` buffers FILTERED OUT
       (a plain strict=False would raise a size-mismatch on those present-but-wrong-shape buffers).
       We then ASSERT every net.*/fc_head.*/fc_mu.* weight loaded and ONLY noise/noise_scales were
       skipped.
     * OBS NORMALIZATION: the trainer runs `actor(normalize_obs(obs))` (obs_normalization=True), so
       we load the checkpoint's `EmpiricalNormalization` state and apply it (eval mode, no update)
       before the actor forward. Skipping this would feed the actor raw obs — a different controller.
     * DETERMINISTIC forward (`actor.forward`), NO exploration noise.

2. DECODE mirrors `env_pd_rl.SprinterPDRLEnv._set_actions` EXACTLY (cross-checked line by line):
       raw            = actor(normalize(obs))                 # (n,75), Tanh-bounded [-1,1]
       _, kp, kd      = decode(raw, kp_scale, kd_scale)       # softplus * per-joint scale
       raw_qdes       = clamp(raw[:, :25], -1, 1)
       q_now          = joint_positions[:, qids]
       q_des          = q_now + QDES_DELTA_CLAMP * raw_qdes   # DELTA reparam (NOT absolute q_des)
       pd_q_des <- q_des[:, act_perm]; pd_kp <- clamp(kp,0)[:, act_perm]; pd_kd <- clamp(kd,0)[...]
       muscles OFF (action muscle slice = -1  ==  excitation 0, matching env's muscle_excitations.zero_())
   kp_scale/kd_scale come from pd_student.pt — the SAME file the RL env loads at __init__, so the
   decode is bit-identical to training.

   ENV: we use `EnvConfigSprinterTorquePD` (the muscle+PD twin), NOT `..._RL`. The RL env's
   `_set_actions` would re-decode the harness's blank action and clobber the PD buffers we write by
   hand (and zero the teacher's muscle drive). The physics is IDENTICAL between the two configs (the
   RL env subclass overrides only `_set_actions`/reward; both build the same sprinter_torque.osim
   with use_pd_actuators=True). This is the same env `evaluate_pd` uses, where teacher-sanity passes.

3. COMPLIANCE (NEW — the muscle-transfer measure): over the RL actor's rollout, at each alive
   (upright) state we record the policy's emitted (kp,kd) [names order] and the impedance oracle's
   (log kp, log kd) at that raw-359 obs, then report the log-space distance
       per-element RMS  = sqrt( mean_{alive, 50 dims} (log g_pol - log g_tea)^2 )   [primary scalar]
       per-env L2       = mean_{alive} || (log kp_pol,log kd_pol) - (log kp_tea,log kd_tea) ||_2
       + per-joint kp / kd RMS-log-distance.
   Lower = more muscle-like. Uses the SAME eps (ORACLE_EPS) as the RL reward's log convention, so the
   two log-spaces align. Same-state comparison (obs_t for both actor and oracle) — the cleanest "is
   the policy compliant in this state" measure; the training reward uses a one-step-lagged obs but
   the same log-distance form.

VERDICT / BUCKETS (plan's three, thresholds documented + untouched from the sprint side):
    teacher-sanity PASS first (else abort + write JSON).
    sprint     = (rl fwd_speed >= 0.5 * teacher fwd_speed) AND (rl not_fallen >= 0.7).
    compliant  = per-element RMS log-distance <= COMPLIANT_RMS_LOG (provisional; documented below).
    buckets    : (a) sprint+compliant, (b) sprint-only, (c) no-sprint.
"""
import os
import glob
import json
import time
import argparse
import re

import torch

import bolt

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.dagger_pd import decode, QDES_DELTA_CLAMP
from msk_envs.distill.dof_utils import (
    joint_dof_indices, build_actuator_perm, build_teacher_obs_cols,
)
from msk_envs.distill.impedance_cache import load_oracle, ORACLE_PATH, ORACLE_EPS
from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
from msk_envs.train.nets.normalizers import EmpiricalNormalization
from msk_envs.utils.reward_lib import has_fallen

# Reuse the Phase-B harness verbatim (sprint_rollout + constants + loaders).
from msk_envs.distill.evaluate_pd import (
    sprint_rollout, load_student, CKPT, STUDENT_PT, N_ENVS, SEED, MAX_STEPS,
    BLANK_REWARD_LAMBDAS,
)

RUN_DIR = "/home/ubuntu/msk_envs/models/rl_pd_2026-08-31_21-54"
EVAL_JSON = os.path.join(RUN_DIR, "eval_rl.json")

# Provisional compliance threshold (per-ELEMENT RMS distance in log space). Documented, not tuned:
# the impedance oracle's OWN held-out fit error is val_mse_log~0.19 => per-element RMS ~0.43, i.e.
# the oracle itself cannot resolve the teacher impedance below ~0.43 in log space. A policy within
# ~e^1 (=2.7x) per-gain of the oracle target on average => per-element RMS ~1.0. We call the policy
# "compliant" at <=1.0 (roughly a 2.7x average gain factor). This threshold gates only the label;
# the raw distance is always reported so the number can be re-judged later / against a no-reg run.
COMPLIANT_RMS_LOG = 1.0


def _default_ckpt():
    """Highest-iter *.pt in RUN_DIR (by the trailing _<iter> in the filename)."""
    cands = glob.glob(os.path.join(RUN_DIR, "*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt checkpoints in {RUN_DIR}")

    def _iter(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(cands, key=_iter)


def _ckpt_iter(path):
    m = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_rl_actor(ckpt_path, n_obs, n_act, num_envs, dev):
    """Load the RL DeterministicPolicy actor + obs normalizer from an RL checkpoint.

    Builds the actor at `num_envs` (256) and loads `actor_state_dict` with the per-env noise
    buffers (noise, noise_scales — sized to the trainer's 4096 envs) FILTERED OUT, then asserts the
    real weights loaded and ONLY those two buffers were skipped. Returns (actor, obs_normalizer|None).
    """
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    args = ck["args"]
    assert args.get("agent", "fasttd3") == "fasttd3", f"expected fasttd3 actor, got {args.get('agent')}"

    actor = DeterministicPolicy(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        hidden_dim=args["actor_hidden_dim"],
        std_min=args["std_min"],
        std_max=args["std_max"],
        sim_type=args.get("sim_type", ""),
        sim_dimension=args.get("sim_dimension", 64),
        seq_len=args.get("actor_seq_len", 8),
        use_layer_norm=args["use_layer_norm"],
        device=dev,
    ).to(dev)

    asd = ck["actor_state_dict"]
    # Per-env exploration buffers are sized to the TRAINER's num_envs (4096); a plain strict=False
    # load would raise a size-mismatch (the keys are PRESENT with the wrong shape). Filter them out.
    skip = {"noise", "noise_scales"}
    filtered = {k: v for k, v in asd.items() if k not in skip}
    missing, unexpected = actor.load_state_dict(filtered, strict=False)

    # VERIFY: every real weight loaded, and ONLY the noise buffers were skipped.
    weight_keys = ["net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias",
                   "fc_head.0.weight", "fc_head.0.bias", "fc_mu.0.weight", "fc_mu.0.bias"]
    sd = actor.state_dict()
    for k in weight_keys:
        assert k in asd, f"checkpoint missing expected actor weight {k}"
        assert torch.equal(sd[k], asd[k].to(dev)), f"actor weight {k} did NOT load (value mismatch)"
    assert set(missing) <= skip, f"unexpected MISSING keys after load (weights failed to load!): {missing}"
    assert len(unexpected) == 0, f"unexpected EXTRA keys in checkpoint: {unexpected}"
    print(f"[actor] loaded {len(filtered)} tensors; skipped per-env buffers {sorted(skip)}; "
          f"verified {len(weight_keys)} weight tensors match", flush=True)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    # Obs normalizer: the trainer runs actor(normalize_obs(obs)). Reproduce it (eval => no update).
    obs_norm = None
    ons = ck.get("obs_normalizer_state", {})
    if args.get("obs_normalization", True) and ons is not None and len(ons) > 0:
        obs_norm = EmpiricalNormalization(shape=n_obs, device=dev)
        obs_norm.load_state_dict(ons)
        obs_norm.eval()
        print(f"[actor] obs_normalizer loaded (count={int(obs_norm.count.item())}); "
              f"applied before actor forward", flush=True)
    else:
        print("[actor] no obs normalization (Identity)", flush=True)
    return actor, obs_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None,
                    help="RL checkpoint .pt (default: highest-iter in the run dir)")
    args = ap.parse_args()
    ckpt_path = args.ckpt or _default_ckpt()
    it = _ckpt_iter(ckpt_path)
    print(f"INTERIM EVAL @ iter={it}  ckpt={ckpt_path}  (training may still be ongoing)", flush=True)

    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=N_ENVS, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)
    assert getattr(cfg, "use_pd_actuators", False), "PD env must have use_pd_actuators=True"

    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "label order (joint_dof_indices) != teacher.names"
    teacher_cols = build_teacher_obs_cols(env, dev)          # 359-obs -> 336-obs teacher adapter
    act_perm = build_actuator_perm(env, names, dev)          # names order -> actuator-slice order
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    dids = torch.tensor([env.dof_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    n_musc = env.num_muscles

    # kp_scale/kd_scale from pd_student.pt (SAME file the RL env loads) => decode bit-identical.
    _student, kp_scale, kd_scale, s_names = load_student(STUDENT_PT, dev)
    assert s_names == names, "pd_student names != joint_dof_indices order"

    obs_dim = env.reset().shape[1]
    actor, obs_norm = load_rl_actor(ckpt_path, n_obs=obs_dim, n_act=75, num_envs=N_ENVS, dev=dev)
    oracle, oracle_meta = load_oracle(ORACLE_PATH, dev)      # obs359 -> (log kp[25], log kd[25])

    full_steps = int(round(env.max_episode_duration / env.delta_t))
    steps = min(full_steps, MAX_STEPS)
    print(f"n_envs={env.num_worlds} full_steps={full_steps} steps={steps} "
          f"n_musc={n_musc} obs={obs_dim}", flush=True)

    # ---- action fns -------------------------------------------------------------------------
    def teacher_fn(obs):
        # Zero PD buffers so persistent gains never leak torque into the teacher-driven advance.
        bolt.pd_kp(env.d).zero_()
        bolt.pd_kd(env.d).zero_()
        bolt.pd_q_des(env.d).zero_()
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        return a

    # Compliance capture buffers (filled by rl_actor_fn on alive/upright emit states).
    comp = {"kp_pol": [], "kd_pol": [], "log_kp_tea": [], "log_kd_tea": [], "alive": []}

    def rl_actor_fn(obs):
        # DETERMINISTIC forward on NORMALIZED obs (matches trainer: actor(normalize_obs(obs))).
        with torch.no_grad():
            norm = obs_norm(obs) if obs_norm is not None else obs
            raw = actor(norm)                                # (n,75), Tanh [-1,1]
        # Decode EXACTLY as env_pd_rl._set_actions (delta reparam q_des; softplus*scale gains).
        _, kp, kd = decode(raw, kp_scale, kd_scale)
        raw_qdes = torch.clamp(raw[:, :25], -1.0, 1.0)
        q_now = env.joint_positions.index_select(1, qids)
        q_des = q_now + QDES_DELTA_CLAMP * raw_qdes
        kp = torch.clamp(kp, min=0.0)
        kd = torch.clamp(kd, min=0.0)
        bolt.pd_q_des(env.d).copy_(q_des.index_select(1, act_perm))
        bolt.pd_kp(env.d).copy_(kp.index_select(1, act_perm))
        bolt.pd_kd(env.d).copy_(kd.index_select(1, act_perm))
        # Compliance: record emitted gains + oracle target at THIS (raw-359) state; mask by upright.
        with torch.no_grad():
            log_kp_tea, log_kd_tea = oracle(obs)             # oracle keyed on RAW 359 obs
        alive = 1.0 - has_fallen(root_pos=env.root_pos,
                                 ground_rotation=env.ground_rotation).float()
        comp["kp_pol"].append(kp.detach().clone())
        comp["kd_pol"].append(kd.detach().clone())
        comp["log_kp_tea"].append(log_kp_tea.detach().clone())
        comp["log_kd_tea"].append(log_kd_tea.detach().clone())
        comp["alive"].append(alive.detach().clone())
        a = env.get_blank_actions()
        a[:, :n_musc] = -1.0                                 # muscles OFF (== excitation 0)
        return a

    # ---- 1. TEACHER SANITY GATE -------------------------------------------------------------
    t0 = time.time()
    m_teacher = sprint_rollout(env, teacher_fn, steps, qids, dids, teacher_cols=teacher_cols)
    print(f"[teacher] not_fallen={m_teacher['not_fallen']:.3f} "
          f"fwd_speed={m_teacher['fwd_speed']:.3f} drift={m_teacher['drift']:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    TEACHER_SPRINTS = (m_teacher["fwd_speed"] > 1.0) and (m_teacher["not_fallen"] > 0.7)
    if not TEACHER_SPRINTS:
        print("TEACHER_SANITY_FAIL: teacher did not sprint in this harness "
              f"(fwd={m_teacher['fwd_speed']:.3f}, not_fallen={m_teacher['not_fallen']:.3f}). "
              "Aborting — harness/adapter bug, NOT reporting RL numbers vs a broken teacher.",
              flush=True)
        with open(EVAL_JSON, "w") as f:
            json.dump({"iter": it, "ckpt": ckpt_path, "teacher": m_teacher,
                       "teacher_sanity": "FAIL"}, f, indent=2)
        return
    print("TEACHER_SANITY_PASS", flush=True)

    # ---- 2. RL actor sprint (matched starts) + compliance capture ---------------------------
    for k in comp:
        comp[k].clear()
    t0 = time.time()
    m_pd_rl = sprint_rollout(env, rl_actor_fn, steps, qids, dids)
    print(f"\n[pd_rl] not_fallen={m_pd_rl['not_fallen']:.3f} "
          f"fwd_speed={m_pd_rl['fwd_speed']:.3f} drift={m_pd_rl['drift']:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- 3. compliance metric (log-space impedance distance) --------------------------------
    kp_pol = torch.stack(comp["kp_pol"])           # (S,n,25) names order, >=0
    kd_pol = torch.stack(comp["kd_pol"])
    log_kp_tea = torch.stack(comp["log_kp_tea"])   # (S,n,25) already log
    log_kd_tea = torch.stack(comp["log_kd_tea"])
    alive = torch.stack(comp["alive"]).bool()      # (S,n)
    S, N, J = kp_pol.shape

    log_kp_pol = kp_pol.clamp_min(ORACLE_EPS).log()
    log_kd_pol = kd_pol.clamp_min(ORACLE_EPS).log()
    d_kp = (log_kp_pol - log_kp_tea).reshape(S * N, J)[alive.reshape(S * N)]  # (M,25)
    d_kd = (log_kd_pol - log_kd_tea).reshape(S * N, J)[alive.reshape(S * N)]
    M = d_kp.shape[0]

    # Primary scalar: per-element RMS log distance over all alive samples & 50 gain dims.
    sq_all = torch.cat([d_kp, d_kd], dim=1)                      # (M,50)
    rms_log = float(sq_all.pow(2).mean().sqrt())
    # Per-env L2 over the 50-vector, averaged over alive samples.
    per_env_l2 = float(sq_all.pow(2).sum(dim=1).sqrt().mean())
    # Per-joint RMS log distance (kp and kd separately) over alive samples.
    kp_rms_log = d_kp.pow(2).mean(dim=0).sqrt()                  # (25,)
    kd_rms_log = d_kd.pow(2).mean(dim=0).sqrt()
    compliance = {
        "eps": ORACLE_EPS,
        "n_alive_samples": int(M),
        "rms_log": rms_log,                     # PRIMARY per-element RMS
        "per_env_l2_log": per_env_l2,           # mean ||.||_2 over 50 dims
        "mean_kp_rms_log": float(kp_rms_log.mean()),
        "mean_kd_rms_log": float(kd_rms_log.mean()),
        "per_joint_kp_rms_log": [float(x) for x in kp_rms_log],
        "per_joint_kd_rms_log": [float(x) for x in kd_rms_log],
        "names": names,
        "oracle_val_mse_log": oracle_meta.get("fit", {}).get("val_mse_log"),
        "threshold_rms_log": COMPLIANT_RMS_LOG,
        "convention": "log(kp/kd clamp_min eps); same-state (obs_t) policy-vs-oracle; alive=upright",
    }
    print(f"\ncompliance: n_alive={M} rms_log={rms_log:.3f} per_env_l2_log={per_env_l2:.3f} "
          f"(kp_rms={compliance['mean_kp_rms_log']:.3f} kd_rms={compliance['mean_kd_rms_log']:.3f}) "
          f"[oracle_val_mse_log={compliance['oracle_val_mse_log']}]", flush=True)
    # Notable per-joint (largest combined log-distance).
    comb = (kp_rms_log.pow(2) + kd_rms_log.pow(2)).sqrt()
    order = torch.argsort(comb, descending=True)
    print(f"{'joint':<22s} {'kp_rms_log':>10s} {'kd_rms_log':>10s}", flush=True)
    for j in order.tolist():
        print(f"{names[j]:<22s} {float(kp_rms_log[j]):>10.3f} {float(kd_rms_log[j]):>10.3f}",
              flush=True)

    # ---- 4. verdict + buckets ---------------------------------------------------------------
    sprint = (m_pd_rl["fwd_speed"] >= 0.5 * m_teacher["fwd_speed"]
              and m_pd_rl["not_fallen"] >= 0.7)
    compliant = rms_log <= COMPLIANT_RMS_LOG
    if sprint and compliant:
        bucket = "sprint+compliant"
    elif sprint and not compliant:
        bucket = "sprint-only"
    else:
        bucket = "no-sprint"

    print(f"\nSPRINT pd_rl={m_pd_rl['fwd_speed']:.3f} teacher={m_teacher['fwd_speed']:.3f} "
          f"(frac={m_pd_rl['fwd_speed']/max(m_teacher['fwd_speed'],1e-6):.2f}); "
          f"UPRIGHT pd_rl={m_pd_rl['not_fallen']:.3f} teacher={m_teacher['not_fallen']:.3f}; "
          f"COMPLIANCE rms_log={rms_log:.3f} (<= {COMPLIANT_RMS_LOG} ? {compliant})", flush=True)
    print(f"BUCKET={bucket}   (INTERIM @ iter={it})", flush=True)

    out = {
        "interim": True,
        "iter": it,
        "ckpt": ckpt_path,
        "n_envs": env.num_worlds, "steps": steps, "full_steps": full_steps, "seed": SEED,
        "teacher": m_teacher,
        "teacher_sanity": "PASS",
        "pd_rl": m_pd_rl,
        "compliance": compliance,
        "verdict": {
            "sprint": bool(sprint), "compliant": bool(compliant), "bucket": bucket,
            "thresholds": {"fwd_frac_min": 0.5, "not_fallen_min": 0.7,
                           "compliant_rms_log_max": COMPLIANT_RMS_LOG},
        },
    }
    with open(EVAL_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {EVAL_JSON}", flush=True)


if __name__ == "__main__":
    main()
