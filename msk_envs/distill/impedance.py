"""Task 2 — Teacher impedance extraction via finite-difference of the muscle model.

Given the muscle env at a teacher-driven state (teacher's muscle activations already applied
by the preceding teacher step), we extract, per non-root joint DoF, the diagonal PD-equivalent
impedance of the muscle net joint torque `ufrc_muscle`:

    kp[j] = -d tau_j / d q_j          (restoring stiffness; positive = spring holding the joint)
    kd[j] = -d tau_j / d qdot_j       (damping)
    q_des[j] = q_j + tau_j / max(kp[j], kp_min)

These serve as distillation targets for a PD student (Phase B).

RECOMPUTE MECHANISM (the crux):
  `bolt._src.forward.fwd(m, d)` realizes position -> velocity -> muscles -> forces and
  populates `d.ufrc_muscle` WITHOUT advancing time (no integration). Crucially, within a
  single `fwd`, muscle *activation* is NOT integrated (`smooth_muscle_act.activation_dynamics`
  only writes the derivative `m_act_dot`; it is never applied inside `fwd`). So repeated `fwd`
  calls at perturbed `qpos`/`qvel` recompute the muscle force purely from the perturbed muscle
  path length and contraction velocity, holding the teacher's activation fixed — exactly the
  impedance-at-current-activation we want.

  Finite-difference recipe per joint j (central difference):
    save qpos, qvel (full clones)
    d.qpos[:, qpos_adr[j]] += eps_p; fwd; tau_plus  = ufrc_muscle[:, dof[j]]; restore
    d.qpos[:, qpos_adr[j]] -= eps_p; fwd; tau_minus = ufrc_muscle[:, dof[j]]; restore
    kp[:, j] = -(tau_plus - tau_minus) / (2 eps_p)
    (same for qvel +/- eps_d -> kd)
  After all perturbations, restore qpos/qvel byte-identically and call `fwd` once more so
  downstream reads are consistent.

ASSUMPTION: extract_impedance reads the CURRENT env state and does NOT re-apply the teacher's
action. It must be called right after a teacher step, when `d` holds the teacher-driven muscle
activation state. We do NOT touch `m_excitations`/`m_act`.

INDEX SPACES: qpos has width 32 (a quaternion), qvel/ufrc_muscle have width 31 (nv/dof space).
Position perturbations use `env.qpos_id_lookup[name]`; velocity perturbations and ufrc_muscle
slicing use `env.dof_id_lookup[name]`. Names come from `joint_dof_indices` (25 non-root joints).
"""
import torch

import bolt
import bolt._src.forward as forward
from msk_envs.distill.dof_utils import joint_dof_indices

KP_MIN = 1.0
EPS_P = 1e-3   # rad
EPS_D = 1e-2   # rad/s


@torch.no_grad()
def extract_impedance(env, teacher, eps_p: float = EPS_P, eps_d: float = EPS_D,
                      kp_min: float = KP_MIN):
    """Extract per-joint (q_des, kp, kd) at the CURRENT (teacher-driven) env state.

    Args:
        env: muscle env (EnvConfigSprinterTorque) just after a teacher step.
        teacher: Teacher instance (used only for `.names`/`.idx` consistency).
        eps_p, eps_d: finite-difference steps (rad, rad/s).
        kp_min: stiffness floor below which q_des falls back to current q.

    Returns:
        (q_des, kp, kd), each (n_envs, 25) torch tensor in names order; finite; kp,kd >= 0.
    """
    m, d = env.m, env.d
    names, idx = joint_dof_indices(env)
    dev = bolt.joint_positions(d).device

    qpos = bolt.joint_positions(d)       # (n_envs, 32) torch view of d.qpos
    qvel = bolt.joint_velocities(d)      # (n_envs, 31) torch view of d.qvel
    n_envs = qpos.shape[0]
    n_j = len(names)

    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], dtype=torch.long, device=dev)
    dids = torch.tensor([env.dof_id_lookup[n] for n in names], dtype=torch.long, device=dev)

    # Baseline (unperturbed) muscle torque at the current activation, for the setpoint.
    forward.fwd(m, d)
    tau0 = bolt.ufrc_muscle(d).index_select(1, dids).clone()   # (n_envs, 25) names order
    q0 = qpos.index_select(1, qids).clone()                    # (n_envs, 25) names order

    # Full-state save for byte-identical restore after each perturbation.
    qpos_save = qpos.clone()
    qvel_save = qvel.clone()

    kp = torch.empty((n_envs, n_j), device=dev)
    kd = torch.empty((n_envs, n_j), device=dev)

    ufrc = lambda: bolt.ufrc_muscle(d)  # re-fetch view each call (same underlying buffer)

    for j in range(n_j):
        qadr = int(qids[j].item())
        dadr = int(dids[j].item())

        # --- stiffness: perturb qpos[qadr] +/- eps_p ---
        qpos[:, qadr] += eps_p
        forward.fwd(m, d)
        tau_pp = ufrc()[:, dadr].clone()
        qpos.copy_(qpos_save)

        qpos[:, qadr] -= eps_p
        forward.fwd(m, d)
        tau_pm = ufrc()[:, dadr].clone()
        qpos.copy_(qpos_save)

        kp[:, j] = -(tau_pp - tau_pm) / (2.0 * eps_p)

        # --- damping: perturb qvel[dadr] +/- eps_d ---
        qvel[:, dadr] += eps_d
        forward.fwd(m, d)
        tau_dp = ufrc()[:, dadr].clone()
        qvel.copy_(qvel_save)

        qvel[:, dadr] -= eps_d
        forward.fwd(m, d)
        tau_dm = ufrc()[:, dadr].clone()
        qvel.copy_(qvel_save)

        kd[:, j] = -(tau_dp - tau_dm) / (2.0 * eps_d)

    # Restore exactly and leave the sim consistent for downstream reads.
    qpos.copy_(qpos_save)
    qvel.copy_(qvel_save)
    forward.fwd(m, d)

    # Clamp to physical (non-negative) impedance.
    kp = kp.clamp_min(0.0)
    kd = kd.clamp_min(0.0)

    # Setpoint: q_des = q + tau / max(kp, kp_min); fall back to q where kp < kp_min.
    kp_eff = kp.clamp_min(kp_min)
    q_des = q0 + tau0 / kp_eff
    fallback = kp < kp_min
    q_des = torch.where(fallback, q0, q_des)

    return q_des, kp, kd


def _fmt_row(name, v_kp, v_kd):
    return f"  {name:<24s} kp={v_kp:9.3f}  kd={v_kd:9.4f}"


def main():
    import torch
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorque
    from msk_envs.distill.teacher import Teacher
    from msk_envs.distill.dof_utils import build_teacher_obs_cols

    CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
    REWARD_LAMBDAS = {
        "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
        "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
    }

    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorque()
    cfg.reward_lambdas = dict(REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=64, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)

    names, idx = joint_dof_indices(env)
    n_musc = env.num_muscles
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "names order mismatch vs teacher"
    teacher_cols = build_teacher_obs_cols(env, dev)

    def teacher_action(obs):
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        return a

    obs = env.reset()
    # Warm up a few teacher steps so the muscle activations are teacher-driven and the
    # gait is underway (stance loads the leg muscles).
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(teacher_action(obs))

    # ---- PROOF: ufrc_muscle actually CHANGES under a qpos perturbation (not all-zero) ----
    m, d = env.m, env.d
    forward.fwd(m, d)
    base = bolt.ufrc_muscle(d).clone()
    qpos = bolt.joint_positions(d)
    qsave = qpos.clone()
    # perturb the first non-root joint's qpos
    probe_name = names[0]
    qadr = env.qpos_id_lookup[probe_name]
    qpos[:, qadr] += 1e-2
    forward.fwd(m, d)
    after = bolt.ufrc_muscle(d).clone()
    qpos.copy_(qsave)
    forward.fwd(m, d)
    delta = (after - base).abs()
    print(f"RESULT proof_ufrc_changes joint={probe_name} "
          f"max|dUfrc|={delta.max().item():.6e} mean|dUfrc|={delta.mean().item():.6e} "
          f"nonzero_dofs={(delta.max(dim=0).values > 1e-9).sum().item()}/{delta.shape[1]}",
          flush=True)

    # ---- Roll the teacher a few steps, extracting impedance each step ----
    n_steps = 8
    kp_acc = torch.zeros(len(names), device=dev)
    kd_acc = torch.zeros(len(names), device=dev)
    kp_pos_frac = torch.zeros(len(names), device=dev)
    all_finite = True
    kp_nonneg = True

    # ε-sensitivity: compare full-ε vs half-ε at the first extraction step.
    q1, kp1, kd1 = extract_impedance(env, teacher, eps_p=EPS_P, eps_d=EPS_D)
    q2, kp2, kd2 = extract_impedance(env, teacher, eps_p=EPS_P / 2, eps_d=EPS_D / 2)
    denom_kp = kp1.abs().clamp_min(1.0)
    denom_kd = kd1.abs().clamp_min(1.0)
    rel_kp = ((kp1 - kp2).abs() / denom_kp).max().item()
    rel_kd = ((kd1 - kd2).abs() / denom_kd).max().item()
    # Also a magnitude-weighted (mean) relative change for context.
    mrel_kp = ((kp1 - kp2).abs() / denom_kp).mean().item()
    mrel_kd = ((kd1 - kd2).abs() / denom_kd).mean().item()

    for step in range(n_steps):
        q_des, kp, kd = extract_impedance(env, teacher)
        assert q_des.shape == (env.num_worlds, 25)
        assert kp.shape == (env.num_worlds, 25) and kd.shape == (env.num_worlds, 25)
        all_finite = all_finite and bool(torch.isfinite(q_des).all() and
                                         torch.isfinite(kp).all() and torch.isfinite(kd).all())
        kp_nonneg = kp_nonneg and bool((kp >= 0).all() and (kd >= 0).all())
        kp_acc += kp.mean(dim=0)
        kd_acc += kd.mean(dim=0)
        kp_pos_frac += (kp.mean(dim=0) > 0).float()

        # advance teacher one step
        obs, _, term, trunc, _ = env.step(teacher_action(obs))

    kp_mean = (kp_acc / n_steps).cpu()
    kd_mean = (kd_acc / n_steps).cpu()
    frac_pos_joints = (kp_pos_frac / n_steps)  # per-joint fraction of steps with mean kp>0

    print(f"RESULT all_finite={all_finite} kp_kd_nonneg={kp_nonneg}", flush=True)
    print(f"RESULT eps_sensitivity max_rel_kp={rel_kp:.4f} max_rel_kd={rel_kd:.4f} "
          f"mean_rel_kp={mrel_kp:.4f} mean_rel_kd={mrel_kd:.4f} "
          f"(target <0.20)", flush=True)
    n_pos = int((kp_mean > 0).sum().item())
    print(f"RESULT kp_positive_joints={n_pos}/{len(names)} "
          f"kp_mean_overall={kp_mean.mean().item():.3f} "
          f"kd_mean_overall={kd_mean.mean().item():.4f}", flush=True)

    print("Per-joint mean kp/kd over teacher rollout (names order):", flush=True)
    for i, nm in enumerate(names):
        print(_fmt_row(nm, kp_mean[i].item(), kd_mean[i].item()), flush=True)

    # Sanity on q_des fallback behavior
    q_des, kp, kd = extract_impedance(env, teacher)
    q_now = bolt.joint_positions(env.d).index_select(
        1, torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev)).clone()
    fb = (kp < KP_MIN)
    if fb.any():
        max_fb_dev = (q_des[fb] - q_now[fb]).abs().max().item()
    else:
        max_fb_dev = 0.0
    print(f"RESULT qdes_fallback_frac={fb.float().mean().item():.3f} "
          f"max|qdes-q|_where_kp<kpmin={max_fb_dev:.3e} (should be ~0)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
