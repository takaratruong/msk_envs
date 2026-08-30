"""Task 4 (PRIMARY USER GATE) — PD-impedance student evaluation.

The decisive experiment for Phase B: does the distilled PD-impedance student reproduce the
muscle teacher's ACTUATION (torque) and its SPRINT?

The user's success criterion is literal: "we have a final torque to compare to". So the primary
metric is the applied PD torque
        tau_PD(t) = clamp( kp*(q_des - q) - kd*qdot , +/- optimal_force )
versus the teacher's muscle net joint torque tau_muscle(t) at MATCHED states, per joint.

--------------------------------------------------------------------------------------------------
WHY THE TORQUE MATCH IS EVALUATED ON THE TEACHER'S TRAJECTORY (matched-state design)
--------------------------------------------------------------------------------------------------
tau_muscle is read from bolt.ufrc_muscle, the muscle net joint torque at the CURRENT muscle
*activation*. Muscle activation is a hidden state that is only integrated by a real env.step under
the teacher's excitations (Task 2 / impedance.py: `forward.fwd` recomputes muscle force from
path-length/velocity but does NOT integrate activation). So tau_muscle is only PHYSICAL at states
the teacher actually drove. You cannot conjure a teacher muscle torque at a pd-student-visited
state by writing excitations + fwd — activation there has decayed (muscles were off).

Therefore the physically-honest matched-state comparison is done ON THE TEACHER ROLLOUT: at each
teacher-driven state we (a) read the real tau_muscle, and (b) query the student at that exact obs
and compute the tau_PD it WOULD command from its decoded (q_des, kp, kd) at the identical (q, qdot).
Both torques are evaluated at the same state -> a clean per-joint tau_PD-vs-tau_muscle comparison.
This is the same relabel idea as Task 2/3 (read the teacher's torque at a visited state), applied
at eval time on states where the muscle torque is genuine.

The q_des clamp (q_des = q + clamp(delta, +/-QDES_DELTA_CLAMP)) and the softplus>=0 gains are
REPLICATED EXACTLY from dagger_pd.decode + rollout_and_label so tau_PD is what the deployed student
would truly apply.

--------------------------------------------------------------------------------------------------
SPRINT / UPRIGHT (single reconciled has_fallen metric, matched starts)
--------------------------------------------------------------------------------------------------
Both teacher and pd_student are rolled out through the SAME manual step loop (mirroring pd_probe.py:
pre_sim_step -> launch_sim_step -> update_metrics -> has_fallen -> _get_terminated -> _perform_reset
-> _get_obs). `not_fallen` is the mean per-step survival under a per-env alive latch (has_fallen),
identical for both controllers. Starts are matched by seeding the RNG identically before each reset.
  - teacher : base-Sprinter muscle policy, obs reduced 359->336, excitations to the muscle slice.
  - pd_student : muscles OFF, PD buffers written from the decoded (q_des, kp, kd) (apply_pd path).

Verdict buckets: match+sprint / match+no-sprint / no-match.
"""
import json
import time

import torch

import bolt
import bolt._src.forward as forward

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.dagger_pd import StudentPD, decode, QDES_DELTA_CLAMP
from msk_envs.distill.dof_utils import (
    joint_dof_indices, build_actuator_perm, build_teacher_obs_cols,
)
from msk_envs.utils.reward_lib import has_fallen
from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_pd/pd_student.pt"
OPTFORCE = "/home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/sprinter_torque_optforce.json"
EVAL_JSON = "/home/ubuntu/msk_envs/models/dagger_pd/eval_pd.json"

N_ENVS = 256
SEED = 0
MAX_STEPS = 300  # cap the full episode (q_des clamp prevents stall, but bound wall-time)

# distill/eval doesn't use rewards; the SprinterTorque(PD) config ships without reward lambdas.
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


def _fwd_speed(env):
    """World-frame forward (FWD) linear velocity of the pelvis root, per env."""
    return env.body_velocities[:, env.root_id, FWD_IDX + 3]


def read_optimal_force(names, device):
    """Per-DoF optimal_force (torque clamp), ordered to match `names`."""
    with open(OPTFORCE) as f:
        of_map = json.load(f)
    return torch.tensor([of_map[n] for n in names], device=device, dtype=torch.float32)


def load_student(path, dev):
    ck = torch.load(path, map_location=dev)
    s = StudentPD(ck["n_obs"], n_dof=25, device=dev)
    s.load_state_dict(ck["state_dict"])
    s.eval()
    kp_scale = ck["kp_scale"].to(dev)
    kd_scale = ck["kd_scale"].to(dev)
    return s, kp_scale, kd_scale, ck["names"]


def student_pd_targets(student, obs, q_now, kp_scale, kd_scale):
    """Decode the student at `obs` and produce the DEPLOYED (q_des, kp, kd) in names order.

    Replicates dagger_pd.decode + the rollout-time q_des clamp EXACTLY:
        q_des = q_now + clamp(raw_qdes - q_now, +/-QDES_DELTA_CLAMP)
    """
    with torch.no_grad():
        raw = student(obs)
    q_des_s, kp_s, kd_s = decode(raw, kp_scale, kd_scale)
    delta = torch.clamp(q_des_s - q_now, -QDES_DELTA_CLAMP, QDES_DELTA_CLAMP)
    q_des_s = q_now + delta
    return q_des_s, kp_s, kd_s


def _reset_matched(env):
    """Reset with a fixed seed so teacher and pd_student see identical start states."""
    torch.manual_seed(SEED)
    return env.reset()


def sprint_rollout(env, action_fn, steps, qids, dids, teacher_cols=None,
                   record=None):
    """Manual-step rollout mirroring pd_probe.rollout. Returns sprint/upright metrics.

    action_fn(obs) -> full action tensor (writes any PD buffers as a side effect).
    record: optional dict of preallocated (steps, n, 25) buffers {tau_pd, tau_mus, alive}
            filled ONLY on the teacher rollout (where tau_muscle is physical). When present,
            `student`, `kp_scale`, `kd_scale`, `optf`, `idx_t` must be in record too.
    """
    obs = _reset_matched(env)
    n = env.num_worlds
    dev = obs.device
    m, d = env.m, env.d
    upright = torch.zeros(n, device=dev)
    fwd_sum = torch.zeros(n, device=dev)
    drift = torch.zeros(n, device=dev)

    for t in range(steps):
        a = action_fn(obs)
        env.pre_sim_step(a)
        env.launch_sim_step()
        env.update_metrics()

        # Per-step survival (NOT latched): env.step-style auto-reset keeps a fallen env from
        # polluting velocity stats, and `not_fallen` is the mean per-step upright rate — the
        # standard locomotion survival metric (matches Phase A evaluate.run_metrics). `alive`
        # here is the per-step (1 - has_fallen) weight, identical for teacher and pd_student.
        fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation).float()
        alive = 1.0 - fallen
        upright += alive
        v = _fwd_speed(env)
        fwd_sum += v * alive
        drift += env.root_pos[:, SIDE_IDX].abs() * alive

        if record is not None:
            # State post-step (teacher-driven activation). Recompute muscle force AT this exact
            # (q, qdot, activation) so tau_muscle matches the state we score tau_PD at.
            q = env.joint_positions.index_select(1, qids).clone()      # (n,25) names order
            qd = env.joint_velocities.index_select(1, dids).clone()    # (n,25) names order
            forward.fwd(m, d)
            tau_mus = bolt.ufrc_muscle(d).index_select(1, record["idx_t"]).clone()
            obs_next = env._get_obs()
            q_des_s, kp_s, kd_s = student_pd_targets(
                record["student"], obs_next, q, record["kp_scale"], record["kd_scale"])
            tau_pd = torch.clamp(kp_s * (q_des_s - q) - kd_s * qd,
                                 -record["optf"], record["optf"])
            record["tau_pd"][t] = tau_pd
            record["tau_mus"][t] = tau_mus
            record["alive"][t] = alive.clone()  # (n,) per-env survival latch

        term = env._get_terminated()
        done = term.clamp(0.0, 1.0).unsqueeze(-1)
        if done.any():
            env._perform_reset(done)
        obs = env._get_obs()

    denom = upright.clamp(min=1)
    return {
        "not_fallen": (upright / steps).mean().item(),
        "fwd_speed": (fwd_sum / denom).mean().item(),
        "drift": (drift / denom).mean().item(),
    }


def per_joint_torque_match(tau_pd, tau_mus, alive, names):
    """Per-joint Pearson correlation + RMS(tau_PD)/RMS(tau_muscle) over alive samples.

    tau_pd/tau_mus: (steps, n, 25); alive: (steps, n). Uses only samples where alive>0
    (drops post-fall garbage).
    """
    S, N, J = tau_pd.shape
    m = alive.reshape(S * N).bool()
    xp = tau_pd.reshape(S * N, J)[m]        # (M,25)
    xm = tau_mus.reshape(S * N, J)[m]
    M = xp.shape[0]
    corr, rms_ratio, rms_pd, rms_mus = [], [], [], []
    for j in range(J):
        a = xp[:, j]
        b = xm[:, j]
        am = a - a.mean()
        bm = b - b.mean()
        denom = am.norm() * bm.norm()
        c = (am @ bm) / denom if denom > 0 else torch.tensor(0.0, device=a.device)
        rp = a.pow(2).mean().sqrt()
        rm = b.pow(2).mean().sqrt()
        corr.append(float(c))
        rms_pd.append(float(rp))
        rms_mus.append(float(rm))
        rms_ratio.append(float(rp / rm) if rm > 0 else float("nan"))
    return {
        "n_samples": int(M),
        "names": names,
        "corr": corr,
        "rms_pd": rms_pd,
        "rms_mus": rms_mus,
        "rms_ratio": rms_ratio,
        "mean_corr": float(sum(corr) / len(corr)),
    }


def main():
    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=N_ENVS, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)
    assert getattr(cfg, "use_pd_actuators", False), "PD env must have use_pd_actuators=True"

    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "label order (joint_dof_indices) != teacher.names"
    teacher_cols = build_teacher_obs_cols(env, dev)   # 359-obs -> 336-obs teacher adapter
    act_perm = build_actuator_perm(env, names, dev)   # names order -> actuator-slice order
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    dids = torch.tensor([env.dof_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    n_musc = env.num_muscles
    optf = read_optimal_force(names, dev)             # (25,) names order, torque clamp

    student, kp_scale, kd_scale, s_names = load_student(STUDENT_PT, dev)
    assert s_names == names, "pd_student names != joint_dof_indices order"

    full_steps = int(round(env.max_episode_duration / env.delta_t))
    steps = min(full_steps, MAX_STEPS)
    print(f"n_envs={env.num_worlds} full_steps={full_steps} steps={steps} "
          f"n_musc={n_musc} obs={env.reset().shape[1]}", flush=True)

    # ---- action fns -------------------------------------------------------------------------
    def teacher_fn(obs):
        # Zero PD buffers so persistent gains never leak torque into the teacher-driven advance.
        bolt.pd_kp(env.d).zero_()
        bolt.pd_kd(env.d).zero_()
        bolt.pd_q_des(env.d).zero_()
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        return a

    def pd_student_fn(obs):
        q_now = env.joint_positions.index_select(1, qids)
        q_des_s, kp_s, kd_s = student_pd_targets(student, obs, q_now, kp_scale, kd_scale)
        # Write PD buffers in actuator-slice order (apply_pd convention), muscles OFF.
        bolt.pd_q_des(env.d).copy_(q_des_s.index_select(1, act_perm))
        bolt.pd_kp(env.d).copy_(torch.clamp(kp_s, min=0.0).index_select(1, act_perm))
        bolt.pd_kd(env.d).copy_(torch.clamp(kd_s, min=0.0).index_select(1, act_perm))
        a = env.get_blank_actions()
        a[:, :n_musc] = -1.0  # muscles OFF
        return a

    # ---- 1. TEACHER SANITY GATE (+ torque recording) ---------------------------------------
    tau_pd = torch.zeros(steps, env.num_worlds, 25, device=dev)
    tau_mus = torch.zeros(steps, env.num_worlds, 25, device=dev)
    aliveb = torch.zeros(steps, env.num_worlds, device=dev)
    record = {"tau_pd": tau_pd, "tau_mus": tau_mus, "alive": aliveb,
              "student": student, "kp_scale": kp_scale, "kd_scale": kd_scale,
              "optf": optf, "idx_t": teacher.idx_t}

    t0 = time.time()
    m_teacher = sprint_rollout(env, teacher_fn, steps, qids, dids,
                               teacher_cols=teacher_cols, record=record)
    print(f"[teacher] not_fallen={m_teacher['not_fallen']:.3f} "
          f"fwd_speed={m_teacher['fwd_speed']:.3f} drift={m_teacher['drift']:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    TEACHER_SPRINTS = (m_teacher["fwd_speed"] > 1.0) and (m_teacher["not_fallen"] > 0.7)
    if not TEACHER_SPRINTS:
        print("TEACHER_SANITY_FAIL: teacher did not sprint in this harness "
              f"(fwd={m_teacher['fwd_speed']:.3f}, not_fallen={m_teacher['not_fallen']:.3f}). "
              "Aborting — harness/adapter bug, NOT reporting student numbers vs a broken teacher.",
              flush=True)
        with open(EVAL_JSON, "w") as f:
            json.dump({"teacher": m_teacher, "teacher_sanity": "FAIL"}, f, indent=2)
        return
    print("TEACHER_SANITY_PASS", flush=True)

    # ---- 2. torque match (per-joint) --------------------------------------------------------
    torque_match = per_joint_torque_match(tau_pd, tau_mus, aliveb, names)
    print(f"\ntorque_match: n_samples={torque_match['n_samples']} "
          f"mean_per_joint_corr={torque_match['mean_corr']:.3f}", flush=True)
    print(f"{'joint':<22s} {'corr':>7s} {'rms_pd':>9s} {'rms_mus':>9s} {'rms_ratio':>9s}",
          flush=True)
    for j, nm in enumerate(names):
        print(f"{nm:<22s} {torque_match['corr'][j]:>7.3f} "
              f"{torque_match['rms_pd'][j]:>9.2f} {torque_match['rms_mus'][j]:>9.2f} "
              f"{torque_match['rms_ratio'][j]:>9.3f}", flush=True)

    # ---- 3. pd_student sprint (matched starts) ----------------------------------------------
    t0 = time.time()
    m_pd_student = sprint_rollout(env, pd_student_fn, steps, qids, dids)
    print(f"\n[pd_student] not_fallen={m_pd_student['not_fallen']:.3f} "
          f"fwd_speed={m_pd_student['fwd_speed']:.3f} drift={m_pd_student['drift']:.3f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- 4. verdict -------------------------------------------------------------------------
    mean_corr = torque_match["mean_corr"]
    match = mean_corr >= 0.5                       # torque trajectories track the teacher
    sprint = (m_pd_student["fwd_speed"] >= 0.5 * m_teacher["fwd_speed"]
              and m_pd_student["not_fallen"] >= 0.7)
    if match and sprint:
        bucket = "match+sprint"
    elif match and not sprint:
        bucket = "match+no-sprint"
    else:
        bucket = "no-match"

    print(f"\nTORQUE_MATCH={mean_corr:.3f}; "
          f"SPRINT pd={m_pd_student['fwd_speed']:.3f} teacher={m_teacher['fwd_speed']:.3f}; "
          f"UPRIGHT pd={m_pd_student['not_fallen']:.3f} teacher={m_teacher['not_fallen']:.3f}",
          flush=True)
    print(f"BUCKET={bucket}", flush=True)

    out = {
        "n_envs": env.num_worlds, "steps": steps, "full_steps": full_steps, "seed": SEED,
        "teacher": m_teacher,
        "pd_student": m_pd_student,
        "teacher_sanity": "PASS",
        "torque_match": {k: v for k, v in torque_match.items()},
        "verdict": {"mean_corr": mean_corr, "match": bool(match), "sprint": bool(sprint),
                    "bucket": bucket,
                    "thresholds": {"corr_min": 0.5, "fwd_frac_min": 0.5, "not_fallen_min": 0.7}},
    }
    with open(EVAL_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {EVAL_JSON}", flush=True)


if __name__ == "__main__":
    main()
