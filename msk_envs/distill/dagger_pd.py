"""DAgger loop (Phase B): distill the muscle teacher's IMPEDANCE into a PD student.

The student MLP maps obs(359) -> 75 = concat(q_des[25], kp_raw[25], kd_raw[25]) and those
outputs drive the sub-step PD actuator (Task 0): tau = clamp(kp*(q_des-q) - kd*qdot, +/-optforce).
Labels come from Task 2's `extract_impedance` (finite-difference of the muscle net joint torque):
per-state (q_des, kp, kd) targets, each (n_envs, 25) in `names` order.

This mirrors the Phase A torque DAgger (`dagger.py`): rollout_and_label / fit / main / beta schedule.
The differences are the 75-dim impedance target, the output scaling, and that the student rollout
advances via the PD actuator path (`pd_control.apply_pd`) instead of open-loop torque.

--------------------------------------------------------------------------------------------------
OUTPUT SCALING (Task-2 finding: kp is O(1000), kd is O(1), q_des is O(1) radians)
--------------------------------------------------------------------------------------------------
The three sub-vectors live on wildly different scales, so a raw linear head would make kp swamp
everything. The student emits a raw 75-vector; we decode it as:

    q_des = raw_qdes                                  (linear, radians)
    kp    = softplus(raw_kp) * KP_SCALE[j]            (>= 0, ballpark of teacher kp)
    kd    = softplus(raw_kd) * KD_SCALE[j]            (>= 0, ballpark of teacher kd)

KP_SCALE / KD_SCALE are PER-JOINT constants set to the teacher's MEAN kp / kd for that joint,
measured by a short impedance calibration pass at startup (a few hundred teacher-driven states),
floored at 1.0. With raw~0 the student emits ~softplus(0)*SCALE = 0.69*SCALE, i.e. right in the
teacher's range; softplus keeps kp,kd >= 0. The scales are saved in the checkpoint.

--------------------------------------------------------------------------------------------------
LOSS NORMALIZATION (chosen: option (a) — train on the RAW, pre-scale targets)
--------------------------------------------------------------------------------------------------
Instead of computing MSE on the physical (q_des,kp,kd) — where kp's O(1000) magnitude would
dominate and swamp q_des's O(1) — we invert the scaling to express the labels as the raw values
the head would have to emit, then MSE on those:

    target_raw_qdes = q_des
    target_raw_kp   = softplus_inv(clamp(kp / KP_SCALE, min=eps))
    target_raw_kd   = softplus_inv(clamp(kd / KD_SCALE, min=eps))

softplus_inv(y) = y + log(-expm1(-y)) (stable). The eps clamp keeps joints with kd==0 (ankle/
subtalar/elbow/mtp, per Task 2) finite (target_raw ~ -6.9 -> decoded kd ~ 0). All three sub-blocks
are now O(1), so the 75-dim MSE weights q_des, kp and kd comparably. val_mse is reported on this
raw scale (consistent across rounds, so the "final < 0.6x round-0" criterion is well defined).

--------------------------------------------------------------------------------------------------
STALL MITIGATION (Task 0/2 finding: q_des far from the pose stalls Bolt's adaptive integrator)
--------------------------------------------------------------------------------------------------
  * beta schedule starts at 1.0 (round 0 = pure teacher, no student authority) and anneals.
  * The student's q_des is clamped to current_q + clamp(delta, +/-0.5 rad) at rollout time, so an
    ill-trained early student can never command a pose far from where the body currently is.
  * kp/kd are softplus>=0 and in the teacher's calibrated range by construction.

--------------------------------------------------------------------------------------------------
PD-ENV BOOKKEEPING
--------------------------------------------------------------------------------------------------
The env is EnvConfigSprinterTorquePD (use_pd_actuators=True). The PD buffers (pd_q_des/pd_kp/pd_kd)
persist in Data across steps. A student PD step leaves them nonzero, so the *teacher probe step*
explicitly ZEROES pd_kp/pd_kd first (actuators contribute no torque) and drives the body with the
teacher's muscle excitations alone -> a clean teacher-driven advance. extract_impedance reads
ufrc_muscle (muscle-only) and is independent of the actuator torque, so labels are unaffected.
Student rollout uses apply_pd (muscles OFF, gains routed names->actuator order via act_perm) —
the identical convention used for the labels.
"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import bolt
from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.impedance import extract_impedance
from msk_envs.distill.pd_control import apply_pd
from msk_envs.distill.dof_utils import (
    joint_dof_indices, build_actuator_perm, build_teacher_obs_cols,
)

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
OUT_DIR = "/home/ubuntu/msk_envs/models/dagger_pd"

# distill doesn't use rewards; SprinterTorque(PD) ships without reward lambdas.
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}

SCALE_FLOOR = 1.0        # floor on per-joint KP_SCALE/KD_SCALE (avoids 0 scale on kd==0 joints)
RAW_EPS = 1e-3           # floor on (physical/scale) before softplus_inv (keeps kd==0 finite)
QDES_DELTA_CLAMP = 0.5   # rad: student q_des is clamped to current_q +/- this at rollout time


class StudentPD(nn.Module):
    """obs(n_obs) -> raw 75 = concat(raw_qdes[25], raw_kp[25], raw_kd[25])."""

    def __init__(self, n_obs, n_dof=25, hidden=(256, 256), device="cuda"):
        super().__init__()
        dims = [n_obs, *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b, device=device), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 3 * n_dof, device=device)]
        self.net = nn.Sequential(*layers)
        self.n_dof = n_dof

    def forward(self, obs):
        return self.net(obs)  # (n, 75) raw


def softplus_inv(y):
    """Stable inverse of softplus: log(exp(y) - 1) = y + log(-expm1(-y)), for y > 0."""
    return y + torch.log(-torch.expm1(-y))


def decode(raw, kp_scale, kd_scale):
    """raw(n,75) -> physical (q_des, kp, kd), each (n,25) in names order."""
    n = raw.shape[1] // 3
    q_des = raw[:, :n]
    kp = F.softplus(raw[:, n:2 * n]) * kp_scale
    kd = F.softplus(raw[:, 2 * n:]) * kd_scale
    return q_des, kp, kd


def encode_targets(q_des, kp, kd, kp_scale, kd_scale):
    """Physical labels -> RAW pre-scale targets (n,75) that the head should emit (loss space)."""
    raw_kp = softplus_inv((kp / kp_scale).clamp_min(RAW_EPS))
    raw_kd = softplus_inv((kd / kd_scale).clamp_min(RAW_EPS))
    return torch.cat([q_des, raw_kp, raw_kd], dim=1)


def teacher_probe_step(env, teacher, teacher_cols, n_musc):
    """Advance the body under the TEACHER's muscles alone. Zeroes the PD buffers first so the
    persistent actuator gains from a prior student step don't corrupt the teacher-driven advance.
    Returns the post-step obs."""
    bolt.pd_kp(env.d).zero_()
    bolt.pd_kd(env.d).zero_()
    bolt.pd_q_des(env.d).zero_()
    obs = env._get_obs()
    a = env.get_blank_actions()
    a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
    obs_next, _, _, _, _ = env.step(a)
    return obs_next


def calibrate_scales(env, teacher, teacher_cols, n_musc, qids, warmup=10, n_steps=4):
    """Short impedance calibration: warm up the teacher gait, then average kp/kd per joint over a
    few hundred teacher-driven states. Returns (KP_SCALE, KD_SCALE) each (1,25), floored."""
    env.reset()
    for _ in range(warmup):
        teacher_probe_step(env, teacher, teacher_cols, n_musc)
    kp_acc = None
    kd_acc = None
    for _ in range(n_steps):
        teacher_probe_step(env, teacher, teacher_cols, n_musc)
        _, kp, kd = extract_impedance(env, teacher)
        kp_m = kp.mean(dim=0)
        kd_m = kd.mean(dim=0)
        kp_acc = kp_m if kp_acc is None else kp_acc + kp_m
        kd_acc = kd_m if kd_acc is None else kd_acc + kd_m
    kp_scale = (kp_acc / n_steps).clamp_min(SCALE_FLOOR).unsqueeze(0)  # (1,25)
    kd_scale = (kd_acc / n_steps).clamp_min(SCALE_FLOOR).unsqueeze(0)
    return kp_scale, kd_scale


def rollout_and_label(env, teacher, student, beta, horizon, teacher_cols, act_perm,
                      kp_scale, kd_scale, qids, n_musc):
    """Roll out `horizon` steps. Every step: teacher probe step (advance under muscles) then
    extract_impedance -> 75-dim RAW label at the visited obs. beta-mix advance: keep the teacher's
    resulting state (prob beta) or take a student PD step via apply_pd (prob 1-beta)."""
    obs = env.reset()
    O, L = [], []
    for _ in range(horizon):
        O.append(obs.detach().clone())
        obs_teacher = teacher_probe_step(env, teacher, teacher_cols, n_musc)
        q_des, kp, kd = extract_impedance(env, teacher)
        L.append(encode_targets(q_des, kp, kd, kp_scale, kd_scale))
        if torch.rand(()) < beta:
            obs = obs_teacher
        else:
            with torch.no_grad():
                raw = student(obs)
            q_des_s, kp_s, kd_s = decode(raw, kp_scale, kd_scale)
            # Clamp q_des to a sane range around the CURRENT pose (integrator stall defense).
            q_now = env.joint_positions.index_select(1, qids)
            delta = torch.clamp(q_des_s - q_now, -QDES_DELTA_CLAMP, QDES_DELTA_CLAMP)
            q_des_s = q_now + delta
            obs, _, _, _, _ = apply_pd(env, q_des_s, kp_s, kd_s, act_perm)
    return torch.cat(O), torch.cat(L)


def fit(student, buf_O, buf_L, epochs, device):
    """80/20 train/val split (disjoint), Adam 1e-3, MSE on the 75-dim RAW target, mb 4096.
    Returns val MSE (raw scale)."""
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    n = buf_O.shape[0]
    idx = torch.randperm(n, device=device)
    n_val = n // 5
    vi, ti = idx[:n_val], idx[n_val:]
    for _ in range(epochs):
        perm = ti[torch.randperm(ti.shape[0], device=device)]
        for mb in perm.split(4096):
            opt.zero_grad()
            loss = ((student(buf_O[mb]) - buf_L[mb]) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        val = ((student(buf_O[vi]) - buf_L[vi]) ** 2).mean().item()
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n_envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=args.n_envs, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)
    assert getattr(cfg, "use_pd_actuators", False), "PD env must have use_pd_actuators=True"

    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "label order (joint_dof_indices) != teacher.names"
    teacher_cols = build_teacher_obs_cols(env, dev)   # 359-obs -> 336-obs teacher adapter
    act_perm = build_actuator_perm(env, names, dev)   # names order -> actuator-slice order
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    n_musc = env.num_muscles

    n_obs = env.reset().shape[1]
    student = StudentPD(n_obs, n_dof=25, device=dev)

    # --- calibrate per-joint output scales from the teacher's impedance ---
    t0 = time.time()
    kp_scale, kd_scale = calibrate_scales(env, teacher, teacher_cols, n_musc, qids)
    print(f"calibration {time.time()-t0:.1f}s  KP_SCALE(mean)={kp_scale.mean().item():.1f} "
          f"KD_SCALE(mean)={kd_scale.mean().item():.3f}", flush=True)

    bufO = bufL = None
    metrics = []
    for r in range(args.rounds):
        beta = 1.0 if r == 0 else max(0.0, 1.0 - r / (args.rounds - 1))
        tr = time.time()
        O, L = rollout_and_label(env, teacher, student, beta, args.horizon, teacher_cols,
                                 act_perm, kp_scale, kd_scale, qids, n_musc)
        bufO = O if bufO is None else torch.cat([bufO, O])
        bufL = L if bufL is None else torch.cat([bufL, L])
        val = fit(student, bufO, bufL, args.epochs, dev)
        dt = time.time() - tr
        print(f"round {r} beta={beta:.2f} buf={bufO.shape[0]} val_mse={val:.4f} time={dt:.1f}s",
              flush=True)
        metrics.append({"round": r, "beta": beta, "buf": int(bufO.shape[0]), "val_mse": val,
                        "time_s": dt, "n_envs": args.n_envs, "horizon": args.horizon,
                        "epochs": args.epochs})

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
        "state_dict": student.state_dict(),
        "n_obs": n_obs,
        "names": names,
        "kp_scale": kp_scale.cpu(),
        "kd_scale": kd_scale.cpu(),
        "scaling": {"raw_eps": RAW_EPS, "scale_floor": SCALE_FLOOR,
                    "qdes_delta_clamp": QDES_DELTA_CLAMP,
                    "encoding": "q_des=raw; kp=softplus(raw)*KP_SCALE; kd=softplus(raw)*KD_SCALE"},
    }, f"{OUT_DIR}/pd_student.pt")
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    val0, valF = metrics[0]["val_mse"], metrics[-1]["val_mse"]
    print(f"SAVED pd_student.pt  val0={val0:.4f} valF={valF:.4f} ratio={valF/val0:.3f}", flush=True)


if __name__ == "__main__":
    main()
