"""Task 4 — Rollout-fidelity evaluation (PRIMARY success criterion) + BC ablation.

Rolls out the distilled TORQUE student in the SprinterTorque env and measures how
faithfully it reproduces the muscle teacher's sprint gait.

--- Metric reconciliation (env layout differs from the task brief) ---
The brief assumed a LocomotionEnv with `_root_vel_xy()` + `command_vel`. In fact SprinterTorque
is a SprintingEnv (LanesEnv branch): a MAX-forward-speed sprint task with NO commanded velocity.
So the fidelity metrics are defined for the sprint task as:
  - upright_fraction : mean fraction of the episode each env stays UPRIGHT (not fallen) before its
                       first fall, via has_fallen(). NOTE we deliberately do NOT use the env's full
                       _get_terminated() (fallen OR out-of-lane OR not-facing-fwd): the lane/facing
                       conditions are far stricter than "upright" and the healthy teacher trips them
                       while never falling, so full-termination latching mis-scores the teacher at
                       ~0.27 vs its true ~1.0 not-fallen rate. See run_metrics docstring.
  - fwd_speed        : mean forward (FWD) root velocity over the alive portion.
  - speed_err        : mean |fwd_speed - target| over the alive portion, where `target` is the
                       TEACHER's own achieved mean sprint speed. This is the natural sprint-fidelity
                       target: "does the torque student sprint at the muscle teacher's speed?" The
                       teacher's own speed_err (its fluctuation about its mean) is the baseline the
                       gate's 1.5x is measured against.
  - drift            : mean |lateral (SIDE) root position| over the alive portion.

Three controllers are compared on identical metrics:
  - teacher : the base-Sprinter muscle policy (obs reduced 359->336 via the teacher adapter,
              excitations written to the muscle slice).
  - dagger  : the 12-round muscles-off torque student trained in Task 3.
  - bc      : a plain behavior-cloning ablation trained only on teacher-visited states
              (beta=1 always) — the DAgger-vs-BC control.

Student rollouts use the EXACT training-time apply convention (student.apply): muscles forced
raw -1 (excitation 0, pure torque control) and per-DoF torques permuted from `names` order into
the env actuator-slice order via act_perm.

PRIMARY GATE:  dagger upright_fraction >= 0.7  AND  dagger speed_err <= 1.5 * teacher speed_err
Printed as PHASE_A_PASS / PHASE_A_FAIL. A FAIL is a legitimate result (motivates Phase B).

env.step returns (obs, rew, terminated, truncated, info); terminated is (num_worlds,).
"""
import json

import torch

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.distill.teacher import Teacher
from msk_envs.distill import student as student_mod
from msk_envs.distill.student import Student
from msk_envs.distill.dof_utils import (
    joint_dof_indices, build_actuator_perm, build_teacher_obs_cols,
)
from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX
from msk_envs.utils.reward_lib import has_fallen

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_sprint/student.pt"
EVAL_JSON = "/home/ubuntu/msk_envs/models/dagger_sprint/eval.json"

# distill/eval doesn't use rewards; the SprinterTorque config ships without reward lambdas.
# Matches dagger.py exactly (Task 3 hit KeyError 'lambda_vel' without this).
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


def _fwd_speed(env):
    """World-frame forward (FWD) linear velocity of the pelvis root, per env."""
    return env.body_velocities[:, env.root_id, FWD_IDX + 3]


def run_metrics(env, controller, is_teacher, of=None, act_perm=None,
                teacher_cols=None, target_speed=None, steps=None):
    """Roll out a full episode over all envs and return sprint-fidelity metrics.

    controller: obs -> muscle excitations (teacher) or per-DoF torque targets (student).
      teacher : obs reduced to 336 via teacher_cols before the policy call; excitations
                written to the muscle slice.
      student : muscles forced OFF (raw -1) and torques permuted via act_perm into the
                actuator slice — identical to the training-time student.apply() convention.

    --- Why we step manually instead of env.step() ---
    env.step() -> rl_step() AUTO-RESETS any env whose _get_terminated() fires, and for the
    sprint task _get_terminated() = (fallen OR toes-out-of-lane OR pelvis-not-facing-forward).
    Two problems for fidelity metrics: (1) the lane/facing conditions are far stricter than
    "still upright" — the healthy teacher trips them constantly while never actually falling,
    so latching on the full termination scores the teacher ~0.27 upright, contradicting its
    true ~1.0 not-fallen rate; (2) after an auto-reset the env state is scrambled, so any state
    read post-step() is unreliable.

    So we drive the physics directly (pre_sim_step -> launch_sim_step, the teacher.py smoke-test
    path) with NO auto-reset, and define "upright" as NOT-FALLEN via has_fallen() on the live
    post-physics state. `upright_fraction` is then the mean fraction of the episode each env
    stays upright before its first fall — the "upright duration" the brief asks for and the
    semantic the coordinator's sanity check (teacher ~0.9+) uses. Once an env falls, the alive
    mask latches it off for the rest of the episode.

    target_speed: if given, speed_err = mean_alive |fwd_speed - target_speed|; else speed_err
                  is left at 0 (used for the teacher's first pass to measure its mean speed).
    """
    obs = env.reset()
    steps = steps or int(round(env.max_episode_duration / env.delta_t))
    n = env.num_worlds
    dev = obs.device
    upright = torch.zeros(n, device=dev)
    fwd_sum = torch.zeros(n, device=dev)
    speed_err = torch.zeros(n, device=dev)
    drift = torch.zeros(n, device=dev)
    alive = torch.ones(n, device=dev)
    n_musc = env.num_muscles

    for _ in range(steps):
        a = env.get_blank_actions()
        if is_teacher:
            exc = controller(obs.index_select(1, teacher_cols))[:, :n_musc]
            a[:, :n_musc] = exc
        else:
            with torch.no_grad():
                tau = controller(obs)                      # (n, 25) in `names` order
            # Match student.apply exactly: muscles OFF, torque->excitation, permute to slice.
            exc = student_mod.torque_to_excitation(tau, of) * 2.0 - 1.0
            a[:, :n_musc] = -1.0                           # muscles OFF (raw -1 -> excitation 0)
            a[:, n_musc:] = exc.index_select(1, act_perm)  # route each torque to its actuator

        # Manual step, NO auto-reset (see docstring). Mirrors teacher.py's smoke path.
        env.pre_sim_step(a)
        env.launch_sim_step()
        env.update_metrics()

        # Latch on falling only (true "upright"), on the live pre-reset state.
        fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation).float()
        alive = alive * (1.0 - fallen)
        upright += alive
        v = _fwd_speed(env)
        fwd_sum += v * alive
        if target_speed is not None:
            speed_err += (v - target_speed).abs() * alive
        drift += env.root_pos[:, SIDE_IDX].abs() * alive

        obs = env._get_obs()

    denom = upright.clamp(min=1)
    return {
        "upright_fraction": (upright / steps).mean().item(),
        "fwd_speed": (fwd_sum / denom).mean().item(),
        "speed_err": (speed_err / denom).mean().item(),
        "drift": (drift / denom).mean().item(),
    }


def load_student(path, dev):
    """Load the trained DAgger student and its per-DoF optimal_force (in `names` order)."""
    ck = torch.load(path, map_location=dev)
    s = Student(ck["n_obs"], n_dof=25, device=dev)
    s.load_state_dict(ck["state_dict"])
    s.eval()
    return s, ck["optimal_force"].to(dev)


def train_bc(env, teacher, n_obs, dev, teacher_cols, n_batches=25, horizon=128):
    """BC ablation: collect teacher-visited states only (beta=1 always), fit a student once.

    The DAgger-vs-BC control. Labels are the teacher's net joint torque (torque_label), same as
    DAgger. The trajectory is advanced purely by the teacher's muscle excitations (no student
    rollout), so this student never sees its own state distribution — the thing DAgger corrects.
    """
    s = Student(n_obs, 25, device=dev)
    opt = torch.optim.Adam(s.parameters(), lr=1e-3)
    n_musc = env.num_muscles
    O, T = [], []
    obs = env.reset()
    for _ in range(horizon):
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        O.append(obs.detach().clone())
        _, _, _, _, _ = env.step(a)
        T.append(teacher.torque_label(env))
        obs = env._get_obs()
    O = torch.cat(O)
    T = torch.cat(T)
    for _ in range(n_batches):
        for mb in torch.randperm(O.shape[0], device=dev).split(4096):
            opt.zero_grad()
            (((s(O[mb]) - T[mb]) ** 2).mean()).backward()
            opt.step()
    s.eval()
    return s


def main():
    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorque()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=256, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)

    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "label order (joint_dof_indices) != teacher.names"
    teacher_cols = build_teacher_obs_cols(env, dev)   # 359-obs -> 336-obs teacher adapter (shared)
    act_perm = build_actuator_perm(env, names, dev)   # student(names) -> actuator-slice order
    n_obs = env.reset().shape[1]
    steps = int(round(env.max_episode_duration / env.delta_t))

    # Pass 1: teacher rollout to establish the reference sprint speed (the fidelity target).
    tref = run_metrics(env, teacher.action, True, teacher_cols=teacher_cols)
    target_speed = tref["fwd_speed"]

    # Pass 2: re-measure all controllers against the teacher's mean sprint speed.
    m_teacher = run_metrics(env, teacher.action, True, teacher_cols=teacher_cols,
                            target_speed=target_speed)

    student, of = load_student(STUDENT_PT, dev)
    assert of.shape[0] == 25, "optimal_force length mismatch"
    m_dagger = run_metrics(env, student, False, of=of, act_perm=act_perm,
                           target_speed=target_speed)

    bc = train_bc(env, teacher, n_obs, dev, teacher_cols)
    m_bc = run_metrics(env, bc, False, of=of, act_perm=act_perm,
                       target_speed=target_speed)

    print(f"n_envs={env.num_worlds} steps={steps} "
          f"teacher_ref_speed={target_speed:.3f} m/s")
    for name, m in [("teacher", m_teacher), ("dagger", m_dagger), ("bc", m_bc)]:
        print(f"{name:8s} upright={m['upright_fraction']:.3f} "
              f"speed_err={m['speed_err']:.3f} drift={m['drift']:.3f} "
              f"(fwd_speed={m['fwd_speed']:.3f})")

    ok = (m_dagger["upright_fraction"] >= 0.7
          and m_dagger["speed_err"] <= 1.5 * m_teacher["speed_err"])
    print("PHASE_A_PASS" if ok else "PHASE_A_FAIL")

    with open(EVAL_JSON, "w") as f:
        json.dump({"teacher": m_teacher, "dagger": m_dagger, "bc": m_bc,
                   "pass": bool(ok),
                   "teacher_ref_speed": target_speed,
                   "n_envs": env.num_worlds, "steps": steps,
                   "gate": {"upright_min": 0.7, "speed_err_max_mult": 1.5}}, f, indent=2)
    print(f"WROTE {EVAL_JSON}")


if __name__ == "__main__":
    main()
