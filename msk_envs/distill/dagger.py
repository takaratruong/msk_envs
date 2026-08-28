"""DAgger loop: distill a trained muscle sprint policy (teacher) into a torque policy (student).

Each round:
  - Roll out `horizon` steps in a SprinterTorque env.
  - At every visited state, apply the teacher's muscle excitations and read the net joint
    torque it produces (ufrc_muscle sliced to the 25 non-root joint DoFs) as the label.
  - Advance the controlled trajectory with a beta coin flip: keep the teacher's resulting
    state with prob beta, else take a student torque step (on-student-distribution data).
  - Aggregate (obs, teacher_torque) pairs across rounds and refit the student by MSE.

beta schedule: round 0 -> beta=1 (pure teacher, seed the buffer on the teacher's trajectory),
then anneal beta -> 0 so later rounds collect data on the student's own state distribution
(the DAgger correction that fixes compounding drift).

--- Interface reconciliation (discovered during Task 3, differs from the task brief) ---
The teacher checkpoint is a *base* Sprinter policy: obs dim 336, action dim 138
(136 muscle excitations + 2 mtp-motor excitations). The SprinterTorque twin, however,
emits obs dim 359 and action dim 161 (136 muscles + 25 coordinate actuators). The only
obs difference is the actuator-activation block (2 mtp motors in base vs 25 actuators in
the twin). We therefore:
  * Build a teacher-obs adapter (359 -> 336): keep everything except the 25-actuator block,
    and re-insert just the two mtp-motor activation columns in the base env's order.
  * Feed only the teacher action's first 136 dims (muscles) into the muscle slice.
  * Permute the student's per-DoF torque output (joint_dof_indices / `names` order) into the
    env's actuator-slice order (actuator_id_lookup order) before writing excitations, so each
    torque lands on the correct joint. Labels, optimal_force and student columns all stay in
    `names` order; only the env write is permuted.
  * env.step returns (obs, rew, terminated, truncated, info).
"""
import os
import json
import argparse
import torch

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.distill.teacher import Teacher
from msk_envs.distill.student import Student, torque_to_excitation
from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm, build_teacher_obs_cols

CKPT = "/home/ubuntu/msk_envs/models/baseline_sprint_2026-08-27_20-59/baseline_sprint_2026-08-27_20-59_149000.pt"
OPTFORCE = "/home/ubuntu/msk_envs/msk_envs/msk_models/sprinter/sprinter_torque_optforce.json"
OUT_DIR = "/home/ubuntu/msk_envs/models/dagger_sprint"

# distill doesn't use rewards; the SprinterTorque config ships without reward lambdas.
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


def read_optimal_force(names, device):
    """Load per-DoF optimal_force, ordered to match `names` (Teacher.torque_label order)."""
    with open(OPTFORCE) as f:
        of_map = json.load(f)
    return torch.tensor([of_map[n] for n in names], device=device, dtype=torch.float32)


def rollout_and_label(env, teacher, student, of, beta, horizon, device, teacher_cols, act_perm):
    """Roll out horizon steps. Every step: apply teacher muscles to READ the torque label at
    the current state, record (obs, label). Then advance the controlled trajectory: with prob
    beta keep the teacher's resulting state, else take a student torque step."""
    n_musc = env.num_muscles
    obs = env.reset()
    O, T = [], []
    for _ in range(horizon):
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        obs_teacher, _, _, _, _ = env.step(a)
        O.append(obs.detach().clone())
        T.append(teacher.torque_label(env))
        if torch.rand(()) < beta:
            obs = obs_teacher
        else:
            with torch.no_grad():
                tau = student(obs)                       # (n, 25) in `names` order
            raw = torque_to_excitation(tau, of) * 2.0 - 1.0
            a2 = env.get_blank_actions()
            # Muscles OFF: raw -1 -> excitation 0. Pure torque control (no 50% muscle tone),
            # so this student can transfer to a muscle-less motor robot (Unitree G1) and Task 4's
            # fidelity comparison is clean. get_blank_actions() returns zeros (raw 0 -> exc 0.5).
            a2[:, :n_musc] = -1.0
            a2[:, n_musc:] = raw.index_select(1, act_perm)  # route each torque to its actuator
            obs, _, _, _, _ = env.step(a2)
    return torch.cat(O), torch.cat(T)


def fit(student, buf_O, buf_T, epochs, device):
    """80/20 train/val split (disjoint indices), Adam lr 1e-3, MSE, minibatches of 4096.
    Returns val MSE."""
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    n = buf_O.shape[0]
    idx = torch.randperm(n, device=device)
    n_val = n // 5
    vi, ti = idx[:n_val], idx[n_val:]
    for _ in range(epochs):
        perm = ti[torch.randperm(ti.shape[0], device=device)]
        for mb in perm.split(4096):
            opt.zero_grad()
            loss = ((student(buf_O[mb]) - buf_T[mb]) ** 2).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        val = ((student(buf_O[vi]) - buf_T[vi]) ** 2).mean().item()
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorque()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(num_envs=args.n_envs, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)
    names, _ = joint_dof_indices(env)
    teacher = Teacher(CKPT, env, dev)

    # Correctness: label order (joint_dof_indices) must match teacher.names and optimal_force order.
    assert names == teacher.names, "label order (joint_dof_indices) != teacher.names"
    of = read_optimal_force(names, dev)
    assert of.shape[0] == len(names) == 25, "optimal_force / names length mismatch"
    teacher_cols = build_teacher_obs_cols(env, dev)     # 359-obs -> 336-obs teacher adapter
    act_perm = build_actuator_perm(env, names, dev)     # student(names) -> actuator-slice order

    n_obs = env.reset().shape[1]
    student = Student(n_obs, n_dof=25, device=dev)

    bufO = bufT = None
    metrics = []
    for r in range(args.rounds):
        beta = 1.0 if r == 0 else max(0.0, 1.0 - r / (args.rounds - 1))
        O, T = rollout_and_label(env, teacher, student, of, beta, args.horizon, dev,
                                 teacher_cols, act_perm)
        bufO = O if bufO is None else torch.cat([bufO, O])
        bufT = T if bufT is None else torch.cat([bufT, T])
        val = fit(student, bufO, bufT, args.epochs, dev)
        print(f"round {r} beta={beta:.2f} buf={bufO.shape[0]} val_mse={val:.4f}")
        metrics.append({"round": r, "beta": beta, "buf": int(bufO.shape[0]), "val_mse": val,
                        "n_envs": args.n_envs, "horizon": args.horizon, "epochs": args.epochs})

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "n_obs": n_obs,
                "optimal_force": of.cpu(), "names": names}, f"{OUT_DIR}/student.pt")
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"SAVED student.pt  val0={metrics[0]['val_mse']:.4f} valF={metrics[-1]['val_mse']:.4f}")


if __name__ == "__main__":
    main()
