"""Diagnostic probe: is the distilled torque student's collapse caused by MISSING JOINT
FEEDBACK (the ActivationCoordinateActuator is pure open-loop torque, no kp/kd) rather than
bad torque prediction?

Reuses evaluate.py's exact env/teacher/student construction and its manual step loop
(pre_sim_step -> launch_sim_step -> update_metrics -> has_fallen -> _get_terminated ->
_perform_reset -> _get_obs), which runs without crashing. All joint state read off the live
env is cloned immediately.

Modes:
  teacher      muscle teacher (reference ceiling); also RECORDS per-step joint angles
               q_ref (STEPS, N, 25) in `names` order.
  pd_track     muscles OFF, tau = clip(kp*(q_ref[t]-q) - kd*qd, -of, of), routed via act_perm.
               Sweep (kp,kd) in [(100,5),(300,15),(600,30)].  Replays the teacher trajectory,
               so we set cfg.apply_start_noise=False for BOTH teacher-record and pd-replay so
               they share the reset distribution (tracking is still only approximate because
               mid-episode resets fire at different steps).
  student_ff   distilled student, feedforward only (SM.apply convention). Sanity: known collapse.
  student_damp student feedforward torque + velocity damping: tau = clip(student(obs) - kd*qd,
               -of, +of), muscles off. Sweep kd in [2, 10].

Prints a "RESULT" line per mode with notfallen (=upright_fraction) and fwd (mean forward speed).
"""
import torch

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.distill.teacher import Teacher
from msk_envs.distill import student as SM
from msk_envs.distill.dof_utils import (
    joint_dof_indices, build_actuator_perm, build_teacher_obs_cols,
)
from msk_envs.utils.global_params import FWD_IDX, SIDE_IDX
from msk_envs.utils.reward_lib import has_fallen
from msk_envs.distill.evaluate import (
    CKPT, STUDENT_PT, BLANK_REWARD_LAMBDAS, load_student, _fwd_speed,
)


def _clip_pm(tau, of):
    """Clamp tau elementwise into [-of, +of] where of is a (25,) per-DoF tensor (names order)."""
    return torch.maximum(torch.minimum(tau, of), -of)


def rollout(env, action_fn, steps, teacher_cols, target_speed=None, record_q=None, qids=None):
    """Manual-step rollout mirroring evaluate.run_metrics. action_fn(obs, t) -> full action.

    record_q: optional (steps, n, 25) tensor filled with joint_positions[:, qids] each step
              (post-physics, pre-reset), in `names` order.
    Returns dict with upright_fraction ('notfallen'), fwd_speed, speed_err, drift.
    """
    obs = env.reset()
    n = env.num_worlds
    dev = obs.device
    upright = torch.zeros(n, device=dev)
    fwd_sum = torch.zeros(n, device=dev)
    speed_err = torch.zeros(n, device=dev)
    drift = torch.zeros(n, device=dev)
    alive = torch.ones(n, device=dev)

    for t in range(steps):
        a = action_fn(obs, t)
        env.pre_sim_step(a)
        env.launch_sim_step()
        env.update_metrics()

        fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation).float()
        alive = alive * (1.0 - fallen)
        upright += alive
        v = _fwd_speed(env)
        fwd_sum += v * alive
        if target_speed is not None:
            speed_err += (v - target_speed).abs() * alive
        drift += env.root_pos[:, SIDE_IDX].abs() * alive
        if record_q is not None:
            record_q[t] = env.joint_positions[:, qids].clone()

        term = env._get_terminated()
        done = term.clamp(0.0, 1.0).unsqueeze(-1)
        if done.any():
            env._perform_reset(done)
        obs = env._get_obs()

    denom = upright.clamp(min=1)
    return {
        "notfallen": (upright / steps).mean().item(),
        "fwd_speed": (fwd_sum / denom).mean().item(),
        "speed_err": (speed_err / denom).mean().item(),
        "drift": (drift / denom).mean().item(),
    }


def main():
    dev = torch.device("cuda")
    cfg = EnvConfigSprinterTorque()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    cfg.apply_start_noise = False  # teacher-record & pd-replay share reset distribution
    env = EnvFactory.create_env(num_envs=64, env_config=cfg,
                                requires_visuals=False, cuda_graph=True, device=dev)

    names, idx = joint_dof_indices(env)
    n_musc = env.num_muscles
    teacher = Teacher(CKPT, env, dev)
    assert names == teacher.names, "label order != teacher.names"
    teacher_cols = build_teacher_obs_cols(env, dev)
    act_perm = build_actuator_perm(env, names, dev)

    # position indices (width-32, has quaternion) differ from velocity/torque indices (width-31)
    qids = torch.tensor([env.qpos_id_lookup[nm] for nm in names], device=dev, dtype=torch.long)
    dids = torch.tensor([env.dof_id_lookup[nm] for nm in names], device=dev, dtype=torch.long)

    student, of = load_student(STUDENT_PT, dev)  # of: (25,) names order
    assert of.shape[0] == 25

    steps = int(round(env.max_episode_duration / env.delta_t))
    steps = min(steps, 250)
    print(f"n_envs={env.num_worlds} steps={steps} apply_start_noise={cfg.apply_start_noise}",
          flush=True)

    # ---- 1. teacher (reference ceiling) + record q_ref ----
    q_ref = torch.zeros(steps, env.num_worlds, 25, device=dev)

    def teacher_fn(obs, t):
        a = env.get_blank_actions()
        a[:, :n_musc] = teacher.action(obs.index_select(1, teacher_cols))[:, :n_musc]
        return a

    m_teacher = rollout(env, teacher_fn, steps, teacher_cols, record_q=q_ref, qids=qids)
    target_speed = m_teacher["fwd_speed"]
    print(f"RESULT mode=teacher notfallen={m_teacher['notfallen']:.3f} "
          f"fwd={m_teacher['fwd_speed']:.3f} speed_err={m_teacher['speed_err']:.3f} "
          f"drift={m_teacher['drift']:.3f}", flush=True)

    # ---- 2. pd_track: muscles OFF, PD toward recorded teacher trajectory ----
    def make_pd_fn(kp, kd):
        def pd_fn(obs, t):
            q = env.joint_positions.index_select(1, qids).clone()
            qd = env.joint_velocities.index_select(1, dids).clone()
            tau = _clip_pm(kp * (q_ref[t] - q) - kd * qd, of)
            exc = SM.torque_to_excitation(tau, of) * 2.0 - 1.0
            a = env.get_blank_actions()
            a[:, :n_musc] = -1.0
            a[:, n_musc:] = exc.index_select(1, act_perm)
            return a
        return pd_fn

    for kp, kd in [(100, 5), (300, 15), (600, 30)]:
        m = rollout(env, make_pd_fn(kp, kd), steps, teacher_cols, target_speed=target_speed)
        print(f"RESULT mode=pd_track kp={kp} kd={kd} notfallen={m['notfallen']:.3f} "
              f"fwd={m['fwd_speed']:.3f} speed_err={m['speed_err']:.3f} "
              f"drift={m['drift']:.3f}", flush=True)

    # ---- 3. student_ff: distilled student, feedforward only ----
    def student_ff_fn(obs, t):
        with torch.no_grad():
            tau = student(obs)
        exc = SM.torque_to_excitation(tau, of) * 2.0 - 1.0
        a = env.get_blank_actions()
        a[:, :n_musc] = -1.0
        a[:, n_musc:] = exc.index_select(1, act_perm)
        return a

    m = rollout(env, student_ff_fn, steps, teacher_cols, target_speed=target_speed)
    print(f"RESULT mode=student_ff notfallen={m['notfallen']:.3f} fwd={m['fwd_speed']:.3f} "
          f"speed_err={m['speed_err']:.3f} drift={m['drift']:.3f}", flush=True)

    # ---- 4. student_damp: student feedforward torque + velocity damping ----
    def make_damp_fn(kd):
        def damp_fn(obs, t):
            with torch.no_grad():
                tau_ff = student(obs)
            qd = env.joint_velocities.index_select(1, dids).clone()
            tau = _clip_pm(tau_ff - kd * qd, of)
            exc = SM.torque_to_excitation(tau, of) * 2.0 - 1.0
            a = env.get_blank_actions()
            a[:, :n_musc] = -1.0
            a[:, n_musc:] = exc.index_select(1, act_perm)
            return a
        return damp_fn

    for kd in [2, 10]:
        m = rollout(env, make_damp_fn(kd), steps, teacher_cols, target_speed=target_speed)
        print(f"RESULT mode=student_damp kd={kd} notfallen={m['notfallen']:.3f} "
              f"fwd={m['fwd_speed']:.3f} speed_err={m['speed_err']:.3f} "
              f"drift={m['drift']:.3f}", flush=True)


if __name__ == "__main__":
    main()
