"""Task 5 — Record the distilled PD-impedance student's rollout for dashboard visualization.

Rolls out the DAgger-trained PD student in SprinterTorquePD env (muscles OFF, PD actuators ON)
and saves the trajectory for the Bolt renderer. Uses the EXACT student.apply convention from
evaluate_pd.py (decode+clamp+apply_pd path) so the render matches how the student was evaluated.

EXPECTATION: The student STAYS UPRIGHT (Task 4 = not_fallen 0.997, matching the teacher) but
does NOT sprint (fwd -0.09 m/s vs teacher +5.12). The video will show a figure that is stable/
balanced but doesn't run forward — a visual demonstration that Phase B fixes Phase A's collapse
but does NOT yet reproduce the sprint gait.
"""
import torch

import bolt

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.distill.dagger_pd import StudentPD, decode, QDES_DELTA_CLAMP
from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm

STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_pd/pd_student.pt"

# SprinterTorquePD env requires reward_lambdas set, even though distill doesn't use rewards.
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


def main():
    dev = torch.device("cuda")

    # Single env, visuals enabled for LoggedSim trajectory capture
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(
        num_envs=1,
        env_config=cfg,
        requires_visuals=True,
        cuda_graph=True,
        device=dev
    )

    # Load the trained PD student (DAgger, 8 rounds)
    ck = torch.load(STUDENT_PT, map_location=dev)
    student = StudentPD(ck["n_obs"], n_dof=25, device=dev)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    kp_scale = ck["kp_scale"].to(dev)
    kd_scale = ck["kd_scale"].to(dev)

    # Build the actuator permutation: student outputs in `names` order (joint_dof_indices),
    # but env actuator slice is in actuator_id_lookup order. act_perm routes them correctly.
    names, _ = joint_dof_indices(env)
    act_perm = build_actuator_perm(env, names, dev)
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)

    # LoggedSim wraps the env, captures frames at 30fps for the renderer
    recording_fps = 30.0
    sim = LoggedSim(env, dev, delta_t_log=1.0 / recording_fps)
    obs = sim.reset()

    # Rollout for full episode (or until termination)
    steps = int(round(env.max_episode_duration / env.delta_t))
    n_musc = env.num_muscles

    print(f"Recording PD student rollout: num_envs=1, steps={steps}, n_musc={n_musc}", flush=True)

    for _ in range(steps):
        # Student PD path: replicate EXACTLY the evaluate_pd.py / dagger_pd.py rollout.
        # student(obs) -> raw -> decode -> (q_des_s, kp, kd); q_des = q_now + clamp(delta, +/-0.5);
        # then apply_pd writes the PD buffers and advances with muscles OFF.
        q_now = env.joint_positions.index_select(1, qids)

        with torch.no_grad():
            raw = student(obs)
        q_des_s, kp_s, kd_s = decode(raw, kp_scale, kd_scale)

        # Clamp q_des to current_q +/- QDES_DELTA_CLAMP (replicates dagger_pd rollout).
        delta = torch.clamp(q_des_s - q_now, -QDES_DELTA_CLAMP, QDES_DELTA_CLAMP)
        q_des_s = q_now + delta

        # Write PD buffers in actuator-slice order (apply_pd convention), muscles OFF.
        bolt.pd_q_des(env.d).copy_(q_des_s.index_select(1, act_perm))
        bolt.pd_kp(env.d).copy_(torch.clamp(kp_s, min=0.0).index_select(1, act_perm))
        bolt.pd_kd(env.d).copy_(torch.clamp(kd_s, min=0.0).index_select(1, act_perm))

        a = env.get_blank_actions()
        a[:, :n_musc] = -1.0  # muscles OFF (raw -1 -> excitation 0)

        finished, obs = sim.step(a)
        if finished.all():
            break

    # Save trajectory for the Bolt renderer (json.gz format)
    # save_animation writes to dashboard/trajectories/<folder>/<base>_<world_idx>.json.gz
    sim.save_animation("dashboard/trajectories/pd_student", "1", use_gzip=True)
    print("WROTE dashboard/trajectories/pd_student/1_0.json.gz")


if __name__ == "__main__":
    main()
