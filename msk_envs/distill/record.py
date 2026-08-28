"""Task 5 — Record the distilled torque student's rollout for dashboard visualization.

Rolls out the DAgger-trained student in SprinterTorque env (muscles OFF, pure torque control)
and saves the trajectory for the Bolt renderer. Uses the EXACT student.apply convention from
evaluate.py (act_perm + muscles forced -1) so the render matches how the student was evaluated.

EXPECTATION: The student COLLAPSES (Task 4 = PHASE_A_FAIL), so the video will show a falling
figure, NOT a running gait. That is the correct, expected output — a diagnostic that visually
documents the failure and motivates Phase B (muscle-as-constraint). Do NOT "fix" the student.
"""
import torch

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorque
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.distill.student import Student, torque_to_excitation
from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm

STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_sprint/student.pt"

# SprinterTorque env requires reward_lambdas set, even though distill doesn't use rewards.
# Matches evaluate.py + dagger.py (Task 3 hit KeyError 'lambda_vel' without this).
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


def main():
    dev = torch.device("cuda")

    # Single env, visuals enabled for LoggedSim trajectory capture
    cfg = EnvConfigSprinterTorque()
    cfg.reward_lambdas = dict(BLANK_REWARD_LAMBDAS)
    env = EnvFactory.create_env(
        num_envs=1,
        env_config=cfg,
        requires_visuals=True,
        cuda_graph=True,
        device=dev
    )

    # Load the trained student (DAgger, 12 rounds)
    ck = torch.load(STUDENT_PT, map_location=dev)
    student = Student(ck["n_obs"], n_dof=25, device=dev)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    of = ck["optimal_force"].to(dev)

    # Build the actuator permutation: student outputs in `names` order (joint_dof_indices),
    # but env actuator slice is in actuator_id_lookup order. act_perm routes them correctly.
    names, _ = joint_dof_indices(env)
    act_perm = build_actuator_perm(env, names, dev)

    # LoggedSim wraps the env, captures frames at 30fps for the renderer
    recording_fps = 30.0
    sim = LoggedSim(env, dev, delta_t_log=1.0 / recording_fps)
    obs = sim.reset()

    # Rollout for full episode (or until termination)
    steps = int(round(env.max_episode_duration / env.delta_t))
    n_musc = env.num_muscles

    for _ in range(steps):
        a = env.get_blank_actions()

        # Student torque path: muscles OFF, torques permuted into actuator slice.
        # EXACT match to evaluate.py's student rollout (lines 118-123).
        with torch.no_grad():
            tau = student(obs)  # (1, 25) in `names` order
        exc = torque_to_excitation(tau, of) * 2.0 - 1.0  # convert to raw [-1,1]
        a[:, :n_musc] = -1.0                              # muscles OFF (raw -1 -> excitation 0)
        a[:, n_musc:] = exc.index_select(1, act_perm)    # route torques to actuator slice

        finished, obs = sim.step(a)
        if finished.all():
            break

    # Save trajectory for the Bolt renderer (json.gz format)
    # save_animation writes to dashboard/trajectories/<name>_<id>.json.gz
    sim.save_animation("dashboard/trajectories/dagger_student", "1", use_gzip=True)
    print("WROTE dashboard/trajectories/dagger_student_1.json.gz")


if __name__ == "__main__":
    main()
