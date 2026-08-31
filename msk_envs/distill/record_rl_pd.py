"""Task RL-T5 — Record the RL-trained PD-impedance student's rollout for dashboard visualization.

Rolls out the RL-trained PD-gain policy (fasttd3 DeterministicPolicy actor) in SprinterTorquePD env
(muscles OFF, PD actuators ON) and saves the trajectory for the Bolt renderer. Uses the EXACT
actor decode+apply convention from evaluate_rl_pd.py so the render matches how the policy was
trained and evaluated.

INTERIM EXPECTATION (iter 8000-9000): The RL policy STAYS UPRIGHT (not_fallen 1.000, fixes Phase A
collapse) but does NOT yet sprint (fwd 0.89 m/s vs teacher 4.91); compliance rms_log 2.15 (kp
teacher-like, distal-joint kd gap). The video will show a figure that is stable/balanced but doesn't
run forward — a visual demonstration that Phase B + RL fixes collapse and achieves muscle-compliant
gains, but does not yet reproduce forward locomotion at this interim checkpoint. Training is ongoing
toward ~150k iters.
"""
import os
import glob
import re
import torch

import bolt

from msk_envs.envs.env_factory import EnvFactory
from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
from msk_envs.utils.logged_sim import LoggedSim
from msk_envs.distill.dagger_pd import decode, QDES_DELTA_CLAMP
from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm
from msk_envs.distill.evaluate_rl_pd import load_rl_actor

RUN_DIR = "/home/ubuntu/msk_envs/models/rl_pd_2026-08-31_21-54"
STUDENT_PT = "/home/ubuntu/msk_envs/models/dagger_pd/pd_student.pt"

# SprinterTorquePD env requires reward_lambdas set, even though distill doesn't use rewards.
BLANK_REWARD_LAMBDAS = {
    "lambda_vel": 0.0, "lambda_mid_lane": 0.0, "lambda_spring": 0.0,
    "lambda_damper": 0.0, "lambda_limit": 0.0, "lambda_muscle_passive": 0.0,
}


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


def main():
    # Use iter 8000 (the evaluated checkpoint from T4) instead of latest, to ensure
    # we render the exact policy that was evaluated and reported.
    ckpt_path = "/home/ubuntu/msk_envs/models/rl_pd_2026-08-31_21-54/rl_pd_2026-08-31_21-54_8000.pt"
    it = _ckpt_iter(ckpt_path)
    print(f"Recording RL-PD student @ iter={it}  ckpt={ckpt_path}", flush=True)

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

    # Load kp_scale/kd_scale from the Phase-B pd_student.pt (SAME file the RL env loads).
    ck = torch.load(STUDENT_PT, map_location=dev)
    kp_scale = ck["kp_scale"].to(dev)
    kd_scale = ck["kd_scale"].to(dev)

    # Load the RL actor + obs_normalizer (handles strict=False noise-buffer skip).
    obs_dim = env.reset().shape[1]
    actor, obs_norm = load_rl_actor(ckpt_path, n_obs=obs_dim, n_act=75, num_envs=1, dev=dev)

    # Build the actuator permutation: student outputs in `names` order (joint_dof_indices),
    # but env actuator slice is in actuator_id_lookup order. act_perm routes them correctly.
    names, _ = joint_dof_indices(env)
    act_perm = build_actuator_perm(env, names, dev)
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)

    # LoggedSim wraps the env, captures frames at 30fps for the renderer
    recording_fps = 30.0
    sim = LoggedSim(env, dev, delta_t_log=1.0 / recording_fps)

    # Use seed 0 to match evaluate_rl_pd
    torch.manual_seed(0)
    obs = sim.reset()
    print(f"Recording with seed=0, num_envs=1 (eval showed not_fallen=1.000)", flush=True)

    # Rollout for full episode (or until termination)
    steps = int(round(env.max_episode_duration / env.delta_t))
    n_musc = env.num_muscles

    print(f"Recording RL-PD rollout: num_envs=1, steps={steps}, n_musc={n_musc}", flush=True)

    for step_i in range(steps):
        # RL-PD path: replicate EXACTLY the evaluate_rl_pd.py / env_pd_rl rollout.
        # raw = actor(normalize_obs(obs)) -> decode -> (q_des_delta, kp, kd); q_des = q_now + delta;
        # then write the PD buffers and advance with muscles OFF.
        if obs is None:
            # LoggedSim returns None once all envs are finished; stop rollout
            print(f"Episode finished at step {step_i}/{steps} (env signaled termination)", flush=True)
            break

        q_now = env.joint_positions.index_select(1, qids)

        with torch.no_grad():
            norm = obs_norm(obs) if obs_norm is not None else obs
            raw = actor(norm)  # (1,75), Tanh [-1,1]

        # Decode EXACTLY as evaluate_rl_pd / env_pd_rl._set_actions
        _, kp, kd = decode(raw, kp_scale, kd_scale)
        raw_qdes = torch.clamp(raw[:, :25], -1.0, 1.0)
        q_des = q_now + QDES_DELTA_CLAMP * raw_qdes  # DELTA reparam (NOT absolute q_des)

        # Write PD buffers in actuator-slice order (permuted), muscles OFF.
        bolt.pd_q_des(env.d).copy_(q_des.index_select(1, act_perm))
        bolt.pd_kp(env.d).copy_(torch.clamp(kp, min=0.0).index_select(1, act_perm))
        bolt.pd_kd(env.d).copy_(torch.clamp(kd, min=0.0).index_select(1, act_perm))

        a = env.get_blank_actions()
        a[:, :n_musc] = -1.0  # muscles OFF (raw -1 -> excitation 0)

        finished, obs = sim.step(a)
        # Unlatch finished so LoggedSim doesn't stop producing obs (matches evaluate_rl_pd's manual
        # loop which resets fallen envs and continues without breaking)
        sim.finished[:] = 0

    # Save trajectory for the Bolt renderer (json.gz format)
    # save_animation writes to dashboard/trajectories/<folder>/<base>_<world_idx>.json.gz
    sim.save_animation("dashboard/trajectories/rl_pd_student", "1", use_gzip=True)
    print(f"WROTE dashboard/trajectories/rl_pd_student/1_0.json.gz (iter {it})")


if __name__ == "__main__":
    main()
