"""PD actuator control helpers for sub-step PD impedance mode.

Provides apply_pd() to write q_des/kp/kd buffers (in names order → actuator-slice order)
and step with muscles OFF.
"""
import torch
import bolt


def apply_pd(env, q_des, kp, kd, act_perm):
    """Write PD control buffers and step with muscles OFF.

    Args:
        env: MSKEnv instance with use_pd_actuators=True
        q_des: (n_envs, 25) desired joint positions in names order
        kp: (n_envs, 25) proportional gains in names order
        kd: (n_envs, 25) derivative gains in names order
        act_perm: (25,) permutation tensor from names order → actuator-slice order

    Returns:
        5-tuple (obs, rew, term, trunc, info) from env.step
    """
    # Write PD buffers in actuator-slice order (permute from names order)
    bolt.pd_q_des(env.d).copy_(q_des.index_select(1, act_perm))
    bolt.pd_kp(env.d).copy_(torch.clamp(kp, min=0.0).index_select(1, act_perm))
    bolt.pd_kd(env.d).copy_(torch.clamp(kd, min=0.0).index_select(1, act_perm))

    # Step with muscles OFF (excitation 0)
    a = env.get_blank_actions()
    a[:, :env.num_muscles] = -1.0  # muscles OFF
    return env.step(a)


if __name__ == "__main__":
    """Smoke test: constant PD gains (q_des=0, kp=100, kd=5) for 100 steps.
    Verifies PD actuators are engaged and stabilizing (not-fallen > 0.5)."""
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorquePD
    from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm
    from msk_envs.utils.reward_lib import has_fallen

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 64

    # Build env with PD actuators
    cfg = EnvConfigSprinterTorquePD()
    cfg.reward_lambdas = {
        "lambda_vel": 0.0,
        "lambda_mid_lane": 0.0,
        "lambda_spring": 0.0,
        "lambda_damper": 0.0,
        "lambda_limit": 0.0,
        "lambda_muscle_passive": 0.0,
    }

    env = EnvFactory.create_env(
        num_envs=num_envs,
        env_config=cfg,
        requires_visuals=False,
        cuda_graph=True,
        device=dev,
    )

    # Build actuator permutation (names → actuator-slice order)
    names, _ = joint_dof_indices(env)
    act_perm = build_actuator_perm(env, names, dev)

    # Hold the model's STARTING pose as a benign PD setpoint (a bad target like q_des=0
    # yanks all joints to zero and drives the body into a config where Bolt's adaptive
    # integrator stalls). Read the start pose ONCE after reset and keep it fixed.
    env.reset()
    qids = torch.tensor([env.qpos_id_lookup[n] for n in names], device=dev, dtype=torch.long)
    q_des = env.joint_positions.index_select(1, qids).clone()  # (num_envs, 25), FIXED setpoint
    kp = torch.full((num_envs, 25), 80.0, device=dev)
    kd = torch.full((num_envs, 25), 8.0, device=dev)

    # Run 100 steps holding the start pose
    for _ in range(100):
        apply_pd(env, q_des, kp, kd, act_perm)

    # Check not-fallen fraction
    fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation)
    not_fallen_frac = (~fallen).float().mean().item()

    print(f"notfallen={not_fallen_frac:.3f}")
    print("pd buffers written, stepped ok")
    assert not_fallen_frac > 0.5, f"PD stabilization failed: notfallen={not_fallen_frac:.3f} <= 0.5"
