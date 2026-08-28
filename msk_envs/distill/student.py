import torch
import torch.nn as nn


class Student(nn.Module):
    def __init__(self, n_obs, n_dof=25, hidden=(256, 256), device="cuda"):
        super().__init__()
        dims = [n_obs, *hidden]
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b, device=device), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_dof, device=device)]
        self.net = nn.Sequential(*layers)
        self.n_dof = n_dof

    def forward(self, obs):
        return self.net(obs)  # (n_envs, 25) torque targets (N·m)


def torque_to_excitation(tau, optimal_force):
    """
    Convert torque targets to actuator excitations.
    Inverse of: torque = (activation - 0.5) * 2 * optimal_force
    where excitation == activation when activation_time_constant = 0.
    """
    return torch.clamp(tau / (2.0 * optimal_force) + 0.5, 0.0, 1.0)


def apply(env, tau, optimal_force, act_perm=None, base_action=None):
    """
    Write student torques into env actuator excitations and step, with muscles fully OFF.

    Pure torque control: the muscle slice is forced to raw -1 (excitation 0), NOT left at
    the blank default. get_blank_actions() returns zeros, and the env maps raw 0 -> excitation
    0.5, i.e. all muscles held at ~50% tone. That is wrong for a torque student: the goal is a
    policy that controls the body with joint torques ALONE so it can transfer to a muscle-less
    motor robot (Unitree G1). Leaning on 50% muscle tone would not transfer and would muddy the
    fidelity comparison. Setting the muscle slice to -1 makes muscles genuinely min-activation.

    Args:
        env: The environment instance
        tau: Torque targets tensor (n_envs, n_dof), in `names` (joint_dof_indices) order
        optimal_force: Optimal force per DoF (n_dof,), in the SAME `names` order
        act_perm: Optional LongTensor from dof_utils.build_actuator_perm routing the
            `names`-ordered vector into the env's actuator-slice order. REQUIRED for real
            use: the env actuator slice (actuator_id_lookup order) does NOT match `names`
            order, so without it torques land on the wrong joints. If None, the raw vector
            is written straight through (back-compat only; correct only if the two orders
            already coincide).
        base_action: Optional base action tensor to modify (n_envs, action_dim)

    Returns:
        Whatever env.step() returns.
    """
    a = base_action if base_action is not None else env.get_blank_actions()
    exc = torque_to_excitation(tau, optimal_force)  # [0,1]
    # env maps raw_action [-1,1] -> excitation (a+1)/2 for actuators; pre-invert to raw:
    raw = exc * 2.0 - 1.0
    if act_perm is not None:
        raw = raw.index_select(1, act_perm)  # route each torque to its actuator slot
    a[:, :env.num_muscles] = -1.0            # muscles OFF (raw -1 -> excitation 0)
    a[:, env.num_muscles:] = raw
    return env.step(a)


if __name__ == "__main__":
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorque

    dev = torch.device("cuda")

    # Roundtrip test
    of = torch.full((25,), 100.0, device=dev)
    tau = torch.randn(16, 25, device=dev) * 20.0
    exc = torque_to_excitation(tau, of)
    tau_rt = (exc - 0.5) * 2.0 * of
    valid = tau.abs() < of  # only unclamped entries round-trip
    err = (tau_rt[valid] - tau[valid]).abs().max().item()
    print(f"roundtrip max err {err:.2e}")

    # Apply test
    cfg = EnvConfigSprinterTorque()
    # Set dummy reward lambdas for smoke test (distill doesn't use rewards)
    cfg.reward_lambdas = {
        "lambda_vel": 0.0,
        "lambda_mid_lane": 0.0,
        "lambda_spring": 0.0,
        "lambda_damper": 0.0,
        "lambda_limit": 0.0,
        "lambda_muscle_passive": 0.0,
    }
    env = EnvFactory.create_env(
        num_envs=16,
        env_config=cfg,
        requires_visuals=False,
        cuda_graph=True,
        device=dev
    )
    from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm
    obs = env.reset()
    s = Student(obs.shape[1], device=dev)
    names, _ = joint_dof_indices(env)
    act_perm = build_actuator_perm(env, names, dev)  # names-order -> actuator-slice order
    apply(env, s(obs), of, act_perm=act_perm)
    print("applied step ok (with actuator permutation)")
