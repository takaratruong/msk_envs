"""SprinterPDRLEnv: RL environment for distilling muscle→torque with PD actuators.

This env exposes a 75-dim PD action space (q_des[25], kp[25], kd[25]) to the TD3 trainer.
The student learns to sprint using sub-step PD impedance control, optionally regularized
by an impedance reward term (lambda_impedance, populated in Task 2).

Action space (75-dim, range [-1,1]):
  - raw_qdes[25]:  desired-joint-position DELTA (q_des = q_now + QDES_DELTA_CLAMP * raw_qdes)
  - raw_kp[25]:    proportional gains (decoded via softplus * KP_SCALE)
  - raw_kd[25]:    derivative gains (decoded via softplus * KD_SCALE)

Reward: inherits base sprint terms (forward velocity, etc.) + rew_impedance placeholder (0.0).

q_des DELTA REPARAM (RL-T1 warm-start contingency)
--------------------------------------------------
The trainer actor (DeterministicPolicy, fasttd3) has a Tanh output head, so every action
component is bounded to [-1,1]. Interpreting raw_qdes as an ABSOLUTE joint angle (the original
dagger_pd convention) made joints whose neutral pose exceeds 1 rad UNREACHABLE by a Tanh actor,
which collapsed the warm-started balancer (verified: clamping absolute q_des to +-1 drops
notfallen 0.77 -> 0.38). We therefore interpret raw_qdes as a DELTA fraction:

    q_des = q_now + QDES_DELTA_CLAMP * clamp(raw_qdes, -1, 1)

Since the actor's raw_qdes is already in [-1,1], the effective delta lives in +-QDES_DELTA_CLAMP
(+-0.5 rad) around the current pose — the same integrator-stall guard the old absolute+delta-clamp
form provided, but now every reachable delta is representable by a Tanh head. kp/kd are unchanged
(softplus * per-joint scale from the warm-start student checkpoint). The warm-start bridge fits the
actor to StudentPD in THIS reparam space (see distill/warmstart_bridge.py).
"""
import os
import torch
import bolt

from .env_sprint import SprintingEnv
from .env_config import EnvConfig
from msk_envs.distill.dof_utils import joint_dof_indices, build_actuator_perm

# Avoid circular import: dagger_pd imports env_factory, which imports this module.
# decode/QDES_DELTA_CLAMP are imported LOCALLY inside _set_actions (single source of truth).


class SprinterPDRLEnv(SprintingEnv):
    """Sprinting env with 75-dim PD action space for RL training."""

    def __init__(
        self,
        num_envs: int,
        env_config: EnvConfig,
        device: torch.device,
        requires_visuals: bool,
        cuda_graph: bool,
    ):
        super().__init__(
            num_envs=num_envs,
            env_config=env_config,
            device=device,
            requires_visuals=requires_visuals,
            cuda_graph=cuda_graph,
        )

        # Load KP_SCALE/KD_SCALE from the warm-start student checkpoint
        # Use absolute path from project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        student_ckpt_path = os.path.join(project_root, "models/dagger_pd/pd_student.pt")
        if not os.path.exists(student_ckpt_path):
            raise FileNotFoundError(
                f"PD student checkpoint not found at {student_ckpt_path}. "
                f"Cannot load KP_SCALE/KD_SCALE. Run dagger_pd.py first or provide the checkpoint."
            )
        ckpt = torch.load(student_ckpt_path, map_location=device)
        self.kp_scale = ckpt["kp_scale"].to(device)  # (1, 25)
        self.kd_scale = ckpt["kd_scale"].to(device)  # (1, 25)

        # Build actuator permutation (names order → actuator-slice order)
        self.names, _ = joint_dof_indices(self)
        self.act_perm = build_actuator_perm(self, self.names, device)

        # qids for reading current joint positions
        self.qids = torch.tensor(
            [self.qpos_id_lookup[n] for n in self.names],
            device=device,
            dtype=torch.long,
        )

        # Buffer to store last applied 75-dim raw action for _get_actions
        self._last_raw_action = torch.zeros((num_envs, 75), device=device)

        # Store last kp/kd (physical, names order) for Task 2 impedance reward
        self._last_kp = torch.zeros((num_envs, 25), device=device)
        self._last_kd = torch.zeros((num_envs, 25), device=device)

    def _get_actions(self) -> torch.Tensor:
        """Return the 75-dim PD action buffer (defines num_actions == 75)."""
        return self._last_raw_action.detach().clone()

    def _set_actions(self, raw_action: torch.Tensor) -> None:
        """Decode 75-dim raw action and write PD buffers. Muscles OFF.

        CRITICAL: This function MUST ONLY write buffers. It must NOT call apply_pd or step the sim.
        env.step() calls pre_sim_step -> _set_actions -> launch_sim_step, so the sim step happens
        AFTER this function returns.
        """
        # Import decode + QDES_DELTA_CLAMP locally to avoid circular import
        # (single source of truth: dagger_pd)
        from msk_envs.distill.dagger_pd import decode, QDES_DELTA_CLAMP

        # Store for _get_actions
        self._last_raw_action.copy_(raw_action)

        # Decode kp/kd (softplus * scale). decode() also returns q_des from raw[:, :25], but under
        # the DELTA reparam we IGNORE that absolute value and re-derive q_des as a delta below.
        _, kp, kd = decode(raw_action, self.kp_scale, self.kd_scale)

        # q_des DELTA reparam (RL-T1 contingency): interpret raw_qdes as a delta fraction so a
        # Tanh-bounded actor can reach any pose. raw in [-1,1] -> delta in +-QDES_DELTA_CLAMP rad.
        raw_qdes = torch.clamp(raw_action[:, :25], -1.0, 1.0)
        q_now = self.joint_positions.index_select(1, self.qids)
        q_des = q_now + QDES_DELTA_CLAMP * raw_qdes

        # Store kp/kd for Task 2 impedance reward
        self._last_kp.copy_(kp)
        self._last_kd.copy_(kd)

        # Write PD buffers (names order → actuator-slice order via act_perm)
        # Reuse pd_control.apply_pd's buffer-writing logic (NOT the function itself, which would step).
        bolt.pd_q_des(self.d).copy_(q_des.index_select(1, self.act_perm))
        bolt.pd_kp(self.d).copy_(torch.clamp(kp, min=0.0).index_select(1, self.act_perm))
        bolt.pd_kd(self.d).copy_(torch.clamp(kd, min=0.0).index_select(1, self.act_perm))

        # Force muscles OFF (excitation = 0 → mapped to [0,1] range via base class, so raw = -1)
        self.muscle_excitations.zero_()

    def _compute_raw_reward_dict(self):
        """Compute sprint rewards + rew_impedance placeholder (0.0 until Task 2)."""
        # Inherit base sprint terms (rew_vel, rew_mid_lane, spring/damper/limit penalties)
        super()._compute_raw_reward_dict()

        # Add impedance reward placeholder (Task 2 will populate this with the actual loss)
        self.reward_dict["rew_impedance"] = torch.zeros(
            self.num_worlds, device=self.device
        )


if __name__ == "__main__":
    """Smoke test: build env, verify num_actions==75, step 50 random actions, print reward keys."""
    from msk_envs.envs.env_factory import EnvFactory
    from msk_envs.envs.env_config import EnvConfigSprinterTorquePD_RL
    from msk_envs.utils.reward_lib import has_fallen

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 64

    print(f"Building SprinterPDRLEnv on {device}...")
    cfg = EnvConfigSprinterTorquePD_RL()
    env = EnvFactory.create_env(
        num_envs=num_envs,
        env_config=cfg,
        requires_visuals=False,
        cuda_graph=True,
        device=device,
    )

    # AC1: num_actions == 75
    assert env.num_actions() == 75, f"num_actions={env.num_actions()}, expected 75"
    print(f"✓ AC1: num_actions == {env.num_actions()}")

    # AC2: random-action rollout (no stall, no exception)
    print("Stepping 50 random actions...")
    env.reset()
    for i in range(50):
        actions = env.get_random_actions()
        obs, rew, term, trunc, info = env.step(actions)
        # Check for NaN (would indicate integrator stall or failure)
        if torch.isnan(obs).any() or torch.isnan(rew).any():
            raise RuntimeError(f"NaN detected at step {i}: obs NaN={torch.isnan(obs).any()}, rew NaN={torch.isnan(rew).any()}")
    print(f"✓ AC2: 50 random steps completed without error or NaN")

    # AC3: reward dict includes sprint terms + rew_impedance
    raw_rewards = info["raw_rewards"]
    reward_keys = set(raw_rewards.keys())
    expected_keys = {"rew_vel", "rew_mid_lane", "rew_spring", "rew_damper", "rew_limit", "rew_muscle_passive", "rew_impedance"}
    assert expected_keys.issubset(reward_keys), f"Missing reward keys: {expected_keys - reward_keys}"
    print(f"✓ AC3: reward_dict keys = {sorted(reward_keys)}")

    # Check not-fallen fraction (should be >0 with random PD, since q_des is clamped)
    fallen = has_fallen(root_pos=env.root_pos, ground_rotation=env.ground_rotation)
    not_fallen_frac = (~fallen).float().mean().item()
    print(f"not_fallen_frac = {not_fallen_frac:.3f} (should be >0; q_des clamp keeps integrator alive)")

    print("\n✓ All acceptance criteria PASSED.")
