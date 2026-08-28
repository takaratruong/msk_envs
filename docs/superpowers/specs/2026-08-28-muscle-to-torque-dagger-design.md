# Muscle → Torque Distillation via DAgger — Design

**Date:** 2026-08-28
**Status:** Design — awaiting review
**Scope:** Phase A of a larger arc. Phases B and C are sketched but out of scope for implementation.

## Goal

The real Unitree G1 is torque/motor actuated. We want good, natural motion on a
**torque-actuated** robot, and we want to use a **muscle-actuated model as a naturalness
prior** — never running muscles on hardware, only distilling their motion into a torque policy.

Phase A tests the core hypothesis on infrastructure that already works:

> Can a torque-driven policy reproduce a trained muscle policy's motion, learned by
> DAgger against the muscle model, well enough to stay upright and track the gait?

If yes, the muscle prior transfers to torque control and later phases (co-learning,
muscle-as-constraint, full G1 tree) are worth the cost. If motion is reproduced but
looks less natural, that degradation is the motivation for Phase B — not a failure.

## Why this is feasible in the current sim (verified)

- Bolt exposes **`ufrc_muscle` (nworld, nv)** — the net generalized force muscles apply per
  DoF, computed as Σ(muscle force × moment arm) then mapped to generalized-speed space.
  For **revolute joints this equals joint torque in N·m** (the mobilizer transform is identity
  for hinges; it only differs for the translational floating-base DoFs). Verified in
  `bolt/_src/smooth_frc.py` (`qfrc_to_ufrc`) and `bolt/_src/types.py` docstrings.
- Bolt supports **`ActivationCoordinateActuator`** (torque actuators with activation dynamics)
  — see `bolt/load_utils/actuator_helper.py`. So a torque twin is a first-class model, not a hack.
  Its torque map (verified in `bolt/_src/smooth_actuator.py`) is
  **`actuation (N·m) = (activation − 0.5) × 2 × optimal_force`**, with excitation ∈ [0,1] driving
  activation through a first-order lag (`activation_time_constant`). Two consequences:
  - **`optimal_force` per joint is the torque ceiling** (±optimal_force). It MUST be sized from
    the teacher's observed `ufrc_muscle` range per DoF (e.g. ~1.2× the max |torque| seen across
    teacher rollouts), else the student cannot physically match the teacher on high-torque joints.
    Sizing is a step in the authoring script (run teacher rollouts first, collect per-DoF max |τ|).
  - **Activation lag means applied torque lags the command.** The student outputs a *target*
    torque; we invert the map to the excitation to write (`excitation = τ_target/(2·optimal_force) + 0.5`,
    clamped to [0,1]). Set `activation_time_constant = 0` (instantaneous) for Phase A so applied
    torque = commanded torque and the DAgger target bookkeeping is unambiguous; revisit lag in Phase B.
- The env action space is already **`[muscle_excitations, actuator_excitations]`**
  (`env_base.py::_get_actions`), so a model can be driven by muscles OR coordinate-actuators
  through the same interface — no new sim plumbing to apply student torques.

## Teacher: the trained sprint checkpoint

We already have a fully trained sprint policy:
`models/baseline_sprint_2026-08-27_20-59/..._149000.pt`, running the **sprinter model
(136 muscles, 31 coordinates)**. This gives Phase A a real expert on day one — and because
the sprinter is full-body, the distillation **covers arms + torso for free**.

- 31 coordinates = 6 floating-base (pelvis) + **25 actuated joint DoFs**:
  legs (hip 3-DoF, knee, ankle, subtalar, mtp ×2), torso (3), shoulders (3 ×2), elbows (×2).
- Student action space = **25 torques** = `ufrc_muscle` sliced to the 25 non-root DoFs.
  The 6 pelvis root DoFs are dropped: they are the unactuated floating base (driven by
  ground contact), which is physically correct — a real robot's base is not a motor.

## Architecture — the DAgger loop

Teacher and student **share one skeleton and one joint space**. The teacher drives it via
muscles; the student drives the *same* skeleton via coordinate-actuators. A torque vector
means the same thing to both.

1. **Teacher rollout:** the trained muscle policy runs the muscle (sprinter) model → natural motion (the expert).
2. **Student rollout (the DAgger part):** the student `π_θ: obs → τ ∈ ℝ²⁵` drives the torque
   twin. At each student-visited state we query the teacher for the torque it *would* produce:
   the teacher's native output is muscle excitations, so we use the **muscle model itself as the
   action translator** — set the muscle env to the student-visited state, apply the teacher's
   activations, step muscle dynamics, and read **`ufrc_muscle`** as the torque label.
3. **Aggregate & retrain:** append `(obs, τ_teacher)` to the buffer; refit `π_θ` by MSE.
   Because labels are collected on **student-visited** states, this fixes the compounding
   drift that plain BC suffers. A β-schedule (more teacher control early, annealed to student)
   bootstraps round 0.

**Key subtlety (the piece to test hardest):** labeling requires evaluating the teacher at
student-visited states. Two envs (muscle env for teacher, torque-twin env for student) are
kept in lockstep: each round, set the muscle env's qpos/qv to the student's visited states,
step muscle dynamics, read `ufrc_muscle`.

## Components & files

All new work is additive in a new `msk_envs/msk_envs/distill/` module. Existing envs, the
muscle model, and training code are untouched.

**1. Teacher wrapper — `distill/teacher.py`**
- Loads the muscle policy checkpoint; given env states, returns muscle excitations, and after
  the env applies them reads `ufrc_muscle[:, joint_dofs]` (25 DoFs) as the torque label.
- Interface: `Teacher.label(muscle_env, obs) -> (muscle_excitation, tau_teacher ∈ ℝ²⁵)`.

**2. Student policy — `distill/student.py`**
- Small MLP `obs → τ ∈ ℝ²⁵`, applied through the env's existing `actuator_excitations`
  channel, scaled to each joint's torque range. `.save/.load`.

**3. Torque-twin model — `bolt/add_sprinter_torque_actuators.py`**
- One-time authoring script (mirrors `author_g1_muscle.py`): load `sprinter_model.osim`, add
  25 `ActivationCoordinateActuator`s (one per joint DoF, `activation_time_constant=0` for Phase A),
  write `sprinter_torque.osim`. Same skeleton, same joint indices, muscles removed/ignored for
  the student env.
- **`optimal_force` sizing:** run the teacher (muscle sprinter) for a batch of rollouts, collect
  per-DoF max |`ufrc_muscle`|, set each actuator's `optimal_force` to ~1.2× that. So this script
  runs after a short teacher-rollout data-collection pass, not blindly.

**4. DAgger loop — `distill/dagger.py`**
- Orchestrator: `run_dagger(cfg) -> trained student + metrics log`. Per round: student rollout
  in torque env → collect visited states → teacher labels via muscle env → aggregate → MSE refit.

**5. Config — extend `msk_envs/msk_envs/envs/env_config.py`**
- Add `EnvConfigSprinterTorque` (sprinter skeleton + 25 coordinate-actuators, no muscle
  actions) + subcommand `sprintertorque`. Additive, like the `g1musclefn` config already added.

## Evaluation / success criteria

**Primary success bar — rollout fidelity (the honest test):** run the student (torque twin)
from the same start states and measure whether it **reproduces the sprint gait**:
stays upright for the episode, tracks the commanded forward speed within tolerance, and keeps
pelvis/CoM trajectory close to the teacher's. Phase A succeeds when the torque student
reproduces the sprint motion in rollout.

**Diagnostics (explain *why*, do not gate success):**
- Torque MSE / R² per DoF on held-out student-visited states — is it learning the mapping.
- DAgger vs plain BC on rollout fidelity — did the loop earn its complexity.
- Naturalness proxy (jerk/smoothness or torque-effort) + side-by-side render on the dashboard
  vs the muscle teacher — how much muscle compliance survived collapse to torque. Sets up Phase B.

## Out of scope (later phases, sketched)

- **Phase B — co-learning / muscle-as-constraint:** train the torque student with RL but
  regularized toward the muscle prior (penalize deviation from muscle-implied torque or
  muscle-implied stiffness) so it keeps compliance raw BC discards. Literature exists on hybrid
  muscle/torque actuation and muscle-as-regularizer.
- **Phase C — full G1 tree:** author the model on G1's real 29-DoF kinematic tree
  (12 leg + 3 waist + 14 arm) and repeat. On the back burner until A/B pay off.

## Risks / honest caveats

- BC/DAgger from a single-step torque readout can still drift; DAgger mitigates but the
  floating-base coupling (student controls 25 joint DoFs, not the 6 root DoFs) means the
  student must recover base motion purely through contact — this is the hardest part and the
  most likely place Phase A underperforms. That result would itself be informative.
- Collapsing 136 muscles → 25 net torques is lossy by construction (compliance, co-contraction,
  force–length/velocity shaping are flattened). Phase A measures motion reproduction, not
  naturalness preservation; the latter is explicitly Phase B.
