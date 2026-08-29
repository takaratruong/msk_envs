# Sub-step PD-Impedance Distillation (Phase B-lite) — Design

**Date:** 2026-08-29
**Status:** Design — awaiting review
**Builds on:** `2026-08-28-muscle-to-torque-dagger-design.md` (Phase A). Phase A's PHASE_A_FAIL is the motivation.

## Goal

Phase A distilled the muscle sprint teacher into a **pure-torque** student and it collapsed. Root cause (diagnosed + confirmed): the muscle's actuation is state-dependent — force = `f(activation, fiber_length, fiber_velocity)`, re-evaluated every integrator substep — so it carries intrinsic impedance (a nonlinear spring-damper) that stabilizes the body *between* the 30 Hz control ticks. Distilling to a **net torque** kept only the feedforward part and applied it as a flat constant across each 33 ms window, blind to how the body moved. Result: open-loop torque, unstable biped, falls in ~0.5 s.

Fix: give the torque student the missing impedance by making the actuator a **state-dependent PD element evaluated every substep** (like the muscle), and distill from the teacher not just its torque but its **impedance** (stiffness + damping) and **setpoint**. The policy outputs `(q_des, kp, kd)` per joint; the applied torque `τ = kp·(q_des − q) − kd·q̇` is recomputed each substep from live state.

This is deployment-realistic: it's exactly how a real Unitree G1 is driven — a policy sets target angles (+ gains) at low rate, a fast joint-level PD loop stabilizes in between.

## Why this is feasible (verified in Bolt source)

- `forward.fwd(m, d)` — which calls `realize_actuators` → `actuator_force` — is invoked **inside the adaptive integrator at every substep** (`integrate_euler_adaptive.py` lines 31, 47, 68). So an actuator force kernel that reads live state produces true per-substep impedance, at the same rate the muscle enjoys.
- The current actuator kernel `_ufrc_actuators` (`smooth_actuator.py`) already runs per-substep but ignores state: `actuation = (activation − 0.5)·2·optimal_force`.
- `d.qpos[world, coord]` and `d.qvel[world, coord]` are available in that kernel; `ActuatorMetadata` already carries `coordinate` (the DoF index). Adding a PD force law there reads only quantities already present.
- Teacher net joint torque `τ_muscle` is read from `ufrc_muscle` (Phase A, verified: = joint torque for revolute joints).

## Architecture

### Component 1 — Sub-step PD actuator (Bolt core)

A NEW actuator type (`PDActuator`) so muscle-driven models are untouched (no regression). Its per-substep force kernel:

```
τ = clamp( kp·(q_des − qpos[coord]) − kd·qvel[coord],  −τ_max, +τ_max )
ufrc_actuator[coord] += τ
```

- `q_des, kp, kd, τ_max` per actuator. `q_des, kp, kd` are the policy's 30 Hz command, held across the control window in per-actuator state buffers (analogous to the existing `a_act`/excitation buffer); the kernel re-evaluates against *current* qpos/qvel every substep → state-responsive torque.
- `τ_max` sized from the teacher's per-joint torque range (as Phase A sized `optimal_force`).
- New metadata fields on the actuator; new command buffers in `Data`; wired into `realize_actuators`/`actuator_force`.
- **Guardrails:** `kp ≥ 0`, `kd ≥ 0` enforced (never active-destabilizing); `τ` clamped to `±τ_max`.

### Component 2 — Teacher impedance extraction (msk_envs distill)

At each teacher-visited state `(q, q̇)` with the teacher's muscle activations applied, decompose the muscle's actuation into setpoint + diagonal impedance per joint `j`:

- **Stiffness** (finite-difference the muscle model): `kp[j] = −(τ(q+εₚ·eⱼ) − τ(q−εₚ·eⱼ)) / (2εₚ)`, where `τ(·)` is `ufrc_muscle` re-read after perturbing joint `j`'s position. Clamp `≥ 0`.
- **Damping**: `kd[j] = −(τ(q̇+ε_d·eⱼ) − τ(q̇−ε_d·eⱼ)) / (2ε_d)`. Clamp `≥ 0`.
- **Setpoint** (from the identity so the PD law reproduces the observed torque at the current state): `q_des[j] = q[j] + τ[j]/kp[j]` (guard `kp[j] > kp_min` to avoid divide-by-zero; if `kp[j] ≈ 0`, fall back to `q_des[j] = q[j]` and rely on damping).

Diagonal impedance only (`∂τᵢ/∂qⱼ` for i≠j ignored) — matches how a real robot's per-joint PD controllers work and keeps the student output at 3×25 = 75. Biarticular coupling is a known simplification (flagged, not modeled).

Cost: ~4 extra `ufrc_muscle` evaluations per joint per labeled state (±q, ±q̇). Acceptable; it's the price of recovering impedance.

### Component 3 — Distillation (DAgger, reuse Phase A)

Student MLP: `obs (359) → (q_des, kp, kd) ∈ ℝ⁷⁵`. Trained by MSE against the per-state `(q_des, kp, kd)` targets, in the same DAgger loop as Phase A (student rollout → teacher relabels visited states → aggregate → refit). Reuse verbatim: `build_actuator_perm`, `build_teacher_obs_cols` (corrected), muscles-off (raw −1) convention, teacher checkpoint, β schedule.

## Evaluation / success criteria

**Primary (the gradeable target):** replay the distilled PD student and compare its *applied* torque trajectory `τ_PD(t) = kp·(q_des−q) − kd·q̇` against the teacher's `τ_muscle(t)` at matched states along the rollout — per-joint correlation + magnitude match. This directly tests "did the PD form reproduce the muscle's actuation trajectory," including its in-window (sub-tick) variation, which the flat torque could not.

**Outcome:** actual sprint vs teacher — forward speed + not-fallen fraction.
- High torque-match + sprints → success: the muscle's impedance was the missing piece.
- High torque-match + no sprint → real finding: PD form captured the muscle actuation but propulsion lives elsewhere (e.g. contact timing) → motivates full RL Phase B.
- Low torque-match → the diagonal PD form can't capture the muscle; report which joints/phases diverge.

**Guardrail gates (from Phase A lessons):**
- Sanity: the *teacher itself* must sprint in the eval harness (the check that caught Phase A's P0 obs-adapter bug) before student numbers are trusted.
- Use ONE reconciled upright metric across all harnesses (Phase A had a 0.08-vs-0.935 disagreement between harness variants — pick per-step not-fallen via `has_fallen`, matched start states, and state it once).

## Out of scope (future)

- **Full RL Phase B:** RL-train `(q_des, kp, kd)` to sprint, regularized toward the teacher (muscle-as-constraint), warm-started from this distilled policy. Only if the distillation result warrants it.
- Full (non-diagonal) stiffness matrix / biarticular coupling.
- Time-varying `τ_max`.

## Risks / caveats

- **Identifiability:** `(q_des, kp, kd)` are not uniquely determined by a single `(q, q̇, τ)` sample; we resolve this by defining kp/kd as the muscle's local `−∂τ/∂q`, `−∂τ/∂q̇` (finite-difference), which IS well-defined, then deriving `q_des` from the torque identity.
- **Finite-difference noise:** muscle force curves can be stiff; ε must be tuned (report chosen ε and a sensitivity check).
- **Learned-gain instability:** mitigated by `kp,kd ≥ 0` clamps and `τ_max` clamp, but a bad distilled gain could still oscillate — the τ_PD-vs-τ_muscle metric will surface it.
- **Diagonal approximation** drops biarticular coupling (hamstrings span hip+knee); may cap torque-match on those joints — expected, and measurable per-joint.
