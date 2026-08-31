# RL Phase B — PD Action Space + Muscle-Impedance Constraint — Design

**Date:** 2026-08-31
**Status:** Design — approved to proceed (user: "go" / "stop asking")
**Builds on:** Phase A (`2026-08-28...`), PD-impedance distillation (`2026-08-29...`). Motivated by the moving-reference experiment (open-loop trajectory replay collapses) → propulsion needs a *closed-loop* state-reactive controller, which only RL-against-the-task can supply.

## Goal

Train an RL policy on the **sub-step PD action space** (Task-0 PD actuator: outputs `q_des, kp, kd` per joint) to **actually sprint**, warm-started from the Phase-B distilled student (which balances but doesn't propel), and regularized to keep its impedance **muscle-like** (kp/kd near the teacher's extracted values). This is the "muscle-as-constraint" formulation: RL supplies the closed-loop propulsion that imitation structurally could not, while the regularizer preserves the natural compliance the muscles encoded.

**Why RL (settled by prior experiments):** Phase A (distill net torque) → collapses. Phase B (distill static impedance) → balances, no sprint (torque-match 0.356 even on-distribution). Moving-reference replay → collapses (open-loop kinematics can't stabilize a passive floating base). The propulsion is a closed-loop, state-reactive quantity; the only clean way to obtain `q_des(state)` that both balances AND propels is to optimize the task directly (RL), using the PD actuator (proven to stabilize) as the action space.

## Architecture

### Component 1 — PD-action-space RL env
A new env config `EnvConfigSprinterTorquePD_RL` (subclass of the locomotion/sprint env on the PD model, `use_pd_actuators=True`). Overrides:
- **`num_actions` → 75** (q_des[25], kp[25], kd[25]), replacing the default `[muscle_excitations|actuator_excitations]` action.
- **`_set_actions(raw75)`**: decode raw→(q_des,kp,kd) with the SAME decode as distillation (dagger_pd.decode: q_des = q_now + clamp(delta, ±QDES_DELTA_CLAMP); kp=softplus(raw)·KP_SCALE; kd=softplus(raw)·KD_SCALE, scales from calibration), permute names→actuator order (build_actuator_perm), write pd_q_des/pd_kp/pd_kd buffers, muscles forced OFF (−1).
- **`_get_actions`**: return current (q_des,kp,kd) reconstruction (for buffer shape / algo bookkeeping).
- **Reward**: reuse the EXISTING sprint/locomotion reward that trained the teacher (forward-velocity + alive + limit/effort penalties). Do NOT invent a reward. obs=359, termination has_fallen — unchanged.
- Slots into the existing TD3/PPO trainer (train.py) exactly as the muscle sprinter did — only the action space differs.

### Component 2 — Warm-start from the distilled balancer
Initialize the RL actor from `models/dagger_pd/pd_student.pt` (obs359→75, balances at upright 0.997). Load compatible weights into the trainer's actor net before training (align hidden dims (256,256), or load the shared trunk). This hands RL a standing controller so exploration targets propulsion, not re-learning balance (which random init cannot — moving-ref test showed instant collapse).

### Component 3 — Impedance regularizer (muscle-as-constraint)
A reward/loss penalty keeping the policy's emitted (kp,kd) near the teacher's extracted impedance (Task-2 `extract_impedance`, validated):
`R_imp = −λ · ‖ log(kp_policy) − log(kp_teacher(s)) ‖² + (kd term)` (log-space because kp spans hundreds–thousands). Computed on visited states; teacher impedance from the finite-difference extractor.
- **λ schedule**: start high (stay muscle-compliant, exploit warm-start stability) → anneal as forward reward grows. Config knob with default + anneal.
- **Tension (the experiment's point)**: too-high λ → stays a balancer (Phase B); too-low λ → plain RL-on-torque-robot (muscle claim gone). Success = a λ regime where it BOTH sprints AND stays compliant.

## Evaluation / success criteria

**Primary:** the policy SPRINTS — forward speed approaching the teacher's (target: a large fraction of teacher fwd, ≫ Phase B's −0.09), staying upright (has_fallen), over ≥256 envs full episode, teacher-sanity-gated first (Phase A/B P0 lesson: teacher must sprint in the eval harness).

**Muscle-transfer metric:** compliance — mean ‖(kp,kd)_policy − (kp,kd)_teacher‖ (log-space) vs a no-regularizer RL baseline. Shows the prior kept the controller muscle-like rather than optimizing to arbitrary stiff gains.

**Ablations:** λ sweep (high→balances / low→plain-RL / find the both regime); regularizer vs none; warm-start vs random-init (expected: random-init fails, per moving-ref).

**Interpretation buckets:** (a) sprints + compliant → muscle-transfer success; (b) sprints only when λ→0 → propulsion needs RL but muscle prior doesn't survive it (still a result); (c) never sprints → distilled-balance + RL insufficient on this body (informative negative).

## Compute reality (honest)
This is real RL: the muscle sprinter took 150k iters. Expect **hours-to-days of GPU per run**, several runs for the λ sweep. Shared, segfault-prone GPUs. The plan must budget for this (long background runs, checkpointing, not a single-session turnaround). A run may not converge to a sprint — that is a valid, recorded outcome, not a failure to implement.

## Out of scope
- Full G1 tree (Phase C, back burner).
- Non-diagonal impedance.
- Reward redesign — reuse the teacher's.

## Risks
- Warm-start architecture mismatch (distilled MLP vs trainer actor) — align or load-subset; verify the loaded policy still balances before RL.
- λ tuning is the crux and may need several runs; budget for it.
- RL on a passive-base biped is hard; convergence not guaranteed — checkpoint often, evaluate intermediate.
- Impedance-regularizer cost: teacher `extract_impedance` per training state is expensive (25×4 forward.fwd). Mitigate: regularize toward a precomputed impedance lookup / amortize, or subsample — decide in plan.
