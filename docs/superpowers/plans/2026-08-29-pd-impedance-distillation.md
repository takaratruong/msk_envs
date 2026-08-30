# Sub-step PD-Impedance Distillation (Phase B-lite) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the distilled torque student the muscle's missing impedance via a sub-step PD actuator (`τ = kp·(q_des−q) − kd·q̇`, recomputed every integrator substep), distill `(q_des, kp, kd)` from the teacher, and grade by how well the applied PD torque matches the muscle torque trajectory + actual sprint.

**Architecture:** Add a PD force mode to Bolt's coordinate actuator: a per-substep Warp kernel reads live `qpos/qvel` and three held per-actuator command buffers (`pd_q_des, pd_kp, pd_kd`) to produce a state-responsive torque, leaving the existing constant-torque path untouched (no muscle-model regression). The muscle teacher's per-state diagonal impedance (`kp=−∂τ/∂q`, `kd=−∂τ/∂q̇` by finite difference of `ufrc_muscle`) and setpoint (`q_des=q+τ/kp`) become distillation targets; the DAgger loop, actuator permutation, teacher-obs adapter, and muscles-off convention are reused verbatim from Phase A.

**Tech Stack:** Python 3.11, Warp (Bolt kernels), PyTorch (CUDA), OpenSim 4.6, existing msk_envs distill module.

**Spec:** `docs/superpowers/specs/2026-08-29-pd-impedance-distillation-design.md`

## Global Constraints

- **PD law (per substep, in the actuator force kernel):** `τ = clamp(kp·(q_des − qpos[coord]) − kd·qvel[coord], −τ_max, +τ_max)`. `τ_max` = the actuator's `optimal_force` (reuse Phase A sizing). `qpos` indexed by the actuator's qpos-adr, `qvel` by dof-adr — NOTE qpos and qvel indices differ (qpos width 32 incl. quaternion, qvel width 31); populate/use both.
- **Guardrails:** enforce `kp ≥ 0`, `kd ≥ 0` (clamp at distillation AND in-kernel defensively); torque clamped to `±τ_max`.
- **PD is stateless** (no activation dynamics for Phase B): kernel reads command buffers + live state directly; does NOT consume integration state `nz`.
- **Command buffers:** three new `(n_worlds, nactuator)` float arrays in `Data`: `pd_q_des`, `pd_kp`, `pd_kd`, held across the 30 Hz control window (written once per control step, read every substep).
- **PD mode is opt-in** via a model-load flag / env-config field (`use_pd_actuators: bool`); when false, the existing constant-torque `_ufrc_actuators` path runs unchanged.
- **Reuse Phase A verbatim:** `dof_utils.build_actuator_perm`, `build_teacher_obs_cols` (corrected act_start=2·num_muscles), muscles-OFF (raw −1) student convention, teacher ckpt `models/baseline_sprint_2026-08-27_20-59/..._149000.pt`, `sprinter_torque.osim` (25 actuators).
- **Model reused as-is:** `sprinter_torque.osim` (ActivationCoordinateActuator ×25). PD mode does not need a new OpenSim type — `q_des/kp/kd` are runtime policy outputs, not model fields; `optimal_force` in the model = `τ_max`.
- **GPU:** Bolt needs CUDA → `dangerouslyDisableSandbox: true`; pick an idle GPU via `nvidia-smi` (SHARED and segfault-prone — low util AND low mem), set `CUDA_VISIBLE_DEVICES`; use `cuda_graph=True` (cuda_graph=False was pathologically slow in Phase A). Retry on a different idle GPU if a run segfaults with no traceback.
- **Env:** `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh; conda activate bolt`. `env.step` returns 5-tuple `(obs, rew, term, trunc, info)`; set `reward_lambdas` (BLANK_REWARD_LAMBDAS all-zero, see evaluate.py) or KeyError. Commit with `git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code"` (never `git config`; stage ONLY plan files — the working tree has unrelated natural-walk changes; never `git add -A`).
- **Branch:** continue on `distill/muscle-to-torque-dagger`.

**User decisions (already made):**
- "Torque G1 is the target; muscles are a prior" — deployment target is a torque robot; PD is how G1 is really driven.
- Time-varying impedance: policy outputs `(q_des, kp, kd)` per joint (75 outputs), NOT fixed gains.
- Distill from teacher (not RL yet); grade by applied-PD-torque vs muscle-torque trajectory match + sprint speed.
- Faithful **sub-step** PD (kernel-level), not a 30 Hz Python approximation.

---

### Task 0: Sub-step PD actuator kernel in Bolt

**Goal:** Add a stateless PD force mode to Bolt's actuator: new command buffers + `use_pd` metadata flag + a per-substep kernel computing `τ = clamp(kp·(q_des−q)−kd·q̇, ±τ_max)`, wired into `actuator_force`. Existing constant-torque path unchanged.

**Files:**
- Modify: `/home/ubuntu/bolt/bolt/_src/types.py` (add `pd_q_des/pd_kp/pd_kd` arrays to Data; add `use_pd` + qpos/qvel adr fields to `ActuatorMetadata`)
- Modify: `/home/ubuntu/bolt/bolt/_src/smooth_actuator.py` (new `_ufrc_pd_actuators` kernel + launch in `actuator_force`)
- Modify: `/home/ubuntu/bolt/bolt/model_loader.py` (allocate the 3 buffers; populate qpos/qvel adr in metadata)
- Modify: `/home/ubuntu/bolt/bolt/bindings.py` (getters/setters for pd_q_des/pd_kp/pd_kd)

**Acceptance Criteria:**
- [ ] With `use_pd` set, driving `pd_q_des` toward a target with `pd_kp>0, pd_kd>0` moves the coordinate toward `q_des` and holds it (a single-joint settling test).
- [ ] With `use_pd` false, actuator behaves exactly as before (constant-torque path untouched — regression check: existing sprinter_torque load + step unchanged).
- [ ] Torque is clamped to `±optimal_force`; `kp<0`/`kd<0` inputs are treated as 0 (defensive clamp).

**Verify:** a single-actuator settling script prints the coordinate converging to `q_des` under PD, and no-PD-effect under the old path — run on GPU.

**Steps:**

- [ ] **Step 1: Add command buffers + metadata fields.**

In `types.py`, on the `Data` struct add (alongside `a_act`):
```python
    pd_q_des: wp.array2d(dtype=float)   # (nworld, nactuator) target angle, held per control step
    pd_kp: wp.array2d(dtype=float)      # (nworld, nactuator) stiffness
    pd_kd: wp.array2d(dtype=float)      # (nworld, nactuator) damping
```
On `ActuatorMetadata` add:
```python
    use_pd: int          # 1 = PD force law, 0 = constant-torque (default 0)
    qpos_adr: int        # qpos index for this actuator's coordinate
    qvel_adr: int        # qvel (dof) index for this actuator's coordinate
```
(existing `coordinate` = qvel/dof index; qpos_adr may differ — populate both.)

- [ ] **Step 2: PD force kernel.**

In `smooth_actuator.py`:
```python
@wp.kernel
def _ufrc_pd_actuators(
        actuator_metadata: wp.array(dtype=ActuatorMetadata),
        integration_done_in: wp.array(dtype=bool),
        qpos_in: wp.array2d(dtype=float),
        qvel_in: wp.array2d(dtype=float),
        pd_q_des_in: wp.array2d(dtype=float),
        pd_kp_in: wp.array2d(dtype=float),
        pd_kd_in: wp.array2d(dtype=float),
        ufrc_actuator_out: wp.array2d(dtype=float),
):
    worldid, aid = wp.tid()
    if integration_done_in[worldid]:
        return
    am = actuator_metadata[aid]
    if am.use_pd == 0:
        return
    q = qpos_in[worldid, am.qpos_adr]
    qd = qvel_in[worldid, am.qvel_adr]
    kp = wp.max(pd_kp_in[worldid, aid], 0.0)
    kd = wp.max(pd_kd_in[worldid, aid], 0.0)
    tau = kp * (pd_q_des_in[worldid, aid] - q) - kd * qd
    tau = wp.clamp(tau, -am.optimal_force, am.optimal_force)
    wp.atomic_add(ufrc_actuator_out[worldid], am.coordinate, tau)
    return
```

- [ ] **Step 3: Launch it in `actuator_force`.**

Modify `actuator_force(m, d)` to launch BOTH kernels (each early-returns on the wrong `use_pd`), so mixed models work and constant-torque stays intact:
```python
def actuator_force(m: Model, d: Data):
    wp.launch(_ufrc_actuators, dim=(d.nworld, m.nactuator),
              inputs=[m.actuator_metadata, d.integration_done, d.a_act],
              outputs=[d.ufrc_actuator])
    wp.launch(_ufrc_pd_actuators, dim=(d.nworld, m.nactuator),
              inputs=[m.actuator_metadata, d.integration_done, d.qpos, d.qvel,
                      d.pd_q_des, d.pd_kp, d.pd_kd],
              outputs=[d.ufrc_actuator])
```
ALSO add `if am.use_pd == 1: return` near the top of `_ufrc_actuators` so a PD actuator isn't double-driven by the constant-torque kernel. Verify that guard is added.

- [ ] **Step 4: Allocate buffers + populate metadata in model_loader.**

Where `a_act` is allocated (`model_loader.py` ~436/496), add `pd_q_des/pd_kp/pd_kd = make_zero((n_worlds, nactuator))`. When building `ActuatorMetadata` (via actuator_helper), set `use_pd` from a load flag, and set `qpos_adr`/`qvel_adr` from the model's coordinate→qpos/qvel address maps (`mob_qposadr`/`mob_dofadr` resolved per coordinate). Thread a `use_pd_actuators: bool=False` param through `load_model`.

- [ ] **Step 5: Bindings.**

In `bindings.py` add torch getters:
```python
def pd_q_des(d): return wp.to_torch(d.pd_q_des)
def pd_kp(d):    return wp.to_torch(d.pd_kp)
def pd_kd(d):    return wp.to_torch(d.pd_kd)
```

- [ ] **Step 6: Single-joint settling test (write, run on GPU).**

Write `/home/ubuntu/bolt/test_pd_actuator.py`: load `sprinter_torque.osim` with `use_pd_actuators=True`, 4 worlds; set `pd_kp=200, pd_kd=10, pd_q_des=` a fixed target for one leg joint (e.g. knee), muscles off; step 100 control ticks; print the joint angle trajectory. Expected: angle converges toward `q_des` and holds (settles), NOT diverges. Then load with `use_pd_actuators=False`, apply the same buffers → confirm no PD effect (regression). Run: `cd /home/ubuntu/bolt && CUDA_VISIBLE_DEVICES=<idle> python test_pd_actuator.py` (sandbox off).

- [ ] **Step 7: Commit.**

```bash
cd /home/ubuntu/bolt
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add bolt/_src/types.py bolt/_src/smooth_actuator.py bolt/model_loader.py bolt/bindings.py test_pd_actuator.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(pd): sub-step PD actuator force mode (Task 0)"
```

```json:metadata
{"files": ["/home/ubuntu/bolt/bolt/_src/types.py", "/home/ubuntu/bolt/bolt/_src/smooth_actuator.py", "/home/ubuntu/bolt/bolt/model_loader.py", "/home/ubuntu/bolt/bolt/bindings.py", "/home/ubuntu/bolt/test_pd_actuator.py"], "verifyCommand": "cd /home/ubuntu/bolt && python test_pd_actuator.py", "acceptanceCriteria": ["PD mode: joint converges to q_des and holds", "use_pd=false: constant-torque path unchanged (regression)", "torque clamped to +/-optimal_force, kp/kd<0 treated as 0"], "modelTier": "frontier"}
```

---

### Task 1: Env config + action plumbing for PD actuators

**Goal:** An env config `EnvConfigSprinterTorquePD` that loads the sprinter_torque model in PD mode, and an env-side path to write a 75-wide action `(q_des, kp, kd)×25` into the `pd_q_des/pd_kp/pd_kd` buffers (names order → actuator-slice order via act_perm).

**Files:**
- Modify: `/home/ubuntu/msk_envs/msk_envs/envs/env_config.py` (+`EnvConfigSprinterTorquePD` with `use_pd_actuators=True`; register subcommand `sprintertorquepd`)
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/pd_control.py` (helpers to write pd buffers + scale raw policy outputs to physical q_des/kp/kd ranges)

**Acceptance Criteria:**
- [ ] `env-config:sprintertorquepd` resolves; env constructs with PD actuators (num_actuators==25, use_pd active).
- [ ] `pd_control.apply_pd(env, q_des, kp, kd, act_perm)` writes the three buffers (permuted names→actuator order) and steps without error.
- [ ] A smoke rollout with constant `(q_des=0, kp=100, kd=5)` keeps the body from instantly collapsing (sanity that PD is engaged: not-fallen > 0.5 over 100 steps).

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.pd_control` → prints `pd buffers written, stepped ok, notfallen>0.5 with basic gains`.

**Steps:**

- [ ] **Step 1: Env config.**
```python
@dataclass
class EnvConfigSprinterTorquePD(EnvConfigSprinterTorque):
    """Sprinter torque twin driven by sub-step PD actuators (q_des,kp,kd per joint)."""
    use_pd_actuators: bool = True
```
Ensure `use_pd_actuators` is threaded from env_config → the env's `load_model` call → Bolt (check how env_base passes config fields to bindings.load_model; add the param). Register `Annotated[EnvConfigSprinterTorquePD, tyro.conf.subcommand(name="sprintertorquepd")]` in the union.

- [ ] **Step 2: pd_control helpers.**
```python
import torch
import bolt
def apply_pd(env, q_des, kp, kd, act_perm):
    """q_des,kp,kd: (n_envs,25) in names order. Write into pd buffers in actuator-slice order, muscles OFF."""
    bolt.pd_q_des(env.d).copy_(q_des.index_select(1, act_perm))
    bolt.pd_kp(env.d).copy_(torch.clamp(kp, min=0.0).index_select(1, act_perm))
    bolt.pd_kd(env.d).copy_(torch.clamp(kd, min=0.0).index_select(1, act_perm))
    a = env.get_blank_actions(); a[:, :env.num_muscles] = -1.0   # muscles OFF; actuator excitation slice unused in PD mode
    return env.step(a)
```

- [ ] **Step 3: smoke __main__** (build sprintertorquepd env with reward_lambdas set, act_perm via build_actuator_perm, apply constant q_des=0/kp=100/kd=5 for 100 steps via apply_pd, report not-fallen via has_fallen). Run on GPU (idle, cuda_graph=True).

- [ ] **Step 4: Commit** (stage only env_config.py + distill/pd_control.py).
```bash
cd /home/ubuntu/msk_envs
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/envs/env_config.py msk_envs/distill/pd_control.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(pd): SprinterTorquePD env config + pd_control apply (Task 1)"
```

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/envs/env_config.py", "/home/ubuntu/msk_envs/msk_envs/distill/pd_control.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.pd_control", "acceptanceCriteria": ["env-config:sprintertorquepd resolves with PD actuators", "apply_pd writes permuted buffers + steps", "basic gains keep notfallen>0.5"], "modelTier": "standard"}
```

---

### Task 2: Teacher impedance extraction (finite-difference stiffness/damping)

**Goal:** `distill/impedance.py` — given the muscle env at a teacher state with teacher activations applied, compute per-joint diagonal `kp=−∂τ/∂q`, `kd=−∂τ/∂q̇` (finite difference of `ufrc_muscle`) and setpoint `q_des=q+τ/kp`, all in names order.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/impedance.py`

**Acceptance Criteria:**
- [ ] `extract_impedance(env, teacher)` returns `(q_des, kp, kd)` each `(n_envs,25)`, finite; kp,kd clamped ≥0; q_des falls back to current q where kp<kp_min.
- [ ] Finite-difference validated: for stance joints, stiffness sign is restoring (kp>0); per-joint kp/kd distribution over a teacher rollout reported.
- [ ] ε sensitivity: kp/kd at ε vs ε/2 differ by <20% (else ε mis-set) — report the numbers.

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.impedance` → prints per-joint mean kp/kd over a short teacher rollout + ε-sensitivity, all finite.

**Steps:**

- [ ] **Step 1: perturbation-based stiffness.** Read current q/qd (qpos_id_lookup for position, dof_id_lookup for velocity, per name). Apply teacher muscle activations. For each of the 25 joints, perturb that DoF's qpos by ±εp, recompute `ufrc_muscle` WITHOUT advancing the sim, central-difference → kp[j]. Same perturbing qvel by ±εd → kd[j]. εp=1e-3 rad, εd=1e-2 rad/s starting values.
- [ ] **Step 2: setpoint** `q_des = q + tau/clamp(kp, min=kp_min)`, kp_min=1.0; where kp<kp_min set q_des=q.
- [ ] **Step 3: clamp** kp,kd = max(·, 0).
- [ ] **Step 4: __main__** rolls the teacher a few steps, calls extract_impedance each step, prints per-joint kp/kd mean + ε/2 sensitivity. Run GPU.
- [ ] **Step 5: Commit** (stage only distill/impedance.py).
```bash
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" add msk_envs/distill/impedance.py
git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code" commit -m "feat(pd): teacher impedance extraction via finite-difference (Task 2)"
```

**IMPLEMENTER NOTE — highest-uncertainty task:** the crux is recomputing `ufrc_muscle` at a perturbed state without corrupting the rollout. Investigate Bolt's realize path (`forward.realize_muscles`, `bolt.fk`): you likely save qpos/qvel, write the perturbed value, call the realize path to recompute muscle forces, read `ufrc_muscle`, then restore the saved state. If the only clean option is a full step, save and restore full state (qpos/qvel/time). Document exactly what you used. If a clean perturbation read is infeasible, report BLOCKED with findings rather than shipping a wrong stiffness.

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/impedance.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.impedance", "acceptanceCriteria": ["extract_impedance returns finite (q_des,kp,kd) (n_envs,25), kp/kd>=0", "finite-diff stiffness sign correct, per-joint distribution reported", "epsilon sensitivity <20%"], "modelTier": "frontier"}
```

---

### Task 3: DAgger distillation of (q_des, kp, kd)

**Goal:** `distill/dagger_pd.py` — DAgger loop distilling the teacher's impedance: student MLP `obs(359)→(q_des,kp,kd)∈ℝ⁷⁵`, labels from Task 2's `extract_impedance`, student rollout drives the PD env via `pd_control.apply_pd`. Saves `pd_student.pt` + metrics.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/dagger_pd.py`
- Create (output): `/home/ubuntu/msk_envs/models/dagger_pd/pd_student.pt`, `metrics.json`

**Acceptance Criteria:**
- [ ] ≥8 DAgger rounds run; buffer grows; per-round val MSE (75-dim target) logged and decreasing (final < 0.6× round-0).
- [ ] Student rollout uses PD env + muscles-off + act_perm (identical convention to labels).
- [ ] Saves pd_student.pt (state_dict, n_obs, names, output scaling) + metrics.json.

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.dagger_pd --rounds 8` → per-round val_mse lines, final<0.6×round-0, `SAVED pd_student.pt`.

**Steps:**

- [ ] **Step 1:** Student head outputs 75 = concat(q_des[25], kp[25], kd[25]). Scaling: q_des in radians (raw linear head); kp,kd via softplus (guarantees ≥0) scaled to the teacher's observed kp/kd ranges from Task 2. Document the mapping in the file + saved in the checkpoint.
- [ ] **Step 2: rollout_and_label** — mirror dagger.py: teacher probe step (muscles, read state) → `extract_impedance(env, teacher)` gives the 75-dim label at obs; β-mix advance: teacher's next obs (prob β) or student PD step via `pd_control.apply_pd(env, q_des, kp, kd, perm)` (prob 1−β). Aggregate (obs, label75).
- [ ] **Step 3: fit** — MSE on 75-dim, 80/20 split, Adam 1e-3 (reuse dagger.py fit).
- [ ] **Step 4: main** — β schedule (r0=1, anneal 1−r/(rounds−1)), save pd_student.pt + metrics.json to models/dagger_pd/. Reuse Teacher, build_actuator_perm, build_teacher_obs_cols, joint_dof_indices.
- [ ] **Step 5: run 8 rounds**, confirm MSE drop. Tuning knobs (--epochs/--horizon/--rounds) allowed; report actuals. (Note per Phase A: more ROUNDS, not epochs, improves the ratio.)
- [ ] **Step 6: Commit** (stage only distill/dagger_pd.py).

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/dagger_pd.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.dagger_pd --rounds 8", "acceptanceCriteria": ["8 rounds, buffer grows, final val_mse<0.6x round0", "student rollout via PD env + muscles-off + act_perm", "saves pd_student.pt + metrics.json"], "modelTier": "frontier"}
```

---

### Task 4: Evaluation — torque-trajectory match + sprint (PRIMARY GATE)

**Goal:** `distill/evaluate_pd.py` — the decisive experiment: replay the distilled PD student, (a) measure applied `τ_PD(t)=kp(q_des−q)−kd·q̇` vs teacher `τ_muscle(t)` per-joint correlation along matched rollouts, and (b) measure sprint (fwd speed, not-fallen) vs teacher. Print verdict.

**USER-ORDERED GATE — NON-SKIPPABLE.** The user set the success criterion as "we have a final torque to compare to" — the applied-PD-torque vs muscle-torque match is the primary, user-specified metric. It MUST NOT be closed by walking around it or substituting a cheaper check. Close only after a real GPU run captures the per-joint torque-match table AND the sprint numbers (teacher vs pd_student), with the teacher-sprints sanity gate passing first.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/evaluate_pd.py`
- Create (output): `/home/ubuntu/msk_envs/models/dagger_pd/eval_pd.json`

**Acceptance Criteria:**
- [ ] Sanity gate: teacher sprints in this harness (fwd>0 clearly, not-fallen high) — else STOP and report (Phase A P0 lesson).
- [ ] Reports per-joint correlation + RMS ratio of τ_PD vs τ_muscle over ≥256 envs, full episode, at matched states.
- [ ] Reports sprint metrics (fwd_speed, not-fallen) for teacher vs pd_student, single reconciled not-fallen metric (has_fallen, matched starts).
- [ ] Prints a verdict line: `TORQUE_MATCH=<mean corr>; SPRINT pd=<fwd> teacher=<fwd>`; and the interpretation bucket (match+sprint / match+no-sprint / no-match).

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.evaluate_pd` → teacher sanity + per-joint torque-match table + sprint table + verdict line.

**Steps:**

- [ ] **Step 1:** reuse evaluate.py's harness (run_metrics, teacher path with obs adapter, has_fallen upright). Add a τ recorder: during the pd_student (muscles-off) rollout, at each visited state compute τ_PD = clamp(kp(q_des−q)−kd·qd, ±τ_max) in names order AND relabel with the teacher: apply teacher muscle activations at that same state, read env.ufrc_muscle[teacher.idx_t] = τ_muscle (same relabel trick as DAgger). Store both trajectories.
- [ ] **Step 2:** teacher-sprints sanity gate FIRST; abort with a clear message if teacher collapses (adapter/harness bug).
- [ ] **Step 3:** per-joint Pearson correlation + RMS(τ_PD)/RMS(τ_muscle) across the stored trajectories; sprint metrics (fwd, not-fallen) for teacher vs pd_student over 256 envs full episode.
- [ ] **Step 4:** verdict line + dump eval_pd.json (teacher, pd_student, per-joint torque_match, buckets).
- [ ] **Step 5: run it** (real GPU), capture output. Report honestly — match+sprint, match+no-sprint, or no-match are all valid results; do NOT tune thresholds.
- [ ] **Step 6: Commit** (stage only distill/evaluate_pd.py; models/ gitignored so eval_pd.json stays on disk).

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/evaluate_pd.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.evaluate_pd", "acceptanceCriteria": ["teacher-sprints sanity gate passes first", "per-joint tau_PD vs tau_muscle correlation + RMS ratio over 256 envs reported", "sprint fwd/not-fallen for teacher vs pd_student reported", "verdict line printed, thresholds untouched"], "modelTier": "frontier", "userGate": true, "tags": ["user-gate"], "requireEvidenceTokens": [["teacher"], ["pd_student"], ["torque_match"]]}
```

---

### Task 5: Render the PD student + dashboard card

**Goal:** Render the distilled PD student's rollout to video and add a dashboard card next to the failed Phase A torque student, for visual before/after.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/record_pd.py`
- Modify: `/home/ubuntu/bolt_baselines/dashboard.html` (add "Distilled PD Student · Phase B" card)
- Create (output): `/home/ubuntu/bolt_baselines/videos/pd_student.mp4`

**Acceptance Criteria:**
- [ ] Single-env PD-student rollout recorded (via LoggedSim, PD env, muscles-off) and rendered to pd_student.mp4.
- [ ] Dashboard shows the PD student card; caption states the outcome truthfully (whatever Task 4 found).
- [ ] Opens in Firefox on :1 (window-capture verified).

**Verify:** `ls -la /home/ubuntu/bolt_baselines/videos/pd_student.mp4` non-empty + Firefox window screenshot shows the card.

**Steps:**

- [ ] **Step 1:** `record_pd.py` mirrors record.py but uses `EnvConfigSprinterTorquePD`, loads pd_student.pt, drives via pd_control.apply_pd, LoggedSim save_animation. Confirm the real output path (Phase A gotcha: save_animation writes `<folder>/<base>_<worldidx>.json.gz`, e.g. `dashboard/trajectories/pd_student/1_0.json.gz`).
- [ ] **Step 2:** `render_trajectory.py <traj>` → ffmpeg → `/home/ubuntu/bolt_baselines/videos/pd_student.mp4`.
- [ ] **Step 3:** add dashboard card after the Phase A "Distilled Torque Student" card; caption reflects Task 4 verdict (e.g. "PD-impedance student — stays upright / sprints at X m/s / etc.").
- [ ] **Step 4:** open Firefox on :1: `export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority HOME=/home/ubuntu; rm -f ~/.mozilla/firefox/*/lock ~/.mozilla/firefox/*/.parentlock; LD_LIBRARY_PATH=/snap/firefox/current/usr/lib/firefox:/snap/firefox/current/usr/lib/firefox/gtk3 setsid /snap/firefox/current/usr/lib/firefox/firefox --new-window "file:///home/ubuntu/bolt_baselines/dashboard.html" &` — then capture the Firefox WINDOW (multi-head display; root capture fails): `WID=$(xdotool search --name "Bolt" | tail -1); import -window "$WID" /tmp/claude/dash_pd.png`. Read it to confirm the card. (All need sandbox off.)
- [ ] **Step 5: Commit** (stage only record_pd.py + dashboard.html — dashboard.html is in /home/ubuntu/bolt_baselines, add it by explicit path; it may be outside a git repo, in which case just produce the file).

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/record_pd.py", "/home/ubuntu/bolt_baselines/dashboard.html"], "verifyCommand": "ls -la /home/ubuntu/bolt_baselines/videos/pd_student.mp4", "acceptanceCriteria": ["PD student rollout rendered to mp4", "dashboard card added with truthful caption", "opens in Firefox on :1, window capture verified"], "modelTier": "standard"}
```

---

## Ordering & dependencies

Strictly sequential: 0 (PD kernel) → 1 (env/plumbing) → 2 (impedance extraction) → 3 (DAgger) → 4 (eval, PRIMARY GATE) → 5 (render). Task 0 is the riskiest (Bolt-core Warp kernel); Task 2 is the highest-uncertainty (perturbation read of ufrc_muscle). Task 4 is the user gate. A no-sprint result at Task 4 does not block Task 5 — the render + torque-match numbers are the finding.
