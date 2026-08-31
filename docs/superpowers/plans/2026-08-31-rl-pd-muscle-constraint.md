# RL Phase B — PD Action Space + Muscle-Impedance Constraint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** RL-train a policy on the sub-step PD action space (outputs q_des,kp,kd) to sprint, warm-started from the Phase-B distilled balancer, regularized to keep impedance muscle-like — supplying the closed-loop propulsion imitation could not.

**Architecture:** A new env exposes a 75-dim PD action (q_des,kp,kd per joint) into the EXISTING TD3 trainer (which made the teacher — action-space-agnostic via `num_actions()`/`_set_actions`/`action_range`). Warm-start uses the trainer's existing `checkpoint_path` mechanism (bridge the distilled MLP into the actor's `actor_state_dict` format). The impedance regularizer is added as one more reward term (`rew_impedance` + `lambda_impedance`) via the existing `reward_dict`/`reward_lambdas` machinery — no new RL code.

**Tech Stack:** Python 3.11, PyTorch CUDA, existing msk_envs TD3 trainer, Bolt PD actuator (Task-0 of prior plan), distill module.

**Spec:** `docs/superpowers/specs/2026-08-31-rl-pd-muscle-constraint-design.md`

## Global Constraints

- **PD action decode** (identical to distillation, single source `dagger_pd`): q_des = q_now + clamp(delta, ±QDES_DELTA_CLAMP=0.5); kp=softplus(raw)·KP_SCALE; kd=softplus(raw)·KD_SCALE (per-joint scales). Permute names→actuator order via `build_actuator_perm`; muscles OFF (−1). Reuse `pd_control.apply_pd` semantics.
- **Reward**: reuse the EXISTING sprint/locomotion reward (forward-velocity + alive + penalties) that trained the teacher. Add ONLY `rew_impedance` as an extra term. Do NOT redesign the base reward.
- **Impedance regularizer**: `rew_impedance = −‖log(kp_pol)−log(kp_tea(s))‖² − ‖log(kd_pol+ε)−log(kd_tea(s)+ε)‖²` (log-space; kp spans hundreds–thousands, kd~0-7). Scaled by `lambda_impedance` (config, annealable). Teacher impedance kp_tea/kd_tea(s) from Task-2 `extract_impedance` — EXPENSIVE (25×4 forward.fwd/state); the plan amortizes it (see Task 2).
- **Warm-start**: bridge `models/dagger_pd/pd_student.pt` (obs359→75 MLP) into the trainer actor's `actor_state_dict` format; pass via `--td3-config.checkpoint_path`. Verify the loaded actor still BALANCES (upright high) before RL — a warm-start that doesn't balance is a bridge bug.
- **GPU**: CUDA, `dangerouslyDisableSandbox: true`; idle GPU via nvidia-smi (low util AND mem), CUDA_VISIBLE_DEVICES; cuda_graph=True. Shared/segfault-prone → retry other GPU. Training is LONG (hours-days) → run in background, checkpoint often (save_interval), evaluate intermediate checkpoints.
- **Env**: `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh; conda activate bolt`; PYTHONPATH=/home/ubuntu/msk_envs. Commit `git -c user.email="claude-code@anthropic.local" -c user.name="Claude Code"`; stage explicit paths only (working tree has unrelated natural-walk changes; never `git add -A`). Branch `distill/muscle-to-torque-dagger`.
- **Trainer integration points (verified):** td3/train.py: `n_act=envs.num_actions()` (line 57), `action_low/high=envs.action_range` (73), actor `DeterministicPolicy` (127), `actor.load_state_dict(ckpt["actor_state_dict"])` (377) when `checkpoint_path` set (372). Reward: env `_compute_raw_reward_dict`→`self.reward_dict`; `get_scaled_reward_dict` multiplies each term by `reward_lambdas[lambda_<key>]`.

**User decisions (already made):**
- Muscle prior = **warm-start + impedance regularizer** (muscle-as-constraint, full form).
- Distill→RL sequence: RL supplies closed-loop propulsion; imitation/moving-ref can't (proven).
- Reuse existing reward + trainer; don't reinvent RL.

---

### Task 0: PD-action-space RL env + config

**Goal:** An env that exposes the 75-dim PD action to the existing trainer and rewards sprinting, driving the Task-0 PD actuator.

**Files:**
- Modify: `/home/ubuntu/msk_envs/msk_envs/envs/env_config.py` (+`EnvConfigSprinterTorquePD_RL`, subcommand `sprintertorquepdrl`; + `lambda_impedance: float` and `impedance_anneal_*` fields)
- Create: `/home/ubuntu/msk_envs/msk_envs/envs/env_pd_rl.py` (env class: num_actions=75, _set_actions decode+apply, _get_actions, reward incl. rew_impedance placeholder=0 until Task 2)
- Modify: `/home/ubuntu/msk_envs/msk_envs/envs/env_factory.py` + `env_variants.py` if the variant needs registering

**Acceptance Criteria:**
- [ ] `env-config:sprintertorquepdrl` resolves; `env.num_actions()==75`; env constructs with PD actuators.
- [ ] A random-action rollout runs without error and without integrator stall (q_des clamped around current pose).
- [ ] Reward dict includes the base sprint terms (forward-vel etc.); `rew_impedance` present (0.0 placeholder until Task 2).

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.envs.env_pd_rl` (a `__main__` smoke: build env, num_actions==75, step 50 random actions, print reward keys + notfallen).

**Steps:**
- [ ] **Step 1:** `env_pd_rl.py` — subclass the sprint/locomotion env the teacher used (check env_variants for SPRINT→SprintingEnv). Override `num_actions`→75; `action_range` stays (-1,1). Import decode/QDES_DELTA_CLAMP + build_actuator_perm; in `__init__` build act_perm + load KP_SCALE/KD_SCALE (from pd_student.pt or recalibrate). `_set_actions(raw75)`: split raw_qdes/raw_kp/raw_kd, decode (q_des=q_now+clamp(delta,±0.5); kp/kd=softplus×scale), write pd buffers permuted, muscles=−1. `_get_actions`: return a 75-vec reconstruction (last-applied or zeros — for TD3 replay shape).
- [ ] **Step 2:** reward — inherit the base sprint `_compute_raw_reward_dict`, then add `self.reward_dict["rew_impedance"] = torch.zeros(num_worlds, device=...)` placeholder. Add `lambda_impedance` (default 0.0) to reward_lambdas.
- [ ] **Step 3:** register config + variant.
- [ ] **Step 4:** `__main__` smoke (random actions, no stall, reward keys, notfallen). Run GPU.
- [ ] **Step 5:** commit (env_config.py, env_pd_rl.py, env_factory.py/env_variants.py — explicit paths).

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/envs/env_pd_rl.py","/home/ubuntu/msk_envs/msk_envs/envs/env_config.py","/home/ubuntu/msk_envs/msk_envs/envs/env_factory.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.envs.env_pd_rl", "acceptanceCriteria": ["env-config:sprintertorquepdrl resolves, num_actions==75","random rollout no stall","reward dict has sprint terms + rew_impedance placeholder"], "modelTier": "frontier"}
```

---

### Task 1: Warm-start bridge (distilled student → actor checkpoint)

**Goal:** Convert `pd_student.pt` into the trainer's `actor_state_dict` checkpoint format so `--checkpoint_path` warm-starts the RL actor from the balancer; verify the loaded actor balances.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/warmstart_bridge.py`
- Create (output): `/home/ubuntu/msk_envs/models/dagger_pd/pd_student_actor_ckpt.pt`

**Acceptance Criteria:**
- [ ] Produces a checkpoint with `actor_state_dict` (+ obs_normalizer_state, and any qnet keys the trainer expects) loadable by td3/train.py's actor at n_act=75, n_obs=359.
- [ ] Loaded actor reproduces StudentPD's raw outputs on sample obs (assert max|diff|<1e-4).
- [ ] Sanity: load into the actor, roll out in the PD-RL env → upright fraction high (balances, ~Phase-B's 0.997), fwd≈0.

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.warmstart_bridge` → writes ckpt + prints `warmstart balances: notfallen=<high>`.

**Steps:**
- [ ] **Step 1:** inspect td3/train.py actor construction (DeterministicPolicy(**actor_kwargs)) + checkpoint keys it loads (actor_state_dict, obs_normalizer_state, qnet_state_dict, qnet_target_state_dict). Map the StudentPD MLP → the actor net: if StudentPD is a plain MLP with matching in/out dims (359→75) + hidden (256,256), copy Linear weights into the actor's matching layers. Document the mapping.
- [ ] **Step 2:** write the bridged checkpoint (actor_state_dict from mapped weights; obs_normalizer_state from pd_student if present else identity; include zeroed qnet_state_dict/qnet_target_state_dict of correct shape ONLY if train.py errors without them — check whether checkpoint_path load tolerates missing qnet keys; if it does, omit).
- [ ] **Step 3:** verify output-match (<1e-4 vs StudentPD) THEN sanity rollout balances in PD-RL env. Run GPU.
- [ ] **Step 4:** commit warmstart_bridge.py.

**IMPLEMENTER NOTE:** the decode (softplus×scale, q_des clamp) lives in the ENV `_set_actions` (Task 0), so the actor only needs to emit the same 75 raw values StudentPD did. Bridge is a state_dict remap if architectures align. If train.py's actor is NOT a plain MLP (e.g. SimbaActor default) force `--td3-config` to use DeterministicPolicy (the sprint teacher used it) so the warm-start maps cleanly; document the algo/actor-class choice.

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/warmstart_bridge.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.warmstart_bridge", "acceptanceCriteria": ["actor_state_dict ckpt loadable by td3 at n_act=75 n_obs=359","loaded actor reproduces StudentPD outputs <1e-4","warm-start balances in PD-RL env"], "modelTier": "frontier"}
```

---

### Task 2: Impedance regularizer reward term

**Goal:** Populate `rew_impedance` in the PD-RL env with the log-space distance between the policy's emitted (kp,kd) and the teacher's impedance at the current state, amortized to be affordable in the RL loop.

**Files:**
- Modify: `/home/ubuntu/msk_envs/msk_envs/envs/env_pd_rl.py` (compute rew_impedance)
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/impedance_cache.py` (amortization: offline impedance oracle)

**Acceptance Criteria:**
- [ ] rew_impedance = −(log kp_pol − log kp_tea)² − (log(kd_pol+ε) − log(kd_tea+ε))², per env, finite.
- [ ] Teacher impedance amortized: an offline "impedance oracle" MLP `obs→(log kp,log kd)[25]` fit on extract_impedance over a teacher rollout (cheap forward pass at train time), OR subsampled live extraction every K steps held between. Document choice + cost. (Live 25×4 forward.fwd/step is too slow for RL — MUST amortize.)
- [ ] lambda_impedance>0 short smoke: rew_impedance active (nonzero, sane sign), no NaN/blowup.

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.impedance_cache` → builds oracle, prints rew_impedance stats over a rollout (finite, responds to gain mismatch).

**Steps:**
- [ ] **Step 1 (oracle, recommended):** roll teacher, call Task-2 `extract_impedance` over many states, fit small MLP obs→(log kp,log kd)[25], save to models/dagger_pd/impedance_oracle.pt. (Alt: subsample live every K steps — document if chosen.)
- [ ] **Step 2:** in env `_compute_raw_reward_dict`, after base terms: kp_pol/kd_pol = policy's last-applied gains (store in _set_actions); kp_tea/kd_tea = oracle(obs); rew_impedance = −MSE(log-space). Guard finite.
- [ ] **Step 3:** smoke verify nonzero/finite/sane sign. Run GPU.
- [ ] **Step 4:** commit env_pd_rl.py + impedance_cache.py.

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/envs/env_pd_rl.py","/home/ubuntu/msk_envs/msk_envs/distill/impedance_cache.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.impedance_cache", "acceptanceCriteria": ["rew_impedance log-space finite per env","teacher impedance amortized (oracle/subsample), cost documented","lambda>0 smoke active no NaN"], "modelTier": "frontier"}
```

---

### Task 3: Launch RL training (warm-start + regularizer) — LONG BACKGROUND RUN

**Goal:** Kick off the RL run with the existing TD3 trainer: PD-RL env, warm-start checkpoint, lambda_impedance>0. Checkpoint often; runs hours-days.

**USER-ORDERED GATE.** This launches a long training run the user is funding compute for. Close only after a stable run is confirmed HEALTHY for ~1-2k iters (warm-start balances at iter0, reward being optimized, checkpoints written) with captured evidence — NOT when training finishes (out of session).

**Files:**
- Create (output): `/home/ubuntu/msk_envs/models/rl_pd_<date>/` (trainer output — gitignored)
- Possibly modify: trainer arg plumbing only if lambda_impedance anneal needs a hook (else constant via reward_lambdas)

**Acceptance Criteria:**
- [ ] Training launches: `python -m msk_envs.train.train <sprint-task> --algo td3 --td3-config.num-envs <N> --td3-config.checkpoint_path <warmstart> env-config:sprintertorquepdrl` with lambda_impedance>0. Background, checkpoints at save_interval.
- [ ] First ~1-2k iters: no crash; warm-start balances at iter0 (fwd~0, upright high); reward dict includes rew_impedance; forward-velocity reward begins rising or being optimized. Report iter/s + ETA.
- [ ] Launch record (command, GPU, dir, start) in report/ledger.

**Verify:** training process alive, checkpoints appearing, log tail shows forward reward moving (or documented plateau). COMPLETES at a confirmed-healthy launch, not at training end.

**Steps:**
- [ ] **Step 1:** dry-run ~50 iters — full stack trains without error, warm-start balances at iter0.
- [ ] **Step 2:** launch real run background, idle GPU, modest save_interval, num-envs as fits. Record command+GPU+dir.
- [ ] **Step 3:** monitor ~1-2k iters; confirm healthy (no NaN, reward moving, checkpoints written); report iter/s + ETA.
- [ ] **Step 4:** ledger the launch; DO NOT block on completion.

```json:metadata
{"files": [], "verifyCommand": "training healthy for 1-2k iters (see report)", "acceptanceCriteria": ["RL launches with warm-start + regularizer no crash","warm-start balances at iter0, fwd reward optimized","checkpoints written, launch recorded"], "modelTier": "frontier", "userGate": true, "tags": ["user-gate"], "requireEvidenceTokens": [["warmstart"],["reward"],["checkpoint"]]}
```

---

### Task 4: Evaluate trained checkpoints — sprint + compliance (PRIMARY GATE)

**Goal:** Evaluate the best available RL checkpoint: does it sprint (fwd approaching teacher, ≫ Phase-B's −0.09), and did it stay muscle-compliant (kp/kd near teacher vs no-reg baseline)? Reuse evaluate_pd harness.

**USER-ORDERED GATE.** Success = sprint speed + compliance. Close only after a real GPU eval of an ACTUAL trained checkpoint captures both, teacher-sanity-gated first. If training hasn't reached a sprint, report current-best honestly (note iters trained) — an interim/negative result is valid.

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/evaluate_rl_pd.py`
- Create (output): `models/rl_pd_<date>/eval_rl.json`

**Acceptance Criteria:**
- [ ] Teacher-sanity gate passes first.
- [ ] pd_rl_student vs teacher: fwd_speed, not-fallen (has_fallen, matched starts, ≥256 envs full episode).
- [ ] Compliance: mean log-space ‖(kp,kd)_pol − (kp,kd)_tea‖ vs teacher impedance (ideally vs a no-regularizer run if available).
- [ ] Verdict + bucket (sprint+compliant / sprint-only-λ0 / no-sprint).

**Verify:** `cd /home/ubuntu/msk_envs && CUDA_VISIBLE_DEVICES=<idle> python -m msk_envs.distill.evaluate_rl_pd --ckpt <path>` → teacher sanity + sprint table + compliance + verdict.

**Steps:**
- [ ] **Step 1:** adapt evaluate_pd.py: load RL actor (DeterministicPolicy@75), drive PD-RL env, same has_fallen/matched-start harness; teacher-sanity gate first.
- [ ] **Step 2:** sprint metrics (fwd, not-fallen) pd_rl vs teacher.
- [ ] **Step 3:** compliance metric (policy kp/kd vs impedance oracle, log-space).
- [ ] **Step 4:** verdict + eval_rl.json on best checkpoint. Report honestly; thresholds untouched.
- [ ] **Step 5:** commit evaluate_rl_pd.py.

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/evaluate_rl_pd.py"], "verifyCommand": "cd /home/ubuntu/msk_envs && python -m msk_envs.distill.evaluate_rl_pd --ckpt <best>", "acceptanceCriteria": ["teacher-sanity gate first","sprint fwd/not-fallen pd_rl vs teacher over 256 envs","compliance log-space vs teacher impedance","verdict + bucket thresholds untouched"], "modelTier": "frontier", "userGate": true, "tags": ["user-gate"], "requireEvidenceTokens": [["teacher"],["pd_rl"],["compliance"]]}
```

---

### Task 5: Render + dashboard card

**Goal:** Render the trained RL-PD student and add a dashboard card (third in the progression: Phase A collapse → Phase B balance → RL sprint-or-not).

**Files:**
- Create: `/home/ubuntu/msk_envs/msk_envs/distill/record_rl_pd.py`
- Modify: `/home/ubuntu/bolt_baselines/dashboard.html`
- Create (output): `/home/ubuntu/bolt_baselines/videos/rl_pd_student.mp4`

**Acceptance Criteria:**
- [ ] RL-PD student rollout rendered to mp4.
- [ ] Dashboard card added, truthful caption reflecting Task-4 verdict.
- [ ] Opens in Firefox on :1, window-capture verified (multi-head: xdotool search Bolt + import -window <wid>).

**Verify:** `ls -la /home/ubuntu/bolt_baselines/videos/rl_pd_student.mp4` + Firefox window screenshot.

**Steps:**
- [ ] **Step 1:** record_rl_pd.py mirrors record_pd.py, loads RL actor, drives PD-RL env via same decode+apply, LoggedSim save_animation (path `<folder>/<base>_<worldidx>.json.gz`).
- [ ] **Step 2:** render_trajectory.py → ffmpeg → mp4.
- [ ] **Step 3:** dashboard card, truthful caption per Task-4 verdict.
- [ ] **Step 4:** Firefox :1 window capture verify.
- [ ] **Step 5:** commit record_rl_pd.py + dashboard.html.

```json:metadata
{"files": ["/home/ubuntu/msk_envs/msk_envs/distill/record_rl_pd.py","/home/ubuntu/bolt_baselines/dashboard.html"], "verifyCommand": "ls -la /home/ubuntu/bolt_baselines/videos/rl_pd_student.mp4", "acceptanceCriteria": ["RL-PD rollout rendered to mp4","dashboard card truthful caption","opens in Firefox :1 window capture verified"], "modelTier": "standard"}
```

---

## Ordering & dependencies
Sequential: 0 (env) → 1 (warm-start) → 2 (regularizer) → 3 (LAUNCH long run, user gate) → 4 (eval best ckpt, user gate) → 5 (render). Tasks 0-2 are buildable in-session; Task 3 launches a run that trains hours-days OUT of session; Task 4 evaluates whatever checkpoint exists (may be interim). A no-sprint outcome is a valid recorded result. Tasks 3 & 4 are user gates.
