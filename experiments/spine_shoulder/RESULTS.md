# Spine + Shoulder model on the Stepping-Stone task — Results

Goal: get a spine-and-shoulder model running on the stepping-stones task and verify it is correct and behaves correctly.

## What was built
A variant of the sprinter, `msk_models/sprinter/sprinter_model_spineshoulder.osim`, built by
`experiments/spine_shoulder/build_sprinter_spineshoulder.py` from `sprinter_model_sym.osim`:

- **Flexible spine (the "tube"):** the single stiff 3-DOF `back` joint is replaced by a
  4-joint lumbar chain `pelvis → lumbar1 → lumbar2 → lumbar3 → torso`, each a 3-DOF CustomJoint
  (extension/bending/rotation). 9 new lumbar DOF + the retained `torso_*` triplet. Torso mass
  carved 26.83 → 19.63 kg (3 × 2.4 kg segments), total conserved. The ribcage/torso stays rigid.
- **Mobile scapula (Seth form):** `scapulothoracic_r/l` WeldJoints → stock OpenSim `EllipsoidJoint`s
  (Bolt's ELLIPSOID mobilizer — the cheap Seth 2019 formulation, scapula rides the ribcage
  ellipsoid by construction, no constraint solver). Ribcage radii 0.083/0.20/0.083 m. 6 new
  scapula DOF (`scap_{abduction,elevation,uprot}_{r,l}`).
- Passive `SpringGeneralizedForce` + `CoordinateLimitForce` on every new coordinate (the model
  has no serratus/trapezius/rhomboid, so the girdle is passively stabilized).
- Muscle paths: `sprinter_model_spineshoulder_fn.xml` drops the 8 trunk-crossing muscles that
  referenced the old `back` coordinate; they fall back to Scholz2015 geometric point-paths that
  correctly follow the multi-segment chain. 128 leg fn-paths untouched.

## Correctness (verified in Bolt — the training engine)
`verify_model.py` + `calibrate_scapula.py`:
- Loads: nq=47, 28 bodies, all coordinates resolved.
- **Neutral pose preserved:** torso Δ=0.000 m; scapula & humerus within 3 mm of the welded
  baseline (2.6 mm = physical ellipsoid-surface curvature, expected).
- **Scapula is mobile and rides the ellipsoid:** 0.118 m displacement across an abduction sweep.
- **Spine bends:** torso moves 0.219 m under a combined lumbar-bend sweep.
- Muscle lengths finite at neutral (0.056–0.591 m).
- Ellipsoid parent-frame math: R_p = R0 (X_torso→scapula), t_p = p0 − R0·(0,0,semi_z).

## Behaves correctly (stepping-stone task)
- Full StoneCourse env builds with the new model: obs (N, 386) finite, reset + step finite.
- **Training run** `stonecourse_spineshoulder` (tmux, GPU3, TD3, 1024 envs, cuda-graph,
  from scratch; run dir `models/stonecourse_spineshoulder_2026-09-09_10-20`) is stable and
  **learning**: mean episode length 15.1 → 16.8 and velocity reward 0.97 → 1.07 over the first
  ~1700 iterations; alive reward pinned at 1.0; no NaNs / divergence.
- Behavior harness `verify_behavior.py` (cuda-graph fast path) on the iter-1000 checkpoint
  (receipt `verify_behavior_1000.log`, 64 envs × 200 steps):
  - **qpos finite for the entire 200-step rollout** (no divergence with a real policy driving it).
  - The freed DOFs are actively used: scapula excursions abduction 0.48 / elevation 0.99 /
    upward-rot 1.44 rad; lumbar bending ~1.4–1.6 rad per segment; shoulder flexion 1.3–1.6 rad.
  - L/R shoulder-flexion correlation ≈ −0.02 at this early checkpoint — not yet the clean
    anti-phase arm swing of a mature gait (expected; iter 1000 has just learned to stay upright
    and inch forward, mean 0.39 m/200 steps). This is the metric to re-check on a longer-trained
    checkpoint to confirm the freed girdle produces more natural arm swing than the welded model.

## Files
- Build: `experiments/spine_shoulder/build_sprinter_spineshoulder.py`
- Verify (Bolt correctness): `verify_model.py`, `calibrate_scapula.py`
- Env smoke: `env_smoke.py`
- Behavior: `verify_behavior.py`
- Model: `msk_models/sprinter/sprinter_model_spineshoulder{.osim,_fn.xml}`
- Config: `EnvConfigSprinterSpineShoulder` (subcommand `env-config:spineshoulder`)
- Symmetry: lumbar bending/rotation added to negate sets; scapula coords L/R-paired.

## Update 2026-09-09: intervertebral stiffness (fixes spine buckling)
Behavior check on the first run revealed the 3-segment lumbar chain was **buckling**: each
lumbar joint swung 1.4-1.6 rad although its range is only ±0.44 rad. Root cause: the trunk
muscles attach only at the endpoints (pelvis, torso) so they control just the TOTAL trunk
angle -- the intermediate segments were internally underdetermined, held by damping alone
(stiffness=0), and the coordinate-limit forces were 9x too soft (1.0 vs baseline 8.73).
Fix (no new muscles): added per-segment restoring stiffness `LUMBAR_STIFFNESS=40 N*m/rad`
(disc/ligament surrogate) via SpringGeneralizedForce, and raised CoordinateLimitForce to the
baseline scale (8.73). Verified Bolt loads it (`dof_stiffness[lumbar1_bending]=40`), neutral
pose still preserved (torso Δ=0.000, scapula 3mm). Retrained run:
`stonecourse_spineshoulder_stiff` (GPU3, run dir `_stiff_2026-09-09_12-27`); the pre-fix run
is archived as `models/stonecourse_spineshoulder_nostiff_2026-09-09_10-20`.

Behavior at iter-1000, stiffened vs pre-fix (both early/exploratory checkpoints):
  lumbar bending excursion:  1.4-1.6 rad  ->  1.0-1.1 rad   (buckling much reduced)
  forward distance/200 steps: 0.39 m      ->  0.51 m        (better task progress)
  qpos finite whole rollout:  yes          ->  yes           (stable)
So stiffness=40 substantially reduced the buckling and improved gait, but excursions still
slightly exceed the ±0.44 rad (0.87 total) nominal range at this near-random checkpoint --
expected during early exploration, but if a longer-trained checkpoint still overshoots, raise
LUMBAR_STIFFNESS further (e.g. 80) and/or steepen the CoordinateLimitForce transition.

## Update 2026-09-09 (later): REGRESSION found vs baseline — ablation running
The full spine+shoulder model does NOT match the baseline skeleton's task performance.
At comparable iterations, curriculum promotion (the real "behaves correctly" bar):
  - baseline sym model (omni/backward runs): curriculum_completion_rate 0.43-0.69, forward
    progress 6-8 m, promotes past level 0.
  - spine+shoulder (stiffened) @45k: completion_rate **0.0 the entire run**, stuck at
    curriculum level 0, forward progress only ~1.25 m. It learns to stand/shuffle (return
    1.1->6.6, episode length 11->55) but never crosses a course, so it never gets promoted.
So the added complexity is a regression, not yet a win. Prime suspects from behavior data:
scap_uprot swings ~2.4 rad (floppy passively-stabilized scapula) and lumbar still overshoots
range — a wobbling trunk + windmilling girdle likely prevents committed locomotion.
ABLATION launched to isolate the cause (both env-configs + models added, symmetric build via
`--spine/--shoulder` flags on build_sprinter_spineshoulder.py):
  - `stonecourse_spineonly`   (env-config:spineonly,   GPU2): multi-seg spine, welded scapula.
  - `stonecourse_shoulderonly`(env-config:shoulderonly,GPU6): ellipsoid scapula, single back joint.
Whichever recovers baseline curriculum promotion identifies the culprit; compare
curriculum_completion_rate + mean_forward_progress against the baseline at matched iterations.

## Update 2026-09-09 (CRITICAL correction): wrong curriculum — earlier "regression" was invalid
The "stuck at level 0 / regression vs baseline" conclusion was an artifact of MY MISTAKE:
all my runs used bare `StoneCourseConfig` dataclass defaults, which are the HARD end-state
curriculum, while every working baseline run overrides ~13 curriculum fields on the CLI.
Diff of my run vs the canonical forward run (symaug), most damaging first:
  course_terminate_on_ground_contact: True (mine) vs False  -> my model died on ANY ground touch
  course_initial_height_scale:        1.0  (mine) vs 0.1    -> full-height stones from step 1 (no ramp)
  course_curriculum_height_scale_increment: 0.0 vs 0.15     -> height dimension disabled -> no promotion path
  course_terminate_below_supports:    True vs False
  course_stone_gated_reward:          False vs True
  course_curriculum_require_stone_support: False vs True
  course_continuation_probability:    0.0 vs 0.95
  course_stones: 5 vs 8; step_length_range (0.65,1.5) vs (0.4,1.5); initial_step_length_max 0.8 vs 0.7;
  require_interior_landing True vs False; landing_margin_inactive 0.0 vs 0.02; recycle_distance_behind 0.15 vs 2.0
FIX: added `_ForwardCourseCurriculumMixin` (values from stonecourse_symaug_launch.log) and mixed
it into all three configs (spineshoulder/spineonly/shoulderonly) so they inherit the PROVEN
forward curriculum by construction. Verified resolved values (height_scale 0.1, term_on_ground
False, gated True, 8 stones). Relaunched all three (exp suffix `_fc`); wrong-curriculum runs
archived with `_wrongcurric`. The spine/shoulder-vs-baseline comparison is only valid from these
`_fc` runs onward.

## Known follow-ups
- `thorax_capsule*` / `lumbar_spine_capsule2` colliders skipped (not in geom lookup) — harmless
  for locomotion, revisit if trunk contact matters.
- Arm behavior comparison vs the welded baseline needs a longer-trained checkpoint to be
  conclusive (first checkpoint learns to stay alive and move forward; arm-swing style emerges later).
- `experiments/stone_course/model_lookups.json` is the baseline 32-coord layout; regenerate for
  the 47-coord model before any tool that depends on it.
