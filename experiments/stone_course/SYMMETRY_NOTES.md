# Symmetry status and known limits

## Verified (2026-09-07)

- `sprinter_model_sym.osim` is exactly L/R mirror-symmetric in all authored
  data: 68 muscle pairs (paths, scalars), bodies, inertias, joint frames,
  contact spheres, wrap objects, obstacle ellipsoids and contact hints.
  Fixes applied: vas_int + pectoralis_major_sup scalars averaged across
  sides, radius_hand_l weld mirrored from the right, pec_major_sup_l Scholz
  insertion z flipped (was copied unmirrored).
- The flip utilities in `experiments/stone_course/symmetry.py`
  (`MirrorSpec.flip_obs`, `flip_action`, `flip_qpos`, `flip_qvel`,
  `mirror_world_layout`) round-trip every directly-mirrorable observation
  block at exactly 0.0 error against a trained policy: course terrain,
  muscle activations, actuator activations, qpos, qvel, actions.

## Known residual (future work)

The strict behavioral test (flipped policy in mirrored world reproduces the
identical mirrored rollout) still FAILs with ~3% fiber-length error at t=0,
confined to the 18 arm muscles using `Scholz2015GeometryPath` (pec/lat/
biceps/triceps groups). The model data for these is mirror-exact; the
residual is Bolt's iterative geodesic path solver converging to slightly
different curved-path solutions from mirrored initializations. Chaos
amplifies this to ~0.4 m root divergence over 2 s.

Possible future step: make Bolt's Scholz geodesic solver deterministic
under mirroring (e.g. canonicalize the initialization by side, or solve one
side and mirror the solution). Until then, mirrored transitions are exact
for all leg mechanics and approximate (~3%) for arm-muscle fiber state —
acceptable as data augmentation, insufficient for exact-equivalence claims.

## Observation: one-foot hopping basin (2026-09-07)

riser3sym (control, no augmentation) converged early onto single-leg
hopping: its iter-16000 full-episode eval spent 73% of frames on the right
foot only, 1% left, 0% double support. Verified NOT a model artifact: the
symmetrization touched leg anatomy by at most 0.015% (vas_int) and made the
legs exactly equal. Hopping is simply a valid basin of the minimal
vel+alive objective - one leg to coordinate, no weight transfer.

symaug (identical recipe + mirrored replay) shows near-balanced contacts
at iter 3000 (34% L / 40% R): mirrored data prevents one-sided strategies
from consolidating, since every right-leg experience trains the left too.
Expectation: terrain difficulty (long gaps, steep elevation) should
eventually price hopping out for the control as well; either way the A/B
now doubles as a gait-symmetry experiment with a pathological baseline.

## Finding: knee-cap experiment (2026-09-07, stonecourse_kneecap)

Capping the knee coordinate at 0.0 rad (no hyperextension) did NOT inhibit
learning: the capped lane matched its uncapped counterpart's pace (48-54%
completion, full platform height by ~13k iterations) and produced visually
excellent straight-knee walking.

However the mechanism is the same exploit relocated: the policy braces
against the hard stop at exactly straight and uses the leg as a passive
fulcrum, just as the uncapped model braces at +10 degrees. Design decision:
keep hyperextension allowed (standard sym model). If the straight-knee
aesthetic is wanted without the limit-bracing exploit, the honest levers are
late-phase effort costs (lambda_limit on limit forces, or the metabolic
term), not geometry caps. Keeper checkpoint of the good-looking gait:
models/stonecourse_kneecap_keeper.pt.

## Observation: staged lane leads per-checkpoint (2026-09-07, user)

Visual note from watching the live cards: stonecourse_staged shows the best
gait quality for its checkpoint of any lane so far, including clear PUSH-OFF
behavior with the back foot (active ankle propulsion at toe-off) - the first
lane where that has been visible. Quantitatively consistent: staged reached
the 1.12 m / ±30° terrain rung in ~25k iterations vs riser6's ~62k under the
same final rules (landing termination + annealed margin), 2.5x faster via the
stage-sequenced schedule (free learning -> placement -> terrain).

Recipe of the lane: symmetric model, mirrored replay augmentation, stone-gated
reward, stones-only passing, rising platform, staged landing rule
(course_defer_landing_to_full_height), annealed margin 0.06 -> 0.01/promotion.
