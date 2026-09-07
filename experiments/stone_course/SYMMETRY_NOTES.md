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
