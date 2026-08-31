# Randomized stepping-stone walking

This experiment trains the muscle-actuated sprinter model to cross a sequence
of raised box slabs. Terrain, rather than a foot-placement reward, makes bad
steps inconvenient: a missed slab drops the model to the lower ground and ends
the episode as a fall.

The task is deliberately minimal:

- reward capped forward velocity and staying upright;
- give the policy the root-relative centers of the next four slabs;
- terminate failures when the pelvis falls or turns away from the course; and
- truncate successfully as soon as a contacting foot reaches the final slab.

There is no heel-strike reward, toe penalty, target-foot assignment, gait phase,
or imitation objective. Heel strike is free to emerge from the physical task.

## How per-world terrain works

Bolt's model contains one shared set of 24 slab collider definitions. Its data
contains a separate local transform for every `(world, collider)` pair. At each
reset, `StoneCourseEnv` samples a new layout for only the resetting worlds and
writes those rows of the transform tensor.

Consequently, every world has the same collider count and shapes but different
placements. Broadphase and narrowphase still run independently inside each
world; a foot in world 7 cannot collide with a slab in world 12. No model or
collision-filter variant is needed for each placement.

Default course geometry:

| Setting | Default |
| --- | ---: |
| Slabs per world | 24 |
| Slab size (forward × vertical × lateral) | 0.36 × 0.10 × 0.36 m |
| Slab top height | 0.45 m |
| Regular center spacing | uniform in 0.40–0.70 m |
| Launch-pair center spacing | uniform in 0.32–0.38 m |
| Alternating lateral centerline | ±0.12 m |
| Regular lateral jitter | uniform in ±0.10 m |
| Observation lookahead | 4 slab centers (8 values) |
| Target walking speed | 1.4 m/s |
| Maximum episode duration | 10 s |

The first pair is intentionally easier so the run-start pose begins supported.
Every later slab, including its forward and lateral placement, is independently
resampled for each world and episode.

## Walking heel contact

`right_foot_7` and `left_foot_7` use the walking-model heel geometry:

- 50 mm sphere radius at local position `(0.01, 0.0018, 0)` m;
- its bottom is aligned with foot contacts 3–6 at `-0.0482` m; and
- stiffness 500 kN/m with static/dynamic friction 0.8/0.8.

Only the heel contact changed. The other sole contacts and muscle model remain
the sprinter defaults.

## Run the experiment

First install Bolt and this repository's requirements as described in the root
[README](../../README.md). Run commands from the repository root in the Python
environment that contains Bolt.

The convenience launcher uses 1,024 environments on physical GPU 0 by default:

```bash
CUDA_DEVICE=0 NUM_ENVS=1024 ./experiments/stone_course/train.sh
```

The equivalent explicit command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m msk_envs.train.train stonecourse \
  --disable-wandb \
  --exp-prefix stonecourse_walkingheel \
  --algo td3 \
  --gpu-id 0 \
  --td3-config.num-envs 1024 \
  env-config:sprinter
```

The environment-model subcommand and all `--env-config.*` overrides must come
last. For example, this runs 32 somewhat wider-spaced slabs:

```bash
CUDA_DEVICE=0 NUM_ENVS=1024 ./experiments/stone_course/train.sh \
  --env-config.course-stones 32 \
  --env-config.course-step-length-range 0.50 0.80
```

Useful launcher variables are `CUDA_DEVICE`, `NUM_ENVS`, and `EXP_PREFIX`. Set
`RESUME=1` to resume the latest checkpoint matching `EXP_PREFIX` in the current
worktree's `models/` directory.

## Outputs and evaluation

Training periodically writes:

- checkpoints to `models/<experiment-name>/`;
- TensorBoard events to the same directory; and
- evaluation trajectories to `dashboard/trajectories/<experiment-name>/`.

Launch the repository viewer with:

```bash
cd dashboard
python dashboard.py
```

The logged trajectory contains the actual box colliders. There is no circular
target overlay, so the viewer and offline renderer show the same slab geometry
used by physics.

## Verify before a long run

Run the focused CPU/tensor checks:

```bash
python -m unittest discover -s tests -p 'test_stone_course*.py' -v
python -m unittest discover -s tests -p 'test_walking_heel_contacts.py' -v
```

They cover layout validation, independent sampling, selective per-world reset,
physical-transform synchronization, root-relative observations, final-slab
success semantics, and both heel geometry/material definitions.

For a low-cost integration check, start with `NUM_ENVS=2`. Building the first
Bolt model may compile and cache Warp kernels, so the initial startup is slower
than later runs.

## Code map

- `msk_envs/envs/env_stone_course.py`: course specification, layout sampling,
  collider updates, observations, reward, and episode boundaries.
- `msk_envs/envs/env_config.py`: user-facing `course_*` defaults and CLI flags.
- `msk_envs/train/hyperparams.py`: the `stonecourse` training preset.
- `msk_envs/msk_models/sprinter/sprinter_model.osim`: heel sphere geometry.
- `msk_envs/msk_models/sprinter/contact_params/contact_params.yaml`: heel
  contact material.
- `tests/test_stone_course.py` and `tests/test_walking_heel_contacts.py`:
  focused regression coverage.
