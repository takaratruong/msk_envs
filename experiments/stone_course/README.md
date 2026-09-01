# Randomized stepping-stone walking

This experiment trains the muscle-actuated sprinter model on an endless stream
of raised box slabs. Terrain, rather than a foot-placement reward, makes bad
steps inconvenient: a missed slab drops the model to the lower ground and ends
the episode as a fall.

The task is deliberately minimal:

- reward capped forward velocity and staying upright;
- give the policy the root-relative centers of the next four slabs;
- reject edge-pivot touchdowns unless the complete foot-contact footprint is on
  the slab top;
- terminate failures when the pelvis falls or turns away from the course; and
- use the ordinary time limit as a neutral truncation, with no final slab.

There is no heel-strike reward, toe penalty, target-foot assignment, gait phase,
or imitation objective. Heel strike is free to emerge from the physical task.

## How per-world terrain works

Bolt's model contains one shared set of five slab collider definitions. Its data
contains a separate local transform for every `(world, collider)` pair. At each
reset, `StoneCourseEnv` samples a new five-slab buffer for only the resetting
worlds and writes those rows of the transform tensor.

During an episode, an inactive slab that is at least 0.15 m behind the pelvis is
moved beyond that world's furthest slab. Its new forward gap and lateral jitter
are sampled at that moment. Collider IDs therefore rotate through the course,
while observations sort their current positions and expose the nearest four
upcoming slabs. A supporting slab is never recycled.

Consequently, every world has the same collider count and shapes but different
placements. Broadphase and narrowphase still run independently inside each
world; a foot in world 7 cannot collide with a slab in world 12. No model or
collision-filter variant is needed for each placement.

Default course geometry:

| Setting | Default |
| --- | ---: |
| Recycled slabs per world | 5 |
| Slab size (forward × vertical × lateral) | 0.36 × 0.10 × 0.36 m |
| Slab top height | 0.45 m |
| Initial center spacing | uniform in 0.40–0.55 m |
| Final curriculum spacing | uniform in 0.40–0.85 m |
| Curriculum promotion | +0.05 m to upper bound |
| Launch-pair center spacing | uniform in 0.32–0.38 m |
| Alternating lateral centerline | ±0.12 m |
| Regular lateral jitter | uniform in ±0.10 m |
| Observation lookahead | 4 slab centers (8 values) |
| Target walking speed | 1.4 m/s |
| Maximum episode duration | 12 s |

The first pair is intentionally easier so the run-start pose begins supported.
Every recycled slab receives a fresh forward and lateral placement. The lower
spacing bound remains fixed at 0.40 m. After a window of 1,024 episodes, the
upper bound increases by 0.05 m when at least 60% survived the full 12 seconds
and traveled at least 12 m. Difficulty never decreases and caps at 0.85 m.
TD3 checkpoints retain this curriculum state, and evaluation environments copy
it from training before each rendered rollout.

## Interior landing rule

A touchdown is interior only when every contact sphere belonging to that foot
projects completely onto the same slab top:

```text
abs(foot_x - slab_x) <= slab_half_length - contact_radius
abs(foot_z - slab_z) <= slab_half_width  - contact_radius
```

At each new left or right touchdown, contact without whole-foot interior support
is an edge-pivot failure. Inactive toe and heel spheres still count toward the
footprint, so a single safe sphere cannot excuse the rest of the foot hanging
over an edge. The first 0.25 seconds after reset are exempt so the launch pose
can settle. This is an episode constraint, not a shaped gait reward.

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
last. For example, this lets the curriculum expand to 0.95 m:

```bash
CUDA_DEVICE=0 NUM_ENVS=1024 ./experiments/stone_course/train.sh \
  --env-config.course-step-length-range 0.40 0.95 \
  --env-config.course-initial-step-length-max 0.55
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

They cover layout validation, independent sampling, inactive-only per-world
recycling, physical-transform synchronization, sorted root-relative
observations, interior landing semantics, neutral timeouts, curriculum
promotion, and both heel geometry/material definitions.

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
