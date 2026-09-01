# Randomized stepping-stone walking

This experiment trains the muscle-actuated sprinter model on an endless stream
of raised box slabs. Its 3D target curriculum is inspired by Xie et al.'s
[ALLSTEPS](https://arxiv.org/abs/2005.04323), while retaining this repository's
minimal locomotion objective. Terrain, rather than a foot-placement reward,
makes bad steps inconvenient: a missed slab drops the model to the lower ground
and ends the episode as a fall.

The task is deliberately minimal:

- reward capped forward velocity and staying upright;
- give the policy the 3D surface centers and roll/pitch of the next four slabs;
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
moved beyond that world's furthest slab. Its new 3D target and surface tilt are
sampled at that moment. Collider IDs therefore rotate through the course, while
observations sort their current positions and expose the nearest four upcoming
slabs. A supporting slab is never recycled.

Consequently, every world has the same collider count and shapes but different
placements. Broadphase and narrowphase still run independently inside each
world; a foot in world 7 cannot collide with a slab in world 12. No model or
collision-filter variant is needed for each placement.

Default course geometry:

| Setting | Default |
| --- | ---: |
| Recycled slabs per world | 5 |
| Slab size (forward × vertical × lateral) | 0.36 × 0.10 × 0.36 m |
| Initial slab top height | 0.45 m |
| Allowed slab-top height | 0.20–1.05 m |
| Initial radial step distance | uniform in 0.65–0.80 m |
| Final radial step distance | uniform in 0.65–1.50 m |
| Final forward turn range | ±20° |
| Final elevation range | ±50° within the absolute height bounds |
| Final slab roll/pitch | independently uniform in ±20° |
| Promotion increments | +0.14 m distance, +10° elevation, +4° turn/tilt |
| Launch-pair center spacing | uniform in 0.32–0.38 m |
| Alternating lateral centerline | ±0.12 m |
| Regular lateral jitter | uniform in ±0.10 m |
| Observation lookahead | 4 surfaces (20 values) |
| Target walking speed | 1.35 m/s |
| Maximum episode duration | 12 s |

The first pair is intentionally easier so the run-start pose begins supported;
the first three reset slabs are flat and straight. Later slabs are generated in
spherical coordinates relative to their predecessor: radial distance,
forward-biased yaw, and elevation. The alternating lateral offset is retained
so the course still presents natural left/right footholds.

After a window of 1,024 episodes, all four difficulty bounds expand when at
least 60% survived the full 12 seconds and traveled at least 12 m. Difficulty
never decreases. This first phase is forward locomotion with moderate turning;
sideways, backwards, and fully omnidirectional target generation are deliberately
left for a later phase. TD3 checkpoints retain curriculum state, and evaluation
environments copy it from training before each rendered rollout.

The policy observation preserves the original eight root-relative X/Z values,
then appends four root-relative top heights and eight roll/pitch values. This
ordering permits a flat-course policy to be migrated with zero weights for the
12 new inputs.

## Interior landing rule

A touchdown is interior only when every contact sphere belonging to that foot
projects completely onto the same slab top. The test is evaluated in the
slab's tilted local frame:

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
  --exp-prefix stonecourse_allsteps_forward \
  --algo td3 \
  --gpu-id 0 \
  --td3-config.num-envs 1024 \
  env-config:sprinter
```

The environment-model subcommand and all `--env-config.*` overrides must come
last. For example, this limits the final radial distance to 1.20 m:

```bash
CUDA_DEVICE=0 NUM_ENVS=1024 ./experiments/stone_course/train.sh \
  --env-config.course-step-length-range 0.65 1.20 \
  --env-config.course-initial-step-length-max 0.80
```

Useful launcher variables are `CUDA_DEVICE`, `NUM_ENVS`, and `EXP_PREFIX`. Set
`RESUME=1` to resume the latest checkpoint matching `EXP_PREFIX` in the current
worktree's `models/` directory.

To preserve an existing eight-feature flat-course gait, first expand its actor,
critics, and observation normalizer:

```bash
python experiments/stone_course/expand_checkpoint_observation.py \
  models/<flat-run>/<flat-checkpoint>.pt \
  models/stonecourse_allsteps_forward_warmstart.pt
```

Then launch the forward 3D curriculum from it:

```bash
CUDA_DEVICE=0 NUM_ENVS=1024 EXP_PREFIX=stonecourse_allsteps_forward \
WARMSTART_CHECKPOINT=models/stonecourse_allsteps_forward_warmstart.pt \
  ./experiments/stone_course/train.sh
```

The migration retains the old X/Z and body features exactly, initializes the
new policy/critic columns to zero, and resets terrain curriculum state to the
flat, straight 0.65–0.80 m stage.

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

They cover layout validation, bounded 3D sampling, surface rotations,
independent per-world recycling, physical-transform synchronization, sorted
lookahead observations, tilted local-frame landing semantics, neutral timeouts,
curriculum promotion, checkpoint migration, and both heel geometry/material
definitions.

For a low-cost integration check, start with `NUM_ENVS=2`. Building the first
Bolt model may compile and cache Warp kernels, so the initial startup is slower
than later runs.

## Code map

- `msk_envs/envs/env_stone_course.py`: course specification, layout sampling,
  collider updates, observations, reward, and episode boundaries.
- `msk_envs/envs/env_config.py`: user-facing `course_*` defaults and CLI flags.
- `experiments/stone_course/expand_checkpoint_observation.py`: flat-policy
  warm-start migration.
- `msk_envs/train/hyperparams.py`: the `stonecourse` training preset.
- `msk_envs/msk_models/sprinter/sprinter_model.osim`: heel sphere geometry.
- `msk_envs/msk_models/sprinter/contact_params/contact_params.yaml`: heel
  contact material.
- `tests/test_stone_course.py` and `tests/test_walking_heel_contacts.py`:
  focused regression coverage.
