# IMG_2296 real-to-simulation reconstruction

## Problem and desired outcome
Reconstruct the scene observed in `/home/ubuntu/Downloads/IMG_2296.MOV` into
one coherent environment for humanoid manipulation, simulation, and real
LiDAR localization. The user requests photographic appearance, accurate
geometry, collision meshes, individually identifiable task objects and
articulations, a measured comparison to the capture, and a report of methods
and bottlenecks.

## Starting evidence and prior attempts
The supplied MOV is approximately 75.57 seconds of 1920×1080 HEVC, about
30 fps, recorded on an iPhone 15 Pro Max. Initial ffprobe reports RGB, audio,
and four Apple metadata tracks; recorded metric depth is not established.
No earlier reconstruction of this capture was found in this checkout.
The checkout has unrelated user changes and active simulation workloads.

## Constraints and non-goals
Keep this work inside `experiments/real2sim_2296/` and
`runs/real2sim-2296/`; preserve existing project work and processes.
Use local compute and locally stored capture data. Do not publish or upload
the private capture. Pin bounded jobs to an available GPU and give long jobs
run/status/logs/stop/restart controls. Do not invent metric accuracy, hidden
surfaces, material parameters, joint limits, or real-world validation.
One world frame must own rendering, geometry, collisions, semantics and
exported simulators. A visually plausible generated room is not acceptance.

## Observable acceptance (to refine after visual inspection)
1. Source is hashed; streams, orientation, timing, camera data and depth are
   audited. Source views and coverage are documented.
2. Camera reconstruction covers the observed scene in one consistent frame,
   with reprojection and independently held-out view diagnostics.
3. Appearance is checked against held-out video frames; measured geometry,
   scale and alignment are checked against independent metric observations.
4. Collision geometry, task objects, movable parts and articulations have
   explicit identities, provenance and uncertainty; dynamic objects do not
   remain baked into the static visual scene.
5. Simulator import, contact, grasp/support, articulation, sensor frames and
   LiDAR localization tests pass for the user's robot and sensor configuration.
6. A single versioned scene package, evidence, reproducible commands and a
   candid final report describe passed and unmet criteria separately.

## Verification and metrics
Proposed thresholds and test procedures will be written in ACCEPTANCE.md.
No self-consistency metric alone certifies real-world accuracy. Keep held-out
views separate from optimization. Independent red-team passes are bound to
source/code snapshot hashes; join a final matching pass before completion.

## Unresolved decisions
The user confirmed only the MOV is available and the eventual robot is G1.
The independent container audit found no recorded depth or camera trajectory.
What real dimensions can anchor metric scale? Which G1 hand, LiDAR model,
manipulation objects, joints and workspace define task-specific tolerances?
What independent real trials establish
simulation-to-real performance correlation?

## Initial path
Inspect the full video and metadata; inventory visible objects and coverage;
write numerical acceptance and current tool comparison; run calibrated
multiview reconstruction; evaluate evidence and iterate before deriving
rendering/collision/semantic layers and simulator exports. Any unavailable
evidence remains an explicit unmet gate rather than an assumed success.
