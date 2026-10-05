# Reconstruction plan and decision gates

## 1. Inspect before reconstructing
Hash the MOV and record ffprobe + Apple metadata atom audit. Decode frames
with actual timestamps; inspect samples across the entire 75.57 s path and
close-ups of low-texture, reflective, occluded and potentially movable regions.
Inventory objects and note unavailable evidence. The user has confirmed there
are no depth/pose sidecars and the target is G1.

## 2. Establish a defensible camera/map baseline
Use an isolated local environment and an available GPU. Choose sharp frames
in short time bins, retaining source indices and PTS; freeze evaluation before
optimization. The current 303-frame split contains 259 training frames,
28 interleaved holdouts, 8 holdouts in buffered 48–49 s and 70–71 s blocks,
and 8 buffer frames excluded from fitting and scoring. Extract SIFT, match local temporal windows
and long-range revisit candidates, geometrically verify and run COLMAP bundle
adjustment with estimated shared lens parameters; verify the shared-lens
assumption rather than treating the focal prior as calibration. Inspect component count,
registered fraction, reprojection residuals and focal plausibility. Use
additional matches or an independent learned initializer only if evidence
shows the ordinary baseline fails. Keep every attempt and its metrics.

## 3. Derive observed surfaces and appearance in the same frame
Undistort the registered training views. Run multiview stereo with geometric
consistency and confidence filtering. Preserve points, normals, visibility and
camera trajectory. Inspect missing regions before meshing; never close gaps
across doors/sinks/glass to manufacture coverage. Train a Gaussian appearance
layer using only training images once cameras pass the baseline. Evaluate
held-out RGB by fixed-map camera localization, and explicitly report pose
failures, changing TVs, reflection artifacts and unseen-region uncertainty.
Compare meshed surfaces and neural geometry only as hypotheses unless measured.

## 4. Iterate on actual failure regions
Review registered/unregistered timestamps and worst held-out errors. Refine
frame selection, matching or calibration if the camera graph is deficient;
mask dynamic imagery where observed; use a surface-oriented method if a splat
surface is inadequate. Keep code/config/source hashes and per-attempt receipts.
Do not spend indefinitely fitting hidden or blank regions unsupported by data.

## 5. Assemble one coherent scene package
Use one canonical manifest and coordinate transform for cameras, appearance,
observed geometry, semantic instances, future collision solids and
articulations. Prefer OpenUSD for scene composition and simulator handoff;
provide common mesh/point/splat formats. Mark reconstruction units as arbitrary
until independent scale/gravity anchors exist. A preview asset is not enabled
as a certified collider. Only derive simulator dynamics from measured or
explicitly accepted uncertain properties; keep unsupported layers unverified.

## 6. Promote to manipulation/localization quality when evidence permits
Acquire independent metric anchors and close scans of G1 contact areas,
door/drawer travel and hidden surfaces. Identify target G1 hand and LiDAR,
calibrate sensor extrinsics, model materials and validate object collisions
through measured support/grasp/slide/articulation trials. Compare simulated
LiDAR and localization against held-out real scans, then conduct matched real
and simulated tasks. Apply every gate in ACCEPTANCE.md. These steps cannot be
replaced by visual inspection of the same RGB video.

## 7. Deliver and independently check
Provide deterministic run controls, source/code/asset hashes, source-vs-render
comparisons, a local visual review, the canonical scene package and REPORT.md
describing what worked, iterations, failures and exact unmet gates. Join an
independent red-team review whose hash matches final code and artifacts. State
clearly whether the requested final environment passes; never label a partial
preview an exact or deployment-ready twin.
