# IMG_2296 reconstruction workspace

The [final independent audit](../../runs/real2sim-2296/20260912T0614Z/reviews/final/report.md)
passed 3,312 integrity checks and confirmed the candidate's unmet acceptance.
It found a four-camera / 211-point SfM component at 45.47–46.04 seconds whose
alignment to the main 188-camera component relies on the learned pose prior.
This additional connectivity failure is detailed in [REPORT.md](REPORT.md).

This experiment reconstructs the user's iPhone lab-kitchen MOV for an eventual
Unitree G1 environment. No usable recorded depth or camera-pose stream was
identified; this pipeline treats the capture as **monocular RGB**. The current
output is one delivered OpenUSD reconstruction candidate with partial camera
coverage, completed Gaussian appearance and observed/inferred diagnostic meshes.
The final composition rendered in Isaac at two source cameras. **The original
exact-replica and deployment goal is not met.** Appearance generalization and
contact surfaces fail the proposed gates; metric scale, articulation, physical
behavior, real LiDAR localization and G1 task transfer remain unverified.

- [Methods and bottlenecks report](REPORT.md), [acceptance](ACCEPTANCE.md),
  [plan](PLAN.md), [tool survey](TOOLS.md), [native USD export](USD_EXPORT.md)
- Source: `/home/ubuntu/Downloads/IMG_2296.MOV`
- Source SHA256: `282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9`
- Current run: `runs/real2sim-2296/20260912T0614Z/` (`r2`).
- Earlier split and reusable environments: `runs/real2sim-2296/20260912T0540Z/`
  (`r1`); its reconstructions are not the current model.
- Selected appearance: `r2/appearance_v2/`, completed step 30,000; its native
  `appearance.usdz` is exported. The larger-capacity alternative is preserved
  separately under `r2/appearance_capacity/` and is not selected.
- Intended authority: one OpenUSD composition for Isaac Sim/Isaac Lab. Blender
  is an optional authoring tool. A common rigid transform may level the
  source-fitted floor for viewing; it introduces no scale and does not certify
  gravity. USD unit metadata is an encoding convention until scale is measured.

## Current reconstruction path

At the 2026-09-12 08:55 UTC report snapshot, `model_stable` contains **192/259**
selected training cameras and 26,746 sparse points. The saved model hashes match
its metric report: median reprojection 0.591 px, p95 2.191 px. Coverage remains
below the proposed 95% target, and the p95 exceeds the proposed 2 px target.
These image residuals do not measure real surface accuracy.

The source inventory identifies two islands and three visible sink stations:
two along the perimeter and one in an island. Repeated labels on the islands,
repeated fixtures at the two perimeter sinks and weak floor matches
made earlier COLMAP maps physically inconsistent, including maps with many
registered cameras and low residuals. The revised path uses local VGGT
inference on all **259 training views only**, triangulates the original SIFT
tracks in that inferred camera frame, applies robust bundle adjustment, and
iteratively removes weakly constrained cameras and unsupported below-floor
points. The 259 inferred cameras are priors, not 259 independently solved poses.
The original `facebook/VGGT-1B` checkpoint is recorded as CC-BY-NC-4.0 and is used
here for a research/noncommercial pilot; its exact revision and hashes are in
[REPORT.md](REPORT.md). This is not a commercial deployment permission claim.

Fixed-map RGB camera localization attempted all **36** held-out selections:
**25 localized, 11 rejected**. The current dataset therefore has 192 training
images and 25 localized validation images. All 36 remain in the review's
denominator: interleaved 20/28, buffered 5/8.

| Completed output | Evidence and limit |
|---|---|
| Dense stereo | 662,374 fused points; source/dataset/cloud hashes match the post-run receipt. Median distinct camera support is 4; fusion's `min_num_pixels=5` is a pixel threshold. |
| Observed mesh | 520,847 vertices, 784,079 triangles and 359,965 boundary edges; not watertight. Sampled image-space ray coverage is only 2.07–43.26%. Collision remains disabled. |
| Inferred mesh | Separate learned-depth TSDF hypothesis: 22,438,032 triangles from 160/192 accepted views. Image-ray coverage rises to 95.60–99.91%, but independent floor controls expose false surfaces. Hidden and collision-disabled. |
| Selected Gaussian baseline | 999,236 Gaussians at step 30,000. Held-out full/static PSNR 17.96/22.12 dB on 25/36 views; static SSIM 0.8650. Buffered static PSNR 17.30 dB on 5/8. Appearance acceptance is unmet. |
| Capacity continuation | Continued from baseline step 10,000 with optimizer/RNG/strategy state preserved and a 3M cap; finished with 1,583,730 Gaussians. Training improved slightly, validation worsened. Not selected. |
| Native final export | Selected baseline USDZ is bound to its raw float32 PLY; no normalization. Maximum position-component float16 drift is 0.0009706 raw units, with no verified metric interpretation. |
| Final native comparison | Actual canonical 30k scene rendered at original-resolution training and buffered cameras; both exit 0 with black hidden-volume controls. Native/gsplat PSNR 37.50/38.56 dB; buffered native/source static PSNR only 16.19 dB. |
| Source inventory | 68 entries / 102 observations, distinguishing three sinks, two islands and two service carts. The packager uses USD observation Scopes for evidence; it does not generate detached object bodies or measured joints. |

The first `appearance_stable` attempt stopped on nonfinite loss at step 2,533.
Revised SSIM disables cuDNN TF32 moment calculations, passed independent
value/gradient checks, and completed under `appearance_fp32` → `appearance_v2/`.
The failed attempt remains preserved. A separate three-view query-pose fit is
diagnostic only; it uses query RGB and does not alter the held-out scores or
canonical cameras. The learned-depth geometry pilot is retained as a separate
diagnostic hypothesis because the observed MVS mesh is incomplete. Its improved
coverage does not certify physical accuracy.

The canonical composition is now
[`r2/scene/scene.usda`](../../runs/real2sim-2296/20260912T0614Z/scene/scene.usda),
with 217 source cameras, selected Gaussian appearance, both hidden diagnostic
surfaces and the observation inventory. Render assets, source records and camera
transforms are bound to the completed evidence. Independent core composition
and final native-render checks passed; the assembled portable documentation has
its own finalization receipt. The [report](REPORT.md) records the evidence and
remaining gates.

## Inspect the current evidence

- [Final source / gsplat / native Isaac comparisons](../../runs/real2sim-2296/20260912T0614Z/final_native/index.html)
  and [60-file evidence receipt](../../runs/real2sim-2296/20260912T0614Z/final_native/evidence.json)
- [Earlier intermediate step-5,000 native preview](../../runs/real2sim-2296/20260912T0614Z/previews/step_005000/index.html)
- [Completed baseline source comparisons](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/eval/step_030000_all/comparison.mp4)
  — sampled-camera playback at 6 fps, not source real time
- [Observed-mesh coverage review](../../runs/real2sim-2296/20260912T0614Z/geometry/review/index.html)
- [Inferred-mesh source review](../../runs/real2sim-2296/20260912T0614Z/geometry_prior/review/index.html)
  and [independent false-floor-hit audit](../../runs/real2sim-2296/20260912T0614Z/reviews/prior1/report.md)
- [Source object inventory and crops](../../runs/real2sim-2296/20260912T0614Z/semantic_inventory/index.html)
- [Independent SSIM repair audit](../../runs/real2sim-2296/20260912T0614Z/reviews/ssim_v2/report.md)

The baseline's worst buffered kitchen views contain substantial floaters and
smearing outside screen masks. At 60.238 seconds, the sink/counter is almost
absent from the observed mesh. These limitations remain visible in the evidence;
the native rendering bridge does not certify reconstruction or physics.

## Run controls

Run from `/home/ubuntu/msk_envs-stone-course`:

```bash
python experiments/real2sim_2296/jobctl.py --root runs/real2sim-2296/20260912T0614Z status dense_stable
python experiments/real2sim_2296/jobctl.py --root runs/real2sim-2296/20260912T0614Z status mesh_stable
python experiments/real2sim_2296/jobctl.py --root runs/real2sim-2296/20260912T0614Z status appearance_fp32
python experiments/real2sim_2296/jobctl.py --root runs/real2sim-2296/20260912T0614Z status appearance_capacity
python experiments/real2sim_2296/jobctl.py --root runs/real2sim-2296/20260912T0614Z logs appearance_fp32
```

Jobs retain attempt directories, immutable entrypoint copies, code hashes,
timestamps and logs. The current pointer is not proof of success: inspect its
status, output files and verification receipts. The controller also supports
`stop NAME` and `restart NAME`; these change the named job and are not status
queries. `restart` retries the owning stage with its original arguments, so
inspect completed artifacts before retrying. Never reuse a run directory for
a different capture or changed split.
Use an explicit command separator, e.g. `start --gpu 2 dense -- <command>`.
Only named task processes are controlled. Existing simulation jobs are untouched.

## Environment isolation

The reusable environments remain in **r1**, although outputs now go to r2:

| Environment | Interpreter | Verified packages |
|---|---|---|
| Reconstruction and review | `runs/real2sim-2296/20260912T0540Z/venv/bin/python` | PyCOLMAP CUDA12 4.2.0, Open3D 0.19.0, usd-core 25.11 |
| Appearance / learned inference | `runs/real2sim-2296/20260912T0540Z/appearance-venv/bin/python` | Torch 2.4.1+cu124, gsplat 1.5.3+pt24cu124 |

Pinned CUDA-compatible wheels avoid compiling these extensions with the host's
CUDA 13 toolchain. The existing Isaac installation is used separately for
native rendering at
`/home/ubuntu/projects/simtoolreal/repo/.venv_isaacsim/bin/python` (Isaac Sim 5.1).
The synthetic capability test is documented in [USD_EXPORT.md](USD_EXPORT.md);
the actual final and intermediate comparisons and process receipts are linked above.

## Evidence notes

Current `frames.json` contains **303** sharp source selections at four time bins
per second, preserving actual PTS, source indices and image hashes:

| Selection | Count | Policy |
|---|---:|---|
| Training | 259 | Excludes validation and temporal buffers |
| Interleaved validation | 28 | Original every-tenth-bin selections outside buffered blocks |
| Buffered validation | 8 | Actual PTS in [48,49) and [70,71) seconds |
| Excluded buffer | 8 | Remaining selections in [47.5,49.5) and [69.5,71.5) seconds |

The initial r1 `inspection/contact_sheet.jpg` has approximate bin-start labels
and is **superseded**. Use the PTS contact sheets or generate a review directly
from the current manifest. The review script does not decode the MOV again:

```bash
runs/real2sim-2296/20260912T0540Z/venv/bin/python \
  experiments/real2sim_2296/review.py \
  --run runs/real2sim-2296/20260912T0614Z \
  --model-name model_stable \
  --appearance runs/real2sim-2296/20260912T0614Z/appearance_v2 \
  --scene runs/real2sim-2296/20260912T0614Z/scene \
  --output /tmp/real2sim2296-review-stable
```

The scene argument names the composed package directory. Without `--output`, the static HTML is written to
`RUN/review/index.html`. It includes source previews, raw camera trajectories,
full cloud extent, validation denominators and unmet acceptance gates.

Both validation groups come from the same capture. Even successful fixed-map
RGB localization and good image scores do not establish real LiDAR performance,
metric accuracy or G1 task transfer. See the report's acceptance table before
calling any later scene final.
