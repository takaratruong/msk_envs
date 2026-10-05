# IMG_2296 real2sim: methods, evidence and bottlenecks

**Final audit addendum (09:12 UTC):** The
[independent integrated audit](../../runs/real2sim-2296/20260912T0614Z/reviews/final/report.md)
passed 3,312 integrity checks across 1,575 files, preserving the distinction
between successful native rendering and unmet environment acceptance. It also
identified **two surviving SfM camera components: 188 and 4 cameras**. The small
component contains `frame_001364`, `frame_001367`, `frame_001373` and
`frame_001381` at 45.471667–46.038333 seconds, with 211 points. Its alignment to
the main component relies on the shared learned pose prior; surviving feature
tracks do not constrain that alignment. The single connected reconstruction
gate therefore also fails. The
[connectivity receipt](../../runs/real2sim-2296/20260912T0614Z/reviews/final/connectivity.json)
binds this finding to the delivered model. No scene, camera, geometry or
appearance assets were changed by this addendum.

This report records the reconstruction work and its limits as of **2026-09-12
08:55 UTC**. The run delivered **one OpenUSD reconstruction candidate**, including
the selected 30,000-step Gaussian appearance, observed and inferred diagnostic
surfaces, source-linked object observations and 217 prepared cameras. The exact
composition rendered successfully in Isaac Sim at a training and a buffered
validation camera. **The original exact-replica and deployment goal is not met.**
Held-out appearance has substantial errors; inferred geometry adds coverage but
also false surfaces. Metric scale, usable collision, articulation, LiDAR and
G1 task transfer remain unverified. Completed artifacts and failed acceptance
gates are distinguished below. Independent reviews verify the final native
rendering path and core composition, while retaining the reconstruction's
quality failures. The assembled documentation/portable-evidence receipt is
tracked separately from those payload checks.

The authoritative run is `runs/real2sim-2296/20260912T0614Z/` (`r2`). Paths below
refer to that run unless stated otherwise. The earlier
`runs/real2sim-2296/20260912T0540Z/` (`r1`) retains the original split, rejected
attempts and reusable Python environments. Existing results were preserved
instead of being silently relabeled as later attempts.

## Capture and intended outcome

The source is `/home/ubuntu/Downloads/IMG_2296.MOV`, SHA256
`282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9`.
The saved [probe](../../runs/real2sim-2296/20260912T0540Z/inspection/ffprobe.json)
reports 75.573 seconds, 2,267 HEVC video frames, 1920 × 1080 pixels, approximately
30 fps, AAC audio and four Apple metadata tracks. No usable recorded depth or
camera-pose stream was identified. The reconstruction therefore uses RGB; the
phone's depth capability does not supply missing range observations.

The footage shows a lab kitchen with two distinct islands and **three visible
sink stations: two along the perimeter and one in an island**. The room also
contains dispensers, cabinets, drawers, appliances, stools, round tables, chairs,
carts, glass partitions, lights, displays and smaller equipment. The
[source inventory](../../runs/real2sim-2296/20260912T0614Z/semantic_inventory/index.html)
contains 68 entries and 102 source observations, including visible parts and
explicitly unresolved regions. Parent/child entries are not independent object
counts. This is not exhaustive segmentation or a mechanical inventory. Cabinet
interiors, undersides, occluded surfaces and joint motion remain incompletely
observed.

The requested destination is one OpenUSD composition for eventual Unitree G1
pick/place simulation and real LiDAR map localization. Photorealistic image
agreement, metric surface accuracy, correct physical interaction and real/sim
task correlation are distinct acceptance gates. None substitutes for the
others.

The [tool survey](TOOLS.md) compares classical and learned reconstruction,
Gaussian/surface methods, RGB-D mapping and simulation formats. The exercised
path combines COLMAP/PyCOLMAP, VGGT, gsplat, Open3D and native Isaac rendering.
OpenUSD is the intended composition authority; Blender remains an optional
authoring tool. MuJoCo and Genesis were considered as simulation destinations,
but no IMG_2296 scene has been validated in those engines.

## Source selection and validation protocol

[frames.json](../../runs/real2sim-2296/20260912T0614Z/frames.json) contains 303
sharp selections: the highest measured image sharpness within four time bins
per second. Every selected image retains its source frame index, actual PTS,
source identity and image SHA256. The source MOV hash was rechecked for this
report. The current frame-manifest SHA256 is
`7eecd606f8f4fcf24d0bb177ff799ffcf9d234c6c87e862ddd4d14d1b559a284`.

| Partition | Selected images | Role |
|---|---:|---|
| Training | 259 | Camera/structure and appearance optimization |
| Interleaved validation | 28 | Same-trajectory image interpolation check |
| Buffered validation | 8 | Actual PTS in [48,49) and [70,71) seconds |
| Excluded temporal buffer | 8 | Remaining frames in [47.5,49.5) and [69.5,71.5) seconds |

The split was fixed before the r2 reconstructions. Cached descriptors and pair
verification were copied, but no earlier poses or 3D points were adopted as a
new solution. Mapping and learned initialization use training images only.
Validation poses are estimated afterward against the frozen training map;
failed pose estimates remain in the 36-image denominator. A model trained on
all 259 selected training images still has not seen the validation or buffer
images.

An early contact sheet used approximate bin-start captions and is superseded.
Later PTS sheets and [review.py](review.py) use the actual frame manifest. The
review hashes its inputs, retains missing time intervals and shows both full
point-cloud bounds and a separately labeled magnified display crop. It does
not normalize reconstruction coordinates or infer timestamps from frame rate.

## What was tried and why the camera solution changed

The first path used local SIFT features, temporal and revisit matching,
geometric verification, COLMAP reconstruction and shared SIMPLE_RADIAL camera
parameters. It produced apparently plausible global statistics but physically
wrong associations. The [first rejected-model audit](../../runs/real2sim-2296/20260912T0614Z/reviews/model_aliasing/report.md)
traced 86 shared points to matching compost labels on different islands, and
77 to matching recycle labels on different islands. Three cameras also escaped
millions of arbitrary units while retaining subpixel reprojection errors.

Masking repeated labels and rejecting narrowly supported nonlocal image pairs
removed those known control correspondences. The camera masks could suppress
ambiguous static labels without erasing those labels from appearance images.
Changing displays and visible dynamic regions received separate masks, with
source-view corrections when automatic detections missed screen strips or
confused dispensers with displays. Mask remapping requires all bilinear RGB
contributors to be valid, avoiding mixed dynamic colors at mask boundaries.

These changes improved specific controls but did not resolve all aliases. A
later 166-camera map had a connected co-visibility graph, compact camera
extent and subpixel median error, yet source inspection exposed eight gross
temporal discontinuities. Some adjacent estimates rotated by 100–173 degrees
while the source showed small motions. Distinct sink paper dispensers and
faucets still shared incorrect landmarks. Those wrong-sink controls compare
the two perimeter stations (frames 854/1785 versus 275/330); the island sink
visible in frame 2174 is a third distinct station. Weak floor texture also supported
bad connections. These findings and source overlays are in the
[subsequent camera-failure audit](../../runs/real2sim-2296/20260912T0614Z/reviews/camera_failure/report.md).

| Attempt | Cameras / selected training | Sparse points | Median / p95 reprojection, px | Interpretation |
|---|---:|---:|---:|---|
| r1 `model` | 233 / 273 | 28,460 | 0.785 / 2.214 | Rejected: wrong island identity and extreme weak component |
| r1 `model_v2` | 252 / 273 | 29,330 | 0.814 / 2.243 | High registration count; diagnostic attempt, not promoted |
| r2 `model` | 226 / 259 | 33,254 | 0.831 / 2.262 | Not promoted after source/camera checks |
| r2 `model_candidate_auto` | 55 / 259 | 6,875 | 0.640 / 2.030 | Rejected fragmented candidate |
| r2 `model_final` | 166 / 259 | 21,471 | 0.787 / 2.207 | Rejected: wrong sink identities and gross pose flips; filename does not mean accepted |
| r2 `model_vggt_all` | 259 inferred priors / 259 | 31,369 triangulated | Not an acceptance measurement | Learned initialization only |
| r2 `model_refined` | 204 / 259 | 27,164 | 0.594 / 2.195 | Intermediate robust refinement; 198 cameras with broad track support |
| r2 `model_stable` | **192 / 259** | **26,746** | **0.591 / 2.191** | Current partial-coverage candidate after floor/support pruning |

Historical values are saved reports, not remeasured real-world errors. Current
COLMAP reads confirm the r2 camera/point counts. Both refined-model metric
reports contain five binary artifact hashes, all of which matched their
respective current models during this report audit. Older metric formats lack
that complete binding; their source-specific rejection evidence remains tied
to the snapshots named in the audit reports.

## Learned initialization, robust refinement and floor evidence

Local VGGT trials used 24, then 64, then all 259 training views. The all-view
[inference receipt](../../runs/real2sim-2296/vggt-pilot/inference_all/result.json)
binds the selected-frame list, camera predictions and current frame manifest.
It records source uploads as false. Learned depth and confidence were kept as
predictions, not measurements. The exported learned point cloud is a diagnostic
and is not a certified collision mesh.

The exact original `facebook/VGGT-1B` checkpoint revision is
`860abec7937da0a4c03c41d3c269c366e82abdf9`; its local SHA256 was independently
rechecked as `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
The [official model card](https://huggingface.co/facebook/VGGT-1B/blob/main/README.md)
and saved local card identify **CC-BY-NC-4.0**. This run uses those weights for a
research/noncommercial pilot and makes no commercial deployment permission
claim. The separately recorded VGGT code commit is
`a288dd0f14786c93483e45524328726ab7b1b4ce`; checkpoint terms must not be inferred
from a different code or model release.

[seed_from_poses.py](seed_from_poses.py) inserted the predicted cameras into a
copied COLMAP database while preserving original SIFT keypoint indices. It
triangulated image correspondences in the learned frame, using a shared focal
prior. These 259 inserted poses were explicitly treated as hypotheses.
[refine_seed.py](refine_seed.py) then used SOFT_L1 bundle adjustment with
successive 8, 5 and 3 pixel observation filters. Weakly supported poses were
held fixed during optimization, retained in an all-poses diagnostic copy, and
excluded from the working model if they lacked at least 30 observations or a
2.5% image-area convex hull. Pruning is repeated because removing one camera
can reduce another camera's usable support.

The working `model_stable` is pruned from `model_refined_all_poses`. Manually
inspected visible-floor polygons identified triangulated floor candidates;
[floor_evidence.json](../../runs/real2sim-2296/20260912T0614Z/floor_evidence.json)
records the fitted plane and source-model hashes. The prune-only pass checked
27,164 common point coordinates against that reference, removed 292 points
more than 0.1 **raw reconstruction units** below the fitted floor, and performed
three support-pruning rounds. It retained 192 cameras, all meeting the reported
broad-support criterion. The conservative temporal test reports no gross jump;
this does not establish absence of smaller pose errors or all false matches.

The floor fit uses RGB-derived SfM points. It is neither a measured gravity
vector nor a surveyed plane, and its numerical residuals are not millimetres.
The same bound floor hypothesis is used by geometry processing. Subsequent
dense checks expose local departures from that plane, described below.

## Fixed-map RGB localization

The prepared [dataset](../../runs/real2sim-2296/20260912T0614Z/dataset.json)
contains 192 undistorted training views. The completed
[RGB localization report](../../runs/real2sim-2296/20260912T0614Z/evaluation/localization.json)
attempted every held-out selection:

| Validation group | Attempted | Localized | Rejected |
|---|---:|---:|---:|
| Interleaved | 28 | 20 | 8 |
| Buffered blocks | 8 | 5 | 3 |
| Total | **36** | **25** | **11** |

The report's source, frame-manifest, model path, binary artifact and mask-receipt
hashes are checked against current inputs. It reports robust PnP against fixed
training-map points with spatial-support and temporal consistency rejection;
it does not update the map with validation observations. Its 25 usable poses
populate the appearance dataset's validation list. All 11 failures must remain
visible beside image metrics. This is RGB camera localization from the same
video, not a real LiDAR localization trial or an independently surveyed pose
benchmark.

## Appearance optimization, numerical repair and capacity experiment

The appearance path uses gsplat with the 192 fixed prepared cameras and static
masks. Neither camera optimization nor coordinate normalization is enabled.
The first `appearance_stable` attempt failed at 07:36:37 UTC with
`Nonfinite training loss at step
2533`, recorded in its [owned log](../../runs/real2sim-2296/20260912T0614Z/jobs/appearance_stable/20260912T073409792738Z/log.txt).
The last logged splat count was 909,131 at step 2,525. Periodic checkpoint files
were saved through step 2,000 and remain explicitly rejected optimization
evidence.
The observed exception is numerical, not a reported out-of-memory exception.

The reconstruction lane's isolated test of the actual step-2,000 checkpoint on
`frame_001280.jpg` reported full-image SSIM of −2.55077 with cuDNN TF32 versus
0.56694 when it was disabled. These are diagnostic values from the failed
model, not accepted appearance scores. The revised code disables TF32 for
cuDNN globally and explicitly inside SSIM, clamps tiny negative moment
variances, guards invalid SSIM values and records distinct `fp32ssim-v2`
training semantics. Disabling matrix-multiplication TF32 alone had not covered
the cuDNN convolutions used for SSIM moments.

The [independent SSIM audit](../../runs/real2sim-2296/20260912T0614Z/reviews/ssim_v2/report.md)
passed on the corrected implementation and actual rejected checkpoint/source
frame. Corrected full-frame SSIM was 0.566931 with global cuDNN TF32 either true
or false. Three actual patches agreed with float64 references within 4.34e-5;
gradient relative L2 error was at most 0.00589, all gradients were finite, and
invalid masked pixels received zero gradient. The audit did not independently
reproduce the old kernel-dependent negative score in its second execution
context; preserved invalid training logs and the reconstruction lane's
reproduction are distinguished from the independently tested repair.

The [fresh job configuration](../../runs/real2sim-2296/20260912T0614Z/jobs/appearance_fp32/20260912T074102372712Z/config.json)
records launch at 07:41:02 UTC on physical GPU 2 with maximum width 1920.
`appearance_v2/` completed all **30,000 steps** at 08:09:02 UTC and exported
**999,236 Gaussians**, under a one-million growth cap. Its archived trainer SHA256 is
`1c3f5a3f5e32481c38f75c256f9db485fdea6c3813b25aa900d0a4879319d9d7`.
The original `appearance/` outputs and failed attempt remain separate.

A separate `appearance_capacity/` experiment continues from the baseline's
step-10,000 checkpoint with a three-million growth cap. The
[continuation metadata](../../runs/real2sim-2296/20260912T0614Z/appearance_capacity/metadata.json)
records that exact parent checkpoint, unchanged training inputs and preserved
optimizer, strategy and random-generator states. The code permits this explicit
cap increase in a fresh output directory while rejecting unrelated configuration
changes. A cap is a limit, not an achieved primitive count. This continuation
completed at 08:13:05 UTC with **1,583,730 Gaussians**. Its
[final evaluation](../../runs/real2sim-2296/20260912T0614Z/appearance_capacity/eval/step_030000_all/metrics.json)
uses the same frozen inputs as the baseline. The completed **baseline
`appearance_v2` is selected** for the canonical scene because its aggregate
held-out measures are better. The capacity continuation remains an evaluated
alternative and is not selected.

The [continuation audit](../../runs/real2sim-2296/20260912T0614Z/reviews/capacity1/report.md)
matched the parent checkpoint and all 74 paired logged sample choices through
step 11,850, supporting restored sampling state. It identified later-resume
lineage and stale-output guard gaps; a
[subsequent 23-test pass](../../runs/real2sim-2296/20260912T0614Z/reviews/capacity2/report.md)
verified those repairs and checkpoint-lineage serialization. The completed
experiments ran their archived implementations with their original, correctly
recorded parent/input identities.

The completed baseline's
[217-view evaluation](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/eval/step_030000_all/metrics.json)
uses **192 training images at 1888 × 1061** and **25 localized validation images
at 1920 × 1080**. Values below are per-view arithmetic means.
PSNR uses uncropped RGB in [0,1]; SSIM uses an 11-pixel Gaussian window with
sigma 1.5. Full-image LPIPS uses the official pretrained AlexNet variant.
Static LPIPS was not implemented and is **unavailable**, not zero.
Appearance rejection follows the static PSNR and SSIM gates. Full-image LPIPS
is diagnostic and does not substitute for the specified static-region LPIPS gate.

| Baseline, step 30,000 | Rendered / selected | Full PSNR, dB | Static PSNR, dB | Full / static SSIM | Full LPIPS |
|---|---:|---:|---:|---:|---:|
| Training | 192 / 259 | 29.04 | 30.56 | 0.9335 / 0.9405 | 0.2095 |
| All validation | **25 / 36** | **17.96** | **22.12** | **0.8422 / 0.8650** | **0.3516** |
| Interleaved validation | 20 / 28 | 18.52 | 23.33 | 0.8545 / 0.8781 | 0.3348 |
| Buffered validation | 5 / 8 | 15.71 | 17.30 | 0.7930 / 0.8125 | 0.4189 |

| Capacity comparison, both step 30,000 | Baseline: 1M cap | Continuation: 3M cap |
|---|---:|---:|
| Final Gaussian count | 999,236 | 1,583,730 |
| Training static PSNR | 30.56 dB | 30.85 dB |
| Validation full / static PSNR, 25 of 36 | 17.96 / 22.12 dB | 17.74 / 21.78 dB |
| Validation full / static SSIM | 0.8422 / 0.8650 | 0.8382 / 0.8608 |
| Validation full LPIPS | 0.3516 | 0.3538 |
| Buffered static PSNR, 5 of 8 | 17.30 dB | 16.51 dB |

The larger representation fits training images slightly better but worsens
these aggregate held-out measures. Increasing primitive capacity alone did
not resolve the generalization problem. This is a continuation experiment,
not an independent from-scratch trial, and the same validation views informed
this comparison; they are not an untouched final test set.

Good training-image scores do not carry over to the buffered views. Source
comparison at [23.935 seconds](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/eval/step_030000_all/00080_val.jpg)
shows smearing and floaters around two service carts; its static PSNR is
15.21 dB. At [48.371667 seconds](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/eval/step_030000_all/00149_val.jpg),
the fridge, island and cabinets are recognizable but heavily overlaid by
incorrect appearance; static PSNR is 16.14 dB. These errors persist outside
changing screens. Their cause can include remaining pose/calibration errors,
weak support and Gaussian overfitting; the present evidence does not isolate
each contribution. The evaluation's 6 fps comparison movie uses one sampled
camera per frame and does not reproduce source playback timing.

A separate [query-pose diagnostic](../../runs/real2sim-2296/20260912T0614Z/query_pose_diagnostic/results.json)
held the final Gaussian tensors and dataset fixed while fitting only six pose
parameters to each of three query RGB images for 250 iterations. Static PSNR
changed from 15.21 to 17.33 dB for the cart view, from 28.52 to 29.51 dB for a
cabinet view, and from 16.14 to 16.53 dB for the buffered kitchen view. This
shows that a modest pose adjustment can improve same-query image fit while
leaving the severe buffered-view failure largely unresolved. Such a fit may
compensate for camera, appearance or geometry errors; it does not establish
which cause dominates or measure physical pose accuracy. Because it fits the same query RGB that is scored,
these are **diagnostic optimization values, not held-out acceptance scores**.
No fitted query pose was written into the canonical dataset.

## Dense reconstruction and observed surface

`dense_stable` completed COLMAP PatchMatch photometric/geometric stereo and
fusion on physical GPUs 3, 4 and 6, with maximum image size 1920. It produced
192 maps of each depth/normal type and **662,374 fused points**. The fused file
was written at 08:00:32 UTC and the owned job exited successfully at 08:00:33.
The [dense provenance receipt](../../runs/real2sim-2296/20260912T0614Z/dense_provenance.json)
binds source, dataset, prepared images/masks, sparse models, configurations,
depth/normal files, fused PLY and visibility data. It is explicitly a
**post-run consistency audit**, not a retroactive preflight capture. Its source
poses, shared point coordinates and prepared intrinsics agree exactly with the
frozen model/dataset in the recorded comparisons.

Fusion used `min_num_pixels=5`, a contributing-**pixel** threshold. It does not
guarantee five distinct supporting cameras. The
[independent visibility audit](../../runs/real2sim-2296/20260912T0614Z/reviews/fused1/results.json)
of `fused.ply.vis` found a minimum of **2** and median of **4** distinct views per
point; **333,423 of 662,374 points (50.3%)** have fewer than five. The full cloud
has finite positions and normals, but these checks establish data integrity,
not surface accuracy.

An [independent audit of 11 geometric-depth maps](../../runs/real2sim-2296/20260912T0614Z/reviews/dense1/report.md)
unprojected camera-Z depths
using the saved intrinsics and half-pixel centers, then checked membership in
source-inspected floor polygons. Near frame 393, source PTS **13.102 seconds**,
the median absolute departure from the common floor plane is **0.01968 raw
units**, with p95 0.02438; the floor patch is systematically above that plane.
Frame 500 has only **6.8% valid depth coverage inside its visible-floor polygon**.
The below-floor rejection does not repair an above-floor warp or missing
surfaces. This is internal consistency against an RGB-derived plane; it does
not supply metric scale, measured gravity or surveyed surface error.

`mesh_stable` completed at 08:06:57 UTC. Its
[geometry receipt](../../runs/real2sim-2296/20260912T0614Z/geometry/metrics.json)
matches the current dense receipt and fused hash. Filtering removed one dense
point more than 0.1 raw units below the floor, no points at the runaway guard,
and 5,789 statistical outliers; voxel reduction left **619,629 meshing points**.
Open3D ball-pivoting produced **520,847 vertices and 784,079 triangles**, with
**359,965 boundary edges**, zero nonmanifold edges and `watertight=false`.
The [independent mesh audit](../../runs/real2sim-2296/20260912T0614Z/reviews/mesh1/report.md)
reproduced those counts and found that the mesh is also **not vertex-manifold**,
with **17,830 connected triangle components**. It rehashed all 1,369 files bound
by the dense receipt, including the maps, and reproduced all 17 source-camera
ray-hit fractions exactly. Successful integrity checks do not make this a
usable collision surface. Ball pivots can also bridge unsupported local gaps;
no global hole completion was applied.

The [source/mesh ray review](../../runs/real2sim-2296/20260912T0614Z/geometry/review/index.html)
shows the missing surfaces directly. Across its 17 diagnostic views, full-image
ray-hit coverage ranges from **2.07% to 43.26%**. At
[60.238 seconds](../../runs/real2sim-2296/20260912T0614Z/geometry/review/12_frame_001807.jpg),
the perimeter sink, counter and cabinets are almost absent from the mesh. These
are image-space triangle-hit fractions, not measured completeness of all
physical surfaces or a LiDAR return simulation. The mesh remains diagnostic,
with collision certification and articulation separation explicitly false.

The completed [learned-depth pilot](../../runs/real2sim-2296/20260912T0614Z/geometry_prior/README.md)
adds a separate **inferred** surface. Each VGGT camera-Z depth image is aligned
to current training tracks with a positive multiplicative scale, median-log
fitting and MAD rejection, without an additive shift. The fit requires at least
20 inlier tracks spanning 2% of image area and accepts **160 of 192** working
views. Predicted pixels are mapped through the original SIMPLE_RADIAL camera
to the current undistorted rays. Camera poses and the common reconstruction
frame remain unchanged; aligning predicted depth to arbitrary SfM units does
not measure physical scale. Confidence is an uncalibrated score, thresholded
by percentile, not a probability.

TSDF fusion at width 640 produced **12,281,334 vertices and 22,438,032 triangles**,
with 1,938,826 reported boundary edges and 155 nonmanifold edges. The
[17-camera source review](../../runs/real2sim-2296/20260912T0614Z/geometry_prior/review/index.html)
has **95.60–99.91% image-ray hit coverage**, recovering much of the missing
counter and tabletop area. The surface still has noise, thick or inaccurate
thin parts, a misshaped partly open cabinet door, and inaccurate sink/faucet
surfaces. Its higher hit coverage does not establish correct geometry.

The [independent prior audit](../../runs/real2sim-2296/20260912T0614Z/reviews/prior1/report.md)
verified the actual input bindings and camera/depth coordinate convention but
found false first-hit surfaces over clearly visible floor. Against the same
internally derived floor plane, fused floor p95 errors reach **0.09123 raw units**
at 36.037 seconds; at 75.540 seconds the median offset is 0.01797 raw units and
median normal deviation 29.70 degrees. These are correlated internal controls,
not surveyed distances. The prior is selected **only as a hidden,
collision-disabled diagnostic hypothesis** alongside the observed MVS surface.
Neither surface is accepted for navigation collision, LiDAR localization or
G1 contacts. The original VGGT checkpoint's noncommercial restriction also
applies to this research pilot.

## One native OpenUSD scene and actual native rendering evidence

The completed canonical entrypoint is
[`scene/scene.usda`](../../runs/real2sim-2296/20260912T0614Z/scene/scene.usda),
with the selected Gaussian appearance, both diagnostic meshes, source-linked
observation Scopes and all **217 prepared cameras** in one composition.
[scene_package.py](scene_package.py) applies one shared rigid transform derived
from the source-fitted floor to make the preview Z-up. The transform has no
scale component; both directions are stored in the
[manifest](../../runs/real2sim-2296/20260912T0614Z/scene/manifest.json).
Its render assets and source-observation files match their bound hashes. Raw
data remain available. USD
`metersPerUnit=1` is explicitly an encoding convention, not a measurement.
Physical gravity remains unverified. Both meshes are hidden and their
CollisionAPI is explicitly disabled; no active physics or articulated G1
environment is claimed.

The [independent composition audit](../../runs/real2sim-2296/20260912T0614Z/reviews/package1/report.md)
checked all 217 camera names/poses/intrinsics, every vertex/index/normal/color
in both mesh layers and all 68 observation Scopes. They match the selected
source data and common transform. A separate
[19-case finalizer test](../../runs/real2sim-2296/20260912T0614Z/reviews/finalizer1/report.md)
verified guards against attaching another model, mismatched surface dataset
or unrelated creation job as provenance. These are composition/integrity checks;
they do not accept the reconstructed geometry or photorealism.

[export_splats_usd.py](export_splats_usd.py) converts standard Gaussian PLY into
the supported legacy NuRec USDZ representation using a pinned official 3DGRUT
template. It preserves the raw coordinate scale and reports float16 position
drift relative to the authoritative float32 PLY. Plain splat PLY is not itself
a native Isaac Gaussian asset. Root render settings must be applied explicitly
because referenced layer metadata does not automatically become root metadata.
See [USD_EXPORT.md](USD_EXPORT.md) for the format and exact commands.

The selected final baseline has now been exported as
[`appearance_v2/appearance.usdz`](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/appearance.usdz).
Its [export receipt](../../runs/real2sim-2296/20260912T0614Z/appearance_v2/appearance.usdz.receipt.json)
binds all **999,236 Gaussians** to the completed float32 PLY, the converter and
official template commit `a37ef721012dea0f29c0fcfff2d525023b4e854a`.
No normalization was applied. Maximum XYZ component quantization drift is
**0.0009706 raw units**, maximum point L2 drift **0.0010971**, and p95 L2 drift
**0.0004083**. The float32 PLY remains authoritative. USD opening, payload/tensor
round trips and package checks passed; this export receipt explicitly does
not claim native rendering of the final asset.

A **synthetic** three-color SH3 Gaussian package rendered through installed
Isaac Sim 5.1 on physical GPU 4. Hiding its NuRec volume produced an all-black
control; 48,657 pixels changed by more than eight intensity levels. All 16 files
and four inputs named in the archived
[check summary](../../runs/real2sim-2296/20260912T0614Z/reviews/isaac_smoke/check_summary.json)
matched their hashes in this audit. The capture phase completed in 33.2 seconds,
but full plugin shutdown exceeded the 180-second watchdog and was terminated.
The archived tested script and later shutdown-only adjustment are distinguished
in the [smoke README](../../runs/real2sim-2296/20260912T0614Z/reviews/isaac_smoke/README.md).
This established the renderer's capability before exporting the real scene.

The subsequent [actual-scene preview](../../runs/real2sim-2296/20260912T0614Z/previews/step_005000/index.html)
exported a frozen **step-5,000** `appearance_v2` checkpoint, containing 706,759
Gaussians, into a separate native USDZ and wrapper. The source camera is
`frame_001499.jpg` at **49.971667 seconds**. Both Gaussian appearance and camera
use the common rigid floor transform; raw PLY positions and higher SH
coefficients exactly match the checkpoint. The composed camera's projection
agrees with the raw camera to within 0.000696 pixels in the recorded test.
Float16 export maximum XYZ component drift is **0.0009742 raw units**, maximum
point L2 drift 0.0010457 and p95 L2 drift 0.0004086. These are not metric errors.

At 640 × 360, native Isaac versus gsplat is **32.98 dB full-image PSNR**, with
mean absolute RGB error **1.90/255**. Native versus source is 22.98 dB full-image
and 25.89 dB static PSNR. No camera fitting, image warp or color correction was
applied. Native framing and overall color agree with the gsplat image, while
source-relative blur and floaters remain. Hiding the NuRec volume makes the
control image all black; **99.93%** of pixels change by more than eight intensity
levels. The owned Isaac 5.1 job on physical GPU 4 completed with **exit code 0**
at 07:55:19 UTC, within its 250-second watchdog, using the shutdown adjustment.
All 42 artifacts in its
[evidence receipt](../../runs/real2sim-2296/20260912T0614Z/previews/step_005000/evidence.json)
matched their hashes during this report audit. This demonstrated the actual
rendering bridge on one supported **training view at an intermediate step**.
The [independent native-preview audit](../../runs/real2sim-2296/20260912T0614Z/reviews/native1/report.md)
also reproduced the source resizing, masks and pixel metrics, checked the USD
camera and rigid shared frame, and verified all decoded tensors against
float16 casts of the original PLY, including the SH3 channel ordering.

The **selected final asset** subsequently rendered through the exact canonical
stage at two original-resolution cameras. The
[final comparison gallery](../../runs/real2sim-2296/20260912T0614Z/final_native/index.html)
and [evidence receipt](../../runs/real2sim-2296/20260912T0614Z/final_native/evidence.json)
bind 60 portable media, comparison, input and job files. Both Isaac jobs exited
0 within their 250-second watchdogs; the process audit found no owned processes
remaining. They used physical GPU 4, explicit root render settings, no AA and
the original fixed cameras. No image warp, color fit or pose optimization was
applied. Scores below compare saved 8-bit PNGs; their rounding differs slightly
from the float evaluation above.

| Final native check | Training: frame 1499 | Buffered validation: frame 1451 |
|---|---:|---:|
| Source PTS / resolution | 49.971667 s / 1888 × 1061 | 48.371667 s / 1920 × 1080 |
| Native vs gsplat, full PSNR | 37.50 dB | 38.56 dB |
| Native vs gsplat, mean absolute RGB error | 1.40 / 255 | 1.51 / 255 |
| Native vs source, static PSNR | 30.10 dB | **16.19 dB** |
| Hidden-control pixels changed by more than 8 | 99.827% | 99.948% |
| Completed job exit | 0 | 0 |

![Final training source, gsplat and native Isaac](../../runs/real2sim-2296/20260912T0614Z/final_native/frame_001499/comparison_small.png)

![Final buffered source, gsplat and native Isaac](../../runs/real2sim-2296/20260912T0614Z/final_native/frame_001451/comparison_small.png)

Both hidden-volume controls are all black. The USD camera projections agree
with the prepared cameras within **0.00001 pixels** on the tested in-frame
sparse points. Native and gsplat framing/color agree closely, while the severe
buffered-view floaters and ghosting remain in both. This isolates a successful
native rendering path from an unsuccessful appearance reconstruction in that
view. Two camera checks do not replace the full 25/36 validation denominator.
The stage and all Gaussian/mesh/semantic payloads remained byte-identical;
portable documentation and provenance additions are recorded separately in
[the metadata update receipt](../../runs/real2sim-2296/20260912T0614Z/final_native/package_metadata_update.json).
Later attachment of render evidence changes packaging metadata, not the
rendered stage or its visual assets.
The [independent final native audit](../../runs/real2sim-2296/20260912T0614Z/reviews/native_final1/report.md)
passed 184 input/artifact hash checks, verified the exact source/mask pixels and
cameras, reproduced pixel metrics within 1.45e-7 dB, and confirmed both black
controls and successful process exits. It visually confirmed the buffered-view
failure. This pass binds the actual final stage, payloads and images; later
portable documentation has its own finalization receipt.

The observed mesh is intended as a hidden diagnostic layer with collision
explicitly disabled. The source inventory's
[verification](../../runs/real2sim-2296/20260912T0614Z/semantic_inventory/verification.json)
binds all 68 entries and 102 observations to training-source images, with
within-frame annotations and separate controls for three sinks, three stools
and two service carts. Its scope is annotation integrity and source provenance;
approximate visible polygons are not segmentation masks. The packager places
these entries in USD **observation Scopes**, retaining instance relationships,
source frames and an evidence link. They have **no detached body geometry**.
Masses, hinge axes, cabinet travel and contact parameters remain unmeasured.
The [independent inventory audit](../../runs/real2sim-2296/20260912T0614Z/reviews/inventory1/report.md)
checked all annotation bindings across 17 source frames and the acyclic parent
relationships, and visually verified the three physical sink identities. It
does not claim an exhaustive review of every semantic label or room object.

## Main bottlenecks and remaining acceptance

| Bottleneck | What worked | What remains |
|---|---|---|
| Repeated islands, sinks, labels and floor texture | Source overlays falsified misleading high-registration maps; instance-aware masks and learned camera initialization improved the working hypothesis | Verify remaining wrong-instance controls and smaller pose drift |
| White counters, glass, metal, thin rails and occlusion | MVS exposes missing surfaces; a separate learned TSDF surface recovers broad worktops | Observed ray coverage is only 2.07–43.26%; inferred coverage is 95.60–99.91% but includes false floor hits and inaccurate thin/contact surfaces |
| Missing depth, scale and gravity | Keep arbitrary units and explicit common transforms | Calibrated metric anchors, independent depth/LiDAR or survey evidence |
| Static video of movable equipment | Inventory visible categories without inventing mechanics | Per-object geometry, measured joint states, limits, mass, inertia, friction and contact trials |
| Same-trajectory evaluation | Buffered holdouts and explicit failed-localization denominators | Independent trajectory and real sensor/task trials |
| CUDA/runtime compatibility | Isolated PyCOLMAP CUDA12 and Torch 2.4.1/gsplat 1.5.3 wheels; selected final scene rendered twice, independently checked, with clean exits | Maintain the exact compatible runtime and package bindings |
| Gaussian numerical stability | Independent corrected-SSIM value/gradient checks passed; both fresh-v2 paths reached step 30,000 | Preserve the rejected first attempt and repaired training semantics |
| Appearance generalization | Completed full/static/group evaluations expose failures; larger-capacity continuation was tested | Buffered views remain poor; isolate pose/calibration, visibility and overfitting contributions |
| Checkpoint usage terms | Record exact original VGGT weight identity and noncommercial model card | A suitable, documented model/licensing route for intended deployment |

The current camera coverage is 74.1%, below the proposed 95% target; p95
reprojection is 2.191 px, above the proposed 2 px target. Baseline held-out
static PSNR is 22.12 dB and SSIM 0.8650, below the proposed 28 dB / 0.90 image
targets. No independent evidence presently certifies centimetre or
millimetre geometry, LiDAR return behavior, physical collision accuracy,
articulation correctness, Unitree G1 pick/place success, or real/sim performance
correlation. The exact G1 hand/configuration, task objects, sensor model,
extrinsics and matched real trials are unresolved.

## Delivered candidate and unmet acceptance

Update this table only from completed artifacts whose source/model/dataset
hashes match the chosen scene. Do not replace missing entries with intended
settings, early smoke tests or results from rejected camera models.

| Result | Current evidence state | Required final evidence |
|---|---|---|
| Dense reconstruction | Complete: 662,374 fused points; current post-run receipt and visibility support verified | Resolve local floor errors and missing surfaces before metric acceptance |
| Observed mesh | Complete diagnostic mesh: 784,079 triangles, 359,965 boundary edges; source ray review available | Watertight task-relevant collider, independent surface accuracy and gap review |
| Inferred mesh | Complete separate hypothesis: 22,438,032 triangles; source and independent floor reviews show both added coverage and false surfaces | Correct task-relevant geometry and independent metric validation; collision remains disabled |
| Actual Gaussian appearance | Both step-30,000 experiments complete; hashes and full/static/group metrics checked; baseline selected for better aggregate validation scores | Selected appearance remains below acceptance; preserve worst-view evidence |
| Native appearance export | Selected final baseline PLY/USDZ hashes and raw-coordinate quantization receipt verified | Preserve raw float32 authority and exact export provenance |
| Canonical scene | Completed `scene/scene.usda`; independent core composition and bound payload/camera checks pass | Preserve final portable-evidence bindings; deployment gates remain unmet |
| Final native Isaac render | Selected 30k composition rendered at training and buffered cameras; independent camera/settings/pixel/control/exit checks pass | Two views verify this rendering path, not whole-environment acceptance |
| Final source alignment | Full appearance evaluation, observed/inferred 17-camera mesh reviews and actual final native comparisons expose substantial deficiencies | Generalization and true surface accuracy remain below requirements; do not promote this candidate as exact |
| Deployment acceptance | Unverified | Independent metric, collision, articulation, LiDAR and G1/real-sim trials in [ACCEPTANCE.md](ACCEPTANCE.md) |

The two model metric reports audited here have SHA256
`337068d80599ce9454a4976453a8592931f14fbc864608d26836a47c76699566`
(`model_refined_metrics.json`) and
`ee16fa113baa57c3bb58fa4275386d618ed5942d7f2c204f6e8cf185bcf82e1b`
(`model_stable_metrics.json`). The all-view VGGT result is
`6a538095ed865a40a0cd30a2c67e36cbb3b831e4179fd29c8100ae05038a97e2`.
The prepared dataset is
`cda62572b733c74e9409730e05a0110e42f63897e69a089ef682691cef23837b`;
the bound fixed-map localization report is
`7738237a367042ba49f15eba35f42eb7a9b487be29884578df22b35f59bfc273`.

This update also rehashed both completed appearance checkpoints, PLYs, videos,
all prepared input images/masks, all 434 final comparison images, the fused cloud
and visibility file, geometry arrays/surfaces, 17 mesh-review images, source
inventory crops and the 42 intermediate native-preview artifacts. No mismatch
was found. The query-pose comparisons and final USDZ source/package/converter/
template hashes were checked separately. This is an artifact-binding audit;
it does not convert the reported quality failures into passed acceptance.

| Snapshot artifact | SHA256 |
|---|---|
| `appearance_v2/result.json` | `3969c14a726f02be8632e9614aa11b07f9c4fc9f39f68892d62b56b81014a67e` |
| `appearance_capacity/result.json` | `a6ef5e522f8871b8491f0a54fcff75436186c4462fc1815feb1ddaa3c56f38fd` |
| `dense_provenance.json` | `32ce5581d92b9c17dde5aa033ca78b9f33036041f693f48d9d6eb1646fbdf7e5` |
| `geometry/metrics.json` | `ce766fae4f926c037eb6bf1c36000181cb5489a50f42069afe233816350a7960` |
| `appearance_v2/appearance.usdz.receipt.json` | `549c68e7747354a5c834b2f0785538766a5b9d24af8a831e946d4febbb765e9d` |
| `previews/step_005000/evidence.json` | `5103e85768c1dc801a0d24740f6f28892cf51b734c3624f9c67ab21bf5ae3ef2` |
| `geometry_prior/metrics.json` | `e77f88940bfcb664a0b1d0bac3e8607dbd02a0754ca09e0883f0668b58c7724c` |
| `scene/scene.usda` | `ea3dad2a8b1242fa9df9356e2230818d2af3191420104944fa07c5323c713bfe` |
| `final_native/evidence.json` | `2029090849a95e01ef73da6eceaa4d68f10aaed5726738ad4f94023e26783106` |

These hashes bind this report's snapshot; later artifacts need their own checks.
