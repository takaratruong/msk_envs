# Tool choices for IMG_2296 real-to-simulation

Research date: 2026-09-12. This is a primary-source tool survey and a proposed
stage decision, not an installation or reconstruction receipt. No capture data
was uploaded during this research. Versions and capabilities below describe
the cited upstream documentation; installed versions still require a probe.

## Decision

Use one calibrated reconstruction and one world frame, with linked appearance,
surface geometry, collision, object and articulation layers. Start with
COLMAP/PyCOLMAP cameras and conventional multiview stereo. Train a Gaussian
appearance model only after camera quality is established. Prefer a pinned
gsplat baseline for a small appearance experiment, then 3DGRUT/OpenUSD if native
Isaac Sim rendering is the selected deployment path. Obtain actual metric
depth, dimensions or an independent LiDAR scan before calling the geometry
accurate in metres. Build the manipulation assets from observed geometry plus
documented measurements, with explicit unresolved properties.

This choice follows the current capture evidence in [BRIEF.md](BRIEF.md): a
75.57-second iPhone MOV with RGB and Apple metadata, but no confirmed recorded
depth or camera poses. The kitchen/lab contains low-texture counters,
reflective appliances, sinks, thin chair legs, glass partitions, repeating
cabinet fronts, and changing screens. These are adverse cases for reconstructing
complete surfaces from RGB. The same images can constrain appearance much more
strongly than hidden shape, scale, contact properties or laser response.

The iPhone's depth capability and a saved RGB-D dataset are separate facts.
Apple exposes scene depth and its confidence through ARKit frame properties;
the capture app must preserve the relevant data. ARKit's front-camera
`capturedDepthData` is also distinct from rear-camera `sceneDepth`.
[Apple scene-depth sample](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth),
[Apple depth-property distinction](https://developer.apple.com/documentation/arkit/arframe/captureddepthdata).

## Camera reconstruction and explicit surfaces

| Tool | Useful capability | Decision for this capture |
|---|---|---|
| **COLMAP / PyCOLMAP** | Calibrated cameras, feature tracks, bundle adjustment, sparse geometry, dense PatchMatch, fusion and meshing APIs. Linux CUDA 12 Python wheels are now available as `pycolmap-cuda12`. | **First path.** Sample sharp frames with overlap and translation; mask changing screens/reflections where needed; compare camera models and inspect disconnected components. Use an isolated CUDA wheel before paying for a source build. Official development docs currently describe 4.3.0.dev0, so match API use to the installed release. [Python API](https://colmap.github.io/pycolmap/index.html), [installation](https://colmap.github.io/install.html). |
| **hloc + LightGlue** | Learned local feature matching with a conventional geometric verification and reconstruction backend. | **First fallback for matching failures.** Test difficult links and loop closures before changing the whole reconstruction method. LightGlue code/weights are Apache-2.0; its SuperPoint feature backend has separate restrictive terms, while DISK is a different option. [hloc](https://github.com/cvg/Hierarchical-Localization), [LightGlue](https://github.com/cvg/LightGlue). |
| **VGGT** | Feed-forward cameras, depth, points and tracks; optional bundle adjustment and COLMAP export. | **Independent initialization/diagnostic.** Run small overlapping view sets, compare resulting poses against multiview observations, then refine. Confidence and predicted depth are not an independent survey. Do not concatenate local reconstructions without verified alignment. Original weights remain non-commercial; the distinct commercial checkpoint is gated and has an acceptable-use policy. [repository](https://github.com/facebookresearch/vggt), [license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt). |
| **MASt3R / DUSt3R** | Learned point maps and matching can recover correspondences when conventional features are weak; MASt3R provides a metric-named checkpoint. | **Research fallback.** A metric prediction is a learned prior, not a measurement of this room. Use supported correspondences in a common optimization and record prior-dependent surfaces. Both repositories use CC BY-NC-SA 4.0, and checkpoints include additional dataset notices. [MASt3R](https://github.com/naver/mast3r), [DUSt3R license](https://github.com/naver/dust3r/blob/main/LICENSE), [checkpoint notices](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE). |
| **OpenMVS** | Imports undistorted COLMAP cameras and points; densifies, meshes, refines and textures. CUDA is optional; upstream now lists Ubuntu release binaries. | **Mesh baseline or dense fallback.** Preserve raw dense points and visibility before cleaning. White counters and glass can remain incomplete; a hole-filled mesh does not constitute observed geometry. AGPL-3.0 software. [COLMAP import and workflow](https://github.com/cdcseacave/openMVS/wiki/Usage), [build options](https://github.com/cdcseacave/openMVS/wiki/Building), [license](https://github.com/cdcseacave/openMVS/blob/master/LICENSE). |
| **Open3D** | RGB-D TSDF integration, point-cloud processing, registration and mesh extraction. | **Transparent geometry processing.** Fuse real depth if available; otherwise explicitly label MVS- or model-derived depth. Keep confidence, source support and the unfilled mesh. TSDF voxel size is numerical resolution, not evidence of measurement accuracy. [integration documentation](https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/integration.html). |

COLMAP's library is BSD licensed, but its third-party dependencies have their
own terms. Distribution of a compiled product needs the pinned dependency
notices, rather than just the top-level license.
[COLMAP licensing](https://colmap.github.io/license.html).

For dense COLMAP on this NVIDIA machine, confirm that the actual installed
binary/wheel exposes GPU support. Default Linux distribution packages do not
include CUDA; the official CUDA Docker image is another contained route.
Start stereo on a small valid set and verify nonempty depth before launching
the room. [COLMAP installation](https://colmap.github.io/install.html).

## Appearance and surface-aware neural methods

| Tool | Strength | Limitation and disposition |
|---|---|---|
| **3D Gaussian Splatting through gsplat** | Fast, editable point-based appearance; standalone COLMAP trainer, image metrics and efficient GPU rasterization. Apache-2.0 implementation. | **Preferred appearance baseline.** Low image error can coexist with floaters, erroneous surfaces or transparency. The current `main` requires PyTorch 2.7+; v1.5.3 is the referenced released line. Pin a version compatible with the isolated runtime. [repository and installation](https://github.com/nerfstudio-project/gsplat). |
| **Nerfstudio Splatfacto** | Integrated preprocessing, viewer, training, evaluation and splat export; initializes well from COLMAP points. | **Convenient alternative if installed cleanly.** Do not assume the general `ns-export` mesh examples apply identically to every Gaussian model/version; probe model/exporter compatibility. Its published installation guide still describes an older Torch/CUDA stack. [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html), [export documentation](https://docs.nerf.studio/quickstart/export_geometry.html), [installation source](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md). |
| **2D Gaussian Splatting** | Oriented surface elements with normal/distortion regularization; bounded and unbounded TSDF mesh extraction. | **Useful geometric comparison after the baseline.** Prefer a pilot in the task workspace. RGB-only supervision still leaves unseen/ambiguous surfaces unresolved. Original implementation has the INRIA research/non-commercial license; gsplat has a separately implemented 2D rasterization API, which is not automatically the entire original training pipeline. [implementation](https://github.com/hbb1/2d-gaussian-splatting), [license](https://github.com/hbb1/2d-gaussian-splatting/blob/main/LICENSE.md). |
| **PGSR** | Planar Gaussian representation designed for multiview surface reconstruction without pretrained depth/normal priors. | **Worth a bounded comparison for counters/walls.** Evaluate observed edges and flat surfaces, not just a whole-scene image score. Custom CUDA extensions and older environment examples add installation work. Educational/research/non-profit license, with separate commercial permission. [implementation](https://github.com/zju3dv/PGSR), [license](https://github.com/zju3dv/PGSR/blob/main/LICENSE.md). |
| **GOF** | Extracts an adaptive mesh from an opacity-field level set through marching tetrahedra. | **Secondary mesh experiment.** Avoid treating an opacity level set as a measured physical boundary. Original implementation uses the INRIA research/non-commercial license. [implementation](https://github.com/autonomousvision/gaussian-opacity-fields), [license](https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/LICENSE.md). |
| **SuGaR** | Surface-aligned Gaussians, mesh extraction/refinement and Blender editing workflow. | **Editing candidate if mesh appearance needs refinement.** It does not infer object dynamics or identify true hinge axes. Original implementation uses the INRIA research/non-commercial license. [implementation](https://github.com/Anttwo/SuGaR), [license](https://github.com/Anttwo/SuGaR/blob/main/LICENSE.md). |
| **DN-Splatter / AGS-Mesh** | Depth/normal-supervised splatting and mesh reconstruction, including smartphone RGB-D workflows. Repository is Apache-2.0. | **Strong candidate if genuine depth is recovered.** Particularly relevant to this room's low texture. Separate measured depth from monocular model priors in the loss and evidence. Audit all downloaded predictor weights. [implementation](https://github.com/maturk/dn-splatter), [paper](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf). |

These are alternatives to compare at a fixed camera solution and split, not
tools that all need to be installed. My inference from their representations
is that appearance quality, surface accuracy and editability require separate
tests. In particular, novel-view PSNR/SSIM/LPIPS do not establish collision
accuracy, and an RGB-trained laser renderer does not establish real return
statistics on glass, black plastic or reflective metal.

## NVIDIA path, RGB-D and SLAM

**3DGRUT / NuRec is a real, relevant route, with a specific boundary.** NVIDIA's
mono workflow uses COLMAP, 3DGUT and USDZ for Isaac Sim. It explicitly describes
the generated scene as visual geometry without inherent collision properties.
Its export-normalization option centers/scales the scene; that does not
recover metric scale. Preserve and compose the exact normalization transform
with the world transform. [Official mono recipe](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html).

Current 3DGRUT code is Apache-2.0, supports Gaussian ray tracing/rasterization,
and provides a local virtual-environment CUDA installation route including
12.4, 12.8 and 13.0. This is newer than the mono tutorial's CUDA 11.8/GCC 11
instructions. The code can export USD for the native Omniverse rendering path.
Select a pinned, verified release/commit rather than mixing the old tutorial's
installer with new configuration files.
[3DGRUT repository](https://github.com/nv-tlabs/3dgrut).

For a calibrated stereo or RGB-D recording, NVIDIA's more complete robotics
recipe adds cuSFM, FoundationStereo and nvblox for poses, depth and mesh
generation alongside neural appearance. It still instructs users to add
SimReady assets and collision properties. That explains why a well-recorded
ROS bag can support a much stronger result than an RGB MOV: the bag can retain
calibration, known baselines, synchronized streams, poses and measured depth.
It is the recorded information and coverage that matter, not the bag container.
[Official stereo recipe](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html).

| Tool | Conditional role |
|---|---|
| **nvblox** | GPU TSDF/ESDF mapping from depth and poses, with C++, Python and ROS 2 surfaces. Use for metric mapping/collision seeds if calibrated depth exists, or for explicitly labeled inferred-depth experiments. It cannot turn missing sensor depth into measurements. Current public license is Apache-2.0 with retained third-party notices. [repository](https://github.com/nvidia-isaac/nvblox), [license](https://github.com/nvidia-isaac/nvblox/blob/public/LICENSE.md). |
| **cuVSLAM** | CUDA odometry/mapping with current source and prebuilt Python routes; NVIDIA Community License. Useful for suitably calibrated sensor recordings. A monocular configuration still needs a metric source to establish scale. [repository](https://github.com/nvidia-isaac/cuVSLAM). |
| **ORB-SLAM3** | Monocular, stereo, RGB-D and visual-inertial trajectory alternative. It estimates maps/trajectories rather than delivering appearance, semantic objects and physical parameters. [official repository](https://github.com/UZ-SLAMLab/ORB_SLAM3). |
| **RTAB-Map** | RGB-D, stereo and LiDAR graph SLAM; useful for an independent future map and localization replay. Current MOV alone does not supply its missing calibrated depth/LiDAR observations. [official project](https://introlab.github.io/rtabmap/), [license](https://github.com/introlab/rtabmap/blob/master/LICENSE). |
| **NuRec NRE containers / Instant NuRec** | Consider NRE when NCore input, container access and its sensor schema are justified. Instant NuRec specifically targets driving logs; its feed-forward outdoor prior is not evidence of this kitchen's true shape. Neither is the first dependency for this short monocular room capture. [NuRec overview](https://docs.nvidia.com/nurec/), [Instant NuRec](https://github.com/NVIDIA/instant-nurec). |

## Authoring, physics and deployment

| Tool | Recommended ownership |
|---|---|
| **OpenUSD / SimReady** | Canonical scene composition: metre scale, named frames, stable object identities, visual/collision geometry, bodies, joints, mass and semantic metadata. USDPhysics describes physics but does not measure its parameters. SimReady supplies authoring/validation conventions, not automatic truth from video. [USDPhysics](https://openusd.org/release/api/usd_physics_page_front.html), [SimReady specification](https://docs.omniverse.nvidia.com/simready/latest/overview/simready-spec.html), [physics practices](https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/physics-best-practices.html). |
| **Blender** | Inspect and repair reconstructed surfaces, align observed parts, create UVs/materials, inspect camera reprojections, and author independently identified movable meshes. Prefer constrained edits with supporting views/measurements. Blender supports a subset of USD data, so inspect exported transforms/materials and author physics through the owning USD/simulator path. [USD manual](https://docs.blender.org/manual/id/5.0/files/import_export/usd.html). |
| **Isaac Sim** | Preferred rich deployment target if available: mixed Gaussian/polygon rendering, USD physics and RTX LiDAR. Dynamic concave contact needs suitable convex decomposition or SDF, not an unchecked raw scan. NuRec rendering has version/extension and single-GPU prerequisites. [neural rendering](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_nurec.html), [NuRec render setup](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/nurec_utils.html), [physics](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html), [RTX LiDAR](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_lidar.html). |
| **Isaac Lab** | Robot/task learning and reproducible scenario evaluation after the scene and robot are validated. It builds on Isaac Sim; it is not a reconstruction engine. Keep the robot's actuator, controller, timestep and observation configuration explicit. [official documentation](https://isaac-sim.github.io/IsaacLab/main/index.html). |
| **MuJoCo** | Efficient contact/articulation checks and a deployment variant when the robot already uses MJCF. General mesh collision operates through convex collision pipelines; cavities and thin obstacles need deliberate decomposition or a supported SDF path. Its standard OpenGL visualization is not the Gaussian appearance target. Apache-2.0 source. [collision documentation](https://mujoco.readthedocs.io/en/latest/computation/), [repository](https://github.com/google-deepmind/mujoco). |
| **Genesis World** | Optional second physics backend when needed for a task. Current USD import is supported, but automatic watertightening, decimation and convexification can alter the collision shape. Compare processed geometry explicitly; do not assume material/joint equivalence across engines. Apache-2.0 source. [mesh processing](https://genesis-world.readthedocs.io/en/latest/user_guide/assets/mesh_processing.html), [USD import](https://genesis-world.readthedocs.io/en/latest/api_reference/engine/entity/morph/file_morph/usd.html), [repository](https://github.com/Genesis-Embodied-AI/genesis-world). |

One scene may contain multiple representations without becoming multiple room
replicas: every object owns one transform and identity, with an appearance
child, a measured or estimated surface child, collision shapes and metadata.
Exports are generated from that authority. Movable objects must be removed
from the static appearance layer as well as separated in physics; otherwise
their old image remains after a simulated grasp. Regions revealed by moving
an object remain unknown unless another view or new capture observes them.

No tool in this survey makes a static room video identify true object mass,
centre of mass, friction, compliance, restitution, hinge limits, joint damping,
handle strength or appliance interior geometry. Physically plausible values
can support a prototype, but real-task correlation requires measurement or
system identification. A matching cabinet model retrieved from an asset
library is a candidate until dimensions and motion are checked.

## Practical stage gates on this host

The reported host has L40S GPUs, Torch 2.6/cu124 and system CUDA 13. Avoid
mutating a shared simulation environment. Use one currently available GPU,
an isolated environment, bounded image/Gaussian counts and deterministic run
paths. Record GPU assignment, source hashes, package versions, compiler/runtime
versions, configuration and outputs for each candidate.

1. **Input audit.** Decode the full video with orientation/timestamps preserved;
   inventory streams/sidecars, focal changes, motion blur, exposure and dynamic
   regions. Reserve views before optimization. Frame count is not independent
   coverage; almost identical adjacent frames are weak validation.
2. **Camera pilot.** Use a small sharp sample spanning the room. Try sequential
   matching plus deliberate loop links and bundle adjustment. Measure
   registered coverage, track lengths, triangulation angles, reprojection tails,
   and consistency across capture portions. A small successful component does
   not pass room coverage. Fall back to learned matching before learned shape.
3. **Geometry pilot.** Test actual CUDA support in an isolated PyCOLMAP wheel;
   run dense stereo/fusion on a subset. Compare to OpenMVS only when output
   completeness or installation warrants it. Inspect countertops, door edges,
   sink cavities, chair legs and glass separately. Keep the source-supported
   raw surface as evidence alongside cleaned geometry.
4. **Scale gate.** Fit one robust similarity transform using distributed metric
   observations; retain independent dimensions/scan points for validation.
   Without them, use explicit arbitrary units and decline metric acceptance.
   Predicted monocular depth or a standard counter-height assumption may guide
   a prototype but cannot pass this gate.
5. **Appearance pilot.** Pin compatible gsplat 1.5.3 or use isolated current
   3DGRUT. For Torch 2.6/cu124, use a matching CUDA 12.4 compiler/toolchain or a
   verified wheel; do not JIT-build an old stack against system CUDA 13 by
   accident. Current gsplat `main` requires upgrading its isolated Torch stack.
   Evaluate held-out images with exposure conventions fixed, plus direct
   side-by-side crops. [gsplat compatibility notes](https://github.com/nerfstudio-project/gsplat).
6. **Targeted iteration.** If camera residuals fail, fix cameras/data. If images
   look good but geometry fails, add real measurements or try one surface-aware
   model on the failing region. Training longer is not a remedy for unobserved
   surfaces. Select methods by the relevant acceptance metric and preserve
   failed candidate receipts.
7. **Asset authoring.** Separate each task-relevant body and articulated part;
   measure its collision boundary, pivots, axis, limits and physical parameters.
   Reconstruct exposed background. Use uncertainty labels for every inferred
   interior/backside. Author semantics and laser material properties separately
   from RGB reflectance.
8. **Single scene verification.** Import the canonical scene into the selected
   runtime. Verify camera/world/LiDAR extrinsics, units, export transforms,
   collision overlays, support/contact and full articulation sweeps. Replay the
   actual LiDAR configuration against real held-out scans; evaluate localization
   success, drift and failure cases. Finally compare real and simulated pick and
   place trials across several conditions/policies, with confidence intervals.

A sparse/appearance pilot is plausibly a minutes-to-hours compute experiment;
that is an engineering estimate, not a benchmark on this capture. Full room
geometry, separated movable assets, articulation measurements and real robot
validation are a larger data/authoring/verification project. More GPUs cannot
recover absent depth, occluded interiors or unobserved dynamics. The correct
stopping rule is passed evidence for the required workspace and tasks, with
unmet gates listed explicitly.

## Remaining bottlenecks and evidence to obtain

- Original recording app and any ARKit/depth/confidence/intrinsics/poses or
  LiDAR sidecars; a MOV transfer can preserve RGB while omitting app data.
- Several distributed measured dimensions or a registered independent survey,
  plus close coverage of the exact grasp objects and support/contact surfaces.
- Open/closed and intermediate views of every required cabinet, drawer,
  dishwasher, appliance door or other articulation, including exposed interiors.
- Robot, hand and LiDAR models, extrinsics, controller/sensor settings and task
  tolerances; photorealism alone cannot set these requirements.
- Real held-out LiDAR/localization runs and manipulation trials. Same-video
  re-rendering is useful evidence of appearance alignment, not proof of
  deployment correlation.

Before product distribution, record code, dependency, checkpoint and retrieved
asset licenses separately. In particular, a permissive renderer does not
relicense restrictive learned weights or imported assets. This survey records
source terms to inform tool selection; no commercial-use clearance is claimed.
