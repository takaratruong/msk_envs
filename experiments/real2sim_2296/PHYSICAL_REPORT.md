# IMG_2296: textured, articulated scene

The selected physical candidate is **mesh09**, a single textured OpenUSD scene
with 79 modeled assets, 3,861 mesh parts, 3,675 collision parts and 85 non-fixed
joint coordinates. It contains no Gaussian volumes. This supersedes the
Gaussian candidate for the user's clarified representation request; the
earlier reconstruction and its failed deployment gates remain evidence.

The physical deliverable uses actual separate mesh surfaces, compound collision
pieces, rigid bodies and USD joints. The source footage supplies the visible
layout, furniture style and texture patches. Hidden interiors and mechanical
parameters are constructed estimates. No claim of an exact metric replica,
real LiDAR localization accuracy or G1 task transfer has been established.

The canonical project run is
`runs/real2sim-2296/20260912T1002Z-physical/`. Paths below are relative to that
directory. The portable archive uses the layout described in its README;
its scene and texture bytes are identical.

| Deliverable | Project path |
|---|---|
| Isaac scene: named meshes, bodies, joints and collisions | `iterations/mesh09/scene.usda` |
| Object/joint inventory and assumptions | `iterations/mesh09/scene.manifest.json`, `iterations/mesh09/inventory_ledger.json` |
| Editable Blender copy, textures packed | `authoring/mesh09/scene_viewable.blend` |
| Four video/render comparisons | `iterations/mesh09/comparison/index.html` |
| Smooth native articulation video | `demos/mesh09/native_take03/articulation_demo.mp4` |
| Native motion and contact evidence | `physics_tests/mesh09_verification.json` |
| Collision OBJ/PLY and ideal scan controls | `lidar/mesh09_verified/` |
| Portable scene, textures, Blender and evidence | `delivery/IMG_2296_physical.zip` |

Canonical scene SHA256:
`f78a48e358080b0079c9d176beea9286ef8e9eb2e4ff827eb90ca686f2647fbc`.
Original MOV SHA256:
`282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9`.

## What was done

1. Reused the inspected MOV, 68-entry annotated inventory, estimated camera
   poses/intrinsics and floor evidence from the earlier reconstruction. The MOV has
   RGB footage but no established usable recorded depth or ARKit poses.
2. Fit the island countertop rectangle jointly in two source views, then used
   a nominal 0.9144 m countertop height to choose a working scale. Mean corner
   reprojection error is 8.88 source pixels on the eight fitting corners;
   this is a fitting residual, not an independent accuracy result.
3. Built separate kitchen assets: perimeter and upper cabinetry, two islands,
   three modeled open sink basins, faucets, appliances, stools, tables, chairs,
   carts, visible fixtures and small movable objects. Interior panels, shelves,
   drawer boxes and handles are geometry. Source-visible retainers keep the
   corresponding door joints locked, and an ajar cabinet starts partly open.
4. Extracted material and sign/display patches from original video frames with
   recorded homographies and pixel provenance. They retain baked lighting and
   do not constitute measured PBR albedo. Grain scale, roughness and metal
   response remain estimates.
5. Authored a Z-up, meters-based OpenUSD composition and tested it in native
   Isaac Sim 5.1. Blender 4.5.13 is an editable authoring copy; native USD
   remains the authority for physics and joint semantics.
6. Compared matching-camera renders and ran independent geometry review and
   native PhysX motion/contact tests. Failed iterations were preserved and
   corrected instead of treating successful file loading as physics validation.
7. Added an ideal geometry scan in the same USD coordinate frame, with glass
   included and omitted as separate controls. This exercises the composed
   collision surfaces; it is not a sensor model or an independently surveyed map.

## Findings that changed the implementation

- An estimated glass pane enclosed four source cameras. Native RTX refraction
  distorted the entire camera view. A glass-only control restored the expected
  projection, and the boundary was revised to clear all source camera centers.
- An island sink crossed the internal wall of an adjacent waste bay. Aligning
  the sink base with the basin preserves its usable interior.
- Per-patch chair-shell UVs repeated the grain visibly. The visible shell was
  welded and assigned continuous UVs while retaining thin compound colliders.
- Swept tube sections could twist at sharp bends, leaving pinched visible
  geometry inside a larger convex collider. Straight convex rounded sections
  remove that discrepancy.
- Two corner-cabinet handles overlapped before simulation. Native contact
  impulses eventually corrupted broadphase state. The corner was given real
  clearance. Verification now rejects backend errors even when stale USD
  transforms still appear finite.
- Nonzero joint initialization initially failed to propagate through nested
  links. Forward kinematics now handles mixed chains; independent nonzero
  swivel and slider checks measure the resulting joint frames.
- An ajar source door needed an explicit USD joint-state position as well as
  transformed mesh links and a drive target. Otherwise PhysX initialized it
  closed and then drove it toward the photographed pose.
- Collision review had to cover neighboring assets as well as individual
  objects. Initial refrigerator/upper-cabinet, cooler/glass and chair/glass
  intersections were removed. Opening sweeps subsequently found corner
  obstructions that were invisible to an initial-pose overlap test.
- Rectifying the waste fronts identified their seam and label locations, but
  handle positions required a separate fit of the mounting endpoints: a
  protruding U-handle does not lie in the door plane. Its centerline was rebuilt
  without scaling down the tube gauge.
- Four separately beveled countertop pieces produced artificial visible seams.
  They were replaced with one watertight beveled slab and a real through-hole,
  with the same mesh used for static triangle collision.
- A ceiling closed the nominal room shell but blocked the earlier dominant
  dome light. Controlled native renders separated illumination from geometry;
  neutral area lights and path tracing produced a better enclosed-room result.
  Light intensities and ceiling height remain authored estimates.
- A full traversal exceeded the initial 300-second watchdog after 41 of 63
  driven cycles. The partial results were retained, including three actual
  corner obstructions. Measured runtime justified a 600-second bound for the
  next complete traversal; a timeout is not a passing verification receipt.
- The corrected corner drawer needed genuine cabinet clearance to reach its
  452.4 mm modeled travel. Two adjacent corner doors have conservative
  100-degree limits because the constructed surrounding geometry obstructs
  larger angles. These are usable modeled limits, not recovered hardware specs.
- A sphere released on the sink's curved side rolled toward the bowl bottom.
  The full run retained a failed flat-support height/drift result despite
  contacting only the expected basin. Separate geometric placement/contact
  diagnostics are retained alongside that failure; they do not rewrite the run.

## Representation and verification boundaries

The current model has 85 non-fixed joint coordinates: 63 driven demonstration
mechanisms, six doors retained by visible handle locks, and 16 passive caster
coordinates. These counts are separate from the 68-entry source inventory,
which includes assemblies, subparts and unresolved regions. Sixty-two entries
map to modeled assets. The unresolved entries are two distant chairs, a
partially seen fourth table, two candidate glazed doors and the laboratory
contents beyond the glass. They are not silently counted as reconstructed.

Each moving link uses separate convex collision pieces. Concave static
surfaces use their triangles. Cabinets, drawer boxes, sink bowls, bin wells
and the cooler have actual openings. Masses, contact parameters, hinge axes,
travel limits, hidden shelves and appliance interiors are constructed estimates.
The position drives are demonstration controllers; they are not measured door
or drawer dynamics. The viewer can disable their stiffness for passive
interaction, but this does not validate a G1 grasp or contact policy.

Native checks measure link motion and joint frames, solver/backend errors,
initial stability, return to the initial state, actual support contacts and a
small clearance under a modeled handle. The standard driven cycle uses 80%
of travel, except oven/dishwasher hinges, which use full travel. Focused corner
checks additionally exercise the declared endpoints. A 12 mm sphere clearance
query is a generic geometric test, not a G1 finger-fit result.

The source comparisons use four reconstructed cameras without fitting an
extra image warp or exposure correction. Display textures preserve selected
recorded instants, so they need not match a different timestamp's screen
content. RGB difference is descriptive and is not a metric geometry score.
Source-derived texture patches retain camera processing, blur and baked light;
they are not intrinsic reflectance measurements or seamlessly captured PBR
materials. Stool profiles, cabinet proportions and unseen room boundaries
remain approximate even where the scene is visually recognizable.

## Acceptance and evidence

| Requirement | Observed result and practical limit |
|---|---|
| Actual textured geometric objects | Met: 79 assets, 16 source image textures, no splat volumes; native Isaac rendering and independently reopened Blender meshes. |
| Collision surfaces correspond to the model | Bounded checks pass: 3,675 composed colliders, zero initial cross-asset moving overlaps in the census; countertop through-hole and sampled cavities remain open. This is not exhaustive contact validation of every possible simultaneous motion. |
| Doors/drawers move under physics | All 63 driven cycles and six retained doors passed their native checks. The 16 passive caster coordinates were observed initially but excluded from drive-cycle claims. |
| Corner endpoint travel | Three additional full-range cycles passed: approximately +99.17 degrees, -99.85 degrees and -451.91 mm against modeled +100 degrees, -100 degrees and -452.4 mm endpoints. |
| Support and handle use | Three named support tests and the controlled 12 mm sphere handle-clearance query have separate receipts; the combined support audit is recorded in the native verification receipt. This is not a G1 grasp test. |
| Editable interchange | Blender fresh-process checks passed for 3,861 meshes, 217 cameras, 116 joint metadata markers including fixed joints, and 16 packed images. Joint markers are metadata, not Blender solver constraints. |
| Complete scene inventory | Incomplete: 62/68 entries mapped; six unresolved entries are listed above. Asset and inventory counts describe different granularity. |
| Photorealistic match throughout video | Not met. Four unwarped camera comparisons show material, furniture, appliance-detail and boundary differences. |
| Accurate metric geometry | Unverified. Nominal scale uses an assumed counter height; no independent dimension or depth reference exists. |
| LiDAR localization on a real map | Unverified. The ideal collision scan is internally aligned but is neither a measured map nor a calibrated sensor simulation. |
| G1 pick/place and real/sim performance correlation | Unverified. No G1 hand model, task configuration, measured dynamics or paired trials were supplied or evaluated. |

The full native sequence completed 28,140 physics steps at 120 Hz with finite
rigid transforms, a clean backend, unchanged source files, and maximum fixed-base
drift of 0.000000375 nominal metres. Its overall receipt remains false because
the original sink support probe failed. The combined receipt names each
component run and its executed helper, and reports the supported union of
checks explicitly. It does not turn the failed run into a passing run.

The basin followup uses a 25 mm-radius sphere, deliberately larger than the
modeled 18 mm drain radius, with at least 5 mm required radial support margin.
It permits downhill rolling and checks containment below the rim, contact
only with the right bowl, a final native overlap query expanded by 1 mm,
and settling under gravity. The measured terminal displacement was zero over
five samples in 0.25 s, and final speed was 0.639 mm/s against a 10 mm/s limit.
No scene geometry, drive parameters or planar-support thresholds changed.
The earlier 18 mm sphere diagnostics remain recorded and are not accepted as
proof of planar stability in the curved bowl.

The countertop uses one outward, watertight 64-vertex/128-triangle visible mesh
and the same static triangle collider. Independent review checked 66,454
bidirectional rays plus five aperture controls with no discrepancies. Thin
stool collision patches exceed a generic 2% hull-volume heuristic; separate
sampling found a maximum 0.764 mm surface gap. This qualifies that approximation
and does not establish a full surface-distance bound.

The four comparisons cover 0.00, 28.24, 49.97 and 71.94 seconds of the 75.57-second
source. Their RGB mean absolute differences are respectively 41.84, 53.99,
33.99 and 31.23 on a 0–255 scale. They include lighting and changing screen
content, share fitted source cameras, and are not independent geometric
validation. The kitchen is recognizable, but stool feet/profile/spacing,
cabinet and floor appearance, island countertop rim labels, appliance details,
and the hallway/lab beyond glass still differ visibly. The dual-display wall
is especially overbright relative to the source.

The collision export has 199,492 vertices and 383,488 triangles in the initial
composed world pose, grouped by 3,675 collider owners. OBJ readers may deduplicate
vertices; independent readback compared exact file arrays and oriented surfaces.
The artificial LiDAR grid casts 46,080 rays from one nominal aisle pose.
Glass inclusion versus omission changes 8,536 rays (18.52%). These are two
ideal limiting cases, not measured glass response. No noise, intensity,
motion distortion, beam footprint or return dropout is modeled.

## Opening and using the scene

Extract the portable ZIP and open `scene/physical/scene.usda` in Isaac Sim.
Preserve the directory tree so the relative textures resolve. The single USD
contains all scene meshes and physics definitions. For Blender, open
`authoring/scene.blend`; all source textures are packed.

On this host, from the project root:

```sh
python experiments/real2sim_2296/physical_control.py open
python experiments/real2sim_2296/physical_control.py status
python experiments/real2sim_2296/physical_control.py logs
python experiments/real2sim_2296/physical_control.py stop
python experiments/real2sim_2296/physical_control.py restart
```

The launcher uses `selected_scene.json` and checks both the USD and adjacent
manifest hashes. `open` uses the current helper; `restart` reuses the archived
command for the previous owned job. Its local Isaac Python installation is
`/home/ubuntu/projects/simtoolreal/repo/.venv_isaacsim/bin/python`; GPU 0 and
display `:1` are host-specific choices.

The controls select a door/drawer, open it, return it, restore the source pose,
switch source cameras, and disable/restore demonstration drive stiffness.
Retained doors stay locked. Opening normally requests 80% of the modeled
travel, with full oven/dishwasher hinge travel. Physics state and commands use
temporary session edits; source scene files remain unchanged. Disabling drive
stiffness is preparation for interaction, not a robot policy or grasp controller.

The Isaac GUI was captured loaded and paused on the source kitchen camera.
The remote desktop was locked, preventing a real mouse open/reset check.
Isolated callback/controller checks and native physics tests are recorded
separately; they do not prove live UI event ordering.

Blender uses a -9.6633-stop display-exposure adjustment to make the imported
USD lighting viewable. Readback confirms unchanged geometry, UVs, materials,
lights and cameras; the adjustment is not a texture or lighting refit. Blender
material conversion and tone mapping can differ from Isaac. USD remains the
authority for articulation semantics and physical simulation.

The 14-second articulation video contains 210 native 1280 x 720 captures at
15 fps while PhysX steps at 120 Hz. It shows a refrigerator hinge reaching
79.90 degrees and returning to 0.15 degrees, then a perimeter drawer reaching
329.84 mm and returning within 0.16 mm. These are commanded demonstrations,
not G1 interaction. Per-frame solver measurements and an independent movie
decoder readback are retained. Source geometry, materials, masses, limits,
drive gains, colliders and camera pose remain unchanged.

The video explicitly uses native RTX realtime rendering for viewing. Two
path-traced capture attempts were stopped and retained as incomplete: installed
Replicator accumulated 64 subframes for every captured frame, exceeding the
intended short-job budget. The four source comparison stills retain native
path tracing. Renderer throughput did not require changing the physical scene.

## Main bottlenecks and next evidence

The dominant limit is missing observation, not lack of a rendering tool. The
RGB video does not establish absolute scale, hidden interiors, true hinge
hardware or contact dynamics. Glass, reflections, repeating cabinets, thin
supports, moving screens and weakly textured surfaces also limit image-based
surface recovery. Constructed CAD details produce usable objects but must not
be described as measured reconstructions.

The partial hallway and laboratory beyond glass have insufficient geometric
evidence for an exact reconstruction. Unknown inventory entries remain explicit.

Real deployment still requires an independent scale/dimension reference,
LiDAR scans and sensor extrinsics/return characteristics, the G1 hand and task
configuration, measured contact/dynamics where task-sensitive, and paired
real/sim task evaluation. None can be certified from image resemblance alone.

The next useful measurements are distributed room/furniture dimensions,
overlapping calibrated depth or LiDAR, close views of the remaining assets,
open/closed mechanism captures, and task-specific G1 hand/sensor parameters.
After fitting those data in this same world frame, validate against held-out
dimensions and scans, then run paired real/sim localization and pick/place
trials. Visual refinement should address the four comparison views before
claiming photorealistic completion.

## Tools and references

The earlier [tool survey](TOOLS.md) records reconstruction alternatives and
the image-based baseline. This physical implementation uses NumPy/SciPy and
PyCOLMAP for geometric evidence, OpenUSD for composition, Isaac Sim/PhysX for
native stepping and rendering, and Blender for editable mesh/material assets.
Joint and articulation behavior follows NVIDIA's
[joint documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/joints.html)
and [articulation documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/rigid_bodies_articulations/articulations.html).
Blender's [USD documentation](https://docs.blender.org/manual/en/4.2/files/import_export/usd.html)
describes the interchange boundary; a mesh import alone does not validate an
Isaac articulation.
NVIDIA's [path-tracing settings](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)
describe the indirect-light rendering controls used for the enclosed scene.
