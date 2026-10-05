# Acceptance and verification contract

These are proposed engineering targets, not measured results or guarantees.
They are intentionally stricter in the manipulation workspace than elsewhere.
The user confirmed the MOV is the only capture and Unitree G1 is the eventual
robot. Hand, LiDAR model, calibration and real task trials are not supplied.

| Gate | Proposed acceptance | Evidence and method | Current availability |
|---|---|---|---|
| Source integrity | All artifacts trace to a SHA256 of the original file; actual frame PTS and orientation retained | ffprobe, MOV atom audit, per-frame manifest | Available |
| Coverage | Every task-relevant visible object inventoried; ≥95% of navigation surfaces and ≥99% of contact workspace observed at useful angles; unseen surfaces explicitly identified | View/visibility map against independent survey, object inventory | Video supports partial view coverage only |
| Camera solution | One connected solution; ≥95% of selected train frames registered; median reprojection ≤1 px, p95 ≤2 px at source resolution; no implausible lens/pose jumps | Sparse tracks, calibration, path inspection, loop/revisit residuals | RGB-derived, measurable |
| Generalization | Predetermined held-out images excluded from geometry and appearance optimization; their cameras localized against fixed training map | Frame split, fixed-map PnP and localization statistics | Measurable; same-trajectory views are a limited test |
| Appearance | Held-out static-region PSNR ≥28 dB, SSIM ≥0.90, LPIPS ≤0.15; inspect worst views, texture seams, floaters, thin geometry and disocclusions | Side-by-side and residual renders at matched cameras; report masks and unmasked scores | Measurable where poses can be recovered |
| Metric frame | Metres; explicit handedness, gravity and origin; all layers and exports agree within 1 mm numerically; scale uncertainty ≤0.5% | Independent non-collinear measured anchors and transform round trips | No metric anchors or gravity measurements supplied |
| Navigation geometry | Independent surface distance median ≤10 mm, p95 ≤30 mm; planar wall/floor normal error ≤0.5°; distances across room ≤1% | Surveyed or calibrated held-out LiDAR/depth, bidirectional surface distances, completeness | Requires new independent measurements |
| Manipulation geometry | Support plane median ≤3 mm, p95 ≤5 mm; handles and grasp surfaces within 3 mm; thin edges and openings preserved | Close-range calibrated scans/calipers, task-specific clearance budget | Not certifiable from compressed room video alone |
| Objects and identity | All task objects have persistent IDs, transforms, bounds, semantic class, static/dynamic state, source views and confidence; no duplicated static/interactive visual geometry | Object inventory and layer audit | Inventory possible; full geometry/mechanics unobserved |
| Articulations | Correct parent/child hierarchy, joint type/axis/origin; hinge origin ≤3 mm and axis ≤1°; measured limits within 2°/3 mm; clearance through full motion | Measured open/closed/intermediate states and contact sweeps | Static/partially open views do not identify limits or mechanics |
| Collision quality | Closed finite solids for dynamic objects, consistent winding, no degenerate faces; appropriate concavity on static surfaces; thin legs, handles, sink/cabinet openings retained | Mesh checks; ray/visual alignment; supported-drop, sliding and swept-volume tests | Requires reliable geometry before promotion |
| Physical parameters | Measured mass/COM/inertia and friction ranges; positive physically consistent inertia; restitution, hinge damping and actuator properties measured or explicitly ranged | Weighing, identification trials, drop/slide/open tests | Unidentifiable from RGB appearance |
| G1 simulation | Named G1 configuration and hand; correct asset scale and gravity; stable stance and navigation; no start penetrations; ≥100 repeatable representative pick/place trials with logged failure modes | Simulator import and contact tests, robot configuration hash and seeds | G1 named; exact configuration and tasks absent |
| LiDAR behavior | Correct sensor model, scan timing, beam pattern, mounting and noise; surface range residual median ≤10 mm, p95 ≤30 mm on opaque static geometry | Real calibrated scans vs raycasts; glass/metal dropout separately assessed | Sensor and real scans absent |
| Localization | ≥95% success over predeclared start poses; held-out real trajectory error p95 ≤5 cm and yaw ≤1°; report failures in repeated/occluded regions | Independent real trajectory ground truth, fixed map, actual localization stack | Cannot test with camera video as sole input |
| Real/sim correlation | Matched real/sim interventions across ≥3 task variants; success difference ≤10 percentage points with confidence intervals, rank correlation ≥0.8 where statistically identifiable | Same robot/control/perception, repeated real trials, matched sensor statistics | No real task trials supplied |
| Portable scene | One canonical scene manifest/frame; deterministic assets and checksums; OpenUSD authoritative composition; validated simulator-specific exports | Load/round-trip tests and cross-layer transform checks | Build only supported layers; no invented physical certification |

## Acceptance rules

- All thresholds require measurement at the specified resolution/units; an
  unavailable measurement is **unverified**, never a pass.
- Reprojection and photometric fit test consistency with RGB, not physical
  accuracy. Monocular reconstruction has an unresolved global scale and frame.
- A neural depth prediction or fitted furniture template is an inferred prior,
  not a measured surface. Record it separately if used.
- Reject a disconnected or degenerate camera trajectory even if its pixel
  residual is low. In the first pilot, three cameras and 313 points escaped
  millions of reconstruction units from the room despite subpixel median fit;
  that result is rejected for geometry. Inspect radius distributions and
  temporal jumps, and use conventional registration without a structure-less
  fallback for the next experiment.
- Gaussian splats can model appearance; they are not closed collision bodies
  and their rendered depth is not automatically a faithful LiDAR map.
- Glass, mirrors, polished metal and TV imagery require explicit treatment.
  Objects seen through glass are not necessarily in the robot's free space.
- Collision decimation is accepted against task clearances, not polygon count.
- A room in which cabinets never move cannot establish joint travel, breakaway
  friction, mass or hidden interiors. No claimed exact articulation from it.
- For real/sim correlation, the experimental unit is a physical robot/task
  condition with repeat trials, not adjacent video frames. Predeclare enough
  conditions and repetitions to estimate confidence intervals and correlation
  uncertainty; three variants alone do not provide strong statistical evidence.
- Do not use a made-up metric scale to label a package ready for deployment.
- A preview may be delivered with unmet gates; it must visibly state that it
  is a reconstruction candidate and cannot be called the final exact replica.

## Scene-specific audit inventory

Observed categories across the source: perimeter base and wall cabinets with
doors, drawers and pulls; refrigerator; range/oven/hood; microwave; dishwasher;
countertops; three sink/faucet stations (two distinct wall sinks and one in the
island); an island with three stools; a second island
with sorting-bin openings, labels and doors; three round pedestal tables;
black chairs; mobile utility carts with thin rails and wheels; bins; wall TVs,
dispensers and outlets; lights and overhead devices; gray floor with dark
strips and tape markers; walls, doorway and glass partitions. The equipment
behind the glass requires separate spatial classification. This is an initial
visual inventory, not an exhaustive instance count. At least one base-cabinet
door appears ajar. The TV contents change, while the cabinet motion is not
demonstrated. Undersides, cabinet interiors and regions outside the trajectory
are not fully observed.
