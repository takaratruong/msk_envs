# G1 fridge motion library task

## Requested outcome
Build and show a replayable G1 motion plan in the IMG_2296 physical scene: walk through the aisle to the refrigerator, establish a reachable stance, reach the handle, pull one hinged door open, and finish stably. The user explicitly prefers a small reusable library of existing motions, spatial placement, manual/IK correction, and stitching; avoid fresh Kimodo/ARDY generation as the primary method.

## Inputs and prior evidence
- Canonical mesh09 scene: runs/real2sim-2296/20260912T1002Z-physical/iterations/mesh09/scene.usda; SHA256 f78a48e358080b0079c9d176beea9286ef8e9eb2e4ff827eb90ca686f2647fbc.
- Refrigerator has two upper-door hinges (0 to 110 and -110 to 0 degrees), separate handle meshes/colliders, and a freezer slide. Earlier drive tests verify modeled mechanics, not G1 manipulation.
- Scene scale, handle details, dynamics and hidden hardware are estimates derived from video; real-world dimensions and opening resistance remain unmeasured.
- Local candidate motion assets exist in kimodo-scene-refactor/runs/reach_dataset, ProtoMotions/data/g1-kimodo-generated and rmr_tracking/motions. Their compatibility and quality must be inspected before selection. Existing generated clips may be curated and edited; no generation service is needed for playback.

## Scope and constraints
- First deliverable is a deterministic, editable kinematic motion reference and scene preview. A physics controller executing the sequence is a separate validation state; never describe pose playback or commanded door motion as physical robot-driven opening.
- Keep original USD, Downloads physical package, and unrelated project/workflow state unchanged. New code lives in experiments/real2sim_2296/g1_motion*; new inputs, outputs, reviews, clips and jobs live in runs/real2sim-2296/20260912T2237Z-g1-motion.
- Reuse a local G1 model and record the exact body/hand configuration and clip provenance. Do not assume compatibility from filenames.
- Detached rendering jobs use the existing jobctl run/status/logs/stop/restart surface and owned process identity. No robot hardware execution.
- Independent red-team reviews begin with this specification and continue through a pass bound to final delivered inputs.

## Implementation path
1. Inspect and curate a small library: walking/start-stop, standing/reaching, and pull/retreat if available; expose gaps honestly.
2. Read fridge handle, hinge and collision geometry from USD. Choose interaction stance and approach path against the actual scene.
3. Place clips, select compatible transitions, and solve local deterministic corrections for joint limits, planted feet and hand targets following the hinge arc. Store all edits as parameters rather than destroying source clips.
4. Export one composed timeline with G1 in the existing scene, an editable motion/clip library, and a watchable video.
5. Measure the resulting reference and independently inspect rendered motion before delivery; document measured passes, approximations and unresolved failures.

## Observable acceptance and evidence
- Same physical room geometry/materials and scene coordinate frame; bind source and selected clip/model hashes.
- Real G1 link geometry and documented joint mapping. All motion values finite and robot joint limits respected.
- Continuous root/link trajectories with no pose jumps at joins; report maximum join discontinuity, planted-foot sliding/height error and hand position/orientation error during the pull.
- Door travels with the planned handle path (initial target at least 60 degrees, subject to reach/collision feasibility); arm/body clear the moving door and nearby furniture. Distinguish intended hand-handle contact from unintended penetration.
- Scene preview and motion data can be reopened without model generation; export timestamps, phase labels and reproducible commands.
- Report kinematic/physical validation separately. No claims of dynamically balanced execution, grasp force capability, real/sim correlation or real deployment readiness without corresponding evidence.

## Open decisions owned during implementation
Select the best available compatible walking/reaching clips; choose left/right door and hand according to reach and sweep clearance; choose stance, start point and timeline duration. The user confirmed Unitree Dex3 hands; the selected local model is Dex3-1, with seven joints per hand. A full physical pull will require matching hardware/controller parameters and measured or explicitly swept door resistance.

## Concrete selected inputs (implementation update)
- Robot: /home/ubuntu/projects/bm_generalist/source/deploy/assets/g1/g1_29dof_with_hand.xml, 29 body plus 14 interleaved finger DOFs, real visual meshes. Unitree Dex3-1 is the selected local configuration matching the user's confirmed Dex3 hands.
- Walking: R4/research/motion_assets/walk_forward_01_qpos50.npz and walk_forward_02_qpos50.npz, exact named-joint conversion from Takara frames 22777–23037 and 55880–56150. Source indices/hashes/mappings are in curation.json and joint_mapping.json.
- Reach: R4/research/motion_assets/reach_radial_06_qpos36.npz, mapped by joint name through the source Kimodo G1 XML, then edited using the selected articulated-hand model FK.
- Task: door1, nominal rail height1.15m, initial stance[-1.94,-0.38], facing yaw2.99032; first pull target65degrees. Parameters remain editable and will be corrected against full-scene collision evidence.
- Numeric reference gates: joint-limit violation<=1e-5rad, finite/unit root quaternions, planted foot origin error<=5mm and sole penetration<=3mm; planned wrist position error<=5mm and orientation<=2degrees during pull; no unintended scene penetration>2mm. All are reference-quality gates, not dynamic or hardware validation.
