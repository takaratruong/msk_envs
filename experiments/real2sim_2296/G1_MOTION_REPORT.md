# G1 fridge motion: method and limits

The delivered reference uses the G1 with Unitree Dex3-1 hands in the existing IMG_2296 physical room. It walks approximately 1.965 m through the aisle, turns into a two-foot stance, reaches the right hand to a fridge rail, closes its fingers, and follows the door through a 65° opening. The reference has 589 poses at 30 Hz, spanning 19.6 seconds, and finishes holding the door open.

This establishes an editable motion and its geometric alignment with the modeled room. It does not establish a dynamically executed pull. Robot poses and door angle are coordinated data; the animation does not demonstrate that hand forces opened the door.

## What was reused

The room is the earlier `mesh09` physical USD. Its source SHA-256 is `f78a48e358080b0079c9d176beea9286ef8e9eb2e4ff827eb90ca686f2647fbc`. Its geometry, materials, textures, coordinate frame, colliders and articulation definitions are preserved. The animation adds a robot layer and a preview overlay. The overlay temporarily disables physics for deterministic pose review; the underlying physical layer is unchanged.

Two walking candidates were extracted from the existing local Takara motion recording: original frames 22777–23037 and 55880–56150, both at 50 Hz. The first was selected using requested travel distance and endpoint joint-pose difference. This is offline selection from a small library followed by stitching and correction; it is not a learned online motion-matching controller.

The reach seed is an existing local saved reach clip, `resel01_s021_radial_r0.6_qpos.npz`. Its pose and wrist-path residual informed the reach. No fresh Kimodo, ARDY or diffusion generation was run. A ready-made refrigerator pull was not available in the audited clips, so the turn, final approach, finger closure, pull and hold were authored explicitly against this scene.

The selected G1 model has actual link meshes, 29 body joints and 14 finger joints. Joint mapping was checked by names and forward kinematics. The source clip models differ slightly from the target in torso/shoulder frames, so all target link poses were regenerated from the selected G1/Dex3 model. The hand motor ordering is interleaved with the body tree and cannot be inferred by appending 14 finger values to a 29-joint vector.

## What was built and edited

1. Read the source USD transforms, refrigerator hinge, handle rail and 3,675 enabled collider parts. Preserve nominal meters and Z-up without applying the room reconstruction scale again.
2. Place the selected walking window into the clear aisle. Decelerate its start/end timing, infer stance intervals, anchor planted feet, and solve the legs to their edited targets.
3. Lower the pelvis 45 mm during the edited walk to leave knee extension margin. Use a positive-knee IK branch and remove spurious one-frame swing intervals that previously caused abrupt leg motion.
4. Author two foot placements to turn into the interaction stance at XY `[-1.94, -0.38]`, facing approximately 171.33°. Keep the soles on the modeled floor and shift the root during each placement.
5. Define a Dex3-specific grip frame from its actual palm/finger geometry. The wrist is not the grip point. The selected wrist-local point is `[0.111948, 0.074234, 0]` m, aligned to the nominal 1.15 m-high straight rail section.
6. Orient the hand at a pregrasp point before approaching the rail, then close the seven right-hand joints to the stored targets. The source reach is a seed; a bounded inverse-kinematics solve supplies the scene-specific arm poses.
7. Derive the rail and wrist targets directly from the source hinge arc while smoothly changing the door coordinate from 0° to 65°. Keep the two stance feet fixed and finish holding the pose.
8. Export the actual target-model FK into a portable USD animation. Render the composed scene natively in Isaac Sim, check sampled USD poses against the saved FK, and inspect both the approach view and fridge view.

The arm solve keeps waist roll and pitch at zero and allows waist yaw within ±0.5 rad. It limits right shoulder roll to at most −0.02 rad, right wrist yaw to at least −1.45 rad and wrist pitch to at most 1.42 rad. A previous-pose preference smooths the redundant arm/waist solution across the grasp-to-pull boundary.

## Measured result

The selected motion is `take06`. These are numerical residuals against the estimated scene and robot model, **not millimeter-accurate measurements of the real kitchen**.

| Measurement | Result |
|---|---:|
| Saved poses checked against scene geometry | 589 / 589 |
| Robot joint-limit violation | 0 |
| Unintended scene-intersection frames | 0 |
| Maximum planted-foot target error | 0.0164 mm |
| Maximum drift within an authored stance interval | 0.0163 mm |
| Minimum visual sole height | −0.0011 mm |
| Maximum stance sole height | 0.0135 mm |
| Maximum grip-frame position error during pull/hold | 0.0465 mm |
| Maximum grip orientation error during pull/hold | 0.0168° |
| Final door opening | 65° |
| Largest joint change between 30 Hz samples | 0.1984 rad |
| Largest root translation between samples | 36.22 mm |

The free pregrasp path is a guide, not a contact constraint: its largest wrist-target deviation is 12.1 mm after applying the conservative wrist limits. The final grip and door-following phase satisfy the much tighter values above. The finger closure is an authored geometric grasp reference; an earlier hand-only mesh check placed the distal finger surfaces about 1 mm from the rail. No force closure or load-bearing grasp is claimed.

Environment clearance uses the robot's visible meshes, rather than relying on undersized foot collision proxies, against the source room colliders. The selected door colliders move with the complete planned hinge transform. The independent self-collision audit checks the visual surfaces separately from the robot model's native collision approximations. Consult the exact selected audit in `evidence/self_collision/receipt.json` for its verdict, pair exclusions, baseline housing seams and native contact census.

The scene checks are performed at saved 30 Hz poses. They are not a continuous swept-volume certificate between samples. The robot's source joint-interface exclusions and existing modeled housing seams are recorded; a blanket claim that every pair of model triangles is disjoint would be inaccurate.

The movie contains the 15 Hz subset of the same timeline, including the final displayed pose. Its encoded duration is approximately 19.667 seconds because the final frame occupies one display interval. The room and robot were rendered with native RTX; the preview is not a Gaussian splat or a generated illustration.

## What failed and why

The first attempts exposed foot sliding/overextension, an IK knee-branch flip, and fingers intersecting the handle during the approach. Fixed foot anchors, pelvis clearance, an explicit knee branch and a staged pregrasp corrected those failures.

Changing the posture preference abruptly at the start of the pull caused a visible elbow step. A previous-pose continuity term removed that change. Allowing waist roll/pitch also produced torso-trim intersections, while a shoulder solution folded slightly into the torso. Keeping the torso upright and restricting the shoulder removed those contacts while retaining the full opening arc.

The remaining wrist problem was subtler: the exact visible wrist surfaces were separated, but the source model's convex collision approximation intersected at a compound wrist bend. Tightening yaw alone reduced the overlap but did not remove it. A diagnostic pitch/yaw grid and an additional pitch bound produced a reference with no native self-contact at any saved pose. The collision model and its masks were left unchanged.

The verification tools also needed stronger input binding. Early checks could be fooled by moving foot/wrist targets along with an invalid trajectory, incomplete collision indexes, or changed door/finger schedules. Independent review added source-geometry/model hashes, fixed-stance drift, task-derived handle targets, complete finger/schedule checks and explicit camera/export bindings. Rejected iterations and their original evidence remain in the authoring run; the transfer package contains the selected result.

## Remaining work before physical execution

The reference needs a G1 locomotion/manipulation controller to track it while maintaining balance. That controller must establish a real Dex3 grasp, overcome door friction/seal/latch resistance, coordinate arm and body forces, and handle tracking or perception error. No dynamic rollout, contact force measurement, actuator torque evaluation, robustness sweep or hardware execution was performed here.

The room originated from the supplied MOV alone. Its scale and unseen mechanisms remain estimates, and door mass, friction, resistance and exact handle dimensions have not been measured. Those uncertainties also limit LiDAR localization and sim-to-real performance claims. This motion does not validate the kitchen as an exact real-world replica or demonstrate localization accuracy.

The delivered library, explicit phase schedule, named joint data, physical room layer and robot model provide the starting inputs for that controller work. Fresh verification is required after changing a clip, stance, grip, robot model or scene geometry.
