# G1 + Dex3 refrigerator motion

Open **G1_fridge_motion.mp4** or **index.html** to watch the 19.6-second sequence: walk about 1.96 m, turn into a stance, reach with the right Dex3 hand, close the fingers around the rail, and open the modeled fridge door to 65°. It finishes holding the door open.

This is an editable **kinematic motion reference**. The door coordinate and robot poses are authored together. A physics controller has not executed the pull, and grip force, balance under load, door resistance, or real-robot transfer have not been established.

## Files

| Path | Use |
|---|---|
| `scene/scene.usda` | One composed room + G1 + door animation, ready to open in Isaac Sim or a USD viewer. |
| `scene/environment/iterations/mesh09/scene.usda` | Unchanged physical environment layer with its original colliders and articulations. |
| `scene/g1_motion.usdc` | Real robot meshes with sampled link poses. |
| `scene/preview_overlay.usda` | Preview-only physics disabling and animated fridge door. |
| `robot/g1_dex3/` | Original G1 MJCF and every referenced mesh, including both articulated Dex3 hands. |
| `motion/trajectory.npz` | All 589 poses at 30 Hz, door angle, timestamps, phases and authored constraints. |
| `motion/trajectory.csv` | Same root/joint/door coordinates with readable column names. |
| `motion/joint_order.json` | Exact coordinate order and units. |
| `motion/task.json` | Scene/task identity, stance, handle frame, finger targets and camera settings. |
| `library/source/` | Two existing walking candidates and one existing reach seed, plus provenance and joint mapping. |
| `library/edited/` | The nine placed and corrected phase clips used in this sequence. |
| `recipe.json` | Selection and placement parameters and the hinge/grip frames. |
| `REPORT.md` | Method, iterations, measured results and remaining limitations. |
| `evidence/` | Bound numerical results, independent self-collision audit, native rendering receipts and source PNG frames. |
| `code/` | Planner, verifier, exporter, native renderer and portable command entrypoint. |

The robot has 29 body joints and seven joints per hand: 43 hinge coordinates plus a seven-coordinate floating root. **Finger coordinates are interleaved with the arms.** Use joint names or `joint_order.json`; do not assume all fingers follow the 29 body coordinates. Root quaternion order is WXYZ; length is in meters, angles in radians, world Z points up.

Room scale and dimensions remain nominal authored estimates from the video; meters describe coordinate units, not independently measured real-world accuracy.

## Open in Isaac Sim

Use File → Open on `scene/scene.usda`. Select `/World/PreviewCameras/Overview` for the approach or `/World/PreviewCameras/Task` for the fridge. Scrub the timeline from frame 0 to 588 at 30 frames per second. The two-camera movie switches views at 10 seconds.

The composed preview disables scene physics so the sampled reference can be reviewed consistently. The underlying physical room layer is preserved byte for byte. The combined robot layer is FK animation; the articulated robot model for later controller work is supplied separately as MJCF. Pressing simulation Play does not turn this reference into a validated force-driven manipulation controller.

## Check and reproduce

Run from this extracted directory. Checking transfer hashes needs only Python:

```bash
python code/g1_motion_bundle.py check
```

For CPU planning, USD export and geometric verification, use a Python environment with `requirements.txt` installed. Python 3.10 was used for this handoff. Rendering additionally uses Isaac Sim 5.1's Python runtime, an RTX-capable GPU and FFmpeg.

```bash
python -m pip install -r requirements.txt
python code/g1_motion_bundle.py verify
python code/g1_motion_bundle.py build
```

`verify` independently reloads the model and unchanged room, recomputes FK, and checks all saved frames against the source scene collision geometry. Its result is written to `reproduced/verify/verification/results.json`. Self-collision evidence is a separate audit in `evidence/self_collision`; this command does not claim to repeat that audit or simulate dynamics.

`build` reconstructs the reference from the three bundled source clips. It writes `reproduced/build/motion/`. All world transforms are regenerated from the target model. No motion-generation service, model download or access to the original machine's project paths is required.

To edit placement, door angle or grip offset, make a copy of `motion/task.json`, edit its `plan` or `grasp` values, and pass `--task your_task.json` to `build`. More involved changes such as turn timing, contact schedule, IK preferences or approach path are explicit in `code/g1_motion_plan.py`. The edited phase clips already share the original kitchen coordinate frame; arbitrary concatenations still require contact/transition correction and fresh verification.

To verify and export a rebuilt or edited trajectory:

```bash
python code/g1_motion_bundle.py verify --trajectory reproduced/build/motion/trajectory.npz --task reproduced/build/motion/task.json
python code/g1_motion_bundle.py export --trajectory reproduced/build/motion/trajectory.npz --task reproduced/build/motion/task.json
```

Open `reproduced/export/scene/scene.usda` to review that export. For native rendering, run the entrypoint with Isaac Sim's Python interpreter:

```bash
python code/g1_motion_bundle.py render --scene reproduced/export/scene/scene.usda --gpu 0
```

Choose the physical GPU ordinal appropriate for your machine. The reference was rendered on physical GPU 3 of the authoring workstation. The portable helper clears `CUDA_VISIBLE_DEVICES` before starting native Isaac rendering to preserve consistent native GPU enumeration.

The archived receipts retain original absolute source paths for provenance. `portable.json` maps bundled dependencies to local relative paths; the command helper rebuilds a local task before execution. Source clips are existing local assets, with their known provenance preserved. The included Kimodo license applies to that source project and does not establish licensing for unrelated robot meshes or walking data.
