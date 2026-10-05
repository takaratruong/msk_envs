# Native Gaussian appearance in the canonical OpenUSD scene

Inspected and tested 2026-09-12. This document covers the export boundary only. The authoritative reconstruction coordinates remain in the float32 Gaussian PLY and reconstruction manifest; visual rendering, metric calibration, collision geometry and articulated objects have separate acceptance checks.

## Decision and supported representation

For the locally installed Isaac Sim 5.1, convert standard Gaussian PLY into a **legacy NuRec Volume USDZ**, then reference that asset from the canonical scene's common reconstruction frame. Generic Gaussian PLY is an interchange format; a PLY reference or ordinary `UsdGeom.Points` prim does not provide native Gaussian rendering.

The newer `UsdVol.ParticleField3DGaussianSplat` representation is preferred upstream for newer runtimes, including Isaac Sim 6.0. It is a different export target. NVIDIA lists the legacy NuRec path for Kit 107.3–110.1 / Isaac Sim 5.0–6.0. These are upstream compatibility claims, not proof that this machine has loaded the required rendering components. [Pinned export documentation](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/README.md), [Isaac Sim 5.1 neural volume documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html).

The delivered [export_splats_usd.py](export_splats_usd.py) creates:

| Package member | Contents |
|---|---|
| `default.usda` | `/World` default prim and `/World/gauss` `UsdVol.Volume`, with identity transform |
| `model.nurec` | Gzip-wrapped MessagePack carrying the pinned official NuRec configuration and Gaussian tensors |
| `PROVENANCE.json` | Source PLY link/hash, converter and upstream hashes, precision statistics, package versions and unverified render status |
| External `<output>.usdz.receipt.json` | The same provenance plus package hash and completed structural checks |

This payload is **`.nurec`, not `.nvdb`**. The Volume has `OmniNuRecFieldAsset` children for density and emissive color. `omni:nurec:useProxyTransform=false` allows its authored and inherited USD transforms to take effect. The bridge matches these upstream schema fields, uses an identity color matrix and preserves the standard legacy render settings. Its bounds include both original and quantized Gaussian centers to avoid excluding a center solely through rounding. [Pinned serializer](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/usd/nurec/serializer.py).

## Run without installing the training stack

The existing reconstruction venv already has the four CPU dependencies. No packages or Isaac environments were installed or modified for this converter.

| Component | Version used by passing tests |
|---|---|
| Python | 3.10 |
| numpy | 2.2.6 |
| plyfile | 1.1.5 |
| msgpack | 1.1.2 |
| usd-core | 25.11 |

From the repository root, replace `PATH_TO_FINISHED_APPEARANCE_PLY` with the finished standard `appearance.ply`, and choose a new output filename:

```bash
runs/real2sim-2296/20260912T0540Z/venv/bin/python \
  experiments/real2sim_2296/export_splats_usd.py \
  --input PATH_TO_FINISHED_APPEARANCE_PLY \
  --output runs/real2sim-2296/20260912T0540Z/appearance.usdz \
  --upstream-cache runs/real2sim-2296/20260912T0540Z/export_upstream
```

The first conversion downloads **only the pinned template source**, verifies its hash and retains its original Apache-2.0 header. It does not send reconstruction data anywhere. Later conversions can add `--offline`; missing or modified cached source causes a failure. Existing outputs or receipts are never overwritten.

| Upstream pin | Value |
|---|---|
| Repository | `nv-tlabs/3dgrut` |
| Commit | `a37ef721012dea0f29c0fcfff2d525023b4e854a` |
| Source | `threedgrut/export/usd/nurec/templates.py` |
| SHA256 | `8632ad7c712ce3fdc59263ee725ef838869f2b972f00c124cdb8ad576599d57f` |

The bridge executes the unchanged ASTs of `fill_3dgut_template` and `_fill_state_dict_tensors` from that verified file. It bypasses an unused import into the full training tree and authors the small USD wrapper locally. This avoids a broad vendored fork and dependencies such as NVIDIA camera/training packages. The full fetched source and license attribution remain in the cache. [Pinned template](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/usd/nurec/templates.py).

For an independent environment, the tested dependency command is `python -m pip install numpy==2.2.6 plyfile==1.1.5 msgpack==1.1.2 usd-core==25.11`; use a dedicated venv if required. This installation was unnecessary locally. The upstream full-stack alternative is `python -m threedgrut.export.scripts.transcode model.ply -o model.usdz --format nurec` at the pinned commit, without `--apply-coordinate-transform` and without dataset normalization. [Transcoder implementation](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/scripts/transcode.py).

## Input, frame and precision contract

Accepted input is a standard **3D** Gaussian PLY with float32 `x/y/z`, opacity logits, three log-scales, four quaternion components in **wxyz** order, three DC SH coefficients, and complete SH degree 0–3. Higher-order PLY coefficients use channel-major order and are rearranged into coefficient-major RGB for NuRec. The local `appearance.py` exporter follows this convention. A colored point-cloud PLY, a 2DGS surfel PLY, partial SH storage or silently narrowed float64 data is rejected. [Upstream PLY importer](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/importers/ply.py).

The actual NuRec implementation stores **pre-activation tensors** and configures sigmoid opacity, exponential scale and quaternion normalization in the renderer. The broad README table calls NuRec values post-activation; the pinned exporter/template implementation is the operative contract for this bridge. Do not exponentiate scales or sigmoid opacity a second time before passing this PLY. [Pinned exporter](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/usd/nurec/exporter.py), [pinned template](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/usd/nurec/templates.py).

No centering, normalization, axis rotation, scale fit, point deletion or geometry completion is performed. `upAxis=Z` and `metersPerUnit=1` are output encoding metadata; they do **not** establish gravity or measured meters. Until calibrated, source coordinates remain raw reconstruction units. A single calibrated transform in the canonical parent must apply equally to appearance, geometric surfaces, collision shapes and cameras.

The legacy payload casts every Gaussian tensor, including XYZ, to float16. Consequently, unchanged frame and scale do not imply bit-identical positions. The receipt reports maximum component error, maximum/p95/RMS point displacement, bounds before/after rounding, and maximum absolute error for every tensor. If the eventual common transform is a rigid rotation plus scale `s` meters per raw unit, multiply reported point-displacement values by `abs(s)` to obtain meters. These figures measure export rounding, not reconstruction accuracy.

The source PLY is linked relative to the USDZ's containing directory and hash-identified as the authority in both provenance records. Keep that float32 file with the deliverable; the USDZ does not contain a duplicate. Nonfinite values, float16 overflow, zero quantized quaternions and invalid activated scales are refused. Input hashes are checked before/after loading. The exporter uses ZIP_STORED members aligned to 64-byte boundaries and refuses ZIP64 rather than emitting an unsupported oversized USDZ. [Upstream packaging implementation](https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/threedgrut/export/usd/stage_utils.py).

## Join the canonical stage

Reference the package's `/World` default prim beneath the existing shared reconstruction parent. This schematic uses `/World/Capture`; substitute the canonical manifest's actual parent and an asset path relative to the canonical stage:

```python
appearance = UsdGeom.Xform.Define(stage, "/World/Capture/Appearance")
appearance.GetPrim().GetReferences().AddReference("appearance.usdz")
# The composed Volume is now /World/Capture/Appearance/gauss.
# The shared calibrated transform belongs on /World/Capture.
```

The renderer defaults live in layer `customLayerData`; a reference does **not** import that layer metadata into the canonical root. The root stage owner must explicitly inspect and apply the package's `renderSettings`, merging with the scene's intended settings. Keep `appearance_stage = Usd.Stage.Open(...)` alive while inspecting its layer; an ephemeral chained Stage/Layer expression triggered a `TfWeakPtr` lifetime crash in this local OpenUSD Python build during test-harness inspection. The converter itself retains its Stage objects and passed package checks.

The package is a visual representation. It has no mesh, collision APIs, semantic segmentation, object separation, joints, inertias or articulation limits. Add those verified representations under the same frame in the canonical scene. A cabinet door or manipulated item baked into this static Gaussian volume will remain visually present when a separate physics object moves unless its appearance has also been separated. LiDAR returns from the selected geometric representation require an actual sensor check; rendering the Gaussian RGB view is insufficient. Do not assume hiding a geometry prim from cameras preserves RTX sensor visibility.

## Local Isaac capability and completed synthetic render check

Read-only filesystem inspection found:

| Environment | Installed build | Consequence |
|---|---|---|
| `/home/ubuntu/projects/simtoolreal/repo/.venv_isaacsim` | `5.1.0-rc.19+release.26219.9c81211b.gl` / package 5.1.0.0 | Candidate legacy NuRec renderer |
| `/home/ubuntu/miniconda3/envs/isaaclab` | `4.5.0-rc.36+release.19112.f59b3005.gl` / package 4.5.0.0 | Outside the pinned exporter compatibility range; do not select for this asset |

The 5.1 cache contains `omni.hydra.rtx`, `omni.volume`, Hydra delegates and `libnrend.so`. Both `omni.hydra.scene_delegate` and `omni.hydra.usdrt_delegate` manifests name **`omni.usd.schema.omni_nurec_types` as optional**. Its directory was not found in the cache; the running app also reported it unavailable and its concrete USD type unregistered. **The subsequent native render succeeded anyway through the installed RTX backend.** This absent optional extension is therefore not a rendering blocker on this installation. The separately documented extension registers NuRec USD types. [NuRec schema extension overview](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_nurec_types/0.0.1/Overview.html).

Converter receipts deliberately retain `isaac_render=not_run`: conversion alone does not launch a renderer, and `usd-core` can compose unknown typed prims without rendering them. The separate [isaac_render.py](isaac_render.py) helper now uses `SimulationApp` and Replicator RGB, checks the requested existing camera, explicitly applies root/referenced render settings, and can produce a hidden-Volume control image. Root render-setting opinions override referenced defaults; unresolved conflicts between referenced defaults are refused.

The SH3 synthetic smoke test on physical GPU **4**, PCI **C6**, produced the expected red, green and blue Gaussians. Hiding the Volume produced an all-black control: **48,657 pixels** changed by more than 8 levels, with maximum difference **253**. The existing identity camera and its render-product relationship were exact. Measured finite-Gaussian color centroids differed by **0.63–1.35 pixels** from pinhole projections of the raw 3D centers; this is not a pixel-equivalence or reconstruction-accuracy claim.

Rendering and receipt generation completed in **33.2 seconds**. The process then exceeded its 180-second watchdog during slow plugin shutdown. All owned renderer process groups were cleaned up. The delivered helper subsequently changed only `fast_shutdown=False` to Isaac's normal `True`; that shutdown-only change was syntax/diff checked and awaits the real-scene run. The exact rendering-tested script is archived with its own hash. No additional render was launched after the requested stop.

Durable evidence, exact invocation, tested script, fixture, both PNGs, logs and hash-linked checks are in [the smoke review folder](../../runs/real2sim-2296/20260912T0614Z/reviews/isaac_smoke/README.md). Only the synthetic SH3 path has native image evidence so far. The actual reconstructed capture still requires known-pose comparison against gsplat renders and source images, with fixed intrinsics, exposure and background. A renderer change can itself change pixels. Plain Gaussian export here includes no PPISP shaders or camera-response training state.

## Verification performed

Final tested converter SHA256: `7ea769e422a4eb9641382eeaa11b700fb4cc4dafe6634c2477a9d6bbcdb3bff1`.

CPU CLI conversions passed for complete SH degrees 0, 1, 2 and 3, including ASCII and big-endian PLY input. Each package passed OpenUSD opening, payload resolution, all-tensor float16 round-trip equality, identity transform, ZIP alignment and CRC checks. Independent property-index probes checked higher-order SH channel ordering. Source hashes remained unchanged and receipts matched converter/input/output hashes.

A composed-stage probe applied translation `(4,-2,1)`, 90° about Z and scale `0.37` on the common parent; the referenced asset inherited that transform and retained valid payload resolution. The same probe confirmed root render metadata is not imported by an ordinary reference. Negative probes rejected colored-only PLY, NaN, float16 overflow, zero quaternion, activated scale underflow, float64 narrowing, empty input, changed cache, missing offline cache and output overwrite.

The four-point SH3 test deliberately included an X coordinate near 100 raw units. Export rounding measured maximum XYZ component error **0.01235198974609375** and maximum point displacement **0.012369757577354139** raw units. These are synthetic precision-test results, not measurements of IMG_2296. That test's input SHA256 is `871c442d5a30c8c56405f072c6bfbf25990aa95a759db45b8ed93817deb10d52`; package SHA256 is `ee72ca9ea619e240fee73bd7589eea2513406752002ea0a6a031b8b05fc1e7bd`. Its machine-local inspection files are `/tmp/real2sim-nurec-final-test-8c5au7jh/final-sh3.usdz` and the adjacent receipt; production conversion will create durable equivalents in the run directory.

What worked was the small CPU serialization bridge, explicit coordinate/precision checks, and native SH3 Gaussian rendering with a hidden-Volume control. Cold pipeline compilation, excessive smoke-test warm-up and slow full plugin shutdown consumed the initial render budget. Remaining bottlenecks are measuring actual gsplat-to-Isaac appearance differences and meeting the scene's independent metric geometry, collision, LiDAR and articulation criteria. None is solved merely by changing the asset container to USDZ.
