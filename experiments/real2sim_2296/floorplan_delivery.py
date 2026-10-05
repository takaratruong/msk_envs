#!/usr/bin/env python3
"""Package the reviewed layout, source comparisons and editable authoring copy."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, html, json, shutil
import cv2
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2)+'\n')
def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(src)==sha(dst)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args(); run=a.run.resolve(); candidate=run/'iterations/layout05'; out=a.output.resolve()
    if out.exists(): raise FileExistsError('Use a fresh package path')
    out.mkdir(parents=True)
    integration=json.loads((candidate/'integration.receipt.json').read_text())
    assert integration['scene']['sha256']==sha(candidate/'scene.usda')
    for item in integration['payload']:
        src=candidate/item['path']; assert sha(src)==item['sha256']; copy(src, out/'scene'/item['path'])
    for name in ['scene.manifest.json', 'build_inputs.json']:
        copy(candidate/name, out/'scene'/name)
    write(out/'evidence/portable_aliases.json',dict(
        schema='real2sim-historical-provenance-aliases/v1',
        status='Original build and Blender receipts retain byte-identical historical paths. These aliases locate included evidence without changing those bound inputs.',
        aliases=[dict(source_file='scene/build_inputs.json',field='native_acceptance_receipt.path',
                      historical_value='../../reviews/pass5/receipt.json',package_path='evidence/review/receipt.json',
                      sha256=sha(run/'reviews/pass5/receipt.json'))]))
    review=run/'reviews/pass5'; receipt=json.loads((review/'receipt.json').read_text())
    # The selected reviewed source and its transitive payload are additionally
    # checked below in the relocated USD, rather than trusting a wrapper hash.
    for name in ['receipt.json','report.md','input_union.json','invariants.json','cooler_scaling.json',
                 'global_overlap.json','camera_clearance.json','cooler_projection.json',
                 'cooler_clearance_estimate.json','cooler_sweep.json','cooler_sweep_translated.json','final_checks.json']:
        copy(review/name,out/'evidence/review'/name)
    copy(candidate/'integration.receipt.json',out/'evidence/integration.receipt.json')
    for src in sorted((run/'astra').glob('*.json')): copy(src,out/'evidence/astra'/src.name)
    copy(run/'astra/REPORT.md',out/'evidence/astra/REPORT.md')
    for name in ['cooler_controls.json','cooler_proposal.json','cooler_compare_002057.jpg','cooler_compare_002081.jpg']:
        src=run/'root_fit'/name
        if src.exists(): copy(src,out/'evidence/cooler'/name)
    for src in sorted((candidate/'diagram').iterdir()):
        if src.is_file(): copy(src,out/'comparison/plan'/src.name)
    for src in sorted((candidate/'comparison').iterdir()):
        if src.is_file(): copy(src,out/'comparison'/src.name)
    comparison=json.loads((candidate/'comparison/results.json').read_text())
    for view in comparison['views']:
        source=Path(view['source']);assert sha(source)==view['source_sha256']
        copy(source,out/'evidence/native/source'/source.name)
    blend=run/'authoring/layout05.blend'; br=blend.with_suffix('.receipt.json')
    b=json.loads(br.read_text()); assert b['source_usd_sha256']==sha(candidate/'scene.usda')
    assert sha(blend)==b['output']['sha256'] and b['packed_textures_reopened_verified']
    for src in sorted(blend.parent.glob('layout05.*')):
        if src.suffix in ['.blend','.json']: copy(src,out/'authoring'/src.name)
    native=[]
    for frame in ['000000','000847','001499','002158']:
        name='frame_'+frame; before=a.baseline/('render_'+name+'.png'); after=candidate/('render_'+frame+'.png')
        assert before.is_file() and after.is_file()
        orig=cv2.imread(str(candidate/'comparison'/(name+'_source.png')))
        views=[orig,cv2.imread(str(before)),cv2.imread(str(after))]
        # Resizing is the sole comparison operation. Source camera poses are fixed.
        w,h=854,480; strip=[]
        for im,label in zip(views,['SOURCE / UNDISTORTED','PREVIOUS MESH LAYOUT','REVISED MESH LAYOUT']):
            im=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA);bar=np.full((38,w,3),25,np.uint8)
            cv2.putText(bar,label,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.63,(245,245,245),1,cv2.LINE_AA)
            strip.append(np.concatenate([bar,im]))
        panel=out/'comparison'/(name+'_before_after.jpg');cv2.imwrite(str(panel),np.concatenate(strip,axis=1))
        ar=after.with_suffix('.png.receipt.json'); nr=json.loads(ar.read_text())
        # Preserve the exact native renderer receipt; the comparison is not a render.
        copy(after,out/'evidence/native/after'/after.name)
        copy(ar,out/'evidence/native/after'/ar.name)
        copy(before,out/'evidence/native/before'/before.name)
        before_receipt=before.with_suffix('.png.receipt.json')
        if before_receipt.exists():copy(before_receipt,out/'evidence/native/before'/before_receipt.name)
        native.append(dict(frame=frame,before_sha256=sha(before),after_sha256=sha(after),
                           before_path='evidence/native/before/'+before.name,after_path='evidence/native/after/'+after.name,
                           native_receipt_sha256=sha(ar),panel=str(panel.relative_to(out)),panel_sha256=sha(panel)))
    s=Usd.Stage.Open(str(out/'scene/scene.usda')); assert s
    assert UsdGeom.GetStageMetersPerUnit(s)==1 and UsdGeom.GetStageUpAxis(s)=='Z'
    layers=[]; assets=[]
    for layer in s.GetUsedLayers():
        if not layer.realPath: continue
        p=Path(layer.realPath).resolve(); assert p.is_relative_to(out)
        layers.append(dict(path=str(p.relative_to(out)),sha256=sha(p)))
    prims=list(s.Traverse()); joints=[str(p.GetPath()) for p in prims if p.IsA(UsdPhysics.Joint)]
    original=Usd.Stage.Open(str(candidate/'scene.usda'))
    assert joints==[str(p.GetPath()) for p in original.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert sum(p.IsA(UsdGeom.Mesh) for p in prims)==3864 and len(joints)==116
    assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)==3678
    for prim in prims:
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in [Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray]:continue
            vv=attr.Get(); vv=vv if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray else [vv]
            for v in vv or []:
                if not v or not v.path:continue
                p=Path(v.resolvedPath).resolve(); assert p.is_file() and p.is_relative_to(out)
                assert not Path(v.path).is_absolute()
                assets.append(dict(attribute=str(attr.GetPath()),path=str(p.relative_to(out)),sha256=sha(p)))
    metrics=json.loads((run/'astra/fit_results.json').read_text())
    line_rows=[]
    for key,label in [('glass_sink_line','Sink-side glass'),('glass_tables_line','Table-side glass'),('glass_back_line','Back glass')]:
        v=metrics[key]['scores']['heldout']; line_rows.append((label,v['n'],v['baseline_median'],v['median']))
    clear=json.loads((candidate/'diagram/model_clearances.json').read_text())['clearances']
    md='''# IMG_2296 — floor-plan revision 2

Open `scene/scene.usda` in Isaac Sim or another OpenUSD application. This is one composed, textured mesh scene with colliders and articulations. `authoring/layout05.blend` is an editable copy of the same geometry with all 16 textures packed. Blender retains the 116 USD joints as inspectable metadata and arrow empties; its solver does not receive working USD articulation constraints.

This revision improves spatial layout against fixed cameras from the original MOV. It is a source-supported relative reconstruction, not an exact surveyed replica. No depth stream, camera-pose log, LiDAR scan or independent room measurement was available. Scale remains conditional on a nominal 0.9144 m countertop height.

## What changed

Three glass partitions now follow the observed floor contacts. A remeshed column connects the sink-side and table-side glass at the observed junction. The cooler position, facing and uniform size were fitted to visible corners; all its lid geometry and joint anchors were scaled together. A previously missing corridor wall surface was added, leaving the passage and unseen far continuation open. Floor and ceiling support canvases extend beneath this visible wall; their outer edges do not represent a measured room perimeter.

The kitchen runs, refrigerator, both islands, tables, stools and carts retain their previous modeled positions. A proposed sorting-island shift and dimensional refit were rejected because they worsened other camera views. The accepted change preserves all 116 joint paths and 4,582 unaffected prims. The composed scene has 3,864 mesh parts and 3,678 collision parts, with no Gaussian splats.

## Image alignment evidence

Fitting and held-out source controls are recorded in `evidence/astra/controls.json`. These line distances use the original radial camera calibration and exact undistortion; the native comparison images use the corresponding undistorted source cameras. Cameras were not adjusted for this layout iteration. The table below reports held-out median pixel-equivalent line residuals, not metric geometry error.

| Region | Held-out controls | Previous | Revised |
|---|---:|---:|---:|
'''
    for label,n,before,after in line_rows:md+=f'| {label} | {n} | {before:.2f} px | {after:.2f} px |\n'
    md+='''
Each accepted glass region improves its held-out median by more than 25%; none worsens. This gate applies to those fitted planes, not to the entire room. The new corridor wall had no baseline surface to score. Its separate floor-line holdouts are 10.79–25.67 px and corner-line holdouts 26.67–32.93 px. Column silhouette residuals improve from 27.52–70.49 px to 7.12–14.41 px across eight controls; those views had already been inspected in a prior diagnostic, so they are exploratory holdouts. Three blurred cooler corner holdouts improve from 285.75/168.83/198.74 px to 48.72/14.87/34.76 px. Nearby viewpoints and shared camera estimates limit independence.

`comparison/` contains fixed-camera source/before/after panels and source/revised overlays. The side-by-side panels use only resizing and labeled composition. Diagnostic overlays are explicitly 50/50 alpha blends of the resized source and render; they do not alter either source image or scene geometry. Lighting, glossy reflections, cabinet proportions and the laboratory beyond the glass still differ visibly. No full-room photorealism or global metric-accuracy pass is claimed.

## Modeled passage measurements

These are shortest distances between projected collider envelopes below 1.2 m, in nominal meters. They are not surveyed aisle widths or G1 clearance certification; articulation state and robot shape matter.

| Collider-envelope separation | Previous | Revised |
|---|---:|---:|
'''
    for label,v in clear.items():md+=f"| {label} | {v['before']['distance_m']:.3f} m | {v['after']['distance_m']:.3f} m |\n"
    md+='''
## Collision and articulation findings

The independent review checks initial cross-asset geometry, finite glass spans, joint identities, cooler anchor scaling and source camera clearance. There are no initial cross-asset hull overlaps in its 3,678-collider census. This is a geometric result, not a new full-room dynamic stability test.

The cooler is mobile and starts clear of the glass. Its lid meets the pane around 29° in the sampled in-place sweep. The authored 105° mechanical range is preserved. A separate diagnostic that moves the cooler 123.8 mm toward the room (CAD delta approximately [-0.12380, -0.00015, 0] m) clears the sampled full range with at least 3 mm separation. The production scene keeps the source-fitted initial placement. Move the cooler forward before opening it fully; the diagnostic does not establish continuous collision clearance or a measured real-world distance.

The sink-side glass receives a 0.25 m unmeasured far-span completion. The table-side glass has a 2.91757 m modeled span with inherited far extent; it did not receive that extra 0.25 m. An older integration change-description repeats the sink phrase under the table entry; the geometry and this explanation distinguish them. Back-glass endpoints and the far corridor truncation remain completion assumptions. No transverse corridor closure was added.

## Process and bottlenecks

1. Reused the original video frames, camera calibration and nominal CAD frame. Excluded the disconnected four-camera reconstruction component from the global fitting work.
2. Astra annotated and froze fit/holdout controls, fitted glass lines, tested island alternatives, and proposed the corridor surface and column junction. Root integrated accepted geometry into the articulated USD, preserving existing object identities and textures.
3. Root refined the cooler against additional visible floor corners, maintaining uniform articulation geometry. An independent lane challenged source associations, radial-camera scoring, missing finite spans, source invariants and moving-part clearance.
4. Rendered four unchanged source cameras in native Isaac, inspected comparisons, generated the top-down collider plan, and imported the final USD into Blender. Blender mesh/hierarchy/UV/material/camera checks passed; all packed textures survived reopening.
5. Packaged relative USD dependencies and the authoring file, reopened the relocated USD, and bound the files with SHA256 hashes.

The main limits are monocular scale ambiguity, camera drift/conditioning near the floor horizon, sparse visible corners, glass and occlusion, and unseen geometry. Cabinet interiors and hinge/mass/friction parameters remain engineering assumptions. These artifacts do not yet validate LiDAR localization accuracy or correlated hardware performance. The G1 physics work uses a separate task adapter derived from this layout; old mesh09 trajectory receipts do not transfer automatically. Its selected fridge door is passive and a synthetic bottle is added for the task. Full pickup/placement acceptance is tracked separately.

## Files and reproducibility

- `scene/scene.usda`: portable simulation source; its relative geometry sublayer and textures are under `scene/environment/`.
- `authoring/layout05.blend`: editable packed authoring copy, with import/reopen receipts alongside it.
- `comparison/plan/floorplan_before_after.png` and `.svg`: projected modeled footprints; `model_clearances.json` and `footprints.json` provide numeric data.
- `evidence/astra/`: controls, proposals, rejected diagnostic scores and source-input hashes. `evidence/cooler/`: additional root fit. Their original machine-local paths are historical provenance, not runtime USD dependencies.
- `evidence/review/`: exact reviewed layout and cooler-sweep findings. `evidence/native/`: matching Isaac render receipts.
- `evidence/portable_aliases.json`: maps the historical review path in `scene/build_inputs.json` to the included review receipt. The build file remains byte-identical because Blender's input receipt binds its original hash. Historical provenance paths are not runtime scene dependencies.
- `package.manifest.json` and `portability.receipt.json`: package hashes and relocated scene checks.

The original MOV and full camera-reconstruction dataset are intentionally not duplicated here. Reproducing the source fitting requires those original inputs; the scene itself opens from this package alone. Prior deliveries remain untouched.
'''
    (out/'README.md').write_text(md); (run/'REPORT.md').write_text(md)
    rows=''.join(f'<figure><img src="{r["panel"]}" loading="lazy"><figcaption>Frame {r["frame"]}: unchanged source camera; resize only.</figcaption></figure>' for r in native)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>IMG_2296 floor-plan revision</title><style>body{font:17px/1.55 system-ui;color:#1e2935;background:#f3f4f1;margin:36px auto;max-width:1500px;padding:0 24px}h1{font-size:34px;margin-bottom:8px}figure{margin:24px 0;background:white;padding:12px;border-radius:10px}img{width:100%;height:auto}a{color:#006d72}figcaption{font-size:14px;color:#48535e}.intro{max-width:900px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card{padding:16px;background:#fff;border-radius:10px}table{border-collapse:collapse;background:#fff}td,th{padding:8px 18px;border-bottom:1px solid #dde3e5;text-align:left}</style><h1>IMG_2296 · revised floor plan</h1><p class="intro">One articulated mesh scene, refined against fixed video cameras. The glass junction, column, cooler and corridor now follow more of the observed layout. Geometry is in nominal meters; no independent metric survey or complete-room photorealism is claimed.</p><div class="grid"><div class="card"><b>Simulation</b><br><a href="scene/scene.usda">OpenUSD scene</a><br>3,864 meshes · 3,678 colliders · 116 joints</div><div class="card"><b>Authoring</b><br><a href="authoring/layout05.blend">Blender scene</a><br>16 packed textures · joint metadata</div><div class="card"><b>Evidence</b><br><a href="README.md">Process, changes and limits</a><br><a href="evidence/review/report.md">Independent layout review</a></div></div><figure><img src="comparison/plan/floorplan_before_after.png"><figcaption>Projected collider footprints. Teal highlights revised geometry. Open boundaries remain unknown.</figcaption></figure><p>The mobile cooler starts clear, but its lid meets the glass around 29°. A separate 12.4 cm forward translation clears its sampled 105° range. The source-fitted scene preserves that initial placement and obstruction.</p>'''+rows
    (out/'index.html').write_text(page)
    write(out/'portability.receipt.json',dict(schema='real2sim-floorplan-package-portability/v1',produced_at=datetime.now(timezone.utc).isoformat(),
        scene_sha256=sha(out/'scene/scene.usda'),layers=layers,assets=assets,mesh_parts=3864,colliders=3678,joints=len(joints),
        all_dependencies_within_package=True,all_joint_paths_preserved=True,blend_sha256=sha(out/'authoring/layout05.blend'),
        source_layout_review_sha256=sha(review/'receipt.json'),blender_receipt_sha256=sha(br),native_views=native,script_sha256=sha(__file__)))
    write(out/'package.manifest.json',dict(schema='real2sim-floorplan-package/v1',run_id=run.name,entrypoint='scene/scene.usda',
        created_at=datetime.now(timezone.utc).isoformat(),claim='Source-supported relative floor-plan refinement with explicit cooler obstruction; no metric survey or full-task physics claim',
        files={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(package=str(out),manifest_sha256=sha(out/'package.manifest.json'),files=len(list(out.rglob('*')))),indent=2))


if __name__=='__main__':main()
