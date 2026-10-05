#!/usr/bin/env python3
"""Package the physical candidate and hash-bound evidence without changing USD."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import zipfile
from pxr import Usd, Sdf, UsdUtils
from physical_verify import mechanical_verdict


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_file(path, expected):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'File differs from evidence: ' + str(path))
    return path


def physical_inputs(union_path, scene, scene_manifest, assets):
    """Validate the supported union without changing a failed component verdict."""
    union_path = union_path.resolve(strict=True)
    report = read(union_path)
    digest = sha(scene)
    require(report.get('schema') == 'real2sim-physical-union-verification/v1', 'Expected the combined native verification receipt')
    require(report.get('source_scene_sha256') == digest, 'Combined receipt scene differs')
    require(report.get('modeled_mechanical_scope_passed') is True, 'Combined mechanical scope has not passed')
    components = report.get('components', {})
    require({'full_sequence', 'full_range', 'support_followup'} <= set(components), 'Missing native component')
    required = {union_path}
    loaded = {}
    actual_assets = {str(Path(p).resolve()): sha(p) for p in assets}
    joint_paths = {j['path'] for j in scene_manifest['joints']}
    for label, entry in components.items():
        path = bound_file(union_path.parent / entry['path'], entry['sha256'])
        native = read(path)
        required.add(path)
        require(native.get('source_scene_sha256') == digest and native.get('source_files_unchanged') is True,
                'Native component does not bind unchanged source: ' + label)
        require(native.get('render_asset_files_sha256') == actual_assets, 'Native texture dependency set differs: ' + label)
        require(native.get('usd_layers_sha256') == {str(scene): digest}, 'Native composed layer set differs: ' + label)
        require(set(native.get('authored_initial_joint_state', {})) == joint_paths, 'Native joint inventory differs: ' + label)
        require(native.get('artifacts'), 'Native component has no artifacts: ' + label)
        native_inputs = native.get('input_files_sha256', {})
        for required_path in (scene, scene.with_suffix('.manifest.json'), scene.parent / 'build_inputs.json'):
            require(native_inputs.get(str(required_path)) == sha(required_path), 'Missing native source provenance: ' + label)
        require(any(Path(p).name == 'physical_verify.py' for p in native_inputs), 'Missing executed native verifier: ' + label)
        for name, expected in native['artifacts'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe native artifact path')
            required.add(bound_file(path.parent / name, expected))
        for key in ('input_files_sha256', 'usd_layers_sha256', 'render_asset_files_sha256'):
            for name, expected in native.get(key, {}).items():
                required.add(bound_file(name, expected))
        loaded[label] = native
    full, focused, support = (loaded[k] for k in ('full_sequence', 'full_range', 'support_followup'))
    require(mechanical_verdict(full, False), 'Actual full motion outcomes do not pass')
    require(mechanical_verdict(focused, False), 'Focused endpoint outcomes do not pass')
    require(mechanical_verdict(support, True), 'Actual support and handle outcomes do not pass')
    expected_driven = {j['path'] for j in scene_manifest['joints'] if not j.get('passive') and not j.get('retainer_locked')}
    expected_locked = {j['path'] for j in scene_manifest['joints'] if j.get('retainer_locked')}
    require({j['path'] for j in full['joint_outcomes'] if j['status'] == 'drive_cycle_measured'} == expected_driven,
            'Full receipt did not exercise every driven joint')
    require({j['path'] for j in full['joint_outcomes'] if j['status'] == 'locked_checked'} == expected_locked,
            'Full receipt did not check every retained joint')
    require(full['joint_denominators'].get('unselected_focused_diagnostic') == 0, 'Full sequence is only focused')
    corner_paths = {
        '/World/Assets/return_corner_pair/Joints/door_1_hinge',
        '/World/Assets/return_corner_pair/Joints/drawer_0_slide',
        '/World/Assets/upper_main_0/Joints/door_0_hinge',
    }
    corner_outcomes = {j['path']: j for j in focused['joint_outcomes'] if j['status'] == 'drive_cycle_measured'}
    require(set(corner_outcomes) == corner_paths and focused.get('request', {}).get('open_fraction') == 1.0,
            'Focused receipt did not test the three intended full endpoints')
    authored = {j['path']: j for j in scene_manifest['joints']}
    for name, outcome in corner_outcomes.items():
        joint = authored[name]
        endpoint = max((joint['lower'], joint['upper']), key=abs)
        require(math.isclose(outcome.get('open_command', math.nan), endpoint, abs_tol=1e-8), 'Focused command is not an authored endpoint')
        require(any(math.isclose(c.get('targets', {}).get(name, math.nan), endpoint, abs_tol=1e-8)
                    for c in focused['drive_commands']), 'Full endpoint was never commanded')
        tolerance = 3.0 if joint['type'] == 'revolute' else .018
        require(abs(outcome.get('observed_open', math.inf) - endpoint) < tolerance,
                'Measured full endpoint is outside the unchanged native tolerance')
    basin = next(p for p in support['support_probes'] if p['label'] == 'sink_bowl')
    settled = basin.get('basin_containment', {})
    require(basin.get('acceptance_kind') == 'curved_basin_containment' and basin.get('radius') == .025,
            'Superseded planar probe is not basin containment evidence')
    require(settled.get('passed') is True and settled.get('sphere_contained_below_rim') is True
            and settled.get('required_radial_margin_m') == .005
            and settled.get('radial_margin_over_drain_m', -math.inf) >= .005
            and settled.get('overlap_query_radius_expansion_m') == .001
            and settled.get('final_expected_overlap_colliders')
            and settled.get('final_unexpected_overlap_colliders') == []
            and settled.get('unexpected_bowl_contact_colliders') == []
            and settled.get('supported_latest_contact_sample_count', 0) > 0
            and settled.get('terminal_pose_sample_count', 0) >= 5
            and settled.get('terminal_window_seconds', 0) >= .25
            and 0 <= settled.get('terminal_max_displacement_m', math.inf) < .001
            and 0 <= settled.get('final_speed_m_s', math.inf) < .01,
            'Basin containment/support/settling criteria do not pass')
    # Additional audits/helpers referenced by the union must also survive relocation.
    for name, expected in report.get('artifacts', {}).items():
        required.add(bound_file(union_path.parent / name, expected))
    return required, report


def blender_inputs(blend, scene):
    directory = blend.parent
    receipt_path = directory / 'scene_viewable.receipt.json'
    fresh_path = directory / 'viewable_fresh_process_verification.json'
    receipt, fresh, delivery = read(receipt_path), read(fresh_path), read(directory / 'delivery.json')
    require(delivery.get('status') == 'passed' and delivery.get('scene_sha256') == sha(scene), 'Blender delivery scene differs')
    bindings = {str(Path(e['path']).resolve()): e['sha256'] for e in delivery['artifacts']}
    require(delivery.get('all_scene_dependencies_unchanged') is True, 'Authoring/geometry source changed')
    for name, expected in bindings.items():
        bound_file(name, expected)
    for p in (blend, receipt_path, fresh_path):
        require(str(p) in bindings, 'Unbound Blender file')
        bound_file(p, bindings[str(p)])
    require(receipt.get('status') == 'verified_authoring_copy' and receipt.get('source_usd_sha256') == sha(scene), 'Blender export is not verified for this USD')
    require(Path(receipt['output']['path']).resolve() == blend and receipt['output']['sha256'] == sha(blend), 'Blender output differs')
    require(fresh.get('status') == 'passed' and fresh.get('blend_sha256') == sha(blend)
            and fresh.get('source_usd_sha256') == sha(scene) and fresh.get('checks') and all(v is True for v in fresh['checks'].values()),
            'Blender fresh-process readback does not pass')
    for name, key in [('scene_viewable.inputs.json', 'inputs_receipt_sha256'), ('scene_viewable.import.json', 'import_receipt_sha256')]:
        bound_file(directory / name, receipt[key])
    # Include authoring receipts and packed-data readback, preserving originals.
    return [p for p in directory.iterdir() if p.is_file() and p.suffix in ('.json', '.md', '.py', '.png', '.exr')]


def demo_inputs(receipt_path, scene, assets):
    receipt_path = receipt_path.resolve(strict=True)
    demo, checked = read(receipt_path), read(receipt_path.parent / 'readback.json')
    require(demo.get('schema') == 'real2sim-commanded-articulation-demo/v1' and demo.get('status') == 'passed', 'Native demo is incomplete')
    require(demo.get('scene_sha256') == sha(scene) and demo.get('frame_count') == 210, 'Native demo source/frame count differs')
    require(demo.get('commanded_demonstration') is True and demo.get('g1_interaction') is False, 'Demo scope differs')
    require(demo.get('checks') and all(c['passed'] is True for c in demo['checks']), 'Native demo checks fail')
    require(checked.get('status') == 'passed' and checked.get('native_receipt_sha256') == sha(receipt_path)
            and checked.get('scene_sha256') == sha(scene) and checked.get('checks')
            and all(c['passed'] is True for c in checked['checks']), 'Demo readback differs')
    require(demo['texture_files_sha256'] == {str(Path(p).resolve()): sha(p) for p in assets}, 'Demo texture set differs')
    required = {receipt_path, receipt_path.parent / 'readback.json'}
    for name, expected in demo['input_files'].items():
        required.add(bound_file(name, expected))
    for frame in demo['frames']:
        required.add(bound_file(frame['path'], frame['sha256']))
    require(len(demo['frames']) == 210 and len({f['path'] for f in demo['frames']}) == 210, 'Demo frame sequence incomplete')
    required.add(bound_file(demo['video']['path'], demo['video']['sha256']))
    require(checked['video_sha256'] == demo['video']['sha256'], 'Demo decoder checked a different movie')
    required.add(bound_file(receipt_path.parent / 'trajectory.jsonl', demo['trajectory_sha256']))
    required.add(bound_file(receipt_path.parent / 'frames.json', demo['frames_receipt_sha256']))
    required.add(bound_file(receipt_path.parent.parent / 'controls/readback_demo.py', checked['readback_helper_sha256']))
    for artifact in checked['artifacts']:
        required.add(bound_file(artifact['path'], artifact['sha256']))
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--physics-receipt', type=Path, required=True)
    parser.add_argument('--blend', type=Path, required=True)
    parser.add_argument('--demo-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene, blend = args.scene.resolve(strict=True), args.blend.resolve(strict=True)
    run, experiment = scene.parents[2], Path(__file__).resolve().parent
    manifest_path = scene.with_suffix('.manifest.json')
    scene_manifest = read(manifest_path)
    require(scene_manifest['scene_sha256'] == sha(scene), 'Scene manifest differs')
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(scene)))
    require(not unresolved and len(layers) == 1, 'Expected one complete scene layer and resolved textures')
    native_files, physics = physical_inputs(args.physics_receipt, scene, scene_manifest, assets)
    authoring_files = blender_inputs(blend, scene)
    demo_files = demo_inputs(args.demo_receipt, scene, assets)
    output = args.output.resolve()
    extracted = output.parent / (output.stem + '_unpacked')
    require(not output.exists() and not extracted.exists(), 'Output or extraction directory already exists')
    files, aliases = {}, {}

    def add(src, dst):
        src = Path(src).resolve(strict=True)
        require(not Path(dst).is_absolute() and '..' not in Path(dst).parts, 'Unsafe package path')
        require(dst not in files or files[dst]['source'] == src, 'Conflicting package path: ' + dst)
        files[dst] = dict(source=src, sha256=sha(src), bytes=src.stat().st_size)
        aliases.setdefault(str(src), []).append(dst)

    for name in ('scene.usda', 'scene.manifest.json', 'build_inputs.json', 'inventory_ledger.json', 'mesh_specs.json.gz'):
        add(scene.parent / name, 'scene/physical/' + name)
    for src in (scene.parent / 'source').glob('*.py'):
        add(src, 'scene/physical/source/' + src.name)
    for asset in assets:
        src = Path(asset).resolve(strict=True)
        add(src, src.relative_to(run).as_posix())
    add(blend, 'authoring/scene.blend')
    for src in authoring_files:
        add(src, 'evidence/authoring/' + src.name)
    for src in sorted(native_files):
        if str(src) in aliases:
            continue
        add(src, 'evidence/' + src.relative_to(run).as_posix())
    demo = read(args.demo_receipt)
    add(demo['video']['path'], 'articulation.mp4')
    for src in sorted(demo_files):
        if str(src) not in aliases:
            add(src, src.relative_to(run).as_posix())
    for name in ('README.md', 'mesh09_verification.md'):
        add(run / 'physics_tests' / name, 'evidence/physics_tests/' + name)
    for name in ('physical_package.py', 'physical_verify.py', 'physical_viewer.py', 'physical_control.py', 'jobctl.py'):
        add(experiment / name, 'tools/' + name)
    for name, dst in [('PHYSICAL_REPORT.md', 'REPORT.md'), ('TOOLS.md', 'TOOLS.md'), ('PHYSICAL_BRIEF.md', 'BRIEF.md')]:
        add(experiment / name, dst)
    for name in ('layout_evidence.json', 'single_sink_fit.json', 'textures/manifest.json', 'outlets/outlets.json'):
        add(run / name, 'provenance/' + name)
    for name in ('selected_scene.json', 'source_preservation.json'):
        add(run / name, 'provenance/' + name)
    for src in (scene.parent / 'comparison').iterdir():
        if src.is_file():
            add(src, 'comparison/' + src.name)
    comparison = read(scene.parent / 'comparison/results.json')
    require(comparison['scene_sha256'] == sha(scene), 'Source comparisons bind a different scene')
    for view in comparison['views']:
        source = bound_file(view['source'], view['source_sha256'])
        add(source, 'evidence/source_frames/' + source.name)
        rendered = bound_file(run.parents[2] / view['render'], view['render_sha256'])
        add(rendered, 'evidence/renders/' + rendered.name)
    for src in scene.parent.glob('render_*.receipt.json'):
        add(src, 'evidence/renders/' + src.name)
    geometry = run / 'lidar' / (scene.parent.name + '_verified')
    for name in ('collision_initial_pose.ply', 'collision_initial_pose.obj', 'collision_export_verified.receipt.json',
                 'receipt.json', 'artifact_readback.json', 'comparison.png', 'glass_included.ply', 'glass_omitted.ply',
                 'scans.npz', 'colliders.json', 'collision_geometry.npz', 'README.md', 'COLLISION_READBACK.md', 'COLLISION_EXPORT.md'):
        add(geometry / name, 'geometry/' + name)
    # Carry each review's declared payload. Identical snapshot bytes can alias
    # an existing packaged file; original receipts remain byte-for-byte intact.
    for review_name in ('export9', 'preflight9'):
        directory = run / 'reviews' / review_name
        for src in directory.iterdir():
            if src.is_file():
                add(src, 'evidence/reviews/' + review_name + '/' + src.name)
        review = read(directory / 'receipt.json')
        for name, expected in review['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe review payload path')
            src = bound_file(directory / name, expected)
            identical = [dst for dst, item in files.items() if item['sha256'] == expected]
            if identical:
                aliases.setdefault(str(src), []).append(identical[0])
            else:
                add(src, 'evidence/reviews/' + review_name + '/' + name)
        input_union = directory / 'input_union.json'
        if input_union.exists():
            for original, expected in read(input_union)['files'].items():
                identical = [dst for dst, item in files.items() if item['sha256'] == expected]
                if identical:
                    aliases.setdefault(str(Path(original).absolute()), []).append(identical[0])
                else:
                    src = bound_file(original, expected)
                    add(src, 'evidence/review_inputs/' + src.relative_to(run.parents[2]).as_posix())
    for name in ('loaded_scene.png', 'loaded_scene.receipt.json'):
        add(run / 'viewer' / scene.parent.name / name, 'evidence/viewer/' + name)

    # Close declared evidence to a fixed point: a newly added receipt may name
    # further payloads. Resolve historical references by path AND digest.
    # Raw capture/external input lists are provenance, not delivery artifacts.
    processed = set()
    closure_reference_count = 0
    while True:
        pending = []
        for original, destinations in list(aliases.items()):
            if Path(original).suffix != '.json':
                continue
            for destination in set(destinations):
                item = files[destination]
                key = (original, item['sha256'])
                if key not in processed:
                    processed.add(key)
                    pending.append((Path(original), item))
        if not pending:
            break
        for original, item in pending:
            record = read(item['source'])
            if not isinstance(record, dict):
                continue
            schema = record.get('schema', '')
            references = []
            declared = record.get('files', {})
            if isinstance(schema, str) and schema.startswith('redteam-') and isinstance(declared, dict):
                references += [(original.parent / name, expected) for name, expected in declared.items()
                               if isinstance(expected, str) and len(expected) == 64]
            artifacts = record.get('artifacts', [])
            if isinstance(artifacts, dict):
                references += [(original.parent / name, expected) for name, expected in artifacts.items()
                               if isinstance(expected, str) and len(expected) == 64]
            elif isinstance(artifacts, list):
                references += [(original.parent / entry['path'], entry['sha256']) for entry in artifacts
                               if isinstance(entry, dict) and 'path' in entry and 'sha256' in entry]
            if schema == 'real2sim-selected-physical-scene/v1':
                references += [(run / entry['path'], entry['sha256']) for entry in record['evidence'].values()]
            for reference, expected in references:
                reference = reference.resolve(strict=False)
                identical = [dst for dst, value in files.items() if value['sha256'] == expected]
                if identical:
                    destinations = aliases.setdefault(str(reference), [])
                    if identical[0] not in destinations:
                        destinations.append(identical[0])
                else:
                    src = bound_file(reference, expected)
                    add(src, 'evidence/closure/' + src.relative_to(run.parents[2]).as_posix())
                closure_reference_count += 1
    packaged = dict(schema='real2sim-portable-physical-package/v1', produced_at=datetime.now(timezone.utc).isoformat(),
        scene='scene/physical/scene.usda', scene_sha256=sha(scene), physics_receipt_sha256=sha(args.physics_receipt),
        helper_sha256=sha(__file__), verdict_helper_sha256=sha(experiment / 'physical_verify.py'),
        scope='Source-informed textured articulated candidate. Metric/photoreal replica and G1/LiDAR transfer unverified.',
        evidence_closure=dict(processed_json_contexts=len(processed), declared_references=closure_reference_count,
                              method='Fixed-point path-and-SHA resolution of selected evidence, declared artifacts and red-team payloads; raw source input lists remain external provenance.'),
        original_path_to_package_paths=aliases,
        files={name: {k: v for k, v in item.items() if k != 'source'} for name, item in sorted(files.items())})
    readme = '''# IMG_2296 physical scene

Open `scene/physical/scene.usda` in Isaac Sim. Keep this directory tree intact:
the unchanged USD resolves its textures by relative paths. It contains the
meshes, rigid bodies, compound colliders and joints in one scene.

`authoring/scene.blend` is the editable packed mesh/material copy. Joint markers
are metadata in Blender; native USD/Isaac remains the physics authority.

Read REPORT.md for verification scope, known omissions and modeled parameters.
Open comparison/index.html for four matching-camera video/render comparisons.
Play articulation.mp4 for the 14-second native physics demonstration. It uses
an explicit RTX realtime viewing override; the still comparisons use path tracing.
The geometry folder contains initial-pose collision exports and an ideal scan,
not an independently measured localization map or a real sensor model.

Native evidence includes the original failed sink-probe receipt and subsequent
bounded contact checks. The combined receipt describes which checks establish
the modeled mechanical result. Original receipts are never rewritten; use
package.json's original_path_to_package_paths to resolve their historical paths
inside this bundle. Captures are discrete phase samples unless labeled otherwise.

Absolute scale, hidden mechanisms, photorealistic replication and G1 task
transfer remain unverified. See the report for visible source discrepancies.
The raw MOV and full reconstruction dataset are external source inputs and
are not included; the selected four reference images and their hashes are included.
The tools directory preserves implementation code; viewer/control host paths
require a local Isaac installation and are not a bundled simulator runtime.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, item in sorted(files.items()):
            bound_file(item['source'], item['sha256'])
            z.write(item['source'], 'IMG_2296_physical/' + name)
        z.writestr('IMG_2296_physical/package.json', json.dumps(packaged, indent=2) + '\n')
        z.writestr('IMG_2296_physical/README.md', readme)
    with zipfile.ZipFile(output) as z:
        z.extractall(extracted)
    root = extracted / 'IMG_2296_physical'
    for name, entry in packaged['files'].items():
        bound_file(root / name, entry['sha256'])
    relocated = root / packaged['scene']
    require(Usd.Stage.Open(str(relocated)) and sha(relocated) == sha(scene), 'Relocated USD did not reopen unchanged')
    new_layers, new_assets, new_unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(relocated)))
    require(not new_unresolved and len(new_layers) == 1 and len(new_assets) == len(assets), 'Relocated dependencies differ')
    for asset in new_assets:
        require(Path(asset).resolve(strict=True).is_relative_to(root), 'External relocated dependency')
    receipt = dict(schema='real2sim-package-readback/v1', produced_at=datetime.now(timezone.utc).isoformat(),
        package=str(output), package_sha256=sha(output), scene_sha256=sha(scene),
        physics_receipt_sha256=sha(args.physics_receipt), helper_sha256=sha(__file__),
        manifest_sha256=sha(root / 'package.json'), all_file_hashes_match=True,
        relocated_stage_opened=True, external_unresolved_assets=new_unresolved, texture_count=len(new_assets),
        extracted_scene=str(relocated), file_count=len(files), status='passed')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
