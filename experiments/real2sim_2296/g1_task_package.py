#!/usr/bin/env python3
"""Verify and transfer the frozen G1 fridge-to-table package (stdlib only).

The source package gains one deterministic package.manifest.json.  A fresh
Downloads child and ZIP contain exactly its declared payload plus the manifest.
Existing destinations are never overwritten.  The external output directory
holds the transfer receipt and verifier logs; no simulation is run here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile


SCHEMA = 'g1-fridge-to-table-transfer/v1'
VERIFIER_SHA256 = '81a6f13dd8e265d1e3cf698f4413cc2c4f015fe6fd755fffea296eefa864baff'
MANIFEST = 'package.manifest.json'
PHYS = 'components/physics_stage05'
CASE = PHYS + '/cases/fridge_to_table'
RAW = CASE + '/raw'
MOVIE = 'media/G1_Dex3_fridge_to_table_physics.mp4'
MEDIA_RECEIPT = 'evidence/media/delivery.receipt.json'
PRIMARY = frozenset({
    'README.md', 'index.html', 'articulated_room.usda', 'PROCESS_AND_RESULTS.md',
    'components/reference_stage03/manifest.json', PHYS + '/manifest.json',
    'recorded_physics/usd/scene.usda', MOVIE, 'evidence/index.json',
})
GATES = frozenset({
    'immutable_declared_inputs', 'complete_reference_executed',
    'complete_task_substep_contacts', 'unassisted_initial_task',
    'torque_only_freebase', 'source_motor_limits', 'no_applied_external_force',
    'upright', 'physical_door_opened', 'opposing_fingers_acquired',
    'physically_lifted', 'carried_with_hand_contact', 'reached_target_table',
    'released_and_supported_two_seconds', 'final_object_stable',
    'no_major_unintended_robot_contacts',
})
EXCLUSIONS = [
    'package.manifest.json (shipped separately, not self-hashed)',
    '**/__pycache__/**', '**/reproduced/**', '**/*.mov (case insensitive)',
    'virtual/Conda environments (named directories or environment markers)',
]
ENV_NAMES = frozenset({'venv', '.venv', 'env', '.env', 'envs', 'environments',
                       'miniconda3', 'anaconda3', 'conda-meta'})
STORED_SUFFIXES = frozenset({'.npz', '.png', '.jpg', '.jpeg', '.mp4', '.zip',
                            '.gz', '.bz2', '.xz', '.7z', '.webp', '.usdz'})
BUFFER = 8 * 1024 * 1024
UNSET = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BUFFER), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError('Non-finite JSON number: ' + value)))


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def save(path, value):
    Path(path).write_bytes(json_bytes(value))


def clean_name(name):
    require(isinstance(name, str) and bool(name), 'Empty or non-string path')
    require(not any(ord(c) < 32 for c in name) and '\\' not in name and ':' not in name,
            'Unsafe path: ' + repr(name))
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and '..' not in pure.parts and name != '.'
            and str(pure) == name, 'Path must be clean and relative: ' + name)
    return name


def safe_file(root, name):
    clean_name(name)
    path = root / name
    require(path.resolve() == path and path.is_file(),
            'Missing, symlinked or non-file package input: ' + name)
    require(stat.S_ISREG(path.lstat().st_mode), 'Not a regular file: ' + name)
    return path


def excluded(name):
    parts = PurePosixPath(name).parts
    return (name == MANIFEST or '__pycache__' in parts or 'reproduced' in parts
            or any(p.lower() in ENV_NAMES or p.lower().endswith('-venv') for p in parts)
            or PurePosixPath(name).suffix.lower() == '.mov')


def inventory(root, apply_exclusions=True):
    names, skipped = [], []
    for current, dirs, files in os.walk(root, followlinks=False):
        current = Path(current)
        for name in sorted(dirs[:]):
            path = current / name
            relative = path.relative_to(root).as_posix()
            marker = (path / 'pyvenv.cfg').is_file() or (path / 'conda-meta').is_dir()
            if apply_exclusions and (excluded(relative) or marker):
                dirs.remove(name)
                skipped.append(relative + '/')
            else:
                clean_name(relative)
                require(not path.is_symlink(), 'Symlinked directory: ' + relative)
        for name in sorted(files):
            relative = (current / name).relative_to(root).as_posix()
            if apply_exclusions and excluded(relative):
                skipped.append(relative)
                continue
            safe_file(root, relative)
            names.append(relative)
    return sorted(names), sorted(skipped)


def validate_rows(rows):
    require(isinstance(rows, list) and bool(rows), 'Manifest rows must be nonempty')
    names = []
    for row in rows:
        require(isinstance(row, dict), 'Manifest row must be an object')
        names.append(clean_name(row.get('path')))
        require(type(row.get('size')) is int and row['size'] >= 0, 'Invalid file size')
        require(isinstance(row.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid SHA256')
    require(len(names) == len(set(names)), 'Duplicate manifest paths')
    return set(names)


class Package:
    def __init__(self, root):
        self.root = root
        self.bound = {}

    def bind(self, name, expected=UNSET):
        path = safe_file(self.root, name)
        digest = sha(path)
        if expected is not UNSET:
            require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
                    'Missing or invalid bound SHA256: ' + name)
            require(digest == expected, 'SHA256 mismatch: ' + name)
        self.bound[name] = digest
        return path

    def document(self, name, expected=UNSET):
        return load(self.bind(name, expected))

    def pointer(self, row, expected_path=None):
        require(isinstance(row, dict), 'Missing evidence pointer')
        name = row.get('path')
        if expected_path is not None:
            require(name == expected_path, 'Wrong evidence pointer: ' + str(name))
        require(isinstance(row.get('sha256'), str), 'Missing evidence SHA256')
        return self.bind(name, row['sha256'])

    def component_rows(self, prefix, document):
        rows = document.get('files')
        validate_rows(rows)
        for row in rows:
            path = self.bind(prefix + '/' + row['path'], row['sha256'])
            require(path.stat().st_size == row['size'], 'Component file size differs')


def check_gates(document):
    gates = document.get('gates')
    require(isinstance(gates, dict) and set(gates) == GATES
            and all(value is True for value in gates.values()),
            'All 16 named physical gates must be present and true')


def preflight(root):
    """Read-only joins; rejects unfinished media before creating a manifest."""
    p = Package(root)
    for name in sorted(PRIMARY):
        require(p.bind(name).stat().st_size > 0, 'Empty primary input: ' + name)
    p.bind('verify_package.py', VERIFIER_SHA256)
    physics = p.document(PHYS + '/manifest.json')
    require(physics.get('schema') == 'g1-physical-stage-manifest/v2'
            and physics.get('physical_pickplace_success') is True,
            'physics_stage05 must declare physical success')
    p.component_rows(PHYS, physics)
    reference = physics.get('reference_component', {})
    require(reference.get('path') == '../reference_stage03', 'Wrong reference component')
    p.bind('components/reference_stage03/manifest.json', reference.get('manifest_sha256'))
    case = p.document(CASE + '/case.json')
    require(case.get('schema') == 'g1-physical-staged-case/v2'
            and case.get('physical_pickplace_success') is True and case.get('raw') == 'cases/fridge_to_table/raw',
            'Wrong or unsuccessful selected physical case')
    selected = case.get('physical_evaluation', {})
    require(selected.get('path') == 'cases/fridge_to_table/evaluation/receipt.json'
            and selected.get('complete_physical_task_passed') is True,
            'Missing selected physical evaluation')
    evaluation_name = PHYS + '/' + selected['path']
    evaluation = p.document(evaluation_name, selected.get('sha256'))
    require(evaluation.get('schema') == 'g1-pickplace-physical-task-evaluation/v1'
            and evaluation.get('complete_physical_task_passed') is True
            and evaluation.get('input_mismatches') == [], 'Physical evaluation did not pass')
    check_gates(evaluation)
    aliases = selected.get('input_aliases', {})
    inputs = evaluation.get('inputs', {})
    require(isinstance(inputs, dict) and bool(inputs), 'Unbound physical evaluation')
    bound_evaluation_names = set()
    for original, digest in inputs.items():
        alias = aliases.get(original)
        require(isinstance(alias, str) and not Path(alias).is_absolute()
                and '\\' not in alias, 'Missing or absolute evaluation alias: ' + original)
        path = (root / PHYS / alias).resolve()
        require(path.is_relative_to(root), 'Evaluation alias escapes package')
        name = path.relative_to(root).as_posix()
        p.bind(name, digest)
        bound_evaluation_names.add(name)
    require({RAW + '/receipt.json', RAW + '/inputs.json', RAW + '/rollout.npz',
             RAW + '/task_contacts_substeps.jsonl', RAW + '/contacts.jsonl'} <= bound_evaluation_names,
            'Evaluation does not bind all raw state/contact inputs')
    for name, digest in evaluation.get('artifacts', {}).items():
        p.bind(CASE + '/evaluation/' + clean_name(name), digest)
    raw = p.document(RAW + '/receipt.json', case.get('historical_receipt_sha256'))
    p.bind(RAW + '/inputs.json', case.get('historical_inputs_sha256'))
    require(raw.get('completed_requested_duration') is True and raw.get('fall') is None
            and raw.get('exception') is None and type(raw.get('samples')) is int
            and raw['samples'] > 0 and raw['samples'] == raw.get('requested_samples'),
            'Raw run is incomplete or failed')
    require(raw.get('inputs_receipt', {}).get('sha256') == p.bound[RAW + '/inputs.json'],
            'Raw receipt does not bind input receipt')
    for name, digest in raw.get('artifacts', {}).items():
        p.bind(RAW + '/' + clean_name(name), digest)
    require(raw.get('artifacts', {}).get('rollout.npz') == p.bound[RAW + '/rollout.npz'],
            'Raw receipt does not bind recorded states')
    triple = {'rollout.npz': 'rollout.npz', 'rollout_inputs.json': 'inputs.json',
              'rollout_receipt.json': 'receipt.json'}
    for replay_name, raw_name in triple.items():
        p.bind('recorded_physics/inputs/' + replay_name, p.bound[RAW + '/' + raw_name])
    media = p.document(MEDIA_RECEIPT)
    require(media.get('schema') == 'agentic-evidence/v1' and media.get('status') == 'passed'
            and media.get('diagnostic_reset') is False, 'Final media receipt is missing or unpassed')
    require(media.get('raw_rollout_sha256') == p.bound[RAW + '/rollout.npz']
            and media.get('sha256') == p.bound[MOVIE], 'Media binds a different movie or rollout')
    require(media.get('all_exported_samples_checked') == raw['samples']
            and type(media.get('all_rendered_frames_checked')) is int
            and media['all_rendered_frames_checked'] > 0, 'Incomplete media sample binding')
    index = p.document('evidence/index.json')
    require(index.get('schema') == 'g1-final-task-evidence-index/v1'
            and index.get('physical_task_passed') is True and index.get('media_status') == 'ready',
            'Final evidence index is not ready')
    check_gates(index)
    p.pointer(index.get('physical_evaluation'), evaluation_name)
    p.pointer(index.get('raw_receipt'), RAW + '/receipt.json')
    p.pointer(index.get('raw_states'), RAW + '/rollout.npz')
    p.pointer(index.get('media'), MOVIE)
    p.pointer(index['media'].get('receipt'), MEDIA_RECEIPT)
    # Bind all additional path/hash pointers displayed by the evidence index.
    def pointers(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                p.pointer(value)
            for child in value.values():
                pointers(child)
        elif isinstance(value, list):
            for child in value:
                pointers(child)
    pointers(index)
    return {'all_16_physical_gates_bound': True, 'recorded_state_triple_matches': True,
            'media_matches_selected_physics': True, 'media_index_ready': True,
            'raw_samples': raw['samples'], 'rendered_frames': media['all_rendered_frames_checked'],
            'required_inputs_sha256': dict(sorted(p.bound.items()))}


def verify_folder(root, rows):
    wanted = validate_rows(rows)
    actual, _ = inventory(root, apply_exclusions=False)
    require(set(actual) == wanted, 'Copied payload contains missing or extra files')
    for row in rows:
        path = safe_file(root, row['path'])
        require(path.stat().st_size == row['size'] and sha(path) == row['sha256'],
                'Copy readback differs: ' + row['path'])
    return {'passed': True, 'files_checked': len(rows),
            'bytes_checked': sum(row['size'] for row in rows), 'extra_files': []}


def verify_zip(path, folder, rows):
    clean_name(folder)
    require('/' not in folder, 'ZIP root must be one folder')
    validate_rows(rows)
    expected = {folder + '/' + row['path']: row for row in rows}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) and set(names) == set(expected),
                'ZIP has missing, extra or duplicate entries')
        for entry in entries:
            clean_name(entry.filename)
            require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
                    and not (entry.flag_bits & 1), 'Invalid ZIP member type')
            row = expected[entry.filename]
            digest, size = hashlib.sha256(), 0
            with archive.open(entry) as stream:
                for block in iter(lambda: stream.read(BUFFER), b''):
                    size += len(block)
                    digest.update(block)
            require(size == entry.file_size == row['size'] and digest.hexdigest() == row['sha256'],
                    'ZIP readback differs: ' + entry.filename)
    return {'passed': True, 'entries_checked': len(rows),
            'uncompressed_bytes_checked': sum(row['size'] for row in rows),
            'duplicate_entries': [], 'unsafe_entries': []}


def run_verifier(package, output, label):
    require(sha(package / 'verify_package.py') == VERIFIER_SHA256, 'Package verifier changed')
    result = subprocess.run([sys.executable, '-I', '-B', str(package / 'verify_package.py')],
                            cwd=package, capture_output=True, text=True, timeout=300)
    (output / (label + '.stdout.json')).write_text(result.stdout)
    (output / (label + '.stderr.txt')).write_text(result.stderr)
    require(result.returncode == 0, 'Package verifier failed: ' + result.stderr[-1000:])
    document = json.loads(result.stdout)
    require(document.get('passed') is True, 'Package verifier did not pass')
    return document


def transfer(package, downloads, output):
    package, downloads, output = map(lambda path: Path(path).absolute(), (package, downloads, output))
    require(package.resolve() == package and package.is_dir(), 'Source package must be a real directory')
    require(downloads.resolve() == downloads and downloads.is_dir(), 'Downloads must be an existing real directory')
    require(output.resolve() == output and not output.exists(), 'Output must be a fresh real path')
    require(not output.is_relative_to(package) and not output.is_relative_to(downloads),
            'External receipt output must be outside source package and Downloads')
    require(not package.is_relative_to(downloads) and not downloads.is_relative_to(package),
            'Source package and Downloads must be disjoint')
    clean_name(package.name)
    target, zip_path = downloads / package.name, downloads / (package.name + '.zip')
    require(not target.exists() and not target.is_symlink() and not zip_path.exists()
            and not zip_path.is_symlink(), 'Destination folder and ZIP must be fresh; preserve existing deliveries')
    output.mkdir(parents=True)
    try:
        bindings = preflight(package)
        names, skipped = inventory(package)
        require(PRIMARY <= set(names), 'Required primary files excluded from payload')
        missing_bound = sorted(set(bindings['required_inputs_sha256']) - set(names))
        require(not missing_bound,
                'Bound preflight inputs excluded from payload: ' + ', '.join(missing_bound))
        rows = [{'path': name, 'size': (package / name).stat().st_size, 'sha256': sha(package / name)}
                for name in names]
        validate_rows(rows)
        manifest = {'schema': SCHEMA, 'files': rows, 'file_count': len(rows),
                    'bytes': sum(row['size'] for row in rows), 'excluded': EXCLUSIONS,
                    'scope': 'Transfer integrity. Physical acceptance remains in its bound evaluation.'}
        manifest_bytes = json_bytes(manifest)
        manifest_path = package / MANIFEST
        if manifest_path.exists() or manifest_path.is_symlink():
            require(safe_file(package, MANIFEST).read_bytes() == manifest_bytes,
                    'Preserve existing different package manifest')
        else:
            with manifest_path.open('xb') as stream:
                stream.write(manifest_bytes)
        (output / MANIFEST).write_bytes(manifest_bytes)
        save(output / 'preflight.json', bindings)
        save(output / 'exclusions.json', {'excluded_paths': skipped, 'policy': EXCLUSIONS})
        source_check = run_verifier(package, output, 'source_verifier')
        shipped = rows + [{'path': MANIFEST, 'size': len(manifest_bytes),
                           'sha256': hashlib.sha256(manifest_bytes).hexdigest()}]
        target.mkdir()
        for row in shipped:
            source = safe_file(package, row['path'])
            destination = target / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst, BUFFER)
        copied = verify_folder(target, shipped)
        copy_check = run_verifier(target, output, 'copy_verifier')
        with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3,
                             allowZip64=True) as archive:
            for row in shipped:
                info = zipfile.ZipInfo(package.name + '/' + row['path'], (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = (zipfile.ZIP_STORED if Path(row['path']).suffix.lower()
                                      in STORED_SUFFIXES else zipfile.ZIP_DEFLATED)
                info._compresslevel = 3
                with (target / row['path']).open('rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, BUFFER)
        zipped = verify_zip(zip_path, package.name, shipped)
        # Reject source edits during the copy/archive operation, including new payload files.
        final_names, _ = inventory(package)
        require(final_names == names, 'Source payload inventory changed during transfer')
        final_check = run_verifier(package, output, 'final_source_verifier')
        require(sha(manifest_path) == shipped[-1]['sha256'], 'Source manifest changed during transfer')
        artifacts = {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
        receipt = {'schema': 'g1-fridge-to-table-transfer-receipt/v1',
                   'produced_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                   'claim': 'Declared package copied and ZIP read back completely; bound nominal simulation evidence retained.',
                   'helper': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                   'source_package': str(package), 'downloads_folder': str(target),
                   'manifest_sha256': shipped[-1]['sha256'], 'file_count': manifest['file_count'],
                   'bytes': manifest['bytes'], 'shipped_file_count': len(shipped),
                   'shipped_bytes': sum(row['size'] for row in shipped),
                   'zip': {'path': str(zip_path), 'size': zip_path.stat().st_size, 'sha256': sha(zip_path),
                           'compression': 'DEFLATED level 3; STORED for compressed media/arrays/archives'},
                   'preflight': bindings, 'copy_readback': copied, 'zip_readback': zipped,
                   'source_verifier': source_check, 'copied_verifier': copy_check,
                   'final_source_verifier': final_check, 'artifacts': artifacts,
                   'limitations': 'No new physics or hardware/room accuracy claim. Python/runtime environments and original MOV are not transferred.'}
        save(output / 'receipt.json', receipt)
        return receipt
    except Exception as error:
        save(output / 'failure.json', {'status': 'failed', 'error': type(error).__name__ + ': ' + str(error),
             'source_package': str(package), 'downloads_folder': str(target), 'zip': str(zip_path),
             'scope': 'Any partial new output is retained for diagnosis; no success receipt was issued.'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = transfer(args.package, args.downloads, args.output)
    print(json.dumps({'status': result['status'], 'receipt': str(args.output / 'receipt.json'),
                      'file_count': result['file_count'], 'shipped_file_count': result['shipped_file_count'],
                      'zip': result['zip']}, indent=2))


if __name__ == '__main__':
    main()
