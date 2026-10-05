#!/usr/bin/env python3
"""Owned run/status/logs/stop/restart controls for the R8 hybrid experiment.

Each worker snapshots the runtime, every explicit *.py configuration path, and
both configurations before launching. Previous trials are never overwritten.
The runtime owns physics and acceptance; a zero process exit is not a skill pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def write_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def snapshot_plan(project, config_path, output, name, stamp):
    """Read-only plan. Files contain exact bytes; no code is imported or run."""
    project, config_path, output = Path(project).resolve(), Path(config_path).resolve(strict=True), Path(output).resolve()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', name) or not re.fullmatch(r'[0-9TZ]+', stamp):
        raise ValueError('Invalid trial name or timestamp')
    original = config_path.read_bytes()
    config = json.loads(original)
    if not isinstance(config, dict):
        raise ValueError('Hybrid configuration must be an object')
    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else project / path).resolve(strict=True)
    transition_source = resolve(config['transition_config'])
    original_transition = transition_source.read_bytes()
    transition = json.loads(original_transition)
    if not isinstance(transition, dict) or str(transition.get('provider_gpu')) != '4':
        raise ValueError('This owned MotionBricks lane requires explicit provider_gpu=4')
    provider_root = resolve(transition['provider_root'])
    if not provider_root.is_dir():
        raise ValueError('MotionBricks provider_root must be an existing directory')
    provider_audit = provider_root / f'hybrid_{name}_{stamp}'
    if provider_audit.exists():
        raise FileExistsError('Fresh provider audit directory already exists')
    transition['provider_audit'] = str(provider_audit)
    files, sources, copied = {}, {}, {}
    def snapshot(source):
        source = resolve(source)
        if source in copied:
            return copied[source]
        if not source.is_file() or source.suffix != '.py':
            raise ValueError(f'Expected a local Python file: {source}')
        destination = output / 'implementation' / f'{len(copied):03d}' / source.name
        data = source.read_bytes()
        files[destination] = data
        sources[str(source)] = {'sha256': sha(data), 'snapshot': str(destination)}
        copied[source] = str(destination)
        return str(destination)
    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value.endswith('.py'):
            return snapshot(value)
        return value
    runtime = snapshot(project / 'experiments/real2sim_2296/interactive_locomotion/hybrid_runtime.py')
    launcher = snapshot(Path(__file__).resolve())
    config = rewrite(config)
    transition = rewrite(transition)
    config['transition_config'] = str(output / 'transition_config.json')
    files[output / 'source_config.json'] = original
    files[output / 'source_transition_config.json'] = original_transition
    files[output / 'config.json'] = encoded(config)
    files[output / 'transition_config.json'] = encoded(transition)
    sources[str(config_path)] = {'sha256': sha(original), 'snapshot': str(output / 'source_config.json')}
    sources[str(transition_source)] = {'sha256': sha(original_transition), 'snapshot': str(output / 'source_transition_config.json')}
    # Reject a concurrent source edit instead of silently mixing live versions.
    for source, binding in sources.items():
        if sha(Path(source).read_bytes()) != binding['sha256']:
            raise ValueError(f'Snapshot source changed while reading: {source}')
    return {'files': files, 'sources': sources, 'runtime': runtime, 'launcher': launcher,
            'config': config, 'transition_config': transition, 'provider_audit': str(provider_audit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status', 'logs', 'stop', 'restart', '_worker'])
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--name', default='fridge_open')
    parser.add_argument('--config', type=Path, help='Default: RUN_ROOT/configs/hybrid04.json; restart override selects a new source configuration')
    parser.add_argument('--grace', type=float, default=30., help='Owned stop grace in seconds')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', args.name):
        parser.error('Name must use lowercase letters, digits and underscores')
    if args.config is not None and args.action not in {'run', 'restart', '_worker'}:
        parser.error('--config is only meaningful for run/restart')
    project, run = args.project_root.resolve(strict=True), args.run_root.resolve(strict=True)
    source = project / 'experiments/real2sim_2296/interactive_locomotion'
    jobctl = project / 'experiments/real2sim_2296/jobctl.py'
    python = project / 'runs/real2sim-2296/20260912T2237Z-g1-motion/venv/bin/python'
    pointer = run / 'trials' / args.name / 'latest.json'
    if args.action == '_worker':
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = pointer.parent / stamp
        output.mkdir(parents=True, exist_ok=False)
        status = {'run_directory': str(output), 'created_at': stamp, 'state': 'snapshotting',
                  'result_status': 'unchecked; process status is not physical success'}
        write_json(pointer, status)
        try:
            config_path = (args.config or run / 'configs/hybrid04.json').resolve(strict=True)
            plan = snapshot_plan(project, config_path, output, args.name, stamp)
            for destination, data in plan['files'].items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(data)
                if sha(destination.read_bytes()) != sha(data):
                    raise ValueError('Snapshot readback mismatch')
            command = [str(python), plan['runtime'], '--config', str(output / 'config.json'),
                       '--output', str(output / 'physics')]
            launch = {'schema': 'g1-hybrid-launch/v1', 'command': command, 'cwd': str(project),
                'sources': plan['sources'], 'snapshots_sha256': {str(path): sha(data) for path, data in plan['files'].items()},
                'provider_audit': plan['provider_audit'], 'runtime_cuda_visible_devices': '',
                'provider_gpu': 4, 'outer_watchdog_seconds': 240,
                'scope': 'Byte-preserving source/config snapshot and owned offline launch; physical result is evaluated separately.'}
            write_json(output / 'launch.json', launch)
            status.update(state='running', launch_receipt=str(output / 'launch.json'),
                          launch_receipt_sha256=sha((output / 'launch.json').read_bytes()))
            write_json(pointer, status)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(command, cwd=project, env=environment)
            status.update(state='finished', returncode=result.returncode,
                          result_status='reported; inspect physics/receipt.json and independent evaluation')
            write_json(pointer, status)
            raise SystemExit(result.returncode)
        except Exception as error:
            status.update(state='failed', error=repr(error))
            write_json(pointer, status)
            raise
    command = [sys.executable, str(jobctl), '--root', str(run)]
    if args.action == 'run' or (args.action == 'restart' and args.config is not None):
        config_path = (args.config or run / 'configs/hybrid04.json').resolve(strict=True)
        if not python.is_file():
            raise FileNotFoundError(python)
        command += ['start', args.name, '--gpu', '', '--', 'timeout', '--signal=TERM', '--kill-after=5s', '240s',
                    sys.executable, str(source / 'hybrid.py'), '_worker', '--project-root', str(project),
                    '--run-root', str(run), '--name', args.name, '--config', str(config_path)]
    else:
        command += [args.action, args.name]
        if args.action == 'stop':
            command += ['--grace', str(args.grace)]
    result = subprocess.run(command, cwd=project)
    if args.action == 'status' and pointer.exists():
        print(pointer.read_text().strip())
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
