#!/usr/bin/env python3
"""Owned live simulator run/status/logs/stop/restart with source snapshots."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def save(path, value):
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'status', 'logs', 'stop', 'restart', '_worker'])
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--name', default='live_room')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--duration', type=float, default=0.)
    p.add_argument('--initial-instance')
    p.add_argument('--auto-interact', action='store_true')
    p.add_argument('--port', type=int, default=8782)
    a = p.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,50}', a.name):
        p.error('Invalid owned process name')
    root, run = a.project_root.resolve(strict=True), a.run_root.resolve(strict=True)
    e = root/'experiments/real2sim_2296'
    python = root/'runs/real2sim-2296/20260912T2237Z-g1-motion/venv/bin/python'
    if a.action == '_worker':
        config_path = (a.config or run/'configs/live_natural24.json').resolve(strict=True)
        config = json.loads(config_path.read_text())
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        trial = run/'trials'/a.name/stamp
        trial.mkdir(parents=True, exist_ok=False)
        sources, copied = {}, {}
        def snapshot(path):
            source = Path(path).resolve(strict=True)
            if source in copied:
                return str(copied[source])
            data = source.read_bytes()
            target = trial/'implementation'/str(len(copied))/source.name
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            copied[source] = target
            sources[str(source)] = {'snapshot': str(target), 'sha256': hashlib.sha256(data).hexdigest()}
            return str(target)
        def rewrite(x):
            if isinstance(x, dict):
                return {k: rewrite(v) for k, v in x.items()}
            if isinstance(x, list):
                return [rewrite(v) for v in x]
            if isinstance(x, str) and x.endswith('.py'):
                return snapshot(x)
            return x
        executable = snapshot(e/'interactive_locomotion/live_runtime.py')
        config = rewrite(config)
        if 'transport_adapter' in config:
            ui_source = Path(config.get('transport_ui_dir', e/'interactive_locomotion/live_ui')).resolve(strict=True)
            ui_target = trial/'ui'
            ui_target.mkdir()
            for name in ('index.html', 'app.js', 'style.css'):
                source = ui_source/name
                data = source.read_bytes()
                target = ui_target/name
                target.write_bytes(data)
                sources[str(source)] = {'snapshot': str(target), 'sha256': hashlib.sha256(data).hexdigest()}
            config['transport_ui_dir'] = str(ui_target)
        (trial/'source_config.json').write_bytes(config_path.read_bytes())
        save(trial/'config.json', config)
        for source, item in sources.items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Source changed during snapshot')
        command = [str(python), executable, '--config', str(trial/'config.json'), '--output', str(trial/'physics'),
                   '--duration', str(a.duration)]
        if not a.headless:
            command += ['--realtime', '--serve', '--render', '--port', str(a.port)]
        if a.initial_instance:
            command += ['--initial-instance', a.initial_instance]
        if a.auto_interact:
            command += ['--auto-interact']
        save(trial/'launch.json', {'command': command, 'sources': sources, 'renderer_gpu': 3,
            'inference': 'CPU with one intra-op/OpenBLAS/OMP thread', 'mode': 'offline test' if a.headless else 'live paced browser control'})
        pointer = trial.parent/'latest.json'
        save(pointer, {'run_directory': str(trial), 'state': 'running', 'result': 'unchecked'})
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='3', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run(command, cwd=root, env=environment)
        save(pointer, {'run_directory': str(trial), 'state': 'finished', 'returncode': result.returncode,
                       'result': 'reported; inspect physics receipt and independent verification'})
        raise SystemExit(result.returncode)
    command = [sys.executable, str(e/'jobctl.py'), '--root', str(run)]
    if a.action == 'run':
        command += ['start', a.name, '--gpu', '', '--', sys.executable, str(e/'interactive_locomotion/live.py'),
            '_worker', '--project-root', str(root), '--run-root', str(run), '--name', a.name,
            '--config', str((a.config or run/'configs/live_natural24.json').resolve(strict=True)),
            '--duration', str(a.duration), '--port', str(a.port)]
        if a.headless:
            command += ['--headless']
        if a.initial_instance:
            command += ['--initial-instance', a.initial_instance]
        if a.auto_interact:
            command += ['--auto-interact']
    else:
        command += [a.action, a.name]
    result = subprocess.run(command, cwd=root)
    if a.action == 'status':
        pointer = run/'trials'/a.name/'latest.json'
        if pointer.exists():
            print(pointer.read_text())
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
