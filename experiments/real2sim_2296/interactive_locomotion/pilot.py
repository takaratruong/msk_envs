#!/usr/bin/env python3
"""Run/status/logs/stop/restart surface for the bounded room transition pilot.

Every execution gets a fresh output and provider audit directory. Restarting
preserves all previous evidence. This launches an offline physics experiment,
not a hardware controller or an interactive game.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run', 'status', 'logs', 'stop', 'restart', '_worker'])
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--name', default='room_transition_pilot')
    a = p.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,40}', a.name):
        raise ValueError('Pilot name must use lowercase letters, digits and underscores')
    project = a.project_root.resolve(strict=True)
    run = a.run_root.resolve(strict=True)
    source = project / 'experiments/real2sim_2296/interactive_locomotion'
    jobctl = project / 'experiments/real2sim_2296/jobctl.py'
    python = project / 'runs/real2sim-2296/20260912T2237Z-g1-motion/venv/bin/python'
    if a.action == '_worker':
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output_root = run / 'pilot' / a.name / stamp
        output_root.mkdir(parents=True, exist_ok=False)
        implementation = output_root / 'implementation'
        implementation.mkdir()
        for name in ['benchmark.py', 'transition_reference.py']:
            shutil.copyfile(source / name, implementation / name)
        config = json.loads((run / 'transition/config_room01.json').read_text())
        config['provider_audit'] = str(Path(config['provider_root']) / ('pilot_' + a.name + '_' + stamp))
        config_path = output_root / 'config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        pointer = output_root.parent / 'latest.json'
        temp = pointer.with_suffix('.tmp')
        temp.write_text(json.dumps({'run_directory': str(output_root), 'created_at': stamp,
                                   'state': 'started', 'result_status': 'unchecked'}) + '\n')
        temp.replace(pointer)
        command = [str(python), str(implementation / 'benchmark.py'), '--controller', 'scalebfm',
                   '--tracker', str(project / 'runs/real2sim-2296/20260913T0013Z-g1-pickplace/tracker'),
                   '--robot', config['robot_model'],
                   '--scene', str(project / 'runs/real2sim-2296/20260913T0013Z-g1-pickplace/scene08/scene.xml'),
                   '--floor-geom', 'room_3499',
                   '--reference', str(project / 'runs/real2sim-2296/20260912T2237Z-g1-motion/iterations/take06/trajectory.npz'),
                   '--duration', '16', '--reference-provider', str(implementation / 'transition_reference.py'),
                   '--provider-config', str(config_path), '--output', str(output_root / 'physics')]
        result = subprocess.run(command, cwd=project)
        pointer.write_text(json.dumps({'run_directory': str(output_root), 'created_at': stamp,
                                      'state': 'finished', 'returncode': result.returncode,
                                      'result_status': 'reported; inspect physics/receipt.json'}) + '\n')
        raise SystemExit(result.returncode)
    command = [sys.executable, str(jobctl), '--root', str(run)]
    if a.action == 'run':
        command += ['start', a.name, '--gpu', '', '--', 'timeout', '--signal=TERM', '--kill-after=5s', '240s',
                    sys.executable, str(source / 'pilot.py'), '_worker', '--project-root', str(project),
                    '--run-root', str(run), '--name', a.name]
    else:
        command += [a.action, a.name]
    result = subprocess.run(command, cwd=project)
    pointer = run / 'pilot' / a.name / 'latest.json'
    if a.action == 'status' and pointer.exists():
        print(pointer.read_text().strip())
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
