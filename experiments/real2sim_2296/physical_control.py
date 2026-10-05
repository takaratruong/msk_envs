#!/usr/bin/env python3
"""Open or inspect the selected physical scene through an owned Isaac GUI job."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys

PROJECT=Path(__file__).resolve().parents[2]
RUN=PROJECT/'runs/real2sim-2296/20260912T1002Z-physical'
ISAAC=Path('/home/ubuntu/projects/simtoolreal/repo/.venv_isaacsim/bin/python')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['open','status','logs','stop','restart'])
    parser.add_argument('--run-root',type=Path,default=RUN)
    parser.add_argument('--scene',type=Path,help='Explicit candidate for open; otherwise use selected_scene.json')
    args=parser.parse_args();run=args.run_root.resolve()
    command=[sys.executable,str(PROJECT/'experiments/real2sim_2296/jobctl.py'),
             '--root',str(run),'start' if args.action=='open' else args.action,'physical_viewer']
    if args.action=='open':
        if args.scene:
            scene=args.scene.resolve(strict=True)
            expected=hashlib.sha256(scene.read_bytes()).hexdigest()
            manifest_path=scene.with_suffix('.manifest.json')
            manifest_expected=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        else:
            selected=json.loads((run/'selected_scene.json').read_text())
            scene=(run/selected['scene']['path']).resolve(strict=True)
            expected=selected['scene']['sha256']
            if hashlib.sha256(scene.read_bytes()).hexdigest()!=expected:
                raise ValueError('Selected scene no longer matches its recorded hash')
            manifest_path=(run/selected['manifest']['path']).resolve(strict=True)
            if manifest_path!=scene.with_suffix('.manifest.json'):
                raise ValueError('Selected manifest must be adjacent to its scene')
            manifest_expected=selected['manifest']['sha256']
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=manifest_expected:
                raise ValueError('Selected manifest no longer matches its recorded hash')
        manifest=json.loads(manifest_path.read_text())
        if manifest['scene_sha256']!=expected:raise ValueError('Scene manifest does not match')
        if not ISAAC.is_file():raise FileNotFoundError('Set ISAAC to a local Isaac Sim Python installation')
        output=run/'viewer'/scene.parent.name
        command+=['--gpu','0','--','env','-u','CUDA_VISIBLE_DEVICES',
                  'DISPLAY='+os.environ.get('DISPLAY',':1'),'OMNI_KIT_ACCEPT_EULA=YES',
                  'CUDA_DEVICE_ORDER=PCI_BUS_ID',str(ISAAC),'-u',
                  str(PROJECT/'experiments/real2sim_2296/physical_viewer.py'),
                  '--scene',str(scene),'--output',str(output),'--expected-sha256',expected,
                  '--expected-manifest-sha256',manifest_expected]
    elif args.scene:parser.error('--scene only applies to open')
    return subprocess.call(command,cwd=PROJECT)

if __name__=='__main__':raise SystemExit(main())
