#!/usr/bin/env python3
"""Portable command surface included with the delivered motion bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','verify','build','export','render'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--trajectory',type=Path,help='Optional edited trajectory for export or verification')
    parser.add_argument('--task',type=Path,help='Optional task associated with the edited trajectory')
    parser.add_argument('--scene',type=Path,help='Optional exported scene for rendering')
    parser.add_argument('--gpu',type=int,default=0,help='Physical GPU ordinal for native Isaac rendering')
    args=parser.parse_args();root=args.root.resolve()
    config=json.loads((root/'portable.json').read_text())
    def path(key):return (root/config[key]).resolve()
    if args.action=='check':
        manifest=json.loads((root/'files.sha256.json').read_text())
        failures=[name for name,digest in manifest.items() if not (root/name).is_file() or sha(root/name)!=digest]
        print(json.dumps(dict(files=len(manifest),mismatches=failures,passed=not failures),indent=2))
        return 1 if failures else 0
    output=(args.output or root/'reproduced'/args.action).resolve()
    output.mkdir(parents=True,exist_ok=True)
    task=json.loads((args.task or path('task')).read_text())
    trajectory=(args.trajectory or path('trajectory')).resolve()
    mapping=config['source_path_map']
    task['scene']['path']=str(path('physical_scene'))
    task['robot']['model']=str(path('robot'))
    for entry in task['robot']['asset_files']:
        entry['path']=str((root/mapping[entry['path']]).resolve()) if entry['path'] in mapping else str(Path(entry['path']).resolve())
    for entry in task['plan']['matching_candidates']+[task['plan']['selected_walk'],task['plan']['reach']]:
        entry['path']=str((root/mapping[entry['path']]).resolve()) if entry['path'] in mapping else str(Path(entry['path']).resolve())
    task_path=output/'task.local.json';task_path.write_text(json.dumps(task,indent=2)+'\n')
    code=root/'code';py=sys.executable
    if args.action=='verify':
        command=[py,str(code/'g1_motion_verify.py'),'--trajectory',str(trajectory),
                 '--task',str(task_path),'--collision-dir',str(path('collision_dir')),
                 '--output',str(output/'verification')]
    elif args.action=='build':
        command=[py,str(code/'g1_motion_plan.py'),'--robot',str(path('robot')),
                 '--walk',str(path('walk01')),'--walk',str(path('walk02')),
                 '--reach',str(path('reach36')),'--reach-model',str(path('reach_model')),
                 '--task',str(task_path),'--output',str(output/'motion'),
                 '--fps',str(task['plan']['fps']),'--door-degrees',str(task['plan']['door_degrees']),
                 '--stance-x',str(task['plan']['stance_xy'][0]),'--stance-y',str(task['plan']['stance_xy'][1]),
                 '--approach-yaw',str(task['plan']['approach_yaw']),
                 '--grip-x',str(task['grasp']['grip_point_wrist'][0]),'--grip-y',str(task['grasp']['grip_point_wrist'][1])]
    elif args.action=='export':
        command=[py,str(code/'g1_motion_usd.py'),'--model',str(path('robot')),
                 '--trajectory',str(trajectory),'--task-json',str(task_path),
                 '--scene',str(path('physical_scene')),'--output-dir',str(output/'scene')]
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        os.environ['OMNI_KIT_ACCEPT_EULA']='YES';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        command=[py,str(code/'g1_motion_render.py'),'--scene',str((args.scene or path('preview_scene')).resolve()),
                 '--output-dir',str(output/'render'),'--gpu',str(args.gpu),
                 '--camera','/World/PreviewCameras/Overview','--second-camera','/World/PreviewCameras/Task',
                 '--cut-time','10','--width','1280','--height','720','--fps','15','--subframes','4']
    print(json.dumps({'command':command,'scope':'kinematic reference; no controller execution'},indent=2),flush=True)
    subprocess.run(command,check=True)
    if args.action=='verify':
        result=json.loads((output/'verification/results.json').read_text())
        return 0 if result['kinematic_checks_passed'] else 1
    return 0


if __name__=='__main__':raise SystemExit(main())
