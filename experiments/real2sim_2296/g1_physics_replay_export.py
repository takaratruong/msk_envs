#!/usr/bin/env python3
"""Export exact recorded MuJoCo poses through the existing textured USD exporter.

Rendering this animation does not re-execute physics. The additional receipt
binds every robot/door/bottle pose to the physical rollout, including failures.
The playback clock starts at zero; physical time starts at the first 20 ms sample.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import mujoco


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--exporter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.rollout.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    inp=json.loads((run/'inputs.json').read_text());raw=json.loads((run/'receipt.json').read_text())
    assert sha(run/'inputs.json')==raw['inputs_receipt']['sha256']
    for path,digest in {**inp['inputs'],**inp['model_dependencies']}.items():
        assert sha(path)==digest, path
    for name,digest in raw['artifacts'].items():assert sha(run/name)==digest, name
    args=inp['arguments'];robot=mujoco.MjModel.from_xml_path(args['robot'])
    model=mujoco.MjModel.from_xml_path(inp['executed_scene']['path'])
    assert sha(inp['executed_scene']['path'])==inp['executed_scene']['sha256']
    z=np.load(run/'rollout.npz',allow_pickle=False);n=len(z['time'])
    assert np.allclose(np.diff(z['time']),.02,rtol=0,atol=1e-8)
    q=np.zeros((n,robot.nq))
    for j in range(robot.njnt):
        match=model.joint(robot.joint(j).name).id
        assert model.jnt_type[match]==robot.jnt_type[j]
        width=7 if robot.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1
        assert width==7 or robot.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
        q[:,robot.jnt_qposadr[j]:robot.jnt_qposadr[j]+width]=z['qpos'][:,model.jnt_qposadr[match]:model.jnt_qposadr[match]+width]
    oq=int(model.joint('task_bottle_free').qposadr[0]);dq=int(model.joint('task_fridge_hinge').qposadr[0])
    object_pose=z['qpos'][:,oq:oq+7].copy();angle=z['qpos'][:,dq].copy()
    assert np.allclose(object_pose,z['object_pose'],rtol=0,atol=1e-8)
    assert np.array_equal(angle,z['door_angle'])
    source=out/'inputs';source.mkdir()
    trajectory=source/'actual_trajectory.npz'
    np.savez_compressed(trajectory,qpos=q,time=np.arange(n,dtype=float)/50.,fps=np.array(50.),
        door_angle=angle,object_pose=object_pose,object_pose_origin=np.array('center'),
        simulation_time=z['time'],phase=z['phase'],reference_time=z['reference_time'],
        source=np.array('Recorded torque simulation; no object attachment or pose interpolation'))
    for name in ['receipt.json','inputs.json']:shutil.copyfile(run/name,source/('rollout_'+name))
    shutil.copyfile(run/'rollout.npz',source/'rollout.npz')
    task=Path(args['task']);adapter=Path(args['scene']).with_name('scene.receipt.json')
    subprocess.run([sys.executable,str(a.exporter.resolve()),'--task',str(task),
        '--trajectory',str(trajectory),'--adapter-receipt',str(adapter),'--output',str(out/'usd')],check=True)
    child=json.loads((out/'usd/pickplace_export.receipt.json').read_text())
    assert child['status']=='passed' and child['samples']==n
    copied=np.load(out/'usd/inputs/trajectory.npz',allow_pickle=False)
    fk=np.load(out/'usd/fk_samples.npz',allow_pickle=False)
    assert np.array_equal(copied['qpos'],q) and np.array_equal(copied['object_pose'],object_pose)
    assert np.array_equal(fk['qpos'],q) and np.array_equal(fk['door_angle'],angle)
    bindings={str(x.resolve()):sha(x) for x in [run/'receipt.json',run/'inputs.json',run/'rollout.npz',
        trajectory,a.exporter,Path(__file__),task,adapter,out/'usd/pickplace_export.receipt.json',out/'usd/export.receipt.json']}
    receipt=dict(schema='g1-physical-pose-replay-export/v1',status='passed',produced_at=datetime.now(timezone.utc).isoformat(),
        source_rollout=str(run),source_scope=raw['scope'],diagnostic_reset=bool(inp.get('diagnostic_task_reset')),
        inputs=bindings,source_scene_sha256=child['source_scene']['sha256'],
        source_rollout_sha256=sha(run/'rollout.npz'),export_trajectory_sha256=sha(trajectory),
        portable_source={name:dict(path='../inputs/'+name,sha256=sha(source/name)) for name in ['rollout.npz','rollout_receipt.json','rollout_inputs.json']},
        robot_qpos_map=[dict(name=robot.joint(j).name,robot_address=int(robot.jnt_qposadr[j]),
            scene_address=int(model.jnt_qposadr[model.joint(robot.joint(j).name).id]),
            width=7 if robot.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1) for j in range(robot.njnt)],
        bottle_qpos_address=oq,door_qpos_address=dq,
        task_export_receipt_sha256=sha(out/'usd/pickplace_export.receipt.json'),
        inherited_export_receipt_sha256=sha(out/'usd/export.receipt.json'),
        source_samples=n,source_sample_rate_hz=50,playback_start_seconds=0.,
        physical_start_seconds=float(z['time'][0]),physical_end_seconds=float(z['time'][-1]),
        robot_pose_arrays_equal=True,door_pose_arrays_equal=True,object_pose_arrays_equal=True,
        physics_reexecuted_by_export=False,physical_task_success_claim=False,
        limitations=['Exact sampled-pose replay of the bound physical rollout, including any failed task.',
            'The rendering engine does not integrate physics; simulation success requires the separate physical evaluator.',
            'Playback time zero corresponds to the first recorded physical sample at20ms; no reset pose is fabricated.'])
    (out/'usd/physics_replay.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='passed',scene=str(out/'usd/scene.usda'),receipt_sha256=sha(out/'usd/physics_replay.receipt.json'))))


if __name__=='__main__':main()
