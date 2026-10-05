#!/usr/bin/env python3
"""Check a task-prop-only scene revision against a frozen native adapter."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def stripped(path):
    root=ET.parse(path).getroot();world=root.find('worldbody');asset=root.find('asset')
    body=world.find("body[@name='task_bottle']");assert body is not None;world.remove(body)
    for mesh in list(asset):
        if mesh.tag=='mesh' and mesh.get('name','').startswith('task_bottle_'):asset.remove(mesh)
    return ET.canonicalize(ET.tostring(root,encoding='unicode'),strip_text=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-base-delta',type=float,nargs=3,default=[0.,0.,0.])
    p.add_argument('--placement-only',action='store_true')
    a=p.parse_args();old=a.before.resolve();new=a.after.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    checks={};bindings={};before=read(old/'scene.receipt.json');after=read(new/'scene.receipt.json')
    for base,rec in [(old,before),(new,after)]:
        for name in ['scene.xml','scene.receipt.json']:bindings[str(base/name)]=sha(base/name)
        assert sha(base/'scene.xml')==rec['output']['sha256']
        path=Path(rec['task']['path']);assert sha(path)==rec['task']['sha256'];bindings[str(path)]=sha(path)
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    checks['native_xml_identical_outside_task_prop']=stripped(old/'scene.xml')==stripped(new/'scene.xml')
    checks['source_scene_and_layers_identical']=before['scene']==after['scene'] and before['scene_layers']==after['scene_layers']
    checks['all_room_decomposition_records_identical']=before['room']==after['room']
    checks['all_source_robot_assets_identical']=before['robot']==after['robot']
    count=0
    for row in before['room']:
        rel=Path('room_meshes')/(row['name']+'.obj')
        assert sha(old/rel)==sha(new/rel)==row['sha256'];count+=1
    for row in before['robot']['assets']:
        assert sha(old/row['copy'])==sha(new/row['copy'])==row['sha256'];count+=1
    checks['all_nonprop_asset_bytes_identical']=True
    x=mujoco.MjModel.from_xml_path(str(old/'scene.xml'));y=mujoco.MjModel.from_xml_path(str(new/'scene.xml'))
    xb=x.body('task_bottle').id;yb=y.body('task_bottle').id;assert xb==yb
    for name in ['nq','nv','nu','nbody','njnt','neq','nmocap']:
        checks['count:'+name]=getattr(x,name)==getattr(y,name)
    for name in ['actuator_ctrlrange','actuator_gear','actuator_gainprm','actuator_biasprm','actuator_trnid',
                 'jnt_type','jnt_range','jnt_axis','jnt_pos','jnt_bodyid','jnt_qposadr','jnt_dofadr',
                 'dof_damping','dof_frictionloss','dof_armature']:
        checks[name]=bool(np.array_equal(getattr(x,name),getattr(y,name)))
    keep=np.arange(x.nbody)!=xb
    for name in ['body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat']:
        checks['nonprop:'+name]=bool(np.array_equal(getattr(x,name)[keep],getattr(y,name)[keep]))
    gx=x.geom_bodyid!=xb;gy=y.geom_bodyid!=yb
    for name in ['geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp',
                 'geom_condim','geom_contype','geom_conaffinity','geom_margin','geom_gap','geom_rgba']:
        checks['nonprop:'+name]=bool(np.array_equal(getattr(x,name)[gx],getattr(y,name)[gy]))
    task=read(after['task']['path']);obj=task['object'];prop=np.flatnonzero(y.geom_bodyid==yb)
    assert len(prop)==1;g=int(prop[0]);radius=float(obj['radius']);height=float(obj['height']);mass=float(obj['mass'])
    expected_inertia=np.array([mass*(3*radius**2+height**2)/12]*2+[mass*radius**2/2])
    checks['single_declared_native_cylinder']=obj['shape']=='cylinder' and y.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER and bool(np.array_equal(y.geom_size[g,:2],[radius,height/2]))
    checks['homogeneous_declared_mass_and_inertia']=abs(y.body_mass[yb]-mass)<1e-12 and bool(np.allclose(y.body_inertia[yb],expected_inertia,rtol=0,atol=1e-12)) and bool(np.array_equal(y.body_ipos[yb],np.zeros(3)))
    delta=np.asarray(a.expected_base_delta)
    assert np.isfinite(delta).all()
    checks['base_matches_declared_delta_and_center']=bool(np.allclose(obj['initial_base_world'],np.asarray(before['bottle']['initial_base_world'])+delta,rtol=0,atol=1e-12)) and bool(np.allclose(y.body_pos[yb],np.array(obj['initial_base_world'])+[0,0,height/2],rtol=0,atol=1e-12))
    if a.placement_only:
        old_prop=ET.parse(old/'scene.xml').getroot().find("worldbody/body[@name='task_bottle']")
        new_prop=ET.parse(new/'scene.xml').getroot().find("worldbody/body[@name='task_bottle']")
        old_prop.attrib.pop('pos');new_prop.attrib.pop('pos')
        checks['placement_only_prop_definition_identical']=ET.canonicalize(ET.tostring(old_prop,encoding='unicode'),strip_text=True)==ET.canonicalize(ET.tostring(new_prop,encoding='unicode'),strip_text=True)
        qexpected=x.qpos0.copy();oq=int(x.joint('task_bottle_free').qposadr[0]);qexpected[oq:oq+3]+=delta
        checks['placement_only_compiled_reset_coordinates']=bool(np.allclose(y.qpos0,qexpected,rtol=0,atol=1e-12))
    checks['declared_friction_and_contact_parameters']=bool(np.array_equal(y.geom_friction[g],[obj['friction'],.005,.0001])) and y.geom_condim[g]==4 and bool(np.array_equal(y.geom_solref[g],[.008,1]))
    checks['native_options_unchanged']=all(np.array_equal(getattr(x.opt,k),getattr(y.opt,k)) for k in ['timestep','gravity','solver','cone','iterations','tolerance','impratio','noslip_iterations','disableflags'])
    out.mkdir(parents=True)
    result={'schema':'g1-task-prop-scene-isolation/v1','produced_at':datetime.now(timezone.utc).isoformat(),
        'passed':all(checks.values()),'checks':checks,'inputs':bindings,'nonprop_asset_files_checked':count,
        'before_prop':before['bottle'],'after_prop':obj,'expected_base_delta':delta.tolist(),'placement_only_requested':a.placement_only,
        'scope':'Static declared task-prop revision and preservation of robot/room/door definitions. Does not certify motion clearance, grasp or physical execution.'}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'files':count},indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
