#!/usr/bin/env python3
"""Assemble a self-contained reference-motion handoff without copying caches."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def include_evidence_dependencies(receipts,out,source_map,run):
    """Copy only declared, hash-bound control/baseline evidence transitively."""
    visited=set()
    def include(path,digest=None):
        path=Path(path).resolve()
        if digest is not None:assert sha(path)==digest,('Evidence hash differs',str(path))
        if str(path) not in source_map:
            relative=path.relative_to(run) if path.is_relative_to(run) else Path('external')/sha(path)[:12]/path.name
            target=out/'evidence/supporting'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);source_map[str(path)]=str(target.relative_to(out))
        assert sha(out/source_map[str(path)])==sha(path)
        if path in visited:return
        visited.add(path)
        if path.suffix=='.json':scan(json.loads(path.read_text()),path.parent)
    def scan(value,base):
        def referenced(raw,digest):
            named=Path(raw)
            candidates=[named] if named.is_absolute() else [anchor/named for anchor in [base,Path.cwd(),*run.parents]]
            matches=[candidate.resolve() for candidate in candidates if candidate.is_file() and sha(candidate)==digest]
            assert matches,('Unresolved bound evidence',raw,digest)
            include(matches[0],digest)
            if named.is_absolute() or (named.parts and named.parts[0] in ['runs','experiments']):
                source_map[raw]=source_map[str(matches[0])]
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                referenced(value['path'],value['sha256'])
            for key,item in value.items():
                if key in ['inputs','source_inputs','control_inputs','artifacts'] and isinstance(item,dict):
                    for name,digest in item.items():
                        if isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest):
                            referenced(name,digest)
                scan(item,base)
        elif isinstance(value,list):
            for item in value:scan(item,base)
    for receipt in receipts:include(receipt)
    return dict(files_examined=len(visited),all_declared_bytes_bound=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','take','export','render','self-audit','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--closeup',type=Path)
    args=p.parse_args();r=args.run.resolve();take=args.take.resolve();out=args.output.resolve()
    code=Path(__file__).resolve().parent
    camera_task=args.export.parent/(args.export.name.replace('_export','_inputs'))/'task_camera_override.json'
    task=json.loads(camera_task.read_text());original_task=json.loads((take/'task.json').read_text())
    assert {k:v for k,v in task.items() if k!='preview_cameras'}=={k:v for k,v in original_task.items() if k!='preview_cameras'}
    trajectory_sha=sha(take/'trajectory.npz')
    verify=json.loads((take/'verification_full/results.json').read_text())
    exported=json.loads((args.export/'export.receipt.json').read_text())
    rendered=json.loads((args.render/'render.receipt.json').read_text())
    self_audit=json.loads((args.self_audit/'receipt.json').read_text())
    build=json.loads((take/'build.json').read_text())
    assert verify['kinematic_checks_passed'] and all(verify['checks'].values()) and verify['full_frame_collision_check']
    assert verify['trajectory_sha256']==trajectory_sha==build['trajectory_sha256']
    assert verify['task_sha256']==sha(take/'task.json')==build['task_sha256']
    assert verify['robot_model_sha256']==task['robot']['sha256']==sha(task['robot']['model'])
    assert verify['script_sha256']==sha(code/'g1_motion_verify.py')
    assert build['script_sha256']==sha(code/'g1_motion_plan.py')
    assert exported['status']=='passed' and rendered['status']=='passed'
    assert all(c['passed'] for c in exported['checks']+exported['input_identity_checks']+rendered['checks'])
    assert exported['implementation_sha256']==sha(code/'g1_motion_usd.py')
    assert rendered['implementation_sha256']==sha(code/'g1_motion_render.py')
    assert {trajectory_sha,sha(camera_task),task['robot']['sha256']} <= {v['sha256'] for v in exported['inputs']}
    assert rendered['export_receipt_sha256']==sha(args.export/'export.receipt.json')
    assert rendered['capture_mode']=='uniform_timeline_movie'
    for record in exported['inputs']+exported['model_files']+exported['artifacts']:
        assert sha(record['path'])==record['sha256']
    for name,digest in rendered['input_files_sha256'].items():assert sha(name)==digest
    for sample in rendered['samples']:assert sha(sample['image']['path'])==sample['image']['sha256']
    assert sha(rendered['movie']['path'])==rendered['movie']['sha256']
    assert sha(args.render/'samples.jsonl')==rendered['samples_jsonl_sha256']
    assert self_audit['no_new_non_neighbor_visual_intersections'] and self_audit['native_saved_pose_contact_free']
    assert all(self_audit['evidence_integrity_checks'].values())
    assert self_audit['inputs'][str(take/'trajectory.npz')]==trajectory_sha
    assert self_audit['inputs'][str(take/'task.json')]==sha(take/'task.json')
    for name,digest in self_audit['inputs'].items():assert sha(name)==digest
    for name,digest in self_audit['artifacts'].items():assert sha(args.self_audit/name)==digest
    out.mkdir(parents=True,exist_ok=False)
    source_map={}
    def copy(source,destination):
        source=Path(source).resolve();target=out/destination
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        source_map[str(source)]=str(target.relative_to(out));return target
    def model(source,name):
        source=Path(source).resolve();xml=ET.parse(source).getroot()
        assert not xml.findall('.//include')
        compiler=xml.find('compiler');meshdir=compiler.get('meshdir','') if compiler is not None else ''
        files={source}|{(source.parent/meshdir/n.attrib['file']).resolve() for n in xml.findall('.//asset/mesh') if 'file' in n.attrib}
        common=Path(os.path.commonpath([str(f.parent) for f in files]))
        for f in sorted(files):copy(f,Path('robot')/name/f.relative_to(common))
        return source_map[str(source)]
    for source in args.export.rglob('*'):
        if source.is_file():copy(source,Path('scene')/source.relative_to(args.export))
    for record in exported['environment_files']:
        source_map[record['source']['path']]=source_map[record['copy']['path']]
    robot_path=model(task['robot']['model'],'g1_dex3')
    reach_model=model('/home/ubuntu/projects/kimodo-scene-refactor/kimodo/assets/skeletons/g1skel34/xml/g1.xml','reach_source')
    source_map[task['scene']['path']]='scene/environment/iterations/mesh09/scene.usda'
    assert sha(out/source_map[task['scene']['path']])==task['scene']['sha256']
    for entry in task['robot']['asset_files']:
        assert entry['path'] in source_map and sha(out/source_map[entry['path']])==entry['sha256']
    assets=r/'research/motion_assets'
    for glob in ['*_qpos*.npz','*diagnostic.png']:
        for f in sorted(assets.glob(glob)):copy(f,Path('library/source')/f.name)
    for name in ['README.md','curation.json','joint_mapping.json','robot_assets.json','audit_receipt.json',
                 'verification.json','reach_source_record.json']:
        copy(assets/name,Path('library/source')/name)
    for name in ['trajectory.npz','timeline.json','build.json']:
        copy(take/name,Path('motion')/name)
    save(out/'motion/task.json',task)
    assert sha(out/'motion/task.json')==sha(camera_task)
    source_map[str(camera_task.resolve())]='motion/task.json'
    for record in exported['inputs']:
        if record['sha256']==trajectory_sha:source_map[record['path']]='motion/trajectory.npz'
    copy(take/'task.json','motion/task.before_camera.json')
    z=np.load(take/'trajectory.npz',allow_pickle=False);phases=json.loads((take/'timeline.json').read_text())
    phase_index=[]
    for phase in phases:
        a=phase['start_frame'];b=phase['end_frame']+1
        data={key:z[key][a:b] for key in z.files if z[key].ndim and len(z[key])==len(z['time'])}
        data['time']=data['time']-data['time'][0];data['fps']=z['fps']
        name=f"library/edited/{phase['name']}.npz";(out/name).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/name,**data)
        phase_index.append(dict(phase,clip=name,sha256=sha(out/name),coordinates='original kitchen world frame'))
    save(out/'library/edited/index.json',phase_index)
    qnames=['root_x','root_y','root_z','root_qw','root_qx','root_qy','root_qz']+task['robot']['joint_names']
    assert len(qnames)==z['qpos'].shape[1]
    save(out/'motion/joint_order.json',dict(qpos=qnames,units='meters, WXYZ quaternion, radians',fps=float(z['fps'])))
    with (out/'motion/trajectory.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['time','phase','door_radians']+qnames)
        for i,q in enumerate(z['qpos']):writer.writerow([float(z['time'][i]),str(z['phase'][i]),float(z['door_angle'][i])]+q.tolist())
    for name in ['collision.npz','collision.json','task.json','top_view.png']:
        copy(r/'scene_inputs'/name,Path('scene_inputs')/name)
    for name in ['g1_motion_plan.py','g1_motion_scene.py','g1_motion_verify.py','g1_motion_usd.py',
                 'g1_motion_render.py','g1_motion_bundle.py','g1_motion_package.py','G1_MOTION_BRIEF.md']:
        copy(code/name,Path('code')/name)
    copy(code/'G1_MOTION_README.md','README.md')
    copy(code/'G1_MOTION_REPORT.md','REPORT.md')
    for f in args.self_audit.rglob('*'):
        if f.is_file():copy(f,Path('evidence/self_collision')/f.relative_to(args.self_audit))
    closure=include_evidence_dependencies([args.self_audit/'receipt.json'],out,source_map,r)
    save(out/'evidence/supporting_closure.json',closure)
    for name in ['results.json','measurements.npz']:
        copy(take/'verification_full'/name,Path('evidence/kinematics')/name)
    for name in ['render.receipt.json','samples.json','samples.jsonl','ffprobe.json']:
        copy(args.render/name,Path('evidence/render')/name)
    for f in sorted((args.render/'frames').glob('*.png')):
        copy(f,Path('evidence/render/frames')/f.name)
    movies=list(args.render.glob('*.mp4'));assert len(movies)==1,movies
    copy(movies[0],'G1_fridge_motion.mp4')
    frames=sorted((args.render/'frames').glob('*.png'));assert frames
    for index,label in [(0,'start'),(len(frames)//3,'approach'),(int(len(frames)*.69),'grasp'),(len(frames)-1,'open')]:
        copy(frames[index],Path('images')/(label+'.png'))
    if args.closeup:
        close=json.loads((args.closeup/'render.receipt.json').read_text())
        close_export_path=Path(close['request']['export_receipt'])
        close_export=json.loads(close_export_path.read_text())
        assert close['status']=='passed' and all(v['passed'] for v in close['checks'])
        assert close_export['status']=='passed' and all(v['passed'] for v in close_export['checks']+close_export['input_identity_checks'])
        assert close['export_receipt_sha256']==sha(close_export_path)
        assert {trajectory_sha,task['robot']['sha256']} <= {v['sha256'] for v in close_export['inputs']}
        assert close_export['source_scene_sha256']==task['scene']['sha256']
        for name,digest in close['input_files_sha256'].items():assert sha(name)==digest
        close_sample=close['samples'][0];close_image=close_sample['image']
        assert sha(close_image['path'])==close_image['sha256']
        copy(close_image['path'],'images/dex3_grasp.png')
        save(out/'evidence/hand_closeup.json',dict(schema='g1-closeup-evidence/v1',
             claim='Native rendering of the selected hand and rail at the saved grasp pose; no physical grasp validation.',
             trajectory_sha256=trajectory_sha,robot_sha256=task['robot']['sha256'],scene_sha256=task['scene']['sha256'],
             source_render_receipt_sha256=sha(args.closeup/'render.receipt.json'),
             source_export_receipt_sha256=sha(close_export_path),
             camera=close_export['cameras'],authored_frame=close_sample['authored_sample_index'],
             time_seconds=close_sample['target_seconds'],path='images/dex3_grasp.png',sha256=close_image['sha256']))
    license_path=Path('/home/ubuntu/projects/kimodo-scene-refactor/LICENSE')
    if license_path.is_file():copy(license_path,'provenance/kimodo_LICENSE')
    hand_doc=Path(task['robot']['model']).parent/'g1_joint_index_dds.md'
    if hand_doc.exists():copy(hand_doc,'robot/g1_joint_index_dds.md')
    # Original receipts remain immutable. Bind every already-bundled receipt
    # input to its relocated bytes instead of relying on authoring-machine paths.
    bound_inputs={v['path']:v['sha256'] for v in exported['inputs']+exported['model_files']}
    bound_inputs.update(rendered['input_files_sha256']);bound_inputs.update(self_audit['inputs'])
    for record in exported['environment_files']:
        bound_inputs[record['source']['path']]=record['source']['sha256']
    for name,digest in bound_inputs.items():
        assert name in source_map,('Missing portable receipt input',name)
        assert sha(out/source_map[name])==digest
    save(out/'portable.json',dict(schema='g1-motion-portable/v1',
         task='motion/task.json',trajectory='motion/trajectory.npz',robot=robot_path,reach_model=reach_model,
         walk01=source_map[str((assets/'walk_forward_01_qpos50.npz').resolve())],
         walk02=source_map[str((assets/'walk_forward_02_qpos50.npz').resolve())],
         reach36=source_map[str((assets/'reach_radial_06_qpos36.npz').resolve())],
         preview_scene='scene/scene.usda',physical_scene='scene/environment/iterations/mesh09/scene.usda',
         collision_dir='scene_inputs',source_path_map=source_map))
    save(out/'recipe.json',dict(task=task['plan'],door=task['door'],grasp=task['grasp'],
         selected_take=take.name,mode='offline small-library clip selection, placement and IK editing',
         physical_execution_verified=False,phase_clips='library/edited/index.json'))
    save(out/'evidence/integration.json',dict(schema='g1-motion-integration/v1',
         claim='One bound kinematic motion, model, scene, export and movie; no dynamic execution claim.',
         trajectory_sha256=trajectory_sha,task_before_camera_sha256=sha(take/'task.json'),
         task_with_camera_sha256=sha(camera_task),robot_sha256=task['robot']['sha256'],
         scene_sha256=task['scene']['sha256'],
         verification_sha256=sha(take/'verification_full/results.json'),
         self_audit_sha256=sha(args.self_audit/'receipt.json'),
         export_receipt_sha256=sha(args.export/'export.receipt.json'),
         render_receipt_sha256=sha(args.render/'render.receipt.json'),
         movie_sha256=sha(out/'G1_fridge_motion.mp4'),
         saved_frame_geometric_checks_passed=True,physical_execution_verified=False))
    (out/'requirements.txt').write_text('numpy==2.2.6\nscipy==1.15.3\nmujoco==3.8.1\ntrimesh==4.12.2\nusd-core==25.11\npython-fcl==0.7.0.11\nPillow\n\n# Rendering separately requires NVIDIA Isaac Sim 5.1 and its Python runtime.\n')
    (out/'index.html').write_text('''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 · Refrigerator motion</title>
<style>
body{margin:0;background:#101519;color:#e8edf0;font:16px/1.5 system-ui,sans-serif}main{max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:32px;margin:0 0 6px}p{color:#bec9cf;max-width:850px}video{width:100%;background:#070b0e;border-radius:12px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}button,a{color:#a9d9ec}button{background:#23323b;border:1px solid #43545e;border-radius:7px;padding:10px 16px;font:inherit;cursor:pointer}.facts{display:flex;gap:30px;margin:18px 0;flex-wrap:wrap}.facts b{font-size:24px;display:block;color:#eff6f8}.facts span{color:#b4c3ca;font-size:14px}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.images img{width:100%;border-radius:8px}small{color:#a0b0b9}footer{margin:30px 0;border-top:1px solid #33434d;padding-top:20px}
</style><main>
<small>IMG_2296 · Editable motion reference</small><h1>G1 walks to the fridge and opens the door</h1>
<p>Existing walking and reach clips, placed and corrected in the modeled kitchen. Unitree Dex3 hands follow the articulated door through a 65° opening.</p>
<video id="motion" controls playsinline preload="metadata" poster="images/start.png"><source src="G1_fridge_motion.mp4" type="video/mp4"></video>
<nav aria-label="Motion phases"><button data-t="0">Start</button><button data-t="0.7">Walk</button><button data-t="6.63">Turn</button><button data-t="10.03">Reach</button><button data-t="13.03">Grasp</button><button data-t="13.83">Open door</button></nav>
<div class="facts"><div><b>1.96 m</b><span>Approach</span></div><div><b>65°</b><span>Door opening</span></div><div><b>589 poses</b><span>30 Hz reference</span></div><div><b>Dex3</b><span>Articulated fingers</span></div></div>
<p>Saved-pose checks pass for room clearance, foot placement, joint limits and handle alignment. The independent audit records native self-contact clearance and retained baseline mesh seams. This is kinematic playback; physical controller execution and grasp forces remain untested. Room dimensions are nominal estimates.</p>
<div class="images"><figure><img src="images/approach.png" alt="G1 approaching the fridge"><figcaption>Approach and turn</figcaption></figure><figure><img src="images/grasp.png" alt="G1 reaching the fridge handle"><figcaption>Dex3 grasp reference</figcaption></figure><figure><img src="images/open.png" alt="G1 holding the open fridge door"><figcaption>Door held open</figcaption></figure></div>
<footer><nav><a href="README.md">Open and rebuild</a><a href="REPORT.md">Method and limitations</a><a href="scene/scene.usda">USD scene</a><a href="motion/trajectory.npz">Motion data</a><a href="evidence/integration.json">Evidence bindings</a></nav><small>One composed scene, with source clips, corrected phase clips, robot meshes and reproducible code included.</small></footer>
</main><script>document.querySelectorAll('[data-t]').forEach(b=>b.addEventListener('click',()=>{const v=document.getElementById('motion');v.currentTime=Number(b.dataset.t);v.play()}));</script></html>''')
    if args.closeup:
        page=out/'index.html';page.write_text(page.read_text().replace('<footer>','<figure><img style="width:100%;border-radius:8px" src="images/dex3_grasp.png" alt="Close view of the articulated Dex3 fingers around the fridge rail"><figcaption>Dex3 grasp pose · geometric reference</figcaption></figure><footer>'))
    print(json.dumps(dict(output=str(out),files=sum(f.is_file() for f in out.rglob('*')),selected_take=take.name),indent=2))


if __name__=='__main__':main()
