#!/usr/bin/env python3
"""Portable CPU ScaleBFM/MuJoCo replay; install as <component>/code/physics.py.

The component manifest selects its sibling reference. Historical components
remain immutable; this source is used only when staging a new component.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,re,subprocess,sys,importlib.metadata as metadata
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
def reference_root(root):
    relative=Path(json.loads((root/'manifest.json').read_text())['reference_component']['path'])
    resolved=(root/relative).resolve()
    if relative.is_absolute() or resolved.parent!=root.parent.resolve() or resolved==root.resolve():
        raise ValueError('Reference must be one adjacent component')
    return resolved
REF=reference_root(ROOT)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def below(p,root):return Path(p).resolve().is_relative_to(root.resolve())
def local(p):
    p=Path(p).resolve()
    if not (below(p,ROOT) or below(p,REF)):raise ValueError('Runtime input leaves staged components: '+str(p))
    return p
def output(p):
    p=Path(p).resolve()
    if not below(p,ROOT/'reproduced'):raise ValueError('Derived writes must be under '+str(ROOT/'reproduced'))
    return p
def case(name):
    row=read(ROOT/'cases.json')['cases'][name];p=local(ROOT/row['path'])
    if sha(p)!=row['sha256']:raise ValueError('Case descriptor changed')
    return read(p)
def verify(quiet=False):
    manifest=read(ROOT/'manifest.json');bad=[]
    for row in manifest['files']:
        p=local(ROOT/row['path'])
        if not p.is_file() or p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:bad.append(row['path'])
    dep=manifest['reference_component']
    if sha(REF/'manifest.json')!=dep['manifest_sha256']:bad.append(dep['path']+'/manifest.json')
    if sha(REF/'stage.receipt.json')!=dep['receipt_sha256']:bad.append(dep['path']+'/stage.receipt.json')
    # Check the complete sibling manifest as well as direct trial input bindings.
    sibling=read(REF/'manifest.json')
    for row in sibling['files']:
        p=local(REF/row['path'])
        if not p.is_file() or p.stat().st_size!=row['size'] or sha(p)!=row['sha256']:bad.append(dep['path']+'/'+row['path'])
    checked=0
    for name in read(ROOT/'cases.json')['cases']:
        c=case(name);raw=ROOT/c['raw'];ins=read(raw/'inputs.json');receipt=read(raw/'receipt.json')
        if sha(raw/'inputs.json')!=c['historical_inputs_sha256']:bad.append(name+': inputs receipt')
        if sha(raw/'receipt.json')!=c['historical_receipt_sha256']:bad.append(name+': rollout receipt')
        for old,h in {**ins['inputs'],**ins['model_dependencies']}.items():
            p=local(ROOT/c['input_aliases'][old]);checked+=1
            if sha(p)!=h:bad.append(name+': '+old)
        for rel,h in receipt['artifacts'].items():
            p=local(raw/rel)
            if sha(p)!=h:bad.append(name+': raw '+rel)
    result={'schema':'g1-physical-stage-transfer-check/v1','passed':not bad,
            'manifest_sha256':sha(ROOT/'manifest.json'),'own_files_checked':len(manifest['files']),
            'reference_files_checked':len(sibling['files']),'historical_input_bindings_checked':checked,
            'mismatches':bad,'scope':'Byte and dependency closure; no new physical task-success claim.'}
    if not quiet:print(json.dumps(result,indent=2))
    if bad:raise RuntimeError('Staged input verification failed: '+str(bad[:5]))
    return result
def runtime_check():
    expected=read(ROOT/'runtime.json');got={n:metadata.version(n) for n in expected['packages']}
    differences={n:{'expected':v,'actual':got[n]} for n,v in expected['packages'].items() if got[n]!=v}
    if sys.version_info[:3]!=(3,10,13):differences['python']={'expected':'3.10.13','actual':sys.version.split()[0]}
    return {'schema':'g1-physical-runtime-check/v1','passed':not differences,'python':sys.version,
            'packages':got,'device':'CPU','differences':differences}
def prepare(name,out,duration=None):
    out=output(out)
    if out.exists():raise FileExistsError('Choose a fresh derived output directory')
    c=case(name);raw=ROOT/c['raw'];original=read(raw/'inputs.json')
    aliases={k:str(local(ROOT/v['staged_path'])) for k,v in read(ROOT/'aliases.json')['paths'].items()}
    # Case-local mappings remain authoritative if a shared helper has multiple exact copies.
    aliases.update({k:str(local(ROOT/v)) for k,v in c['input_aliases'].items()})
    def relocate(v):
        if isinstance(v,str):return aliases.get(v,v)
        if isinstance(v,list):return [relocate(x) for x in v]
        if isinstance(v,dict):return {aliases.get(k,k):relocate(x) for k,x in v.items()}
        return v
    out.mkdir(parents=True)
    inp=out/'inputs';inp.mkdir()
    a=relocate(original['arguments']);a['models']=str(ROOT/'models/scalebfm')
    task=relocate(read(Path(a['task'])))
    scene_task_source=local(ROOT/c['scene_task_source'])
    scene_task=relocate(read(scene_task_source))
    arm=relocate(read(Path(a['arm_config'])))
    save(inp/'task.json',task);save(inp/'scene_task.json',scene_task);save(inp/'arm_config.json',arm)
    adapter=relocate(read(REF/'native_scene/scene.receipt.json'))
    adapter['task']={'path':str(inp/'scene_task.json'),'sha256':sha(inp/'scene_task.json')}
    adapter['output']['path']=str(REF/'native_scene/scene.xml')
    adapter['relocation_only']={'original_receipt_sha256':sha(REF/'native_scene/scene.receipt.json'),
         'scope':'Only file paths and derived scene-task hash changed; native scene/model/object geometry is unchanged.'}
    save(inp/'scene.receipt.json',adapter)
    a.update(task=str(inp/'task.json'),scene_receipt=str(inp/'scene.receipt.json'),arm_config=str(inp/'arm_config.json'),
             scene=str(REF/'native_scene/scene.xml'),output=str(out/'rollout'))
    if duration is not None:
        if not .02<=duration<=160:raise ValueError('Duration must be between .02 and 160 seconds')
        a['duration']=float(duration)
    for key in ['robot','scene','reference','task','models','parameters','policy_plugin','arm_plugin','arm_config','scene_receipt','physics_options']:
        if key=='physics_options' and not a.get(key):continue
        p=local(a[key])
        if not p.exists():raise FileNotFoundError(p)
    assert sha(a['scene'])==adapter['output']['sha256']
    # Exact physical semantic contract, apart from paths and historical descriptive reference metadata.
    for k in ['scene','object','place','door','robot','grasp']:assert task[k]==scene_task[k],k
    assert task['scene']==adapter['scene'],'Task and adapter room identity'
    room=local(task['scene']['path'])
    assert room.is_file() and sha(room)==task['scene']['sha256'],'Local room source bytes'
    assert ('object_grasp' in task)==('object_grasp' in scene_task),'object_grasp presence'
    if 'object_grasp' in task:assert task['object_grasp']==scene_task['object_grasp'],'object_grasp'
    script=local(ROOT/c['script'])
    cmd=[sys.executable,str(script)]
    for k,v in a.items():
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v:cmd.append(flag)
            elif k=='stop_on_fall':cmd.append('--no-stop-on-fall')
        elif v is not None:cmd.extend([flag,str(v)])
    save(inp/'arguments.json',a)
    receipt={'schema':'g1-physical-input-relocation/v1','case':name,'produced_at':now(),'passed':True,
       'portable_helper_sha256':sha(Path(__file__)), 'input_manifest_sha256':sha(ROOT/'manifest.json'),
       'historical_input_receipt_sha256':sha(raw/'inputs.json'),'historical_rollout_receipt_sha256':sha(raw/'receipt.json'),
       'scene_sha256':sha(a['scene']),'reference_sha256':sha(a['reference']),
       'archived_executed_scene_sha256':sha(raw/'executed_scene.xml'),
       'reference_component':str(REF.relative_to(ROOT.parent)),
       'scene_task_source_sha256':sha(scene_task_source),
       'fresh_scene_wrapper':'The unchanged archived helper compiles rollout/executed_scene.xml from the selected sibling native_scene/scene.xml.',
       'geometry_or_solver_edits':False,'object_attachment':False,'diagnostic_initial_task_reset':a['diagnostic_reset_task_from_reference'],
       'duration_override_seconds':duration,'controller_script_sha256':sha(script),
       'derived_inputs':{str(p.relative_to(out)):sha(p) for p in inp.iterdir() if p.is_file()},'command':cmd}
    save(out/'relocation.receipt.json',receipt)
    return out,cmd
def compare(name,out):
    import numpy as np
    out=output(out);c=case(name);old=ROOT/c['raw'];new=out/'rollout'
    checks={};metrics={}
    rec=read(new/'receipt.json');ins=read(new/'inputs.json');rel=read(out/'relocation.receipt.json')
    # Join the native receipts to actual bytes before accepting any prefix.
    checks['historical_input_receipt_hash']=sha(old/'inputs.json')==c['historical_inputs_sha256']
    checks['historical_rollout_receipt_hash']=sha(old/'receipt.json')==c['historical_receipt_sha256']
    required={'inputs.json','executed_scene.xml','rollout.npz','first_inference.npz','contacts.jsonl','task_contacts_substeps.jsonl'}
    for label,base,receipt in [('historical',old,read(old/'receipt.json')),('new',new,rec)]:
        artifacts=receipt.get('artifacts',{})
        checks[label+':required_artifact_bindings']=required<=set(artifacts)
        for fname,h in artifacts.items():
            path=(base/fname).resolve()
            checks[label+':artifact:'+fname]=below(path,base) and path.is_file() and sha(path)==h
        checks[label+':native_input_hash']=receipt.get('inputs_receipt',{}).get('sha256')==sha(base/'inputs.json')
    checks['new_native_input_path']=Path(rec.get('inputs_receipt',{}).get('path','')).resolve()==(new/'inputs.json').resolve()
    checks['new_native_scene_binding']=ins.get('executed_scene',{}).get('sha256')==sha(new/'executed_scene.xml') and Path(ins.get('executed_scene',{}).get('path','')).resolve()==(new/'executed_scene.xml').resolve()
    checks['relocation_case_and_historical_inputs']=rel.get('case')==name and rel.get('historical_input_receipt_sha256')==c['historical_inputs_sha256'] and rel.get('historical_rollout_receipt_sha256')==c['historical_receipt_sha256']
    for fname,h in rel.get('derived_inputs',{}).items():
        path=(out/fname).resolve()
        checks['relocation_input:'+fname]=below(path,out) and path.is_file() and sha(path)==h
    checks['native_arguments_equal_prepared_arguments']=ins.get('arguments')==read(out/'inputs/arguments.json')
    args=ins['arguments'];dt=float(args['timestep']);ratio=.02/dt
    checks['positive_integral_substeps']=math.isfinite(dt) and dt>0 and abs(ratio-round(ratio))<1e-9
    substeps=int(round(ratio))
    duration=args.get('duration')
    if duration is None:
        with np.load(local(args['reference']),allow_pickle=False) as reference:
            duration=float(reference['time'][-1])-float(args['start'])
    expected_samples=int(math.ceil(float(duration)*50))
    n=rec.get('samples');steps=rec.get('contact_sampling',{}).get('task_contact_substeps')
    valid_n=type(n) is int and n>0 and n==expected_samples and n==rec.get('requested_samples')
    checks['nonzero_requested_and_actual_sample_counts']=valid_n
    checks['exact_physics_step_count']=valid_n and type(steps) is int and steps==n*substeps
    checks['declared_contact_rates']=rec.get('contact_sampling',{}).get('general_contacts_hz')==50 and rec.get('contact_sampling',{}).get('task_contacts_hz')==1/dt
    checks['declared_simulation_duration']=valid_n and math.isfinite(float(rec['simulated_seconds'])) and abs(float(rec['simulated_seconds'])-n*.02)<1e-8
    for fname in ['first_inference.npz','rollout.npz']:
        a=np.load(old/fname,allow_pickle=False);b=np.load(new/fname,allow_pickle=False)
        checks[fname+':field_names']=bool(a.files) and set(a.files)==set(b.files)
        for k in sorted(set(a.files)&set(b.files)):
            x=a[k];y=b[k]
            if fname=='rollout.npz':
                expected=expected_samples*substeps if k.startswith('substep_') else expected_samples
                checks[fname+':coverage:'+k]=y.ndim>0 and len(y)==expected and len(x)>=expected
                x=x[:expected]
            numeric=x.dtype.kind not in 'USO' and y.dtype.kind not in 'USO'
            if numeric:checks[fname+':finite:'+k]=bool(np.isfinite(y).all())
            # NumPy sizes text arrays to the longest label actually present;
            # a short prefix may legitimately have a smaller Unicode width.
            checks[fname+':dtype:'+k]=x.dtype==y.dtype if numeric else x.dtype.kind==y.dtype.kind
            if k=='inference_s':
                checks[fname+':inference_timing_shape']=x.shape==y.shape
                continue # Wall-clock performance cannot be bit-identical.
            same=x.shape==y.shape and np.array_equal(x,y)
            key=fname+':'+k;checks[key]=bool(same)
            if numeric and x.shape==y.shape:
                delta=np.abs(x.astype(float)-y.astype(float));finite=delta[np.isfinite(delta)]
                metrics[key]={'maximum_absolute_difference':float(finite.max()) if finite.size else 0.,'shape':list(y.shape)}
        if fname=='rollout.npz':
            t=b['time'];target=np.arange(1,expected_samples+1)*.02
            checks['state_timestamps_cover_every_control_tick']=t.shape==target.shape and bool(np.allclose(t,target,rtol=0,atol=1e-8))
        a.close();b.close()
    # Empty contact arrays still require one record per control/physics step.
    line_counts={}
    for fname in ['contacts.jsonl','task_contacts_substeps.jsonl']:
        count=0;equal=True;timestamps=True
        expected=expected_samples if fname=='contacts.jsonl' else expected_samples*substeps
        period=.02 if fname=='contacts.jsonl' else dt
        with (old/fname).open() as a,(new/fname).open() as b:
            for line in b:
                count+=1
                value=json.loads(line);oldline=a.readline()
                if not oldline or value!=json.loads(oldline):equal=False
                tm=float(value['time'])
                if not math.isfinite(tm) or abs(tm-count*period)>1e-8 or not isinstance(value.get('contacts'),list):timestamps=False
        checks[fname]=equal;checks[fname+':exact_nonzero_coverage']=count==expected and count>0
        checks[fname+':timestamps']=timestamps;line_counts[fname]=count
    badpaths=[]
    for path,h in {**ins['inputs'],**ins['model_dependencies']}.items():
        p=Path(path)
        if not (below(p,ROOT) or below(p,REF)) or not p.is_file() or sha(p)!=h:badpaths.append(path)
    checks['all_new_runtime_inputs_within_two_components_and_hash_bound']=not badpaths
    checks['completed_duration_without_fall_or_exception']=rec['completed_requested_duration'] and not rec['fall'] and not rec['exception']
    result={'schema':'g1-physical-relocated-prefix-comparison/v1','case':name,'produced_at':now(),
       'comparison_helper_sha256':sha(Path(__file__)),
       'passed':all(checks.values()),'checks':checks,'array_metrics':metrics,'contact_prefix_line_counts':line_counts,
       'historical_inputs_sha256':sha(old/'inputs.json'),'historical_rollout_sha256':sha(old/'rollout.npz'),
       'new_inputs_sha256':sha(new/'inputs.json'),'new_rollout_sha256':sha(new/'rollout.npz'),
       'new_receipt_sha256':sha(new/'receipt.json'),'relocation_receipt_sha256':sha(out/'relocation.receipt.json'),
       'new_runtime_input_count':len(ins['inputs'])+len(ins['model_dependencies']),'bad_runtime_paths':badpaths,
       'simulated_seconds':rec['simulated_seconds'],'saved_samples':rec['samples'],
       'physics_steps':rec['contact_sampling']['task_contact_substeps'],
       'excluded_fields':['inference_s (wall-clock duration)'],
       'scope':'Only the produced historical prefix is compared. No full-task physical success or later stability is inferred.'}
    save(out/'comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['array_metrics','checks']},indent=2))
    return result
def execute(name,run_name,duration):
    verify(quiet=True);runtime=runtime_check()
    if not runtime['passed']:raise RuntimeError('Pinned runtime differs: '+str(runtime['differences']))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'reproduced/runs'/run_name/stamp
    out,cmd=prepare(name,out,duration);save(out/'runtime.json',runtime)
    save(ROOT/'reproduced/runs'/run_name/'latest.json',{'case':name,'path':str(out.relative_to(ROOT)),'started_at':now()})
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run(cmd,cwd=ROOT,env=env)
    summary={'schema':'g1-physical-replay-status/v1','case':name,'output':str(out.relative_to(ROOT)),
             'returncode':result.returncode,'finished_at':now(),'command':cmd}
    save(out/'execution.json',summary)
    if result.returncode:raise SystemExit(result.returncode)
    result=compare(name,out)
    if not result['passed']:raise SystemExit(2)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','runtime','prepare','run','status','logs','stop','restart','execute','compare'])
    p.add_argument('--case',default=read(ROOT/'cases.json').get('default_case','antipodal_hold1'));p.add_argument('--name');p.add_argument('--duration',type=float)
    p.add_argument('--timeout',type=int);p.add_argument('--output',type=Path);a=p.parse_args()
    name=a.name or a.case
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name):raise ValueError('Invalid owned run name')
    if a.command=='verify':verify();return
    if a.command=='runtime':
        r=runtime_check();print(json.dumps(r,indent=2))
        if not r['passed']:raise SystemExit(1)
        return
    if a.command=='execute':execute(a.case,name,a.duration);return
    if a.command=='compare':
        if not a.output:raise ValueError('--output must name the replay directory')
        if not compare(a.case,a.output)['passed']:raise SystemExit(2)
        return
    if a.command=='prepare':
        verify();out=a.output or ROOT/'reproduced/prepared'/name
        where,cmd=prepare(a.case,out,a.duration);print(json.dumps({'output':str(where),'command':cmd},indent=2));return
    job=[sys.executable,str(ROOT/'code/jobctl.py'),'--root',str(ROOT/'reproduced/jobs')]
    if a.command=='run':
        verify(quiet=True);c=case(a.case);bound=a.timeout or c['default_timeout_seconds']
        if not 15<=bound<=600:raise ValueError('Timeout must be 15–600 seconds')
        cmd=['env','PYTHONPATH='+str(ROOT/'code'),'PYTHONDONTWRITEBYTECODE=1',
             'timeout','--signal=TERM','--kill-after=5s',str(bound)+'s',sys.executable,'-m','physics',
             'execute','--case',a.case,'--name',name]
        if a.duration is not None:cmd+=['--duration',str(a.duration)]
        job+=['start',name,'--gpu','','--',*cmd]
    else:job+=[a.command,name]
    subprocess.run(job,check=True,cwd=ROOT)
if __name__=='__main__':main()
