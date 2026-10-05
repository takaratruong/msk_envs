#!/usr/bin/env python3
"""Stage one exact physical trial beside its immutable portable reference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),str(dst)
    else:shutil.copyfile(src,dst)


def validate_evaluation(run, path, evaluator, python=None):
    """Join an evaluator result to this exact raw run before carrying its claim."""
    result=read(path)
    if result.get('schema')!='g1-pickplace-physical-task-evaluation/v1':
        raise ValueError('Unsupported physical evaluation schema')
    if Path(result.get('rollout','')).resolve()!=run.resolve():
        raise ValueError('Evaluation is for a different rollout')
    required_gates={'immutable_declared_inputs','complete_reference_executed',
        'complete_task_substep_contacts','unassisted_initial_task','torque_only_freebase',
        'source_motor_limits','no_applied_external_force','upright','physical_door_opened',
        'opposing_fingers_acquired','physically_lifted','carried_with_hand_contact',
        'reached_target_table','released_and_supported_two_seconds','final_object_stable',
        'no_major_unintended_robot_contacts'}
    gates=result.get('gates',{})
    if set(gates)!=required_gates or any(type(v) is not bool for v in gates.values()):
        raise ValueError('Incomplete or non-Boolean physical gates')
    if type(result.get('complete_physical_task_passed')) is not bool or result['complete_physical_task_passed']!=all(gates.values()):
        raise ValueError('Physical success flag disagrees with its gates')
    if result.get('input_mismatches')!=[]:
        raise ValueError('Evaluation has unresolved input mismatches')
    bindings=result.get('inputs',{})
    for name in ['receipt.json','inputs.json','rollout.npz','contacts.jsonl',
                 'task_contacts_substeps.jsonl','executed_scene.xml']:
        source=(run/name).resolve()
        if bindings.get(str(source))!=sha(source):
            raise ValueError('Evaluation lacks this exact raw artifact: '+name)
    checkers=[p for p in bindings if Path(p).name=='g1_pickplace_physics_verify.py']
    if len(checkers)!=1 or bindings[checkers[0]]!=sha(evaluator):
        raise ValueError('Evaluation is not bound to the supplied evaluator')
    for old,h in bindings.items():
        if sha(old)!=h:raise ValueError('Changed evaluation input: '+old)
    for rel,h in result.get('artifacts',{}).items():
        source=(path.parent/rel).resolve()
        if not source.is_relative_to(path.parent.resolve()) or sha(source)!=h:
            raise ValueError('Changed or escaping evaluation artifact: '+rel)
    if not {'measurements.npz','task_contact_measurements.npz','unexpected_contacts.json'}<=set(result.get('artifacts',{})):
        raise ValueError('Missing physical evaluation measurements')
    # Hashes alone do not establish that gate flags were computed from the raw
    # states. Re-run the pinned checker and require its complete result, rather
    # than maintaining a second, easily divergent copy of its threshold logic.
    with tempfile.TemporaryDirectory(prefix='g1-stage-evaluation-') as temporary:
        fresh=Path(temporary)/'evaluation'
        command=[str(python or sys.executable),str(evaluator.resolve()),
                 '--rollout',str(run.resolve()),'--output',str(fresh)]
        subprocess.run(command,check=True,timeout=180,capture_output=True,text=True)
        reproduced=read(fresh/'receipt.json')
        def normalized(value):
            value=dict(value);value.pop('produced_at',None)
            value['inputs']={('pinned_evaluator' if Path(k).name=='g1_pickplace_physics_verify.py' else k):v
                             for k,v in value['inputs'].items()}
            return value
        if normalized(result)!=normalized(reproduced):
            raise ValueError('Evaluation disagrees with a fresh execution of the pinned checker')
        for rel,h in reproduced['artifacts'].items():
            if sha(fresh/rel)!=h:raise ValueError('Recomputed measurement digest differs: '+rel)
        result['_staging_recheck']=reproduced
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runtime-template',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--jobctl',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--claim',required=True)
    p.add_argument('--evaluation',type=Path)
    p.add_argument('--evaluator',type=Path)
    p.add_argument('--evaluation-python',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',a.case):raise ValueError('Invalid case name')
    run=a.rollout.resolve();ref=a.reference.resolve();out=a.output.resolve()
    if bool(a.evaluation)!=bool(a.evaluator):
        raise ValueError('Supply both --evaluation and --evaluator')
    evaluation=validate_evaluation(run,a.evaluation.resolve(),a.evaluator.resolve(),a.evaluation_python) if a.evaluation else None
    physical_success=evaluation['complete_physical_task_passed'] if evaluation else False
    if ref.parent!=out.parent:raise ValueError('Reference and physics must be adjacent components')
    out.mkdir(parents=True,exist_ok=False)
    now=datetime.now(timezone.utc).isoformat()
    ins=read(run/'inputs.json');raw=read(run/'receipt.json');args=ins['arguments']
    assert args['controller']=='scalebfm'
    assert sha(run/'inputs.json')==raw['inputs_receipt']['sha256']
    refm=read(ref/'manifest.json');refr=read(ref/'stage.receipt.json')
    assert refr['manifest_sha256']==sha(ref/'manifest.json')
    declared={row['path']:row for row in refm['files']}
    for rel,row in declared.items():
        source=(ref/rel).resolve()
        assert source.is_relative_to(ref) and sha(source)==row['sha256'] and source.stat().st_size==row['size'],rel
    prefix=Path('..')/ref.name
    aliases={}
    for old,row in read(ref/'aliases.json')['paths'].items():
        source=(ref/row['staged_path']).resolve()
        assert source.is_relative_to(ref) and sha(source)==row['sha256'],old
        assert str(source.relative_to(ref)) in declared,old
        aliases[old]={'staged_path':str(prefix/row['staged_path']),'sha256':row['sha256']}
    case_root=out/'cases'/a.case
    for rel,h in raw['artifacts'].items():
        source=(run/rel).resolve()
        assert source.is_relative_to(run) and sha(source)==h,rel
        copy(source,case_root/'raw'/rel)
        aliases[str(source)]={'staged_path':str((case_root/'raw'/rel).relative_to(out)),'sha256':h}
    copy(run/'receipt.json',case_root/'raw/receipt.json')
    aliases[str(run/'receipt.json')]={'staged_path':str((case_root/'raw/receipt.json').relative_to(out)),
        'sha256':sha(run/'receipt.json')}
    models=Path(args['models']).resolve()
    input_aliases={}
    bindings={**ins['inputs'],**ins['model_dependencies']}
    scripts=[k for k in ins['inputs'] if Path(k).name=='sonic_rollout.py']
    assert len(scripts)==1
    for old,h in bindings.items():
        source=Path(old).resolve();assert sha(source)==h,old
        if old in aliases and aliases[old]['sha256']==h:
            rel=Path(aliases[old]['staged_path'])
        else:
            if source.is_relative_to(models):rel=Path('models/scalebfm')/source.relative_to(models)
            else:rel=Path('cases')/a.case/'executed'/h[:16]/source.name
            copy(source,out/rel)
            aliases[old]={'staged_path':str(rel),'sha256':h}
        assert sha(out/rel)==h
        input_aliases[old]=str(rel)
    # Preserve the pinned model loader's relative resource tree, even if a
    # matching digest happened to exist in the sibling reference.
    for old,h in ins['inputs'].items():
        source=Path(old).resolve()
        if source.is_relative_to(models):
            rel=Path('models/scalebfm')/source.relative_to(models)
            copy(source,out/rel);input_aliases[old]=str(rel)
            aliases[old]={'staged_path':str(rel),'sha256':h}
    scene_receipt=read(ref/'native_scene/scene.receipt.json')
    assert sha(ref/'native_scene/scene.xml')==scene_receipt['output']['sha256']==sha(args['scene'])
    scene_task_path=prefix/'inputs/scene_task.original.json'
    assert sha(out/scene_task_path)==scene_receipt['task']['sha256']
    evaluation_binding=None
    if evaluation:
        evaluation_root=case_root/'evaluation'
        copy(a.evaluation,evaluation_root/'receipt.json')
        save(evaluation_root/'recomputed.receipt.json',evaluation['_staging_recheck'])
        for rel in evaluation['artifacts']:copy(a.evaluation.parent/rel,evaluation_root/rel)
        evaluation_aliases={}
        for old,h in {**evaluation['inputs'],**evaluation['_staging_recheck']['inputs']}.items():
            if old in aliases and aliases[old]['sha256']==h:
                rel=Path(aliases[old]['staged_path'])
            else:
                rel=evaluation_root.relative_to(out)/'inputs'/h[:16]/Path(old).name
                copy(old,out/rel);aliases[old]={'staged_path':str(rel),'sha256':h}
            evaluation_aliases[old]=str(rel)
        evaluation_binding={'path':str((evaluation_root/'receipt.json').relative_to(out)),
            'sha256':sha(a.evaluation),'evaluator_sha256':sha(a.evaluator),
            'recomputed_receipt_path':str((evaluation_root/'recomputed.receipt.json').relative_to(out)),
            'recomputed_receipt_sha256':sha(evaluation_root/'recomputed.receipt.json'),
            'input_aliases':evaluation_aliases,'complete_physical_task_passed':physical_success,
            'scope':'Bound nominal simulation evaluation; independent review and hardware validity are separate.'}
    descriptor={'schema':'g1-physical-staged-case/v2','id':a.case,'historical_name':run.name,
        'claim':a.claim,'physical_pickplace_success':physical_success,
        'physical_evaluation':evaluation_binding,
        'raw':str(Path('cases')/a.case/'raw'),'historical_inputs_sha256':sha(run/'inputs.json'),
        'historical_receipt_sha256':sha(run/'receipt.json'),'arguments':args,
        'input_aliases':input_aliases,'script':input_aliases[scripts[0]],
        'scene_task_source':str(scene_task_path),'default_timeout_seconds':450}
    descriptor_path=case_root/'case.json';save(descriptor_path,descriptor)
    save(out/'cases.json',{'schema':'g1-physical-cases/v2','default_case':a.case,
        'cases':{a.case:{'path':str(descriptor_path.relative_to(out)),
            'sha256':sha(descriptor_path),'claim':a.claim}},
        'scope':'Exact archived trial; physical success is determined by a separate bound evaluator.'})
    save(out/'aliases.json',{'schema':'g1-physical-stage-aliases/v2','paths':aliases,
        'scope':'Exact historical file bytes mapped to paths inside these two components.'})
    copy(a.helper,out/'code/physics.py');copy(a.jobctl,out/'code/jobctl.py')
    copy(Path(__file__),out/'code/g1_physics_stage.py')
    copy(a.runtime_template/'runtime.json',out/'runtime.json')
    copy(a.runtime_template/'requirements-runtime.txt',out/'requirements.txt')
    # Include the pinned source/model notices as well as inference resources.
    # Any same-path resource already copied from the selected trial must match.
    model_template=a.runtime_template/'models'
    for source in model_template.rglob('*'):
        if source.is_file():copy(source,out/'models'/source.relative_to(model_template))
    notices=a.runtime_template/'source_notices'
    for source in notices.rglob('*'):
        if not source.is_file():continue
        destination=out/'source_notices'/source.relative_to(notices)
        if source.name=='README.md':
            copy(source,destination.with_name('historical_template_README.md'))
            destination.write_text(source.read_text().replace('../../reference_stage01/', '../../'+ref.name+'/').replace('../requirements-runtime.txt','../requirements.txt'))
        else:copy(source,destination)
    (out/'README.md').write_text('# Physical trial reproduction\n\n'+a.claim+'\n\n'
        'The sibling `'+ref.name+'` is required. Keep both directories beside each other. '
        'The checkpoint is included; install the pinned Python/runtime dependencies first.\n\n'
        '```sh\npython code/physics.py verify\npython code/physics.py runtime\n'
        'python code/physics.py run --name smoke --duration 0.04\n'
        'python code/physics.py status --name smoke\npython code/physics.py logs --name smoke\n'
        'python code/physics.py run --name full\n```\n\n'
        'Use `stop` or `restart` with the same `--name`. Runs are bounded and write under '
        '`reproduced/`. Replay comparisons measure only the actual reproduced interval. '
        'Physical task completion and hardware robustness are separate claims.\n')
    files=[{'path':str(f.relative_to(out)),'size':f.stat().st_size,'sha256':sha(f)}
           for f in sorted(out.rglob('*')) if f.is_file()]
    dep={'path':str(prefix),'manifest_sha256':sha(ref/'manifest.json'),
         'receipt_sha256':sha(ref/'stage.receipt.json')}
    save(out/'manifest.json',{'schema':'g1-physical-stage-manifest/v2','component':out.name,
        'produced_at':now,'physical_pickplace_success':physical_success,'reference_component':dep,
        'case_registry':'cases.json','files':files,
        'excluded':['reproduced/**','__pycache__/**','manifest.json','stage.receipt.json'],
        'scope':'Immutable historical inputs and raw trial; portability smoke test pending.'})
    save(out/'stage.receipt.json',{'schema':'g1-physical-stage-receipt/v2',
        'produced_at':now,'status':'staged_pending_portable_execution',
        'manifest_sha256':sha(out/'manifest.json'),'file_count':len(files),
        'bytes':sum(f['size'] for f in files),'reference_component':dep,
        'physical_pickplace_success':physical_success,'physical_evaluation':evaluation_binding,
        'portability_verified':False})
    print(json.dumps({'output':str(out),'files':len(files),'bytes':sum(f['size'] for f in files),
        'manifest_sha256':sha(out/'manifest.json')}))


if __name__=='__main__':main()
