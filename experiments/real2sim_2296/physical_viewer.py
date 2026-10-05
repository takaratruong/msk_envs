#!/usr/bin/env python3
"""Interactive Isaac mesh scene; controls edit only the anonymous session layer."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
p.add_argument('--expected-manifest-sha256',required=True)
p.add_argument('--camera',default='/World/Cameras/frame_001499');args=p.parse_args()
args.scene=args.scene.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
assert sha(args.scene)==args.expected_sha256,'Scene differs from the reviewed input'
manifest_path=args.scene.with_suffix('.manifest.json')
assert sha(manifest_path)==args.expected_manifest_sha256,'Manifest differs from the reviewed input'
manifest=json.loads(manifest_path.read_text())
assert manifest['scene_sha256']==args.expected_sha256,'Manifest belongs to a different scene'
def status(state,**extra):
    x=dict(state=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
           scene=str(args.scene),scene_sha256=sha(args.scene),manifest_sha256=sha(manifest_path),script_sha256=sha(__file__),
           display=os.environ.get('DISPLAY'),source_unchanged=sha(args.scene)==args.expected_sha256,
           manifest_unchanged=sha(manifest_path)==args.expected_manifest_sha256,**extra)
    tmp=args.output/'status.tmp.json';tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(args.output/'status.json')
    print(json.dumps(x),flush=True)
status('starting');sys.argv=[sys.argv[0]]
from isaacsim import SimulationApp
app=SimulationApp(dict(headless=False,hide_ui=False,active_gpu=0,physics_gpu=0,multi_gpu=False,
    width=1280,height=720,window_width=1500,window_height=900,renderer='RaytracedLighting',
    anti_aliasing=0,sync_loads=True,create_new_stage=False,fast_shutdown=True,enable_crashreporter=False,
    extra_args=['--portable-root',str(args.output/'kit-portable'),'--/app/settings/persistent=false',
      '--/app/telemetry/enabled=false','--/app/window/title=IMG_2296 - Textured articulated model',
      '--/app/runLoops/main/rateLimitEnabled=true','--/app/runLoops/main/rateLimitFrequency=30',
      '--/log/file='+str(args.output/'isaac.kit.log')]))
try:
    import carb.settings,omni.usd,omni.timeline,omni.ui as ui
    import omni.replicator.core as rep
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd,UsdPhysics,UsdGeom,Gf
    project=Path('/home/ubuntu/msk_envs-stone-course')
    sys.path.insert(0,str(project/'experiments/real2sim_2296'))
    from isaac_render import render_settings
    ctx=omni.usd.get_context()
    if not ctx.open_stage(str(args.scene)):raise RuntimeError('Could not open physical scene')
    for _ in range(12):app.update()
    stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
    for layer in stage.GetUsedLayers():
        if not layer.anonymous:layer.SetPermissionToSave(False)
    timeline=omni.timeline.get_timeline_interface();timeline.stop();rep.orchestrator.set_capture_on_play(False)
    app.reset_render_settings();settings=carb.settings.get_settings();requested,_=render_settings(stage)
    for key,value in requested.items():settings.set('/'+key.replace(':','/').lstrip('/'),value)
    settings.set('/app/viewport/grid/enabled',False)
    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('Interactive viewport unavailable')
    if not stage.GetPrimAtPath(args.camera).IsA(UsdGeom.Camera):raise ValueError('Requested camera is missing')
    viewport.camera_path=args.camera;viewport.set_texture_resolution((1888,1061))
    joints=[j for j in manifest['joints'] if not j.get('passive')]
    assets_by_name={a['name']:a for a in manifest['assets']}
    for j in joints:
        prim=stage.GetPrimAtPath(j['path'])
        if not prim or not prim.IsA(UsdPhysics.Joint):raise ValueError('Manifest joint missing: '+j['path'])
    source_bodies={str(p.GetPath()):[(op.GetOpName(),op.Get()) for op in UsdGeom.Xformable(p).GetOrderedXformOps()]
                   for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    choices=[j['asset']+' / '+j['path'].rsplit('/',1)[1]+(' [retained]' if j.get('retainer_locked') else '') for j in joints]
    initial_choice=next((i for i,j in enumerate(joints) if j['asset']=='refrigerator_01' and j['path'].endswith('/upper_door_0_hinge')),0)
    current={'index':initial_choice,'mode':'position drives','reset_pending':False}
    def selected(model,item=None):current['index']=model.get_item_value_model().as_int
    def set_target(opening):
        j=joints[current['index']];prim=stage.GetPrimAtPath(j['path'])
        d=UsdPhysics.DriveAPI(prim,'angular' if j['type']=='revolute' else 'linear')
        q=max((j['lower'],j['upper']),key=abs) if opening else j['initial']
        if opening:
            appliance=assets_by_name[j['asset']]['metadata'].get('asset_type') in ('dishwasher','range_oven')
            # Match the ordinary native demonstration cycle. Full hardware
            # bounds remain independently authored and available to simulation.
            q*=1. if appliance and j['type']=='revolute' else .8
        with Usd.EditContext(stage,stage.GetSessionLayer()):d.GetTargetPositionAttr().Set(float(q))
        timeline.play()
    def reset_source():
        timeline.stop();current['reset_pending']=True
    def restore_source_state():
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for path,ops in source_bodies.items():
                prim=stage.GetPrimAtPath(path)
                for name,value in ops:prim.GetAttribute(name).Set(value)
                rigid=UsdPhysics.RigidBodyAPI(prim)
                rigid.CreateVelocityAttr().Set(Gf.Vec3f(0.));rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.))
            for j in manifest['joints']:
                prim=stage.GetPrimAtPath(j['path']);kind='angular' if j['type']=='revolute' else 'linear'
                UsdPhysics.DriveAPI(prim,kind).GetTargetPositionAttr().Set(float(j['initial']))
                prim.GetAttribute('state:'+kind+':physics:position').Set(float(j['initial']))
                prim.GetAttribute('state:'+kind+':physics:velocity').Set(0.)
        current['reset_pending']=False
    def passive():
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for j in joints:
                if j.get('retainer_locked'):continue
                kind='angular' if j['type']=='revolute' else 'linear'
                UsdPhysics.DriveAPI(stage.GetPrimAtPath(j['path']),kind).GetStiffnessAttr().Set(0.)
        current['mode']='passive manipulation';timeline.play()
    def restore_drives():
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            for j in joints:
                kind='angular' if j['type']=='revolute' else 'linear'
                UsdPhysics.DriveAPI(stage.GetPrimAtPath(j['path']),kind).GetStiffnessAttr().Set(80. if kind=='angular' else 500.)
        current['mode']='position drives'
    window=ui.Window('IMG_2296 scene controls',width=410,height=385)
    with window.frame:
        with ui.VStack(spacing=7):
            ui.Label('Textured meshes and physical joints',height=24)
            ui.Label('Modeled dimensions and mechanisms; real-world accuracy is unverified.',word_wrap=True,height=42)
            with ui.HStack(height=28):
                ui.Button('Play physics',clicked_fn=timeline.play);ui.Button('Pause',clicked_fn=timeline.pause)
                ui.Button('Source pose',clicked_fn=reset_source)
            ui.Label('Select a modeled joint',height=18)
            combo=ui.ComboBox(initial_choice,*choices,height=28);combo.model.add_item_changed_fn(selected)
            with ui.HStack(height=28):
                ui.Button('Open selected',clicked_fn=lambda:set_target(True))
                ui.Button('Return selected',clicked_fn=lambda:set_target(False))
            with ui.HStack(height=28):
                ui.Button('Passive hand interaction',clicked_fn=passive)
                ui.Button('Restore position drives',clicked_fn=restore_drives)
            with ui.HStack(height=28):
                for label,frame in [('Kitchen','001499'),('Island reverse','002158'),('Tables','000000')]:
                    ui.Button(label,clicked_fn=lambda f=frame:setattr(viewport,'camera_path','/World/Cameras/frame_'+f))
            with ui.HStack(height=28):
                ui.Button('Path traced lighting',clicked_fn=lambda:settings.set('/rtx/rendermode','PathTracing'))
                ui.Button('Fast navigation',clicked_fn=lambda:settings.set('/rtx/rendermode','RaytracedLighting'))
            ui.Label('Drag the view to navigate. Retained doors stay locked. All edits are temporary.',word_wrap=True,height=38)
    for _ in range(60):app.update()
    status('open',camera=str(viewport.camera_path),playing=timeline.is_playing(),
           edit_target=stage.GetEditTarget().GetLayer().identifier,assets=len(manifest['assets']),
           joints=len(manifest['joints']),gaussian_volumes=sum(p.GetTypeName()=='OmniNuRecVolume' for p in stage.Traverse()),
           gpu=settings.get('/renderer/activeGpu'))
    last=time.monotonic()
    while app.is_running():
        app.update()
        if current['reset_pending']:restore_source_state()
        if time.monotonic()-last>30:
            status('open',camera=str(viewport.camera_path),playing=timeline.is_playing(),mode=current['mode']);last=time.monotonic()
    status('closed')
except BaseException as error:
    status('error',error=repr(error));raise
finally:app.close()
