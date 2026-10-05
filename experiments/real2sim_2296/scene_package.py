#!/usr/bin/env python3
"""Compose one OpenUSD preview with explicit geometry/physics evidence limits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,Vt


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_inputs(root,app):
    data=json.loads((root/'dataset.json').read_text())
    result=json.loads((app/'result.json').read_text())
    inputs=json.loads((app/'input_manifest.json').read_text())
    body={k:v for k,v in inputs.items() if k!='inputs_sha256'}
    if hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()!=inputs['inputs_sha256']:
        raise RuntimeError('Appearance input manifest content hash mismatch.')
    if result['inputs_sha256']!=inputs['inputs_sha256']:
        raise RuntimeError('Appearance receipt mismatch.')
    if inputs['dataset_sha256']!=sha(root/'dataset.json') or inputs['source_sha256']!=data['source_sha256']:
        raise RuntimeError('Dataset or source differs from appearance training.')
    if sha(Path(inputs['points_path']))!=inputs['points_sha256']:
        raise RuntimeError('Initial points changed after training.')
    for item in inputs['images']:
        if sha(Path(item['image_path']))!=item['sha256']:
            raise RuntimeError('Training/evaluation image changed: '+item['name'])
        if item.get('mask_path') and sha(Path(item['mask_path']))!=item['mask_sha256']:
            raise RuntimeError('Training/evaluation mask changed: '+item['name'])
    if data.get('model_artifacts'):
        for name,expected in data['model_artifacts'].items():
            if sha(Path(data['model_path'])/name)!=expected:
                raise RuntimeError('Camera model changed after preparation: '+name)
    recorded={Path(x['path']).name:x for x in result['artifacts']}
    if 'appearance.ply' not in recorded or sha(app/'appearance.ply')!=recorded['appearance.ply']['sha256']:
        raise RuntimeError('Trained appearance PLY does not match its result receipt.')
    export=json.loads((app/'appearance.usdz.receipt.json').read_text())
    if export['source_ply']['sha256']!=sha(app/'appearance.ply'):
        raise RuntimeError('USDZ was exported from a different PLY.')
    if export['output']['sha256']!=sha(app/'appearance.usdz'):
        raise RuntimeError('USDZ bytes do not match exporter receipt.')
    return data,result


def camera_at(stage,path,view):
    cam=UsdGeom.Camera.Define(stage,path)
    K=np.array(view['K']);w,h=view['width'],view['height']
    aperture=36.
    cam.CreateHorizontalApertureAttr(aperture)
    focal=float(K[0,0]/w*aperture)
    vertical=focal*h/K[1,1]
    cam.CreateVerticalApertureAttr(float(vertical))
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureOffsetAttr(float((w/2-K[0,2])/w*aperture))
    cam.CreateVerticalApertureOffsetAttr(float((K[1,2]-h/2)/h*vertical))
    cam.CreateClippingRangeAttr(Gf.Vec2f(.001,10000))
    c2w=np.linalg.inv(np.array(view['w2c']))@np.diag([1,-1,-1,1])
    cam.AddTransformOp().Set(Gf.Matrix4d(c2w.T.tolist()))
    cam.GetPrim().SetCustomDataByKey('sourceFrame',view['name'])
    cam.GetPrim().SetCustomDataByKey('sourceTimeSeconds',view.get('time_seconds',0.))
    cam.GetPrim().SetCustomDataByKey('imageWidth',int(w))
    cam.GetPrim().SetCustomDataByKey('imageHeight',int(h))
    return cam


def scene_frame(root,data):
    """Rigidly level the source-observed floor, preserving the unknown scale."""
    model=Path(data['model_path'])
    metrics=json.loads((root/(model.name+'_metrics.json')).read_text())
    floor=metrics.get('floor_filter')
    if not floor:return np.eye(4),None
    if metrics['model_artifacts']!=data['model_artifacts']:
        raise RuntimeError('Floor transform and prepared camera model differ.')
    if sha(Path(floor['evidence_path']))!=floor['evidence_sha256']:
        raise RuntimeError('Floor evidence changed.')
    n=np.array(floor['plane_normal']);offset=floor['plane_offset']
    centers=np.array([np.linalg.inv(np.array(v['w2c']))[:3,3] for v in data['train']])
    origin=np.median(centers,axis=0);origin-=n*(origin@n+offset)
    up=-n
    first=min(data['train'],key=lambda v:v['time_seconds'])
    right=np.linalg.inv(np.array(first['w2c']))[:3,0]
    right-=up*(right@up);right/=np.linalg.norm(right)
    forward=np.cross(up,right)
    transform=np.eye(4);transform[:3,:3]=np.stack([right,forward,up])
    transform[:3,3]=-transform[:3,:3]@origin
    if not np.allclose(transform[:3,:3]@transform[:3,:3].T,np.eye(3),atol=1e-8) or np.linalg.det(transform[:3,:3])<.999999:
        raise RuntimeError('Invalid floor rigid transform.')
    return transform,floor


def inventory_records(root,data):
    directory=root/'semantic_inventory'
    inventory=json.loads((directory/'objects.json').read_text())
    receipt=json.loads((directory/'verification.json').read_text())
    if (inventory['source_sha256']!=data['source_sha256'] or
        inventory['frames_manifest_sha256']!=sha(root/'frames.json') or
        inventory['sparse_model']['binary_hashes']!=data['model_artifacts'] or
        receipt['objects_sha256']!=sha(directory/'objects.json')):
        raise RuntimeError('Semantic inventory does not match current source/cameras.')
    frames={v['name']:v for v in json.loads((root/'frames.json').read_text())['frames']}
    identities={v['instance_id'] for v in inventory['objects']}
    if len(identities)!=len(inventory['objects']):raise RuntimeError('Repeated semantic instance ID.')
    for obj in inventory['objects']:
        if obj['parent_instance_id'] and obj['parent_instance_id'] not in identities:
            raise RuntimeError('Unresolved semantic parent.')
        for observation in obj['observations']:
            frame=frames[observation['frame']]
            if (observation['image_sha256']!=frame['sha256'] or
                abs(observation['source_pts_seconds']-frame['time_seconds'])>1e-6 or
                sha(Path(observation['image_path']))!=observation['image_sha256'] or
                sha(directory/observation['review_crop'])!=observation['review_crop_sha256']):
                raise RuntimeError('Semantic observation/source/crop mismatch.')
    for media in inventory['annotated_frame_media']:
        if sha(directory/media['path'])!=media['sha256']:
            raise RuntimeError('Annotated semantic frame changed.')
    return inventory


def native_evidence(directory,out,data):
    receipt=json.loads((directory/'evidence.json').read_text())
    inputs=receipt['inputs']
    if hashlib.sha256(json.dumps(inputs,sort_keys=True,allow_nan=False).encode()).hexdigest()!=receipt['inputs_sha256']:
        raise RuntimeError('Native evidence input digest mismatch.')
    if inputs['source_sha256']!=data['source_sha256'] or inputs['stage_sha256']!=sha(out/'scene.usda'):
        raise RuntimeError('Native evidence belongs to a different source or scene.')
    packaged_geometry=json.loads((out/'assets'/'observed_surface_metrics.json').read_text())
    if inputs['dataset_sha256']!=packaged_geometry['dataset_sha256']:
        raise RuntimeError('Native evidence uses a different camera dataset.')
    render_files={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and
                  (p==out/'scene.usda' or p.relative_to(out).parts[0] in ('assets','semantics'))}
    if not render_files.issubset(inputs['render_payloads']):
        raise RuntimeError('Native evidence omits current render or semantic files.')
    for name in render_files:
        if sha(out/name)!=inputs['render_payloads'][name]:
            raise RuntimeError('Native-tested payload differs: '+name)
    files=receipt['artifacts']
    for required in ('index.html','comparisons.json','process_audit.json','scene_binding.json'):
        if required not in files:raise RuntimeError('Native evidence omits '+required)
    target=(out/'verification'/'native').resolve()
    if not target.is_relative_to(out.resolve()):raise RuntimeError('Native evidence target leaves the scene directory.')
    for name in [*files,'evidence.json']:
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Native artifact must be a clean relative path: '+name)
        if not (target/name).resolve().is_relative_to(target):
            raise RuntimeError('Native artifact target leaves the evidence directory: '+name)
        source=(directory/name).resolve()
        if not source.is_relative_to(directory) or not source.is_file() or (name in files and sha(source)!=files[name]):
            raise RuntimeError('Native artifact path/hash mismatch: '+name)
    process=json.loads((directory/'process_audit.json').read_text())
    for name in ('native_train','native_buffered'):
        job=process['jobs'][name]
        if job['state']['status']!='succeeded' or job['state']['returncode']!=0 or job['owned_processes']:
            raise RuntimeError('Native capture did not finish successfully: '+name)
    comparisons=json.loads((directory/'comparisons.json').read_text())
    views=comparisons['views']
    if len(views)!=2 or {v['frame'] for v in views}!={'frame_001499.jpg','frame_001451.jpg'}:
        raise RuntimeError('Both declared native camera checks are required.')
    for view in views:
        if view['scene_sha256']!=inputs['stage_sha256'] or not view['hidden_control_all_black'] or view['changed_pixels_gt8']<1000:
            raise RuntimeError('Native view/control evidence mismatch.')
    return receipt,files


def finalize_evidence(root,app,out,creation_attempt,native_directory=None):
    """Add portable provenance without changing the already-rendered USD layers."""
    data,result=verify_inputs(root,app)
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    before=sha(manifest_path)
    for name,expected in manifest['artifacts'].items():
        if sha(out/name)!=expected:raise RuntimeError('Existing scene artifact changed: '+name)
    if manifest['source_sha256']!=data['source_sha256']:
        raise RuntimeError('Scene and supplied dataset source differ.')
    for name in ('appearance.ply','appearance.usdz'):
        if sha(app/name)!=sha(out/'assets'/name):
            raise RuntimeError('Supplied appearance differs from the packaged asset: '+name)
    dataset_hash=sha(root/'dataset.json')
    if manifest.get('dataset_sha256',dataset_hash)!=dataset_hash:
        raise RuntimeError('Existing scene dataset binding differs.')
    for layer in manifest['geometry_layers']:
        name='inferred_surface' if layer['inferred_prior'] else 'observed_surface'
        bound_geometry=json.loads((out/'assets'/(name+'_metrics.json')).read_text())
        if bound_geometry.get('dataset_sha256')!=dataset_hash:
            raise RuntimeError('Packaged surface was built from a different dataset.')
    selection=json.loads((root/'appearance_selection.json').read_text())
    selected=[v for v in selection['candidates'] if v['name']==selection['selected']]
    if (root/selection['selected']).resolve()!=app or len(selected)!=1 or selected[0]['result_sha256']!=sha(app/'result.json'):
        raise RuntimeError('Selection receipt does not select the supplied appearance result.')
    job=json.loads((creation_attempt/'status.json').read_text())
    creator=creation_attempt/'scene_package.py'
    if job.get('status')!='succeeded' or sha(creator) not in job.get('script_sha256',{}).values():
        raise RuntimeError('Successful creation job and archived implementation are required.')
    creation_output=json.loads((creation_attempt/'log.txt').read_text())
    if Path(creation_output.get('scene','')).resolve()!=out/'scene.usda' or creation_output.get('sha256')!=sha(out/'scene.usda'):
        raise RuntimeError('Creation job output path/hash does not match this scene.')
    native_receipt,native_files=native_evidence(native_directory,out,data) if native_directory else (None,{})
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    sources={'dataset.json':root/'dataset.json','frames.json':root/'frames.json',
             'appearance_input_manifest.json':app/'input_manifest.json',
             'appearance_result.json':app/'result.json','appearance_selection.json':root/'appearance_selection.json',
             'camera_metrics.json':root/(Path(data['model_path']).name+'_metrics.json'),
             'localization.json':root/'evaluation'/'localization.json',
             'dense_provenance.json':root/'dense_provenance.json',
             'creation_job.json':creation_attempt/'status.json','creation_config.json':creation_attempt/'config.json',
             'creation_output.json':creation_attempt/'log.txt',
             'creation_scene_package.py':creator}
    for name,source in sources.items():shutil.copy2(source,evidence/name)
    shutil.copy2(__file__,evidence/'finalize_scene_package.py')
    if native_receipt:
        target=out/'verification'/'native'
        for name in [*native_files,'evidence.json']:
            destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(native_directory/name,destination)
        manifest.update(native_isaac_render_verified=True,
            native_render_evidence='verification/native/evidence.json',
            native_render_evidence_sha256=sha(target/'evidence.json'),
            native_render_scope='Two fixed source cameras in Isaac Sim 5.1: rendering, pose/export consistency and hidden-volume controls only; image-quality and physical acceptance remain unmet or unverified.')
    manifest.update(dataset_sha256=sha(root/'dataset.json'),
        appearance_input_manifest_sha256=sha(app/'input_manifest.json'),
        appearance_result_sha256=sha(app/'result.json'),
        appearance_selection_sha256=sha(root/'appearance_selection.json'),
        creation_implementation_sha256=sha(creator),finalization_implementation_sha256=sha(__file__),
        portable_provenance='evidence/',
        provenance_scope='Copied manifests retain original capture/training paths for provenance. Rendering uses relative packaged assets; full training inputs remain in the original run.',
        finalized_at=datetime.now(timezone.utc).isoformat())
    manifest['artifacts']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'schema':'agentic-evidence/v1','run_id':root.name+'-scene-evidence',
        'claim':'Portable reconstruction provenance added; existing canonical USD layers and assets unchanged. No geometry or deployment acceptance.',
        'produced_at':manifest['finalized_at'],'inputs_sha256':before,
        'path':str(manifest_path),'sha256':sha(manifest_path),
        'scene_sha256':sha(out/'scene.usda'),'creation_implementation_sha256':sha(creator)}
    (root/'scene_finalization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--appearance',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--include-inferred',action='store_true',help='Include the separately audited learned-depth surface hypothesis.')
    p.add_argument('--finalize-evidence',action='store_true',help='Bind portable receipts without modifying existing USD scene layers.')
    p.add_argument('--creation-attempt',type=Path,help='Successful owned package job containing its archived implementation.')
    p.add_argument('--native-evidence',type=Path,help='Completed native-render evidence directory to copy into an existing package.')
    a=p.parse_args();root=a.run.resolve();app=a.appearance.resolve();out=(a.output or root/'scene').resolve()
    if a.finalize_evidence:
        if not a.creation_attempt:p.error('--creation-attempt is required with --finalize-evidence')
        finalize_evidence(root,app,out,a.creation_attempt.resolve(),a.native_evidence.resolve() if a.native_evidence else None);return
    if (out/'scene.usda').exists():raise RuntimeError('Scene exists: use a new output directory for a revised package.')
    data,result=verify_inputs(root,app)
    transform,floor=scene_frame(root,data)
    inventory=inventory_records(root,data)
    out.mkdir(parents=True,exist_ok=True);(out/'assets').mkdir(exist_ok=True)
    for source in [app/'appearance.ply',app/'appearance.usdz',app/'appearance.usdz.receipt.json']:
        if source.exists():shutil.copy2(source,out/'assets'/source.name)
    stage=Usd.Stage.CreateNew(str(out/'scene.usda'))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    frame=UsdGeom.Xform.Define(stage,'/World/ReconstructionFrame')
    frame.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    if floor:shutil.copy2(floor['evidence_path'],out/'assets'/'floor_evidence.json')
    stage.GetRootLayer().customLayerData={'real2sim':{
        'sourceSHA256':data['source_sha256'],'status':'UNVERIFIED_RECONSTRUCTION_PREVIEW',
        'sourceUnits':'arbitrary','metricScaleVerified':False,'gravityAlignmentVerified':False,
        'floorAlignmentEstimatedFromImages':floor is not None,
        'metresPerUnitIsEncodingConventionOnly':True,'physicsValidated':False,
        'description':'One common unscaled camera/appearance/surface frame, leveled using visible floor evidence when available. No real-world deployment acceptance.'}}
    if not (out/'assets'/'appearance.usdz').exists():
        raise RuntimeError('Native NuRec appearance package is required; a PLY is not an Isaac splat renderer.')
    native=Usd.Stage.Open(str(out/'assets'/'appearance.usdz'))
    settings=native.GetRootLayer().customLayerData.get('renderSettings',{})
    layer_data=dict(stage.GetRootLayer().customLayerData)
    layer_data['renderSettings']=settings
    stage.GetRootLayer().customLayerData=layer_data
    visual=stage.DefinePrim('/World/ReconstructionFrame/Appearance','Xform')
    visual.GetReferences().AddReference('./assets/appearance.usdz','/World')
    visual.SetCustomDataByKey('role','RGB-fitted appearance; not a collider or physical depth map')
    geometry_layers=[]
    sources=[('ObservedSurface','observed_surface',root/'geometry',False)]
    if a.include_inferred:sources.append(('InferredSurface','inferred_surface',root/'geometry_prior',True))
    for prim_name,asset_name,geom_dir,inferred in sources:
        geometry=geom_dir/'surface_arrays.npz'
        if not geometry.exists():raise RuntimeError('Required scene surface is missing: '+str(geometry))
        geom_receipt=json.loads((geom_dir/'metrics.json').read_text())
        if (geom_receipt['source_sha256']!=data['source_sha256'] or
            geom_receipt['artifacts']['surface_arrays.npz']!=sha(geometry) or
            geom_receipt.get('dataset_sha256')!=sha(root/'dataset.json')):
            raise RuntimeError('Geometry receipt differs from current geometry/dataset.')
        if inferred and not geom_receipt.get('inferred_prior'):
            raise RuntimeError('Inferred surface lacks its explicit prior declaration.')
        arr=np.load(geometry)
        meshstage=Usd.Stage.CreateNew(str(out/'assets'/(asset_name+'.usdc')))
        mesh=UsdGeom.Mesh.Define(meshstage,'/'+prim_name);meshstage.SetDefaultPrim(mesh.GetPrim())
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(arr['vertices']))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(arr['triangles']),3,dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(arr['triangles'].reshape(-1)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none);mesh.CreateDoubleSidedAttr(True)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(arr['normals']));mesh.SetNormalsInterpolation('vertex')
        if len(arr['colors'])==len(arr['vertices']):
            # PLY colors are image-encoded sRGB; USD displayColor is linear.
            c=arr['colors'];linear=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4).astype(np.float32)
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(linear))
        meshstage.GetRootLayer().Save()
        diagnostic=stage.DefinePrim('/World/ReconstructionFrame/'+prim_name,'Mesh')
        diagnostic.GetReferences().AddReference('./assets/'+asset_name+'.usdc')
        UsdGeom.Imageable(diagnostic).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collision=UsdPhysics.CollisionAPI.Apply(diagnostic)
        collision.CreateCollisionEnabledAttr(False)
        diagnostic.SetCustomDataByKey('role','Learned-depth TSDF hypothesis; not measured or validated collision geometry' if inferred else 'RGB multiview surface estimate; incomplete collision geometry')
        diagnostic.SetCustomDataByKey('learnedDepthPrior',inferred)
        diagnostic.SetCustomDataByKey('metricScaleVerified',False)
        diagnostic.SetCustomDataByKey('localGapBridgingPossible',True)
        diagnostic.SetCustomDataByKey('objectSegmentationComplete',False)
        shutil.copy2(geom_dir/'metrics.json',out/'assets'/(asset_name+'_metrics.json'))
        geometry_layers.append(dict(prim=str(diagnostic.GetPath()),inferred_prior=inferred,
            source_arrays_sha256=sha(geometry),source_metrics_sha256=sha(geom_dir/'metrics.json'),
            vertices=len(arr['vertices']),triangles=len(arr['triangles']),collision_enabled=False))
    UsdGeom.Xform.Define(stage,'/World/ReconstructionFrame/Cameras')
    for view in data['train']+data['val']:
        path='/World/ReconstructionFrame/Cameras/'+Path(view['name']).stem
        camera_at(stage,path,view)
    stage.GetRootLayer().Save()
    shutil.copytree(root/'semantic_inventory',out/'semantics')
    UsdGeom.Scope.Define(stage,'/World/ReconstructionFrame/Observations')
    for obj in inventory['objects']:
        prim=UsdGeom.Scope.Define(stage,'/World/ReconstructionFrame/Observations/'+obj['instance_id']).GetPrim()
        for key,value in [('instanceId',obj['instance_id']),('category',obj['category']),('label',obj['label']),
                          ('parentInstanceId',obj['parent_instance_id'] or ''),('experimentRole',obj['experiment_role'])]:
            prim.CreateAttribute('real2sim:'+key,Sdf.ValueTypeNames.String).Set(value)
        prim.CreateAttribute('real2sim:evidence',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./semantics/objects.json'))
        prim.CreateAttribute('real2sim:sourceFrames',Sdf.ValueTypeNames.StringArray).Set([v['frame'] for v in obj['observations']])
        prim.CreateAttribute('real2sim:geometrySegmented',Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute('real2sim:physicalParametersMeasured',Sdf.ValueTypeNames.Bool).Set(False)
        prim.SetCustomDataByKey('role','Source-linked observation record; no detached body geometry or measured joint')
    stage.GetRootLayer().Save()
    manifest={
        'schema':'real2sim-scene/v1','source_sha256':data['source_sha256'],
        'status':'candidate_not_accepted_for_deployment','entrypoint':'scene.usda',
        'coordinate_system':'Right-handed Z-up using a source-fitted floor estimate; all assets and cameras share one rigid transform',
        'metres_per_reconstruction_unit':None,'gravity_frame_verified':False,
        'sfm_to_scene':transform.tolist(),'scene_to_sfm':np.linalg.inv(transform).tolist(),
        'floor_alignment_estimated':floor is not None,'floor_evidence':floor,
        'appearance_model_normalized':False,
        'geometry':'RGB MVS surface evidence and, when included, a separately identified learned-depth hypothesis; hidden surfaces unverified',
        'geometry_layers':geometry_layers,
        'collision_enabled':False,'articulations_authored':False,'semantics':'semantics/objects.json',
        'inventory_entries':inventory['object_count'],'inventory_observations':inventory['observation_count'],
        'inventory_scope':inventory['scope'],'object_geometry_segmented':False,
        'native_isaac_render_verified':False,'geometry_metric_verified':False,
        'real_lidar_localization_verified':False,'g1_manipulation_verified':False,
        'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()},
        'appearance_metrics':result['appearance_metrics'],
        'validation_localization':data.get('validation_localization',{}),
        'instructions':'Open scene.usda in an Isaac runtime supporting legacy NuRec. This is a preview: do not infer metric scale from USD units. Surface layers are hidden for appearance rendering; make them visible to inspect geometry. Their CollisionAPI is explicitly disabled until geometry, scale and contacts are validated. Observation scopes are image evidence, not separate movable bodies.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reopened=Usd.Stage.Open(str(out/'scene.usda'))
    assert reopened and reopened.GetPrimAtPath('/World/ReconstructionFrame/Appearance/gauss')
    print(json.dumps({'scene':str(out/'scene.usda'),'sha256':sha(out/'scene.usda'),
          'composed_prims':len(list(reopened.Traverse())),'status':manifest['status']},indent=2))


if __name__=='__main__':main()
