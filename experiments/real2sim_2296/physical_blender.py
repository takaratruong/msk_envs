#!/usr/bin/env python3
"""Import the exact physical USD into an editable, texture-packed Blender file.

The USD remains authoritative for simulation. Blender's native USD importer does
not reproduce USD/PhysX articulations: every joint is preserved as inspectable
metadata/empties, without inventing a Blender rigid-body solver configuration.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera',default='/World/Cameras/frame_001499')
    p.add_argument('--blender',type=Path)
    p.add_argument('--render-width',type=int,default=960)
    p.add_argument('--samples',type=int,default=24)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--skip-render',action='store_true')
    p.add_argument('--inside-blender',action='store_true',help=argparse.SUPPRESS)
    return p


def blender_path(args):
    if args.blender:return args.blender.resolve()
    for parent in args.scene.resolve().parents:
        receipt=parent/'tools/blender.receipt.json'
        if receipt.exists():
            data=json.loads(receipt.read_text())
            if not data.get('official_checksum_verified'):raise ValueError('Blender installation receipt is not verified.')
            path=Path(data['executable'])
            if path.is_file():return path.resolve()
    raise FileNotFoundError('Supply --blender or place a verified tools/blender.receipt.json beside the run.')


def plain(value):
    """Lossless-enough readable USD metadata values; original typed text retained."""
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if hasattr(value,'path') and hasattr(value,'resolvedPath'):
        return {'path':value.path,'resolved_path':value.resolvedPath}
    if hasattr(value,'GetReal') and hasattr(value,'GetImaginary'):
        return {'real':float(value.GetReal()),'imaginary':list(value.GetImaginary())}
    try:return [plain(v) for v in value]
    except TypeError:return str(value)


def _run(args):
    import bpy
    import numpy as np
    from mathutils import Matrix,Vector,Quaternion
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Sdf,Gf

    started=time.time();source=args.scene.resolve();output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix!='.blend':raise ValueError('--output must be a .blend path.')
    if output.exists() or output.with_suffix('.receipt.json').exists():
        raise FileExistsError('Refusing to replace a completed authoring artifact; use a fresh output.')
    if args.render_width<128 or args.render_width>4096 or not 1<=args.samples<=256:
        raise ValueError('Render limits are width 128..4096 and samples 1..256.')
    source_sha=sha(source);manifest_path=source.with_suffix('.manifest.json');build_path=source.parent/'build_inputs.json'
    manifest=json.loads(manifest_path.read_text());build=json.loads(build_path.read_text())
    if manifest['scene_sha256']!=source_sha or build['scene_sha256']!=source_sha:
        raise ValueError('USD does not match the frozen scene/build receipts.')
    if manifest.get('representation')!='mesh' or manifest.get('gaussian_volumes')!=0:
        raise ValueError('This authoring converter requires the physical mesh scene, without splats.')
    stage=Usd.Stage.Open(str(source))
    if not stage:raise RuntimeError('USD stage could not be opened.')
    xforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_prims={str(p.GetPath()):p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
    camera_prims={str(p.GetPath()):p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)}
    material_prims={str(p.GetPath()):p for p in stage.Traverse() if p.IsA(UsdShade.Material)}
    forbidden=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() in ('Points','PointInstancer','Volume','ParticleSystem')]
    if forbidden:raise ValueError('Unexpected point/volume representation: '+str(forbidden[:5]))
    if args.camera not in camera_prims:raise ValueError('Requested source camera is missing.')
    source_camera=UsdGeom.Camera(camera_prims[args.camera]);camera_data=camera_prims[args.camera].GetCustomData()
    source_width=int(camera_data.get('imageWidth',1920));source_height=int(camera_data.get('imageHeight',1080))
    layers={str(layer.realPath):sha(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    dependencies={};texture_inputs=[]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray):continue
            values=attr.Get();values=values if attr.GetTypeName()==Sdf.ValueTypeNames.AssetArray else [values]
            for value in values or []:
                if not value or not value.path:continue
                resolved=Path(value.resolvedPath) if value.resolvedPath else (source.parent/value.path).resolve()
                if not resolved.is_file():raise FileNotFoundError('Unresolved USD asset: '+str(attr.GetPath())+' '+str(value.path))
                dependencies[str(resolved)]={'sha256':sha(resolved),'size':resolved.stat().st_size}
                if resolved.suffix.lower() in ('.png','.jpg','.jpeg','.exr','.tif','.tiff','.bmp'):
                    texture_inputs.append({'attribute':str(attr.GetPath()),'resolved_path':str(resolved),'sha256':dependencies[str(resolved)]['sha256']})
    joints=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        attrs={attr.GetName():{'type':str(attr.GetTypeName()),'value':plain(attr.Get())}
               for attr in prim.GetAttributes() if attr.HasAuthoredValueOpinion()}
        rels={rel.GetName():[str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships()}
        joints.append({'path':str(prim.GetPath()),'type':str(prim.GetTypeName()),'attributes':attrs,
                       'relationships':rels,'custom_data':plain(prim.GetCustomData())})
    provenance={'schema':'real2sim-blender-authoring-inputs/v1','source_usd':str(source),'source_usd_sha256':source_sha,
                'source_video_sha256':build['source_sha256'],'manifest_sha256':sha(manifest_path),'build_inputs_sha256':sha(build_path),
                'implementation_sha256':sha(__file__),'blender_version':bpy.app.version_string,'blender_build_hash':bpy.app.build_hash.decode(),
                'layers':layers,'dependencies':dependencies,'texture_inputs':texture_inputs,
                'expected_meshes':len(mesh_prims),'expected_cameras':len(camera_prims),'expected_materials':len(material_prims),
                'usd_joints':joints,'selected_camera':args.camera,'source_camera_metadata':plain(camera_data),
                'physics_authority':'Original USD; this Blender file does not import USD articulations into a solver.',
                'metric_accuracy_verified':False,'source_uploads':False}
    write_json(output.with_suffix('.inputs.json'),provenance)
    print(json.dumps({'phase':'audited','meshes':len(mesh_prims),'cameras':len(camera_prims),'joints':len(joints),'textures':len(dependencies)}),flush=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    imported_ids={}
    class AuthoringUSDPaths(bpy.types.USDHook):
        bl_idname='real2sim_authoring_usd_paths'
        bl_label='Preserve physical USD paths'
        @staticmethod
        def on_import(context):
            for path,ids in context.get_prim_map().items():
                for data in ids:
                    data['usd_prim_path']=str(path)
                    if isinstance(data,bpy.types.Object):imported_ids[str(path)]=data
            return True
    bpy.utils.register_class(AuthoringUSDPaths)
    result=bpy.ops.wm.usd_import(filepath=str(source),scale=1.0,set_frame_range=False,
        import_cameras=True,import_curves=False,import_lights=True,import_materials=True,import_meshes=True,
        import_volumes=False,import_shapes=False,import_skeletons=False,import_blendshapes=False,
        import_points=False,import_visible_only=False,create_collection=False,read_mesh_uvs=True,
        read_mesh_colors=True,read_mesh_attributes=True,import_all_materials=True,import_usd_preview=True,
        set_material_blend=True,import_textures_mode='IMPORT_PACK',attr_import_mode='ALL',
        validate_meshes=True,create_world_material=True,merge_parent_xform=False,
        apply_unit_conversion_scale=True,support_scene_instancing=False)
    bpy.utils.unregister_class(AuthoringUSDPaths)
    if 'FINISHED' not in result:raise RuntimeError('Native Blender USD import did not finish.')
    imported_meshes=[o for o in bpy.data.objects if o.type=='MESH'];imported_cameras=[o for o in bpy.data.objects if o.type=='CAMERA']
    if len(imported_meshes)!=len(mesh_prims) or len(imported_cameras)!=len(camera_prims):
        raise RuntimeError(f'Imported counts differ: meshes {len(imported_meshes)}/{len(mesh_prims)}, cameras {len(imported_cameras)}/{len(camera_prims)}.')
    # Native importer may drop implicitly-defined organizational prims. Its
    # official import hook maps IDs to exact USD paths despite duplicate names.
    # Rebuild those missing parent empties and author source world matrices,
    # preserving the USD hierarchy without inferring names or moving vertices.
    by_path=dict(imported_ids)
    if not set(mesh_prims).issubset(by_path) or not set(camera_prims).issubset(by_path):
        raise RuntimeError('Native import hook did not map every mesh and camera.')
    original_native_matrix_errors=[];restored_groups=[]
    for path,obj in by_path.items():
        prim=stage.GetPrimAtPath(path)
        if prim.IsA(UsdGeom.Imageable):
            original_native_matrix_errors.append(float(np.max(np.abs(np.array(obj.matrix_world)-np.array(xforms.GetLocalToWorldTransform(prim)).T))))
    paths=set(by_path)
    for path in list(paths):
        parent=Sdf.Path(path).GetParentPath()
        while parent!=Sdf.Path.absoluteRootPath:
            paths.add(str(parent));parent=parent.GetParentPath()
    for path in sorted(paths,key=lambda p:(p.count('/'),p)):
        prim=stage.GetPrimAtPath(path)
        if path not in by_path:
            obj=bpy.data.objects.new(Sdf.Path(path).name,None);scene_collection=bpy.context.scene.collection
            scene_collection.objects.link(obj);by_path[path]=obj;restored_groups.append(path)
        obj=by_path[path];parent_path=str(Sdf.Path(path).GetParentPath())
        obj.parent=by_path.get(parent_path)
        obj.matrix_parent_inverse=Matrix.Identity(4)
        obj.matrix_world=Matrix(np.array(xforms.GetLocalToWorldTransform(prim)).T.tolist())
        obj['usd_prim_path']=path
    bpy.context.view_layer.update()
    for path,obj in by_path.items():
        prim=stage.GetPrimAtPath(path)
        if prim:
            custom=prim.GetCustomData()
            if custom:obj['usd_custom_data_json']=json.dumps(plain(custom),sort_keys=True)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):obj['usd_rigid_body']=True
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                obj['usd_collision_enabled']=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                obj['usd_collision_approximation']=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
    if not set(mesh_prims).issubset(by_path) or not set(camera_prims).issubset(by_path):
        raise RuntimeError('Native import did not retain an auditable USD object hierarchy.')
    matrix_errors=[];geometry_errors=[];uv_counts=[]
    for path,prim in mesh_prims.items():
        obj=by_path[path];usd_mesh=UsdGeom.Mesh(prim)
        expected=np.asarray(usd_mesh.GetPointsAttr().Get(),dtype=np.float64)
        actual=np.array([v.co[:] for v in obj.data.vertices])
        if expected.shape!=actual.shape:raise RuntimeError('Mesh vertex count changed: '+path)
        geometry_errors.append(float(np.max(np.abs(actual-expected))))
        usd_counts=list(usd_mesh.GetFaceVertexCountsAttr().Get())
        if len(usd_counts)!=len(obj.data.polygons) or sum(usd_counts)!=len(obj.data.loops):
            raise RuntimeError('Mesh face/corner counts changed: '+path)
        if not obj.data.uv_layers:raise RuntimeError('Missing imported mesh UVs: '+path)
        uv_counts.append(len(obj.data.uv_layers.active.data))
        bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if bound:
            expected_material=str(bound.GetPath())
            actual_material=obj.data.materials[0] if obj.data.materials else None
            if actual_material is None or actual_material.get('usd_prim_path', '/World/Materials/'+actual_material.name)!=expected_material:
                raise RuntimeError('Mesh material binding differs from USD: '+path)
        matrix_errors.append(float(np.max(np.abs(np.array(obj.matrix_world)-np.array(xforms.GetLocalToWorldTransform(prim)).T))))
    if max(geometry_errors,default=0)>1e-6 or max(matrix_errors,default=0)>1e-5:
        raise RuntimeError('Native import changed mesh coordinates or world transforms.')
    camera_matrix_errors=[float(np.max(np.abs(np.array(by_path[path].matrix_world)-np.array(xforms.GetLocalToWorldTransform(prim)).T))) for path,prim in camera_prims.items()]
    camera=by_path[args.camera];camera_matrix_error=max(camera_matrix_errors)
    if camera_matrix_error>1e-5:raise RuntimeError('Source camera world transform changed.')
    scene=bpy.context.scene;scene.camera=camera;scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1.0
    scene.render.resolution_x=args.render_width;scene.render.resolution_y=round(args.render_width*source_height/source_width);scene.render.resolution_percentage=100
    # Preserve the USD filmback exactly even when a small preview's integer
    # dimensions cannot reproduce the original aspect ratio perfectly.
    h_ap=float(source_camera.GetHorizontalApertureAttr().Get());v_ap=float(source_camera.GetVerticalApertureAttr().Get())
    correction=(scene.render.resolution_x/scene.render.resolution_y)/(h_ap/v_ap)
    scene.render.pixel_aspect_x=1.0 if correction>=1 else 1/correction
    scene.render.pixel_aspect_y=correction if correction>=1 else 1.0
    bpy.context.view_layer.update()
    usd_projection=np.array(source_camera.GetCamera(Usd.TimeCode.Default()).frustum.ComputeProjectionMatrix()).T
    blender_projection=np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
       x=scene.render.resolution_x,y=scene.render.resolution_y,
       scale_x=scene.render.pixel_aspect_x,scale_y=scene.render.pixel_aspect_y))
    points=np.array([[x,y,-1,1] for x,y in ((0,0),(-.3,-.2),(.3,.2),(-.3,.2),(.3,-.2))])
    def project(matrix):
        q=points@matrix.T;ndc=q[:,:2]/q[:,3:4]
        return (ndc*np.array([.5,-.5])+.5)*[scene.render.resolution_x,scene.render.resolution_y]
    projection_error=float(np.max(np.abs(project(usd_projection)-project(blender_projection))))
    if projection_error>.10:raise RuntimeError('Blender source-camera projection differs by '+str(projection_error)+' pixels.')

    # Preserve actual USD joints as metadata empties, not fake working constraints.
    metadata_collection=bpy.data.collections.new('USD Physics Metadata — not Blender simulation')
    scene.collection.children.link(metadata_collection)
    joint_names=[]
    for j in joints:
        path=j['path'];prim=stage.GetPrimAtPath(path);joint=UsdPhysics.Joint(prim)
        name='USDJoint_'+path.split('/Assets/')[-1].replace('/','__')
        obj=bpy.data.objects.new(name,None);metadata_collection.objects.link(obj);obj.empty_display_type='ARROWS';obj.empty_display_size=.06;obj.hide_render=True
        obj['usd_joint_path']=path;obj['usd_joint_type']=j['type'];obj['usd_joint_json']=json.dumps(j,sort_keys=True)
        obj['blender_simulation_status']='Metadata only: USD articulation was not imported into Blender physics.'
        for key in ('physics:axis','physics:lowerLimit','physics:upperLimit','physics:jointEnabled'):
            value=prim.GetAttribute(key).Get()
            if value is not None:obj[key.replace(':','_')]=value
        body0=joint.GetBody0Rel().GetTargets();body1=joint.GetBody1Rel().GetTargets()
        targets=body0 or body1;local=joint.GetLocalPos0Attr().Get() if body0 else joint.GetLocalPos1Attr().Get()
        if targets and str(targets[0]) in by_path:
            obj.parent=by_path[str(targets[0])];obj.location=tuple(local)
        else:obj.location=tuple(local)
        joint_names.append(obj.name)
    texture_nodes=[]
    for material in bpy.data.materials:
        if not material.use_nodes:continue
        for node in material.node_tree.nodes:
            if node.type=='TEX_IMAGE' and node.image:
                texture_nodes.append({'material':material.name,'image':node.image.name,'path':bpy.path.abspath(node.image.filepath)})
    expected_texture_names={Path(x['resolved_path']).name for x in texture_inputs}
    actual_texture_names={Path(x['path']).name for x in texture_nodes}
    if not expected_texture_names.issubset(actual_texture_names):
        raise RuntimeError('Imported materials lost source textures: '+str(sorted(expected_texture_names-actual_texture_names)))
    image_records=[]
    for image in bpy.data.images:
        if image.source!='FILE':continue
        original=bpy.path.abspath(image.filepath)
        if not image.has_data:image.reload()
        image.pack()
        if not image.packed_file:raise RuntimeError('Texture could not be packed: '+image.name)
        image['source_usd_resolved_path']=original
        image_records.append({'name':image.name,'original_path':original,'packed_bytes':image.packed_file.size,
                              'size':list(image.size),'packed_sha256':hashlib.sha256(image.packed_file.data).hexdigest()})
        if original not in dependencies or image_records[-1]['packed_sha256']!=dependencies[original]['sha256']:
            raise RuntimeError('Packed texture bytes differ from their source USD dependency: '+image.name)
        # Deliberately remove operational dependence on original filesystem paths.
        # Packed data remains authoritative when this .blend is opened elsewhere.
        image.filepath='//packed_textures/'+Path(original).name
    if not image_records:raise RuntimeError('No file textures were packed.')
    scene['source_usd_sha256']=source_sha;scene['source_video_sha256']=build['source_sha256']
    scene['source_usd_path']=str(source);scene['physics_authority']='Canonical source USD; Blender joints are inspectable metadata only.'
    scene['usd_joint_count']=len(joints);scene['representation']='Textured polygon meshes; no Gaussian splats or volumes.'
    scene['source_camera_path']=args.camera
    source_text=bpy.data.texts.new('USD_SOURCE_PROVENANCE.json');source_text.write(json.dumps(provenance,indent=2))
    readme='''Editable authoring copy imported by Blender's native USD importer.\n\nThe original physical USD remains authoritative for simulation. Blender does not automatically import USD/PhysX articulations into its physics solver. Each joint is retained as an inspectable arrow empty and JSON custom properties in the USD Physics Metadata collection. Do not treat these empties as working constraints. USD revolute limit metadata retains degrees, and prismatic limits retain stage metres.\n\nThe native importer dropped implicit organizational groups and their parent transforms in this input. Blender's official USD import hook supplied exact primitive paths; the converter recreated missing parent empties and restored their authoritative USD world matrices. No image-based pose fitting or camera optimization occurred. Mesh coordinates, hierarchy, world transforms, UVs, material bindings and all source camera transforms were checked against the frozen USD. The chosen camera's projection was also checked.\n\nSource image textures are packed byte-for-byte into the blend. Operational image paths intentionally point to a nonexistent packed_textures folder, so reopening/rendering exercises packed data. USD primitive paths and custom metadata remain on objects.\n\nSource-based dimensions and physical parameters remain inferred and unmeasured. This authoring import is not a new metric-accuracy or robot-deployment validation. Blender's material conversion and AgX tone mapping can differ from Isaac rendering.\n\nReferences: [Blender 4.5 USD import documentation](https://docs.blender.org/manual/en/4.5/files/import_export/usd.html), [native importer API](https://docs.blender.org/api/4.5/bpy.ops.wm.html#bpy.ops.wm.usd_import). The path hook is runtime-verified in the installed Blender 4.5.13 build.\n'''
    text_block=bpy.data.texts.new('AUTHORING_README.md');text_block.write(readme)
    (output.parent/'AUTHORING_README.md').write_text(readme+'\nSource USD SHA256: '+source_sha+'\nSource video SHA256: '+build['source_sha256']+'\n')
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=args.threads
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.film_transparent=False
    scene.render.filepath=str(output.with_suffix('.png'))
    scene.view_settings.view_transform='AgX'
    # Opening the file presents the chosen source camera without controlling a desktop viewer.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
    checks={'mesh_count':len(imported_meshes),'camera_count':len(imported_cameras),'material_count':len(bpy.data.materials),
            'joint_metadata_count':len(joint_names),'mesh_local_coordinate_max_error':max(geometry_errors,default=0),
            'native_matrix_error_before_hierarchy_restoration':max(original_native_matrix_errors,default=0),
            'restored_organizational_groups':restored_groups,
            'mesh_world_matrix_max_error':max(matrix_errors,default=0),'source_camera_matrix_error':camera_matrix_error,
            'source_camera_projection_max_error_pixels':projection_error,'uv_corner_count':sum(uv_counts),
            'packed_images':image_records,'image_texture_nodes':texture_nodes,'point_or_volume_prims':forbidden,
            'physics_simulation_imported':False}
    write_json(output.with_suffix('.import.json'),checks)
    bpy.ops.wm.save_as_mainfile(filepath=str(output),compress=True,check_existing=False)
    print(json.dumps({'phase':'saved','blend':str(output),'meshes':len(imported_meshes),'camera_error_pixels':projection_error,'packed_images':len(image_records)}),flush=True)
    # Reopen the saved artifact before rendering: packed textures and camera state
    # must survive serialization, not just exist in the import process's memory.
    bpy.ops.wm.open_mainfile(filepath=str(output))
    reopened=bpy.context.scene
    reopened_meshes=[o for o in bpy.data.objects if o.type=='MESH'];reopened_cameras=[o for o in bpy.data.objects if o.type=='CAMERA']
    if len(reopened_meshes)!=len(mesh_prims) or len(reopened_cameras)!=len(camera_prims):raise RuntimeError('Reopened blend changed object counts.')
    for row in image_records:
        image=bpy.data.images.get(row['name'])
        if image is None or not image.packed_file:raise RuntimeError('Packed texture did not survive reopening: '+row['name'])
        if hashlib.sha256(image.packed_file.data).hexdigest()!=row['packed_sha256']:raise RuntimeError('Packed texture bytes changed after reopening.')
        # Blender loads packed pixel buffers lazily after opening a file.
        # Access pixels explicitly before testing has_data; do not reload a
        # deliberately absent external path or mistake laziness for lost data.
        if len(image.pixels)==0 or not math.isfinite(float(image.pixels[0])) or not image.has_data:
            raise RuntimeError('Packed texture pixels could not be read after reopening: '+row['name'])
    render_path=None
    if not args.skip_render:
        print(json.dumps({'phase':'render','engine':'Cycles','device':'CPU','samples':args.samples,'width':reopened.render.resolution_x,'height':reopened.render.resolution_y}),flush=True)
        bpy.ops.render.render(write_still=True);render_path=output.with_suffix('.png')
        if not render_path.exists() or render_path.stat().st_size<1024:raise RuntimeError('Source-camera Blender still was not written.')
    if sha(source)!=source_sha or any(sha(path)!=info['sha256'] for path,info in dependencies.items()):
        raise RuntimeError('Frozen USD or referenced textures changed during import/render.')
    receipt={'schema':'real2sim-blender-authoring/v1','status':'verified_authoring_copy',
             'produced_at':dt.datetime.now(dt.timezone.utc).isoformat(),'source_usd':str(source),'source_usd_sha256':source_sha,
             'source_video_sha256':build['source_sha256'],'implementation_sha256':sha(__file__),
             'blender_version':bpy.app.version_string,'output':{'path':str(output),'sha256':sha(output),'size':output.stat().st_size},
             'inputs_receipt_sha256':sha(output.with_suffix('.inputs.json')),'import_receipt_sha256':sha(output.with_suffix('.import.json')),
             'mesh_count':len(reopened_meshes),'camera_count':len(reopened_cameras),'joint_metadata_count':len(joints),
             'packed_image_count':len(image_records),'packed_textures_reopened_verified':True,'selected_camera':args.camera,
             'camera_projection_error_pixels':projection_error,'gaussian_splats':0,'physics_simulation_imported':False,
             'metric_accuracy_verified':False,'elapsed_seconds':time.time()-started,
             'render':None if render_path is None else {'path':str(render_path),'sha256':sha(render_path),'engine':'Cycles CPU','samples':args.samples},
             'limitations':['Native USD mesh/material/camera authoring import; USD physics remains authoritative.',
                            'Joints are preserved as metadata/empties, not Blender solver constraints.',
                            'Blender material/lighting conversion and AgX differ from Isaac rendering.',
                            'Nominal source-informed geometry is not independently measured metric geometry.']}
    write_json(output.with_suffix('.receipt.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args=parser().parse_args(argv)
    if args.inside_blender:
        _run(args);return
    executable=blender_path(args)
    command=[str(executable),'--background','--factory-startup','--threads',str(args.threads),
             '--python-exit-code','1','--python',str(Path(__file__).resolve()),'--',*argv,'--inside-blender']
    result=subprocess.run(command,check=False)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
