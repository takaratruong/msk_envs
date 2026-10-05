#!/usr/bin/env python3
"""Ideal instantaneous ray probe of authored USD collision geometry.

This is neither a physical LiDAR simulator nor a localization validation. It uses
nominal scene metres, default authored initial transforms, individual convex hulls
where requested by USD, and an artificial angular grid with glass opaque/omitted.
No intensity, refraction, noise, multiple returns, beam footprint or timing model
is inferred. The scene is read only; every output stays beneath --output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,data):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def category(asset,material):
    if asset=='floor_surface_region_01' or material=='floor':
        return 'floor'
    if any(word in asset for word in ('wall','partition','ceiling','column')):
        return 'room'
    return 'furniture'


def inputs(stage,scene):
    from pxr import Sdf
    result = {'scene':{'path':str(scene),'sha256':sha(scene)},'layers':[],'asset_dependencies':[],
              'sidecar_manifests':[],'archived_build_helpers':[]}
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        path = Path(layer.realPath or layer.identifier).resolve()
        result['layers'].append({'path':str(path),'sha256':sha(path)})
    dependencies = {}
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset,Sdf.ValueTypeNames.AssetArray):
                continue
            value = attr.Get()
            values = [value] if isinstance(value,Sdf.AssetPath) else list(value or [])
            for asset in values:
                if not asset.path:
                    continue
                authored_layer = attr.GetPropertyStack()[0].layer
                resolved = asset.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(authored_layer,asset.path)
                path = Path(resolved).resolve()
                if not path.is_file():
                    raise ValueError('Unresolved USD asset dependency: '+str(attr.GetPath()))
                dependencies[str(path)] = {'path':str(path),'sha256':sha(path),'size':path.stat().st_size}
    result['asset_dependencies'] = list(dependencies.values())
    manifest_path = scene.with_suffix('.manifest.json')
    build_path = scene.parent/'build_inputs.json'
    for path in (manifest_path,build_path):
        if not path.is_file():
            raise ValueError('Frozen physical scene is missing its provenance sidecar: '+str(path))
        data = json.loads(path.read_text())
        if data['scene_sha256'] != result['scene']['sha256']:
            raise ValueError('Scene hash differs from sidecar: '+str(path))
        result['sidecar_manifests'].append({'path':str(path),'sha256':sha(path)})
    build = json.loads(build_path.read_text())
    result['source_sha256'] = build['source_sha256']
    result['scale_status'] = build['scale_status']
    for name,expected in build['code'].items():
        path = scene.parent/'source'/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Archived scene helper differs from build receipt: '+str(path))
        result['archived_build_helpers'].append({'path':str(path),'sha256':expected})
    return result


def extract_colliders(stage):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    from scipy.spatial import ConvexHull
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices,faces,owners,rows = [],[],[],[]
    vertex_offset = 0
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsInstance():
            raise ValueError('Instance proxies require explicit expansion: '+str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if any(child.IsA(UsdGeom.Subset) for child in prim.GetChildren()):
            raise ValueError('Per-face material subsets need explicit glass treatment')
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or counts.sum()!=len(indices):
            raise ValueError('Invalid USD collision mesh: '+str(prim.GetPath()))
        if np.any(counts!=3):
            raise ValueError('This bounded probe expects the frozen scene triangular mesh contract')
        triangles = indices.reshape(-1,3).copy()
        if triangles.min()<0 or triangles.max()>=len(points):
            raise ValueError('Collision mesh index outside points')
        transform = np.asarray(cache.GetLocalToWorldTransform(prim),dtype=np.float64)
        world = points@transform[:3,:3]+transform[3,:3]
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        material_path = str(material.GetPath()) if material else ''
        material_name = material.GetPrim().GetName() if material else ''
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or 'none')
        rigid = prim
        while rigid and not rigid.IsPseudoRoot() and not rigid.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid = rigid.GetParent()
        dynamic = bool(rigid and not rigid.IsPseudoRoot() and UsdPhysics.RigidBodyAPI(rigid).GetRigidBodyEnabledAttr().Get())
        if approximation=='convexHull':
            hull = ConvexHull(world)
            triangles = hull.simplices.copy()
            t = world[triangles]
            reverse = np.einsum('ij,ij->i',np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),hull.equations[:,:3])<0
            triangles[reverse] = triangles[reverse][:,[0,2,1]]
        elif approximation=='none':
            flip = (str(mesh.GetOrientationAttr().Get())=='leftHanded') ^ (np.linalg.det(transform[:3,:3])<0)
            if flip:
                triangles = triangles[:,[0,2,1]]
        else:
            raise ValueError('Unsupported USD collision approximation: '+approximation)
        t = world[triangles]
        if not np.isfinite(world).all() or np.any(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)<1e-12):
            raise ValueError('Degenerate extracted collision triangle: '+str(prim.GetPath()))
        path = str(prim.GetPath())
        asset = path.split('/')[3] if path.startswith('/World/Assets/') else prim.GetName()
        collider_id = len(rows)
        row = {'id':collider_id,'usd_path':path,'asset':asset,'category':category(asset,material_name),
               'material':material_name,'material_path':material_path,'glass':material_name=='glass',
               'dynamic_body':dynamic,'rigid_body_path':str(rigid.GetPath()) if dynamic else None,
               'authored_approximation':approximation,'extracted_representation':'individual_convex_hull' if approximation=='convexHull' else 'authored_triangles',
               'authored_visibility':str(mesh.GetVisibilityAttr().Get()),'visibility_does_not_disable_collision':True,
               'world_transform_row_vector_convention':transform.tolist(),
               'vertices':len(world),'triangles':len(triangles),
               'bounds_nominal_m':[world.min(0).tolist(),world.max(0).tolist()]}
        rows.append(row)
        vertices.append(world.astype(np.float32))
        faces.append((triangles+vertex_offset).astype(np.uint32))
        owners.append(np.full(len(triangles),collider_id,dtype=np.int32))
        vertex_offset += len(world)
    if not rows:
        raise ValueError('USD has no enabled collision meshes')
    return np.concatenate(vertices),np.concatenate(faces),np.concatenate(owners),rows,mesh_count


def ray_scene(vertices,faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(vertices,dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces,dtype=o3d.core.Dtype.UInt32))
    return scene


def selftest():
    import open3d as o3d
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf
    box = o3d.geometry.TriangleMesh.create_box(1,1,1).translate([1.5,-.5,-.5])
    test = ray_scene(np.asarray(box.vertices,dtype=np.float32),np.asarray(box.triangles,dtype=np.uint32))
    rays = np.array([[0,0,0,1,0,0],[0,0,0,-1,0,0],[2,0,0,1,0,0]],dtype=np.float32)
    distance = test.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    if not np.allclose(distance[[0,2]],[1.5,.5],atol=1e-6) or not np.isinf(distance[1]):
        raise ValueError('Independent analytic box intersection control failed')
    # An independent in-memory USD control checks composed transform ordering,
    # moving-part hull extraction and world-space range, without reading our
    # production geometry archive back through the same numerical transform.
    usd = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(usd,'/World/Assets/transform_control/Links/base')
    root.AddTranslateOp().Set(Gf.Vec3d(2,-3,1))
    root.AddRotateZOp().Set(90.)
    root.AddScaleOp().Set(Gf.Vec3f(2,1,.5))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    mesh = UsdGeom.Mesh.Define(usd,str(root.GetPath())+'/cube')
    cube_vertices = np.asarray(box.vertices)-[2.,0.,0.]
    # A redundant interior point must not inflate the individual hull.
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in np.vstack([cube_vertices,[0,0,0]])])
    mesh.CreateFaceVertexCountsAttr([3]*len(box.triangles))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(box.triangles).reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    material = UsdShade.Material.Define(usd,'/World/Materials/black')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    vertices,faces,_,rows,_ = extract_colliders(usd)
    expected_bounds = np.array([[1.5,-4,.75],[2.5,-2,1.25]])
    if not np.allclose([vertices.min(0),vertices.max(0)],expected_bounds,atol=1e-6) or len(faces)!=12 or not rows[0]['dynamic_body']:
        raise ValueError('USD transform/individual-convex-hull extraction control failed')
    transformed = ray_scene(vertices,faces)
    hit = float(transformed.cast_rays(o3d.core.Tensor([[0,-3,1,1,0,0]],dtype=o3d.core.Dtype.Float32))['t_hit'][0].item())
    if abs(hit-1.5)>1e-6:
        raise ValueError('Composed USD world-space range control failed')
    return {'known_box_entry_distance':float(distance[0]),'known_inside_exit_distance':float(distance[2]),
            'away_ray_misses':True,'composed_usd_translate_rotate_scale_bounds':expected_bounds.tolist(),
            'individual_hull_triangles':len(faces),'composed_world_ray_distance':hit}


def scan(scene,owner_map,origin,directions,min_range,max_range):
    import open3d as o3d
    rays = np.c_[np.broadcast_to(origin,directions.shape),directions].astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    distance = result['t_hit'].numpy()
    primitive = result['primitive_ids'].numpy()
    valid = np.isfinite(distance)&(distance>=min_range)&(distance<=max_range)
    ids = np.full(len(distance),-1,dtype=np.int32)
    ids[valid] = owner_map[primitive[valid]]
    ranges = np.where(valid,distance,0).astype(np.float32)
    points = origin+directions[valid]*ranges[valid,None]
    return dict(valid=valid,ranges=ranges,collider_ids=ids,points=points.astype(np.float32),
                normals=result['primitive_normals'].numpy()[valid],
                ray_ids=np.flatnonzero(valid).astype(np.uint32),
                too_close_count=int(np.sum(np.isfinite(distance)&(distance<min_range))),
                beyond_range_count=int(np.sum(np.isfinite(distance)&(distance>max_range))))


COLORS = {'floor':(103,154,105),'room':(126,133,151),'furniture':(218,115,59),'glass':(30,168,211)}


def write_ply(path,result,rows):
    from plyfile import PlyData,PlyElement
    count = len(result['points'])
    data = np.empty(count,dtype=[(n,'<f4') for n in ('x','y','z','nx','ny','nz','range')]+
                    [(n,'u1') for n in ('red','green','blue')]+[('collider_id','<u4'),('ray_id','<u4')])
    for i,n in enumerate(('x','y','z')):
        data[n] = result['points'][:,i]
    for i,n in enumerate(('nx','ny','nz')):
        data[n] = result['normals'][:,i]
    data['range'] = result['ranges'][result['valid']]
    owner_ids = result['collider_ids'][result['valid']]
    data['collider_id'] = owner_ids
    data['ray_id'] = result['ray_ids']
    colors = np.array([COLORS['glass' if rows[i]['glass'] else rows[i]['category']] for i in owner_ids],dtype=np.uint8)
    for i,n in enumerate(('red','green','blue')):
        data[n] = colors[:,i]
    PlyData([PlyElement.describe(data,'vertex')],text=False,
            comments=['Ideal geometry probe; nominal metres; colors identify modeled return classes; no measured LiDAR data']).write(path)


def summary(result,rows):
    ids = result['collider_ids'][result['valid']]
    counts = Counter(map(int,ids))
    return {'rays':len(result['valid']),'hits':len(ids),'misses':int(np.sum(~result['valid'])),
            'range_min_nominal_m':float(result['ranges'][result['valid']].min()),
            'range_max_nominal_m':float(result['ranges'][result['valid']].max()),
            'too_close_count':result['too_close_count'],'beyond_max_range_count':result['beyond_range_count'],
            'category_counts':dict(Counter(rows[i]['category'] for i in ids)),
            'glass_hits':int(sum(rows[i]['glass'] for i in ids)),
            'unique_collider_returns':len(counts),'unique_asset_returns':len({rows[i]['asset'] for i in ids}),
            'return_ids':[{'collider_id':i,'usd_path':rows[i]['usd_path'],'asset':rows[i]['asset'],
                           'category':rows[i]['category'],'glass':rows[i]['glass'],'count':n}
                          for i,n in counts.most_common()]}


def diagnostic(path,results,rows,origin,n_azimuth,n_elevation):
    from PIL import Image,ImageDraw,ImageFont
    image = Image.new('RGB',(1440,1040),(244,245,247))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    draw.text((22,16),'Ideal collision-geometry scan | nominal scale, no physical LiDAR model',font=font,fill=(23,32,43))
    all_points = np.concatenate([r['points'] for r in results.values()])
    lo,hi = all_points[:,:2].min(0),all_points[:,:2].max(0)
    span = np.maximum(hi-lo,.1)
    factor = min(650/span[0],570/span[1])
    for i,(label,result) in enumerate(results.items()):
        left = i*720
        draw.text((left+24,56),label.replace('_',' ').title(),font=font,fill=(23,32,43))
        center = np.array([left+360,400.])
        pts = (result['points'][:,:2]-(lo+hi)/2)*factor
        pts[:,1] *= -1
        pts += center
        owners = result['collider_ids'][result['valid']]
        for cls in ('floor','room','furniture','glass'):
            for xy,owner in zip(pts,owners):
                actual = 'glass' if rows[owner]['glass'] else rows[owner]['category']
                if actual==cls:
                    draw.point(tuple(np.round(xy).astype(int)),fill=COLORS[cls])
        xy = (origin[:2]-(lo+hi)/2)*factor
        xy[1] *= -1
        xy += center
        draw.ellipse((xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5),fill=(188,32,54))
        draw.text((left+24,711),'Top view: +X right, +Y up; red point is the sensor origin',font=small,fill=(40,50,60))
        ranges = result['ranges'].reshape(n_elevation,n_azimuth)
        keep = result['valid'].reshape(n_elevation,n_azimuth)
        q = np.clip(ranges/12.,0,1)
        rgb = np.stack([45+195*q,80+105*(1-np.abs(2*q-1)),205-160*q],axis=2).astype(np.uint8)
        rgb[~keep] = (220,223,228)
        panel = Image.fromarray(rgb[::-1]).resize((672,160),Image.Resampling.NEAREST)
        image.paste(panel,(left+24,746))
        draw.text((left+24,916),'Range grid: azimuth 0–360°; high elevation at top; gray = no return',font=small,fill=(40,50,60))
    draw.text((24,952),'Return colors: floor green | room gray | furniture orange | glass cyan. Range-grid color saturates at 12 nominal m.',font=small,fill=(40,50,60))
    draw.text((24,983),'Glass control changes only glass-material colliders. No refraction, reflectivity, noise, timing or multiple-return behavior is modeled.',font=small,fill=(40,50,60))
    image.save(path)


def run(args):
    from pxr import Usd,UsdGeom
    import open3d as o3d
    import scipy
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    source = args.scene.resolve()
    start = time.monotonic()
    def progress(phase,**values):
        write_json(output/'progress.json',{'phase':phase,'updated_at':datetime.now(timezone.utc).isoformat(),**values})
        print(json.dumps({'phase':phase,**values}),flush=True)
    progress('loading_usd')
    stage = Usd.Stage.Open(str(source))
    if not stage or stage.GetCompositionErrors():
        raise ValueError('USD composition failed')
    if UsdGeom.GetStageUpAxis(stage)!='Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-12:
        raise ValueError('This physical scene probe expects nominal metre units and Z up')
    provenance = inputs(stage,source)
    progress('extracting_colliders')
    vertices,triangles,owners,rows,mesh_count = extract_colliders(stage)
    np.savez_compressed(output/'collision_geometry.npz',vertices=vertices,triangles=triangles,collider_ids=owners)
    write_json(output/'colliders.json',{'schema':'real2sim-ideal-ray-colliders/v1','scene_sha256':provenance['scene']['sha256'],'colliders':rows})
    progress('building_rays',colliders=len(rows),triangles=len(triangles),glass_colliders=sum(r['glass'] for r in rows))
    origin = np.asarray(args.pose,dtype=np.float32)
    full = ray_scene(vertices,triangles)
    closest = float(full.compute_distance(o3d.core.Tensor(origin[None]))[0].item())
    bounds = np.array([r['bounds_nominal_m'] for r in rows])
    containing_boxes = np.flatnonzero(np.all(origin>=bounds[:,0],axis=1)&np.all(origin<=bounds[:,1],axis=1))
    if len(containing_boxes) or closest<.30:
        raise ValueError('Requested probe origin is not demonstrably clear of collision geometry')
    azimuth = np.linspace(0,2*np.pi,args.azimuth_count,endpoint=False,dtype=np.float64)
    elevation = np.radians(np.linspace(args.elevation_min,args.elevation_max,args.elevation_count))
    az,el = np.meshgrid(azimuth,elevation)
    directions = np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)],axis=2).reshape(-1,3).astype(np.float32)
    no_glass = ~np.array([rows[i]['glass'] for i in owners])
    omitted = ray_scene(vertices,triangles[no_glass])
    progress('casting_ideal_scans',pose=origin.tolist(),nearest_collider_nominal_m=closest,rays=len(directions))
    results = {'glass_included':scan(full,owners,origin,directions,args.min_range,args.max_range),
               'glass_omitted':scan(omitted,owners[no_glass],origin,directions,args.min_range,args.max_range)}
    summaries = {name:summary(result,rows) for name,result in results.items()}
    checks = []
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
    check('analytic_box_raycast_controls',True,selftest())
    check('unit_direction_rays',np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-6))
    check('free_origin_outside_all_collider_bounds',not len(containing_boxes),closest)
    for name,result in results.items():
        check(name+'_finite_points_normals_and_stored_ranges',np.isfinite(result['points']).all() and np.isfinite(result['normals']).all() and np.isfinite(result['ranges']).all())
        ranges = result['ranges'][result['valid']]
        check(name+'_bounded_valid_ranges',np.all(ranges>=args.min_range)&np.all(ranges<=args.max_range))
        check(name+'_floor_room_furniture_return_ids',all(summaries[name]['category_counts'].get(c,0)>0 for c in ('floor','room','furniture')))
        check(name+'_return_points_agree_with_ranges',np.allclose(np.linalg.norm(result['points']-origin,axis=1),ranges,atol=3e-5))
        write_ply(output/(name+'.ply'),result,rows)
    a,b = results.values()
    shared = a['valid']&b['valid']
    difference = b['ranges'][shared]-a['ranges'][shared]
    changed = (a['valid']!=b['valid']) | (a['collider_ids']!=b['collider_ids'])
    changed |= shared&(np.abs(b['ranges']-a['ranges'])>1e-4)
    check('omitted_scan_has_no_glass_returns',summaries['glass_omitted']['glass_hits']==0)
    check('glass_control_exercises_a_visible_surface',summaries['glass_included']['glass_hits']>0 and np.any(changed))
    check('removing_glass_never_creates_a_closer_return',not np.any(b['valid']&~a['valid']) and np.all(difference>=-1e-4))
    fields = {'origin':origin,'directions':directions,'azimuth_radians':az.astype(np.float32),
              'elevation_radians':el.astype(np.float32)}
    for name,result in results.items():
        for key in ('valid','ranges','collider_ids','points','normals','ray_ids'):
            fields[name+'_'+key] = result[key]
    np.savez_compressed(output/'scans.npz',**fields)
    diagnostic(output/'comparison.png',results,rows,origin,args.azimuth_count,args.elevation_count)
    bound_inputs = [provenance['scene']]+provenance['layers']+provenance['asset_dependencies']+provenance['sidecar_manifests']+provenance['archived_build_helpers']
    check('all_bound_scene_inputs_unchanged_after_probe',all(sha(p['path'])==p['sha256'] for p in bound_inputs))
    artifacts = [{'path':str(p),'sha256':sha(p),'size':p.stat().st_size} for p in sorted(output.iterdir()) if p.name in
                 {'collision_geometry.npz','colliders.json','glass_included.ply','glass_omitted.ply','scans.npz','comparison.png'}]
    receipt = {'schema':'real2sim-ideal-lidar-geometry-probe/v1','status':'passed' if all(c['passed'] for c in checks) else 'failed',
               'produced_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
               'probe_type':'ideal_instantaneous_collision_geometry_rays','real_lidar_validation':False,
               'deployment_ready_map':False,'metric_accuracy_verified':False,
               'source_sha256':provenance['source_sha256'],'scene_sha256':provenance['scene']['sha256'],
               'helper_sha256':sha(__file__),'inputs':provenance,'checks':checks,'artifacts':artifacts,
               'pose':{'position_nominal_m':origin.tolist(),'yaw_degrees':0,'pitch_degrees':0,'roll_degrees':0,
                       'description':('Central aisle between the stool island and tables/carts; test point chosen in nominal scene coordinates'
                                      if np.allclose(origin,[1.50,-1.75,1.05]) else 'CLI-specified nominal scene point; geometric clearance checked'),
                       'closest_collision_surface_nominal_m':closest,'inside_any_collider_aabb':False,
                       'robot_mount_calibrated':False},
               'scan_pattern':{'azimuth_count':args.azimuth_count,'azimuth_interval_degrees':[0,360],
                       'azimuth_end_exclusive':True,'elevation_count':args.elevation_count,
                       'elevation_interval_degrees':[args.elevation_min,args.elevation_max],
                       'ray_count':len(directions),'range_limits_nominal_m':[args.min_range,args.max_range],
                       'range_definition':'Euclidean first-hit distance along unit world-space rays',
                       'invalid_stored_range':0,'valid_mask_required':True,'timestamps':'single instantaneous authored initial pose; no rolling acquisition'},
               'collision_extraction':{'authored_mesh_count':mesh_count,'enabled_collision_mesh_count':len(rows),
                       'collision_triangle_count':len(triangles),'convex_hull_mesh_count':sum(r['authored_approximation']=='convexHull' for r in rows),
                       'glass_material_mesh_count':sum(r['glass'] for r in rows),'time_code':'Default authored initial transforms',
                       'physics_advanced':False,'simulation_contact_offsets_applied':False,
                       'hidden_collision_patches_included':True,'whole_asset_hulls_created':False},
               'scans':summaries,'glass_sensitivity':{'changed_ray_count':int(changed.sum()),'changed_ray_fraction':float(changed.mean()),
                       'same_ray_shared_hit_range_delta_quantiles_nominal_m':np.quantile(difference,[0,.5,.9,1]).tolist(),
                       'control':'All collision meshes bound to material key glass are opaque first-return surfaces, versus omitted entirely'},
               'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
                          'open3d':o3d.__version__,'usd':'.'.join(map(str,Usd.GetVersion()))},
               'limitations':['Scene scale and hidden geometry are inferred, not independently measured.',
                    'The sensor model, beam pattern, rig extrinsics, calibration, material response and timing are unknown.',
                    'No beam divergence, incidence response, reflectivity, intensity, glass transmission/refraction, noise, dropout or multiple returns.',
                    'Uses current authored initial object poses only; no articulation motion or physical settling is run.',
                    'This probe checks modeled geometric intersections and glass sensitivity. It does not validate real LiDAR localization or sim-to-real performance.']}
    write_json(output/'receipt.json',receipt)
    (output/'README.md').write_text('# Ideal collision-geometry ray probe\n\n'
        'This is a geometry smoke test, not real LiDAR validation or a deployment-ready localization map. '
        'The scene is nominally scaled from an assumed counter height; dimensions are not independently measured.\n\n'
        f'USD: `{source}`\n\nSource scene SHA256: `{receipt["scene_sha256"]}`\n\n'
        f'Scan origin: {origin.tolist()} nominal metres, world +Z up; yaw/pitch/roll zero. '
        f'Nearest collision surface is {closest:.4f} nominal metres away. The scan is an artificial '
        f'{args.azimuth_count}×{args.elevation_count} angular grid covering 360 degrees; '
        'it represents one instantaneous initial scene state and no specific sensor.\n\n'
        'Actual composed USD transforms are applied to collision meshes. Each moving/compound part with '
        '`convexHull` is replaced by its own convex hull, while `none` retains the authored triangles. '
        'Invisible collision patches are included. No whole-object convex hull closes cabinet or sink cavities.\n\n'
        'The glass-included control treats glass as opaque; the second control omits every glass-material collider. '
        'These are ideal limiting cases, not a physical glass response model. PLY colors identify return categories; '
        'the clouds include collider IDs and ray IDs. Full angular grids and valid masks are in `scans.npz`; '
        'zero stored range means no valid return. `colliders.json` maps IDs to USD paths and transforms.\n\n'
        f'Checks: {receipt["status"]}, {sum(c["passed"] for c in checks)}/{len(checks)} passed. '
        f'Glass sensitivity changes {int(changed.sum())}/{len(changed)} rays. '
        'No noise, intensity, motion, beam footprint, return dropout or physical sensor calibration is modeled.\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ideal geometry scan</title>'
        '<style>body{font:18px system-ui;margin:24px;background:#f4f5f7}img{max-width:1440px;width:100%}</style>'
        '<h1>Ideal collision-geometry scan</h1><p>Nominal scene geometry. No real LiDAR validation or deployment-ready map claim.</p>'
        '<img src="comparison.png"><p><a href="receipt.json">Bound receipt</a> · <a href="README.md">Method and limitations</a> · '
        '<a href="glass_included.ply">Glass included PLY</a> · <a href="glass_omitted.ply">Glass omitted PLY</a></p>')
    progress('finished',status=receipt['status'],receipt_sha256=sha(output/'receipt.json'),
             scene_sha256=receipt['scene_sha256'],changed_rays=int(changed.sum()))
    if receipt['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pose',nargs=3,type=float,default=[1.50,-1.75,1.05])
    parser.add_argument('--azimuth-count',type=int,default=720)
    parser.add_argument('--elevation-count',type=int,default=64)
    parser.add_argument('--elevation-min',type=float,default=-35.)
    parser.add_argument('--elevation-max',type=float,default=15.)
    parser.add_argument('--min-range',type=float,default=.05)
    parser.add_argument('--max-range',type=float,default=20.)
    args = parser.parse_args()
    if not (16<=args.azimuth_count<=4096 and 2<=args.elevation_count<=256 and
            -89<args.elevation_min<args.elevation_max<89 and 0<args.min_range<args.max_range<=100 and np.isfinite(args.pose).all()):
        parser.error('Invalid bounded scan parameters')
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'failure.json',{'status':'failed','produced_at':datetime.now(timezone.utc).isoformat(),
                   'error_type':type(error).__name__,'error':str(error),'helper_sha256':sha(__file__)})
        raise
