#!/usr/bin/env python3
"""Export source-informed mesh assets, compound colliders and joints to OpenUSD."""
from pathlib import Path
import hashlib, json, math
import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdPhysics, UsdLux, Vt

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotation_z(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def quat_z(angle):
    return Gf.Quatf(float(math.cos(angle/2)),Gf.Vec3f(0.,0.,float(math.sin(angle/2))))

def axis_rotation(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    return Gf.Quatf(float(math.cos(angle/2)),Gf.Vec3f(*(axis*math.sin(angle/2))))

def rotation_axis(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)*math.cos(angle)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*skew

def initial_body_poses(bodies,joints,initial):
    """Forward kinematics for rest-frame meshes, including nested moving links."""
    origins={n:np.asarray(b['origin'],dtype=float) for n,b in bodies.items()}
    incoming={}
    for j in joints:
        if j['body1'] in incoming:raise ValueError('A link cannot have multiple parent joints')
        incoming[j['body1']]=j
    poses={n:(np.eye(3),o.copy(),Gf.Quatf(1.)) for n,o in origins.items() if n not in incoming}
    remaining=list(joints)
    while remaining:
        progressed=False
        for j in remaining[:]:
            if j['body0'] not in poses:continue
            parent,child=j['body0'],j['body1'];rp,tp,qp=poses[parent]
            q=float(initial.get(j['name'],j.get('initial',0.)))
            anchor=np.asarray(j['anchor']);axis=np.asarray(j['axis'])
            if j['type']=='revolute':
                rc=rp@rotation_axis(axis,q)
                tc=tp+rp@(anchor-origins[parent])+rc@(origins[child]-anchor)
                qc=qp*axis_rotation(axis,q)
            elif j['type']=='prismatic':
                rc=rp;tc=tp+rp@(origins[child]-origins[parent]+axis*q);qc=qp
            else:raise ValueError('Unsupported joint type '+j['type'])
            poses[child]=(rc,tc,qc);remaining.remove(j);progressed=True
        if not progressed:raise ValueError('Joint tree is cyclic or references an unknown parent')
    return poses

class SceneWriter:
    def __init__(self,path,texture_root=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stage=Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage,1.)
        world=UsdGeom.Xform.Define(self.stage,'/World')
        self.stage.SetDefaultPrim(world.GetPrim())
        self.stage.SetTimeCodesPerSecond(60)
        self.stage.GetRootLayer().customLayerData={
          'representation':'Textured mesh assets with compound collisions and articulations',
          'metricScaleStatus':'Nominal counter height assumption; metric dimensions not measured',
          'renderSettings':{'rtx:post:histogram:enabled':False,'rtx:post:tonemap:op':2,
                            'rtx:post:tonemap:enableSrgbToGamma':True,
                            'rtx:post:tonemap:filmIso':200.,
                            'rtx:rendermode':'PathTracing',
                            'rtx:pathtracing:spp':4,'rtx:pathtracing:totalSpp':256,
                            'rtx:pathtracing:maxBounces':6,
                            'rtx:pathtracing:optixDenoiser:enabled':True}}
        physics=UsdPhysics.Scene.Define(self.stage,'/World/PhysicsScene')
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
        self.materials={};self.assets=[];self.joints=[]
        self.texture_root=Path(texture_root) if texture_root else None
        self.make_materials()

    def material(self,name,color,roughness=.45,metallic=0.,texture=None,opacity=1.):
        path='/World/Materials/'+name
        mat=UsdShade.Material.Define(self.stage,path)
        shader=UsdShade.Shader.Define(self.stage,path+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(float(opacity))
        if texture:
            import os
            tex=UsdShade.Shader.Define(self.stage,path+'/Texture')
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(os.path.relpath(texture,self.path.parent)))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            for axis in ('S','T'):tex.CreateInput('wrap'+axis,Sdf.ValueTypeNames.Token).Set('repeat')
            reader=UsdShade.Shader.Define(self.stage,path+'/UV')
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        self.materials[name]=mat
        return mat

    def make_materials(self):
        defaults={
         'maple':((.61,.40,.18),.42,0.),'steel':((.48,.50,.51),.28,1.),
         'chrome':((.72,.73,.74),.16,1.),'white':((.85,.85,.81),.45,0.),
         'black':((.023,.025,.027),.40,0.),'rubber':((.014,.017,.02),.82,0.),
         'floor':((.48,.45,.41),.78,0.),'wall':((.78,.77,.73),.84,0.),
         'trim':((.085,.072,.06),.67,0.),'glass':((.58,.70,.71),.10,0.),
         'blue':((.015,.065,.38),.42,0.),'green':((.035,.16,.085),.58,0.),
         'paper':((.82,.82,.80),.75,0.),'cardboard':((.35,.23,.12),.85,0.)}
        for name,(color,rough,metal) in defaults.items():
            self.material(name,color,rough,metal,opacity=.12 if name=='glass' else 1.)

    def mesh(self,path,part,origin,dynamic=False):
        vertices=np.asarray(part['vertices'],dtype=np.float64)-origin
        faces=part['faces'];counts=[len(f) for f in faces];indices=np.concatenate(faces).astype(np.int32)
        if not len(vertices) or not np.isfinite(vertices).all():raise ValueError('Invalid mesh '+path)
        if len(indices) and (indices.min()<0 or indices.max()>=len(vertices)):raise ValueError('Invalid face indices '+path)
        mesh=UsdGeom.Mesh.Define(self.stage,path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(counts);mesh.CreateFaceVertexIndicesAttr(indices.tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        if part.get('visibility')=='invisible':mesh.CreateVisibilityAttr('invisible')
        # Smooth curved/beveled regions while preserving sharp planar edges.
        # Face-varying normals prevent a shared cube corner from bowing walls.
        fv=np.asarray(faces,dtype=int)
        crosses=np.cross(vertices[fv[:,1]]-vertices[fv[:,0]],vertices[fv[:,2]]-vertices[fv[:,0]])
        face_normals=crosses/np.maximum(np.linalg.norm(crosses,axis=1)[:,None],1e-15)
        adjacent=[[] for _ in vertices]
        for fi,f in enumerate(faces):
            for index in f:adjacent[index].append(fi)
        normals=[]
        for fi,f in enumerate(faces):
            for index in f:
                candidates=np.asarray(adjacent[index],dtype=int)
                aligned=candidates[(face_normals[candidates]@face_normals[fi])>.5]
                n=crosses[aligned].sum(axis=0);n/=max(np.linalg.norm(n),1e-15);normals.append(n)
        normals=np.asarray(normals)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0)),Gf.Vec3f(*vertices.max(0))])
        mat=part.get('material','white')
        uv=part.get('uv')
        if uv is None:
            uv=[];full=np.asarray(part['vertices'],dtype=float)
            for f in faces:
                v=full[f];n=np.cross(v[1]-v[0],v[2]-v[0]);dominant=int(np.argmax(np.abs(n)))
                axes=[i for i in (0,1,2) if i!=dominant]
                q=v[:,axes]
                if mat not in ('floor','trim'):
                    lo=full[:,axes].min(0);size=np.maximum(full[:,axes].max(0)-lo,1e-8);q=(q-lo)/size
                uv.extend(q.tolist())
        uv=np.asarray(uv,dtype=np.float32).reshape(-1,2)
        if len(uv)!=len(indices):raise ValueError('UV must be face-varying '+path)
        primvar=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
        primvar.Set(Vt.Vec2fArray.FromNumpy(uv))
        if mat not in self.materials:raise KeyError('Unknown material '+mat)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.materials[mat])
        mesh.GetPrim().SetCustomDataByKey('assetPart',part['name'])
        if part.get('collision',False):
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            # Author extension attributes explicitly so the portable USD writer
            # does not require Isaac's Python ABI. Isaac registers these schemas.
            mesh.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
            mesh.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.001)
            mesh.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
            # Each mesh is a separately supplied convex piece for moving links.
            # Static concave surfaces retain their triangles; never hull an asset.
            approximation=part.get('collision_approximation','convexHull' if dynamic else 'none')
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
            mesh.GetPrim().SetCustomDataByKey('collisionShapeStatus','Modeled geometry; not measured collision accuracy')
        return mesh

    def add_asset(self,name,spec,position=(0,0,0),yaw=0.,initial=None,locked=None,source_ids=None):
        path='/World/Assets/'+name
        root=UsdGeom.Xform.Define(self.stage,path)
        position=np.asarray(position,dtype=float);rot=rotation_z(yaw)
        root.AddTranslateOp().Set(Gf.Vec3d(*position));root.AddOrientOp().Set(quat_z(yaw))
        bodies=spec['bodies'];joint_list=spec.get('joints',[])
        articulated=bool(joint_list)
        initial={j['name']:float(j.get('initial',0.)) for j in joint_list}|(initial or {})
        locked=set(locked or [])
        body_paths={n:path+'/Links/'+Tf.MakeValidIdentifier(n) for n in bodies}
        body_poses=initial_body_poses(bodies,joint_list,initial)
        body_origins={n:pose[1] for n,pose in body_poses.items()}
        rigid={}
        for n,b in bodies.items():
            p=UsdGeom.Xform.Define(self.stage,body_paths[n]);p.AddTranslateOp().Set(Gf.Vec3d(*body_origins[n]))
            p.AddOrientOp().Set(body_poses[n][2])
            dynamic=articulated or not b.get('static',False)
            rigid[n]=dynamic
            if dynamic:
                UsdPhysics.RigidBodyAPI.Apply(p.GetPrim()).CreateRigidBodyEnabledAttr(True)
                UsdPhysics.MassAPI.Apply(p.GetPrim()).CreateMassAttr(max(.01,float(b.get('mass',1.))))
                p.GetPrim().AddAppliedSchema('PhysxRigidBodyAPI')
                for attr,value in [('solverPositionIterationCount',16),('solverVelocityIterationCount',4)]:
                    p.GetPrim().CreateAttribute('physxRigidBody:'+attr,Sdf.ValueTypeNames.Int).Set(value)
                p.GetPrim().CreateAttribute('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float).Set(2.)
                p.GetPrim().SetCustomDataByKey('massStatus','Modeled estimate, not measured')
        for part in spec['parts']:
            body=part.get('body','base')
            # Rest-pose asset-frame vertices stay relative to the original link
            # origin; the body transform, not vertex baking, supplies articulation.
            self.mesh(body_paths[body]+'/'+Tf.MakeValidIdentifier(part['name']),part,np.asarray(bodies[body]['origin']),rigid[body])
        if articulated:
            bases=[n for n,b in bodies.items() if b.get('static',False)]
            if len(bases)>1:raise ValueError('At most one fixed base per articulated asset: '+name)
            if bases:
                base=bases[0]
                anchor=UsdPhysics.FixedJoint.Define(self.stage,path+'/Joints/FixedBase')
                anchor.CreateBody1Rel().SetTargets([body_paths[base]])
                world_pos=position+rot@body_origins[base]
                anchor.CreateLocalPos0Attr(Gf.Vec3f(*world_pos));anchor.CreateLocalPos1Attr(Gf.Vec3f(0.))
                anchor.CreateLocalRot0Attr(quat_z(yaw));anchor.CreateLocalRot1Attr(Gf.Quatf(1.))
                UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
            else:
                # Mobile carts use a floating articulation rooted at their base.
                UsdPhysics.ArticulationRootAPI.Apply(self.stage.GetPrimAtPath(body_paths['base']))
        for j in joint_list:
            jpath=path+'/Joints/'+Tf.MakeValidIdentifier(j['name']);kind=j['type'];axis=np.asarray(j['axis'],dtype=float)
            idx=int(np.argmax(np.abs(axis)))
            if not np.allclose(np.abs(axis),np.eye(3)[idx]):raise ValueError('Constructor joint axis must be cardinal')
            sign=float(axis[idx]);limits=np.asarray(j['limits'],dtype=float)
            initial_q=float(initial.get(j['name'],0.))
            if j['name'] in locked:limits=np.array([initial_q,initial_q])
            if kind=='revolute':
                joint=UsdPhysics.RevoluteJoint.Define(self.stage,jpath);drive_type='angular';limits=np.degrees(limits)
                target=math.degrees(initial_q)
            elif kind=='prismatic':
                joint=UsdPhysics.PrismaticJoint.Define(self.stage,jpath);drive_type='linear';target=initial_q
            else:raise ValueError('Unsupported joint type '+kind)
            # The USD cardinal axis is positive. Encode any sign in limits/target.
            limits=np.sort(limits*sign);target*=sign
            joint.CreateAxisAttr(('X','Y','Z')[idx])
            joint.CreateLowerLimitAttr(float(limits[0]));joint.CreateUpperLimitAttr(float(limits[1]))
            joint.CreateBody0Rel().SetTargets([body_paths[j['body0']]])
            joint.CreateBody1Rel().SetTargets([body_paths[j['body1']]])
            anchor=np.asarray(j['anchor'],dtype=float)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(anchor-np.asarray(bodies[j['body0']]['origin']))))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(anchor-np.asarray(bodies[j['body1']]['origin']))))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.));joint.CreateLocalRot1Attr(Gf.Quatf(1.))
            joint.CreateCollisionEnabledAttr(False)
            joint.GetPrim().AddAppliedSchema('PhysicsJointStateAPI:'+drive_type)
            joint.GetPrim().CreateAttribute('state:'+drive_type+':physics:position',Sdf.ValueTypeNames.Float).Set(float(target))
            joint.GetPrim().CreateAttribute('state:'+drive_type+':physics:velocity',Sdf.ValueTypeNames.Float).Set(0.)
            drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),drive_type)
            drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(float(target))
            passive=spec.get('metadata',{}).get('asset_type')=='service_cart'
            drive.CreateStiffnessAttr(0. if passive else (80. if kind=='revolute' else 500.))
            drive.CreateDampingAttr(.01 if passive else (15. if kind=='revolute' else 60.))
            drive.CreateMaxForceAttr(120. if kind=='revolute' else 350.)
            joint.GetPrim().SetCustomDataByKey('mechanismStatus','Inferred mechanism; hardware/limits unmeasured')
            joint.GetPrim().SetCustomDataByKey('driveStatus','Nominal position controller for demonstration; disable position stiffness for passive hand manipulation')
            joint.GetPrim().SetCustomDataByKey('sourceRetainerLocked',j['name'] in locked)
            self.joints.append(dict(asset=name,path=jpath,type=kind,body0=body_paths[j['body0']],body1=body_paths[j['body1']],
              lower=float(limits[0]),upper=float(limits[1]),unit='degrees' if kind=='revolute' else 'meters',
              initial=float(target),retainer_locked=j['name'] in locked,passive=passive))
        root.GetPrim().SetCustomDataByKey('sourceInstanceIds',','.join(source_ids or []))
        root.GetPrim().SetCustomDataByKey('geometryStatus','Source-informed modeled geometry; hidden surfaces inferred')
        self.assets.append(dict(name=name,path=path,source_ids=source_ids or [],parts=len(spec['parts']),
          bodies=list(bodies),joints=len(joint_list),position=position.tolist(),yaw_radians=yaw,
          metadata=spec.get('metadata',{})))
        return path

    def add_lighting(self):
        dome=UsdLux.DomeLight.Define(self.stage,'/World/Lights/Ambient')
        dome.CreateIntensityAttr(2250.);dome.CreateColorAttr(Gf.Vec3f(.93,.95,1.))
        for i,(x,y) in enumerate([(0,1),(0,-2),(4,-2),(-1,-1),(4,-5)]):
            light=UsdLux.RectLight.Define(self.stage,f'/World/Lights/Ceiling_{i:02d}')
            light.CreateWidthAttr(1.2);light.CreateHeightAttr(.6)
            light.CreateIntensityAttr(22500.);light.CreateColorAttr(Gf.Vec3f(1.,.99,.97))
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x,y,3.))

    def add_cameras(self,dataset,transform):
        from scene_package import camera_at
        transform=np.asarray(transform,dtype=float)
        scale=float(np.linalg.norm(transform[0,:3]))
        for view in dataset['train']+dataset['val']:
            camera_c2w=np.linalg.inv(np.asarray(view['w2c'])).T@transform
            camera_c2w[:3,:3]/=scale
            new=dict(view,w2c=np.linalg.inv(camera_c2w.T).tolist())
            camera_at(self.stage,'/World/Cameras/'+Path(view['name']).stem,new)

    def save(self):
        self.stage.GetRootLayer().Save()
        manifest=dict(schema='real2sim-physical-scene/v1',scene=self.path.name,scene_sha256=sha(self.path),
           assets=self.assets,joints=self.joints,representation='mesh',gaussian_volumes=0,
           nominal_scale=True,metric_accuracy_verified=False,physics_validation='not_yet_run')
        self.path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return manifest
