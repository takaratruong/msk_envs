#!/usr/bin/env python3
"""Build a task-specific MuJoCo contact scene from the physical USD colliders.

Unused room articulations are held at their authored poses in this adapter.
The selected fridge leaf is passive and the added bottle is a free body.
The source USD retains every articulation and is never edited here.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, xml.etree.ElementTree as ET
import numpy as np
import mujoco, trimesh
from scipy.optimize import linprog
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from g1_motion_scene import triangles


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def nums(x): return ' '.join(format(float(v), '.12g') for v in x)


def collision_chunks(mesh,path,approximation):
    """Preserve the known beveled through-aperture with exact convex pieces."""
    hull=mesh.convex_hull
    if approximation!='none' or abs(hull.volume-mesh.volume)<1e-7:
        return [mesh]
    if not path.endswith('/island_countertop/Links/base/top_right') or len(mesh.vertices)!=64:
        # Small warped shell cells can be concave without a through-hole. Find
        # an interior kernel point and fan the original boundary triangles.
        mesh=mesh.copy();mesh.fix_normals();normal=mesh.face_normals
        limits=np.einsum('ij,ij->i',normal,mesh.vertices[mesh.faces[:,0]])
        result=linprog([0,0,0,-1],A_ub=np.c_[normal,np.ones(len(normal))],b_ub=limits,
                        bounds=[*zip(mesh.bounds[0],mesh.bounds[1]),(0,None)],method='highs')
        if not result.success:raise ValueError('Non-star-shaped triangle collider needs a dedicated decomposition: '+path)
        center=result.x[:3];tetvol=np.linalg.det(mesh.vertices[mesh.faces]-center)/6
        if np.any(tetvol<-1e-12):raise ValueError('Invalid shell kernel '+path)
        parts=[trimesh.convex.convex_hull(np.vstack([center,mesh.vertices[f]]))
               for f,v in zip(mesh.faces,tetvol) if v>1e-12]
        if abs(sum(p.volume for p in parts)-mesh.volume)>1e-7:raise ValueError('Shell fan volume differs '+path)
        return parts
    # physical_countertop.slab's four outer eight-vertex rings then four inner
    # rings. A full sector hull slightly fills the inner bevel, so split at all
    # height rings and tetrahedralize any nonconvex cell about its interior.
    parts=[]
    quads=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
    faces=np.array([[q[0],q[1],q[2]] for q in quads]+[[q[0],q[2],q[3]] for q in quads])
    for level in range(3):
        for k in range(8):
            ids=[side*32+lev*8+corner for side in range(2) for lev in [level,level+1] for corner in [k,(k+1)%8]]
            vertices=mesh.vertices[ids]
            cell=trimesh.Trimesh(vertices,faces.copy(),process=False);cell.fix_normals()
            if abs(cell.convex_hull.volume-cell.volume)<1e-10:
                parts.append(cell);continue
            center=vertices.mean(0)
            signed=np.linalg.det(vertices[cell.faces]-center)/6
            if np.any(signed<-1e-12):raise ValueError('Cell is not star shaped')
            for face,vol in zip(cell.faces,signed):
                if vol>1e-12:parts.append(trimesh.convex.convex_hull(np.vstack([center,vertices[face]])))
    error=abs(sum(abs(x.volume) for x in parts)-abs(mesh.volume))
    if error>1e-7:raise ValueError('Slab sector decomposition changes volume: '+str(error))
    return parts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--robot',type=Path,required=True);ap.add_argument('--task',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'scene.xml').exists():raise FileExistsError('Use a new candidate directory')
    task=json.loads(a.task.read_text());s=Usd.Stage.Open(str(a.scene.resolve()))
    assert UsdGeom.GetStageMetersPerUnit(s)==1 and UsdGeom.GetStageUpAxis(s)=='Z'
    assert sha(a.scene)==task['scene']['sha256'],'Task scene hash differs'
    assert sha(a.robot)==task['robot']['sha256'],'Task robot hash differs'
    for binding in task['robot']['asset_files']:
        assert sha(binding['path'])==binding['sha256'],'Bound robot asset changed'
    tree=ET.parse(a.robot);root=tree.getroot();assert not root.findall('.//include')
    compiler=root.find('compiler');meshdir=compiler.get('meshdir','')
    source_assets=[]
    for node in root.findall('asset/mesh'):
        src=(a.robot.parent/meshdir/node.get('file')).resolve();dst=a.output/'robot_meshes'/src.name
        dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst);node.set('file',str(dst.relative_to(a.output)))
        source_assets.append({'path':str(src),'sha256':sha(src),'copy':str(dst.relative_to(a.output))})
    compiler.set('meshdir','.');compiler.set('balanceinertia','true')
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.attrib.update(timestep='0.002',gravity='0 0 -9.81',integrator='implicitfast',
                         solver='Newton',iterations='60',tolerance='1e-9',cone='elliptic')
    asset=root.find('asset');world=root.find('worldbody')
    ET.SubElement(world,'light',pos='0 -1 5',dir='0 0 -1',diffuse='.8 .8 .8')
    # Existing robot has no world ground. Fail rather than silently duplicate it.
    assert not world.findall('geom')
    cache=UsdGeom.XformCache();door_path=task['door']['body_path']
    pivot=np.asarray(task['door']['world_pivot']);door_prim=s.GetPrimAtPath(door_path)
    joint=UsdPhysics.RevoluteJoint(s.GetPrimAtPath(task['door']['joint_path']))
    assert str(joint.GetBody1Rel().GetTargets()[0])==door_path and joint.GetAxisAttr().Get()=='Z'
    parent=s.GetPrimAtPath(str(joint.GetBody0Rel().GetTargets()[0]))
    T=np.asarray(cache.GetLocalToWorldTransform(parent))
    actual_pivot=(np.r_[np.asarray(joint.GetLocalPos0Attr().Get()),1.]@T)[:3]
    assert np.allclose(pivot,actual_pivot,atol=1e-7),'Task hinge pivot differs from USD'
    assert np.allclose(task['door']['limits_degrees'],[joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()])
    mass=float(UsdPhysics.MassAPI(door_prim).GetMassAttr().Get())
    door=ET.SubElement(world,'body',name='task_fridge_door',pos=nums(pivot))
    ET.SubElement(door,'joint',name='task_fridge_hinge',type='hinge',axis='0 0 1',
                  range=nums(np.deg2rad(task['door']['limits_degrees'])),limited='true',
                  damping='0.15',frictionloss='0.08',armature='0.003')
    records=[];door_parts=[];meshpath=a.output/'room_meshes';meshpath.mkdir(exist_ok=True)
    for prim in s.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        path=str(prim.GetPath());mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),float)
        T=np.asarray(cache.GetLocalToWorldTransform(prim));v=(np.c_[v,np.ones(len(v))]@T)[:,:3]
        f=triangles(mesh);moving=path.startswith(door_path+'/');origin=pivot if moving else v.mean(0)
        tm=trimesh.Trimesh(v-origin,f,process=False)
        approximation=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        chunks=collision_chunks(tm,path,approximation)
        color=[.65,.64,.61];material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            shader=UsdShade.Shader(s.GetPrimAtPath(str(material.GetPath())+'/Surface'))
            inp=shader.GetInput('diffuseColor')
            if inp and inp.Get() is not None:color=list(inp.Get())
        for sector,part in enumerate(chunks):
            hull=part.convex_hull;vol=abs(float(part.volume));hv=abs(float(hull.volume))
            if hv<1e-12:raise ValueError('Degenerate collider '+path)
            name='room_'+str(len(records)).zfill(4);dst=meshpath/(name+'.obj')
            part.export(dst);ET.SubElement(asset,'mesh',name=name,file=str(dst.relative_to(a.output)))
            node=ET.SubElement(door if moving else world,'geom',name=name,type='mesh',mesh=name,
                      pos='0 0 0' if moving else nums(origin),rgba=nums(color+[1.]),
                      contype='1',conaffinity='1',friction='0.8 0.005 0.0001',
                      condim='4',group='2',margin='0',solref='0.008 1',mass='0')
            records.append(dict(name=name,usd_path=path,selected_door=moving,sha256=sha(dst),
                     vertices=len(part.vertices),triangles=len(part.faces),source_volume=vol,convex_volume=hv,
                     source_approximation=approximation,decomposition_sector=sector,decomposition_count=len(chunks),
                     bounds=[v.min(0).tolist(),v.max(0).tolist()]))
            if moving:door_parts.append((node,hv))
    for node,vol in door_parts:node.set('mass',str(mass*vol/sum(x[1] for x in door_parts)))
    obj=task['object'];base=np.asarray(obj['initial_base_world']);center=base+np.array([0,0,obj['height']/2])
    bottle=ET.SubElement(world,'body',name='task_bottle',pos=nums(center))
    ET.SubElement(bottle,'freejoint',name='task_bottle_free')
    shape=obj.get('shape','compound_bottle')
    if shape not in ('compound_bottle','cylinder'):raise ValueError('Unsupported task prop shape')
    if shape=='cylinder':
        radius=float(obj['radius'])
        if not np.isfinite([radius,obj['height'],obj['mass']]).all() or min(radius,obj['height'],obj['mass'])<=0:
            raise ValueError('Cylinder dimensions and mass must be positive and finite')
        # One continuous collider avoids artificial edges inside the grasp region.
        pieces=[('body',radius,obj['height']/2,0.,1.)]
    else:
        pieces=[('body',obj['body_radius'],obj['body_height']/2,
                 obj['body_height']/2-obj['height']/2,.80),
                ('neck',obj['neck_radius'],obj['neck_height']/2,
                 obj['height']/2-obj['neck_height']/2,.08)]
    for name,radius,halfheight,z,fraction in pieces:
        ET.SubElement(bottle,'geom',name='task_bottle_'+name,type='cylinder',size=nums([radius,halfheight]),
            pos=nums([0,0,z]),rgba='.12 .40 .75 1',mass=str(obj['mass']*fraction),
            friction=nums([obj['friction'],.005,.0001]),condim='4',solref='0.008 1')
    if shape=='compound_bottle':
        n=32;angles=np.arange(n)*2*np.pi/n;verts=[]
        for z,r in [(obj['body_height']-obj['height']/2,obj['body_radius']),
                    (obj['height']/2-obj['neck_height'],obj['neck_radius'])]:
            verts.extend(np.c_[r*np.cos(angles),r*np.sin(angles),np.full(n,z)])
        shoulder=trimesh.convex.convex_hull(np.asarray(verts));dst=a.output/'bottle_shoulder.obj';shoulder.export(dst)
        ET.SubElement(asset,'mesh',name='task_bottle_shoulder_mesh',file=dst.name)
        ET.SubElement(bottle,'geom',name='task_bottle_shoulder',type='mesh',mesh='task_bottle_shoulder_mesh',
            rgba='.12 .40 .75 1',mass=str(obj['mass']*.12),friction=nums([obj['friction'],.005,.0001]),
            condim='4',solref='0.008 1')
    ET.indent(tree,space='  ');xml=a.output/'scene.xml';tree.write(xml,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    receipt=dict(schema='g1-task-mujoco-adapter/v1',scene=dict(path=str(a.scene.resolve()),sha256=sha(a.scene)),
        scene_layers=[dict(path=x.realPath,sha256=sha(x.realPath)) for x in s.GetUsedLayers() if x.realPath],
        robot=dict(path=str(a.robot.resolve()),sha256=sha(a.robot),assets=source_assets),
        task=dict(path=str(a.task.resolve()),sha256=sha(a.task)),script_sha256=sha(__file__),
        output=dict(path=str(xml.resolve()),sha256=sha(xml)),nq=model.nq,nv=model.nv,nu=model.nu,
        room_collision_parts=len(records),source_collision_parts=len(set(r['usd_path'] for r in records)),door_collision_parts=len(door_parts),room=records,
        door_mass_kg=mass,door_damping=.15,door_frictionloss=.08,passive_door=True,
        door_drive_replacement='USD pose-holding servo removed; nominal passive hinge resistance, unmeasured',
        bottle=obj,unused_room_articulations='Held at authored rest poses only in this task-specific adapter',
        object_geometry=dict(shape=shape,geom_names=[model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]==model.body('task_bottle').id],
                             mass_kg=float(model.body('task_bottle').mass[0]),inertia_kg_m2=model.body('task_bottle').inertia.tolist()),
        source_usd_articulations='Unchanged',physics_executed=False,
        assisted_constraints=int(model.neq),world_base_stabilizers=False)
    (a.output/'scene.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['nq','nv','nu','room_collision_parts','door_collision_parts','door_mass_kg']}))


if __name__=='__main__':main()
