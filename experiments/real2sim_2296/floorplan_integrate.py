#!/usr/bin/env python3
"""Integrate evidence-backed static layout changes into a portable mesh scene."""
import argparse, json, hashlib, math
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf, Vt
from g1_motion_usd import environment_copy
from physical_usd import SceneWriter
from physical_assets import box as primitive_box


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def box(size,center=(0,0,0),**kwargs):
    material=kwargs.pop('material','white')
    part=primitive_box(size,center,material='white',**kwargs)
    part['material']=material
    return part
def spec(parts):
    return dict(parts=parts,bodies={'base':dict(origin=[0,0,0],mass=1.,static=True)},joints=[],
                metadata=dict(asset_type='source_supported_layout_revision',dimensions_measured=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cooler-proposal',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'scene.usda').exists():raise FileExistsError('Use a new iteration')
    proposal=json.loads((a.evidence/'coherent_junction_proposal.json').read_text())
    corridor=json.loads((a.evidence/'corridor_patch.json').read_text())
    fits=json.loads((a.evidence/'fit_results.json').read_text())
    assert sha(a.source)==proposal['baseline_sha256']
    _,copied,dependencies=environment_copy(a.source.resolve(),a.output)
    s=Usd.Stage.Open(str(copied));writer=object.__new__(SceneWriter)
    writer.path=copied;writer.stage=s;writer.assets=[];writer.joints=[]
    writer.materials={x.GetName():UsdShade.Material(x) for x in s.GetPrimAtPath('/World/Materials').GetChildren()}
    changed=[];ledger=[]
    def add(name,parts,position,yaw=0.,reason=''):
        old=s.GetPrimAtPath('/World/Assets/'+name)
        if old:
            assert not any(x.IsA(UsdPhysics.Joint) for x in Usd.PrimRange(old)), 'Never remesh articulated asset'
            s.RemovePrim(old.GetPath())
        writer.add_asset(name,spec(parts),position,yaw)
        changed.append(name);ledger.append(dict(asset=name,position=list(position),yaw=yaw,reason=reason))
    def glass(name,aa,bb,reason):
        aa=np.asarray(aa);bb=np.asarray(bb);length=float(np.linalg.norm(bb-aa));yaw=math.atan2(*(bb-aa)[::-1])
        parts=[box((length,.018,2.84),(length/2,0,1.48),name='glazing',material='glass',bevel=0),
               box((length,.06,.10),(length/2,0,.06),name='base',material='black',bevel=0)]
        # Preserve thickness and spacing instead of scaling bars with wall length.
        for i,x in enumerate(np.arange(.025,length-.025,1.35)):
            parts.append(box((.045,.055,3.1),(x,0,1.55),name=f'mullion_{i}',material='steel',bevel=.001))
        for i,z in enumerate([.83,.91,1.]):
            parts.append(box((length,.020,.015),(length/2,0,z),name=f'frosted_stripe_{i}',material='white',bevel=0,collision=False))
        add(name,parts,[*aa,0.],yaw,reason)
        ledger[-1].update(endpoints=[aa.tolist(),bb.tolist()],length=length)
    c=proposal['column'];w,d,h=c['width'],c['depth'],c['height']
    add(c['asset'],[box((w,d,h),(0,0,h/2),name='column',material='wall',bevel=.008),
         box((w+.015,d+.015,.12),(0,0,.06),name='plinth',material='black')],
        [*c['center_xy'],0.],c['yaw'],'Static column crosssection fitted to visible base corners; height remains nominal')
    if a.cooler_proposal:
        cooler=json.loads(a.cooler_proposal.read_text());x,y,yaw,scale=cooler['parameters']
        root=s.GetPrimAtPath('/World/Assets/cooler_01')
        root.GetAttribute('xformOp:translate').Set(Gf.Vec3d(x,y,0))
        root.GetAttribute('xformOp:orient').Set(Gf.Quatf(math.cos(yaw/2),Gf.Vec3f(0,0,math.sin(yaw/2))))
        # Bake isotropic size into geometry and anchor coordinates. No scale
        # transform is inherited by the physics articulation.
        for prim in Usd.PrimRange(root):
            if prim==root:continue
            for attr in prim.GetAuthoredAttributes():
                n=attr.GetName();value=attr.Get()
                if n in ['xformOp:translate','physics:localPos0','physics:localPos1']:
                    attr.Set(type(value)(*map(float,np.asarray(value)*scale)))
                elif n=='physics:mass':attr.Set(float(value)*scale**3)
            if prim.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)*scale
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
                mesh.GetExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))])
        root.SetCustomDataByKey('layoutUniformSizeRatio',float(scale))
        root.SetCustomDataByKey('layoutSizeStatus','Fitted source floor corners with separate-view checks; unmeasured uniform size and nominal density')
        changed.append('cooler_01');ledger.append(dict(asset='cooler_01',position=[x,y,0],yaw=yaw,uniform_size_ratio=scale,
            reason='Correct observed facing and footprint, clear fitted glazing; all lid geometry/anchors scaled consistently, nominal mass follows volume',
            proposal_sha256=sha(a.cooler_proposal)))
    for line in proposal['glass_segments']:
        aa=np.array(line['a_xy']);bb=np.array(line['b_xy'])
        # A quarter-metre completion beyond the observed fitting segment is
        # explicitly unmeasured, allowing the pane to meet the nominal room shell.
        if 'sink_end' in line['asset']:bb+=.25*(bb-aa)/np.linalg.norm(bb-aa)
        glass(line['asset'],aa,bb,line['far_span_status']+'; sink far span includes 0.25 m unmeasured completion')
    back=np.asarray(fits['glass_back_line']['candidate_segment'])[:,:2]
    direction=(back[1]-back[0]);direction/=np.linalg.norm(direction)
    # Source fitting view explicitly sees glass at x=-2.55; cover that observed
    # section plus 0.10 m margin without asserting a physical endpoint.
    end=back[0]+((-2.65-back[0,0])/direction[0])*direction
    glass('glass_partition_back',back[0],end,'Line fitted across views; near x=-2.65 covers source fitting controls, far span inherited; endpoints not measured')
    near=np.array(corridor['near_floor_endpoint']);far=np.array(corridor['far_visible_patch_endpoint'])
    vec=near-far;length=float(np.linalg.norm(vec[:2]));yaw=math.atan2(vec[1],vec[0]);h=corridor['nominal_wall_height'];th=corridor['thickness_m']
    add('observed_corridor_wall_patch',[
        box((length,th,h),(length/2,-th/2,h/2),name='wall',material='wall',bevel=0),
        box((length,.018,.10),(length/2,.009,.05),name='skirting',material='black',bevel=0)],
        far.tolist(),yaw,'Observed missing surface patch; remote truncation is not a corridor termination; no transverse closure')
    # Extend existing open support/lighting canvases to cover the observed patch.
    # These large rectangles are explicitly not a recovered closed floor plan.
    for name in ['floor_surface_region_01','ceiling_boundary']:
        root=s.GetPrimAtPath('/World/Assets/'+name)
        if not root:continue
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),float)
            # Mesh09 canvas is x[-5,+11]. Its left edge moves to -8.7
            # without changing the other edges or the physical floor height.
            lo=v[:,0].min();mask=np.isclose(v[:,0],lo,atol=1e-5);v[mask,0]-=3.7
            mesh.GetPointsAttr().Set([Gf.Vec3f(*x) for x in v]);mesh.GetExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))])
            if name=='floor_surface_region_01':
                indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Set(Vt.Vec2fArray.FromNumpy((v[indices,:2]/.61).astype(np.float32)))
        changed.append(name);ledger.append(dict(asset=name,reason='Open completion canvas extended 3.7 m toward negative X to support observed corridor; not a measured boundary'))
    s.GetRootLayer().customLayerData=dict(s.GetRootLayer().customLayerData)|{
        'layoutRevision':'Astra multiview planes, coherent column junction and observed corridor patch',
        'metricScaleStatus':'Nominal counter height assumption; no surveyed metric dimensions',
        'perimeterStatus':'Open, incomplete observed boundaries; rectangular floor/ceiling are completion canvases'}
    s.GetRootLayer().Save()
    wrapper=Usd.Stage.CreateNew(str(a.output/'scene.usda'))
    wrapper.GetRootLayer().subLayerPaths=[str(copied.relative_to(a.output))]
    UsdGeom.SetStageUpAxis(wrapper,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(wrapper,1.)
    wrapper.SetTimeCodesPerSecond(s.GetTimeCodesPerSecond());wrapper.SetFramesPerSecond(s.GetFramesPerSecond())
    wrapper.SetDefaultPrim(wrapper.GetPrimAtPath('/World'));wrapper.GetRootLayer().Save()
    src=Usd.Stage.Open(str(a.source));sc=UsdGeom.XformCache();cc=UsdGeom.XformCache()
    invariant_errors=[];checked=0
    for prim in src.Traverse():
        path=str(prim.GetPath());asset=path.split('/')[3] if path.startswith('/World/Assets/') else ''
        if asset in changed:continue
        cp=s.GetPrimAtPath(path)
        if not cp:invariant_errors.append('missing '+path);continue
        if prim.IsA(UsdGeom.Xformable):
            if not np.allclose(np.asarray(sc.GetLocalToWorldTransform(prim)),np.asarray(cc.GetLocalToWorldTransform(cp)),atol=1e-10):invariant_errors.append('transform '+path)
        for attr in prim.GetAuthoredAttributes():
            av=attr.Get();bv=cp.GetAttribute(attr.GetName()).Get()
            if isinstance(av,Sdf.AssetPath):
                same=av.path==bv.path and sha(av.resolvedPath)==sha(bv.resolvedPath)
            else:same=repr(av)==repr(bv)
            if not same:invariant_errors.append('attribute '+path+'.'+attr.GetName())
        checked+=1
    # Asset-valued attrs resolve to the copied files; compare their source values
    # above, and hash every transitive payload below for a portable receipt.
    collisions=[str(x.GetPath()) for x in s.Traverse() if x.IsA(UsdGeom.Mesh) and x.HasAPI(UsdPhysics.CollisionAPI)]
    joints=[str(x.GetPath()) for x in s.Traverse() if x.IsA(UsdPhysics.Joint)]
    source_joints=[str(x.GetPath()) for x in src.Traverse() if x.IsA(UsdPhysics.Joint)]
    assert joints==source_joints
    if invariant_errors:raise RuntimeError('Unexpected invariant changes '+repr(invariant_errors[:10]))
    payload=[dict(path=str(x.relative_to(a.output)),sha256=sha(x)) for x in sorted(a.output.rglob('*')) if x.is_file()]
    receipt=dict(schema='real2sim-floorplan-integration/v1',source=dict(path=str(a.source.resolve()),sha256=sha(a.source)),
        script_sha256=sha(__file__),evidence={x.name:sha(x) for x in sorted(a.evidence.glob('*.json'))},
        cooler_proposal=None if not a.cooler_proposal else dict(path=str(a.cooler_proposal.resolve()),sha256=sha(a.cooler_proposal)),
        scene=dict(path=str((a.output/'scene.usda').resolve()),sha256=sha(a.output/'scene.usda')),
        changed_assets=changed,changes=ledger,payload=payload,unchanged_prims_checked=checked,
        unchanged_prim_errors=invariant_errors,all_joint_paths_preserved=True,joints=len(joints),colliders=len(collisions),
        native_render_checked=False,collision_clearance_checked=False,metric_accuracy_verified=False)
    (a.output/'integration.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['scene','changed_assets','joints','colliders','unchanged_prims_checked']}))


if __name__=='__main__':main()
