"""Assembly-specific corrections and continuous visible shells over convex pieces."""
from collections import Counter
import numpy as np

def shift_link(spec,body,offset):
    offset=np.asarray(offset,dtype=float)
    for p in spec['parts']:
        if p['body']==body:p['vertices']=(np.asarray(p['vertices'])+offset).tolist()
    spec['bodies'][body]['origin']=(np.asarray(spec['bodies'][body]['origin'])+offset).tolist()
    for j in spec['joints']:
        if j['body1']==body:j['anchor']=(np.asarray(j['anchor'])+offset).tolist()

def continuous_shell(spec,prefix):
    selected=[p for p in spec['parts'] if p['name'].startswith(prefix+'_')]
    if not selected:return
    # Preserve each thin convex patch for contact, but render one welded outer
    # surface with continuous UV coordinates and shared smoothing vertices.
    vertices=[];quads=[];known={}
    for p in selected:
        remap=[]
        for xyz in p['vertices']:
            key=tuple(np.round(xyz,9))
            if key not in known:known[key]=len(vertices);vertices.append(xyz)
            remap.append(known[key])
        triangles=[[remap[k] for k in f] for f in p['faces']]
        quads.extend([triangles[i:i+2] for i in range(0,len(triangles),2)])
        p['visibility']='invisible'
    keys=[tuple(sorted({v for f in quad for v in f})) for quad in quads]
    counts=Counter(keys)
    faces=[face for key,quad in zip(keys,quads) if counts[key]==1 for face in quad]
    v=np.asarray(vertices);f=np.asarray(faces)
    # Along the bent seat/back rather than resetting at every collision patch.
    u=(v[:,0]-v[:,0].min())/np.ptp(v[:,0])
    along=v[:,1]+v[:,2];t=(along-along.min())/np.ptp(along)
    uv=np.c_[u,t][f]
    spec['parts'].append(dict(name=prefix+'_continuous_visual',vertices=vertices,
        faces=faces,uv=uv.tolist(),material=selected[0]['material'],body=selected[0]['body'],collision=False))
    spec['metadata']['visible_shell']='Welded union of the same thin collision patches; continuous UVs'

def polish(name,spec,library):
    kind=spec.get('metadata',{}).get('asset_type')
    # Replace differently oriented sweep rings with a genuinely convex straight
    # capsule. Rounded ends join neighboring sections without pinched surfaces.
    for index,p in enumerate(spec['parts']):
        if '_segment_' not in p['name'] or not p.get('collision'):continue
        v=np.asarray(p['vertices']);sides=(len(v)-2)//2
        if sides<6 or 2*sides+2!=len(v):continue
        start,end=v[-2:];direction=end-start;length=np.linalg.norm(direction)
        if length<1e-9:continue
        radius=float(np.median(np.linalg.norm(v[:sides]-start,axis=1)))
        spec['parts'][index]=library.capsule(radius,length,(start+end)/2,direction,
            name=p['name'],body=p['body'],material=p['material'],segments=sides,rings=3)
    if kind=='maple_sled_stool':continuous_shell(spec,'maple_shell')
    if kind=='black_shell_chair':continuous_shell(spec,'black_shell')
    if kind=='cabinet':
        top=next((p for p in spec['parts'] if p['name']=='case_top'),None)
        if top:
            underside=float(np.asarray(top['vertices'])[:,2].min())-.004
            for p in spec['parts']:
                if not p['body'].startswith('drawer_') or not ('_side_' in p['name'] or p['name'].endswith('_back')):continue
                v=np.asarray(p['vertices']);lo,hi=v[:,2].min(),v[:,2].max()
                if hi>underside:v[:,2]=lo+(v[:,2]-lo)*(underside-lo)/(hi-lo)
                p['vertices']=v.tolist()
    if kind=='french_door_refrigerator':
        # The front and joint anchor retain their photographed placement. Only
        # the hidden box fits within the actual lower compartment boundaries.
        base=np.asarray(next(p['vertices'] for p in spec['parts'] if p['name']=='case_bottom'))
        divider=np.asarray(next(p['vertices'] for p in spec['parts'] if p['name']=='upper_compartment_floor'))
        floor_low=float(base[:,2].max())+.004
        side_high=float(divider[:,2].min())-.004
        for p in spec['parts']:
            if p['body']!='freezer_drawer':continue
            v=np.asarray(p['vertices']);lo,hi=v[:,2].min(),v[:,2].max()
            if p['name']=='freezer_drawer_floor':v[:,2]+=floor_low-lo
            elif '_side_' in p['name'] or p['name']=='freezer_drawer_back':
                v[:,2]=floor_low+.012+(v[:,2]-lo)/(hi-lo)*(side_high-floor_low-.012)
            elif '_runner_' in p['name']:v[:,2]+=floor_low+.016-lo
            p['vertices']=v.tolist()
        spec['metadata']['freezer_interior_clearance_m']=.004
    if kind=='microwave':
        shift_link(spec,'door',[0,-.008,0])
        spec['metadata']['door_clearance_m']=.003
    if kind=='cooler':
        shift_link(spec,'lid',[0,0,.005])
        spec['metadata']['lid_clearance_m']=.0025
    for j in spec['joints']:
        if j['type']!='revolute' or not np.allclose(j['axis'],[0,0,1]):continue
        body=j['body1'];leaf=next((p for p in spec['parts'] if p['body']==body and p['name'].endswith('_leaf')),None)
        if leaf is None:continue
        y=np.asarray(leaf['vertices'])[:,1]
        target=float(y.max()+.0002) if body.startswith('back_door_') else float(y.min()-.0002)
        # An approximate concealed hinge axis at the front face leaves the
        # closed silhouette intact and avoids sweeping thickness into its neighbor.
        j['anchor'][1]=target;spec['bodies'][body]['origin'][1]=target
        spec['metadata']['hinge_axis_status']='Front-face pivot approximation; hidden mechanism is unmeasured'
    if name.startswith('upper_'):
        for body in spec['bodies']:
            parts=[p for p in spec['parts'] if p['name'].startswith(body+'_pull')]
            if not parts:continue
            z=np.concatenate([np.asarray(p['vertices']) for p in parts])[:,2]
            dz=.12-(z.min()+z.max())/2
            for p in parts:p['vertices']=(np.asarray(p['vertices'])+[0,0,dz]).tolist()
    return spec
