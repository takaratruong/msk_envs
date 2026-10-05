#!/usr/bin/env python3
"""Source-inspired, parameterized articulated mesh assets for IMG_2296.MOV.

Public constructors return ordinary JSON-serializable dictionaries. Nominal metres,
asset x right/y back/z up, x centred, front y=0. Cabinet/appliance z=0 is the floor;
sink z=0 is its rim. Revolute joint values are RADIANS; prismatic values are metres.
Every vertex and joint anchor is in the ASSET frame, including moving parts.
Bodies supply their closed-pose origins; consumers subtract these origins before
attaching geometry to rigid bodies. Dimensions, interiors, axes, limits and masses
are explicit engineering assumptions, not measurements from the RGB video.

Only numpy is needed to construct assets. `--preview DIR` adds Pillow for isolated
diagnostic previews; this module never places or exports the reconstructed scene.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'real2sim-physical-asset/v1'
MATERIAL_KEYS = ('maple', 'steel', 'chrome', 'white', 'black', 'glass', 'floor', 'rubber')
SOURCE_SHA256 = '282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9'


def _vec(value):
    return np.asarray(value, dtype=float)


def _unit(value):
    value = _vec(value)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError('A direction must have nonzero length.')
    return value / norm


def _uv_planar(vertices, faces):
    vertices = _vec(vertices)
    lo, span = vertices.min(axis=0), np.maximum(np.ptp(vertices, axis=0), 1e-8)
    uv = []
    for face in faces:
        p = vertices[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        drop = int(np.argmax(np.abs(normal)))
        axes = ((1, 2), (0, 2), (0, 1))[drop]
        uv.append(((p[:, axes] - lo[list(axes)]) / span[list(axes)]).tolist())
    return uv


def _box_mesh(size, center, bevel=.002):
    """Closed rounded cuboid; each face grid rounds into a common inner box."""
    half = _vec(size) / 2
    if np.any(half <= 0):
        raise ValueError('All box sizes must be positive.')
    bevel = min(max(float(bevel), 0), float(half.min()) * .8)
    if bevel < 1e-8:
        vertices = [[sx * half[0], sy * half[1], sz * half[2]]
                    for sx, sy, sz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                                      (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        quads = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        faces = [tri for q in quads for tri in ([q[0],q[1],q[2]],[q[0],q[2],q[3]])]
        return _vec(vertices) + _vec(center), faces
    vertices, faces, known = [], [], {}
    def vertex(p):
        inner = np.clip(p, -half + bevel, half - bevel)
        rounded = inner + bevel * _unit(p - inner)
        key = tuple(np.round(rounded, 10))
        if key not in known:
            known[key] = len(vertices)
            vertices.append(rounded + _vec(center))
        return known[key]
    values = [[-h, -h + .3 * bevel, -h + bevel, h - bevel, h - .3 * bevel, h] for h in half]
    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1, 1):
            grid = []
            for va in values[a]:
                row = []
                for vb in values[b]:
                    p = np.zeros(3); p[axis] = sign * half[axis]; p[a] = va; p[b] = vb
                    row.append(vertex(p))
                grid.append(row)
            for i in range(5):
                for j in range(5):
                    q = [grid[i][j], grid[i+1][j], grid[i+1][j+1], grid[i][j+1]]
                    if sign < 0:
                        q.reverse()
                    faces += [[q[0],q[1],q[2]], [q[0],q[2],q[3]]]
    return _vec(vertices), faces


def _tube_mesh(points, radius, sides=12, cap=True):
    points = _vec(points)
    if len(points) < 2 or radius <= 0:
        raise ValueError('Tube requires two distinct points and positive radius.')
    tangent = np.empty_like(points)
    tangent[0], tangent[-1] = points[1] - points[0], points[-1] - points[-2]
    tangent[1:-1] = points[2:] - points[:-2]
    tangent = np.array([_unit(t) for t in tangent])
    ref = np.array([0.,0.,1.]) if abs(tangent[0,2]) < .9 else np.array([1.,0.,0.])
    normal = _unit(np.cross(tangent[0], ref))
    vertices, uv = [], []
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    for i, (point, t) in enumerate(zip(points, tangent)):
        normal = _unit(normal - t * np.dot(normal, t))
        binormal = np.cross(t, normal)
        for j in range(sides):
            theta = j * 2 * math.pi / sides
            vertices.append(point + radius * (normal * math.cos(theta) + binormal * math.sin(theta)))
    faces = []
    for i in range(len(points) - 1):
        for j in range(sides):
            k = (j + 1) % sides
            a,b,c,d = i*sides+j,i*sides+k,(i+1)*sides+k,(i+1)*sides+j
            faces += [[a,b,c],[a,c,d]]
            uv += [[[j/sides,length[i]],[((j+1)/sides),length[i]],[((j+1)/sides),length[i+1]]],
                   [[j/sides,length[i]],[((j+1)/sides),length[i+1]],[j/sides,length[i+1]]]]
    if cap:
        for end, outward in ((0,-1),(len(points)-1,1)):
            center = len(vertices); vertices.append(points[end])
            for j in range(sides):
                q = [center,end*sides+j,end*sides+(j+1)%sides]
                if outward < 0:
                    q.reverse()
                faces.append(q); uv.append([[.5,.5],[0,0],[1,0]])
    return _vec(vertices), faces, uv


def _rounded_loop(width, depth, radius, z=0, center=(0,0), segments=8):
    radius = min(radius, width*.49, depth*.49)
    points = []
    for cx,cy,start in ((width/2-radius,depth/2-radius,0),(-width/2+radius,depth/2-radius,90),
                        (-width/2+radius,-depth/2+radius,180),(width/2-radius,-depth/2+radius,270)):
        for theta in np.linspace(start,start+90,segments,endpoint=False):
            a=math.radians(theta)
            points.append([cx+radius*math.cos(a)+center[0],cy+radius*math.sin(a)+center[1],z])
    return _vec(points)


def _loops_mesh(loops, close=False):
    vertices = np.concatenate(loops)
    count = len(loops[0])
    if any(len(loop) != count for loop in loops):
        raise ValueError('Corresponding mesh loops must have the same vertex count.')
    faces = []
    for i in range(len(loops) if close else len(loops)-1):
        ni=(i+1)%len(loops)
        for j in range(count):
            nj=(j+1)%count
            a,b,c,d=i*count+j,i*count+nj,ni*count+nj,ni*count+j
            faces += [[a,b,c],[a,c,d]]
    return vertices,faces


class _Asset:
    def __init__(self, kind, dimensions, evidence, static=True, mass=0):
        self.spec = {'schema':SCHEMA, 'parts':[], 'bodies':{}, 'joints':[], 'metadata':{
            'asset_type':kind,'dimensions_nominal_m':dimensions,'source_sha256':SOURCE_SHA256,
            'source_evidence':evidence,'coordinate_system':'x right, y back, z up; x centred; front y=0',
            'vertices_frame':'asset','joint_anchor_frame':'asset','body_origins':'closed pose in asset frame',
            'joint_angle_units':'radians','units':'nominal_meters','dimensions_measured':False,
            'physical_parameters_measured':False,'source_specific_proportions':True,
            'hidden_interiors':'constructed engineering assumptions; not visible evidence',
            'collision_note':'Preserve separate convex panels and concave openings. Never make a single convex hull of an assembled cabinet/sink/handle.',
            'uv':'Per-face-corner UV, aligned with face vertex order; planar part-normalized unless swept/shell-specific.',
            'assumptions':[]}}
        self.body('base',[0,0,0],mass,static)

    def body(self,name,origin,mass,static=False):
        if name in self.spec['bodies']:
            raise ValueError('Duplicate body: '+name)
        self.spec['bodies'][name]={'origin':list(map(float,origin)),'mass':float(mass),'static':bool(static)}

    def joint(self,name,kind,body0,body1,anchor,axis,limits,initial=0):
        self.spec['joints'].append({'name':name,'type':kind,'body0':body0,'body1':body1,
            'anchor':list(map(float,anchor)),'axis':_unit(axis).tolist(),'limits':list(map(float,limits)),
            'initial':float(initial),'parameters_status':'assumed_not_measured'})

    def mesh(self,name,vertices,faces,material='maple',body='base',collision=True,uv=None,hint='convex_hull'):
        if material not in MATERIAL_KEYS or body not in self.spec['bodies']:
            raise ValueError('Unknown material/body.')
        v=_vec(vertices);f=np.asarray(faces,dtype=int)
        self.spec['parts'].append({'name':name,'vertices':v.tolist(),'faces':f.tolist(),
            'material':material,'body':body,'collision':bool(collision),'collision_hint':hint,
            'uv':_uv_planar(v,f) if uv is None else uv})

    def box(self,name,size,center,material='maple',body='base',bevel=.002,collision=True):
        v,f=_box_mesh(size,center,bevel)
        self.mesh(name,v,f,material,body,collision)

    def tube(self,name,points,radius,material='chrome',body='base',sides=12,collision=True):
        v,f,uv=_tube_mesh(points,radius,sides)
        if len(points)==2 or not collision:
            self.mesh(name,v,f,material,body,collision,uv,'convex_hull')
            return
        # Compound convex swept segments preserve finger clearance and faucet
        # arches without relying on an engine's whole-handle convex decomposition.
        for i in range(len(points)-1):
            vertices=np.concatenate([v[i*sides:(i+2)*sides],_vec(points)[i:i+2]])
            faces=[]
            for j in range(sides):
                k=(j+1)%sides
                faces += [[j,k,sides+k],[j,sides+k,sides+j],
                          [2*sides,k,j],[2*sides+1,sides+j,sides+k]]
            self.mesh(name+f'_segment_{i:02d}',vertices,faces,material,body,True,hint='convex_hull')

    def handle(self,name,center,length=.14,vertical=True,standout=.031,body='base',radius=.0045,material='chrome'):
        # A real U profile leaves finger clearance between the bar and door.
        center=_vec(center);bend=min(.018,length/6)
        path=[]
        for q in np.linspace(0,1,7):
            path.append([0,-standout*(1-(1-q)**2),-length/2+bend*q*q])
        for q in np.linspace(0,1,7)[1:]:
            path.append([0,-standout,-length/2+bend+q*(length-2*bend)])
        for q in np.linspace(0,1,7)[1:]:
            path.append([0,-standout*(1-q*q),length/2-bend*(1-q)**2])
        path=_vec(path)
        if not vertical:
            path=path[:,[2,1,0]]
        self.tube(name,path+center,radius,material,body)

    def finish(self):
        validate(self.spec)
        return self.spec


def _case(asset,width,depth,height,bottom=.08,thickness=.018,material='maple',top=True,back=True):
    asset.box('case_left',[thickness,depth,height-bottom],[-width/2+thickness/2,depth/2,(height+bottom)/2],material)
    asset.box('case_right',[thickness,depth,height-bottom],[width/2-thickness/2,depth/2,(height+bottom)/2],material)
    asset.box('case_bottom',[width-2*thickness,depth,thickness],[0,depth/2,bottom+thickness/2],material)
    if top:
        asset.box('case_top',[width-2*thickness,depth,thickness],[0,depth/2,height-thickness/2],material)
    if back:
        asset.box('case_back',[width-2*thickness,.012,height-bottom-2*thickness],
                  [0,depth-.006,(height+bottom)/2],material)
    if bottom>0:
        asset.box('toe_kick',[width-.016,depth-.06,bottom],[0,(depth+.06)/2,bottom/2],'black',bevel=.001)


def _door(asset,name,x0,x1,z0,z1,hinge='left',material='maple',thickness=.02,handle=True,latched=False):
    anchor=[x0 if hinge=='left' else x1,-thickness/2,(z0+z1)/2]
    asset.body(name,anchor,max(.3,(x1-x0)*(z1-z0)*thickness*620))
    limits=[-math.radians(110),0] if hinge=='left' else [0,math.radians(110)]
    if latched:
        limits=[0,0]
    asset.joint(name+'_hinge','revolute','base',name,anchor,[0,0,1],limits)
    asset.box(name+'_leaf',[x1-x0,thickness,z1-z0],[(x0+x1)/2,-thickness/2,(z0+z1)/2],material,name,.0025)
    if handle:
        hx=x1-.04 if hinge=='left' else x0+.04
        hz=z1-min(.13,(z1-z0)*.24)
        asset.handle(name+'_pull',[hx,-thickness-.001,hz],min(.13,(z1-z0)*.5),body=name)
    # Visible hinge knuckles are separate from the cabinet's empty opening.
    for i,z in enumerate((z0+.10*(z1-z0),z1-.10*(z1-z0))):
        asset.tube(name+f'_hinge_knuckle_{i}',[[anchor[0],.003,z-.016],[anchor[0],.003,z+.016]],.0035,'steel',name,collision=False)
    return name


def _drawer(asset,name,width,depth,z0,z1,x=0,material='maple',front_thickness=.02,box_width=None):
    anchor=[x,0,z0]
    asset.body(name,anchor,max(.5,width*depth*3))
    asset.joint(name+'_slide','prismatic','base',name,anchor,[0,-1,0],[0,depth*.78])
    asset.box(name+'_front',[width,front_thickness,z1-z0],[x,-front_thickness/2,(z0+z1)/2],material,name,.0025)
    # Open drawer box, not a solid block behind the moving front.
    wall=.012; inner_w=width-.035 if box_width is None else box_width; box_z=z0+.025; box_h=max(.035,z1-z0-.035)
    asset.box(name+'_floor',[inner_w,depth-.035,wall],[x,(depth+.015)/2,box_z],'white',name,.001)
    for side in (-1,1):
        asset.box(name+f'_side_{side}',[wall,depth-.035,box_h],
                  [x+side*(inner_w-wall)/2,(depth+.015)/2,box_z+box_h/2],'white',name,.001)
    asset.box(name+'_back',[inner_w,wall,box_h],[x,depth-.01,box_z+box_h/2],'white',name,.001)
    asset.handle(name+'_pull',[x,-front_thickness-.001,(z0+z1)/2],min(.15,width*.5),False,body=name)
    for side in (-1,1):
        asset.box(name+f'_runner_{side}',[.003,depth*.8,.016],[x+side*(inner_w/2+.001),depth*.46,box_z+.015],'steel',name,.0005)
    return name


def _counter(asset,width,depth,z,thickness=.032,cutout=None):
    if cutout is None:
        asset.box('countertop',[width,depth,thickness],[0,depth/2,z+thickness/2],'white',bevel=.004)
        return
    x,y,w,d=map(float,cutout)
    left,right,front,back=x-w/2,x+w/2,y-d/2,y+d/2
    if not (-width/2<left<right<width/2 and 0<front<back<depth):
        raise ValueError('Sink cutout must lie strictly within the countertop.')
    for name,sx,sy,cx,cy in [('left',left+width/2,depth,(-width/2+left)/2,depth/2),
                            ('right',width/2-right,depth,(width/2+right)/2,depth/2),
                            ('front',w,front,x,front/2),('back',w,depth-back,x,(depth+back)/2)]:
        asset.box('countertop_'+name,[sx,sy,thickness],[cx,cy,z+thickness/2],'white',bevel=.002)


def cabinet(width=.90,depth=.60,height=.90,doors=2,drawers=1,*,countertop=True,
            shelves=1,wall_mounted=False,sink_cutout=None,latched=False,initial_angles=None):
    """Maple slab cabinet; drawers=0..5, doors=0..2. Cutout=(x,y,w,d)."""
    if not (0<=doors<=2 and 0<=drawers<=5) or min(width,depth,height)<=.1:
        raise ValueError('Unsupported cabinet dimensions or front count.')
    a=_Asset('cabinet',dict(width=width,depth=depth,height=height),
             ['frame_000398.jpg@13.268333s','frame_000297.jpg@9.901667s'])
    toe=0 if wall_mounted else .08; ct=.032 if countertop else 0; top=height-ct
    _case(a,width,depth,top,toe,top=sink_cutout is None)
    gap=.004
    if doors and drawers:
        drawer_zone=min((top-toe)*.35,.15*drawers)
    elif drawers:
        drawer_zone=top-toe-gap
    else:
        drawer_zone=0
    door_top=top-drawer_zone-gap
    effective_shelves=0 if drawers and not doors else shelves
    for n in range(effective_shelves):
        z=toe+(door_top-toe)*(n+1)/(effective_shelves+1)
        a.box(f'interior_shelf_{n}',[width-.043,depth-.05,.017],[0,(depth+.025)/2,z],'white',bevel=.001)
    if doors:
        leaf_w=(width-(doors+1)*gap)/doors
        for n in range(doors):
            x0=-width/2+gap+n*(leaf_w+gap)
            _door(a,f'door_{n}',x0,x0+leaf_w,toe+gap,door_top,
                  'left' if n==0 else 'right',latched=latched)
    for n in range(drawers):
        h=drawer_zone/drawers
        _drawer(a,f'drawer_{n}',width-2*gap,depth-.04,top-(n+1)*h+gap,top-n*h-gap)
    if latched and doors==2:
        z=door_top-min(.13,(door_top-toe)*.24)
        a.box('visible_handle_retainer',[.15,.014,.018],[0,-.052,z],'black',bevel=.004,collision=False)
        a.spec['metadata']['assumptions'].append('Visible handle retainer locks nominal door joints; removal mechanism is not inferred.')
    if countertop:
        _counter(a,width+.04,depth+.035,top,.032,sink_cutout)
    a.spec['metadata']['sink_cutout']=sink_cutout
    if initial_angles is not None:
        for j in a.spec['joints']:
            if j['type']=='revolute':
                j['initial']=float(initial_angles.get(j['body1'],0))
    a.spec['metadata']['assumptions'] += ['Hidden cabinet shelf layout and door hinges are constructed, not observed motion.',
                                        'Cabinet dimensions and hardware masses are nominal assumptions.']
    return a.finish()


def refrigerator(width=.91,depth=.76,height=1.80):
    """French-door refrigerator with two hinged upper leaves and freezer drawer."""
    a=_Asset('french_door_refrigerator',dict(width=width,depth=depth,height=height),
             ['frame_001488.jpg@49.605s','frame_000796.jpg@26.535s'])
    _case(a,width,depth,height,.045,.045,'steel')
    freezer_top=height*.30
    a.box('upper_compartment_floor',[width-.10,depth-.06,.03],[0,depth/2,freezer_top+.015],'white')
    # Bright inner liner uses individual walls, leaving both compartments open.
    for side in (-1,1):
        a.box(f'inner_liner_{side}',[.006,depth-.08,height-freezer_top-.065],
              [side*(width/2-.048),depth/2,(height+freezer_top)/2],'white',bevel=.001)
    a.box('inner_back',[width-.10,.008,height-freezer_top-.055],[0,depth-.052,(height+freezer_top)/2],'white')
    for n,z in enumerate(np.linspace(freezer_top+.22,height-.29,3)):
        front,back=.165,depth-.065
        a.box(f'glass_shelf_{n}',[width-.12,back-front,.006],[0,(front+back)/2,z],'glass',bevel=.001)
        a.box(f'shelf_front_trim_{n}',[width-.12,.015,.014],[0,front+.008,z],'white',bevel=.002)
    for n,(x0,x1,hinge) in enumerate(((-width/2+.003,-.003,'left'),(.003,width/2-.003,'right'))):
        name=f'upper_door_{n}'
        _door(a,name,x0,x1,freezer_top+.005,height-.004,hinge,'steel',.055,False)
        hx=x1-.055 if n==0 else x0+.055
        a.handle(name+'_long_pull',[hx,-.056,(freezer_top+height)/2],height*.38,True,.055,name,.009)
        # Door bins are shallow open trays attached to each leaf.
        for k,z in enumerate((freezer_top+.22,freezer_top+.50)):
            bw=x1-x0-.10;cx=(x0+x1)/2
            a.box(name+f'_bin_floor_{k}',[bw,.09,.01],[cx,.09,z],'white',name,.002)
            a.box(name+f'_bin_lip_{k}',[bw,.009,.07],[cx,.135,z+.035],'white',name,.002)
            for side in (-1,1):
                a.box(name+f'_bin_end_{k}_{side}',[.009,.09,.07],[cx+side*(bw-.009)/2,.09,z+.035],'white',name,.002)
    _drawer(a,'freezer_drawer',width-.008,depth-.075,.055,freezer_top-.006,material='steel',front_thickness=.06,box_width=width-.11)
    # Replace the small generic drawer pull with the source-like full-width pull.
    a.spec['parts']=[p for p in a.spec['parts'] if not p['name'].startswith('freezer_drawer_pull')]
    a.handle('freezer_full_width_pull',[0,-.061,freezer_top-.08],width*.72,False,.055,'freezer_drawer',.010)
    a.spec['metadata']['assumptions'] += ['Interior bins/shelves and French-door opening limits are inferred engineering defaults.',
                                        'No appliance brand, capacity, compressor internals or door seal force is asserted.']
    return a.finish()


def _wire_rack(a,name,width,depth,z,height=.09,body='base'):
    for n,x in enumerate(np.linspace(-width/2,width/2,9)):
        a.tube(name+f'_long_{n}',[[x,.035,z],[x,depth,z]],.0025,'steel',body,sides=8)
    for n,y in enumerate(np.linspace(.035,depth,8)):
        a.tube(name+f'_cross_{n}',[[-width/2,y,z],[width/2,y,z]],.0025,'steel',body,sides=8)
    for side in (-1,1):
        a.tube(name+f'_side_{side}',[[side*width/2,.035,z],[side*width/2,.035,z+height],
                    [side*width/2,depth,z+height],[side*width/2,depth,z]],.003,'steel',body,sides=8)


def range_oven(width=.76,depth=.66,height=.91):
    a=_Asset('range_oven',dict(width=width,depth=depth,height=height),['frame_000250.jpg@8.333333s'])
    _case(a,width,depth,height-.025,.11,.025,'steel')
    a.box('oven_inner_back',[width-.07,.012,height-.30],[0,depth-.038,(height+.11)/2],'black')
    a.box('cooktop',[width,depth,.024],[0,depth/2,height-.012],'black',bevel=.004)
    for n,(x,y) in enumerate([(x,y) for x in (-width*.24,width*.24) for y in (depth*.27,depth*.72)]):
        # Visible source cooktop is dark; rings are geometric burner hypotheses.
        ring=np.array([[x+.078*math.cos(t),y+.078*math.sin(t),height+.002] for t in np.linspace(0,2*math.pi,49)])
        a.tube(f'burner_ring_{n}',ring,.0025,'steel',sides=8,collision=False)
    a.box('control_fascia',[width,.075,.085],[0,-.002,height-.07],'steel',bevel=.004)
    for n,x in enumerate(np.linspace(-width*.36,width*.36,5)):
        a.tube(f'control_knob_{n}',[[x,-.045,height-.073],[x,-.065,height-.073]],.017,'black',sides=16)
    name='oven_door';anchor=[0,-.01,.15];a.body(name,anchor,7)
    a.joint('oven_door_hinge','revolute','base',name,anchor,[1,0,0],[0,math.pi/2])
    z0,z1=.15,height-.13
    for label,size,center in [('bottom',[width-.012,.046,.10],[0,-.018,z0+.05]),
                              ('top',[width-.012,.046,.10],[0,-.018,z1-.05]),
                              ('left',[.075,.046,z1-z0-.20],[-width/2+.043,-.018,(z0+z1)/2]),
                              ('right',[.075,.046,z1-z0-.20],[width/2-.043,-.018,(z0+z1)/2])]:
        a.box(name+'_'+label,size,center,'steel',name,.004)
    a.box('oven_glass',[width-.16,.012,z1-z0-.19],[0,-.035,(z0+z1)/2],'glass',name,.005)
    a.handle('oven_bar_pull',[0,-.044,z1-.055],width*.72,False,.045,name,.01)
    for n,z in enumerate((.32,.52)):
        _wire_rack(a,f'oven_rack_{n}',width-.085,depth-.045,z,.012)
    a.spec['metadata']['assumptions'].append('Burner/rack details and oven interior are plausible source-informed geometry, not identified appliance specifications.')
    return a.finish()


def dishwasher(width=.60,depth=.60,height=.86,racks=2):
    a=_Asset('dishwasher',dict(width=width,depth=depth,height=height),['frame_000297.jpg@9.901667s'])
    _case(a,width,depth,height,.065,.022,'steel')
    name='door';anchor=[0,-.012,.09];a.body(name,anchor,5)
    a.joint('door_hinge','revolute','base',name,anchor,[1,0,0],[0,math.pi/2])
    a.box('door_skin',[width-.008,.044,height-.095],[0,-.014,(height+.085)/2],'steel',name,.006)
    a.box('inner_door_liner',[width-.044,.012,height-.14],[0,.011,(height+.09)/2],'steel',name,.008)
    a.box('black_control_strip',[width-.025,.014,.045],[0,-.042,height-.037],'black',name,.004)
    a.handle('top_door_pull',[0,-.039,height-.095],width*.65,False,.032,name,.008)
    for n in range(racks):
        z=.22+n*(height-.39)/max(1,racks-1);body=f'rack_{n}';anchor=[0,0,z]
        a.body(body,anchor,1.3)
        a.joint(body+'_slide','prismatic','base',body,anchor,[0,-1,0],[0,depth*.68])
        _wire_rack(a,body,width-.075,depth-.055,z,.105,body)
    a.spec['metadata']['assumptions'].append('Rack layout and slides are constructed; slide operation assumes the bottom-hinged door is already open.')
    return a.finish()


def sink(width=.79,depth=.46,bowl_depth=.19,basins=2):
    """Open rounded stainless bowl(s), thick shell and drain aperture. Rim at z=0."""
    if basins not in (1,2) or min(width,depth,bowl_depth)<=.025:
        raise ValueError('Sink requires one or two positive-sized basins.')
    a=_Asset('double_sink' if basins==2 else 'single_sink',dict(width=width,depth=depth,bowl_depth=bowl_depth),
             ['frame_000250.jpg@8.333333s','frame_000297.jpg@9.901667s'] if basins==2 else ['frame_001807.jpg@60.238333s'])
    for bowl in range(basins):
        bw=(width-.008*(basins-1))/basins;cx=-width/2+bw/2+bowl*(bw+.008);cy=depth/2
        count=32
        reference=_rounded_loop(bw-.11,depth-.13,.065,0,(0,0))
        angles=np.arctan2(reference[:,1],reference[:,0])
        def drain(radius,z):return np.c_[cx+radius*np.cos(angles),cy+radius*np.sin(angles),np.full(count,z)]
        # Loop path goes from inner drain up the inner wall, over the rim, down
        # the outer wall and back through the drain neck: an actual hollow shell.
        inner=[drain(.018,-bowl_depth-.012),drain(.018,-bowl_depth+.005),
               _rounded_loop(bw-.11,depth-.13,.065,-bowl_depth+.016,(cx,cy)),
               _rounded_loop(bw-.06,depth-.075,.055,-.035,(cx,cy)),
               _rounded_loop(bw-.038,depth-.040,.045,-.005,(cx,cy)),
               _rounded_loop(bw,depth,.055,0,(cx,cy))]
        outer=[drain(.021,-bowl_depth-.015),drain(.021,-bowl_depth+.002),
               _rounded_loop(bw-.105,depth-.125,.065,-bowl_depth+.010,(cx,cy)),
               _rounded_loop(bw-.055,depth-.070,.055,-.039,(cx,cy)),
               _rounded_loop(bw-.032,depth-.034,.045,-.008,(cx,cy)),
               _rounded_loop(bw,depth,.055,-.003,(cx,cy))]
        # Each small wall/floor cell is a closed convex collision piece. Their
        # union is the hollow basin; no collider ever spans the basin opening.
        for row in range(len(inner)-1):
            for j in range(count):
                nj=(j+1)%count
                v=np.array([inner[row][j],inner[row][nj],inner[row+1][nj],inner[row+1][j],
                            outer[row][j],outer[row][nj],outer[row+1][nj],outer[row+1][j]])
                f=_hexa_faces(v)
                a.mesh(f'bowl_{bowl}_wall_{row}_{j:02d}',v,f,'steel',hint='convex_hull')
        # Dark recess is recessed below the geometric drain opening, not a cap over the basin.
        a.tube(f'drain_{bowl}_recess',[[cx,cy,-bowl_depth-.032],[cx,cy,-bowl_depth-.029]],.017,'black',sides=24,collision=False)
    a.spec['metadata']['placement_anchor']=[0,0,0]
    a.spec['metadata']['sink_opening']='No solid cube spans the basin. Compound wall/floor patches preserve cavity and drain aperture.'
    a.spec['metadata']['assumptions'].append('Bowl depth, wall thickness and drain diameter are nominal; hidden plumbing is omitted.')
    return a.finish()


def faucet(height=.46,reach=.21,base_radius=.026):
    """Arched gooseneck faucet; base at [0,reach,0], spout towards y=0."""
    a=_Asset('arched_faucet',dict(height=height,reach=reach),['frame_000250.jpg@8.333333s','frame_000297.jpg@9.901667s'])
    a.tube('base_mount',[[0,reach,0],[0,reach,.012]],base_radius,'chrome',sides=24)
    a.tube('mixing_body',[[0,reach,.008],[0,reach,.12]],.019,'chrome',sides=20)
    arch_radius=reach/2;spring=height-arch_radius
    points=[[0,reach,.10],[0,reach,spring]]
    points += [[0,reach/2+arch_radius*math.cos(t),spring+arch_radius*math.sin(t)] for t in np.linspace(0,math.pi,25)[1:]]
    points += [[0,0,max(.15,spring-.055)]]
    a.tube('arched_spout',points,.011,'chrome',sides=16)
    a.tube('aerator',[[0,0,spring-.050],[0,0,spring-.067]],.014,'steel',sides=20)
    # The observed control lever is modeled as a separate, nominal revolute part.
    anchor=[.022,reach,.085];a.body('lever',anchor,.08)
    a.joint('lever_rotation','revolute','base','lever',anchor,[1,0,0],[-.6,.6])
    a.tube('control_lever',[[.022,reach,.085],[.050,reach,.09],[.065,reach,.145]],.0055,'chrome','lever',sides=12)
    a.spec['metadata']['placement_anchor']=[0,reach,0]
    a.spec['metadata']['assumptions'].append('Spout/lever mechanism was not operated; control joint and range are illustrative geometry assumptions.')
    return a.finish()


def validate(spec):
    """Validate the serializable mesh/body/joint interface, without physics claims."""
    names=set()
    for part in spec['parts']:
        if part['name'] in names:raise ValueError('Duplicate part name '+part['name'])
        names.add(part['name'])
        v=_vec(part['vertices']);f=np.asarray(part['faces'],dtype=int)
        if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Invalid vertices '+part['name'])
        if f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid triangles '+part['name'])
        if len(part['uv'])!=len(f) or np.asarray(part['uv']).shape!=(len(f),3,2):raise ValueError('UV face corners mismatch '+part['name'])
        if not np.isfinite(part['uv']).all():raise ValueError('Nonfinite UV '+part['name'])
        if part['body'] not in spec['bodies']:raise ValueError('Missing body '+part['body'])
    children=set()
    for joint in spec['joints']:
        if joint['body0'] not in spec['bodies'] or joint['body1'] not in spec['bodies']:raise ValueError('Joint has missing body')
        if joint['body1'] in children:raise ValueError('Body has multiple parent joints')
        children.add(joint['body1'])
        if joint['limits'][0]>joint['limits'][1] or abs(np.linalg.norm(joint['axis'])-1)>1e-8:raise ValueError('Invalid joint limits/axis')
        if not joint['limits'][0]-1e-8<=joint.get('initial',0)<=joint['limits'][1]+1e-8:raise ValueError('Initial joint value outside limits')
    json.dumps(spec,allow_nan=False)
    return True


def _hexa_faces(vertices):
    faces=[];center=np.mean(vertices,axis=0)
    for q in ([0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]):
        q=list(q);p=np.asarray(vertices)[q]
        if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),np.mean(p,axis=0)-center)<0:q.reverse()
        faces += [[q[0],q[1],q[2]],[q[0],q[2],q[3]]]
    return faces


def box(size,center=(0,0,0),*,name='box',material='white',body='base',bevel=.002,collision=True):
    """Return one part dictionary; size is full XYZ extent, center in asset frame."""
    v,f=_box_mesh(size,center,bevel)
    return _primitive_part(name,v,f,material,body,collision)


def cylinder(radius,length,center=(0,0,0),axis=(0,0,1),*,name='cylinder',material='steel',body='base',segments=24,collision=True):
    """Return a closed cylindrical part; length is end-to-end along axis."""
    direction=_unit(axis)*length/2
    v,f,uv=_tube_mesh([_vec(center)-direction,_vec(center)+direction],radius,segments)
    return _primitive_part(name,v,f,material,body,collision,uv)


def capsule(radius,length,center=(0,0,0),axis=(0,0,1),*,name='capsule',material='black',body='base',segments=16,rings=6,collision=True):
    """Return a convex capsule; length is straight-section length, total length+2r."""
    if radius<=0 or length<0:raise ValueError('Invalid capsule dimensions.')
    direction=_unit(axis);ref=np.array([0,0,1]) if abs(direction[2])<.9 else np.array([1,0,0])
    u=_unit(np.cross(direction,ref));v=np.cross(direction,u);vertices=[];rows=[]
    profile=[(-length/2-radius,0)]
    profile += [(-length/2+radius*math.sin(t),radius*math.cos(t)) for t in np.linspace(-math.pi/2,0,rings+1)[1:]]
    if length>0:profile.append((length/2,radius))
    profile += [(length/2+radius*math.sin(t),radius*math.cos(t)) for t in np.linspace(0,math.pi/2,rings+1)[1:-1]]
    profile.append((length/2+radius,0))
    for z,r in profile:
        ids=[]
        for theta in ([0] if r<1e-10 else np.linspace(0,2*math.pi,segments,endpoint=False)):
            ids.append(len(vertices));vertices.append(_vec(center)+direction*z+r*(u*math.cos(theta)+v*math.sin(theta)))
        rows.append(ids)
    faces=[]
    for a,b in zip(rows[:-1],rows[1:]):
        for j in range(segments):
            k=(j+1)%segments
            if len(a)==1:faces.append([a[0],b[k],b[j]])
            elif len(b)==1:faces.append([a[j],a[k],b[0]])
            else:faces += [[a[j],a[k],b[k]],[a[j],b[k],b[j]]]
    return _primitive_part(name,_vec(vertices),faces,material,body,collision)


def _ear_clip(profile):
    p=np.asarray(profile,dtype=float)
    area=np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])/2
    order=list(range(len(p))) if area>0 else list(range(len(p)-1,-1,-1))
    triangles=[]
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    while len(order)>3:
        found=False
        for j in range(len(order)):
            a,b,c=order[j-1],order[j],order[(j+1)%len(order)]
            if cross(p[b]-p[a],p[c]-p[b])<=1e-12:continue
            def inside(k):
                return all(cross(p[v]-p[u],p[k]-p[u])>=-1e-12 for u,v in ((a,b),(b,c),(c,a)))
            if any(inside(k) for k in order if k not in (a,b,c)):continue
            triangles.append([a,b,c]);order.pop(j);found=True;break
        if not found:raise ValueError('Profile must be a simple polygon without repeated/collinear vertices.')
    triangles.append(order)
    return triangles,area


def extruded_profile(profile,depth,*,axis='y',offset=(0,0,0),name='profile',material='white',body='base',collision=True):
    """Extrude a simple 2D polygon from 0 to depth along axis; profile uses other axes in XYZ order.

    Thus axis='y' uses profile (x,z). Concave profiles get a decomposition hint;
    callers can use separate convex pieces when exact dynamic collision matters.
    """
    if axis not in ('x','y','z') or depth<=0:raise ValueError('Invalid extrusion.')
    axis_id='xyz'.index(axis);axes=[k for k in range(3) if k!=axis_id]
    p=np.asarray(profile,dtype=float);cap,area=_ear_clip(p);n=len(p)
    if area<0:p=p[::-1];cap,_=_ear_clip(p)
    vertices=np.zeros((2*n,3));vertices[:n,axes]=p;vertices[n:,axes]=p;vertices[n:,axis_id]=depth
    parity=1 if axis_id!=1 else -1;faces=[]
    for tri in cap:
        faces += [tri[::-1] if parity==1 else tri, [v+n for v in (tri if parity==1 else tri[::-1])]]
    for j in range(n):
        k=(j+1)%n;q=[j,k,k+n,j+n]
        if parity<0:q.reverse()
        faces += [[q[0],q[1],q[2]],[q[0],q[2],q[3]]]
    result=_primitive_part(name,vertices+_vec(offset),faces,material,body,collision)
    turns=[np.cross(p[(j+1)%n]-p[j],p[(j+2)%n]-p[(j+1)%n]) for j in range(n)]
    if any(t<-1e-10 for t in turns):result['collision_hint']='convex_decomposition'
    return result


def _primitive_part(name,vertices,faces,material,body,collision,uv=None):
    if material not in MATERIAL_KEYS:raise ValueError('Unknown material key '+material)
    vertices=_vec(vertices);faces=np.asarray(faces,dtype=int)
    return {'name':name,'vertices':vertices.tolist(),'faces':faces.tolist(),'material':material,'body':body,
            'collision':bool(collision),'collision_hint':'convex_hull',
            'uv':_uv_planar(vertices,faces) if uv is None else uv}


def _shell_seat(asset,name,width,depth,seat_height,back_height,material,thickness=.012):
    # Profile follows the continuous bent shell visible on stools and black chairs.
    controls=np.array([[.012,seat_height+.025],[depth*.22,seat_height],
                       [depth*.57,seat_height+.012],[depth*.76,seat_height+.062],
                       [depth*.86,seat_height+back_height*.40],[depth*.94,seat_height+back_height*.77],
                       [depth,seat_height+back_height]])
    profiles=[]
    padded=np.vstack([2*controls[0]-controls[1],controls,2*controls[-1]-controls[-2]])
    for i in range(1,len(padded)-2):
        p0,p1,p2,p3=padded[i-1:i+3]
        for t in np.linspace(0,1,4,endpoint=False):
            profiles.append(.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t))
    profiles=np.vstack([profiles,controls[-1]])
    xs=np.linspace(-1,1,7);front=[];back=[]
    for i,(y,z) in enumerate(profiles):
        tangent=profiles[min(i+1,len(profiles)-1)]-profiles[max(0,i-1)]
        normal=_unit([0,-tangent[1],tangent[0]])
        along=i/(len(profiles)-1)
        taper=1-.09*(abs(along-.35)/.65)**3
        row=[];rear=[]
        for u in xs:
            # Shallow transverse cupping plus tapered shell corners.
            point=np.array([u*width/2*taper,y-.013*u*u*along,z+.014*u*u*(1-along)])
            row.append(point+normal*thickness/2);rear.append(point-normal*thickness/2)
        front.append(row);back.append(rear)
    front,back=np.array(front),np.array(back)
    # A compact union of thin shell patches preserves open space below the seat
    # and between backrest and legs when each part receives convex collision.
    for i in range(len(profiles)-1):
        for j in range(len(xs)-1):
            v=np.array([front[i,j],front[i,j+1],front[i+1,j+1],front[i+1,j],
                        back[i,j],back[i,j+1],back[i+1,j+1],back[i+1,j]])
            asset.mesh(name+f'_{i:02d}_{j}',v,_hexa_faces(v),material)


def stool(width=.43,depth=.45,seat_height=.64,back_height=.30):
    """Maple bent-shell counter stool with black sled frame and footrest."""
    a=_Asset('maple_sled_stool',dict(width=width,depth=depth,seat_height=seat_height,back_height=back_height),
             ['frame_001488.jpg@49.605s','frame_000796.jpg@26.535s'],False,5.5)
    _shell_seat(a,'maple_shell',width,depth*.90,seat_height,back_height,'maple',.012)
    for side in (-1,1):
        x=side*width*.42
        a.tube(f'sled_{side}',[[x,.075,seat_height-.016],[x,.008,.021],[x,depth,.021],
                             [x,depth*.77,seat_height+.024]],.008,'black',sides=12)
        for n,y in enumerate((.04,depth-.03)):
            a.box(f'rubber_foot_{side}_{n}',[.030,.055,.012],[x,y,.012],'rubber',bevel=.003)
    a.tube('footrest',[[-width*.42,.055,.27],[width*.42,.055,.27]],.0075,'black')
    a.tube('rear_frame_brace',[[-width*.42,depth*.77,seat_height-.016],[width*.42,depth*.77,seat_height-.016]],.007,'black')
    a.spec['metadata']['assumptions'].append('Bent shell curvature and steel wire gauge are inferred from the three visibly matching stools.')
    return a.finish()


def chair(width=.47,depth=.50,seat_height=.455,back_height=.36):
    """Black continuous shell chair with four splayed tubular legs."""
    a=_Asset('black_shell_chair',dict(width=width,depth=depth,seat_height=seat_height,back_height=back_height),
             ['frame_000000.jpg@0s'],False,4.0)
    _shell_seat(a,'black_shell',width,depth*.9,seat_height,back_height,'black',.009)
    for side in (-1,1):
        for end in (0,1):
            top=[side*width*.34,.10+end*depth*.49,seat_height-.008]
            foot=[side*width*.46,.014+end*depth*.91,.018]
            a.tube(f'leg_{side}_{end}',[foot,top],.0095,'black')
            a.spec['parts'].append(capsule(.011,.012,foot,[0,0,1],name=f'foot_{side}_{end}',material='rubber'))
        a.tube(f'underseat_rail_{side}',[[side*width*.34,.10,seat_height-.008],
                                     [side*width*.34,.10+depth*.49,seat_height-.008]],.009,'black')
    return a.finish()


def round_table(diameter=.86,height=.74):
    """Round maple top and steel pedestal with four spreading feet, as in source."""
    a=_Asset('round_pedestal_table',dict(diameter=diameter,height=height),['frame_000000.jpg@0s'],False,15)
    center=np.array([0,diameter/2,0]);radius=diameter/2;thickness=.028
    # Lathed edge profile forms a small actual roundover on the tabletop.
    rings=[]
    for r,z in [(radius-.004,height-thickness),(radius,height-thickness+.004),
                (radius,height-.004),(radius-.004,height)]:
        theta=np.linspace(0,2*math.pi,64,endpoint=False)
        rings.append(np.c_[r*np.cos(theta),r*np.sin(theta)+center[1],np.full(64,z)])
    v,f=_loops_mesh(rings)
    vertices=v.tolist()
    for ring,reverse in ((0,True),(3,False)):
        ci=len(vertices);vertices.append([0,center[1],rings[ring][0,2]])
        for j in range(64):
            face=[ci,ring*64+j,ring*64+(j+1)%64]
            if reverse:face.reverse()
            f.append(face)
    a.mesh('maple_round_top',vertices,f,'maple')
    a.tube('steel_pedestal',[[0,center[1],.075],[0,center[1],height-thickness]],.037,'steel',sides=32)
    a.spec['parts'].append(cylinder(.10,.012,[0,center[1],height-thickness-.007],name='underside_mount_plate',material='steel',segments=32))
    for n,angle in enumerate((0,math.pi/2,math.pi,3*math.pi/2)):
        # Low tapered curved foot profile; separate convex sections avoid solid
        # disk collision through the open gaps between the four feet.
        direction=np.array([math.cos(angle),math.sin(angle),0]);side=np.cross([0,0,1],direction)
        stations=[(.015,.065,.105),(.13,.077,.092),(.25,.050,.066),(.33,.034,.047)]
        for j,((r0,z0,w0),(r1,z1,w1)) in enumerate(zip(stations[:-1],stations[1:])):
            verts=[]
            for radius_,z,width_ in ((r0,z0,w0),(r1,z1,w1)):
                for s,t in ((-1,-1),(1,-1),(1,1),(-1,1)):
                    verts.append(center+direction*radius_+side*s*width_/2+np.array([0,0,z+t*.008]))
            a.mesh(f'pedestal_foot_{n}_{j}',verts,_hexa_faces(verts),'steel')
        foot=center+direction*.33
        a.spec['parts'].append(cylinder(.026,.014,[foot[0],foot[1],.014],name=f'leveling_pad_{n}',material='rubber',segments=20))
        a.spec['parts'].append(cylinder(.010,.025,[foot[0],foot[1],.032],name=f'leveling_screw_{n}',material='steel',segments=12))
    return a.finish()


def service_cart(width=.51,depth=.86,height=.96,trays=3,casters=True):
    """Black open trays, silver posts, black end handles and four swivel casters."""
    if trays not in (2,3):raise ValueError('Source carts support the two/three-tray hypothesis.')
    a=_Asset('service_cart',dict(width=width,depth=depth,height=height,trays=trays),
             ['frame_000689.jpg@22.968333s'],False,8)
    tray_z=np.linspace(.18,height-.13,trays)
    for n,z in enumerate(tray_z):
        a.box(f'tray_{n}_floor',[width,depth,.025],[0,depth/2,z],'black',bevel=.007)
        for label,size,center in [('left',[.018,depth,.06],[-width/2+.009,depth/2,z+.029]),
                                 ('right',[.018,depth,.06],[width/2-.009,depth/2,z+.029]),
                                 ('front',[width-.036,.018,.043],[0,.009,z+.021]),
                                 ('back',[width-.036,.018,.043],[0,depth-.009,z+.021])]:
            a.box(f'tray_{n}_rim_{label}',size,center,'black',bevel=.005)
    for side in (-1,1):
        for end in (0,1):
            x=side*(width/2-.025);y=.026+end*(depth-.052)
            a.tube(f'post_{side}_{end}',[[x,y,.125],[x,y,height-.105]],.017,'steel',sides=16)
            for n,z in enumerate(tray_z):
                a.tube(f'post_collar_{side}_{end}_{n}',[[x,y,z-.025],[x,y,z+.043]],.022,'black',sides=16)
            if not casters:continue
            swivel=f'caster_{side}_{end}';anchor=[x,y,.135];a.body(swivel,anchor,.12)
            a.joint(swivel+'_swivel','revolute','base',swivel,anchor,[0,0,1],[-math.pi,math.pi])
            a.tube(swivel+'_stem',[[x,y,.10],[x,y,.151]],.009,'steel',swivel)
            wy=y+.018;wheel_center=[x,wy,.051]
            for s in (-1,1):
                a.box(swivel+f'_fork_{s}',[.006,.058,.053],[x+s*.020,y+.015,.078],'steel',swivel,.003)
            wheel=swivel+'_wheel';a.body(wheel,wheel_center,.16)
            a.joint(wheel+'_axle','revolute',swivel,wheel,wheel_center,[1,0,0],[-100*math.pi,100*math.pi])
            a.spec['parts'].append(cylinder(.050,.032,wheel_center,[1,0,0],name=wheel+'_tire',material='rubber',body=wheel,segments=28))
            a.spec['parts'].append(cylinder(.024,.035,wheel_center,[1,0,0],name=wheel+'_hub',material='steel',body=wheel,segments=20))
    for n,y in enumerate((.008,depth-.008)):
        points=[[-width/2+.026,y,height-.105],[-width/2+.026,y,height-.025],
                [-width/2+.045,y,height-.010],[width/2-.045,y,height-.010],
                [width/2-.026,y,height-.025],[width/2-.026,y,height-.105]]
        a.tube(f'cart_handle_{n}',points,.018,'black',sides=16)
    a.spec['metadata']['assumptions'].append('Caster swivel/spin axes are nominal; wheel friction, brakes and bearing dynamics were not measured.')
    return a.finish()


def sorting_island(width=2.20,depth=.91,height=.9144,bays=4,aperture_height=.17,back_doors=True):
    """Shared four-bay sorting island; low leaves and bin apertures on both faces."""
    if bays<1:raise ValueError('At least one bay is required.')
    a=_Asset('sorting_island',dict(width=width,depth=depth,height=height,bays=bays),
             ['frame_000507.jpg@16.901667s','frame_001582.jpg@52.738333s'])
    toe=.08;top=height-.032;door_top=top-aperture_height;panel=.018;gap=.004
    _case(a,width-.06,depth-.055,top,toe,panel,top=False,back=False)
    # Carcass front remains at 0; far-face leaves sit at its actual back plane.
    case_depth=depth-.055;case_width=width-.06;bay_width=case_width/bays
    for n in range(1,bays):
        a.box(f'bay_partition_{n}',[panel,case_depth,top-toe],[-case_width/2+n*bay_width,case_depth/2,(top+toe)/2])
    for n in range(bays):
        x0=-case_width/2+n*bay_width+gap;x1=x0+bay_width-2*gap
        _door(a,f'front_door_{n}',x0,x1,toe+gap,door_top,'left' if n%2==0 else 'right')
        if back_doors:
            start_parts,start_bodies,start_joints=len(a.spec['parts']),set(a.spec['bodies']),len(a.spec['joints'])
            _door(a,f'back_door_{n}',x0,x1,toe+gap,door_top,'left' if n%2==0 else 'right')
            rotation=np.diag([-1.,-1.,1.]);offset=np.array([0,case_depth,0])
            for part in a.spec['parts'][start_parts:]:part['vertices']=(_vec(part['vertices'])@rotation.T+offset).tolist()
            for name in set(a.spec['bodies'])-start_bodies:a.spec['bodies'][name]['origin']=(rotation@_vec(a.spec['bodies'][name]['origin'])+offset).tolist()
            for joint in a.spec['joints'][start_joints:]:
                joint['anchor']=(rotation@_vec(joint['anchor'])+offset).tolist();joint['axis']=(rotation@_vec(joint['axis'])).tolist()
        # Open bin well. Bags are not made into hard solid blocks.
        bin_w=bay_width-.07;cx=-case_width/2+(n+.5)*bay_width;bin_bottom=.12;bin_top=door_top+.035
        a.box(f'bin_{n}_bottom',[bin_w,case_depth-.13,.012],[cx,case_depth/2,bin_bottom],'black',bevel=.003)
        for s in (-1,1):
            a.box(f'bin_{n}_side_{s}',[.009,case_depth-.13,bin_top-bin_bottom],
                  [cx+s*(bin_w-.009)/2,case_depth/2,(bin_top+bin_bottom)/2],'black',bevel=.003)
        for end in (0,1):
            a.box(f'bin_{n}_end_{end}',[bin_w,.009,bin_top-bin_bottom],
                  [cx,.065+end*(case_depth-.13),(bin_top+bin_bottom)/2],'black',bevel=.003)
    # Roof sits above the apertures; no separate cabinet top seals the bay access.
    _counter(a,width,depth,top,.032)
    a.spec['metadata']['assumptions'].append('Four accessible open bin wells replace unmeasured soft bag shapes; disposal labels/textures are supplied by the texture lane.')
    a.spec['metadata']['apertures']={'faces':['front','back'] if back_doors else ['front'],'count_per_face':bays,'height':aperture_height}
    return a.finish()


def microwave(width=.55,depth=.42,height=.32):
    a=_Asset('microwave',dict(width=width,depth=depth,height=height),['frame_000398.jpg@13.268333s'])
    _case(a,width,depth,height,.014,.018,'steel')
    panel=.09;door_right=width/2-panel;door_left=-width/2+.003
    a.box('control_panel',[panel,.027,height-.025],[width/2-panel/2,-.001,height/2],'black',bevel=.002)
    a.box('control_display',[panel*.72,.003,.045],[width/2-panel/2,-.017,height-.058],'glass',bevel=.001)
    for r in range(4):
        for c in range(3):
            a.spec['parts'].append(cylinder(.0045,.003,[width/2-panel*.73+c*.020,-.018,height*.52-r*.028],
                                  [0,1,0],name=f'key_{r}_{c}',material='steel',segments=10,collision=False))
    name='door';anchor=[door_left,-.01,height/2];a.body(name,anchor,1.0)
    a.joint('door_hinge','revolute','base',name,anchor,[0,0,1],[-math.radians(110),0])
    cx=(door_left+door_right)/2;dw=door_right-door_left
    a.box('window',[dw-.075,.01,height-.07],[cx,-.013,height/2],'glass',name,.003)
    for label,size,center in [('bottom',[dw,.026,.035],[cx,-.008,.0175]),
                              ('top',[dw,.026,.035],[cx,-.008,height-.0175]),
                              ('left',[.035,.026,height-.07],[door_left+.0175,-.008,height/2]),
                              ('right',[.035,.026,height-.07],[door_right-.0175,-.008,height/2])]:
        a.box('door_frame_'+label,size,center,'steel',name,.003)
    a.spec['parts'].append(cylinder(min(width-panel-.055,depth-.055)/2,.006,[cx,depth/2,.037],name='turntable',material='glass',segments=40))
    a.spec['metadata']['assumptions'].append('Door mechanism and interior turntable are inferred; control graphics and manufacturer are not invented.')
    return a.finish()


def range_hood(width=.88,depth=.50,height=.74):
    a=_Asset('range_hood',dict(width=width,depth=depth,height=height),['frame_000250.jpg@8.333333s'])
    bottom=[[-width/2,0,.03],[width/2,0,.03],[width/2,depth,.03],[-width/2,depth,.03]]
    upper=[[-width*.25,depth*.50,.25],[width*.25,depth*.50,.25],
           [width*.25,depth,.25],[-width*.25,depth,.25]]
    for i in range(4):
        j=(i+1)%4;face=np.array([bottom[i],bottom[j],upper[j],upper[i]])
        normal=_unit(np.cross(face[1]-face[0],face[2]-face[0]));v=np.vstack([face,face-normal*.003])
        a.mesh(f'canopy_panel_{i}',v,_hexa_faces(v),'steel')
    for label,size,center in [('front',[width,.023,.03],[0,.0115,.015]),
                              ('back',[width,.023,.03],[0,depth-.0115,.015]),
                              ('left',[.023,depth-.046,.03],[-width/2+.0115,depth/2,.015]),
                              ('right',[.023,depth-.046,.03],[width/2-.0115,depth/2,.015])]:
        a.box('canopy_rim_'+label,size,center,'steel',bevel=.003)
    for label,size,center in [('front',[width*.50,.006,height-.25],[0,depth*.50+.003,(height+.25)/2]),
                              ('left',[.006,depth*.50,height-.25],[-width*.25+.003,depth*.75,(height+.25)/2]),
                              ('right',[.006,depth*.50,height-.25],[width*.25-.003,depth*.75,(height+.25)/2]),
                              ('top',[width*.50,depth*.50,.006],[0,depth*.75,height-.003])]:
        a.box('chimney_'+label,size,center,'steel',bevel=.001)
    a.box('underside_filter',[width-.09,depth-.09,.004],[0,depth/2,.024],'black',bevel=.001,collision=False)
    for i,x in enumerate(np.linspace(-width*.42,width*.42,15)):
        a.tube(f'filter_wire_{i}',[[x,.05,.019],[x,depth-.05,.019]],.0014,'steel',sides=6,collision=False)
    return a.finish()


def posed_parts(spec,joint_values=None):
    """Return copied part dictionaries in an articulated asset-space pose for previews."""
    values=joint_values or {};transforms={'base':np.eye(4)};remaining=list(spec['joints'])
    while remaining:
        changed=False
        for joint in remaining[:]:
            if joint['body0'] not in transforms:continue
            q=float(values.get(joint['name'],joint.get('initial',0)))
            if not joint['limits'][0]-1e-8<=q<=joint['limits'][1]+1e-8:raise ValueError('Preview joint outside limits.')
            local=np.eye(4);axis=_vec(joint['axis']);anchor=_vec(joint['anchor'])
            if joint['type']=='revolute':
                k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                rotation=np.eye(3)+math.sin(q)*k+(1-math.cos(q))*(k@k)
                local[:3,:3]=rotation;local[:3,3]=anchor-rotation@anchor
            else:local[:3,3]=axis*q
            transforms[joint['body1']]=transforms[joint['body0']]@local
            remaining.remove(joint);changed=True
        if not changed:raise ValueError('Cyclic or disconnected articulation graph.')
    result=[]
    for part in spec['parts']:
        transform=transforms.get(part['body'],np.eye(4));copy=dict(part)
        copy['vertices']=(_vec(part['vertices'])@transform[:3,:3].T+transform[:3,3]).tolist();result.append(copy)
    return result


def _preview_png(spec,path,joint_values=None,size=600):
    from PIL import Image,ImageDraw,ImageFont
    parts=posed_parts(spec,joint_values);vertices=np.concatenate([_vec(p['vertices']) for p in parts])
    view=_unit([1.45,-2.5,1.55]);right=_unit(np.cross([0,0,1],view));up=np.cross(view,right)
    basis=np.array([right,up,view]);projected=vertices@basis.T
    lo,hi=projected[:,:2].min(axis=0),projected[:,:2].max(axis=0);scale=(size-70)/max(hi-lo)
    center=(hi+lo)/2
    pixels=np.full((size,size,3),(233,236,240),dtype=np.uint8)
    zbuffer=np.full((size,size),-np.inf,dtype=np.float64)
    palette={'maple':(200,156,88),'steel':(154,167,178),'chrome':(203,218,229),'white':(230,233,227),
             'black':(34,38,42),'glass':(48,66,77),'floor':(158,162,166),'rubber':(23,26,29)}
    light=_unit([-1,-2,4])
    for part in parts:
        verts=_vec(part['vertices']);faces=np.asarray(part['faces']);p=verts@basis.T
        for face in faces:
            xyz=verts[face];normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);length=np.linalg.norm(normal)
            if length<1e-12:continue
            normal/=length
            if np.dot(normal,view)<-1e-6:continue
            shade=.48+.45*max(0,float(np.dot(normal,light)))
            color=tuple(int(min(255,max(0,c*shade))) for c in palette[part['material']])
            xy=(p[face,:2]-center)*scale;xy[:,1]*=-1;xy+=size/2
            # True depth buffering, rather than triangle-average painter order,
            # prevents false holes where cabinet interiors overlap in projection.
            low=np.maximum(np.floor(xy.min(axis=0)).astype(int),0)
            high=np.minimum(np.ceil(xy.max(axis=0)).astype(int),size-1)
            if np.any(high<low):continue
            x0,y0=xy[0];x1,y1=xy[1];x2,y2=xy[2]
            denominator=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(denominator)<1e-10:continue
            yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1].astype(float);xx+=.5;yy+=.5
            alpha=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denominator
            beta=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denominator
            gamma=1-alpha-beta
            depths=alpha*p[face[0],2]+beta*p[face[1],2]+gamma*p[face[2],2]
            region=zbuffer[low[1]:high[1]+1,low[0]:high[0]+1]
            keep=(alpha>=-1e-9)&(beta>=-1e-9)&(gamma>=-1e-9)&(depths>region)
            region[keep]=depths[keep]
            pixels[low[1]:high[1]+1,low[0]:high[0]+1][keep]=color
    image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
    draw.text((14,12),spec['metadata']['asset_type'].replace('_',' '),fill=(23,29,36))
    draw.text((14,size-24),'Nominal mesh geometry | dimensions and mechanics inferred',fill=(58,64,72))
    image.save(path)


def _preview(output):
    from PIL import Image,ImageDraw
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    examples={'cabinet':cabinet(),'cabinet_ajar':cabinet(initial_angles={'door_0':-.60}),
              'drawer_bank':cabinet(doors=0,drawers=4,shelves=0),
              'refrigerator':refrigerator(width=.81),'range_oven':range_oven(),'dishwasher':dishwasher(),
              'double_sink':sink(),'single_sink':sink(.58,.45,basins=1),'faucet':faucet(),
              'stool':stool(),'chair':chair(),'round_table':round_table(),'service_cart':service_cart(),
              'sorting_island':sorting_island(),'microwave':microwave(),'range_hood':range_hood()}
    manifest=[]
    for name,spec in examples.items():
        path=output/(name+'.png');_preview_png(spec,path)
        (output/(name+'.json')).write_text(json.dumps(spec,separators=(',',':'),allow_nan=False)+'\n')
        manifest.append({'name':name,'image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'parts':len(spec['parts']),'bodies':len(spec['bodies']),'joints':len(spec['joints'])})
    open_states={'cabinet':{'door_0_hinge':-1.15,'door_1_hinge':1.15,'drawer_0_slide':.27},
                 'refrigerator':{'upper_door_0_hinge':-1.2,'upper_door_1_hinge':1.2,'freezer_drawer_slide':.40},
                 'range_oven':{'oven_door_hinge':1.35},'dishwasher':{'door_hinge':1.5,'rack_1_slide':.25},
                 'sorting_island':{'front_door_0_hinge':-1.2,'front_door_1_hinge':1.2}}
    for name,q in open_states.items():
        path=output/(name+'_open.png');_preview_png(examples[name],path,q)
        manifest.append({'name':name+'_open','image':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'joint_values':q})
    columns=4;thumb=300;rows=math.ceil(len(manifest)/columns);contact=Image.new('RGB',(columns*thumb,rows*(thumb+26)),(249,250,252));draw=ImageDraw.Draw(contact)
    for n,item in enumerate(manifest):
        x=(n%columns)*thumb;y=(n//columns)*(thumb+26);im=Image.open(output/item['image']);im.thumbnail((thumb,thumb))
        contact.paste(im,(x,y));draw.text((x+8,y+thumb+4),item['name'],fill=(25,30,37))
    contact.save(output/'contact.jpg',quality=92)
    cards=''.join(f'<figure><img src="{x["image"]}" alt="{x["name"]}"><figcaption>{x["name"]}</figcaption></figure>' for x in manifest)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Physical mesh asset previews</title><style>body{font:16px system-ui;background:#eef0f3;color:#17202c;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;background:white;border-radius:12px;overflow:hidden}img{width:100%}figcaption{padding:12px}</style><h1>Physical mesh assets</h1><p>Source-informed geometry, with explicit open/closed articulation examples. Materials here are diagnostic colors; the separate texture lane supplies appearance. Dimensions, hidden interiors and physics are inferred, not measured.</p><main>'+cards+'</main>')
    receipt={'schema':'real2sim-physical-asset-previews/v1','implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_sha256':SOURCE_SHA256,'examples':manifest,'assembled_scene_exported':False}
    (output/'previews.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(output),'assets':len(examples),'previews':len(manifest),'implementation_sha256':receipt['implementation_sha256']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',type=Path,required=True)
    _preview(parser.parse_args().preview)
