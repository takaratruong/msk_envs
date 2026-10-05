"""Source-visible fixtures and small task objects for the physical scene."""
import math
import numpy as np
from physical_refinements import source_hood

def add_details(scene,add,A,box,simple,c):
    r,b,theta,h=c['r'],c['b'],c['theta'],c['h']
    add('range_hood_01',source_hood(A,.91,.48,.43),c['range_pos']+b*.19+np.array([0,0,1.866]),theta,['range_hood_01'])
    # Photographed pendant silhouettes: hollow black shade, white inner disc,
    # and suspension wire, independent of the illumination approximation.
    for i,x in enumerate([-.77,.02,.88]):
        n=48;vertices=[]
        for radius,z in [(.125,0),(.064,.255),(.061,.255),(.122,.006)]:
            vertices.extend([[radius*math.cos(a),radius*math.sin(a),z] for a in np.linspace(0,2*math.pi,n,endpoint=False)])
        faces=[]
        for ring in range(4):
            for j in range(n):
                nxt=(j+1)%n;ra=ring*n;rb=((ring+1)%4)*n
                faces.extend([[ra+j,ra+nxt,rb+nxt],[ra+j,rb+nxt,rb+j]])
        part=dict(name='hollow_shade',vertices=vertices,faces=faces,material='black',body='base',collision=False)
        spec=simple([part,A.cylinder(.118,.004,(0,0,.008),name='diffuser',material='white',segments=48,collision=False),
            A.cylinder(.0018,.705,(0,0,.6075),name='suspension',material='black',segments=8,collision=False)])
        add(f'pendant_{i+1}',spec,[x,1.0,2.24],ids=[['pendant_left_01','pendant_middle_02','pendant_right_03'][i]])
    # Glass-front countertop enclosure; its hidden hinge mechanism is inferred.
    spec=A.cabinet(.42,.39,.54,doors=1,drawers=0,countertop=False,wall_mounted=True,shelves=2)
    for p in spec['parts']:p['material']='glass' if p['name']=='door_0_leaf' else 'black'
    add('counter_glass_appliance_01',spec,c['appliance_pos']+b*.13+np.array([0,0,h+.002]),theta,['counter_glass_appliance_01'])
    # Compact countertop oven: open carcass and bottom-hinged glazed leaf.
    w,d,z=.41,.30,.26;parts=[]
    for name,size,pos in [('left',(.015,d,z),(-w/2+.0075,d/2,z/2)),('right',(.015,d,z),(w/2-.0075,d/2,z/2)),
                           ('bottom',(w-.03,d,.015),(0,d/2,.015)),('top',(w,d,.015),(0,d/2,z-.0075)),
                           ('back',(w-.03,.015,z-.03),(0,d-.0075,z/2))]:
        parts.append(box(size,pos,name=name,material='steel'))
    parts.extend([box((w-.10,.012,z-.07),(-.022,-.008,z/2),name='glazed_door',body='door',material='glass'),
                  A.cylinder(.009,w*.66,(-.03,-.036,z-.05),(1,0,0),name='pull',body='door',material='steel')])
    for i in range(3):parts.append(A.cylinder(.017,.026,(w/2-.035,-.011,.05+i*.073),(0,1,0),name=f'knob_{i}',material='black'))
    anchor=[-.02,-.008,.025]
    spec=simple(parts);spec['bodies']['door']=dict(origin=anchor,mass=.8,static=False)
    spec['joints']=[dict(name='door_hinge',type='revolute',body0='base',body1='door',anchor=anchor,axis=[1,0,0],limits=[0,math.pi/2])]
    spec['metadata']['mechanism']='Bottom hinge is inferred from source appearance; not observed moving'
    pos=c['single_end']+c['single_back']*.11+np.array([0,0,h+.002])
    add('toaster_oven_01',spec,pos,c['single_yaw'],['toaster_oven_01'])
    bottle=simple([A.cylinder(.037,.18,(0,0,.09),name='body',material='glass',segments=32),
                   A.cylinder(.031,.13,(0,0,.075),name='blue_contents',material='black',segments=32,collision=False),
                   A.cylinder(.015,.065,(0,0,.206),name='neck',material='glass',segments=24),
                   A.cylinder(.019,.020,(0,0,.245),name='cap',material='white',segments=24)],False,.4,'bottle')
    bottle['parts'][1]['material']='blue'
    add('blue_liquid_bottle_01',bottle,pos-c['single_right']*.28+np.array([0,0,.006]),ids=['blue_liquid_bottle_01'])
    # Open pitcher consists of individual thin wall sectors and a bottom.
    p=[];radius=.055;segments=32
    for i,a in enumerate(np.linspace(0,2*math.pi,segments,endpoint=False)):
        part=box((2*math.pi*radius/segments,.005,.19),(0,0,.095),name=f'wall_{i}',material='glass')
        angle=a-math.pi/2;rot=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1.]])
        v=np.asarray(part['vertices']);part['vertices']=(v@rot.T+[radius*math.cos(a),radius*math.sin(a),0]).tolist();p.append(part)
    p.append(A.cylinder(radius,.007,(0,0,.0035),name='floor',material='glass',segments=32))
    add('water_pitcher_candidate_01',simple(p,False,.25,'pitcher'),c['single_pos']-c['single_right']*.96+c['single_back']*.28+np.array([0,0,h+.001]),ids=['water_pitcher_candidate_01'])
    # Tall open blue floor bin beside the two service carts.
    w,d,z=.20,.39,.82;p=[]
    for name,size,center in [('bottom',(w,d,.018),(0,d/2,.018)),('left',(.018,d,z),(-w/2+.009,d/2,z/2)),
                             ('right',(.018,d,z),(w/2-.009,d/2,z/2)),('front',(w-.036,.018,z),(0,.009,z/2)),
                             ('back',(w-.036,.018,z),(0,d-.009,z/2))]:
        p.append(box(size,center,name=name,material='blue',bevel=.006))
    add('blue_floor_bin_01',simple(p,False,2.8,'open_blue_bin'),[1.225,-3.7845,0.],ids=['blue_floor_bin_01'])
    # Blue/white cooler with hollow cavity and separate rear-hinged lid.
    w,d,z=.58,.43,.55;p=[]
    for name,size,center in [('bottom',(w,d,.04),(0,d/2,.025)),('left',(.035,d,z),(-w/2+.0175,d/2,z/2)),
                             ('right',(.035,d,z),(w/2-.0175,d/2,z/2)),('front',(w-.07,.035,z),(0,.0175,z/2)),
                             ('back',(w-.07,.035,z),(0,d-.0175,z/2))]:
        p.append(box(size,center,name=name,material='blue',bevel=.012))
    p.append(box((w+.018,d+.018,.055),(0,d/2,z+.025),name='lid',material='white',body='lid',bevel=.018))
    p.append(box((w,d,.034),(0,d/2,z+.061),name='lid_top',material='trim',body='lid',bevel=.015))
    for side in (-1,1):
        p.append(box((.055,.19,.04),(side*(w/2+.01),d/2,.40),name=f'carry_handle_{side}',material='white',bevel=.012))
    anchor=[0,d,z];spec=simple(p,False,5.3,'cooler');spec['bodies']['lid']=dict(origin=anchor,mass=.9,static=False)
    spec['joints']=[dict(name='lid_hinge',type='revolute',body0='base',body1='lid',anchor=anchor,axis=[1,0,0],limits=[-math.radians(105),0])]
    center=np.asarray(c['layout']['point_controls']['cooler_floor']['nominal_xyz'])
    # A 15 mm source-pick adjustment clears the estimated glass plane.
    glass_clearance=np.array([-.9776,-.2103,0.])*.015
    add('cooler_01',spec,center-b*(d/2)+glass_clearance,theta,['cooler_01'])
