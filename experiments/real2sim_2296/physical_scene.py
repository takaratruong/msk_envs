#!/usr/bin/env python3
"""Assemble the source-informed physical model; all dimensions are adjustable estimates."""
from pathlib import Path
import argparse, copy, gzip, json, math, hashlib, types
import numpy as np
from pxr import Gf, UsdGeom, UsdShade, Sdf
# Read once so parallel asset improvements cannot change this build's code
# provenance after Python has already imported its constructor definitions.
ASSET_BYTES=Path(__file__).with_name('physical_assets.py').read_bytes()
A=types.ModuleType('physical_assets_snapshot')
A.__file__=str(Path(__file__).with_name('physical_assets.py'))
exec(compile(ASSET_BYTES,A.__file__,'exec'),A.__dict__)
from physical_usd import SceneWriter, rotation_z, sha
from physical_polish import polish
from physical_refinements import refine_stool
from physical_supports import adjust_stool_supports
from physical_countertop import slab

PROJECT=Path(__file__).resolve().parents[2]
REFERENCE=PROJECT/'runs/real2sim-2296/20260912T0614Z'
RUN=PROJECT/'runs/real2sim-2296/20260912T1002Z-physical'

def box(size,center=(0,0,0),**kwargs):
    material=kwargs.pop('material','white')
    part=A.box(size,center,material='white',**kwargs)
    part['material']=material
    return part

def simple(parts,static=True,mass=1.,kind='source_informed_details'):
    return dict(parts=parts,bodies={'base':dict(origin=[0,0,0],mass=mass,static=static)},joints=[],
      metadata=dict(asset_type=kind,dimensions_measured=False,physical_parameters_measured=False))

def decal(spec,name,body,center,size,material,flip=False):
    p=box((size[0],.0006,size[1]),center,name=name,body=body,
                              material=material,bevel=0,collision=False)
    if flip:
        uv=np.asarray(p['uv']);uv[...,0]=1-uv[...,0];p['uv']=uv.tolist()
    spec['parts'].append(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=RUN/'iterations/mesh01/scene.usda')
    args=parser.parse_args();out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
    layout=json.loads((RUN/'layout_evidence.json').read_text())
    data=json.loads((REFERENCE/'dataset.json').read_text())
    scene=SceneWriter(out,RUN/'textures');records=[]
    textures={'maple':'wood_maple','floor':'floor_gray_tile','trim':'floor_charcoal_border','white':'white_countertop'}
    for name,texture in textures.items():
        scene.material(name,(.8,.8,.8),.70 if name in ('floor','trim') else .42,
                       texture=RUN/'textures'/f'{texture}.png')
    for key in ('label_recycle','label_compost','label_landfill','rim_recycle','rim_compost','rim_landfill',
                'fridge_notice','display_kitchen','display_dual_left','display_dual_right'):
        scene.material(key,(.8,.8,.8),.7,texture=RUN/'textures'/f'{key}.png')
    outlets=json.loads((RUN/'outlets/outlets.json').read_text())
    for plate in outlets['plates']:
        texture=Path(plate['texture']['path'])
        if sha(texture)!=plate['texture']['sha256']:raise ValueError('Outlet texture differs from its source evidence')
        scene.material(plate['instance_id'],(.8,.8,.8),.60,texture=texture)
    def add(name,spec,pos=(0,0,0),yaw=0.,ids=None,initial=None,locked=None):
        if spec.get('metadata',{}).get('asset_type')=='maple_sled_stool':spec=adjust_stool_supports(refine_stool(spec),A)
        spec=polish(name,spec,A)
        pos=np.asarray(pos,dtype=float).copy()
        if spec.get('metadata',{}).get('asset_type') in ('maple_sled_stool','black_shell_chair','round_pedestal_table','service_cart','open_blue_bin','cooler'):
            floor_min=min(np.asarray(p['vertices'])[:,2].min() for p in spec['parts'] if p.get('collision'))
            pos[2]-=floor_min
        records.append(dict(name=name,spec=spec,position=list(pos),yaw=yaw,source_ids=ids or [],initial=initial or {},locked=locked or []))
        return scene.add_asset(name,spec,pos,yaw,initial,locked,ids)
    # The two-view island fit owns this nominal scene frame.
    island=layout['island_rectangle_fit'];iw,idepth,h=island['width'],island['depth'],island['height']
    r=np.array([.9886,-.1507,0.]);r/=np.linalg.norm(r);b=np.array([-r[1],r[0],0.]);theta=math.atan2(r[1],r[0])
    corner=np.array([-2.43,2.54,0.]);depth=.62
    # Perimeter units, from the left return corner toward the glass partition.
    modules=[('main_left_pair',1.37,'cab'),('range_oven_01',.83,'range'),('main_drawer_bank',.80,'drawers'),
             ('dishwasher_01',.60,'dishwasher'),('wall_sink_station_01',.88,'sink'),('counter_appliance_base',.62,'cab')]
    at=0.;module_positions={}
    for name,width,kind in modules:
        pos=corner+r*(at+width/2);module_positions[name]=pos
        if kind=='range':spec=A.range_oven(width,depth,h)
        elif kind=='dishwasher':spec=A.dishwasher(width,depth,h-.035)
        elif kind=='sink':spec=A.cabinet(width,depth,h,doors=2,drawers=0,shelves=0,sink_cutout=(0,.32,.76,.42))
        elif kind=='drawers':spec=A.cabinet(width/2,depth,h,doors=0,drawers=4,shelves=0)
        else:
            filler=.080 if name=='main_left_pair' else 0.
            spec=A.cabinet(width-filler,depth,h,doors=2,drawers=1,shelves=1)
            pos=pos+r*filler/2
            if filler:
                add('corner_filler',simple([box((filler,.025,h-.08),(0,.0165,(h+.08)/2),name='front',material='maple'),
                    box((filler+.022,depth+.035,.032),(.011,(depth+.035)/2,h-.016),name='counter_bridge',material='white')]),corner+r*filler/2,theta)
        ids=[name] if kind in ('range','dishwasher','sink') else []
        if kind=='sink':ids+=['wall_sink_door_left_01','wall_sink_door_right_01']
        if kind=='drawers':
            add(name+'_left',spec,pos-r*width/4,theta)
            add(name+'_right',A.cabinet(width/2,depth,h,doors=0,drawers=4,shelves=0),pos+r*width/4,theta)
        else:add(name,spec,pos,theta,ids)
        if kind=='dishwasher':
            add(name+'_countertop',simple([box((width+.008,depth+.035,.032),(0,(depth+.035)/2,h-.016),name='top',material='white')]),pos,theta)
        at+=width
    # Corner return and photographed latch/ajar state.
    return_len=2.28;start=corner-b*return_len;offset=0.
    returns=[('microwave_niche',.62,'open'),('lower_cabinet_latched_pair_01',.82,'latched'),
             ('lower_cabinet_ajar_door_01',.42,'ajar'),('return_corner_pair',.42,'plain')]
    for name,width,kind in returns:
        inset=.08 if name=='return_corner_pair' else 0.
        pos=start+b*(offset+width/2-inset/2)
        spec=A.cabinet(width-inset,depth,h,doors=0 if kind=='open' else (1 if kind=='ajar' else 2),
                        drawers=1 if kind!='open' else 0,shelves=0 if kind=='open' else 1,latched=kind=='latched')
        if kind=='open':
            spec['parts'].append(box((width-.04,depth-.04,.018),(0,depth/2,.22),name='microwave_shelf',material='maple'))
        if inset:
            spec['metadata']['corner_clearance_status']='Nominal 80 mm corner filler gives the drawer clearance from perpendicular fronts; not measured'
            next(j for j in spec['joints'] if j['name']=='door_1_hinge')['limits'][1]=math.radians(100)
            spec['metadata']['corner_stop']='Nominal 100 degree usable-travel cap before modeled neighboring-cabinet contact near104 degrees; actual hardware unmeasured'
            filler_pos=start+b*(offset+width-inset/2)
            filler=simple([box((inset,.025,h-.08),(0,.0165,(h+.08)/2),name='recessed_front',material='maple',bevel=0),
                           box((inset,depth+.035,.032),(0,(depth+.035)/2,h-.016),name='counter_bridge',material='white',bevel=0)])
            add('corner_return_filler',filler,filler_pos,theta+math.pi/2)
        add(name,spec,pos,theta+math.pi/2,([name] if kind in ('latched','ajar') else [])+['corner_counter_surface_01'],
            initial={'door_0_hinge':-math.radians(24)} if kind=='ajar' else None,
            locked=['door_0_hinge','door_1_hinge'] if kind=='latched' else None)
        if kind=='open':
            mw=(A.microwave(width=.53,depth=.38,height=.30) if hasattr(A,'microwave') else
                simple([box((.53,.38,.30),(0,.19,.15),name='enclosure',material='steel'),
                        box((.40,.006,.22),(-.045,-.004,.16),name='window',material='black')]))
            add('microwave_01',mw,pos+np.array([0.,0.,.235]),theta+math.pi/2,['microwave_01'])
        offset+=width
    # Stainless refrigerator projects forward of the shallower cabinetry.
    fridge_pos=np.array([-2.47,-.08,0.])-b*.060;fridge=A.refrigerator(.82,.79,1.80)
    decal(fridge,'source_notice','upper_door_0',[-.19,-.056,1.40],[.15,.22],'fridge_notice')
    add('refrigerator_01',fridge,fridge_pos,theta+math.pi/2,
        ['refrigerator_01','refrigerator_upper_left_door_01','refrigerator_upper_right_door_01','refrigerator_lower_drawer_01'])
    # Upper cabinets follow the same two perpendicular walls, including the
    # shorter over-fridge pair; each leaf is a separate link.
    for i,width in enumerate([.76,.76,1.05]):
        pos=corner+r*(-.52+sum([.76,.76,1.05][:i])+width/2)+b*.30+np.array([0,0,1.27])
        if i==2:break
        spec=A.cabinet(width,.32,.84,doors=2,drawers=0,countertop=False,wall_mounted=True)
        if i==0:
            next(j for j in spec['joints'] if j['name']=='door_0_hinge')['limits'][0]=-math.radians(100)
            spec['metadata']['corner_stop']='Nominal 100 degree usable-travel cap before modeled wall contact near104 degrees; actual hardware unmeasured'
        add(f'upper_main_{i}',spec,pos,theta,['upper_cabinet_bank_01'])
    for i in range(3):
        inset=.09 if i==0 else (.12 if i==2 else 0.)
        shift=inset/2 if i==0 else -inset/2
        pos=start+b*((i+.5)*.76+shift)-r*.30+np.array([0,0,1.27])
        spec=A.cabinet(.76-inset,.32,.84,doors=2,drawers=0,countertop=False,wall_mounted=True)
        if i==2:spec['metadata']['corner_clearance_status']='Nominal 120 mm end clearance for the perpendicular upper leaf; hidden corner dimensions unmeasured'
        add(f'upper_return_{i}',spec,pos,theta+math.pi/2,['upper_cabinet_bank_01'])
    add('upper_fridge',A.cabinet(.82,.32,.30,doors=2,drawers=0,countertop=False,wall_mounted=True),fridge_pos-r*.42+np.array([0,0,1.83]),theta+math.pi/2,['upper_cabinet_bank_01'])
    # Source-matched sink island: doors on the kitchen side, an overhang and
    # solid rear panel on the stool side. The source shows two wide horizontal
    # pulls with two labels each; a pullout mechanism is an explicit hypothesis.
    sx=-.6311;sy=.6736;sw=.7151;sd=.3832
    body_width=iw-.07
    # Three source views constrain the front seams. The basin keeps its
    # independently fitted position, slightly off-center within the sink bay.
    front_edges=[body_width/2,.50,-.10,-body_width/2]
    label_centers=[(.967,.672),(.365,.065)]
    for i in range(3):
        width=front_edges[i]-front_edges[i+1]
        center=(front_edges[i]+front_edges[i+1])/2
        spec=A.cabinet(width,.65,h-.034,doors=0 if i<2 else 2,drawers=1 if i<2 else 0,countertop=False,shelves=0,
                        sink_cutout=(center-sx,.32,.72,.42) if i==2 else None)
        if i<2:
            labels=['recycle','recycle'] if i==0 else ['compost','landfill']
            for j,label in enumerate(labels):
                decal(spec,f'sorting_label_{j}','drawer_0',[center-label_centers[i][j],-.022,.780],[.130,.175],'label_'+label)
            spec['parts']=[p for p in spec['parts'] if not p['name'].startswith('drawer_0_pull')]
            handle_asset=A._Asset('source_handle',{},[]);handle_asset.spec=spec
            handle_asset.handle('drawer_0_pull',[center-np.mean(label_centers[i]),-.021,.690],
                                length=.100,vertical=False,standout=.031,body='drawer_0',radius=.0045)
            spec['metadata']['waste_front_evidence']='Two labels and one horizontal U-handle per wide front; pullout slide is inferred, not observed moving'
        if i==0:
            # The plates mount on the fixed cabinet body. The nominal 3 mm
            # thickness projects out from the fitted end-panel plane.
            for plate in outlets['plates']:
                c=np.asarray(plate['center_world_nominal_m']);w=plate['width_along_y_nominal_m'];z=plate['height_z_nominal_m']
                p=box((.003,w,z),c+[.0015,0,0],name=plate['instance_id'],material=plate['instance_id'],bevel=0)
                world=np.asarray(p['vertices']);faces=np.asarray(p['faces'])
                uv=np.c_[(world[:,1]-c[1])/w+.5,(world[:,2]-c[2])/z+.5]
                p['uv']=uv[faces].tolist()
                p['vertices']=((world-[center,idepth-.026,0.])@rotation_z(math.pi)).tolist()
                spec['parts'].append(p)
            spec['metadata']['outlet_plates']='Source-textured 3 mm mounted plates; socket interiors and electrical behavior unmodeled'
        add(f'island_base_{i}',spec,[center,idepth-.026,0.],math.pi,['island_sink_station_01'])
    # Standalone countertop frame has an actual opening for the double basin.
    counter=simple([slab(iw,idepth,h,(sx,sy,sw,sd))],kind='continuous_countertop_with_aperture')
    counter['metadata']['surface_collision']='Same watertight beveled mesh, static triangle collision; true sink through-hole'
    add('island_countertop',counter,ids=['island_sink_station_01'])
    add('island_sink_basin_01',A.sink(sw+.008,sd+.008,.19,2),[sx,sy-(sd+.008)/2,h+.002],ids=['island_sink_basin_01'])
    add('island_faucet_01',A.faucet(.47,.22),[-.6293,.6304,h+.004],math.pi,ids=['island_faucet_01'])
    # Perimeter double and single stations remain distinct instances.
    sinkpos=module_positions['wall_sink_station_01']
    add('wall_sink_basin_01',A.sink(.78,.44,.19,2),sinkpos+b*.10+np.array([0,0,h+.002]),theta,['wall_sink_basin_01'])
    add('wall_faucet_01',A.faucet(.48,.21),sinkpos+b*.31+np.array([0,0,h+.003]),theta,['wall_faucet_01'])
    single_yaw=1.60213
    single_right=np.array([math.cos(single_yaw),math.sin(single_yaw),0.])
    single_back=np.array([-single_right[1],single_right[0],0.])
    single_center=np.array([-2.51308,-2.50224,0.])-single_back*.31
    single_start=single_center-single_right*1.35;single_positions=[]
    for i in range(3):
        pos=single_start+single_right*((i+.5)*.90);single_positions.append(pos)
        spec=A.cabinet(.90,depth,h,doors=2,drawers=0,shelves=0 if i==1 else 1,
                       sink_cutout=(0,.31,.3861,.2884) if i==1 else None,latched=i<2)
        add(f'single_station_base_{i}',spec,pos,single_yaw,['wall_sink_station_02','wall_sink_02_latched_pair'] if i==1 else [],
            locked=['door_0_hinge','door_1_hinge'] if i<2 else None)
    ss=single_positions[1];back=single_back
    add('wall_sink_basin_02',A.sink(.3941,.2964,.19,1),ss+back*(.31-.2964/2)+np.array([0,0,h+.002]),single_yaw,['wall_sink_basin_02'])
    add('wall_faucet_02',A.faucet(.46,.21),ss+back*.30+np.array([0,0,h+.003]),single_yaw,['wall_faucet_02'])
    add('wall_secondary_tap_02',A.faucet(.27,.14,.019),ss+single_right*.19+back*.36+np.array([0,0,h+.003]),single_yaw,['wall_secondary_tap_02'])
    # Room shell is modeled mesh geometry, with distinct glazed wall sections.
    floor=box((16,13,.10),(3,-2.5,-.05),name='floor',material='floor',bevel=0)
    f=np.asarray(floor['faces']);v=np.asarray(floor['vertices']);floor['uv']=(v[f][...,:2]/.61).tolist()
    architecture=[floor]
    add('floor_surface_region_01',simple(architecture),ids=['floor_surface_region_01'])
    ceiling=simple([box((16,13,.08),(3,-2.5,3.24),name='ceiling',material='wall',bevel=0)],kind='estimated_ceiling_boundary')
    ceiling['metadata']['height_status']='Nominal 3.20 m ceiling closes the modeled room shell; not independently measured'
    add('ceiling_boundary',ceiling)
    add('main_wall',simple([box((at+.88,.15,3.2),((at-.70)/2,.70,1.6),name='wall',material='wall',bevel=0),
                            box((at+.18,.035,.095),(at/2,.61,.048),name='skirting',material='black',bevel=0)]),corner,theta)
    add('left_wall',simple([box((6.65,.15,3.2),(0,0,1.6),name='wall',material='wall',bevel=0)]),corner-r*.70-b*2.6,theta+math.pi/2)
    add('single_counter_back_wall',simple([box((2.90,.15,3.2),(0,.70,1.6),name='wall',material='wall',bevel=0)]),ss,single_yaw)
    def glass_segment(name,a,z,length=None):
        a=np.asarray(a,float);z=np.asarray(z,float);vector=z-a;length=np.linalg.norm(vector);yaw=math.atan2(vector[1],vector[0])
        parts=[box((length,.018,2.84),(length/2,0,1.48),name='glazing',material='glass',bevel=0),
               box((length,.06,.10),(length/2,0,.06),name='base',material='black',bevel=0)]
        for i,x in enumerate(np.arange(0,length+.01,1.35)):
            parts.append(box((.045,.055,3.1),(float(x),0,1.55),name=f'mullion_{i}',material='steel',bevel=.001))
        for i,height in enumerate([.83,.91,1.00]):
            parts.append(box((length,.020,.015),(length/2,0,height),name=f'frosted_stripe_{i}',material='white',bevel=0,collision=False))
        add(name,simple(parts),[a[0],a[1],0.],yaw,[name] if name!='glass_partition_back' else [])
    glass_segment('glass_partition_sink_end_01',[2.89,2.50],[3.60,-.80])
    glass_segment('glass_partition_tables_side_02',[3.90,-1.05],[6.20,-.70])
    glass_segment('glass_partition_back',[6.65,-5.65],[-1.6,-5.65])
    add('column_near_cooler_01',simple([box((.33,.40,3.2),(0,0,1.6),name='column',material='wall',bevel=.008),
                                       box((.345,.415,.12),(0,0,.06),name='plinth',material='black')]),[3.76,-.87,0.],theta,['column_near_cooler_01'])
    # Dark floor inlays are thin faces on the same floor and have no independent
    # collision shelf; they are modeled surface finish.
    for name,width,dep,cx,cy in [('island_mat',iw+.72,idepth+.58,0,.32),('sorting_mat',2.8,1.4,0,-3.70)]:
        part=box((width,dep,.001),(0,0,.0006),name='inlay',material='trim',bevel=0,collision=False)
        add(name,simple([part]),[cx,cy,0.])
    for name,center,width,yaw in [('main_floor_border',corner+r*at/2-b*.33,at+.70,theta),
                                 ('return_floor_border',start+b*return_len/2+r*.32,return_len+.7,theta+math.pi/2)]:
        add(name,simple([box((width,.53,.001),(0,0,.0006),name='border',material='trim',bevel=0,collision=False)]),center,yaw)
    # Displays and plumbing dispensers, attached to their corresponding wall.
    def wall_details(prefix,pos,yaw,dual=False):
        for i,dx in enumerate([-.70,.40] if dual else [0.]):
            tex=('display_dual_left' if i==0 else 'display_dual_right') if dual else 'display_kitchen'
            dw,dh,dz=(1.06,.60,2.04) if dual else (.94,.54,1.98)
            spec=simple([box((dw,.045,dh),(dx,.59,dz),name='bezel',material='black',bevel=.005),
                         box((dw-.045,.001,dh-.055),(dx,.566,dz),name='screen',material=tex,bevel=0,collision=False)])
            source_display='display_kitchen_01' if not dual else ('display_dual_left_02' if i==0 else 'display_dual_right_03')
            add(prefix+f'_display_{i}',spec,pos,yaw,[source_display])
        for name,x,width,height in [('paper',-.36,.23,.34),('soap',.36,.075,.21)]:
            spec=simple([box((width,.11,height),(x,.565,1.30),name='housing',material='black',bevel=.020)])
            source_name='soap_dispenser_candidate' if name=='soap' else 'paper_dispenser'
            add(prefix+'_'+name,spec,pos,yaw,[source_name+('_01' if prefix=='main' else '_02')])
    wall_details('main',sinkpos,theta)
    wall_details('single',ss,single_yaw,True)
    # Remaining reusable constructors are integrated when available. Their
    # absence is recorded rather than silently replaced with generic boxes.
    inventory=json.loads((REFERENCE/'semantic_inventory/objects.json').read_text())
    picks=layout['point_controls']
    if hasattr(A,'stool'):
        for i,x in enumerate([-.74,0,.74]):add(f'stool_island_{i+1}',A.stool(),[x,-.32,0.],math.pi,[['stool_island_left_01','stool_island_middle_01','stool_island_right_01'][i]])
    if hasattr(A,'sorting_island'):
        sorting=A.sorting_island(2.20,.91,h)
        for side in ('front','back'):
            # Back leaf indices reverse X, but the same bin keeps its category
            # on both faces. Source frame001499 sees the far rim from the front.
            categories=['landfill','compost','recycle','recycle']
            if side=='back':categories=categories[::-1]
            for i,label in enumerate(categories):
                body=f'{side}_door_{i}';origin=np.asarray(sorting['bodies'][body]['origin'])
                leaf=next(p for p in sorting['parts'] if p['name']==body+'_leaf')
                center=np.asarray(leaf['vertices']).mean(0);center[2]=.44
                center[1]=-.022 if side=='front' else .877
                decal(sorting,f'{side}_label_{i}',body,center,[.20,.285],'label_'+label,flip=side=='back')
                top=box((.17,.038,.0006),[center[0],.025 if side=='front' else .885,h+.0005],
                    name=f'{side}_rim_label_{i}',material='rim_'+label,collision=False,bevel=0)
                if side=='back':top['uv']=(1-np.asarray(top['uv'])).tolist()
                sorting['parts'].append(top)
        add('sorting_island_01',sorting,[0,-4.12,0.],0.,['sorting_island_01','sorting_panel_recycle_left_01',
          'sorting_panel_recycle_right_01','sorting_panel_compost_01','sorting_panel_landfill_01'])
    for kind,names in [('round_table',['table_01_center','table_02_top_center','table_03_center']),
                       ('service_cart',['cart_near_bin_floor','cart_outer_floor']),
                       ('chair',['chair_01_floor','chair_02_floor'])]:
        if not hasattr(A,kind):continue
        ids={'round_table':['table_round_foreground_01','table_round_kitchen_side_02','table_round_glass_side_03'],
             'service_cart':['service_cart_near_bin_01','service_cart_outer_02'],
             'chair':['chair_table_left_01','chair_table_right_02']}[kind]
        for i,key in enumerate(names):
            pos=np.array(picks[key]['nominal_xyz'][:2]+[0.]);yaw=0. if i==0 else math.pi
            if kind=='chair':
                table_key='table_02_top_center' if i==0 else 'table_03_center'
                toward=np.asarray(picks[table_key]['nominal_xyz'][:2])-pos[:2]
                yaw=math.atan2(toward[0],-toward[1])
                # The floor pick does not identify the seat center exactly.
                # Preserve the table-facing pose while clearing the glass.
                if i==1:pos[:2]+=.045*toward/np.linalg.norm(toward)
            spec=getattr(A,kind)();dims=spec['metadata']['dimensions_nominal_m']
            center_y=dims['diameter']/2 if kind=='round_table' else dims['depth']/2
            pos-=rotation_z(yaw)@np.array([0,center_y,0.])
            add(ids[i],spec,pos,yaw,[ids[i]])
    # Small manipulable source objects are geometric objects in their own bodies.
    dish=A.cylinder(.065,.012,(0,0,.006),name='dish',material='white',segments=48)
    add('counter_dish_stack_01',simple([dish],False,.12,'dish'),start+b*1.15-r*.27+np.array([0,0,h+.006]),ids=['counter_dish_stack_01'])
    tray=[box((.23,.14,.012),(0,0,.006),name='board',material='maple',bevel=.006)]
    add('island_sink_board_or_tray_01',simple(tray,False,.20,'board'),[sx-(sw+.008)/4,sy,h-.17],ids=['island_sink_board_or_tray_01'])
    from physical_details import add_details
    add_details(scene,add,A,box,simple,dict(r=r,b=b,theta=theta,h=h,corner=corner,
        range_pos=module_positions['range_oven_01'],appliance_pos=module_positions['counter_appliance_base'],
        single_pos=ss,single_yaw=single_yaw,single_right=single_right,single_back=single_back,
        single_end=single_positions[2],layout=layout))
    scene.add_lighting();scene.add_cameras(data,layout['raw_to_cad_row_matrix'])
    manifest=scene.save()
    with gzip.open(out.parent/'mesh_specs.json.gz','wt') as f:json.dump(records,f)
    implemented={i for record in records for i in record['source_ids']}
    ledger=[dict(instance_id=o['instance_id'],label=o['label'],
                 status='modeled' if o['instance_id'] in implemented else 'not_yet_mapped',
                 assets=[record['name'] for record in records if o['instance_id'] in record['source_ids']],
                 source_articulation_hypothesis=o.get('articulation_hypothesis')) for o in inventory['objects']]
    (out.parent/'inventory_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    source_dir=out.parent/'source';source_dir.mkdir(exist_ok=True)
    (source_dir/'physical_assets.py').write_bytes(ASSET_BYTES)
    code={p.name:sha(p) for p in [Path(__file__)]+[Path(__file__).with_name(n) for n in
        ['physical_usd.py','physical_details.py','physical_polish.py','physical_refinements.py','physical_supports.py','physical_countertop.py','physical_outlets.py','scene_package.py','physical_fixture_fit.py']]}
    code['physical_assets.py']=hashlib.sha256(ASSET_BYTES).hexdigest()
    for name in code:
        if name!='physical_assets.py':(source_dir/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out.parent/'build_inputs.json').write_text(json.dumps(dict(
      source_sha256=data['source_sha256'],layout_sha256=sha(RUN/'layout_evidence.json'),
      code=code,
      fixture_evidence={str(p.relative_to(RUN)):sha(p) for p in [RUN/'single_sink_fit.json',
          RUN/'reviews/source_control1/faucet_plane_controls.json',RUN/'reviews/source_control1/basin_plane_controls.json',
          RUN/'reviews/waste_fit1/recommendation.json',RUN/'reviews/handle_fit1/results.json',RUN/'outlets/outlets.json'] if p.exists()},
      texture_manifest_sha256=sha(RUN/'textures/manifest.json'),
      scene_sha256=sha(out),representation='textured_mesh',physics_exercised=False,
      scale_status=layout['scale_status']),indent=2)+'\n')
    print(json.dumps(dict(path=str(out),assets=len(records),joints=len(manifest['joints']),
              source_entries_mapped=sum(o['status']=='modeled' for o in ledger),total_source_entries=len(ledger)),indent=2))

if __name__=='__main__':main()
