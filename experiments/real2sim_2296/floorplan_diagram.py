#!/usr/bin/env python3
"""Plan diagrams from composed mesh geometry, with nominal model clearances."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from shapely.geometry import MultiPoint
from shapely.ops import unary_union,nearest_points
from pxr import Usd,UsdGeom,UsdPhysics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def geometry(path):
    s=Usd.Stage.Open(str(path));c=UsdGeom.XformCache();assets={}
    for prim in s.GetPrimAtPath('/World/Assets').GetChildren():
        n=prim.GetName()
        if n in ['floor_surface_region_01','ceiling_boundary']:continue
        vertices=[]
        for p in Usd.PrimRange(prim):
            if not p.IsA(UsdGeom.Mesh) or not p.HasAPI(UsdPhysics.CollisionAPI):continue
            v=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get());T=np.asarray(c.GetLocalToWorldTransform(p))
            v=(np.c_[v,np.ones(len(v))]@T)[:,:3]
            # Overhead cupboards and ceiling fixtures do not define floor aisles.
            if v[:,2].min()>1.2:continue
            vertices.append(v)
        if not vertices:continue
        v=np.concatenate(vertices);poly=MultiPoint(v[:,:2]).convex_hull
        if poly.geom_type=='Polygon':assets[n]=poly
    return s,assets


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    before,bb=geometry(a.source);after,aa=geometry(a.candidate)
    changed={'glass_partition_sink_end_01','glass_partition_tables_side_02','glass_partition_back','column_near_cooler_01','cooler_01','observed_corridor_wall_patch'}
    labels={'refrigerator_01':'Fridge','island_countertop':'Sink island','sorting_island_01':'Sorting island',
       'single_counter_back_wall':'Single-sink wall','table_round_kitchen_side_02':'Table 2',
       'table_round_foreground_01':'Table 1','table_round_glass_side_03':'Table 3',
       'observed_corridor_wall_patch':'Observed corridor wall','cooler_01':'Cooler'}
    fig,axes=plt.subplots(1,2,figsize=(18,8),constrained_layout=True)
    for ax,assets,title in zip(axes,[bb,aa],['Previous mesh layout','Revised source-supported layout']):
        for name,poly in assets.items():
            color='#58b7b0' if name in changed and assets is aa else ('#9ab4c4' if 'glass' in name else '#d3c5aa')
            if 'wall' in name or 'column' in name:color='#67727c' if name not in changed or assets is bb else '#158b82'
            ax.add_patch(PlotPolygon(np.asarray(poly.exterior.coords),closed=True,facecolor=color,edgecolor='#56616b',lw=.55,alpha=.9))
        for name,label in labels.items():
            if name in assets:
                xy=assets[name].centroid;ax.text(xy.x,xy.y,label,ha='center',va='center',fontsize=7,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.72,pad=1))
        ax.set(xlim=(-9.0,6.8),ylim=(-6.8,3.6),xlabel='CAD X — nominal metres',ylabel='CAD Y — nominal metres',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_axisbelow(True)
    groups={
      'Fridge to sink island':(['refrigerator_01'],['island_countertop']),
      'Main counter to sink island':(['main_left_pair','range_oven_01','main_drawer_bank_left','main_drawer_bank_right','dishwasher_01','wall_sink_station_01'],['island_countertop']),
      'Stools to sorting island':(['stool_island_1','stool_island_2','stool_island_3'],['sorting_island_01']),
      'Sink island to glazing':(['island_countertop'],['glass_partition_sink_end_01']),
      'Corridor entrance between wall and glass':(['single_counter_back_wall'],['glass_partition_back'])}
    distances={}
    for name,(left,right) in groups.items():
        distances[name]={}
        for label,assets in [('before',bb),('after',aa)]:
            ll=[assets[x] for x in left if x in assets];rr=[assets[x] for x in right if x in assets]
            if not ll or not rr:raise ValueError('Missing clearance group '+name)
            one,two=unary_union(ll),unary_union(rr);p1,p2=nearest_points(one,two)
            distances[name][label]=dict(distance_m=one.distance(two),from_xy=list(p1.coords)[0],to_xy=list(p2.coords)[0])
    fig.suptitle('IMG_2296 floor-plan iteration',fontsize=18)
    fig.text(.5,.025,'Open boundaries and unseen continuation remain unknown. Floor/ceiling canvases are omitted. Scale is assumed from a 0.9144 m counter.',ha='center',fontsize=10)
    for ext in ['png','svg']:fig.savefig(a.output/f'floorplan_before_after.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    (a.output/'model_clearances.json').write_text(json.dumps(dict(status='Distances between projected modeled collider envelopes; not surveyed aisle widths or robot-clearance certification',clearances=distances),indent=2)+'\n')
    (a.output/'footprints.json').write_text(json.dumps({k:list(v.exterior.coords) for k,v in aa.items()},indent=2)+'\n')
    (a.output/'diagram.receipt.json').write_text(json.dumps(dict(schema='real2sim-plan-diagram/v1',source_sha256=sha(a.source),candidate_sha256=sha(a.candidate),
        source_layers={x.realPath:sha(x.realPath) for x in before.GetUsedLayers() if x.realPath},
        candidate_layers={x.realPath:sha(x.realPath) for x in after.GetUsedLayers() if x.realPath},
        script_sha256=sha(__file__),artifacts={x.name:sha(x) for x in a.output.iterdir() if x.is_file()}),indent=2)+'\n')
    print(json.dumps({k:{kk:round(vv['distance_m'],3) for kk,vv in v.items()} for k,v in distances.items()},indent=2))


if __name__=='__main__':main()
