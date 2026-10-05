#!/usr/bin/env python3
"""Fit source-linked geometric controls; nominal scale is an explicit assumption."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'runs/real2sim-2296/20260912T0614Z'
OUTPUT = ROOT / 'runs/real2sim-2296/20260912T1002Z-physical'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class LayoutEvidence:
    def __init__(self):
        self.dataset = json.loads((REFERENCE/'dataset.json').read_text())
        self.views = {v['name']:v for v in self.dataset['train']+self.dataset['val']}
        self.rec = pycolmap.Reconstruction(str(REFERENCE/'model_stable'))
        self.images = {v.name:v for v in self.rec.images.values()}
        stage=Usd.Stage.Open(str(REFERENCE/'scene/scene.usda'))
        self.floor=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath('/World/ReconstructionFrame')).ComputeLocalToWorldTransform(0))
        self.unfloor=np.linalg.inv(self.floor)
    def camera(self,frame):
        return self.rec.cameras[self.images[frame].camera_id]
    def ray(self,frame,xy):
        c2w=np.linalg.inv(self.views[frame]['w2c'])
        q=np.asarray(self.camera(frame).cam_from_img(np.asarray(xy,dtype=float)))
        direction=np.r_[q,1.]@c2w[:3,:3].T@self.floor[:3,:3]
        origin=np.r_[c2w[:3,3],1.]@self.floor
        return origin[:3],direction
    def project(self,frame,xyz):
        xyz=np.asarray(xyz)
        raw=np.c_[xyz,np.ones(len(xyz))]@self.unfloor
        c=raw@np.asarray(self.views[frame]['w2c']).T
        return np.asarray(self.camera(frame).img_from_cam(np.ascontiguousarray(c[:,:3])))
    def plane_point(self,frame,xy,z=0):
        c,d=self.ray(frame,xy)
        t=(z-c[2])/d[2]
        if t<=0:raise ValueError('Control intersects plane behind camera')
        return c+t*d

def rectangle(p):
    cx,cy,w,d,theta,z=p
    xy=np.array([[-w/2,-d/2],[w/2,-d/2],[w/2,d/2],[-w/2,d/2]])
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.c_[xy@rot.T+[cx,cy],np.full(4,z)]

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    e=LayoutEvidence()
    controls={
      'frame_001488.jpg':[[643,455],[1336,444],[1316,390],[728,408]],
      'frame_002158.jpg':[[973,84],[59,347],[785,558],[1400,146]],
    }
    def residual(p):
        corners=rectangle(p)
        return np.concatenate([(e.project(f,corners)-xy).ravel() for f,xy in controls.items()])
    fit=least_squares(residual,[-.03,.35,.32,.12,-.46,.114],
        bounds=([-.5,0,.1,.03,-1,.05],[.5,1,.6,.3,0,.22]),loss='soft_l1',f_scale=8,max_nfev=1000)
    cx,cy,w,d,theta,z=fit.x
    nominal_counter_height=.9144
    scale=nominal_counter_height/z
    x=np.array([math.cos(theta),math.sin(theta),0.])
    y=np.array([-math.sin(theta),math.cos(theta),0.])
    origin=np.array([cx,cy,0.])-y*d/2
    basis=np.stack([x,y,np.array([0.,0.,1.])],axis=1)
    cad=np.eye(4);cad[:3,:3]=basis*scale;cad[3,:3]=-origin@basis*scale
    raw_to_cad=e.floor@cad
    errors={f:np.linalg.norm(e.project(f,rectangle(fit.x))-xy,axis=1).tolist() for f,xy in controls.items()}
    def pick(frame,xy,height=0):
        q=e.plane_point(frame,xy,height/scale)
        return (np.r_[q,1]@cad)[:3].tolist()
    picks={
      'main_counter_left':('frame_001488.jpg',[851,356],.9144),
      'main_counter_right':('frame_001488.jpg',[1746,342],.9144),
      'corner_counter_inner':('frame_001488.jpg',[570,369],.9144),
      'fridge_front_near':('frame_001488.jpg',[195,776],0),
      'fridge_front_far':('frame_001488.jpg',[367,695],0),
      'sorting_left_floor':('frame_000507.jpg',[253,550],0),
      'sorting_right_floor':('frame_000507.jpg',[1197,691],0),
      'sorting_top_left':('frame_000507.jpg',[164,167],.9144),
      'sorting_top_right':('frame_000507.jpg',[1256,275],.9144),
      'single_sink_counter_left':('frame_000847.jpg',[875,665],.9144),
      'single_sink_center':('frame_000847.jpg',[1410,680],.9144),
      'table_01_center':('frame_000000.jpg',[657,953],0),
      'table_02_top_center':('frame_000000.jpg',[604,213],.75),
      'table_03_center':('frame_000000.jpg',[1304,601],0),
      'chair_01_floor':('frame_000000.jpg',[177,517],0),
      'chair_02_floor':('frame_000000.jpg',[1605,542],0),
      'cart_near_bin_floor':('frame_002081.jpg',[1612,505],0),
      'cart_outer_floor':('frame_002081.jpg',[1441,477],0),
      'blue_bin_floor':('frame_002081.jpg',[1744,527],0),
      'cooler_floor':('frame_002081.jpg',[445,849],0),
      'column_cooler_floor':('frame_002081.jpg',[542,583],0),
      'far_chair_left_floor':('frame_002229.jpg',[124,202],0),
      'far_chair_right_floor':('frame_002229.jpg',[203,201],0),
      'far_table_floor':('frame_002229.jpg',[648,403],0),
    }
    source_controls={name:dict(frame=frame,source_image_sha256=sha(REFERENCE/'images'/frame),pixel_xy=xy,assumed_height=height,
         nominal_xyz=pick(frame,xy,height)) for name,(frame,xy,height) in picks.items()}
    result=dict(schema='real2sim-physical-layout-evidence/v1',
      source_sha256=e.dataset['source_sha256'],dataset_sha256=sha(REFERENCE/'dataset.json'),
      implementation_sha256=sha(__file__),nominal_counter_height=nominal_counter_height,
      source_camera_artifacts={p.name:sha(p) for p in (REFERENCE/'model_stable').glob('*.bin')},
      floor_evidence_sha256=sha(REFERENCE/'floor_evidence.json'),
      reference_scene_sha256=sha(REFERENCE/'scene/scene.usda'),
      scale_status='ASSUMED nominal 0.9144 m island counter height; no independent metric observation',
      meter_per_raw_unit=scale,raw_to_cad_row_matrix=raw_to_cad.tolist(),floor_to_cad_row_matrix=cad.tolist(),
      island_rectangle_fit=dict(parameters_raw=fit.x.tolist(),width=w*scale,depth=d*scale,height=z*scale,
         source_controls=controls,reprojection_errors_pixels=errors,
         mean_corner_error_pixels=float(np.mean(list(errors.values()))),
         inputs=[dict(frame=f,sha256=sha(REFERENCE/'images'/f)) for f in controls]),
      point_controls=source_controls,uncertainties=[
         'Hand-selected source pixels are approximate visible corners, not a survey.',
         'Perspective ray intersections assume the fitted floor and nominal object heights.',
         'Source camera error can distort point estimates; orthogonal CAD fitting is an additional prior.',
         'Body masses, joint mechanisms and hidden surfaces must be separately modeled assumptions.'])
    (OUTPUT/'layout_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
