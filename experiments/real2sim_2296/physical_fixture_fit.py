#!/usr/bin/env python3
"""Reproduce source-controlled nominal fixture fits; these are not measurements."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from physical_layout import LayoutEvidence

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/real2sim-2296/20260912T1002Z-physical'
REF=ROOT/'runs/real2sim-2296/20260912T0614Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    e=LayoutEvidence();layout=json.loads((RUN/'layout_evidence.json').read_text())
    cad=np.asarray(layout['floor_to_cad_row_matrix']);inverse=np.linalg.inv(cad)
    controls={'frame_000847.jpg':[[1343,686],[1530,700],[1509,675],[1332,663]],
              'frame_001807.jpg':[[800,357],[1212,377],[1172,289],[836,276]]}
    def corners(p):
        cx,cy,w,d,theta=p;c,s=np.cos(theta),np.sin(theta)
        xy=np.array([[-w/2,-d/2],[w/2,-d/2],[w/2,d/2],[-w/2,d/2]])@np.array([[c,s],[-s,c]])+[cx,cy]
        return np.c_[xy,np.full(4,.9164)]
    def project(frame,p):return e.project(frame,(np.c_[corners(p),np.ones(4)]@inverse)[:,:3])
    def residual(p):return np.concatenate([(project(f,p)-xy).ravel() for f,xy in controls.items()])
    fit=least_squares(residual,[-2.4,-2.55,.6,.38,1.42],
        bounds=([-5,-5,.2,.10,.8],[0,0,1,.6,2.2]),loss='soft_l1',f_scale=8,max_nfev=500)
    result=dict(scope='Manual source rim corners, estimated cameras and assumed counter plane; nominal fit, not measured accuracy',
        controls_original_pixels=controls,parameters_cx_cy_w_d_yaw=fit.x.tolist(),
        residuals_pixels={f:np.linalg.norm(project(f,fit.x)-xy,axis=1).tolist() for f,xy in controls.items()},
        source_image_sha256={f:sha(REF/'images'/f) for f in controls},layout_sha256=sha(RUN/'layout_evidence.json'),
        dataset_sha256=sha(REF/'dataset.json'),implementation_sha256=sha(__file__),
        layout_helper_sha256=sha(Path(__file__).with_name('physical_layout.py')))
    (RUN/'single_sink_fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
