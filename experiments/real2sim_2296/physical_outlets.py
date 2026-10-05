#!/usr/bin/env python3
"""Source-textured outlet plates fitted to the supplied nominal island end plane.

Only visible plate appearance is reconstructed. Socket internals, conductivity,
wiring and electrical functionality are deliberately unmodeled. Rectified PNGs
preserve the photographed detail/illumination, without generated pixels or OCR.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


SCENE_SHA256 = '4ff8ae22b13a4270532b7fc0eeb3ac9870a9a5989d1eb794797e89950263db26'
IMAGE_SHA256 = '56629a7a3f3b95a2c080f57cd3fc6fd005c99fb819b4a14e587169b1989253c6'
SOURCE_SHA256 = '282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9'
OBSERVATIONS = [
    {'instance_id':'island_end_outlet_plate_left','description':'Left white plate with two visible receptacle patterns',
     'quad_tl_tr_br_bl':[[411.8,543.3],[506.5,573.3],[508.0,641.8],[415.5,610.5]]},
    {'instance_id':'island_end_outlet_plate_right','description':'Right white plate with one visible circular receptacle pattern',
     'quad_tl_tr_br_bl':[[550.3,590.5],[644.8,622.8],[643.5,683.5],[551.0,649.0]]},
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera(scene):
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(scene));prim=stage.GetPrimAtPath('/World/Cameras/frame_002158');cam=UsdGeom.Camera(prim)
    metadata=prim.GetCustomData();w,h=int(metadata['imageWidth']),int(metadata['imageHeight'])
    f=float(cam.GetFocalLengthAttr().Get());ha=float(cam.GetHorizontalApertureAttr().Get());va=float(cam.GetVerticalApertureAttr().Get())
    ox=float(cam.GetHorizontalApertureOffsetAttr().Get());oy=float(cam.GetVerticalApertureOffsetAttr().Get())
    K=np.array([[w*f/ha,0,w/2-w*ox/ha],[0,h*f/va,h/2+h*oy/va],[0,0,1]])
    world_gl=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    world_cv=world_gl.T@np.diag([1,-1,-1,1])
    frustum=cam.GetCamera().frustum
    gf_view_projection=np.asarray(frustum.ComputeViewMatrix())@np.asarray(frustum.ComputeProjectionMatrix())
    return dict(K=K,c2w=world_cv,w2c=np.linalg.inv(world_cv),width=w,height=h,
                source_time_seconds=float(metadata['sourceTimeSeconds']),camera_path=str(prim.GetPath()),
                world_gl_row=world_gl,gf_view_projection=gf_view_projection)


def project(points,cam):
    q=np.asarray(points)@cam['w2c'][:3,:3].T+cam['w2c'][:3,3]
    if np.any(q[:,2]<=0):raise ValueError('Plate falls behind source camera')
    h=q@cam['K'].T
    return h[:,:2]/h[:,2,None]


def gf_project(points,cam):
    q=np.c_[points,np.ones(len(points))]@cam['gf_view_projection'];q=q[:,:3]/q[:,3,None]
    return np.c_[(q[:,0]+1)*cam['width']/2,(1-q[:,1])*cam['height']/2]


def rectangle(parameters,x):
    y,z,width,height=parameters
    return np.array([[x,y-width/2,z+height/2],[x,y+width/2,z+height/2],
                     [x,y+width/2,z-height/2],[x,y-width/2,z-height/2]])


def fit(quad,cam,x):
    ray=np.c_[quad,np.ones(4)]@np.linalg.inv(cam['K']).T@cam['c2w'][:3,:3].T
    origin=cam['c2w'][:3,3];t=(x-origin[0])/ray[:,0]
    if np.any(t<=0):raise ValueError('Plane intersections are behind camera')
    intersections=origin+t[:,None]*ray
    initial=[intersections[:,1].mean(),intersections[:,2].mean(),np.ptp(intersections[:,1]),np.ptp(intersections[:,2])]
    result=least_squares(lambda p:(project(rectangle(p,x),cam)-quad).ravel(),initial,
                         bounds=([-10,0,.02,.02],[10,2,1,1]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if not result.success:raise ValueError('Rectangle fit failed')
    return result.x,intersections


def run(args):
    from PIL import Image,ImageDraw,ImageFont
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    scene=args.scene.resolve();source=args.image.resolve()
    if sha(scene)!=SCENE_SHA256:raise ValueError('This source-detail fit is bound to frozen mesh07')
    if sha(source)!=IMAGE_SHA256:raise ValueError('Undistorted source image does not match inspected input')
    build=json.loads((scene.parent/'build_inputs.json').read_text())
    if build['source_sha256']!=SOURCE_SHA256 or build['scene_sha256']!=SCENE_SHA256:raise ValueError('Scene/source receipt mismatch')
    cam=camera(scene);image=cv2.imread(str(source),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(cam['height'],cam['width']):raise ValueError('Source image/camera dimensions differ')
    checks=[]
    def check(name,value,detail=None):checks.append(dict(name=name,passed=bool(value),detail=detail))
    check('exact_source_size_and_pts',image.shape[:2]==(1061,1888) and cam['source_time_seconds']==71.94)
    plates=[];overlay=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(overlay)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for index,observation in enumerate(OBSERVATIONS):
        quad=np.array(observation['quad_tl_tr_br_bl'],dtype=float)
        parameters,intersections=fit(quad,cam,args.plane_x)
        corners=rectangle(parameters,args.plane_x);fitted=project(corners,cam);error=np.linalg.norm(fitted-quad,axis=1)
        check(observation['instance_id']+'_corner_reprojection_below_2px',error.max()<2.,error.tolist())
        check(observation['instance_id']+'_ray_plane_roundtrip',np.max(np.abs(project(intersections,cam)-quad))<1e-8)
        check(observation['instance_id']+'_independent_USD_projection',np.max(np.abs(gf_project(corners,cam)-fitted))<1e-5)
        check(observation['instance_id']+'_axis_aligned_on_supplied_plane',np.all(corners[:,0]==args.plane_x) and np.all(parameters[2:]>0))
        check(observation['instance_id']+'_inside_nominal_end_panel',corners[:,1].min()>=0 and corners[:,1].max()<=.928125 and corners[:,2].min()>0 and corners[:,2].max()<.9144)
        # Sampling the fitted rectangle makes the texture/world rectangle mapping
        # internally consistent; picked white edges remain the independent fit data.
        width=384;height=round(width*parameters[3]/parameters[2])
        dst=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
        H=cv2.getPerspectiveTransform(fitted.astype(np.float32),dst)
        texture=cv2.warpPerspective(image,H,(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        path=output/(observation['instance_id']+'.png');cv2.imwrite(str(path),texture)
        # Actual pixel-space size is recorded: resampling does not recover detail.
        source_edge_lengths=np.linalg.norm(np.roll(fitted,-1,axis=0)-fitted,axis=1)
        uv_vertices=corners[[3,2,1,0]] # BL,BR,TR,TL; +Y cross +Z => outward +X.
        normal=np.cross(uv_vertices[1]-uv_vertices[0],uv_vertices[2]-uv_vertices[0]);normal/=np.linalg.norm(normal)
        check(observation['instance_id']+'_outward_front_and_UV_axes',np.allclose(normal,[1,0,0]))
        sensitivity=[]
        for delta in (-.01,.01):
            shifted,_=fit(quad,cam,args.plane_x+delta)
            sensitivity.append({'assumed_plane_x_shift_nominal_m':delta,'center_y_z_width_y_height_z':shifted.tolist()})
        plate=dict(observation,center_world_nominal_m=[args.plane_x,float(parameters[0]),float(parameters[1])],
                   width_along_y_nominal_m=float(parameters[2]),height_z_nominal_m=float(parameters[3]),
                   source_frame='frame_002158.jpg',source_time_seconds=71.94,
                   source_image_sha256=IMAGE_SHA256,manual_corner_accuracy_pixels='approximately 1–2 px; not a calibrated error distribution',
                   raw_plane_intersections=intersections.tolist(),fitted_corners_tl_tr_br_bl=corners.tolist(),
                   fitted_source_corners_tl_tr_br_bl=fitted.tolist(),corner_fit_errors_pixels=error.tolist(),
                   rms_corner_error_pixels=float(np.sqrt(np.mean(error**2))),
                   texture={'path':str(path),'sha256':sha(path),'width':width,'height':height,
                            'source_to_texture_homography':H.tolist(),'sampled_source_edge_lengths_pixels':source_edge_lengths.tolist(),
                            'status':'Source photo rectified by homography; illumination and blur remain baked in; oversampling adds no measured detail'},
                   front_face={'normal':[1,0,0],'vertices_bl_br_tr_tl':uv_vertices.tolist(),
                               'triangles':[[0,1,2],[0,2,3]],'uv_per_vertex':[[0,0],[1,0],[1,1],[0,1]],
                               'uv_per_triangle':[[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]],
                               'u_axis_world':[0,1,0],'v_axis_world':[0,0,1],
                               'texture_top_left_uv':[0,1]},
                   plane_sensitivity_controls=sensitivity,thickness_status='Unmeasured; root may add a thin plate outward of the panel, with a documented small offset',
                   internal_geometry_modeled=False,electrical_functionality_modeled=False)
        plates.append(plate)
        draw.line([tuple(p) for p in quad]+[tuple(quad[0])],fill=(244,192,32),width=2)
        draw.line([tuple(p) for p in fitted]+[tuple(fitted[0])],fill=(30,235,230),width=1)
        draw.text((float(quad[0,0]),float(quad[0,1]-22)),f'P{index+1}: {error.max():.2f}px',font=font,fill=(0,255,235))
    check('plates_distinct_and_nonoverlapping',plates[0]['center_world_nominal_m'][1]+plates[0]['width_along_y_nominal_m']/2 <
          plates[1]['center_world_nominal_m'][1]-plates[1]['width_along_y_nominal_m']/2)
    overlay.save(output/'source_overlay.png')
    crop=overlay.crop((385,510,675,710)).resize((1160,800),Image.Resampling.LANCZOS)
    crop.save(output/'source_fit_crop.png')
    sheet=Image.new('RGB',(1160,1140),(243,244,246));sheet.paste(crop,(0,0));d=ImageDraw.Draw(sheet)
    for index,plate in enumerate(plates):
        texture=Image.open(plate['texture']['path']);sheet.paste(texture,(35+index*570,850))
        d.text((35+index*570,817),f'Plate {index+1}: {plate["width_along_y_nominal_m"]*1000:.2f} × {plate["height_z_nominal_m"]*1000:.2f} nominal mm',font=font,fill=(30,40,50))
    sheet.save(output/'comparison.jpg',quality=95)
    check('source_and_scene_unchanged',sha(source)==IMAGE_SHA256 and sha(scene)==SCENE_SHA256)
    result={'schema':'real2sim-source-outlet-plates/v1','status':'passed' if all(c['passed'] for c in checks) else 'failed',
            'produced_at':datetime.now(timezone.utc).isoformat(),'helper_sha256':sha(__file__),
            'source_sha256':SOURCE_SHA256,'source_image':str(source),'source_image_sha256':IMAGE_SHA256,
            'scene':str(scene),'scene_sha256':SCENE_SHA256,'build_inputs_sha256':sha(scene.parent/'build_inputs.json'),
            'camera':{'usd_path':cam['camera_path'],'K_source_pixels':cam['K'].tolist(),'c2w_OpenCV':cam['c2w'].tolist(),
                      'width':cam['width'],'height':cam['height'],'source_time_seconds':71.94},
            'fit_plane':{'equation':'X=constant','x_nominal_m':args.plane_x,'outward_normal':[1,0,0],'status':'Root-supplied modeled panel plane; not a metric measurement'},
            'checks':checks,'plates':plates,
            'diagnostics':[{'path':str(output/name),'sha256':sha(output/name)} for name in ('source_overlay.png','source_fit_crop.png','comparison.jpg')],
            'limitations':['Source camera, global scale and panel placement remain modeled hypotheses.',
                'One observed frame constrains visible plate appearance; hidden socket geometry and electrical behavior are unmodeled.',
                'The PNGs are photo crops with perspective removed, not estimated intrinsic albedo or higher-resolution measurements.',
                'A plate thickness/forward offset is an assembly assumption; moving its face off the fitted plane slightly changes reprojection.']}
    (output/'outlets.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for plate in plates:
        rows.append('| '+plate['instance_id']+' | '+str(plate['center_world_nominal_m'][1])+' | '+str(plate['center_world_nominal_m'][2])+' | '+str(plate['width_along_y_nominal_m'])+' | '+str(plate['height_z_nominal_m'])+' |')
    (output/'README.md').write_text('# Island end outlet-plate details\n\n'
        'Source: undistorted frame_002158.jpg, 1888×1061, PTS 71.94 s. The exact image and mesh07 camera are hashed in outlets.json. '
        'The source corners were visually inspected and refined; cyan in the overlay is the constrained fitted rectangle, yellow is the observed white-plate outline.\n\n'
        'Face plane: X='+str(args.plane_x)+' nominal m, outward +X. Dimensions are nominal camera/plane fits, not measured sizes.\n\n'
        '| Instance | Center Y | Center Z | Width +Y | Height +Z |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
        'UV orientation: +U is world +Y, +V is world +Z. PNG top-left belongs to UV(0,1). '
        'Use the supplied BL,BR,TR,TL vertices with triangles [0,1,2], [0,2,3] for outward +X winding. '
        'A thin white plate can carry the texture; any thickness and offset from the panel are unmeasured assembly assumptions. '
        'Socket holes, contacts, wiring and electrical functionality are not modeled. '
        'Source lighting/blur remains in these crops; 384-pixel resampling adds no recovered detail.\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Island outlet plates</title>'
        '<style>body{font:18px system-ui;margin:24px;background:#f3f4f6}img{max-width:1160px;width:100%}</style>'
        '<h1>Source-derived outlet plates</h1><p>Nominal plane fits and rectified source photo textures. Socket internals and electrical behavior are unmodeled.</p>'
        '<img src="comparison.jpg"><p><a href="outlets.json">Fit, textures and provenance</a> · <a href="README.md">Integration guidance</a></p>')
    print(json.dumps({'status':result['status'],'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),
                      'outlets':str(output/'outlets.json'),'receipt_sha256':sha(output/'outlets.json'),'helper_sha256':sha(__file__)}))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plane-x',type=float,default=1.081114539)
    run(parser.parse_args())
