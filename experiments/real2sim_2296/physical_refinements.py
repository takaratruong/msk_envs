#!/usr/bin/env python3
"""Bounded source-guided shape refinements; no placement or scene export.

``refine_stool(spec)`` returns a copy of the raw physical_assets.stool() spec.
Call it BEFORE physical_polish.polish(), which welds the same shell patches into
one visible shell. Bodies, joints, supports and nominal placement stay unchanged.
``source_hood(library, width=.91, depth=.48, height=.43)`` returns a new asset spec
with the same x-right/y-back/z-up, front-y=0, bottom-z=0 convention as the library.

These are engineering shapes inferred from RGB silhouettes. Corner radii, shell
thickness, hidden hood construction and all dimensions remain unmeasured.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np


SOURCE_SHA256 = '282a9ccd41ea5291b187ce04699d4f93c2e40a891098ae0250184a71da336fb9'
SOURCE_EVIDENCE = [
    {'name': 'frame_001499.jpg', 'source_index': 1499, 'time_seconds': 49.971667,
     'sha256': '8d84e5e894ae4fc1a206c69660160ead6dfecf969c16df5276e506ac9c9030ab',
     'stool_bbox_xyxy': [805, 426, 1384, 748], 'hood_bbox_xyxy': [873, 30, 1111, 149]},
    {'name': 'frame_002158.jpg', 'source_index': 2158, 'time_seconds': 71.94,
     'sha256': '61e11589a655747c48921ebb45041cef1b2556e21b370c281a92c1ea11082287',
     'stool_bbox_xyxy': [62, 63, 879, 290]},
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hexa_faces(vertices):
    """Keep consecutive triangle pairs for physical_polish's shared-face weld."""
    v = np.asarray(vertices, dtype=float)
    center = v.mean(axis=0)
    faces = []
    for q in ([0,1,2,3], [4,7,6,5], [0,4,5,1], [1,5,6,2], [2,6,7,3], [3,7,4,0]):
        q = list(q)
        p = v[q]
        if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), p.mean(axis=0)-center) < 0:
            q.reverse()
        faces.extend(([q[0],q[1],q[2]], [q[0],q[2],q[3]]))
    return faces


def _uv(vertices, faces):
    v = np.asarray(vertices, dtype=float)
    lo, span = v.min(axis=0), np.maximum(np.ptp(v, axis=0), 1e-9)
    result = []
    for face in faces:
        p = v[face]
        drop = np.argmax(np.abs(np.cross(p[1]-p[0], p[2]-p[0])))
        axes = [k for k in range(3) if k != drop]
        result.append(((p[:, axes]-lo[axes])/span[axes]).tolist())
    return result


def _smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def refine_stool(spec):
    """Round the source stool's upper corners and cup its back, returning a copy.

    Shared surface vertices are transformed once on a shared grid; visible and
    collision patches therefore have identical seams. The last backrest rows get
    five extra samples to resolve the rounded corner. Collision remains the
    library's compound convex hulls of thin patches, never a hull of the stool.
    Repeated calls are idempotent. Applying this after polish is an explicit error.
    """
    if spec.get('metadata', {}).get('asset_type') != 'maple_sled_stool':
        raise ValueError('refine_stool requires an unplaced maple_sled_stool spec')
    result = copy.deepcopy(spec)
    if any(p['name'] == 'maple_shell_continuous_visual' for p in spec['parts']):
        raise ValueError('Call refine_stool before physical_polish.polish')
    if result['metadata'].get('source_shape_refinement', {}).get('version') == 1:
        return result
    patches = {}
    for p in spec['parts']:
        match = re.fullmatch(r'maple_shell_(\d+)_(\d+)', p['name'])
        if match:
            patches[tuple(map(int, match.groups()))] = p
    if not patches:
        raise ValueError('Raw shell patches are missing')
    rows = max(i for i,j in patches)+1
    cols = max(j for i,j in patches)+1
    if len(patches) != rows*cols or cols % 2:
        raise ValueError('Expected a complete, symmetric shell patch grid')
    front = np.full((rows+1,cols+1,3), np.nan)
    back = front.copy()
    for (i,j), p in patches.items():
        v = np.asarray(p['vertices'], dtype=float)
        if v.shape != (8,3):
            raise ValueError('Shell patches must have eight corresponding vertices')
        for k,(r,c) in enumerate(((i,j),(i,j+1),(i+1,j+1),(i+1,j))):
            for array,point in ((front,v[k]),(back,v[k+4])):
                if np.isfinite(array[r,c]).all() and not np.allclose(array[r,c],point,atol=1e-11,rtol=0):
                    raise ValueError('Input shell has inconsistent shared seams')
                array[r,c] = point
    center = (front+back)/2
    thickness = np.linalg.norm(front-back,axis=2)
    dimensions = spec['metadata']['dimensions_nominal_m']
    width = float(dimensions['width'])
    back_height = float(dimensions['back_height'])
    seat_height = float(dimensions['seat_height'])
    if min(width,back_height) <= 0:
        raise ValueError('Positive stool width and back height required')
    # Refine only the last part of the existing back profile, keeping seat/rail
    # placement. These additional samples are shared by both visible and collision.
    sample_rows = np.unique(np.r_[np.arange(rows+1), rows-2.5, rows-1.5,
                                  rows-.75, rows-.50, rows-.25])
    sample_rows = sample_rows[(sample_rows >= 0) & (sample_rows <= rows)]
    c = np.empty((len(sample_rows),cols+1,3))
    t = np.empty((len(sample_rows),cols+1))
    for j in range(cols+1):
        for axis in range(3):
            c[:,j,axis] = np.interp(sample_rows,np.arange(rows+1),center[:,j,axis])
        t[:,j] = np.interp(sample_rows,np.arange(rows+1),thickness[:,j])
    top_z = float(center[-1,cols//2,2])
    radius = min(width*.115,back_height*.18)
    base_z = c[:,cols//2,2].copy()
    half_width = np.max(np.abs(c[:,:,0]),axis=1)
    rise = _smoothstep((base_z-seat_height-back_height*.20)/(back_height*.80))
    corner_z = np.clip(base_z-(top_z-radius),0.,radius)
    inset = radius-np.sqrt(np.maximum(0.,radius*radius-corner_z*corner_z))
    if np.any(inset >= half_width*.45):
        raise ValueError('Rounded corner would excessively narrow this stool')
    u = c[:,:,0]/half_width[:,None]
    width_factor = (1.-inset/half_width)[:,None]
    c[:,:,0] *= width_factor
    # A mild transverse cup is visible from both sides of the source island.
    extra_cup = width*.0325
    crown = width*.0093
    # Corner clipping changes sampled x, not the underlying bent surface's
    # curvature. Re-evaluate transverse cupping at that new x; squeezing an
    # unchanged u-based cup into the narrow last row twists collision cells.
    along = sample_rows/rows
    # Differentiate the underlying surface at fixed physical x. Differentiating
    # the clipped boundary grid instead spuriously tilts the last corner normals.
    y_axis = np.interp(sample_rows,np.arange(rows+1),center[:,cols//2,1])+extra_cup*rise
    y_curve = (.013*along+extra_cup*rise)/(half_width*half_width)
    z_curve = (.014*(1-along)-crown*rise)/(half_width*half_width)
    c[:,:,1] = y_axis[:,None]-y_curve[:,None]*c[:,:,0]**2
    c[:,:,2] = base_z[:,None]+z_curve[:,None]*c[:,:,0]**2
    derivative = np.gradient(np.c_[y_axis,y_curve,base_z,z_curve],sample_rows,axis=0,edge_order=2)
    tangent_x = np.empty_like(c)
    tangent_x[:,:,0] = 1.
    tangent_x[:,:,1] = -2*y_curve[:,None]*c[:,:,0]
    tangent_x[:,:,2] = 2*z_curve[:,None]*c[:,:,0]
    tangent_profile = np.empty_like(c)
    tangent_profile[:,:,0] = 0.
    tangent_profile[:,:,1] = derivative[:,0,None]-derivative[:,1,None]*c[:,:,0]**2
    tangent_profile[:,:,2] = derivative[:,2,None]+derivative[:,3,None]*c[:,:,0]**2
    normal = np.cross(tangent_x,tangent_profile)
    length = np.linalg.norm(normal,axis=2)
    if np.any(length < 1e-10):
        raise ValueError('Refinement produced a singular shell surface')
    normal /= length[:,:,None]
    f = c+normal*t[:,:,None]/2
    b = c-normal*t[:,:,None]/2
    new_parts = []
    for i in range(len(sample_rows)-1):
        old_row = min(rows-1,int(math.floor(sample_rows[i])))
        for j in range(cols):
            p = copy.deepcopy(patches[old_row,j])
            v = np.array([f[i,j],f[i,j+1],f[i+1,j+1],f[i+1,j],
                          b[i,j],b[i,j+1],b[i+1,j+1],b[i+1,j]])
            faces = _hexa_faces(v)
            p.update(name=f'maple_shell_{i:02d}_{j}',vertices=v.tolist(),
                     faces=faces,uv=_uv(v,faces),collision_hint='convex_hull')
            new_parts.append(p)
    old_names = {p['name'] for p in patches.values()}
    result['parts'] = new_parts+[p for p in result['parts'] if p['name'] not in old_names]
    result['metadata']['source_shape_refinement'] = {
        'version': 1, 'status': 'inferred_from_RGB_not_measured',
        'source_sha256': SOURCE_SHA256, 'evidence': copy.deepcopy(SOURCE_EVIDENCE),
        'upper_corner_radius_nominal_m': radius,
        'additional_transverse_cup_nominal_m': extra_cup,
        'top_edge_crown_nominal_m': crown,
        'original_shell_patch_count': len(patches), 'refined_shell_patch_count': len(new_parts),
        'shell_thickness_preserved': True, 'bodies_joints_supports_unchanged': True,
        'collision': 'Separate convex hulls of thin curved patches; hull deviation audited in preview receipt',
        'visual': 'Same shell patches; physical_polish welds matching internal faces before export',
    }
    return result


def source_hood(library, width=.91, depth=.48, height=.43):
    """Full-width sloping stainless enclosure visible in frame_001499.

    Origin is its lowest front edge, with x centred and the wall behind at +depth.
    The interior stays hollow. Individual finite-thickness panels are convex
    colliders. Hidden filter/support details are explicitly inferred.
    """
    width,depth,height = map(float,(width,depth,height))
    if min(width,depth,height) <= .06:
        raise ValueError('Hood dimensions must exceed 0.06 nominal metres')
    sheet = min(.003,width*.004,depth*.007)
    lip_height = height*.115
    top_front = depth*.46
    spec = {'schema': library.SCHEMA, 'parts': [],
            'bodies': {'base': {'origin':[0.,0.,0.], 'mass':0., 'static':True}},
            'joints': [], 'metadata': {
        'asset_type':'range_hood', 'dimensions_nominal_m':dict(width=width,depth=depth,height=height),
        'source_sha256': SOURCE_SHA256, 'source_evidence':copy.deepcopy(SOURCE_EVIDENCE[:1]),
        'coordinate_system':'x right, y back, z up; x centred; front y=0; bottom z=0',
        'vertices_frame':'asset', 'joint_anchor_frame':'asset',
        'body_origins':'closed pose in asset frame', 'joint_angle_units':'radians',
        'units':'nominal_meters', 'dimensions_measured':False,
        'physical_parameters_measured':False, 'source_specific_proportions':True,
        'source_shape_refinement':{'version':1,'status':'inferred_from_RGB_not_measured',
            'shape':'Full-width sloped front, narrow lower lip, full-width roof; no narrow chimney',
            'front_slope_from_vertical_degrees':math.degrees(math.atan2(top_front,height-lip_height)),
            'sheet_thickness_nominal_m':sheet},
        'hidden_interiors':'Filter, roof/back panels and sheet thickness are engineering assumptions',
        'collision_note':'Separate convex thin panels; hollow enclosure retained',
        'uv':'Per-face-corner UV in face order; planar part coordinates',
        'assumptions':['Overall dimensions and front slope are not measured.',
                       'Visible broad silhouette and lower band follow the source; underside construction is inferred.',
                       'No controls, manufacturer branding or powered articulation is invented.'],
    }}
    parts = spec['parts']
    # Left/right side walls are convex profile extrusions. The sloping front
    # retains full width from its lower edge to its top, as seen in the source.
    side_profile = [[0.,0.],[depth,0.],[depth,height],[top_front,height],[0.,lip_height]]
    for side,x in (('left',-width/2),('right',width/2-sheet)):
        parts.append(library.extruded_profile(side_profile,sheet,axis='x',offset=[x,0,0],
                     name=f'enclosure_side_{side}',material='steel'))
    face = np.array([[-width/2+sheet,0,lip_height],[width/2-sheet,0,lip_height],
                     [width/2-sheet,top_front,height],[-width/2+sheet,top_front,height]])
    normal = np.cross(face[1]-face[0],face[2]-face[0])
    normal /= np.linalg.norm(normal)
    v = np.vstack([face,face-normal*sheet])
    faces = _hexa_faces(v)
    parts.append(dict(name='full_width_sloped_front',vertices=v.tolist(),faces=faces,
                      uv=_uv(v,faces),material='steel',body='base',collision=True,
                      collision_hint='convex_hull'))
    for name,size,center in (
        ('lower_front_lip',[width-2*sheet,sheet,lip_height],[0,sheet/2,lip_height/2]),
        ('full_width_roof',[width-2*sheet,depth-top_front,sheet],
                           [0,(depth+top_front)/2,height-sheet/2]),
        ('back_panel',[width-2*sheet,sheet,height-sheet],
                      [0,depth-sheet/2,(height-sheet)/2]),
        ('lower_rear_rail',[width-2*sheet,.025,sheet],[0,depth-.0125,sheet/2]),
    ):
        parts.append(library.box(size,center,name=name,material='steel',bevel=0))
    # Recessed mesh filters are visual thin pieces, so neither an opaque physics
    # box nor a whole-canopy convex hull closes the hollow interior.
    filter_depth = depth-.065
    filter_width = (width-.07)/2
    for index,sign in enumerate((-1,1)):
        cx = sign*(filter_width/2+.008)
        parts.append(library.box([filter_width,filter_depth,.002],
                     [cx,depth/2,.014],name=f'underside_filter_{index}',
                     material='black',bevel=0,collision=False))
        for k,x in enumerate(np.linspace(cx-filter_width/2+.009,cx+filter_width/2-.009,12)):
            parts.append(library.cylinder(.0009,filter_depth-.006,[x,depth/2,.0115],[0,1,0],
                         name=f'filter_wire_{index}_{k}',material='steel',segments=6,collision=False))
    library.validate(spec)
    return spec


def _audit(spec):
    """Triangle topology plus explicitly sampled convex-envelope deviation."""
    import trimesh
    rows = []
    for p in spec['parts']:
        m = trimesh.Trimesh(p['vertices'],p['faces'],process=False)
        row = {'name':p['name'], 'collision':bool(p.get('collision')), 'vertices':len(m.vertices),
               'triangles':len(m.faces), 'watertight':bool(m.is_watertight),
               'winding_consistent':bool(m.is_winding_consistent), 'signed_volume':float(m.volume),
               'minimum_triangle_area':float(m.area_faces.min()), 'strictly_convex':bool(m.is_convex)}
        if p.get('collision'):
            h = m.convex_hull
            row.update(collision_envelope_closed=bool(h.is_watertight),
                       collision_envelope_outward=bool(h.is_winding_consistent and h.volume>0),
                       collision_envelope_strictly_convex=bool(h.is_convex),
                       hull_volume_excess_fraction=float(max(0,h.volume/m.volume-1)))
            if p['name'].startswith('maple_shell_'):
                triangles = h.triangles
                samples = np.concatenate([h.vertices,triangles.mean(1),
                    (triangles[:,0]+triangles[:,1])/2,(triangles[:,1]+triangles[:,2])/2,
                    (triangles[:,2]+triangles[:,0])/2])
                _,dist,_ = trimesh.proximity.closest_point_naive(m,samples)
                row['hull_to_patch_sampled_max_distance_nominal_m'] = float(dist.max())
                row['hull_distance_samples'] = len(samples)
        if p['name'].endswith('_continuous_visual'):
            row['connected_components'] = len(m.split(only_watertight=False))
        rows.append(row)
    colliders = [r for r in rows if r['collision']]
    shell = [r for r in colliders if r['name'].startswith('maple_shell_')]
    return {'part_count':len(rows), 'collider_count':len(colliders),
            'all_closed_outward_nondegenerate':all(r['watertight'] and r['winding_consistent']
                        and r['signed_volume']>0 and r['minimum_triangle_area']>1e-13 for r in rows),
            'strictly_convex_part_count':sum(r['strictly_convex'] for r in rows),
            'all_collision_envelopes_closed_outward_convex':all(r['collision_envelope_closed']
                       and r['collision_envelope_outward'] and r['collision_envelope_strictly_convex']
                       for r in colliders),
            'shell_hull_max_volume_excess_fraction':max((r['hull_volume_excess_fraction'] for r in shell),default=0),
            'shell_hull_sampled_max_gap_nominal_m':max((r['hull_to_patch_sampled_max_distance_nominal_m']
                                                      for r in shell),default=0),
            'hull_gap_is_sampled_not_a_continuous_bound':True,
            'parts':rows}


def _preview(output,source_run):
    """Isolated reproducible geometry diagnostics; does not export a scene."""
    import html
    from PIL import Image, ImageDraw, ImageFont
    import physical_assets as A
    import physical_polish as P

    output = Path(output).resolve()
    source_run = Path(source_run).resolve()
    output.mkdir(parents=True,exist_ok=True)
    checks = []
    def check(name,value,detail=None):
        checks.append({'name':name,'passed':bool(value),'detail':detail})
    frames = json.loads((source_run/'frames.json').read_text())
    by_name = {v['name']:v for v in frames['frames']}
    check('source_movie_hash_matches',frames['source_sha256']==SOURCE_SHA256)
    evidence = copy.deepcopy(SOURCE_EVIDENCE)
    for e in evidence:
        p = source_run/'images'/e['name']
        row = by_name[e['name']]
        e['image_path'] = str(p)
        check('image_sha_'+e['name'],_sha(p)==e['sha256']==row['sha256'])
        check('exact_pts_'+e['name'],row['time_seconds']==e['time_seconds'])
    before = A.stool()
    before_copy = copy.deepcopy(before)
    after = refine_stool(before)
    check('input_spec_not_mutated',before==before_copy)
    check('refinement_idempotent',refine_stool(after)==after)
    check('bodies_and_joints_unchanged',before['bodies']==after['bodies'] and before['joints']==after['joints'])
    non_shell = lambda s: [p for p in s['parts'] if not p['name'].startswith('maple_shell_')]
    check('all_supports_unchanged',non_shell(before)==non_shell(after))
    check('dimension_metadata_unchanged',before['metadata']['dimensions_nominal_m']==after['metadata']['dimensions_nominal_m'])
    before_polished = P.polish('stool_before',copy.deepcopy(before),A)
    after_polished = P.polish('stool_after',copy.deepcopy(after),A)
    try:
        refine_stool(before_polished)
    except ValueError:
        check('wrong_polish_order_rejected',True)
    else:
        check('wrong_polish_order_rejected',False)
    hood_before = A.range_hood(width=.91,depth=.48,height=.43)
    hood_after = source_hood(A)
    specs = {'stool_before':before_polished,'stool_after':after_polished,
             'hood_before':hood_before,'hood_after':hood_after}
    audits = {}
    for name,s in specs.items():
        A.validate(s)
        audits[name] = _audit(s)
        check(name+'_closed_outward_nondegenerate',audits[name]['all_closed_outward_nondegenerate'])
        check(name+'_convex_collision_envelopes',audits[name]['all_collision_envelopes_closed_outward_convex'])
        if name.startswith('stool'):
            welded = next(r for r in audits[name]['parts'] if r['name']=='maple_shell_continuous_visual')
            check(name+'_one_closed_visible_shell',welded['watertight'] and welded['connected_components']==1)
    check('refined_shell_sampled_hull_gap_below_1mm_nominal',
          audits['stool_after']['shell_hull_sampled_max_gap_nominal_m']<.001,
          audits['stool_after']['shell_hull_sampled_max_gap_nominal_m'])
    check('hood_all_parts_strictly_convex',
          audits['hood_after']['strictly_convex_part_count']==audits['hood_after']['part_count'])
    check('hood_has_full_width_front_and_no_chimney',
          any(p['name']=='full_width_sloped_front' for p in hood_after['parts']) and
          not any('chimney' in p['name'] for p in hood_after['parts']))
    # Independent cavity point control: no hull of any hood collider contains the
    # central air volume. This is an open shell, not a solid decorative appliance.
    from scipy.spatial import ConvexHull
    cavity_point = np.array([0.,.36,.21])
    contains = []
    for p in hood_after['parts']:
        if p.get('collision'):
            hull = ConvexHull(np.asarray(p['vertices']))
            contains.append(np.all(hull.equations[:,:3]@cavity_point+hull.equations[:,3]<-1e-7))
    check('hood_central_cavity_remains_empty',not any(contains))
    for index,dims in enumerate((dict(width=.38,depth=.41,seat_height=.61,back_height=.25),
                                  dict(width=.48,depth=.50,seat_height=.68,back_height=.35))):
        variant = P.polish('parameter_probe',refine_stool(A.stool(**dims)),A)
        A.validate(variant)
        result = _audit(variant)
        check('parameterized_stool_'+str(index),result['all_closed_outward_nondegenerate']
              and result['all_collision_envelopes_closed_outward_convex'],dims)
    artifacts = []
    def record(p,kind):
        artifacts.append({'path':str(p.resolve()),'name':p.name,'type':kind,'sha256':_sha(p)})
    # Use the existing CPU depth-buffered preview renderer. Invisible collision
    # patches are excluded; the one welded visible shell is what is rendered.
    for name,s in specs.items():
        visible = copy.deepcopy(s)
        visible['parts'] = [p for p in visible['parts'] if p.get('visibility')!='invisible']
        if name.startswith('stool'):
            for p in visible['parts']:
                v = np.asarray(p['vertices'])
                v[:,:2] *= -1  # Rear oblique view exposes the source-facing backs.
                p['vertices'] = v.tolist()
        p = output/(name+'.png')
        A._preview_png(visible,p,size=720)
        record(p,'diagnostic_render')
        p = output/(name+'.json')
        p.write_text(json.dumps(s,separators=(',',':'),allow_nan=False)+'\n')
        record(p,'asset_spec')
    for e in evidence:
        im = Image.open(e['image_path']).convert('RGB')
        for subject in ('stool','hood'):
            key = subject+'_bbox_xyxy'
            if key not in e:
                continue
            p = output/(e['name'].replace('.jpg','')+'_'+subject+'_crop.jpg')
            im.crop(e[key]).save(p,quality=96)
            record(p,'source_crop')
    # A single review sheet keeps source evidence and before/after geometry
    # visible together; perspective and lighting are intentionally not fitted.
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path,22)
    small = ImageFont.truetype(font_path,17)
    sheet = Image.new('RGB',(1440,1720),(243,244,246))
    draw = ImageDraw.Draw(sheet)
    draw.text((24,18),'Source-guided shape refinement | nominal dimensions, not measured',font=font,fill=(20,28,36))
    placements = [('frame_001499_stool_crop.jpg',(24,62,720,285)),
                  ('frame_002158_stool_crop.jpg',(746,62,1416,285)),
                  ('stool_before.png',(24,333,708,1017)),('stool_after.png',(732,333,1416,1017)),
                  ('frame_001499_hood_crop.jpg',(24,1060,456,1320)),
                  ('hood_before.png',(466,1060,944,1538)),('hood_after.png',(938,1060,1416,1538))]
    for filename,rect in placements:
        im = Image.open(output/filename).convert('RGB')
        im.thumbnail((rect[2]-rect[0],rect[3]-rect[1]),Image.Resampling.LANCZOS)
        sheet.paste(im,((rect[0]+rect[2]-im.width)//2,(rect[1]+rect[3]-im.height)//2))
    for text,xy in [('49.971667 s: three stool backs',(24,291)),('71.940000 s: rounded tops from reverse side',(746,291)),
                    ('Before: current raw library + polish',(24,309)),('After: rounded corners + same polish',(732,309)),
                    ('49.971667 s: actual broad hood',(24,1029)),('Before: narrow chimney',(466,1029)),
                    ('After: full-width slope',(958,1029))]:
        draw.text(xy,text,font=small,fill=(28,36,44))
    draw.text((24,1555),'Geometry diagnostics use neutral colors; scene textures, cameras and articulation integration stay with the main scene.',font=small,fill=(32,38,46))
    draw.text((24,1590),'Stool: 174 thin collision patches + one welded visible shell. Hood: separate convex panels around a hollow enclosure.',font=small,fill=(32,38,46))
    draw.text((24,1625),'Collision-envelope deviation is sampled; native contact/placement verification is performed separately by the scene owner.',font=small,fill=(32,38,46))
    p = output/'comparison.jpg'
    sheet.save(p,quality=94)
    record(p,'comparison_sheet')
    receipt = {'schema':'real2sim-physical-refinement-audit/v1',
               'status':'passed' if all(c['passed'] for c in checks) else 'failed',
               'produced_at':datetime.now(timezone.utc).isoformat(),
               'source_sha256':SOURCE_SHA256, 'source_evidence':evidence,
               'source_frames_manifest_sha256':_sha(source_run/'frames.json'),
               'script_sha256':_sha(__file__), 'library_sha256':_sha(A.__file__),
               'polish_sha256':_sha(P.__file__), 'checks':checks,'audits':audits,'artifacts':artifacts,
               'limits':['RGB silhouette inference; no measured dimensions or physical mechanisms.',
                         'Diagnostic colors and viewpoints are not a claim of photorealistic scene matching.',
                         'Curved raw shell patches are not strictly convex. Their separately audited convex hulls are the dynamic collision representation.',
                         'Hull gap is sampled at hull vertices, edge midpoints and triangle centroids; it is not an analytic global bound.',
                         'No root scene, camera, material, body placement or joint state is changed by this module.']}
    p = output/'tests.json'
    p.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    status = receipt['status']
    (output/'README.md').write_text(
        '# Source-guided stool and hood geometry\n\n'
        '`refine_stool(spec)` returns a copy and must run before `physical_polish.polish`; '
        '`source_hood(library, width=.91, depth=.48, height=.43)` returns a new plain asset specification.\n\n'
        'Both functions preserve the asset library frame: centered X, +Y toward the back wall, +Z up. '
        'The hood origin is its lowest front edge. Root placement should supply the inferred base height separately. '
        'Revolute joint units remain radians; these refinements add no joints.\n\n'
        'The stool profile is clipped to a rounded upper outline and its shallow transverse cup is strengthened. '
        'The continuous shell is welded from exactly the same shared vertices as the thin collision patches. '
        'Supports, body origin, mass and existing joint information are unchanged. The native simulator receives '
        'convex hulls per thin patch; it must never receive one convex hull of the whole stool.\n\n'
        'The hood uses full-width sloped sheet geometry and a narrow lower lip, as visible at 49.971667 s. '
        'Its enclosure is hollow; its underside filter construction and nominal dimensions are inferred. '
        'The old/new hood diagnostic uses identical overall dimensions to isolate the silhouette change.\n\n'
        f'Verification status: **{status}**, {sum(c["passed"] for c in checks)}/{len(checks)} checks. '
        f'Default refined shell maximum sampled hull gap: {audits["stool_after"]["shell_hull_sampled_max_gap_nominal_m"]*1000:.4f} nominal mm. '
        'This sampled result is neither a continuous geometric bound nor a measurement of real-world accuracy.\n\n'
        'See `tests.json` for exact image PTS/hashes, source/code bindings, per-part normals/topology/convexity diagnostics '
        'and the comparison artifact hashes. Native collision and source-camera placement checks remain separate.\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Source shape refinement</title>'
        '<style>body{font:17px system-ui;margin:24px;background:#f3f4f6;color:#17212b}img{width:100%;max-width:1440px}a{color:#205d91}</style>'
        '<h1>Stool and hood shape refinement</h1><p>Source silhouettes guide nominal geometry; dimensions and hidden construction remain inferred.</p>'
        '<img src="comparison.jpg" alt="Source crops and before/after mesh diagnostic renders">'
        '<p>Verification: '+html.escape(status)+'. <a href="tests.json">Exact receipt and per-part audit</a> · '
        '<a href="README.md">Interfaces and limitations</a></p>')
    print(json.dumps({'status':status,'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),
                      'receipt':str(p),'receipt_sha256':_sha(p),'script_sha256':receipt['script_sha256'],
                      'sampled_shell_hull_gap_nominal_m':audits['stool_after']['shell_hull_sampled_max_gap_nominal_m']}))
    if status != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    args = parser.parse_args()
    _preview(args.preview,args.source_run)
