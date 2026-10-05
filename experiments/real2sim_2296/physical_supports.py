#!/usr/bin/env python3
"""Attach the four stool uprights to the actual shell without moving the feet.

Call ``adjust_stool_supports(refine_stool(A.stool()), A)`` before polish. This
returns a copy. The source shows slender uprights meeting the shell; the exact
hidden attachment mechanism remains inferred. No separate scene is exported.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _shell_triangles(spec):
    parts = [p for p in spec['parts'] if p['name'].startswith('maple_shell_')]
    if not parts or any(p['name'].endswith('_continuous_visual') for p in parts):
        raise ValueError('Support adjustment requires raw stool patches, before polish')
    return np.concatenate([np.asarray(p['vertices'],dtype=float)[np.asarray(p['faces'],dtype=int)] for p in parts])


def _ray_hits(origin,direction,triangles):
    e1 = triangles[:,1]-triangles[:,0]
    e2 = triangles[:,2]-triangles[:,0]
    h = np.cross(np.broadcast_to(direction,e2.shape),e2)
    determinant = np.einsum('ij,ij->i',e1,h)
    inverse = np.divide(1.,determinant,out=np.zeros_like(determinant),where=np.abs(determinant)>1e-11)
    offset = origin-triangles[:,0]
    u = inverse*np.einsum('ij,ij->i',offset,h)
    q = np.cross(offset,e1)
    v = inverse*(q@direction)
    distance = inverse*np.einsum('ij,ij->i',e2,q)
    valid = (np.abs(determinant)>1e-11)&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(distance>1e-8)
    ids = np.flatnonzero(valid)
    return ids, distance


def adjust_stool_supports(spec, library):
    """Extend existing leg axes to the shell; keep bottom endpoints and gauge.

    Floor skids, rubber feet, braces, shell, bodies, joints and mass are unchanged.
    Each raw upright is replaced by one closed two-ring cylinder with the original
    endpoint ordering. The existing polish step then makes its rounded capsule.
    A 0.15 mm nominal weld overlap absorbs capsule tessellation at the attachment;
    this is a modeled mechanical assumption, not measured joining hardware.
    """
    if spec.get('metadata',{}).get('asset_type') != 'maple_sled_stool':
        raise ValueError('Expected an unplaced maple_sled_stool asset')
    triangles = _shell_triangles(spec)
    result = copy.deepcopy(spec)
    if result['metadata'].get('support_attachment_adjustment',{}).get('version') == 1:
        return result
    expected = {f'sled_{side}_segment_{segment:02d}' for side in (-1,1) for segment in (0,2)}
    present = {p['name'] for p in spec['parts'] if p['name'] in expected}
    if present != expected:
        raise ValueError('Expected the four raw upright segments from the frozen stool library')
    attachments = []
    weld = .00015
    for index,p in enumerate(spec['parts']):
        if p['name'] not in expected:
            continue
        vertices = np.asarray(p['vertices'],dtype=float)
        sides = (len(vertices)-2)//2
        if len(vertices) != 2*sides+2 or sides != 12:
            raise ValueError('Expected original 12-sided two-ring upright, before polish')
        endpoints = vertices[-2:].copy()
        lower_index = int(np.argmin(endpoints[:,2]))
        upper_index = 1-lower_index
        foot = endpoints[lower_index]
        old_tip = endpoints[upper_index].copy()
        direction = old_tip-foot
        old_length = np.linalg.norm(direction)
        direction /= old_length
        radius = float(np.median(np.linalg.norm(vertices[:sides]-endpoints[0],axis=1)))
        ids,distances = _ray_hits(foot,direction,triangles)
        if not len(ids):
            raise ValueError('Existing upright axis misses the stool shell: '+p['name'])
        hit_index = int(ids[np.argmin(distances[ids])])
        hit = foot+direction*distances[hit_index]
        triangle = triangles[hit_index]
        normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
        normal /= np.linalg.norm(normal)
        incidence = abs(float(np.dot(normal,direction)))
        if incidence < .2:
            raise ValueError('Grazing support attachment would need a different mount design')
        # A capsule is a segment swept by a sphere. Its endpoint center must stop
        # radius/incidence before the centerline hit, not one radius along Z.
        new_tip = hit-direction*(radius-weld)/incidence
        new_length = np.linalg.norm(new_tip-foot)
        if new_length < old_length or new_length-old_length > .4*spec['metadata']['dimensions_nominal_m']['back_height']:
            raise ValueError('Support correction exceeds the bounded source-guided adjustment')
        endpoints[upper_index] = new_tip
        part = library.cylinder(radius,new_length,endpoints.mean(axis=0),endpoints[1]-endpoints[0],
                               name=p['name'],material=p['material'],body=p['body'],segments=sides,
                               collision=p.get('collision',True))
        result['parts'][index] = part
        attachments.append({'part':p['name'],'floor_endpoint':foot.tolist(),
            'old_upper_endpoint':old_tip.tolist(),'new_upper_endpoint':new_tip.tolist(),
            'shell_axis_intersection':hit.tolist(),'shell_triangle':triangle.tolist(),
            'outward_shell_normal':normal.tolist(),'axis':direction.tolist(),
            'radius_nominal_m':radius,'axis_extension_nominal_m':float(new_length-old_length),
            'normal_incidence':incidence,'nominal_weld_overlap_m':weld})
    result['metadata']['support_attachment_adjustment'] = {
        'version':1,'status':'inferred_geometry_not_measured',
        'method':'Extend original upright centerlines to first actual shell hit, stopping by capsule radius / normal incidence',
        'source_evidence':'frame_001499.jpg at 49.971667 s shows uprights meeting shell; frame_002158.jpg at 71.94 s constrains reverse silhouette',
        'source_sha256':spec['metadata'].get('source_sha256'),
        'floor_feet_skids_braces_shell_bodies_joints_unchanged':True,
        'tube_gauge_unchanged':True,'new_mount_parts':0,
        'attachment_weld_overlap_nominal_m':weld,
        'attachments':attachments,
        'limitation':'No hidden screw, weld or mounting bracket was measured or recovered from the RGB source',
    }
    library.validate(result)
    return result


def _welded_shell_mesh(spec):
    import trimesh
    p = next(p for p in spec['parts'] if p['name']=='maple_shell_continuous_visual')
    return trimesh.Trimesh(p['vertices'],p['faces'],process=False)


def _attachment_audit(spec):
    """Dense deterministic surface samples, with segment/triangle crossing control."""
    import trimesh
    shell = _welded_shell_mesh(spec)
    result = []
    names = {f'sled_{side}_segment_{segment:02d}' for side in (-1,1) for segment in (0,2)}
    # Eight subdivisions per triangle sample the rounded end caps more densely
    # than their visual mesh. This reports a sampled gap, not a global proof.
    bary = np.array([[i/8,j/8,1-(i+j)/8] for i in range(9) for j in range(9-i)])
    for p in spec['parts']:
        if p['name'] not in names:
            continue
        m = trimesh.Trimesh(p['vertices'],p['faces'],process=False)
        low = m.vertices[:,2].min()
        tri = m.triangles
        tri = tri[tri[:,:,2].max(1)>low+(m.vertices[:,2].max()-low)*.90]
        samples = np.unique(np.round(np.einsum('bi,tij->tbj',bary,tri).reshape(-1,3),12),axis=0)
        _,distance,_ = trimesh.proximity.closest_point_naive(shell,samples)
        # Independent odd/even containment on the welded shell estimates the
        # intentionally tiny same-body weld penetration. All interior points of
        # the 12 mm shell are within 10 mm of its boundary.
        ray_direction = np.array([.371,.599,.709])
        ray_direction /= np.linalg.norm(ray_direction)
        penetration = []
        for sample_index in np.flatnonzero(distance<.01):
            ids,ray_distances = _ray_hits(samples[sample_index],ray_direction,shell.triangles)
            ts = np.sort(ray_distances[ids])
            unique_count = int(len(ts)>0)+int(np.sum(np.diff(ts)>1e-8))
            if unique_count%2:
                penetration.append(float(distance[sample_index]))
        crossings = 0
        for edge in m.edges_unique:
            a,b = m.vertices[edge]
            delta = b-a
            length = np.linalg.norm(delta)
            ids,distances = _ray_hits(a,delta/length,shell.triangles)
            crossings += int(np.any(distances[ids]<length-1e-8))
        result.append({'part':p['name'],'attachment_surface_samples':len(samples),
                       'sampled_min_gap_nominal_m':float(distance.min()),
                       'sampled_inside_count':len(penetration),
                       'sampled_max_weld_penetration_nominal_m':max(penetration,default=0.),
                       'surface_edge_crossings_with_shell':crossings,
                       'watertight':bool(m.is_watertight),'outward':bool(m.is_winding_consistent and m.volume>0),
                       'strictly_convex':bool(m.is_convex),'volume_nominal_m3':float(m.volume)})
    return result


def _preview(output,source_run):
    from PIL import Image,ImageDraw,ImageFont
    import physical_assets as A
    import physical_refinements as R
    import physical_polish as P
    import trimesh
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    source_run = Path(source_run).resolve()
    source_frame = source_run/'images/frame_001499.jpg'
    frame_row = next(r for r in json.loads((source_run/'frames.json').read_text())['frames'] if r['name']==source_frame.name)
    before = R.refine_stool(A.stool())
    saved_before = copy.deepcopy(before)
    after = adjust_stool_supports(before,A)
    checks = []
    def check(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
    check('source_frame_exact_hash',_sha(source_frame)==frame_row['sha256']=='8d84e5e894ae4fc1a206c69660160ead6dfecf969c16df5276e506ac9c9030ab')
    check('source_exact_pts',frame_row['time_seconds']==49.971667)
    check('input_not_mutated',before==saved_before)
    check('idempotent',adjust_stool_supports(after,A)==after)
    check('body_joint_mass_unchanged',before['bodies']==after['bodies'] and before['joints']==after['joints'])
    changed = {r['part'] for r in after['metadata']['support_attachment_adjustment']['attachments']}
    check('only_four_upright_parts_change',len(changed)==4 and
          all(a==b for a,b in zip(before['parts'],after['parts']) if a['name'] not in changed))
    for a,b in zip(before['parts'],after['parts']):
        if a['name'] not in changed:
            continue
        va,vb = np.asarray(a['vertices']),np.asarray(b['vertices'])
        ia,ib = np.argmin(va[-2:,2]),np.argmin(vb[-2:,2])
        check(a['name']+'_floor_endpoint_preserved',np.allclose(va[-2:][ia],vb[-2:][ib],atol=1e-12,rtol=0))
        da,db = np.diff(va[-2:],axis=0)[0],np.diff(vb[-2:],axis=0)[0]
        check(a['name']+'_upright_axis_preserved',np.linalg.norm(np.cross(da,db))<1e-12)
        check(a['name']+'_radius_preserved',abs(np.linalg.norm(va[0]-va[-2])-np.linalg.norm(vb[0]-vb[-2]))<1e-12)
    pb,pa = P.polish('stool_before',copy.deepcopy(before),A),P.polish('stool_after',copy.deepcopy(after),A)
    A.validate(pa)
    audit_before,audit_after = _attachment_audit(pb),_attachment_audit(pa)
    check('all_adjusted_capsules_closed_outward_convex',all(r['watertight'] and r['outward'] and r['strictly_convex'] for r in audit_after))
    max_gap = max(r['sampled_min_gap_nominal_m'] for r in audit_after)
    check('maximum_sampled_attachment_gap_under_half_mm_nominal',max_gap<.0005,max_gap)
    max_penetration = max(r['sampled_max_weld_penetration_nominal_m'] for r in audit_after)
    check('sampled_same_body_weld_penetration_under_half_mm',max_penetration<.0005,max_penetration)
    volume_before = sum(r['volume_nominal_m3'] for r in audit_before)
    volume_after = sum(r['volume_nominal_m3'] for r in audit_after)
    check('upright_volume_growth_under_8_percent',volume_after/volume_before<1.08,volume_after/volume_before-1)
    # All unchanged parts, including shell and lower skid capsules, remain exact
    # through the same polish operation. Supports are a same-body welded join.
    check('polished_shell_skids_feet_braces_unchanged',all(a==b for a,b in zip(pb['parts'],pa['parts']) if a['name'] not in changed))
    shell = _welded_shell_mesh(pa)
    check('shell_still_one_closed_outward_component',shell.is_watertight and shell.is_winding_consistent and
          shell.volume>0 and len(shell.split(only_watertight=False))==1)
    for i,dimensions in enumerate((dict(width=.38,depth=.41,seat_height=.61,back_height=.25),
                                    dict(width=.48,depth=.50,seat_height=.68,back_height=.35))):
        s = adjust_stool_supports(R.refine_stool(A.stool(**dimensions)),A)
        A.validate(s)
        check('parameterized_stool_'+str(i),len(s['metadata']['support_attachment_adjustment']['attachments'])==4)
    artifacts = []
    def record(path):
        artifacts.append({'path':str(path),'sha256':_sha(path)})
    for name,s in [('before',pb),('after',pa)]:
        visible = copy.deepcopy(s)
        visible['parts'] = [p for p in visible['parts'] if p.get('visibility')!='invisible']
        for p in visible['parts']:
            v = np.asarray(p['vertices']);v[:,:2] *= -1;p['vertices'] = v.tolist()
        path = output/(name+'.png');A._preview_png(visible,path,size=800);record(path)
    Image.open(source_frame).crop((805,426,1384,748)).save(output/'source_stools.jpg',quality=96)
    record(output/'source_stools.jpg')
    sheet = Image.new('RGB',(1600,1230),(243,244,246));draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    draw.text((24,18),'Stool support attachment | same feet, axes, tube gauge and shell',font=font,fill=(20,28,36))
    im = Image.open(output/'source_stools.jpg');im.thumbnail((575,300));sheet.paste(im,(24,58))
    draw.text((640,78),'Source: frame 1499, 49.971667 s',font=font,fill=(20,28,36))
    draw.text((640,120),'Support axes extend to the actual shell underside.',font=small,fill=(35,45,55))
    draw.text((640,158),'Four upright tips change. Feet and lower skids remain fixed.',font=small,fill=(35,45,55))
    draw.text((640,196),'Hidden attachment construction and dimensions are inferred.',font=small,fill=(35,45,55))
    sheet.paste(Image.open(output/'before.png'),(0,390));sheet.paste(Image.open(output/'after.png'),(800,390))
    draw.text((24,360),'Before: visible support gap',font=font,fill=(20,28,36))
    draw.text((824,360),'After: uprights meet the shell',font=font,fill=(20,28,36))
    draw.text((24,1200),'Diagnostic colors; source-camera rendering and native physics validation remain with the scene integration.',font=small,fill=(35,45,55))
    sheet.save(output/'comparison.jpg',quality=94);record(output/'comparison.jpg')
    path = output/'adjusted_stool_raw.json';path.write_text(json.dumps(after,separators=(',',':'))+'\n');record(path)
    status = 'passed' if all(c['passed'] for c in checks) else 'failed'
    receipt = {'schema':'real2sim-stool-support-audit/v1','status':status,
               'produced_at':datetime.now(timezone.utc).isoformat(),'script_sha256':_sha(__file__),
               'library_sha256':_sha(A.__file__),'refinements_sha256':_sha(R.__file__),'polish_sha256':_sha(P.__file__),
               'source_sha256':before['metadata']['source_sha256'],'source_frame':frame_row,
               'checks':checks,'attachments':after['metadata']['support_attachment_adjustment']['attachments'],
               'before':audit_before,'after':audit_after,'max_sampled_attachment_gap_nominal_m':max_gap,
               'max_sampled_weld_penetration_nominal_m':max_penetration,
               'upright_volume_growth_fraction':volume_after/volume_before-1,'artifacts':artifacts,
               'limits':['Nominal geometry, not measured scale or recovered hardware.',
                         'Support-to-shell distance uses deterministic surface samples; it is not a continuous distance bound.',
                         'The 0.15 mm nominal weld overlap is intentional between pieces on the same rigid body.',
                         'No body mass, source camera, scene placement, floor contact location or shell shape is modified.']}
    path = output/'tests.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    (output/'README.md').write_text('# Stool support attachment\n\n'
        '`adjust_stool_supports(spec, library)` returns a copy. Apply after `refine_stool` and before `polish`. '
        'It extends the four original upright axes to the first actual shell intersection; lower endpoints, '
        'skids, feet, bars, shell, body origins, joints and mass remain unchanged. No new mount parts are added.\n\n'
        'Raw two-ring cylinders retain the endpoint convention required by the existing polish operation, '
        'which then produces four convex capsules. A 0.15 mm nominal same-body weld overlap is intentional. '
        'The hidden attachment hardware and physical dimensions were not observed or measured.\n\n'
        f'Status: {status}. Maximum sampled support attachment gap: {max_gap*1000:.5f} nominal mm. '
        f'Maximum sampled same-body weld penetration: {max_penetration*1000:.5f} nominal mm. '
        f'Upright volume increase: {(volume_after/volume_before-1)*100:.3f}%; tube radius is unchanged. '
        'This receipt verifies modeled geometry rather than real-world dimensional accuracy.\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Stool supports</title>'
        '<style>body{font:18px system-ui;margin:24px;background:#f3f4f6}img{width:100%;max-width:1600px}</style>'
        '<h1>Stool support attachment</h1><img src="comparison.jpg"><p><a href="tests.json">Bound receipt</a> · '
        '<a href="README.md">Interface and limitations</a></p>')
    print(json.dumps({'status':status,'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),
                      'script_sha256':receipt['script_sha256'],'receipt':str(path),'receipt_sha256':_sha(path),
                      'max_sampled_attachment_gap_nominal_m':max_gap,'upright_volume_growth_fraction':volume_after/volume_before-1}))
    if status != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    args = p.parse_args()
    _preview(args.preview,args.source_run)
