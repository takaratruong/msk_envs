#!/usr/bin/env python3
"""Apply source-inspected TV overrides and restore clear detector false positives."""
import hashlib,json
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--labels',default='geometry_masks')
    p.add_argument('--manual-camera',type=Path)
    a=p.parse_args();root=a.run.resolve();out=root/'appearance_masks';geo=root/'camera_masks'
    out.mkdir(exist_ok=True);geo.mkdir(exist_ok=True)
    frames=json.loads((root/'frames.json').read_text());detections=json.loads((root/'source_masks'/'detections.json').read_text())
    labels=json.loads((root/a.labels/'receipt.json').read_text())
    manual=json.loads(a.manual_camera.read_text()) if a.manual_camera else {'polygons':{}}
    if a.manual_camera and manual['source_sha256']!=frames['source_sha256']:
        raise RuntimeError('Manual camera annotations are from a different source.')
    if a.manual_camera:
        recorded={x['name']:x for x in manual['inputs']}
        actual={x['name']:x for x in frames['frames']}
        for name in manual['polygons']:
            if name not in recorded or name not in actual or recorded[name]['sha256']!=actual[name]['sha256']:
                raise RuntimeError('Manual annotation source binding missing: '+name)
            if hashlib.sha256(Path(actual[name]['image_path']).read_bytes()).hexdigest()!=recorded[name]['sha256']:
                raise RuntimeError('Annotated source pixels changed: '+name)
    by={x['name']:x for x in detections['views']};label_by={x['name']:x for x in labels['views']}
    # Conservative enclosing rectangles traced from inspection/tv_closeups.jpg.
    # Coordinates are fractions of the source image, keyed by exact frame ID.
    boxes={1698:[0,0,.46,.18],1708:[0,0,.66,.43],1717:[0,0,.87,.61],1719:[0,0,.91,.65],
      1730:[0,0,1,.69],1739:[0,0,1,.69],1741:[0,0,1,.69],1753:[.07,0,1,.74],
      1755:[.09,0,1,.74],1765:[.23,0,1,.51],1770:[.25,0,1,.43],
      1825:[0,0,.55,.11],1832:[0,0,.55,.36],1844:[0,.14,.53,.60],
      1845:[0,.19,.53,.72],1853:[0,.19,.41,.74],1860:[0,.14,.22,.68]}
    boxes[1844]=[0,.14,.53,.72]
    boxes.update({796:[0,0,.14,.12],808:[0,.08,.25,.33],815:[0,.12,.36,.40],
      820:[0,.15,.44,.43],827:[.10,.17,.54,.46],838:[.33,.15,.76,.42],
      847:[.49,.07,.99,.37],854:[.58,.03,1,.35],858:[.64,.025,1,.35],863:[.55,.03,1,.35]})
    secondary={847:[0,.15,.10,.37],854:[0,.13,.18,.37],858:[0,.13,.18,.37],863:[0,.13,.13,.36]}
    assert set(boxes).issubset({x['source_index'] for x in frames['frames']})
    reports=[];sheets=[]
    for item in frames['frames']:
        w,h=item['width'],item['height'];t=item['time_seconds']
        mask=np.full((h,w),255,np.uint8);kept=[];restored=[]
        for d in by[item['name']]['detections']:
            x0,y0,x1,y1=d['bbox_xyxy'];aspect=(x1-x0)/max(y1-y0,1)
            valid=(d['category']=='tv' and aspect>=1.25 and not (55.5<=t<63.5 or 64.5<=t<67.5)) or (d['category']=='person' and t>=72.5)
            if valid:
                cv2.rectangle(mask,(x0,y0),(x1,y1),0,-1);kept.append(d)
            else:restored.append(d)
        if item['source_index'] in boxes:
            b=np.array(boxes[item['source_index']])*[w,h,w,h]
            cv2.rectangle(mask,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),0,-1)
        if item['source_index'] in secondary:
            b=np.array(secondary[item['source_index']])*[w,h,w,h]
            cv2.rectangle(mask,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),0,-1)
        path=out/(item['name']+'.png');cv2.imwrite(str(path),mask)
        gm=mask.copy()
        for label in label_by[item['name']]['labels']:
            cv2.fillConvexPoly(gm,np.round(label['quad']).astype(int),0)
        for poly in manual['polygons'].get(item['name'],[]):
            cv2.fillPoly(gm,[np.array(poly,dtype=np.int32)],0)
        cv2.imwrite(str(geo/(item['name']+'.png')),gm)
        reports.append(dict(name=item['name'],time_seconds=t,mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          mask_path=str(path),valid_fraction=float((mask>0).mean()),retained_detections=kept,
          restored_detections=restored,manual_tv_bbox=boxes.get(item['source_index']),
          secondary_display_bbox=secondary.get(item['source_index']),
          camera_mask_sha256=hashlib.sha256((geo/(item['name']+'.png')).read_bytes()).hexdigest()))
        if 55.5<=t<63.5 or (kept and item['bin']%4==0):
            im=cv2.imread(item['image_path']);red=np.full_like(im,(20,20,240));im[mask==0]=(im[mask==0]*.4+red[mask==0]*.6).astype(np.uint8)
            sheets.append((cv2.resize(im,(480,270)),t))
    meta=dict(schema='real2sim-reviewed-masks/v1',source_sha256=frames['source_sha256'],
       source_detections_sha256=hashlib.sha256((root/'source_masks'/'detections.json').read_bytes()).hexdigest(),
       sign_templates_receipt_sha256=hashlib.sha256((root/a.labels/'receipt.json').read_bytes()).hexdigest(),
       manual_camera_sha256=hashlib.sha256(a.manual_camera.read_bytes()).hexdigest() if a.manual_camera else None,
       protocol='Detector boxes restricted by inspected object aspect/time; source-inspected screen close-up rectangles. Conservative rectangles can exclude some static pixels; not pixel-perfect segmentation.',views=reports)
    (out/'detections.json').write_text(json.dumps(meta,indent=2)+'\n')
    (geo/'receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
    for page in range((len(sheets)+23)//24):
        chunk=sheets[page*24:(page+1)*24];sheet=Image.new('RGB',(1920,298*((len(chunk)+3)//4)),(20,22,28));draw=ImageDraw.Draw(sheet)
        for i,(im,t) in enumerate(chunk):
            x,y=i%4*480,i//4*298;sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+5,y+277),f'{t:.3f}s red=excluded',fill='white')
        sheet.save(out/f'review_{page+1}.jpg',quality=93)
    print('Corrected',len(reports),'appearance/camera masks',flush=True)


if __name__=='__main__':main()
