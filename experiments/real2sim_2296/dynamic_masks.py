#!/usr/bin/env python3
"""Conservative TV/person masks; detector outputs are candidates, not object truth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torchvision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.run.resolve()
    manifest=json.loads((root/'frames.json').read_text())
    out=root/'source_masks';out.mkdir(exist_ok=True)
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights).eval().cuda()
    categories=weights.meta['categories']
    reports=[];thumbs=[]
    with torch.inference_mode():
        for i,entry in enumerate(manifest['frames']):
            image=Image.open(entry['image_path']).convert('RGB')
            arr=np.array(image)
            x=torch.from_numpy(arr).permute(2,0,1).float().cuda()/255
            pred=model([x])[0]
            mask=Image.new('L',image.size,255);d=ImageDraw.Draw(mask)
            overlay=image.copy();ov=ImageDraw.Draw(overlay)
            boxes=[]
            for box,score,label in zip(pred['boxes'].cpu().numpy(),pred['scores'].cpu().numpy(),pred['labels'].cpu().numpy()):
                category=categories[int(label)]
                if category not in ('tv','person') or score<.65: continue
                box=np.round(box).astype(int)
                box[[0,2]]=np.clip(box[[0,2]]+np.array([-6,6]),0,image.width-1)
                box[[1,3]]=np.clip(box[[1,3]]+np.array([-6,6]),0,image.height-1)
                d.rectangle(box.tolist(),fill=0)
                ov.rectangle(box.tolist(),outline=(255,60,60),width=6)
                boxes.append(dict(category=category,score=float(score),bbox_xyxy=box.tolist()))
            path=out/(entry['name']+'.png');mask.save(path)
            reports.append(dict(name=entry['name'],time_seconds=entry['time_seconds'],detections=boxes,
              valid_fraction=float((np.array(mask)>0).mean()),mask_path=str(path),
              mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if boxes:
                overlay.thumbnail((480,270));thumbs.append((overlay,entry['time_seconds'],boxes))
            if i%20==0:print(f'masked {i+1}/{len(manifest["frames"])}',flush=True)
    (out/'detections.json').write_text(json.dumps(dict(source_sha256=manifest['source_sha256'],
       model='torchvision FasterRCNN_ResNet50_FPN_Weights.DEFAULT',score_threshold=.65,
       caveat='Axis-aligned conservative detector boxes; inspect contact sheets. Glass/reflections are not masked.',
       views=reports),indent=2)+'\n')
    for page in range((len(thumbs)+23)//24):
        chunk=thumbs[page*24:(page+1)*24]
        sheet=Image.new('RGB',(1920,298*((len(chunk)+3)//4)),(20,22,28));draw=ImageDraw.Draw(sheet)
        for i,(im,t,boxes) in enumerate(chunk):
            x,y=i%4*480,i//4*298;sheet.paste(im,(x,y))
            draw.text((x+4,y+277),f'{t:.3f}s '+','.join(b['category'] for b in boxes),fill='white')
        sheet.save(out/f'review_{page+1}.jpg',quality=92)
    print('Finished detector candidates:',len(thumbs),'views with masks',flush=True)


if __name__=='__main__':main()
