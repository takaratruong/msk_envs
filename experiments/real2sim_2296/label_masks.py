#!/usr/bin/env python3
"""Find repeated printed signs by planar template matching for SfM masks only."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-name',default='geometry_masks')
    a=p.parse_args();root=a.run.resolve();out=root/a.output_name;out.mkdir(exist_ok=True)
    manifest=json.loads((root/'frames.json').read_text());by={x['name']:x for x in manifest['frames']}
    reference=cv2.imread(by['frame_001582.jpg']['image_path'])
    # Manually inspected corners of the three physical printed sign designs.
    polygons={
      'landfill':[[648,539],[841,477],[827,732],[648,802]],
      'compost':[[1046,394],[1168,350],[1144,573],[1025,615]],
      'recycle':[[1327,298],[1411,267],[1376,450],[1287,493]],
    }
    sift=cv2.SIFT_create(nfeatures=14000,contrastThreshold=.015)
    templates=[]
    corners=np.float32([[0,0],[299,0],[299,419],[0,419]])
    for name,poly in polygons.items():
        H=cv2.getPerspectiveTransform(np.float32(poly),corners)
        patch=cv2.warpPerspective(reference,H,(300,420))
        cv2.imwrite(str(out/f'template_{name}.jpg'),patch)
        kp,desc=sift.detectAndCompute(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),None)
        templates.append((name,np.float32([x.pt for x in kp]),desc))
    # The original blue template is small and oblique. A separately inspected
    # closer view supplies legible detail, without altering any source image.
    blue_source=cv2.imread(by['frame_000664.jpg']['image_path'])
    blue_quad=np.float32([[1050,476],[1193,380],[1148,638],[1018,733]])
    patch=cv2.warpPerspective(blue_source,cv2.getPerspectiveTransform(blue_quad,corners),(300,420))
    cv2.imwrite(str(out/'template_recycle_close.jpg'),patch)
    kp,desc=sift.detectAndCompute(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),None)
    templates.append(('recycle_close',np.float32([x.pt for x in kp]),desc))
    matcher=cv2.BFMatcher();reports=[];reviews=[]
    for idx,item in enumerate(manifest['frames']):
        image=cv2.imread(item['image_path']);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp,desc=sift.detectAndCompute(gray,None)
        xy=np.float32([x.pt for x in kp])
        mask=cv2.imread(str(root/'source_masks'/(item['name']+'.png')),cv2.IMREAD_GRAYSCALE)
        if mask is None:raise RuntimeError('Complete source dynamic masks first.')
        overlay=image.copy();found=[]
        for name,refxy,refdesc in templates:
            matches=matcher.knnMatch(desc,refdesc,k=2) if desc is not None else []
            good=[m for m,n in matches if m.distance<.78*n.distance]
            for attempt in range(8):
                if len(good)<10:break
                src=np.float32([refxy[m.trainIdx] for m in good])
                dst=np.float32([xy[m.queryIdx] for m in good])
                H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=3000,confidence=.999)
                if H is None or inliers is None:break
                keep=inliers.ravel().astype(bool)
                if sum(keep)<10:break
                quad=cv2.perspectiveTransform(corners[None],H)[0]
                coverage=cv2.contourArea(cv2.convexHull(src[keep]))/(300*420)
                area=abs(cv2.contourArea(quad))
                sensible=(np.isfinite(quad).all() and cv2.isContourConvex(quad.astype(np.int32))
                   and coverage>.035 and 100<area<image.shape[0]*image.shape[1]*.2
                   and np.min(quad)>-image.shape[1] and np.max(quad)<image.shape[1]*2)
                if sensible:
                    mid=quad.mean(axis=0);expanded=mid+(quad-mid)*1.08
                    cv2.fillConvexPoly(mask,np.round(expanded).astype(int),0)
                    cv2.polylines(overlay,[np.round(expanded).astype(int)],True,(30,40,255),4)
                    found.append(dict(label=name,quad=expanded.tolist(),inliers=int(sum(keep)),template_hull_fraction=coverage))
                    good=[m for m in good if cv2.pointPolygonTest(quad,tuple(map(float,xy[m.queryIdx])),False)<0]
                else:
                    good=[m for j,m in enumerate(good) if not keep[j]]
        path=out/(item['name']+'.png');cv2.imwrite(str(path),mask)
        reports.append(dict(name=item['name'],time_seconds=item['time_seconds'],labels=found,
          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),masked_fraction=float((mask==0).mean())))
        if found:
            overlay=cv2.resize(overlay,(480,270));reviews.append((overlay,item['time_seconds']))
        if idx%25==0:print(f'label masks {idx+1}/{len(manifest["frames"])}; lastcount {len(found)}',flush=True)
    for page in range((len(reviews)+23)//24):
        chunk=reviews[page*24:(page+1)*24];sheet=Image.new('RGB',(1920,298*((len(chunk)+3)//4)),(20,22,28));draw=ImageDraw.Draw(sheet)
        for j,(im,t) in enumerate(chunk):
            x,y=j%4*480,j//4*298;sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(x,y));draw.text((x+6,y+277),f'{t:.3f}s',fill='white')
        sheet.save(out/f'review_{page+1}.jpg',quality=93)
    (out/'receipt.json').write_text(json.dumps(dict(schema='real2sim-camera-label-masks/v1',
       source_sha256=manifest['source_sha256'],template_source='frame_001582.jpg',template_polygons=polygons,
       additional_template=dict(source='frame_000664.jpg',quad=blue_quad.tolist()),
       usage='Camera correspondence exclusion only. Appearance retains printed labels.',
       limitations='Planar SIFT template candidates require review. Small/blurred/partial labels may be missed.',views=reports),indent=2)+'\n')
    print('Frames with sign detections',len(reviews),flush=True)


if __name__=='__main__':main()
