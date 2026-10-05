#!/usr/bin/env python3
"""Bind same-camera source/render comparisons; no fitted image warp or relighting."""
import argparse,hashlib,html,json
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--scene',type=Path,required=True)
    a.add_argument('--dataset',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--render',action='append',required=True,help='source_frame.jpg=/absolute/render.png')
    args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.dataset.read_text());views={v['name']:v for v in data['train']+data['val']}
    records=[];rows=[]
    for entry in args.render:
        frame,render_path=entry.split('=',1);render_path=Path(render_path);v=views[frame]
        source=Path(v['image_path']);render=cv2.imread(str(render_path));original=cv2.imread(str(source))
        if original is None or render is None:raise ValueError('Missing source or render '+entry)
        h,w=render.shape[:2];reference=cv2.resize(original,(w,h),interpolation=cv2.INTER_AREA)
        # Keep native source and result separate; the overlay is only diagnostic.
        stem=Path(frame).stem;cv2.imwrite(str(args.output/(stem+'_source.png')),reference)
        cv2.imwrite(str(args.output/(stem+'_render.png')),render)
        overlay=cv2.addWeighted(reference,.5,render,.5,0)
        cv2.imwrite(str(args.output/(stem+'_overlay.png')),overlay)
        panel=np.concatenate([reference,render],axis=1);bar=np.full((34,w*2,3),24,np.uint8)
        for x,label in [(12,'UNDISTORTED VIDEO FRAME '+frame),(w+12,'TEXTURED MESH / SAME CAMERA')]:
            cv2.putText(bar,label,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
        panel=np.concatenate([bar,panel]);panel_path=args.output/(stem+'_comparison.jpg');cv2.imwrite(str(panel_path),panel)
        metrics=dict(rgb_mean_absolute_error_255=float(np.abs(reference.astype(float)-render).mean()),
                     source_mean_rgb=reference.mean((0,1))[::-1].tolist(),render_mean_rgb=render.mean((0,1))[::-1].tolist())
        receipt=render_path.with_suffix(render_path.suffix+'.receipt.json')
        record=dict(frame=frame,time_seconds=v['time_seconds'],source=str(source),source_sha256=sha(source),
          render=str(render_path),render_sha256=sha(render_path),render_receipt=str(receipt),
          render_receipt_sha256=sha(receipt) if receipt.exists() else None,metrics=metrics,
          panel=panel_path.name,panel_sha256=sha(panel_path),resize_only=True,alignment_optimization=False)
        records.append(record)
        rows.append(f'<article><h2>{html.escape(frame)} · {v["time_seconds"]:.2f} s</h2><img src="{panel_path.name}"><details><summary>50% overlay</summary><img src="{stem}_overlay.png"></details></article>')
    report=dict(schema='real2sim-physical-comparison/v1',produced_at=datetime.now(timezone.utc).isoformat(),
       scene=str(args.scene),scene_sha256=sha(args.scene),dataset_sha256=sha(args.dataset),script_sha256=sha(__file__),
       method='Resize the calibrated undistorted source to render resolution; no post-hoc registration, exposure matching or image synthesis.',
       limitations=['RGB error includes lighting, view-dependent materials and changed display content; it is not a geometry accuracy measurement.',
                    'Camera estimates and nominal scene scale share source evidence; no independent metric ground truth.'],views=records)
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>IMG_2296 mesh comparison</title><style>body{background:#171a1f;color:#eceff3;font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 20px}img{width:100%;height:auto}article{margin:32px 0}summary{cursor:pointer}</style><h1>Video and articulated mesh</h1><p>Matching reconstructed cameras. Source images are undistorted and resized; no image warp or exposure correction. Geometry and materials remain modeled estimates.</p>'+''.join(rows))
    print(json.dumps(dict(scene_sha256=report['scene_sha256'],views=len(records),output=str(args.output)),indent=2))

if __name__=='__main__':main()
