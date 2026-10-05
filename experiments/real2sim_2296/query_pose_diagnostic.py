#!/usr/bin/env python3
"""Diagnose query-pose error with a frozen appearance; never update map or dataset."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from gsplat import rasterization


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def exp_pose(delta):
    rx,ry,rz,tx,ty,tz=delta;z=rx*0
    return torch.matrix_exp(torch.stack([z,-rz,ry,tx,rz,z,-rx,ty,-ry,rx,z,tz,z,z,z,z]).reshape(4,4))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--appearance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=250)
    a=p.parse_args();root=a.run.resolve();app=a.appearance.resolve();out=a.output.resolve()
    if out.exists():raise RuntimeError('Use a fresh diagnostic output.')
    out.mkdir(parents=True)
    meta=json.loads((app/'metadata.json').read_text());latest=json.loads((app/'latest_checkpoint.json').read_text())
    checkpoint=Path(latest['path'])
    if sha(checkpoint)!=latest['sha256']:raise RuntimeError('Checkpoint hash mismatch.')
    model=torch.load(checkpoint,map_location='cpu',weights_only=True)
    # Use the exact archived implementation of the completed baseline.
    trainer=Path(meta['operational_config']['dataset']).resolve().parent/'jobs/appearance_fp32'
    job=json.loads((trainer/'current.json').read_text());source=Path(job['attempt'])/'appearance.py'
    if sha(source)!=model['implementation_sha256']:raise RuntimeError('Archived trainer differs from checkpoint.')
    spec=importlib.util.spec_from_file_location('trained_appearance',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    _,views,manifest=m.load_dataset(root/'dataset.json',1920)
    if manifest['inputs_sha256']!=model['inputs_sha256']:raise RuntimeError('Frozen inputs changed.')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    splats={k:v.cuda().detach() for k,v in model['splats'].items()};extent=model['initialization']['scene_extent']
    rows=[]
    for name in ['frame_000718.jpg','frame_001014.jpg','frame_001451.jpg']:
        v=next(x for x in views['val'] if x['name']==name)
        ref=torch.from_numpy(np.array(Image.open(v['image_path']).convert('RGB'))).cuda().float()[None]/255
        mask=torch.from_numpy(np.array(Image.open(v['mask_path']).convert('L'))==255).cuda()[None,None]
        base=torch.tensor(v['w2c'],dtype=torch.float32,device='cuda')
        twist=torch.nn.Parameter(torch.zeros(6,device='cuda'))
        scales=torch.tensor([.01,.01,.01,.01*extent,.01*extent,.01*extent],device='cuda')
        opt=torch.optim.Adam([twist],lr=.025)
        def render():
            pose=exp_pose(twist*scales)@base
            rgb,_,_=rasterization(means=splats['means'],quats=splats['quats'],scales=splats['scales'].exp(),
                opacities=splats['opacities'].sigmoid(),colors=torch.cat([splats['sh0'],splats['shN']],1),
                viewmats=pose[None],Ks=torch.tensor(v['K'],device='cuda')[None],width=v['width'],height=v['height'],
                sh_degree=3,near_plane=extent*1e-4,far_plane=extent*1e4,packed=True,render_mode='RGB')
            return rgb.clamp(0,1),pose
        def scores(rgb):
            x=rgb.permute(0,3,1,2);y=ref.permute(0,3,1,2)
            mse=float(((x-y).square()*mask).sum()/(3*mask.sum()))
            return dict(static_psnr_db=-10*math.log10(max(mse,1e-12)),static_ssim=float(m.ssim(x,y,mask)))
        with torch.no_grad():before,_=render();before_score=scores(before)
        for i in range(a.iterations):
            rgb,_=render();x=rgb.permute(0,3,1,2);y=ref.permute(0,3,1,2)
            l1=((x-y).abs()*mask).sum()/(3*mask.sum());ssim=m.ssim(x,y,mask)
            normalized=torch.cat([(twist*scales)[:3]/math.radians(3),(twist*scales)[3:]/(.05*extent)])
            loss=.8*l1+.2*(1-ssim)+.001*normalized.square().sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite pose loss.')
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():after,pose=render();after_score=scores(after)
        images=[ref[0],before[0],after[0]];w,h=v['width'],v['height']
        canvas=Image.new('RGB',(w*3,h+42),(18,22,30));draw=ImageDraw.Draw(canvas)
        labels=['SOURCE','FROZEN PNP CAMERA','QUERY RGB-FITTED CAMERA (DIAGNOSTIC)']
        for j,img in enumerate(images):
            canvas.paste(Image.fromarray(np.rint(img.cpu().numpy()*255).astype('uint8')),(j*w,42));draw.text((j*w+8,8),labels[j]+' | '+name,fill='white')
        file=out/(Path(name).stem+'.jpg');canvas.save(file,quality=94)
        row=dict(name=name,source_time_seconds=v['source_timestamp_s'],before=before_score,after=after_score,
            delta_camera_twist=(twist*scales).detach().cpu().tolist(),w2c=pose.detach().cpu().tolist(),
            comparison=str(file),comparison_sha256=sha(file))
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(schema='real2sim-query-pose-diagnostic/v1',inputs_sha256=manifest['inputs_sha256'],
        source_sha256=manifest['source_sha256'],dataset_sha256=sha(root/'dataset.json'),
        checkpoint_sha256=latest['sha256'],implementation_sha256=sha(__file__),iterations=a.iterations,
        query_rgb_used_for_pose_optimization=True,gaussians_updated=False,dataset_updated=False,
        acceptance_metrics=False,interpretation='Diagnostic fit of six query-pose parameters to the same query RGB. Improved scores are not untouched held-out scores or a real-pose accuracy measurement.',views=rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
