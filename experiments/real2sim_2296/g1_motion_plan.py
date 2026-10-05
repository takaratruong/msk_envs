#!/usr/bin/env python3
"""Deterministic clip selection, placement and contact-aware editing for G1.

This produces a kinematic reference, not a physics-controller rollout. Original
clips remain unchanged. The door angle and all corrections are explicit data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_files(path):
    path=Path(path).resolve();root=ET.parse(path).getroot()
    if root.findall('.//include'):raise ValueError('Explicit include dependency support required')
    compiler=root.find('compiler');directory=compiler.get('meshdir','') if compiler is not None else ''
    files=[path]+[(path.parent/directory/node.attrib['file']).resolve() for node in root.findall('.//asset/mesh') if 'file' in node.attrib]
    return [dict(path=str(f),sha256=sha(f)) for f in sorted(set(files))]
def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))
def rz(a): return Rotation.from_euler('z', a).as_matrix()
def wxyz(r): return np.roll(Rotation.from_matrix(r).as_quat(), 1)
def rquat(q): return Rotation.from_quat(np.roll(q, -1)).as_matrix()
def blend_rot(a, b, u):
    return Slerp([0.,1.], Rotation.from_matrix(np.stack([a,b])))(float(np.clip(u,0,1))).as_matrix()


def sample_q(q, index):
    index=np.asarray(index); lo=np.floor(index).astype(int); hi=np.minimum(lo+1,len(q)-1)
    a=(index-lo)[:,None]; result=q[lo]*(1-a)+q[hi]*a
    qa=q[lo,3:7].copy(); qb=q[hi,3:7].copy()
    qb[np.sum(qa*qb,axis=1)<0]*=-1
    qq=qa*(1-a)+qb*a; result[:,3:7]=qq/np.linalg.norm(qq,axis=1)[:,None]
    return result


class Robot:
    def __init__(self, path):
        self.path=Path(path); self.m=mujoco.MjModel.from_xml_path(str(path)); self.d=mujoco.MjData(self.m)
        self.addr={self.m.joint(j).name:int(self.m.jnt_qposadr[j]) for j in range(1,self.m.njnt)}
        self.dof={self.m.joint(j).name:int(self.m.jnt_dofadr[j]) for j in range(1,self.m.njnt)}
        self.lo=self.m.jnt_range[1:,0].copy(); self.hi=self.m.jnt_range[1:,1].copy()
        self.foot_ids=[self.m.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.wrist=self.m.body('right_wrist_yaw_link').id
        self.leg_names=[s+'_'+j+'_joint' for s in ('left','right') for j in
                        ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        self.arm_names=[s+'_joint' for s in
                        ('waist_yaw','right_shoulder_pitch',
                         'right_shoulder_roll','right_shoulder_yaw','right_elbow',
                         'right_wrist_roll','right_wrist_pitch','right_wrist_yaw')]
        self.sole_z=-.035409145

    def forward(self,q):
        self.d.qpos[:]=q; mujoco.mj_forward(self.m,self.d)

    def feet(self,q):
        self.forward(q)
        return self.d.xpos[self.foot_ids].copy(),self.d.xmat[self.foot_ids].reshape(2,3,3).copy()

    def wrist_pose(self,q):
        self.forward(q)
        return self.d.xpos[self.wrist].copy(),self.d.xmat[self.wrist].reshape(3,3).copy()

    def ik(self,q, targets, names, prior=None, iterations=70, prior_weight=.035):
        q=q.copy(); qi=np.array([self.addr[n] for n in names]); vi=np.array([self.dof[n] for n in names])
        jids=np.array([self.m.joint(n).id for n in names]); lower=self.m.jnt_range[jids,0]+1e-6; upper=self.m.jnt_range[jids,1]-1e-6
        for j,name in enumerate(names):
            if name.endswith('knee_joint'):
                lower[j]=max(lower[j],.045)
                q[qi[j]]=max(q[qi[j]],.25)
        if prior is None:prior=q.copy()
        if len(targets)==1 and targets[0][0]==self.wrist:
            # Bounded least squares avoids the joint-limit branch jumps that a
            # clipped pseudoinverse can produce. Keep the torso near upright
            # and prevent the upper arm from folding into the torso housing.
            for j,name in enumerate(names):
                if name=='waist_yaw_joint':lower[j],upper[j]=-.5,.5
                if name=='right_shoulder_roll_joint':upper[j]=min(upper[j],-.02)
                if name=='right_wrist_yaw_joint':lower[j]=max(lower[j],-1.45)
                if name=='right_wrist_pitch_joint':upper[j]=min(upper[j],1.42)
            body,point,rotation=targets[0]
            # Continue the previous solution through phase boundaries. A much
            # smaller source-pose preference preserves the reach seed without
            # an elbow/null-space jump when the pulling phase takes over.
            previous=q[qi].copy();preference=prior[qi].copy()
            def residual(x):
                trial=q.copy();trial[qi]=x;self.forward(trial)
                rr=self.d.xmat[body].reshape(3,3)
                return np.r_[20*(self.d.xpos[body]-point),
                             3*Rotation.from_matrix(rotation@rr.T).as_rotvec(),
                             .04*(x-previous),.002*(x-preference)]
            solution=least_squares(residual,np.clip(q[qi],lower+1e-7,upper-1e-7),
                                   bounds=(lower,upper),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q[qi]=solution.x;self.forward(q);return q
        last=float('inf')
        for _ in range(iterations):
            self.forward(q); errs=[]; jacs=[]
            for body,point,rotation in targets:
                jp=np.zeros((3,self.m.nv));jr=np.zeros((3,self.m.nv))
                mujoco.mj_jacBody(self.m,self.d,jp,jr,body)
                cur=self.d.xmat[body].reshape(3,3)
                errs.extend([point-self.d.xpos[body], .35*Rotation.from_matrix(rotation@cur.T).as_rotvec()])
                jacs.extend([jp[:,vi],.35*jr[:,vi]])
            e=np.concatenate(errs);j=np.concatenate(jacs,axis=0)
            if np.linalg.norm(e)<2e-5:break
            inv=np.linalg.solve(j@j.T+np.eye(len(e))*2e-5,np.eye(len(e)))
            pinv=j.T@inv
            delta=pinv@e + prior_weight*(np.eye(len(qi))-pinv@j)@(prior[qi]-q[qi])
            if np.max(np.abs(delta))>.14:delta*=.14/np.max(np.abs(delta))
            q[qi]=np.clip(q[qi]+delta,lower,upper)
            if abs(last-np.linalg.norm(e))<1e-10:break
            last=np.linalg.norm(e)
        self.forward(q)
        return q

    def map_clip(self,q, source_model):
        if source_model is None:
            if q.shape[1]!=self.m.nq:raise ValueError('Clip has no compatible joint mapping')
            return q.copy()
        source=mujoco.MjModel.from_xml_path(str(source_model))
        if q.shape[1]!=source.nq:raise ValueError('Source model differs from clip width')
        result=np.tile(self.m.qpos0,(len(q),1));result[:,:7]=q[:,:7]
        for j in range(1,source.njnt):
            name=source.joint(j).name
            if name not in self.addr:raise ValueError('Missing named target joint '+name)
            result[:,self.addr[name]]=q[:,source.jnt_qposadr[j]]
        return result


def load_clip(path):
    z=np.load(path,allow_pickle=False)
    q=np.asarray(z['qpos'],dtype=float);fps=float(np.ravel(z['fps'])[0])
    if not np.isfinite(q).all():raise ValueError('Nonfinite source clip')
    return q,fps


def place_clip(q, endpoint, desired_heading):
    q=q.copy(); direction=q[-1,:2]-q[0,:2]
    angle=desired_heading-np.arctan2(direction[1],direction[0]);rot=rz(angle)
    offset=np.r_[endpoint,0.] - rot@np.r_[q[-1,:2],0.]
    q[:,:3]=q[:,:3]@rot.T+offset
    for row in q:row[3:7]=wxyz(rot@rquat(row[3:7]))
    return q


def lock_walk_feet(robot,q,fps):
    feet=[];rots=[]
    for row in q:
        p,r=robot.feet(row);feet.append(p);rots.append(r)
    feet=np.asarray(feet);rots=np.asarray(rots)
    contacts=np.zeros((len(q),2),bool);targets=np.empty_like(feet);r_targets=np.empty_like(rots)
    sole_height=-robot.sole_z
    for side in range(2):
        speed=np.linalg.norm(np.gradient(feet[:,side,:2],1/fps,axis=0),axis=1)
        height=feet[:,side,2]-sole_height
        raw=(height<.025)&(speed<.38)
        raw=binary_closing(raw,structure=np.ones(3));raw=binary_opening(raw,structure=np.ones(3))
        raw[:3]=True;raw[-3:]=True
        raw=binary_closing(raw,structure=np.ones(5),border_value=1)
        runs=[];start=None
        for i,active in enumerate(np.r_[raw,False]):
            if active and start is None:start=i
            if not active and start is not None:
                runs.append((start,i-1));start=None
        for start,end in runs:
            anchor=np.median(feet[start:end+1,side],axis=0);anchor[2]=sole_height
            yaw=np.median(np.unwrap(Rotation.from_matrix(rots[start:end+1,side]).as_euler('xyz')[:,2]))
            targets[start:end+1,side]=anchor;r_targets[start:end+1,side]=rz(yaw);contacts[start:end+1,side]=True
        for (a,b),(c,d) in zip(runs[:-1],runs[1:]):
            for i in range(b+1,c):
                u=(i-b)/(c-b);s=smooth(u)
                targets[i,side]=(1-s)*targets[b,side]+s*targets[c,side]
                # Preserve the source's swing clearance profile, with a visible
                # minimum arc for the rescheduled endpoint contacts.
                targets[i,side,2]=sole_height+.065*np.sin(np.pi*u)**2
                r_targets[i,side]=blend_rot(r_targets[b,side],r_targets[c,side],s)
    edited=[]
    for i,row in enumerate(q):
        # Flattening and locking soles can overextend a straight source leg.
        # Lowering the pelvis leaves knee margin and avoids an IK branch flip.
        row=row.copy();row[2]-=.045
        ts=[(robot.foot_ids[s],targets[i,s],r_targets[i,s]) for s in range(2)]
        fixed=robot.ik(row,ts,robot.leg_names,prior=row,iterations=55,prior_weight=0.)
        edited.append(fixed)
    return np.asarray(edited),contacts,targets,r_targets


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',type=Path,required=True);p.add_argument('--walk',type=Path,action='append',required=True)
    p.add_argument('--reach',type=Path,required=True);p.add_argument('--reach-model',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=30);p.add_argument('--door-degrees',type=float,default=65.)
    p.add_argument('--stance-x',type=float,default=-1.94);p.add_argument('--stance-y',type=float,default=-.38)
    p.add_argument('--approach-yaw',type=float,default=2.25)
    p.add_argument('--grip-x',type=float,default=.111948);p.add_argument('--grip-y',type=float,default=.074234)
    p.add_argument('--release-and-retract',action='store_true',help='Optional extension beyond the requested open-and-hold task')
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    task=json.loads(args.task.read_text());robot=Robot(args.robot);fps=args.fps
    candidates=[]
    for path in args.walk:
        q,f=load_clip(path)
        if q.shape[1]!=robot.m.nq:raise ValueError('Walking clip must already have exact target model order')
        distance=float(np.linalg.norm(q[-1,:2]-q[0,:2]))
        _,r0=robot.feet(q[0]);_,r1=robot.feet(q[-1])
        # Offline motion matching: requested travel plus endpoint stance/pose.
        score=abs(distance-2.)+.02*np.linalg.norm(q[-1,7:]-q[0,7:])
        candidates.append(dict(path=str(path.resolve()),sha256=sha(path),distance_m=distance,score=float(score),fps=f))
    selected=min(candidates,key=lambda x:x['score']);walk,wfps=load_clip(selected['path'])
    duration=(len(walk)-1)/wfps
    nt=int(round((duration+.7)*fps))+1
    # Smoothly enter and leave the source gait; interior time follows the clip.
    t=np.linspace(0,1,nt);source_t=t.copy();edge=.12
    # A monotone cubic easing at each edge keeps finite, zero endpoint velocity.
    left=t<edge;right=t>1-edge
    source_t[left]=edge*(2*(t[left]/edge)**2-(t[left]/edge)**3)
    u=(1-t[right])/edge;source_t[right]=1-edge*(2*u*u-u*u*u)
    walk=sample_q(walk,source_t*(len(walk)-1))
    for side,sign in [('right',1.),('left',-1.)]:
        for finger in ('index','middle'):
            walk[:,robot.addr[f'{side}_hand_{finger}_0_joint']]=sign*.42
            walk[:,robot.addr[f'{side}_hand_{finger}_1_joint']]=sign*.58
        walk[:,robot.addr[f'{side}_hand_thumb_2_joint']]=-sign*.35
    goal=np.array([args.stance_x,args.stance_y])
    walk=place_clip(walk,goal,args.approach_yaw)
    walk,wc,wp,wr=lock_walk_feet(robot,walk,fps)
    reach,rfps=load_clip(args.reach);reach=robot.map_clip(reach,args.reach_model)
    frame=[];angles=[];labels=[];contact=[];foot_targets=[];wrist_targets=[];wrist_rotations=[];grasp=[]
    def append(q,angle,label,feet_c,feet_p,wrist_p=None,wrist_r=None,grasp_active=False):
        frame.append(q.copy());angles.append(angle);labels.append(label);contact.append(feet_c)
        foot_targets.append(np.asarray(feet_p));wrist_targets.append(np.full(3,np.nan) if wrist_p is None else wrist_p)
        wrist_rotations.append(np.full((3,3),np.nan) if wrist_r is None else wrist_r);grasp.append(grasp_active)
    for _ in range(round(.7*fps)):append(walk[0],0.,'ready',[True,True],wp[0])
    for i,q in enumerate(walk):append(q,0.,'walk',wc[i],wp[i])
    qlast=walk[-1].copy();initial_p,initial_r=robot.feet(qlast)
    yaw=task['facing_yaw_radians'];facing=rz(yaw)
    # Comfortable two-foot stance; use this model's actual FK to establish the
    # sole offsets, including its selected hand/waist morphology.
    stand=robot.m.qpos0.copy();stand[:2]=goal;stand[3:7]=wxyz(facing)
    for name,addr in robot.addr.items():
        if '_hand_' in name:stand[addr]=walk[-1,addr]
    for side in ('left','right'):
        stand[robot.addr[side+'_hip_pitch_joint']]=-.16
        stand[robot.addr[side+'_knee_joint']]=.32
        stand[robot.addr[side+'_ankle_pitch_joint']]=-.16
    neutral_p,neutral_r=robot.feet(stand)
    stand[2]+=(-robot.sole_z)-np.mean(neutral_p[:,2]);neutral_p,neutral_r=robot.feet(stand)
    final_feet=neutral_p.copy();final_feet[:,2]=-robot.sole_z
    final_r=np.stack([facing,facing])
    # Two explicitly scheduled placement steps turn into the interaction stance.
    settle_duration=3.0
    for t in np.arange(1,round(settle_duration*fps)+1)/fps:
        u=smooth(t/settle_duration);q=qlast*(1-u)+stand*u
        q[2]-=.015*np.sin(np.pi*t/settle_duration)**2
        q[3:7]=wxyz(blend_rot(rquat(qlast[3:7]),facing,u))
        fp=initial_p.copy();fr=initial_r.copy();fc=[True,True]
        for s,begin,end in [(1,.35,1.15),(0,1.65,2.45)]:
            v=np.clip((t-begin)/(end-begin),0,1);vsm=smooth(v)
            fp[s]=(1-vsm)*initial_p[s]+vsm*final_feet[s]
            fp[s,2]+= .065*np.sin(np.pi*v)**2
            fr[s]=blend_rot(initial_r[s],facing,vsm)
            fc[s]=not (begin<t<end)
        # Lateral weight shift during each single-support placement step.
        shift=np.zeros(2)
        if t<1.4:shift=(fp[0,:2]-q[:2])*.68*smooth(t/.3)*(1-smooth((t-1.15)/.25))
        elif t<2.7:shift=(fp[1,:2]-q[:2])*.68*smooth((t-1.4)/.25)*(1-smooth((t-2.45)/.25))
        q[:2]+=shift
        q=robot.ik(q,[(robot.foot_ids[s],fp[s],fr[s]) for s in range(2)],robot.leg_names,prior=q,prior_weight=0.)
        append(q,0.,'turn_and_settle',fc,fp)
    stand=frame[-1].copy();feet_p,feet_r=robot.feet(stand)
    for _ in range(round(.4*fps)):append(stand,0.,'stand',[True,True],feet_p)
    door=np.asarray(task['door']['source_world_transform']);pivot=np.asarray(task['door']['world_pivot'])
    rail=np.asarray(task['grasp']['rail_point_world'])
    # Right hand's local +Z is parallel to the rail; fingers curl toward local
    # +Y. The grip-frame origin is deliberately offset from the wrist/palm mesh.
    grip_local=np.array([args.grip_x,args.grip_y,0.])
    grasp_rotation=door[:3,:3]
    def target(angle):
        rotation=rz(angle);rail_at=pivot+rotation@(rail-pivot)
        wrist_r=rotation@grasp_rotation
        return rail_at-wrist_r@grip_local,wrist_r
    closed_pos,closed_rot=target(0.)
    initial_wrist,initial_wrist_rot=robot.wrist_pose(stand)
    # Retain the source reach's wrist-path residual and its elbow/waist pose as
    # a posture preference; exact task targets and foot locks are solved here.
    rcount=round(3.*fps)
    rq=sample_q(reach,np.linspace(0,len(reach)-1,rcount))
    rp=np.array([robot.wrist_pose(q)[0] for q in rq])
    source_forward=rquat(rq[0,3:7]);map_r=facing@source_forward.T
    rp=(rp-rp[0])@map_r.T
    previous=stand.copy()
    outward=np.asarray(task['fridge_outward_world'])
    pregrasp=closed_pos+.14*outward
    for i,source in enumerate(rq):
        u=(i+1)/len(rq);s=smooth(u)
        residual=rp[i]-smooth(i/max(1,len(rq)-1))*rp[-1]
        if u<.7:
            v=u/.7;a=smooth(v)
            pos=initial_wrist*(1-a)+pregrasp*a+.15*residual*np.sin(np.pi*v)
            rot=blend_rot(initial_wrist_rot,closed_rot,a)
        else:
            a=smooth((u-.7)/.3);pos=pregrasp*(1-a)+closed_pos*a;rot=closed_rot
        prior=stand.copy()
        for name in robot.arm_names:prior[robot.addr[name]]=(1-s)*stand[robot.addr[name]]+s*source[robot.addr[name]]
        q=robot.ik(previous,[(robot.wrist,pos,rot)],robot.arm_names,prior=prior,iterations=100)
        hand_open=smooth((u-.62)/.16)
        for name,addr in robot.addr.items():
            if name.startswith('right_hand_'):q[addr]=(1-hand_open)*stand[addr]
        append(q,0.,'reach',[True,True],feet_p,pos,rot);previous=q
    hand_close={'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':-.091890,'right_hand_thumb_2_joint':-.847978,
                'right_hand_index_0_joint':1.055898,'right_hand_index_1_joint':1.440395,
                'right_hand_middle_0_joint':1.055898,'right_hand_middle_1_joint':1.440395}
    for i in range(round(.8*fps)):
        q=previous.copy();s=smooth((i+1)/round(.8*fps))
        for n,v in hand_close.items():q[robot.addr[n]]=s*v
        append(q,0.,'close_hand',[True,True],feet_p,closed_pos,closed_rot)
    previous=frame[-1].copy();endangle=np.deg2rad(args.door_degrees)
    for i in range(round(4.*fps)):
        angle=endangle*smooth((i+1)/round(4.*fps));pos,rot=target(angle)
        q=robot.ik(previous,[(robot.wrist,pos,rot)],robot.arm_names,prior=previous,iterations=100)
        append(q,angle,'pull',[True,True],feet_p,pos,rot,True);previous=q
    for _ in range(round(1.*fps)):
        pos,rot=target(endangle);append(previous,endangle,'hold_open',[True,True],feet_p,pos,rot,True)
    for i in range(round(.7*fps) if args.release_and_retract else 0):
        q=previous.copy();s=1-smooth((i+1)/round(.7*fps))
        for n,v in hand_close.items():q[robot.addr[n]]=s*v
        pos,rot=target(endangle);append(q,endangle,'release',[True,True],feet_p,pos,rot)
    q0=frame[-1].copy();p0,r0=robot.wrist_pose(q0);previous=q0.copy()
    withdraw=p0+.15*(rz(endangle)@outward)
    clear_side=np.array([goal[0]+.16,goal[1]+.12,1.08])
    for i in range(round(2.7*fps) if args.release_and_retract else 0):
        u=(i+1)/round(2.7*fps)
        if u<1/3:
            s=smooth(u*3);pos=p0*(1-s)+withdraw*s;rot=r0
        elif u<2/3:
            s=smooth((u-1/3)*3);pos=withdraw*(1-s)+clear_side*s;rot=blend_rot(r0,initial_wrist_rot,.5*s)
        else:
            s=smooth((u-2/3)*3);pos=clear_side*(1-s)+initial_wrist*s;rot=blend_rot(r0,initial_wrist_rot,.5+.5*s)
        q=robot.ik(previous,[(robot.wrist,pos,rot)],robot.arm_names,prior=stand,iterations=100)
        append(q,endangle,'retract',[True,True],feet_p,pos,rot);previous=q
    for _ in range(round(.8*fps)):
        pos,rot=target(endangle)
        append(previous,endangle,'finish',[True,True],feet_p,
               None if args.release_and_retract else pos,None if args.release_and_retract else rot,
               not args.release_and_retract)
    frames=np.asarray(frame);times=np.arange(len(frames))/fps
    trajectory=args.output/'trajectory.npz'
    np.savez_compressed(trajectory,qpos=frames,time=times,fps=fps,door_angle=np.asarray(angles),
                        phase=np.asarray(labels),foot_contact=np.asarray(contact),
                        foot_target=np.asarray(foot_targets),wrist_target=np.asarray(wrist_targets),
                        wrist_rotation_target=np.asarray(wrist_rotations),grasp_active=np.asarray(grasp))
    task['robot']=dict(model=str(args.robot.resolve()),sha256=sha(args.robot),nq=robot.m.nq,
                       asset_files=model_files(args.robot),
                       joint_names=[robot.m.joint(j).name for j in range(1,robot.m.njnt)],
                       hand_configuration='Unitree Dex3-1, seven finger joints per hand; user-selected Dex3 configuration')
    task['grasp'].update(grip_point_wrist=grip_local.tolist(),wrist_closed_rotation=grasp_rotation.tolist(),
                         finger_joint_targets=hand_close,
                         frame_status='authored articulated-hand reference; grasp force not tested')
    task['plan']=dict(fps=fps,stance_xy=goal.tolist(),door_degrees=args.door_degrees,
                      approach_yaw=args.approach_yaw,matching_candidates=candidates,selected_walk=selected,
                      reach=dict(path=str(args.reach.resolve()),sha256=sha(args.reach)),
                      arm_edit=dict(waist_roll_pitch_fixed_radians=0.,waist_yaw_bounds=[-.5,.5],
                                    right_shoulder_roll_max=-.02,right_wrist_yaw_min=-1.45,right_wrist_pitch_max=1.42,
                                    continuity_weight=.04,source_pose_weight=.002),
                      generator_used=False,description='Select and place existing clips; feet and wrist corrected by deterministic IK; turn/pull authored against scene geometry.')
    (args.output/'task.json').write_text(json.dumps(task,indent=2)+'\n')
    phases=[]
    for label in dict.fromkeys(labels):
        ii=np.flatnonzero(np.asarray(labels)==label)
        phases.append(dict(name=label,start_frame=int(ii[0]),end_frame=int(ii[-1]),start_seconds=float(times[ii[0]]),end_seconds=float(times[ii[-1]])))
    (args.output/'timeline.json').write_text(json.dumps(phases,indent=2)+'\n')
    (args.output/'build.json').write_text(json.dumps(dict(schema='g1-library-motion-build/v1',
        trajectory_sha256=sha(trajectory),script_sha256=sha(__file__),task_sha256=sha(args.output/'task.json'),
        frames=len(frames),duration_seconds=float(times[-1]),physical_execution_verified=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),frames=len(frames),duration_seconds=float(times[-1])),indent=2))


if __name__=='__main__':main()
