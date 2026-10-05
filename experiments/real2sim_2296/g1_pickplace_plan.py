#!/usr/bin/env python3
"""Editable G1/Dex3 pick/place reference; attachment here is kinematic only.

The original take06 opening poses are copied exactly.  New motions reuse the
same named-joint Robot/IK implementation and curated walking clip, with explicit
foot targets and object attachment state.  This is not a controller rollout.
"""
from __future__ import annotations

import argparse
import json
import time as walltime
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import g1_motion_plan as original
from g1_motion_plan import Robot, blend_rot, load_clip, lock_walk_feet, model_files, rquat, rz, sample_q, sha, smooth, wxyz


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def yaw_of(q):
    return float(Rotation.from_matrix(rquat(q[3:7])).as_euler('xyz')[2])


def checked_attachment(robot, q, base, grip, grasp_height, position_tolerance=.002, axis_tolerance=.02):
    """Reject a distant/tilted hand before constructing a reference attachment."""
    p,r=robot.wrist_pose(q); neck=np.asarray(base)+[0.,0.,grasp_height]
    position_error=float(np.linalg.norm(p+r@grip-neck))
    axis_error=float(np.arccos(np.clip(r[2,2],-1.,1.)))
    if position_error>position_tolerance or axis_error>axis_tolerance:
        raise ValueError(f'Refusing attachment: grip error {position_error:.6f} m, axis error {axis_error:.6f} rad')
    return (r.T@(base-p),r.T.copy()),position_error,axis_error


class Reference:
    def __init__(self, task, prefix, walk, fps, output):
        self.task, self.fps, self.output = task, fps, output
        self.robot = Robot(task['robot']['model'])
        self.walk, self.walk_fps = load_clip(walk)
        if self.walk.shape[1] != self.robot.m.nq:
            raise ValueError('Walk model order/width mismatch')
        declared = task['robot']['joint_names']
        actual = [self.robot.m.joint(j).name for j in range(1, self.robot.m.njnt)]
        if declared != actual or task['robot']['nq'] != self.robot.m.nq:
            raise ValueError('Task named joint order differs from compiled model')
        if task['robot']['sha256'] != sha(task['robot']['model']):
            raise ValueError('Robot XML identity mismatch')
        declared_assets = {str(Path(x['path']).resolve()): x['sha256'] for x in task['robot']['asset_files']}
        actual_assets = {x['path']: x['sha256'] for x in model_files(task['robot']['model'])}
        if declared_assets != actual_assets:
            raise ValueError('Robot mesh/XML dependency mismatch')
        if task['scene']['sha256'] != sha(task['scene']['path']):
            raise ValueError('Source scene identity mismatch')
        self.prefix = np.load(prefix, allow_pickle=False)
        if abs(float(self.prefix['fps'])-fps) > 1e-9:
            raise ValueError('Preserving prefix requires identical sample rate')
        self.prefix_count = len(self.prefix['qpos'])
        # Geometry is an input contract. The long-neck task prop and shelf
        # position were explicitly revised in task_scene06; never apply an
        # adjustment twice when consuming that already prepared task.
        if abs(task['object']['height']-.185)>1e-9 or abs(task['object']['grasp_height']-.145)>1e-9:
            raise ValueError('This reference requires the scene06 long-neck task prop')
        self.base0 = np.array(task['object']['initial_base_world'], float)
        self.object_pose = np.r_[self.base0, 1., 0., 0., 0.]
        self.attachment = None
        self.rows = []
        for i, q in enumerate(self.prefix['qpos']):
            self.append(q, str(self.prefix['phase'][i]), self.prefix['foot_contact'][i],
                        self.prefix['foot_target'][i], self.prefix['wrist_target'][i],
                        self.prefix['wrist_rotation_target'][i],
                        door_grasp=bool(self.prefix['grasp_active'][i]),
                        angle=float(self.prefix['door_angle'][i]))
        stand_indices = np.flatnonzero(self.prefix['phase'] == 'stand')
        self.neutral = self.prefix['qpos'][stand_indices[-1]].copy()
        self.finger_names = list(task['grasp']['finger_joint_targets'])
        self.finger_addresses = [self.robot.addr[n] for n in self.finger_names]
        self.closed = np.array([task['grasp']['finger_joint_targets'][n] for n in self.finger_names])
        self.opened = np.zeros(len(self.finger_names))
        self.grip = np.array(task['grasp']['grip_point_wrist'])
        self.angle = float(self.prefix['door_angle'][-1])
        self.started = walltime.time()

    @property
    def q(self):
        return self.rows[-1]['qpos'].copy()

    def progress(self, phase):
        data = dict(schema='g1-pickplace-progress/v1', status='building', phase=phase,
                    frames=len(self.rows), seconds=(len(self.rows)-1)/self.fps,
                    elapsed_seconds=walltime.time()-self.started)
        (self.output/'status.json').write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(data), flush=True)

    def append(self, q, phase, feet_contact, feet_target, wrist=None, wrist_rotation=None,
               door_grasp=False, object_grasp=False, angle=None, support=None):
        if object_grasp:
            if self.attachment is None:
                raise ValueError('Attachment must be explicitly initialized')
            p, r = self.robot.wrist_pose(q)
            translation, rotation = self.attachment
            self.object_pose = np.r_[p+r@translation, wxyz(r@rotation)]
        if support is None:
            support = 'kinematic_hand_attachment' if object_grasp else (
                'table_surface_reference' if self.attachment is not None else 'fridge_shelf_reference')
        contacts = []
        for side, active in zip(('left_foot_floor', 'right_foot_floor'), feet_contact):
            if active: contacts.append(side)
        if door_grasp: contacts.append('right_dex3_door_handle_commanded')
        if object_grasp: contacts.append('right_dex3_bottle_neck_commanded')
        if support != 'kinematic_hand_attachment': contacts.append('bottle_'+support)
        self.rows.append(dict(qpos=q.copy(), phase=phase,
                              foot_contact=np.asarray(feet_contact, bool).copy(),
                              foot_target=np.asarray(feet_target).copy(),
                              wrist_target=np.full(3, np.nan) if wrist is None else np.asarray(wrist).copy(),
                              wrist_rotation_target=np.full((3,3), np.nan) if wrist_rotation is None else np.asarray(wrist_rotation).copy(),
                              door_angle=self.angle if angle is None else angle,
                              grasp_active=door_grasp or object_grasp,
                              door_grasp_active=door_grasp, object_grasp_active=object_grasp,
                              object_pose=self.object_pose.copy(), object_support=support,
                              contact_labels=';'.join(contacts)))

    def stand(self, xy, yaw, template=None):
        q = self.neutral.copy() if template is None else template.copy()
        q[:2] = xy; q[3:7] = wxyz(rz(yaw)); q[20:22] = 0.
        for side in ('left', 'right'):
            for joint, value in [('hip_pitch',-.16), ('hip_roll',0), ('hip_yaw',0),
                                 ('knee',.32), ('ankle_pitch',-.16), ('ankle_roll',0)]:
                q[self.robot.addr[f'{side}_{joint}_joint']] = value
        feet, _ = self.robot.feet(q)
        q[2] += -self.robot.sole_z - float(np.mean(feet[:,2]))
        return q

    def carry_arm(self, q, previous):
        for name in self.robot.arm_names:
            q[self.robot.addr[name]] = previous[self.robot.addr[name]]
        q[self.finger_addresses] = self.closed
        if getattr(self,'left_carry_rest',False):
            for n in self.robot.addr:
                if n.startswith('left_') and not n.startswith('left_hand_') and any(x in n for x in ('shoulder','elbow','wrist')):q[self.robot.addr[n]]=0.
        r = rz(yaw_of(q))
        target = q[:3] + r@np.array([.26,-.22,.30])
        out = self.robot.ik(q, [(self.robot.wrist,target,r)], self.robot.arm_names, prior=q)
        return out, target, r

    def guided_arm_ik(self,q,target,rotation,guide):
        """Continue a native-clear guide's redundant arm branch explicitly."""
        qi=np.array([self.robot.addr[n] for n in self.robot.arm_names])
        ji=np.array([self.robot.m.joint(n).id for n in self.robot.arm_names])
        lower=self.robot.m.jnt_range[ji,0]+1e-6;upper=self.robot.m.jnt_range[ji,1]-1e-6
        for i,name in enumerate(self.robot.arm_names):
            if name=='waist_yaw_joint':lower[i],upper[i]=-.5,.5
            if name=='right_shoulder_roll_joint':upper[i]=min(upper[i],-.02)
            if name=='right_wrist_yaw_joint':lower[i]=max(lower[i],-1.45)
            if name=='right_wrist_pitch_joint':upper[i]=min(upper[i],1.42)
        previous=q[qi].copy();preference=guide[qi].copy()
        def residual(x):
            trial=q.copy();trial[qi]=x;p,r=self.robot.wrist_pose(trial)
            return np.r_[200*(p-target),30*Rotation.from_matrix(rotation@r.T).as_rotvec(),
                         .12*(x-preference),.005*(x-previous)]
        solution=least_squares(residual,np.clip(preference,lower+1e-7,upper-1e-7),bounds=(lower,upper),
                               max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q=q.copy();q[qi]=solution.x;return q

    def step_to(self, qgoal, phase, duration=2.8, carry=False, goal_feet=None, goal_rots=None,
                wrist_goal=None, wrist_rotation=None, pelvis_dip=.10, weight_shift=.68, upper_guide=None,
                swing_order=(1,0)):
        self.progress(phase)
        start = self.q; initial_p, initial_r = self.robot.feet(start)
        final_p, final_r = self.robot.feet(qgoal)
        if goal_feet is not None: final_p = np.array(goal_feet)
        if goal_rots is not None: final_r = np.array(goal_rots)
        final_p[:,2] = -self.robot.sole_z
        wrist_start, wrist_rstart = self.robot.wrist_pose(start)
        # One explicit swing at a time; the support shift follows the previous
        # verified turn pattern. These targets still require physical tracking.
        for i in range(1, round(duration*self.fps)+1):
            t=i/self.fps; u=smooth(t/duration)
            q=(1-u)*start+u*qgoal; q[3:7]=wxyz(blend_rot(rquat(start[3:7]),rquat(qgoal[3:7]),u))
            # Translating between planted feet needs more knee margin than the
            # original almost stationary turn. A smooth 10 cm pelvis dip avoids
            # asking a nearly straight stance leg to reach beyond its length.
            q[2]-=pelvis_dip*np.sin(np.pi*t/duration)**2
            targets=initial_p.copy(); rotations=initial_r.copy(); contact=np.ones(2,bool)
            for side,a,b in [(swing_order[0],.12*duration,.40*duration),(swing_order[1],.55*duration,.83*duration)]:
                v=np.clip((t-a)/(b-a),0.,1.); s=smooth(v)
                targets[side]=(1-s)*initial_p[side]+s*final_p[side]
                targets[side,2]+=.065*np.sin(np.pi*v)**2
                rotations[side]=blend_rot(initial_r[side],final_r[side],s)
                contact[side]=not (a<t<b)
            left_weight=smooth(t/(.12*duration))*(1-smooth((t-.40*duration)/(.15*duration)))
            right_weight=smooth((t-.40*duration)/(.15*duration))*(1-smooth((t-.83*duration)/(.17*duration)))
            center=q[:2].copy()
            q[:2]+=weight_shift*(left_weight*(targets[1-swing_order[0],:2]-center)+right_weight*(targets[1-swing_order[1],:2]-center))
            for name in self.robot.leg_names:
                q[self.robot.addr[name]]=self.q[self.robot.addr[name]]
            q=self.robot.ik(q,[(self.robot.foot_ids[s],targets[s],rotations[s]) for s in range(2)],
                            self.robot.leg_names, prior=q, iterations=70, prior_weight=0.)
            target=rotation=None
            if wrist_goal is not None:
                target=(1-u)*wrist_start+u*np.asarray(wrist_goal)
                rotation=blend_rot(wrist_rstart,np.asarray(wrist_rotation),u)
                for name in self.robot.arm_names:q[self.robot.addr[name]]=self.q[self.robot.addr[name]]
                if upper_guide is not None:
                    guide=sample_q(np.asarray(upper_guide),np.array([u*(len(upper_guide)-1)]))[0]
                    q=self.guided_arm_ik(q,target,rotation,guide)
                else:q=self.robot.ik(q,[(self.robot.wrist,target,rotation)],self.robot.arm_names,prior=q)
                q[self.finger_addresses]=self.closed if carry else self.opened
            elif carry: q,target,rotation=self.carry_arm(q,self.q)
            self.append(q,phase,contact,targets,target,rotation,object_grasp=carry)

    def arm_to(self, target, rotation, phase, duration, carry=False, fingers=None, guide_goal=None):
        self.progress(phase)
        start=self.q; p0,r0=self.robot.wrist_pose(start); feet,_=self.robot.feet(start)
        finger0=start[self.finger_addresses].copy()
        for i in range(1,round(duration*self.fps)+1):
            u=smooth(i/(duration*self.fps)); p=(1-u)*p0+u*target; r=blend_rot(r0,rotation,u)
            if guide_goal is None:q=self.robot.ik(self.q,[(self.robot.wrist,p,r)],self.robot.arm_names,prior=start)
            else:q=self.guided_arm_ik(self.q,p,r,(1-u)*start+u*np.asarray(guide_goal))
            if fingers is not None: q[self.finger_addresses]=(1-u)*finger0+u*fingers
            self.append(q,phase,[True,True],feet,p,r,object_grasp=carry)

    def joint_arm_to(self, goal, phase, duration, include_left=False, include_waist_pitch=False, door_grasp=False, carry=False):
        """Authored smooth joint-space retraction, with no claimed wrist path."""
        self.progress(phase)
        start=self.q; feet,_=self.robot.feet(start)
        addresses=[self.robot.addr[n] for n in self.robot.arm_names]
        if include_left:addresses += [self.robot.addr[n] for n in self.robot.addr if n.startswith('left_') and not n.startswith('left_hand_') and any(x in n for x in ('shoulder','elbow','wrist'))]
        if include_waist_pitch:addresses += [self.robot.addr['waist_pitch_joint'],self.robot.addr['waist_roll_joint']]
        for i in range(1,round(duration*self.fps)+1):
            u=smooth(i/(duration*self.fps)); q=start.copy()
            q[addresses]=(1-u)*start[addresses]+u*goal[addresses]
            self.append(q,phase,[True,True],feet,door_grasp=door_grasp,object_grasp=carry)

    def object_to(self, target_base, wrist_rotation, phase, duration):
        self.progress(phase)
        start=self.q; base0=self.object_pose[:3].copy();_,r0=self.robot.wrist_pose(start)
        feet,_=self.robot.feet(start)
        for i in range(1,round(duration*self.fps)+1):
            u=smooth(i/(duration*self.fps));base=(1-u)*base0+u*target_base
            r=blend_rot(r0,wrist_rotation,u);p=base-r@self.attachment[0]
            q=self.robot.ik(self.q,[(self.robot.wrist,p,r)],self.robot.arm_names,prior=start)
            self.append(q,phase,[True,True],feet,p,r,object_grasp=True)

    def finger_to(self, fingers, phase, duration, object_grasp=False):
        self.progress(phase)
        q0=self.q; feet,_=self.robot.feet(q0); p,r=self.robot.wrist_pose(q0)
        for i in range(1,round(duration*self.fps)+1):
            u=smooth(i/(duration*self.fps));q=q0.copy()
            q[self.finger_addresses]=(1-u)*q0[self.finger_addresses]+u*fingers
            self.append(q,phase,[True,True],feet,p,r,object_grasp=object_grasp)

    def hold(self, phase, duration, carry=False):
        self.progress(phase)
        q=self.q;feet,_=self.robot.feet(q);p,r=self.robot.wrist_pose(q)
        for _ in range(round(duration*self.fps)):
            self.append(q,phase,[True,True],feet,p,r,object_grasp=carry)

    def walk_to(self, endpoint, phase):
        startxy=self.q[:2]; endpoint=np.array(endpoint,float)
        distance=float(np.linalg.norm(endpoint-startxy)); heading=float(np.arctan2(*(endpoint-startxy)[::-1]))
        chunks=max(1,int(np.ceil(distance/2.05)))
        for chunk in range(chunks):
            a=startxy+(endpoint-startxy)*chunk/chunks; b=startxy+(endpoint-startxy)*(chunk+1)/chunks
            angle=wrap(heading-yaw_of(self.q))
            if abs(angle)>.12:
                initial_yaw=yaw_of(self.q);turns=int(np.ceil(abs(angle)/.65))
                for j in range(turns):
                    self.step_to(self.stand(a,initial_yaw+angle*(j+1)/turns,self.q),
                                 phase+'_align_heading_'+str(chunk)+'_'+str(j),2.4,True)
            n=round(((len(self.walk)-1)/self.walk_fps+.7)*self.fps)+1
            u=np.linspace(0,1,n); s=u.copy();edge=.12
            left=u<edge;right=u>1-edge
            s[left]=edge*(2*(u[left]/edge)**2-(u[left]/edge)**3)
            v=(1-u[right])/edge;s[right]=1-edge*(2*v*v-v*v*v)
            q=sample_q(self.walk,s*(len(self.walk)-1))
            direction=q[-1,:2]-q[0,:2]; oldheading=np.arctan2(direction[1],direction[0])
            rot=rz(heading-oldheading)
            oldstart=q[0,:2].copy(); local=(q[:,:2]-oldstart)@rot[:2,:2].T
            forward=np.array([np.cos(heading),np.sin(heading)]); lateral=np.array([-forward[1],forward[0]])
            along=local@forward; cross=local@lateral
            q[:,:2]=a+along[:,None]/along[-1]*(b-a)+cross[:,None]*lateral
            for row in q:
                row[3:7]=wxyz(rot@rquat(row[3:7])); row[self.finger_addresses]=self.closed
            # Additional leg margin for this new gait edit. Actual visible sole
            # planes, rather than ankle points alone, are checked at save time.
            q,contacts,targets,rotations=lock_walk_feet(self.robot,q,self.fps)
            # Transfer to the source's first planted stance, then play the clip
            # with rescheduled flat soles. Keep hand edits continuous across it.
            self.step_to(q[0],phase+'_enter_'+str(chunk),2.8,True,targets[0],rotations[0],pelvis_dip=.04)
            self.progress(phase+'_walk_'+str(chunk))
            for j in range(1,len(q)):
                row,p,r=self.carry_arm(q[j].copy(),self.q)
                self.append(row,phase+'_walk_'+str(chunk),contacts[j],targets[j],p,r,object_grasp=True)
            goal=self.stand(b,heading,self.q)
            self.step_to(goal,phase+'_settle_'+str(chunk),2.8,True,pelvis_dip=.04)

    def build(self):
        outward=np.array(self.task['fridge_outward_world']); yaw=self.task['facing_yaw_radians']
        seed=self.task['reference_planning']['pickup_seed']
        near=np.array(seed['near_qpos']);far=np.array(seed['far_qpos'])
        guide=np.array(seed['guide_qpos']);ready=far.copy();ready[:2]+=[.30,-.20]
        tucked=self.q
        for name,value in zip(['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'],[0,.8,0,.5,0,0,0]):
            tucked[self.robot.addr['left_'+name+'_joint']]=value
        self.joint_arm_to(tucked,'clear_left_hand_from_hip',1.2,include_left=True,door_grasp=True)
        for name,value in [('left_shoulder_pitch_joint',.8),('left_shoulder_roll_joint',.12),('left_elbow_joint',1.2)]:
            tucked[self.robot.addr[name]]=value
        self.joint_arm_to(tucked,'tuck_unused_left_arm',1.2,include_left=True,door_grasp=True)
        self.finger_to(self.opened,'release_door_handle',.8)
        p,r=self.robot.wrist_pose(self.q)
        self.arm_to(p-.12*r[:,0],r,'withdraw_from_open_door_edge',1.5)
        tucked=self.q;tucked[19:22]=0.
        for key,value in [('right_shoulder_pitch_joint',.6),('right_shoulder_roll_joint',-.15),('right_shoulder_yaw_joint',0),('right_elbow_joint',1.1),('right_wrist_roll_joint',0),('right_wrist_pitch_joint',0),('right_wrist_yaw_joint',0)]:
            tucked[self.robot.addr[key]]=value
        self.joint_arm_to(tucked,'retract_after_opening',2.,include_waist_pitch=True)
        midpoint=(self.q[:2]+ready[:2])/2
        for i,xy in enumerate([midpoint,ready[:2]]):
            goal=self.stand(xy,yaw_of(far),self.q)
            self.step_to(goal,'reposition_outside_fridge_'+str(i),3.)
        bottle_r=rz(2.18286)
        neck=self.base0+np.array([0,0,self.task['object']['grasp_height']])
        grasp_p=neck-bottle_r@self.grip
        raised=self.q;raised[self.robot.addr['right_shoulder_roll_joint']]=-.5;raised[self.robot.addr['right_elbow_joint']]=.3
        self.joint_arm_to(raised,'clear_right_arm_for_pregrasp',1.2)
        self.joint_arm_to(ready,'reach_fridge_pregrasp',2.4,include_waist_pitch=True)
        for i in range(2):
            u=(i+1)/2;goal=self.stand((1-u)*ready[:2]+u*far[:2],yaw_of(far),self.q);goal[21]=far[21]
            armguide=np.stack([ready.copy(),far.copy()]);p_far,r_far=self.robot.wrist_pose(far)
            self.step_to(goal,'approach_raised_hand_'+str(i),3.,wrist_goal=p_far+np.r_[(1-u)*(ready[:2]-far[:2]),0.],
                         wrist_rotation=r_far,pelvis_dip=.01,weight_shift=0.,upper_guide=armguide,swing_order=(0,1))
        for i in range(2):
            start=36-18*i;end=18-18*i
            goal=self.stand(guide[end,:2],yaw_of(near),self.q);goal[21]=near[21]
            target,rotation=self.robot.wrist_pose(guide[end])
            self.step_to(goal,'approach_open_fridge_'+str(i),3.,
                         wrist_goal=target,wrist_rotation=rotation,
                         pelvis_dip=.005,weight_shift=0.,upper_guide=guide[end:start+1][::-1],swing_order=(0,1))
        self.arm_to(grasp_p,bottle_r,'reach_bottle_neck',1.2,guide_goal=near)
        self.finger_to(self.closed,'close_dex3_on_bottle',.8)
        p,r=self.robot.wrist_pose(self.q)
        self.attachment,self.attachment_grip_error,self.attachment_axis_error=checked_attachment(
            self.robot,self.q,self.base0,self.grip,self.task['object']['grasp_height'])
        self.hold('establish_reference_bottle_grasp',.3,True)
        self.arm_to(p+np.array([0,0,.035]),r,'lift_bottle_clear_shelf',1.1,True,guide_goal=guide[0])
        extract_start,_=self.robot.wrist_pose(self.q)
        for i in range(2):
            start=18*i;end=18*(i+1)
            goal=self.stand(guide[end,:2],yaw_of(near),self.q);goal[21]=near[21]
            target,rotation=self.robot.wrist_pose(guide[end])
            self.step_to(goal,'extract_bottle_with_backstep_'+str(i),3.,True,
                         wrist_goal=target,wrist_rotation=rotation,
                         pelvis_dip=.005,weight_shift=0.,upper_guide=guide[start:end+1])
        for i in range(2):
            u=(i+1)/2;goal=self.stand((1-u)*far[:2]+u*ready[:2],yaw_of(far),self.q);goal[21]=far[21]
            p_far,r_far=self.robot.wrist_pose(far)
            self.step_to(goal,'extract_bottle_with_backstep_'+str(i+2),3.,True,
                         wrist_goal=p_far+np.r_[u*(ready[:2]-far[:2]),0.],wrist_rotation=r_far,
                         pelvis_dip=.01,weight_shift=0.,upper_guide=np.stack([far,ready]))
        # Bring the bottle into the torso-relative carrying envelope before
        # moving the feet. Bottle motion remains actual FK attachment data.
        r=rz(yaw_of(self.q));target=self.q[:3]+r@np.array([.26,-.22,.30])
        self.arm_to(target,r,'lower_to_carry_pose',1.5,True)
        left=self.q
        for name,value in zip(['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'],[0,.8,0,.5,0,0,0]):left[self.robot.addr['left_'+name+'_joint']]=value
        self.joint_arm_to(left,'clear_unused_arm_for_carry',1.2,include_left=True,carry=True)
        for name in self.robot.addr:
            if name.startswith('left_') and not name.startswith('left_hand_') and any(x in name for x in ('shoulder','elbow','wrist')):left[self.robot.addr[name]]=0.
        self.joint_arm_to(left,'rest_unused_arm_for_carry',1.2,include_left=True,carry=True)
        self.left_carry_rest=True
        self.walk_to([-1.7,-1.6],'carry_to_central_aisle')
        self.walk_to([2.7,-1.4],'carry_across_central_aisle')
        self.walk_to([3.20,-1.60],'approach_table_02')
        place_yaw=-.65
        self.step_to(self.stand([3.20,-1.60],place_yaw,self.q),'turn_to_table_02',2.8,True)
        target_base=np.array([3.38,-1.91,self.task['place']['support_height']+.0003])
        self.task['place'].update(target_base_world=target_base.tolist(),stance_xy=[3.20,-1.60],
                                  facing_yaw_radians=place_yaw,
                                  adjustment_reason='Original stance/target was unreachable. Target lies inside radius .43 table with ~.041 m radial bottle margin; closer stance reduces wrist/waist reach.')
        final_r=rz(place_yaw)
        final_p=target_base-final_r@self.attachment[0]
        self.arm_to(final_p+[0,0,.10],final_r,'position_bottle_over_table',1.5,True)
        self.arm_to(final_p,final_r,'lower_bottle_to_table',1.5,True)
        self.hold('support_bottle_before_release',.3,True)
        self.finger_to(self.opened,'release_bottle_on_table',.9,False)
        p,r=self.robot.wrist_pose(self.q)
        self.arm_to(p-.13*(rz(place_yaw)@np.array([1.,0.,0.])),r,'withdraw_hand_after_place',1.2,False)
        p,r=self.robot.wrist_pose(self.q);delta=np.array([3.02,-1.47])-self.q[:2]
        self.step_to(self.stand([3.02,-1.47],place_yaw,self.q),'retreat_from_table_02',2.8,False,
                     wrist_goal=p+np.r_[delta,0.],wrist_rotation=r,pelvis_dip=.04)
        neutral=self.stand(self.q[:2],place_yaw)
        self.joint_arm_to(neutral,'return_hand_to_rest',1.4)
        self.hold('released_bottle_reference_settle',2.5,False)

    def save(self, inputs):
        arrays={k:np.asarray([row[k] for row in self.rows]) for k in self.rows[0]}
        arrays['time']=np.arange(len(self.rows))/self.fps;arrays['fps']=np.array(self.fps)
        arrays['task_contact_labels']=arrays['contact_labels'].copy()
        arrays['object_base_pose']=arrays['object_pose'].copy()
        for row in arrays['object_pose']:
            row[:3]+=rquat(row[3:])@np.array([0.,0.,self.task['object']['height']/2])
        arrays['object_pose_convention']=np.array('world_xyz_then_quaternion_wxyz; geometric body center origin')
        arrays['object_pose_origin']=np.array('center')
        self.task['object_pose_origin']='center'
        arrays['kinematic_attachment_reference']=np.array(True)
        qpos=arrays['qpos'];qvel=np.zeros((len(qpos),self.robot.m.nv))
        for i in range(len(qpos)-1):
            mujoco.mj_differentiatePos(self.robot.m,qvel[i],1/self.fps,qpos[i],qpos[i+1])
        qvel[-1]=qvel[-2];arrays['qvel']=qvel
        phases=[]
        for i,label in enumerate(arrays['phase']):
            if i==0 or label!=arrays['phase'][i-1]:
                phases.append(dict(name=str(label),start_frame=i,start_time_seconds=i/self.fps))
            phases[-1].update(end_frame=i,end_time_seconds=i/self.fps)
        errors=[];feet_errors=[];orientation_errors=[];sole_min=[];sole_max=[];actual_foot_points=[]
        sole_vertices=[]
        for body in self.robot.foot_ids:
            vertices=[]
            for g in range(self.robot.m.ngeom):
                m=self.robot.m
                if m.geom_bodyid[g]!=body or m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[g])
                vertices.append(v@rotation.reshape(3,3).T+m.geom_pos[g])
            vertices=np.concatenate(vertices);sole_vertices.append(vertices[vertices[:,2]<vertices[:,2].min()+.0001])
        for i,q in enumerate(qpos):
            p,r=self.robot.wrist_pose(q);f,fr=self.robot.feet(q)
            errors.append(float(np.linalg.norm(p-arrays['wrist_target'][i])) if np.isfinite(arrays['wrist_target'][i]).all() else np.nan)
            orientation_errors.append(float(np.linalg.norm(Rotation.from_matrix(arrays['wrist_rotation_target'][i]@r.T).as_rotvec())) if np.isfinite(arrays['wrist_rotation_target'][i]).all() else np.nan)
            feet_errors.append(np.linalg.norm(f-arrays['foot_target'][i],axis=1))
            world=[v@fr[s].T+f[s] for s,v in enumerate(sole_vertices)]
            sole_min.append([float(v[:,2].min()) for v in world]);sole_max.append([float(v[:,2].max()) for v in world])
            actual_foot_points.append(world)
        arrays['wrist_position_error']=np.array(errors);arrays['wrist_orientation_error']=np.array(orientation_errors)
        arrays['foot_position_error']=np.array(feet_errors)
        arrays['sole_min_z']=np.array(sole_min);arrays['sole_max_z']=np.array(sole_max)
        drift=np.full((len(qpos),2),np.nan)
        for side in range(2):
            anchor=None
            for i,active in enumerate(arrays['foot_contact'][:,side]):
                if not active:anchor=None;continue
                if anchor is None:anchor=actual_foot_points[i][side].copy()
                drift[i,side]=np.max(np.linalg.norm(actual_foot_points[i][side]-anchor,axis=1))
        arrays['planted_sole_point_drift']=drift
        lower=self.robot.m.jnt_range[1:,0];upper=self.robot.m.jnt_range[1:,1]
        joint_violation=np.maximum(lower-qpos[:,7:],qpos[:,7:]-upper)
        phase_metrics={}
        for phase in phases:
            sl=slice(phase['start_frame'],phase['end_frame']+1)
            e=arrays['wrist_position_error'][sl];er=arrays['wrist_orientation_error'][sl]
            phase_metrics[phase['name']]=dict(frames=len(qpos[sl]),
                wrist_position_error_max_m=float(np.nanmax(e)) if np.isfinite(e).any() else None,
                wrist_orientation_error_max_rad=float(np.nanmax(er)) if np.isfinite(er).any() else None,
                foot_target_error_max_m=float(np.max(arrays['foot_position_error'][sl])),
                root_bounds=[qpos[sl,:3].min(0).tolist(),qpos[sl,:3].max(0).tolist()])
        extension=slice(self.prefix_count,None)
        final_error=float(np.linalg.norm(arrays['object_base_pose'][-1,:3]-self.task['place']['target_base_world']))
        checks=dict(prefix_qpos_exact=bool(np.array_equal(qpos[:self.prefix_count],self.prefix['qpos'])),
                    prefix_time_exact=bool(np.array_equal(arrays['time'][:self.prefix_count],self.prefix['time'])),
                    prefix_door_exact=bool(np.array_equal(arrays['door_angle'][:self.prefix_count],self.prefix['door_angle'])),
                    all_qpos_finite=bool(np.isfinite(qpos).all()),
                    all_object_poses_finite=bool(np.isfinite(arrays['object_pose']).all()),
                    joint_limits=bool(np.max(joint_violation)<1e-6),
                    root_quaternions_normalized=bool(np.max(abs(np.linalg.norm(qpos[:,3:7],axis=1)-1))<1e-6),
                    object_quaternions_normalized=bool(np.max(abs(np.linalg.norm(arrays['object_pose'][:,3:],axis=1)-1))<1e-6),
                    final_object_released=bool(not arrays['object_grasp_active'][-1]),
                    extension_wrist_error_under_2mm=bool(np.nanmax(arrays['wrist_position_error'][extension])<.002),
                    extension_foot_target_error_under_5mm=bool(np.max(arrays['foot_position_error'][extension])<.005),
                    extension_joint_step_under_025rad=bool(np.max(abs(np.diff(qpos[self.prefix_count-1:,7:],axis=0)))<.25),
                    bottle_grasp_alignment_under_2mm=bool(self.attachment_grip_error<.002),
                    bottle_final_target_under_2mm=bool(final_error<.002),
                    extension_planted_soles_above_minus1mm=bool(np.min(arrays['sole_min_z'][extension][arrays['foot_contact'][extension]])>-.001),
                    extension_planted_soles_below3mm=bool(np.max(arrays['sole_max_z'][extension][arrays['foot_contact'][extension]])<.003),
                    extension_planted_sole_point_drift_under3mm=bool(np.nanmax(drift[extension])<.003))
        result_status='reference_candidate' if all(checks.values()) else 'rejected_reference'
        self.task['reference']=dict(schema='g1-pickplace-reference/v1',physical_execution_verified=False,
            object_attachment='Kinematic reference only; wrist FK drives bottle while object_grasp_active. Released support is a reference pose, not simulated settling.',
            grasp_active_semantics='Any commanded door or bottle grasp; separate door_grasp_active and object_grasp_active identify the entity.',
            phases=phases, fps=self.fps, prefix_frames=self.prefix_count,
            carry_wrist_root_relative=[.26,-.22,.30],bottle_stance_xy=[-2.20,-.26],bottle_wrist_yaw_radians=2.18286,
            planned_route=[[-1.75,-.55],[-1.7,-1.6],[2.7,-1.4],[3.20,-1.60]],
            unresolved=['New poses require independent scene/self-collision and visual review.','No dynamic balance, force, grasp stability, SONIC tracking, or real-world alignment claim.'])
        np.savez_compressed(self.output/'trajectory.npz',**arrays)
        (self.output/'task.json').write_text(json.dumps(self.task,indent=2)+'\n')
        (self.output/'timeline.json').write_text(json.dumps(phases,indent=2)+'\n')
        metrics=dict(schema='g1-pickplace-reference-build/v1',status=result_status,
            physical_execution_verified=False,scene_collision_verified=False,self_collision_verified=False,
            source_scene_sha256=self.task['scene']['sha256'],inputs=inputs,
            script_sha256=sha(__file__),imported_motion_helper_sha256=sha(original.__file__),
            frames=len(qpos),duration_seconds=float(arrays['time'][-1]),checks=checks,
            joint_limit_violation_max_rad=float(max(0,np.max(joint_violation))),
            joint_step_max_rad=float(np.max(abs(np.diff(qpos[:,7:],axis=0)))),
            root_step_max_m=float(np.max(np.linalg.norm(np.diff(qpos[:,:3],axis=0),axis=1))),
            wrist_position_error_max_m=float(np.nanmax(errors)),
            foot_target_error_max_m=float(np.max(feet_errors)),
            extension_planted_sole_min_z_m=float(np.min(arrays['sole_min_z'][extension][arrays['foot_contact'][extension]])),
            extension_planted_sole_max_z_m=float(np.max(arrays['sole_max_z'][extension][arrays['foot_contact'][extension]])),
            extension_planted_sole_point_drift_max_m=float(np.nanmax(drift[extension])),
            final_bottle_base_world=arrays['object_base_pose'][-1,:3].tolist(),
            final_bottle_body_center_world=arrays['object_pose'][-1,:3].tolist(),
            final_bottle_target_error_m=final_error,bottle_attachment_grip_error_m=self.attachment_grip_error,
            bottle_attachment_axis_error_rad=self.attachment_axis_error,
            released_table_reference_duration_seconds=2.5,phase_metrics=phase_metrics,
            artifacts={name:dict(path=str((self.output/name).resolve()),sha256=sha(self.output/name))
                       for name in ['trajectory.npz','task.json','timeline.json']})
        (self.output/'build.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (self.output/'status.json').write_text(json.dumps(dict(status=result_status,frames=len(qpos),
            build_sha256=sha(self.output/'build.json'),physical_execution_verified=False),indent=2)+'\n')
        print(json.dumps({k:metrics[k] for k in ['status','frames','duration_seconds','checks','wrist_position_error_max_m','foot_target_error_max_m','joint_step_max_rad','final_bottle_target_error_m']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--walk',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=int,default=30)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'trajectory.npz').exists():raise FileExistsError('Use a fresh reference iteration directory')
    task=json.loads(args.task.read_text())
    reference=Reference(task,args.prefix,args.walk,args.fps,args.output)
    try:
        reference.build()
    except Exception as error:
        failure=dict(schema='g1-pickplace-build-failure/v1',status='failed_before_complete_reference',
                     error=repr(error),frames=len(reference.rows),script_sha256=sha(__file__),
                     physical_execution_verified=False)
        (args.output/'status.json').write_text(json.dumps(failure,indent=2)+'\n')
        np.savez_compressed(args.output/'partial_diagnostic.npz',qpos=np.array([r['qpos'] for r in reference.rows]),
                            phase=np.array([r['phase'] for r in reference.rows]))
        raise
    inputs={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in
            [('task',args.task),('prefix',args.prefix),('walk',args.walk)]}
    reference.save(inputs)


if __name__=='__main__':main()
