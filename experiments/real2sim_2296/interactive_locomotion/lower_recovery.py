"""Reference-only standing recovery after a crouched drawer interaction.

No actual state, force, model, actuator, or policy history is assigned. After
the opening recipe completes, its motor overrides and native observations
continue while ScaleBFM follows a feet-preserving standing reference. Then
the original AMO handback runs. The runtime owns action sequencing and failure handling.
"""
import json,time,math
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp

class RecoveryReference:
    def __init__(self, engine):
        for name,default,low,high in (('lower_recovery_seconds',3.,1.,5.),
                ('lower_standing_height_m',.747,.70,.80), ('lower_recovery_xy_gain',0.,0.,1.),
                ('lower_recovery_xy_cap_m',.06,0.,.10), ('lower_recovery_xy_speed_m_s',.04,.001,.04)):
            value=engine.c.get(name,default)
            if type(value) not in (int,float) or not low<=value<=high or not math.isfinite(value):
                raise ValueError(name+' is outside the finite recovery bounds')
        self.engine=engine;self.original=engine.skill;self.start=float(engine.data.time)
        self.attempt_id=engine.attempt_id
        self.artifact_prefix='lower_recovery_'+str(engine.attempt_id).zfill(3)
        self.duration=float(engine.c.get('lower_recovery_seconds',3.))
        self.settle=0.;self.timed_out=False;self.finished=False
        self.xy_shift=np.zeros(2);self.feedback_gain=float(engine.c.get('lower_recovery_xy_gain',0.))
        self.feedback_cap=float(engine.c.get('lower_recovery_xy_cap_m',.06));self.feedback_speed=float(engine.c.get('lower_recovery_xy_speed_m_s',.04));self.feedback_rows=[]
        if not 0<=self.feedback_gain<=1. or not 0<=self.feedback_cap<=.10 or not 0<self.feedback_speed<=.04:raise ValueError('Recovery reference feedback exceeds its declared bounds')
        self.r=engine.robot;rd=mujoco.MjData(self.r);q=engine.i.measured(engine.data)
        self.left_park=dict(engine.c.get('lower_recovery_left_park',{}));self.left_park_start={}
        for name,value in self.left_park.items():
            if name not in engine.arm_rest or not name.startswith('left_') or not np.isfinite(value):raise ValueError('Recovery park must name an existing left-arm motor target')
            joint=self.r.joint(name)
            if not joint.range[0]+.002<=value<=joint.range[1]-.002:raise ValueError('Recovery park must preserve source stops')
            self.left_park_start[name]=float(engine.arm_rest[name])
        if self.left_park:engine.arm_rest=dict(engine.arm_rest)
        rd.qpos[:]=q;mujoco.mj_forward(self.r,rd)
        self.feet=[self.r.body(s+'_ankle_roll_link').id for s in ('left','right')]
        fp=rd.xpos[self.feet].copy();fr=rd.xmat[self.feet].reshape(2,3,3).copy()
        names=[s+'_'+n+'_joint' for s in ('left','right') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        joints=[self.r.joint(n).id for n in names];self.legq=self.r.jnt_qposadr[joints];bounds=self.r.jnt_range[joints]
        qgoal=q.copy();qgoal[2]=float(engine.c.get('lower_standing_height_m',.747))
        qgoal[3:7]=Rotation.from_euler('z',engine.h.yaw(q[3:7])).as_quat()[[3,0,1,2]]
        qgoal[self.r.joint('waist_yaw_joint').qposadr[0]]=0.
        qgoal[self.r.joint('waist_roll_joint').qposadr[0]]=0.
        qgoal[self.r.joint('waist_pitch_joint').qposadr[0]]=engine.torso_target
        q[self.original.arm.q]=self.original.arm.target;qgoal[self.original.arm.q]=self.original.arm.target
        q[engine.i.rq[self.original.fi]]=0.;qgoal[engine.i.rq[self.original.fi]]=0.
        for name,value in engine.arm_rest.items():
            if name.startswith('left_'):q[self.r.joint(name).qposadr[0]]=value;qgoal[self.r.joint(name).qposadr[0]]=value
        for name,value in self.left_park.items():qgoal[self.r.joint(name).qposadr[0]]=value
        rotations=Slerp([0.,1.],Rotation.from_quat([q[[4,5,6,3]],qgoal[[4,5,6,3]]]))
        self.times=np.linspace(0.,self.duration,61);poses=[];errors=[];seed=q[self.legq].copy();begun=time.monotonic()
        for elapsed in self.times:
            u=engine.h.smooth(elapsed/self.duration);qr=(1-u)*q+u*qgoal;qr[3:7]=rotations([u]).as_quat()[0][[3,0,1,2]]
            park_u=engine.h.smooth(elapsed/.8)
            for name,value in self.left_park.items():qr[self.r.joint(name).qposadr[0]]=(1-park_u)*self.left_park_start[name]+park_u*value
            rd.qpos[:]=qr
            def residual(x):
                rd.qpos[self.legq]=x;mujoco.mj_kinematics(self.r,rd)
                return np.concatenate([np.r_[rd.xpos[b]-fp[k],.2*Rotation.from_matrix(rd.xmat[b].reshape(3,3)@fr[k].T).as_rotvec()] for k,b in enumerate(self.feet)])
            opt=least_squares(residual,np.clip(seed,bounds[:,0]+.0021,bounds[:,1]-.0021),bounds=(bounds[:,0]+.002,bounds[:,1]-.002),max_nfev=60,gtol=1e-9,ftol=1e-9,xtol=1e-9)
            qr[self.legq]=opt.x;seed=opt.x;err=residual(opt.x).reshape(2,6);errors.append([float(np.max(np.linalg.norm(err[:,:3],axis=1))),float(np.max(np.linalg.norm(err[:,3:],axis=1))/.2)]);poses.append(qr)
        self.physical_poses=np.array(poses);self.target=self.physical_poses[-1].copy()
        self.bias=engine.scale.reference_bias.copy();engine.scale.feedback=None
        self.poses=self.physical_poses.copy();self.poses[:,:3]-=self.bias
        # This is a declared virtual-reference translation, never actual alignment.
        self.position_tolerance=.03;self.joint_tolerance=.10
        evidence={'schema':'g1-private-standing-reference/v1','start_time':self.start,'duration_s':self.duration,'standing_height_m':float(qgoal[2]),'reference_bias_frozen':self.bias.tolist(),'leg_joint_names':names,'foot_targets_m':fp.tolist(),'foot_rotation_targets':fr.tolist(),'max_private_foot_error_m_rad':np.max(errors,axis=0).tolist(),'fit_wall_seconds':time.monotonic()-begun,'leg_source_margin_rad':float(np.min(np.minimum(self.physical_poses[:,self.legq]-bounds[:,0],bounds[:,1]-self.physical_poses[:,self.legq]))),'reference_translation_only':True,'native_path_contacts':[]}
        # Native geometry is checked in separate private data at every knot.
        md=mujoco.MjData(engine.model);md.qpos[:]=engine.data.qpos
        for k,qr in enumerate(self.physical_poses):
            md.qpos[engine.i.rootq:engine.i.rootq+7]=qr[:7];md.qpos[engine.i.q]=qr[engine.i.rq];mujoco.mj_forward(engine.model,md)
            for c in md.contact:
                gs=[int(c.geom1),int(c.geom2)];bs=[int(engine.model.geom_bodyid[g])for g in gs]
                if not any(b in engine.i.robot_bodies for b in bs):continue
                if any(g in engine.floor for g in gs) and any(b in engine.i.feet for b in bs):continue
                if c.dist<0.:evidence['native_path_contacts'].append({'knot':k,'time_after_start':float(self.times[k]),'bodies':[engine.model.body(b).name for b in bs],'geoms':[engine.model.geom(g).name for g in gs],'distance_m':float(c.dist)})
        self.native_clear=not any(x['distance_m']<-.003 for x in evidence['native_path_contacts'])
        self.fit_clear=bool(np.max(np.array(errors)[:,0])<.001 and np.max(np.array(errors)[:,1])<.005)
        engine.h.save(engine.out/(self.artifact_prefix+'_reference.json'),evidence)
        np.savez_compressed(engine.out/(self.artifact_prefix+'_reference.npz'),time=self.times+self.start,qpos=self.poses,physical_qpos=self.physical_poses)

    def __getattr__(self,name):return getattr(self.original,name)
    @property
    def success(self):return self.original.success
    @success.setter
    def success(self,value):self.original.success=value
    @property
    def failure(self):return self.original.failure
    @failure.setter
    def failure(self,value):self.original.failure=value
    @property
    def release_pending(self):return self.original.release_pending
    @release_pending.setter
    def release_pending(self,value):self.original.release_pending=value
    @property
    def phase(self):return 'stand_recovery'
    @property
    def done(self):return False
    def at(self,times):
        t=np.clip(np.atleast_1d(times)-self.start,0.,self.duration);q=np.column_stack([np.interp(t,self.times,self.poses[:,k]) for k in range(self.r.nq)]);q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1)[:,None]
        if self.feedback_gain:q[:,:2]+=self.xy_shift
        return q,np.zeros((len(t),43))
    def update(self,now):
        self.original.arm.update(self.original.pos,self.original.rot,posture=self.original.arm.target)
        q,heading,tilt,speed=self.engine.actual();age=now-self.start
        for name,value in self.left_park.items():self.engine.arm_rest[name]=(1-self.engine.h.smooth(age/.8))*self.left_park_start[name]+self.engine.h.smooth(age/.8)*value
        if self.feedback_gain:
            delta=(self.target[:2]-q[:2])*self.feedback_gain*.02;delta*=min(1.,self.feedback_speed*.02/(np.linalg.norm(delta)+1e-12));self.xy_shift+=delta;self.xy_shift*=min(1.,self.feedback_cap/(np.linalg.norm(self.xy_shift)+1e-12))
        self.feedback_rows.append({'time':float(now),'virtual_xy_shift_m':self.xy_shift.tolist(),'actual_root_xy':q[:2].tolist(),'physical_goal_root_xy':self.target[:2].tolist(),'left_park_motor_targets':{n:float(self.engine.arm_rest[n])for n in self.left_park}})
        foot=self.engine.i.floor_force(self.engine.data,self.engine.floor)
        ready=(age>=self.duration and abs(q[2]-self.target[2])<self.position_tolerance and np.linalg.norm(q[:2]-self.target[:2])<.035 and np.sqrt(np.mean((q[self.legq]-self.target[self.legq])**2))<self.joint_tolerance and speed<.10 and tilt<12 and np.all(foot>20) and self.original.clear_dwell>=.3)
        self.settle=self.settle+.02 if ready else 0.;self.finished=self.settle>=.5;self.timed_out=age>8.
    def motor_overrides(self):return self.original.motor_overrides()
    def observe(self,now):
        self.original.observe(now);return self.state()
    def state(self):
        result=self.original.state();result.update(phase=self.phase,done=False,standing_recovery_dwell=self.settle);return result

    def save_feedback(self):
        self.engine.h.save(self.engine.out/(self.artifact_prefix+'_feedback.json'), {
            'schema':'g1-lower-recovery-reference-feedback/v1',
            'attempt_id':self.attempt_id, 'gain':self.feedback_gain,
            'maximum_speed_m_s':self.feedback_speed, 'maximum_shift_m':self.feedback_cap,
            'rows':self.feedback_rows, 'finished':self.finished, 'timed_out':self.timed_out,
            'scope':'Only the virtual whole-body reference root is translated. Actual state is never assigned.'})
