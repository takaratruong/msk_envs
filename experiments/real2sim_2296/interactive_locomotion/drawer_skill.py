"""Handle-frame drawer opening using private IK and original G1 motor torques.

The slide is passive. This module never writes actual qpos/qvel, applies a
floating-base force, creates an attachment, or steps the simulation. The owning
runtime supplies actual state, applies bounded torques, and observes every step.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x**3 * (10 + x * (-15 + 6*x))


def mix_rotation(a, b, u):
    return Slerp([0., 1.], Rotation.from_matrix([a, b]))([np.clip(u, 0, 1)]).as_matrix()[0]


class ArmIK:
    """Robot-only private FK/Jacobians, with named mappings into the room."""
    def __init__(self, model, data, robot, interface):
        self.m, self.d, self.r, self.i = model, data, robot, interface
        self.rd = mujoco.MjData(robot)
        self.names = ['right_'+n+'_joint' for n in (
            'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
            'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
        joints = [robot.joint(n).id for n in self.names]
        self.q = robot.jnt_qposadr[joints]
        self.v = robot.jnt_dofadr[joints]
        self.idx = np.array([interface.index[n] for n in self.names])
        self.sq, self.sv = interface.q[self.idx], interface.v[self.idx]
        self.act = interface.motors[self.idx]
        self.lo, self.hi = robot.jnt_range[joints, 0]+.002, robot.jnt_range[joints, 1]-.002
        self.lo[-1], self.hi[-1] = max(self.lo[-1], -1.45), min(self.hi[-1], 1.45)
        self.hi[1] = min(self.hi[1], -.02)
        self.body = robot.body('right_wrist_yaw_link').id
        self.actual_body = model.body('right_wrist_yaw_link').id
        self.jp, self.jr = np.zeros((3, robot.nv)), np.zeros((3, robot.nv))
        self.target = data.qpos[self.sq].copy()
        self.velocity = np.zeros(7)
        self.error = np.zeros(2)
        self.kp = np.array([80.]*4+[40.]*3)
        self.kd = np.array([4.]*4+[2.]*3)

    def solve(self, pose, position, rotation, *, seed=None, iterations=24):
        self.rd.qpos[:] = pose
        self.rd.qpos[self.q] = np.clip(self.target if seed is None else seed, self.lo, self.hi)
        for _ in range(iterations):
            mujoco.mj_kinematics(self.r, self.rd)
            mujoco.mj_comPos(self.r, self.rd)
            dp = position - self.rd.xpos[self.body]
            dr = Rotation.from_matrix(rotation @ self.rd.xmat[self.body].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(dp) < 2e-5 and np.linalg.norm(dr) < 2e-4:
                break
            mujoco.mj_jacBody(self.r, self.rd, self.jp, self.jr, self.body)
            j = np.vstack([self.jp[:, self.v], .25*self.jr[:, self.v]])
            step = j.T @ np.linalg.solve(j@j.T + np.eye(6)*1e-5, np.r_[dp, .25*dr])
            step *= min(1., .18/(np.linalg.norm(step)+1e-12))
            self.rd.qpos[self.q] = np.clip(self.rd.qpos[self.q]+step, self.lo, self.hi)
        mujoco.mj_kinematics(self.r, self.rd)
        self.error[:] = [np.linalg.norm(position-self.rd.xpos[self.body]),
            np.linalg.norm(Rotation.from_matrix(rotation@self.rd.xmat[self.body].reshape(3, 3).T).as_rotvec())]
        return self.rd.qpos[self.q].copy()

    def update(self, position, rotation, *, posture=None, dt=.02):
        prev = self.target.copy()
        if posture is None:
            result = self.solve(self.i.measured(self.d), position, rotation)
        else:
            result = np.asarray(posture)
            self.error[:] = 0.  # This segment follows joint posture, not an IK constraint.
        self.target = np.clip(np.clip(result, self.lo, self.hi), prev-3*dt, prev+3*dt)
        self.velocity += (1-np.exp(-dt/.06))*((self.target-prev)/dt-self.velocity)

    def torques(self):
        return self.kp*(self.target-self.d.qpos[self.sq]) + self.kd*(
            self.velocity-self.d.qvel[self.sv]) + self.d.qfrc_bias[self.sv]


class DrawerSkill:
    CLOSED = np.array([0., -.09189, -.847978, 1.055898, 1.440395, 1.055898, 1.440395])
    FINGER_SUFFIXES = ('thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1', 'middle_0', 'middle_1')
    GRIP = np.array([.111948, .074234, 0.])

    def __init__(self, model, data, robot, interface, descriptor):
        self.m, self.d, self.r, self.i = model, data, robot, interface
        self.desc = descriptor
        self.integral_control = descriptor.get('integral_control', False)
        if not isinstance(self.integral_control, bool):
            raise ValueError('Integral motor correction must be enabled with a boolean')
        self.method = descriptor.get('method', 'opposed_pinch')
        if self.method not in ('opposed_pinch', 'contact_hook'):
            raise ValueError('Unknown drawer interaction method')
        self.retraction_mode = descriptor.get('retraction_mode', 'neutral_backoff')
        if self.retraction_mode not in ('neutral_backoff', 'park_clear', 'unwind_retract'):
            raise ValueError('Unknown drawer retraction mode')
        self.joint = model.joint(descriptor['joint_name']).id
        if model.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError('Drawer must have a sliding joint')
        if np.any(model.actuator_trnid[:, 0] == self.joint):
            raise ValueError('Drawer joint must be passive')
        self.dq, self.dv = int(model.jnt_qposadr[self.joint]), int(model.jnt_dofadr[self.joint])
        self.bid = model.body(descriptor['body_name']).id
        self.axis = np.asarray(descriptor['opening_axis_world'], float)
        if self.axis.shape != (3,) or not np.isfinite(self.axis).all() or not np.isclose(np.linalg.norm(self.axis), 1.):
            raise ValueError('Unit opening axis required')
        self.rail = {model.geom(n).id for n in descriptor['rail_geom_names']}
        if not self.rail or any(model.geom_bodyid[g] != self.bid for g in self.rail):
            raise ValueError('Grasp rail must belong to the selected moving drawer')
        self.closed_handle = np.asarray(descriptor['handle_world'], float)
        self.grip = np.asarray(descriptor.get('grip_point_wrist', self.GRIP), float)
        self.rotation = np.column_stack([[0., 0., -1.], -self.axis, np.cross([0., 0., 1.], self.axis)])
        roll = float(descriptor.get('grasp_roll_degrees', 0.))
        if not np.isfinite(roll) or abs(roll) > 90.:
            raise ValueError('Grasp rotation about the rail must be finite and within90 degrees')
        self.rotation = self.rotation @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
        self.pregrasp_offset = np.asarray(descriptor.get('pregrasp_offset_m', [.1, .04]), float)
        if (self.pregrasp_offset.shape != (2,) or not np.isfinite(self.pregrasp_offset).all() or
                not 0.02 <= self.pregrasp_offset[0] <= .2 or not 0. <= self.pregrasp_offset[1] <= .12):
            raise ValueError('Pregrasp offsets must be20..200 mm outward and0..120 mm upward')
        self.clearance_lift = float(descriptor.get('clearance_lift_m', .04))
        if not np.isfinite(self.clearance_lift) or not 0. <= self.clearance_lift <= .08:
            raise ValueError('Clearance lift must be finite and between0 and80 mm')
        self.withdraw_distance = float(descriptor.get('withdraw_distance_m', .10))
        self.defer_unfold = descriptor.get('defer_finger_unfold', False)
        if not np.isfinite(self.withdraw_distance) or not .02 <= self.withdraw_distance <= .15:
            raise ValueError('Withdrawal distance must be finite and between 20 and 150 mm')
        if not isinstance(self.defer_unfold, bool) or (self.defer_unfold and self.retraction_mode != 'unwind_retract'):
            raise ValueError('Deferred finger opening requires the unwind/retract sequence')
        self.index_pressure = float(descriptor.get('hook_index_pressure_N', 0.))
        self.takeup_distance = float(descriptor.get('hook_takeup_m', .01))
        if (not np.isfinite([self.index_pressure, self.takeup_distance]).all() or
                not 0. <= self.index_pressure <= 10. or not 0. <= self.takeup_distance <= .01):
            raise ValueError('Hook pressure must be0..10 N and take-up0..10 mm')
        if self.index_pressure and (self.method != 'contact_hook' or descriptor.get('force_grip', False)):
            raise ValueError('Coupled hook pressure requires the contact-hook method without pinch feedback')
        if np.linalg.det(self.rotation) < .999:
            raise ValueError('Horizontal drawer grasp frame must be right handed')
        self.fnames = ['right_hand_'+n+'_joint' for n in self.FINGER_SUFFIXES]
        self.fi = np.array([interface.index[n] for n in self.fnames])
        self.fq, self.fv, self.fa = interface.q[self.fi], interface.v[self.fi], interface.motors[self.fi]
        self.digits = {model.body('right_hand_'+n+'_link').id: n for n in ('thumb_2', 'index_1', 'middle_1')}
        self.hand_bodies = {b for b in interface.robot_bodies if model.body(b).name.startswith(('right_hand_', 'right_wrist_'))}
        self.arm = ArmIK(model, data, robot, interface)
        if descriptor.get('arm_working_limits') is not None:
            limits = np.asarray(descriptor['arm_working_limits'], float)
            if (limits.shape != (2, 7) or not np.isfinite(limits).all() or
                    np.any(limits[0] >= limits[1]) or np.any(limits[0] < self.arm.lo) or
                    np.any(limits[1] > self.arm.hi)):
                raise ValueError('Arm working limits must strictly fit inside the original solver limits')
            self.arm.lo, self.arm.hi = limits[0].copy(), limits[1].copy()
        self.arm.kp *= float(descriptor.get('arm_gain_scale', 1.))
        self.arm.kd *= float(descriptor.get('arm_damping_scale', 1.))
        self.started = False

    def start(self, now):
        self.started = True
        self.begun = self.phase_time = float(now)
        self.phase = 'reach'
        self.stance = self.i.measured(self.d)
        self.reference_q = self.stance.copy()
        self.waist_q = int(self.r.joint('waist_yaw_joint').qposadr[0])
        self.waist_sq = int(self.m.joint('waist_yaw_joint').qposadr[0])
        self.waist_target = float(self.stance[self.waist_q])
        self.initial_arm = self.d.qpos[self.arm.sq].copy()
        self.retraction_arm = np.asarray(self.desc.get('retraction_arm') if self.desc.get('retraction_arm') is not None
                                         else self.initial_arm, float)
        if (self.retraction_arm.shape != (7,) or not np.isfinite(self.retraction_arm).all() or
                np.any(self.retraction_arm < self.arm.lo) or np.any(self.retraction_arm > self.arm.hi)):
            raise ValueError('Retraction pose must obey the original arm joint limits')
        self.initial_pos = self.d.xpos[self.arm.actual_body].copy()
        self.initial_rot = self.d.xmat[self.arm.actual_body].reshape(3, 3).copy()
        self.initial_slide = float(self.d.qpos[self.dq])
        self.pull_origin = self.initial_slide
        self.goal = min(float(self.desc.get('target_open_m', .22)), float(self.m.jnt_range[self.joint, 1])*.7)
        self.goal = max(self.goal, self.initial_slide)
        self.minimum_retained_position = self.goal-.015
        self.finger_target = self.d.qpos[self.fq].copy()
        self.arm_integral = np.zeros(7)
        self.finger_integral = np.zeros(7)
        self.arm_integral_cap = 1.5*self.m.dof_frictionloss[self.arm.sv]
        self.finger_integral_cap = 1.5*self.m.dof_frictionloss[self.fv]
        self.finger_correction = np.zeros(3)
        self.hook_pressure = 0.
        self.hook_digit_correction = np.zeros(2)
        self.loaded_pair_dwell = 0.
        self.digit_force = np.zeros(3)
        self.digit_force_raw = np.zeros(3)
        self.digit_normals = np.zeros((3, 3))
        self.opposed = False
        self.opposed_dwell = 0.
        self.distal_dwell = 0.
        self.distal_axis_force = self.other_axis_force = 0.
        self.distal_opening_work = self.other_opening_work = 0.
        self.distal_signed_work = self.other_signed_work = 0.
        self.initial_fingers = self.finger_target.copy()
        self.closed = np.asarray(self.desc.get('finger_closed', self.CLOSED), float).copy()
        squeeze = float(self.desc.get('finger_squeeze_rad', 0.))
        self.closed += squeeze*np.array([0., -1., -.5, .5, 1., .5, 1.])
        self.closed = np.clip(self.closed, self.i.ranges[self.fi, 0], self.i.ranges[self.fi, 1])
        fitted = np.asarray(self.desc.get('finger_closed', self.CLOSED), float)
        self.release_fingers = fitted+np.array([0., .04, -.04, -.04, .04, -.04, .04])
        self.release_fingers = np.clip(self.release_fingers, self.i.ranges[self.fi, 0], self.i.ranges[self.fi, 1])
        if self.desc.get('finger_release') is not None:
            release = np.asarray(self.desc['finger_release'], float)
            if (release.shape != (7,) or not np.isfinite(release).all() or
                    np.any(release < self.i.ranges[self.fi, 0]) or
                    np.any(release > self.i.ranges[self.fi, 1])):
                raise ValueError('Release fingers must obey the original seven joint ranges')
            self.release_fingers = release.copy()
        self.pregrasp_pos = (self.closed_handle+self.axis*self.initial_slide-self.rotation@self.grip +
                            self.pregrasp_offset[0]*self.axis + [0., 0., self.pregrasp_offset[1]])
        self.pregrasp_arm = self.arm.solve(self.stance, self.pregrasp_pos, self.rotation,
            seed=np.asarray(self.desc.get('arm_seed', self.initial_arm)), iterations=100)
        self.pregrasp_error = self.arm.error.copy()
        self.pregrasp_current_error = self.pregrasp_error.copy()
        self.retreat = 0.
        self.pos = self.initial_pos.copy()
        self.rot = self.initial_rot.copy()
        self.contacts_seconds = self.two_digit_seconds = self.open_dwell = self.clear_dwell = 0.
        self.max_displacement = self.initial_slide
        self.previous_slide = self.progress_high_water = self.initial_slide
        self.paired_open_travel = 0.
        self.hooked_open_travel = 0.
        self.digits_seen = set()
        self.last_observe = float(now)
        self.success = False
        self.release_pending = False
        self.done = False
        self.failure = None
        self.contact_force = 0.
        self.contact_digits = set()
        self.hand_contact = False
        self.history = []

    def at(self, times):
        return np.tile(self.reference_q, (len(np.atleast_1d(times)), 1)), np.zeros((len(np.atleast_1d(times)), 43))

    def change(self, phase, now):
        self.history.append({'time': float(now), 'from': self.phase, 'to': phase,
                             'slide_m': float(self.d.qpos[self.dq])})
        self.phase, self.phase_time = phase, float(now)
        self.segment_pos, self.segment_rot = self.pos.copy(), self.rot.copy()
        self.segment_fingers = self.finger_target.copy()
        self.segment_arm = self.arm.target.copy()
        self.segment_retreat = self.retreat
        self.segment_waist = self.waist_target

    def opening_evidence(self):
        if self.method == 'contact_hook':
            return (self.contacts_seconds >= .1 and
                    self.hooked_open_travel >= .5*(self.goal-self.initial_slide) and
                    self.distal_opening_work >= self.other_opening_work)
        return (self.two_digit_seconds >= .1 and len(self.digits_seen) >= 2 and
                self.paired_open_travel >= .5*(self.goal-self.initial_slide))

    def update(self, now):
        # Simulation time accumulates floating error. The same one-second
        # segment must not last an extra control tick when F is pressed later.
        age = round(float(now-self.phase_time), 9)
        handle = self.closed_handle+self.axis*float(self.d.qpos[self.dq])
        wrist = handle-self.rotation@self.grip
        posture = None
        if self.phase == 'reach':
            u = smooth(age/2.5)
            if self.desc.get('joint_reach', False):
                if self.desc.get('body_compensated_reach', False):
                    # The support controller can move the torso while the arm
                    # rises. Retarget the same world pregrasp in the measured
                    # body, retaining the selected arm branch, so the following
                    # wrist-hold segment does not start with a pose jump.
                    self.pregrasp_arm = self.arm.solve(self.i.measured(self.d),
                        self.pregrasp_pos, self.rotation, seed=self.pregrasp_arm)
                    self.pregrasp_current_error = self.arm.error.copy()
                posture = (1-u)*self.initial_arm+u*self.pregrasp_arm
                self.arm.rd.qpos[:] = self.i.measured(self.d)
                self.arm.rd.qpos[self.arm.q] = posture
                mujoco.mj_kinematics(self.r, self.arm.rd)
                self.pos = self.arm.rd.xpos[self.arm.body].copy()
                self.rot = self.arm.rd.xmat[self.arm.body].reshape(3, 3).copy()
            else:
                self.pos = self.initial_pos*(1-u)+self.pregrasp_pos*u
                self.rot = mix_rotation(self.initial_rot, self.rotation, u)
            self.finger_target = (1-u)*self.initial_fingers
            if age >= 2.5:
                self.change('preshape', now)
        elif self.phase == 'preshape':
            self.pos, self.rot = self.pregrasp_pos, self.rotation
            self.finger_target = self.release_fingers*smooth(age/1.)
            if age >= 1.:
                self.change('approach', now)
        elif self.phase == 'approach':
            u = smooth(age/1.0)
            self.pos = (1-u)*self.segment_pos+u*(wrist+[0., 0., self.pregrasp_offset[1]])
            self.rot = self.rotation
            if age >= 1.0:
                self.change('lower' if self.pregrasp_offset[1] > 0. else 'close', now)
        elif self.phase == 'lower':
            u = smooth(age/1.)
            self.pos = (1-u)*self.segment_pos+u*wrist
            self.rot = self.rotation
            if age >= 1.:
                self.change('close', now)
        elif self.phase == 'close':
            self.pos, self.rot = wrist, self.rotation
            u = smooth(age/1.)
            self.finger_target = (1-u)*self.release_fingers+u*self.closed
            acquired = not self.desc.get('force_grip', False) or self.opposed_dwell >= .1
            if age >= 1.4 and self.method == 'contact_hook':
                self.change('takeup', now)
            elif age >= 1.4 and acquired:
                self.change('pull', now)
            elif age >= 4.:
                self.failure = 'Could not acquire '+('distal handle contact' if self.method == 'contact_hook' else 'an opposed handle grip')
                self.change('release', now)
        elif self.phase == 'takeup':
            # An unloaded hook can graze the rail with almost no normal force.
            # Take up at most10 mm of slack before requiring loaded contact.
            # This phase receives no opening-travel/work success credit.
            self.pos = self.segment_pos+self.takeup_distance*smooth(age/1.)*self.axis
            self.rot, self.finger_target = self.rotation, self.closed.copy()
            loaded = self.loaded_pair_dwell >= .1 if self.index_pressure else self.distal_dwell >= .1
            if loaded:
                self.pull_origin = max(self.initial_slide, float(self.d.qpos[self.dq]))
                self.change('pull', now)
            elif age >= 2.:
                self.failure = 'Could not load the distal hook within the bounded acquisition'
                self.change('release', now)
        elif self.phase == 'pull':
            intended = min(self.goal, self.pull_origin+.055*age)
            if self.desc.get('retreat_follow_drawer', False):
                # Step back with measured drawer travel. Walking away on the
                # clock while a grip slips pushes the hand outside its reach.
                travel = np.clip(float(self.d.qpos[self.dq])-self.initial_slide, 0., self.goal-self.initial_slide)
                desired_retreat = self.desc.get('retreat_ratio', .7)*travel
                self.retreat += float(np.clip(desired_retreat-self.retreat, -.065*.02, .065*.02))
            else:
                self.retreat = self.desc.get('retreat_ratio', .7)*(intended-self.initial_slide)
            # Limit forward pull lead, but keep an absolute task target when
            # the passive drawer overshoots. Following it with a fixed tiny
            # backward offset can feed motion into a sliding handle indefinitely.
            compliance_offset = float(self.desc.get('hook_load_offset_m', 0.))*smooth(age/1.)
            lead = min(intended+compliance_offset-float(self.d.qpos[self.dq]), .035)
            self.pos, self.rot = wrist+lead*self.axis, self.rotation
            self.finger_target = self.closed.copy()
            if (self.desc.get('release_on_target', False) and
                    float(self.d.qpos[self.dq]) >= self.goal and self.opening_evidence()):
                # Release once the actual target is reached. The existing open
                # dwell still has to pass; it is evaluated during release.
                self.release_pending = True
                self.change('release', now)
            elif self.open_dwell >= .5:
                self.success = self.opening_evidence()
                self.failure = None if self.success else 'Opening lacks the declared distal-handle travel/work evidence'
                self.change('release', now)
            elif age >= 9.:
                self.failure = 'Drawer did not reach the contact-qualified opening target'
                self.change('release', now)
        elif self.phase == 'release':
            u = smooth(age/1.2)
            self.finger_target = (1-u)*self.segment_fingers+u*self.release_fingers
            self.pos, self.rot = self.segment_pos, self.segment_rot
            if age >= 1.2:
                if self.clearance_lift > 0.:
                    self.change('lift_clear', now)
                elif self.clear_dwell >= .1:
                    # A horizontal release must be physically clear before
                    # withdrawing; a phase label alone is not disengagement.
                    self.change('withdraw', now)
                elif age >= 3.:
                    self.failure = self.failure or 'Fingers did not clear the handle; holding release posture'
                    self.release_pending = False
                    self.success = False
        elif self.phase == 'lift_clear':
            self.pos = self.segment_pos+smooth(age/1.)*np.array([0., 0., self.clearance_lift])
            self.rot = self.segment_rot
            if age >= 1. and self.clear_dwell >= .1:
                self.change('withdraw', now)
            elif age >= 3.:
                self.failure = self.failure or 'Lift did not clear the handle; holding clearance posture'
                self.release_pending = False
                self.success = False
        elif self.phase == 'withdraw':
            u = smooth(age/1.5)
            self.pos = self.segment_pos + u*self.withdraw_distance*self.axis
            self.rot = self.segment_rot
            if age >= 1.5:
                self.change('unwind' if self.defer_unfold else 'unfold', now)
        elif self.phase == 'unfold':
            self.pos, self.rot = self.segment_pos, self.segment_rot
            self.finger_target = (1-smooth(age/1.))*self.segment_fingers
            if age >= 1.:
                self.change('backoff' if self.retraction_mode == 'neutral_backoff' else 'unwind', now)
        elif self.phase == 'unwind':
            # AMO returns the waist toward its walking posture. Complete that
            # turn with wrist feedback still active so the hand stays clear.
            self.waist_target = self.segment_waist*(1-smooth(age/2.))
            self.pos, self.rot = self.segment_pos, self.segment_rot
            self.finger_target = self.release_fingers.copy() if self.defer_unfold else np.zeros(7)
            unwind_seconds = 2. if abs(self.segment_waist) >= .08 else 0.
            if age >= unwind_seconds and abs(self.d.qpos[self.waist_sq]) < .08 and self.clear_dwell >= .3:
                self.change('retract' if self.retraction_mode == 'unwind_retract' else 'park', now)
            elif age >= 4.:
                self.failure = self.failure or 'Could not unwind the torso with a clear hand'
                self.done = True
                self.change('failed', now)
        elif self.phase == 'park':
            # The hand has withdrawn 100 mm and risen 40 mm before unfolding.
            # Preserve that measured clear posture for locomotion handback;
            # lowering to the old arm posture can sweep through the open front.
            self.pos, self.rot = self.segment_pos, self.segment_rot
            self.finger_target = np.zeros(7)
            if age >= 1. and self.clear_dwell >= .3 and np.max(np.abs(self.d.qpos[self.fq])) < .18:
                self.done = True
                self.change('done', now)
            elif age >= 3.:
                self.failure = self.failure or 'Withdrawn hand did not establish a clear ready posture'
                self.done = True
                self.change('failed', now)
        elif self.phase == 'backoff':
            # The open front occupies the neutral arm's old return path. Move
            # the support reference back while holding the released hand in
            # world space, then retract only after the actual base follows.
            self.retreat = self.segment_retreat+.15*smooth(age/3.)
            self.pos, self.rot = self.segment_pos, self.segment_rot
            target_xy = (self.stance[:3]+self.retreat*self.axis)[:2]
            actual_xy = self.d.qpos[self.i.rootq:self.i.rootq+2]
            settled = (np.linalg.norm(actual_xy-target_xy) < .035 and
                       np.linalg.norm(self.d.qvel[self.i.rootv:self.i.rootv+2]) < .15)
            if age >= 3. and settled and self.clear_dwell >= .3:
                self.change('retract', now)
            elif age >= 5.:
                self.failure = self.failure or 'Could not step clear before lowering the arm'
                self.done = True
                self.change('failed', now)
        elif self.phase == 'retract':
            u = smooth(age/2.)
            posture = self.segment_arm*(1-u)+self.retraction_arm*u
            if self.defer_unfold:
                # Move the released, compact hand clear of the front before
                # extending its fingers during the joint-space return.
                self.finger_target = self.segment_fingers*(1-smooth((age-.8)/.8))
            if age >= 2. and self.clear_dwell >= .3 and np.max(np.abs(self.d.qpos[self.fq])) < .18:
                self.done = True
                self.change('done', now)
            elif age >= 5.:
                self.failure = self.failure or 'Hand retraction did not establish clearance'
                self.done = True
                self.change('failed', now)
        if self.done and self.release_pending:
            self.failure = self.failure or 'Released drawer did not settle open for the required dwell'
            self.release_pending = False
        self.arm.update(self.pos, self.rot, posture=posture)
        self.reference_q = self.stance.copy()
        self.reference_q[:3] += self.retreat*self.axis
        self.reference_q[self.waist_q] = self.waist_target
        self.reference_q[self.arm.q] = self.arm.target
        self.reference_q[self.i.rq[self.fi]] = self.finger_target

    def motor_overrides(self):
        target = self.finger_target.copy()
        if self.index_pressure:
            weight = 0.
            if self.phase in ('close', 'takeup', 'pull'):
                weight = smooth((self.d.time-self.phase_time-1.)/.3) if self.phase == 'close' else 1.
                # Coupled proximal/distal motion closes the two lower pads
                # without changing their distal orientation. Pressure comes
                # only from measured selected-rail contacts and source motors.
                self.hook_digit_correction += weight*np.clip(
                    (self.index_pressure-self.digit_force[1:])*.2, -.25, .25)*.002
                self.hook_digit_correction = np.clip(self.hook_digit_correction, 0., .25)
            elif self.phase == 'release':
                weight = 1-smooth((self.d.time-self.phase_time)/1.2)
            b, c = weight*self.hook_digit_correction
            target += [0., 0., 0., b, -b, c, -c]
        if self.method == 'contact_hook' and self.desc.get('hook_pressure_feedback', False):
            weight = 0.
            if self.phase in ('close', 'takeup', 'pull'):
                weight = smooth((self.d.time-self.phase_time-1.)/.3) if self.phase == 'close' else 1.
                self.hook_pressure = float(np.clip(self.hook_pressure+
                    weight*np.clip((3.-self.digit_force[0])*.01, -.02, .02)*.002, 0., .02))
            elif self.phase == 'release':
                weight = 1-smooth((self.d.time-self.phase_time)/1.2)
            # Keep distal-thumb orientation while increasing rail preload.
            target += weight*self.hook_pressure*np.array([0., -1., 1., 0., 0., 0., 0.])
        if self.desc.get('force_grip', False) and self.phase in ('close', 'pull'):
            if self.phase == 'close':
                weight = smooth((self.d.time-self.phase_time-1.)/.3)
            else:
                weight = 1.
            self.finger_correction += weight*np.clip((3.-self.digit_force)*.08, -.1, .12)*.002
            self.finger_correction = np.clip(self.finger_correction, -.04, .10)
            a, b, c = self.finger_correction*weight
            target += [0., -a, -.5*a, .6*b, b, .6*c, c]
        target = np.clip(target, self.i.ranges[self.fi, 0], self.i.ranges[self.fi, 1])
        arm_tau = self.arm.torques()
        finger_kp = self.desc.get('finger_kp', 5.)
        finger_tau = finger_kp*(target-self.d.qpos[self.fq])-.15*self.d.qvel[self.fv]
        if self.integral_control:
            # Source joint friction leaves a small steady PD position error,
            # enough to miss this thin rail. Integrate only small joint errors
            # with a one-second time constant and a friction-scaled torque cap.
            # Larger transients decay the correction instead of winding it up.
            for accumulated, cap, error, gain in (
                    (self.arm_integral, self.arm_integral_cap,
                     self.arm.target-self.d.qpos[self.arm.sq], self.arm.kp),
                    (self.finger_integral, self.finger_integral_cap,
                     target-self.d.qpos[self.fq], finger_kp)):
                near = np.abs(error) <= .04
                accumulated[:] = np.where(near, accumulated+.002*gain*error,
                                           accumulated*np.exp(-.002/.1))
                accumulated[:] = np.clip(accumulated, -cap, cap)
            arm_tau = arm_tau+self.arm_integral
            finger_tau = finger_tau+self.finger_integral
        return self.arm.act, arm_tau, self.fa, finger_tau

    def observe(self, now):
        dt = float(now-self.last_observe)
        if not np.isfinite(dt) or not np.isclose(dt, .002, rtol=0., atol=1e-8):
            raise ValueError('Drawer contact observations require each 500 Hz physics step')
        self.last_observe = float(now)
        # Accumulate the declared fixed cadence, not timestamp subtraction
        # noise that can make exactly 100 ms fail a clearance threshold.
        dt = .002
        digits, force, hand_contact, robot_contact = set(), 0., False, False
        forces, normals = np.zeros(3), np.zeros((3, 3))
        distal_axis_force = other_axis_force = 0.
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            a, b = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if self.bid not in (a, b):
                continue
            other = b if a == self.bid else a
            if other in self.hand_bodies:
                hand_contact = True
            if other not in self.i.robot_bodies:
                continue
            robot_contact = True
            f = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, k, f)
            world_force = c.frame.reshape(3, 3).T@f[:3]
            force_on_drawer = world_force*(1 if b == self.bid else -1)
            on_distal_rail = (c.geom1 in self.rail or c.geom2 in self.rail) and other in self.digits
            if on_distal_rail:
                distal_axis_force += float(np.dot(force_on_drawer, self.axis))
                if f[0] > .02:
                    digits.add(self.digits[other])
                    force += float(f[0])
                    idx = ('thumb_2', 'index_1', 'middle_1').index(self.digits[other])
                    forces[idx] += f[0]
                    normals[idx] += c.frame[:3]*f[0]*(1 if a == other else -1)
            else:
                other_axis_force += float(np.dot(force_on_drawer, self.axis))
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        opposed = forces[0] > .2 and any(forces[k] > .2 and np.dot(normals[0], normals[k]) < -.5 for k in (1, 2))
        self.opposed_dwell = self.opposed_dwell+dt if opposed else 0.
        self.digit_force += (1-np.exp(-dt/.02))*(forces-self.digit_force)
        self.digit_force_raw = forces.copy()
        self.digit_normals = normals
        self.opposed = bool(opposed)
        self.distal_dwell = self.distal_dwell+dt if forces.sum() > .2 else 0.
        self.loaded_pair_dwell = self.loaded_pair_dwell+dt if np.all(forces[1:] >= 1.) else 0.
        self.distal_axis_force, self.other_axis_force = distal_axis_force, other_axis_force
        self.digits_seen.update(digits)
        self.contacts_seconds += dt if digits else 0.
        self.two_digit_seconds += dt if len(digits) >= 2 else 0.
        slide = float(self.d.qpos[self.dq])
        advance = max(0., min(slide-self.previous_slide, slide-self.progress_high_water))
        if self.phase == 'pull' and opposed:
            self.paired_open_travel += advance
        if self.phase == 'pull':
            if digits and distal_axis_force > .1:
                self.hooked_open_travel += advance
            positive_delta = max(0., slide-self.previous_slide)
            self.distal_signed_work += distal_axis_force*(slide-self.previous_slide)
            self.other_signed_work += other_axis_force*(slide-self.previous_slide)
            self.distal_opening_work += max(0., distal_axis_force)*positive_delta
            self.other_opening_work += max(0., other_axis_force)*positive_delta
        self.previous_slide = slide
        self.progress_high_water = max(self.progress_high_water, slide)
        self.max_displacement = max(slide, self.max_displacement)
        self.open_dwell = self.open_dwell+dt if slide >= self.goal-.008 and abs(self.d.qvel[self.dv]) <= .035 else 0.
        if self.release_pending and slide < self.goal-.008:
            self.release_pending = False
            self.failure = 'Drawer lost its target opening during controlled release'
        if self.release_pending and self.open_dwell >= .5 and self.opening_evidence():
            self.success = True
            self.release_pending = False
            self.failure = None
        self.clear_dwell = self.clear_dwell+dt if not robot_contact and self.phase in ('release', 'lift_clear', 'withdraw', 'unfold', 'unwind', 'park', 'backoff', 'retract', 'done') else 0.
        self.contact_digits, self.contact_force, self.hand_contact = digits, force, hand_contact
        return self.state()

    def state(self):
        return {'phase': self.phase, 'slide_m': float(self.d.qpos[self.dq]), 'goal_m': self.goal,
                'joint_position': float(self.d.qpos[self.dq]), 'joint_velocity': float(self.d.qvel[self.dv]),
                'goal': self.goal, 'joint_unit': 'm',
                'slide_velocity': float(self.d.qvel[self.dv]), 'contact_N': self.contact_force,
                'digits': sorted(self.contact_digits), 'digits_seen': sorted(self.digits_seen),
                'contact_seconds': self.contacts_seconds, 'two_digit_seconds': self.two_digit_seconds,
                'open_dwell': self.open_dwell, 'clear_dwell': self.clear_dwell,
                'success': self.success, 'done': self.done, 'failure': self.failure,
                'opening_achieved': self.success,
                'digit_force_N': self.digit_force.tolist(), 'opposed_dwell': self.opposed_dwell,
                'finger_correction': self.finger_correction.tolist(),
                'hook_pressure_rad': self.hook_pressure,
                'integral_motor_correction': self.integral_control,
                'arm_integral_torque_Nm': self.arm_integral.tolist(),
                'finger_integral_torque_Nm': self.finger_integral.tolist(),
                'hook_index_correction_rad': self.hook_digit_correction.tolist(),
                'loaded_pair_dwell': self.loaded_pair_dwell,
                'paired_open_travel_m': self.paired_open_travel,
                'method': self.method, 'hooked_open_travel_m': self.hooked_open_travel,
                'retraction_mode': self.retraction_mode,
                'withdraw_distance_m': self.withdraw_distance,
                'defer_finger_unfold': self.defer_unfold,
                'distal_axis_force_N': self.distal_axis_force, 'other_robot_axis_force_N': self.other_axis_force,
                'distal_opening_work_J': self.distal_opening_work, 'other_robot_opening_work_J': self.other_opening_work,
                'distal_signed_work_J': self.distal_signed_work, 'other_robot_signed_work_J': self.other_signed_work,
                'opening_work_definition': 'Positive opening effort: max(axis force, 0) times max(slide increment, 0); signed work is recorded separately',
                'pregrasp_plan_error': self.pregrasp_error.tolist(),
                'pregrasp_current_plan_error': self.pregrasp_current_error.tolist(),
                'finger_open_error_rad': float(np.max(np.abs(self.d.qpos[self.fq]))),
                'max_slide_m': self.max_displacement, 'ik_error': self.arm.error.tolist()}
