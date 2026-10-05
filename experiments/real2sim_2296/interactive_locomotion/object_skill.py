"""One free-cylinder pickup, persistent motor-controlled carry, and tray placement.

The owner supplies ArmIK explicitly, steps physics, clips the original motors,
and enforces body/room collision and torso guards. This module writes only its
own targets/private FK data. No object attachment, actual-state setter, applied
force, or physics step is used. The pressure law is adapted from the exact R6
arm_carry_feedback.py d543154c89f066ef863774273b777ffc3cae83ec5b04be20d8a8750e8699d260.
Its body reference/posture controller is deliberately not imported.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


R6_INPUTS = {
    "arm_carry_feedback.py": "d543154c89f066ef863774273b777ffc3cae83ec5b04be20d8a8750e8699d260",
    "arm_lower_shelf_torso1_config.json": "11ef3031d6e1720d01979ca72d7b096b495e3896ec18676cf8d029966af4117c",
    "trajectory.npz": "bc3827ebf293c83dc10e07f32b82ec1bc8ee8bdadbeae4945416d0ab0c9bff85",
    "task.json": "3873f285bf3344ac231945768804ec92821d6e080b7bc4866ae2519c3c150f15",
}
SUFFIXES = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
CLOSED = np.array([0., .048076026601645366, -.3726966863369886,
                   .9458678722053423, 1.2446440195855035,
                   .9458678722053423, 1.2446440195855035])
GRIP = np.array([.11320344202949058, .09054644792990237, 0.])
GRASP_SEED = np.array([-.4573664333047325, -.4225538027537598, -.15449260548226748,
                      .29013187505201354, .3676839194512442, .03100363412008353, -1.0005412025761449])
CARRY_LOCAL_POSITION = np.array([.2885911908711106, -.08333487963902628, .3095037545092827])
CARRY_LOCAL_ROTATION = np.array([
    [.8600896063595952, -.46986859475786497, -.19866900284761932],
    [.4794251938870862, .8775827502100646, .0000002650854954746566],
    [.1743483653451556, -.09524715320685337, .9800666443193871]])


def smooth(age):
    x = float(np.clip(age, 0., 1.))
    return x*x*x*(10.+x*(-15.+6.*x))


def vector(value, shape, name):
    try:
        a = np.asarray(value, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid " + name) from exc
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("Invalid finite shape for " + name)
    return a.copy()


def number(value, lo, hi, name):
    if type(value) not in (int, float) or not lo <= value <= hi:
        raise ValueError("Invalid bounded " + name)
    return float(value)


def rotation_mix(a, b, u):
    return Rotation.from_rotvec(float(np.clip(u, 0., 1.)) *
                                Rotation.from_matrix(b@a.T).as_rotvec()).as_matrix()@a


class CylinderForceGrip:
    """R6 three-pad acquisition and opposed two-pad retention, in motor targets.

    Inputs are CURRENT native normal magnitudes and force-weighted normals.
    Filtered values are controller memory and are never release evidence.
    """
    PAIRS = (np.array([1, 2]), np.array([3, 4]), np.array([5, 6]))

    def __init__(self, ranges):
        self.ranges = np.asarray(ranges, float).copy()
        self.active = False
        self.reset(np.zeros(7))

    def reset(self, actual):
        self.active = False
        self.latched = False
        self.ready = self.bad = self.hold_age = 0.
        self.contacted = np.zeros(3, bool)
        self.searching = np.zeros(3, bool)
        self.filtered = np.zeros(3)
        self.goals = np.asarray(actual, float).copy()
        self.targets = np.zeros(3)
        self.supported = self.opposed = False
        self.pair_dot = self.opposition_dot = 1.
        self.mode = 0

    def step(self, desired, actual, normals, vectors, dt, *, active, retain_pair):
        desired = np.asarray(desired, float)
        if not active:
            self.active = self.latched = self.supported = self.opposed = False
            self.targets[:] = 0.
            self.mode = 0
            return desired.copy()
        if not self.active:
            self.reset(actual)
            self.active = True
        normals, vectors = np.asarray(normals), np.asarray(vectors)
        self.filtered += (1.-np.exp(-dt/.01))*(normals-self.filtered)
        opposing = vectors[1]+vectors[2]
        den = float(np.linalg.norm(vectors[0])*np.linalg.norm(opposing))
        self.opposition_dot = float(np.clip(np.dot(vectors[0], opposing)/den, -1, 1)) if den > 1e-12 else 1.
        present = normals >= .04
        self.opposed = bool(np.all(present) and self.opposition_dot <= -.8)
        counters = np.flatnonzero(present[1:])+1
        pair = None
        self.pair_dot = 1.
        if retain_pair and self.latched and present[0] and len(counters) == 1:
            k = int(counters[0])
            den = float(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[k]))
            self.pair_dot = float(np.clip(np.dot(vectors[0], vectors[k])/den, -1, 1)) if den > 1e-12 else 1.
            if self.pair_dot <= -.9:
                pair = k
        self.supported = bool(self.opposed or pair is not None)
        if self.latched:
            self.bad = 0. if self.supported else self.bad+dt
            if self.bad >= .05:
                self.latched = False
                self.hold_age = 0.
        self.ready = self.ready+dt if np.all(self.filtered >= .04) and self.opposed else 0.
        if not self.latched and self.ready >= .03:
            self.latched = True
            self.hold_age = 0.
        hold = np.array([2., 1., 1.])
        if pair is not None:
            hold[:] = 0.
            hold[0] = hold[pair] = 2.
        self.targets[:] = .15+(hold-.15)*min(self.hold_age/.25, 1.) if self.latched else .15
        self.mode = 2 if self.latched and pair is not None else 3 if self.latched and self.opposed else 1
        if self.latched:
            self.hold_age += dt
        result = desired.copy()
        for k, idx in enumerate(self.PAIRS):
            closed = CLOSED[idx]
            direction = np.array([-1., 1.]) if k == 0 else np.array([1., -1.])
            if not self.contacted[k] and normals[k] >= .04:
                self.contacted[k] = True
                self.goals[idx] = np.asarray(actual)[idx]
            goal = self.goals[idx].copy()
            if not self.contacted[k]:
                if np.max(abs(desired[idx]-closed)) < .01 and np.max(abs(goal-closed)) < .005:
                    self.searching[k] = True
                goal += direction*.08*dt if self.searching[k] else np.clip(desired[idx]-goal, -1.5*dt, 1.5*dt)
            else:
                current_mean = float(np.mean(goal-closed))
                desired_mean = float(np.mean(desired[idx]-closed))
                goal += np.clip(desired_mean-current_mean, -1.5*dt, 1.5*dt)
                speed = float(np.clip(.6*(self.targets[k]-self.filtered[k]), -.5, .5))
                if pair is not None and k not in (0, pair):
                    speed = .04
                goal += direction*speed*dt
            lo = self.ranges[idx, 0]+1e-4
            hi = self.ranges[idx, 1]-1e-4
            if self.contacted[k] or self.searching[k]:
                lo, hi = np.maximum(lo, closed-.3), np.minimum(hi, closed+.3)
                mean = float(np.mean(goal-closed))
                pinch = float(np.dot(goal-closed, direction)/2)
                bounds = np.sort(np.column_stack(((lo-closed-mean)/direction,
                                                   (hi-closed-mean)/direction)), axis=1)
                pinch = float(np.clip(pinch, max(bounds[:, 0]), min(bounds[:, 1])))
                goal = closed+mean+direction*pinch
            else:
                goal = np.clip(goal, lo, hi)
            self.goals[idx] = result[idx] = goal
        return result


class ObjectSkill:
    CONTROL_DT = .02
    PHYSICS_DT = .002

    def __init__(self, model, data, robot, interface, descriptor, arm_api):
        self.m, self.d, self.r, self.i = model, data, robot, interface
        self.desc = dict(descriptor)
        self.action = descriptor.get('action')
        if self.action not in ('grasp', 'place'):
            raise ValueError('Object action must be grasp or place')
        self.bid = model.body(descriptor['object_body']).id
        self.og = model.geom(descriptor['object_geom']).id
        if int(model.geom_bodyid[self.og]) != self.bid:
            raise ValueError('Object geom must belong to the selected object body')
        self.joint = int(model.body_jntadr[self.bid])
        if (int(model.body_jntnum[self.bid]) != 1 or
                model.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_FREE or
                np.any((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) &
                       (model.actuator_trnid[:, 0] == self.joint))):
            raise ValueError('Soda requires one unactuated free joint')
        self.dq, self.dv = int(model.jnt_qposadr[self.joint]), int(model.jnt_dofadr[self.joint])
        self.radius = number(descriptor.get('cylinder_radius', .025), .025, .025, 'tested cylinder radius')
        self.halfheight = number(descriptor.get('halfheight', .1), .1, .1, 'tested cylinder halfheight')
        self.mass = number(descriptor.get('mass', .15), .15, .15, 'tested cylinder mass')
        if (model.geom_type[self.og] != mujoco.mjtGeom.mjGEOM_CYLINDER or
                not np.allclose(model.geom_size[self.og, :2], [self.radius, self.halfheight], rtol=0, atol=1e-9) or
                not np.isclose(model.body_mass[self.bid], self.mass, rtol=0, atol=1e-9)):
            raise ValueError('Native cylinder geometry/mass differs from the tested grip domain')
        self.source_support = model.geom(descriptor['source_support_geom']).id
        self._configure_place(descriptor)
        self.fnames = ['right_hand_'+s+'_joint' for s in SUFFIXES]
        self.fi = np.array([interface.index[n] for n in self.fnames], int)
        self.fq, self.fv, self.fa = interface.q[self.fi], interface.v[self.fi], interface.motors[self.fi]
        self.franges = interface.ranges[self.fi].copy()
        if np.any(CLOSED < self.franges[:, 0]) or np.any(CLOSED > self.franges[:, 1]):
            raise ValueError('Source finger ranges exclude the tested cylinder grip')
        self.digits = [model.body('right_hand_'+s+'_link').id for s in ('thumb_2', 'index_1', 'middle_1')]
        self.hand_bodies = {k for k in range(model.nbody) if model.body(k).name.startswith('right_hand_')}
        self.arm = arm_api.ArmIK(model, data, robot, interface)
        self.fk = mujoco.MjData(robot)
        self.torso = robot.body('torso_link').id
        self.wrist = robot.body('right_wrist_yaw_link').id
        self.ik_limits = vector(descriptor.get('ik_error_limits_m_rad', [.005, .025]), (2,), 'IK limits')
        if np.any(self.ik_limits <= 0) or np.any(self.ik_limits > [.005, .025]):
            raise ValueError('IK limits cannot weaken the existing5mm/.025rad guard')
        self.pregrasp_out = number(descriptor.get('pregrasp_out_m', .06), .02, .12, 'pregrasp offset')
        self.pregrasp_up = number(descriptor.get('pregrasp_up_m', .035), 0., .08, 'pregrasp lift')
        self.lift_height = number(descriptor.get('lift_height_m', .05), .035, .08, 'lift height')
        self.retract_distance = number(descriptor.get('retract_distance_m', .08), .08, .20, 'loaded withdrawal')
        self.carry_local_pos = vector(descriptor.get('carry_local_position', CARRY_LOCAL_POSITION), (3,), 'carry position')
        self.carry_local_rot = vector(descriptor.get('carry_local_rotation', CARRY_LOCAL_ROTATION), (3, 3), 'carry rotation')
        if (np.linalg.norm(self.carry_local_pos) > .6 or
                not np.allclose(self.carry_local_rot.T@self.carry_local_rot, np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(self.carry_local_rot), 1., atol=1e-6)):
            raise ValueError('Invalid torso-relative carry frame')
        self.arm_seed = vector(descriptor.get('arm_seed', GRASP_SEED), (7,), 'arm seed')
        if np.any(self.arm_seed < self.arm.lo) or np.any(self.arm_seed > self.arm.hi):
            raise ValueError('Arm seed must remain within source solver bounds')
        self.grip = CylinderForceGrip(self.franges)
        self.started = False
        self.phase = 'not_started'
        self.done = self.success = self.carrying = self.carry_lost = self.requires_pause = False
        self.failure = None
        self.input_bindings = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _configure_place(self, descriptor):
        self.stance_xy = vector(descriptor['stance_xy'], (2,), 'stance XY')
        self.stance_yaw = number(descriptor['stance_yaw'], -100., 100., 'stance yaw')
        self.tray_bid = self.m.body(descriptor['tray_body']).id
        self.tray_floor = self.m.geom(descriptor['tray_floor_geom']).id
        if int(self.m.geom_bodyid[self.tray_floor]) != self.tray_bid:
            raise ValueError('Tray floor must belong to the selected tray')
        value = descriptor['tray_center']
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) not in (2, 3):
            raise ValueError('Tray center requires XY or XYZ')
        center = vector(value, (len(value),), 'tray center')
        self.tray_center = center[:2].copy()
        self.tray_inner = vector(descriptor['tray_inner_halfsize'], (2,), 'tray interior')
        if np.any(self.tray_inner <= .03) or np.any(self.tray_inner > .3):
            raise ValueError('Tray interior must contain the tested cylinder')
        self.tray_z = number(descriptor['tray_floor_z'], .4, 1.2, 'tray support height')

    def _fk(self):
        self.fk.qpos[:] = self.i.measured(self.d)
        mujoco.mj_kinematics(self.r, self.fk)
        return (self.fk.xpos[self.wrist].copy(), self.fk.xmat[self.wrist].reshape(3, 3).copy(),
                self.fk.xpos[self.torso].copy(), self.fk.xmat[self.torso].reshape(3, 3).copy())

    def _object(self):
        pose = self.d.qpos[self.dq:self.dq+7].copy()
        if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1.) > 1e-5:
            raise ValueError('Invalid actual free-object pose')
        rot = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        return pose[:3], rot, pose[:3]-self.halfheight*rot[:, 2]

    def _contacts(self):
        normal, vectors = np.zeros(3), np.zeros((3, 3))
        hand = nonhand = support = tray = other_support = 0.
        rows = []
        f = np.zeros(6)
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if self.bid not in (b1, b2):
                continue
            other = b2 if b1 == self.bid else b1
            mujoco.mj_contactForce(self.m, self.d, k, f)
            mag = max(0., float(f[0]))
            on_object_normal = c.frame[:3]*(-1 if b1 == self.bid else 1)
            if other in self.hand_bodies:
                hand += mag
            else:
                nonhand += mag
                if other != self.tray_bid:
                    other_support += mag
            if other in self.digits:
                idx = self.digits.index(other)
                normal[idx] += mag
                vectors[idx] += mag*on_object_normal
            if self.source_support in (c.geom1, c.geom2):
                support += mag
            if self.tray_floor in (c.geom1, c.geom2):
                tray += mag
            rows.append({'geom1': self.m.geom(c.geom1).name, 'geom2': self.m.geom(c.geom2).name,
                         'body1': self.m.body(b1).name, 'body2': self.m.body(b2).name,
                         'distance_m': float(c.dist), 'normal_N': mag,
                         'normal_world_on_object': on_object_normal.tolist(),
                         'force_world_on_object': ((c.frame.reshape(3, 3).T@f[:3])*
                                                   (-1 if b1 == self.bid else 1)).tolist()})
        return normal, vectors, hand, nonhand, support, tray, other_support, rows

    def start(self, now):
        if self.started:
            raise RuntimeError('An existing grasp instance must persist; do not reset it')
        if self.action != 'grasp':
            raise ValueError('Placement requires begin_place on an actually held object')
        self.begun = self.phase_time = self.last_observe = float(now)
        self.last_update = self.last_motor = None
        self.started = True
        self.stance = self.i.measured(self.d).copy()
        self.reference_q = self.stance.copy()
        self.t = np.array([float(now), float(now)+180.])
        self.pos, self.rot, _, _ = self._fk()
        self.initial_arm = self.d.qpos[self.arm.sq].copy()
        self.finger_reference = self.finger_target = self.d.qpos[self.fq].copy()
        self.initial_fingers = self.finger_target.copy()
        self.grip.reset(self.finger_target)
        self.initial_object, _, self.initial_base = self._object()
        self.grasp_dwell = self.lift_dwell = self.held_dwell = self.clear_dwell = 0.
        self.support_dwell = self.place_dwell = self.grip_loss_dwell = 0.
        self.grasp_achieved = self.lift_achieved = False
        self.object_contacts = []
        self.contact_N = self.nonhand_N = self.source_support_N = self.tray_support_N = self.other_support_N = 0.
        self.digit_force_raw, self.digit_normals = np.zeros(3), np.zeros((3, 3))
        self.opposed = self.current_held = self.inside_tray = self.place_retained = False
        self.opposition_dot = 1.
        self.object_tilt = self.object_linear_speed = self.object_angular_speed = 0.
        self.held_offset = self.held_rotation = None
        self.outward = -np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw), 0.])
        self.grasp_rotation = Rotation.from_euler('z', self.stance_yaw+2.18286-2.9).as_matrix()
        self.closed_pos = self.initial_object + [0., 0., .045] - self.grasp_rotation@GRIP
        self.pregrasp = self.closed_pos+self.pregrasp_out*self.outward+[0., 0., self.pregrasp_up]
        self.pregrasp_arm = self.arm.solve(self.stance, self.pregrasp, self.grasp_rotation,
                                          seed=self.arm_seed, iterations=100)
        if not self._ik_ok():
            return self.state()
        # Flex before raising: the direct rest-to-pregrasp joint interpolation
        # crosses the cabinet front. These two source-range postures leave the
        # open fingertips outside it before the Cartesian approach begins.
        self.tuck_arm = self.initial_arm.copy()
        self.tuck_arm[3] = -.6
        self.raise_arm = self.tuck_arm.copy()
        self.raise_arm[0] = -.7
        self._phase('approach_tuck', now)
        return self.state()

    def _phase(self, phase, now):
        self.phase, self.phase_time = phase, float(now)
        self.segment_pos, self.segment_rot = self.pos.copy(), self.rot.copy()
        self.segment_fingers = self.finger_target.copy()
        self.segment_arm = self.arm.target.copy()

    def _fail(self, reason, *, pause=True):
        if self.failure is None:
            self.failure = reason
        self.success = False
        if pause:
            self.requires_pause = self.done = True

    def _ik_ok(self):
        if not np.isfinite(self.arm.error).all() or np.any(self.arm.error > self.ik_limits):
            self._fail('Object wrist planning exceeded the5mm/.025rad IK guard')
            return False
        return True

    def _command(self, position, rotation):
        self.pos, self.rot = np.asarray(position).copy(), np.asarray(rotation).copy()
        old = self.arm.target.copy()
        result = self.arm.solve(self.i.measured(self.d), self.pos, self.rot, iterations=60)
        if not self._ik_ok():
            return
        self.arm.target = np.clip(result, old-3*self.CONTROL_DT, old+3*self.CONTROL_DT)
        self.arm.velocity += (1.-np.exp(-self.CONTROL_DT/.06))*((self.arm.target-old)/self.CONTROL_DT-self.arm.velocity)

    def _move(self, position, rotation, duration, age):
        u = smooth(age/duration)
        self._command((1-u)*self.segment_pos+u*np.asarray(position), rotation_mix(self.segment_rot, rotation, u))

    def _joint_move(self, target, duration, age):
        u = smooth(age/duration)
        self.arm.update(self.pos, self.rot, posture=(1-u)*self.segment_arm+u*target)
        # During a joint segment the logged Cartesian goal is FK of the
        # commanded arm in the current measured body, not a stale endpoint.
        self.fk.qpos[:] = self.i.measured(self.d)
        self.fk.qpos[self.arm.q] = self.arm.target
        mujoco.mj_kinematics(self.r, self.fk)
        self.pos = self.fk.xpos[self.wrist].copy()
        self.rot = self.fk.xmat[self.wrist].reshape(3, 3).copy()

    def at(self, times):
        values = np.atleast_1d(np.asarray(times, float))
        if not self.started or values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Started skill and finite simulation times required')
        # Keep the measured body/root hold, but expose the same arm/finger
        # targets that the motor overrides request. A stale rest-arm reference
        # makes the whole-body tracker compensate for intentional arm motion.
        self.reference_q[self.arm.q] = self.arm.target
        self.reference_q[self.i.rq[self.fi]] = self.finger_target
        return np.repeat(self.reference_q[None], len(values), axis=0), np.zeros((len(values), self.r.nq-7))

    def phase_at(self, now):
        return self.phase

    def update(self, now):
        if not self.started:
            raise RuntimeError('Start before update')
        now = float(now)
        if self.last_update is not None and now < self.last_update-1e-8:
            raise ValueError('Control clock moved backwards')
        if self.last_update is not None and abs(now-self.last_update) < 1e-9:
            return self.state()
        self.last_update = now
        if self.requires_pause:
            return self.state()
        age = round(now-self.phase_time, 9)
        actual, _, _, _ = self._fk()
        if self.phase in ('approach_tuck', 'approach_raise'):
            tucking = self.phase == 'approach_tuck'
            target = self.tuck_arm if tucking else self.raise_arm
            duration = 1.2 if tucking else 1.4
            self._joint_move(target, duration, age)
            self.finger_reference = (1-smooth(age/1.2))*self.initial_fingers if tucking else np.zeros(7)
            if age >= duration and np.max(abs(self.d.qpos[self.arm.sq]-target)) <= .08:
                self.pos, self.rot, _, _ = self._fk()
                self._phase('approach_raise' if tucking else 'pregrasp', now)
            elif age > duration+2.:
                self._fail('Object clearance posture did not settle')
        elif self.phase == 'pregrasp':
            self._move(self.pregrasp, self.grasp_rotation, 1.6, age)
            self.finger_reference = np.zeros(7)
            if age >= 1.6 and np.linalg.norm(actual-self.pregrasp) < .015 and not self.requires_pause:
                self._phase('approach', now)
            elif age > 3.6:
                self._fail('Object pregrasp tracking did not settle')
        elif self.phase == 'approach':
            self._move(self.closed_pos, self.grasp_rotation, 1.4, age)
            self.finger_reference = np.zeros(7)
            if age >= 1.4 and not self.requires_pause:
                self._phase('close', now)
        elif self.phase == 'close':
            actual_object, _, _ = self._object()
            correction = actual_object-self.initial_object
            if np.linalg.norm(correction) > .08:
                self._fail('Object moved outside the bounded acquisition domain')
            else:
                self._command(self.closed_pos+correction, self.grasp_rotation)
                self.finger_reference = smooth(age/.8)*CLOSED
                if age >= .8 and self.grasp_dwell >= .1 and self.grip.latched:
                    self.grasp_achieved = True
                    self.held_offset, self.held_rotation = self._held_transform()
                    self._phase('lift', now)
                elif age >= 4.:
                    self._fail('Could not acquire an opposed cylinder grasp', pause=False)
                    self._phase('abort_release', now)
        elif self.phase == 'lift':
            self._move(self.segment_pos+[0., 0., self.lift_height], self.segment_rot, 1.2, age)
            self.finger_reference = CLOSED.copy()
            if age >= 1.2 and self.lift_dwell >= .4:
                self.lift_achieved = self.carrying = True
                self._phase('retract', now)
            elif age > 3.5:
                self._fail('Cylinder did not leave its support under a held grasp')
        elif self.phase == 'retract':
            self._move(self.segment_pos+self.retract_distance*self.outward, self.segment_rot, 1.6, age)
            self.finger_reference = CLOSED.copy()
            if age >= 1.6 and self.current_held:
                self._phase('carry_blend', now)
        elif self.phase in ('carry_blend', 'carry'):
            _, _, tp, tr = self._fk()
            target, rot = tp+tr@self.carry_local_pos, tr@self.carry_local_rot
            if self.phase == 'carry_blend':
                self._move(target, rot, 2., age)
                if age >= 2. and self.current_held and self.held_dwell >= .4:
                    self._phase('carry', now)
                    self.done = self.success = True
            else:
                self._command(target, rot)
            self.finger_reference = CLOSED.copy()
        elif self.phase == 'place_hover':
            self._move(self.place_hover, self.place_rotation, 2.4, age)
            self.finger_reference = CLOSED.copy()
            if age >= 2.4 and not self.requires_pause:
                self._phase('place_lower', now)
        elif self.phase == 'place_lower':
            self._move(self.place_target, self.place_rotation, 2., age)
            self.finger_reference = CLOSED.copy()
            if age >= 2. and not self.requires_pause:
                self._phase('place_support', now)
        elif self.phase == 'place_support':
            self._command(self.place_target, self.place_rotation)
            self.finger_reference = CLOSED.copy()
            if self.support_dwell >= .3:
                self._phase('release', now)
            elif age > 3.:
                self._fail('Tray support was not measured before release')
        elif self.phase in ('release', 'abort_release'):
            self._command(self.segment_pos, self.segment_rot)
            self.finger_reference = (1-smooth(age/.9))*self.segment_fingers
            if age >= .9 and self.clear_dwell >= .1:
                self.carrying = False
                self._phase('withdraw' if self.phase == 'release' else 'abort_withdraw', now)
            elif age >= 3.:
                self._fail('Open fingers did not release the actual cylinder')
        elif self.phase in ('withdraw', 'abort_withdraw'):
            self._move(self.segment_pos+.13*self.outward, self.segment_rot, 1.2, age)
            self.finger_reference = np.zeros(7)
            if age >= 1.2 and self.clear_dwell >= .3:
                if self.phase == 'abort_withdraw':
                    self.done = True
                    self._phase('failed_clear', now)
                elif self.place_dwell >= 2.:
                    self.done = self.success = True
                    self._phase('placed', now)
            if age > 5. and not self.done:
                self._fail('Released cylinder did not settle inside the tray')
        if not self.done and now-self.begun > 30.:
            self._fail('Object sequence exceeded its bounded30second deadline')
        return self.state()

    def update_carry(self, now):
        if self.phase not in ('carry', 'carry_blend'):
            raise RuntimeError('Carry updates require a held carry phase')
        return self.update(now)

    def _held_transform(self):
        wp, wr, _, _ = self._fk()
        op, ort, _ = self._object()
        return wr.T@(op-wp), wr.T@ort

    def begin_place(self, descriptor, now):
        if (not self.started or not self.carrying or not self.retained_ok() or
                descriptor.get('action') != 'place' or
                descriptor.get('object_body') != self.desc['object_body'] or
                descriptor.get('object_geom') != self.desc['object_geom']):
            raise ValueError('Placement requires the same actually held cylinder')
        self._configure_place(descriptor)
        self.desc, self.action = dict(descriptor), 'place'
        self.stance = self.reference_q = self.i.measured(self.d).copy()
        self.t = np.array([float(now), float(now)+180.])
        self.begun = float(now)
        self.done = self.success = False
        self.support_dwell = self.place_dwell = self.clear_dwell = 0.
        wp, wr, _, _ = self._fk()
        op, ort, _ = self._object()
        offset, relative_rot = wr.T@(op-wp), wr.T@ort
        wrist_yaw = float(np.arctan2(wr[1, 0], wr[0, 0]))
        object_yaw = float(np.arctan2(ort[1, 0], ort[0, 0]))
        desired_object_rot = Rotation.from_euler('z', object_yaw+self.stance_yaw-wrist_yaw).as_matrix()
        self.place_rotation = desired_object_rot@relative_rot.T
        object_target = np.r_[self.tray_center, self.tray_z+self.halfheight+.0003]
        self.place_target = object_target-self.place_rotation@offset
        self.place_hover = self.place_target+[0., 0., .12]
        self.outward = -np.array([np.cos(self.stance_yaw), np.sin(self.stance_yaw), 0.])
        self.pos, self.rot = wp, wr
        self._phase('place_hover', now)
        self.last_update = None
        return self.state()

    def motor_overrides(self):
        if not self.started:
            raise RuntimeError('Start before motor feedback')
        now = float(self.d.time)
        if self.last_motor is not None and abs(now-self.last_motor) < 1e-9:
            return tuple(x.copy() for x in self.last_overrides)
        if self.last_motor is not None and abs(now-self.last_motor-self.PHYSICS_DT) > 1e-8:
            self._fail('Missing500Hz cylinder motor-feedback coverage')
        self.last_motor = now
        normal, vectors, *_ = self._contacts()
        active = self.phase in ('close', 'lift', 'retract', 'carry_blend', 'carry',
                                'place_hover', 'place_lower', 'place_support')
        self.finger_target = self.grip.step(self.finger_reference, self.d.qpos[self.fq], normal, vectors,
                                          self.PHYSICS_DT, active=active, retain_pair=self.grasp_achieved)
        finger_tau = 5.*(self.finger_target-self.d.qpos[self.fq])-.1*self.d.qvel[self.fv]
        self.last_overrides = (self.arm.act.copy(), self.arm.torques(), self.fa.copy(), finger_tau)
        return tuple(x.copy() for x in self.last_overrides)

    def observe(self, now):
        if not self.started:
            raise RuntimeError('Start before observing')
        now = float(now)
        dt = now-self.last_observe
        if abs(dt) < 1e-9:
            return self.state()
        if not np.isfinite(dt) or abs(dt-self.PHYSICS_DT) > 1e-8:
            self._fail('Missing500Hz object contact observation coverage')
            return self.state()
        self.last_observe = now
        (self.digit_force_raw, self.digit_normals, self.contact_N, self.nonhand_N,
         self.source_support_N, self.tray_support_N, self.other_support_N, self.object_contacts) = self._contacts()
        op, ort, base = self._object()
        velocity = self.d.qvel[self.dv:self.dv+6]
        self.object_linear_speed = float(np.linalg.norm(velocity[:3]))
        self.object_angular_speed = float(np.linalg.norm(velocity[3:]))
        self.object_tilt = float(np.degrees(np.arccos(np.clip(ort[2, 2], -1, 1))))
        opposing = self.digit_normals[1]+self.digit_normals[2]
        den = np.linalg.norm(self.digit_normals[0])*np.linalg.norm(opposing)
        self.opposition_dot = float(np.dot(self.digit_normals[0], opposing)/den) if den > 1e-12 else 1.
        self.opposed = bool(self.digit_force_raw[0] > .05 and np.any(self.digit_force_raw[1:] > .05)
                            and self.opposition_dot < -.8)
        self.grasp_dwell = self.grasp_dwell+self.PHYSICS_DT if self.opposed else 0.
        lifting = base[2] > self.initial_base[2]+.025 and self.nonhand_N <= .05 and self.contact_N > .05
        self.lift_dwell = self.lift_dwell+self.PHYSICS_DT if lifting and self.opposed else 0.
        wp, wr, _, _ = self._fk()
        held_error = float(np.linalg.norm(op-(wp+wr@self.held_offset))) if self.held_offset is not None else 0.
        self.held_error = held_error
        current = self.digit_force_raw >= .04
        three = bool(np.all(current) and self.opposition_dot <= -.8)
        pair = bool(current[0] and np.count_nonzero(current[1:]) == 1 and self.opposition_dot <= -.9)
        held = bool((three or pair) and self.grip.latched and self.grasp_achieved
                    and held_error < .04 and self.object_tilt < 30.)
        self.current_held = held and self.nonhand_N <= .05 and base[2] > .65
        self.held_dwell = self.held_dwell+self.PHYSICS_DT if self.current_held else 0.
        monitor = self.phase in ('lift', 'retract', 'carry_blend', 'carry', 'place_hover', 'place_lower', 'place_support')
        if monitor:
            self.grip_loss_dwell = 0. if held else self.grip_loss_dwell+self.PHYSICS_DT
            if self.grip_loss_dwell >= .05:
                self.carry_lost = True
                self._fail('Actual opposed cylinder retention was lost')
        self.clear_dwell = self.clear_dwell+self.PHYSICS_DT if self.contact_N <= .05 else 0.
        # The tray is fixed, but read its actual world orientation for footprint containment.
        tray_rot = self.d.xmat[self.tray_bid].reshape(3, 3)
        xy = (tray_rot.T@np.r_[op[:2]-self.tray_center, 0.])[:2]
        axis = tray_rot.T@ort[:, 2]
        extent = self.halfheight*np.abs(axis[:2])+self.radius*np.sqrt(np.maximum(0., 1.-axis[:2]**2))
        self.inside_tray = bool(np.all(np.abs(xy)+extent <= self.tray_inner-.002))
        stable = bool(self.object_tilt < 15. and self.object_linear_speed < .04 and self.object_angular_speed < .3)
        supported = bool(self.tray_support_N >= .5*self.mass*9.81 and self.other_support_N <= .05
                         and abs(base[2]-self.tray_z) < .01
                         and self.inside_tray and stable)
        self.support_dwell = self.support_dwell+self.PHYSICS_DT if supported else 0.
        self.place_retained = bool(supported and self.contact_N <= .05 and not self.grip.active)
        self.place_dwell = self.place_dwell+self.PHYSICS_DT if self.place_retained else 0.
        if self.action == 'place' and self.phase == 'placed' and not self.place_retained:
            self.success = False
        if np.any(self.d.qfrc_applied) or np.any(self.d.xfrc_applied):
            self._fail('Object observer detected external applied assistance')
        return self.state()

    def handback_ready(self):
        if self.requires_pause:
            return False
        if self.action == 'grasp' and self.failure is None:
            return bool(self.done and self.phase == 'carry' and self.current_held and self.held_dwell >= .4)
        if self.phase == 'failed_clear':
            return bool(self.done and self.clear_dwell >= .3 and self.source_support_N > .05
                        and np.max(abs(self.d.qpos[self.fq])) <= .18)
        return bool(self.done and self.success and self.place_retained and self.place_dwell >= 2.
                    and self.clear_dwell >= .3 and np.max(abs(self.d.qpos[self.fq])) <= .18)

    def retained_ok(self):
        if self.failure is not None or self.carry_lost or self.requires_pause:
            return False
        return bool(self.current_held) if self.action == 'grasp' else bool(self.place_retained and self.place_dwell >= 2.)

    def state(self):
        if not self.started:
            return {'kind': 'object', 'action': self.action, 'phase': self.phase, 'success': False,
                    'done': False, 'carrying': False, 'failure': self.failure}
        op, ort, base = self._object()
        safe_error = [float(v) if np.isfinite(v) else None for v in self.arm.error]
        return {'kind': 'object', 'action': self.action, 'phase': self.phase, 'done': bool(self.done),
                'success': bool(self.success and self.failure is None), 'failure': self.failure,
                'requires_pause': bool(self.requires_pause), 'carrying': bool(self.carrying),
                'carry_lost': bool(self.carry_lost), 'object_center': op.tolist(), 'object_base': base.tolist(),
                'object_tilt_degrees': float(self.object_tilt), 'linear_speed_m_s': float(self.object_linear_speed),
                'angular_speed_rad_s': float(self.object_angular_speed), 'ik_error': safe_error,
                'grasp_achieved': bool(self.grasp_achieved), 'lift_achieved': bool(self.lift_achieved),
                'grasp_dwell': float(self.grasp_dwell), 'lift_dwell': float(self.lift_dwell),
                'held_dwell': float(self.held_dwell), 'held_position_error_m': float(getattr(self, 'held_error', 0.)),
                'current_held': bool(self.current_held), 'opposed': bool(self.opposed),
                'opposition_dot': float(self.opposition_dot), 'contact_N': float(self.contact_N),
                'nonhand_support_N': float(self.nonhand_N), 'source_support_N': float(self.source_support_N),
                'other_than_hand_or_tray_support_N': float(self.other_support_N),
                'tray_support_N': float(self.tray_support_N), 'inside_tray': bool(self.inside_tray),
                'support_dwell': float(self.support_dwell), 'release_clear_dwell': float(self.clear_dwell),
                'placed_dwell': float(self.place_dwell), 'place_retained': bool(self.place_retained),
                'force_grip_active': bool(self.grip.active), 'force_grip_latched': bool(self.grip.latched),
                'force_support_mode': int(self.grip.mode), 'digit_force_N': self.digit_force_raw.tolist(),
                'digit_normals_world': self.digit_normals.tolist(), 'filtered_digit_force_N': self.grip.filtered.tolist(),
                'force_targets_N': self.grip.targets.tolist(), 'finger_target': self.finger_target.tolist(),
                'finger_open_error_rad': float(np.max(abs(self.d.qpos[self.fq]))),
                'object_contacts': self.object_contacts}
