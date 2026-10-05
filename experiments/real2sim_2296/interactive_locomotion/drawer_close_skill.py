"""Native-contact drawer push, using the existing bounded arm controller.

Factory injection shares the exact opening implementation without importing a
machine-specific path. All state assignment is to private reference state;
the live robot and passive drawer move only through their native dynamics.
"""
import math
import mujoco
import numpy as np


def make_skill(base, model, data, robot, interface, descriptor):
    for name, default, lower, upper in (
            ('push_offset_m', .015, -.030, .050),
            ('push_speed_m_s', .035, .005, .080),
            ('retreat_ratio', .75, 0., 1.2),
            ('terminal_hold_at_m', 0., 0., .010)):
        value = descriptor.get(name, default)
        if (type(value) not in (int, float) or not lower <= value <= upper or
                not math.isfinite(value)):
            raise ValueError(name+' is outside the finite closing range')
    class DrawerCloseSkill(base.DrawerSkill):
        def start(self, now):
            super().start(now)
            self.goal = 0.
            self.maximum_retained_position = .010
            self.closed_dwell = self.closing_contact_seconds = 0.
            self.hand_closing_travel = self.hand_closing_work = self.other_closing_work = 0.
            self.hand_signed_closing_work = self.other_signed_closing_work = 0.
            self.low_water = self.initial_slide
            self.push_origin = self.initial_slide
            self.push_axis_force = self.other_push_axis_force = 0.
            self.push_root_lag = self.push_lead = 0.
            self.terminal_hold_at = float(self.desc.get('terminal_hold_at_m', 0.))
            self.terminal_hold_position = self.terminal_hold_rotation = None
            self.terminal_hold_time = self.terminal_hold_slide = None
            self.push_offset = float(self.desc.get('push_offset_m', .015))
            self.closed_handle = self.closed_handle+self.push_offset*self.axis
            self.pregrasp_pos += self.push_offset*self.axis
            self.pregrasp_arm = self.arm.solve(self.stance, self.pregrasp_pos, self.rotation,
                seed=self.pregrasp_arm, iterations=100)
            self.pregrasp_error = self.pregrasp_current_error = self.arm.error.copy()

        def change(self, phase, now):
            if phase == 'close':
                phase = 'push'
                self.push_origin = float(self.d.qpos[self.dq])
            super().change(phase, now)

        def retained_position_ok(self):
            return abs(float(self.d.qpos[self.dq])) <= self.maximum_retained_position

        def closing_evidence(self):
            return (self.closing_contact_seconds >= .1 and
                    self.hand_closing_travel >= .5*max(self.initial_slide-.010, 0.) and
                    self.hand_closing_work > 0. and self.hand_closing_work >= self.other_closing_work)

        def update(self, now):
            age = round(float(now-self.phase_time), 9)
            if self.phase not in ('push', 'release'):
                return super().update(now)
            if self.phase == 'push':
                slide = float(self.d.qpos[self.dq])
                if (self.terminal_hold_position is None and self.terminal_hold_at > 0.
                        and abs(slide) <= self.terminal_hold_at):
                    # The preceding commanded pose already passed the owning
                    # runtime's IK guard. Stop advancing it once the passive
                    # slide enters tolerance, then qualify actual settled state.
                    # The latch is not a closure result and never changes state.
                    self.terminal_hold_position = self.pos.copy()
                    self.terminal_hold_rotation = self.rot.copy()
                    self.terminal_hold_time = float(now)
                    self.terminal_hold_slide = slide
                intended = max(-self.push_offset-.006, self.push_origin-float(self.desc.get('push_speed_m_s', .035))*age)
                travel = np.clip(self.initial_slide-slide, 0., self.initial_slide)
                desired = -float(self.desc.get('retreat_ratio', .75))*travel
                if self.terminal_hold_position is None:
                    self.retreat += float(np.clip(desired-self.retreat, -.055*.02, .055*.02))
                # Bounded inward lead; a stalled drawer cannot drag the target
                # through the cabinet or make the body keep advancing on time.
                physical_root_goal = self.stance[:3]+self.retreat*self.axis
                self.push_root_lag = float(np.linalg.norm(self.i.measured(self.d)[:2]-physical_root_goal[:2]))
                lead_limit = min(.025, max(0., .045-self.push_root_lag)*.8)
                lead = float(np.clip(intended-slide, -lead_limit, 0.))
                self.push_lead = lead
                handle = self.closed_handle+self.axis*slide
                if self.terminal_hold_position is None:
                    self.pos = handle-self.rotation@self.grip+lead*self.axis
                    self.rot = self.rotation
                else:
                    self.pos = self.terminal_hold_position.copy()
                    self.rot = self.terminal_hold_rotation.copy()
                    self.push_lead = float((self.pos-(handle-self.rotation@self.grip))@self.axis)
                self.finger_target = self.release_fingers.copy()
                if self.closed_dwell >= .5 and self.closing_evidence():
                    self.success = True
                    self.change('release', now)
                elif age >= 16.:
                    self.failure = 'Drawer did not settle closed with hand-contact travel and work'
                    self.change('release', now)
            elif self.phase == 'release':
                self.pos = self.segment_pos+.030*base.smooth(age/1.)*self.axis
                self.rot = self.segment_rot
                self.finger_target = self.release_fingers.copy()
                if age >= 1. and self.clear_dwell >= .1:
                    self.change('withdraw', now)
                elif age >= 3.:
                    self.failure = self.failure or 'Pushing hand did not clear the drawer'
                    self.success = False
            # Re-solving from the chosen arm branch prevents a redundant
            # elbow from drifting into a limit during a long straight push.
            # Rate limits and native feedback still govern the actual motors.
            result = self.arm.solve(self.i.measured(self.d), self.pos, self.rot,
                seed=np.asarray(self.desc['arm_seed']), iterations=100)
            error = self.arm.error.copy()
            self.arm.update(self.pos, self.rot, posture=result)
            self.arm.error[:] = error
            self.reference_q = self.stance.copy()
            self.reference_q[:3] += self.retreat*self.axis
            self.reference_q[self.waist_q] = self.waist_target
            self.reference_q[self.arm.q] = self.arm.target
            self.reference_q[self.i.rq[self.fi]] = self.finger_target

        def observe(self, now):
            previous = self.previous_slide
            super().observe(now)
            hand_force = other_force = 0.
            for k in range(self.d.ncon):
                c = self.d.contact[k]
                a, b = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
                if self.bid not in (a, b):
                    continue
                other = b if a == self.bid else a
                if other not in self.i.robot_bodies:
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, k, f)
                on_drawer = (c.frame.reshape(3, 3).T@f[:3])*(1 if b == self.bid else -1)
                inward = -float(on_drawer@self.axis)
                if other in self.hand_bodies:
                    hand_force += inward
                else:
                    other_force += inward
            slide = float(self.d.qpos[self.dq])
            if self.phase == 'push':
                increment = max(0., min(previous-slide, self.low_water-slide))
                self.closing_contact_seconds += .002 if hand_force > .1 else 0.
                self.hand_closing_travel += increment if hand_force > .1 else 0.
                delta = previous-slide
                self.hand_closing_work += max(0., hand_force)*max(0., delta)
                self.other_closing_work += max(0., other_force)*max(0., delta)
                self.hand_signed_closing_work += hand_force*delta
                self.other_signed_closing_work += other_force*delta
            self.low_water = min(self.low_water, slide)
            self.push_axis_force, self.other_push_axis_force = hand_force, other_force
            self.closed_dwell = self.closed_dwell+.002 if self.retained_position_ok() and abs(self.d.qvel[self.dv]) < .035 else 0.
            if self.success and not self.retained_position_ok():
                self.success = False
                self.failure = 'Drawer lost its closed position during release or handback'
            return self.state()

        def state(self):
            result = super().state()
            result.update(action='close', opening_achieved=False, closing_achieved=self.success,
                interaction_achieved=self.success, closed_dwell=self.closed_dwell,
                closing_contact_seconds=self.closing_contact_seconds,
                hand_closing_travel_m=self.hand_closing_travel,
                hand_closing_work_J=self.hand_closing_work, other_robot_closing_work_J=self.other_closing_work,
                hand_signed_closing_work_J=self.hand_signed_closing_work,
                other_signed_closing_work_J=self.other_signed_closing_work,
                hand_inward_force_N=self.push_axis_force, other_robot_inward_force_N=self.other_push_axis_force,
                maximum_retained_position_m=self.maximum_retained_position,
                push_root_lag_m=self.push_root_lag, push_lead_m=self.push_lead,
                terminal_hold_at_m=self.terminal_hold_at,
                terminal_hold_active=self.terminal_hold_position is not None,
                terminal_hold_time=self.terminal_hold_time,
                terminal_hold_slide_m=self.terminal_hold_slide,
                terminal_hold_position_m=(self.terminal_hold_position.tolist()
                    if self.terminal_hold_position is not None else None))
            return result

    return DrawerCloseSkill(model, data, robot, interface, descriptor)
