"""Observed soda-to-tray mission state; no physics writes or actuation.

The object remains a native free body. Success is based on measured lift,
unsupported hand carry, complete release, and supported containment in the tray.
"""
from collections import deque
import json
from pathlib import Path
import mujoco
import numpy as np


class ObjectMission:
    def __init__(self, model, data, interface, path):
        self.m, self.d, self.i = model, data, interface
        self.config = json.loads(Path(path).read_text())
        c = self.config
        self.body = model.body(c['object_body']).id
        self.geom = model.geom(c['object_geom']).id
        self.joint = model.joint(c['object_joint']).id
        if (model.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_FREE or
                int(model.jnt_bodyid[self.joint]) != self.body or
                int(model.geom_bodyid[self.geom]) != self.body or
                np.any(model.actuator_trnid[:, 0] == self.joint)):
            raise ValueError('Mission soda must have its own unactuated native free joint')
        if model.geom_type[self.geom] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError('Mission containment uses the declared native cylinder')
        self.radius, self.halfheight = map(float, model.geom_size[self.geom, :2])
        if not np.allclose([self.radius,self.halfheight], [c['cylinder_radius'],c['halfheight']], atol=1e-9):
            raise ValueError('Declared soda size differs from the native collider')
        self.tray_body = model.body(c['tray_body']).id
        self.tray_floor = model.geom(c['tray_floor_geom']).id
        self.tray_geoms = {model.geom(n).id for n in c['tray_geoms']}
        if any(int(model.geom_bodyid[g]) != self.tray_body for g in self.tray_geoms):
            raise ValueError('Tray colliders must belong to the declared receiving fixture')
        self.q = int(model.jnt_qposadr[self.joint])
        self.v = int(model.jnt_dofadr[self.joint])
        self.hand_bodies = {b for b in interface.robot_bodies if model.body(b).name.startswith('right_hand_')}
        self.digit_bodies = [model.body('right_hand_'+n+'_link').id for n in ('thumb_2','index_1','middle_1')]
        self.started = False
        self.holding = False
        self.placing = False
        self.complete = False
        self.failure = None
        self.skill = None
        self.lift_dwell = self.acquisition_dwell = 0.
        self.acquired = self.lifted = False
        self.carry_seconds = self.max_carry_displacement = 0.
        self.carry_origin = None
        self.segment_origin = None
        self.segment_seconds = 0.
        self.last_time = float(data.time)
        self.landing = deque()
        self.max_digit_force = 0.
        self.latest = {}
        self.events = []
        self.initial_center = data.xpos[self.body].copy()

    def begin_grasp(self):
        if self.holding or self.complete:
            raise ValueError('A new pickup requires an empty hand and an unfinished task')
        self.started = True
        self.acquired = self.lifted = False
        self.acquisition_dwell = self.lift_dwell = 0.
        self.carry_seconds = self.max_carry_displacement = self.segment_seconds = 0.
        self.segment_origin = self.carry_origin = None
        self.landing.clear()
        self.events.append({'time':float(self.d.time),'event':'pickup_attempt'})

    def descriptors(self):
        c = self.config
        common = {**c, 'kind':'object', 'qualified': bool(c.get('qualified',False)),
            'handle_height':float(self.d.xpos[self.body,2])}
        soda = {**common, 'id':c['object_id'], 'name':'Soda', 'action':'grasp',
            'body_name':c['object_body'], 'handle_local':[0.,0.,0.],
            'rail_geom_names':[c['object_geom'],'soda_visual_top','soda_visual_bottom','soda_visual_tab'],
            'stance_xy':c['source_stance_xy'], 'stance_yaw':c['source_stance_yaw'],
            'target_position_world':self.d.xpos[self.body].tolist()}
        tray = {**common, 'id':c['tray_id'], 'name':'Blue tray', 'action':'place',
            'body_name':c['tray_body'], 'handle_local':[0.,0.,.004],
            'rail_geom_names':c['tray_geoms'], 'stance_xy':c['tray_stance_xy'],
            'stance_yaw':c['tray_stance_yaw'], 'target_position_world':c['tray_center'],
            'handle_height':c['tray_floor_z']}
        return [soda,tray]

    def blocker(self, action):
        if self.failure:
            return self.failure
        if self.complete:
            return 'Soda delivered'
        if action == 'place':
            return None if self.holding and self.lifted else 'Pick up the soda first'
        if self.holding:
            return 'Soda is already in your hand'
        delta = self.d.xpos[self.body]-np.asarray(self.config['source_center'])
        if np.linalg.norm(delta[:2]) > .04 or abs(delta[2]) > .025:
            return 'The soda moved outside the pickup area'
        if self.latest and self.latest['object_tilt_degrees'] > 15:
            return 'The soda must be standing upright'
        return None

    def observe(self, now):
        dt = float(now)-self.last_time
        if not 0.0019 <= dt <= .0021:
            raise ValueError('Mission observation must cover every 500 Hz physics step')
        self.last_time = float(now)
        m,d = self.m,self.d
        normal = np.zeros(3)
        vectors = np.zeros((3,3))
        hand = support = tray_support = other_robot = 0.
        minimum_distance = 0.
        for k in range(d.ncon):
            con = d.contact[k]
            if self.geom not in (con.geom1,con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.geom else con.geom1
            body = int(m.geom_bodyid[other])
            force = np.zeros(6)
            mujoco.mj_contactForce(m,d,k,force)
            minimum_distance = min(minimum_distance,float(con.dist))
            if body in self.hand_bodies:
                hand += float(force[0])
            elif body in self.i.robot_bodies:
                other_robot += float(force[0])
            else:
                support += float(force[0])
                if other == self.tray_floor:
                    tray_support += float(force[0])
            if body in self.digit_bodies:
                index = self.digit_bodies.index(body)
                # Direction points from each contacting digit toward the soda.
                direction = con.frame[:3] * (1. if con.geom2 == self.geom else -1.)
                normal[index] += force[0]
                vectors[index] += force[0]*direction
        lengths = np.linalg.norm(vectors,axis=1)
        unit = vectors/np.maximum(lengths[:,None],1e-12)
        pair_dot = unit[1:]@unit[0]
        opposed = bool(normal[0] > .04 and any(normal[k+1]>.04 and pair_dot[k]<-.8 for k in range(2)))
        three = bool(np.all(normal>.04) and np.all(pair_dot<-.8))
        center = d.xpos[self.body].copy()
        axis = d.xmat[self.body].reshape(3,3)[:,2]
        extent = self.halfheight*np.abs(axis)+self.radius*np.sqrt(np.maximum(0.,1-axis*axis))
        bottom = float(center[2]-extent[2])
        tilt = float(np.degrees(np.arccos(np.clip(axis[2],-1.,1.))))
        speed = float(np.linalg.norm(d.qvel[self.v:self.v+3]))
        angular = float(np.linalg.norm(d.qvel[self.v+3:self.v+6]))
        if self.started and not self.acquired:
            self.acquisition_dwell = self.acquisition_dwell+dt if three else 0.
            if self.acquisition_dwell >= .1:
                self.acquired = True
                self.events.append({'time':now,'event':'three_pad_acquisition'})
        unsupported = bool(opposed and hand > .1 and support < .04 and other_robot < .04)
        lift = bottom-(self.initial_center[2]-self.halfheight)
        if self.acquired and not self.lifted:
            self.lift_dwell = self.lift_dwell+dt if unsupported and lift>.025 else 0.
            if self.lift_dwell >= .4:
                self.lifted = True
                self.carry_origin = center.copy()
                self.events.append({'time':now,'event':'physical_lift','lift_m':lift})
        if self.lifted and unsupported and not self.complete:
            self.carry_seconds += dt
            if self.segment_origin is None:
                self.segment_origin=center.copy()
                self.segment_seconds=0.
            self.segment_seconds += dt
            if self.segment_seconds>=1.:
                self.max_carry_displacement = max(self.max_carry_displacement,
                    float(np.linalg.norm(center[:2]-self.segment_origin[:2])))
        else:
            self.segment_origin=None
            self.segment_seconds=0.
        tray_xy = np.asarray(self.config['tray_center'][:2])
        inside = bool(np.all(np.abs(center[:2]-tray_xy)+extent[:2] <= np.asarray(self.config['tray_inner_halfsize'])-1e-4))
        released = bool(hand < .01 and other_robot < .01)
        landing_ok = bool(self.placing and self.lifted and self.carry_seconds>=1. and
            self.max_carry_displacement>1. and inside and released and tilt<15. and
            speed<.04 and angular<.3 and support-tray_support<.01 and
            abs(bottom-self.config['tray_floor_z'])<.004)
        if not landing_ok:
            self.landing.clear()
        else:
            self.landing.append((float(now),tray_support>.5*self.config['mass']*9.81))
            while self.landing and now-self.landing[0][0]>2.002:
                self.landing.popleft()
        dwell = float(now-self.landing[0][0]+dt) if self.landing else 0.
        fraction = float(np.mean([x[1] for x in self.landing])) if self.landing else 0.
        if not self.complete and dwell>=2. and fraction>=.95:
            self.complete = True
            self.holding = False
            self.events.append({'time':now,'event':'soda_delivered','tray_support_fraction':fraction})
        self.max_digit_force = max(self.max_digit_force,float(normal.max()))
        self.latest = {'time':float(now), 'object_position':center, 'object_quaternion_wxyz':d.xquat[self.body].copy(),
            'object_velocity':d.qvel[self.v:self.v+6].copy(), 'object_bottom_z':bottom,
            'object_tilt_degrees':tilt,'object_speed':speed,'object_angular_speed':angular,
            'digit_force_N':normal,'digit_normals_world':unit,'opposition_dot':pair_dot,
            'hand_force_N':hand,'support_force_N':support,'tray_floor_force_N':tray_support,
            'other_robot_force_N':other_robot,'minimum_contact_distance':minimum_distance,
            'opposed':opposed,'three_pad_opposed':three,'unsupported':unsupported,
            'acquired':self.acquired,'lifted':self.lifted,'lift_m':float(lift),
            'lift_dwell':self.lift_dwell,'carry_seconds':self.carry_seconds,
            'carry_displacement_m':self.max_carry_displacement,'inside_tray':inside,
            'released':released,'landing_dwell':dwell,'tray_support_fraction':fraction,
            'holding':self.holding,'placing':self.placing,'complete':self.complete}
        return self.latest

    def status(self):
        stage = ('failed' if self.failure else 'complete' if self.complete else
            'placing' if self.placing else 'carrying' if self.holding else 'find')
        objective = {'find':'Find the soda','carrying':'Carry the soda to the blue tray',
            'placing':'Place the soda in the blue tray','complete':'Soda delivered',
            'failed':'Soda task stopped'}[stage]
        return {'stage':stage,'objective':objective,'held_object':self.config['object_id'] if self.holding else None,
            'completed':self.complete,'message':self.failure or objective,
            'physical_lift':self.lifted,'carry_distance_m':self.max_carry_displacement,
            'landing_dwell_s':self.latest.get('landing_dwell',0.)}
