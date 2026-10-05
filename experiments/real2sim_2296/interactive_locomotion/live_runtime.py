#!/usr/bin/env python3
"""Live local G1 control and contact-driven contextual interaction.

One initialized free-base state; all subsequent physical movement uses the 43
original torque motors. A separate renderer receives copies of recorded state.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def load_module(path, name):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def alignment_velocity(delta_world, heading, handoff_radius=.10, minimum_speed=.11):
    """Keep AMO translating outside the radial handoff region.

    AMO's reviewed any-motion gate uses a 0.1 m/s threshold per axis. A small
    diagonal error can otherwise fall below both thresholds while remaining
    outside the configured alignment radius. Preserve the commanded direction and
    existing speed caps, with a small floor only in that stalled region.
    """
    c, s = np.cos(heading), np.sin(heading)
    velocity = np.clip(np.array([[c, s], [-s, c]])@delta_world*1.2,
                       [-.2, -.18], [.2, .18])
    largest = float(np.max(np.abs(velocity)))
    if np.linalg.norm(delta_world) >= handoff_radius and 0. < largest < minimum_speed:
        velocity *= minimum_speed/largest
    return velocity


def route_velocity(delta_world, heading, *, terminal=False):
    """Preserve the checked segment's direction when limiting body-axis speed."""
    c, s = np.cos(heading), np.sin(heading)
    velocity = np.array([[c, s], [-s, c]])@delta_world*1.2
    caps = np.array([.20, .18] if terminal else [.30, .20])
    velocity *= min(1., float(np.min(caps/np.maximum(np.abs(velocity), 1e-12))))
    largest = float(np.max(np.abs(velocity)))
    if np.linalg.norm(delta_world) >= .09 and 0. < largest < .11:
        velocity *= .11/largest
    return velocity


def put_latest(channel, packet):
    """Nonblocking single-producer mailbox: replace a waiting older packet."""
    try:
        channel.put_nowait(packet)
        return True
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            # A multiprocessing feeder or consumer may be between operations.
            # Never stall the physics loop; the next tick supplies newer state.
            pass
        try:
            channel.put_nowait(packet)
            return True
        except queue.Full:
            return False


def render_worker(scene, incoming, outgoing, stop, game_view_path=None):
    """Private visualization state; no simulator state or control ownership."""
    import cv2
    render_stopping = False
    def stop_render(signum, frame):
        nonlocal render_stopping
        render_stopping = True
    signal.signal(signal.SIGTERM, stop_render)
    signal.signal(signal.SIGINT, stop_render)
    outgoing.cancel_join_thread()
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .75
    model.vis.headlight.specular[:] = .15
    model.vis.global_.fovy = 49.
    robot_bodies = {model.body('pelvis').id}
    for body in range(model.nbody):
        if int(model.body_parentid[body]) in robot_bodies:
            robot_bodies.add(body)
    for geom in range(model.ngeom):
        if (int(model.geom_bodyid[geom]) in robot_bodies and model.geom_group[geom] != 1
                and (model.geom_contype[geom] or model.geom_conaffinity[geom])):
            model.geom_rgba[geom, 3] = 0.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 540)
    renderer = mujoco.Renderer(model, height=540, width=960)
    game = load_module(game_view_path, "live_game_view").GameView(model) if game_view_path else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = 2., -15.
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    try:
        while not render_stopping and not stop.is_set():
            try:
                packet = incoming.get(timeout=.1)
            except queue.Empty:
                continue
            while True:
                try:
                    packet = incoming.get_nowait()
                except queue.Empty:
                    break
            if isinstance(packet, dict):
                stamp, q, heading = packet["frame_sim_time"], packet["qpos"], packet["heading"]
            else:
                stamp, q, heading = packet
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            if game is not None and isinstance(packet, dict):
                look = dict(packet['look'])
                if 'heading_world' in look:
                    look['yaw'] = float(np.arctan2(np.sin(look['heading_world']-heading), np.cos(look['heading_world']-heading)))
                result = game.render(renderer, data, heading, look, packet['candidates'], scene_option=option)
                result.setdefault('camera', {}).update(native_heading_world_rad=float(heading),
                    desired_heading_world_rad=float(look.get('heading_world', heading+look['yaw'])),
                    mode=look.get('mode', 'head'))
                result.update(frame_sim_time=stamp, captured_at=packet['captured_at'], view_session=packet['view_session'])
                put_latest(outgoing, result)
                continue
            camera.lookat[:] = q[:3]+np.array([0., 0., -.02])
            camera.azimuth = np.degrees(heading)+75.
            renderer.update_scene(data, camera, scene_option=option)
            image = renderer.render()
            ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                put_latest(outgoing, (stamp, jpg.tobytes()))
    finally:
        renderer.close()


class Engine:
    def __init__(self, config, output, *, initial_instance=None):
        self.c, self.out = config, Path(output)
        self.alignment_radius = float(config.get('alignment_xy_tolerance_m', .10))
        if not np.isfinite(self.alignment_radius) or not .02 <= self.alignment_radius <= .10:
            raise ValueError('Alignment handoff radius must be finite and within20..100 mm')
        self.alignment_minimum_speed = float(config.get('alignment_minimum_speed_m_s', .11))
        if (not np.isfinite(self.alignment_minimum_speed) or
                not .11 <= self.alignment_minimum_speed <= .18):
            raise ValueError('Alignment minimum speed must be finite and within0.11..0.18 m/s')
        self.drawer_entry_lateral_tolerance = float(config.get('drawer_entry_lateral_tolerance_m', .18))
        if (not np.isfinite(self.drawer_entry_lateral_tolerance) or
                not .01 <= self.drawer_entry_lateral_tolerance <= .18):
            raise ValueError('Drawer entry lateral tolerance must be finite and within10..180 mm')
        self.drawer_ik_limits = None
        if config.get('drawer_ik_error_limits_m_rad') is not None:
            limits = np.asarray(config['drawer_ik_error_limits_m_rad'], float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits <= 0.):
                raise ValueError('Drawer IK limits must be positive finite position and rotation errors')
            self.drawer_ik_limits = limits.copy()
        self.h = load_module(config['hybrid_helpers'], 'live_hybrid_helpers')
        h = self.h
        self.legacy = h.module(config['tracker_driver'], 'live_legacy')
        self.robot = mujoco.MjModel.from_xml_path(config['robot'])
        self.source = self.legacy.Reference(Path(config['reference']), self.robot)
        self.model, self.compiled = self.legacy.compile_model(Path(config['scene']), self.out, False, .002,
            json.loads(Path(config['physics_options']).read_text()))
        self.data = mujoco.MjData(self.model)
        self.i = h.RobotInterface(self.model, self.robot)
        self.drawer_entry_template = None
        if config.get('drawer_entry_template'):
            template = json.loads(Path(config['drawer_entry_template']).read_text())
            self.drawer_entry_template = np.asarray(template['qpos'], float)
            if (template['robot_sha256'] != h.sha(config['robot']) or
                    self.drawer_entry_template.shape != (self.robot.nq,) or
                    not np.isfinite(self.drawer_entry_template).all() or
                    abs(np.linalg.norm(self.drawer_entry_template[3:7])-1.) > 1e-6):
                raise ValueError('Invalid named-robot drawer entry reference')
        self.params = json.loads(Path(config['parameters']).read_text())
        self.draw_api = h.module(config['drawer_adapter'], 'live_drawer_skill')
        self.close_api = h.module(config['drawer_close_adapter'], 'live_drawer_close') if config.get('drawer_close_adapter') else None
        self.lower_api = h.module(config['lower_recovery_adapter'], 'live_lower_recovery') if config.get('lower_recovery_adapter') else None
        self.recovery = None
        self.recoveries = []
        self.lower_recovery_started = False
        self.instance_entry_templates = {}
        for identity, path in config.get('drawer_entry_templates', {}).items():
            template = json.loads(Path(path).read_text())
            values = np.asarray(template['qpos'], float)
            if (template['robot_sha256'] != h.sha(config['robot']) or values.shape != (self.robot.nq,) or
                    not np.isfinite(values).all() or abs(np.linalg.norm(values[3:7])-1.) > 1e-6):
                raise ValueError('Invalid per-instance drawer entry template')
            if np.any(values[self.i.rq] < self.i.ranges[:, 0]) or np.any(values[self.i.rq] > self.i.ranges[:, 1]):
                raise ValueError('Per-instance entry template exceeds original motor joint ranges')
            self.instance_entry_templates[identity] = values.copy()
        self.catalog = json.loads(Path(config['drawer_catalog']).read_text())
        self.instances = []
        for row in self.catalog['drawers']:
            handle = row['handles'][0]
            axis = np.asarray(row['axis_world'])
            center = np.asarray(handle['world_center_m'])
            left = np.cross([0., 0., 1.], -axis)
            stance = center + float(config.get('stance_distance_m', .52))*axis + float(config.get('stance_left_m', .16))*left
            descriptor = {'id': row['instance_id'], 'name': row['instance_id'].replace('__', ' · ').replace('_', ' '),
                'kind': 'drawer', 'joint_name': row['native_joint_name'], 'body_name': row['native_body_name'],
                'handle_world': center.tolist(), 'opening_axis_world': axis.tolist(),
                'rail_geom_names': handle['rail_geom_names'], 'stance_xy': stance[:2].tolist(),
                'stance_yaw': float(np.arctan2(-axis[1], -axis[0])), 'target_open_m': config.get('target_open_m', .22),
                'retreat_ratio': config.get('drawer_retreat_ratios', {}).get(row['instance_id'], config.get('retreat_ratio', .7)),
                'grip_point_wrist': config.get('grip_point_wrist', [.111948, .074234, 0.]),
                'finger_closed': config.get('finger_closed', [0., -.09189, -.847978, 1.055898, 1.440395, 1.055898, 1.440395]),
                'finger_release': config.get('drawer_finger_release'),
                'finger_squeeze_rad': config.get('finger_squeeze_rad', 0.),
                'force_grip': config.get('force_grip', False),
                'joint_reach': config.get('joint_reach', False),
                'body_compensated_reach': config.get('drawer_body_compensated_reach', False),
                'integral_control': config.get('drawer_integral_control', False),
                'withdraw_distance_m': config.get('drawer_withdraw_distance_m', .10),
                'defer_finger_unfold': config.get('drawer_defer_finger_unfold', False),
                'method': config.get('drawer_method', 'opposed_pinch'),
                'hook_pressure_feedback': config.get('hook_pressure_feedback', False),
                'hook_load_offset_m': config.get('hook_load_offset_m', 0.),
                'hook_index_pressure_N': config.get('drawer_hook_index_pressure_N', 0.),
                'hook_takeup_m': config.get('drawer_hook_takeup_m', .01),
                'retraction_mode': config.get('drawer_retraction', 'neutral_backoff'),
                'retraction_arm': config.get('drawer_retraction_arm'),
                'arm_seed': config.get('arm_seed', [-.5, -.5, .5, 1., 0., 0., 0.]),
                'grasp_roll_degrees': config.get('drawer_grasp_roll_degrees', 0.),
                'arm_working_limits': config.get('drawer_arm_working_limits'),
                'pregrasp_offset_m': config.get('drawer_pregrasp_offset_m', [.1, .04]),
                'clearance_lift_m': config.get('drawer_clearance_lift_m', .04),
                'retreat_follow_drawer': config.get('drawer_retreat_follow_actual', False),
                'release_on_target': config.get('drawer_release_on_target', False),
                'finger_kp': config.get('finger_kp', 5.),
                'arm_gain_scale': config.get('arm_gain_scale', 1.),
                'arm_damping_scale': config.get('arm_damping_scale', 1.),
                'range_m': row['displacement_limits_m'][1], 'handle_height': center[2],
                'joint_unit': 'm',
                'qualified': row['instance_id'] in config.get('qualified_drawers', []),
                'qualified_close': row['instance_id'] in config.get('qualified_close_drawers', [])}
            descriptor.update(config.get('drawer_instance_overrides', {}).get(row['instance_id'], {}))
            self.instances.append(descriptor)
        self.fridge_api = None
        if config.get('fridge_adapter'):
            self.fridge_api = h.module(config['fridge_adapter'], 'live_fridge_adapter')
            fridge_goal = self.source.at(9.8)[0][0]
            self.instances.append({'id': 'refrigerator_01__upper_door_1', 'kind': 'fridge',
                'name': 'Fridge door', 'stance_xy': fridge_goal[:2].tolist(), 'stance_yaw': h.yaw(fridge_goal[3:7]),
                'handle_height': 1.15, 'joint_name': 'task_fridge_hinge', 'body_name': 'task_fridge_door',
                'qualified': bool(config.get('qualified_fridge', False)), 'target_open_m': float(np.radians(60)),
                'range_m': float(np.radians(110)), 'joint_unit': 'rad',
                'frozen_adapter': config['frozen_fridge_adapter'], 'bundle_dir': config['fridge_bundle']})
        # Scene-specific preconditions belong to the instance configuration.
        # These checks only read actual passive-joint positions; they do not
        # prescribe a pose, close a drawer, or replace physical collision tests.
        requirements = config.get('drawer_entry_requires_closed', {})
        if not isinstance(requirements, dict):
            raise ValueError('Drawer entry preconditions must be a mapping')
        named = {item['id']: item for item in self.instances}
        if any(key not in named or named[key]['kind'] != 'drawer' for key in self.instance_entry_templates):
            raise ValueError('Per-instance entry template must name a catalog drawer')
        for item in self.instances:
            if type(item.get('standing_recovery', False)) is not bool:
                raise ValueError('Standing recovery flag must be boolean')
            if item.get('standing_recovery') and (self.lower_api is None or item['id'] not in self.instance_entry_templates):
                raise ValueError('Crouched drawers require both named entry and standing recovery adapters')
        self.entry_clearance_requirements = {}
        for target, dependencies in requirements.items():
            if target not in named or named[target]['kind'] != 'drawer' or not isinstance(dependencies, dict):
                raise ValueError('Entry preconditions require a known drawer and dependency mapping')
            cached = []
            for neighbor, maximum in dependencies.items():
                if (neighbor == target or neighbor not in named or named[neighbor]['kind'] != 'drawer'
                        or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                        or not np.isfinite(maximum) or not 0 <= maximum <= .02):
                    raise ValueError('A closed-drawer precondition needs a different known drawer and a finite 0..20 mm tolerance')
                item = named[neighbor]
                cached.append((neighbor, item['name'], int(self.model.joint(item['joint_name']).qposadr[0]), float(maximum)))
            self.entry_clearance_requirements[target] = cached
        self.mission = self.object_api = None
        if config.get('object_mission'):
            mujoco.mj_forward(self.model, self.data)
            self.object_api = h.module(config['object_adapter'], 'live_object_skill')
            mission_api = h.module(config['object_mission_adapter'], 'live_object_mission')
            self.mission = mission_api.ObjectMission(self.model, self.data, self.i, config['object_mission'])
            self.instances.extend(self.mission.descriptors())
        initial = self.source.at(0 if initial_instance is None else 9.8)[0][0]
        if initial_instance is not None:
            item = next(x for x in self.instances if x['id'] == initial_instance)
            initial[:2] = item['stance_xy']
            rotate = Rotation.from_euler('z', item['stance_yaw']-h.yaw(initial[3:7]))
            initial[3:7] = (rotate*Rotation.from_quat(initial[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
        offset = np.asarray(config.get('initial_offset', [0, 0, 0]), float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > [.3, .3, .3]):
            raise ValueError('Initial offset exceeds the declared local test domain')
        initial[:2] += offset[:2]
        initial[3:7] = (Rotation.from_euler('z', offset[2])*Rotation.from_quat(initial[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
        if config.get('initial_heading_rad') is not None:
            spawn_heading = config['initial_heading_rad']
            if type(spawn_heading) not in (int, float) or not -np.pi <= spawn_heading <= np.pi:
                raise ValueError('Initial heading must be finite within -pi..pi')
            adjustment = Rotation.from_euler('z', spawn_heading-h.yaw(initial[3:7]))
            initial[3:7] = (adjustment*Rotation.from_quat(initial[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
        self.torso_target = np.radians(float(config.get('torso_target_degrees', 0.)))
        self.enforce_torso = bool(config.get('enforce_forward_torso', False))
        if self.enforce_torso and not 0 < self.torso_target <= np.radians(15):
            raise ValueError('Forward torso target must be positive and at most15 degrees')
        if self.enforce_torso:
            initial[self.robot.joint('waist_pitch_joint').qposadr[0]] = self.torso_target
        self.arm_rest = config.get('arm_rest', {})
        for name, value in self.arm_rest.items():
            if not name.startswith(('left_shoulder_', 'right_shoulder_', 'left_elbow_', 'right_elbow_', 'left_wrist_', 'right_wrist_')):
                raise ValueError('Arm rest configuration may only set arm joints')
            initial[self.robot.joint(name).qposadr[0]] = float(value)
        self.data.qpos[self.i.rootq:self.i.rootq+7] = initial[:7]
        self.data.qpos[self.i.q] = initial[self.i.rq]
        mujoco.mj_forward(self.model, self.data)
        self.initial_state = {'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist()}
        self.torso = self.model.body('torso_link').id
        self.torso_motor_index = self.i.index['waist_pitch_joint']
        self.torso_roll_motor_index = self.i.index['waist_roll_joint']
        self.waist_q_indices = [int(self.model.joint(n).qposadr[0]) for n in
                                ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')]
        self.torso_velocity = np.zeros(6)
        self.minimum_torso_pitch = self.torso_pitch()
        self.minimum_pelvis_heading_pitch = self.torso_angles()[2]
        self.torso_violations = []
        names = self.params['body_joint_names_hardware']
        self.bq = np.array([self.model.joint(n).qposadr[0] for n in names])
        self.bv = np.array([self.model.joint(n).dofadr[0] for n in names])
        self.bi = np.array([self.i.index[n] for n in names])
        rbq = np.array([self.robot.joint(n).qposadr[0] for n in names])
        self.scale = h.module(config['scalebfm_adapter'], 'live_scale').create_policy(
            Path(config['scalebfm_weights']), self.params, rbq, 1, self.model, self.data, self.robot, self.i.rootq)
        self.amo_api = h.module(config['amo_adapter'], 'live_amo')
        native = self.amo_api.AMOPolicy(config['amo_weights'], device='cpu')
        self.amo = h.module(config['amo_command_adapter'], 'live_gate').AMOHeadingGate(native, mode='any_motion')
        self.native = native
        self.posture = initial.copy()
        self.upper = {n: float(initial[self.robot.joint(n).qposadr[0]]) for n in self.amo_api.ARM_JOINT_NAMES}
        self.base_kp, self.base_kd = np.full(43, 5.), np.full(43, .1)
        for name in self.i.names:
            if 'wrist' in name:
                self.base_kp[self.i.index[name]], self.base_kd[self.i.index[name]] = 20., 1.
        self.floor = {self.model.geom(n).id for n in config['floor_geometries']}
        # Native geometry ownership is fixed for this model. Cache the lookup,
        # retaining the same contact classification at every physics substep.
        self.geom_bodies = tuple(int(b) for b in self.model.geom_bodyid)
        self.robot_geoms = tuple(b in self.i.robot_bodies for b in self.geom_bodies)
        self.foot_geoms = tuple(b in self.i.feet for b in self.geom_bodies)
        self.navigator = None
        self.navigation_pool = None
        self.navigation_future = None
        self.approach = None
        self.approach_distance = float(config.get('interaction_approach_distance_m', 6.))
        if not np.isfinite(self.approach_distance) or not 1.5 <= self.approach_distance <= 8.:
            raise ValueError('Interaction approach distance must be within 1.5..8 metres')
        if config.get('approach_navigation_adapter'):
            navigation = h.module(config['approach_navigation_adapter'], 'live_approach_navigation')
            self.navigator = navigation.NativeApproachMap(self.model, self.data,
                options={**config.get('approach_navigation_options', {}), 'floor_geometries': config['floor_geometries']})
            self.navigation_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='room_approach')
        self.empty_navigator = self.navigator
        self.carry_navigator = None
        if self.navigator is not None and self.mission is not None:
            self.carry_navigator = navigation.NativeApproachMap(self.model, self.data,
                options={**config.get('approach_navigation_options', {}),
                    'floor_geometries': config['floor_geometries'],
                    'carried_geom_names': [self.mission.config['object_geom']]})
        self.heading = h.yaw(initial[3:7])
        self.look_blocked_key = None
        self.last_look_key = None
        self.mode, self.message = 'walking', 'Steer toward a supported handle. Press F when prompted.'
        self.entered = self.dwell = 0.
        self.selected = self.skill = None
        self.last_active = False
        self.events, self.results = [], []
        self.attempt_id, self.attempt_result = 0, None
        self.interaction_feedback = None
        self.rows, self.task_rows = collections.defaultdict(list), collections.defaultdict(list)
        self.chunks, self.task_chunks = [], []
        self.physics_rows, self.physics_chunks = collections.defaultdict(list), []
        self.object_rows, self.object_chunks = collections.defaultdict(list), []
        self.contacts = (self.out/'contacts.jsonl').open('w')
        self.paused = False
        self.failed = None
        self.ticks = []
        self.task_state = None
        self.peak_tilt = 0.
        self.stable_after = 0.
        self.qualifications = config.get('qualified_drawers', [])
        self.dev = bool(config.get('development_instances', False))
        self.prepared_fridge = self.fridge_api.FridgeAdapter(self.model, self.data, self.robot, self.i,
            next(x for x in self.instances if x['kind'] == 'fridge')) if self.fridge_api and (self.mission is None or config.get('qualified_fridge',False)) else None
        self.bind_inputs()

    def bind_inputs(self):
        h = self.h
        bindings = {}
        for key, value in self.c.items():
            if isinstance(value, str) and Path(value).is_file():
                bindings[str(Path(value).resolve())] = h.sha(value)
        for path in self.c.get('drawer_entry_templates', {}).values():
            bindings[str(Path(path).resolve())] = h.sha(path)
        bindings.update(self.scale.input_bindings)
        bindings.update(self.native.input_bindings)
        if self.prepared_fridge:
            bindings.update(self.prepared_fridge.input_bindings)
        bindings.update(self.legacy.model_files(Path(self.c['scene'])))
        bindings.update(self.legacy.model_files(Path(self.c['robot'])))
        if self.c.get('transport_ui_dir'):
            for name in ('index.html', 'app.js', 'style.css'):
                path = Path(self.c['transport_ui_dir'])/name
                bindings[str(path.resolve())] = h.sha(path)
        bindings[str(Path(__file__).resolve())] = h.sha(__file__)
        h.save(self.out/'inputs.json', {'schema': 'g1-live-inputs/v1', 'config': self.c,
            'files': bindings, 'initial_state': self.initial_state, 'compiled_scene': str(self.compiled),
            'compiled_scene_sha256': h.sha(self.compiled), 'physics_hz': 500, 'control_hz': 50,
            'actuation': '43 original bounded direct motors; passive drawer slides; no external forces or attachments',
            'initialization': 'The only actual qpos assignment in the simulator, before execution',
            'policy_history': 'AMO runs each tick; ScaleBFM warms during alignment and keeps its native inferred-action history',
            'drawer_inventory_ids': [x['id'] for x in self.instances]})

    def change(self, mode, message, **details):
        previous = self.mode
        self.events.append({'time': float(self.data.time), 'from': self.mode, 'to': mode,
                            'message': message, **details})
        self.mode, self.entered, self.dwell, self.message = mode, float(self.data.time), 0., message
        state = {'planning':'planning', 'approach':'approach', 'align':'align',
            'enter_skill':'interacting', 'interaction_skill':'interacting',
            'handback':'interacting', 'paused':'failed'}.get(mode)
        if state and self.selected is not None:
            self.interaction_notice(state, message)
        elif mode == 'walking' and self.attempt_result is not None:
            self.interaction_notice('done' if self.attempt_result.get('success') else 'failed', message)
        elif mode == 'walking' and previous in ('planning', 'approach', 'align'):
            self.interaction_notice('failed', message)

    def interaction_notice(self, state, message):
        if self.selected is not None:
            self.interaction_feedback = {'id':self.selected['id'],
                'action':self.selected['action'], 'state':state, 'message':message,
                'attempt_id':self.attempt_id}

    def actual(self):
        q = self.i.measured(self.data)
        heading = self.h.yaw(q[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1))))
        speed = float(np.linalg.norm(self.data.qvel[self.i.rootv:self.i.rootv+2]))
        return q, heading, tilt, speed

    def torso_angles(self):
        # mj_step integrates qpos after evaluating body transforms. Reconstruct
        # this named robot's zero-fixed-rotation waist chain from current qpos
        # so the logged post-step angle has no one-step kinematics lag.
        base = np.empty(9)
        mujoco.mju_quat2Mat(base, self.data.qpos[self.i.rootq+3:self.i.rootq+7])
        yaw, roll, pitch = self.data.qpos[self.waist_q_indices]
        cy, sy, cr, sr, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
        body = base.reshape(3, 3)
        torso = body@np.column_stack([
            [cy*cp-sy*sr*sp, sy*cp+cy*sr*sp, -cr*sp],
            [-sy*cr, cy*cr, sr],
            [cy*sp+sy*sr*cp, sy*sp-cy*sr*cp, cr*cp]])
        forward = torso[:, 0]
        # Positive pitch points the chest's forward axis below the horizontal:
        # a forward torso lean, independent of yaw and robot position.
        pitch = float(np.arctan2(-forward[2], np.hypot(forward[0], forward[1])))
        roll = float(np.arctan2(torso[2, 1], torso[2, 2]))
        pelvis_forward = body[:2, 0]/max(np.linalg.norm(body[:2, 0]), 1e-9)
        pelvis_pitch = float(np.arctan2(np.dot(torso[:2, 2], pelvis_forward), torso[2, 2]))
        return pitch, roll, pelvis_pitch

    def torso_pitch(self):
        return self.torso_angles()[0]

    def entry_blocker(self, identity):
        for _, name, address, maximum in self.entry_clearance_requirements.get(identity, []):
            slide = float(self.data.qpos[address])
            if not np.isfinite(slide) or slide > maximum:
                return name
        return None

    def candidates(self, *, include_distant=False):
        q, heading, tilt, speed = self.actual()
        choices = []
        for original in self.instances:
            if original['kind'] == 'object':
                continue
            item = dict(original)
            slide = float(self.data.qpos[self.model.joint(item['joint_name']).qposadr[0]])
            action = 'close' if self.close_api is not None and item['kind'] == 'drawer' and slide > .02 else 'open'
            item['action'] = action
            if action == 'close':
                item['qualified'] = bool(item.get('qualified_close', False))
                offset = min(max(slide, 0.), item['range_m'])*float(item['retreat_ratio'])
                extra = self.c.get('drawer_close_stance_extra_by_instance_m', {}).get(item['id'], self.c.get('drawer_close_stance_extra_m'))
                if extra is not None:
                    offset = extra
                    if type(offset) not in (int, float) or not 0. <= offset <= .25:
                        raise ValueError('Closing stance offset must be finite within0..250mm')
                item['stance_xy'] = (np.asarray(item['stance_xy'])+offset*np.asarray(item['opening_axis_world'])[:2]).tolist()
                item.update(self.c.get('drawer_close_parameters', {}))
                item.update(self.c.get('drawer_close_parameters_by_instance', {}).get(item['id'], {}))
            distance = float(np.linalg.norm(q[:2]-item['stance_xy']))
            stance_left = np.array([-np.sin(item['stance_yaw']), np.cos(item['stance_yaw'])])
            lateral_error = float(np.dot(q[:2]-item['stance_xy'], stance_left))
            angle = abs(self.h.wrap(heading-item['stance_yaw']))
            slide = float(self.data.qpos[self.model.joint(item['joint_name']).qposadr[0]])
            if distance > (self.approach_distance if self.navigator is not None else 1.5) and not include_distant:
                continue
            qualified = item['qualified'] or self.dev
            blocker = self.entry_blocker(item['id'])
            reason = ('Simulation paused' if self.paused else 'Interaction in progress' if self.mode != 'walking' else
                      'Place the soda in the blue tray first' if self.mission is not None and self.mission.holding else
                      'Not qualified for '+action+' yet' if not qualified else
                      'Already open' if action == 'open' and slide >= item['target_open_m']-.008 else
                      'Blocked by open '+blocker if blocker is not None else
                      'Finishing cancelled route planning' if self.navigation_future is not None and not self.navigation_future.done() else
                      'Target is too far away' if self.navigator is not None and distance > self.approach_distance else
                      'Move closer to the selected handle' if self.navigator is None and distance > .18 else
                      'Center yourself in front of the handle' if self.navigator is None and item['kind'] == 'drawer' and
                          abs(lateral_error) > self.drawer_entry_lateral_tolerance else
                      'Face the handle' if self.navigator is None and angle > .35 else
                      'Stop walking first' if self.navigator is None and speed > .25 else
                      'Robot must be standing upright' if tilt > 15 else 'Press F to '+action)
            choices.append((distance, {**item, 'distance_m': distance, 'heading_error_rad': angle,
                'lateral_error_m': lateral_error,
                'eligible': reason == 'Press F to '+action and self.mode == 'walking', 'reason': reason,
                'slide_m': slide}))
        if self.mission is not None:
            for item in self.mission.descriptors():
                distance = float(np.linalg.norm(q[:2]-item['stance_xy']))
                if distance > (self.approach_distance if self.navigator is not None else 1.5) and not include_distant:
                    continue
                angle = abs(self.h.wrap(heading-item['stance_yaw']))
                blocker = self.mission.blocker(item['action'])
                reason = ('Simulation paused' if self.paused else 'Interaction in progress' if self.mode != 'walking' else
                    blocker if blocker else 'Not qualified for '+item['action']+' yet' if not (item['qualified'] or self.dev) else
                    'Finishing cancelled route planning' if self.navigation_future is not None and not self.navigation_future.done() else
                    'Target is too far away' if self.navigator is not None and distance > self.approach_distance else
                    'Move closer' if self.navigator is None and distance > .18 else
                    'Face the object' if self.navigator is None and angle > .35 else
                    'Stop walking first' if self.navigator is None and speed > .25 else
                    'Robot must be standing upright' if tilt > 15 else 'Press F to '+item['action'])
                choices.append((distance,{**item,'distance_m':distance,'heading_error_rad':angle,
                    'eligible':reason=='Press F to '+item['action'], 'reason':reason}))
        return [x[1] for x in sorted(choices, key=lambda x: (x[0], -x[1]['handle_height']))]

    @staticmethod
    def target_moved(current, previous):
        if current['kind'] == 'object':
            return np.linalg.norm(np.asarray(current['target_position_world'])-previous['target_position_world']) > .01
        return abs(current['slide_m']-previous['slide_m']) > .01

    def candidate(self, requested_id=None):
        choices = self.candidates()
        if requested_id is not None:
            return next((x for x in choices if x['id'] == requested_id), None)
        qualified = [x for x in choices if x['qualified'] or self.dev]
        return qualified[0] if qualified else choices[0] if choices else None

    def cancel_approach(self, reason, heading, look_key):
        self.events.append({'time': float(self.data.time), 'event': 'approach_cancelled',
            'attempt_id': self.attempt_id, 'instance': self.selected['id'], 'reason': reason})
        self.approach = None
        self.heading = heading
        self.look_blocked_key = look_key
        self.change('walking', reason)
        self.interaction_notice('cancelled' if reason.startswith('Approach cancelled.') else 'failed',
            'Approach cancelled. You have control.' if reason.startswith('Approach cancelled.') else
            'Cannot find a clear approach from this position.' if reason.startswith('Cannot approach:') else reason)

    def plan_approach(self, measured, candidate, captured_qpos=None):
        """Reject malformed planner output without enabling direct alignment."""
        try:
            planning_data = self.data
            if captured_qpos is not None:
                # The worker owns this private planning state exclusively.
                # It never reads changing actual transforms or writes actual q.
                planning_data = mujoco.MjData(self.model)
                planning_data.qpos[:] = captured_qpos
                mujoco.mj_kinematics(self.model, planning_data)
            route = self.navigator.plan(planning_data, measured[:2], candidate['stance_xy'], candidate['stance_yaw'])
            if not isinstance(route, dict) or type(route.get('success')) is not bool:
                raise ValueError('Planner did not return an explicit route result')
            if not route['success']:
                return {'success': False, 'reason': str(route.get('reason', 'No safe route'))[:300]}
            points = np.asarray(route.get('waypoints'), float)
            if (points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 128 or
                    not np.isfinite(points).all() or np.any(np.abs(points) > 1000.) or
                    np.linalg.norm(points[-1]-candidate['stance_xy']) > 1e-5):
                raise ValueError('Route must end at the exact requested stance with bounded finite waypoints')
            length = float(np.sum(np.linalg.norm(np.diff(np.vstack([measured[:2], points]), axis=0), axis=1)))
            if length > 25.:
                raise ValueError('Route exceeds the 25 metre approach bound')
            terminal = route.get('terminal_waypoint_index')
            anchor = route.get('terminal_anchor_index')
            if (type(terminal) is not int or type(anchor) is not int or
                    not 0 <= anchor < terminal == len(points)-1 or terminal != anchor+1):
                raise ValueError('Route must declare its final fixed-heading corridor and turning anchor')
            corridor = route.get('start_corridor')
            if corridor is not None:
                if (not isinstance(corridor, dict) or
                        set(corridor) != {'end_index','heading_world','maximum_yaw_error_rad','distance_m'} or
                        type(corridor['end_index']) is not int or corridor['end_index'] != 1 or len(points) < 3 or
                        any(type(corridor[k]) not in (int,float) or not np.isfinite(corridor[k])
                            for k in ('heading_world','maximum_yaw_error_rad','distance_m')) or
                        not 0 < corridor['maximum_yaw_error_rad'] <= .15 or
                        not .10 <= corridor['distance_m'] <= 1.0 or
                        np.linalg.norm(points[0]-measured[:2]) > 1e-5 or
                        abs(np.linalg.norm(points[1]-points[0])-corridor['distance_m']) > 1e-5 or
                        abs(self.h.wrap(corridor['heading_world']-self.h.yaw(measured[3:7]))) > 1e-5):
                    raise ValueError('Invalid fixed-heading start corridor')
            return json.loads(json.dumps(route, allow_nan=False))
        except Exception as exc:
            return {'success': False, 'reason': 'Route planning failed: '+str(exc)[:250]}

    def approach_velocity(self, measured, heading, tilt, speed, now):
        """AMO follows a checked route; this method never assigns native state."""
        h, plan = self.h, self.approach
        points = plan['waypoints']
        corridor = plan.get('start_corridor')
        while plan['index'] < len(points)-1:
            at_exit = corridor is not None and plan['index'] == corridor['end_index']
            if np.linalg.norm(points[plan['index']]-measured[:2]) >= (.06 if at_exit else .10):break
            # An approximate waypoint arrival must never allow an early turn
            # while the measured body still lacks the full cruise reserve.
            if at_exit and not self.navigator.cruise_clear(self.data, measured[:2], measured[:2]):break
            plan['index'] += 1
            plan['best_distance'] = float('inf')
            plan['progress_at'] = now
        waypoint = points[plan['index']]
        delta = waypoint-measured[:2]
        distance = float(np.linalg.norm(delta))
        final = plan['index'] == len(points)-1
        terminal = plan['index'] >= plan['terminal_index']
        exiting = corridor is not None and plan['index'] <= corridor['end_index']
        desired_heading = (corridor['heading_world'] if exiting else self.selected['stance_yaw'] if terminal else
                           float(np.arctan2(delta[1], delta[0])))
        heading_error = abs(h.wrap(desired_heading-heading))
        if exiting and heading_error > corridor['maximum_yaw_error_rad']:
            self.cancel_approach('Approach stopped: heading left the checked starting corridor.', heading, self.look_blocked_key)
            return np.zeros(2)
        if exiting != plan.get('start_corridor_active', False):
            plan['next_check'] = now
            self.message = 'Making room to walk to '+self.selected['name'] if exiting else 'Walking to '+self.selected['name']
            self.interaction_notice('approach', self.message)
        plan['start_corridor_active'] = exiting
        old_ready = plan.get('terminal_aligned', False)
        if terminal:
            if heading_error <= .10:
                plan['terminal_aligned'] = True
            elif heading_error > .13:
                plan['terminal_aligned'] = False
        if terminal != plan.get('terminal_active', False) or old_ready != plan.get('terminal_aligned', False):
            plan['next_check'] = now
        plan['terminal_active'] = terminal
        next_heading = self.heading+np.clip(h.wrap(desired_heading-self.heading), -.01, .01)
        self.heading = h.wrap(heading+np.clip(h.wrap(next_heading-heading), -.35, .35))
        if distance < plan['best_distance']-.025:
            plan['best_distance'], plan['progress_at'] = distance, now
        if now-plan['started'] > 90 or now-plan['progress_at'] > 12:
            self.cancel_approach('Approach stopped: no safe progress. You have control.', heading, self.look_blocked_key)
            return np.zeros(2)
        if now >= plan['next_check']:
            plan['next_check'] = now+.10
            if exiting:
                band = corridor['maximum_yaw_error_rad']
                clear = self.navigator.corridor_clear(self.data, measured[:2], waypoint,
                    desired_heading-band, desired_heading+band)
            elif terminal and not plan.get('terminal_aligned', False):
                clear = self.navigator.turn_clear(self.data, measured[:2], heading, desired_heading)
            elif terminal:
                clear = self.navigator.corridor_clear(self.data, measured[:2], waypoint, heading, desired_heading)
            else:
                clear = self.navigator.segment_clear(self.data, measured[:2], waypoint)
            if not clear:
                self.events.append({'time': now, 'event': 'approach_clearance_failed', 'attempt_id': self.attempt_id,
                    'navigation_reason': self.navigator.last_reason, 'position_xy': measured[:2].tolist(),
                    'waypoint_xy': waypoint.tolist(), 'heading': heading, 'desired_heading': desired_heading,
                    'waypoint_index': plan['index'], 'terminal_segment': terminal, 'starting_corridor': exiting})
                self.cancel_approach('Approach stopped: the route is obstructed. You have control.', heading, self.look_blocked_key)
                return np.zeros(2)
        velocity = route_velocity(delta, heading, terminal=terminal)
        if exiting:
            largest = float(np.max(np.abs(velocity)))
            if distance >= .04 and 0. < largest < .11:
                velocity *= .11/largest
            velocity *= min(1., .14/max(float(np.max(np.abs(velocity))), 1e-12))
            if distance < .04:velocity[:] = 0.
        if (terminal and not plan.get('terminal_aligned', False)) or (not terminal and heading_error > .35) or (final and distance < .09):
            velocity[:] = 0.
        if (final and plan.get('terminal_aligned', False) and distance < .12 and abs(h.wrap(heading-self.selected['stance_yaw'])) < .13
                and speed < .15 and tilt < 15):
            plan['next_check'] = now
            self.change('align', 'Aligning with '+self.selected['name'], instance=self.selected['id'])
        return velocity

    def step(self, command):
        if self.paused:
            return
        if self.recovery is not None and self.skill is self.recovery:
            recovery = self.recovery
            if recovery.timed_out:
                self.failed = 'Standing recovery did not settle within 8 seconds'
                self.paused = True
                recovery.original.success = False
                recovery.original.failure = self.failed
                if self.attempt_result is not None:
                    self.attempt_result.update(success=False, failure=self.failed)
                recovery.save_feedback()
                self.change('paused', self.failed)
                return
            if recovery.finished:
                recovery.save_feedback()
                self.skill = recovery.original
                self.events.append({'time': float(self.data.time), 'event': 'standing_recovery_finished', 'attempt_id': self.attempt_id})
                self.begin_handback(self.skill.success and self.skill.failure is None)
        tic = time.perf_counter()
        h, m, d, i = self.h, self.model, self.data, self.i
        now = float(d.time)
        measured, actual_heading, tilt, speed = self.actual()
        if self.mission is not None:
            self.navigator = self.carry_navigator if self.mission.holding else self.empty_navigator
        active = bool(command.get('active', False))
        look_key = (command.get('view_session'), command.get('look_heading'))
        if (self.last_look_key is not None and
                look_key[0] != self.last_look_key[0] and
                look_key[1] == self.last_look_key[1]):
            # Reclaiming controls keeps the camera orientation. A new lease
            # alone is not a fresh horizontal steering request, even when
            # the native loop did not observe the intervening inactive tick.
            self.look_blocked_key = look_key
            if self.mode == 'walking':
                self.heading = actual_heading
        self.last_look_key = look_key
        if self.mode != 'walking':
            # A camera turn during a hand task must not become a queued body
            # turn when walking resumes. Only a changed horizontal demand
            # resumes it; pitch and camera-mode changes remain visual.
            self.look_blocked_key = look_key
        if self.last_active and not active and self.mode == 'walking':
            self.heading = actual_heading
            self.look_blocked_key = look_key
            self.events.append({'time': now, 'event': 'input_deadman', 'virtual_heading_reanchored': self.heading})
        self.last_active = active
        if command.get('interact'):
            requested_id = command.get('interaction_candidate_id')
            candidate = self.candidate(requested_id) if isinstance(requested_id, str) and requested_id else None
            action_matches = candidate is not None and command.get('interaction_action', None if self.c.get('game_view_adapter') else candidate['action']) == candidate['action']
            if self.mode != 'walking' or candidate is None or not candidate['eligible'] or not action_matches:
                self.events.append({'time': now, 'event': 'interaction_rejected', 'mode': self.mode,
                    'reason': 'Displayed action changed' if candidate and not action_matches else candidate['reason'] if candidate else 'No nearby supported mechanism'})
            if candidate is not None and self.mode == 'walking' and candidate['eligible'] and action_matches:
                self.attempt_id += 1
                self.attempt_result = None
                self.skill = self.task_state = None
                self.recovery = None
                self.lower_recovery_started = False
                self.failed = None
                self.selected = candidate
                self.scale.reference_bias[:] = 0.
                self.scale.feedback = None
                self.reference = h.HoldReference(measured)
                if self.navigator is not None:
                    axes = np.asarray([command.get('forward', 0.), command.get('lateral', 0.), command.get('yaw', 0.)], float)
                    self.approach = {'started': now, 'progress_at': now,
                        'best_distance': float('inf'), 'next_check': now, 'look_key': look_key,
                        'initial_axes': axes, 'cancel_armed': bool(np.max(np.abs(axes)) < .05),
                        'target_slide': candidate.get('slide_m')}
                    self.heading = actual_heading
                    self.navigation_future = self.navigation_pool.submit(self.plan_approach, measured.copy(),
                        json.loads(json.dumps(candidate)), d.qpos.copy())
                    self.change('planning', 'Finding a route to '+candidate['name'], instance=candidate['id'],
                        action=candidate['action'], input_sequence=command.get('seq'),
                        interaction_frame_seq=command.get('interaction_frame_seq'), view_session=command.get('view_session'),
                        view_revision=command.get('look_revision'))
                else:
                    self.approach = None
                    self.change('align', 'Aligning with '+candidate['name'], instance=candidate['id'])
        if self.approach is not None and self.mode in ('planning', 'approach', 'align'):
            axes = np.asarray([command.get('forward', 0.), command.get('lateral', 0.), command.get('yaw', 0.)], float)
            current = self.candidate(self.selected['id'])
            manual = (self.approach['cancel_armed'] and np.max(np.abs(axes)) > .1)
            if not self.approach['cancel_armed']:
                if np.max(np.abs(axes)) < .05:
                    self.approach['cancel_armed'] = True
                elif np.max(np.abs(axes-self.approach['initial_axes'])) > .1:
                    manual = True
            reason = ('Approach cancelled. You have control.' if not active or manual else
                'Approach stopped: the selected action changed.' if current is None or current['action'] != self.selected['action'] else
                'Approach stopped: target is no longer qualified.' if not (current['qualified'] or self.dev) else
                'Approach stopped: the target moved.' if self.target_moved(current,self.selected) else None)
            if reason:
                self.cancel_approach(reason, actual_heading, look_key)
        if self.approach is not None and self.mode == 'align' and now >= self.approach['next_check']:
            self.approach['next_check'] = now+.10
            if not self.navigator.corridor_clear(d, measured[:2], self.selected['stance_xy'], actual_heading, self.selected['stance_yaw']):
                self.cancel_approach('Alignment stopped: the route is obstructed. You have control.', actual_heading, look_key)
        if self.mode in ('planning', 'approach', 'align', 'enter_skill'):
            blocker = self.entry_blocker(self.selected['id'])
            if blocker is not None:
                reason = 'Blocked by open '+blocker
                if self.mode in ('planning', 'approach', 'align'):
                    self.heading = actual_heading
                    self.approach = None
                    self.change('walking', reason+'; alignment cancelled')
                else:
                    self.failed = reason+' during entry'
                    self.begin_handback(False)
        velocity = np.zeros(2)
        weight = 0.
        mode_age = round(now-self.entered, 9)
        if self.mode == 'walking':
            if active:
                requested = np.asarray([command.get('forward', 0.), command.get('lateral', 0.), command.get('yaw', 0.)], float)
                if not np.isfinite(requested).all():
                    raise ValueError('Nonfinite normalized live input')
                requested = np.clip(requested, -1, 1)
                velocity = requested[:2]*[.45, .3]
                look_heading = command.get('look_heading')
                if self.c.get('look_steers_robot') and look_heading is not None:
                    if (type(look_heading) not in (int, float) or not -np.pi <= look_heading <= np.pi
                            or not np.isfinite(look_heading)):
                        raise ValueError('Invalid game heading demand')
                    angle = h.wrap(look_heading-actual_heading)
                    c, s = np.cos(angle), np.sin(angle)
                    velocity = np.array([[c, -s], [s, c]])@velocity
                    # Free camera orbit can put a forward view command fully
                    # sideways in the body. Preserve its direction while
                    # retaining the existing body-axis walking speed bounds.
                    velocity *= min(1., float(np.min(np.array([.45, .3])/
                        np.maximum(np.abs(velocity), 1e-12))))
                    if look_key != self.look_blocked_key:
                        # AMO receives a virtual facing target, never a native
                        # pose assignment. Demand is absolute, so it cannot
                        # accumulate into perpetual turning as the body follows.
                        next_heading = self.heading+np.clip(h.wrap(look_heading-self.heading), -.01, .01)
                        self.heading = h.wrap(actual_heading+np.clip(h.wrap(next_heading-actual_heading), -.35, .35))
                else:
                    self.heading = h.wrap(self.heading+requested[2]*.5*.02)
        elif self.mode == 'planning':
            self.reference = h.HoldReference(measured)
            if self.navigation_future.done():
                route = self.navigation_future.result()
                if not route['success']:
                    self.cancel_approach('Cannot approach: '+route['reason'], actual_heading, look_key)
                else:
                    self.approach.update(waypoints=np.asarray(route['waypoints'], float), index=0,
                        terminal_index=route['terminal_waypoint_index'], progress_at=now,
                        start_corridor=route.get('start_corridor'))
                    self.change('approach', 'Walking to '+self.selected['name'], instance=self.selected['id'], route=route)
        elif self.mode == 'approach':
            velocity = self.approach_velocity(measured, actual_heading, tilt, speed, now)
            self.reference = h.HoldReference(measured)
        elif self.mode == 'align':
            delta = np.asarray(self.selected['stance_xy'])-measured[:2]
            velocity = alignment_velocity(delta, actual_heading, self.alignment_radius, self.alignment_minimum_speed)
            self.heading = self.selected['stance_yaw']
            self.reference = h.HoldReference(measured)
            ready = (np.linalg.norm(delta) < self.alignment_radius and abs(h.wrap(actual_heading-self.heading)) < .13
                and speed < .12 and tilt < 15 and np.all(i.floor_force(d, self.floor) > 20))
            self.dwell = self.dwell+.02 if ready else 0.
            if self.dwell >= .4:
                self.approach = None
                self.reference = h.HoldReference(measured)
                self.entry_start = measured.copy()
                self.entry_goal = measured.copy() if self.selected['kind'] in ('drawer','object') else self.source.at(9.8)[0][0]
                template = self.instance_entry_templates.get(self.selected['id'], self.drawer_entry_template)
                if self.selected['kind'] == 'drawer' and template is not None:
                    self.entry_goal = template.copy()
                    adjustment = Rotation.from_euler('z', self.selected['stance_yaw']-h.yaw(self.entry_goal[3:7]))
                    self.entry_goal[3:7] = (adjustment*Rotation.from_quat(self.entry_goal[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
                self.entry_goal[:2] = self.selected['stance_xy']
                if self.enforce_torso:
                    self.entry_goal[self.robot.joint('waist_pitch_joint').qposadr[0]] = self.torso_target
                if self.selected['kind'] in ('drawer','object'):
                    self.entry_goal[self.robot.joint('waist_yaw_joint').qposadr[0]] = self.c.get('drawer_waist_yaw', .5)
                    # The inactive arm uses the configured rest throughout the
                    # interaction. Give the tracker and entry gate that same
                    # posture instead of an obsolete template arm position.
                    for name, target in self.arm_rest.items():
                        if name.startswith('left_'):
                            self.entry_goal[self.robot.joint(name).qposadr[0]] = target
                self.entry_root = self.entry_goal[:3].copy()
                self.scale.feedback = self.params.get('root_reference_feedback')
                self.change('enter_skill', 'Settling under ScaleBFM')
            elif mode_age > 12:
                self.heading = actual_heading
                self.approach = None
                self.change('walking', 'Could not align; steer closer and try again')
        elif self.mode == 'enter_skill':
            weight = h.smooth(mode_age/.6)
            u = h.smooth(mode_age/1.5)
            self.reference.q = (1-u)*self.entry_start+u*self.entry_goal
            ra, rb = Rotation.from_quat(self.entry_start[[4, 5, 6, 3]]), Rotation.from_quat(self.entry_goal[[4, 5, 6, 3]])
            self.reference.q[3:7] = (ra*Rotation.from_rotvec((ra.inv()*rb).as_rotvec()*u)).as_quat()[[3, 0, 1, 2]]
            waist = self.robot.joint('waist_yaw_joint').qposadr[0]
            ready = (mode_age > 2 and speed < .12 and tilt < 15 and
                abs(measured[waist]-self.entry_goal[waist]) < .1 and
                np.linalg.norm(measured[:2]-self.entry_root[:2]) < self.c.get('drawer_entry_tolerance_m', .015)
                and np.all(i.floor_force(d, self.floor) > 20))
            if self.selected['kind'] == 'fridge':
                ready = (mode_age > 2 and speed < .1 and tilt < 15 and
                    np.linalg.norm(measured[:2]-self.entry_root[:2]) < .02 and
                    abs(h.wrap(actual_heading-h.yaw(self.entry_goal[3:7]))) <= np.radians(10) and
                    np.sqrt(np.mean((measured[self.i.rq[self.bi]]-self.entry_goal[self.i.rq[self.bi]])**2)) < .15
                    and np.all(i.floor_force(d, self.floor) > 20))
            elif self.selected['kind'] == 'drawer' and (self.selected['id'] in self.instance_entry_templates or self.drawer_entry_template is not None):
                ready = (ready and abs(measured[2]-self.entry_goal[2]) < .03 and
                    abs(h.wrap(actual_heading-self.selected['stance_yaw'])) < .13 and
                    np.sqrt(np.mean((measured[self.i.rq[self.bi]]-self.entry_goal[self.i.rq[self.bi]])**2)) < .1)
            self.dwell = self.dwell+.02 if ready else 0.
            if self.dwell >= .5:
                current = self.candidate(self.selected['id'])
                if (current is None or current['action'] != self.selected['action'] or
                        not (current['qualified'] or self.dev) or
                        self.target_moved(current,self.selected)):
                    self.failed = 'Selected mechanism changed during entry'
                    self.begin_handback(False)
                    return
                factory = (self.draw_api.DrawerSkill if self.selected['kind'] == 'drawer' else
                    self.fridge_api.FridgeAdapter if self.selected['kind'] == 'fridge' else None)
                if self.selected['kind'] == 'object':
                    if self.selected['action'] == 'place':
                        self.skill = self.mission.skill
                        self.skill.begin_place(self.selected,now)
                        self.mission.placing = True
                    else:
                        self.skill = self.object_api.ObjectSkill(m,d,self.robot,i,self.selected,self.draw_api)
                        self.skill.start(now)
                        self.mission.skill = self.skill
                        self.mission.begin_grasp()
                elif self.selected['kind'] == 'fridge' and self.prepared_fridge is not None:
                    self.skill, self.prepared_fridge = self.prepared_fridge, None
                else:
                    if self.selected.get('action') == 'close':
                        self.skill = self.close_api.make_skill(self.draw_api, m, d, self.robot, i, self.selected)
                    else:
                        self.skill = factory(m, d, self.robot, i, self.selected)
                if self.selected['kind'] != 'object':
                    self.skill.start(now)
                if self.selected['kind'] == 'drawer':
                    self.skill.stance = self.reference.q.copy()
                    self.skill.reference_q = self.skill.stance.copy()
                    self.skill.waist_target = float(self.skill.stance[self.skill.waist_q])
                self.skill.update(now)
                self.reference = self.skill
                self.task_state = self.skill.state()
                verb = {'open':'Opening ','close':'Closing ','grasp':'Picking up ','place':'Placing in '}
                self.change('interaction_skill', verb[self.selected['action']]+self.selected['name'])
            elif mode_age > 20:
                self.failed = 'ScaleBFM could not establish entry stance'
                self.begin_handback(False)
        elif self.mode == 'interaction_skill':
            weight = 1.
            self.skill.update(now)
            if mode_age > 30:
                self.failed = 'Skill exceeded its bounded release/retraction deadline'
                self.paused = True
                self.change('paused', self.failed)
                return
        elif self.mode == 'handback':
            weight = 1-h.smooth(mode_age/.6)
        object_controller = self.mission.skill if self.mission is not None else None
        if object_controller is not None and self.mode != 'interaction_skill':
            if object_controller.phase in ('carry','carry_blend'):
                object_controller.update_carry(now)
            else:
                object_controller.update(now)
        if object_controller is not None and object_controller.requires_pause:
            self.failed = self.mission.failure = object_controller.failure
            self.paused = True
            self.task_state = object_controller.state()
            self.change('paused',self.failed)
            return
        if self.mission is not None and self.mission.holding and self.mode == 'walking' and self.navigator is not None:
            # Free carrying remains physical. Check the measured robot AND soda
            # sweep before manual translation/turning; no collision mask changes.
            ca,sa=np.cos(actual_heading),np.sin(actual_heading)
            ahead=measured[:2]+np.array([[ca,-sa],[sa,ca]])@velocity*.4
            if not self.navigator.corridor_clear(d,measured[:2],ahead,actual_heading,self.heading):
                velocity[:]=0.
                self.heading=actual_heading
                self.message='There is not enough room to carry the soda that way.'
        if (self.mode == 'interaction_skill' and self.selected['kind'] == 'drawer'
                and self.drawer_ik_limits is not None):
            error = np.asarray(self.skill.arm.error)
            if not np.isfinite(error).all() or np.any(error > self.drawer_ik_limits):
                # A bounded IK result can still miss its task-space goal.
                # Do not execute that off-goal arm posture or continue cleanup.
                # This checks private planning; actual contacts remain checked
                # independently after every physics step.
                self.failed = 'Drawer arm cannot meet the planned wrist pose within its declared error limits'
                self.skill.failure = self.failed
                self.skill.success = False
                self.skill.release_pending = False
                self.task_state = self.skill.state()
                # Preserve invalid planner diagnostics explicitly in JSON;
                # never repair the private solution or actual robot state.
                for key in ('ik_error', 'pregrasp_plan_error', 'pregrasp_current_plan_error'):
                    values = np.asarray(self.task_state[key], float)
                    invalid = ~np.isfinite(values)
                    self.task_state[key] = [None if bad else float(value)
                                           for value, bad in zip(values, invalid)]
                    self.task_state[key+'_invalid_components'] = invalid.tolist()
                self.paused = True
                self.change('paused', self.failed, wrist_error_m_rad=self.task_state['ik_error'],
                            wrist_error_invalid_components=self.task_state['ik_error_invalid_components'],
                            wrist_error_limits_m_rad=self.drawer_ik_limits.tolist(),
                            task_phase=self.skill.phase)
                return
        amo_result = self.amo.step(m, d, float(velocity[0]), float(velocity[1]), self.heading,
                                  upper_body_targets=self.upper, dt=.02)
        atarget = i.source_targets(self.posture)
        akp, akd, limits = self.base_kp.copy(), self.base_kd.copy(), i.limits.copy()
        for name, value in amo_result['joint_targets'].items():
            k = i.index[name]
            atarget[k] = value
            akp[k], akd[k] = amo_result['kp'][name], amo_result['kd'][name]
            cap = float(amo_result['torque_limits'][name])
            limits[k] = max(limits[k, 0], -cap), min(limits[k, 1], cap)
        atarget = np.clip(atarget, i.ranges[:, 0], i.ranges[:, 1])
        starget, skp, skd = atarget.copy(), self.base_kp.copy(), self.base_kd.copy()
        if self.mode in ('align', 'enter_skill', 'interaction_skill', 'handback'):
            raw, _ = self.scale.infer(self.reference, now, d.qpos[self.bq], d.qvel[self.bv],
                measured[3:7], d.qvel[i.rootv+3:i.rootv+6], False)
            starget = i.source_targets(self.reference.at(now)[0][0])
            starget[self.bi] = raw
            skp[self.bi], skd[self.bi] = self.params['kps'], self.params['kds']
            starget = np.clip(starget, i.ranges[:, 0], i.ranges[:, 1])
        if weight >= 1.:
            limits = i.limits.copy()
        category_count = np.zeros(4, int)
        category_force, category_depth = np.zeros(4), np.zeros(4)
        contact_violation = None
        external = 0.
        control_phase = self.mode
        for substep in range(10):
            at = akp*(atarget-d.qpos[i.q])-akd*d.qvel[i.v]
            st = skp*(starget-d.qpos[i.q])-skd*d.qvel[i.v]
            if self.mode in ('enter_skill', 'interaction_skill') and self.selected['kind'] in ('drawer','object'):
                wi = i.index['waist_yaw_joint']
                wq = self.robot.joint('waist_yaw_joint').qposadr[0]
                target = self.reference.at(now)[0][0, wq]
                st[wi] = 60.*(target-d.qpos[i.q[wi]])-3.*d.qvel[i.v[wi]]+d.qfrc_bias[i.v[wi]]
            if self.mode == 'interaction_skill' and self.selected['kind'] != 'object':
                aa, arm_tau, fa, finger_tau = self.skill.motor_overrides()
                arm_weight = getattr(self.skill, 'arm_weight', 1.)
                st[self.skill.arm.idx] = (1-arm_weight)*st[self.skill.arm.idx]+arm_weight*arm_tau
                st[self.skill.fi] = finger_tau
            motor_tau = (1-weight)*at+weight*st
            if object_controller is not None:
                _,arm_tau,_,finger_tau=object_controller.motor_overrides()
                if object_controller.requires_pause:
                    self.failed=self.mission.failure=object_controller.failure
                    self.paused=True
                    self.change('paused',self.failed)
                    return
                motor_tau[object_controller.arm.idx]=arm_tau
                motor_tau[object_controller.fi]=finger_tau
            for name, target in self.arm_rest.items():
                if name.startswith('left_') and (self.selected is None or self.selected['kind'] in ('drawer','object')):
                    ai = i.index[name]
                    motor_tau[ai] = 60.*(target-d.qpos[i.q[ai]])-3.*d.qvel[i.v[ai]]+d.qfrc_bias[i.v[ai]]
            if self.enforce_torso:
                ti = self.torso_motor_index
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.torso, self.torso_velocity, 1)
                current_pitch, current_roll, _ = self.torso_angles()
                required_q = np.clip(d.qpos[i.q[ti]]+self.torso_target-current_pitch,
                                     i.ranges[ti, 0], i.ranges[ti, 1])
                motor_tau[ti] = (160.*(required_q-d.qpos[i.q[ti]])-
                                 10.*self.torso_velocity[1]+d.qfrc_bias[i.v[ti]])
                ri = self.torso_roll_motor_index
                required_roll = np.clip(d.qpos[i.q[ri]]-current_roll, i.ranges[ri, 0], i.ranges[ri, 1])
                if self.mode not in ('walking', 'planning', 'approach'):
                    motor_tau[ri] = (160.*(required_roll-d.qpos[i.q[ri]])-
                                     10.*self.torso_velocity[0]+d.qfrc_bias[i.v[ri]])
            d.ctrl[i.motors] = np.clip(motor_tau, limits[:, 0], limits[:, 1])
            external = max(external, float(np.abs(d.qfrc_applied).max()), float(np.abs(d.xfrc_applied).max()))
            mujoco.mj_step(m, d)
            pitch, torso_roll, pelvis_pitch = self.torso_angles()
            self.minimum_torso_pitch = min(self.minimum_torso_pitch, pitch)
            self.minimum_pelvis_heading_pitch = min(self.minimum_pelvis_heading_pitch, pelvis_pitch)
            if self.enforce_torso and min(pitch, pelvis_pitch) < -1e-6:
                self.torso_violations.append({'time': float(d.time), 'pitch_degrees': float(np.degrees(pitch)),
                    'pelvis_heading_pitch_degrees': float(np.degrees(pelvis_pitch)), 'mode': self.mode})
            for key, value in {'time': d.time, 'torso_pitch_rad': pitch, 'mode': self.mode,
                               'torso_roll_rad': torso_roll, 'pelvis_heading_torso_pitch_rad': pelvis_pitch,
                               'root_quaternion_wxyz': d.qpos[i.rootq+3:i.rootq+7].copy(),
                               'waist_yaw_roll_pitch_rad': d.qpos[self.waist_q_indices].copy(),
                               'attempt_id': self.attempt_id, 'waist_pitch_torque': float(d.ctrl[i.motors[self.torso_motor_index]])}.items():
                self.physics_rows[key].append(value)
            if self.mission is not None:
                observed=self.mission.observe(d.time)
                if object_controller is not None:
                    self.task_state=object_controller.observe(d.time)
                    if object_controller.requires_pause:
                        self.failed=self.mission.failure=object_controller.failure
                        self.paused=True
                        self.change('paused',self.failed)
                for key,value in {**observed,'mode':self.mode,'attempt_id':self.attempt_id,
                    'skill_phase':object_controller.phase if object_controller else 'none',
                    'external':external,
                    'arm_ik_error':object_controller.arm.error.copy() if object_controller else np.zeros(2),
                    'finger_target':object_controller.finger_target.copy() if object_controller else np.zeros(7)}.items():
                    self.object_rows[key].append(value)
            if (self.mode in ('interaction_skill', 'handback') and self.skill is not None
                    and self.selected['kind'] != 'object'):
                self.task_state = self.skill.observe(d.time)
                for key, value in {'time': d.time, 'phase': self.skill.phase, 'joint_position': self.task_state['joint_position'],
                    'velocity': self.task_state['joint_velocity'], 'contact_N': self.task_state['contact_N'],
                    'digits': [n in self.skill.contact_digits for n in ('thumb_2', 'index_1', 'middle_1')],
                    'open_dwell': self.skill.open_dwell, 'clear_dwell': self.skill.clear_dwell,
                    'attempt_id': self.attempt_id, 'action': self.selected.get('action', 'open'),
                    'closed_dwell': float(getattr(self.skill, 'closed_dwell', 0.)),
                    'hand_closing_travel_m': float(getattr(self.skill, 'hand_closing_travel', 0.)),
                    'hand_closing_work_J': float(getattr(self.skill, 'hand_closing_work', 0.)),
                    'other_robot_closing_work_J': float(getattr(self.skill, 'other_closing_work', 0.)),
                    'hand_inward_force_N': float(getattr(self.skill, 'push_axis_force', 0.)),
                    'other_robot_inward_force_N': float(getattr(self.skill, 'other_push_axis_force', 0.)),
                    'raw_digit_force_N': getattr(self.skill, 'digit_force_raw', np.zeros(3)).copy(),
                    'digit_normals_world': getattr(self.skill, 'digit_normals', np.zeros((3, 3))).copy(),
                    'hook_index_correction_rad': getattr(self.skill, 'hook_digit_correction', np.zeros(2)).copy(),
                    'arm_integral_torque_Nm': getattr(self.skill, 'arm_integral', np.zeros(7)).copy(),
                    'finger_integral_torque_Nm': getattr(self.skill, 'finger_integral', np.zeros(7)).copy(),
                    # I arrays are stored controller state. Only this mode
                    # includes them in the motor request before torque clipping.
                    'integral_motor_override_active': (self.mode == 'interaction_skill' and
                        bool(getattr(self.skill, 'integral_control', False))),
                    'loaded_pair_dwell': float(getattr(self.skill, 'loaded_pair_dwell', 0.)),
                    'opposed': bool(getattr(self.skill, 'opposed', False)),
                    'paired_open_travel_m': float(getattr(self.skill, 'paired_open_travel', 0.)),
                    'hooked_open_travel_m': float(getattr(self.skill, 'hooked_open_travel', 0.)),
                    'distal_axis_force_N': float(getattr(self.skill, 'distal_axis_force', 0.)),
                    'other_robot_axis_force_N': float(getattr(self.skill, 'other_axis_force', 0.)),
                    'distal_opening_work_J': float(getattr(self.skill, 'distal_opening_work', 0.)),
                    'other_robot_opening_work_J': float(getattr(self.skill, 'other_opening_work', 0.)),
                    'distal_signed_work_J': float(getattr(self.skill, 'distal_signed_work', 0.)),
                    'other_robot_signed_work_J': float(getattr(self.skill, 'other_signed_work', 0.)),
                    'kind': self.selected['kind'],
                    'external': external}.items():
                    self.task_rows[key].append(value)
            for index in range(d.ncon):
                con = d.contact[index]
                g1, g2 = con.geom1, con.geom2
                soda_contact=self.mission is not None and self.mission.geom in (g1,g2)
                if not (self.robot_geoms[g1] or self.robot_geoms[g2] or soda_contact):
                    continue
                bodies = (self.geom_bodies[g1], self.geom_bodies[g2])
                floor = g1 in self.floor or g2 in self.floor
                cat = (0 if self.foot_geoms[g1] or self.foot_geoms[g2] else 1) if floor else (2 if self.robot_geoms[g1] and self.robot_geoms[g2] else 3)
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, index, f)
                category_count[cat] += 1
                category_force[cat] = max(category_force[cat], f[0])
                category_depth[cat] = min(category_depth[cat], con.dist)
                if cat != 0:
                    self.contacts.write(json.dumps({'time': float(d.time), 'phase': self.mode,
                        'body1': m.body(bodies[0]).name, 'body2': m.body(bodies[1]).name,
                        'geom1': m.geom(con.geom1).name, 'geom2': m.geom(con.geom2).name,
                        'normal_N': float(f[0]), 'distance_m': float(con.dist),
                        'normal_world_geom1_to_geom2': con.frame[:3].tolist(),
                        'force_world_on_geom2': (con.frame.reshape(3, 3).T@f[:3]).tolist(),
                        'position_world': con.pos.tolist(), 'force_local': f.tolist()})+'\n')
                    guarded=control_phase in ('planning', 'approach', 'align', 'enter_skill', 'interaction_skill', 'handback')
                    guarded=guarded or (self.mission is not None and self.mission.holding)
                    if guarded and con.dist < -.003:
                        contact_violation = {'time': float(d.time), 'distance_m': float(con.dist),
                            'body1': m.body(bodies[0]).name, 'body2': m.body(bodies[1]).name}
            if contact_violation is not None or self.paused:
                break
        q, heading, tilt, speed = self.actual()
        self.peak_tilt = max(self.peak_tilt, tilt)
        if contact_violation is not None:
            self.failed = 'Interaction exceeded the 3 mm non-foot collision limit'
            self.paused = True
            if self.skill is not None:
                self.skill.failure = self.failed
                if self.selected['kind'] == 'drawer':
                    self.skill.success = False
                    self.skill.release_pending = False
                self.task_state = {**self.skill.state(), 'success': False}
            if self.attempt_result is not None:
                self.attempt_result.update(success=False, failure=self.failed)
            self.change('paused', self.failed, contact=contact_violation)
        if self.mode == 'interaction_skill' and self.skill.done:
            self.attempt_result = {'instance': self.selected['id'], 'attempt_id': self.attempt_id,
                'time': float(d.time), 'action': self.selected.get('action', 'open'), **self.skill.state(), 'success': False, 'stable_amo_handback': False}
            self.results.append(self.attempt_result)
            handback_ready=(self.skill.handback_ready() if self.selected['kind']=='object' else self.skill.clear_dwell>=.3)
            if not handback_ready:
                self.failed = self.skill.failure or 'Contact-free handback was not established'
                self.paused = True
                self.change('paused', self.failed)
            else:
                if self.selected['kind']=='object' and self.selected['action']=='grasp' and self.skill.success:
                    if not self.mission.lifted:
                        self.failed=self.mission.failure='Measured soda lift did not satisfy the mission gate'
                        self.paused=True
                        self.change('paused',self.failed)
                        return
                    self.mission.holding=True
                self.begin_handback(self.skill.success and self.skill.failure is None)
        if self.mode == 'handback' and weight == 0.:
            stable = (speed < .1 and np.linalg.norm(d.qvel[i.rootv+3:i.rootv+6]) < .13
                and tilt < 15 and q[2] > .55 and np.all(i.floor_force(d, self.floor) > 20))
            self.dwell = self.dwell+.02 if stable else 0.
            if self.dwell >= 1.:
                self.stable_after = self.dwell
                if self.attempt_result is not None:
                    self.attempt_result['stable_amo_handback'] = True
                if self.attempt_result is not None and self.skill is not None:
                    state = self.skill.state()
                    if self.selected['kind']=='object':
                        terminal=(self.skill.handback_ready() and self.skill.retained_ok() and
                            (self.mission.lifted if self.selected['action']=='grasp' else self.mission.complete))
                        self.attempt_result['object_state']=state
                    else:
                        self.attempt_result['final_joint_position'] = float(d.qpos[self.skill.dq])
                        retained = (self.skill.retained_position_ok() if hasattr(self.skill, 'retained_position_ok') else
                                    state['joint_position'] >= self.skill.minimum_retained_position)
                        terminal = (state.get('interaction_achieved', state['opening_achieved']) and state['failure'] is None and state['clear_dwell'] >= .3
                            and state['finger_open_error_rad'] <= .18 and retained)
                    self.attempt_result['success'] = bool(terminal)
                    self.attempt_result['final_finger_open_error_rad'] = state['finger_open_error_rad']
                    self.task_state['success'] = bool(terminal)
                if self.selected['kind']=='object':
                    message=('Soda held. Walk to the blue tray.' if self.selected['action']=='grasp' else 'Soda delivered.') if terminal else 'Object interaction did not pass.'
                    if self.selected['action']=='place' or not terminal:
                        self.mission.skill=None
                    self.change('walking',message)
                else:
                    self.change('walking', (('Closed. ' if self.selected.get('action') == 'close' else 'Opened. ') if self.attempt_result and self.attempt_result['success'] else 'Interaction did not pass. ')+
                                'AMO control resumed.')
            elif round(d.time-self.entered, 9) > 8.:
                self.failed = 'AMO handback did not settle'
                self.paused = True
                self.change('paused', self.failed)
        for key, value in {'time': d.time, 'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(), 'ctrl': d.ctrl.copy(),
            'mode': self.mode, 'control_mode': control_phase, 'command': np.r_[velocity, self.heading],
            'input_sequence': command.get('seq', -1), 'input_active': active,
            'input_age_s': float(command['age_s']) if command.get('age_s') is not None else -1.,
            'input_age_valid': command.get('age_s') is not None,
            'look_heading_input': command.get('look_heading') if command.get('look_heading') is not None else 0.,
            'look_heading_valid': command.get('look_heading') is not None,
            'look_revision_input': command.get('look_revision', -1),
            'look_body_turn_held': look_key == self.look_blocked_key,
            'scale_weight': weight, 'scale_bias': self.scale.reference_bias.copy(),
            'tilt_degrees': tilt, 'foot_force': i.floor_force(d, self.floor), 'external_force': external,
            'contact_counts': category_count, 'contact_max_force': category_force, 'contact_min_distance': category_depth,
            'arm_target': self.skill.arm.target.copy() if self.skill else np.zeros(7),
            'arm_goal': self.skill.pos.copy() if self.skill else np.zeros(3),
            'arm_rotation_goal': self.skill.rot.copy() if self.skill else np.eye(3),
            'arm_ik_error': self.skill.arm.error.copy() if self.skill else np.zeros(2),
            'reach_pregrasp_plan_error': getattr(self.skill, 'pregrasp_current_error', np.zeros(2)).copy(),
            'reference_qpos': self.reference.at(now)[0][0].copy() if self.mode != 'walking' else self.posture.copy(),
            'last_physics_ncon': int(d.ncon)}.items():
            self.rows[key].append(value)
        self.ticks.append(time.perf_counter()-tic)
        if external or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or q[2] < .45 or tilt > 65:
            self.failed = 'External assistance, nonfinite state, or fall detected'
            self.paused = True
            self.change('paused', self.failed)
        if self.enforce_torso and self.torso_violations:
            self.failed = 'Backward torso lean violated the reaching/locomotion constraint'
            self.paused = True
            if self.attempt_result:
                self.attempt_result['success'] = False
                self.attempt_result['failure'] = self.failed
            self.change('paused', self.failed)
        if len(self.rows['time']) >= 1500:
            self.flush()

    def begin_handback(self, success):
        if (self.selected.get('standing_recovery', False) and
                self.lower_api is not None and not self.lower_recovery_started):
            if self.skill is None or not self.skill.done or self.skill.clear_dwell < .1:
                self.failed = self.failed or 'Lower entry did not finish with a clear, retracted hand; recovery paused'
                self.paused = True
                if self.skill is not None:
                    self.skill.success = False
                    self.skill.failure = self.failed
                if self.attempt_result is not None:
                    self.attempt_result.update(success=False, failure=self.failed)
                self.change('paused', self.failed)
                return
            self.lower_recovery_started = True
            self.recovery = self.lower_api.RecoveryReference(self)
            self.recoveries.append(self.recovery)
            if not self.recovery.native_clear or not self.recovery.fit_clear:
                self.failed = 'Standing recovery failed its private feet and collision checks'
                self.paused = True
                self.skill.success = False
                self.skill.failure = self.failed
                if self.attempt_result is not None:
                    self.attempt_result.update(success=False, failure=self.failed)
                self.change('paused', self.failed)
                return
            self.skill = self.recovery
            self.reference = self.recovery
            self.change('interaction_skill', 'Standing up with hands clear', standing_recovery=True, attempt_id=self.attempt_id)
            return
        if self.attempt_result is None:
            self.attempt_result = {'instance': self.selected['id'], 'attempt_id': self.attempt_id,
                'success': False, 'failure': self.failed or 'Skill did not start', 'phase': self.mode,
                'time': float(self.data.time), 'stable_amo_handback': False}
            self.results.append(self.attempt_result)
        q = self.i.measured(self.data)
        self.posture = q.copy()
        self.upper = {n: float(q[self.robot.joint(n).qposadr[0]]) for n in self.amo_api.ARM_JOINT_NAMES}
        self.heading = self.h.yaw(q[3:7])
        q[:3] -= self.scale.reference_bias
        self.scale.feedback = None
        self.reference = self.h.HoldReference(q)
        self.change('handback', 'Returning control to AMO', opening_success=bool(success),
                    reference_bias_frozen=self.scale.reference_bias.tolist())

    def status(self, wall_seconds=0.):
        q, heading, tilt, speed = self.actual()
        return {'mode': self.mode, 'sim_time': float(self.data.time),
            'real_time_factor': float(self.data.time/max(wall_seconds, 1e-6)), 'message': self.message,
            'interaction_feedback': self.interaction_feedback,
            'candidate': self.candidate(), 'candidates': self.candidates(), 'controller_ms': float(np.median(self.ticks[-100:])*1000) if self.ticks else 0.,
            'robot_state': {'position': q[:3].tolist(), 'heading': heading, 'tilt': tilt, 'speed': speed},
            'torso_pitch_degrees': float(np.degrees(self.torso_pitch())),
            'drawer_fraction': self.task_state['joint_position']/self.selected['range_m'] if self.task_state and self.selected['kind'] != 'object' else None,
            'mission':self.mission.status() if self.mission is not None else None,
            'task': self.task_state, 'paused': self.paused,
            'scope': 'Live simulated torque control; nominal geometry/materials. Space stops walking; active skills finish their release.'}

    def flush(self):
        for rows, chunks, prefix in ((self.rows, self.chunks, 'states'), (self.task_rows, self.task_chunks, 'task'),
                                    (self.physics_rows, self.physics_chunks, 'physics'),
                                    (self.object_rows,self.object_chunks,'object')):
            if not rows or not rows['time']:
                continue
            path = self.out/f'{prefix}_{len(chunks):04d}.npz'
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in rows.items()})
            chunks.append({'path': path.name, 'sha256': self.h.sha(path), 'samples': len(rows['time'])})
            rows.clear()
        self.contacts.flush()

    def finish(self, wall_seconds):
        if self.navigation_pool is not None:
            self.navigation_pool.shutdown(wait=True, cancel_futures=True)
        for recovery in self.recoveries:
            recovery.save_feedback()
        self.flush()
        self.contacts.close()
        self.h.save(self.out/'events.json', self.events)
        self.h.save(self.out/'results.json', self.results)
        if self.mission is not None:
            self.h.save(self.out/'mission.json',{'status':self.mission.status(),'events':self.mission.events})
        # Headless guards can stop between periodic status writes. Persist the
        # final mode and task diagnostics even when no further tick will run.
        self.h.save(self.out/'status.json', self.status(wall_seconds))
        receipt = {'schema': 'g1-live-session/v1', 'produced_at': datetime.now(timezone.utc).isoformat(),
            'simulated_seconds': float(self.data.time), 'wall_seconds': wall_seconds,
            'real_time_factor': float(self.data.time/max(wall_seconds, 1e-9)), 'failure': self.failed,
            'paused': self.paused, 'peak_tilt_degrees': self.peak_tilt, 'results': self.results,
            'control_ms_median': float(np.median(self.ticks)*1000) if self.ticks else None,
            'control_ms_p95': float(np.percentile(self.ticks, 95)*1000) if self.ticks else None,
            'inputs_sha256': self.h.sha(self.out/'inputs.json'), 'state_chunks': self.chunks, 'task_chunks': self.task_chunks,
            'physics_chunks': self.physics_chunks, 'minimum_torso_pitch_degrees': float(np.degrees(self.minimum_torso_pitch)),
            'object_chunks':self.object_chunks,
            'mission':self.mission.status() if self.mission is not None else None,
            'minimum_pelvis_heading_torso_pitch_degrees': float(np.degrees(self.minimum_pelvis_heading_pitch)),
            'torso_target_degrees': float(np.degrees(self.torso_target)), 'torso_constraint_enabled': self.enforce_torso,
            'torso_violations': self.torso_violations,
            'standing_recoveries': [{'attempt_id': recovery.attempt_id, 'finished': recovery.finished,
                'files': {recovery.artifact_prefix+suffix: self.h.sha(self.out/(recovery.artifact_prefix+suffix))
                          for suffix in ('_reference.json', '_reference.npz', '_feedback.json')}} for recovery in self.recoveries],
            'contacts_sha256': self.h.sha(self.out/'contacts.jsonl'),
            'status_sha256': self.h.sha(self.out/'status.json'),
            'events_sha256': self.h.sha(self.out/'events.json'), 'results_sha256': self.h.sha(self.out/'results.json'),
            'scope': 'Actual torque physics session; per-instance qualification and live transport are evaluated separately'}
        self.h.save(self.out/'receipt.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-instance')
    p.add_argument('--auto-interact', action='store_true')
    p.add_argument('--duration', type=float, default=0., help='Simulation seconds; zero means until owned shutdown')
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--port', type=int, default=8782)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    render_hz = config.get('render_hz', 30.)
    if type(render_hz) not in (int, float) or not 10 <= render_hz <= 50:
        raise ValueError('Render cadence must be finite within 10..50 Hz')
    render_period, next_render = 1./render_hz, 0.
    args.output.mkdir(parents=True, exist_ok=False)
    engine = Engine(config, args.output, initial_instance=args.initial_instance)
    engine.h.save(args.output/'launch_arguments.json', vars(args)|{'config': str(args.config), 'output': str(args.output)})
    transport = None
    if args.serve:
        transport = load_module(config['transport_adapter'], 'live_transport').LiveTransport(
            port=args.port, deadman_s=.35, ui_dir=config.get('transport_ui_dir'), require_view=bool(config.get('game_view_adapter')),
            steer_with_look=bool(config.get('look_steers_robot')))
        transport.publish(engine.status(0.), None)
        transport.start()
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    renderer = None
    if args.render:
        context = mp.get_context('spawn')
        incoming, outgoing, render_stop = context.Queue(1), context.Queue(1), context.Event()
        incoming.cancel_join_thread()
        renderer = context.Process(target=render_worker, args=(str(engine.compiled), incoming, outgoing, render_stop, config.get('game_view_adapter')))
        renderer.start()
    begun = time.monotonic()
    tick = 0
    last_status = 0.
    fired = False
    try:
        while not stopped and (args.duration <= 0 or engine.data.time < args.duration):
            command = transport.poll_command() if transport else {'active': False, 'interact': False, 'seq': tick, 'age_s': 0.}
            if args.auto_interact and engine.data.time >= 2. and not fired:
                automatic = engine.candidate(args.initial_instance)
                command = {**command, 'interact': True, 'interaction_candidate_id': args.initial_instance,
                           'interaction_action': automatic['action'] if automatic else None}
                fired = True
            if not engine.paused:
                engine.step(command)
                tick += 1
            elif not args.serve:
                break
            elapsed = time.monotonic()-begun
            if elapsed-last_status >= .1:
                status = engine.status(elapsed)
                engine.h.save(args.output/'status.json', status)
                if transport:
                    transport.publish(status, None)
                last_status = elapsed
            if renderer and elapsed >= next_render:
                if config.get('game_view_adapter'):
                    view = transport.view_request() if transport else {'look': {'yaw': 0., 'pitch': -.15, 'revision': 0}, 'view_session': 'offline'}
                    put_latest(incoming, {**view, 'frame_sim_time': float(engine.data.time), 'qpos': engine.data.qpos.copy(),
                        'heading': engine.actual()[1], 'captured_at': time.monotonic(),
                        'candidates': engine.candidates(include_distant=True)})
                else:
                    put_latest(incoming, (float(engine.data.time), engine.data.qpos.copy(), engine.actual()[1]))
                # Keep cadence across physics ticks instead of quantizing each
                # period upward. Missed frames are skipped, never queued.
                next_render += render_period
                if next_render <= elapsed:
                    next_render = elapsed+render_period
            if renderer:
                try:
                    frame = outgoing.get_nowait()
                    if transport:
                        if isinstance(frame, dict):
                            transport.publish_view(frame)
                        else:
                            frame_time, jpeg = frame
                            transport.publish({**engine.status(elapsed), 'frame_sim_time': frame_time}, jpeg)
                except queue.Empty:
                    pass
            if args.realtime:
                time.sleep(max(0., min(.02, tick*.02-(time.monotonic()-begun))))
            if engine.paused:
                time.sleep(.02)
    finally:
        if transport:
            transport.stop()
        if renderer:
            render_stop.set()
            renderer.join(timeout=5.)
            if renderer.is_alive():
                renderer.terminate()
                renderer.join(timeout=2.)
        engine.finish(time.monotonic()-begun)


if __name__ == '__main__':
    main()
