#!/usr/bin/env python3
"""Continuous AMO travel, ScaleBFM fridge skill and AMO handback experiment.

One free-base MuJoCo state and original torque motors. The automatic action
request is an experiment input; this entrypoint is not a hardware controller.
"""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def yaw(q):
    r = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
    return float(np.arctan2(r[1, 0], r[0, 0]))


def wrap(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


def smooth(value):
    u = float(np.clip(value, 0., 1.))
    return u ** 3 * (10 + u * (-15 + 6 * u))


class HoldReference:
    def __init__(self, q, duration=180.):
        self.q = np.asarray(q, float).copy()
        self.t = np.array([0., duration])

    def at(self, times):
        count = len(np.atleast_1d(times))
        return np.tile(self.q, (count, 1)), np.zeros((count, len(self.q) - 7))


class RobotInterface:
    def __init__(self, model, robot):
        self.model, self.robot = model, robot
        self.names = [robot.joint(j).name for j in range(robot.njnt) if robot.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.index = {name: i for i, name in enumerate(self.names)}
        joints = [model.joint(name).id for name in self.names]
        self.q, self.v = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.rq = np.array([robot.joint(name).qposadr[0] for name in self.names])
        self.ranges = model.jnt_range[joints].copy()
        motors = []
        for name, joint in zip(self.names, joints):
            ids = np.flatnonzero((model.actuator_trnid[:, 0] == joint)
                                & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            source_ids = np.flatnonzero((robot.actuator_trnid[:, 0] == robot.joint(name).id)
                                       & (robot.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            if len(ids) != 1 or len(source_ids) != 1:
                raise ValueError('Every original joint must have exactly one motor')
            i, j = int(ids[0]), int(source_ids[0])
            if (model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED or model.actuator_gainprm[i, 0] != 1
                    or model.actuator_biastype[i] != 0 or model.actuator_dyntype[i] != 0
                    or not model.actuator_ctrllimited[i] or not np.array_equal(model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
                    or not np.array_equal(model.actuator_ctrlrange[i], robot.actuator_ctrlrange[j])):
                raise ValueError('Original direct torque motors and bounds must be preserved')
            motors.append(i)
        self.motors = np.array(motors)
        self.limits = model.actuator_ctrlrange[self.motors].copy()
        self.rootq = int(model.joint('floating_base_joint').qposadr[0])
        self.rootv = int(model.joint('floating_base_joint').dofadr[0])
        if len(self.motors) != 43 or model.nu != 43 or model.neq or model.nmocap or model.joint('floating_base_joint').type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Require original 43 motors, free base, no constraints or mocap')
        self.feet = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
        self.robot_bodies = {model.body(robot.body(i).name).id for i in range(1, robot.nbody)}

    def measured(self, data):
        result = np.empty(self.robot.nq)
        result[:7] = data.qpos[self.rootq:self.rootq + 7]
        result[self.rq] = data.qpos[self.q]
        return result

    def source_targets(self, q):
        return np.asarray(q)[self.rq].copy()

    def floor_force(self, data, floor):
        result = np.zeros(2)
        for i in range(data.ncon):
            con = data.contact[i]
            if con.geom1 not in floor and con.geom2 not in floor:
                continue
            bodies = [int(self.model.geom_bodyid[g]) for g in (con.geom1, con.geom2)]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, force)
            for k, foot in enumerate(self.feet):
                if foot in bodies:
                    result[k] += max(0., force[0])
        return result


def run_episode(config, output, model, data, robot, interface, source_ref, params, scale, amo,
                skill, entry, goal, travel_posture, upper, base_kp, base_kd, bq, bv, bindex, floor, begun):
    rows = collections.defaultdict(list)
    skill_rows = collections.defaultdict(list)
    events, contact_rows = [], []
    phase = 'amo_travel'
    phase_entered = 0.
    approach_dwell = 0.
    bridge_started = None
    handback_started = None
    handback_heading = None
    settled_seconds = 0.
    task_opened = False
    failure = None
    fall = None
    final_success = False
    skill_state = None
    ticks = []
    source_motor_limits = interface.limits.copy()
    door_id = model.body('task_fridge_door').id
    door_q = int(model.joint('task_fridge_hinge').qposadr[0])
    wrists = [model.body(side + '_wrist_yaw_link').id for side in ('left', 'right')]
    measured_fk = mujoco.MjData(model)

    def change(next_phase, now, **detail):
        nonlocal phase, phase_entered
        events.append({'time': float(now), 'from': phase, 'to': next_phase, **detail})
        phase, phase_entered = next_phase, float(now)

    try:
        for tick in range(round(config['maximum_seconds'] * 50)):
            tic = time.perf_counter()
            now = tick * .02
            measured = interface.measured(data)
            heading = yaw(measured[3:7])
            tilt = float(np.arccos(np.clip(Rotation.from_quat(measured[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1)))
            speed = float(np.linalg.norm(data.qvel[interface.rootv:interface.rootv + 2]))
            foot_forces = interface.floor_force(data, floor)
            distance = float(np.linalg.norm(measured[:2] - goal[:2]))
            facing = abs(wrap(heading - yaw(goal[3:7])))
            ready = (distance <= config['approach_xy_tolerance_m'] and facing <= config['approach_heading_tolerance_rad']
                     and speed <= .12 and tilt <= np.radians(15) and np.all(foot_forces > 20))
            if phase == 'amo_travel':
                approach_dwell = approach_dwell + .02 if now >= config['approach_reference_end'] and ready else 0.
                if approach_dwell >= .3:
                    change('request_entry_transition', now, root_xy_error_m=distance, heading_error_rad=facing)
                elif now >= config['travel_timeout_seconds']:
                    failure = 'AMO did not reach the entry preparation domain before timeout'
                    handback_heading, handback_started = heading, now
                    change('failed_amo_hold', now, reason=failure)
            if phase in ('amo_travel', 'request_entry_transition', 'scale_entry'):
                state_before = [x.copy() for x in (data.qpos, data.qvel, data.ctrl, data.qfrc_applied, data.xfrc_applied)]
                entry.before_step(now, data, allow_request=phase != 'amo_travel')
                if not all(np.array_equal(before, after) for before, after in zip(state_before,
                           (data.qpos, data.qvel, data.ctrl, data.qfrc_applied, data.xfrc_applied))):
                    raise ValueError('Reference generation changed actual robot state')
                if entry.state == 'failed':
                    failure = 'Entry reference was rejected'
                    handback_heading, handback_started = heading, now
                    change('failed_amo_hold', now, reason=failure)
                elif entry.activated is not None and phase == 'request_entry_transition':
                    bridge_started = now
                    change('scale_entry', now, reference_target_time=entry.target_time,
                           physical_state_reset=False, scale_history='Native first-state initialization at this tick')
                elif phase == 'scale_entry' and entry.entry_verified:
                    skill.start(now)
                    change('fridge_skill', now)

            control_phase = phase
            command = np.zeros(2)
            commanded_heading = handback_heading if handback_heading is not None else yaw(goal[3:7])
            active_reference = HoldReference(measured)
            reference_time = now
            scale_weight = 0.
            if phase == 'amo_travel':
                rt = min(now, config['approach_reference_end'])
                qr = source_ref.at(rt)[0][0]
                if now < config['approach_reference_end']:
                    ta, tb = max(0., rt - .01), min(config['approach_reference_end'], rt + .01)
                    qa, qb = source_ref.at([ta, tb])[0]
                    velocity = (qb[:2] - qa[:2]) / max(tb - ta, 1e-8)
                else:
                    velocity = np.zeros(2)
                world_command = velocity + config['approach_position_gain'] * (qr[:2] - measured[:2])
                c, s = np.cos(heading), np.sin(heading)
                command = np.clip(np.array([[c, s], [-s, c]]) @ world_command, [-.5, -.35], [.5, .35])
                commanded_heading = yaw(qr[3:7])
            elif phase == 'scale_entry':
                active_reference = entry
                scale_weight = smooth((now - bridge_started) / config['handoff_blend_seconds'])
            elif phase == 'fridge_skill':
                reference_time = skill.source_time(now)
                active_reference = skill.reference
                scale_weight = 1.
                skill.update(now)
                commanded_heading = yaw(active_reference.at(reference_time)[0][0, 3:7])
            elif phase == 'return_to_amo':
                active_reference = handback_reference
                scale_weight = 1 - smooth((now - handback_started) / config['handoff_blend_seconds'])
            elif phase == 'failed_amo_hold':
                scale_weight = 0.

            amo_result = amo.step(model, data, float(command[0]), float(command[1]), float(commanded_heading),
                                  upper_body_targets=upper, dt=.02)
            amo_target = interface.source_targets(travel_posture)
            amo_kp, amo_kd = base_kp.copy(), base_kd.copy()
            limits = source_motor_limits.copy()
            for name, value in amo_result['joint_targets'].items():
                i = interface.index[name]
                amo_target[i] = value
                amo_kp[i], amo_kd[i] = amo_result['kp'][name], amo_result['kd'][name]
                cap = float(amo_result['torque_limits'][name])
                if not np.isfinite(cap) or cap <= 0:
                    raise ValueError('Invalid native motor cap')
                limits[i] = [max(limits[i, 0], -cap), min(limits[i, 1], cap)]
            amo_target = np.clip(amo_target, interface.ranges[:, 0], interface.ranges[:, 1])
            scale_target = amo_target.copy()
            scale_kp, scale_kd = base_kp.copy(), base_kd.copy()
            if phase in ('scale_entry', 'fridge_skill', 'return_to_amo'):
                raw, _ = scale.infer(active_reference, reference_time, data.qpos[bq], data.qvel[bv],
                                     measured[3:7], data.qvel[interface.rootv + 3:interface.rootv + 6], False)
                scale_target = interface.source_targets(active_reference.at(reference_time)[0][0])
                scale_target[bindex] = raw
                scale_kp[bindex], scale_kd[bindex] = params['kps'], params['kds']
                scale_target = np.clip(scale_target, interface.ranges[:, 0], interface.ranges[:, 1])
            # AMO-native caps constrain travel and the blended handoff. Once
            # ScaleBFM fully owns the motors, original source caps apply.
            if scale_weight >= 1.:
                limits = source_motor_limits.copy()
            saturation = np.zeros(43)
            max_external = 0.
            categories = np.zeros(4, dtype=np.int64)
            max_force = np.zeros(4)
            min_distance = np.zeros(4)
            for substep in range(10):
                amo_tau = amo_kp * (amo_target - data.qpos[interface.q]) - amo_kd * data.qvel[interface.v]
                scale_tau = scale_kp * (scale_target - data.qpos[interface.q]) - scale_kd * data.qvel[interface.v]
                if phase == 'fridge_skill':
                    override = skill.motor_overrides(now + substep * .002, dt=.002)
                    for actuator, torque in zip(override['arm_actuator_ids'], override['arm_torque']):
                        i = int(np.flatnonzero(interface.motors == actuator)[0])
                        weight = override['arm_weight']
                        scale_tau[i] = (1 - weight) * scale_tau[i] + weight * torque
                    for actuator, torque in zip(override['hand_actuator_ids'], override['hand_torque']):
                        i = int(np.flatnonzero(interface.motors == actuator)[0])
                        scale_tau[i] = torque
                wanted = (1 - scale_weight) * amo_tau + scale_weight * scale_tau
                clipped = np.clip(wanted, limits[:, 0], limits[:, 1])
                saturation += (np.abs(wanted - clipped) > 1e-8) / 10
                data.ctrl[interface.motors] = clipped
                max_external = max(max_external, float(np.abs(data.qfrc_applied).max()), float(np.abs(data.xfrc_applied).max()))
                mujoco.mj_step(model, data)
                if phase == 'fridge_skill':
                    skill_state = skill.observe(data.time)
                    task_opened = task_opened or bool(skill_state['success_latched'])
                    for key in ('source_time', 'door_angle', 'door_velocity', 'open_hold_seconds',
                                'contact_seconds', 'distal_rail_normal_force_N', 'success_latched',
                                'handback_ready', 'hand_clear_seconds', 'hand_to_door_contact',
                                'grip_to_rail_anchor_distance_m', 'right_finger_open_error_max_rad',
                                'coverage_valid', 'external_forces_zero'):
                        skill_rows[key].append(skill_state[key])
                    skill_rows['time'].append(float(data.time))
                    skill_rows['distal_contact_mask'].append([name in skill_state['distal_rail_digits']
                        for name in ('thumb_2_link', 'index_1_link', 'middle_1_link')])
                for i in range(data.ncon):
                    con = data.contact[i]
                    bodies = [int(model.geom_bodyid[g]) for g in (con.geom1, con.geom2)]
                    if not set(bodies) & interface.robot_bodies:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    on_floor = con.geom1 in floor or con.geom2 in floor
                    category = (0 if any(b in interface.feet for b in bodies) else 1) if on_floor else (2 if all(b in interface.robot_bodies for b in bodies) else 3)
                    categories[category] += 1
                    max_force[category] = max(max_force[category], max(0., force[0]))
                    min_distance[category] = min(min_distance[category], con.dist)
                    if category != 0:
                        contact_rows.append({'time': float(data.time), 'phase': phase,
                            'geom1': model.geom(con.geom1).name, 'geom2': model.geom(con.geom2).name,
                            'body1': model.body(bodies[0]).name, 'body2': model.body(bodies[1]).name,
                            'distance_m': float(con.dist), 'normal_N': float(force[0])})
            measured_fk.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, measured_fk)
            end = now + .02
            if phase in ('amo_travel', 'request_entry_transition', 'scale_entry'):
                entry.after_step(end, data)
            if phase == 'fridge_skill':
                if skill_state.get('handback_ready'):
                    handback_started = end
                    handback_pose = interface.measured(data)
                    handback_heading = yaw(handback_pose[3:7])
                    # Keep the actually retracted arms and fingers. Replacing
                    # them with the initial travel pose can sweep the open leaf.
                    travel_posture = handback_pose.copy()
                    upper = {name: float(travel_posture[robot.joint(name).qposadr[0]]) for name in upper}
                    # The ScaleBFM wrapper adds its accumulated task bias to all
                    # world targets. Compensate this virtual hold pose and freeze
                    # that integrator so the effective target is the actual pose.
                    held = handback_pose.copy()
                    held[:3] -= scale.reference_bias
                    scale.feedback = None
                    handback_reference = HoldReference(held)
                    change('return_to_amo', end, opening_success=task_opened,
                           upper_body_ownership='Measured retracted arm/wrist/finger pose held by AMO',
                           reference_bias_frozen=scale.reference_bias.tolist(),
                           physical_state_reset=False)
                elif end - phase_entered > config['skill_timeout_seconds']:
                    failure = 'Skill could not establish safe handback before timeout'
                    change('failed_skill_hold', end, reason=failure)
            if phase in ('return_to_amo', 'failed_amo_hold') and scale_weight == 0.:
                q_check = interface.measured(data)
                up = Rotation.from_quat(q_check[[4, 5, 6, 3]]).as_matrix()[2, 2]
                stopped = (np.linalg.norm(data.qvel[interface.rootv:interface.rootv + 2]) < .1
                           and np.linalg.norm(data.qvel[interface.rootv + 3:interface.rootv + 6]) < .1
                           and abs(wrap(yaw(q_check[3:7]) - handback_heading)) <= np.radians(5)
                           and np.all(interface.floor_force(data, floor) > 20)
                           and q_check[2] >= .55 and up >= np.cos(np.radians(15))
                           and (phase == 'failed_amo_hold' or data.qpos[door_q] >= np.radians(50)))
                settled_seconds = settled_seconds + .02 if stopped else 0.
                if settled_seconds >= config['handback_hold_seconds']:
                    final_success = task_opened and failure is None
                    change('complete' if final_success else 'failed', end)
            q_measured = interface.measured(data)
            tilt_end = float(np.degrees(np.arccos(np.clip(Rotation.from_quat(q_measured[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1))))
            for key, value in {'time': data.time, 'phase': phase, 'control_phase': control_phase,
                'control_time': now, 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
                'ctrl': data.ctrl.copy(), 'command_body_velocity': command, 'command_heading': commanded_heading,
                'scale_weight': scale_weight, 'amo_joint_target': amo_target, 'scale_joint_target': scale_target,
                'scale_reference_bias': scale.reference_bias.copy(),
                'amo_kp': amo_kp, 'amo_kd': amo_kd, 'scale_kp': scale_kp, 'scale_kd': scale_kd,
                'effective_motor_limits': limits, 'source_reference_time': reference_time,
                'active_reference_qpos': active_reference.at(reference_time)[0][0],
                'root_xy_error_to_entry': q_measured[:2] - goal[:2], 'tilt_degrees': tilt_end,
                'foot_normal_force': interface.floor_force(data, floor), 'wrist_world': measured_fk.xpos[wrists].copy(),
                'door_angle_radians': data.qpos[door_q], 'saturation_fraction': saturation,
                'contact_counts_foot_floor_body_floor_self_room': categories,
                'contact_max_normal_N': max_force, 'contact_min_distance_m': min_distance,
                'max_external_force': max_external,
                'skill_opening_latched': bool(skill_state and skill_state['success_latched']),
                'skill_open_hold_seconds': float(skill_state['open_hold_seconds']) if skill_state else 0.,
                'skill_hand_clear_seconds': float(skill_state['hand_clear_seconds']) if skill_state else 0.,
                'skill_handback_ready': bool(skill_state and skill_state['handback_ready'])}.items():
                rows[key].append(value)
            rows['arm_ik_target'].append(skill.arm.target.copy() if skill.started and skill.arm.target is not None else np.zeros(7))
            rows['arm_goal_position'].append(skill.arm.goal_pos.copy() if skill.started else np.zeros(3))
            rows['arm_goal_orientation'].append(skill.arm.goal_rot.copy() if skill.started else np.eye(3))
            rows['arm_ik_position_error'].append(float(skill.arm.diagnostics.get('ik_target_position_error_m', 0.)) if skill.started else 0.)
            rows['arm_ik_orientation_error'].append(float(skill.arm.diagnostics.get('ik_target_rotation_error_rad', 0.)) if skill.started else 0.)
            ticks.append(time.perf_counter() - tic)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = 'Nonfinite physical state'
                break
            if q_measured[2] < .45 or tilt_end > 65:
                fall = {'time': float(data.time), 'root_z': float(q_measured[2]), 'tilt_degrees': tilt_end}
                break
            if phase in ('complete', 'failed', 'failed_skill_hold'):
                break
            if tick % 50 == 0:
                save(output / 'status.json', {'phase': phase, 'time': float(data.time), 'root_entry_error_m': distance,
                    'door_angle_degrees': float(np.degrees(data.qpos[door_q])), 'entry_ready': entry.entry_verified})
        else:
            failure = failure or 'Episode deadline exceeded'
    except Exception as exc:
        failure = repr(exc)
    entry_result = entry.finish()
    np.savez_compressed(output / 'rollout.npz', **{k: np.asarray(v) for k, v in rows.items()})
    np.savez_compressed(output / 'skill_trace.npz', **{k: np.asarray(v) for k, v in skill_rows.items()})
    save(output / 'events.json', events)
    with (output / 'contacts.jsonl').open('w') as stream:
        for row in contact_rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    save(output / 'skill_result.json', skill_state)
    result = {'schema': 'g1-amo-scalebfm-skill-run/v1', 'produced_at': datetime.now(timezone.utc).isoformat(),
        'success': bool(final_success and failure is None and fall is None), 'final_phase': phase,
        'failure': failure, 'fall': fall, 'samples': len(rows['time']), 'simulated_seconds': float(data.time),
        'opening_measured': task_opened, 'stable_amo_handback_seconds': settled_seconds,
        'entry': entry_result, 'wall_seconds': time.perf_counter() - begun,
        'tick_ms_median': float(np.median(ticks) * 1000) if ticks else None,
        'tick_ms_p95': float(np.percentile(ticks, 95) * 1000) if ticks else None,
        'inputs_sha256': sha(output / 'inputs.json'),
        'artifacts': {name: sha(output / name) for name in ('rollout.npz', 'skill_trace.npz', 'events.json', 'contacts.jsonl', 'skill_result.json')},
        'scope': 'Single configured AMO-to-ScaleBFM fridge skill experiment, with explicit initial state; no family-level or arbitrary-navigation reliability claim.'}
    save(output / 'receipt.json', result)
    save(output / 'status.json', result)
    print(json.dumps(result, indent=2))
    if not result['success']:
        raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if not 10 <= config['maximum_seconds'] <= 180:
        raise ValueError('Bounded episode duration is required')
    a.output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    legacy = module(config['tracker_driver'], 'hybrid_legacy_tracker')
    robot = mujoco.MjModel.from_xml_path(config['robot'])
    source_ref = legacy.Reference(Path(config['reference']), robot)
    physics_options = json.loads(Path(config['physics_options']).read_text())
    model, compiled = legacy.compile_model(Path(config['scene']), a.output, False, .002, physics_options)
    data = mujoco.MjData(model)
    interface = RobotInterface(model, robot)
    initial = source_ref.at(0)[0][0]
    data.qpos[interface.rootq:interface.rootq + 7] = initial[:7]
    data.qpos[interface.q] = initial[interface.rq]
    offset = np.asarray(config.get('initial_root_offset', [0., 0., 0.]))
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > [.3, .3, .2]):
        raise ValueError('Initial perturbation is outside the local test domain')
    data.qpos[interface.rootq:interface.rootq + 2] += offset[:2]
    if offset[2]:
        r = Rotation.from_euler('z', offset[2]) * Rotation.from_quat(initial[[4, 5, 6, 3]])
        data.qpos[interface.rootq + 3:interface.rootq + 7] = r.as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    initial_state = {'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist()}
    params = json.loads(Path(config['parameters']).read_text())
    body_names = params['body_joint_names_hardware']
    bq = np.array([model.joint(n).qposadr[0] for n in body_names])
    bv = np.array([model.joint(n).dofadr[0] for n in body_names])
    rbq = np.array([robot.joint(n).qposadr[0] for n in body_names])
    bindex = np.array([interface.index[n] for n in body_names])
    scale_api = module(config['scalebfm_adapter'], 'hybrid_scale')
    scale = scale_api.create_policy(Path(config['scalebfm_weights']), params, rbq, 2, model, data, robot, interface.rootq)
    amo_api = module(config['amo_adapter'], 'hybrid_amo_native')
    native_amo = amo_api.AMOPolicy(config['amo_weights'], device='cpu')
    gate_api = module(config['amo_command_adapter'], 'hybrid_amo_gate')
    amo = gate_api.AMOHeadingGate(native_amo, mode=config['amo_gate'])
    skill_api = module(config['skill_adapter'], 'hybrid_fridge_skill')
    skill = skill_api.FridgeOpenSkill(model, data, robot, bundle_dir=Path(config['skill_bundle']))
    transition_api = module(config['transition_adapter'], 'hybrid_transition_source')

    class RuntimeEntryReference(transition_api.TransitionReference):
        def before_step(self, now, current, *, allow_request=False):
            if self.state == 'travel' and not allow_request:
                self.history.append((float(now), self._measured(current)))
            else:
                super().before_step(now, current)

        def _activate(self, now, measured, candidate):
            # Match the new tracker reference to the CURRENT measured AMO pose.
            # This is a virtual reference, not an assignment to the physical robot.
            self.source = HoldReference(measured)
            super()._activate(now, measured, candidate)

    entry = RuntimeEntryReference(source_ref, model, data, robot, a.output, Path(config['transition_config']))
    floor = {model.geom(n).id for n in config['floor_geometries']}
    if floor != entry.floor_geoms:
        raise ValueError('Controller and entry gate must share the actual named floor')
    goal = source_ref.at(config['skill_source_start'])[0][0]
    travel_posture = source_ref.at(0)[0][0]
    upper = {name: float(travel_posture[robot.joint(name).qposadr[0]]) for name in amo_api.ARM_JOINT_NAMES}
    base_kp, base_kd = np.full(43, 5.), np.full(43, .1)
    for name in interface.names:
        if 'wrist' in name:
            base_kp[interface.index[name]], base_kd[interface.index[name]] = 20., 1.
    dependencies = legacy.model_files(Path(config['scene'])) | legacy.model_files(Path(config['robot']))
    bound = {str(a.config.resolve()): sha(a.config), str(Path(__file__).resolve()): sha(__file__)}
    for key in ('tracker_driver', 'reference', 'physics_options', 'parameters', 'scalebfm_adapter',
                'amo_adapter', 'amo_command_adapter', 'skill_adapter', 'transition_adapter', 'transition_config'):
        bound[str(Path(config[key]).resolve())] = sha(config[key])
    bound.update(scale.input_bindings)
    bound.update(native_amo.input_bindings)
    bound.update(entry.input_bindings)
    skill_manifest = Path(config['skill_bundle']) / 'manifest.json'
    bound[str(skill_manifest.resolve())] = sha(skill_manifest)
    for row in json.loads(skill_manifest.read_text())['files']:
        file = Path(config['skill_bundle']) / row['local_path']
        if sha(file) != row['sha256']:
            raise ValueError('Skill bundle changed during initialization')
        bound[str(file.resolve())] = row['sha256']
    for file in Path(config['amo_weights']).glob('*.pt'):
        bound[str(file.resolve())] = sha(file)
    save(a.output / 'inputs.json', {'schema': 'g1-hybrid-runtime-inputs/v1', 'config': config, 'inputs': bound,
        'model_dependencies': dependencies, 'compiled_xml': {'path': str(compiled), 'sha256': sha(compiled)},
        'initial_state': initial_state, 'motor_names': interface.names, 'motor_indices': interface.motors.tolist(),
        'source_ctrlrange': interface.limits.tolist(), 'physics_hz': 500, 'control_hz': 50,
        'policy_history': 'AMO continues observing/inferencing through interaction with its own predicted action history; '
            'ScaleBFM initializes native current-state history at the first bridge tick. No physical state reset at handoff.',
        'actuation': 'Original bounded torque motors; two PD torque fields blend across controller changes. '
            'Skill owns only arm/finger overrides; no root forces, joint drives or attachments.',
        'skill_binding': getattr(skill, 'input_bindings', None)})
    # The state machine and integration loop follow below.
    run_episode(config, a.output, model, data, robot, interface, source_ref, params, scale, amo,
                skill, entry, goal, travel_posture, upper, base_kp, base_kd, bq, bv, bindex, floor, begun)


if __name__ == '__main__':
    main()
