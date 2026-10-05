#!/usr/bin/env python3
"""Bounded common-embodiment controller experiments; all motion is motor-driven.

Reference controllers receive the declared pose sequence. AMO receives the
sequence's velocity/heading through an explicit goal adapter. This compares
complete controller configurations, not isolated neural-network quality.
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


def wrapped(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['amo', 'sonic', 'sonic_v11', 'scalebfm'], required=True)
    p.add_argument('--tracker', type=Path, required=True)
    p.add_argument('--robot', type=Path, required=True)
    p.add_argument('--scene', type=Path)
    p.add_argument('--floor-geom', action='append', help='Explicit collision floor geometry name; repeat for a tiled room floor')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--duration', type=float, default=20.)
    p.add_argument('--amo-adapter', type=Path)
    p.add_argument('--amo-weights', type=Path)
    p.add_argument('--position-gain', type=float, default=0.)
    p.add_argument('--initial-root-offset', type=float, nargs=3, default=[0., 0., 0.], metavar=('X_M', 'Y_M', 'YAW_RAD'))
    p.add_argument('--arm-mode', choices=['policy', 'reference_pd'], default='reference_pd')
    p.add_argument('--reference-provider', type=Path)
    p.add_argument('--provider-config', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (not np.isfinite([a.start, a.duration, a.position_gain]).all()
            or not .02 <= a.duration <= 180 or not 0 <= a.position_gain <= 1
            or abs(a.duration * 50 - round(a.duration * 50)) > 1e-7):
        raise ValueError('Finite start, positive whole 50 Hz sample count and bounded position gain required')
    if bool(a.reference_provider) != bool(a.provider_config) or (a.reference_provider and (a.controller == 'amo' or a.start != 0)):
        raise ValueError('Reference provider requires its configuration, a pose tracker and a zero reference start')
    if a.scene and not a.floor_geom:
        raise ValueError('A room trial requires explicit floor geometry names')
    offset = np.asarray(a.initial_root_offset)
    if not np.isfinite(offset).all() or np.any(np.abs(offset) > [.5, .5, .35]):
        raise ValueError('Initial pose perturbation must be finite and within the local test domain')
    a.output.mkdir(parents=True, exist_ok=False)
    start_wall = time.perf_counter()
    legacy = module(a.tracker / 'sonic_rollout.py', 'benchmark_tracker_source')
    robot = mujoco.MjModel.from_xml_path(str(a.robot))
    ref = legacy.Reference(a.reference, robot)
    if a.start < ref.t[0] - 1e-8 or a.start + a.duration > ref.t[-1] + .021:
        raise ValueError('Requested interval is outside the supplied reference')
    m, executed = legacy.compile_model(a.scene or a.robot, a.output, not bool(a.scene), .002)
    d = mujoco.MjData(m)
    measured_fk = mujoco.MjData(m)
    names = [robot.joint(j).name for j in range(robot.njnt) if robot.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    jids = np.array([m.joint(n).id for n in names])
    qids, vids = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    rqids = np.array([robot.joint(n).qposadr[0] for n in names])
    motors = []
    for jid in jids:
        ix = np.flatnonzero((m.actuator_trnid[:, 0] == jid) & (m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
        if len(ix) != 1:
            raise ValueError('Expected one motor per source joint')
        k = int(ix[0])
        if (not m.actuator_ctrllimited[k] or m.actuator_biastype[k] != 0
                or m.actuator_dyntype[k] != 0 or m.actuator_gainprm[k, 0] != 1
                or m.actuator_gaintype[k] != mujoco.mjtGain.mjGAIN_FIXED
                or not np.array_equal(m.actuator_gear[k], [1, 0, 0, 0, 0, 0])):
            raise ValueError('Expected source-limited direct torque motors')
        source_joint = robot.joint(m.joint(jid).name).id
        source_motor = np.flatnonzero((robot.actuator_trnid[:, 0] == source_joint)
                                     & (robot.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
        if len(source_motor) != 1 or not np.array_equal(m.actuator_ctrlrange[k], robot.actuator_ctrlrange[source_motor[0]]):
            raise ValueError('Compiled scene must preserve the original robot motor limits')
        motors.append(k)
    motors = np.array(motors)
    root = m.joint('floating_base_joint')
    rootq, rootv = int(root.qposadr[0]), int(root.dofadr[0])
    if m.neq or m.nmocap or root.type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('No equality constraints, mocap or fixed base permitted')
    initial = ref.at(a.start)[0][0]
    d.qpos[rootq:rootq + 7] = initial[:7]
    d.qpos[qids] = initial[rqids]
    d.qpos[rootq:rootq + 2] += offset[:2]
    if offset[2]:
        rotation = Rotation.from_euler('z', offset[2]) * Rotation.from_quat(d.qpos[rootq + np.array([4, 5, 6, 3])])
        d.qpos[rootq + 3:rootq + 7] = rotation.as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    params_path = a.tracker / ('parameters_scalebfm.json' if a.controller == 'scalebfm' else 'parameters.json')
    params = json.loads(params_path.read_text())
    body_names = params['body_joint_names_hardware']
    bq = np.array([m.joint(n).qposadr[0] for n in body_names])
    bv = np.array([m.joint(n).dofadr[0] for n in body_names])
    rbq = np.array([robot.joint(n).qposadr[0] for n in body_names])
    dependencies = legacy.model_files(a.robot)
    if a.scene:
        dependencies.update(legacy.model_files(a.scene))
    inputs = {str(p.resolve()): sha(p) for p in [Path(__file__), a.tracker / 'sonic_rollout.py', a.reference, params_path]}
    if a.controller == 'amo':
        if not a.amo_adapter or not a.amo_weights:
            raise ValueError('AMO adapter and weights must be explicit')
        api = module(a.amo_adapter, 'benchmark_amo')
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        policy = api.AMOPolicy(a.amo_weights, device='cpu')
        policy.reset()
        inputs[str(a.amo_adapter.resolve())] = sha(a.amo_adapter)
        for file in a.amo_weights.glob('*.pt'):
            inputs[str(file.resolve())] = sha(file)
    elif a.controller.startswith('sonic'):
        models = a.tracker / 'models'
        if a.controller == 'sonic_v11':
            models /= 'sonic_v1_1'
        policy = legacy.Sonic(models, params, rbq, 2)
        for name in ['model_encoder.onnx', 'model_decoder.onnx', 'observation_config.yaml']:
            inputs[str((models / name).resolve())] = sha(models / name)
    else:
        file = a.tracker / 'scalebfm_policy.py'
        api = module(file, 'benchmark_scalebfm')
        policy = api.create_policy(a.tracker / 'scalebfm', params, rbq, 2, m, d, robot, rootq)
        inputs[str(file.resolve())] = sha(file)
        inputs.update(policy.input_bindings)
    index = {n: i for i, n in enumerate(names)}
    upper = [n for n in names if any(s in n for s in ['shoulder', 'elbow', 'wrist'])]
    extra = [n for n in names if n not in body_names]
    feet = [m.body(s + '_ankle_roll_link').id for s in ['left', 'right']]
    wrists = [m.body(s + '_wrist_yaw_link').id for s in ['left', 'right']]
    torso = m.body('torso_link').id
    robot_bodies = {m.body(robot.body(i).name).id for i in range(1, robot.nbody)}
    floor_names = a.floor_geom or ['tracker_ground']
    floor_geoms = {m.geom(name).id for name in floor_names}
    if any(m.geom_bodyid[g] in robot_bodies for g in floor_geoms):
        raise ValueError('The floor cannot be a robot geometry')
    reference_provider = None
    if a.reference_provider:
        provider_api = module(a.reference_provider, 'benchmark_reference_provider')
        reference_provider = provider_api.create_reference(ref, m, d, robot, a.output, a.provider_config)
        ref = reference_provider
        inputs[str(a.reference_provider.resolve())] = sha(a.reference_provider)
        inputs.update(reference_provider.input_bindings)
    save(a.output / 'inputs.json', {'schema': 'g1-common-controller-inputs/v1', 'inputs': inputs,
        'model_dependencies': dependencies, 'compiled_xml': {'path': str(executed), 'sha256': sha(executed)},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'scope': 'Common-robot configuration comparison; AMO velocity/heading goal adapter versus pose-reference trackers. No pickup or free-object carrying claim.',
        'model': {'nq': m.nq, 'nv': m.nv, 'nu': m.nu, 'mass_kg': float(m.body_mass.sum()),
            'joint_names': names, 'motor_indices': motors.tolist(), 'ctrlrange': m.actuator_ctrlrange[motors].tolist(),
            'initial_qpos': d.qpos.tolist(), 'initial_qvel': d.qvel.tolist(), 'initial_state_assignments': 1,
            'per_step_pose_assignment': False, 'timestep': .002, 'control_hz': 50, 'neq': m.neq, 'nmocap': m.nmocap},
        'floor_geometries': floor_names,
        'arm_override': {'mode': a.arm_mode, 'kp': 40., 'kd': 2., 'wrist_kp': 20., 'wrist_kd': 1.},
        'amo_goal_adapter': {'vx_limit': .5, 'vy_limit': .4, 'heading': 'absolute world heading of reference',
            'position_gain': a.position_gain, 'native_stand_gate_preserved': True,
            'torque_caps': 'Intersection of source model bounds and native AMO per-joint caps'},
        'timing': {'commands_and_targets': 'control interval start', 'measured_state_and_goal_error': 'control interval end',
                   'body_fk': 'Separate MjData kinematics recomputed from saved poststep qpos; active policy/sensors untouched',
                   'contacts': 'All 10 solver samples in each control interval; contact forces refer to solver poses before each integration',
                   'initial_velocity': 'zero', 'torch_threads': 2}})
    rows = collections.defaultdict(list)
    diagnostics = []
    failure = None
    fall = None
    tick_seconds = []
    previous_foot = d.xpos[feet].copy()
    previous_yaw = yaw(d.qpos[rootq + 3:rootq + 7])
    required = round(a.duration * 50)
    loop_start = time.perf_counter()
    try:
        for tick in range(required):
            tic = time.perf_counter()
            rt = a.start + tick * .02
            if reference_provider:
                prior_state = [v.copy() for v in (d.qpos, d.qvel, d.ctrl, d.qfrc_applied, d.xfrc_applied)]
                reference_provider.before_step(rt, d)
                if not all(np.array_equal(old, new) for old, new in zip(prior_state, (d.qpos, d.qvel, d.ctrl, d.qfrc_applied, d.xfrc_applied))):
                    raise ValueError('Reference provider changed actual simulation state or controls')
            qr = ref.at(rt)[0][0]
            before = ref.at(max(ref.t[0], rt - .01))[0][0]
            after = ref.at(min(ref.t[-1], rt + .01))[0][0]
            den = min(ref.t[-1], rt + .01) - max(ref.t[0], rt - .01)
            world_v = (after[:2] - before[:2]) / max(den, 1e-8)
            heading = yaw(qr[3:7])
            desired_yaw_rate = wrapped(yaw(after[3:7]) - yaw(before[3:7])) / max(den, 1e-8)
            actual_heading = yaw(d.qpos[rootq + 3:rootq + 7])
            c, s = np.cos(actual_heading), np.sin(actual_heading)
            to_body = np.array([[c, s], [-s, c]])
            command_world = world_v + a.position_gain * (qr[:2] - d.qpos[rootq:rootq + 2])
            command = to_body @ command_world
            command = np.clip(command, [-.5, -.4], [.5, .4])
            target = qr[rqids].copy()
            kp = np.full(len(names), 5.)
            kd = np.full(len(names), .1)
            limits = m.actuator_ctrlrange[motors].copy()
            if a.controller == 'amo':
                native_upper = {n: float(qr[robot.joint(n).qposadr[0]]) for n in upper if 'wrist' not in n}
                result = policy.step(m, d, float(command[0]), float(command[1]), heading,
                                     upper_body_targets=native_upper, dt=.02)
                for name, value in result['joint_targets'].items():
                    target[index[name]] = value
                    kp[index[name]] = result['kp'][name]
                    kd[index[name]] = result['kd'][name]
                    cap = float(result['torque_limits'][name])
                    if not np.isfinite(cap) or cap <= 0:
                        raise ValueError('Invalid native AMO torque cap')
                    limits[index[name], 0] = max(limits[index[name], 0], -cap)
                    limits[index[name], 1] = min(limits[index[name], 1], cap)
                for name in upper:
                    if 'wrist' in name:
                        kp[index[name]], kd[index[name]] = 20., 1.
                diag = result.get('diagnostics', {})
                diagnostics.append({'time': tick * .02, **{k: v for k, v in diag.items() if isinstance(v, (float, int, str, bool))}})
            else:
                value, _ = policy.infer(ref, rt, d.qpos[bq], d.qvel[bv],
                                        d.qpos[rootq + 3:rootq + 7], d.qvel[rootv + 3:rootv + 6], False)
                for i, name in enumerate(body_names):
                    target[index[name]], kp[index[name]], kd[index[name]] = value[i], params['kps'][i], params['kds'][i]
            if a.arm_mode == 'reference_pd':
                for name in upper:
                    i = index[name]
                    target[i] = qr[robot.joint(name).qposadr[0]]
                    kp[i], kd[i] = (20., 1.) if 'wrist' in name else (40., 2.)
            target = np.clip(target, m.jnt_range[jids, 0], m.jnt_range[jids, 1])
            foot_force = np.zeros(2)
            contact_counts = np.zeros(4, dtype=np.int64)
            contact_min_distance = np.zeros(4)
            contact_max_normal = np.zeros(4)
            saturation = np.zeros(len(names))
            max_external = 0.
            for _ in range(10):
                torque = kp * (target - d.qpos[qids]) - kd * d.qvel[vids]
                clipped = np.clip(torque, limits[:, 0], limits[:, 1])
                saturation += (np.abs(torque - clipped) > 1e-8) / 10
                d.ctrl[motors] = clipped
                max_external = max(max_external, float(np.max(np.abs(d.qfrc_applied))), float(np.max(np.abs(d.xfrc_applied))))
                mujoco.mj_step(m, d)
                for ci in range(d.ncon):
                    contact = d.contact[ci]
                    bodies = [int(m.geom_bodyid[g]) for g in [contact.geom1, contact.geom2]]
                    if not (set(bodies) & robot_bodies):
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, force)
                    on_floor = contact.geom1 in floor_geoms or contact.geom2 in floor_geoms
                    if on_floor and any(body in feet for body in bodies):
                        category = 0
                    elif on_floor:
                        category = 1
                    elif all(body in robot_bodies for body in bodies):
                        category = 2
                    else:
                        category = 3
                    contact_counts[category] += 1
                    contact_min_distance[category] = min(contact_min_distance[category], float(contact.dist))
                    contact_max_normal[category] = max(contact_max_normal[category], max(0., float(force[0])))
                    for fi, body in enumerate(feet):
                        if body in bodies and on_floor:
                            foot_force[fi] += max(0., force[0]) / 10
            current_yaw = yaw(d.qpos[rootq + 3:rootq + 7])
            tilt = np.arccos(np.clip(Rotation.from_quat(d.qpos[rootq + np.array([4, 5, 6, 3])]).as_matrix()[2, 2], -1, 1))
            measured_fk.qpos[:] = d.qpos
            mujoco.mj_kinematics(m, measured_fk)
            foot_speed = np.linalg.norm((measured_fk.xpos[feet, :2] - previous_foot[:, :2]) / .02, axis=1)
            previous_foot = measured_fk.xpos[feet].copy()
            torso_r = measured_fk.xmat[torso].reshape(3, 3)
            end_ref = ref.at(min(ref.t[-1], rt + .02))[0][0]
            if reference_provider:
                reference_provider.after_step(rt + .02, d)
            for key, value in {'time': d.time, 'reference_time': min(ref.t[-1], rt + .02), 'control_reference_time': rt,
                'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(),
                'ctrl': d.ctrl.copy(), 'joint_target': target, 'kp': kp, 'kd': kd,
                'effective_torque_limits': limits,
                'root_xy_error': d.qpos[rootq:rootq + 2] - end_ref[:2], 'heading_error': wrapped(current_yaw - yaw(end_ref[3:7])),
                'desired_world_velocity': world_v, 'command_body_velocity': command, 'desired_heading': heading,
                'desired_yaw_rate': desired_yaw_rate, 'actual_world_velocity': d.qvel[rootv:rootv + 2].copy(),
                'actual_yaw_rate': wrapped(current_yaw - previous_yaw) / .02,
                'foot_normal_force': foot_force, 'loaded_foot_body_speed': foot_speed,
                'contact_counts_foot_ground_body_ground_self_room': contact_counts,
                'contact_min_distance_foot_ground_body_ground_self_room': contact_min_distance,
                'contact_max_normal_foot_ground_body_ground_self_room': contact_max_normal,
                'wrist_world': measured_fk.xpos[wrists].copy(), 'wrist_torso': (measured_fk.xpos[wrists] - measured_fk.xpos[torso]) @ torso_r,
                'tilt_degrees': np.degrees(tilt), 'saturation_fraction': saturation, 'max_external_force': max_external}.items():
                rows[key].append(value)
            if reference_provider:
                rows['reference_qpos'].append(end_ref)
            previous_yaw = current_yaw
            tick_seconds.append(time.perf_counter() - tic)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                failure = 'Nonfinite simulation state'
                break
            if d.qpos[rootq + 2] < .45 or tilt > np.radians(65):
                fall = {'time': d.time, 'root_z': float(d.qpos[rootq + 2]), 'tilt_degrees': float(np.degrees(tilt))}
                break
            if tick % 100 == 0:
                save(a.output / 'status.json', {'state': 'running', 'samples': tick + 1, 'requested': required,
                    'time': d.time, 'root_xy_error': float(np.linalg.norm(rows['root_xy_error'][-1])), 'tilt_degrees': float(np.degrees(tilt))})
    except Exception as exc:
        failure = repr(exc)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    provider_result = reference_provider.finish() if reference_provider else None
    np.savez_compressed(a.output / 'rollout.npz', **arrays)
    save(a.output / 'diagnostics.json', diagnostics)
    completed = len(rows['time']) == required and fall is None and failure is None
    result = {'schema': 'g1-common-controller-run/v1', 'produced_at': datetime.now(timezone.utc).isoformat(),
        'completed': completed, 'controller': a.controller, 'samples': len(rows['time']), 'requested_samples': required,
        'fall': fall, 'failure': failure, 'loop_wall_seconds': time.perf_counter() - loop_start,
        'total_wall_seconds': time.perf_counter() - start_wall, 'control_tick_ms_median': float(np.median(tick_seconds) * 1000) if tick_seconds else None,
        'control_tick_ms_p95': float(np.percentile(tick_seconds, 95) * 1000) if tick_seconds else None,
        'physical_interaction_success_claim': False, 'inputs_sha256': sha(a.output / 'inputs.json'),
        'reference_provider_result': provider_result,
        'artifacts': {'rollout.npz': sha(a.output / 'rollout.npz'), 'diagnostics.json': sha(a.output / 'diagnostics.json')},
        'scope': 'Physics-driven common-robot controller trial; final task-goal ranking is evaluated separately.'}
    save(a.output / 'receipt.json', result)
    save(a.output / 'status.json', {'state': 'completed' if completed else 'failed', **result})
    print(json.dumps(result, indent=2))
    if failure:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
