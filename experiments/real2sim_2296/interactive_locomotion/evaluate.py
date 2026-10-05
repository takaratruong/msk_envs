#!/usr/bin/env python3
"""Evaluate bound common-controller trials against an explicit screening protocol."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def segments(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(boundaries[::2], boundaries[1::2]))


def rms_norm(values):
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=-1)))) if len(values) else None


def yaw(quaternions):
    r = Rotation.from_quat(np.asarray(quaternions)[..., [1, 2, 3, 0]]).as_matrix()
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def evaluate(folder, protocol):
    raw = json.loads((folder / 'receipt.json').read_text())
    inp = json.loads((folder / 'inputs.json').read_text())
    args = inp['arguments']
    matches = [case for case in protocol['cases']
               if Path(args['reference']).resolve() == (Path(protocol['project_root']) / case['reference']).resolve()
               and args['start'] == case['start_seconds'] and args['duration'] == case['duration_seconds']]
    if len(matches) != 1 or args['arm_mode'] != protocol['comparison']['arm_mode']:
        raise ValueError('Trial is outside the declared case, interval or arm configuration')
    if args.get('scene') or args.get('reference_provider') or any(args.get('initial_root_offset', [0., 0., 0.])):
        raise ValueError('Room, generated-reference and perturbed-start trials require their own protocol')
    if args['controller'] != raw['controller'] or raw['controller'] not in ('amo', 'scalebfm', 'sonic', 'sonic_v11'):
        raise ValueError('Controller identity mismatch')
    if args['position_gain'] not in (0., 1.) or (args['controller'] != 'amo' and args['position_gain'] != 0.):
        raise ValueError('Unsupported or misleading goal-feedback configuration')
    if set(('rollout.npz', 'diagnostics.json')) - set(raw['artifacts']):
        raise ValueError('Missing mandatory raw artifact binding')
    required = round(matches[0]['duration_seconds'] * 50)
    if (raw['requested_samples'] != required or type(raw['samples']) is not int
            or not 0 <= raw['samples'] <= required or type(raw['completed']) is not bool
            or raw['completed'] != (raw['samples'] == required and raw['fall'] is None and raw['failure'] is None)):
        raise ValueError('Run completion or coverage disagrees with the declared interval')
    if sha(folder / 'inputs.json') != raw['inputs_sha256']:
        raise ValueError('Run input receipt mismatch: ' + str(folder))
    for name, expected in raw['artifacts'].items():
        if sha(folder / name) != expected:
            raise ValueError('Run artifact changed: ' + str(folder / name))
    for file, expected in {**inp['inputs'], **inp['model_dependencies']}.items():
        if sha(file) != expected:
            raise ValueError('Bound run dependency changed: ' + file)
    if sha(inp['compiled_xml']['path']) != inp['compiled_xml']['sha256']:
        raise ValueError('Compiled scene changed')
    with np.load(folder / 'rollout.npz', allow_pickle=False) as source:
        z = {key: source[key] for key in source.files}
    result = {'name': folder.name, 'controller': raw['controller'], 'run_receipt_sha256': sha(folder / 'receipt.json'),
              'inputs_sha256': raw['inputs_sha256'], 'arguments': inp['arguments'], 'completed': raw['completed'],
              'fall': raw['fall'], 'failure': raw['failure'], 'samples': raw['samples']}
    if not raw['samples']:
        result.update(classification='infrastructure_failure', metrics=None, gates=None)
        return result
    n = raw['samples']
    mandatory = ('time', 'reference_time', 'control_reference_time', 'qpos', 'qvel', 'ctrl',
                 'root_xy_error', 'heading_error', 'desired_world_velocity', 'desired_heading',
                 'desired_yaw_rate', 'actual_world_velocity', 'actual_yaw_rate', 'command_body_velocity',
                 'foot_normal_force', 'wrist_world', 'wrist_torso', 'saturation_fraction', 'max_external_force',
                 'tilt_degrees')
    if set(mandatory) - set(z):
        raise ValueError('Missing required physical telemetry')
    if any(len(value) != n for value in z.values()):
        raise ValueError('Inconsistent sample counts')
    finite = all(np.isfinite(value).all() for value in z.values() if value.dtype.kind in 'fc')
    if not finite or not np.allclose(z['time'], (np.arange(n) + 1) / 50, atol=1e-8):
        raise ValueError('Invalid finite state or physics sample times')
    source_path = Path(args['tracker']) / 'sonic_rollout.py'
    spec = importlib.util.spec_from_file_location('screen_reference_source', source_path)
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    model = mujoco.MjModel.from_xml_path(args['robot'])
    reference = api.Reference(Path(args['reference']), model)
    expected_time = np.minimum(reference.t[-1], args['start'] + z['time'])
    expected_control_time = args['start'] + np.arange(n) / 50
    if (not np.allclose(z['reference_time'], expected_time, atol=1e-9, rtol=0)
            or not np.allclose(z['control_reference_time'], expected_control_time, atol=1e-9, rtol=0)):
        raise ValueError('Reference clock does not cover the declared interval')
    desired = reference.at(z['reference_time'])[0]
    actual_model = mujoco.MjModel.from_xml_path(inp['compiled_xml']['path'])
    actual_data = mujoco.MjData(actual_model)
    root_id = actual_model.joint('floating_base_joint').qposadr[0]
    root_velocity_id = actual_model.joint('floating_base_joint').dofadr[0]
    if z['qpos'].shape != (n, actual_model.nq) or z['qvel'].shape != (n, actual_model.nv) or z['ctrl'].shape != (n, actual_model.nu):
        raise ValueError('Recorded embodiment shape mismatch')
    if np.any(z['ctrl'] < actual_model.actuator_ctrlrange[:, 0] - 1e-9) or np.any(z['ctrl'] > actual_model.actuator_ctrlrange[:, 1] + 1e-9):
        raise ValueError('Recorded control exceeds model motor limits')
    actual_xy = z['qpos'][:, root_id:root_id + 2]
    if not np.allclose(z['root_xy_error'], actual_xy - desired[:, :2], atol=1e-9):
        raise ValueError('Recorded position error is not aligned to poststep target')
    before_time = np.maximum(reference.t[0], expected_control_time - .01)
    after_time = np.minimum(reference.t[-1], expected_control_time + .01)
    before, after = reference.at(before_time)[0], reference.at(after_time)[0]
    denominator = (after_time - before_time)
    velocity_goal = (after[:, :2] - before[:, :2]) / denominator[:, None]
    yaw_rate_goal = wrap(yaw(after[:, 3:7]) - yaw(before[:, 3:7])) / denominator
    control_reference = reference.at(expected_control_time)[0]
    heading_goal = yaw(control_reference[:, 3:7])
    actual_heading = yaw(z['qpos'][:, root_id + 3:root_id + 7])
    heading_error = wrap(actual_heading - yaw(desired[:, 3:7]))
    actual_velocity = z['qvel'][:, root_velocity_id:root_velocity_id + 2]
    initial = np.asarray(inp['model']['initial_qpos'])
    previous_heading = np.r_[yaw(initial[root_id + 3:root_id + 7]), actual_heading[:-1]]
    previous_xy = np.vstack([initial[root_id:root_id + 2], actual_xy[:-1]])
    goal_world = velocity_goal + args['position_gain'] * (control_reference[:, :2] - previous_xy)
    c, s = np.cos(previous_heading), np.sin(previous_heading)
    command = np.clip(np.column_stack([c * goal_world[:, 0] + s * goal_world[:, 1],
                                      -s * goal_world[:, 0] + c * goal_world[:, 1]]), [-.5, -.4], [.5, .4])
    derived = {'desired_world_velocity': velocity_goal, 'desired_heading': heading_goal,
               'desired_yaw_rate': yaw_rate_goal, 'heading_error': heading_error,
               'actual_world_velocity': actual_velocity, 'actual_yaw_rate': wrap(actual_heading - previous_heading) * 50,
               'command_body_velocity': command}
    for key, value in derived.items():
        if z[key].shape != value.shape or not np.allclose(z[key], value, atol=1e-8, rtol=0):
            raise ValueError('Derived telemetry mismatch: ' + key)
        z[key] = value
    reference_data = mujoco.MjData(model)
    wrist_ids = [model.body(side + '_wrist_yaw_link').id for side in ('left', 'right')]
    torso_id = model.body('torso_link').id
    wanted_world, wanted_torso = [], []
    for q in desired:
        reference_data.qpos[:] = q
        mujoco.mj_forward(model, reference_data)
        wrist = reference_data.xpos[wrist_ids].copy()
        wanted_world.append(wrist)
        wanted_torso.append((wrist - reference_data.xpos[torso_id]) @ reference_data.xmat[torso_id].reshape(3, 3))
    # mj_step leaves xpos at the final solver pose before its last integration.
    # Independently derive all end-state body metrics from recorded qpos.
    actual_wrist_ids = [actual_model.body(side + '_wrist_yaw_link').id for side in ('left', 'right')]
    actual_foot_ids = [actual_model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    actual_torso_id = actual_model.body('torso_link').id
    actual_data.qpos[:] = inp['model']['initial_qpos']
    mujoco.mj_kinematics(actual_model, actual_data)
    previous_foot = actual_data.xpos[actual_foot_ids].copy()
    actual_world, actual_torso, actual_foot_speed = [], [], []
    for q in z['qpos']:
        actual_data.qpos[:] = q
        mujoco.mj_kinematics(actual_model, actual_data)
        wrist = actual_data.xpos[actual_wrist_ids].copy()
        actual_world.append(wrist)
        actual_torso.append((wrist - actual_data.xpos[actual_torso_id]) @ actual_data.xmat[actual_torso_id].reshape(3, 3))
        actual_foot_speed.append(np.linalg.norm(actual_data.xpos[actual_foot_ids, :2] - previous_foot[:, :2], axis=1) * 50)
        previous_foot = actual_data.xpos[actual_foot_ids].copy()
    actual_world, actual_torso = np.asarray(actual_world), np.asarray(actual_torso)
    logged_wrist_discrepancy = float(np.linalg.norm(actual_world - z['wrist_world'], axis=-1).max())
    if 'body_fk' in inp['timing'] and logged_wrist_discrepancy > 1e-9:
        raise ValueError('Poststep FK telemetry disagrees with recorded physics poses')
    windows = protocol['windows']
    speed = np.linalg.norm(z['desired_world_velocity'], axis=1)
    moving = speed > windows['moving_speed_threshold_m_s']
    stopped = ((speed < windows['stopped_speed_threshold_m_s'])
               & (np.abs(z['desired_yaw_rate']) < windows['stopped_yaw_rate_threshold_rad_s']))
    stops = []
    for begin, end in segments(stopped):
        if (end - begin) / 50 < windows['stopped_segment_min_seconds']:
            continue
        settled = begin + round(windows['stop_settle_seconds'] * 50)
        path = actual_xy[settled:end]
        stops.append({'from': float(z['time'][begin]), 'to': float(z['time'][end - 1]),
                      'measured_after_settling_at': float(z['time'][settled]),
                      'max_excursion_m': float(np.linalg.norm(path - path[0], axis=1).max()),
                      'endpoint_displacement_m': float(np.linalg.norm(path[-1] - path[0])),
                      'path_length_m': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())})
    loaded = z['foot_normal_force'] > windows['foot_loaded_force_newtons']
    foot = np.asarray(actual_foot_speed)[loaded]
    wrist_torso_error = actual_torso - np.asarray(wanted_torso)
    wrist_world_error = actual_world - np.asarray(wanted_world)
    categories = ('foot_ground', 'body_ground', 'self', 'room')
    contacts = {}
    contact_key = 'contact_counts_foot_ground_body_ground_self_room'
    if contact_key in z:
        for i, name in enumerate(categories):
            contacts[name] = {
                'contact_observations_500hz': int(z[contact_key][:, i].sum()),
                'max_penetration_m': max(0., float(-z['contact_min_distance_foot_ground_body_ground_self_room'][:, i].min())),
                'max_normal_newtons': float(z['contact_max_normal_foot_ground_body_ground_self_room'][:, i].max())}
    targets = protocol['screen_targets']
    velocity_error = z['actual_world_velocity'] - z['desired_world_velocity']
    metrics = {
        'seconds_simulated': float(z['time'][-1]), 'moving_samples': int(moving.sum()),
        'moving_velocity_rmse_m_s': rms_norm(velocity_error[moving]),
        'all_velocity_rmse_m_s': rms_norm(velocity_error),
        'root_xy_rmse_m': rms_norm(z['root_xy_error']),
        'root_xy_final_error_m': float(np.linalg.norm(z['root_xy_error'][-1])),
        'heading_abs_p95_degrees': float(np.degrees(np.percentile(np.abs(z['heading_error']), 95))),
        'heading_final_error_degrees': float(np.degrees(z['heading_error'][-1])),
        'yaw_rate_rmse_rad_s': float(np.sqrt(np.mean((z['actual_yaw_rate'] - z['desired_yaw_rate']) ** 2))),
        'stop_windows': stops,
        'stop_max_excursion_m': max((v['max_excursion_m'] for v in stops), default=None),
        'loaded_ankle_body_xy_speed_p95_m_s': float(np.percentile(foot, 95)) if len(foot) else None,
        'body_metrics_source': 'Independent end-state FK from saved qpos',
        'raw_logged_wrist_max_timestamp_discrepancy_m': logged_wrist_discrepancy,
        'wrist_torso_position_rmse_m': rms_norm(wrist_torso_error.reshape(-1, 3)),
        'wrist_torso_rmse_left_right_m': [rms_norm(wrist_torso_error[:, i]) for i in range(2)],
        'wrist_world_position_rmse_m': rms_norm(wrist_world_error.reshape(-1, 3)),
        'motor_saturation_fraction_mean': float(z['saturation_fraction'].mean()),
        'most_saturated_joint': inp['model']['joint_names'][int(np.argmax(z['saturation_fraction'].mean(axis=0)))],
        'motor_saturation_fraction_worst_joint': float(z['saturation_fraction'].mean(axis=0).max()),
        'max_external_force': float(z['max_external_force'].max()),
        'max_tilt_degrees': float(z['tilt_degrees'].max()),
        'contacts': contacts,
        'control_tick_ms_median': raw['control_tick_ms_median'], 'control_tick_p95_ms': raw['control_tick_ms_p95'],
        'simulation_seconds_per_wall_second': float(z['time'][-1] / raw['loop_wall_seconds'])}
    gates = {'complete_without_fall': bool(raw['completed'] and not raw['fall'] and not raw['failure']),
             'external_force_zero': metrics['max_external_force'] == 0}
    for key in ('moving_velocity_rmse_m_s', 'stop_max_excursion_m', 'heading_abs_p95_degrees',
                'wrist_torso_position_rmse_m', 'control_tick_p95_ms'):
        gates[key] = metrics[key] <= targets[key] if metrics[key] is not None else None
    for key, metric in [('body_ground_max_normal_newtons', 'max_normal_newtons'),
                        ('body_ground_max_penetration_m', 'max_penetration_m')]:
        gates[key] = contacts['body_ground'][metric] <= targets[key] if contacts else None
    result.update(classification='completed_physics_trial' if raw['completed'] else 'physics_failure',
                  metrics=metrics, gates=gates, all_observed_targets_met=all(v is True for v in gates.values()),
                  interpretation='Configuration-specific floor screening only. Missing coverage is untested. No manipulation success claim.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--trials', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    results = [evaluate(folder, protocol) for folder in a.trials]
    save(a.output / 'comparison.json', {'schema': 'g1-controller-screen/v1', 'protocol': protocol,
                                      'protocol_sha256': sha(a.protocol), 'results': results})
    receipt = {'schema': 'agentic-evidence/v1', 'run_id': a.output.name,
               'claim': 'Bound controller screening measurements for the listed trials; no interactive or manipulation reliability claim.',
               'produced_at': datetime.now(timezone.utc).isoformat(),
               'inputs_sha256': hashlib.sha256(json.dumps({'protocol': sha(a.protocol), 'evaluator': sha(__file__),
                   'trials': {str(f.resolve()): sha(f / 'receipt.json') for f in a.trials}}, sort_keys=True).encode()).hexdigest(),
               'artifact': {'path': str((a.output / 'comparison.json').resolve()), 'sha256': sha(a.output / 'comparison.json')},
               'evaluator_sha256': sha(__file__)}
    save(a.output / 'receipt.json', receipt)
    for r in results:
        print(json.dumps({k: r[k] for k in ('name', 'completed', 'metrics', 'gates')}, indent=2))


if __name__ == '__main__':
    main()
