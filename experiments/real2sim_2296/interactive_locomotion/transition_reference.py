#!/usr/bin/env python3
"""Measured-state MotionBricks references for a continuous motor-driven trial.

This module reads the simulated robot. It never assigns its pose, velocity,
forces or controls. A separate persistent process generates approximate poses;
the existing tracking policy and its history remain alive across activation.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def smooth(t):
    x = np.clip(t, 0., 1.)
    return x * x * x * (10 + x * (-15 + 6 * x))


def interpolate(t, q, wanted):
    wanted = np.atleast_1d(np.clip(wanted, t[0], t[-1]))
    result = np.column_stack([np.interp(wanted, t, q[:, k]) for k in range(q.shape[1])])
    result[:, 3:7] = Slerp(t, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(wanted).as_quat()[:, [3, 0, 1, 2]]
    return result


def blend(a, b, weight):
    weight = np.asarray(weight)
    result = a * (1 - weight[:, None]) + b * weight[:, None]
    left = Rotation.from_quat(a[:, [4, 5, 6, 3]])
    right = Rotation.from_quat(b[:, [4, 5, 6, 3]])
    result[:, 3:7] = (left * Rotation.from_rotvec((left.inv() * right).as_rotvec() * weight[:, None])).as_quat()[:, [3, 0, 1, 2]]
    return result


class TransitionReference:
    def __init__(self, source, model, data, robot, output, config_path):
        self.source = source
        self.t = source.t
        self.config = json.loads(Path(config_path).read_text())
        self.output = Path(output) / 'transition'
        self.output.mkdir(exist_ok=False)
        self.events = []
        self.samples = []
        self.history = []
        self.state = 'travel'
        self.request_time = None
        self.active_time = None
        self.target_time = None
        self.stable_seconds = 0.
        self.entry_verified = False
        self.activated = None
        self.pending = False
        self.buffer = b''
        self.closed = False
        self.last_observation = None
        self.model = model
        self.fk = mujoco.MjData(model)
        self.robot_fk = mujoco.MjData(robot)
        self.robot = robot
        self.rootq = int(model.joint('floating_base_joint').qposadr[0])
        self.rootv = int(model.joint('floating_base_joint').dofadr[0])
        names = [robot.joint(i).name for i in range(robot.njnt) if robot.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        self.actual_q = np.array([model.joint(n).qposadr[0] for n in names])
        self.robot_q = np.array([robot.joint(n).qposadr[0] for n in names])
        self.body_q = np.array([robot.joint(n).qposadr[0] for n in names if '_hand_' not in n])
        self.foot_bodies = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
        self.floor_geoms = {model.geom(name).id for name in self.config.get('ground_geom_names', ['tracker_ground'])}
        self.target = source.at(self.config['target_reference_time'] + np.arange(4) / 30)[0]
        for name in ('request_after_seconds', 'maximum_candidate_age_seconds', 'prefix_blend_seconds',
                     'tail_blend_seconds', 'entry_hold_seconds'):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError('Transition timing must be finite and positive: ' + name)
        self.provider_root = Path(self.config['provider_root'])
        server = self.provider_root / 'provider_server.py'
        provider = self.provider_root / 'provider.py'
        if sha(server) != self.config['server_sha256'] or sha(provider) != self.config['provider_sha256']:
            raise ValueError('Pinned reference provider changed')
        self.input_bindings = {str(Path(config_path).resolve()): sha(config_path), str(server): sha(server), str(provider): sha(provider)}
        self.log = (self.output / 'provider.stderr.log').open('wb')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.config['provider_gpu']))
        command = [self.config['provider_python'], str(server), '--root', str(self.provider_root),
                   '--robot-model', self.config['robot_model'], '--audit-output', self.config['provider_audit'],
                   '--max-runtime', '240', '--max-requests', '4', '--idle-timeout', '60']
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env, bufsize=0)
        try:
            ready = self._receive(60)
            if ready is None or ready.get('status') != 'ready':
                raise RuntimeError('MotionBricks provider did not become ready')
            save(self.output / 'provider_startup.json', ready)
            self.provider_identity = ready['identity']
            self.input_bindings[str(self.output / 'provider_startup.json')] = sha(self.output / 'provider_startup.json')
            # Pay first-inference initialization before starting the physics clock.
            initial = self._measured(data)
            warm = {'id': 'warmup_discarded', 'context_qpos50': np.tile(initial, (4, 1)).tolist(),
                    'context_time': (-.1 + np.arange(4) / 30).tolist(),
                    'target_qpos50': np.tile(initial, (4, 1)).tolist()}
            self._send(warm)
            response = self._receive(30)
            if response is None or response.get('status') != 'reference_candidate':
                raise RuntimeError('MotionBricks warmup failed')
            save(self.output / 'warmup_discarded.json', response)
        except BaseException:
            self.close()
            raise

    def _measured(self, data):
        q = np.empty(self.robot.nq)
        q[:7] = data.qpos[self.rootq:self.rootq + 7]
        q[self.robot_q] = data.qpos[self.actual_q]
        return q

    def _send(self, request):
        self.process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode())
        self.process.stdin.flush()

    def _receive(self, timeout=0.):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            available, _, _ = select.select([self.process.stdout], [], [], max(0., deadline - time.monotonic()))
            if not available:
                return None
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError('Reference provider closed unexpectedly')
            self.buffer += block
            if len(self.buffer) > 2_000_000:
                raise ValueError('Oversized reference response')
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def before_step(self, now, data):
        measured = self._measured(data)
        self.history.append((float(now), measured))
        if self.pending:
            response = self._receive()
            if response is not None:
                self.pending = False
                save(self.output / 'raw_response.json', response)
                if response.get('status') != 'reference_candidate' or response.get('id') != 'skill_entry':
                    self._reject(now, 'provider_rejected_or_failed')
                elif now - self.request_time > self.config['maximum_candidate_age_seconds']:
                    self._reject(now, 'candidate_expired')
                else:
                    self._activate(now, measured, response['result'])
            elif now - self.request_time > self.config['maximum_candidate_age_seconds']:
                self.pending = False
                self._reject(now, 'candidate_timeout')
        if self.state == 'travel' and now >= self.config['request_after_seconds'] - 1e-9:
            distance = float(np.linalg.norm(measured[:2] - self.target[0, :2]))
            if distance > self.config['max_request_distance_m']:
                self._reject(now, 'outside_interaction_approach_region')
                return
            times = now - .1 + np.arange(4) / 30
            history = self.history[-10:]
            context = interpolate(np.array([x[0] for x in history]), np.array([x[1] for x in history]), times)
            request = {'id': 'skill_entry', 'context_qpos50': context.tolist(),
                       'context_time': times.tolist(), 'target_qpos50': self.target.tolist()}
            save(self.output / 'request.json', request)
            self._send(request)
            self.sent_request = request
            self.request_time = float(now)
            self.pending = True
            self.state = 'generating'
            self.events.append({'time': float(now), 'state': self.state, 'approach_distance_m': distance})

    def _reject(self, now, reason):
        self.state = 'failed'
        self.events.append({'time': float(now), 'state': self.state, 'reason': reason})

    def _activate(self, now, measured, result):
        times = np.asarray(result['time'], float)
        raw = np.asarray(result['qpos50'], float)
        mask = np.asarray(result['future_mask'], bool)
        if (times.ndim != 1 or len(times) < 8 or raw.shape != (len(times), self.robot.nq)
                or not np.isfinite(raw).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
                or not np.allclose(np.diff(times), 1 / 30, atol=1e-7, rtol=0)
                or not np.allclose(times[:4], self.sent_request['context_time'], atol=1e-7, rtol=0)
                or result['metadata']['identity'] != self.provider_identity
                or not np.array_equal(result['context_qpos50'], self.sent_request['context_qpos50'])
                or not np.array_equal(result['target_qpos50'], self.sent_request['target_qpos50'])
                or not np.allclose(np.linalg.norm(raw[:, 3:7], axis=1), 1., atol=1e-3, rtol=0)
                or not np.array_equal(mask, times > self.request_time + 1e-9)
                or times[-1] - now < self.config['prefix_blend_seconds'] + self.config['tail_blend_seconds']):
            self._reject(now, 'malformed_or_insufficient_future_reference')
            return
        future_now = interpolate(times, raw, now)[0]
        mismatch = {'root_m': float(np.linalg.norm(future_now[:3] - measured[:3])),
                    'joint_max_rad': float(np.max(np.abs(future_now[self.body_q] - measured[self.body_q]))),
                    'root_orientation_rad': float((Rotation.from_quat(future_now[[4, 5, 6, 3]]).inv()
                                                  * Rotation.from_quat(measured[[4, 5, 6, 3]])).magnitude())}
        if (mismatch['root_m'] > self.config['max_activation_root_m']
                or mismatch['joint_max_rad'] > self.config['max_activation_joint_rad']
                or mismatch['root_orientation_rad'] > self.config.get('max_activation_orientation_rad', .3)):
            self._reject(now, 'current_state_mismatch')
            return
        used_time = np.r_[now, times[times > now + 1e-9]]
        candidate = interpolate(times, raw, used_time)
        prior = self.source.at(np.minimum(used_time, self.config['target_reference_time']))[0]
        prefix = smooth((used_time - now) / self.config['prefix_blend_seconds'])
        candidate = blend(prior, candidate, prefix)
        target = np.tile(self.target[-1], (len(used_time), 1))
        # Dex3 ownership remains with the holding controller throughout this transition.
        finger_q = np.setdiff1d(np.arange(7, self.robot.nq), self.body_q)
        target[:, finger_q] = measured[finger_q]
        tail = smooth((used_time - (used_time[-1] - self.config['tail_blend_seconds'])) / self.config['tail_blend_seconds'])
        candidate = blend(candidate, target, tail)
        dt = np.diff(used_time)
        joint_speed = float(np.max(np.abs(np.diff(candidate[:, self.body_q], axis=0)) / dt[:, None]))
        root_speed = float(np.max(np.linalg.norm(np.diff(candidate[:, :3], axis=0), axis=1) / dt))
        rotations = Rotation.from_quat(candidate[:, [4, 5, 6, 3]])
        angular_speed = float(np.max((rotations[:-1].inv() * rotations[1:]).magnitude() / dt))
        if (joint_speed > self.config['max_reference_joint_speed_rad_s'] or root_speed > self.config['max_reference_root_speed_m_s']
                or angular_speed > self.config.get('max_reference_root_angular_speed_rad_s', 2.)):
            self._reject(now, 'reference_rate_limit')
            return
        ranges = self.robot.jnt_range[1:]
        if np.any(candidate[:, self.robot_q] < ranges[:, 0] - 1e-7) or np.any(candidate[:, self.robot_q] > ranges[:, 1] + 1e-7):
            self._reject(now, 'reference_joint_limit')
            return
        self.activated = (used_time, candidate)
        self.active_time = float(now)
        self.target_time = float(used_time[-1])
        self.state = 'transition'
        np.savez_compressed(self.output / 'activated_reference.npz', time=used_time, qpos=candidate,
                            raw_time=times, raw_qpos=raw)
        self.events.append({'time': float(now), 'state': self.state, 'request_age_seconds': now - self.request_time,
                            'state_mismatch': mismatch, 'target_time': self.target_time,
                            'max_reference_joint_speed_rad_s': joint_speed, 'max_reference_root_speed_m_s': root_speed,
                            'max_reference_root_angular_speed_rad_s': angular_speed,
                            'raw_quaternion_max_norm_error': float(np.abs(np.linalg.norm(raw[:, 3:7], axis=1) - 1).max()),
                            'quaternion_projection': 'SLERP normalizes the generator float quaternions; raw response is preserved',
                            'policy_history_reset': False, 'actual_state_assignment': False})

    def at(self, requested):
        times = np.atleast_1d(requested).astype(float)
        q, qd = self.source.at(np.minimum(times, self.config['target_reference_time']))
        after_seed = times > self.config['target_reference_time']
        qd[after_seed] = 0.
        if self.activated is not None:
            active_t, active_q = self.activated
            selected = times >= active_t[0]
            if np.any(selected):
                q[selected] = interpolate(active_t, active_q, times[selected])
            derivative = np.gradient(active_q[:, 7:], active_t, axis=0)
            for k in range(derivative.shape[1]):
                qd[selected, k] = np.interp(np.clip(times[selected], active_t[0], active_t[-1]), active_t, derivative[:, k])
            qd[times >= active_t[-1]] = 0.
        return q, qd

    def after_step(self, now, data):
        q = self._measured(data)
        self.fk.qpos[:] = data.qpos
        mujoco.mj_kinematics(self.model, self.fk)
        target = self.target[-1]
        root_error = float(np.linalg.norm(q[:2] - target[:2]))
        actual_r = Rotation.from_quat(q[[4, 5, 6, 3]])
        target_r = Rotation.from_quat(target[[4, 5, 6, 3]])
        heading_error = float(abs((target_r.inv() * actual_r).as_euler('zyx')[0]))
        body_error = float(np.sqrt(np.mean((q[self.body_q] - target[self.body_q]) ** 2)))
        speed = float(np.linalg.norm(data.qvel[self.rootv:self.rootv + 2]))
        tilt = float(np.arccos(np.clip(actual_r.as_matrix()[2, 2], -1., 1.)))
        forces = np.zeros(2)
        for i in range(data.ncon):
            contact = data.contact[i]
            bodies = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if contact.geom1 not in self.floor_geoms and contact.geom2 not in self.floor_geoms:
                continue
            for k, foot in enumerate(self.foot_bodies):
                if foot in bodies:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, data, i, force)
                    forces[k] += max(0., force[0])
        c = self.config['entry_gate']
        gates = {'arrived_xy': root_error <= c['root_xy_m'], 'facing': heading_error <= np.radians(c['heading_degrees']),
                 'body_pose': body_error <= c['body_joint_rmse_rad'], 'stopped': speed <= c['root_speed_m_s'],
                 'upright': tilt <= np.radians(c['tilt_degrees']), 'both_feet_loaded': bool(np.all(forces > c['foot_normal_N']))}
        gates = {name: bool(value) for name, value in gates.items()}
        eligible = self.activated is not None and now >= self.target_time
        self.stable_seconds = self.stable_seconds + .02 if eligible and all(gates.values()) else 0.
        if eligible and self.state == 'transition':
            self.state = 'settling'
            self.events.append({'time': float(now), 'state': self.state})
        if self.stable_seconds >= self.config['entry_hold_seconds'] and not self.entry_verified:
            self.entry_verified = True
            self.state = 'entry_ready'
            self.events.append({'time': float(now), 'state': self.state, 'gates': gates})
        self.last_observation = {'time': float(now), 'state': self.state, 'root_xy_error_m': root_error,
             'heading_error_degrees': float(np.degrees(heading_error)), 'body_joint_rmse_rad': body_error,
             'root_speed_m_s': speed, 'tilt_degrees': float(np.degrees(tilt)), 'foot_normal_N': forces.tolist(),
             'stable_seconds': self.stable_seconds, 'gates': gates}
        self.samples.append(self.last_observation)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if getattr(self, 'process', None):
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        if getattr(self, 'log', None):
            self.log.close()

    def finish(self):
        self.close()
        save(self.output / 'events.json', self.events)
        save(self.output / 'observations.json', self.samples)
        result = {'schema': 'g1-measured-transition-result/v1', 'state': self.state,
                  'reference_activated': self.activated is not None, 'entry_gate_held': self.entry_verified,
                  'last_observation': self.last_observation, 'physical_interaction_success_claim': False,
                  'scope': 'Continuous approach-to-stance reference transition under the same physics tracking policy; no door manipulation claim.',
                  'artifacts': {p.name: sha(p) for p in self.output.iterdir() if p.is_file()}}
        save(self.output / 'result.json', result)
        return result


def create_reference(*args):
    return TransitionReference(*args)
