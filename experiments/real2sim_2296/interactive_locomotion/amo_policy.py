"""Native-semantics AMO G1 inference adapter; no simulation/state/torque writes.

Observation/action formulas derived from OpenTeleVision/AMO play_amo.py,
copyright 2025 Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang;
Apache-2.0, https://www.apache.org/licenses/LICENSE-2.0 .

The command frontend owns yaw-rate integration.  ``heading`` is absolute world
yaw, exactly as upstream.  Call once per 20 ms, then apply the returned PD
targets at the physics rate, intersecting native caps with source motor limits.
Six wrist and fourteen Dex3 joints are deliberately outside this policy.
"""
from collections import deque
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch


SOURCE_URL = 'https://github.com/OpenTeleVision/AMO'
SOURCE_REVISION = '34caaf943660e6f9420e35f64e86dd56fb51dd0e'
WEIGHTS_SHA256 = {
    'amo_jit.pt': '6d867ed2dd2261d0f02a5e81d2b7f92802be30f3d36570fb7c4b18707649ef3f',
    'adapter_jit.pt': '159c5f691e55f68c68e2d98e287f4f87baa54e79759c04e06af3bdbc8f8edc98',
    'adapter_norm_stats.pt': '5adc634a7c5928b7d0a4ede15dbed4371b665b6972d706f32416350028e1956a',
}
JOINT_NAMES = tuple(name + '_joint' for name in (
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow'))
ACTION_JOINT_NAMES = JOINT_NAMES[:15]
ARM_JOINT_NAMES = JOINT_NAMES[15:]
DEFAULT_POSE = np.array([
    -.1, 0, 0, .3, -.2, 0, -.1, 0, 0, .3, -.2, 0, 0, 0, 0,
    .5, 0, .2, .3, .5, 0, -.2, .3], dtype=np.float64)
KP = np.array([150, 150, 150, 300, 80, 20] * 2 + [400] * 3
              + [80, 80, 40, 60] * 2, dtype=np.float64)
KD = np.array([2, 2, 2, 4, 2, 1] * 2 + [15] * 3 + [2, 2, 1, 1] * 2, dtype=np.float64)
TORQUE_LIMITS = np.array([88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50]
                        + [25] * 8, dtype=np.float64)
CONTROL_DT = .02
N_PROPRIO = 93
METADATA = {
    'source_repository': SOURCE_URL, 'source_revision': SOURCE_REVISION,
    'joint_names': JOINT_NAMES, 'action_joint_names': ACTION_JOINT_NAMES,
    'arm_joint_names': ARM_JOINT_NAMES, 'default_pose': dict(zip(JOINT_NAMES, DEFAULT_POSE.tolist())),
    'policy_hz': 50, 'native_physics_hz': 500, 'native_action_scale': .25,
    'proprio_size': 93, 'observation_size': 1043, 'extra_history_size': 2325,
    'history_frames': 10, 'extra_history_frames': 25,
    'heading_mode': 'absolute world yaw radians; no rate integration in policy',
    'native_home_freebase_xyz': [0., 0., 1.],
    'native_home_freebase_quat_wxyz': [1., 0., 0., 0.],
    'native_home_joint_pose': dict(zip(JOINT_NAMES, [
        -.2, 0, 0, .4, -.2, 0, -.2, 0, 0, .4, -.2, 0, 0, 0, 0,
        0, .4, 0, 1.2, 0, -.4, 0, 1.2])),
    'default_torso_height_command_m': .75,
    'initialization': 'Upstream resets native home at freebase Z=1 m, then releases under gravity. '
        'The 0.75 m torso command is not a prescribed freebase height. Target-model floor clearance '
        'and settling must be measured by the simulation driver.',
    'native_quirks_preserved': [
        'Stand flag depends only on abs(vx)<0.1, with one-observation heading-error lag.',
        'Pure lateral/heading commands keep the stand gate active; no yaw-rate observation.',
        'Upstream computes a masked dof-velocity copy but actually observes all raw dof velocities.',
        'Proprio history excludes the current observation; extra history includes it.',
    ],
    'extra_joints': 'No wrist or Dex3 output. Caller owns source-limited PD for those 20 joints.',
    'device_bridge': 'Pinned actor hard-codes one cuda:0 Device constant for a zeros[batch,105] allocation. '
        'For other requested devices, only that in-memory constant is relocated; checkpoint bytes are unchanged.',
}
for _array in (DEFAULT_POSE, KP, KD, TORQUE_LIMITS):
    _array.flags.writeable = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quat_to_euler(quat):
    """Preserve upstream scalar arithmetic and WXYZ convention."""
    euler = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    euler[0] = np.arctan2(sinr, cosr)
    sinp = 2 * (qw * qy - qz * qx)
    euler[1] = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    euler[2] = np.arctan2(siny, cosy)
    return euler


def load_normalization(path):
    """Only pinned upstream NumPy arrays, via Torch's restricted unpickler."""
    core = getattr(np, '_core', None)
    if core is None:
        core = np.core
    allowed = [(core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               np.ndarray, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        stats = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'input_mean': 12, 'input_std': 12, 'output_mean': 15, 'output_std': 15}
    if set(stats) != set(expected):
        raise ValueError('Unexpected normalization fields')
    for key, size in expected.items():
        value = np.asarray(stats[key])
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError('Invalid normalization array: ' + key)
        if key.endswith('_std') and not np.all(value > 0):
            raise ValueError('Nonpositive normalization standard deviation')
    return stats


def relocate_actor_device(actor, device):
    """Fix only the pinned trace's single zero-allocation Device constant."""
    requested = str(torch.device(device))
    found = []
    for module_name, module in actor.named_modules():
        for method in module._c._method_names():
            graph = module._c._get_method(method).graph
            def visit(block):
                for node in block.nodes():
                    if node.kind() == 'prim::Constant' and str(node.output().type()) == 'Device':
                        found.append((module_name, method, node))
                    for child in node.blocks():
                        visit(child)
            visit(graph)
    if len(found) != 1 or found[0][0:2] != ('', 'forward') or found[0][2].s('value') != 'cuda:0':
        raise ValueError('Pinned actor device-constant structure differs')
    node = found[0][2]
    if requested == 'cuda:0':
        return []
    node.s_('value', requested)
    return [{'module': '', 'method': 'forward', 'original': 'cuda:0', 'requested': requested,
             'scope': 'Device-only relocation of fixed zero-padding allocation; weights unchanged.'}]


class AMOPolicy:
    def __init__(self, weights_dir, device='cpu'):
        self.weights_dir = Path(weights_dir).resolve()
        self.device = str(device)
        self.input_bindings = {}
        for name, expected in WEIGHTS_SHA256.items():
            path = self.weights_dir / name
            actual = sha(path)
            if actual != expected:
                raise ValueError('AMO weight hash mismatch: ' + str(path))
            self.input_bindings[str(path)] = actual
        self.policy = torch.jit.load(str(self.weights_dir / 'amo_jit.pt'), map_location=self.device).eval()
        self.adapter = torch.jit.load(str(self.weights_dir / 'adapter_jit.pt'), map_location=self.device).eval()
        self.device_rewrites = relocate_actor_device(self.policy, self.device)
        for net in (self.policy, self.adapter):
            for parameter in net.parameters():
                parameter.requires_grad_(False)
        stats = load_normalization(self.weights_dir / 'adapter_norm_stats.pt')
        for name, value in stats.items():
            setattr(self, name, torch.tensor(value, device=self.device, dtype=torch.float32))
        self._model = None
        self.reset()

    def reset(self):
        """Clear policy history only; never set robot or task state."""
        self.last_action = np.zeros(23)
        self.gait_cycle = np.array([.25, .25])
        self.in_place_stand = True
        self.history = deque((np.zeros(N_PROPRIO) for _ in range(10)), maxlen=10)
        self.extra_history = deque((np.zeros(N_PROPRIO) for _ in range(25)), maxlen=25)
        self.steps = 0
        self.last_observation = None
        self.last_extra_history = None

    def _bind_model(self, model):
        if self._model is model:
            return
        if self._model is not None and self.steps:
            raise ValueError('Call reset() before switching robot model')
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if any(joint < 0 for joint in joint_ids):
            raise ValueError('Model lacks native AMO joint names')
        if any(model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE for joint in joint_ids):
            raise ValueError('AMO body joints must be scalar hinges')
        self.qpos_addresses = model.jnt_qposadr[joint_ids].copy()
        self.qvel_addresses = model.jnt_dofadr[joint_ids].copy()
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis < 0:
            raise ValueError('Model lacks named pelvis body')
        self.pelvis_id = pelvis
        self.sensor_pair = None
        for quaternion_name, gyro_name in [('orientation', 'angular-velocity'), ('imu_quat', 'imu_gyro')]:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                   for name in (quaternion_name, gyro_name)]
            if min(ids) < 0:
                continue
            if (model.sensor_type[ids[0]] != mujoco.mjtSensor.mjSENS_FRAMEQUAT
                    or model.sensor_type[ids[1]] != mujoco.mjtSensor.mjSENS_GYRO):
                raise ValueError('Named IMU sensors have wrong types')
            if model.sensor_refid[ids[0]] != -1:
                raise ValueError('AMO orientation must be measured in the world frame')
            for sensor in ids:
                if model.sensor_objtype[sensor] != mujoco.mjtObj.mjOBJ_SITE:
                    raise ValueError('Named IMU sensor must belong to a pelvis site')
                site = model.sensor_objid[sensor]
                if (model.site_bodyid[site] != pelvis
                        or not np.array_equal(model.site_quat[site], [1., 0., 0., 0.])):
                    raise ValueError('Named IMU frame differs from AMO pelvis frame')
            self.sensor_pair = (quaternion_name, gyro_name)
            break
        self._model = model

    def _state(self, model, data):
        self._bind_model(model)
        q = data.qpos[self.qpos_addresses].astype(np.float32)
        dq = data.qvel[self.qvel_addresses].astype(np.float32)
        if self.sensor_pair is not None:
            quat = data.sensor(self.sensor_pair[0]).data.astype(np.float32).copy()
            omega = data.sensor(self.sensor_pair[1]).data.astype(np.float32).copy()
        else:
            quat = data.xquat[self.pelvis_id].astype(np.float32).copy()
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_id, velocity, 0)
            omega = (data.xmat[self.pelvis_id].reshape(3, 3).T @ velocity[:3]).astype(np.float32)
        if (not all(np.isfinite(value).all() for value in (q, dq, quat, omega))
                or abs(float(np.linalg.norm(quat)) - 1) > 1e-4):
            raise ValueError('AMO state is nonfinite or quaternion is not unit WXYZ')
        return q, dq, quat, omega

    @torch.inference_mode()
    def step(self, model, data, vx, vy, heading, upper_body_targets=None, dt=CONTROL_DT,
             *, torso_height=.75, torso_yaw=0., torso_pitch=0., torso_roll=0.):
        if not np.isfinite(dt) or abs(float(dt) - CONTROL_DT) > 1e-9:
            raise ValueError('AMO requires exactly one inference per 0.02 seconds')
        command = np.asarray([vx, vy, heading, torso_height, torso_yaw, torso_pitch, torso_roll], dtype=np.float64)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError('Commands must be finite scalars')
        upper = {} if upper_body_targets is None else dict(upper_body_targets)
        if not set(upper) <= set(ARM_JOINT_NAMES) or not all(np.isfinite(float(v)) for v in upper.values()):
            raise ValueError('Upper-body targets must be finite values for the eight native arm joints')
        q, dq, quat, omega = self._state(model, data)
        rpy = quat_to_euler(quat)
        dyaw = np.remainder(rpy[2] - heading + np.pi, 2 * np.pi) - np.pi
        previous_stand = self.in_place_stand
        if previous_stand:
            dyaw = 0.
        gait = np.sin(self.gait_cycle * 2 * np.pi)
        adapter_input = np.concatenate([np.zeros(4), q[15:]])
        adapter_input[:4] = [torso_height, torso_yaw, torso_pitch, torso_roll]
        normalized = torch.tensor(adapter_input, device=self.device, dtype=torch.float32).unsqueeze(0)
        normalized = (normalized - self.input_mean) / (self.input_std + 1e-8)
        adapter_output = self.adapter(normalized.view(1, -1)) * self.output_std + self.output_mean
        prop = np.concatenate([omega * .25, rpy[:2], [np.sin(dyaw), np.cos(dyaw)],
                               q - DEFAULT_POSE, dq * .05, self.last_action, gait,
                               adapter_output.cpu().numpy().squeeze()])
        demo = np.zeros(17)
        demo[:8] = q[15:]
        demo[8:10] = [vx, vy]
        self.in_place_stand = np.abs(vx) < .1
        demo[11:14] = [torso_yaw, torso_pitch, torso_roll]
        demo[14:17] = torso_height
        obs = np.concatenate([prop, demo, np.zeros(3), np.array(self.history).flatten()])
        self.history.append(prop)
        self.extra_history.append(prop)
        extra = np.array(self.extra_history).flatten().copy()
        action = self.policy(torch.from_numpy(obs).float().unsqueeze(0).to(self.device),
                             torch.tensor(extra, dtype=torch.float).view(1, -1).to(self.device)).cpu().numpy().squeeze()
        if action.shape != (15,) or not np.isfinite(action).all():
            raise ValueError('AMO policy produced an invalid action')
        raw_action = np.clip(action, -40., 40.)
        self.last_action = np.concatenate([raw_action.copy(), (q - DEFAULT_POSE)[15:] / .25])
        targets = np.concatenate([raw_action * .25, np.zeros(8)]) + DEFAULT_POSE
        for index, name in enumerate(ARM_JOINT_NAMES, 15):
            if name in upper:
                targets[index] = float(upper[name])
        self.gait_cycle = np.remainder(self.gait_cycle + CONTROL_DT * 1.3, 1.)
        if self.in_place_stand and (abs(self.gait_cycle[0] - .25) < .05 or abs(self.gait_cycle[1] - .25) < .05):
            self.gait_cycle = np.array([.25, .25])
        if not self.in_place_stand and (abs(self.gait_cycle[0] - .25) < .05 and abs(self.gait_cycle[1] - .25) < .05):
            self.gait_cycle = np.array([.25, .75])
        self.steps += 1
        self.last_observation = obs.copy()
        self.last_extra_history = extra.copy()
        return {'joint_targets': dict(zip(JOINT_NAMES, targets.tolist())),
                'kp': dict(zip(JOINT_NAMES, KP.tolist())), 'kd': dict(zip(JOINT_NAMES, KD.tolist())),
                'torque_limits': dict(zip(JOINT_NAMES, TORQUE_LIMITS.tolist())),
                'diagnostics': {'policy_step': self.steps, 'heading_absolute_rad': float(heading),
                    'heading_error_observed_rad': float(dyaw), 'previous_stand_flag': bool(previous_stand),
                    'stand_flag': bool(self.in_place_stand), 'gait_cycle': self.gait_cycle.tolist(),
                    'raw_action': raw_action.tolist(), 'adapter_output': adapter_output.cpu().numpy().reshape(-1).tolist(),
                    'imu_source': list(self.sensor_pair) if self.sensor_pair else 'pelvis FK quaternion/local angular velocity',
                    'target_model_nq': model.nq, 'target_model_nu': model.nu,
                    'actor_device_rewrites': self.device_rewrites,
                    'emitted_joint_count': 23, 'source_motor_clipping_owned_by_caller': True}}
