"""Common live-skill facade around the exact previously tested fridge controller.

Only clocks, named motor indices and status presentation are adapted. The frozen
controller still owns private arm/finger feedback and all local opening/release
guards. The caller owns body-policy history, torque clipping, physics steps,
full collision/stability acceptance and eventual locomotion handback.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


FROZEN_ADAPTER_SHA256 = "f39c98c9c2aa72b365475f421294ef81af39da24fc9800f25e5cb596d37d99cb"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _ArmView:
    """Read-only, named-index view; never substitutes a new arm solver."""
    def __init__(self, owner):
        self.owner = owner
        self.names = ["right_" + n + "_joint" for n in (
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw")]
        self.idx = np.asarray([owner.i.index[n] for n in self.names], int)
        self.act = owner.i.motors[self.idx].copy()
        self.sq, self.sv = owner.i.q[self.idx].copy(), owner.i.v[self.idx].copy()
        self.q = np.asarray([owner.r.joint(n).qposadr[0] for n in self.names], int)

    @property
    def target(self):
        s = self.owner.native
        target = s.arm.target if s.started else None
        return self.owner.d.qpos[self.sq].copy() if target is None else target.copy()

    @property
    def error(self):
        diagnostics = self.owner.native.arm.diagnostics if self.owner.native.started else {}
        return np.asarray([diagnostics.get("ik_target_position_error_m", 0.),
                           diagnostics.get("ik_target_rotation_error_rad", 0.)], float)


class FridgeAdapter:
    """Constructor matches DrawerSkill; hinge positions are explicitly radians.

    descriptor requires ``frozen_adapter`` and ``bundle_dir``. ``at`` accepts current
    and future SIMULATION seconds, translating all values into source time before
    the frozen bounded reference clamps them. Other calls use actual data.time.
    Returned arm torque requires the ORIGINAL ``arm_weight`` blend with the body
    controller. All 14 returned finger torques replace their named motor slots.
    """
    def __init__(self, model, data, robot, interface, descriptor):
        self.m, self.d, self.r, self.i = model, data, robot, interface
        self.desc = dict(descriptor)
        adapter = Path(descriptor["frozen_adapter"]).resolve()
        bundle = Path(descriptor["bundle_dir"]).resolve()
        if sha256(adapter) != FROZEN_ADAPTER_SHA256:
            raise ValueError("Require the exact full-retraction frozen fridge adapter")
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("adapter_sha256") != FROZEN_ADAPTER_SHA256:
            raise ValueError("Fridge bundle adapter identity mismatch")
        if (manifest.get("clock", {}).get("source_start_seconds") != 9.8 or
                manifest.get("clock", {}).get("source_end_seconds") != 26.3):
            raise ValueError("Require the full 9.8..26.3 second opening/retraction seed")
        spec = importlib.util.spec_from_file_location("live_frozen_fridge_" + sha256(manifest_path)[:12], adapter)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.FridgeOpenSkill(model, data, robot, bundle_dir=bundle)
        self.input_bindings = dict(self.native.input_bindings)
        self.input_bindings[str(Path(__file__).resolve())] = sha256(__file__)
        self.input_bindings[str(adapter)] = sha256(adapter)
        self.domain = manifest.get("new_scene_domain", {})
        self.scope = "Frozen fridge controller facade; new-scene physical qualification belongs to the caller"
        self.joint, self.bid = self.native.door_jid, self.native.door_bid
        self.dq, self.dv = self.native.door_q, self.native.door_v
        if descriptor.get("joint_name", "task_fridge_hinge") != "task_fridge_hinge" or descriptor.get("body_name", "task_fridge_door") != "task_fridge_door":
            raise ValueError("This frozen skill is qualified only for its named fridge hinge/body")
        if "rail_geom_names" in descriptor and set(descriptor["rail_geom_names"]) != set(manifest["interaction"]["rail_geom_names"]):
            raise ValueError("Fridge rail names differ from the frozen domain")
        self.goal = float(np.deg2rad(60.))
        self.retained_minimum = float(np.deg2rad(50.))
        self.minimum_retained_position = self.retained_minimum
        self.position_units = "rad"
        self.arm = _ArmView(self)
        self.fi = np.asarray([interface.index[n] for n in self.native.hand_names], int)
        self.fa = interface.motors[self.fi].copy()
        self.fq, self.fv = interface.q[self.fi].copy(), interface.v[self.fi].copy()
        if not (np.array_equal(self.fa, self.native.hand_act) and
                np.array_equal(self.fq, self.native.hand_q) and
                np.array_equal(self.fv, self.native.hand_v)):
            raise ValueError("Caller interface disagrees with named frozen hand mapping")
        if len(self.fi) != 14 or len(set(self.fi.tolist())) != 14:
            raise ValueError("Require all 14 interleaved Dex3 motors")
        self.started = False
        self._observed = None
        self._goals = None
        self._last_overrides = None
        self.failure = None
        self.two_digit_seconds = 0.
        self.max_displacement = float(data.qpos[self.dq])

    def start(self, now):
        if self.started:
            raise RuntimeError("Create a fresh facade for each interaction; do not reset an active skill")
        result = self.native.start(now)
        if not np.array_equal(np.asarray(self.native.arm.act, int), self.arm.act):
            raise ValueError("Caller arm motor mapping disagrees with the frozen controller")
        self.begun = float(now)
        self.started = True
        self.stance = self.i.measured(self.d).copy()
        self.reference_q = self.at(now)[0][0].copy()
        self.t = self.begun + self.native.reference.t - self.native.source_start
        self._last_observe = float(now)
        return result

    def at(self, times):
        if not self.started:
            raise RuntimeError("Start the skill before requesting simulation-time references")
        values = np.atleast_1d(np.asarray(times, float))
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Reference times must be a finite scalar or one-dimensional array")
        # Do not first call source_time(), which clips and would lose the frozen
        # reference's zero-velocity rule for lookahead beyond the clip endpoints.
        return self.native.reference.at(self.native.source_start + values - self.begun)

    def phase_at(self, now):
        return self.native.reference.phase_at(self.native.source_time(float(now)))

    def update(self, now):
        self._goals = self.native.update(now)
        self.reference_q = self._goals["qpos"].copy()
        return self._goals

    @property
    def arm_weight(self):
        return float(self.native.arm.weight) if self.started else 0.

    def motor_overrides(self):
        result = self.native.motor_overrides(float(self.d.time), dt=float(self.m.opt.timestep))
        self._last_overrides = result
        return (result["arm_actuator_ids"].copy(), result["arm_torque"].copy(),
                result["hand_actuator_ids"].copy(), result["hand_torque"].copy())

    def observe(self, now):
        observed = self.native.observe(now)
        dt = float(now) - self._last_observe
        self._last_observe = float(now)
        if len(observed["distal_rail_digits"]) >= 2:
            self.two_digit_seconds += min(dt, self.native.PHYSICS_DT)
        self._observed = observed
        self.max_displacement = max(self.max_displacement, observed["door_angle"])
        if not observed["coverage_valid"]:
            self.failure = "Missing native 500 Hz contact observation coverage"
        elif not observed["external_forces_zero"]:
            self.failure = "Native opening observed external applied forces"
        return self.state()

    @property
    def phase(self):
        return "not_started" if not self.started else self.phase_at(float(self.d.time))

    @property
    def done(self):
        return bool(self._observed and self._observed["handback_ready"])

    @property
    def success(self):
        return bool(self._observed and self._observed["opening_success_latched"])

    @property
    def clear_dwell(self):
        return float(self.native.hand_clear_seconds) if self.started else 0.

    @property
    def open_dwell(self):
        return float(self.native.open_hold_seconds) if self.started else 0.

    @property
    def contact_digits(self):
        return {n.removesuffix("_link") for n in (self._observed or {}).get("distal_rail_digits", [])}

    @property
    def pos(self):
        return self.native.arm.goal_pos.copy() if self._goals is not None else self.d.xpos[self.native.wrist_bid].copy()

    @property
    def rot(self):
        return self.native.arm.goal_rot.copy() if self._goals is not None else self.d.xmat[self.native.wrist_bid].reshape(3, 3).copy()

    def state(self):
        n = self._observed or {}
        position, velocity = float(self.d.qpos[self.dq]), float(self.d.qvel[self.dv])
        return {"phase": self.phase, "kind": "fridge", "joint_position": position,
                "joint_velocity": velocity, "goal": self.goal, "joint_unit": "rad", "position_units": "rad",
                "door_angle": position, "door_velocity": velocity,
                "retained_minimum_rad": self.retained_minimum,
                "contact_N": float(n.get("distal_rail_normal_force_N", 0.)),
                "digits": sorted(self.contact_digits),
                "digits_seen": sorted(x.removesuffix("_link") for x in n.get("digits_seen", [])),
                "contact_seconds": float(n.get("contact_seconds", 0.)),
                "two_digit_seconds": self.two_digit_seconds, "open_dwell": self.open_dwell,
                "clear_dwell": self.clear_dwell, "opening_achieved": self.success,
                "success": self.success, "done": self.done, "handback_ready": self.done,
                "failure": self.failure, "max_angle_rad": self.max_displacement,
                "finger_open_error_rad": n.get("right_finger_open_error_max_rad"),
                "ik_error": self.arm.error.tolist(), "arm_weight": self.arm_weight,
                "scope": self.scope, "native": dict(n)}
