"""Explicit joystick frontend for the frozen AMO policy; no physics state writes.

Body-frame velocities are commands, not pose-reference derivatives. Yaw rate is
integrated to AMO's absolute world-heading command from a supplied actual initial
heading. Zero rate holds that virtual target. There is no root-position feedback.

``native`` preserves AMO's vx-only stand behavior. ``any_motion`` is a separately
labeled gate experiment: after native inference, replace only the next stand flag
and next gait phase using commanded translation or an outstanding heading error.
Native inference computes
its current action before those updates, so this preserves the one-observation
gate lag and avoids copying or modifying the frozen observation/network code.
Native checkpoints, inference formulas, gains and torque caps are unchanged.
Subsequent network inputs/actions change through the declared gate and gait state.
"""
import math

import numpy as np


class AMOHeadingGate:
    """Same absolute-heading API as AMOPolicy; optional gate-only intervention.

    Any-motion means |vx|>=.1 m/s, |vy|>=.1 m/s, or wrapped measured absolute
    heading error>=.05 rad. The final heading can finish after translation stops.
    This is an explicit controller variant, not upstream native behavior.
    """
    def __init__(self, policy, *, mode="native", heading_gate_threshold_rad=.05):
        if mode not in {"native", "any_motion"}:
            raise ValueError("Unknown stand-gate mode")
        if not np.isfinite(heading_gate_threshold_rad) or not 0 < heading_gate_threshold_rad <= .2:
            raise ValueError("Heading threshold must be in (0,.2] rad")
        self.policy = policy
        self.mode = mode
        self.heading_threshold = float(heading_gate_threshold_rad)
        self.input_bindings = dict(policy.input_bindings)

    def reset(self):
        self.policy.reset()

    def step(self, model, data, vx, vy, heading, upper_body_targets=None, dt=.02, **torso_commands):
        values = np.asarray([vx, vy, heading, dt], dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or abs(dt-.02)>1e-9:
            raise ValueError("Finite absolute-heading command and 20 ms tick required")
        _, _, quat, _ = self.policy._state(model, data)
        qw,qx,qy,qz = quat
        actual_heading = np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        heading_error = np.remainder(actual_heading-heading+np.pi,2*np.pi)-np.pi
        previous_gait = self.policy.gait_cycle.copy()
        result = self.policy.step(model,data,vx,vy,heading,upper_body_targets=upper_body_targets,
                                  dt=dt,**torso_commands)
        diagnostic=result['diagnostics']
        native_stand=bool(self.policy.in_place_stand)
        native_gait=self.policy.gait_cycle.copy()
        if self.mode=='any_motion':
            stand=abs(vx)<.1 and abs(vy)<.1 and abs(heading_error)<self.heading_threshold
            gait=np.remainder(previous_gait+dt*1.3,1.)
            if stand and (abs(gait[0]-.25)<.05 or abs(gait[1]-.25)<.05):
                gait=np.array([.25,.25])
            if not stand and (abs(gait[0]-.25)<.05 and abs(gait[1]-.25)<.05):
                gait=np.array([.25,.75])
            self.policy.in_place_stand=bool(stand)
            self.policy.gait_cycle=gait
            diagnostic['stand_flag']=bool(stand)
            diagnostic['gait_cycle']=gait.tolist()
        diagnostic['heading_gate']={
            'mode':self.mode,'heading_error_for_gate_rad':float(heading_error),
            'heading_gate_threshold_rad':self.heading_threshold,
            'translation_gate_threshold_m_s':.1,
            'native_vx_only_stand_candidate':native_stand,
            'native_vx_only_next_gait_candidate':native_gait.tolist(),
            'pending_heading_error_keeps_alternate_gate_moving':True,
            'inference_code_and_source_gains_caps_unchanged':True}
        return result


class AMOJoystick:
    def __init__(self, policy, *, mode="native", vx_limit=.5, vy_limit=.4,
                 yaw_rate_limit=.6, heading_gate_threshold_rad=.05):
        if mode not in {"native", "any_motion"}:
            raise ValueError("Unknown stand-gate mode")
        values = [vx_limit, vy_limit, yaw_rate_limit]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Command limits and gate thresholds must be positive and finite")
        self.policy = policy
        self.gate = AMOHeadingGate(policy,mode=mode,heading_gate_threshold_rad=heading_gate_threshold_rad)
        self.mode = mode
        self.limits = np.asarray([vx_limit, vy_limit, yaw_rate_limit], dtype=float)
        self.heading = None

    def reset(self, actual_heading_rad, *, reset_policy=True):
        if not np.isfinite(actual_heading_rad):
            raise ValueError("Initial measured heading must be finite")
        self.heading = math.atan2(math.sin(actual_heading_rad), math.cos(actual_heading_rad))
        if reset_policy:
            self.policy.reset()

    def step(self, model, data, vx, vy, yaw_rate, upper_body_targets=None, dt=.02):
        if self.heading is None:
            raise ValueError("Reset joystick heading from actual state before use")
        if not np.isfinite(dt) or abs(dt - .02) > 1e-9:
            raise ValueError("Frozen AMO inference interval is 20 ms")
        requested = np.asarray([vx, vy, yaw_rate], dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError("Joystick commands must be three finite scalars")
        command = np.clip(requested, -self.limits, self.limits)
        self.heading = math.atan2(math.sin(self.heading + command[2] * dt),
                                  math.cos(self.heading + command[2] * dt))
        result = self.gate.step(model, data, float(command[0]), float(command[1]),
                                self.heading, upper_body_targets=upper_body_targets, dt=dt)
        diagnostic = result["diagnostics"]
        diagnostic["joystick"] = {
            "mode": self.mode, "requested_body_vx_vy_yaw_rate": requested.tolist(),
            "clipped_body_vx_vy_yaw_rate": command.tolist(),
            "integrated_absolute_heading_rad": self.heading,
            "stand_gate": "AMOHeadingGate; see heading_gate diagnostics",
            "heading_frontend": "Rate integration from actual initial heading; zero rate holds virtual target",
        }
        return result
