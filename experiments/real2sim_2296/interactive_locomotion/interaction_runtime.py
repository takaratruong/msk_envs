"""Small, synchronous interaction coordinator. It emits directives, never physics.

The catalogue describes geometry, not executable skills. The registry starts empty.
An integrator must supply a hash-bound native-physics qualification for one exact
action/instance/stance domain, plus observations from the actual simulated state.
This module verifies that receipt's declared scope and file bindings; it cannot
establish the truth of a fabricated external physics receipt. A provider must bind
its actual scene composition to ``scene_fingerprint`` and derive contacts, support,
reachability and stable-controller flags from actual state, never reference poses.

Time is monotonic seconds supplied by the caller. SI units, world Z up. Hand IDs
are provider names. Opening IDs are exact catalogue joint paths; take/put identify
an explicit object and source/destination (support -> hand / hand -> support).
No navigation, IK, contact controller, object attachment, or family executor is
implemented here. Even a qualified executor is invoked by the external driver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping


Action = Literal["open", "close", "take", "put"]
OPENING_FAMILIES = frozenset({"vertical_hinged_door", "sliding_drawer", "sliding_rack",
                              "downward_hinged_door", "upward_lid"})
QUALIFICATION_GATES = frozenset({"actual_motor_driven", "domain_trials_passed",
    "no_state_assistance", "source_limits_preserved", "state_based_completion",
    "cancellation_and_stable_handback"})


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, n: int, name: str):
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(f"{name} must have {n} components")
    return tuple(_number(v, name) for v in value)


def _bool(value, name: str):
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")


def _name(value, name: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identifier")


def _angle_error(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@dataclass(frozen=True)
class Request:
    request_id: str
    action: Action
    source_id: str
    destination_id: str | None = None
    object_id: str | None = None
    # Opening fractions follow catalogue closed -> signed nominal open endpoint.
    target_fraction: float | None = None
    timeout_s: float = 30.0

    def __post_init__(self):
        _name(self.request_id, "request_id")
        _name(self.source_id, "source_id")
        if self.action not in {"open", "close", "take", "put"}:
            raise ValueError("Unsupported request action")
        if not 0 < _number(self.timeout_s, "timeout_s") <= 600:
            raise ValueError("timeout_s must be in (0, 600]")
        if self.action in {"open", "close"}:
            if self.destination_id is not None or self.object_id is not None:
                raise ValueError("Opening request identifies one exact joint only")
            f = self.fraction
            if not 0 <= f <= 1 or (self.action == "open" and f <= 0) or (self.action == "close" and f != 0):
                raise ValueError("open requires fraction (0,1]; close requires zero")
        else:
            _name(self.destination_id, "destination_id")
            _name(self.object_id, "object_id")
            if self.source_id == self.destination_id or self.target_fraction is not None:
                raise ValueError("Transfer requires distinct source/destination and no joint fraction")

    @property
    def fraction(self) -> float:
        return _number(self.target_fraction, "target_fraction") if self.target_fraction is not None else float(self.action == "open")


@dataclass(frozen=True)
class JointState:
    position_si: float
    velocity_si: float = 0.0
    locked: bool = False


@dataclass(frozen=True)
class HandState:
    held_object_id: str | None = None
    # Provider's measured intended/opposing-contact guard, not commanded fingers.
    secure_contact: bool = False


@dataclass(frozen=True)
class ObjectState:
    position_xyz: tuple[float, float, float]
    linear_speed_m_s: float
    angular_speed_rad_s: float
    support_id: str | None = None
    support_contact: bool = False


@dataclass(frozen=True)
class SupportState:
    available: bool


@dataclass(frozen=True)
class Observation:
    timestamp_s: float
    scene_fingerprint: str
    base_xyz: tuple[float, float, float]
    base_heading_rad: float
    robot_supported: bool
    robot_stable: bool
    joints: Mapping[str, JointState] = field(default_factory=dict)
    hands: Mapping[str, HandState] = field(default_factory=dict)
    objects: Mapping[str, ObjectState] = field(default_factory=dict)
    supports: Mapping[str, SupportState] = field(default_factory=dict)
    # Actual approach/IK/collision provider approval for named qualified skills.
    reachable_skill_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stance:
    center_xyz: tuple[float, float, float]
    radius_xy_m: float
    tolerance_z_m: float
    heading_rad: float
    heading_tolerance_rad: float

    def contains(self, obs: Observation) -> bool:
        return (math.hypot(obs.base_xyz[0] - self.center_xyz[0], obs.base_xyz[1] - self.center_xyz[1]) <= self.radius_xy_m
            and abs(obs.base_xyz[2] - self.center_xyz[2]) <= self.tolerance_z_m
            and _angle_error(obs.base_heading_rad, self.heading_rad) <= self.heading_tolerance_rad)


@dataclass(frozen=True)
class JointPrerequisite:
    joint_id: str
    minimum_fraction: float
    maximum_fraction: float


@dataclass(frozen=True)
class SkillDomain:
    action: Action
    source_id: str
    destination_id: str | None
    object_id: str | None
    family: str
    hand_id: str
    stance: Stance
    # Exact requested fraction tested; open and close require separate evidence.
    target_fraction: float | None = None
    start_fraction_range: tuple[float, float] = (0.0, 1.0)
    prerequisites: tuple[JointPrerequisite, ...] = ()
    # A put target is a tested region for the actual object's world center.
    destination_bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    minimum_lift_m: float = 0.025
    completion_dwell_s: float = 0.5
    entry_dwell_s: float = 0.3
    maximum_duration_s: float = 30.0
    joint_fraction_tolerance: float = 0.03
    maximum_joint_speed_si: float = 0.05
    maximum_object_linear_speed_m_s: float = 0.03
    maximum_object_angular_speed_rad_s: float = 0.1


@dataclass(frozen=True)
class SkillReport:
    request_id: str
    skill_id: str
    timestamp_s: float
    status: Literal["running", "succeeded", "failed"]
    safe_to_handback: bool = False
    detail: str = ""


@dataclass(frozen=True)
class Directive:
    request_id: str
    kind: Literal["navigate", "transition", "skill_start", "skill_monitor", "stable_handback", "complete"]
    state: str
    outcome: str | None
    reason: str
    payload: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Backend:
    skill_id: str
    domain: SkillDomain
    receipt_sha256: str
    executor: dict
    input_bindings: dict


class Catalogue:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        content = json.loads(self.path.read_text())
        if content.get("schema") != "g1-scene-interaction-catalog/v1":
            raise ValueError("Wrong catalogue schema")
        if content.get("units", {}).get("up_axis") != "Z" or content["units"].get("distance") != "metre" or content["units"].get("angle") != "radian":
            raise ValueError("Catalogue must use SI / world Z up")
        layers = content.get("used_usd_layers", [])
        if not layers or len({r["path"] for r in layers}) != len(layers):
            raise ValueError("Catalogue needs unique bound scene layers")
        for row in layers:
            if row.get("dirty") is not False or file_sha256(row["path"]) != row["sha256"]:
                raise ValueError("Catalogue scene layer is dirty or changed")
        self.scene_fingerprint = _json_hash(sorted((r["path"], r["sha256"]) for r in layers))
        self.input_bindings = {str(self.path): self.sha256, **{r["path"]: r["sha256"] for r in layers}}
        self.joints = {j["joint_path"]: copy.deepcopy(j) for j in content["joints"]}
        if len(self.joints) != len(content["joints"]):
            raise ValueError("Duplicate catalogue joint")

    def fraction(self, joint_id: str, state: JointState) -> float:
        row = self.joints[joint_id]
        q = _number(state.position_si, "joint position")
        low, high = row["limits_si"]["lower"], row["limits_si"]["upper"]
        if q < low - 1e-5 or q > high + 1e-5:
            raise ValueError("joint_outside_authored_limits")
        span = row["nominal_open_endpoint_si"] - row["nominal_closed_state_si"]
        if abs(span) <= 1e-10:
            return 0.0
        return (q - row["nominal_closed_state_si"]) / span


class Coordinator:
    """One active request. Empty registry is the production default.

    ``begin`` and ``update`` return a directive; an external driver performs it.
    Failure/cancel/timeout are latched. Stable handback never releases an object.
    Once a backend has started it must acknowledge safe handback, and the actual
    robot must remain supported/stable for the handback dwell. A missing/stale
    provider leaves handback pending; elapsed time cannot assert a safe state.
    """
    def __init__(self, catalogue: Catalogue, *, max_observation_age_s: float = 0.2,
                 handback_dwell_s: float = 0.3, object_loss_grace_s: float = 0.1):
        self.catalogue = catalogue
        for name, value in (("max_observation_age_s", max_observation_age_s),
                            ("handback_dwell_s", handback_dwell_s), ("object_loss_grace_s", object_loss_grace_s)):
            if not 0 < _number(value, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        self.max_age = max_observation_age_s
        self.handback_dwell = handback_dwell_s
        self.loss_grace = object_loss_grace_s
        self._registry: dict[str, _Backend] = {}
        self._request: Request | None = None
        self._seen: set[str] = set()
        self._state = "idle"
        self._last_now = -math.inf
        self._last_observation = -math.inf

    @property
    def registered_skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry))

    def register(self, skill_id: str, domain: SkillDomain, qualification_path: str | Path,
                 qualification_sha256: str) -> None:
        """Validate a concrete external executor's exact declared tested domain.

        Receipt schema g1-interaction-skill-qualification/v1: skill_id,
        catalogue_sha256, scene_fingerprint, domain (=asdict(domain)), executor
        {path,sha256,entrypoint}, gates (the six QUALIFICATION_GATES all true),
        and nonempty evidence [{path,sha256}]. Paths resolve at the receipt.
        Production should admit only independently reviewed native test receipts.
        The historical one-task fridge receipt is not this skill qualification.
        """
        if self._state not in {"idle", "complete"}:
            raise ValueError("Do not change backends during an active request")
        _name(skill_id, "skill_id")
        if skill_id in self._registry:
            raise ValueError("Duplicate skill ID")
        self._validate_domain(domain)
        path = Path(qualification_path).resolve()
        if file_sha256(path) != qualification_sha256:
            raise ValueError("Qualification hash mismatch")
        receipt = json.loads(path.read_text())
        if (receipt.get("schema") != "g1-interaction-skill-qualification/v1"
                or receipt.get("skill_id") != skill_id
                or receipt.get("catalogue_sha256") != self.catalogue.sha256
                or receipt.get("scene_fingerprint") != self.catalogue.scene_fingerprint
                or receipt.get("domain") != json.loads(json.dumps(asdict(domain)))):
            raise ValueError("Qualification scope/domain mismatch")
        if not all(receipt.get("gates", {}).get(k) is True for k in QUALIFICATION_GATES):
            raise ValueError("Missing or failed native skill qualification gate")
        executor = receipt.get("executor", {})
        _name(executor.get("entrypoint"), "executor entrypoint")
        evidence = receipt.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Native evidence must be nonempty")
        files = [executor] + evidence
        if len({str(Path(r["path"])) for r in files}) != len(files):
            raise ValueError("Qualification evidence rows must be unique")
        bindings = {str(path): qualification_sha256, **self.catalogue.input_bindings}
        for row in files:
            input_path = (path.parent / row["path"]).resolve()
            if not input_path.is_file() or file_sha256(input_path) != row["sha256"]:
                raise ValueError("Qualification input mismatch")
            bindings[str(input_path)] = row["sha256"]
        # No family or wildcard registration; each domain includes exact IDs.
        signature = (domain.action, domain.source_id, domain.destination_id, domain.object_id, domain.target_fraction)
        if any((b.domain.action, b.domain.source_id, b.domain.destination_id, b.domain.object_id, b.domain.target_fraction) == signature for b in self._registry.values()):
            raise ValueError("Ambiguous duplicate skill domain")
        resolved_executor = copy.deepcopy(executor)
        resolved_executor["path"] = str((path.parent / executor["path"]).resolve())
        self._registry[skill_id] = _Backend(skill_id, copy.deepcopy(domain), qualification_sha256, resolved_executor, bindings)

    def _validate_domain(self, d: SkillDomain):
        Request("domain_validation", d.action, d.source_id, d.destination_id, d.object_id,
                d.target_fraction, d.maximum_duration_s)
        _name(d.hand_id, "hand_id")
        if any("*" in x for x in (d.source_id, d.destination_id or "", d.object_id or "", d.hand_id)):
            raise ValueError("Wildcards are not tested instance domains")
        _vector(d.stance.center_xyz, 3, "stance center")
        for name in ("radius_xy_m", "tolerance_z_m", "heading_tolerance_rad"):
            if _number(getattr(d.stance, name), name) <= 0:
                raise ValueError("Stance tolerances must be positive")
        _number(d.stance.heading_rad, "stance heading")
        for name in ("completion_dwell_s", "entry_dwell_s", "minimum_lift_m", "joint_fraction_tolerance",
                     "maximum_joint_speed_si", "maximum_object_linear_speed_m_s", "maximum_object_angular_speed_rad_s"):
            if _number(getattr(d, name), name) <= 0:
                raise ValueError("Domain tolerances/dwells must be positive")
        if d.joint_fraction_tolerance > 0.1:
            raise ValueError("Joint completion tolerance exceeds 10% travel")
        lo, hi = _vector(d.start_fraction_range, 2, "start fraction range")
        if not 0 <= lo <= hi <= 1:
            raise ValueError("Invalid start fraction domain")
        if d.action in {"open", "close"}:
            row = self.catalogue.joints.get(d.source_id)
            if not row or row["family"] not in OPENING_FAMILIES or row["family"] != d.family or not row["enabled"] or row["source_retainer_locked"]:
                raise ValueError("Unsupported, disabled or retained mechanism domain")
            if d.target_fraction is None:
                raise ValueError("Qualification must state the exact tested target fraction")
            declared = {p.joint_id for p in d.prerequisites}
            if not {p["joint_path"] for p in row["prerequisites"]}.issubset(declared):
                raise ValueError("Catalogue prerequisites need tested numerical thresholds")
            if d.action == "close":
                dependents = {r["joint_path"] for r in self.catalogue.joints.values()
                              if any(p["joint_path"] == d.source_id for p in r["prerequisites"])}
                thresholds = {p.joint_id: p.maximum_fraction for p in d.prerequisites}
                if any(thresholds.get(j, 1.0) > 0.05 for j in dependents):
                    raise ValueError("Closing prerequisite doors requires tested retracted dependents")
        elif d.family != "object_transfer" or (d.action == "take" and d.destination_id != d.hand_id) or (d.action == "put" and d.source_id != d.hand_id):
            raise ValueError("Transfer domain must name its exact source/destination hand")
        if len({p.joint_id for p in d.prerequisites}) != len(d.prerequisites):
            raise ValueError("Duplicate prerequisite")
        for p in d.prerequisites:
            if p.joint_id not in self.catalogue.joints or not 0 <= _number(p.minimum_fraction, "minimum fraction") <= _number(p.maximum_fraction, "maximum fraction") <= 1:
                raise ValueError("Invalid tested prerequisite")
        if d.action == "put":
            if d.destination_bounds_xyz is None or len(d.destination_bounds_xyz) != 2:
                raise ValueError("Put needs tested world-center destination bounds")
            low, high = [_vector(v, 3, "destination bounds") for v in d.destination_bounds_xyz]
            if any(a >= b for a, b in zip(low, high)):
                raise ValueError("Empty destination region")

    def begin(self, request: Request, observation: Observation, now_s: float) -> Directive:
        if not isinstance(request, Request) or _number(now_s, "now_s") < self._last_now:
            raise ValueError("A typed request and monotonic clock are required")
        if request.request_id in self._seen:
            if self._request == request:
                return self.update(observation, now_s)
            raise ValueError("Request ID reuse with different or historical request")
        if self._state not in {"idle", "complete"}:
            raise ValueError("An active request must complete stable handback first")
        self._request = request
        self._seen.add(request.request_id)
        self._state = "preparing"
        self._backend: _Backend | None = None
        self._started_at = _number(now_s, "now_s")
        self._skill_started = False
        self._outcome = None
        self._reason = ""
        self._stable_since = self._completion_since = self._loss_since = None
        self._initial_object_z = None
        self._terminal_payload = None
        self._last_observation = -math.inf
        return self.update(observation, now_s)

    def cancel(self, observation: Observation, now_s: float, reason: str = "cancelled_by_caller") -> Directive:
        if self._state == "idle":
            raise ValueError("No active request")
        if self._state != "complete":
            self._handback("cancelled", reason)
        return self.update(observation, now_s)

    def _handback(self, outcome: str, reason: str):
        if self._state != "handback" or (self._outcome in {"succeeded", "already_satisfied"} and outcome not in {"succeeded", "already_satisfied"}):
            self._state = "handback"
            self._stable_since = None
            self._outcome, self._reason = outcome, reason

    def _observation_error(self, o: Observation, now: float) -> str | None:
        try:
            stamp = _number(o.timestamp_s, "observation timestamp")
            if stamp > now + 1e-9 or now - stamp > self.max_age or stamp < self._last_observation:
                return "stale_or_out_of_order_observation"
            if math.isfinite(self._last_observation) and stamp - self._last_observation > self.max_age + 1e-9:
                # A fresh endpoint does not prove continuous stable/contact state.
                self._last_observation = stamp
                return "observation_coverage_gap"
            if o.scene_fingerprint != self.catalogue.scene_fingerprint:
                return "scene_binding_mismatch"
            _vector(o.base_xyz, 3, "base position")
            _number(o.base_heading_rad, "base heading")
            _bool(o.robot_supported, "robot supported")
            _bool(o.robot_stable, "robot stable")
            for v in o.joints.values():
                _number(v.position_si, "joint position")
                _number(v.velocity_si, "joint velocity")
                _bool(v.locked, "joint lock")
            for v in o.hands.values():
                _bool(v.secure_contact, "secure contact")
                if v.held_object_id is not None:
                    _name(v.held_object_id, "held object")
                    if v.held_object_id not in o.objects:
                        return "held_object_missing"
                elif v.secure_contact:
                    return "contradictory_empty_hand_contact"
            for v in o.objects.values():
                _vector(v.position_xyz, 3, "object position")
                if _number(v.linear_speed_m_s, "object speed") < 0 or _number(v.angular_speed_rad_s, "object angular speed") < 0:
                    return "negative_object_speed"
                _bool(v.support_contact, "support contact")
                if v.support_contact and (v.support_id not in o.supports or not o.supports[v.support_id].available):
                    return "support_contact_without_available_support"
            for v in o.supports.values():
                _bool(v.available, "support available")
            self._last_observation = stamp
        except (TypeError, ValueError, AttributeError):
            return "invalid_observation"
        return None

    def _opening_state(self, o: Observation):
        r = self._request
        row = self.catalogue.joints.get(r.source_id)
        if row is None or row["family"] not in OPENING_FAMILIES or not row["enabled"]:
            raise ValueError("unsupported_mechanism")
        if r.source_id not in o.joints:
            raise ValueError("missing_joint_observation")
        state = o.joints[r.source_id]
        fraction = self.catalogue.fraction(r.source_id, state)
        if (row["source_retainer_locked"] or state.locked) and not (r.action == "close" and abs(fraction) <= 0.03):
            raise ValueError("locked_mechanism")
        return row, state, fraction

    def _secure(self, o: Observation, hand_id: str, object_id: str) -> bool:
        hand = o.hands.get(hand_id)
        return bool(hand and hand.held_object_id == object_id and hand.secure_contact)

    def _preconditions(self, o: Observation, *, entry: bool) -> str | None:
        r, b = self._request, self._backend
        d = b.domain
        if b.skill_id not in o.reachable_skill_ids:
            return "stance_or_skill_not_currently_reachable"
        hand = o.hands.get(d.hand_id)
        if hand is None:
            return "missing_hand_observation"
        for p in d.prerequisites:
            state = o.joints.get(p.joint_id)
            if state is None:
                return "missing_prerequisite_state"
            try:
                f = self.catalogue.fraction(p.joint_id, state)
            except ValueError as error:
                return str(error)
            if not p.minimum_fraction <= f <= p.maximum_fraction:
                return "prerequisite_joint_not_ready"
        if r.action in {"open", "close"}:
            try:
                _, _, fraction = self._opening_state(o)
            except ValueError as error:
                return str(error)
            if entry and not d.start_fraction_range[0] <= fraction <= d.start_fraction_range[1]:
                return "outside_tested_initial_joint_domain"
            if hand.held_object_id is not None:
                return "interaction_hand_occupied"
        else:
            obj = o.objects.get(r.object_id)
            if obj is None:
                return "missing_object_observation"
            support_id = r.source_id if r.action == "take" else r.destination_id
            if support_id not in o.supports or not o.supports[support_id].available:
                return "source_or_destination_support_unavailable"
            if entry and r.action == "take":
                if hand.held_object_id is not None:
                    return "interaction_hand_occupied"
                if not obj.support_contact or obj.support_id != r.source_id:
                    return "object_not_supported_at_requested_source"
                if any(h.held_object_id == r.object_id for h in o.hands.values()):
                    return "source_object_already_held"
            if entry and r.action == "put" and not self._secure(o, d.hand_id, r.object_id):
                return "requested_object_not_securely_held"
            if not entry:
                if any(k != d.hand_id and h.held_object_id == r.object_id for k, h in o.hands.items()):
                    return "object_held_by_untested_hand"
                secure = self._secure(o, d.hand_id, r.object_id)
                expected_support = obj.support_contact and obj.support_id == support_id
                if secure or expected_support:
                    self._loss_since = None
                elif self._loss_since is None:
                    self._loss_since = o.timestamp_s
                elif o.timestamp_s - self._loss_since >= self.loss_grace:
                    return "object_lost_or_supported_at_wrong_location"
                if hand.held_object_id not in {None, r.object_id}:
                    return "unexpected_object_in_interaction_hand"
        return None

    def _goal_observed(self, o: Observation) -> bool:
        r, d = self._request, self._backend.domain
        if r.action in {"open", "close"}:
            _, state, fraction = self._opening_state(o)
            return abs(fraction - r.fraction) <= d.joint_fraction_tolerance and abs(state.velocity_si) <= d.maximum_joint_speed_si
        obj = o.objects[r.object_id]
        slow = obj.linear_speed_m_s <= d.maximum_object_linear_speed_m_s and obj.angular_speed_rad_s <= d.maximum_object_angular_speed_rad_s
        if r.action == "take":
            return (self._secure(o, d.hand_id, r.object_id) and not obj.support_contact and
                    obj.position_xyz[2] - self._initial_object_z >= d.minimum_lift_m and slow)
        lo, hi = d.destination_bounds_xyz
        return (obj.support_contact and obj.support_id == r.destination_id and slow
                and all(a <= v <= z for a, v, z in zip(lo, obj.position_xyz, hi))
                and all(h.held_object_id != r.object_id for h in o.hands.values()))

    def update(self, observation: Observation, now_s: float, report: SkillReport | None = None) -> Directive:
        if self._request is None:
            raise ValueError("No request: call begin first")
        now = _number(now_s, "now_s")
        if now < self._last_now:
            raise ValueError("Coordinator clock moved backwards")
        self._last_now = now
        if self._state == "complete":
            return self._directive("complete", {})
        error = self._observation_error(observation, now)
        if error:
            self._handback("failed", error)
            self._stable_since = None
            return self._directive("stable_handback", {"observation_valid": False,
                "stop_navigation": True, "preserve_last_verified_grip": True,
                "release_object": False, "requires_fresh_supported_state": True})
        r, o = self._request, observation
        valid_report = None
        if report is not None:
            try:
                report_time = _number(report.timestamp_s, "report timestamp")
            except (ValueError, AttributeError):
                report_time = -math.inf
            if (not isinstance(report, SkillReport) or not self._skill_started or report.request_id != r.request_id or report.skill_id != self._backend.skill_id
                    or report.status not in {"running", "succeeded", "failed"}
                    or type(report.safe_to_handback) is not bool
                    or report_time > now or now - report_time > self.max_age):
                self._handback("failed", "invalid_or_stale_executor_report")
            else:
                valid_report = report
                if report.status == "failed":
                    self._handback("failed", "executor_failed: " + report.detail)
        if self._state != "handback" and now - self._started_at >= min(r.timeout_s, self._backend.domain.maximum_duration_s if self._backend else r.timeout_s):
            self._handback("timed_out", "request_deadline_exceeded")
        if self._state != "handback" and not o.robot_supported:
            self._handback("failed", "robot_support_lost")
        if self._state == "preparing" and self._backend is None:
            if r.action in {"open", "close"}:
                try:
                    _, state, fraction = self._opening_state(o)
                    if abs(fraction - r.fraction) <= 0.03 and abs(state.velocity_si) <= 0.05:
                        self._handback("already_satisfied", "fresh_actual_joint_already_at_target")
                except ValueError as error:
                    self._handback("failed", str(error))
            if self._state != "handback":
                candidates = [b for b in self._registry.values() if (b.domain.action, b.domain.source_id, b.domain.destination_id, b.domain.object_id,
                    b.domain.target_fraction) == (r.action, r.source_id, r.destination_id, r.object_id,
                        r.fraction if r.action in {"open", "close"} else None)]
                if not candidates:
                    self._handback("failed", "no_qualified_executor_for_exact_domain")
                else:
                    self._backend = candidates[0]
                    if r.object_id in o.objects:
                        self._initial_object_z = o.objects[r.object_id].position_xyz[2]
        if self._state in {"preparing", "executing"}:
            error = self._preconditions(o, entry=self._state == "preparing")
            if error:
                self._handback("failed", error)
        if self._state == "preparing":
            b = self._backend
            if not b.domain.stance.contains(o):
                self._stable_since = None
                return self._directive("navigate", {"skill_id": b.skill_id, "stance": asdict(b.domain.stance),
                    "requires_actual_collision_aware_navigation": True, "preserve_grip": True})
            if not o.robot_stable:
                self._stable_since = None
            elif self._stable_since is None:
                self._stable_since = o.timestamp_s
            if self._stable_since is None or o.timestamp_s - self._stable_since < b.domain.entry_dwell_s:
                return self._directive("transition", {"skill_id": b.skill_id, "stance": asdict(b.domain.stance), "wait_for_stable_actual_state": True})
            try:
                bindings_current = all(file_sha256(path) == digest for path, digest in b.input_bindings.items())
            except OSError:
                bindings_current = False
            if not bindings_current:
                self._handback("failed", "registered_qualification_or_input_changed")
            else:
                self._state = "executing"
                self._skill_started = True
                if r.object_id in o.objects:
                    self._initial_object_z = o.objects[r.object_id].position_xyz[2]
                return self._directive("skill_start", {"skill_id": b.skill_id, "executor": copy.deepcopy(b.executor),
                    "qualification_sha256": b.receipt_sha256, "request": asdict(r), "domain": asdict(b.domain),
                    "observation_timestamp_s": o.timestamp_s, "input_bindings_sha256": copy.deepcopy(b.input_bindings),
                    "driver_must_recheck_bindings_at_execution": True})
        if self._state == "executing":
            if not self._backend.domain.stance.contains(o):
                self._handback("failed", "left_tested_skill_stance_domain")
            elif self._goal_observed(o):
                if self._completion_since is None:
                    self._completion_since = o.timestamp_s
                if (o.timestamp_s - self._completion_since >= self._backend.domain.completion_dwell_s
                        and valid_report is not None and valid_report.status == "succeeded"):
                    self._handback("succeeded", "sustained_actual_postcondition_and_executor_complete")
            else:
                self._completion_since = None
            if self._state == "executing":
                return self._directive("skill_monitor", {"skill_id": self._backend.skill_id,
                    "actual_postcondition_since_s": self._completion_since, "executor_success_is_not_sufficient": True})
        if self._state == "handback":
            if self._outcome in {"succeeded", "already_satisfied"}:
                try:
                    if self._outcome == "already_satisfied":
                        _, state, fraction = self._opening_state(o)
                        intact = abs(fraction - r.fraction) <= 0.03 and abs(state.velocity_si) <= 0.05
                    else:
                        intact = self._goal_observed(o)
                except (KeyError, ValueError):
                    intact = False
                if not intact:
                    self._handback("failed", "postcondition_lost_during_handback")
            safe = not self._skill_started or (valid_report is not None and valid_report.safe_to_handback)
            stable = o.robot_supported and o.robot_stable and safe
            if not stable:
                self._stable_since = None
            elif self._stable_since is None:
                self._stable_since = o.timestamp_s
            if self._stable_since is not None and o.timestamp_s - self._stable_since >= self.handback_dwell:
                self._state = "complete"
                return self._directive("complete", {"stable_handback_observed": True,
                    "held_objects": {k: v.held_object_id for k, v in o.hands.items() if v.held_object_id is not None}})
            return self._directive("stable_handback", {"observation_valid": True, "stop_navigation": True,
                "request_executor_safe_stop": self._skill_started, "preserve_grip": True,
                "release_object": False, "stable_supported_state": stable})
        raise RuntimeError("Unreachable coordinator state")

    def _directive(self, kind: str, payload: dict) -> Directive:
        if kind == "complete":
            if self._terminal_payload is None:
                self._terminal_payload = copy.deepcopy(payload)
            payload = copy.deepcopy(self._terminal_payload)
        return Directive(self._request.request_id, kind, self._state, self._outcome, self._reason, payload)
