"""Local input/view transport; it never imports or modifies a physics model.

The runtime owns physical preconditions, heading re-anchoring on inactive input,
and autonomous skill completion. Space/blur only stop locomotion input. Browser
and scripted clients share /api/session and /api/input. All age calculations use
the server monotonic clock, including short-lived input tickets.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit


class InputError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class LiveTransport:
    """Thread-safe singleton operator lease and latest-frame HTTP server.

    ``start`` returns self. ``poll_command`` consumes a pending F edge exactly
    once. ``publish`` replaces JSON status; jpeg=None retains the previous frame.
    No command survives a 350 ms input/ticket expiry or inactive/blur packet.
    F after expiry/re-claim requires an observed release before another press.
    """

    def __init__(self, port: int = 8782, deadman_s: float = .35,
                 host: str = "127.0.0.1", ui_dir: str | Path | None = None, require_view: bool = False,
                 steer_with_look: bool = False):
        if host != "127.0.0.1":
            raise ValueError("This application is localhost-only (127.0.0.1)")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("invalid port")
        if not math.isfinite(deadman_s) or not .05 <= deadman_s <= .35:
            raise ValueError("deadman_s must be 0.05..0.35 seconds")
        if type(require_view) is not bool:
            raise ValueError("require_view must be bool")
        self.require_view = require_view
        if type(steer_with_look) is not bool or (steer_with_look and not require_view):
            raise ValueError("steer_with_look requires camera-bound input")
        self.steer_with_look = steer_with_look
        self._view_session = secrets.token_urlsafe(18)
        self._look = {"yaw": 0., "pitch": -.15, "revision": 0}
        if self.steer_with_look:
            self._look["mode"] = "third_person"
        self._views = OrderedDict()
        self.host, self.port, self.deadman_s = host, port, float(deadman_s)
        self.ui_dir = Path(ui_dir or Path(__file__).with_name("live_ui"))
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._server = self._thread = None
        self._running = False
        self._token = secrets.token_urlsafe(32)
        self._tickets: OrderedDict[str, float] = OrderedDict()
        self._client_id = None
        self._seq = -1
        self._received = self._ticket_issued = None
        self._active = False
        self._axes = (0., 0., 0.)
        self._f_down = self._f_armed = False
        self._pending_edge = None
        self._edge_id = 0
        self._selection_id = None
        self._reason = "no operator"
        self._status = {"mode": "waiting", "message": "Waiting for physics runtime",
                        "candidate": {"eligible": False, "reason": "No runtime state"}}
        self._status_time = None
        self._jpeg = None
        self._frame_seq = 0
        self._frame_time = None
        self._accepted = self._rejected = 0

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    def start(self):
        with self._lock:
            if self._running:
                return self
            owner = self

            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def setup(self):
                    super().setup()
                    self.connection.settimeout(2.)
                    # Small control acknowledgements follow HTTP headers in a
                    # second write. Disable Nagle buffering so they do not wait
                    # for the peer's delayed ACK (about 40 ms on localhost).
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                def log_message(self, *_):
                    pass  # tokens and arbitrary client content never enter logs

                def _headers(self, code, mime, length=None):
                    self.send_response(code)
                    self.send_header("Content-Type", mime)
                    if length is not None:
                        self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
                    self.end_headers()

                def _json(self, code, value):
                    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
                    self._headers(code, "application/json", len(data))
                    self.wfile.write(data)

                def _origin(self, post=False):
                    if self.headers.get("Host") != f"127.0.0.1:{owner.port}":
                        raise InputError("invalid local Host", 403)
                    origin = self.headers.get("Origin")
                    if origin and origin != owner.url.rstrip("/"):
                        raise InputError("cross-origin request rejected", 403)
                    if post and self.headers.get("Sec-Fetch-Site") == "cross-site":
                        raise InputError("cross-site request rejected", 403)

                def do_GET(self):
                    try:
                        self._origin()
                        path = urlsplit(self.path).path
                        if path == "/api/bootstrap":
                            result = owner.state_snapshot()
                            result.update(token=owner._token, protocol="g1-live-input/v1")
                            self._json(200, result)
                        elif path == "/api/state":
                            self._json(200, owner.state_snapshot())
                        elif path == "/api/view":
                            self._json(200, owner.view_snapshot())
                        elif path == "/frame.jpg":
                            with owner._lock:
                                frame = owner._jpeg
                            if frame is None:
                                self._json(503, {"error": "No frame published"})
                            else:
                                self._headers(200, "image/jpeg", len(frame))
                                self.wfile.write(frame)
                        elif path == "/stream.mjpg":
                            self._headers(200, "multipart/x-mixed-replace; boundary=frame")
                            last = -1
                            while owner._running:
                                with owner._frame_ready:
                                    owner._frame_ready.wait_for(lambda: not owner._running or
                                        (owner._jpeg is not None and owner._frame_seq != last), timeout=.5)
                                    if not owner._running:
                                        break
                                    if owner._jpeg is None or owner._frame_seq == last:
                                        continue
                                    frame, last = owner._jpeg, owner._frame_seq
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                                 str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                                self.wfile.flush()
                        else:
                            files = {"/": ("index.html", "text/html; charset=utf-8"),
                                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                                     "/style.css": ("style.css", "text/css; charset=utf-8")}
                            if path not in files:
                                raise InputError("not found", 404)
                            name, mime = files[path]
                            content = (owner.ui_dir / name).read_bytes()
                            self._headers(200, mime, len(content))
                            self.wfile.write(content)
                    except InputError as e:
                        self._json(e.status, {"error": str(e)})
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass

                def do_POST(self):
                    try:
                        self._origin(post=True)
                        if self.headers.get("Transfer-Encoding"):
                            raise InputError("chunked input not accepted")
                        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                            raise InputError("application/json required", 415)
                        try:
                            length = int(self.headers.get("Content-Length", "-1"))
                        except ValueError:
                            raise InputError("invalid Content-Length")
                        if not 1 <= length <= 4096:
                            raise InputError("input body must be 1..4096 bytes", 413)
                        raw = self.rfile.read(length)
                        if len(raw) != length:
                            raise InputError("incomplete input body")
                        try:
                            value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
                        except (ValueError, UnicodeDecodeError):
                            raise InputError("invalid finite JSON")
                        if not isinstance(value, dict):
                            raise InputError("JSON object required")
                        path = urlsplit(self.path).path
                        if path == "/api/session":
                            result = owner.claim_session(value)
                        elif path == "/api/input":
                            result = owner.accept_input(value)
                        else:
                            raise InputError("not found", 404)
                        self._json(200, result)
                    except InputError as e:
                        with owner._lock:
                            owner._rejected += 1
                        self.close_connection = True
                        self._json(e.status, {"error": str(e)})
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass

            class Server(ThreadingHTTPServer):
                daemon_threads = True
                block_on_close = False
                allow_reuse_address = True

            for name in ("index.html", "app.js", "style.css"):
                if not (self.ui_dir / name).is_file():
                    raise FileNotFoundError(self.ui_dir / name)
            self._server = Server((self.host, self.port), Handler)
            self.port = self._server.server_address[1]
            self._running = True
            self._thread = threading.Thread(target=self._server.serve_forever,
                                            kwargs={"poll_interval": .1}, name="g1-live-http", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        with self._frame_ready:
            self._running = False
            self._clear("transport stopped")
            self._frame_ready.notify_all()
            server, thread = self._server, self._thread
        if server:
            server.shutdown()
            server.server_close()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.)
        self._server = self._thread = None

    def _clear(self, reason):
        self._active = False
        self._axes = (0., 0., 0.)
        self._pending_edge = None
        self._f_down = self._f_armed = False
        self._reason = reason

    def _expire(self, now):
        if self._active and (self._received is None or now - self._received > self.deadman_s or
                            self._ticket_issued is None or now - self._ticket_issued > self.deadman_s):
            self._clear("input expired")

    def _new_ticket(self, now):
        while self._tickets and (now - next(iter(self._tickets.values())) > self.deadman_s or len(self._tickets) >= 32):
            self._tickets.popitem(last=False)
        ticket = secrets.token_urlsafe(18)
        self._tickets[ticket] = now
        return ticket

    def _authenticate(self, value):
        token = value.get("token")
        if not isinstance(token, str) or not token.isascii() or not secrets.compare_digest(token, self._token):
            raise InputError("invalid local session token", 403)

    def claim_session(self, value):
        if set(value) != {"token"}:
            raise InputError("session expects only token")
        self._authenticate(value)
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            if self._active:
                raise InputError("another operator holds the active input lease", 409)
            self._clear("operator must enable input")
            self._client_id = secrets.token_urlsafe(18)
            self._view_session = secrets.token_urlsafe(18)
            # A fresh input lease invalidates old action/frame identities, not
            # the player's camera choice. Keep the view continuous across
            # focus loss, Escape and re-enable.
            self._look = {**self._look, "revision": 0}
            if self.steer_with_look:
                native_heading = self._status.get("robot_state", {}).get("heading", 0.)
                self._look.setdefault("heading_world", float(native_heading))
                self._look.setdefault("mode", "third_person")
            self._views.clear()
            self._selection_id = None
            self._seq = -1
            self._received = self._ticket_issued = None
            return {"client_id": self._client_id, "ticket": self._new_ticket(now),
                    "deadman_s": self.deadman_s, "next_seq": 0, "view_session": self._view_session, "look": dict(self._look)}

    def _eligible_candidate(self, now, selected=None):
        candidate = self._status.get("candidate")
        if self._status_time is None or now - self._status_time > self.deadman_s:
            return None
        if selected is not None:
            candidates = self._status.get("candidates", [])
            if not isinstance(candidates, list):
                return None
            matches = [c for c in candidates if isinstance(c, dict) and c.get("id") == selected]
            if not matches and isinstance(candidate, dict) and candidate.get("id") == selected:
                matches = [candidate]
            if len(matches) != 1:
                return None
            candidate = matches[0]
        if not isinstance(candidate, dict) or candidate.get("eligible") is not True:
            return None
        cid = candidate.get("id")
        return cid if isinstance(cid, str) and cid else None

    def accept_input(self, value):
        required = {"token", "client_id", "seq", "ticket", "forward", "lateral", "yaw", "interact", "active"}
        optional = {"selection_id", "interaction_candidate_id", "look_yaw", "look_pitch", "look_revision",
                    "interaction_frame_seq", "interaction_action", "interaction_view_session",
                    "look_heading", "view_mode"}
        if not required <= set(value) or not set(value) <= required | optional:
            raise InputError("input fields must match g1-live-input/v1 exactly")
        self._authenticate(value)
        seq = value["seq"]
        if type(seq) is not int or not 0 <= seq <= 2**53 - 1:
            raise InputError("seq must be a nonnegative safe integer")
        if type(value["active"]) is not bool or type(value["interact"]) is not bool:
            raise InputError("active and interact must be bool")
        axes = []
        for name in ("forward", "lateral", "yaw"):
            v = value[name]
            if type(v) not in (int, float) or not -1 <= v <= 1 or not math.isfinite(v):
                raise InputError(f"{name} must be finite and within [-1,1]")
            axes.append(float(v))
        if not isinstance(value["client_id"], str) or not isinstance(value["ticket"], str):
            raise InputError("invalid client/ticket type")
        selected = value.get("selection_id")
        if selected is not None and (not isinstance(selected, str) or not 1 <= len(selected) <= 1024):
            raise InputError("selection_id must be null or a nonempty string up to 1024 characters")
        pressed_candidate = value.get("interaction_candidate_id")
        if pressed_candidate is not None and (not isinstance(pressed_candidate, str) or not 1 <= len(pressed_candidate) <= 1024):
            raise InputError("interaction_candidate_id must be null or a nonempty string up to 1024 characters")
        look_fields = {"look_yaw", "look_pitch", "look_revision"}
        look = None
        if look_fields & set(value):
            if not look_fields <= set(value):
                raise InputError("all three camera look fields are required together")
            for field, low, high in (("look_yaw", -math.pi, math.pi), ("look_pitch", -1.15, .60)):
                v = value[field]
                if type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v):
                    raise InputError("invalid bounded camera look")
            revision = value["look_revision"]
            if type(revision) is not int or not 0 <= revision <= 2**53-1:
                raise InputError("invalid camera revision")
            look = {"yaw": float(value["look_yaw"]), "pitch": float(value["look_pitch"]), "revision": revision}
        if {"look_heading", "view_mode"} & set(value):
            if look is None or not {"look_heading", "view_mode"} <= set(value):
                raise InputError("world heading and view mode require a complete look packet")
            heading = value["look_heading"]
            if type(heading) not in (int, float) or not -math.pi <= heading <= math.pi or not math.isfinite(heading):
                raise InputError("invalid bounded world heading")
            if value["view_mode"] not in ("head", "third_person") or look["yaw"] != 0.:
                raise InputError("world-heading camera requires a supported mode and look_yaw=0")
            look.update(heading_world=float(heading), mode=value["view_mode"])
        if self.require_view and value["active"] and look is None:
            raise InputError("camera-bound input requires look fields")
        if self.steer_with_look and value["active"] and (look is None or "heading_world" not in look):
            raise InputError("game controls require world heading and view mode")
        frame_seq = value.get("interaction_frame_seq")
        if frame_seq is not None and (type(frame_seq) is not int or not 1 <= frame_seq <= 2**53-1):
            raise InputError("invalid interaction frame sequence")
        action = value.get("interaction_action")
        if action is not None and action not in ("open", "close", "grasp", "place"):
            raise InputError("invalid interaction action")
        session = value.get("interaction_view_session")
        if session is not None and (not isinstance(session, str) or not 1 <= len(session) <= 128):
            raise InputError("invalid interaction view session")
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            if value["client_id"] != self._client_id:
                raise InputError("operator lease was replaced", 409)
            if seq <= self._seq:
                raise InputError("replayed or reordered input sequence", 409)
            issued = self._tickets.get(value["ticket"])
            # Even inactive packets are sequenced/authenticated; they can always
            # stop an existing lease after ticket expiry, but never enable it.
            if value["active"] and (issued is None or now - issued > self.deadman_s):
                raise InputError("expired input ticket", 409)
            if value["active"] and look is not None:
                if "heading_world" in look and look["heading_world"] != self._look.get("heading_world"):
                    native_heading = self._status.get("robot_state", {}).get("heading")
                    if native_heading is None or self._status_time is None or now-self._status_time > self.deadman_s:
                        raise InputError("world heading needs a current robot frame", 409)
                # Camera orientation is independent of the native body's yaw.
                # Autonomous walking can turn that body behind a fixed view.
                # The runtime separately limits the actual facing command's
                # rate and lead; an offset camera is not invalid input.
                if (look["revision"] < self._look["revision"] or
                        (look["revision"] == self._look["revision"] and look != self._look)):
                    raise InputError("camera revision changed or went backward", 409)
                if look != self._look:
                    self._pending_edge = None
                self._look = look
            self._seq, self._received, self._ticket_issued = seq, now, issued
            if selected != self._selection_id:
                self._pending_edge = None
            self._selection_id = selected
            self._accepted += 1
            if not value["active"]:
                self._clear("operator stopped or focus lost")
            else:
                self._active, self._axes = True, tuple(axes)
                self._reason = "live input"
                down = value["interact"]
                if not down:
                    self._f_armed = True
                elif not self._f_down and self._f_armed:
                    self._f_armed = False
                    candidate = self._eligible_candidate(now, pressed_candidate if pressed_candidate is not None else self._selection_id)
                    if "interaction_candidate_id" in value and pressed_candidate is None:
                        candidate = None
                    if self._selection_id is not None and candidate != self._selection_id:
                        candidate = None
                    view_key = {"frame_seq": frame_seq, "action": action,
                                "view_revision": look["revision"] if look else None, "view_session": session}
                    if self.require_view and not self._valid_view_edge(now, candidate, view_key):
                        candidate = None
                        self._reason = "Aim or displayed frame changed; release F and try again"
                    if candidate is not None:
                        self._edge_id += 1
                        self._pending_edge = (self._edge_id, candidate, now, self._selection_id, view_key)
                    # Ineligible/busy edges are discarded, never deferred.
                self._f_down = down
            return {"accepted": True, "seq": self._seq, "active": self._active}

    def poll_command(self):
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            edge, self._pending_edge = self._pending_edge, None
            valid_edge = bool(edge and self._active and now - edge[2] <= self.deadman_s and
                              edge[3] == self._selection_id and edge[1] == self._eligible_candidate(now, edge[1]) and
                              (not self.require_view or self._valid_view_edge(now, edge[1], edge[4])))
            forward, lateral, yaw = self._axes if self._active else (0., 0., 0.)
            return {"forward": forward, "lateral": lateral, "yaw": yaw,
                    "interact": valid_edge, "active": self._active,
                    "age_s": None if self._received is None else now - self._received,
                    "seq": self._seq, "client_id": self._client_id, "reason": self._reason,
                    "selection_id": self._selection_id,
                    "look_heading": self._look.get("heading_world"),
                    "look_revision": self._look["revision"], "view_session": self._view_session,
                    "interaction_candidate_id": edge[1] if valid_edge else None,
                    "interaction_event_id": edge[0] if valid_edge else None,
                    "interaction_action": edge[4]["action"] if valid_edge and edge[4]["action"] else
                        next((c.get("action", "open") for c in self._status.get("candidates", []) if valid_edge and c.get("id") == edge[1]), None),
                    "interaction_frame_seq": edge[4]["frame_seq"] if valid_edge else None}

    def state_snapshot(self):
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            return {"status": json.loads(json.dumps(self._status)), "ticket": self._new_ticket(now),
                    "transport": {"active": self._active, "input_age_s": None if self._received is None else now - self._received,
                                  "status_age_s": None if self._status_time is None else now - self._status_time,
                                  "frame_age_s": None if self._frame_time is None else now - self._frame_time,
                                  "seq": self._seq, "frame_seq": self._frame_seq, "reason": self._reason,
                                  "accepted": self._accepted, "rejected": self._rejected,
                                  "deadman_s": self.deadman_s, "view_session": self._view_session, "look": dict(self._look)}}

    def view_request(self):
        with self._lock:
            return {"look": dict(self._look), "view_session": self._view_session}

    def _valid_view_edge(self, now, candidate, key):
        if candidate is None or key.get("view_session") != self._view_session:
            return False
        frame = self._views.get(key.get("frame_seq"))
        if frame is None or not 0 <= now-frame["captured_at"] <= self.deadman_s:
            return False
        if key.get("view_revision") != self._look["revision"] or frame["view_revision"] != key.get("view_revision"):
            return False
        selection = frame.get("selection")
        if (not isinstance(selection, dict) or selection.get("eligible") is not True or
                selection.get("id") != candidate or selection.get("action") != key.get("action")):
            return False
        current = [c for c in self._status.get("candidates", []) if c.get("id") == candidate]
        return (len(current) == 1 and current[0].get("action", "open") == key.get("action") and
                candidate == self._eligible_candidate(now, candidate))

    def publish_view(self, packet):
        jpeg = packet.get("jpeg")
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 8*1024*1024 or
                not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise ValueError("invalid view JPEG")
        clean = json.loads(json.dumps({k: v for k, v in packet.items() if k != "jpeg"}, allow_nan=False))
        required = {"captured_at", "frame_sim_time", "view_revision", "view_session", "selection", "width", "height"}
        if not required <= clean.keys():
            raise ValueError("view packet lacks capture-bound metadata")
        if (type(clean["view_revision"]) is not int or not 0 <= clean["view_revision"] <= 2**53-1 or
                type(clean["captured_at"]) not in (int, float) or
                not -1e15 < clean["captured_at"] < 1e15 or
                type(clean["frame_sim_time"]) not in (int, float) or
                not 0 <= clean["frame_sim_time"] < 1e12):
            raise ValueError("invalid capture time or view revision")
        with self._frame_ready:
            now = time.monotonic()
            # A completed frame remains useful while newer mouse input is in
            # flight. Exact-current revision is an F admission rule, not a
            # display rule; applying it here starves continuous mouse movement.
            previous = next(reversed(self._views.values()), None)
            if (clean["view_session"] != self._view_session or clean["view_revision"] > self._look["revision"] or
                    (previous is not None and (clean["captured_at"] <= previous["captured_at"] or
                        clean["frame_sim_time"] < previous["frame_sim_time"] or
                        clean["view_revision"] < previous["view_revision"])) or
                    not 0 <= now-clean["captured_at"] <= 2.):
                return False
            self._frame_seq += 1
            clean.update(frame_seq=self._frame_seq, jpeg_base64=base64.b64encode(jpeg).decode("ascii"))
            self._views[self._frame_seq] = clean
            while self._views and (len(self._views) > 16 or now-next(iter(self._views.values()))["captured_at"] > 2.):
                self._views.popitem(last=False)
            self._jpeg, self._frame_time = jpeg, clean["captured_at"]
            self._frame_ready.notify_all()
            return True

    def view_snapshot(self):
        with self._lock:
            if not self._views:
                return {"frame_seq": None, "view_session": self._view_session, "selection": None, "frame_age_s": None}
            frame = json.loads(json.dumps(next(reversed(self._views.values()))))
            frame["frame_age_s"] = max(0., time.monotonic()-frame.pop("captured_at"))
            return frame

    def publish(self, status: dict, jpeg: bytes | None = None):
        if not isinstance(status, dict):
            raise ValueError("status must be a JSON object")
        clean = json.loads(json.dumps(status, allow_nan=False))
        if jpeg is not None:
            if not isinstance(jpeg, (bytes, bytearray)) or not 4 <= len(jpeg) <= 8 * 1024 * 1024:
                raise ValueError("JPEG must contain 4 bytes..8 MiB")
            jpeg = bytes(jpeg)
            if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                raise ValueError("invalid JPEG markers")
        with self._frame_ready:
            now = time.monotonic()
            self._status, self._status_time = clean, now
            if self._pending_edge and self._pending_edge[1] != self._eligible_candidate(now, self._pending_edge[1]):
                self._pending_edge = None
            if jpeg is not None:
                self._jpeg, self._frame_time = jpeg, now
                self._frame_seq += 1
                self._frame_ready.notify_all()
