"""Private native head/third-person rendering and reticle selection; no physics writes.

Runtime contract
----------------
``GameView(model, width=960, height=540).render(renderer, data, heading, look,
candidates, scene_option=None)`` returns JPEG bytes plus JSON-safe metadata.
The caller owns renderer/model/data, copies actual qpos into that private data,
and calls ``mj_kinematics`` before rendering. This module never writes model,
data, joints, actuators, collision masks or physics state. It changes only the
supplied visual scene/camera. ``look`` is {yaw, pitch, revision, mode}: radians,
positive yaw left relative to actual heading, positive pitch up; yaw +/-pi,
pitch [-1.15, .60], mode ``head`` or ``third_person`` (legacy default: head).
Transport derives yaw from the user's world heading demand and actual heading;
the runtime alone owns physical turning. Native yaw catch-up does not create a
new user revision. The transport owns monotonic revision/session validation.

Candidates carry id, name, action (open/close), eligible, reason, body_name,
joint_name, handle_world (CLOSED source position), opening_axis_world and
optionally rail_geom_names. Slide displacement is read from actual qpos.
``handle_local`` instead of handle_world supports an arbitrary articulated body.
Missing/invalid geometry is unselectable. Ineligible visible features remain in
the reticle ranking, so no hidden fallback to a farther eligible mechanism occurs.
The default room-selection radius is eight nominal meters from the camera;
runtime routing and interaction eligibility remain separate physical checks.

Output selection screen_x/screen_y are normalized, top-left coordinates. The
caller atomically publishes JPEG and metadata with frame_seq, frame_sim_time,
frame_age_s and view_session. Selection is documentary camera evidence; runtime
must recheck the exact displayed frame, action, session, age and physical gates.
Robot bodies are excluded only from the camera-obstruction ray so the camera
does not collapse onto its own head. Selection respects visible robot geometry
as well as room occlusion. Rays use native visible mesh triangles/primitives,
never broad-phase spheres as final geometry.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


LOOK_YAW_LIMIT = math.pi
LOOK_PITCH_LIMITS = (-1.15, .60)
VIEW_MODES = ('head', 'third_person')


def _vec3(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


def validate_look(look):
    """Validate one snapshot; session/revision ordering belongs to transport."""
    if not isinstance(look, dict):
        raise ValueError("look must be a mapping")
    yaw, pitch, revision = look.get('yaw'), look.get('pitch'), look.get('revision')
    mode = look.get('mode', 'head')
    if (isinstance(yaw, bool) or isinstance(pitch, bool) or
            not isinstance(yaw, (float, int)) or not isinstance(pitch, (float, int)) or
            not -LOOK_YAW_LIMIT <= yaw <= LOOK_YAW_LIMIT or
            not LOOK_PITCH_LIMITS[0] <= pitch <= LOOK_PITCH_LIMITS[1] or
            not math.isfinite(yaw) or not math.isfinite(pitch) or
            type(revision) is not int or not 0 <= revision <= 2**53-1 or
            type(mode) is not str or mode not in VIEW_MODES):
        raise ValueError("invalid finite bounded look/revision")
    return float(yaw), float(pitch), revision


def project_point(point, eye, forward, up, fovy_degrees, width, height):
    """Perspective projection in the same camera frame as native rendering."""
    point, eye = _vec3(point, 'point'), _vec3(eye, 'eye')
    forward, up = _vec3(forward, 'forward'), _vec3(up, 'up')
    right = np.cross(forward, up)
    delta = point-eye
    depth = float(delta @ forward)
    if depth <= .01:
        return None
    x, y = float(delta @ right), float(delta @ up)
    half = math.tan(math.radians(fovy_degrees)/2)
    return {'screen_x': .5+x/(2*depth*half*(width/height)),
            'screen_y': .5-y/(2*depth*half), 'depth_m': depth,
            'angle_radians': math.atan2(math.hypot(x, y), depth),
            'distance_m': float(np.linalg.norm(delta))}


class GameView:
    def __init__(self, model, width=960, height=540, fovy_degrees=68.,
                 selection_cone_degrees=18., selection_distance_m=8.):
        if (type(width) is not int or type(height) is not int or
                not 160 <= width <= 2560 or not 90 <= height <= 1440 or
                not 40 <= fovy_degrees <= 90 or not 1 <= selection_cone_degrees <= 25 or
                isinstance(selection_distance_m, bool) or
                not isinstance(selection_distance_m, (float, int, np.floating)) or
                not .5 <= selection_distance_m <= 8):
            raise ValueError('Invalid bounded camera configuration')
        self.model, self.width, self.height = model, width, height
        self.fovy = float(fovy_degrees)
        self.cone = math.radians(selection_cone_degrees)
        self.max_distance = float(selection_distance_m)
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis < 0:
            raise ValueError('A named G1 pelvis is required')
        robot = {int(pelvis)}
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in robot:
                robot.add(b)
        self.robot_bodies = robot
        self.robot_geom = np.isin(model.geom_bodyid, list(robot))
        heads = [g for g in range(model.ngeom)
                 if self.robot_geom[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and 'head' in (model.mesh(int(model.geom_dataid[g])).name or '').lower()
                 and model.geom_group[g] == 1]
        if not heads:
            raise ValueError('No native visible head mesh; refusing an invented camera anchor')
        self.head_geom = heads[0]
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._native_geometry = np.arange(model.ngeom)

    def _ray(self, data, origin, direction, cutoff, geom_ids=None, exclude_robot=False):
        """Exact per-geom rays after conservative sphere culling."""
        ids = self._native_geometry if geom_ids is None else np.asarray(geom_ids, int)
        if not len(ids):
            return None
        if exclude_robot:
            ids = ids[~self.robot_geom[ids]]
        centers = data.geom_xpos[ids]
        delta = centers-origin
        along = delta @ direction
        radii = self.model.geom_rbound[ids]
        perp2 = np.einsum('ij,ij->i', delta, delta)-along*along
        planes = self.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_PLANE
        keep = planes | ((along+radii >= 0) & (along-radii <= cutoff) &
                         (perp2 <= radii*radii+1e-10))
        best, nearest = float(cutoff), -1
        for gid in ids[keep]:
            gid = int(gid)
            kind = int(self.model.geom_type[gid])
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                distance = mujoco.mj_rayMesh(self.model, data, gid, origin, direction)
            elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                distance = mujoco.mj_rayHfield(self.model, data, gid, origin, direction)
            else:
                distance = mujoco.mju_rayGeom(data.geom_xpos[gid], data.geom_xmat[gid],
                    self.model.geom_size[gid], origin, direction, kind)
            if 0 <= distance < best:
                best, nearest = float(distance), gid
        return None if nearest < 0 else {'geom_id': nearest, 'distance_m': best,
            'geom_name': self.model.geom(nearest).name,
            'body_name': self.model.body(int(self.model.geom_bodyid[nearest])).name}

    def _handle(self, data, item):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, item['body_name'])
        if body < 0:
            raise ValueError('unknown handle body')
        if item.get('handle_local') is not None:
            return data.xpos[body]+data.xmat[body].reshape(3, 3) @ _vec3(item['handle_local'], 'handle_local')
        point = _vec3(item['handle_world'], 'handle_world')
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, item['joint_name'])
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError('non-slide handles require a body-local point')
        axis = _vec3(item['opening_axis_world'], 'opening_axis_world')
        if abs(np.linalg.norm(axis)-1) > 1e-5 or int(self.model.jnt_bodyid[joint]) != body:
            raise ValueError('handle axis/body does not match its slide')
        address = int(self.model.jnt_qposadr[joint])
        return point+axis*(float(data.qpos[address])-float(self.model.qpos0[address]))

    def prepare(self, renderer, data, heading, look, scene_option=None):
        """Update ONLY renderer scene/camera, and return exact camera metadata."""
        yaw, pitch, revision = validate_look(look)
        mode = look.get('mode', 'head')
        if (isinstance(heading, bool) or not isinstance(heading, (float, int, np.floating)) or
                not -2*math.pi <= heading <= 2*math.pi or not math.isfinite(heading)):
            raise ValueError('heading must be finite and bounded')
        angle = float(heading)+yaw
        forward = np.array([math.cos(angle)*math.cos(pitch), math.sin(angle)*math.cos(pitch), math.sin(pitch)])
        right = np.array([math.sin(angle), -math.cos(angle), 0.])
        up = np.cross(right, forward)
        head = data.geom_xpos[self.head_geom].copy()
        # Both anchors come from the actual visible head. The longer view lowers
        # its orbit center so feet remain in frame at the default look pitch.
        anchor = head if mode == 'head' else head+np.array([0., 0., -.35])
        # Keep the shoulder offset inside the nearby task's lateral reach:
        # a larger orbit radius can make centering a close handle impossible
        # while the actual body remains within its stance heading allowance.
        offset = (-.62*forward+.25*right+.16*up if mode == 'head' else
                  -2.10*forward+.25*right+.14*up)
        visible = np.flatnonzero(self.model.geom_rgba[:, 3] > 0)
        if scene_option is not None:
            visible = visible[np.asarray(scene_option.geomgroup)[np.clip(self.model.geom_group[visible], 0, 5)] != 0]
        blockage = self._ray(data, anchor, offset/np.linalg.norm(offset), np.linalg.norm(offset)+.06,
                             visible, exclude_robot=True)
        distance = float(np.linalg.norm(offset))
        if blockage is not None:
            distance = max(.08, min(distance, blockage['distance_m']-.055))
        eye = anchor+offset/np.linalg.norm(offset)*distance
        self.camera.lookat[:] = eye+forward
        self.camera.distance = 1.
        self.camera.azimuth = math.degrees(angle)
        self.camera.elevation = math.degrees(pitch)
        renderer.update_scene(data, self.camera, scene_option=scene_option)
        # Set both monocular GL views explicitly. No model camera or vis settings change.
        near = .01
        half = near*math.tan(math.radians(self.fovy)/2)
        for camera in renderer.scene.camera:
            camera.pos[:] = eye
            camera.forward[:] = forward
            camera.up[:] = up
            camera.orthographic = 0
            camera.frustum_near = near
            camera.frustum_far = max(30., float(camera.frustum_far))
            camera.frustum_top, camera.frustum_bottom = half, -half
            camera.frustum_center = 0.
            camera.frustum_width = 2*half*self.width/self.height
        camera = renderer.scene.camera[0]
        return {'mode': mode, 'eye_world': camera.pos.astype(float).tolist(),
                'forward_world': camera.forward.astype(float).tolist(),
                'up_world': camera.up.astype(float).tolist(), 'head_world': head.tolist(),
                'anchor_world': anchor.tolist(),
                'native_heading_world_rad': float(heading),
                'desired_heading_world_rad': math.atan2(math.sin(angle), math.cos(angle)),
                'head_geom_id': self.head_geom, 'fovy_degrees': self.fovy,
                'look': {'yaw': yaw, 'pitch': pitch, 'revision': revision, 'mode': mode},
                'wall_shortened': bool(distance < np.linalg.norm(offset)-1e-6),
                'camera_obstruction': blockage,
                'selection_cone_degrees': math.degrees(self.cone),
                'selection_distance_m': self.max_distance,
                'ray_robot_exclusion': 'camera obstruction only; selection respects visible robot surfaces'}

    def select(self, renderer, data, camera, candidates):
        """Rank visible screen features before looking at their eligibility."""
        eye, forward, up = (np.asarray(camera[k], float) for k in
                           ('eye_world', 'forward_world', 'up_world'))
        # Native IDs present in this exact visual scene, not just collidable IDs.
        visible_geoms = sorted({int(g.objid) for g in renderer.scene.geoms[:renderer.scene.ngeom]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid >= 0 and g.rgba[3] > 0})
        rows, ranked, used = [], [], set()
        for item in list(candidates)[:64]:
            row = {'id': item.get('id'), 'visible': False}
            try:
                if not isinstance(item.get('id'), str) or item['id'] in used:
                    raise ValueError('missing/duplicate candidate id')
                used.add(item['id'])
                if item.get('action') not in ('open', 'close', 'grasp', 'place'):
                    raise ValueError('unknown displayed interaction action')
                point = self._handle(data, item)
                projection = project_point(point, eye, forward, up, self.fovy, self.width, self.height)
                row['handle_world'] = point.tolist()
                if projection is None:
                    row['view_reason'] = 'behind camera'
                    rows.append(row)
                    continue
                row.update(projection)
                if not (0 <= row['screen_x'] <= 1 and 0 <= row['screen_y'] <= 1):
                    row['view_reason'] = 'offscreen'
                elif row['angle_radians'] > self.cone:
                    row['view_reason'] = 'outside reticle cone'
                elif row['distance_m'] > self.max_distance:
                    row['view_reason'] = 'beyond view-selection distance'
                else:
                    direction = (point-eye)/row['distance_m']
                    hit = self._ray(data, eye, direction, row['distance_m']+.03, visible_geoms)
                    rail_names = set(item.get('rail_geom_names') or [])
                    own_rail = hit is not None and hit['geom_name'] in rail_names
                    row['occlusion_hit'] = hit
                    row['visible'] = hit is not None and (own_rail or hit['distance_m'] >= row['distance_m']-.001)
                    row['view_reason'] = 'visible' if row['visible'] else 'occluded or no native handle surface'
                    if row['visible']:
                        selection = {k: row[k] for k in ('id', 'screen_x', 'screen_y', 'distance_m', 'angle_radians')}
                        selection.update(name=str(item.get('name') or item['id']), action=item['action'],
                            eligible=item.get('eligible') is True,
                            reason=str(item.get('reason') or ('Ready' if item.get('eligible') is True else 'Unavailable')))
                        ranked.append(selection)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                row['view_reason'] = 'invalid geometry: '+str(error)
            rows.append(row)
        ranked.sort(key=lambda x: (x['angle_radians'], x['distance_m'], x['id']))
        return (ranked[0] if ranked else None), rows

    def render(self, renderer, data, heading, look, candidates, scene_option=None):
        import cv2
        camera = self.prepare(renderer, data, heading, look, scene_option)
        selection, projections = self.select(renderer, data, camera, candidates)
        image = renderer.render()
        if image.shape != (self.height, self.width, 3):
            raise ValueError('Renderer dimensions differ from the camera contract')
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError('Native frame JPEG encoding failed')
        return {'jpeg': jpeg.tobytes(), 'width': self.width, 'height': self.height,
                'view_revision': camera['look']['revision'], 'selection': selection,
                'camera': camera, 'candidates_projected': projections}
