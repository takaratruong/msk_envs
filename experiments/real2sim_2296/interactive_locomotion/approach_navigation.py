"""Read-only native-geometry routes for the room's walking approach.

This module neither authorizes a skill nor executes motion. The caller owns
exact ID/action/frame checks, cancellation, speed, deadlines and physics guards.
Geometric clearance is not a dynamic stability or contact qualification.
"""
import hashlib
import heapq
import math
import time
import numpy as np
import mujoco
import shapely
from shapely.geometry import LineString, MultiPoint, Point, box
from shapely.strtree import STRtree


def _xy(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError(name+' must be a finite pair')
    return a


def _angle(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('yaw must be a finite number')
    return math.atan2(math.sin(value), math.cos(value))


class NativeApproachMap:
    """NativeApproachMap(model,data,options=dict), with immutable source model.

    plan(data,start_xy,goal_xy,goal_yaw) returns a JSON-safe dictionary.
    Planning uses a swept cruise circle at least 0.45 m in radius, enlarged
    whenever the current native footprint is larger. Runtime segment_clear
    uses the complete current native footprint plus the declared margins; the
    planning reserve remains available for tracking drift. An explicit
    yaw checks the current native robot geometry plus declared walking margins
    at that fixed heading. corridor_clear checks translation with a bounded
    yaw sweep; turn_clear uses the same check at a fixed position.

    Call after native kinematics is current. Source model edits require a new
    map. No forward/step call or model/data assignment is performed here.
    """
    def __init__(self, model, data, options=None):
        o = dict(options or {})
        known = {'floor_geometries','robot_root','grid_m','max_nodes',
            'planning_seconds','body_margin_m','leg_margin_m','obstacle_margin_m',
            'refresh_tolerance_m','navigation_height_m','terminal_anchor_m',
            'minimum_cruise_radius_m','carried_geom_names'}
        if set(o)-known:
            raise ValueError('Unknown navigation options: '+str(sorted(set(o)-known)))
        self.model = model
        self.grid = self._number(o, 'grid_m', .06, .03, .12)
        self.deadline = self._number(o, 'planning_seconds', 2., .05, 10.)
        self.body_margin = self._number(o, 'body_margin_m', .025, .015, .10)
        self.leg_margin = self._number(o, 'leg_margin_m', .05, .025, .15)
        self.obstacle_margin = self._number(o, 'obstacle_margin_m', .003, .001, .02)
        self.refresh_tolerance = self._number(o, 'refresh_tolerance_m', .0001, 0., .001)
        if self.obstacle_margin < self.refresh_tolerance:
            raise ValueError('Obstacle margin must cover cached-pose tolerance')
        self.height = self._number(o, 'navigation_height_m', 1.50, 1.35, 1.80)
        self.anchor_distance = self._number(o, 'terminal_anchor_m', .35, .25, .80)
        # Saved AMO walking/turning native bounds reached 0.398674 m with the
        # margins below already included. Keep at least 51 mm extra cruise
        # planning reserve; never cap a larger current posture to this value.
        # Runtime queries retain current native bounds and the original margins.
        self.minimum_cruise_radius = self._number(o, 'minimum_cruise_radius_m', .45, .45, .80)
        self.max_nodes = o.get('max_nodes', 30000)
        if type(self.max_nodes) is not int or not 100 <= self.max_nodes <= 100000:
            raise ValueError('max_nodes must be an integer in100..100000')
        names = o.get('floor_geometries')
        if not isinstance(names, list) or not names or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('Explicit nonempty floor_geometries list required')
        self.floor_ids = {model.geom(n).id for n in names}
        self.root = model.body(o.get('robot_root','pelvis')).id
        bodies = {self.root}
        for body in range(model.nbody):
            if int(model.body_parentid[body]) in bodies:bodies.add(body)
        carried_names=o.get('carried_geom_names',[])
        if not isinstance(carried_names,list) or len(carried_names)>4 or len(set(carried_names))!=len(carried_names):
            raise ValueError('Carried geometry requires a small unique name list')
        self.carried_ids={model.geom(n).id for n in carried_names}
        for g in self.carried_ids:
            body=int(model.geom_bodyid[g]);joint=int(model.body_jntadr[body])
            if (body in bodies or model.body_jntnum[body]!=1 or
                    model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE or
                    not (model.geom_contype[g] or model.geom_conaffinity[g]) or
                    np.any(model.actuator_trnid[:,0]==joint)):
                raise ValueError('Carried footprint must be an unactuated free-object collider')
        self.robot_ids = np.array([g for g in range(model.ngeom)
            if (int(model.geom_bodyid[g]) in bodies or g in self.carried_ids)
            and (model.geom_contype[g] or model.geom_conaffinity[g])],int)
        if not len(self.robot_ids):raise ValueError('No native robot colliders')
        self._leg = np.array([any(s in model.body(int(model.geom_bodyid[g])).name
            for s in ('hip_','knee','ankle')) for g in self.robot_ids])
        self._corners = np.array([[a,b,c] for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)])
        self._local = {g:model.geom_aabb[g,:3]+self._corners*model.geom_aabb[g,3:]
            for g in range(model.ngeom)}
        obstacle_ids = [g for g in range(model.ngeom) if g not in self.floor_ids
            and int(model.geom_bodyid[g]) not in bodies and g not in self.carried_ids
            and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.dynamic_ids=[];self.static_ids=[]
        for g in obstacle_ids:
            b=int(model.geom_bodyid[g]);dynamic=False
            while b:
                dynamic |= bool(model.body_dofnum[b]);b=int(model.body_parentid[b])
            (self.dynamic_ids if dynamic else self.static_ids).append(g)
        self._dynamic_indices=np.asarray(self.dynamic_ids,int)
        self._dynamic_local=np.asarray([self._local[g] for g in self.dynamic_ids]).reshape(-1,8,3)
        self._robot_local=np.asarray([self._local[g] for g in self.robot_ids])
        # Obstacle bounds may overestimate solids; floor support must never be
        # overestimated. Only an actual horizontal top face is accepted here.
        floors=[self._support_shape(data,g) for g in self.floor_ids]
        self.floor=shapely.union_all([x[0] for x in floors])
        self.floor_z=float(max(x[2] for x in floors))
        if any(abs(x[2]-self.floor_z)>.01 for x in floors):
            raise ValueError('Navigation requires a common flat support elevation')
        if not self.floor.is_valid or self.floor.area<1.:raise ValueError('Invalid native floor bounds')
        self.static=[(g,*self._shape(data,g)) for g in self.static_ids]
        self.static=[x for x in self.static if x[3]>self.floor_z+.008 and x[2]<self.floor_z+self.height]
        self.static_tree=STRtree([x[1] for x in self.static])
        self.static_union=shapely.union_all([x[1] for x in self.static])
        self.dynamic=[];self.dynamic_tree=STRtree([]);self.dynamic_union=shapely.GeometryCollection()
        self._dynamic_world={};self._dynamic_rows={};self._dynamic_points=None
        self.revision=0;self.last_reason='Not checked';self._grid_cache=None
        self.options={'floor_geometries':sorted(names),'grid_m':self.grid,
            'body_margin_m':self.body_margin,'leg_margin_m':self.leg_margin,
            'obstacle_margin_m':self.obstacle_margin,'refresh_tolerance_m':self.refresh_tolerance,
            'navigation_height_m':self.height,'terminal_anchor_m':self.anchor_distance,
            'minimum_cruise_radius_m':self.minimum_cruise_radius,
            'planning_seconds':self.deadline,'max_nodes':self.max_nodes}
        if carried_names:
            self.options['carried_geom_names']=list(carried_names)
        self.model_geometry_digest=hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes()
            for x in (model.geom_aabb,model.geom_pos,model.geom_quat,model.geom_bodyid,
                      model.geom_contype,model.geom_conaffinity,model.body_parentid))).hexdigest()
        self.refresh(data)

    @staticmethod
    def _number(o,k,default,lo,hi):
        v=o.get(k,default)
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(k+' is outside its finite bound')
        return float(v)

    def _world(self,data,g):
        points=self._local[g]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        if not np.isfinite(points).all():raise ValueError('Nonfinite native collider transform')
        return points

    def _shape(self,data,g):
        pts=self._world(data,g)
        return MultiPoint(pts[:,:2]).convex_hull,float(pts[:,2].min()),float(pts[:,2].max())

    def _support_shape(self,data,g):
        m=self.model;body=int(m.geom_bodyid[g])
        while body:
            if m.body_dofnum[body]:raise ValueError('Navigation floor must be fixed')
            body=int(m.body_parentid[body])
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            pts=self._world(data,g)
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
            pts=m.mesh_vert[start:start+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        else:
            raise ValueError('Floor support requires a finite box or a mesh with a horizontal top face')
        if not np.isfinite(pts).all():raise ValueError('Nonfinite floor geometry')
        z=float(pts[:,2].max());top=pts[np.abs(pts[:,2]-z)<1e-6,:2]
        shape=MultiPoint(top).convex_hull
        if shape.geom_type!='Polygon' or shape.area<.01:
            raise ValueError('No usable horizontal top face in the native floor')
        # The top face of the native convex collision mesh is supported; using
        # only its vertices avoids the unsafe projected-AABB floor shortcut.
        return shape,float(pts[:,2].min()),z

    def refresh(self,data):
        """Refresh changed passive/free obstacle bounds; static geometry stays cached."""
        ids=self._dynamic_indices
        points=self._dynamic_local@data.geom_xmat[ids].reshape(-1,3,3).transpose(0,2,1)+data.geom_xpos[ids,None,:]
        if not np.isfinite(points).all():raise ValueError('Nonfinite native obstacle transform')
        changed=np.ones(len(ids),bool) if self._dynamic_points is None else np.max(np.linalg.norm(points-self._dynamic_points,axis=2),axis=1)>self.refresh_tolerance
        if np.any(changed):
            if self._dynamic_points is None:self._dynamic_points=points.copy()
            else:self._dynamic_points[changed]=points[changed]
            for k in np.flatnonzero(changed):
                g=int(ids[k]);pts=points[k];self._dynamic_world[g]=pts
                lo,hi=float(pts[:,2].min()),float(pts[:,2].max())
                if hi>self.floor_z+.008 and lo<self.floor_z+self.height:
                    self._dynamic_rows[g]=(g,MultiPoint(pts[:,:2]).convex_hull,lo,hi)
                else:self._dynamic_rows.pop(g,None)
            self.dynamic=[row for _,row in sorted(self._dynamic_rows.items())]
            self.dynamic_tree=STRtree([x[1] for x in self.dynamic])
            self.dynamic_union=shapely.union_all([x[1] for x in self.dynamic])
            self.revision+=1;self._grid_cache=None
        return bool(np.any(changed))

    def _footprint(self,data,yaw=None):
        R=data.xmat[self.root].reshape(3,3)
        heading=math.atan2(R[1,0],R[0,0])
        yaw=heading if yaw is None else _angle(yaw)
        delta=yaw-heading;c,s=math.cos(delta),math.sin(delta)
        turn=np.array([[c,-s],[s,c]])
        center=np.asarray(data.xpos[self.root,:2]);shapes=[];radius=0.
        points=self._robot_local@data.geom_xmat[self.robot_ids].reshape(-1,3,3).transpose(0,2,1)+data.geom_xpos[self.robot_ids,None,:]
        if not np.isfinite(points).all():raise ValueError('Nonfinite native robot transform')
        for k,g in enumerate(self.robot_ids):
            pts=points[k];xy=(pts[:,:2]-center)@turn.T
            margin=(self.leg_margin if self._leg[k] else self.body_margin)
            radius=max(radius,float(np.linalg.norm(xy,axis=1).max())+margin)
            shapes.append((int(g),xy,float(pts[:,2].min())-.02,float(pts[:,2].max())+.02,margin))
        if max(x[3] for x in shapes)>self.floor_z+self.height:
            raise ValueError('Actual robot exceeds the cached navigation height band')
        return shapes,radius,heading

    def _floor_clear(self,line,radius):
        return self.floor.covers(line) and line.distance(self.floor.boundary)>=radius

    def _circle_clear(self,a,b,radius):
        line=Point(a) if np.array_equal(a,b) else LineString([a,b])
        if not self._floor_clear(line,radius):self.last_reason='Walking footprint leaves the finite native floor';return False
        for label,shape in [('static',self.static_union),('passive',self.dynamic_union)]:
            if not shape.is_empty and line.distance(shape)<=radius+self.obstacle_margin:
                self.last_reason='Cruise footprint intersects '+label+' native geometry';return False
        self.last_reason='Clear cruise sweep';return True

    def _fixed_clear(self,a,b,footprint,extra=0.):
        # Require native-footprint floor support. The full cruise turning disc
        # is intentionally not substituted into this fixed-heading corridor.
        radius=max(float(np.linalg.norm(x[1],axis=1).max())+x[4] for x in footprint)+self.obstacle_margin+extra
        line=Point(a) if np.array_equal(a,b) else LineString([a,b])
        floor_circle_clear=self._floor_clear(line,radius)
        low=np.minimum(a,b)-radius;high=np.maximum(a,b)+radius
        # An expanded AABB is a guaranteed broad-phase superset; a polygonal
        # buffer's approximate circular arcs need not have that property.
        search=box(low[0],low[1],high[0],high[1]);near=[]
        for rows,tree in [(self.static,self.static_tree),(self.dynamic,self.dynamic_tree)]:
            near.extend(rows[int(index)] for index in tree.query(search))
        if not near and floor_circle_clear:
            self.last_reason='Clear fixed-heading native footprint sweep';return True
        bounds=np.asarray([row[1].bounds for row in near]).reshape(-1,4)
        intervals=np.asarray([[row[2],row[3]] for row in near]).reshape(-1,2)
        for robot_g,xy,zlo,zhi,padding in footprint:
            swept=MultiPoint(np.vstack([xy+a,xy+b])).convex_hull
            pad=padding+self.obstacle_margin+extra
            if not floor_circle_clear and (not self.floor.covers(swept) or swept.distance(self.floor.boundary)<pad):
                self.last_reason='Fixed-heading footprint leaves the native floor';return False
            if not near:continue
            x0,y0,x1,y1=swept.bounds
            relevant=np.flatnonzero((intervals[:,1]+self.obstacle_margin>=zlo)&
                (intervals[:,0]-self.obstacle_margin<=zhi)&(bounds[:,0]<=x1+pad)&
                (bounds[:,2]>=x0-pad)&(bounds[:,1]<=y1+pad)&(bounds[:,3]>=y0-pad))
            if not len(relevant):continue
            distances=shapely.distance(swept,[near[int(index)][1] for index in relevant])
            blocked=np.flatnonzero(distances<=pad)
            if len(blocked):
                g=near[int(relevant[int(blocked[0])])][0]
                self.last_reason='Native obstacle '+(self.model.geom(g).name or str(g))+' blocks '+self.model.body(int(self.model.geom_bodyid[robot_g])).name
                return False
        self.last_reason='Clear fixed-heading native footprint sweep';return True

    def segment_clear(self,data,a,b,yaw=None):
        a,b=_xy(a,'segment start'),_xy(b,'segment end');self.refresh(data)
        footprint,radius,_=self._footprint(data,yaw)
        return self._circle_clear(a,b,radius) if yaw is None else self._fixed_clear(a,b,footprint)

    def corridor_clear(self,data,a,b,yaw_a,yaw_b):
        """Conservative translation and shortest yaw sweep of the current pose.

        Every position along a->b is checked against every intermediate yaw.
        This deliberately permits more combinations than a synchronized path.
        For each <=5 degree interval, each native vertex's circular arc lies
        within its endpoint chord plus R*(1-cos(interval/2)). Taking the convex
        hull of both endpoint footprints, then its translation sweep and this
        sagitta expansion, contains the entire native footprint. All existing
        body/leg/obstacle margins remain; this does not predict future gait.
        """
        a,b=_xy(a,'corridor start'),_xy(b,'corridor end')
        yaw_a,yaw_b=_angle(yaw_a),_angle(yaw_b);self.refresh(data)
        delta=_angle(yaw_b-yaw_a)
        count=max(1,int(math.ceil(abs(delta)/math.radians(5))))
        previous,radius_a,_=self._footprint(data,yaw_a)
        if delta==0.:
            return self._fixed_clear(a,b,previous)
        for index in range(1,count+1):
            following,radius_b,_=self._footprint(data,yaw_a+delta*index/count)
            # 2*sin(x/2)^2 avoids cancellation in 1-cos(x) for tiny angles.
            # These radii include padding and therefore exceed vertex radii.
            extra=2.*max(radius_a,radius_b)*math.sin(delta/count/4.)**2
            combined=[(p[0],np.vstack([p[1],q[1]]),min(p[2],q[2]),
                       max(p[3],q[3]),max(p[4],q[4]))
                      for p,q in zip(previous,following)]
            if not self._fixed_clear(a,b,combined,extra):return False
            previous,radius_a=following,radius_b
        self.last_reason='Clear bounded native translation/yaw corridor';return True

    def turn_clear(self,data,xy,yaw_a,yaw_b):
        """The conservative corridor check with no translation."""
        return self.corridor_clear(data,xy,xy,yaw_a,yaw_b)

    def _grid(self,radius):
        radius=math.ceil(radius*1000.)/1000.
        key=(radius,self.revision)
        if self._grid_cache is not None and self._grid_cache[0]==key:return self._grid_cache[1:]
        xmin,ymin,xmax,ymax=self.floor.bounds
        xs=np.arange(xmin,xmax+self.grid/2,self.grid);ys=np.arange(ymin,ymax+self.grid/2,self.grid)
        if len(xs)*len(ys)>200000:raise ValueError('Floor grid exceeds bounded cell budget')
        X,Y=np.meshgrid(xs,ys);points=shapely.points(X,Y)
        r=radius+self.grid/math.sqrt(2)+self.obstacle_margin
        free=shapely.contains(self.floor,points)&(shapely.distance(points,self.floor.boundary)>r)
        for shape in [self.static_union,self.dynamic_union]:
            if not shape.is_empty:free &= ~shapely.dwithin(points,shape,r)
        self._grid_cache=(key,xs,ys,free)
        return xs,ys,free

    def cruise_clear(self,data,a,b):
        """Test the full planning reserve at a measured cruise entry position."""
        a,b=_xy(a,'cruise start'),_xy(b,'cruise end');self.refresh(data)
        _,radius,_=self._footprint(data)
        return self._circle_clear(a,b,max(radius,self.minimum_cruise_radius))

    def plan(self,data,start_xy,goal_xy,goal_yaw):
        return self._plan(data,start_xy,goal_xy,goal_yaw,time.monotonic(),True)

    def _plan(self,data,start_xy,goal_xy,goal_yaw,begun,allow_start_corridor):
        start,goal=_xy(start_xy,'start'),_xy(goal_xy,'goal');yaw=_angle(goal_yaw)
        self.refresh(data)
        footprint,radius,heading=self._footprint(data,yaw)
        actual_radius=radius
        radius=max(radius,self.minimum_cruise_radius)
        base={'success':False,'waypoints':[],'reason':'Not planned','goal_xy':goal.tolist(),
            'goal_yaw':yaw,'geometry_revision':int(self.revision),'model_geometry_sha256':self.model_geometry_digest,
            'cruise_radius_m':float(radius),'current_footprint_radius_m':float(actual_radius),
            'planning_radius_reserve_m':float(radius-actual_radius),
            'terminal_anchor_index':None,'terminal_waypoint_index':None,
            'options':dict(self.options),
            'planning_budget_semantics':'Soft elapsed budget checked between geometry/search operations; an individual NumPy/Shapely operation is not preemptible',
            'scope':'Conservative native-bounds geometry only; no motion or skill authorization'}
        def result(ok,reason,waypoints=None,anchor=None):
            if ok and time.monotonic()-begun>self.deadline:
                ok=False;reason='Planning time budget exhausted';waypoints=None;anchor=None
            if ok and (len(waypoints)>128 or sum(math.dist(a,b) for a,b in zip(waypoints,waypoints[1:]))>25.):
                ok=False;reason='Route exceeds waypoint/length budget';waypoints=None;anchor=None
            self.last_reason=reason
            return {**base,'success':bool(ok),'reason':reason,'waypoints':waypoints or [],
                'terminal_anchor_index':anchor,'terminal_waypoint_index':anchor+1 if anchor is not None else None,
                'planning_seconds':time.monotonic()-begun}
        if not self._fixed_clear(goal,goal,footprint):return result(False,'Goal stance blocked: '+self.last_reason)
        if time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted at goal check')
        if np.linalg.norm(goal-start)<=.80 and self.turn_clear(data,start,heading,yaw) and self._fixed_clear(start,goal,footprint):
            return result(True,'Direct checked terminal corridor',[start.tolist(),goal.tolist()],0)
        if not self._circle_clear(start,start,radius):
            if not allow_start_corridor:
                return result(False,'Start lacks cruise clearance: '+self.last_reason)
            actual_footprint,_,_=self._footprint(data,heading)
            if not self._fixed_clear(start,start,actual_footprint):
                return result(False,'Current body clearance is blocked: '+self.last_reason)
            # A legal standing pose may fit beside furniture while lacking the
            # larger cruise turning reserve. Exit at the current heading with
            # the same native bounds and margins, then enter the unchanged
            # cruise planner. A common deadline bounds every attempted exit.
            candidates=[]
            for distance in (.20,.30,.45,.60,.80,1.0):
                for offset in (0.,math.pi,-math.pi/2,math.pi/2,
                               -math.pi/4,math.pi/4,-3*math.pi/4,3*math.pi/4):
                    endpoint=start+distance*np.array([math.cos(heading+offset),math.sin(heading+offset)])
                    candidates.append((distance+float(np.linalg.norm(goal-endpoint)),distance,endpoint))
            candidates.sort(key=lambda x:(x[0],x[1]))
            for _,distance,endpoint in candidates:
                if time.monotonic()-begun>self.deadline:
                    return result(False,'Planning time budget exhausted at start corridor')
                # Extra endpoint space absorbs arrival tolerance. Runtime also
                # requires the measured full cruise circle before turning.
                if not self._circle_clear(endpoint,endpoint,radius+.06):continue
                if not self.corridor_clear(data,start,endpoint,heading-.15,heading+.15):continue
                remainder=self._plan(data,endpoint,goal,yaw,begun,False)
                if not remainder['success']:continue
                route=result(True,'Checked start corridor, cruise route and terminal corridor',
                    [start.tolist()]+remainder['waypoints'],remainder['terminal_anchor_index']+1)
                if route['success']:
                    route['start_corridor']={'end_index':1,'heading_world':heading,
                        'maximum_yaw_error_rad':.15,'distance_m':distance}
                return route
            return result(False,'No clear step from this position into the walking route')
        forward=np.array([math.cos(yaw),math.sin(yaw)]);anchors=[]
        for distance in (self.anchor_distance,self.anchor_distance+.15,self.anchor_distance+.30):
            anchor=goal-forward*distance
            if self._circle_clear(anchor,anchor,radius) and self._fixed_clear(anchor,goal,footprint):anchors.append(anchor)
        if not anchors:
            # The bank aisle can admit a side-step while facing the drawer even
            # when the island blocks every straight-out turning anchor. These
            # bounded common candidates retain one continuously checked final
            # segment; no selected drawer or furniture is excluded.
            left=np.array([-forward[1],forward[0]])
            for side in (-1.,1.):
                for lateral in (.65,1.,1.35,1.70):
                    for outward in (0.,.20,.40):
                        if time.monotonic()-begun>self.deadline:
                            return result(False,'Planning time budget exhausted at lateral anchor check')
                        anchor=goal+side*lateral*left-outward*forward
                        if self._circle_clear(anchor,anchor,radius) and self._fixed_clear(anchor,goal,footprint):
                            anchors.append(anchor)
        if not anchors:return result(False,'No clear terminal turning anchor/corridor')
        if time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted at anchor check')
        xs,ys,free=self._grid(radius)
        if time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted while building occupancy')
        def nearest(p):
            ix=int(round((p[0]-xs[0])/self.grid));iy=int(round((p[1]-ys[0])/self.grid));candidates=[]
            for dy in range(-2,3):
                for dx in range(-2,3):
                    x,y=ix+dx,iy+dy
                    if 0<=y<len(ys) and 0<=x<len(xs) and free[y,x]:
                        q=np.array([xs[x],ys[y]])
                        if self._circle_clear(p,q,radius):candidates.append((float(np.linalg.norm(q-p)),(y,x)))
            return min(candidates)[1] if candidates else None
        first=nearest(start)
        if first is None:return result(False,'No safe grid connection from current position')
        targets={}
        for anchor in anchors:
            node=nearest(anchor)
            if node is not None:targets[node]=anchor
        if not targets:return result(False,'No safe grid connection to terminal anchor')
        if time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted at grid connections')
        costs={first:0.};prev={};heap=[(0.,first)];visited=set();found=None
        def heuristic(node):return min(math.hypot(node[0]-t[0],node[1]-t[1])*self.grid for t in targets)
        while heap and len(visited)<self.max_nodes:
            if len(visited)%64==0 and time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted')
            _,node=heapq.heappop(heap)
            if node in visited:continue
            visited.add(node)
            if node in targets:found=node;break
            y,x=node
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                ny,nx=y+dy,x+dx
                if not(0<=ny<len(ys) and 0<=nx<len(xs)) or not free[ny,nx]:continue
                if dx and dy and not(free[y,nx] and free[ny,x]):continue
                n=(ny,nx);cost=costs[node]+math.hypot(dx,dy)*self.grid
                if cost<costs.get(n,float('inf')):
                    costs[n]=cost;prev[n]=node;heapq.heappush(heap,(cost+heuristic(n),n))
        if found is None:return result(False,'No safe route within bounded native grid search')
        nodes=[found]
        while nodes[-1]!=first:nodes.append(prev[nodes[-1]])
        points=[start]+[np.array([xs[x],ys[y]]) for y,x in reversed(nodes)]+[targets[found]]
        smooth=[points[0]];i=0
        while i<len(points)-1:
            if time.monotonic()-begun>self.deadline:return result(False,'Planning time budget exhausted during route simplification')
            j=len(points)-1
            while j>i+1 and not self._circle_clear(points[i],points[j],radius):j-=1
            smooth.append(points[j]);i=j
        # Revalidate exact final route after all shortcuts; no corner cut or
        # silent selected-object exclusion can be introduced by simplification.
        if any(not self._circle_clear(a,b,radius) for a,b in zip(smooth,smooth[1:])):
            return result(False,'Simplified route failed continuous segment check')
        waypoints=[p.tolist() for p in smooth]+[goal.tolist()]
        return result(True,'Cruise route and fixed-heading terminal corridor clear',waypoints,len(smooth)-1)
