import bpy
import math


def setup_track():
    # ==========================================
    # --- IAAF Track Dimensions (in meters) ---
    # ==========================================
    straight_length = 84.39
    inner_radius = 36.5
    lane_width = 1.22
    num_lanes = 8
    line_width = 0.05
    line_z = 0.005  # INCREASED: Prevents Cycles raytracing shadow acne
    segments_per_curve = 128
    chute_length = 30.0

    # --- Stadium Background Dimensions ---
    grass_buffer_width = 15.0
    grass_z = -0.02

    r_track_out = inner_radius + (lane_width * num_lanes)
    r_grass_out = r_track_out + grass_buffer_width

    # ==========================================
    # --- ORIGIN SHIFT CALCULATION           ---
    # ==========================================
    finish_x = straight_length / 2
    start_x_original = finish_x - 100.5
    lane_4_y_original = -(inner_radius + (3.5 * lane_width))

    O_X = -start_x_original
    O_Y = -lane_4_y_original

    def shift_verts(verts):
        return [(v[0] + O_X, v[1] + O_Y, v[2]) for v in verts]

    # ==============================================
    # --- Helper Function: Generate a track ring ---
    # ==============================================
    def get_track_ring_data(r_in, r_out, z_elev):
        verts = []
        faces = []

        def get_semi_circle(center_x, radius, is_right_side):
            pts = []
            start_angle = -math.pi / 2 if is_right_side else math.pi / 2
            end_angle = math.pi / 2 if is_right_side else 3 * math.pi / 2
            for i in range(segments_per_curve + 1):
                angle = start_angle + (end_angle - start_angle) * (i / segments_per_curve)
                pts.append((center_x + radius * math.cos(angle), radius * math.sin(angle), z_elev))
            return pts[:-1]

        in_r = get_semi_circle(straight_length / 2, r_in, True)
        in_l = get_semi_circle(-straight_length / 2, r_in, False)
        inner_verts = in_r + in_l

        out_r = get_semi_circle(straight_length / 2, r_out, True)
        out_l = get_semi_circle(-straight_length / 2, r_out, False)
        outer_verts = out_r + out_l

        verts = inner_verts + outer_verts
        num_inner = len(inner_verts)
        for i in range(num_inner):
            next_i = (i + 1) % num_inner
            # FIX: Perfect Counter-Clockwise Winding to force Normals UP (+Z)
            faces.append((i, i + num_inner, next_i + num_inner, next_i))
        return verts, faces

    # ============================
    # --- Materials Setup      ---
    # ============================
    def create_materials():
        mat_track = bpy.data.materials.new(name="Stadium_Tartan_Rubber")
        mat_track.use_nodes = True
        nodes = mat_track.node_tree.nodes
        links = mat_track.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = (0.5, 0.08, 0.05, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.85
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs['Scale'].default_value = 350.0
        bump = nodes.new(type="ShaderNodeBump")
        bump.inputs['Strength'].default_value = 0.6
        bump.inputs['Distance'].default_value = 0.005
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

        mat_white = bpy.data.materials.new(name="Stadium_White_Paint")
        mat_white.use_nodes = True
        w_bsdf = mat_white.node_tree.nodes.get("Principled BSDF")
        w_bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1.0)
        w_bsdf.inputs['Roughness'].default_value = 0.6

        mat_turf = bpy.data.materials.new(name="Stadium_Infield_Turf")
        mat_turf.use_nodes = True
        t_nodes = mat_turf.node_tree.nodes
        t_links = mat_turf.node_tree.links
        t_bsdf = t_nodes.get("Principled BSDF")
        t_bsdf.inputs['Base Color'].default_value = (0.05, 0.25, 0.03, 1.0)
        t_bsdf.inputs['Roughness'].default_value = 0.95
        t_noise = t_nodes.new(type="ShaderNodeTexNoise")
        t_noise.inputs['Scale'].default_value = 200.0
        t_noise.inputs['Detail'].default_value = 15.0
        t_bump = t_nodes.new(type="ShaderNodeBump")
        t_bump.inputs['Strength'].default_value = 0.8
        t_bump.inputs['Distance'].default_value = 0.02
        links.new(t_noise.outputs['Fac'], t_bump.inputs['Height'])
        links.new(t_bump.outputs['Normal'], t_bsdf.inputs['Normal'])

        return mat_track, mat_white, mat_turf

    mat_track, mat_white, mat_turf = create_materials()

    # =========================================
    # --- Geometry Generation & Placement ---
    # =========================================
    stadium_col = bpy.data.collections.new("0_Stadium_Environment")
    bpy.context.scene.collection.children.link(stadium_col)
    track_col = bpy.data.collections.new("1_Track")
    markers_col = bpy.data.collections.new("2_Markers_100m")
    stadium_col.children.link(track_col)
    stadium_col.children.link(markers_col)

    # 2. Generate Base Track & Chute
    t_verts, t_faces = get_track_ring_data(inner_radius, r_track_out, 0.0)
    c_off = len(t_verts)
    t_verts.extend([
        (-straight_length / 2 - chute_length, -inner_radius, 0.0),
        (-straight_length / 2, -inner_radius, 0.0),
        (-straight_length / 2, -r_track_out, 0.0),
        (-straight_length / 2 - chute_length, -r_track_out, 0.0)
    ])
    # FIX: Chute Winding Counter-Clockwise
    t_faces.append((c_off, c_off + 3, c_off + 2, c_off + 1))

    track_mesh = bpy.data.meshes.new("Track_Base_Mesh")
    track_mesh.from_pydata(shift_verts(t_verts), [], t_faces)
    track_obj = bpy.data.objects.new("1_Track_Base", track_mesh)
    track_col.objects.link(track_obj)
    track_obj.data.materials.append(mat_track)
    for poly in track_obj.data.polygons: poly.use_smooth = True

    # 3. Generate Lane Lines & Chute Lines
    l_verts, l_faces = [], []
    offset = 0
    for i in range(num_lanes + 1):
        center_radius = inner_radius + (i * lane_width)
        v, f = get_track_ring_data(center_radius - (line_width / 2), center_radius + (line_width / 2), line_z)
        cy = -center_radius
        c_v_off = len(v)
        v.extend([
            (-straight_length / 2 - chute_length, cy - line_width / 2, line_z),
            (-straight_length / 2, cy - line_width / 2, line_z),
            (-straight_length / 2, cy + line_width / 2, line_z),
            (-straight_length / 2 - chute_length, cy + line_width / 2, line_z)
        ])
        f.append((c_v_off, c_v_off + 1, c_v_off + 2, c_v_off + 3))
        for face in f: l_faces.append(tuple(idx + offset for idx in face))
        l_verts.extend(v)
        offset += len(v)

    lines_mesh = bpy.data.meshes.new("Track_Lines_Mesh")
    lines_mesh.from_pydata(shift_verts(l_verts), [], l_faces)
    lines_obj = bpy.data.objects.new("2_Track_Lines", lines_mesh)
    track_col.objects.link(lines_obj)
    lines_obj.data.materials.append(mat_white)

    # 4. Generate Finish & 100m Start Lines
    start_x = finish_x - 100

    def create_horizontal_line(name, x_pos):
        verts = [(x_pos - line_width, -inner_radius, line_z * 2), (x_pos + line_width, -inner_radius, line_z * 2),
                 (x_pos + line_width, -r_track_out, line_z * 2), (x_pos - line_width, -r_track_out, line_z * 2)]
        mesh = bpy.data.meshes.new(f"{name}_Mesh")
        # FIX: Winding Counter-Clockwise
        mesh.from_pydata(shift_verts(verts), [], [(0, 3, 2, 1)])
        obj = bpy.data.objects.new(f"3_{name}", mesh)
        track_col.objects.link(obj)
        obj.data.materials.append(mat_white)

    create_horizontal_line("Finish_Line", finish_x)
    create_horizontal_line("100m_Start_Line", start_x)

    # 4.5 Generate 5m Notches on BOTH sides of ALL lane lines
    def create_lane_notch(x_pos, y_center, length=0.35, width=0.025, mode="both"):
        verts = []
        faces = []
        idx = 0

        # TOP side
        if mode in ("both", "top"):
            verts.extend([
                (x_pos - width / 2, y_center + width / 2, line_z * 2),
                (x_pos + width / 2, y_center + width / 2, line_z * 2),
                (x_pos + width / 2, y_center + width / 2 + length, line_z * 2),
                (x_pos - width / 2, y_center + width / 2 + length, line_z * 2),
            ])
            faces.append((idx, idx + 3, idx + 2, idx + 1))
            idx += 4

        # BOTTOM side
        if mode in ("both", "bottom"):
            verts.extend([
                (x_pos - width / 2, y_center - width / 2, line_z * 2),
                (x_pos + width / 2, y_center - width / 2, line_z * 2),
                (x_pos + width / 2, y_center - width / 2 - length, line_z * 2),
                (x_pos - width / 2, y_center - width / 2 - length, line_z * 2),
            ])
            faces.append((idx, idx + 1, idx + 2, idx + 3))

        mesh = bpy.data.meshes.new("Notch_Mesh")
        mesh.from_pydata(shift_verts(verts), [], faces)

        obj = bpy.data.objects.new("5m_Notch", mesh)
        track_col.objects.link(obj)
        obj.data.materials.append(mat_white)

    # Create notches every 5m for all lane boundaries
    for i in range(5, 100, 5):
        x = start_x + i

        for lane_idx in range(num_lanes + 1):
            y = -(inner_radius + lane_idx * lane_width)
            if lane_idx == 0:
                create_lane_notch(x, y, mode="bottom")  # only into track
            elif lane_idx == num_lanes:
                create_lane_notch(x, y, mode="top")  # only into track
            else:
                create_lane_notch(x, y, mode="both")

    # 5. Generate Infield Turf
    turf_verts = []
    # FIX: Sweeping Counter-Clockwise
    for i in range(segments_per_curve):
        angle = -math.pi / 2 + math.pi * (i / segments_per_curve)
        turf_verts.append(
            (straight_length / 2 + inner_radius * math.cos(angle), inner_radius * math.sin(angle), grass_z))
    for i in range(segments_per_curve):
        angle = math.pi / 2 + math.pi * (i / segments_per_curve)
        turf_verts.append(
            (-straight_length / 2 + inner_radius * math.cos(angle), inner_radius * math.sin(angle), grass_z))
    turf_mesh = bpy.data.meshes.new("Turf_Mesh")
    turf_mesh.from_pydata(shift_verts(turf_verts), [], [list(range(len(turf_verts)))])
    turf_obj = bpy.data.objects.new("4_Infield_Turf", turf_mesh)
    track_col.objects.link(turf_obj)
    turf_obj.data.materials.append(mat_turf)

    # 6. Generate 3D Vertical 100m Markers
    for i in range(0, 101, 10):
        text_curve = bpy.data.curves.new(type="FONT", name=f"Marker_{i}m")
        text_curve.body = f"{i}m"
        text_curve.extrude = 0.05
        text_curve.size = 2.0
        text_curve.align_x = 'CENTER'
        text_curve.align_y = 'BOTTOM'

        text_obj = bpy.data.objects.new(f"Marker_{i}m", text_curve)
        markers_col.objects.link(text_obj)
        text_obj.location = (start_x + i + O_X, -inner_radius + 5.0 + O_Y, 0.05)
        text_obj.rotation_euler = (math.pi / 2, 0, 0)
        text_obj.data.materials.append(mat_white)

    # ==========================================
    # --- 7. Wrap-Around Grass Buffer        ---
    # ==========================================
    def get_stadium_path(radius):
        pts = []
        for i in range(segments_per_curve + 1):
            angle = -math.pi / 2 + (math.pi * i / segments_per_curve)
            pts.append((straight_length / 2 + radius * math.cos(angle), radius * math.sin(angle)))
        for i in range(segments_per_curve + 1):
            angle = math.pi / 2 + (math.pi * i / segments_per_curve)
            pts.append((-straight_length / 2 + radius * math.cos(angle), radius * math.sin(angle)))
        return pts

    # Outer Grass Buffer
    g_verts, g_faces = [], []
    in_path = get_stadium_path(r_track_out)
    out_path = get_stadium_path(r_grass_out)
    for p in in_path: g_verts.append((p[0], p[1], grass_z))
    for p in out_path: g_verts.append((p[0], p[1], grass_z))
    n = len(in_path)
    for i in range(n - 1):
        # FIX: Winding Counter-Clockwise
        g_faces.append((i, i + n, i + 1 + n, i + 1))
    g_faces.append((n - 1, 2 * n - 1, n, 0))

    grass_mesh = bpy.data.meshes.new("Outer_Grass_Mesh")
    grass_mesh.from_pydata(shift_verts(g_verts), [], g_faces)
    grass_obj = bpy.data.objects.new("5_Outer_Grass", grass_mesh)
    stadium_col.objects.link(grass_obj)
    grass_obj.data.materials.append(mat_turf)

    print("Stadium generated with track, infield, and outer grass. Cycles is safe!")
