#!/usr/bin/env python3
"""Source-grounded instance inventory for IMG_2296.MOV; CPU-only, no scene mutation.

Annotations below were manually reviewed in the original 1920x1080 source frames.
They are approximate visible-object regions, not pixel-accurate segmentation.
Optional sparse support is a conservative envelope of observed feature points,
never a complete object AABB, collision shape, metric measurement, or mesh label.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def observation(frame, region, note='Visible extent; approximate manual boundary.'):
    result = {'frame': f'frame_{frame:06d}.jpg', 'note': note}
    result['polygon_xy' if isinstance(region[0], (tuple, list)) else 'bbox_xyxy'] = region
    return result


def entity(instance_id, category, label, observations, *, parent=None, uncertainty='', articulation=None, role='navigation_obstacle'):
    return dict(instance_id=instance_id, category=category, label=label, parent_instance_id=parent,
                observations=observations, identity_basis='Manually assigned from the cited visible instance and its surroundings.',
                uncertainty=uncertainty or 'Hidden faces and dimensions are unobserved; identity is limited to the cited source evidence.',
                experiment_role=role, articulation_hypothesis=None if articulation is None else
                dict(type=articulation[0], evidence_status='inferred_from_appearance_not_measured', note=articulation[1],
                     axis=None, pivot=None, limits=None, damping=None, friction=None),
                physical_parameters_status='not_measured', complete_geometry_available=False)


def annotations():
    O = observation
    E = entity
    objects = [
        E('island_sink_station_01', 'sink_station', 'Seating island with sink and three stools',
          [O(1488, [(640,449),(723,403),(1315,389),(1338,442),(1324,697),(687,701)], 'Island assembly; stools are separate children.'),
           O(250, [(0,636),(235,569),(1159,687),(1125,1080),(0,1080)], 'Partly cropped island; foreground stools occlude its base.'),
           O(2158, [(57,346),(991,75),(1412,142),(1327,517),(831,1080),(551,1080),(297,999),(207,417)], 'Oblique island view, with lower front partly cropped.')],
          uncertainty='Distinct from sorting_island_01: this island has an inset sink, tall faucet, seating overhang, and side electrical outlets.', role='manipulation_surface_and_obstacle'),
        E('island_sink_basin_01', 'sink_basin', 'Inset double basin in seating island',
          [O(250, [(144,625),(275,590),(521,625),(405,664),(215,642)]), O(1488, [726,400,951,431]),
           O(1687,[1545,312,1880,368], 'Distant island double basin; the separate single wall basin is visible at lower left.')],
          parent='island_sink_station_01', uncertainty='Separate from wall_sink_basin_01, visible simultaneously in both cited frames. Basin interior depth is unmeasured.', role='manipulation_target'),
        E('island_faucet_01', 'faucet', 'Tall arched faucet on seating island',
          [O(250,[177,368,301,619]), O(1488,[795,267,835,430])], parent='island_sink_station_01',
          articulation=('possible_spout_or_handle_rotation','Appearance suggests plumbing controls; no motion, axes, or limits were observed.'), role='manipulation_candidate'),
        E('wall_sink_station_01', 'sink_station', 'Wall double-sink station by dishwasher and glass-front appliance',
          [O(297,[709,259,1199,708]), O(1488,[1364,317,1584,510])],
          uncertainty='Counter/sink-base assembly only. Distinct from wall_sink_station_02 beneath two displays on the other side of the refrigerator.', role='manipulation_surface_and_obstacle'),
        E('wall_sink_basin_01', 'sink_basin', 'Double basin in wall counter',
          [O(297,[(767,308),(786,270),(1136,270),(1157,314)]), O(250,[1513,522,1840,578]), O(1488,[1380,318,1558,344])],
          parent='wall_sink_station_01', uncertainty='Not the island sink. Both are visible in frame000250 and frame001488.', role='manipulation_target'),
        E('wall_faucet_01', 'faucet', 'Tall faucet at wall sink',
          [O(297,[938,4,1010,269]), O(1488,[1450,204,1499,338])], parent='wall_sink_station_01',
          articulation=('possible_spout_or_handle_rotation','Control mechanism inferred; no motion sequence or physical parameters measured.'), role='manipulation_candidate'),
        E('wall_sink_door_left_01', 'cabinet_door', 'Left cabinet door below wall sink',
          [O(297,[718,338,959,675])], parent='wall_sink_station_01',
          articulation=('revolute_candidate','Vertical door with pull handle; hinge side, pivot, and limits not measured.'), role='manipulation_candidate'),
        E('wall_sink_door_right_01', 'cabinet_door', 'Right cabinet door below wall sink',
          [O(297,[959,341,1194,681])], parent='wall_sink_station_01',
          articulation=('revolute_candidate','Vertical door with pull handle; no observed opening motion.'), role='manipulation_candidate'),
        E('dishwasher_01', 'dishwasher', 'Stainless dishwasher immediately left of wall sink',
          [O(297,[392,329,730,692]), O(250,[1243,544,1482,814]), O(1488,[1234,347,1365,513], 'Lower part partly occluded by seating island.')],
          uncertainty='One dishwasher is positively identified in the reviewed kitchen views; no second distinct dishwasher is asserted.',
          articulation=('bottom_hinged_door_candidate','Appliance appearance suggests a downward-opening door; mechanism not exercised.'), role='manipulation_candidate'),
        E('refrigerator_01', 'refrigerator', 'Tall stainless refrigerator with two upper doors and lower drawer',
          [O(1488,[0,182,370,789]), O(796,[477,93,815,621])],
          uncertainty='Door and drawer layout is visible; model, contents, seals, mass, and opening range are unknown.', role='manipulation_candidate'),
        E('refrigerator_upper_left_door_01', 'appliance_door', 'Refrigerator upper left door',
          [O(1488,[(78,205),(233,192),(265,521),(162,567)])], parent='refrigerator_01',
          articulation=('revolute_candidate','Left upper door inferred from split face and handle; hinge/pivot unmeasured.'), role='manipulation_candidate'),
        E('refrigerator_upper_right_door_01', 'appliance_door', 'Refrigerator upper right door',
          [O(1488,[(233,192),(307,187),(355,482),(265,521)])], parent='refrigerator_01',
          articulation=('revolute_candidate','Right upper door inferred from split face and handle; no motion observed.'), role='manipulation_candidate'),
        E('refrigerator_lower_drawer_01', 'appliance_drawer', 'Refrigerator lower pull-out compartment',
          [O(1488,[(163,566),(354,505),(368,674),(196,769)])], parent='refrigerator_01',
          articulation=('prismatic_candidate','Lower horizontal handle suggests drawer translation; travel and contents unknown.'), role='manipulation_candidate'),
        E('range_oven_01', 'range_oven', 'Range and oven beneath stainless hood',
          [O(250,[697,458,1031,666], 'Lower oven partly occluded by island.'), O(1488,[838,339,1041,408], 'Most lower appliance occluded by island.')],
          articulation=('oven_door_revolute_candidate','Oven door suggested by appliance front; no opening motion recorded.'), role='manipulation_candidate'),
        E('range_hood_01', 'extractor_hood', 'Stainless wall hood over range',
          [O(250,[674,0,1056,198]), O(1488,[812,0,1056,166])], role='static_landmark'),
        E('microwave_01', 'microwave', 'Countertop microwave in lower open cabinet niche',
          [O(398,[0,460,373,647]), O(2158,[1328,83,1526,265])],
          articulation=('door_revolute_candidate','Front window and side controls identify a microwave; hinge and operating state unmeasured.'), role='manipulation_candidate'),
        E('lower_cabinet_ajar_door_01', 'cabinet_door', 'Ajar lower cabinet door near microwave and corner',
          [O(398,[(932,492),(1131,532),(1110,828),(922,772)], 'Door visibly projects from cabinet face.'),
           O(1488,[(511,416),(572,411),(586,531),(522,554)], 'Same corner door seen at a distance; boundary approximate.')],
          articulation=('revolute_candidate','Door is visibly ajar. Hinge axis, opening angle, limits, latch, and required force are unmeasured.'), role='manipulation_target'),
        E('lower_cabinet_latched_pair_01', 'cabinet_pair', 'Closed lower cabinet pair with dark handle strap',
          [O(398,[399,490,932,778])],
          uncertainty='A dark strap/retainer spans the two handles. Its locking function is not established.',
          articulation=('two_revolute_candidates','Two visible leaves; strap may constrain opening. Do not simulate free opening without inspection.'), role='manipulation_candidate'),
        E('upper_cabinet_bank_01', 'cabinet_bank', 'Wood upper cabinet bank around kitchen corner',
          [O(398,[0,0,1920,343], 'Bank is partly cropped; this group is not per-door segmentation.'), O(1488,[78,88,781,348])],
          uncertainty='Several separate leaves are visible; complete leaf count and cross-view correspondence are not assigned.',
          articulation=('multiple_revolute_candidates','Door bank only; individual hinges and hidden interior remain unresolved.'), role='manipulation_candidate'),
        E('corner_counter_surface_01', 'counter_surface', 'White counter surface above microwave/corner cabinets',
          [O(398,[(0,314),(1374,329),(1919,593),(1919,803),(1587,778),(1358,411),(0,397)])], role='pick_and_place_surface'),
        E('sorting_island_01', 'sorting_bin_island', 'Separate island with open waste-bin apertures',
          [O(507,[(163,155),(388,86),(1306,157),(1261,276),(1201,691),(247,550)]),
           O(1582,[(0,80),(634,0),(1479,0),(1702,56),(1548,541),(677,1080),(381,1080),(145,827)], 'Opposite side / oblique view; some boundaries cropped.')],
          uncertainty='Distinct from island_sink_station_01. No sink or faucet is visible on this unit. Printed recycling/compost labels repeat on both islands; labels alone do not identify instances.',
          role='manipulation_surface_and_obstacle'),
        E('sorting_panel_recycle_left_01', 'cabinet_door', 'Sorting island: left recycle panel in frame000507',
          [O(507,[(194,254),(433,280),(463,580),(247,550)])], parent='sorting_island_01',
          articulation=('revolute_candidate','Handle and panel seam suggest hinge motion; counterpart on opposite side not assigned.'), role='manipulation_candidate'),
        E('sorting_panel_recycle_right_01', 'cabinet_door', 'Sorting island: second recycle panel in frame000507',
          [O(507,[(433,280),(677,306),(693,615),(463,580)])], parent='sorting_island_01',
          articulation=('revolute_candidate','Door structure inferred; no opening motion observed.'), role='manipulation_candidate'),
        E('sorting_panel_compost_01', 'cabinet_door', 'Sorting island: compost panel in frame000507',
          [O(507,[(677,306),(944,337),(939,652),(693,615)])], parent='sorting_island_01',
          articulation=('revolute_candidate','Panel label is not a unique place identifier; hinge parameters unknown.'), role='manipulation_candidate'),
        E('sorting_panel_landfill_01', 'cabinet_door', 'Sorting island: landfill panel in frame000507',
          [O(507,[(944,337),(1238,361),(1200,691),(939,652)])], parent='sorting_island_01',
          articulation=('revolute_candidate','Door type is inferred from handle/seam; motion not observed.'), role='manipulation_candidate'),
        E('stool_island_left_01', 'bar_stool', 'Left stool in frontal frame001488', [O(1488,[711,445,908,745])], parent='island_sink_station_01'),
        E('stool_island_middle_01', 'bar_stool', 'Middle stool in frontal frame001488', [O(1488,[890,437,1085,744])], parent='island_sink_station_01'),
        E('stool_island_right_01', 'bar_stool', 'Right stool in frontal frame001488', [O(1488,[1084,429,1279,751])], parent='island_sink_station_01'),
        E('table_round_foreground_01', 'round_table', 'Foreground round table at start of clip', [O(0,[241,368,1023,1080])], role='pick_and_place_surface'),
        E('table_round_kitchen_side_02', 'round_table', 'Round table nearest kitchen in frame000000', [O(0,[369,151,836,382])],
          uncertainty='Base partly occluded by foreground table. Similar tables are kept distinct by same-frame evidence.', role='pick_and_place_surface'),
        E('table_round_glass_side_03', 'round_table', 'Round table beside glass in frame000000', [O(0,[1073,191,1606,714])], role='pick_and_place_surface'),
        E('chair_table_left_01', 'chair', 'Black chair left of tables in frame000000', [O(0,[0,203,326,657])]),
        E('chair_table_right_02', 'chair', 'Black chair right of tables in frame000000', [O(0,[1418,185,1846,672])]),
        E('chair_wall_left_03', 'chair', 'Left black chair against distant whiteboard wall', [O(2229,[71,103,167,233])],
          uncertainty='Small blurred distant observation. Distinct from nearby table chairs by simultaneous visibility.'),
        E('chair_wall_right_04', 'chair', 'Right black chair against distant whiteboard wall', [O(2229,[157,102,241,227])], uncertainty='Small blurred distant observation; hidden legs/shape unresolved.'),
        E('table_occluded_beyond_column_04', 'table_candidate', 'Additional wood-top pedestal table partly behind column', [O(2229,[554,208,704,412])],
          uncertainty='An additional tabletop and pedestal are partly visible beyond the three round tables. Full outline and exact table shape are unresolved.', role='navigation_obstacle'),
        E('service_cart_near_bin_01', 'service_cart', 'Black service cart directly beside blue floor bin',
          [O(689,[(826,164),(1148,144),(1684,499),(1717,656),(1412,1080),(1116,1080),(849,390)]), O(2081,[1519,174,1738,535])],
          uncertainty='One of two simultaneously visible carts. Individual casters are only partly visible.',
          articulation=('caster_wheels_candidate','Wheel rotation and caster swivel are inferred; brakes, axes, masses and rolling resistance unknown.')),
        E('service_cart_outer_02', 'service_cart', 'Second service cart on outer side of cart pair',
          [O(689,[(1157,98),(1453,72),(1919,397),(1919,702),(1763,954),(1549,791),(1681,476),(1210,239)]), O(2081,[1357,176,1579,513])],
          uncertainty='Separate from near-bin cart in the same frames; back portions can occlude each other.',
          articulation=('caster_wheels_candidate','Wheel and swivel motion inferred from casters, not measured.')),
        E('blue_floor_bin_01', 'freestanding_bin', 'Tall blue bin between sorting island and carts',
          [O(689,[765,413,1198,1006], 'Lower bin partly occluded.'), O(2081,[1683,257,1844,548])]),
        E('cooler_01', 'portable_cooler', 'Blue and white cooler by glass column',
          [O(2081,[164,557,612,968]), O(1488,[1850,527,1920,742], 'Mostly cropped at right frame edge.')],
          articulation=('lid_revolute_candidate','Lid is visible; hinge, latch, and opening forces not measured.'), role='manipulation_candidate'),
        E('island_sink_board_or_tray_01', 'loose_object_candidate', 'Tan rectangular item partly inside island sink',
          [O(250,[333,628,492,677])], parent='island_sink_station_01',
          uncertainty='May be a board or tray. Partly occluded; material, dimensions, thickness and graspable edges unresolved.', role='pick_and_place_candidate'),
        E('counter_dish_stack_01', 'dish_candidate', 'Small round pale dish or dish stack on corner counter',
          [O(398,[951,326,1045,360])], parent='corner_counter_surface_01',
          uncertainty='Number of dishes and exact object type are unresolved at source resolution.', role='pick_and_place_candidate'),
        E('display_kitchen_01', 'display', 'Wall display above wall sink', [O(250,[1338,69,1803,332]), O(1488,[1252,54,1517,216])],
          uncertainty='Physical screen is a static landmark; changing video content must not become static map geometry.', role='static_housing_dynamic_content'),
        E('counter_glass_appliance_01', 'appliance_unknown', 'Dark glass-front countertop enclosure at wall sink',
          [O(297,[1190,0,1462,347]), O(1488,[1563,196,1698,339])],
          uncertainty='Function/model is not established. Reflective transparent front can confuse reconstruction.',
          articulation=('door_candidate','Visible framed front suggests an opening panel; mechanism unknown.'), role='manipulation_candidate'),
        E('paper_dispenser_01', 'dispenser', 'Black wall dispenser left of wall faucet', [O(297,[619,28,800,234])],
          uncertainty='Paper-like sheet is visible below the housing; exact dispenser mechanism unknown.', role='manipulation_candidate'),
        E('soap_dispenser_candidate_01', 'dispenser_candidate', 'Slim black dispenser right of wall faucet', [O(297,[1086,64,1153,217])],
          uncertainty='Function inferred from location and shape; contents and actuation unknown.', role='manipulation_candidate'),
        E('pendant_left_01', 'ceiling_light', 'Left pendant in frontal kitchen view', [O(1488,[721,0,785,103])], role='appearance_landmark'),
        E('pendant_middle_02', 'ceiling_light', 'Middle pendant in frontal kitchen view', [O(1488,[956,0,1018,99])], role='appearance_landmark'),
        E('pendant_right_03', 'ceiling_light', 'Right pendant in frontal kitchen view', [O(1488,[1201,0,1269,76])], role='appearance_landmark'),
        E('glass_partition_sink_end_01', 'glass_partition_region', 'Glazed partition bay beside wall sink',
          [O(1081,[(369,0),(1533,0),(1496,613),(397,634)])],
          uncertainty='Visible equipment behind the glass is not the glass surface. Pane thickness, coating, seams and LiDAR return behavior are unmeasured.', role='localization_and_collision_critical'),
        E('glass_partition_tables_side_02', 'glass_partition_region', 'Glazed bay behind round tables',
          [O(0,[(912,0),(1919,0),(1878,399),(922,312)]), O(2229,[(828,0),(1919,0),(1844,642),(830,469)])],
          uncertainty='Glass and transmitted/reflected laboratory contents are separate depth layers. This region is not a fitted plane or complete pane segmentation.', role='localization_and_collision_critical'),
        E('glazed_door_candidate_01', 'glazed_door_candidate', 'Door-like framed glazed panel beyond sink-end bay',
          [O(1081,[642,0,891,424])],
          uncertainty='A framed glazed panel with lower kickplate is visible, but front-versus-background plane and door motion are not resolved.',
          articulation=('revolute_candidate_unconfirmed','Door classification and hinge placement require closer inspection.'), role='localization_boundary_candidate'),
        E('column_near_cooler_01', 'structural_column', 'White column beside cooler and glass bay',
          [O(2081,[439,0,602,605])], role='localization_and_collision_critical'),
        E('floor_surface_region_01', 'floor_region', 'Visible kitchen circulation floor',
          [O(1488,[(265,789),(1427,761),(1799,802),(1729,921),(479,903)])],
          uncertainty='Annotated clear patch only, not a complete floor or measured plane.', role='locomotion_surface'),
        E('laboratory_contents_region_01', 'unresolved_background_group', 'Equipment and furniture visible through glass',
          [O(1081,[413,171,1453,610]), O(2229,[836,0,1920,612])],
          uncertainty='Non-exhaustive group of machinery, cases, benches, monitors and chairs behind glass. Individual identities and depth layers are unresolved; people/screens can be dynamic.', role='appearance_context_unresolved_geometry'),
        E('wall_sink_station_02', 'sink_station', 'Second wall station: single basin beneath two displays, beside toaster',
          [O(847,[859,631,1919,1079]), O(1807,[506,270,1478,1000]),
           O(1687,[0,284,964,1079], 'Station beside refrigerator, visibly separate from island sink at right.')],
          uncertainty='THIRD sink station overall. Single basin, secondary small tap, dark handle straps, dual overhead displays and toaster distinguish it from the dishwasher-side double basin.', role='manipulation_surface_and_obstacle'),
        E('wall_sink_basin_02', 'sink_basin', 'Single basin at dual-display wall station',
          [O(847,[1326,660,1529,707]), O(1807,[(817,271),(1171,290),(1225,380),(795,366)]),
           O(1687,[(86,938),(370,744),(645,758),(487,983)], 'Island double basin is also visible at upper right of this source frame.')],
          parent='wall_sink_station_02', uncertainty='Physically distinct from wall_sink_basin_01 and island_sink_basin_01; this basin has no central divider.', role='manipulation_target'),
        E('wall_faucet_02', 'faucet', 'Tall faucet at dual-display wall station',
          [O(847,[1392,473,1470,667]), O(1807,[975,0,1102,276])], parent='wall_sink_station_02',
          articulation=('possible_spout_or_handle_rotation','No opening/rotation motion, axes, limits or forces measured.'), role='manipulation_candidate'),
        E('wall_secondary_tap_02', 'tap', 'Small secondary tap beside tall faucet at single basin',
          [O(847,[1466,585,1525,671]), O(1807,[1122,90,1200,276])], parent='wall_sink_station_02',
          uncertainty='A distinct small tap is visible; water service and function are unverified.',
          articulation=('handle_rotation_candidate','Small lever suggested by appearance only.'), role='manipulation_candidate'),
        E('wall_sink_02_latched_pair', 'cabinet_pair', 'Latched cabinet pair beneath single wall basin',
          [O(1807,[(525,410),(1475,454),(1394,995),(548,952)])], parent='wall_sink_station_02',
          uncertainty='Dark strap/retainer spans both handles; locking behavior and hidden hinge mechanism are unknown.',
          articulation=('two_revolute_candidates','Do not treat both leaves as freely opening while the visible retainer is present.'), role='manipulation_candidate'),
        E('toaster_oven_01', 'toaster_oven', 'Small countertop oven beside single wall sink',
          [O(1807,[1477,88,1872,333]), O(796,[106,308,267,400])], parent='wall_sink_station_02',
          articulation=('front_door_revolute_candidate','Door and controls visible; exact model and operating mechanism not measured.'), role='manipulation_candidate'),
        E('display_dual_left_02', 'display', 'Left of two displays above single wall sink', [O(847,[960,90,1395,354])],
          uncertainty='Separate from display_kitchen_01 above the dishwasher-side wall basin. Screen content is dynamic.', role='static_housing_dynamic_content'),
        E('display_dual_right_03', 'display', 'Right of two displays above single wall sink', [O(847,[1380,91,1852,362])],
          uncertainty='Screen shows another sink in video content; that screen image is not an additional physical sink in this room.', role='static_housing_dynamic_content'),
        E('paper_dispenser_02', 'dispenser', 'Black wall dispenser at single wall basin', [O(847,[1211,485,1342,633])], parent='wall_sink_station_02', role='manipulation_candidate'),
        E('soap_dispenser_candidate_02', 'dispenser_candidate', 'Slim dispenser at single wall basin', [O(847,[1542,519,1602,635])], parent='wall_sink_station_02',
          uncertainty='Function and contents inferred from shape and position only.', role='manipulation_candidate'),
        E('water_pitcher_candidate_01', 'loose_object_candidate', 'Translucent pitcher-like vessel left of single wall basin',
          [O(847,[904,537,1005,640])], parent='wall_sink_station_02',
          uncertainty='Separate body and nearby lid-like part are visible; actual material, contents, mass and object count unresolved.', role='pick_and_place_candidate'),
        E('blue_liquid_bottle_01', 'bottle', 'Clear bottle with blue contents beside toaster',
          [O(1807,[1380,75,1502,294]), O(796,[76,293,120,399])], parent='wall_sink_station_02',
          uncertainty='Contents, brand, cap mechanism and mass are unverified.', role='pick_and_place_candidate'),
        E('glass_door_entry_candidate_02', 'glazed_door_candidate', 'Glazed panel with long metal bar beside single-sink wall',
          [O(858,[(0,0),(416,0),(431,891),(0,1032)])],
          uncertainty='Long metal bar suggests a door pull; pane boundaries, hinge placement and opening direction are unresolved.',
          articulation=('revolute_candidate_unconfirmed','Door classification is inferred from visible hardware, not demonstrated motion.'), role='localization_boundary_candidate'),
    ]
    return objects


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):h.update(block)
    return h.hexdigest()


def polygon(obs):
    if 'polygon_xy' in obs:return np.asarray(obs['polygon_xy'], np.float32)
    x0, y0, x1, y1 = obs['bbox_xyxy']
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)


def sparse_support(model, objects, frames):
    """Intersect actual tracked features from >=2 annotated observations.

    The quantile envelope intentionally excludes unseen object extents. It must
    never be interpreted as a collision AABB or whole-object segmentation.
    """
    import pycolmap
    names = {p.name: sha(p) for p in model.glob('*.bin')}
    reconstruction = pycolmap.Reconstruction(str(model))
    images = {im.name: im for im in reconstruction.images.values()}
    for obj in objects:
        support = defaultdict(set)
        used = []
        for obs in obj['observations']:
            im = images.get(obs['frame'])
            if im is None:continue
            poly = polygon(obs)
            used.append(obs['frame'])
            for point in im.points2D:
                if not point.has_point3D():continue
                p3 = reconstruction.points3D[point.point3D_id]
                if p3.error > 2 or p3.track.length() < 3:continue
                if cv2.pointPolygonTest(poly, tuple(map(float, point.xy)), False) >= 0:
                    support[int(point.point3D_id)].add(obs['frame'])
        ids = [pid for pid, views in support.items() if len(views) >= 2 and
               max(frames[n]['time_seconds'] for n in views) - min(frames[n]['time_seconds'] for n in views) >= 2]
        obj['sparse_visible_support'] = dict(status='insufficient_multiview_support', candidate_points=len(ids),
                                            registered_annotation_frames=used, units='arbitrary', collision_use_allowed=False,
                                            is_complete_object_extent=False, segmentation_verified=False)
        if len(ids) < 15:continue
        xyz = np.array([reconstruction.points3D[pid].xyz for pid in ids])
        center = np.median(xyz, axis=0)
        distance = np.linalg.norm(xyz-center, axis=1)
        cutoff = np.median(distance) + 4.5 * max(np.median(np.abs(distance-np.median(distance))), 1e-8)
        keep = distance <= cutoff
        if keep.sum() < 15:continue
        xyz = xyz[keep];ids = np.array(ids)[keep]
        obj['sparse_visible_support'].update(status='visible_feature_envelope_only', retained_points=len(ids),
                                             point3D_ids=ids.tolist(), median_xyz=np.median(xyz, axis=0).tolist(),
                                             q02_xyz=np.quantile(xyz,.02,axis=0).tolist(), q98_xyz=np.quantile(xyz,.98,axis=0).tolist(),
                                             outlier_rule='radial median + 4.5*MAD; envelope is per-axis 2nd–98th percentile',
                                             caveat='Background and occluder features may enter approximate regions. This is neither full instance segmentation nor physical size.')
    after = {p.name: sha(p) for p in model.glob('*.bin')}
    if names != after:raise RuntimeError('Sparse model changed during read-only inventory support analysis')
    return dict(path=str(model.resolve()), binary_hashes=names, pycolmap_version=pycolmap.__version__, units='arbitrary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'crops').mkdir(exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    raw = args.frames.read_bytes();manifest = json.loads(raw);frames = {v['name']:v for v in manifest['frames']}
    objects = annotations();ids = [o['instance_id'] for o in objects]
    assert len(ids) == len(set(ids))
    model_input = sparse_support(args.model,objects,frames) if args.model else None
    per_frame = defaultdict(list);media = []
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fontpath,18) if Path(fontpath).exists() else ImageFont.load_default()
    for number,obj in enumerate(objects,1):
        assert obj['parent_instance_id'] is None or obj['parent_instance_id'] in ids
        obj['display_number'] = number
        for index,obs in enumerate(obj['observations']):
            frame = frames[obs['frame']];path = Path(frame['image_path']);image_hash = sha(path)
            assert image_hash == frame['sha256'], f'Source hash changed: {path}'
            assert frame['split'] == 'train', 'This inventory intentionally uses training views only'
            poly = polygon(obs);assert poly.shape[1] == 2 and len(poly) >= 3 and np.isfinite(poly).all()
            assert (poly[:,0]>=0).all() and (poly[:,0]<=frame['width']).all()
            assert (poly[:,1]>=0).all() and (poly[:,1]<=frame['height']).all()
            lo = poly.min(axis=0);hi = poly.max(axis=0);assert (hi > lo).all()
            obs.update(source_pts_seconds=frame['time_seconds'], source_index=frame['source_index'],
                       image_path=str(path.resolve()), image_sha256=image_hash, source_width=frame['width'], source_height=frame['height'],
                       split=frame['split'], annotation_method='manual_visual_review',
                       annotation_semantics='approximate visible-instance region; not a segmentation mask',
                       box_extent='visible_or_cropped_not_amodal', canonical_bbox_xyxy=np.r_[lo,hi].tolist())
            pad = max(12,int(max(hi-lo)*.07));x0,y0=np.maximum(lo-pad,0).astype(int);x1,y1=np.minimum(hi+pad,[frame['width'],frame['height']]).astype(int)
            with Image.open(path) as im:
                crop=im.convert('RGB').crop((x0,y0,x1,y1));draw=ImageDraw.Draw(crop)
                draw.line([tuple(p-[x0,y0]) for p in poly]+[tuple(poly[0]-[x0,y0])],fill=(255,70,30),width=3)
                crop.thumbnail((560,400));tile=Image.new('RGB',(560,435),(22,26,33));tile.paste(crop,((560-crop.width)//2,0))
                ImageDraw.Draw(tile).text((7,408),f'{number:02d}  {obs["frame"]}  {frame["time_seconds"]:.3f}s',font=font,fill='white')
                name=f'crops/{obj["instance_id"]}_{index:02d}.jpg';tile.save(out/name,quality=94)
            obs['review_crop'] = name;obs['review_crop_sha256']=sha(out/name)
            per_frame[obs['frame']].append((number,obj['instance_id'],poly))
    for name,annotations_for_frame in per_frame.items():
        with Image.open(frames[name]['image_path']) as image:
            image=image.convert('RGB');draw=ImageDraw.Draw(image)
            for number,iid,poly in annotations_for_frame:
                color=(255,70+number*37%170,20+number*71%220)
                draw.line([tuple(p) for p in poly]+[tuple(poly[0])],fill=color,width=4)
                x,y=poly.min(axis=0);draw.rectangle((x,y,x+36,y+25),fill=(20,20,20));draw.text((x+3,y+1),str(number),font=font,fill=color)
            image.save(out/'frames'/name,quality=94)
        media.append(dict(path='frames/'+name,sha256=sha(out/'frames'/name),source_image_sha256=frames[name]['sha256']))
    for obj in objects:
        for obs in obj['observations']:obs['annotated_frame']='frames/'+obs['frame']
    # Explicit visual controls for the repeated-sink ambiguity. The display
    # content itself contains sink imagery; only physical basin regions count.
    control_specs = [(250,['island_sink_basin_01','wall_sink_basin_01']),
                     (1687,['wall_sink_basin_02','island_sink_basin_01']),
                     (297,['wall_sink_basin_01']), (847,['wall_sink_basin_02'])]
    lookup = {o['instance_id']:o for o in objects}
    controls=[];control_sheet=Image.new('RGB',(1920,1180),(18,23,31))
    for i,(frame_index,instance_ids) in enumerate(control_specs):
        name=f'frame_{frame_index:06d}.jpg';frame=frames[name]
        with Image.open(frame['image_path']) as image:
            image=image.convert('RGB');draw=ImageDraw.Draw(image);regions=[]
            for j,instance_id in enumerate(instance_ids):
                obs=next(o for o in lookup[instance_id]['observations'] if o['frame']==name)
                poly=polygon(obs);color=[(255,110,15),(40,240,120)][j%2]
                draw.line([tuple(p) for p in poly]+[tuple(poly[0])],fill=color,width=8)
                x,y=poly.min(axis=0);y=max(2,y-30);draw.rectangle((x,y,x+420,y+28),fill=(18,23,31));draw.text((x+4,y+3),instance_id,font=font,fill=color)
                regions.append(dict(instance_id=instance_id,polygon_xy=poly.tolist()))
            image.thumbnail((960,540));x=(i%2)*960;y=(i//2)*590
            control_sheet.paste(image,(x,y+45));d=ImageDraw.Draw(control_sheet)
            d.text((x+8,y+6),f'{name} | PTS {frame["time_seconds"]:.6f}s | physical sink regions',font=font,fill='white')
        controls.append(dict(frame=name,source_pts_seconds=frame['time_seconds'],image_sha256=frame['sha256'],regions=regions))
    control_sheet.save(out/'sink_identity_controls.jpg',quality=96)
    identity=dict(schema='real2sim-instance-identity-controls/v1',source_sha256=manifest['source_sha256'],
                  physical_sink_station_count=3,screen_content_is_not_physical_geometry=True,controls=controls,
                  interpretation='Island double basin and dishwasher-side double wall basin co-occur in frame000250; island double basin and separate single wall basin co-occur in frame001687. The two wall stations differ in basin subdivision, secondary tap, adjacent appliances, display count and cabinet retainers.',
                  caveat='There is no single selected frame showing all three physical basins together. Regions are manually reviewed image evidence; this does not verify 3D placement.',
                  artifact_sha256=sha(out/'sink_identity_controls.jpg'))
    (out/'identity_controls.json').write_text(json.dumps(identity,indent=2)+'\n')
    result=dict(schema='real2sim-semantic-inventory/v1',produced_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=manifest['source_sha256'],frames_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                implementation_sha256=sha(__file__),coordinate_system='original source 1920x1080 pixel corners, x right/y down',
                source_images_uploaded=False,training_only_evidence=True,scene_geometry_mutated=False,
                object_count=len(objects),observation_count=sum(len(o['observations']) for o in objects),
                scope='Observed instances, visible parts, and explicitly unresolved regions. Parent/child entries are not independent object counts.',
                exhaustive_instance_segmentation=False,metric_dimensions_verified=False,physical_parameters_measured=False,
                critical_distinctions=[['island_sink_basin_01','wall_sink_basin_01','wall_sink_basin_02'],['island_sink_station_01','sorting_island_01'],
                                       ['service_cart_near_bin_01','service_cart_outer_02']],
                unresolved=['Individual cabinet leaves outside the specifically annotated targets are grouped.',
                            'Complete hidden/back/bottom surfaces, articulation axes, limits and dynamics are not observed.',
                            'Similar chairs/tables outside canonical evidence are not automatically re-identified.',
                            'Glass may transmit or reflect background imagery; laser interaction is unmeasured.',
                            'Laboratory equipment beyond glass is an unresolved group, not individual collision bodies.'],
                sparse_model=model_input,objects=objects,annotated_frame_media=media)
    (out/'objects.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    cards=[]
    for obj in objects:
        observations=''.join(f'<figure><img loading="lazy" src="{o["review_crop"]}"><figcaption>{html.escape(o["note"])}<br>'
                             f'<details><summary>Source frame and hash</summary><code>{o["frame"]} · PTS {o["source_pts_seconds"]:.6f}s<br>'
                             f'SHA-256 {o["image_sha256"]}</code><img loading="lazy" src="{o["annotated_frame"]}"></details></figcaption></figure>' for o in obj['observations'])
        art=obj['articulation_hypothesis'];art_text='No articulation hypothesis assigned.' if not art else 'Inferred only: '+art['type']+'. '+art['note']
        support=obj.get('sparse_visible_support',{});support_text=support.get('status','not_requested')
        if support.get('retained_points'):support_text+=f' ({support["retained_points"]} features; incomplete extent, collision use disabled)'
        cards.append(f'<article data-search="{html.escape((obj["instance_id"]+" "+obj["category"]+" "+obj["label"]).lower())}">'
                     f'<h2>{obj["display_number"]:02d}. {html.escape(obj["label"])}</h2><code>{obj["instance_id"]}</code>'
                     f'<p>{html.escape(obj["uncertainty"])}</p><p>{html.escape(art_text)}</p><p>Feature support: {html.escape(support_text)}</p>'
                     f'<div class="observations">{observations}</div></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Source object inventory · IMG_2296.MOV</title><style>
body{margin:0;background:#10151d;color:#e8edf5;font:16px/1.45 system-ui}header,main{max-width:1400px;margin:auto;padding:24px}header{position:relative}
h1{font-size:30px}h2{font-size:20px}p{max-width:1100px}code{font-size:12px;overflow-wrap:anywhere}input{width:100%;padding:12px;border:1px solid #516078;background:#1b2330;color:white;font-size:16px;box-sizing:border-box}
article{padding:22px;background:#1b2330;margin-bottom:20px;border-radius:12px}article[hidden]{display:none}.observations{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;max-width:560px;flex:1 1 350px}img{width:100%;height:auto;display:block}figcaption{font-size:13px}details img{margin-top:10px}summary{cursor:pointer;color:#98cdfa}a{color:#98cdfa}.notice{border-left:4px solid #e6bd68;padding-left:14px}
</style><header><h1>Source-grounded object inventory</h1><p>Persistent instance IDs, visible parts, and exact video-frame evidence. THREE sink stations (one island, two wall stations) and two separate islands remain distinct. The orange outline marks an approximate annotation, not a pixel-accurate mask.</p>
<p class="notice">No metric dimensions, complete collision shapes, articulation axes, limits, or dynamics have been measured. Sparse feature envelopes describe visible evidence only. Parent/child entries and unresolved regions are included, so the entry count is not a count of independent physical objects.</p>
<p><a href="objects.json">Machine-readable inventory</a> · <a href="verification.json">Verification receipt</a> · <a href="identity_controls.json">Sink identity evidence</a></p><details><summary>Show three-sink identity control sheet</summary><img src="sink_identity_controls.jpg" alt="Source frame evidence separating the island and two wall basins"></details><input id="filter" placeholder="Filter by instance ID, label, or category" aria-label="Filter inventory"></header><main>'''+''.join(cards)+'''</main><script>
document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();for(const a of document.querySelectorAll('article'))a.hidden=!a.dataset.search.includes(q)});
</script></html>'''
    (out/'index.html').write_text(page)
    thumbs=[]
    for obj in objects:
        with Image.open(out/obj['observations'][0]['review_crop']) as crop:
            thumb=crop.resize((280,218));thumbs.append(thumb.copy())
    sheet=Image.new('RGB',(1120,((len(thumbs)+3)//4)*218),(15,18,24))
    for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%4)*280,(i//4)*218))
    sheet.save(out/'inventory_contact.jpg',quality=92)
    checks=dict(unique_instance_ids=True,parent_references_resolve=True,source_image_hashes_verified=True,
                all_annotations_within_source_extent=True,all_evidence_training_only=True,
                annotated_three_distinct_sink_basins=len([o for o in objects if o['category']=='sink_basin'])==3,
                annotated_three_distinct_island_stools=len([o for o in objects if o['category']=='bar_stool'])==3,
                annotated_two_distinct_service_carts=len([o for o in objects if o['category']=='service_cart'])==2,
                no_measured_articulation_claims=all(o['articulation_hypothesis'] is None or o['articulation_hypothesis']['evidence_status']=='inferred_from_appearance_not_measured' for o in objects))
    assert all(checks.values())
    receipt=dict(schema='real2sim-inventory-verification/v1',produced_at=datetime.now(timezone.utc).isoformat(),
                 implementation_sha256=sha(__file__),frames_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 objects_sha256=sha(out/'objects.json'),index_sha256=sha(out/'index.html'),contact_sha256=sha(out/'inventory_contact.jpg'),
                 identity_controls_sha256=sha(out/'identity_controls.json'),sink_control_image_sha256=sha(out/'sink_identity_controls.jpg'),
                 checks=checks,scope='Source provenance and annotation integrity; semantic interpretation still requires independent visual review.',
                 object_entries=len(objects),observations=result['observation_count'],
                 sparse_feature_envelopes=sum(o.get('sparse_visible_support',{}).get('status')=='visible_feature_envelope_only' for o in objects))
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'README.md').write_text(f'''# Source-grounded scene inventory

Open [index.html](index.html) for the self-contained review. [objects.json](objects.json) contains {len(objects)} persistent entries and {result['observation_count']} image observations, with exact source PTS, decoded frame index, image SHA-256, original-pixel region, visible-extent caveats, and parent/child references. Entries include parts and unresolved regions; this is not a count of independent physical bodies.

The corrected source evidence identifies **three physical sink stations**: the double island sink, the double wall sink by the dishwasher/glass-front appliance, and the single wall sink under two displays beside the toaster. [Identity controls](identity_controls.json) and the associated image distinguish actual basins from sink images playing on the displays. An earlier two-station draft was incomplete and is superseded by this inventory.

What worked: inspecting the full original frames, anchoring similar furniture to simultaneous observations, assigning separate parent/part IDs, and retaining exact image evidence. Two islands share recycling/compost labels; such labels must not establish object identity. Distinct sinks also repeat dispenser and faucet designs.

Main bottlenecks: blur, reflections, occluded bases/interiors, incomplete views of distant furniture, and repeated patterns. Articulation is a hypothesis inferred from visible doors/handles only. No masses, friction, joint pivots, limits, dimensions, or full collision shapes were measured. Glass transmission/reflection and LiDAR behavior remain unresolved.

The optional sparse support pass reads the frozen model without mutation, requires at least 15 shared feature points seen inside at least two annotated regions separated by 2 seconds, rejects tracks shorter than 3 and point errors above 2 pixels, and applies a radial median/MAD outlier gate. It produced {receipt['sparse_feature_envelopes']} accepted visible-feature envelopes. Insufficient support is left empty rather than converted into guessed object geometry. Even an accepted envelope is incomplete and cannot serve as a collision shape.

[verification.json](verification.json) binds the implementation, frames manifest, inventory, review page, contact sheet and identity controls. Its checks concern provenance and annotation integrity; independent visual review is still required for semantic judgments. No source images were uploaded, and no scene, mask, camera, or training dataset was modified.
''')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
