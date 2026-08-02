"""Serialize a DesignState to plain JSON (local meters, centered on the
intersection) so the headless Blender script can build a scene without needing
shapely/geopandas inside Blender's bundled Python.

This module only orchestrates: coordinate transforms live in src/render/coords.py,
crosswalk-to-leg matching in src/render/crosswalks.py, and street-furniture placement
in src/render/props.py."""
import json
import math
from pathlib import Path

from shapely.geometry import Point, Polygon

from src.render.coords import FT_TO_M, building_footprint_ft, pt_to_local_m, ring_to_local_m, wgs84_ring_to_local_m
from src.render.crosswalks import (CROSSWALK_CLEARANCE_FT, CROSSWALK_DEPTH_M, STOP_BAR_CURB_CLEARANCE_M,
                                   crosswalk_bands_ft, stop_bar_bands_ft,
                                   resolve_crosswalk_offsets, resolve_crosswalk_skews,
                                   resolve_stop_bar_offsets, stop_bar_width_ft)
from src.geometry.model import (
    build_pavement_polygon, corner_overlay_polygon, hatch_lines_ft, lane_narrowing_edge_lines_ft,
    lane_narrowing_polygons_ft, lane_narrowing_taper_ft, lane_narrowing_taper_polygons_ft, leg_clearance_ft,
    parking_lane_edge_line_ft, parking_stall_lines_ft,
)
from src.geometry.intersection import IntersectionModel, kerb_lines_with_tags_ft
from src.render.mesh_utils import build_decimated_building_mesh
from src.sources.osm_context import (fetch_buildings, fetch_crossings, fetch_kerbs,
                                     fetch_street_furniture, fetch_traffic_control)
from src.checks import assert_scene_valid
from src.render.props import build_props, control_nodes_ft, osm_tree_points_ft
from src.geometry.treatments import DEFAULT_CENTERLINE_STYLE, LEGAL_PARKING_SETBACK_FT, DesignState, build_sidewalk_pieces

BUILDING_CONTEXT_RADIUS_M = 130
KERB_RADIUS_M = 120
TRAFFIC_CONTROL_RADIUS_M = 60  # control nodes govern THIS junction; a wider net just pulls in neighbours
SIDEWALK_WIDTH_FT = 6
NEAR_ZONE_BUFFER_FT = 10  # how far past the farthest crosswalk the "near" (4k texture) pavement zone extends
PAINT_HATCH_SPACING_FT = 8.0  # spacing between rendered diagonal hatch lines - a rendering choice, not MUTCD-specified.
                               # At the original 2.5ft spacing, each stroke (which runs the buffer's full diagonal
                               # width, edge-to-edge, per hatch_lines_ft) touched the inner lane-edge line so
                               # frequently that the buffer read as one solid painted mass reaching the double
                               # yellow, drowning out the solid edge line and making the 11ft lane unreadable in
                               # the render even though the underlying geometry was already correct (verified via
                               # plan_view.py's top-down plot and by projecting each hatch line's real endpoints).
def _leg_heading_deg(leg) -> float:
    """Compass-agnostic heading (standard math degrees) of a leg's own
    centerline, outward from the intersection - used to angle lane-narrowing
    hatch lines consistently relative to the road itself (like a real gore/
    chevron marking), not a fixed world-space angle that would look diagonal
    on one leg and nearly parallel on another depending on the leg's bearing."""
    (x0, y0), (x1, y1) = leg.centerline.coords[0], leg.centerline.coords[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _split_near_far(polygons: list[Polygon], center_ft: Point, near_radius_ft: float):
    """
    Split a list of polygons (the pavement, the sidewalk pieces, ...) into a
    near-camera zone and everything else, by intersecting each with a circle
    around the intersection - used to texture what viewers will actually
    scrutinize (pavement/sidewalk right at the crosswalks) at a higher
    resolution than the rest. Any piece can become a MultiPolygon on either
    side of the split; always returns flat lists of simple Polygons.
    """
    circle = center_ft.buffer(near_radius_ft)
    near_polys, far_polys = [], []
    for poly in polygons:
        near = poly.intersection(circle)
        far = poly.difference(circle)
        near_polys += list(near.geoms) if near.geom_type == "MultiPolygon" else [near] if not near.is_empty else []
        far_polys += list(far.geoms) if far.geom_type == "MultiPolygon" else [far] if not far.is_empty else []
    return near_polys, far_polys


def export_scenario(model: IntersectionModel, state: DesignState, name: str, out_path: Path,
                     buildings: list[dict] | None = None, crossings: list[dict] | None = None,
                     theme: dict | None = None, traffic_control: list[dict] | None = None,
                     street_furniture: list[dict] | None = None) -> Path:
    center_ft = model.center_ft
    if theme is None:
        from src.render.theme import build_default_theme
        theme = build_default_theme()
    pavement = build_pavement_polygon(state.corner_fillets)
    sidewalk_pieces = build_sidewalk_pieces(state, sidewalk_width_ft=SIDEWALK_WIDTH_FT)
    if buildings is None:
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if crossings is None:
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if traffic_control is None:
        traffic_control = fetch_traffic_control(model.center_wgs84, radius_m=TRAFFIC_CONTROL_RADIUS_M)
    if street_furniture is None:
        street_furniture = fetch_street_furniture(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crosswalk_offsets = resolve_crosswalk_offsets(state, crossings)
    crosswalk_skews = resolve_crosswalk_skews(state, crossings)
    # Stop bars only make sense at a signalized intersection (this site's
    # config.yaml `signals` block is what "signalized" means - see
    # src/render/props.py's _traffic_signal_props/_no_turn_on_red_props, which gate
    # the same way).
    stop_bar_offsets = resolve_stop_bar_offsets(state, crosswalk_offsets) if model.config.get("signals") else {}

    # OSM building footprints are independent of (and coarser than) our SLD/field-measured
    # curb geometry - a few end up drawn overlapping the actual pavement. Drop those rather
    # than render buildings sitting in the middle of the road.
    buildings = [b for b in buildings if not building_footprint_ft(b["coords_wgs84"]).intersects(pavement)]

    near_radius_ft = max((v[0] for v in crosswalk_offsets.values()), default=30) + NEAR_ZONE_BUFFER_FT
    pavement_near, pavement_far = _split_near_far([pavement], center_ft, near_radius_ft)
    sidewalks_near, sidewalks_far = _split_near_far(sidewalk_pieces, center_ft, near_radius_ft)

    # Street trees come only from real OSM natural=tree nodes. They were previously
    # generated by walking each sidewalk piece at TREE_SPACING_FT, which invented 6-24
    # trees per site; nothing recorded says a tree is there, so nothing is drawn.
    tree_points_ft = osm_tree_points_ft(control_nodes_ft(street_furniture))

    props = build_props(model, state, crosswalk_offsets, center_ft, traffic_control, street_furniture,
                         crossings, fetch_kerbs(model.center_wgs84, radius_m=KERB_RADIUS_M))
    # Invariants, not warnings: a pad in the carriageway is a false claim about an
    # accessibility feature, and a curb drawn across the intersection is a false claim
    # about the street. Checked on the same shared band geometry the plan view checks, so
    # the two views can't diverge on what they consider valid. See src/checks.py.
    assert_scene_valid(
        model, state, props, pavement,
        crosswalk_bands=crosswalk_bands_ft(state, crosswalk_offsets, crosswalk_skews,
                                            CROSSWALK_DEPTH_M / FT_TO_M),
        stop_bars=stop_bar_bands_ft(state, stop_bar_offsets, crosswalk_skews),
        scenario=name)

    # Paint-only / no-curb-change proposal treatments (see src/geometry/treatments.py:
    # add_lane_narrowing / add_corner_hatching / add_mountable_apron) - all flush
    # with the existing pavement, never touching pavement_near/far or corner_parcels.
    # A lane-narrowing buffer is drawn as a solid edge line - the new lane's
    # real edge, explicitly delineated (not just implied by wherever the
    # diagonal hatching happens to start) so the lane width actually reads on
    # the render - plus diagonal hatching filling the buffer beyond it, like a
    # real gore/chevron marking, NOT a solid filled block of paint (which at
    # this render's scale is visually indistinguishable from a sidewalk or
    # apron). That edge line doesn't stop in a straight cut at the crosswalk/
    # stop-bar clearance line either - it tapers back out to meet the real
    # curb on the SAME leg, closer to the intersection (see
    # src/geometry/model.py:lane_narrowing_taper_ft), like a parking lane
    # curving back to the curb before an intersection - NOT a sweep around
    # the corner to the cross leg, which would inevitably cut through that
    # leg's own crosswalk (it sits right at the corner by definition). Per
    # the real source photo this treatment is modeled on, the diagonal
    # chevron paint keeps going in the same pattern all the way around the
    # taper to the curb - it doesn't stop dead where the straight run ends -
    # so the taper is hatched too (lane_narrowing_taper_polygons_ft), using
    # the same angle/spacing as the straight run's hatch so the two read as
    # one continuous stripe with no visible seam. It's still bounded by the
    # real curb at target_ft, safely clear of the crosswalk, so it can never
    # cross into the crosswalk, which has priority over this paint.
    lane_narrowing_edge_lines = []
    lane_narrowing_taper_lines = []
    lane_narrowing_hatch_lines = []
    if state.lane_narrowing:
        # Where each side's straight run should start/end (see
        # lane_narrowing_polygons_ft) - the same clearance/target points the
        # taper is anchored to, so the straight edge line, the hatch fill, and
        # the taper all meet exactly with no gap/overlap.
        start_ft = {}
        target_ft_by_leg = {}
        for leg_name, stripe_width_ft in state.lane_narrowing.items():
            sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
            anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
            target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
            start_ft[leg_name] = anchor_ft
            target_ft_by_leg[leg_name] = target_ft
            lane_narrowing_edge_lines += [
                ring_to_local_m(line.coords, center_ft)
                for line in lane_narrowing_edge_lines_ft(state.legs[leg_name], stripe_width_ft,
                                                          start_left_ft=anchor_ft, start_right_ft=anchor_ft,
                                                          sides=sides)
            ]
            lane_narrowing_taper_lines += [
                ring_to_local_m(taper.coords, center_ft)
                for taper in lane_narrowing_taper_ft(state.legs[leg_name], stripe_width_ft, anchor_ft, target_ft,
                                                      sides=sides)
            ]

        for leg_name, stripe_width_ft in state.lane_narrowing.items():
            if leg_name in state.lane_narrowing_line_only:
                continue  # just the edge/taper line above, no chevron fill - see add_lane_narrowing's line_only
            sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
            hatch_angle_deg = _leg_heading_deg(state.legs[leg_name]) + 45
            lane_narrowing_hatch_lines += [
                [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
                for poly in lane_narrowing_polygons_ft(
                    state.legs[leg_name], stripe_width_ft,
                    start_left_ft=start_ft[leg_name], start_right_ft=start_ft[leg_name], sides=sides)
                for line in hatch_lines_ft(poly, spacing_ft=PAINT_HATCH_SPACING_FT, angle_deg=hatch_angle_deg)
            ]
            lane_narrowing_hatch_lines += [
                [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
                for poly in lane_narrowing_taper_polygons_ft(
                    state.legs[leg_name], stripe_width_ft, start_ft[leg_name], target_ft_by_leg[leg_name],
                    sides=sides)
                for line in hatch_lines_ft(poly, spacing_ft=PAINT_HATCH_SPACING_FT, angle_deg=hatch_angle_deg)
            ]
    corner_hatching_lines = [
        [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
        for corner, depth_ft in state.corner_hatching.items()
        if "error" not in state.corner_fillets[corner]
        for line in hatch_lines_ft(corner_overlay_polygon(state.corner_fillets[corner], center_ft, depth_ft),
                                    spacing_ft=PAINT_HATCH_SPACING_FT)
    ]
    # Marked curbside parking (add_marked_parking): a real parking lane, not a paint-only buffer -
    # a lane-edge line depth_ft in from the curb plus perpendicular divider ticks at each stall
    # boundary. The MARKED STALLS themselves only start at parking_start_ft - max(the physical
    # past-the-corner-curve point, leg_clearance_ft, and the real legal minimum distance from the
    # actual crosswalk, LEGAL_PARKING_SETBACK_FT - NJSA 39:4-138) - whichever is farther from the
    # intersection, so parking is never marked somewhere a car couldn't legally stop even if the curb
    # geometry alone would allow painting it there. If curb_offset_ft > 0, the parking lane doesn't
    # start at the curb - there's a striped no-parking buffer between it and the curb (built with the
    # exact same lane_narrowing_* geometry a travel-lane buffer uses, `sides=` restricted to just this
    # one side - see add_marked_parking's curb_offset_ft docstring); THAT buffer still starts at the
    # ordinary anchor_ft and tapers into the corner like any other paint-only buffer here, since a
    # striped no-parking zone is still accurate (arguably more so) right up to the intersection -
    # optionally with bollards centered in it (add_parking_buffer_bollards).
    parking_edge_lines = []
    parking_stall_divider_lines = []
    parking_buffer_hatch_lines = []
    parking_buffer_edge_lines = []  # straight run only - see add_paint_line's docstring for why a curve
    parking_buffer_taper_lines = []  # (below) needs its own list, drawn with add_paint_polyline instead
    for (leg_name, side), zone in state.parking_zones.items():
        leg = state.legs[leg_name]
        depth_ft, stall_length_ft, curb_offset_ft = zone["depth_ft"], zone["stall_length_ft"], zone["curb_offset_ft"]
        anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        legal_start_ft = crosswalk_offsets[leg_name][0] + LEGAL_PARKING_SETBACK_FT
        parking_start_ft = max(anchor_ft, legal_start_ft)
        parking_edge = parking_lane_edge_line_ft(leg, side, depth_ft, parking_start_ft,
                                                  curb_offset_ft=curb_offset_ft)
        if parking_edge is None:
            continue  # corner return leaves no room on this leg - see the plan view's note
        parking_edge_lines.append(ring_to_local_m(parking_edge.coords, center_ft))
        parking_stall_divider_lines += [
            ring_to_local_m(line.coords, center_ft)
            for line in parking_stall_lines_ft(leg, side, depth_ft, stall_length_ft, parking_start_ft,
                                                curb_offset_ft=curb_offset_ft)
        ]
        if curb_offset_ft:
            target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
            buffer_angle_deg = _leg_heading_deg(leg) + 45
            parking_buffer_edge_lines.append(ring_to_local_m(
                lane_narrowing_edge_lines_ft(leg, curb_offset_ft, start_left_ft=anchor_ft, start_right_ft=anchor_ft,
                                              sides=(side,))[0].coords, center_ft))
            for poly in lane_narrowing_polygons_ft(leg, curb_offset_ft, start_left_ft=anchor_ft,
                                                    start_right_ft=anchor_ft, sides=(side,)):
                parking_buffer_hatch_lines += [
                    [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
                    for line in hatch_lines_ft(poly, spacing_ft=PAINT_HATCH_SPACING_FT, angle_deg=buffer_angle_deg)
                ]
            for poly in lane_narrowing_taper_polygons_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                parking_buffer_hatch_lines += [
                    [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
                    for line in hatch_lines_ft(poly, spacing_ft=PAINT_HATCH_SPACING_FT, angle_deg=buffer_angle_deg)
                ]
            for taper in lane_narrowing_taper_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                parking_buffer_taper_lines.append(ring_to_local_m(taper.coords, center_ft))
    corner_apron_polygons = [
        ring_to_local_m(corner_overlay_polygon(state.corner_fillets[corner], center_ft, extent_ft).exterior.coords,
                         center_ft)
        for corner, extent_ft in state.corner_aprons.items()
        if "error" not in state.corner_fillets[corner]
    ]

    building_entries = []
    for b in buildings:
        footprint_ft = building_footprint_ft(b["coords_wgs84"])
        mesh = build_decimated_building_mesh(footprint_ft, b["height_m"] / FT_TO_M)
        if mesh is not None:
            verts_ft, faces = mesh
            building_entries.append({
                "mesh": True,
                "vertices_m": [pt_to_local_m(x, y, center_ft)[:2] + [z * FT_TO_M] for x, y, z in verts_ft],
                "faces": faces,
            })
        else:
            building_entries.append({
                "mesh": False,
                "coords": wgs84_ring_to_local_m(b["coords_wgs84"], center_ft),
                "height_m": b["height_m"],
            })

    data = {
        "name": name,
        "units": "meters",
        "notes": state.notes,
        "theme": theme,
        # Shared with src/render/plan_view.py via src/render/crosswalks.py:CROSSWALK_DEPTH_M so the
        # 2D reconstruction and this 3D render draw the same crosswalk - Blender can't import from src/.
        "crosswalk_depth_m": CROSSWALK_DEPTH_M,
        # Likewise shared, so the 2D stop bar and the rendered one are the same bar.
        "stop_bar_curb_clearance_m": STOP_BAR_CURB_CLEARANCE_M,
        "existing_marked_crosswalks": model.config["intersection"].get("existing_marked_crosswalks", []),
        "pavement_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_near],
        "pavement_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_far],
        "sidewalks_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_near],
        "sidewalks_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_far],
        "tree_points": [pt_to_local_m(x, y, center_ft) for x, y in tree_points_ft],
        "lane_narrowing_edge_lines": lane_narrowing_edge_lines,
        "lane_narrowing_taper_lines": lane_narrowing_taper_lines,
        "lane_narrowing_hatch_lines": lane_narrowing_hatch_lines,
        "corner_hatching_lines": corner_hatching_lines,
        "parking_edge_lines": parking_edge_lines,
        "parking_stall_divider_lines": parking_stall_divider_lines,
        "parking_buffer_hatch_lines": parking_buffer_hatch_lines,
        "parking_buffer_edge_lines": parking_buffer_edge_lines,
        "parking_buffer_taper_lines": parking_buffer_taper_lines,
        "corner_apron_polygons": corner_apron_polygons,
        "props": [
            {
                **p,
                "position_m": pt_to_local_m(p["position_ft"][0], p["position_ft"][1], center_ft),
                **({"arm_length_m": p["arm_length_ft"] * FT_TO_M} if "arm_length_ft" in p else {}),
                # Pad dimensions travel with the prop so Blender draws the pad this
                # module positioned - the sidewalk offset is derived from the depth.
                **({"pad_depth_m": p["pad_depth_ft"] * FT_TO_M,
                    "pad_width_m": p["pad_width_ft"] * FT_TO_M} if "pad_depth_ft" in p else {}),
            }
            for p in props
        ],
        "legs": [
            {
                "name": leg_name,
                "near_m": [(leg.centerline.coords[0][0] - center_ft.x) * FT_TO_M,
                           (leg.centerline.coords[0][1] - center_ft.y) * FT_TO_M],
                "far_m": [(leg.centerline.coords[-1][0] - center_ft.x) * FT_TO_M,
                          (leg.centerline.coords[-1][1] - center_ft.y) * FT_TO_M],
                "width_m": leg.curb_to_curb_ft * FT_TO_M,
                "confirmed": model.config["legs"][leg_name].get("confirmed", False),
                "crosswalk_offset_m": crosswalk_offsets[leg_name][0] * FT_TO_M,
                "crosswalk_offset_source": crosswalk_offsets[leg_name][1],
                # How far the surveyed crossing is rotated off square to this leg
                # (src/render/crosswalks.py:_crossing_skew_deg). 0 for a leg with no
                # matched crossing - we don't invent an orientation we didn't survey.
                "crosswalk_skew_deg": crosswalk_skews.get(leg_name, 0.0),
                # A treatment (e.g. upgrade_crosswalk_markings) can override the style;
                # otherwise default to what OSM says exists today ("lines" if unmapped).
                "crosswalk_style": state.crosswalk_styles.get(leg_name, "lines"),
                # None (not drawn) unless this site's intersection is signalized (see stop_bar_offsets above).
                "stop_bar_offset_m": stop_bar_offsets[leg_name] * FT_TO_M if leg_name in stop_bar_offsets else None,
                # A stop bar only ever belongs across the real entering travel lane, not the full
                # curb-to-curb half (which can include a painted no-parking buffer or a marked-parking
                # lane next to the curb that a stopped vehicle would never actually occupy) - see
                # src/render/crosswalks.py:entering_lane_width_ft, shared with the 2D plan view,
                # i.e. unchanged behavior for any leg that hasn't been narrowed on its entering side.
                "stop_bar_width_m": stop_bar_width_ft(state, leg_name) * FT_TO_M,
                # Real per-leg fact from config.yaml (street-view confirmed), not an OSM tag - see
                # src/geometry/treatments.py:set_centerline_style / DEFAULT_CENTERLINE_STYLE.
                "centerline_style": state.centerline_styles.get(leg_name, DEFAULT_CENTERLINE_STYLE),
            }
            for leg_name, leg in state.legs.items()
        ],
        "refuge_islands": [
            {
                "name": island_name,
                "coords": ring_to_local_m(island["polygon"].exterior.coords, center_ft),
                "height_m": 0.15,
            }
            for island_name, island in state.refuge_islands.items()
        ],
        "raised_crossings": [
            {"name": leg_name, "coords": ring_to_local_m(poly.exterior.coords, center_ft), "height_m": 0.10}
            for leg_name, poly in state.raised_crossings.items()
        ],
        "corner_parcels": [
            {"name": str(row["quadrant"]), "coords": ring_to_local_m(row.geometry.exterior.coords, center_ft)}
            for _, row in model.corner_parcels.iterrows()
        ],
        "buildings": building_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path
