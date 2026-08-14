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
from shapely.ops import unary_union

from src.render.coords import FT_TO_M, building_footprint_ft, pt_to_local_m, ring_to_local_m, wgs84_ring_to_local_m
from src.render.crosswalks import (CROSSWALK_DEPTH_M, STOP_BAR_CURB_CLEARANCE_M,
                                   centerline_paint_ft, continental_bar_count, crosswalk_axes,
                                   centerline_start_ft,
                                   entering_lane_width_ft, resolve_crosswalk_style,
                                   stop_bar_band_geometry_ft, stop_bar_width_ft)
from src.geometry.model import hatch_lines_ft
from src.geometry.intersection import (IntersectionModel, drawn_kerb_radius_ft,
                                       kerb_lines_with_tags_ft)
from src.geometry.kerbs import KerbType
from src.geometry.markings import CHANNELS, KINDS, Role, kinds_in
from src.geometry.paint import RimCause, in_channel
from src.render.frame import junction_frame
from src.render.mesh_utils import build_decimated_building_mesh
from src.render.scene import SceneGeometry
from src.sources.assessor import (BuildingHeight, assessor_path, describe_building_heights,
                                   height_of, parcels_near_buildings, storeys_by_pin)
from src.sources.osm_context import (fetch_buildings, fetch_crossings, fetch_kerbs,
                                     fetch_street_furniture, fetch_traffic_control)
from src.render.props import build_props, control_nodes_ft, osm_tree_points_ft
from src.geometry.treatments import (DesignState, RaiseCrossing, RefugeIsland,
                                      build_sidewalk_pieces)

BUILDING_CONTEXT_RADIUS_M = 130
KERB_RADIUS_M = 120
TRAFFIC_CONTROL_RADIUS_M = 60  # control nodes govern THIS junction; a wider net just pulls in neighbours
SIDEWALK_WIDTH_FT = 6
NEAR_ZONE_BUFFER_FT = 10  # how far past the farthest crosswalk the "near" (4k texture) pavement zone extends
HATCH_ANGLE_DEG = 45.0  # for a corner treatment, which belongs to no single leg's heading
# How tall each kind of kerb is built, measured from z=0 like the pavement slab - so the REVEAL
# above the road is this minus the pavement's own 0.05 m. A raised kerb gets a 0.15 m reveal, the
# ordinary 6 in; a lowered one 0.02 m, which reads as a dropped kerb a car can cross rather than
# as no kerb at all. UNKNOWN is built at the lowered height on purpose: an untagged kerb must not
# render as a claim that a vehicle cannot cross it.
#
# NOTE the sidewalk band is extruded 0.03 m, i.e. BELOW the 0.05 m pavement, so a footway here
# currently sits lower than the road it borders. That predates this and is left alone rather than
# changed in passing - raising it is a visible change to every render and its own decision.
KERB_HEIGHT_M = {KerbType.RAISED: 0.20, KerbType.LOWERED: 0.07,
                 KerbType.FLUSH: 0.055, KerbType.UNKNOWN: 0.07}
PAINT_HATCH_SPACING_FT = 8.0  # spacing between rendered diagonal hatch lines - a rendering choice, not MUTCD-specified.
                               # At the original 2.5ft spacing, each stroke (which runs the buffer's full diagonal
                               # width, edge-to-edge, per hatch_lines_ft) touched the inner lane-edge line so
                               # frequently that the buffer read as one solid painted mass reaching the double
                               # yellow, drowning out the solid edge line and making the 11ft lane unreadable in
                               # the render even though the underlying geometry was already correct (verified via
                               # plan_view.py's top-down plot and by projecting each hatch line's real endpoints).
# Which paint kinds go into which JSON list blender_scene.py reads is no longer written here.
# It was, in a table beside the one every marking is declared in, and keeping the two in step
# was manual: renaming a kind in paint.py and forgetting this table dropped a whole treatment
# from the 3D render with nothing to say so, which happened when the parking buffer's taper
# became the daylight zone's and again when daylighting was added.
#
# Now each marking names its own channel (src/geometry/markings.py) and this is derived from
# that one declaration, so the two cannot disagree. Kept as a name because tests read it.
PAINT_KIND_LISTS = {channel.key: tuple(kind.name for kind in kinds_in(channel))
                    for channel in CHANNELS}
# Markings that reach the render some other way. An OBJECT is built from a prop - the render
# never turns a marking into an object - and check_bollards_are_props fails the build if one
# is painted with no prop behind it.
PAINT_KINDS_NOT_IN_LISTS = frozenset(kind.name for kind in KINDS.values() if kind.is_object)


def _marking_frame_m(prefix: str, leg, station_ft, center_ft) -> dict:
    """{prefix}_centre_m and {prefix}_axis for a marking at `station_ft` along `leg`.

    The crossing frame is defined once, in src/render/crosswalks.py:crosswalk_axes, and
    Blender cannot import it - so like crosswalk_reach_*_m and crosswalk_bar_count it has to
    travel as numbers. blender_scene.py was rebuilding it instead, from near_m and far_m,
    which is the leg's CHORD: identical on a straight centerline, and 4.54 deg out on
    broad_st_east, whose centerline kinks 4.5 deg where NJDOT rounds the corner 43.1 ft from
    the junction. That put the 3D bars somewhere the 2D bands were not, and the plan view
    then cleared paint from a footprint the render did not use.

    The axis is a unit vector and the export frame is a translate-and-scale of state-plane
    feet, so it needs no conversion - only the centre does. Returns {} for a marking this leg
    does not have, so the key is absent rather than null and the renderer's fallback is
    reached the same way it is for older geometry files.
    """
    if station_ft is None:
        return {}
    centre_ft, (ux, uy), _n, _cos = crosswalk_axes(leg, station_ft, 0.0)
    return {f"{prefix}_centre_m": pt_to_local_m(centre_ft[0], centre_ft[1], center_ft),
            f"{prefix}_axis": [ux, uy]}


def _stop_bar_span_m(state: DesignState, leg_name: str, has_bar: bool) -> dict:
    """The stop bar's resolved span and lateral offset, in metres, or {} for a leg with none.

    A dict rather than two values so an absent bar leaves the keys out entirely and
    blender_crosswalks.add_stop_bar falls back to its own arithmetic, the same way the
    crossing frame does.
    """
    if not has_bar:
        return {}
    span_ft, lateral_ft = stop_bar_band_geometry_ft(
        stop_bar_width_ft(state, leg_name), entering_lane_width_ft(state, leg_name) is None)
    return {"stop_bar_span_m": span_ft * FT_TO_M,
            "stop_bar_lateral_offset_m": lateral_ft * FT_TO_M}


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
    if buildings is None:
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if crossings is None:
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if traffic_control is None:
        traffic_control = fetch_traffic_control(model.center_wgs84, radius_m=TRAFFIC_CONTROL_RADIUS_M)
    if street_furniture is None:
        street_furniture = fetch_street_furniture(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)

    # Every marking position this scenario implies, resolved once (src/render/scene.py) and
    # shared with the plan view and the invariants. Crosswalks outrank every other marking,
    # so the paint below is cut around the bands geometrically rather than merely started far
    # enough out - a skewed crossing reaches further along one kerb than its centre offset
    # implies. Stop bars are resolved only at a signalized junction, the same gate
    # src/render/props.py's _traffic_signal_props/_no_turn_on_red_props use.
    scene = SceneGeometry.resolve(model, state, crossings)
    pavement = scene.pavement
    if pavement is None:
        # export_scenario has always required a closed ring (build_pavement_polygon raised
        # here before the scene resolved it), and everything below - the near/far texture
        # split, the building filter, the sidewalk band - is measured against it.
        raise ValueError("Can't export this scenario - the pavement ring did not close. "
                         "See src/geometry/model.py:build_pavement_polygon.")
    crosswalk_offsets = scene.crosswalk_offsets
    crosswalk_skews = scene.crosswalk_skews
    crosswalk_reaches = scene.crosswalk_reaches
    stop_bar_offsets = scene.stop_bar_offsets
    marked_crosswalks = scene.marked_crosswalks
    sidewalk_pieces = build_sidewalk_pieces(state, sidewalk_width_ft=SIDEWALK_WIDTH_FT)

    # OSM building footprints are independent of (and coarser than) our SLD/field-measured
    # curb geometry - a few end up drawn overlapping the actual pavement. Drop those rather
    # than render buildings sitting in the middle of the road.
    buildings = [b for b in buildings if not building_footprint_ft(b["coords_wgs84"]).intersects(pavement)]

    # Resolved once and read twice - written into the JSON for the camera, and used to decide
    # which traced kerbs are in the picture at all. Two calls would be two chances to disagree.
    frame = junction_frame(model)

    near_radius_ft = max((v[0] for v in crosswalk_offsets.values()), default=30) + NEAR_ZONE_BUFFER_FT
    pavement_near, pavement_far = _split_near_far([pavement], center_ft, near_radius_ft)
    sidewalks_near, sidewalks_far = _split_near_far(sidewalk_pieces, center_ft, near_radius_ft)

    # Street trees come only from real OSM natural=tree nodes. They were previously
    # generated by walking each sidewalk piece at TREE_SPACING_FT, which invented 6-24
    # trees per site; nothing recorded says a tree is there, so nothing is drawn.
    tree_points_ft = osm_tree_points_ft(control_nodes_ft(street_furniture))

    props = build_props(model, state, crosswalk_offsets, center_ft, traffic_control, street_furniture,
                         crossings, fetch_kerbs(model.center_wgs84, radius_m=KERB_RADIUS_M))
    paint, props = scene.build_paint_and_posts(props)
    # Invariants, not warnings: a pad in the carriageway is a false claim about an
    # accessibility feature, and a curb drawn across the intersection is a false claim
    # about the street. Checked on the same shared band geometry the plan view checks, so
    # the two views can't diverge on what they consider valid. See src/checks.py.
    scene.assert_valid(props, paint, scenario=name)

    # Paint-only / no-curb-change proposal treatments - lane-narrowing buffers, marked
    # parking, corner hatching, aprons. All of it is built by src/geometry/paint.py, which
    # the plan view also draws from and src/checks.py inspects, so the three cannot disagree
    # about where a marking goes. This function's job is only to sort the pieces into the
    # lists blender_paint.py expects and convert them to local meters.
    def _line(piece):
        return ring_to_local_m(piece.geometry.coords, center_ft)

    # Only the OPENING rims: the hatching runs into a crossing's diagonal, which is what gives a
    # zone its clean end there, and stops short of a driveway's fillet. See paint.RimCause.
    rims_by_side = {}
    for piece in paint:
        if piece.rim is RimCause.OPENING:
            rims_by_side.setdefault((piece.leg, piece.side), []).append(piece.geometry)

    def _hatch(piece):
        """A fill polygon becomes the diagonal strokes that actually get painted.

        Every piece is phased off the intersection centre, so the straight run, the taper,
        the daylight zone and the offcuts left by clipping around a crossing all land on one
        continuous family of lines. Phased off each polygon's own extent instead, the
        strokes stepped sideways at every seam and read as sheared.

        THE HATCHING KEEPS HALF A SPACING OFF A RIM, so the line that closes the zone reads as its
        edge and not as one more stroke. It has to, because the driveway fillet's chord is at the
        hatch angle by construction - its radius is the strip's depth, so the chord is at 45
        degrees, and the strokes are at 45 degrees. On Broad St's east kerb that put a 19 ft stroke
        2 ft from the sweep at the lane edge and 5 ft from it at the kerb: in the render, a fork.
        The tapered tip is then bounded by paint and left unhatched, which is what a striper does
        with a point too narrow to hatch. Same idea as PAINT_TO_CROSSWALK_GAP_FT one layer up.

        WHOLE STROKES ONLY. Taking the gap out of the polygon before hatching instead truncated the
        offending stroke rather than dropping it, and its far half survived as a 3 ft mark floating
        alone against the kerb - a stray, which is worse than the fork it replaced. A stroke either
        clears the sweep or it is not painted.
        """
        angle_deg = (_leg_heading_deg(state.legs[piece.leg]) + 45 if piece.leg
                      else HATCH_ANGLE_DEG)
        strokes = hatch_lines_ft(piece.geometry, spacing_ft=PAINT_HATCH_SPACING_FT,
                                  angle_deg=angle_deg,
                                  phase_origin=(center_ft.x, center_ft.y))
        rims = rims_by_side.get((piece.leg, piece.side))
        if rims:
            keep_off = unary_union(rims).buffer(PAINT_HATCH_SPACING_FT / 2)
            strokes = [line for line in strokes if not line.intersects(keep_off)]
        return [[pt_to_local_m(x, y, center_ft) for x, y in line.coords] for line in strokes]

    def _surface(piece):
        return ring_to_local_m(piece.geometry.exterior.coords, center_ft)

    # One serializer per role, so a new marking is drawn correctly in 3D the moment it is
    # declared. This used to be eleven hand-written assignments naming eleven JSON keys, and
    # picking the wrong helper for a key was silent: a sampled polyline routed through the
    # straight-chord builder deviated 0.7 ft on Broad St's daylight zone.
    BY_ROLE = {Role.LINE: lambda piece: [_line(piece)],
               Role.FILL: _hatch,
               Role.SURFACE: lambda piece: [_surface(piece)],
               # A coloured stretch of carriageway travels as its ring, like a surface - it has
               # no strokes to generate. The two roles serialize the same way and are still
               # different things: only a SURFACE is built ground the markings are cut around.
               Role.COLOUR: lambda piece: [_surface(piece)]}

    def channel_data(channel):
        serialize = BY_ROLE[channel.role]
        return [item for piece in in_channel(paint, channel) for item in serialize(piece)]

    paint_channels = {channel.key: channel_data(channel) for channel in CHANNELS}

    # HOW TALL EACH BUILDING IS, from whoever recorded it. OSM gives this project real outlines
    # and, here, no heights at all: 0 of 1150 building ways carry `height` and 7 carry
    # `building:levels`, so every building was extruded to one default and a borough of
    # storey-and-a-half houses rendered as identical boxes. The assessor counted the storeys -
    # see src/sources/assessor.py for the join and for what happens where nobody counted.
    parcels_for_heights = parcels_near_buildings(model)
    storeys = storeys_by_pin(assessor_path(model))
    building_entries = []
    heights = []
    for b in buildings:
        footprint_ft = building_footprint_ft(b["coords_wgs84"])
        recorded = (BuildingHeight(b["height_m"], b["height_source"])
                    if b.get("height_m") is not None else None)
        height = height_of(footprint_ft, parcels_for_heights, storeys, osm_height=recorded)
        heights.append(height)
        mesh = build_decimated_building_mesh(footprint_ft, height.height_m / FT_TO_M)
        if mesh is not None:
            verts_ft, faces = mesh
            building_entries.append({
                "mesh": True,
                "vertices_m": [[*pt_to_local_m(x, y, center_ft)[:2], z * FT_TO_M] for x, y, z in verts_ft],
                "faces": faces,
                "height_source": height.source,
            })
        else:
            building_entries.append({
                "mesh": False,
                "coords": wgs84_ring_to_local_m(b["coords_wgs84"], center_ft),
                "height_m": height.height_m,
                "height_source": height.source,
            })
    print(f"  NOTE: {describe_building_heights(heights)}.")

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
        # Where the camera points and how much it takes in, resolved by src/render/frame.py so
        # this render and the plan view frame the same ground. Blender used to compute an extent
        # of its own from the pavement below, which is how the two views came to disagree by up
        # to 1.57x on the same junction.
        "frame": frame.as_local_m(center_ft),
        "pavement_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_near],
        "pavement_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_far],
        "sidewalks_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_near],
        "sidewalks_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_far],
        "tree_points": [pt_to_local_m(x, y, center_ft) for x, y in tree_points_ft],
        # Every marking channel, in the order src/geometry/markings.py declares them. Splatted
        # rather than listed key by key: a channel Blender reads and this file forgot to write
        # is a treatment that vanishes between the two, and that is now impossible to type.
        **paint_channels,
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
                "crosswalk_offset_m": crosswalk_offsets[leg_name].offset_ft * FT_TO_M,
                "crosswalk_offset_source": crosswalk_offsets[leg_name][1],
                # How far the surveyed crossing is rotated off square to this leg
                # (src/render/crosswalks.py:_crossing_skew_deg). 0 for a leg with no
                # matched crossing - we don't invent an orientation we didn't survey.
                "crosswalk_skew_deg": crosswalk_skews.get(leg_name, 0.0),
                # How far the crossing actually runs to each kerb, left and right of the
                # centerline. A crosswalk goes kerb to kerb, and since the curb lines became
                # the surveyor's traced kerbs they are neither symmetric about NJDOT's
                # centerline nor at the nominal half-width - so `width_m` alone drew a
                # crossing that stopped short of the kerb on one side. See
                # src/render/crosswalks.py:crosswalk_reach_to_curbs_ft.
                "crosswalk_reach_left_m": crosswalk_reaches.get(leg_name, (None, None))[0] * FT_TO_M
                    if leg_name in crosswalk_reaches else None,
                "crosswalk_reach_right_m": crosswalk_reaches.get(leg_name, (None, None))[1] * FT_TO_M
                    if leg_name in crosswalk_reaches else None,
                # WHERE the crossing sits and WHICH WAY it faces, resolved here rather than
                # re-derived in Blender. blender_scene.py was taking the leg's axis as the
                # near->far CHORD and stepping the offset along it, which is a different
                # answer from crosswalk_axes' on any leg whose centerline bends: on
                # broad_st_east (3 vertices, 4.5 deg kink 43.1 ft out) it rotated the bars
                # 4.54 deg away from where the plan view puts them, and swung them into
                # 12.6 ft of paint the plan view had correctly cleared. Unskewed - the
                # renderer still applies crosswalk_skew_deg itself, because the span factor
                # that keeps a rotated crossing reaching both kerbs lives with it.
                **_marking_frame_m("crosswalk", leg, crosswalk_offsets[leg_name].offset_ft, center_ft),
                # Same for the stop bar, which is a second marking at a second station and
                # inherited the same chord.
                **_marking_frame_m("stop_bar", leg, stop_bar_offsets.get(leg_name), center_ft),
                # An UpgradeCrosswalkMarkings treatment if the design has one, else "lines" -
                # see src/render/crosswalks.py:resolve_crosswalk_style.
                "crosswalk_style": resolve_crosswalk_style(state, leg_name),
                # How many bars a continental/ladder crossing gets across that reach. Sized
                # in src/render/crosswalks.py:continental_bar_count so the arithmetic is
                # testable in one place; the renderer just lays out this many.
                "crosswalk_bar_count": continental_bar_count(sum(crosswalk_reaches[leg_name]))
                    if leg_name in crosswalk_reaches else None,
                # None (not drawn) unless this site's intersection is signalized (see stop_bar_offsets above).
                "stop_bar_offset_m": stop_bar_offsets[leg_name] * FT_TO_M if leg_name in stop_bar_offsets else None,
                # A stop bar only ever belongs across the real entering travel lane, not the full
                # curb-to-curb half (which can include a painted no-parking buffer or a marked-parking
                # lane next to the curb that a stopped vehicle would never actually occupy) - see
                # src/render/crosswalks.py:entering_lane_width_ft, shared with the 2D plan view,
                # i.e. unchanged behavior for any leg that hasn't been narrowed on its entering side.
                "stop_bar_width_m": stop_bar_width_ft(state, leg_name) * FT_TO_M,
                # ...and the resolved span and lateral offset that width produces, so
                # blender_crosswalks.add_stop_bar draws the bar this module measured rather
                # than repeating its arithmetic. The two copies had already diverged twice
                # over: on where the bar starts across the road, and on whether the skew's
                # span factor applies to the lateral offset as well as the span (Blender
                # applied it to both, the plan view to the span only). See
                # src/render/crosswalks.py:stop_bar_band_geometry_ft.
                **_stop_bar_span_m(state, leg_name, leg_name in stop_bar_offsets),
                # A SetCenterlineStyle treatment if the design has one, else the real per-leg
                # fact from config.yaml (street-view confirmed) or OSM's overtaking=no - see
                # src/geometry/treatments.py:DesignState.centerline_style.
                "centerline_style": state.centerline_style(leg_name),
                # Where the centerline paint starts. Resolved here, not in Blender, so the
                # rule ("stop at the stop bar") lives with the geometry and is testable -
                # see src/render/crosswalks.py:centerline_start_ft.
                "centerline_start_m": centerline_start_ft(
                    crosswalk_offsets[leg_name].offset_ft,
                    stop_bar_offsets.get(leg_name),
                    leg_name in marked_crosswalks) * FT_TO_M,
                # THE STRIPES THEMSELVES, following the leg's own centerline. Blender used to be
                # handed near_m and far_m and draw a straight stripe between them, which is the
                # CHORD - up to 3.98 ft off the real centerline on broad_st_east and 7.58 ft on
                # louellen_st_west, putting the double yellow where the stop bar it meets is not
                # and making the lanes either side of it read as different widths. The plan view
                # was drawing it correctly from the same DesignState the whole time; see
                # src/render/crosswalks.py:centerline_paint_ft, which both now call.
                "centerline_paint_m": [
                    ring_to_local_m(line.coords, center_ft)
                    for line in centerline_paint_ft(
                        leg,
                        centerline_start_ft(crosswalk_offsets[leg_name].offset_ft,
                                            stop_bar_offsets.get(leg_name),
                                            leg_name in marked_crosswalks),
                        state.centerline_style(leg_name))
                ],
            }
            for leg_name, leg in state.legs.items()
        ],
        # Both of these are DERIVED GEOMETRY rather than parameters: the treatment builds the
        # polygon, against this design, at the moment it is asked (RefugeIsland.polygon /
        # RaiseCrossing.polygon). They used to be materialised onto the state at apply time,
        # which made a raised crossing's footprint depend on where in a scenario it was applied -
        # its start station reads the corner fillets, and AddCurbExtension re-cuts them.
        "refuge_islands": [
            {
                "name": island.island_name,
                "coords": ring_to_local_m(island.polygon(state).exterior.coords, center_ft),
                "height_m": 0.15,
            }
            for island in state.treatments_of(RefugeIsland)
        ],
        "raised_crossings": [
            {"name": raised.target.leg,
             "coords": ring_to_local_m(raised.polygon(state).exterior.coords, center_ft),
             "height_m": 0.10}
            for raised in state.treatments_of(RaiseCrossing)
        ],
        # THE TRACED KERBS, with what OSM says each one is. There was no kerb in this render at
        # all: the pavement is one slab and the "kerb" was the material boundary where the
        # concrete band started, so a dropped kerb and a 6 in stood-up kerb looked identical and
        # the raised/lowered tagging on all 95 mapped ways reached nothing.
        #
        # The SAME set the plan view draws - kerb_lines_with_tags_ft at the FRAME radius, because
        # the two views disagreeing about which kerbs exist is the failure this project is built
        # around, and because what a drawing contains is a question about the drawing.
        #
        # This took the near set (within 80 ft of the centre) until it was noticed that the near
        # set is the corner-radius fit's test, not a renderer's: at Broad & Greenwood both kerbs
        # are traced the length of the corridor and all but the four returns were being dropped.
        "kerbs": [
            {"coords": ring_to_local_m(line.coords, center_ft),
             "kerb": str(KerbType.from_tags(tags)),
             "height_m": KERB_HEIGHT_M[KerbType.from_tags(tags)]}
            for line, tags, _way_id in kerb_lines_with_tags_ft(model.center_wgs84, center_ft,
                                                                radius_ft=drawn_kerb_radius_ft())
        ],
        # The driveways the kerb openings exist for. Drawn as a narrow strip of the same asphalt
        # rather than as a marking: it is a minor carriageway, and its job in the render is to
        # explain why the kerbside markings stop where they do.
        # The POLYGON, not a centreline plus a width for Blender to re-widen: the plan view draws
        # this same shape, so neither view can disagree about where a driveway is. `surveyed` says
        # whether the outline was traced (a parking lot is mapped as an area) or widened from a
        # line by this project (a driveway, an aisle) - see PavedSurface.extent_is_surveyed.
        "paved_surfaces": [
            {"kind": str(paved.kind),
             "surveyed": paved.extent_is_surveyed,
             "coords": ring_to_local_m(paved.surface.exterior.coords, center_ft)}
            for paved in model.paved_surfaces if paved.surface is not None
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
