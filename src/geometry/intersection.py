"""Assemble the full intersection model (legs, curb lines, corner fillets, parcels)
from a site's config.yaml + the data sources it points to. Shared by every phase
script, for every site - nothing in this module is specific to any one
intersection (see sites/README.md for what a site provides instead)."""
from dataclasses import dataclass, field
from pathlib import Path

from enum import StrEnum

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from src.sources.data_loader import load_parcels_near, load_road_network
from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_kerbs, fetch_roads
from src.geometry.model import (
    CURB_POINT_BEHIND_TOLERANCE_FT,
    CURB_POINT_MAX_WIDTH_RATIO,
    Leg,
    _line_direction,
    assign_curb_points_to_legs,
    assign_kerbs_to_corners,
    curb_line_from_points,
    curb_offsets_at_stations,
    through_street_sides,
    curb_station_span,
    build_corner_fillets,
    build_pavement_polygon,
    corner_radii_from_kerbs,
    buffer_point_wgs84,
    clip_to_radius,
    label_quadrants,
    nearest_per_quadrant,
    reproject_to_state_plane,
    split_leg_centerlines,
    station_offset_many,
)
from src.site import load_site_config

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # src/geometry/intersection.py -> repo root


@dataclass
class IntersectionModel:
    config: dict
    center_wgs84: Point
    center_ft: Point
    legs: dict[str, Leg]
    corner_fillets: dict
    parcels: gpd.GeoDataFrame
    corner_parcels: gpd.GeoDataFrame
    # {leg name: [RoadSpan]} - every OSM highway way lying along the leg, with the stretch of
    # it each one covers. A LIST because what a way says varies along a street and OSM
    # expresses that by splitting the way; see RoadSpan for the restriction that was being
    # dropped when this was one way per leg.
    leg_road_spans: dict = field(default_factory=dict)
    # {leg name: OSM tags of the way covering MOST of the leg}. For whole-leg facts only -
    # overtaking=no is what a double-yellow centerline means, and reading it beats defaulting
    # every leg to a dashed line. Anything that varies along the leg has to read
    # leg_road_spans instead, which is exactly the distinction kerbside parking needs.
    leg_osm_tags: dict = field(default_factory=dict)
    # {leg name: True if that way runs the same way the leg points outward}. OSM's left/right
    # are relative to the WAY's direction; a leg's are relative to its own outward direction.
    # Where they disagree the sides swap. Per span in leg_road_spans, since two ways covering
    # one leg can be drawn in opposite directions.
    leg_osm_aligned: dict = field(default_factory=dict)
    # [PavedSurface] - every mapped driveway, parking aisle and parking lot near this junction,
    # projected once. Paved ground that is not carriageway: drawn as asphalt in both views, and in
    # the driveways' case read as an opening signal too (src/geometry/kerbs.py). See PavedSurface
    # for why they live on the model rather than being fetched per consumer, and why all three are
    # one type.
    paved_surfaces: tuple = ()

    @property
    def driveways(self) -> tuple:
        """Just the driveways - what opens a kerb. A parking lot behind a building crosses none of
        ours, and an aisle inside one reaches the street through a driveway that is mapped
        separately, so neither may put a gap in a marking."""
        return tuple(s for s in self.paved_surfaces if s.kind == PavedKind.DRIVEWAY)

    def parking_restriction_spans(self, leg_name: str) -> list[tuple]:
        """[(start_ft, end_ft, {"left": value, "right": value}, way_id)] in the LEG's frame.

        Every way along this leg, its stretch, and what it says about each kerb - already
        flipped into the leg's own left/right where the way runs against it. The spans are
        contiguous where OSM split a way and may be absent where nothing is mapped; a value of
        None means that way says nothing about that side, which is NOT the same as "none".
        """
        return [(span.start_ft, span.end_ft,
                 parking_restriction_by_side(span.tags, span.aligned), span.way_id)
                for span in self.leg_road_spans.get(leg_name, [])]


def _bearing_deg(from_pt, to_pt) -> float:
    """Compass bearing (0=N, 90=E, clockwise) from from_pt to to_pt."""
    dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
    return (90 - np.degrees(np.arctan2(dy, dx))) % 360


def _bearing_diff(a: float, b: float) -> float:
    """Smallest angular difference between two compass bearings, in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


# How far a road network centerline may sit from the resolved intersection node before
# the snap below is worth reporting. Sub-foot gaps are digitizing noise; anything larger
# is a real disagreement between the two sources and worth seeing in the phase output.
SNAP_REPORT_THRESHOLD_FT = 2.0
ROAD_CONTEXT_RADIUS_M = 130
KERB_CONTEXT_RADIUS_M = 120  # fetch radius, metres - generous enough to catch a whole return
KERB_NEAR_JUNCTION_FT = 80   # but a return belonging to THIS junction is within this of centre
# How far outside a leg's plausible half-width band a traced vertex may sit and still count as
# that leg's kerb, for deciding whether a whole kerb WAY is relevant to this junction.
KERB_ALONG_LEG_TOLERANCE_FT = 8.0
KERB_PLAUSIBLE_HALF_WIDTH_FT = (8.0, 45.0)  # a kerb this far off a centerline is that leg's kerb


class OSMDataUnavailableError(RuntimeError):
    """Overpass could not be reached, so OSM-derived geometry can't be built.

    Distinct from "OSM has no data here", which is a legitimate finding this project
    reports and renders honestly. An unreachable server is not evidence of absence, and
    treating it as such silently downgrades every OSM-derived value to a placeholder.
    """


def _snap_distance_ft(line, center_ft: Point) -> float:
    """Perpendicular distance from the resolved intersection node to a road centerline."""
    return line.distance(center_ft)


def _snap_to_center(piece, center_ft: Point):
    """Translate a leg centerline piece so it starts exactly at the resolved intersection
    node, instead of at the nearest point on the road network's own line.

    WHY: the intersection LOCATION comes from OSM (a real shared junction node,
    cross-checked against the NJDOT SLD milepost - see data_loader.geocode_intersection),
    and so does every piece of context placed against it: the surveyed pedestrian
    crossings (src/render/crosswalks.py), buildings, and mapped footways. The leg
    CENTERLINES come from NJDOT's SRI linear-referencing layer. Those two frames don't
    agree, and on the state/county routes the disagreement is large and systematic:

        Route 518 at Greenwood Ave    8.4 ft        Greenwood Ave (local)   0.3 ft
        Route 518 at Princeton Ave    8.7 ft        Princeton Ave (CR 569)  1.4 ft
        CR 654 at Louellen St        16.3 ft        Columbia Ave (local)    1.8 ft

    ~8.5 ft on Route 518 at two independent junctions is a parallel offset in NJDOT's
    route alignment, not noise - an SRI line is a linear-referencing alignment, not a
    surveyed physical centerline, and some of these diagrams are 15 years old. Left
    uncorrected it shifts the whole modelled roadway sideways relative to the OSM
    crossings: at E Broad & Princeton it put 100% of the Princeton crosswalk inside the
    E Broad roadway. Snapping the centerline onto the node drops that to 22%, and the
    remainder is the leg's own (estimated) width.

    TRADE-OFF: this translates the entire piece, so its far end moves off NJDOT's
    alignment by the same amount (up to ~8 ft at 130 ft out). That's accepted
    deliberately - accuracy at the intersection, which is the whole subject of this
    project, beats accuracy at the far end of a leg that's only there for context. The
    offset is reported above whenever it exceeds SNAP_REPORT_THRESHOLD_FT so it's never
    silent. Bearings, lengths and widths are all unchanged; only position moves.
    """
    x0, y0 = piece.coords[0]
    return affinity.translate(piece, xoff=center_ft.x - x0, yoff=center_ft.y - y0)


def _assign_leg_pieces(pieces: list, leg_names: list[str], legs_cfg: dict, center_ft: Point) -> dict[str, object]:
    """
    Match centerline pieces (all sharing one SRI, split at the intersection) to
    the configured leg names that reference that SRI, by nearest compass bearing.
    Generalizes to any number of pieces per SRI (2 for a through road, 1 for a
    dead-end/stub leg) and any intersection shape - nothing here assumes a
    4-way or perpendicular roads, only that each leg's config entry has an
    accurate `bearing_deg`.
    """
    assigned = {}
    remaining_names = list(leg_names)
    for piece in pieces:
        far_bearing = _bearing_deg((center_ft.x, center_ft.y), piece.coords[-1])
        best_name = min(remaining_names, key=lambda n: _bearing_diff(far_bearing, legs_cfg[n]["bearing_deg"]))
        assigned[best_name] = piece
        remaining_names.remove(best_name)
    return assigned


class PavedKind(StrEnum):
    """What a piece of paved ground beside the carriageway IS.

    A StrEnum so it travels to the 3D render and into the exported JSON as the OSM value it came
    from, the same reason KerbType is one - a reader of the geometry file sees `parking_aisle`,
    not an integer they have to look up.
    """
    DRIVEWAY = "driveway"                # highway=service + service=driveway
    PARKING_AISLE = "parking_aisle"      # highway=service + service=parking_aisle
    PARKING_LOT = "parking_lot"          # amenity=parking, mapped as an AREA


# One radius for driveways, here rather than in each consumer. Matches the building/crossing
# context radius the renderers use, so a driveway drawn in a view is a driveway the openings were
# derived from - the divergence Driveway's docstring is about.
DRIVEWAY_CONTEXT_RADIUS_M = 130
# How wide each LINEAR kind is DRAWN. Assumptions, flagged as such wherever they surface - see
# PavedSurface.width_ft - because OSM maps these as centrelines and none of them carries a width
# here. A residential driveway; a two-way parking aisle at the low end of the 20-24 ft ITE range,
# and a one-way at 12, which is the one place a real tag (`oneway`) picks between them.
DRIVEWAY_DRAWN_WIDTH_FT = 10.0
PARKING_AISLE_WIDTH_FT = 20.0
PARKING_AISLE_ONEWAY_WIDTH_FT = 12.0
DRAWN_WIDTH_FT = {PavedKind.DRIVEWAY: DRIVEWAY_DRAWN_WIDTH_FT,
                  PavedKind.PARKING_AISLE: PARKING_AISLE_WIDTH_FT}


def _to_state_plane(coords) -> list:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
    return list(zip(xs, ys))


def _paved_surfaces_ft(center_wgs84: Point) -> tuple:
    """Every mapped driveway, parking aisle and parking lot near this junction, projected once.

    The lots are built first because they SUBTRACT from the aisles. An aisle inside a mapped lot
    is already paved by the lot's own surveyed outline, and drawing both leaves two coplanar
    surfaces at the same height - which in Blender is not redundancy, it is z-fighting (the
    project has hit that before; see MARKING_CLEARANCE_M). 6 of the borough's 20 aisles are inside
    a lot, so the other 14 still need their strips.
    """
    from src.sources.osm_context import (fetch_driveways, fetch_parking_aisles,
                                         fetch_parking_lots)

    lots = []
    for lot in fetch_parking_lots(center_wgs84, radius_m=DRIVEWAY_CONTEXT_RADIUS_M):
        coords = lot.get("coords_wgs84") or []
        if len(coords) < 4:
            continue
        surface = Polygon(_to_state_plane(coords)).buffer(0)
        if surface.geom_type == "Polygon" and not surface.is_empty:
            lots.append(PavedSurface(kind=PavedKind.PARKING_LOT, surface=surface,
                                     way_id=lot.get("id"), tags=lot.get("tags", {})))
    paved_by_lots = unary_union([lot.surface for lot in lots]) if lots else None

    out = list(lots)
    for kind, fetch in ((PavedKind.DRIVEWAY, fetch_driveways),
                        (PavedKind.PARKING_AISLE, fetch_parking_aisles)):
        for way in fetch(center_wgs84, radius_m=DRIVEWAY_CONTEXT_RADIUS_M):
            coords = way.get("coords_wgs84") or []
            if len(coords) < 2:
                continue
            tags = way.get("tags", {})
            line = LineString(_to_state_plane(coords))
            width_ft = (PARKING_AISLE_ONEWAY_WIDTH_FT
                        if kind == PavedKind.PARKING_AISLE and tags.get("oneway") == "yes"
                        else DRAWN_WIDTH_FT[kind])
            # Flat caps: a round cap would put a half-disc on the end of every driveway, out in
            # the garden it leads to.
            surface = line.buffer(width_ft / 2, cap_style=2)
            if kind == PavedKind.PARKING_AISLE and paved_by_lots is not None:
                surface = surface.difference(paved_by_lots)
            for piece in getattr(surface, "geoms", [surface]):
                if piece.geom_type == "Polygon" and not piece.is_empty:
                    out.append(PavedSurface(kind=kind, line=line, way_id=way.get("id"),
                                            tags=tags, surface=piece))
    return tuple(out)


def _runs_along_a_leg(line: LineString, legs: dict) -> bool:
    """Whether any vertex of `line` sits where one of these legs' kerbs would be.

    The test for "is this OUR kerb", as opposed to "is this near the middle of the junction".
    KERB_NEAR_JUNCTION_FT is the latter, and it is right for fitting a corner radius - a
    return belonging to this junction is within 80 ft of its centre, and at 120 m the fetch
    otherwise drags in neighbouring junctions' returns, which produced a nonsense 7.9-30.2 ft
    radius spread at Columbia & Princeton.

    It is wrong for building the CURB LINES, which want kerb anywhere along a 130 ft leg. On a
    130 ft leg a kerb at station 100 is 100 ft from the junction centre and was being thrown
    away: 14 traced ways across the four junctions, at stations 76-127 and plausible kerb
    offsets, including both sides of greenwood_ave_south from 87 ft out. Their absence was
    invisible because curb_line_from_points EXTRAPOLATES to the working length, so the outer
    half of those legs was drawn from a bearing instead of from the tracing that existed.
    """
    for leg in legs.values():
        if leg.curb_to_curb_ft is None:
            continue
        stations, offsets = station_offset_many(leg.centerline,
                                                np.asarray(line.coords, dtype=float))
        half_ft = leg.curb_to_curb_ft / 2
        along = ((stations > -CURB_POINT_BEHIND_TOLERANCE_FT)
                 & (stations < leg.centerline.length))
        beside = np.abs(offsets) < half_ft * CURB_POINT_MAX_WIDTH_RATIO + KERB_ALONG_LEG_TOLERANCE_FT
        if (along & beside).any():
            return True
    return False


def kerb_lines_with_tags_ft(center_wgs84: Point, center_ft: Point, legs: dict | None = None) -> list:
    """[(LineString, tags)] for traced kerbs near the junction - geometry plus what OSM
    says about each (kerb=lowered, tactile_paving=yes, wheelchair=yes).

    Two relevance tests, because "is this kerb ours" has two different answers. With `legs`
    omitted this returns the NEAR set: everything within KERB_NEAR_JUNCTION_FT of the junction
    CENTRE, which is the right test for fitting a corner RADIUS and for measuring a width -
    a return belonging to this junction is close to it, and at the 120 m fetch radius anything
    looser drags in the neighbouring junctions' returns. Pass `legs` and the test becomes
    _runs_along_a_leg instead: kerb anywhere along a leg, however far out, which is what a
    curb LINE wants. 14 traced ways across the four junctions sit at stations 76-127 with
    plausible kerb offsets and fail the near test - both sides of greenwood_ave_south from
    ~90 ft out among them.

    The wide set is deliberately NOT fed to the fit. Admitting those ways shifts
    w_broad_st_southwest's measured width, that reshuffles the vertex contest at the one
    junction with an acute Y and partial tracing, and louellen_st_west drops from two traced
    kerbs to one - more data in, less data used. So the fit runs on the near set and
    _extend_curbs_with_far_tracing rebuilds the curb lines from the wide set afterwards, once
    the widths are settled and the extra ways can only lengthen a curb, never redefine one.
    See tests/test_leg_frame.py.
    """
    return [(line, tags, way_id) for line, tags, way_id in _projected_kerbs(center_wgs84)
            if (_runs_along_a_leg(line, legs) if legs
                else line.distance(center_ft) <= KERB_NEAR_JUNCTION_FT)]


# Every traced kerb way at one junction, projected into state-plane feet once. The projection
# is a fact about the fetched ways, and the ways are read four times over per model load (the
# radius fit, the width fit, the far-tracing pass, the plan view) plus once per scenario. Held
# alongside the fetched list and served only while that is still the list fetch_kerbs returns,
# the same identity rule src/sources/osm_context.py:_LAYER_VIEWS uses - so a re-pulled snapshot
# cannot be served a projection of the old one.
_PROJECTED_KERBS: dict[tuple, tuple] = {}


def _projected_kerbs(center_wgs84: Point) -> list[tuple]:
    """[(LineString in feet, tags, way id)] for every traced kerb WAY near the junction.

    The id is carried because a marking this project BREAKS for a kerb has to be traceable to
    the kerb that broke it - see src/geometry/kerbs.py:KerbOpening.citation. It was dropped here
    before, and nothing read the tags either, so a caller wanting to know which way a line came
    from had to re-fetch and re-project to find out.

    Lone `barrier=kerb` NODES are dropped: they carry no arc to fit and no line to draw.
    """
    try:
        kerbs = fetch_kerbs(center_wgs84, radius_m=KERB_CONTEXT_RADIUS_M)
    except RuntimeError as e:
        # An outage must not look like "nothing is mapped here". Returning [] silently
        # would drop the traced kerbs, and with them the measured widths, the per-corner
        # radii and every tactile pad - producing a confident-looking render built on
        # absent data. Overpass mirrors are flaky enough that this happens in practice.
        raise OSMDataUnavailableError(
            f"could not fetch OSM kerbs for this junction ({e}). Refusing to build geometry: "
            "without them the widths, corner radii and tactile paving would silently fall back "
            "to placeholders, and the render would look finished while being wrong. Retry when "
            "Overpass is reachable."
        ) from e

    key = (round(center_wgs84.x, 7), round(center_wgs84.y, 7))
    cached = _PROJECTED_KERBS.get(key)
    if cached is not None and cached[0] is kerbs:
        return cached[1]
    projected = []
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords:
            continue
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        projected.append((LineString(zip(xs, ys)), kerb.get("tags", {}), kerb.get("id")))
    _PROJECTED_KERBS[key] = (kerbs, projected)
    return projected


def _kerb_lines_ft(center_wgs84: Point, center_ft: Point) -> list[LineString]:
    """Traced OSM kerb ways near this junction, in state-plane feet.

    Only kerbs within KERB_NEAR_JUNCTION_FT of the centre are kept. The fetch radius has
    to be generous enough to catch a whole corner return, but at 120 m it also pulls in
    kerbs belonging to NEIGHBOURING junctions - and those were being assigned to this
    junction's corners, producing a nonsense 7.9-30.2 ft spread at Columbia & Princeton.
    A corner return sits within a few tens of feet of the junction it belongs to.

    The near set of kerb_lines_with_tags_ft without the tags. It used to be a second copy of
    that function's fetch-and-project loop, which meant an Overpass outage reached this one
    first and came out as a bare RuntimeError rather than the OSMDataUnavailableError written
    to explain it.
    """
    return [line for line, *_ in kerb_lines_with_tags_ft(center_wgs84, center_ft)]


def _widths_from_traced_kerbs(legs: dict, kerb_lines: list, legs_cfg: dict) -> dict:
    """First-pass leg half-widths from traced OSM kerbs: nearest approach, doubled.

    A corner return is TANGENT to its leg's curb line, so the closest approach of a traced
    kerb to a leg centerline is that leg's half-width. (Away from the tangent point the
    return curves off around the corner, so the measurement is an upper bound - the real
    curb is at most this far out, never further.)

    That one-sidedness is what makes it safe to apply: a traced kerb closer to the
    centerline than our modelled curb PROVES the modelled road is too wide. It is only
    used to narrow a leg, never to widen one, and never against a field measurement -
    those win outright (src/provenance.py), though a conflict is reported.

    WHY THIS IS ONLY A FIRST PASS. Doubling one side's distance assumes the centerline sits
    midway between the two kerbs, and NJDOT's route alignment frequently does not - it is
    10.4 ft off centre on w_broad_st_northeast, where CR 518 turns onto Louellen and the
    alignment cuts the corner. Doubling the NEAR kerb there gives 30.3 ft for a street the
    two traced kerbs measure at 35.6. Every leg at every site came out too narrow this way,
    by 1-6 ft. _resize_and_centre_from_traced_kerbs below re-measures each leg properly
    once both kerbs are in its frame, and that measurement governs.

    What this pass is still needed for: assign_curb_points_to_legs disqualifies a traced
    vertex whose |offset| / half-width falls outside CURB_POINT_MIN/MAX_WIDTH_RATIO, so a
    badly wrong configured width (w_broad_st_southwest's 50 ft parcel-gap estimate) throws
    away the very kerbs the second pass needs. This gets the width close enough to collect
    them, and nothing else depends on its result.
    """
    from src.provenance import field_measurement_governs_corner

    updates = {}
    for name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        if field_measurement_governs_corner(legs_cfg.get(name, {})):
            continue        # a width measured at this cross-section outranks any trace
        candidates = []
        for kerb in kerb_lines:
            distance = kerb.distance(leg.centerline)
            nearest_along = leg.centerline.project(kerb.interpolate(kerb.project(leg.centerline.centroid)))
            if not (0 < nearest_along < leg.centerline.length):
                continue
            if KERB_PLAUSIBLE_HALF_WIDTH_FT[0] <= distance <= KERB_PLAUSIBLE_HALF_WIDTH_FT[1]:
                candidates.append(distance)
        if not candidates:
            continue

        measured_half_ft = min(candidates)
        if measured_half_ft >= leg.curb_to_curb_ft / 2 - 0.5:
            continue  # traced kerb agrees, or sits outside our curb - nothing proven
        updates[name] = measured_half_ft * 2
    return updates


# How far outside its assumed half-width the SEEDING assignment will still claim a traced
# vertex. Deliberately loose: that pass is collecting the kerbs the widths will be measured
# FROM, so it must not throw one away for disagreeing with a width nobody has measured yet.
# At W Broad & Louellen the two real kerbs sit at 0.43x and 3.5x the seed half-width, and
# the normal window (0.45-2.6) excluded one of them on every leg. Later rounds use the
# normal window against a width that by then is a measurement.
SEED_RATIO_BOUNDS = (0.2, 5.0)

# A traced kerb within this of the junction is a corner RETURN, flaring out to as much as
# 2.3x the half-width (CURB_POINT_MAX_WIDTH_RATIO). A cross-section measured across two
# returns is the width of the corner, not of the street, so the measurement below starts
# beyond them. Same distance UNTRACED_CORNER_THRESHOLD_FT uses for the same reason.
TRACED_SECTION_START_FT = 35.0
# ...and it ENDS here, which is not the same as ending where the leg does. The window used to
# run to the traced curb line's far end, and a curb line is drawn to the leg's working length -
# so lengthening a leg to show more of it silently re-measured its width. Carrying
# broad_st_east from 130 to 170 ft moved it 52.0 -> 49.9 ft, because East Broad narrows as it
# leaves the junction and the extra 40 ft of narrower street pulled the median down. A width
# is a fact about the approach; how far we chose to DRAW the approach is a presentation
# choice, and a presentation choice may not move a measurement. So the cross-section is always
# taken over the same stretch of road whatever working_length_ft says.
TRACED_SECTION_END_FT = 130.0
# Below this there isn't enough of a traced overlap to call it a cross-section.
MIN_TRACED_SECTION_FT = 20.0
TRACED_SECTION_SAMPLES = 40
# How much the kerbs' midpoint may wander along a leg before a single constant shift stops
# describing it. A parallel offset between NJDOT's alignment and the real carriageway is
# constant by definition; a midpoint that swings several feet means the alignment is
# BENDING relative to the street, and there is no one number that centres it.
MAX_CENTRE_SPREAD_FT = 5.0
# And a sanity bound on the shift itself. Past this the two "kerbs" are more likely to be
# one street's kerb and a neighbouring one's than the two sides of this leg.
MAX_CENTRE_SHIFT_FT = 15.0
# What counts as a real change between two rounds of the fit below, as opposed to the
# geometry jittering on the last decimal place and the loop never terminating.
MATERIAL_WIDTH_CHANGE_FT = 0.25
MATERIAL_SHIFT_FT = 0.1
MAX_FIT_ITERATIONS = 6


def _traced_cross_section(leg) -> tuple[np.ndarray, np.ndarray] | None:
    """(width, centre-offset) sampled along the run where BOTH of a leg's kerbs are traced.

    Both kerbs are read at the SAME centerline station, so the width there is left minus
    right and the centre is their midpoint. Neither quantity needs the alignment to be
    centred, or the street to be symmetrical, or the tracing to have started at the same
    place on the two sides - which is what makes this a measurement rather than a guess.
    Returns None unless both sides are traced: with one kerb there is no cross-section,
    only a distance to one edge.
    """
    if not {"left", "right"} <= leg.traced_sides:
        return None
    spans = [curb_station_span(leg, side) for side in ("left", "right")]
    if any(span is None for span in spans):
        return None
    lo = max(max(span[0] for span in spans), TRACED_SECTION_START_FT)
    hi = min(min(span[1] for span in spans), TRACED_SECTION_END_FT)
    if hi - lo < MIN_TRACED_SECTION_FT:
        return None
    stations = np.linspace(lo, hi, TRACED_SECTION_SAMPLES)
    left = curb_offsets_at_stations(leg, "left", stations)
    right = curb_offsets_at_stations(leg, "right", stations)
    if left is None or right is None:
        return None
    return left - right, (left + right) / 2


def _resize_from_one_traced_kerb(legs: dict, name: str, legs_cfg: dict, quiet: bool) -> bool:
    """Width for a leg with only ONE kerb traced: that kerb's distance out, doubled.

    This is a guess and is labelled as one, because there is no way to make it a
    measurement - the other edge of the street was never mapped. It assumes the alignment
    runs down the middle, which the legs that DO have both kerbs traced show is wrong by
    0.2-10.3 ft. No leg at any of the four junctions needs it today; all twelve sides are
    traced. It is here for the next site, and it says so loudly in the phase output rather
    than presenting a doubled half-width as though it were a cross-section.

    Better than the nearest-approach figure it replaces only in that it uses the MEDIAN
    offset along the street rather than the single closest vertex, which lands on the
    tightest point of a corner return and biases every such leg narrow.
    """
    from src.provenance import field_measurement_governs_corner

    leg = legs[name]
    if len(leg.traced_sides) != 1 or field_measurement_governs_corner(legs_cfg.get(name, {})):
        return False
    side = next(iter(leg.traced_sides))
    span = curb_station_span(leg, side)
    if span is None:
        return False
    lo, hi = max(span[0], TRACED_SECTION_START_FT), min(span[1], TRACED_SECTION_END_FT)
    if hi - lo < MIN_TRACED_SECTION_FT:
        lo, hi = span            # short trace: all of it, corner return and all
    offsets = curb_offsets_at_stations(leg, side, np.linspace(lo, hi, TRACED_SECTION_SAMPLES))
    if offsets is None:
        return False
    width_ft = 2 * float(np.median(np.abs(offsets)))
    if not quiet:
        print(f"  NOTE: {name} is {width_ft:.1f} ft wide, but only its {side} kerb is traced - this "
              f"ASSUMES the street is symmetrical about NJDOT's alignment, which on the legs with both "
              f"kerbs traced is wrong by up to 10 ft. Trace the "
              f"{'right' if side == 'left' else 'left'} kerb to replace the assumption with a "
              f"measurement.")
    if abs(width_ft - leg.curb_to_curb_ft) < MATERIAL_WIDTH_CHANGE_FT:
        return False
    legs[name] = Leg(name=name, centerline=leg.centerline, curb_to_curb_ft=width_ft)
    return True


def _resize_and_centre_from_traced_kerbs(legs: dict, legs_cfg: dict, quiet: bool = False) -> bool:
    """Take each leg's width and its working centerline from its two traced kerbs.

    This is the measurement _widths_from_traced_kerbs could only approximate, and it
    replaces two separate approximations that were both wrong in the same direction:

      WIDTH. Doubling the nearer kerb's distance understates any leg whose alignment is off
      centre, which is all of them. Measured properly: greenwood_ave_south 25.1 -> 31.2,
      columbia_ave_west 21.8 -> 26.4, w_broad_st_northeast 30.3 -> 35.6. The last one is why
      W Broad at Louellen rendered as a road too narrow to hold the lanes drawn on it, and
      columbia_ave_west is why that leg looked like it had no room for a shoulder.

      CENTRE. An SRI line is a linear-referencing alignment, not a surveyed carriageway
      centre (see _snap_to_center), and every width in a proposal is an offset from it. Off
      centre, the paint comes out symmetrical about the wrong line and the drawing looks
      wrong even where it measures right.

    The shift is a single constant per leg - a straight line parallel to the street, which
    is what a striper would lay, not a centreline that wanders to track every wobble in the
    tracing. Mutates `legs` in place (replacing the Leg, so its derived curb lines are
    rebuilt) and returns whether anything moved materially; the caller re-reads the traced
    kerbs in the new frame and comes back, until the two agree.
    """
    from src.provenance import (FIELD_MEASURED, OSM_DERIVED, field_measurement_governs_corner,
                                 leg_width_provenance)

    changed = False
    for name in sorted(legs):
        leg = legs[name]
        section = _traced_cross_section(leg)
        if section is None:
            if _resize_from_one_traced_kerb(legs, name, legs_cfg, quiet):
                changed = True
            continue
        widths, centres = section
        width_ft = float(np.median(widths))
        shift_ft = float(np.median(centres))
        spread_ft = float(centres.max() - centres.min())

        cfg = legs_cfg.get(name, {})
        keep_width = field_measurement_governs_corner(cfg)
        if not quiet:
            if keep_width and abs(width_ft - leg.curb_to_curb_ft) > 1.0:
                print(f"  CONFLICT: {name} is field-measured AT THE INTERSECTION at "
                      f"{leg.curb_to_curb_ft:.1f} ft, but its two traced kerbs are {width_ft:.1f} ft "
                      f"apart. The measurement stands - it was taken at this cross-section. Check "
                      f"the tracing, or whether the measurement spanned a shoulder beyond the kerb.")
            elif not keep_width:
                tier_note = ("Its field measurement is not recorded as taken AT the intersection "
                             "(width_measured_at), so the kerbs traced at this corner govern here. "
                             if leg_width_provenance(cfg) == FIELD_MEASURED else "")
                print(f"  NOTE: {name} is {width_ft:.1f} ft curb to curb, measured between its two "
                      f"traced OSM kerbs at the same station (osm_derived; config says "
                      f"{cfg.get('curb_to_curb_ft', float('nan')):.1f}). {tier_note}Cross-sections "
                      f"range {widths.min():.1f}-{widths.max():.1f} ft.")

        moved_ft = 0.0
        if abs(shift_ft) < MATERIAL_SHIFT_FT:
            pass
        elif spread_ft > MAX_CENTRE_SPREAD_FT or abs(shift_ft) > MAX_CENTRE_SHIFT_FT:
            if not quiet:
                print(f"  NOTE: {name}'s kerb midpoint is {shift_ft:+.1f} ft off the NJDOT alignment "
                      f"and wanders {spread_ft:.1f} ft along the leg - no single shift centres that, "
                      f"so the alignment is left as surveyed. Check whether this leg is a bend, or "
                      f"whether one kerb's tracing strays onto a neighbouring street.")
        else:
            moved_ft = shift_ft
            if not quiet:
                print(f"  NOTE: {name}'s centerline moved {shift_ft:+.1f} ft to sit midway between its "
                      f"two traced kerbs (the NJDOT alignment is a route reference, not a carriageway "
                      f"centre; midpoint holds to {spread_ft:.1f} ft along the leg).")

        if keep_width:
            width_ft = leg.curb_to_curb_ft
        if not moved_ft and abs(width_ft - leg.curb_to_curb_ft) < MATERIAL_WIDTH_CHANGE_FT:
            continue
        centerline = leg.centerline
        if moved_ft:
            (x0, y0), (x1, y1) = centerline.coords[0], centerline.coords[1]
            length = np.hypot(x1 - x0, y1 - y0)
            centerline = affinity.translate(centerline, -(y1 - y0) / length * moved_ft,
                                             (x1 - x0) / length * moved_ft)
        legs[name] = Leg(name=name, centerline=centerline, curb_to_curb_ft=width_ft,
                          width_provenance=None if keep_width else OSM_DERIVED)
        changed = True
    return changed


def _traced_side_count(legs: dict) -> int:
    """How many leg sides are currently drawn from a traced kerb rather than an offset.

    The fit's monotonicity measure: whatever else a round changes, it must never leave this
    lower than it found it. See _fit_legs_to_traced_kerbs.
    """
    return sum(len(leg.traced_sides) for leg in legs.values())


def _extend_curbs_with_far_tracing(legs: dict, center_wgs84: Point, center_ft: Point,
                                    near_coverage: dict | None = None) -> None:
    """Rebuild the curb lines once more, this time including kerb traced further out.

    KERB_NEAR_JUNCTION_FT keeps the fit's input to the ways around the junction, which is
    right for it: those are the ways a corner radius is fitted from, and at the 120 m fetch
    radius anything looser drags in neighbouring junctions. But a curb LINE wants kerb
    anywhere along a 130 ft leg, and 14 traced ways across the four junctions sit at stations
    76-127 with plausible kerb offsets and were being dropped for being >80 ft from the
    junction CENTRE - both sides of greenwood_ave_south from ~90 ft out among them. It never
    showed, because curb_line_from_points extrapolates to the working length: the outer half
    of those legs was drawn from a bearing while the tracing sat unused.

    Done AFTER the fit and with no re-measurement, which is the whole point. Feeding those
    ways to the fit itself shifts w_broad_st_southwest's measured width by half a foot, that
    reshuffles the vertex contest at the one junction with an acute Y and partial tracing, and
    louellen_st_west drops from two traced kerbs to one - more data in, less data used. With
    the widths already settled the extra ways can only lengthen a curb, never redefine one.

    Guarded anyway, on the same rule the fit uses: if the wider set somehow builds FEWER leg
    sides from tracing, the narrower result stands.
    """
    wide = kerb_lines_with_tags_ft(center_wgs84, center_ft, legs)
    before = _traced_side_count(legs)
    saved = {name: (leg.left_curb, leg.right_curb, set(leg.traced_sides))
             for name, leg in legs.items()}
    near_coverage = near_coverage or {}
    coverage = _apply_traced_curb_lines(legs, wide, center_ft, quiet=True)
    if _traced_side_count(legs) < before:
        for name, (left, right, traced) in saved.items():
            legs[name].left_curb, legs[name].right_curb = left, right
            legs[name].traced_sides = traced
        print(f"  NOTE: kerb traced beyond {KERB_NEAR_JUNCTION_FT:.0f} ft would have built "
              f"{before - _traced_side_count(legs)} fewer leg side(s) here - not used.")
        return

    # Correct the record. The fit reported each side's coverage from the NEAR set, because that
    # is all it was allowed to see, and those numbers are what a reader takes as "how much of
    # this curb is real". Left uncorrected they understate it - broad_st_east's left kerb reads
    # as traced to 76 ft when the drawing actually follows tracing to 173.7 - and a curb that
    # looks extrapolated but isn't is the same reporting failure as one that looks traced but
    # isn't, pointed the other way.
    grew = [(name, side, was, now_far)
            for (name, side), (_near, now_far) in sorted(coverage.items())
            for was in [near_coverage.get((name, side))]
            if was is not None and now_far > was[1] + 1.0]
    for name, side, was, now_far in grew:
        print(f"  NOTE: {name} {side} curb follows traced kerb out to {now_far:.0f} ft, not the "
              f"{was[1]:.0f} ft reported above - the rest is traced beyond the "
              f"{KERB_NEAR_JUNCTION_FT:.0f} ft junction radius the fit is restricted to.")


def _fit_legs_to_traced_kerbs(legs: dict, kerb_ways: list, center_ft: Point, legs_cfg: dict
                               ) -> dict[tuple[str, str], tuple[float, float]]:
    """Iterate assignment and measurement until they agree, then report the result.

    These two steps each need the other's answer. A traced vertex is assigned to the leg
    side whose half-width it best matches (assign_curb_points_to_legs), and the width is
    measured from the vertices assigned - so a bad starting width throws away the kerbs
    that would have corrected it, and keeps its own error. Both failure directions showed
    up at W Broad & Louellen: seeded too narrow, Louellen St's south kerb (155 ft of it, at
    a steady 34 ft offset) was 3.5x the assumed half-width and got discarded, leaving the
    leg "19 ft wide"; seeded from one shared width instead, W Broad's northeast leg lost
    its right kerb to the leg next door and came out at 56 ft.

    Iterating to a fixed point removes the dependence on where it starts. The first pass
    judges every leg against SEED_HALF_WIDTH_FT so no plausible kerb is excluded by a width
    nobody has measured yet; after that each leg is judged against its own current width,
    and the loop stops as soon as a round changes nothing material. It converges in 2-3
    rounds at all four junctions - and if some junction ever fails to settle, the cap ends
    it and the printed widths are still the ones actually used.

    Returns the {(leg, side): (near_ft, far_ft)} coverage it reported, so the far-tracing pass
    that runs next can correct those figures where it extends a curb past them.
    """
    started_at = {name: leg.centerline.coords[0] for name, leg in legs.items()}
    reported: dict[tuple[str, str], tuple[float, float]] = {}

    def apply_curbs(quiet=True, ratio_bounds=None):
        coverage = _apply_traced_curb_lines(legs, kerb_ways, center_ft, quiet=quiet,
                                             ratio_bounds=ratio_bounds)
        # Only the loud round, because `reported` exists to be corrected against what the
        # reader was actually shown. The quiet rounds print nothing to correct.
        if not quiet:
            reported.clear()
            reported.update(coverage)

    def snapshot():
        return {name: (leg.curb_to_curb_ft, leg.centerline) for name, leg in legs.items()}

    def restore(saved):
        for name, (width_ft, centerline) in saved.items():
            legs[name] = Leg(name=name, centerline=centerline, curb_to_curb_ft=width_ft)
        apply_curbs()       # a fresh Leg has no traced_sides until the kerbs are re-read

    apply_curbs(ratio_bounds=SEED_RATIO_BOUNDS)
    best, best_sides = snapshot(), _traced_side_count(legs)
    for iteration in range(MAX_FIT_ITERATIONS):
        # THE FIT MAY NEVER USE LESS GROUND TRUTH THAN IT ALREADY HAD. A width feeds the
        # window that decides which traced vertices the NEXT round may claim, so a round can
        # talk itself out of a kerb it was already using - and the loss compounds. At W Broad
        # & Louellen, admitting three more (correct) kerb ways made w_broad_st_southwest
        # measure slightly differently, louellen_st_west lost its north kerb in the reshuffle,
        # its width was then guessed by doubling the south kerb's 40 ft offset into an 80 ft
        # "street", and at 80 ft its own north kerb fell below CURB_POINT_MIN_WIDTH_RATIO and
        # could never be recovered. More data in, less data used, every step defensible.
        #
        # So a round is provisional until it proves it kept every traced side. That makes the
        # fit monotone in the one quantity that matters, which is what makes the runaway
        # impossible rather than merely unlikely.
        changed = _resize_and_centre_from_traced_kerbs(legs, legs_cfg, quiet=True)
        apply_curbs()
        sides = _traced_side_count(legs)
        if sides >= best_sides:      # >=, so among equally good rounds the most converged wins
            best, best_sides = snapshot(), sides
        if not changed:
            break
    # Not "stop at the first round that does not improve" - a round may drop a side and the
    # next recover two. Run the fit out and keep the best state it visited, which is monotone
    # in the outcome without being greedy about the path.
    if _traced_side_count(legs) < best_sides:
        lost = best_sides - _traced_side_count(legs)
        restore(best)
        print(f"  NOTE: the width fit's last round built {lost} fewer leg side(s) from traced "
              f"kerb than its best round did. Kept the better geometry.")

    # Once more out loud, on the geometry that survived - the notes above describe the
    # scaffold, and a note about a width superseded two rounds later is worse than none. The
    # reporting resize replaces Legs, so the kerbs are re-read after it or every leg ends up
    # claiming no traced sides at all.
    apply_curbs(quiet=False)
    _resize_and_centre_from_traced_kerbs(legs, legs_cfg)
    apply_curbs()

    # The per-round shifts were reported quietly and are individually meaningless; what a
    # reader needs is how far the working centerline ended up from the alignment NJDOT
    # published, because every dimension in the proposal is measured off it.
    for name, leg in sorted(legs.items()):
        (x0, y0), (x1, y1) = started_at[name], leg.centerline.coords[0]
        moved_ft = float(np.hypot(x1 - x0, y1 - y0))
        if moved_ft >= MATERIAL_SHIFT_FT:
            print(f"  NOTE: {name}'s working centerline sits {moved_ft:.1f} ft off NJDOT's alignment, "
                  f"midway between its two traced kerbs. The alignment is a linear-referencing "
                  f"reference, not a surveyed carriageway centre; the paint is measured from here.")
    return dict(reported)


# Past this distance out from the junction, a traced kerb says nothing about the corner.
UNTRACED_CORNER_THRESHOLD_FT = 35.0


# A leg is matched to the OSM way whose geometry it lies along: within this far of the
# leg's own centerline, and pointing the same way. Both are needed - the cross street
# passes just as close to the junction, and a parallel service road points the same way.
ROAD_MATCH_MAX_OFFSET_FT = 40.0
ROAD_MATCH_MAX_ANGLE_DEG = 30.0
# ...and it has to be a CARRIAGEWAY. Geometry alone is not enough to identify one: east of
# Princeton Ave, OSM has a `highway=service, service=parking_aisle` way (772378208) running
# 0.5 ft from East Broad Street's centerline at 0.2 deg to it - indistinguishable from the
# street on distance and bearing, and it won the nearest-way tie. So the leg's operational
# tags were read off a parking aisle, which carries none, and East Broad Street's own
# `parking:both:restriction=no_stopping` (way 1546878992) was never seen: the proposal
# hatched that kerb for having 7.5 ft spare and reported it as untagged, while the
# restriction sat in the data the whole time.
#
# A driveway, a parking aisle, a footway and a cycleway all fail this; every leg at all four
# sites is one of these classes. A leg that matches nothing keeps its defaults and says so,
# which is the safe direction - it invents no restriction it cannot source.
ROAD_MATCH_HIGHWAY_CLASSES = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
})


def _match_legs_to_osm_roads(legs: dict, center_wgs84: Point, center_ft: Point) -> dict:
    """{leg name: (tags, aligned)} for the OSM highway way each leg runs along.

    `aligned` is True when the way is drawn in the same direction the leg points outward.
    It decides whether OSM's left/right mean the leg's left/right or the reverse.

    Matched on geometry rather than on the street name in config.yaml: names disagree
    between sources ("W Broad St" vs "West Broad Street"), and a leg is a piece of a
    specific way, not of a name.
    """
    try:
        roads = fetch_roads(center_wgs84, radius_m=ROAD_CONTEXT_RADIUS_M)
    except Exception as e:  # noqa: BLE001 - operational tags are an enhancement, not a dependency
        print(f"  NOTE: couldn't read OSM road tags ({type(e).__name__}); centerline styles "
              f"fall back to the site config.")
        return {}

    # Projected once, outside the leg loop. Each candidate way was re-transformed for every
    # leg, so a 4-leg junction did the same coordinate transform four times per way.
    carriageways = [
        (road, LineString(zip(*wgs84_to_state_plane.transform(
            [c[0] for c in road["coords_wgs84"]], [c[1] for c in road["coords_wgs84"]]))))
        for road in roads
        if road["tags"].get("highway") in ROAD_MATCH_HIGHWAY_CLASSES
    ]

    out: dict[str, list[RoadSpan]] = {}
    for name, leg in legs.items():
        leg_dir = _line_direction(leg.centerline)
        spans = []
        for road, line in carriageways:
            along = np.dot(_line_direction(line), leg_dir)
            angle = np.degrees(np.arccos(np.clip(abs(along), -1, 1)))
            if angle > ROAD_MATCH_MAX_ANGLE_DEG:
                continue
            # The stretch of THIS leg the way actually covers, in the leg's own frame. A way is
            # matched on lying along the leg over that stretch, not on being nearest the leg's
            # midpoint - see RoadSpan for why the difference discarded a real restriction.
            stations, offsets = station_offset_many(leg.centerline,
                                                    np.asarray(line.coords, dtype=float))
            lo = max(float(stations.min()), 0.0)
            hi = min(float(stations.max()), leg.centerline.length)
            if hi - lo < MIN_ROAD_SPAN_FT:
                continue
            # Measured over the part that overlaps, so a way running alongside for miles is not
            # judged by how far away its far end wanders.
            covering = (stations >= -MIN_ROAD_SPAN_FT) & (stations <= leg.centerline.length
                                                          + MIN_ROAD_SPAN_FT)
            if not covering.any():
                continue
            if float(np.abs(offsets[covering]).min()) > ROAD_MATCH_MAX_OFFSET_FT:
                continue
            spans.append(RoadSpan(start_ft=lo, end_ft=hi, tags=road["tags"],
                                   aligned=bool(along >= 0), way_id=road.get("id")))
        if spans:
            out[name] = sorted(spans, key=lambda span: span.start_ft)
    return out


# Below this a way barely touches a leg - usually the cross street clipping the junction node -
# and its tags describe a different street.
MIN_ROAD_SPAN_FT = 5.0


@dataclass(frozen=True)
class PavedSurface:
    """One piece of paved ground beside the carriageway - a driveway, a parking aisle, a lot.

    PART OF THE MODELLED STREET, and it took a correction to put it here. A driveway was added as
    render dressing, fetched and projected independently by the plan view and by the export; then
    it became a signal for where the kerbside markings open and got a THIRD independent fetch in
    src/geometry/kerbs.py, each with its own radius constant. That is the exact shape of the bug
    src/render/scene.py exists to prevent - three consumers each assembling the same geometry, free
    to diverge - committed again one layer down.

    So it is resolved once, at load, beside corner_parcels and leg_road_spans: a surveyed fact
    about this junction's street network, not something each renderer looks up for itself. A
    driveway IS street geometry - it is where vehicles cross the kerb, and it is the reason a
    marking stops.

    ONE TYPE FOR ALL THREE, because they differ in exactly two ways and are otherwise the same
    thing: paved ground that is not carriageway, drawn as asphalt in both views. What differs is
    whether the extent was surveyed (a lot is mapped as an area; a driveway and an aisle are
    centrelines this project widens) and whether it opens the kerb (a driveway does, and
    src/geometry/kerbs.py reads only those - a lot behind a building crosses no kerb of ours).
    Adding parking as its own parallel field, with its own fetch, its own export key and its own
    branch in each renderer, is the same mistake the docstring above is about.
    """
    kind: str = PavedKind.DRIVEWAY
    #: The centreline, for the kinds OSM maps as a way. None for a lot, which is mapped as an area.
    line: LineString | None = None
    way_id: int | None = None
    tags: dict = field(default_factory=dict)
    #: The paved ground itself, built once so the plan view and the 3D render draw the SAME
    #: polygon rather than each widening the line their own way.
    surface: Polygon | None = None

    @property
    def extent_is_surveyed(self) -> bool:
        """Whether somebody traced this outline, or this project widened a line into it.

        A parking lot is mapped as an area and its extent is as surveyed as a building footprint.
        A driveway and an aisle are centrelines with no width on them - 0 of the borough's 43
        driveways and 0 of its 20 aisles carry a `width` tag - so their strips are as wide as
        DRAWN_WIDTH_FT says, which is an assumption and is labelled as one in the legend.
        """
        return self.kind == PavedKind.PARKING_LOT

    @property
    def width_ft(self) -> float | None:
        """How wide the strip is drawn, for the kinds that needed a width. ASSUMED.

        For a driveway, the number that is NOT this: the width of the OPENING it makes in the kerb,
        which IS surveyed (the extent of the `kerb=lowered` section) and is what the gap in the
        markings uses (src/geometry/kerbs.py). The two must not be swapped - at E Broad the dropped
        kerb runs 37 ft while the driveway centreline enters near one end of it, so that section is
        a frontage the driveway opens onto, not the driveway's own width.
        """
        return None if self.extent_is_surveyed else DRAWN_WIDTH_FT[self.kind]


@dataclass(frozen=True)
class RoadSpan:
    """One OSM highway way, and the stretch of one leg it covers.

    A LIST of these per leg, because the thing they carry varies along a street and OSM says so
    by SPLITTING THE WAY. That is how a kerbside parking restriction covering only the approach
    to a junction is expressed, and it is what this project was throwing away: the matcher kept
    the single way nearest the leg's MIDPOINT and dropped the rest, so at Broad & Greenwood a
    `parking:both:restriction=no_parking` tagged over East Broad's first 79.5 ft (way 1547092834)
    lost to the unrestricted way beyond it (11647647) by 1.9 ft against 5.8 - the leg's midpoint
    sits at station 85, past the split. The render then marked parking exactly where the mapper
    had just said there is none, and nothing anywhere reported a problem: a pipeline reading one
    way and finding no restriction looks identical to one that read the restriction and dropped it.

    `aligned` is per SPAN, not per leg: OSM's left/right are relative to the way's own direction,
    and two ways covering one leg can be drawn in opposite directions.
    """
    start_ft: float
    end_ft: float
    tags: dict
    aligned: bool
    way_id: int | None = None

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)

    def covers(self, station_ft: float) -> bool:
        return self.start_ft <= station_ft <= self.end_ft


def _apply_traced_curb_lines(legs: dict, kerb_ways: list, center_ft: Point,
                              quiet: bool = False,
                              ratio_bounds: tuple[float, float] | None = None
                              ) -> dict[tuple[str, str], tuple[float, float]]:
    """Replace a leg's derived curb lines with the surveyor's traced kerbs.

    Returns {(leg, side): (near_ft, far_ft)} - the station span of each side that a traced
    kerb actually covers, which is what the phase output reports as "how much of this curb is
    real" and what _extend_curbs_with_far_tracing corrects where it lengthens one.

    This is the last place NJDOT's alignment was still leaking into the geometry. Position
    was fixed earlier by snapping the centerline to the OSM junction node, but the BEARING
    stayed NJDOT's - and it measured 4-8 deg off the real street at these junctions. An
    offset curb inherits that error and splays ~10 ft away from the true kerb over a 100 ft
    leg, however accurate the width is. Measured on the traced runs: greenwood_ave_north's
    offset varies 11.9 ft along 105 ft while the curb itself bends only 1.3 ft.

    So where a side is traced, the traced points ARE the curb. Untraced sides keep the
    centerline offset. Mutates `legs` in place; curb_to_curb_ft is left as the reported
    width, which no longer drives that side's geometry.

    Every traced kerb way counts, whatever its `kerb` value. A corner return is tagged
    kerb=lowered because it's a ramp - that is a statement about its height, not about
    whether it is the edge of the roadway, and filtering to kerb=raised dropped whole
    traced corners (the SW corner of Broad & Greenwood) in favour of a fitted guess.

    How far to carry a curb comes from the leg's OWN centerline, not from a site-wide working
    length: legs may be different lengths (see load_intersection_model's leg_lengths), and a
    global would draw every curb to the longest leg's end.
    """
    lines = [line for line, *_ in kerb_ways]
    if not lines:
        return {}
    coverage: dict[tuple[str, str], tuple[float, float]] = {}
    assigned = assign_curb_points_to_legs(legs, lines, ratio_bounds)
    # Which kerbs have no corner return at their junction end, and so should be extended in
    # to the node rather than stopping where the tracing happens to stop.
    straight_through = through_street_sides(legs)
    for name, sides in assigned.items():
        leg = legs[name]
        for side, points in sides.items():
            curb = curb_line_from_points(points, leg, leg.centerline.length,
                                          extend_to_junction=(name, side) in straight_through)
            if curb is None:
                continue
            setattr(leg, f"{side}_curb", curb)
            leg.traced_sides.add(side)
            near, far = min(p[0] for p in points), max(p[0] for p in points)
            coverage[(name, side)] = (near, far)
            if not quiet:
                print(f"  NOTE: {name} {side} curb is the traced OSM kerb itself, {len(points)} points "
                      f"covering {near:.0f}-{far:.0f} ft out from the junction.")

    # A side with nothing traced near the junction leaves its corner to be bridged across a
    # gap (or, with nothing traced at all, fitted from a radius). Both are weaker than a
    # traced corner, and a big enough gap is what stops the pavement ring closing - so name
    # the sides. That's the difference between "this junction isn't representable" and
    # "trace these two and it will be".
    gaps = []
    for name, leg in legs.items():
        for side in ("left", "right"):
            points = assigned.get(name, {}).get(side, [])
            if not points:
                gaps.append(f"{name} {side} (nothing traced)")
            elif min(p[0] for p in points) > UNTRACED_CORNER_THRESHOLD_FT:
                gaps.append(f"{name} {side} (traced only from {min(p[0] for p in points):.0f} ft out)")
    if gaps and not quiet:
        print(f"  NOTE: no traced kerb within {UNTRACED_CORNER_THRESHOLD_FT:.0f} ft of the junction on: "
              f"{'; '.join(gaps)}. Those corners are bridged, not traced - tracing the kerb up to "
              f"the corner return would fix them.")
    return coverage


def _build_corners(legs: dict, radius_ft: float, corner_radii: dict, kerb_lines: list) -> dict:
    """Corner geometry, preferring the surveyor's traced kerb over a fitted fillet.

    A traced kerb IS the curb, so it is used as the corner directly and the leg curb lines
    are trimmed to meet its ends. Fitting a circle to it and redrawing that circle off our
    own estimated curb lines kept the curvature but lost the position - the synthesised
    arcs sat 0.2-5.9 ft from the mapped kerb at Broad & Greenwood.

    Falls back to the fitted fillet for the WHOLE junction if the traced corners can't
    close into a valid pavement ring. A traced kerb can be too short, or stop before the
    tangent point, and half a ring built from real geometry is worse than a consistent
    approximation - so the fallback is all-or-nothing and says so.
    """
    corner_arcs = assign_kerbs_to_corners(legs, kerb_lines)
    fillets = build_corner_fillets(legs, radius_ft, corner_radii, corner_arcs)
    traced = [k for k, v in fillets.items() if v.get("source") == "traced_kerb"]
    if not traced:
        return fillets
    try:
        build_pavement_polygon(fillets)
    except ValueError as e:
        print(f"  NOTE: traced kerbs don't close into a valid pavement ring here ({e}). Falling back to "
              f"fitted fillets for this junction; the corners will sit a few feet off the mapped kerb.")
        return build_corner_fillets(legs, radius_ft, corner_radii)
    names = ", ".join("/".join(sorted(c)) for c in traced)
    print(f"  NOTE: corner geometry taken directly from the traced OSM kerb at: {names}.")
    return fillets


def _corner_radii_from_osm(center_wgs84: Point, center_ft: Point, legs: dict, fallback_ft: float) -> dict:
    """Per-corner radii from traced kerbs, reporting what was and wasn't usable."""
    radii, notes = corner_radii_from_kerbs(legs, _kerb_lines_ft(center_wgs84, center_ft), fallback_ft)
    for note in notes:
        print(f"  NOTE: {note}")
    return radii


def load_intersection_model(config: dict | None = None, site: str | None = None) -> IntersectionModel:
    """Pass either a pre-loaded `config` dict, or a `site` name to load it fresh
    (defaults to src.site.DEFAULT_SITE if neither is given)."""
    if config is None:
        from src.site import DEFAULT_SITE
        config = load_site_config(site or DEFAULT_SITE)

    lon, lat = config["intersection"]["center_wgs84"]
    center = Point(lon, lat)
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs("EPSG:3424").iloc[0]

    data_sources = config.get("data_sources", {})
    road_network_path = ROOT_DIR / data_sources["road_network"]
    parcels_path = ROOT_DIR / data_sources["parcels"]

    clip_radius_m = config["intersection"]["clip_radius_m"]
    bbox = buffer_point_wgs84(center, clip_radius_m * 1.3)
    network = load_road_network(bbox=bbox, path=road_network_path)
    clipped = clip_to_radius(network, center, clip_radius_m)
    clipped_ft = reproject_to_state_plane(clipped)

    working_len = config["intersection"]["leg_working_length_ft"]
    legs_cfg = config["legs"]
    # How far to carry each leg, PER LEG. The site-wide value is the default; a leg may
    # override it with legs.<name>.working_length_ft. Legs are not interchangeable in how far
    # they can honestly be drawn - a curb is traced as far as somebody traced it, and past
    # that curb_line_from_points extrapolates from a bearing. So the length that shows the
    # most ground truth on an arterial is longer than the one that starts inventing kerb on
    # the cross street. One global forces the shorter answer on both.
    leg_lengths = {name: float(cfg.get("working_length_ft", working_len))
                   for name, cfg in legs_cfg.items()}
    sri_to_leg_names: dict[str, list[str]] = {}
    for name, leg_cfg in legs_cfg.items():
        sri_to_leg_names.setdefault(leg_cfg["sri"], []).append(name)

    legs: dict[str, Leg] = {}
    for sri, leg_names in sri_to_leg_names.items():
        rows = clipped_ft[clipped_ft["SRI"] == sri]
        if rows.empty:
            print(f"  WARNING: SRI {sri} not found in clipped network - skipping legs {leg_names}.")
            continue
        line = rows.iloc[0].geometry
        # NJDOT digitization sometimes has sub-foot vertex noise near
        # intersections; that noise creates enough local curvature to make
        # offset_curve() self-intersect (returns a MultiLineString) at typical
        # curb-to-curb widths. A few feet of simplification removes it without
        # affecting real road geometry at this scale.
        # Split to the LONGEST length wanted on this SRI - which leg a piece is cannot be
        # known until _assign_leg_pieces has matched it by bearing, so cutting to a specific
        # leg's length here would truncate the wrong one. Each piece is trimmed to its own
        # leg's length below, after it has a name.
        split_len = max(leg_lengths[n] for n in leg_names)
        pieces = [p.simplify(3.0) for p in split_leg_centerlines(line, center_ft, split_len)]
        pieces = [_snap_to_center(p, center_ft) for p in pieces]
        snap_ft = _snap_distance_ft(line, center_ft)
        if snap_ft > SNAP_REPORT_THRESHOLD_FT:
            print(f"  NOTE: SRI {sri} centerline passes {snap_ft:.1f} ft from the resolved intersection "
                  f"node - snapped onto it (legs {', '.join(leg_names)}).")
        for name, piece in _assign_leg_pieces(pieces, leg_names, legs_cfg, center_ft).items():
            # Trimmed from the snapped junction end outward, so a shortened leg keeps the
            # station-0 origin every measurement in the project is taken from.
            if piece.length > leg_lengths[name]:
                piece = substring(piece, 0, leg_lengths[name])
            legs[name] = Leg(name=name, centerline=piece, curb_to_curb_ft=legs_cfg[name].get("curb_to_curb_ft"))

    kerb_lines = _kerb_lines_ft(center, center_ft)
    for name, width_ft in _widths_from_traced_kerbs(legs, kerb_lines, legs_cfg).items():
        legs[name] = Leg(name=name, centerline=legs[name].centerline, curb_to_curb_ft=width_ft)

    # The NEAR set for the fit: the ways around the junction, which is what a width and a
    # corner radius are measured from. _extend_curbs_with_far_tracing adds the rest afterwards,
    # once the widths can no longer be moved by them.
    kerb_ways = kerb_lines_with_tags_ft(center, center_ft)
    # Twice, deliberately. The first pass only has to be good enough to collect each leg's
    # traced vertices; that gives _resize_and_centre_from_traced_kerbs a real cross-section
    # to measure, and the corrected width and centre then change which vertices belong to
    # which leg side, so the assignment is redone in the corrected frame. Silent the first
    # time round - the coverage it reports is about the final geometry, not the scaffold.
    near_coverage = _fit_legs_to_traced_kerbs(legs, kerb_ways, center_ft, legs_cfg)
    _extend_curbs_with_far_tracing(legs, center, center_ft, near_coverage)
    leg_road_spans = _match_legs_to_osm_roads(legs, center, center_ft)
    # The way covering MOST of the leg carries its whole-leg tags. Not the way nearest the
    # leg's midpoint, which is what this used to pick: on a split leg those differ, and the
    # nearest-to-midpoint rule has no claim to describing the leg as a whole.
    dominant = {name: max(spans, key=lambda span: span.length_ft)
                for name, spans in leg_road_spans.items()}
    leg_osm_tags = {name: span.tags for name, span in dominant.items()}
    leg_osm_aligned = {name: span.aligned for name, span in dominant.items()}
    for name, spans in sorted(leg_road_spans.items()):
        if len(spans) < 2:
            continue
        # A split leg is worth saying out loud: it is how OSM records a fact that changes part
        # way along a street, and reading only one of the ways is how such a fact disappears.
        detail = "; ".join(f"way {s.way_id} over {s.start_ft:.0f}-{s.end_ft:.0f} ft" for s in spans)
        print(f"  NOTE: {name} is covered by {len(spans)} OSM ways ({detail}). Kerbside parking "
              f"is read per span; whole-leg tags come from the longest.")

    radius_ft = config["treatments"]["existing_corner_radius_ft"]
    # Traced OSM kerbs give a real measured radius per corner where they exist; the
    # config value stays as the fallback for untraced corners. Nothing in OSM carries a
    # corner radius as a tag, so a traced kerb line is the only way to source this.
    corner_radii = {}
    corner_fillets = {}
    if radius_ft:
        corner_radii = _corner_radii_from_osm(center, center_ft, legs, radius_ft)
        corner_fillets = _build_corners(legs, radius_ft, corner_radii, kerb_lines)

    parcels = load_parcels_near(center, radius_ft=300, path=parcels_path)
    corner_parcels = nearest_per_quadrant(label_quadrants(parcels, center_ft))

    return IntersectionModel(
        config=config,
        center_wgs84=center,
        center_ft=center_ft,
        legs=legs,
        corner_fillets=corner_fillets,
        leg_road_spans=leg_road_spans,
        leg_osm_tags=leg_osm_tags,
        leg_osm_aligned=leg_osm_aligned,
        parcels=parcels,
        corner_parcels=corner_parcels,
        paved_surfaces=_paved_surfaces_ft(center),
    )


# OSM records kerbside parking per side of the way: parking:left:restriction,
# parking:right:restriction, or parking:both:restriction. Any value other than "none" is a
# prohibition of some kind (no_parking, no_standing, no_stopping); "none" is an explicit
# statement that parking IS allowed, which is different from the tag being absent.
PARKING_RESTRICTION_KEYS = {"left": "parking:left:restriction",
                            "right": "parking:right:restriction",
                            "both": "parking:both:restriction"}


def parking_restriction_by_side(tags: dict, aligned: bool) -> dict:
    """{"left": value|None, "right": value|None} in the LEG's frame.

    OSM's left and right are relative to the direction the way was drawn; a leg's are
    relative to the direction it points outward from the junction. Half this project's legs
    run against their way - Columbia Ave's west leg, Princeton Ave's north leg - so reading
    parking:left straight through would put the restriction on the wrong kerb for them, and
    it would look entirely plausible in the render.

    A value of None means OSM says nothing about that side. That is NOT the same as "none",
    which is a positive statement that parking is permitted.
    """
    both = tags.get(PARKING_RESTRICTION_KEYS["both"])
    if both is not None:
        return {"left": both, "right": both}
    osm = {side: tags.get(key) for side, key in PARKING_RESTRICTION_KEYS.items() if side != "both"}
    if aligned:
        return {"left": osm["left"], "right": osm["right"]}
    return {"left": osm["right"], "right": osm["left"]}


def parking_is_restricted(restriction: str | None) -> bool:
    """True where OSM prohibits kerbside parking. Absent or "none" is not a prohibition."""
    return restriction is not None and restriction != "none"
