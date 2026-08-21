"""STATION-RANGED FACTS on a corridor: what is actually there, stationed once for the whole street.

STEP 3 OF docs/network-model.md. Every question here is one that has no answer in the per-leg model
because the answer straddles a junction or falls in the gap between two of them: where the
driveways are, where the cross streets land, which stretches of kerb may hold parking, how much
marked parking a corridor could carry.

WHY THESE ARE READ OFF THE CORRIDOR AND NOT OFF OSM DIRECTLY. A raw OSM feature has a location but
no station, and a station is what makes two facts comparable - a driveway mouth and a no-parking
sign are the same kind of object here only once both are (start_ft, end_ft) on the same corridor.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from src.geometry.model import (frame_at, line_direction, station_offset_many)
from src.geometry.network.corridor import (Corridor)
from src.geometry.network.kerb import (CORRIDOR_KERB_RADIUS_M, _complement_spans, _corridor_kerb_ways,
                                       _intersect_spans, _kerb_samples_on, _merged_spans)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: this type is layered above this package, so importing it
    # for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel



# ---------------------------------------------------------------------------
# The corridor QUESTIONS. Step 2 of docs/network-model.md: these move onto the road first,
# because they have no goldens and are currently wrong in scratch scripts.
#
# Every rule below is the rule the per-leg code already applies, with its constants IMPORTED
# rather than retyped, asked of a road instead of a leg.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorridorFacts:
    """What is positioned along one corridor: openings, crossings, and where parking is legal.

    Resolved ONCE against one road and handed round: two consumers each assembling the same
    geometry are free to diverge the moment one of their tolerances is touched.
    """
    #: (side, KerbOpening) - every place a vehicle crosses one of the two kerbs.
    openings: tuple = ()
    #: CrossStreet - every other street that meets this one, at any of its junctions.
    crossings: tuple = ()
    #: (station_ft, crossing:markings or None, way id) per surveyed pedestrian crossing.
    marked_crossings: tuple = ()
    #: {side: (NoParkingZone, ...)} - R.S. 39:4-138 and what OSM records, stretch by stretch.
    no_parking: tuple = ()
    #: {side: ((lo, hi), ...)} - where a stall may legally be marked.
    parkable: tuple = ()

    def by_side(self, field: str, side: str) -> tuple:
        return next((values for name, values in getattr(self, field) if name == side), ())


def corridor_facts(corridor: Corridor, models: dict[str, "IntersectionModel"]) -> CorridorFacts:
    """Answer every corridor question about one road, in one pass over the OSM snapshot."""
    kerb_ways = _corridor_kerb_ways(models)
    crossings = _cross_streets_on(corridor, models)
    openings = _openings_on(corridor, models, kerb_ways)
    no_parking = tuple((side, _no_parking_zones_on(corridor, side, models, crossings))
                       for side in ("left", "right"))
    parkable = tuple((side, _complement_spans([(zone.start_ft, zone.end_ft)
                                               for zone in zones], 0.0, corridor.length_ft))
                     for side, zones in no_parking)
    return CorridorFacts(openings=openings, crossings=crossings,
                         marked_crossings=_marked_crossings_on(corridor, models),
                         no_parking=no_parking, parkable=parkable)


def _placed_on_corridor(corridor: Corridor, line: LineString) -> tuple | None:
    """(side, first station, last station) for a traced line lying along one of the road's kerbs.

    THE WAY IS PLACED AS A WHOLE and then its whole extent measured: a dropped kerb across a
    driveway mouth is often drawn ACROSS the opening rather than along the street, so taking only
    the samples that individually sit near the kerb collapsed four of six real openings to zero
    length.
    """
    stations, offsets, _points, keep = _kerb_samples_on(
        corridor.centerline, [junction.node_ft for junction in corridor.junctions], line)
    if keep.sum() < 2:
        return None
    side = "left" if float(np.median(offsets[keep])) > 0 else "right"
    return side, float(stations[keep].min()), float(stations[keep].max())


def _openings_on(corridor: Corridor, models: dict[str, "IntersectionModel"], kerb_ways: dict) -> tuple:
    """((side, KerbOpening), ...) for every place a vehicle crosses one of this road's kerbs.

    BOTH SIGNALS, for the reason src/geometry/kerbs.py's module docstring gives: a dropped kerb
    carries a surveyed extent, a mapped driveway carries none but is tagged in places no dropped
    kerb is. The surveyed extent wins where the two overlap.
    """
    from src.geometry.kerbs import (DRIVEWAY_WIDTH_FT, MIN_OPENING_LENGTH_FT, KerbOpening,
                                    KerbType, OpeningSource, opens_the_kerb)
    from src.sources.osm_context import fetch_driveways

    found = []
    for way_id, (line, tags) in sorted(kerb_ways.items()):
        if not opens_the_kerb(tags):
            continue
        placed = _placed_on_corridor(corridor, line)
        if placed is None:
            continue
        side, start, end = placed
        if end - start < MIN_OPENING_LENGTH_FT:
            continue
        found.append((side, KerbOpening(start_ft=start, end_ft=end,
                                        source=OpeningSource.DROPPED_KERB,
                                        kerb=KerbType.from_tags(tags), way_id=way_id)))

    seen = set()
    for model in models.values():
        for drive in fetch_driveways(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if drive["id"] in seen:
                continue
            seen.add(drive["id"])
            meeting = _driveway_meeting(corridor, drive)
            if meeting is None:
                continue
            side, station = meeting
            if any(other_side == side and opening.start_ft <= station <= opening.end_ft
                   for other_side, opening in found):
                continue
            found.append((side, KerbOpening(start_ft=max(station - DRIVEWAY_WIDTH_FT / 2, 0.0),
                                            end_ft=station + DRIVEWAY_WIDTH_FT / 2,
                                            source=OpeningSource.DRIVEWAY, way_id=drive["id"])))
    return tuple(sorted(found, key=lambda pair: (pair[0], pair[1].start_ft)))


def _driveway_meeting(corridor: Corridor, drive: dict) -> tuple | None:
    """(side, station) where a mapped driveway reaches one of this road's kerb runs, or None.

    Which kerb it meets is decided by distance to the kerb LINE, and the station is taken from the
    point on the KERB nearest it - both for the reasons kerbs.py:_driveway_meetings gives.
    """
    from shapely.ops import nearest_points

    from src.geometry.intersection import to_state_plane
    from src.geometry.kerbs import DRIVEWAY_REACH_FT

    coords = drive.get("coords_wgs84") or []
    if len(coords) < 2:
        return None
    line = LineString(to_state_plane(coords))
    best = None
    for run in corridor.kerb_runs:
        gap = line.distance(run.line)
        if gap > DRIVEWAY_REACH_FT or (best is not None and gap >= best[0]):
            continue
        _on_drive, on_kerb = nearest_points(line, run.line)
        station = float(corridor.centerline.project(Point(on_kerb)))
        best = (gap, run.side, station)
    return None if best is None else (best[1], best[2])


def _corridor_ways(models: dict[str, "IntersectionModel"], predicate) -> dict:
    """{way id: (LineString in feet, tags)} for OSM ways near any member junction that match."""
    from src.geometry.intersection import to_state_plane
    from src.sources.osm_context import fetch_roads

    ways = {}
    for model in models.values():
        for way in fetch_roads(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if not predicate(way.get("tags", {})) or len(way.get("coords_wgs84") or []) < 2:
                continue
            ways[way["id"]] = (LineString(to_state_plane(way["coords_wgs84"])), way["tags"])
    return ways


def _half_width_at(corridor: Corridor, station_ft: float) -> float:
    """Half the traced width at a station, falling back to the corridor's typical half-width.

    Needed wherever a test is "does this reach the carriageway" at a station that may sit in an
    untraced stretch. The fallback is this road's OWN median, not a class assumption.
    """
    width = corridor.width_at_ft(station_ft)
    if width is not None:
        return width / 2
    sampled = [corridor.width_at_ft(float(station))
               for lo, hi in corridor.both_traced_spans()
               for station in np.linspace(lo, hi, 5)]
    real = [w for w in sampled if w is not None]
    from src.geometry.context_roads import ROADWAY_DEFAULT_WIDTH_FT

    return float(np.median(real)) / 2 if real else ROADWAY_DEFAULT_WIDTH_FT / 2


def _cross_streets_on(corridor: Corridor, models: dict[str, "IntersectionModel"]) -> tuple:
    """Every other street that meets this one, anywhere along it.

    The corridor form of src/geometry/cross_streets.py. R.S. 39:4-138(e) applies at EVERY
    intersection, and this is the object that can finally say where they all are: on a corridor
    every junction is somebody's own.
    """
    from shapely.ops import nearest_points

    from src.geometry.context_roads import MAX_HALF_WIDTH_FT, assumed_width_ft, is_carriageway
    from src.geometry.cross_streets import (JOIN_TOLERANCE_FT, MIN_CROSS_ANGLE_DEG, CrossStreet,
                                            _crossing_angle_deg, _sides_of)

    out = []
    for way_id, (line, tags) in sorted(_corridor_ways(models, is_carriageway).items()):
        # NOT a geometric intersection, for the reason cross_streets_ft gives: a side street's OSM
        # way stops on OSM's centreline for the main road while this road is NJDOT's alignment, so
        # `intersects` is False for every real side street on the block. The test is APPROACH.
        if corridor.centerline.distance(line) > MAX_HALF_WIDTH_FT + JOIN_TOLERANCE_FT:
            continue
        on_road, _on_way = nearest_points(corridor.centerline, line)
        station = float(corridor.centerline.project(on_road))
        if on_road.distance(line) > _half_width_at(corridor, station) + JOIN_TOLERANCE_FT:
            continue
        if _crossing_angle_deg(corridor.centerline, line, on_road) < MIN_CROSS_ANGLE_DEG:
            continue
        sides = _sides_of(corridor.centerline, line, on_road)
        if not sides:
            continue
        out.append(CrossStreet(leg=corridor.name, station_ft=station,
                               half_width_ft=assumed_width_ft(tags) / 2, sides=sides,
                               name=tags.get("name"), way_id=way_id))
    return tuple(sorted(out, key=lambda cross: cross.station_ft))


def _marked_crossings_on(corridor: Corridor, models: dict[str, "IntersectionModel"]) -> tuple:
    """((station, crossing:markings or None), ...) for the surveyed pedestrian crossings on it.

    Counted, not drawn - drawing them as traced is stream A of docs/network-renderer-plan.md. What
    this is for is the corridor figure: how many crossings a walk down this street actually has,
    and how many of those the surveyor recorded MARKINGS for.
    """
    from src.geometry.cross_streets import MIN_CROSS_ANGLE_DEG, _crossing_angle_deg
    from src.geometry.intersection import to_state_plane
    from src.sources.osm_context import fetch_crossings

    found = {}
    for model in models.values():
        for crossing in fetch_crossings(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            coords = crossing.get("coords_wgs84") or []
            if len(coords) < 2:
                continue
            line = LineString(to_state_plane(coords))
            middle = line.interpolate(0.5, normalized=True)
            station = float(corridor.centerline.project(middle))
            if middle.distance(corridor.centerline) > _half_width_at(corridor, station):
                continue
            if _crossing_angle_deg(corridor.centerline, line, middle) < MIN_CROSS_ANGLE_DEG:
                continue
            key = tuple(crossing.get("node_ids") or [tuple(coords[0])])[:1]
            found[key] = (station, crossing["tags"].get("crossing:markings"))
    return tuple(sorted(found.values()))


def _no_parking_zones_on(corridor: Corridor, side: str, models: dict[str, "IntersectionModel"],
                         crossings: tuple) -> tuple:
    """Every stretch of one kerb where parking is forbidden, with the clause that forbids it.

    THE SAME FOUR RULES src/geometry/daylighting.py applies to a leg, with its statutory distances
    imported rather than retyped. What is different is only that they are applied along a ROAD:

      * (e) 25 ft from the side line of EVERY intersecting street, which is what the statute says.
      * (h) 50 ft from a stop sign and (i) 10 ft from a hydrant, as radii round a point.
      * what OSM RECORDS about this kerb, per stretch of it.

    The junction end of a leg needed a corner fillet's tangent point to find the side line
    (daylighting.sideline_station_ft); on a corridor the member junctions arrive through the same
    cross-street list as every other side street.
    """
    from src.geometry.daylighting import (FIRE_HYDRANT_SETBACK_FT, FOOTWAY_REACH_FT,
                                          SIDELINE_SETBACK_FT, STOP_SIGN_SETBACK_FT, NoParkingZone)
    from src.geometry.intersection import (PARKING_RESTRICTION_KEYS, parking_is_restricted,
                                           parking_restriction_by_side)
    from src.sources.osm_context import fetch_street_furniture, fetch_traffic_control

    zones = []
    for cross in crossings:
        if side not in cross.sides:
            continue
        near_ft, far_ft = cross.mouth_ft
        zones.append(NoParkingZone(
            near_ft - SIDELINE_SETBACK_FT, far_ft + SIDELINE_SETBACK_FT,
            f"R.S. 39:4-138(e), {SIDELINE_SETBACK_FT:.0f} ft from the side line of "
            f"{cross.citation}"))

    for radius_ft, citation, nodes in (
            (STOP_SIGN_SETBACK_FT, "R.S. 39:4-138(h), 50 ft from a stop sign",
             _corridor_nodes(models, fetch_traffic_control, lambda t: t.get("highway") == "stop")),
            (FIRE_HYDRANT_SETBACK_FT, "R.S. 39:4-138(i), 10 ft from a fire hydrant",
             _corridor_nodes(models, fetch_street_furniture,
                             lambda t: t.get("emergency") == "fire_hydrant"))):
        for point in nodes:
            stations, offsets = station_offset_many(corridor.centerline,
                                                    np.asarray([point], dtype=float))
            station, offset = float(stations[0]), float(offsets[0])
            if (offset > 0) != (side == "left"):
                continue        # a hydrant on the north kerb says nothing about the south one
            if abs(offset) > _half_width_at(corridor, station) + FOOTWAY_REACH_FT:
                continue        # further out than a footway reaches: somebody else's kerb
            if not -radius_ft <= station <= corridor.length_ft + radius_ft:
                continue
            zones.append(NoParkingZone(station - radius_ft, station + radius_ft, citation))

    for way_id, span, tags, aligned in _road_spans_on(corridor, models):
        restriction = parking_restriction_by_side(tags, aligned)[side]
        if not parking_is_restricted(restriction):
            continue
        key = PARKING_RESTRICTION_KEYS["both" if tags.get(PARKING_RESTRICTION_KEYS["both"])
                                       else side]
        zones.append(NoParkingZone(span[0], span[1],
                                   f"OSM {key}={restriction} on way {way_id}"))
    return tuple(sorted(zones, key=lambda zone: zone.start_ft))


def _corridor_nodes(models: dict[str, "IntersectionModel"], fetch, predicate) -> list[tuple]:
    """Every OSM node near any member junction that matches, in state-plane feet, once each."""
    from src.geometry.intersection import to_state_plane

    found = {}
    for model in models.values():
        for node in fetch(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if predicate(node.get("tags", {})):
                found[(round(node["lon"], 7), round(node["lat"], 7))] = None
    return to_state_plane(list(found)) if found else []


def _road_spans_on(corridor: Corridor, models: dict[str, "IntersectionModel"]) -> list[tuple]:
    """(way id, (start_ft, end_ft), tags, aligned) for every OSM way lying ALONG this road.

    A LIST and not one way, because what a way says varies along a street and OSM says so by
    SPLITTING THE WAY.

    Matched on lying along the road over a real stretch of it, with the same two thresholds
    _match_legs_to_osm_roads uses, and restricted to CARRIAGEWAY classes for the reason recorded
    there.
    """
    from src.geometry.intersection import (MIN_ROAD_SPAN_FT, ROAD_MATCH_HIGHWAY_CLASSES,
                                           ROAD_MATCH_MAX_ANGLE_DEG, ROAD_MATCH_MAX_OFFSET_FT)

    spans = []
    ways = _corridor_ways(models, lambda t: t.get("highway") in ROAD_MATCH_HIGHWAY_CLASSES)
    road_dir = line_direction(corridor.centerline)
    for way_id, (line, tags) in sorted(ways.items()):
        along = float(np.dot(line_direction(line), road_dir))
        stations, offsets = station_offset_many(corridor.centerline,
                                                np.asarray(line.coords, dtype=float))
        lo = max(float(stations.min()), 0.0)
        hi = min(float(stations.max()), corridor.length_ft)
        if hi - lo < MIN_ROAD_SPAN_FT:
            continue
        covering = (stations >= -MIN_ROAD_SPAN_FT) & (stations <= corridor.length_ft
                                                      + MIN_ROAD_SPAN_FT)
        if not covering.any() or float(np.abs(offsets[covering]).min()) > ROAD_MATCH_MAX_OFFSET_FT:
            continue
        # Per-station rather than on the chord: a 2,849 ft road bends, so its end-to-end direction
        # is a poor description of it anywhere in particular. Taken where the way actually overlaps.
        local = np.asarray([frame_at(corridor.centerline, float(s))[1]
                            for s in stations[covering]])
        way_dir = line_direction(line)
        angle = np.degrees(np.arccos(np.clip(np.abs(local @ way_dir).max(), -1.0, 1.0)))
        if angle > ROAD_MATCH_MAX_ANGLE_DEG:
            continue
        spans.append((way_id, (lo, hi), tags, bool(along >= 0)))
    return spans


def marked_parking_capacity(corridor: Corridor, facts: CorridorFacts, side: str,
                            within: tuple = ()) -> tuple[int, float]:
    """(stalls, ft of kerb they were counted over) where parking is legal on one side.

    `within` narrows the count to a set of spans - which is how the report prints a figure and its
    own coverage from ONE function rather than from two that could drift apart. Returns the length
    as well as the count because "486 stalls" over a corridor that is 43% traced is not a corridor
    figure.

    Stall length is PARKING_STALL_LENGTH_DEFAULT_FT, imported - the same figure the per-leg
    marking uses.
    """
    from src.geometry.model import whole_stalls_ft
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT

    runs = facts.by_side("parkable", side)
    if within:
        runs = _intersect_spans(runs, within)
    stalls, measured_ft = 0, 0.0
    for lo, hi in runs:
        measured_ft += hi - lo
        # Through the same counter parking_stall_lines_ft uses, so the reported number is the
        # number that would be drawn over a run this open - it does not know about the driveway
        # and crossing cuts a real paint pass would also apply, same as the "stalls" figure
        # elsewhere in this module.
        stalls += whole_stalls_ft(hi - lo, PARKING_STALL_LENGTH_DEFAULT_FT)
    return stalls, measured_ft


# How finely the OSM fetch window is walked along a road. Only used to say how much of the road
# the fetch actually covered, so a couple of feet of edge either way does not matter.
_WINDOW_SAMPLE_FT = 10.0


def osm_window_spans(corridor: Corridor,
                     models: dict[str, "IntersectionModel"]) -> tuple[tuple[float, float], ...]:
    """The stretches of a road that fall inside the OSM fetch window round its member junctions.

    THE DENOMINATOR FOR ANYTHING COUNTED OUT OF OSM, and it exists because "nothing fetched" and
    "nothing mapped" arrive identically. At the junction radius of 120 m the three circles on Broad
    St leave 173 m outside every window; at CORRIDOR_KERB_RADIUS_M the same road is fully covered.
    Reported rather than believed.
    """
    reach_ft = CORRIDOR_KERB_RADIUS_M / 0.3048
    centres = [model.center_ft for model in models.values()]
    n = max(int(np.ceil(corridor.length_ft / _WINDOW_SAMPLE_FT)) + 1, 2)
    inside = []
    for station in np.linspace(0.0, corridor.length_ft, n):
        point = corridor.centerline.interpolate(float(station))
        if any(point.distance(centre) <= reach_ft for centre in centres):
            inside.append((float(station) - _WINDOW_SAMPLE_FT / 2,
                           float(station) + _WINDOW_SAMPLE_FT / 2))
    return _intersect_spans(_merged_spans(inside), ((0.0, corridor.length_ft),))
