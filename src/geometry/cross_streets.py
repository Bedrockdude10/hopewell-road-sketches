"""Where a leg crosses ANOTHER street, and what that costs the kerb.

The fact needed was already fetched: `fetch_roads` pulls every `highway=*` way in range and has
been read only for `overtaking=no`; the geometry was thrown away. This finds the crossings and
hands them to two existing mechanisms:

  * a NO-PARKING ZONE either side of the cross street (src/geometry/daylighting.py), the same
    statutory rule the junction end already gets;
  * a KERB OPENING across its mouth (src/geometry/kerbs.py), the same mechanism a driveway
    already gets. NOT because a cross street IS a driveway (MUTCD 11th ed. 1C.02(113)(b),
    N.J.S.A. 39:1-1): the MECHANISM is shared; the RULE is not.

AND ITS CROSSWALKS COME WITH IT. R.S. 39:4-138(e) has two arms - 25 ft from the nearest
crosswalk, and 25 ft from the side line - and only the side line one was applied here.

N.J.S.A. 39:1-1 defines a crosswalk as "either MARKED OR UNMARKED existing at each approach
of every roadway intersection", so every cross street has two across our leg whether or not a
surveyor traced any paint.

WHICH SIDE. The side is read off the way's own geometry: which side of our centerline its
vertices fall on, near the meeting. A way with vertices both sides is a crossroads and opens both.

WHAT IS NOT A CROSS STREET. A driveway and a parking aisle are already PavedSurfaces and
already open the kerb. The junction this drawing is about is excluded by distance: its arms
meet at the centre and its setback is already applied from the side line.
"""
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from src.geometry.context_roads import assumed_width_ft, is_carriageway
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# THIS JUNCTION'S OWN ARMS, and the test is about the WAY rather than about distance along a leg.
# Its arms all meet at the centre and the (e) setback is already measured from the side line there.
#
# An own arm is not merely NEAR the centre, it PASSES THROUGH it. The discriminator is the WAY's
# own distance to the centre, and the tolerance only has to cover NJDOT's alignment disagreeing
# with OSM's centreline - well under the 29 ft widest corner return.
JUNCTION_OWN_ARM_TOLERANCE_FT = 20.0

# How far along the cross street to look when deciding which side of us it leaves on. Far
# enough to clear the meeting itself, short enough not to follow a curving street back round.
SIDE_PROBE_FT = 40.0

# Below this a "side" is digitizing noise - a way that touches ours and runs on in line with it.
MIN_SIDE_OFFSET_FT = 3.0

# How far outside our own carriageway a street may stop and still be meeting us. A side street's
# way ends on OSM's centreline for the main road while our leg is the NJDOT alignment, so the
# two do not touch; this is the slack that covers the disagreement plus a corner return.
JOIN_TOLERANCE_FT = 20.0

# How far off parallel a way has to run to be crossing us rather than lying alongside us. This
# leg's OWN way runs alongside it for its whole length a few feet off, so without this every leg
# would report itself as its own cross street. Same discriminator, and roughly the same angle,
# that _matched_crossings uses to stop a side street's crosswalk being credited to the main road.
MIN_CROSS_ANGLE_DEG = 25.0

# How far outside a cross street's mouth a traced crossing may sit and still be that junction's.
# Measured: the surveyed crossings fit a mean 8.3 ft beyond the kerb line, sigma 2.4, worst case
# 13.9. Comfortably past that worst case and nowhere near the next junction (130 ft apart).
MAX_CROSSWALK_FROM_MOUTH_FT = 25.0

# How far off square a traced way has to run to be a crossing OF OUR LEG rather than a crossing of
# the cross street. The same threshold src/render/crosswalks.py:MIN_CROSSING_ANGLE_DEG picks.
MIN_CROSSWALK_SQUARENESS_DEG = 30.0


def _crossing_angle_deg(leg_line: LineString, way_line: LineString, at: Point) -> float:
    """Angle between the two centrelines where they meet, folded into 0-90 degrees."""
    import math

    def bearing(line: LineString, point, span=15.0):
        along = line.project(point)
        a = line.interpolate(max(along - span, 0.0))
        b = line.interpolate(min(along + span, line.length))
        return math.atan2(b.y - a.y, b.x - a.x)

    delta = math.degrees(abs(bearing(leg_line, at) - bearing(way_line, at))) % 180.0
    return min(delta, 180.0 - delta)


@dataclass(frozen=True)
class CrossStreetCrosswalk:
    """One crosswalk across OUR leg at a cross street, as a station along our centerline.

    N.J.S.A. 39:1-1 defines a crosswalk as "either MARKED OR UNMARKED existing at each approach
    of every roadway intersection", so the thing R.S. 39:4-138(e) measures 25 ft from is there
    whether or not there is paint on it.

    `is_surveyed` says which of the two this is. Unsurveyed: CROSSWALK_OFFSET_FROM_KERB_FT
    beyond the mouth - the same measured 8.3 ft that places THIS junction's own crossings,
    reused so a crosswalk at Blackwell and one at Greenwood are placed by one rule.
    """
    station_ft: float
    is_surveyed: bool
    #: Nodes of the traced crossing way, empty for a placed one. Nodes and not a way id because
    #: src/sources/osm_context.py:fetch_crossings does not carry one - the same thing
    #: src/geometry/surveyed.py cites when it has to name a crossing.
    node_ids: tuple = ()

    @property
    def citation(self) -> str:
        if self.is_surveyed:
            where = f" at OSM node {self.node_ids[0]}" if self.node_ids else ""
            return f"the surveyed crosswalk{where}"
        return ("the unmarked crosswalk N.J.S.A. 39:1-1 puts at every intersection approach "
                "(position estimated - no crossing is traced here)")


@dataclass(frozen=True)
class CrossStreet:
    """Where one other street meets one of our legs."""
    leg: str
    #: Station along the leg's centerline where the two cross.
    station_ft: float
    #: Half the cross street's carriageway, so its mouth spans station +/- this.
    half_width_ft: float
    #: Which of our kerbs it opens - "left", "right", or both for a crossroads.
    sides: frozenset
    name: str | None = None
    way_id: int | None = None
    #: The crosswalks across OUR leg here - normally two, one each side of the mouth.
    crosswalks: tuple = ()

    @property
    def mouth_ft(self) -> tuple[float, float]:
        return self.station_ft - self.half_width_ft, self.station_ft + self.half_width_ft

    @property
    def citation(self) -> str:
        where = f" on way {self.way_id}" if self.way_id is not None else ""
        return f"{self.name or 'an unnamed street'}{where}"


def _sides_of(leg_line: LineString, way_line: LineString, at: Point) -> frozenset:
    """Which side(s) of our leg the cross street leaves on, from its own vertices."""
    from src.geometry.model import station_offset_many

    along = way_line.project(at)
    probes = [along + d for d in (-SIDE_PROBE_FT, -SIDE_PROBE_FT / 2,
                                   SIDE_PROBE_FT / 2, SIDE_PROBE_FT)]
    points = [way_line.interpolate(p) for p in probes if 0 <= p <= way_line.length]
    if not points:
        return frozenset()
    _stations, offsets = station_offset_many(
        leg_line, np.asarray([(p.x, p.y) for p in points], dtype=float))
    sides = set()
    if (offsets > MIN_SIDE_OFFSET_FT).any():
        sides.add("left")
    if (offsets < -MIN_SIDE_OFFSET_FT).any():
        sides.add("right")
    return frozenset(sides)


def cross_streets_from_model(model: "IntersectionModel") -> dict:
    """{leg name: [CrossStreet]} for this model, RESOLVED ONCE at load.

    Reads `model.cross_streets` rather than deriving it again. Falls back to computing it for
    a stand-in model that carries legs and no resolved field (the centerline-precedence tests).
    """
    resolved = getattr(model, "cross_streets", None)
    if resolved is not None:
        return resolved
    if not all(hasattr(model, attr) for attr in ("center_wgs84", "center_ft", "legs")):
        return {}
    return cross_streets_ft(model.center_wgs84, model.center_ft, model.legs)


def _traced_crosswalks_on(leg_line: LineString, half_width_ft: float, crossing_lines: list
                           ) -> list[tuple[float, tuple]]:
    """(station, node ids) for every traced crossing that runs ACROSS this leg.

    Resolved once for the whole leg and handed to each cross street.

    Two guards, the same two _matched_crossings uses at the modelled junction - stated here
    rather than shared because its 80 ft junction bound would throw away every crossing this
    one exists to find:

      * LATERALLY inside our own carriageway, plus slack for a corner return.
      * SQUARE-ISH to us - the direction test that catches a crossing of the cross street
        running along our own kerb.
    """
    found = []
    for line, node_ids in crossing_lines:
        mid = line.interpolate(0.5, normalized=True)
        station_ft = leg_line.project(mid)
        if not 0 <= station_ft <= leg_line.length:
            continue
        if leg_line.interpolate(station_ft).distance(mid) > half_width_ft + JOIN_TOLERANCE_FT:
            continue
        if _crossing_angle_deg(leg_line, line, mid) < MIN_CROSSWALK_SQUARENESS_DEG:
            continue
        found.append((station_ft, node_ids))
    return found


def _crosswalks_of(cross: "CrossStreet", traced: list[tuple[float, tuple]]) -> tuple:
    """The crosswalks across our leg at one cross street - surveyed where traced, else placed.

    ONE PER END, decided separately: a junction routinely has a zebra on one approach and
    nothing on the other. Nearest to the mouth wins where several qualify.
    """
    from src.geometry.model import CROSSWALK_OFFSET_FROM_KERB_FT

    near_ft, far_ft = cross.mouth_ft
    out = []
    for edge_ft, direction in ((near_ft, -1), (far_ft, +1)):
        # Outside the mouth only: a crossing INSIDE it is the crossing of the cross street, or
        # this junction's other approach, and neither is this end's.
        candidates = sorted((abs(station - edge_ft), station, node_ids)
                            for station, node_ids in traced
                            if 0 <= direction * (station - edge_ft) <= MAX_CROSSWALK_FROM_MOUTH_FT)
        if candidates:
            _gap, station_ft, node_ids = candidates[0]
            out.append(CrossStreetCrosswalk(station_ft, is_surveyed=True, node_ids=node_ids))
        else:
            out.append(CrossStreetCrosswalk(
                edge_ft + direction * CROSSWALK_OFFSET_FROM_KERB_FT, is_surveyed=False))
    return tuple(out)


def _crossing_lines_ft(center_wgs84) -> list:
    """Every traced OSM crossing near this junction, in state-plane feet. [] if none reachable.

    Fetched at the radius the crossings are DRAWN at, so a crossing in the picture is a crossing
    the statute is measured from.
    """
    from src.geometry.intersection import to_state_plane
    from src.geometry.treatments.crossings import CROSSING_CONTEXT_RADIUS_M
    from src.render.frame import context_radius_m
    from src.sources.osm_context import fetch_crossings

    try:
        records = fetch_crossings(center_wgs84, radius_m=context_radius_m(CROSSING_CONTEXT_RADIUS_M))
    except Exception:
        return []
    lines = []
    for record in records:
        coords = record.get("coords_wgs84") or []
        if len(coords) < 2:
            continue        # a crossing NODE has no direction - see surveyed._traced_line_ft
        lines.append((LineString(to_state_plane(coords)),
                      tuple(record.get("node_ids") or ())))
    return lines


def cross_streets_ft(center_wgs84, center_ft: Point, legs: dict) -> dict:
    """{leg name: [CrossStreet]} for every other street these legs run across.

    Takes the pieces rather than a model so `load_intersection_model` can call it while the
    model is still being assembled. Guarded: no OSM reachable answers "none" rather than raising.
    """
    from src.geometry.intersection import ROAD_CONTEXT_RADIUS_M, to_state_plane
    from src.render.frame import context_radius_m
    from src.sources.osm_context import fetch_roads

    try:
        ways = fetch_roads(center_wgs84, radius_m=context_radius_m(ROAD_CONTEXT_RADIUS_M))
    except Exception:
        return {}
    crossing_lines = _crossing_lines_ft(center_wgs84)

    out: dict[str, list[CrossStreet]] = {}
    for leg_name, leg in legs.items():
        # Once per leg, not once per cross street - see _traced_crosswalks_on.
        traced = _traced_crosswalks_on(leg.centerline, leg.curb_to_curb_ft / 2, crossing_lines)
        for way in ways:
            tags = way.get("tags", {})
            coords = way.get("coords_wgs84") or []
            if not is_carriageway(tags) or len(coords) < 2:
                continue
            way_line = LineString(to_state_plane(coords))
            # NOT a geometric intersection. A side street's OSM way stops on OSM's centreline for
            # the main road, and our leg is the NJDOT alignment - the two are a few feet apart.
            # So the test is APPROACH: a street that comes within our own carriageway is meeting us.
            reach_ft = leg.curb_to_curb_ft / 2 + JOIN_TOLERANCE_FT
            if leg.centerline.distance(way_line) > reach_ft:
                continue
            if way_line.distance(center_ft) <= JUNCTION_OWN_ARM_TOLERANCE_FT:
                continue    # one of this junction's own arms - see the constant
            on_leg, _on_way = nearest_points(leg.centerline, way_line)
            station_ft = leg.centerline.project(on_leg)
            # A way ALONGSIDE us is not a way across us - and this leg's own OSM way runs
            # alongside it for its whole length, a few feet off. The same orientation test
            # _matched_crossings uses to stop a side street's crosswalk being read as ours.
            if _crossing_angle_deg(leg.centerline, way_line, on_leg) < MIN_CROSS_ANGLE_DEG:
                continue
            sides = _sides_of(leg.centerline, way_line, on_leg)
            if not sides:
                continue
            cross = CrossStreet(
                leg=leg_name, station_ft=station_ft,
                half_width_ft=assumed_width_ft(tags) / 2, sides=sides,
                name=tags.get("name"), way_id=way.get("id"))
            out.setdefault(leg_name, []).append(
                replace(cross, crosswalks=_crosswalks_of(cross, traced)))
    for legs in out.values():
        legs.sort(key=lambda c: c.station_ft)
    return out
