"""Where a leg crosses ANOTHER street, and what that costs the kerb.

Legs are drawn as far as the frame asks now, which on Broad St is 374 ft - past Blackwell
Avenue, past Model Avenue, past every side street on the block. The markings did not know:
stalls were marked straight across the mouth of a cross street, and the statutory setback was
applied only at the junction this drawing is about. Both are wrong on the ground, and R.S.
39:4-138(e) does not say "the intersection you happen to be drawing".

The fact needed to fix it was already fetched. `fetch_roads` pulls every `highway=*` way in
range and has been read only for `overtaking=no`; the geometry - where those ways actually
cross ours - was thrown away. So this finds the crossings and hands them to the two mechanisms
that already exist for exactly this shape of thing:

  * a NO-PARKING ZONE either side of the cross street (src/geometry/daylighting.py), which is
    the same statutory rule the junction end already gets, applied where the statute applies;
  * a KERB OPENING across its mouth (src/geometry/kerbs.py), which is what a driveway already
    gets - a cross street is a driveway that a whole street drives out of, and the paint has to
    break over its width for the same reason.

WHICH SIDE. A side street joins on one side and the kerb opposite is untouched, so an opening
on both sides would break paint that is really there. The side is read off the way's own
geometry: which side of our centerline its vertices fall on, near the meeting. A way with
vertices both sides is a crossroads and opens both.

WHAT IS NOT A CROSS STREET. A driveway and a parking aisle are already PavedSurfaces and
already open the kerb (is_carriageway rejects them). And the junction this drawing is about is
excluded by distance: its own arms meet at the centre, its setback is already applied from the
side line, and reporting it twice would credit one corner with two citations.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from src.geometry.context_roads import assumed_width_ft, is_carriageway

# A meeting closer than this to the junction centre IS this junction, not a cross street. Its
# arms all meet at the centre and the (e) setback is already measured from the side line there
# (src/geometry/daylighting.py), so admitting them would double-cite one corner. Comfortably
# past the widest corner return in the set (29 ft) and short of the nearest real side street.
JUNCTION_OWN_REACH_FT = 50.0

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


def _crossing_angle_deg(leg_line: LineString, way_line: LineString, at: Point) -> float:
    """Angle between the two centrelines where they meet, folded into 0-90 degrees."""
    import math

    def bearing(line, point, span=15.0):
        along = line.project(point)
        a = line.interpolate(max(along - span, 0.0))
        b = line.interpolate(min(along + span, line.length))
        return math.atan2(b.y - a.y, b.x - a.x)

    delta = math.degrees(abs(bearing(leg_line, at) - bearing(way_line, at))) % 180.0
    return min(delta, 180.0 - delta)


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


def cross_streets_from_model(model) -> dict:
    """{leg name: [CrossStreet]} for this model, RESOLVED ONCE at load.

    Reads what `load_intersection_model` already worked out (IntersectionModel.cross_streets)
    rather than deriving it again. It was derived twice - once for the kerb openings and once
    seeding the DesignState - which is the shape src/geometry/intersection.py:PavedSurface's
    docstring is about: two consumers assembling the same geometry, free to diverge the moment
    one of their tolerances is touched.

    Falls back to computing it for a stand-in model that carries legs and no resolved field,
    which is what the centerline-precedence tests build.
    """
    resolved = getattr(model, "cross_streets", None)
    if resolved is not None:
        return resolved
    if not all(hasattr(model, attr) for attr in ("center_wgs84", "center_ft", "legs")):
        return {}
    return cross_streets_ft(model.center_wgs84, model.center_ft, model.legs)


def cross_streets_ft(center_wgs84, center_ft, legs: dict) -> dict:
    """{leg name: [CrossStreet]} for every other street these legs run across.

    Takes the pieces rather than a model, so `load_intersection_model` can call it while the
    model is still being assembled - the same shape _paved_surfaces_ft has, and for the same
    reason. Guarded: no OSM reachable answers "none" rather than raising.
    """
    from src.geometry.intersection import ROAD_CONTEXT_RADIUS_M, _to_state_plane
    from src.render.frame import context_radius_m
    from src.sources.osm_context import fetch_roads

    try:
        ways = fetch_roads(center_wgs84, radius_m=context_radius_m(ROAD_CONTEXT_RADIUS_M))
    except Exception:
        return {}

    out: dict[str, list[CrossStreet]] = {}
    for leg_name, leg in legs.items():
        for way in ways:
            tags = way.get("tags", {})
            coords = way.get("coords_wgs84") or []
            if not is_carriageway(tags) or len(coords) < 2:
                continue
            way_line = LineString(_to_state_plane(coords))
            # NOT a geometric intersection. A side street's OSM way stops on OSM's centreline for
            # the main road, and our leg is the NJDOT alignment - the two are a few feet apart, so
            # `intersects` is False for every real side street on the block. Measured at Broad &
            # Greenwood, requiring a true crossing found 2 ways and both were Broad Street itself.
            # So the test is APPROACH: a street that comes within our own carriageway is meeting
            # us, whoever drew which centreline where.
            reach_ft = leg.curb_to_curb_ft / 2 + JOIN_TOLERANCE_FT
            if leg.centerline.distance(way_line) > reach_ft:
                continue
            on_leg, _on_way = nearest_points(leg.centerline, way_line)
            if on_leg.distance(center_ft) <= JUNCTION_OWN_REACH_FT:
                continue
            station_ft = leg.centerline.project(on_leg)
            # A way ALONGSIDE us is not a way across us - and this leg's own OSM way runs
            # alongside it for its whole length, a few feet off. The same orientation test
            # _matched_crossings uses to stop a side street's crosswalk being read as ours.
            if _crossing_angle_deg(leg.centerline, way_line, on_leg) < MIN_CROSS_ANGLE_DEG:
                continue
            sides = _sides_of(leg.centerline, way_line, on_leg)
            if not sides:
                continue
            out.setdefault(leg_name, []).append(CrossStreet(
                leg=leg_name, station_ft=station_ft,
                half_width_ft=assumed_width_ft(tags) / 2, sides=sides,
                name=tags.get("name"), way_id=way.get("id")))
    for legs in out.values():
        legs.sort(key=lambda c: c.station_ft)
    return out
