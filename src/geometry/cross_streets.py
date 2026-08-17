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
    gets - the paint has to break over its width for the same reason. NOT because a cross street
    IS a driveway: it is emphatically not (MUTCD 11th ed. 1C.02(113)(b), N.J.S.A. 39:1-1, and
    kerbs.OpeningSource.is_an_intersection), and reading the two as one thing is what carried a
    parking edge line straight across Blackwell Avenue. The MECHANISM is shared; the RULE is not.

AND ITS CROSSWALKS COME WITH IT. R.S. 39:4-138(e) has two arms - 25 ft from the nearest
crosswalk, and 25 ft from the side line - and only the side line one was applied here, so a
junction with a marked zebra across our own street got the setback owed to a junction with
nothing. Both arms are resolved on the CrossStreet now, because the statute measures from the
crosswalk and finding it is a question about this street, not about parking.

N.J.S.A. 39:1-1 is what makes that a rule rather than a per-site data question: a crosswalk is
"that part of a highway at an intersection, EITHER MARKED OR UNMARKED EXISTING AT EACH APPROACH
OF EVERY ROADWAY INTERSECTION". So every cross street has two of them across our leg whether or
not a surveyor traced any paint, and CrossStreetCrosswalk carries whether this one was traced or
placed - the setback is the same either way, and the citation is not.

WHICH SIDE. A side street joins on one side and the kerb opposite is untouched, so an opening
on both sides would break paint that is really there. The side is read off the way's own
geometry: which side of our centerline its vertices fall on, near the meeting. A way with
vertices both sides is a crossroads and opens both.

WHAT IS NOT A CROSS STREET. A driveway and a parking aisle are already PavedSurfaces and
already open the kerb (is_carriageway rejects them). And the junction this drawing is about is
excluded by distance: its own arms meet at the centre, its setback is already applied from the
side line, and reporting it twice would credit one corner with two citations.
"""
from dataclasses import dataclass, replace

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

# How far outside a cross street's mouth a traced crossing may sit and still be that junction's.
# Measured: src/geometry/model/context.py:CROSSWALK_OFFSET_FROM_KERB_FT fits the 11 surveyed
# crossings at this project's four sites to a mean 8.3 ft beyond the kerb line, sigma 2.4, worst
# case 13.9. This is comfortably past that worst case and nowhere near the next junction down the
# block - the closest pair of cross streets in the set (Blackwell and Model Avenue) is 130 ft
# apart, so a crossing cannot be credited to the wrong one.
MAX_CROSSWALK_FROM_MOUTH_FT = 25.0

# How far off square a traced way has to run to be a crossing OF OUR LEG rather than a crossing of
# the cross street. The same threshold src/render/crosswalks.py:MIN_CROSSING_ANGLE_DEG picks out of
# the same bimodal spread (true matches 82.3-89.8 deg, false ones 0.6-5.9), and it has real work to
# do here: at Blackwell three crossings are traced, and the one running along our own kerb at
# offset 26.5-28.2 ft is the crossing of BLACKWELL, which puts no setback on Broad St's kerb.
MIN_CROSSWALK_SQUARENESS_DEG = 30.0


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
class CrossStreetCrosswalk:
    """One crosswalk across OUR leg at a cross street, as a station along our centerline.

    A crosswalk and not a marked crosswalk, which is the whole reason this type exists rather
    than the crossings simply being looked up where they are drawn. N.J.S.A. 39:1-1 defines one
    as "either MARKED OR UNMARKED existing at each approach of every roadway intersection", so
    the thing R.S. 39:4-138(e) measures 25 ft from is there whether or not there is paint on it,
    and a setback that appeared only where somebody had traced a zebra would be reporting the
    survey's coverage as if it were the law's reach.

    `is_surveyed` says which of the two this is, and it changes the citation and nothing else.
    Surveyed: the traced way's own position. Unsurveyed: CROSSWALK_OFFSET_FROM_KERB_FT beyond the
    mouth - the same measured 8.3 ft that src/geometry/model/context.py:crosswalk_estimate_ft
    places THIS junction's own crossings with, reused rather than re-guessed so a crosswalk at
    Blackwell and a crosswalk at Greenwood are placed by one rule.
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


def cross_streets_from_model(model) -> dict:
    """{leg name: [CrossStreet]} for this model, RESOLVED ONCE at load.

    Reads what `load_intersection_model` already worked out (IntersectionModel.cross_streets)
    rather than deriving it again. It was derived twice - once for the kerb openings and once
    seeding the DesignState - which is the shape src/geometry/intersection/junction.py:PavedSurface's
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


def _traced_crosswalks_on(leg_line: LineString, half_width_ft: float, crossing_lines: list
                           ) -> list[tuple[float, tuple]]:
    """(station, node ids) for every traced crossing that runs ACROSS this leg.

    Resolved once for the whole leg and then handed to each cross street, rather than searched
    per cross street: a leg carried out with the frame passes several, and re-projecting every
    crossing way per street is the same work done n times.

    Two guards, and they are the two _matched_crossings uses at the modelled junction, for the
    same reasons - stated here rather than shared because that function answers a different
    question (which of THIS junction's four legs owns this crossing) and its 80 ft junction bound
    would throw away every crossing this one exists to find:

      * LATERALLY inside our own carriageway, plus slack for a traced kerb that flares through
        a corner return, which is exactly where a crossing sits.
      * SQUARE-ISH to us. Blackwell has a crossing traced along Broad St's own kerb at offset
        26.5-28.2 ft - the crossing OF Blackwell - and it passes the lateral test comfortably.
        It is not a crossing of Broad St and puts no setback on Broad St's kerb.
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

    ONE PER END, and the ends are decided separately. A junction routinely has a zebra on one
    approach and nothing on the other (Blackwell has both of its traced; Model Avenue has
    neither), and taking the estimate for both because one end was missing would throw away the
    end somebody surveyed, which is the same mistake kerbs._mouth_from_the_tracing declines to
    make about the mouth itself.

    Nearest to the mouth wins where several qualify - the crossing governing this approach is the
    one at it, not one further down the block that happens to be inside the window.
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

    Fetched at the same radius the crossings are DRAWN at (src/geometry/surveyed.py takes
    CROSSING_CONTEXT_RADIUS_M through context_radius_m for exactly this reason), so a crossing
    that is in the picture is a crossing the statute is measured from. A narrower radius here
    would put a setback on the kerbs near the junction and none on the ones further out, which
    reads as a drawing error rather than as a fetch bound.

    Guarded like the rest of this module: no OSM reachable answers "none traced", which falls
    every crosswalk back to the unmarked position the statute gives it anyway.
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


def cross_streets_ft(center_wgs84, center_ft, legs: dict) -> dict:
    """{leg name: [CrossStreet]} for every other street these legs run across.

    Takes the pieces rather than a model, so `load_intersection_model` can call it while the
    model is still being assembled - the same shape _paved_surfaces_ft has, and for the same
    reason. Guarded: no OSM reachable answers "none" rather than raising.
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
            cross = CrossStreet(
                leg=leg_name, station_ft=station_ft,
                half_width_ft=assumed_width_ft(tags) / 2, sides=sides,
                name=tags.get("name"), way_id=way.get("id"))
            out.setdefault(leg_name, []).append(
                replace(cross, crosswalks=_crosswalks_of(cross, traced)))
    for legs in out.values():
        legs.sort(key=lambda c: c.station_ft)
    return out
