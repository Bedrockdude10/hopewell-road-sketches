"""Where a car may legally park near these intersections, and marking the rest as clear.

"Daylighting" is keeping the approach to a crossing free of parked vehicles. New Jersey does it
through the ordinary parking prohibitions in R.S. 39:4-138. Marking the setback is the treatment.

WHAT THE LAW SAYS (R.S. 39:4-138, as amended by P.L. 2009 c.257). Parking is prohibited:

  (a)  within an intersection
  (e)  within 25 ft of the nearest crosswalk or SIDE LINE of a street or intersecting
       highway - reduced to 10 ft where a curb extension or bulbout has been built
  (h)  within 50 ft of a "stop" sign
  (i)  within 10 ft of a fire hydrant

All four are floors, and the binding one is whichever sits furthest from the junction.

WHAT A MUNICIPALITY MAY CHANGE (R.S. 39:4-138.6). Hopewell Borough may set its own distances
by ordinance, but may not permit parking within 25 ft of a crosswalk or side line, nor within
50 ft of a stop sign in a school zone. The Borough's traffic chapter is not machine-readable,
so these are the STATE figures - the correct default in the absence of a local ordinance.

None of this is a rendering preference. A proposal that marks a stall inside one of these
distances is proposing something illegal, which is why check_parking_is_legal treats it as a
scene invariant.
"""
from dataclasses import dataclass

import numpy as np

from src.geometry.model import corner_tangent_station_ft, station_offset_many

# R.S. 39:4-138(e). The headline daylighting distance.
CROSSWALK_SETBACK_FT = 25.0
# R.S. 39:4-138(e), second clause: a curb extension has already taken the parking lane out
# of the sight line, so the statute allows parking to resume 10 ft from the crossing.
CROSSWALK_SETBACK_WITH_BULBOUT_FT = 10.0
# R.S. 39:4-138(e), the other arm - measured from the side line of the intersecting street,
# not from its centre. This is what governs a leg with no marked crossing.
SIDELINE_SETBACK_FT = 25.0
# The clause reads "within 25 feet of the nearest crosswalk OR SIDE LINE ... or within 10
# feet of the nearest crosswalk or side line ... if a curb extension or bulbout has been
# constructed". The reduction applies to both arms - applying it to the crosswalk alone
# leaves the side line binding at 25 ft and the extension buys nothing.
SIDELINE_SETBACK_WITH_BULBOUT_FT = 10.0
# R.S. 39:4-138(h).
STOP_SIGN_SETBACK_FT = 50.0
# R.S. 39:4-138(i).
FIRE_HYDRANT_SETBACK_FT = 10.0

# Prop types (src/render/props.py) that carry a statutory setback of their own.
_SETBACK_BY_PROP = {
    "stop_sign": (STOP_SIGN_SETBACK_FT, "R.S. 39:4-138(h), 50 ft from a stop sign"),
    "fire_hydrant": (FIRE_HYDRANT_SETBACK_FT, "R.S. 39:4-138(i), 10 ft from a fire hydrant"),
}


# How far beyond the kerb a sign or hydrant can stand and still govern the parking beside
# it. These things sit ON the footway by definition, so the test cannot be "is it in the roadway".
FOOTWAY_REACH_FT = 15.0


@dataclass(frozen=True)
class NoParkingZone:
    """A stretch of one kerb where R.S. 39:4-138 forbids parking, and which clause says so.

    An interval, not a start station. (e) bounds the near end of the leg, but (h) and (i) are
    RADII around a point - a hydrant mid-block forbids parking for 10 ft either side of itself.

    `reason` is carried so the proposal can say why - "50 ft from the stop sign per 39:4-138(h)"
    can be checked against the statute.
    """
    start_ft: float
    end_ft: float
    reason: str

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)


def sideline_station_ft(leg_name: str, side: str, legs: dict, corner_fillets: dict) -> float:
    """Where the intersecting street's side line crosses this leg, as a station.

    The side line is the curb line; at a rounded corner this takes the corner fillet's tangent
    point - where this leg's own curb stops running straight and begins turning. That point lies
    at or beyond the true side line, so measuring from it is conservative.

    THE POINT ITSELF IS GEOMETRY and lives with src/geometry/model/corners.py:corner_tangent_station_ft.
    What is statutory is the READING of it: R.S. 39:4-138(e) measures from the side line.
    """
    return corner_tangent_station_ft(leg_name, side, legs, corner_fillets)


def _prop_station_ft(leg, side: str, position_ft, setback_ft: float) -> float | None:
    """A prop's station along this leg side, if it governs parking there at all.

    Three ways it does not: wrong side, too far laterally (past FOOTWAY_REACH_FT beyond the
    kerb), or too far along (its zone does not reach the leg).
    """
    stations, offsets = station_offset_many(leg.centerline, np.asarray([position_ft], dtype=float))
    station, offset = float(stations[0]), float(offsets[0])
    if (offset > 0) != (side == "left"):
        return None
    if abs(offset) > leg.curb_to_curb_ft / 2 + FOOTWAY_REACH_FT:
        return None
    if station < -setback_ft or station > leg.centerline.length + setback_ft:
        return None
    return station


def no_parking_zones_ft(state, leg_name: str, side: str, crosswalk_offsets: dict,
                         props: list[dict] | None = None) -> list[NoParkingZone]:
    """Every stretch of this kerb where R.S. 39:4-138 forbids parking, nearest first."""
    leg = state.legs[leg_name]
    # Local, for the usual cycle: src/geometry/treatments/ reads this module's statutory
    # figures. This module deliberately depends on nothing in src/render.
    from src.geometry.targets import LegSide
    from src.geometry.treatments import CURB_EXTENSION_DEVICES, ProtectDaylightZone

    device = state.treatment_for(ProtectDaylightZone, LegSide(leg_name, side))
    bulbout = device is not None and device.kind in CURB_EXTENSION_DEVICES
    crosswalk_setback = CROSSWALK_SETBACK_WITH_BULBOUT_FT if bulbout else CROSSWALK_SETBACK_FT
    sideline_setback = SIDELINE_SETBACK_WITH_BULBOUT_FT if bulbout else SIDELINE_SETBACK_FT
    zones = []

    # (e) - the junction end of the leg. Both arms are measured, and the further one wins.
    # Indexed rather than read as .offset_ft (src/render/crosswalks.py:CrosswalkOffset) so a
    # caller may hand this a plain (station, source) pair - the statutory rules below need the
    # station only, and this module deliberately depends on nothing in src/render.
    crossing = crosswalk_offsets.get(leg_name)
    crossing_ft = crossing[0] if crossing is not None else None
    sideline_ft = sideline_station_ft(leg_name, side, state.legs, state.corner_fillets)
    junction = [(sideline_ft + sideline_setback,
                 f"R.S. 39:4-138(e), {sideline_setback:.0f} ft from the side line of the "
                 f"intersecting street"
                 + (" (curb extension built)" if bulbout else ""))]
    if crossing_ft is not None:
        junction.append((crossing_ft + crosswalk_setback,
                          f"R.S. 39:4-138(e), {crosswalk_setback:.0f} ft from the crosswalk"
                          + (" (curb extension built)" if bulbout else "")))
    end_ft, reason = max(junction)
    zones.append(NoParkingZone(0.0, end_ft, reason))

    # What OSM says about this kerb, per stretch of it. A mapped prohibition is a no-parking
    # zone exactly like a statutory one - same shape of fact, same consequence.
    #
    # This is where a restriction covering only part of a leg finally lands - it has to be a span
    # and not a flag on the side.
    for restriction in getattr(state, "parking_restrictions", {}).get((leg_name, side), []):
        if not restriction.prohibits:
            continue
        zones.append(NoParkingZone(restriction.start_ft, restriction.end_ft,
                                    restriction.citation))

    # (e) again, at EVERY OTHER intersection this leg runs across - and BOTH ARMS of it.
    # THE SIDE LINE ARM is measured from the edge of the cross street's own carriageway.
    # THE CROSSWALK ARM is measured from a crosswalk that exists whether or not it is painted:
    # N.J.S.A. 39:1-1 defines one as "either marked or unmarked existing at each approach".
    # Both only on the side the street actually leaves on - a T-junction does not daylight the
    # kerb opposite it.
    for cross in getattr(state, "cross_streets", {}).get(leg_name, []):
        if side not in cross.sides:
            continue
        near_ft, far_ft = cross.mouth_ft
        zones.append(NoParkingZone(
            near_ft - sideline_setback, far_ft + sideline_setback,
            f"R.S. 39:4-138(e), {sideline_setback:.0f} ft from the side line of "
            f"{cross.citation}"))
        for crosswalk in getattr(cross, "crosswalks", ()):
            zones.append(NoParkingZone(
                crosswalk.station_ft - crosswalk_setback,
                crosswalk.station_ft + crosswalk_setback,
                f"R.S. 39:4-138(e), {crosswalk_setback:.0f} ft from {crosswalk.citation} at "
                f"{cross.citation}"
                + (" (curb extension built)" if bulbout else "")))

    # (h) and (i) - radii around a point feature, anywhere along the leg.
    for prop in props or []:
        setback = _SETBACK_BY_PROP.get(prop.get("type"))
        if setback is None or prop.get("position_ft") is None:
            continue
        distance_ft, citation = setback
        station = _prop_station_ft(leg, side, prop["position_ft"], distance_ft)
        if station is None:
            continue
        zones.append(NoParkingZone(station - distance_ft, station + distance_ft, citation))

    return sorted(zones, key=lambda zone: zone.start_ft)


def merged_no_parking_spans_ft(zones: list[NoParkingZone]) -> list[tuple[float, float]]:
    """The zones as disjoint spans, for PAINTING.

    A zone list is allowed to overlap - each entry is a separate statutory fact. Paint is not:
    coincident sets of strokes cause z-fighting in the render.
    """
    spans = []
    for zone in sorted(zones, key=lambda z: z.start_ft):
        if spans and zone.start_ft <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], zone.end_ft))
        else:
            spans.append((zone.start_ft, zone.end_ft))
    return [(a, b) for a, b in spans if b > a]


def parkable_runs_ft(state, leg_name: str, side: str, crosswalk_offsets: dict,
                      props: list[dict] | None = None,
                      physical_clearance_ft: float = 0.0,
                      min_run_ft: float = 0.0) -> list[tuple[float, float]]:
    """The (start, end) station spans of this kerb where a stall may legally be marked.

    Everything from the physical clearance point to the end of the leg, less every zone in
    no_parking_zones_ft. Runs shorter than min_run_ft are dropped - a 4 ft gap between two
    prohibitions is not a parking space, and marking one would be worse than leaving the
    kerb blank.
    """
    leg = state.legs[leg_name]
    cursor = max(physical_clearance_ft, 0.0)
    end_of_leg = leg.centerline.length
    runs = []
    for zone in no_parking_zones_ft(state, leg_name, side, crosswalk_offsets, props):
        if zone.start_ft > cursor:
            runs.append((cursor, min(zone.start_ft, end_of_leg)))
        cursor = max(cursor, zone.end_ft)
    if cursor < end_of_leg:
        runs.append((cursor, end_of_leg))
    return [(a, b) for a, b in runs if b - a >= min_run_ft and b > a]


def legal_parking_start_ft(state, leg_name: str, side: str, crosswalk_offsets: dict,
                            props: list[dict] | None = None,
                            physical_clearance_ft: float = 0.0,
                            min_run_ft: float = 0.0) -> float | None:
    """The first station where a stall may be marked, or None if the kerb has no room.

    A convenience over parkable_runs_ft for the common case of a leg with one usable run.
    """
    runs = parkable_runs_ft(state, leg_name, side, crosswalk_offsets, props,
                             physical_clearance_ft, min_run_ft)
    return runs[0][0] if runs else None
