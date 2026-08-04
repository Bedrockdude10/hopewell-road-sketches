"""Where a car may legally park near these intersections, and marking the rest as clear.

"Daylighting" is keeping the approach to a crossing free of parked vehicles so that a driver
and a person crossing can see each other. New Jersey has no daylighting statute by that
name - it does it through the ordinary parking prohibitions in R.S. 39:4-138, which already
forbid parking within 25 ft of a crosswalk. The gap between the law and the street is that
the setback is usually unmarked, so it gets parked in anyway. Marking it is the treatment.

WHAT THE LAW SAYS (R.S. 39:4-138, as amended by P.L. 2009 c.257). Parking is prohibited:

  (a)  within an intersection
  (e)  within 25 ft of the nearest crosswalk or SIDE LINE of a street or intersecting
       highway - reduced to 10 ft where a curb extension or bulbout has been built
  (h)  within 50 ft of a "stop" sign
  (i)  within 10 ft of a fire hydrant

All four are floors, and the binding one is whichever sits furthest from the junction. Two
of them were missing here: only the crosswalk arm of (e) was applied, so the side line was
ignored on legs with no marked crossing, and (h) and (i) were not applied at all even though
this repo already knows where every stop sign and hydrant is from OSM.

WHAT A MUNICIPALITY MAY CHANGE (R.S. 39:4-138.6). Hopewell Borough may set its own
permissible distances by ordinance, but may not permit parking within 25 ft of a crosswalk
or side line, nor within 50 ft of a stop sign in a school zone while school is in session.
The Borough's traffic chapter is not machine-readable here (ecode360 refuses automated
requests), so these are the STATE figures - which is the correct default in the absence of a
local ordinance, and for (e) is a hard floor a local ordinance cannot lower anyway. If the
Borough has adopted something stricter, raise the numbers below; nothing here can be lowered
without checking 39:4-138.6 first.

None of this is a rendering preference. A proposal that marks a stall inside one of these
distances is proposing something illegal, which is why check_parking_is_legal treats it as a
scene invariant rather than a note.
"""
from dataclasses import dataclass

import numpy as np

from src.geometry.model import station_offset_many

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
# constructed". The reduction applies to both arms, so a curb extension has to cut the side
# line setback too - applying it to the crosswalk alone leaves the side line binding at 25 ft
# and the extension buys nothing, which is not what the statute says.
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
# it. These things sit ON the footway by definition - a hydrant 2 ft behind the kerb still
# forbids parking at that kerb - so the test cannot be "is it in the roadway".
FOOTWAY_REACH_FT = 15.0


@dataclass(frozen=True)
class NoParkingZone:
    """A stretch of one kerb where R.S. 39:4-138 forbids parking, and which clause says so.

    An interval, not a start station. That distinction is the whole correction here: the
    junction setbacks in (e) bound the near end of the leg, but (h) and (i) are RADII around
    a point - a hydrant mid-block forbids parking for 10 ft either side of itself and leaves
    the kerb beyond it parkable. Modelling them as "parking starts after this" pushed the
    entire parking lane off a 130 ft leg because of a hydrant on the next block.

    `reason` is carried so the proposal can say why. "Parking starts at 61 ft" is
    unreviewable; "50 ft from the stop sign per 39:4-138(h)" can be checked against the
    statute by someone who is not reading this code.
    """
    start_ft: float
    end_ft: float
    reason: str

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)


def sideline_station_ft(leg_name: str, side: str, legs: dict, corner_fillets: dict) -> float:
    """Where the intersecting street's side line crosses this leg, as a station.

    The side line of the intersecting highway is that street's curb line. At a rounded
    corner there is no single crossing point, so this takes the corner fillet's tangent
    point - where this leg's own curb stops running straight and begins turning into the
    corner. That point lies at or beyond the true side line (the fillet bulges outward from
    the corner the two straight curb lines would otherwise form), so measuring from it is
    conservative: it never places the legal setback closer to the junction than the statute
    does.
    """
    leg = legs.get(leg_name)
    if leg is None:
        return 0.0
    station = 0.0
    for (leg_a, leg_b), pieces in corner_fillets.items():
        if "error" in pieces:
            continue
        # build_corner_fillets pairs leg_a's LEFT curb with leg_b's RIGHT curb.
        if leg_a == leg_name and side == "left":
            tangent = pieces["trimmed_a"].coords[0]
        elif leg_b == leg_name and side == "right":
            tangent = pieces["trimmed_b"].coords[0]
        else:
            continue
        stations, _offsets = station_offset_many(leg.centerline,
                                                  np.asarray([tangent], dtype=float))
        station = max(station, float(stations[0]))
    return station


def _prop_station_ft(leg, side: str, position_ft, setback_ft: float) -> float | None:
    """A prop's station along this leg side, if it governs parking there at all.

    Three ways it does not. Wrong side: a hydrant on the north kerb says nothing about the
    south one. Too far laterally: a sign more than a footway's reach beyond the kerb belongs
    to the cross street or to a property, not to this kerb. Too far along: the check that
    was missing, and the one that mattered - a hydrant 209 ft past the end of a 130 ft leg
    was taken as governing it, which pushed every stall off the leg. Its zone has to
    actually reach the leg, hence the setback in the bound.
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
    from src.geometry.treatments import CURB_EXTENSION_DEVICES

    device = getattr(state, "daylight_devices", {}).get((leg_name, side), {})
    bulbout = device.get("kind") in CURB_EXTENSION_DEVICES
    crosswalk_setback = CROSSWALK_SETBACK_WITH_BULBOUT_FT if bulbout else CROSSWALK_SETBACK_FT
    sideline_setback = SIDELINE_SETBACK_WITH_BULBOUT_FT if bulbout else SIDELINE_SETBACK_FT
    zones = []

    # (e) - the junction end of the leg. Both arms are measured, and the further one wins:
    # the crosswalk governs where one is marked, the side line governs where none is.
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

    A zone list is allowed to overlap - each entry is a separate statutory fact, and the
    reporting and the invariant both want them individually. Paint is not: at broad_st_west
    the hydrant's zone (18.9-38.9 ft) sits entirely inside the junction's (0-45.7 ft), and
    hatching both painted 98 sq ft of the kerb twice. Two coincident sets of strokes is a
    collision - z-fighting in the render, double ink on the plan.
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
