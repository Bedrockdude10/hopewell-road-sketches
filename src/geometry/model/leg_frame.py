"""THE LEG FRAME: (station along the centreline, lateral offset from it).

Every marking in this project is positioned in this frame rather than in state-plane coordinates,
because a street bends and a stripe has to bend with it. So this module owns the Leg itself, the
conversion both ways (station_offset_many / point_at), and the two datums a position can be
measured against - the centreline, and the TRACED KERB (curb_offsets_at_stations).

Those two datums disagreeing is this project's most productive source of bugs: config says the
road is 68 ft wide, the traced kerbs say 44. Anything that asks WHETHER THERE IS ROOM must go
through narrowest_half_width_ft; anything that decides WHERE PAINT GOES measures from the
centreline. See docs/network-model.md, which proposes removing the contradiction entirely."""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from shapely.geometry import LineString, Point, Polygon



@dataclass(frozen=True)
class Alignment:
    """A centreline with a traced kerb on one or both sides - ALL this module reads.

    Every function below is annotated `leg: "Leg"` and not one of them touches a leg: they read
    `.centerline` and one of `.left_curb` / `.right_curb`, and nothing else. So the frame was
    never about legs, and saying so in a type is the first step of docs/network-model.md's step 4.
    A `Road` (src/geometry/network.py) carries the same three attributes and now goes through
    these functions unchanged, which is what makes moving the datum a change of caller rather
    than a rewrite of the frame.

    It also gives the one-sided case a name. Reading a single traced kerb against some
    centreline - a corridor's KerbRun, one piece of a chained road - was done by a private shim
    in network.py that duck-typed those attributes into existence. That shim was right about the
    contract and wrong about where it belongs: the contract is this module's, so the type is too.
    """
    centerline: LineString
    left_curb: LineString | None = None
    right_curb: LineString | None = None

    @classmethod
    def one_sided(cls, centerline: LineString, side: str, curb: LineString) -> "Alignment":
        """One kerb read against one centreline, on the named side."""
        return cls(centerline, **{f"{side}_curb": curb})


@dataclass
class Leg:
    """One approach to an intersection: a centerline plus (if known) a curb-to-curb
    width, from which parallel curb lines are derived automatically."""
    name: str
    centerline: LineString  # starts at the point nearest the intersection, extends outward
    curb_to_curb_ft: float | None = None
    left_curb: LineString | None = None
    right_curb: LineString | None = None
    # Sides whose curb line is the surveyor's traced kerb rather than a centerline offset.
    # The corner between two traced sides is traced too, so it is joined and smoothed
    # instead of being replaced by a fitted fillet.
    traced_sides: set = field(default_factory=set)
    # Tier of curb_to_curb_ft AS BUILT, when that is better than what the config claimed.
    # A width measured between two traced kerbs is osm_derived however the config describes
    # its own estimate, and the phase summary and the plan view's curb styling both have to
    # say which one they are showing - "ESTIMATE / PLACEHOLDER" over a line drawn from a
    # surveyor's trace is the project's own principle stated backwards.
    width_provenance: str | None = None

    def __post_init__(self):
        if self.curb_to_curb_ft is not None:
            half = self.curb_to_curb_ft / 2
            self.left_curb = self.centerline.offset_curve(half)
            self.right_curb = self.centerline.offset_curve(-half)


def unit_vector(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _leg_bearing(leg: "Leg") -> float:
    p0 = np.array(leg.centerline.coords[0])
    p1 = np.array(leg.centerline.coords[1])
    d = p1 - p0
    return np.arctan2(d[1], d[0])


def leg_bearing_deg(leg) -> float:
    """Compass bearing from the junction outward along a leg, from its chord.

    The chord and not the first segment: a bent leg's opening segment points somewhere the
    leg as a whole does not, which is the error that put a crosswalk 29.4 deg off square at
    Louellen (see crosswalk_axes).
    """
    (x0, y0), (x1, y1) = leg.centerline.coords[0], leg.centerline.coords[-1]
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360)


def leg_clearance_ft(leg_name: str, legs: dict, corner_fillets: dict, buffer_ft: float = 3.0,
                     side: str | None = None) -> float:
    """
    Distance from a leg's near point out past BOTH of its corner fillets'
    tangent points, plus a small buffer - the point beyond which the leg's
    curb lines run straight rather than curving through the corner. Use this
    to place crosswalks / raised crossings outside the curve, not inside it -
    a fixed small offset from the intersection center lands inside the curve
    for any leg wide enough or with a generous enough corner radius.

    `side` narrows it to the corners that constrain THAT KERB. A corner return belongs to one
    side of each leg it touches - build_corner_fillets pairs leg A's LEFT curb with leg B's
    RIGHT curb - so a per-leg maximum holds one kerb back for a curve that is on the other.
    At E Broad & Princeton the stem runs south, so both corners constrain the south curbs and
    the north side of E Broad has no return on it at all; the per-leg figure held its kerbside
    paint 28-32 ft out from a curve that is not there:

        e_broad_st_east  left  (north)   per-side  3.0 ft   per-leg 32.1 ft
        e_broad_st_east  right (south)   per-side 32.1 ft   per-leg 32.1 ft

    Without `side` the answer is the per-leg maximum, which is what a CROSSWALK wants - it
    spans kerb to kerb, so it has to clear the returns on both sides. Only paint that belongs
    to one kerb should ask per side.
    """
    # Project onto the centerline (not raw Euclidean distance from the near
    # point) - the tangent point lives on the CURB line, laterally offset from
    # the centerline by half the leg's width, so a plain .distance() call
    # conflates that lateral offset with the actual along-the-road distance,
    # wildly overshooting for wide legs (a 68 ft leg has a 34 ft half-width,
    # which alone would dominate the distance even with zero along-leg offset).
    centerline = legs[leg_name].centerline
    max_along_dist = 0.0
    for (leg_a, leg_b), pieces in corner_fillets.items():
        if "error" in pieces or pieces.get("through_street"):
            # A through-street join is not a corner return: the curb does not curve there, so
            # it constrains nothing. Its "tangent points" are just wherever the two curb lines
            # happen to start, which on a partially-traced side is far up the leg.
            continue
        # trimmed_a is leg_a's LEFT curb, trimmed_b is leg_b's RIGHT curb - see
        # build_corner_fillets. That is what makes a corner side-specific.
        if leg_a == leg_name and side in (None, "left"):
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_a"].coords[0])))
        if leg_b == leg_name and side in (None, "right"):
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_b"].coords[0])))
    return max_along_dist + buffer_ft


# How finely a curbside strip is sampled along the leg. The curb is a traced polyline whose
# vertices fall wherever the surveyor clicked, so the strip is resampled on a regular station
# grid instead: both of its boundaries then have matching vertices at matching stations, and
# the strip stays a strip rather than a wedge.
STRIP_SAMPLE_FT = 2.0

# The steepest lateral shift a marking will follow the kerb through: 1 ft across per 10 along.
#
# A kerb-referenced marking must not inherit every kink in the tracing, and the question of WHICH
# kinks is a rate, not a total. Measured over these three sites, the two kinds of kerb movement
# separate cleanly by rate and not at all by amount: the legs where the street genuinely bends run
# at 1:6 or gentler, and the two whose tracing takes in a corner flare or a parking apron kink at
# 1:2. Sizing the decision off the TOTAL swing instead made it depend on how long the leg was, so
# the same street followed its kerb at a 130 ft frame and abandoned it at a 325 ft one - see
# tests/test_two_way_bike_lane.py::test_kerb_follow_does_not_depend_on_the_frame.
#
# At 1:10 a marking gives up a mean 0.11-0.28 ft of the drift on the legs that drift, and refuses
# up to 12.2 ft of the kink on broad_st_east. Both figures hold at either frame scale, because a
# rate limit is a local property of the kerb and total swing is not.
MAX_KERB_FOLLOW_TAPER = 0.10


@lru_cache(maxsize=256)
def _tapered_curb_frame(centerline: LineString, curb: LineString, max_taper: float):
    """A kerb profile with its steep kinks flattened - see MAX_KERB_FOLLOW_TAPER.

    The largest profile lying at or inside the traced kerb whose lateral slope never exceeds
    `max_taper`, which is a cone erosion: at each station, the lowest any cone of that slope
    rising from another station's kerb offset reaches. So a drift gentler than the limit comes
    back unchanged, while a kink is cut back to the limit from both directions.

    Computed once over the WHOLE traced span, not over each caller's window, because otherwise
    two markings on one kerb get different profiles from asking about different stretches of it -
    and a bike lane's outer stripe disagreeing with its own surface by a few inches is exactly
    what MarkingsDoNotCollide reports.

    Inside the kerb, never outside: erosion can only reduce an offset, so a marking built off
    this cannot cross a kerb it did not already cross.
    """
    stations, offsets = _curb_in_leg_frame(centerline, curb)
    lo, hi = float(stations[0]), float(stations[-1])
    grid = np.linspace(lo, hi, max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2))
    sampled = np.abs(np.interp(grid, stations, offsets))
    eroded = (sampled[None, :] + max_taper * np.abs(grid[:, None] - grid[None, :])).min(axis=1)
    grid.flags.writeable = False
    eroded.flags.writeable = False
    return grid, eroded


@lru_cache(maxsize=256)
def _curb_in_leg_frame(centerline: LineString, curb: LineString):
    """A curb's own vertices as (stations, offsets) in the leg's frame, sorted by station.

    Cached on the two geometries, for the same reason and with the same safety as
    polyline_frame: this is a fact about a pair of immutable lines, and everything that
    measures against a real kerb needs it. Uncached it was recomputed from scratch on every
    query - once per bollard, once per parking-stall divider, once per zone end line - each
    time re-projecting every traced vertex (a traced kerb can carry 30+) to answer about one
    station.

    Sorted here rather than by each caller, because np.interp requires it and two callers
    were each doing their own argsort of the same array.
    """
    stations, offsets = station_offset_many(centerline, np.asarray(curb.coords, dtype=float))
    order = np.argsort(stations)
    stations, offsets = stations[order], offsets[order]
    stations.flags.writeable = False
    offsets.flags.writeable = False
    return stations, offsets


def _traced_curb_frame(leg: "Leg", side: str):
    """(stations, offsets) for a side's real curb, or None where that side has none."""
    curb = getattr(leg, f"{side}_curb", None)
    if curb is None or curb.is_empty:
        return None
    return _curb_in_leg_frame(leg.centerline, curb)


def curb_offsets_at_stations(leg: "Leg", side: str, stations: np.ndarray) -> np.ndarray | None:
    """Signed offsets of a side's real curb at the given CENTERLINE stations.

    The vectorized form of curb_point_at_station's interpolation - see that function for why
    a curb cannot be addressed by its own arc length.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    curb_stations, curb_offsets = frame
    return np.interp(stations, curb_stations, curb_offsets)


def tapered_curb_offsets(leg: "Leg", side: str, stations: np.ndarray,
                          max_taper: float = MAX_KERB_FOLLOW_TAPER) -> np.ndarray | None:
    """UNSIGNED offsets of a side's kerb with its steep kinks flattened, at the given stations.

    What a marking that FOLLOWS the kerb should follow. curb_offsets_at_stations answers where the
    kerb is, which is the right question for "is there room" and the wrong one for "where does the
    paint go": paint that tracked the tracing exactly inherited a 1:2 corner flare and read as
    snaking. See MAX_KERB_FOLLOW_TAPER for why the limit is a rate.
    """
    curb = getattr(leg, f"{side}_curb", None)
    if curb is None or curb.is_empty:
        return None
    grid, eroded = _tapered_curb_frame(leg.centerline, curb, max_taper)
    return np.interp(stations, grid, eroded)


def curb_station_span(leg: "Leg", side: str,
                       behind_ft: float = 0.0) -> tuple[float, float] | None:
    """The stations along the leg that this side's curb was actually traced across, clipped
    to the leg itself.

    A traced kerb runs on its own terms: it starts 13-47 ft out (the corner return is traced
    separately) and several of them carry on 11-78 ft PAST the end of the 130 ft leg, because
    the tracing continues down the block. Paint has to be built inside that span - outside it
    there is no curb to measure from, only extrapolation.

    `behind_ft` lets the span begin BEHIND the junction node, as far back as the kerb is really
    traced. Only a THROUGH-RUNNING kerb should ask: there the kerb does not stop at the node, it
    carries on as the far leg's kerb, and the two legs' paint has to overlap to fuse rather than
    stopping dead at station 0 on each side. At W Broad & Louellen that clamp left a 1.28 ft hole
    in a lane the corridor is supposed to run continuously through - each half ended at its own
    station 0, in its own frame, with different section widths either side. Still bounded by the
    tracing, so this never invents kerb; it only stops discarding what is there (that side is
    traced to station -3.0).
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    stations, _offsets = frame
    lo = max(float(stations[0]), -abs(behind_ft))      # sorted by _curb_in_leg_frame
    hi = min(float(stations[-1]), leg.centerline.length)
    return (lo, hi) if hi > lo else None


# How close to due north-south a leg must run before it has no meaningful compass side. sin(12
# deg): the perpendicular's northing component is then under a fifth of the leg's length, which
# is the point at which a slight survey lean could flip the answer.
NORTH_SOUTH_LEG_TOLERANCE = 0.2


def side_facing(leg: "Leg", compass: str) -> str:
    """Which of this leg's two sides ("left"/"right") faces `compass` ("north"/"south").

    A leg's left/right is in the LEG'S OWN frame - the sign of a lateral offset, outward along
    its bearing - so the same kerb of an east-west street is "left" on one approach and "right"
    on the other. Any decision about a real side of a real street therefore has to be
    translated per leg, and doing it by hand is how a corridor treatment ends up on the north
    kerb of one leg and the south kerb of the next.

    Taken from the centerline's own geometry rather than from config's bearing_deg: the bearing
    is the outward direction, but the centerline is what offsets are actually measured from, and
    on a leg with a kink the two differ. Measured at the leg's midpoint, where a lateral offset
    is least affected by either end.

    REFUSES ONLY A LEG RUNNING NEARLY DUE NORTH-SOUTH, which is the case that genuinely has no
    answer: its sides face east and west, and a compass side would come from whichever way its
    slight lean happened to fall.

    The test is on the PERPENDICULAR's northing component, not on whether the leg is "more
    east-west than north-south". Those are different questions and the second one is wrong: the
    left side is 90 deg anticlockwise of the heading, so its northing component is dx, and dx is
    a strong signal long before the leg is predominantly east-west. A |dx| < |dy| cut is a hard
    45 deg threshold, and it refused w_broad_st_southwest - a leg at 222.3 deg, 2.3 deg past the
    cut, whose south side is entirely unambiguous. That dropped the corridor bike lane from one
    of the two Broad St legs at Louellen and left the treatment stopping inside the junction.
    """
    if compass not in ("north", "south"):
        raise ValueError(f"side_facing takes 'north' or 'south', not {compass!r}")
    line = leg.centerline
    ahead = line.interpolate(min(line.length * 0.55, line.length))
    behind = line.interpolate(max(line.length * 0.45, 0.0))
    dx, dy = ahead.x - behind.x, ahead.y - behind.y
    length = float(np.hypot(dx, dy)) or 1.0
    if abs(dx) / length < NORTH_SOUTH_LEG_TOLERANCE:
        raise ValueError(
            f"Leg {leg.name!r} runs within "
            f"{float(np.degrees(np.arcsin(NORTH_SOUTH_LEG_TOLERANCE))):.0f} deg of due north-south "
            f"(its midpoint heading moves {dx:+.1f} ft east for {dy:+.1f} ft north), so its sides "
            f"face east and west and it has no 'north side' to speak of.")
    # The left side is the +offset side, 90 degrees anticlockwise of the heading: (-dy, dx). Its
    # northing component is dx, so the left side faces north exactly when the leg heads east.
    left_faces = "north" if dx > 0 else "south"
    return "left" if left_faces == compass else "right"


def narrowest_half_width_ft(leg: "Leg", side: str, from_ft: float = 0.0,
                             to_ft: float | None = None) -> float:
    """The least room this side has between the centerline and the real kerb, over a span.

    The bound a cross-section has to fit if it is to be promised for the whole of a leg rather
    than at one station. The nominal half-width is a summary and is routinely the wrong number
    for this: broad_st_east's nominal half is 26.0 ft, and its kerbs come within 22.8 ft of the
    alignment somewhere along the traced run. A 48 ft parking-protected section sized off the
    nominal figure would be drawn 5.2 ft over the kerb at that point.

    Falls back to the nominal half-width where the side has no traced kerb, since then there is
    no measurement to prefer.
    """
    span = curb_station_span(leg, side)
    if span is not None:
        lo = max(span[0], from_ft)
        hi = min(span[1], leg.centerline.length if to_ft is None else to_ft)
        if hi - lo >= STRIP_SAMPLE_FT:
            offsets = curb_offsets_at_stations(
                leg, side, np.linspace(lo, hi, max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)))
            return float(np.abs(offsets).min())
    return _nominal_half_ft(leg)


def _nominal_half_ft(alignment) -> float:
    """The DECLARED half-width, consulted ONLY where nothing is traced to measure instead.

    Reached by getattr because it is the one thing an Alignment deliberately does not carry. A
    declared width and a traced kerb that disagree is this project's most productive source of
    bugs - five in one session, all of the form "config says 68 ft, the kerb says 44" - and
    docs/network-model.md step 5 deletes the declared figure outright, replacing it with a
    measurement that holds over the station range it was actually taken at. Until then a Leg
    still carries one and a Road never does, so this reads whichever is in front of it and
    answers 0 ft - no promised room - where there is no evidence either way.

    It is also consulted LAST now rather than computed first. Evaluating the nominal on every
    call, including the ones about to measure a real kerb and throw it away, is how a figure
    nothing should depend on stays wired into every code path that touches the frame.
    """
    nominal = getattr(alignment, "curb_to_curb_ft", None)
    return nominal / 2 if nominal is not None else 0.0


def paint_stations(leg: "Leg", side: str, start_ft: float,
                    end_ft: float | None = None,
                    beyond_the_tracing: bool = False) -> np.ndarray | None:
    """The station grid every marking on one side of a leg is sampled on.

    Extracted because four functions here had this same eight lines inlined, and a marking whose
    band was built on one grid while its own edge stripes were built on another is a marking whose
    pieces do not line up - which is the failure inset_line_ft and offset_band_polygon already
    warn about in their own docstrings. One grid, one definition.

    Bounded by the stations where this side's kerb is traced, because that is the only stretch
    where a lateral offset can be checked against anything. None where there is no such span or
    it is shorter than a single sample.

    A NEGATIVE start_ft is a request to begin behind the junction node, and it is honoured only
    as far as the kerb is actually traced there - see curb_station_span's behind_ft. Callers that
    start at 0 or beyond are unaffected, which is all of them except a through-running kerb.

    `beyond_the_tracing` LIFTS THAT BOUND, and almost nothing may ask for it. The bound is right
    for a marking that is a DESIGN CHOICE - a stall, a bike lane, a buffer - because drawing one
    where the kerb is unmapped is proposing something on ground nobody has measured. It is wrong
    for a marking that is a STATEMENT OF LAW. R.S. 39:4-138 forbids parking within 25 ft of a
    crosswalk whether or not a surveyor traced the kerb there, and a daylight zone that stops
    where the tracing starts is not a shorter zone, it is the same zone drawn incompletely -
    which is worse than not drawing it, because a gap in hatching reads as permission.

    Measured at W Broad & Louellen, whose south kerb is traced only from station 60.3: the
    statutory zone runs 0-93.3 ft and the hatching was drawn 60.3-93.3, stopping 7.5 ft short of
    a crosswalk it exists to daylight. 92% of the zone was hatched and the missing 8% was the
    part beside the crossing.

    Outside the tracing the kerb is held at the offset it has where the tracing begins, which is
    what curb_offsets_at_stations already returns there (np.interp clamps) and what
    curb_edge_by_station already builds its end vertex from. So this invents no geometry that the
    rest of this module was not already using - it stops discarding it. The assumption is still an
    assumption, and it is the reason this is opt-in and named rather than being the default.
    """
    span = curb_station_span(leg, side, behind_ft=max(-start_ft, 0.0))
    if span is None:
        return None
    lo, hi = span
    if beyond_the_tracing:
        # The LEG's own extent, not the tracing's. Still bounded - a marking off the end of the
        # leg belongs to the next block - and still requiring the kerb to be traced SOMEWHERE,
        # since with no tracing at all there is no offset to hold.
        lo = max(start_ft, -abs(min(lo, 0.0)))
        hi = min(leg.centerline.length if end_ft is None else end_ft, leg.centerline.length)
    else:
        lo = max(lo, start_ft)
        hi = min(hi, leg.centerline.length if end_ft is None else end_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return None
    return np.linspace(lo, hi, max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2))


def kerb_inset_offsets(leg: "Leg", side: str, stations, inset_ft: float,
                        keep_inside_ft: float = 0.0,
                        floor_ft: float = 0.0) -> np.ndarray | None:
    """Offsets that sit `inset_ft` in from the TRACED KERB at each station - a line that follows
    the kerb rather than the alignment, through tapered_curb_offsets so it follows the street
    bending without following a kink in the tracing.

    THE DIFFERENCE THIS EXISTS FOR. A centreline offset is a constant; the kerb is not. On
    w_broad_st_northeast's south-east side the kerb runs 25.13 ft out at the junction throat and
    17.24 ft out 39 ft later - a real, mapped 8 ft convergence where two streets of different
    widths meet. A bike lane drawn at a constant 16.44 ft from the centreline held straight
    through all of it and left a hatched wedge widening from 0.87 ft to 8.68 ft against the kerb.
    In the render the lane reads as swerving away from the kerb it is supposed to be protected by.

    So a marking that BELONGS TO THE KERB - the outer edge of a kerbside bike lane, the parking
    lane against it - is measured from the kerb, and the variable part of the section is absorbed
    where a designer would put it: the buffer between the lane and the travel way. What must NOT
    be measured from here is anything that belongs to the travel lane, or the travel lane inherits
    the kerb's wobble and stops holding its target width.

    `floor_ft` IS WHAT KEEPS THIS ONE-DIRECTIONAL, and it matters more than it looks. A section is
    sized at the leg's NARROWEST traced point, so that position is the one place it is known to
    fit; following the kerb is only ever about the stations where the street has MORE room. Pass
    the section's own alignment-referenced offset here and the marking moves outward with the kerb
    and never inward past its design. Without it, a station where the sampled kerb comes inside the
    narrowest figure compresses the whole section - on broad_st_east's right kerb the lane's outer
    stripe and its buffer stripe converged and ran along each other for 2.3 ft, which
    MarkingsDoNotCollide reported.

    Returns absolute distances, unsigned - see line_from_offsets for the side convention.
    """
    curb_offsets = tapered_curb_offsets(leg, side, stations)
    if curb_offsets is None:
        return None
    return np.maximum(curb_offsets - inset_ft - keep_inside_ft, floor_ft)


def _advancing(centerline: LineString, points) -> np.ndarray:
    """Mask of placed vertices that ADVANCE along the centreline, dropping any that double back.

    A lateral offset is placed perpendicular to the centreline at each station, and near a vertex
    the perpendiculars on either side converge - so past a certain offset consecutive placements
    come out in the wrong order and the polyline reverses. tests/test_frame_properties.py already
    established this as a property of the frame rather than a defect: around a kink there is a band
    of stations no offset can reach.

    It only bites at LARGE offsets, which is why it surfaced when the bike lane's edges moved onto
    the kerb. broad_st_east's centreline kinks 4.5 deg at station 43.2 and that edge sits 28 ft out;
    one vertex, asked for station 42.5, landed back at 38.96. The line stays SIMPLE - it does not
    cross itself - so nothing caught it geometrically, but clipping it by station then produced two
    fragments whose station ranges overlap, and MarkingsDoNotCollide reported 2.3 ft of one line
    painted twice. It was right: a stripe that runs forward, back, then forward again is drawn over
    itself on the ground.

    Dropping the offending vertex loses nothing real - the band it sits in is unreachable, so there
    is no correct position for it - and leaves the stripe monotonic, which is what every
    station-based clip downstream assumes.
    """
    stations, _offsets = station_offset_many(centerline, np.asarray(points, dtype=float))
    keep = np.ones(len(stations), dtype=bool)
    highest = -np.inf
    for i, station in enumerate(stations):
        if station <= highest:
            keep[i] = False
        else:
            highest = station
    return keep


def line_from_offsets(leg: "Leg", side: str, stations, offsets_ft) -> LineString | None:
    """A polyline through `offsets_ft` (absolute, unsigned) at `stations` on one side."""
    sign = 1.0 if side == "left" else -1.0
    pts = place_in_measured_frame(leg.centerline, stations, sign * np.asarray(offsets_ft))
    pts = np.asarray(pts, dtype=float)[_advancing(leg.centerline, pts)]
    return LineString(pts) if len(pts) >= 2 else None


def band_from_offsets(leg: "Leg", side: str, stations, inner_ft, outer_ft) -> Polygon | None:
    """The strip between two per-station offset arrays on one side.

    The shared tail of offset_band_polygon and its kerb-referenced sibling: once both boundaries
    are arrays on the same grid, how they were derived stops mattering, which is what lets a band
    have one centreline-referenced edge and one kerb-referenced edge - exactly what the buffer
    beside a kerb-hugging bike lane is.
    """
    sign = 1.0 if side == "left" else -1.0
    inner_pts = np.asarray(place_in_measured_frame(leg.centerline, stations,
                                                   sign * np.asarray(inner_ft)), dtype=float)
    outer_pts = np.asarray(place_in_measured_frame(leg.centerline, stations,
                                                   sign * np.asarray(outer_ft)), dtype=float)
    # ONE mask for both edges - see _advancing. Filtering them independently would drop a station
    # from the outer edge and keep it on the inner, which shears the band rather than shortening it:
    # the two boundaries are paired by station and a polygon built from unpaired lists closes across
    # the gap. The outer edge doubles back first (it is further out), so requiring BOTH to advance is
    # what keeps the pairing.
    keep = _advancing(leg.centerline, inner_pts) & _advancing(leg.centerline, outer_pts)
    inner_pts, outer_pts = inner_pts[keep], outer_pts[keep]
    if len(inner_pts) < 2:
        return None
    band = Polygon(list(inner_pts) + list(reversed(list(outer_pts))))
    if not band.is_valid:
        band = band.buffer(0)
    return band if not band.is_empty and band.area > 0 else None


def kerb_parallel_line_ft(leg: "Leg", side: str, inset_ft: float, start_ft: float,
                           end_ft: float | None = None,
                           keep_inside_ft: float = 0.0,
                           floor_ft: float = 0.0) -> LineString | None:
    """A stripe that runs `inset_ft` in from the traced kerb - see kerb_inset_offsets."""
    stations = paint_stations(leg, side, start_ft, end_ft)
    if stations is None:
        return None
    offsets = kerb_inset_offsets(leg, side, stations, inset_ft, keep_inside_ft, floor_ft)
    if offsets is None:
        return None
    return line_from_offsets(leg, side, stations, offsets)


def kerb_referenced_band_polygon(leg: "Leg", side: str, outer_inset_ft: float, width_ft: float,
                                  start_ft: float, end_ft: float | None = None,
                                  keep_inside_ft: float = 0.0,
                                  floor_ft: float = 0.0) -> Polygon | None:
    """A band of constant `width_ft` whose OUTER edge sits `outer_inset_ft` in from the kerb.

    A kerbside bike lane's own asphalt: constant width, following the kerb. Its inner edge moves
    with the kerb too, which is the point - the buffer inside it then absorbs whatever the street
    happens to do, and the travel lane keeps its target.
    """
    stations = paint_stations(leg, side, start_ft, end_ft)
    if stations is None:
        return None
    outer = kerb_inset_offsets(leg, side, stations, outer_inset_ft, keep_inside_ft, floor_ft)
    if outer is None:
        return None
    return band_from_offsets(leg, side, stations, np.maximum(outer - width_ft, 0.0), outer)


def inset_line_ft(leg: "Leg", side: str, offset_ft: float,
                   start_ft: float, end_ft: float | None = None,
                   keep_inside_ft: float = 0.0,
                   beyond_the_tracing: bool = False) -> LineString | None:
    """A line offset_ft from the centerline on one side, over the stations where that side's
    curb exists - the inner boundary of curbside_strip_polygon, drawn on its own.

    Built on the same station grid as the strip so the two cannot disagree, and clamped
    inside the real curb for the same reason. NOT `offset_curve(...).interpolate(d)`: an
    offset curve's arc length differs from the centerline's, so `d` there is not station `d`,
    which is what let the parking stall ticks drift along the leg.

    keep_inside_ft is how far short of the kerb the line must stop when it gets clamped
    there - half the painted stripe's width, so the stripe sits inside the road instead of
    straddling the kerb. Clamping the AXIS to the kerb hung half the paint over it wherever
    the road was narrower than the offset asked for.

    beyond_the_tracing is paint_stations'; it is passed through so a statutory zone's edge line
    reaches exactly as far as the zone it bounds. A fill with no line along its first 34 ft is
    the same disagreement between a marking's pieces this function's docstring is about.
    """
    stations = paint_stations(leg, side, start_ft, end_ft, beyond_the_tracing)
    if stations is None:
        return None
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    sign = 1.0 if side == "left" else -1.0
    room = np.maximum(np.abs(curb_offsets) - keep_inside_ft, 0.0)
    inner = sign * np.minimum(offset_ft, room)
    # Same measured-frame placement curbside_strip_polygon uses, and for the same reason - this
    # line is the inner boundary of that strip and the two must not disagree about where it is.
    return LineString(_place_no_further_in_than(leg.centerline, stations, inner))


def offset_band_polygon(leg: "Leg", side: str, inner_offset_ft: float, outer_offset_ft: float,
                         start_ft: float, end_ft: float | None = None,
                         keep_inside_ft: float = 0.0,
                         beyond_the_tracing: bool = False) -> Polygon | None:
    """The strip of roadway between TWO lateral offsets from the centerline, on one side.

    For a marking whose own two boundaries are what define it - a bike lane's green surface
    sits between the lane's two edge stripes and is exactly as wide as the lane. Built on the
    same station grid and with the same kerb clamping inset_line_ft uses, so the band and the
    two stripes drawn at its edges cannot disagree about where those edges are.

    NOT the difference of two curbside_strip_polygons, which is what this replaced and which
    was subtly wrong: both of those are bounded by the stations where the TRACED KERB exists,
    so wherever the kerb is unmapped the inner strip still reached the nominal half-width while
    the outer one contributed nothing to subtract, and the leftover spilled past the marking's
    own outer edge - 6.6 ft past it on broad_st_west's right bike lane, onto asphalt that is not
    the lane. Nothing reported it, because the ground it spilled onto has no other paint on it
    to collide with and no traced kerb to be outside of.

    `beyond_the_tracing` lifts the bound at the tracing, exactly as it does for inset_line_ft and
    for the same short list of callers - a band that states a FACT about the street rather than
    proposing a marking on it. A kerb opening is one: where a vehicle crosses the kerb does not
    stop being true where nobody traced the kerb, and an opening built only over the traced
    stretch is a cut narrower than the marking it has to cut. On W Broad & Louellen's south kerb,
    traced only from station 60.3, the junction's own 0-68 ft mouth came out 7.7 ft long and left
    the daylight hatching it was supposed to remove in fragments inside the intersection.

    Returns None where there is no room or no span to draw over, like its siblings.
    """
    stations = paint_stations(leg, side, start_ft, end_ft, beyond_the_tracing=beyond_the_tracing)
    if stations is None:
        return None
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    room = (np.maximum(np.abs(curb_offsets) - keep_inside_ft, 0.0) if curb_offsets is not None
            else np.full(stations.shape, abs(leg.curb_to_curb_ft) / 2 - keep_inside_ft))
    return band_from_offsets(leg, side, stations, np.minimum(inner_offset_ft, room),
                              np.minimum(outer_offset_ft, room))



def curb_point_at_station(leg: "Leg", side: str, station_ft: float) -> np.ndarray | None:
    """The point on a leg's real curb at `station_ft` ALONG THE CENTERLINE.

    Not `curb.interpolate(station_ft)`. That measures distance along the curb line from the
    curb line's own start, which coincides with the centerline station only while the curb
    is a symmetric offset of the centerline starting at the junction. Since the curbs became
    traced kerbs neither holds - they start 14-47 ft out and run at their own bearing - and
    asking for station 40 landed anywhere from 51 to 86 ft down the leg.
    """
    offsets = curb_offsets_at_stations(leg, side, np.asarray([station_ft], dtype=float))
    if offsets is None:
        return None
    return np.asarray(point_at(leg.centerline, station_ft, float(offsets[0])), dtype=float)


def inset_point_at_station(leg: "Leg", station_ft: float, offset_ft: float) -> np.ndarray:
    """A point offset laterally from the centerline at a given station - exactly, via the
    leg frame, rather than by interpolating along an offset_curve whose own arc length
    differs from the centerline's."""
    return np.asarray(point_at(leg.centerline, station_ft, offset_ft), dtype=float)


def points_at_offset_ft(leg: "Leg", side: str, offset_ft: float, start_ft: float,
                         end_ft: float | None = None, spacing_ft: float = 10.0) -> list[tuple]:
    """Points at a FIXED lateral offset from the centerline, spaced along the leg.

    For anything standing in a strip whose position is measured from the centerline rather than
    from the kerb - a delineator in a bike lane's buffer, say. bollard_points_ft centres its
    points between the kerb and a lane edge, which is the right rule for a kerbside buffer and
    the wrong one for a buffer sitting between two lanes: it would drift outward with the kerb
    instead of holding the line the paint holds.

    Clipped to the stations this side's kerb actually covers, so nothing is placed where there is
    no measured roadway, and never outside the kerb itself.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return []
    lo = max(span[0], start_ft)
    hi = min(span[1], leg.centerline.length if end_ft is None else end_ft)
    if hi <= lo or spacing_ft <= 0:
        return []
    n = int((hi - lo) // spacing_ft) + 1
    stations = lo + np.arange(n) * spacing_ft
    curb_offsets = np.abs(curb_offsets_at_stations(leg, side, stations))
    sign = 1.0 if side == "left" else -1.0
    lateral = np.minimum(offset_ft, curb_offsets)
    return [tuple(point_at(leg.centerline, float(s), sign * float(o)))
            for s, o in zip(stations, lateral)]


# How many corrective passes place_in_measured_frame takes. Two is enough at every leg here -
# the residual falls from 0.59 ft to under a thousandth - and a cap means a pathological frame
# ends the loop rather than spinning in it.
#
# Raising it is not the way to chase a stubborn residual, which is worth stating because it
# looks like the obvious lever and it is not: each pass re-asks at a corrected (station,
# offset) and keeps whichever attempt has the smallest COMBINED station-and-offset error, so
# another pass can legitimately trade station accuracy for offset accuracy and land somewhere
# else entirely. Going 2 -> 3 moved enough geometry to fail 18 tests across four junctions
# while still not fixing the fold that prompted it. Where one line must not drift a particular
# way, bias that line - see inset_line_ft.
_FRAME_CORRECTION_PASSES = 2


def place_in_measured_frame(centerline: LineString, stations: np.ndarray,
                             offsets: np.ndarray) -> list[tuple]:
    """World points that MEASURE BACK as (station, offset), not merely that were built from it.

    point_at and station_offset_many are inverses along a straight centerline and drift apart
    near a kink, because they resolve the ambiguity differently: point_at extrapolates the frame
    of the segment the station falls on, while station_offset_many assigns a point to whichever
    segment it is perpendicular-nearest to. In the wedge outside a bend those are different
    segments, and the further from the centerline the wider the gap.

    It matters here and not for a traced kerb because a traced kerb's stations were DERIVED by
    station_offset_many from surveyed points, so they agree with it by construction. An extension
    imposes stations instead, at ±19 ft of offset, and broad_st_east's centerline kinks 4.5 deg
    43.1 ft out where NJDOT rounds the corner: a vertex placed at station 44.0 read back at
    41.59, and across the taper's 0.2 ft-per-ft slope that is 0.59 ft of offset - enough to put
    the kerbside hatching built against this kerb 0.6 ft over it, which check_paint_inside_the_curb
    duly caught.

    So the placement is corrected against the measuring frame rather than trusted: place, measure,
    move by the residual. Everything downstream - the paint, the crossing reach, the invariants -
    measures with station_offset_many, so that is the frame the geometry has to be right in.

    Corrected per point and only where it HELPS. The two frames do not merely drift: at an offset
    larger than the bend's radius of curvature the offset curve FOLDS, and inside the fold the
    station order reverses - on broad_st_east's right kerb, 19 ft in from the bend at station
    43.1, asking for station 42 lands at 44.35 and asking for 44 lands at 41.59. A correction step
    across that discontinuity overshoots instead of converging, so each point keeps whichever
    estimate measures closest to what was asked and the fold is left alone rather than chased.
    """
    target_s = np.asarray(stations, dtype=float)
    target_o = np.asarray(offsets, dtype=float)
    ask_s, ask_o = target_s.copy(), target_o.copy()
    best = np.array([point_at(centerline, float(s), float(o)) for s, o in zip(ask_s, ask_o)])
    got_s, got_o = station_offset_many(centerline, best)
    best_error = np.hypot(got_s - target_s, got_o - target_o)
    for _ in range(_FRAME_CORRECTION_PASSES):
        ask_s = ask_s + (target_s - got_s)
        ask_o = ask_o + (target_o - got_o)
        trial = np.array([point_at(centerline, float(s), float(o))
                          for s, o in zip(ask_s, ask_o)])
        got_s, got_o = station_offset_many(centerline, trial)
        error = np.hypot(got_s - target_s, got_o - target_o)
        better = error < best_error
        best[better], best_error[better] = trial[better], error[better]
    return [tuple(p) for p in best]


# How many times the outward bias below re-asks. One pass closes most folds; broad_st_west's
# is deep enough at a 2.0x frame that a single correction still left 0.076 ft of the parking
# edge line inside the travel lane, and the residual only showed at that ONE frame scale -
# 1.0, 2.2, 2.5 and 3.0 were all clean. Each pass shrinks what is left, and the loop stops as
# soon as nothing is short, so this costs nothing where there is no fold.
_OUTWARD_BIAS_PASSES = 4


def _place_no_further_in_than(centerline: LineString, stations: np.ndarray,
                               offsets: np.ndarray) -> list[tuple]:
    """place_in_measured_frame, biased so no point lands INSIDE the offset it was asked for.

    A kerbside marking's two possible placement errors are not equivalent. This offset is the
    edge of a travel lane: landing a hair wide of it costs a hair of kerbside treatment, and
    landing a hair narrow puts paint in the lane, which is what PaintStaysOutOfTheTravelLane
    exists to catch. Inside a frame fold - broad_st_east's 7.2 degree kink 43 ft out - the
    placement settles 0.05 ft short, and 0.05 ft short is a reported violation.

    Only the points that fell short move. Re-placing the whole line instead shifts every OTHER
    point too, because place_in_measured_frame searches from the ask and a changed ask
    anywhere reshuffles the lot: that moved enough geometry across all four junctions to fail
    18 tests, in service of 0.05 ft on one vertex of one leg.

    Used by curbside_strip_polygon AND inset_line_ft, which is not optional - the line IS the
    strip's inner boundary, and biasing one without the other breaks the property inset_line_ft
    exists to hold. Biased on its own it put the rim of a hatched zone 1.5 ft alongside the
    edge line it continues, far enough off to stop reading as the same stroke and near enough
    for MarkingsDoNotCollide to call it two.
    """
    offsets = np.asarray(offsets, dtype=float)
    placed = np.asarray(place_in_measured_frame(centerline, stations, offsets), dtype=float)
    ask = offsets.copy()
    for _ in range(_OUTWARD_BIAS_PASSES):
        _stations, got = station_offset_many(centerline, placed)
        short = np.maximum(np.abs(offsets) - np.abs(got), 0.0)
        if not short.any():
            break
        # Eased into the neighbouring vertices at half height rather than applied to the short
        # one alone. A single vertex pushed out of line with the two either side of it is a
        # kink, and a kink in a line that is then clipped around crossings and driveways comes
        # back as overlapping fragments: a 1.5 ft offcut of w_broad_st_southwest's buffer edge
        # line lying on top of the 125 ft one it was cut from, which MarkingsDoNotCollide reads
        # - correctly - as two lines painted down the same stretch of road.
        padded = np.pad(short, 1, mode="edge")
        short = np.maximum(short, 0.5 * np.maximum(padded[:-2], padded[2:]))
        ask = ask + np.sign(offsets) * short
        nudged = np.asarray(place_in_measured_frame(centerline, stations, ask), dtype=float)
        moved = short > 0
        placed[moved] = nudged[moved]
    return [tuple(p) for p in placed]


def curb_edge_by_station(leg: "Leg", side: str, lo_ft: float, hi_ft: float) -> list[tuple] | None:
    """The kerb's OWN world coordinates between two stations, with exact ends.

    For the outer boundary of anything that runs along a kerb. Resampling the kerb onto a station
    grid and re-placing it with point_at was near enough while every kerb offset changed slowly,
    and stopped being so once a curb extension's taper made one change at 0.2 ft per ft: the
    placement drifts from the frame the checks measure in, and inside a fold (see
    place_in_measured_frame) it cannot be corrected at all.

    Taking the kerb's real coordinates sidesteps the frame entirely. Nothing is interpolated
    except the two end vertices, which are held at exactly lo_ft and hi_ft so the strip's two
    boundaries still start and finish at the same stations - the property the resampling existed
    to guarantee, and the one that keeps a strip a strip rather than a wedge.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    curb_stations, curb_offsets = frame
    coords = np.asarray(getattr(leg, f"{side}_curb").coords, dtype=float)
    order = np.argsort(np.asarray(_traced_curb_station_order(leg, side)))
    inside = [tuple(coords[order[i]]) for i in range(len(order))
              if lo_ft < curb_stations[i] < hi_ft]
    ends = place_in_measured_frame(leg.centerline, np.array([lo_ft, hi_ft]),
                                    np.interp([lo_ft, hi_ft], curb_stations, curb_offsets))
    return [ends[0], *inside, ends[1]]


def _traced_curb_station_order(leg: "Leg", side: str) -> np.ndarray:
    """The kerb's vertex indices in station order - the same sort _curb_in_leg_frame applies."""
    curb = getattr(leg, f"{side}_curb")
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    return np.argsort(stations)


def vertex_tangents(line: LineString) -> np.ndarray:
    """Unit direction of a polyline at each of its own vertices.

    Averages the segments either side of a vertex (one-sided at the ends), so a vertex on a
    curve gets the curve's local heading rather than one arbitrary neighbouring segment's.
    """
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return np.zeros_like(coords)
    steps = np.diff(coords, axis=0)
    tangents = np.zeros_like(coords)
    tangents[:-1] += steps
    tangents[1:] += steps
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)


def line_direction(line: LineString) -> np.ndarray:
    coords = np.asarray(line.coords)
    vec = coords[-1] - coords[0]
    norm = np.hypot(*vec)
    return vec / norm if norm else np.array([1.0, 0.0])


@lru_cache(maxsize=512)
def polyline_frame(centerline: LineString):
    """(vertices, unit segment directions, segment lengths, station at each vertex).

    The one description of a leg's frame. Both directions of the transform read it, so
    station_offset(point_at(...)) round-trips exactly - it did not when the forward
    direction used segment tangents and the inverse estimated one from a +/-2 ft window.

    Cached on the centerline itself. Every point this project places goes through the frame -
    a scenario resolves it ~6,000 times per site - and a leg centerline is a 2-3 vertex line,
    so rebuilding the four arrays each time cost more than the projection it exists to serve.
    Shapely geometries hash by value and are immutable, so the key is exactly the input: a leg
    whose centerline is replaced (which is how the width fit re-centres one) gets a new entry
    rather than a stale frame.

    The returned arrays are shared, so callers must not write to them. Nothing here does -
    every consumer indexes or does arithmetic producing new arrays - and marking them
    read-only is what keeps that true rather than conventional.
    """
    verts = np.asarray(centerline.coords, dtype=float)
    seg_vec = verts[1:] - verts[:-1]
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    seg_dir = seg_vec / np.where(seg_len > 0, seg_len, 1.0)[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    for array in (verts, seg_dir, seg_len, cumulative):
        array.flags.writeable = False
    return verts, seg_dir, seg_len, cumulative


def frame_at(centerline: LineString, station: float) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit tangent) of the leg frame at `station`, extrapolating past either end."""
    verts, seg_dir, _seg_len, cumulative = polyline_frame(centerline)
    i = int(np.clip(np.searchsorted(cumulative, station, side="right") - 1, 0, len(seg_dir) - 1))
    return verts[i] + seg_dir[i] * (station - cumulative[i]), seg_dir[i]


def station_offset(centerline: LineString, xy) -> tuple[float, float]:
    """A point in the leg's frame: distance along the centerline, and signed distance from
    it - positive to the left, matching Leg.left_curb / right_curb.

    The station is signed. LineString.project() clamps to [0, length], so everything behind
    the junction comes back as station 0 with a small offset - which let a leg claim the
    curb of the leg OPPOSITE it and draw it straight back through the intersection. Behind
    the junction the station is measured against the leg's own starting tangent instead, so
    it comes out negative and those points are rejected.

    One point through the vectorized path, so there is only ever one definition of the frame.
    """
    stations, offsets = station_offset_many(centerline, np.asarray([xy], dtype=float))
    return float(stations[0]), float(offsets[0])


def point_at(centerline: LineString, station: float, offset: float) -> tuple[float, float]:
    origin, tangent = frame_at(centerline, station)
    return tuple(origin + np.array([-tangent[1], tangent[0]]) * offset)


def station_offset_many(centerline: LineString, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """station_offset() for many points at once: (stations, offsets) arrays.

    Same frame and the same signed-station convention as the scalar version, but the whole
    (points x centerline segments) projection is one numpy expression instead of two shapely
    calls per point. Centerlines carry a handful of vertices, so the matrix is small and
    this collapses the dominant cost of reading a junction's traced kerbs.
    """
    verts, seg_dir, seg_len, cumulative = polyline_frame(centerline)
    seg_start = verts[:-1]

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    rel = pts[:, None, :] - seg_start[None, :, :]             # (p, s, 2)
    along = np.einsum("psc,sc->ps", rel, seg_dir)
    clamped = np.clip(along, 0.0, seg_len[None, :])
    perp = rel - clamped[:, :, None] * seg_dir[None, :, :]
    nearest = np.argmin(np.hypot(perp[:, :, 0], perp[:, :, 1]), axis=1)

    rows = np.arange(len(pts))
    stations = cumulative[nearest] + clamped[rows, nearest]
    tangents = seg_dir[nearest]
    rel_nearest = rel[rows, nearest]
    offsets = tangents[:, 0] * rel_nearest[:, 1] - tangents[:, 1] * rel_nearest[:, 0]

    # Past either end, measure against that end's tangent rather than letting the projection
    # clamp. Behind the junction this is what keeps a station negative, so a leg can't claim
    # the curb of the leg opposite it (see station_offset). Past the far end it stops every
    # point beyond the leg's working length from collapsing onto the same station, and makes
    # this an exact inverse of point_at over the whole line.
    for outside, vertex, direction, base in (
            (stations <= 0, verts[0], seg_dir[0], 0.0),
            (stations >= cumulative[-1], verts[-1], seg_dir[-1], cumulative[-1])):
        if outside.any():
            rel_end = pts[outside] - vertex
            stations[outside] = base + rel_end @ direction
            offsets[outside] = direction[0] * rel_end[:, 1] - direction[1] * rel_end[:, 0]
    return stations, offsets
