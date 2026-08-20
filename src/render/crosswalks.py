"""Match OSM-surveyed pedestrian crossings to intersection legs, and resolve
each leg's crosswalk offset: the real surveyed position when a crossing was
matched, else a geometric estimate (needed for hypothetical/proposed
crossings that don't exist yet). See README.md "Crosswalk styles: real data
over guessing"."""
import math
from typing import NamedTuple

import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.geometry import LineString, Polygon

from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.geometry.model import (crosswalk_estimate_ft, inset_line_ft, leg_clearance_ft,
                                station_offset_many)
from src.geometry.targets import Everywhere, LegSide, LegTarget, Side
from src.geometry.treatments import (AddBikeLane, DesignState, LaneNarrowing, MarkedParking,
                                     ShiftCrosswalk, UpgradeCrosswalkMarkings,
                                     divider_shift_toward_ft)

# OSM crossing:markings values -> our 3 rendered styles. "lines" (two simple
# transverse boundary lines) is the least visible; FHWA/NACTO guidance treats
# continental and ladder as visibility upgrades over it - unmapped/missing
# values default to "lines" since that's the least assumption-laden guess.
OSM_MARKINGS_TO_STYLE = {
    "lines": "lines",
    "zebra": "continental",
    "ladder": "ladder",
}

# Margin beyond a leg's resolved crosswalk offset that a lane-narrowing taper must stay clear of.
# Deliberately more than half CROSSWALK_DEPTH_FT: a curved taper's closest approach to the
# intersection is not exactly at its target_ft endpoint, so this errs generous rather than
# tracking the depth. Single home for both views - export.py (3D) and plan_view.py (2D) compute
# the taper from this one constant so they cannot drift apart.
CROSSWALK_CLEARANCE_FT = 5.0

# Typical MUTCD gap between a stop line and the near edge of the crosswalk ahead of it.
STOP_BAR_TO_CROSSWALK_GAP_FT = 4.0

# Depth of a painted crosswalk - the gap between the two transverse lines, i.e. its dimension
# along the direction of travel. 6 ft is Mercer County's recommended width for a transverse
# crosswalk (Danny, 2026-08-02).
#
# Authoritative in FEET because the source standard is in feet; the metres value is derived for
# the renderer. Single home for both views: plan_view.py draws it directly, and export.py writes
# it into the geometry JSON as `crosswalk_depth_m` for scripts/blender/blender_crosswalks.py,
# which cannot import from src/ (it runs under Blender's bundled Python). That JSON hand-off is
# what keeps the 2D reconstruction and the 3D render from drifting apart.
CROSSWALK_DEPTH_FT = 6.0
CROSSWALK_DEPTH_M = CROSSWALK_DEPTH_FT * FT_TO_M

# Distance back toward the intersection from a leg's resolved crosswalk offset (which is the
# crosswalk's CENTRE) to its stop bar: half the depth reaches the crosswalk's near edge, then the
# MUTCD gap. Derived from the depth rather than hardcoded, so narrowing the crosswalk moves the
# stop bar with it. An approximation - no site here is surveyed down to exact striping.
STOP_BAR_SETBACK_FT = CROSSWALK_DEPTH_FT / 2 + STOP_BAR_TO_CROSSWALK_GAP_FT

# A stop bar spans only the ENTERING half of the roadway - a driver stops in their own
# lanes, never across the opposing ones - unlike a crosswalk, which spans curb to curb.
# This clearance keeps it off the centerline and the curb at each end. Single home for both
# views, like CROSSWALK_DEPTH_M: plan_view.py draws the 2D bar from it, and export.py writes it
# into the geometry JSON as `stop_bar_curb_clearance_m` for blender_crosswalks.py:add_stop_bar.
STOP_BAR_CURB_CLEARANCE_M = 0.5

# Plan/check depth of a stop bar. The painted bar is ~1.5 ft deep; this is only used to
# give it an area to test and draw, never to size the 3D stripe.
STOP_BAR_PLAN_DEPTH_FT = 1.5


def stop_bar_band_geometry_ft(width_ft: float, edge_is_kerb: bool = True,
                               inner_ft: float = 0.0,
                               skew_deg: float = 0.0) -> tuple[float, float]:
    """(span_ft, lateral_offset_ft) for a stop bar on a roadway `width_ft` wide.

    The bar spans the entering half, from the road centerline out to the edge of the lane the
    stopped vehicle is in, toward the leg's own 'left' side (see
    blender_crosswalks.add_stop_bar for why that is the entering driver's side under
    right-hand traffic).

    It STARTS AT THE CENTERLINE, with no clearance there: MUTCD's stop line runs across the
    approach lanes and at the centerline it meets the centerline. A bar standing off it leaves a
    gap with nothing on the other side, which reads as a striping error. Only the far end is held
    back, and only when that end is the KERB, because paint does not run into the gutter. Where a
    treatment has narrowed the lane the far end is a painted edge line instead, and a bar stopping
    short of its own edge line is the same error - so `edge_is_kerb` is False there.
    """
    clearance_ft = STOP_BAR_CURB_CLEARANCE_M / FT_TO_M
    outer_ft = width_ft / 2 - (clearance_ft if edge_is_kerb else 0.0)
    # IT STARTS AT THE PAINTED CENTRELINE, WHICH IS NOT ALWAYS THE ALIGNMENT. `inner_ft` is where
    # that line actually is - zero while the two travel lanes straddle the alignment, and the
    # divider's own offset where a two-way bike lane has shifted them (see
    # treatments.divider_shift_toward_ft). Starting at the alignment regardless paints the stop
    # line across the opposing lanes.
    span_ft = max(outer_ft - inner_ft, clearance_ft)
    # AND THE OFFSET IS IN THE ROTATED FRAME, because both of the bar's ends are. Every builder
    # that lays this bar - crosswalk_band_ft here, blender_crosswalks.add_stop_bar there - grows
    # the SPAN by 1/cos(skew) so a rotated bar still reaches the lane edge, and neither grows the
    # offset. So the near end lands at `offset - span/(2 cos)`, which is only `inner_ft` if the
    # offset took the same stretch: unstretched it sits (span/2)(1 - 1/cos) short of the line the
    # bar rests against, and on louellen_st_west's -43.19 deg crossing that is 2.04 ft of stop bar
    # painted across the opposing lanes. Returned stretched, once, rather than by each caller -
    # the plan view stretched it and the export did not, which is the whole disagreement.
    stretch = 1.0 / max(math.cos(math.radians(abs(skew_deg))), 0.2)
    return span_ft, (inner_ft + span_ft / 2) * stretch


def travel_lane_edge_ft(state: DesignState, leg_name: str, side) -> float | None:
    """How far from the centerline the MOTOR travel lane reaches on `side`, or None for the
    full curb-to-curb half where no treatment has narrowed it.

    One rule with two callers asking the same question - where the ground a car drives on ends.
    A stop bar must not run across a bike lane, a painted buffer or a parking lane, and a person
    crossing is not in front of motor traffic while they are in one either. `side` is a parameter
    because src/metrics.py needs both; entering_lane_width_ft is the left-hand case.

    THE ORDER IS A DECISION. Three treatments can each narrow this kerb:

      1. a BIKE LANE, which answers with the travel lane's own edge;
      2. a LANE NARROWING on this side, whose buffer is spare asphalt the bar stops at;
      3. MARKED PARKING, whose stalls plus kerb buffer the bar stops at.

    The bike lane wins because it is the most specific claim about the same ground: it names the
    travel lane's edge directly, where the other two name what lies outside it. No site here
    applies two to one kerb, so the order states what should happen if one ever does.
    (test_a_stop_bar_stops_where_the_bike_lane_starts pins the bike-lane case.)

    THE SECTION IS ASKED OF THE DESIGN, not read off the treatment. `treatment.lane` is the
    section as REQUESTED - nominal, measured from the alignment; `treatment.section(state)` is the
    section as RESOLVED against the street it is painted on. For a two-way lane those differ,
    because the travel lanes have been shifted off the alignment to make room for it, and reading
    the requested one runs the stop bar past the travel lane's end and into the bike lane.
    """
    side = Side(side)
    kerb = LegSide(leg_name, side)
    half_ft = state.legs[leg_name].curb_to_curb_ft / 2
    bike_lane = state.treatment_for(AddBikeLane, kerb)
    if bike_lane is not None:
        return bike_lane.section(state).offsets_from_centerline_ft()["travel_lane_edge_ft"]
    narrowing = state.treatment_for(LaneNarrowing, kerb.leg_target)
    if narrowing is not None and side in narrowing.sides:
        return half_ft - narrowing.stripe_width_ft
    parking = state.treatment_for(MarkedParking, kerb)
    if parking is not None:
        return half_ft - parking.curb_offset_ft - parking.depth_ft
    return None


def entering_lane_width_ft(state: DesignState, leg_name: str) -> float | None:
    """The entering driver's own side - LEFT under right-hand traffic, the same swap
    src/render/props.py:APPROACHING_DRIVER_RIGHT documents. See travel_lane_edge_ft."""
    return travel_lane_edge_ft(state, leg_name, Side.LEFT)


def stop_bar_width_ft(state: DesignState, leg_name: str) -> float:
    """Full roadway width the stop bar is sized against (twice the entering lane width
    where a treatment narrowed it, else the leg's own curb-to-curb width)."""
    entering_ft = entering_lane_width_ft(state, leg_name)
    return 2 * entering_ft if entering_ft is not None else state.legs[leg_name].curb_to_curb_ft


# Minimum angle between a crossing way and a leg centerline for the crossing to be
# credited to that leg. True crossings and false matches split sharply: 30 deg sits
# well clear of both clusters and keeps the one real-but-skewed case.
MIN_CROSSING_ANGLE_DEG = 30.0

# Past this, a surveyed crossing's angle is worth REPORTING but not discarding. Squareness
# is our assumption; the traced line is the survey, and where the two disagree at a skewed
# junction it is the assumption that is wrong. Drawn as surveyed, said out loud so a real
# tracing error still surfaces.
REPORT_CROSSING_SKEW_DEG = 20.0


def _crossing_angle_deg(crossing_line: LineString, centerline: LineString) -> float:
    """Angle in [0, 90] between a crossing way and a leg centerline, using each one's
    overall start-to-end direction. 90 means the crossing runs square across the leg,
    0 means it runs straight along it."""
    (cx0, cy0), (cx1, cy1) = crossing_line.coords[0], crossing_line.coords[-1]
    (lx0, ly0), (lx1, ly1) = centerline.coords[0], centerline.coords[-1]
    crossing_dir = math.atan2(cy1 - cy0, cx1 - cx0)
    leg_dir = math.atan2(ly1 - ly0, lx1 - lx0)
    # A line has no direction, so fold into [0, 180) then into [0, 90].
    diff = abs(math.degrees(crossing_dir - leg_dir)) % 180
    return 180 - diff if diff > 90 else diff


def _crossing_skew_deg(crossing_line: LineString, centerline: LineString,
                        station_ft: float | None = None) -> float:
    """Signed angle, in (-90, 90], from square-across-the-leg to the crossing's real
    direction. 0 means the surveyed crossing runs exactly perpendicular to the leg
    centerline; positive is counter-clockwise.

    "Square" is measured against the leg AT `station_ft`, because that is the frame
    crosswalk_axes applies the answer in. Measuring against the whole-leg chord makes
    the two disagree by however much the centerline bends between them. The skew is
    carried precisely so the marking lines up with the surveyed one.

    Real crosswalks are not always square to the road they cross - they are painted to
    line up with the curb ramps and the sidewalks either side, which at a skewed
    junction can be several degrees off. Reconstructed perpendicular to the road
    centerline they draw the band at a visible angle to the surveyed crossing line it
    came from. Carried through so both the plan view and the Blender render orient the
    crosswalk the way it was actually surveyed.
    """
    (cx0, cy0), (cx1, cy1) = crossing_line.coords[0], crossing_line.coords[-1]
    crossing_dir = math.atan2(cy1 - cy0, cx1 - cx0)
    if station_ft is None:
        (lx0, ly0), (lx1, ly1) = centerline.coords[0], centerline.coords[-1]
        leg_dir = math.atan2(ly1 - ly0, lx1 - lx0)
    else:
        from src.geometry.model import frame_at

        _origin, tangent = frame_at(centerline, station_ft)
        leg_dir = math.atan2(tangent[1], tangent[0])
    square_dir = leg_dir + math.pi / 2  # perpendicular to the leg where the crossing sits
    # A line has no direction, so fold the difference into (-90, 90].
    diff = (math.degrees(crossing_dir - square_dir) + 90) % 180 - 90
    return diff


# One match per (legs, crossings). The matcher is asked the same question three times per
# scenario - resolve_crosswalk_offsets, resolve_crosswalk_skews and match_crossing_lines_to_legs
# each start by calling it - and it projects every crossing way and scores it against every leg.
_CROSSING_MATCHES: dict[tuple, tuple] = {}
_CROSSING_MATCH_LIMIT = 64


def _match_key(legs: dict):
    """Exactly what the matcher reads off each leg: its name, its centerline and its width.

    Spelled out rather than keyed on Leg identity, because a Leg is mutable (curb lines and
    traced_sides are assigned in place during fitting) and a cache keyed on mutable objects
    is a trap. Centerlines are shapely geometries, which hash by value.
    """
    return tuple((name, leg.centerline, leg.curb_to_curb_ft) for name, leg in sorted(legs.items()))


def _match_crossings_to_legs(legs: dict, crossings: list[dict]) -> dict:
    # `crossings` by identity: it is the list src/sources/osm_context.py's layer cache returns,
    # so it is the same object for the same (centre, radius) until the snapshot is re-pulled -
    # and hashing every crossing dict to key on the contents would cost more than the match.
    key = _match_key(legs)
    cached = _CROSSING_MATCHES.get(key)
    if cached is not None and cached[0] is crossings:
        return cached[1]
    match = _matched_crossings(legs, crossings)
    if len(_CROSSING_MATCHES) >= _CROSSING_MATCH_LIMIT:
        _CROSSING_MATCHES.clear()   # a batch build walks many scenarios; don't grow forever
    _CROSSING_MATCHES[key] = (crossings, match)
    return match


# How far down a leg a mapped crossing may sit and still be THIS junction's crossing.
# 80 ft, same as KERB_NEAR_JUNCTION_FT: a feature belonging to this junction is close to it.
# The lateral and orientation guards below are unchanged; this is the missing third one.
CROSSING_NEAR_JUNCTION_FT = 80.0


def _matched_crossings(legs: dict, crossings: list[dict]) -> dict:
    """
    Match each OSM-mapped crossing way to whichever leg it actually crosses -
    real surveyed geometry beats a geometric estimate of where a crosswalk
    probably sits. A crossing is assigned to the leg whose centerline it's
    closest to (perpendicular distance), as long as its midpoint projects onto
    that leg between the intersection and its far end, it isn't absurdly far
    off to the side (i.e. it's actually this leg's crossing, not some other
    nearby crossing that happened to fall within the fetch radius), and it runs
    ACROSS that leg rather than alongside it.

    Two constraints stop one crossing being credited to the wrong leg:

    * Orientation. A crossing spans the road it crosses, so it has to be roughly
      perpendicular to that leg's centerline. Without this check, a crossing over
      the side street also reads as a plausible candidate for the main street: it
      sits near the main street's centerline and runs parallel to it. At E Broad &
      Princeton, the Princeton crossing (OSM way 376498301) was matched to
      e_broad_st_east at an offset of 0.8 ft - the dead centre of the junction.
    * One crossing, one leg. Assignment was greedy over legs but didn't track which
      crossings had already been used, so a single OSM way could be credited to two
      different legs at two different offsets - as that same way was.

    Returns {leg_name: (offset_ft, style, skew_deg)} for legs with a matched real
    crossing. `skew_deg` is how far the surveyed crossing is rotated off square to the
    leg - see _crossing_skew_deg.
    """
    candidates = []
    for index, crossing in enumerate(crossings):
        xs, ys = wgs84_to_state_plane.transform(
            [c[0] for c in crossing["coords_wgs84"]], [c[1] for c in crossing["coords_wgs84"]]
        )
        line = LineString(zip(xs, ys))
        mid = line.interpolate(0.5, normalized=True)
        style = OSM_MARKINGS_TO_STYLE.get(crossing["tags"].get("crossing:markings"), "lines")
        for leg_name, leg in legs.items():
            centerline = leg.centerline
            along = centerline.project(mid)
            if not (0 < along < centerline.length) or along > CROSSING_NEAR_JUNCTION_FT:
                continue
            perp = centerline.interpolate(along).distance(mid)
            if perp > leg.curb_to_curb_ft / 2 + 10:  # not plausibly this leg's crossing
                continue
            if _crossing_angle_deg(line, centerline) < MIN_CROSSING_ANGLE_DEG:
                continue  # runs alongside this leg, not across it
            skew = _crossing_skew_deg(line, centerline, along)
            candidates.append((perp, leg_name, along, style, skew, index, line, crossing.get("tags", {})))

    best_by_leg: dict[str, tuple] = {}  # leg_name -> (best_perp, along, style, skew, line, tags)
    claimed_crossings: set[int] = set()
    for perp, leg_name, along, style, skew, index, line, tags in sorted(candidates, key=lambda c: c[0]):
        if leg_name in best_by_leg or index in claimed_crossings:
            continue
        best_by_leg[leg_name] = (perp, along, style, skew, line, tags)
        claimed_crossings.add(index)
    return {leg_name: (along, style, skew, line, tags) for leg_name, (_, along, style, skew, line, tags)
            in best_by_leg.items()}


def match_crossing_lines_to_legs(legs: dict, crossings: list[dict]) -> dict:
    """{leg_name: (crossing_line_ft, tags)} for legs with a matched surveyed crossing.

    Public view of the same matching the crosswalk geometry uses, so anything else keyed
    off a crossing - the kerbside hardware in src/render/props.py, say - associates it to
    exactly the same leg rather than re-deriving the association slightly differently.
    """
    return {leg_name: (line, tags) for leg_name, (_along, _style, _skew, line, tags)
            in _match_crossings_to_legs(legs, crossings).items()}


class CrosswalkOffset(NamedTuple):
    """Where a leg's crossing sits, and how well that position is sourced.

    A NamedTuple rather than a bare (float, str): this travels through the paint builder,
    the props, the daylighting rules, the invariants and the export, and every one of them
    reached into it as `offsets[leg][0]`. Half of those were placing statutory setbacks and
    kerbside paint, where "the first element of a tuple" is not what the reader needs to know
    about the number. Still a tuple, so unpacking (`offset_ft, source = ...`) reads the same.

    `source` is one of "osm_survey" (a real surveyed crossing way was matched to this leg) or
    "geometric_estimate" (nothing surveyed here - see crosswalk_estimate_ft), either of which
    may carry a "+scenario_shift(...)" suffix from shift_crosswalk_offset. It is written into
    the exported JSON, so a render always says which of the two it is showing.
    """
    offset_ft: float
    source: str

    @property
    def is_surveyed(self) -> bool:
        return self.source.startswith("osm_survey")


def resolve_crosswalk_offsets(state: DesignState, crossings: list[dict]) -> dict[str, CrosswalkOffset]:
    """{leg_name: CrosswalkOffset} - real OSM survey position if matched, else
    the geometric past-the-curve estimate (needed for hypothetical/proposed
    crossings). A scenario's shift_crosswalk_offset() override (if any) is
    applied on top and noted in the source string, rather than silently
    replacing the real/estimated base value."""
    matched = _match_crossings_to_legs(state.legs, crossings)
    out = {}
    for leg_name in state.legs:
        if leg_name in matched:
            offset_ft, source = matched[leg_name][0], "osm_survey"
        else:
            offset_ft = crosswalk_estimate_ft(leg_name, state.legs)
            source = "geometric_estimate"
        # Every ShiftCrosswalk on this leg, summed. This one ACCUMULATES rather than being
        # last-applied-wins, which is why it asks for all of them and not treatment_for: two
        # shifts of the same leg are two decisions to move the crossing, and the dict this
        # replaced added them up for the same reason.
        delta_ft = sum(shift.delta_ft for shift in
                       state.every_treatment(ShiftCrosswalk, LegTarget(leg_name)))
        if delta_ft:
            offset_ft += delta_ft
            source += f"+scenario_shift({delta_ft:+g}ft)"
        out[leg_name] = CrosswalkOffset(offset_ft, source)
    return out


def resolve_crosswalk_style(state: DesignState, leg_name: str) -> str:
    """Which of the three rendered marking styles this leg's crossing is painted in.

    An UpgradeCrosswalkMarkings treatment if the design has one, else "lines" - the two simple
    transverse boundary lines, which is what every real crossing at these four junctions is
    tagged `crossing:markings=lines` in OSM and what an unmapped one is assumed to be. That
    fallback is the least assumption-laden guess and not a claim about the survey; a matched
    crossing's own tag is read in _match_crossings_to_legs (OSM_MARKINGS_TO_STYLE) but has never
    been wired through to here, which is a gap worth closing separately rather than in passing.

    Here rather than in export.py so the rule has one home, the same reason
    entering_lane_width_ft does.

    THE LEG FIRST, THEN THE FRAME. A per-leg UpgradeCrosswalkMarkings is a decision about this
    approach and outranks a frame-wide one; a frame-wide one (targets.Everywhere) is the policy
    every proposal here starts from, and it is what all_crosswalks_continental now applies.
    Checking only the leg is what left the crossings at unmodelled junctions unstyled - see
    crossing_style_in, which asks the same two questions for a crossing that has no leg at all.
    """
    treatment = state.treatment_for(UpgradeCrosswalkMarkings, LegTarget(leg_name))
    if treatment is None:
        treatment = state.treatment_for(UpgradeCrosswalkMarkings, Everywhere())
    return treatment.style if treatment is not None else "lines"


def resolve_crosswalk_skews(state: DesignState, crossings: list[dict]) -> dict[str, float]:
    """{leg_name: skew_deg} - how far each leg's crosswalk is rotated off square to the
    leg, taken from the surveyed crossing way (see _crossing_skew_deg).

    Only legs with a real matched crossing get an entry. A leg falling back to the
    geometric estimate has no surveyed orientation to copy, so it stays square to the
    road - inventing a skew for it would be a guess dressed up as survey data.

    A surveyed skew is USED, however large. This used to discard anything past 20 deg on the
    grounds that real crossings are square, which threw away the one crossing at W Broad &
    Louellen (-44 deg) and drew a square crosswalk in its place - at the one junction in the
    set whose legs meet at 48 deg, where a crossing joining the actual kerb ramps has to run
    diagonally. The squareness rule was fitted to three ordinary junctions and then applied
    to the one it does not describe. Where our assumption and the surveyor's line disagree,
    the line wins; see REPORT_CROSSING_SKEW_DEG.
    """
    matched = _match_crossings_to_legs(state.legs, crossings)
    skews = {}
    for leg_name, (_along, _style, skew, _line, _tags) in matched.items():
        if abs(skew) > REPORT_CROSSING_SKEW_DEG:
            print(f"  NOTE: {leg_name}'s crossing is drawn {skew:+.1f} deg off square, as surveyed. "
                  f"Its span is {1 / math.cos(math.radians(abs(skew))):.2f}x the roadway width "
                  f"because of it. Worth an eye on the traced way if that looks wrong.")
        skews[leg_name] = skew
    return skews


# A stop bar crosses its approach, so it should sit square-ish to the leg, and it belongs
# to the leg it is painted on rather than the one it happens to point at. Same shape of test
# the crossing matcher uses, and the same threshold: a stop bar is painted parallel to the
# crossing ahead of it, and a tolerance that fits one fits the other.
STOP_LINE_MIN_ANGLE_DEG = MIN_CROSSING_ANGLE_DEG
STOP_LINE_MAX_OFFSET_FT = 40.0   # lateral distance from the leg centerline to the bar's midpoint
STOP_LINE_MAX_ALONG_FT = 120.0   # beyond this it belongs to a neighbouring junction


def match_stop_lines_to_legs(legs: dict, stop_lines: list[dict]) -> dict:
    """{leg_name: LineString} for legs with a surveyed stop bar.

    A bar is credited to the leg whose centerline it crosses most nearly square, ahead of
    the junction and within a plausible distance. One bar per leg: where several qualify the
    nearest to the junction wins, which is the one governing this approach.
    """
    lines_ft = []
    for entry in stop_lines:
        coords = entry["coords_wgs84"]
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        lines_ft.append(LineString(zip(xs, ys)))

    out = {}
    for name, leg in legs.items():
        best = None
        for line in lines_ft:
            mid = line.interpolate(0.5, normalized=True)
            along = leg.centerline.project(mid)
            if not 0 < along <= STOP_LINE_MAX_ALONG_FT:
                continue
            if leg.centerline.distance(mid) > STOP_LINE_MAX_OFFSET_FT:
                continue
            if _crossing_angle_deg(line, leg.centerline) < STOP_LINE_MIN_ANGLE_DEG:
                continue
            if best is None or along < best[0]:
                best = (along, line)
        if best is not None:
            out[name] = best[1]
    return out


def resolve_stop_bar_offsets(state: DesignState, crosswalk_offsets: dict[str, tuple[float, str]],
                              stop_lines: list[dict] | None = None) -> dict[str, float]:
    """{leg_name: offset_ft} - where a signalized approach's stop bar sits.

    Prefers the SURVEYED position: a road_marking=stop_line way is the painted bar itself, so
    its distance along the leg is the answer, not something to infer. Ten are mapped across
    this project's three signalized junctions.

    Falls back to the previous derivation - the leg's resolved crosswalk offset minus
    STOP_BAR_SETBACK_FT - for any approach nobody has traced. Either way the result is
    clamped to leg_clearance_ft() so a short leg or tight corner never pushes the bar back
    into the curb return.
    """
    surveyed = match_stop_lines_to_legs(state.legs, stop_lines or [])
    out = {}
    for leg_name, (crosswalk_offset_ft, _source) in crosswalk_offsets.items():
        min_offset_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        line = surveyed.get(leg_name)
        if line is None:
            # Derived: clamp, because nothing here knows where the bar really is and the
            # corner return is the one place it certainly isn't.
            out[leg_name] = max(crosswalk_offset_ft - STOP_BAR_SETBACK_FT, min_offset_ft)
            continue

        # Surveyed: use it as painted. Clamping a real position to our own corner clearance
        # is the modelled geometry overruling the street, and it fired on 3 of 9 traced bars
        # - two of them at W Broad, whose corners fall back to fitted fillets and whose
        # throat is therefore known to be too big. A bar inside our corner return is
        # evidence about OUR geometry, so it is reported rather than quietly moved.
        offset_ft = state.legs[leg_name].centerline.project(line.interpolate(0.5, normalized=True))
        if offset_ft < min_offset_ft:
            print(f"  SOURCE CONFLICT: {leg_name}'s surveyed stop bar sits {offset_ft:.1f} ft out, "
                  f"inside this junction's modelled corner return ({min_offset_ft:.1f} ft). Drawing it "
                  f"where it is painted - the modelled corner is the thing more likely to be wrong.")
        out[leg_name] = offset_ft
    return out


# Clearance past the crosswalk where a leg has no stop bar to end the centerline at. The
# painted centerline stops before the crossing, it doesn't run into it.
CENTERLINE_CROSSWALK_GAP_FT = 2 / FT_TO_M   # the historical 2 m, kept as-is


def centerline_start_ft(crosswalk_offset_ft: float, stop_bar_offset_ft: float | None,
                        crosswalk_is_marked: bool = True) -> float:
    """How far out along a leg its centerline paint begins.

    A centerline terminates AT the stop bar - it does not continue into the junction past
    the line drivers stop on.

    max() rather than just the bar, so a leg whose surveyed bar sits unusually close in
    still doesn't get centerline paint laid across its crosswalk - but ONLY where there is a
    crosswalk. An unmarked leg still gets a crosswalk_offset_ft, because every leg needs a
    resolved station for a crossing a proposal might add; using that station to hold the
    centerline back behind a crossing that is not painted keeps clear of nothing. Where
    nothing is painted there is nothing to keep clear of.
    """
    past_crosswalk_ft = crosswalk_offset_ft + CENTERLINE_CROSSWALK_GAP_FT
    if stop_bar_offset_ft is None:
        return past_crosswalk_ft
    if not crosswalk_is_marked:
        return stop_bar_offset_ft
    return max(stop_bar_offset_ft, past_crosswalk_ft)


# MUTCD/AASHTO proportions for a no-passing centerline: two ~6 in stripes about 4 in apart.
DOUBLE_YELLOW_GAP_FT = 0.1 / FT_TO_M
# The dashed style's mark and gap. Real segments in feet rather than a dash PATTERN, so the
# plan view and the render break the line in the same places - the same reason the bike lane's
# dotted extension is geometry (see paint.py:_dashes_along).
CENTERLINE_DASH_FT = 1.0 / FT_TO_M
CENTERLINE_GAP_FT = 1.0 / FT_TO_M


def centerline_paint_ft(leg, start_ft: float, style: str,
                         shift_ft: float = 0.0, shift_side: str | None = None) -> list[LineString]:
    """The stripes actually painted down this leg's middle, from `start_ft` to its far end.

    ONE definition for both views, because they had two and only one of them followed the
    road. The plan view offset the leg's real centerline; the 3D render was handed the
    leg's near and far points and drew a straight stripe between them - the CHORD. Same
    on a straight leg, but off by several feet on a curving one.
    """
    if style == "none" or start_ft >= leg.centerline.length:
        return []
    if shift_ft and shift_side is not None:
        # A TWO-WAY BIKE LANE ON ONE SIDE PUSHES THE TRAVEL LANES OFF THE ALIGNMENT, so the
        # divider between them is no longer the alignment itself - see
        # treatments.travel_lane_divider_shift_ft for where the distance comes from and why the
        # two lanes come out equal.
        #
        # Through inset_line_ft rather than offset_curve, for the reason this function exists at
        # all: it is the same lateral-offset machinery the bike lane's own stripes use, on the
        # same station grid, with the same clamping inside the traced kerb. An offset curve's arc
        # length differs from the centerline's, so stationing along it is not stationing along
        # the road - exactly the divergence the single-definition rule forbids.
        # A NEGATIVE shift means the other side, not the same distance on this one. The sign is
        # resolved here rather than assumed away.
        if shift_ft < 0:
            shift_side = str(Side(shift_side).other)
            shift_ft = -shift_ft
        painted = inset_line_ft(leg, shift_side, shift_ft, start_ft)
        if painted is None or painted.is_empty or painted.geom_type != "LineString":
            return []
    else:
        painted = shapely.ops.substring(leg.centerline, start_ft, leg.centerline.length)
    if painted.is_empty or painted.geom_type != "LineString":
        return []
    if style == "double_yellow":
        return [line for line in
                (painted.offset_curve(sign * DOUBLE_YELLOW_GAP_FT / 2) for sign in (1, -1))
                if line.geom_type == "LineString" and not line.is_empty]
    period_ft = CENTERLINE_DASH_FT + CENTERLINE_GAP_FT
    dashes = []
    at_ft = 0.0
    while at_ft + CENTERLINE_DASH_FT <= painted.length:
        dash = shapely.ops.substring(painted, at_ft, at_ft + CENTERLINE_DASH_FT)
        if dash.geom_type == "LineString" and dash.length > 0:
            dashes.append(dash)
        at_ft += period_ft
    return dashes


def crosswalk_axes(leg, offset_ft: float, skew_deg: float = 0.0):
    """(centre, along-travel axis, across-road axis, cos skew) for a crossing on this leg.

    One definition of the crossing's frame, so the band the plan view draws, the reach the
    3D export writes and the check in src/checks.py are all measured off the same axes.

    Read AT the station, off the polyline, not extrapolated from the leg's first segment.
    A centerline that bends between the first segment and the crossing's station throws the
    axes off - by 29.4 deg on Louellen, whose NJDOT alignment rounds the corner where CR
    518 turns off W Broad.

    frame_at is the same leg-frame transform station_offset_many inverts, so the centre a
    crossing is built on and the station everything else measures it by are the same
    arithmetic. See src/geometry/model/leg_frame.py:polyline_frame.
    """
    from src.geometry.model import frame_at

    origin, tangent = frame_at(leg.centerline, offset_ft)
    centre = (float(origin[0]), float(origin[1]))
    ux, uy = float(tangent[0]), float(tangent[1])

    skew = np.radians(skew_deg)
    cos_s, sin_s = np.cos(skew), np.sin(skew)
    # Rotate the leg's own axes by the skew: n is the across-road axis the crosswalk
    # spans, u the along-travel axis its depth runs down.
    nx, ny = -uy, ux
    nx, ny = nx * cos_s - ny * sin_s, nx * sin_s + ny * cos_s
    ux, uy = ux * cos_s - uy * sin_s, ux * sin_s + uy * cos_s
    return centre, (ux, uy), (nx, ny), cos_s


def crosswalk_reaches_ft(state, offsets: dict, skews: dict, roadway=None,
                          marked: set | None = None) -> dict:
    """{leg_name: (left_ft, right_ft)} - how far each crossing runs to reach its kerbs.

    Written into the geometry JSON so the 3D render spans the same real, asymmetric width
    the plan view draws, instead of half the nominal width either side of NJDOT's
    centerline. Blender can't import from src/, so this has to travel as numbers.

    Two passes, because at a shared corner the two crossings run into EACH OTHER. Each
    reaches for its own kerb, and near the corner those kerbs are the same kerb - Greenwood
    north's bars and Broad east's overlapped by 2.07 sq ft of doubled paint. The second pass
    re-measures each crossing with the others' footprints cut out of the roadway it is
    allowed to occupy, which is the same mechanism that already keeps a crossing off the
    footway.
    """
    def measure(name, leg, allowed):
        centre, u, normal, _cos = crosswalk_axes(leg, offsets[name].offset_ft, skews.get(name, 0.0))
        return crosswalk_reach_to_curbs_ft(leg, centre, normal, u, CROSSWALK_DEPTH_FT, allowed)

    legs = {name: leg for name, leg in state.legs.items() if name in offsets}
    provisional = {name: measure(name, leg, roadway) for name, leg in legs.items()}
    if roadway is None or roadway.is_empty:
        return provisional

    depth_ft = CROSSWALK_DEPTH_FT
    bands = {name: crosswalk_band_ft(legs[name], offsets[name].offset_ft, depth_ft,
                                      skews.get(name, 0.0), reach=provisional[name])
             for name in legs if marked is None or name in marked}
    reaches = {}
    for name, leg in legs.items():
        others = [b for other, b in bands.items()
                  if other != name and b is not None and not b.is_empty]
        allowed = roadway.difference(unary_union(others)) if others else roadway
        reaches[name] = measure(name, leg, allowed)
    return reaches


def crosswalk_band_ft(leg, offset_ft: float, depth_ft: float, skew_deg: float = 0.0,
                     span_ft: float | None = None, lateral_offset_ft: float = 0.0,
                     roadway=None, reach: tuple | None = None) -> Polygon:
    """The rectangle a painted crosswalk occupies: `depth_ft` along the leg, centered on
    `offset_ft`, spanning the leg's full curb-to-curb width. Built from the leg's own
    centerline and width, the same inputs blender_crosswalks.py uses (near + u*offset,
    then out to +/- width/2 along n), so the band drawn here is the footprint the 3D
    render will fill with stripes.

    `skew_deg` rotates the band off square to the leg, to match the orientation of the
    surveyed crossing it came from (src/render/crosswalks.py:_crossing_skew_deg). The
    span is divided by cos(skew) so a rotated band still reaches both curb lines rather
    than falling short of them - a crossing at an angle has further to go.

    `span_ft` overrides the full curb-to-curb width, and `lateral_offset_ft` shifts the
    band off the road centerline - together these draw a stop bar, which covers only the
    entering half of the roadway (see src/render/crosswalks.py:stop_bar_band_geometry_ft).
    """
    (cx, cy), (ux, uy), (nx, ny), cos_s = crosswalk_axes(leg, offset_ft, skew_deg)
    half_d = depth_ft / 2
    if span_ft is None and reach is not None:
        left_ft, right_ft = reach
    elif span_ft is None:
        left_ft, right_ft = crosswalk_reach_to_curbs_ft(leg, (cx, cy), (nx, ny), (ux, uy),
                                                         depth_ft, roadway)
    else:
        # An explicit span is a stop bar, which is sized from the entering lane rather than
        # from the kerbs - see stop_bar_band_geometry_ft.
        half = span_ft / (2 * max(cos_s, 0.2))
        left_ft = right_ft = half
    cx += nx * lateral_offset_ft
    cy += ny * lateral_offset_ft
    return Polygon([
        (cx + nx * left_ft + ux * half_d, cy + ny * left_ft + uy * half_d),
        (cx - nx * right_ft + ux * half_d, cy - ny * right_ft + uy * half_d),
        (cx - nx * right_ft - ux * half_d, cy - ny * right_ft - uy * half_d),
        (cx + nx * left_ft - ux * half_d, cy + ny * left_ft - uy * half_d),
    ])


# How far past the nominal half-width to look for the kerb before giving up. Generous: the
# traced kerb flares well past half-width approaching a corner, which is exactly where a
# crosswalk sits.
CURB_SEARCH_FACTOR = 3.0


# How finely the reach is stepped outward looking for the kerb. Well under a paint stripe's
# width, so the crossing lands on the kerb rather than visibly short of or over it.
REACH_STEP_FT = 0.25


def crosswalk_reach_to_curbs_ft(leg, center, normal, along=None,
                                 depth_ft: float = 0.0, roadway=None) -> tuple[float, float]:
    """(left, right) distance from `center` out to this leg's two kerbs along `normal`.

    A crosswalk runs kerb to kerb. Cast rays outward from the centre until they leave the
    roadway - the traced kerb's offset AT ITS OWN STATION, not the corner return, which
    flares away from the road and catches diagonal rays far outside the carriageway.

    `along` and `depth_ft` describe the crossing's own depth. The reach is the smallest found
    across the near edge, centre and far edge, so the corners of the painted band stay inside
    the roadway too, not just its centre line.

    `roadway`, when given, is the junction's pavement polygon, and points must stay inside
    it too. At a corner the pavement's edge is the fillet ARC, which cuts inside the traced
    kerb - the kerb test alone still let a bar corner poke ~1 ft over it on three legs.

    Falls back to the nominal half-width per side when a side has no traced kerb, so a leg
    without one behaves exactly as before.
    """
    from src.geometry.model import curb_offsets_at_stations, station_offset_many

    fallback = leg.curb_to_curb_ft / 2
    normal = np.asarray(normal, dtype=float)
    centre = np.asarray(center, dtype=float)
    # Sampled ACROSS the crossing's depth, not just down its centre line. Three samples
    # (near edge, centre, far edge) is not enough: a corner fillet arc is convex toward the
    # road, so it can cut through the middle of a bar's outer edge while both of that bar's
    # corners are still inside. Seven keeps the residual under the paint's own thickness.
    if along is not None and depth_ft:
        along = np.asarray(along, dtype=float)
        origins = centre + np.outer(np.linspace(-depth_ft / 2, depth_ft / 2, 7), along)
    else:
        origins = centre[None, :]

    steps = np.arange(0.0, fallback * CURB_SEARCH_FACTOR + REACH_STEP_FT, REACH_STEP_FT)
    reach = []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        if getattr(leg, f"{side}_curb", None) is None:
            reach.append(fallback)
            continue
        # (origin, step, xy) for every sample at once - one station_offset_many call and one
        # point-in-polygon call rather than one per origin.
        points = origins[:, None, :] + normal * sign * steps[None, :, None]
        flat = points.reshape(-1, 2)
        stations, offsets = station_offset_many(leg.centerline, flat)
        curb_offsets = curb_offsets_at_stations(leg, side, stations)
        if curb_offsets is None:
            reach.append(fallback)
            continue
        outside = np.abs(offsets) > np.abs(curb_offsets)
        if roadway is not None and not roadway.is_empty:
            outside |= ~shapely.contains_xy(roadway, flat[:, 0], flat[:, 1])
        outside = outside.reshape(points.shape[:2])
        # The last step still INSIDE, on whichever sample leaves earliest - not the first
        # step outside. Taking the first step outside put the paint's outer edge exactly ON
        # the boundary, which the kerb test reads as inside (|offset| <= |curb|) and the
        # pavement test reads as outside (a polygon does not contain its own boundary), so
        # the bars kept landing a hair over the kerb.
        any_out = outside.any(axis=1)
        first_out = outside.argmax(axis=1)
        limits = np.where(any_out, steps[np.maximum(first_out - 1, 0)], steps[-1])
        reach.append(float(limits.min()))
    return reach[0], reach[1]


# A continental/ladder crossing's bars. Authoritative here in FEET, like CROSSWALK_DEPTH_FT,
# and handed to the renderer as a bar COUNT per crossing so the layout arithmetic exists once
# and is testable - scripts/blender/blender_crosswalks.py cannot import from src/.
CONTINENTAL_BAR_WIDTH_FT = 0.5 / FT_TO_M     # the renderer's long-standing 0.5 m bar
CONTINENTAL_BAR_GAP_FT = 0.5 / FT_TO_M       # and 0.5 m gap between bars


def continental_bar_count(span_ft: float,
                           bar_ft: float = CONTINENTAL_BAR_WIDTH_FT,
                           gap_ft: float = CONTINENTAL_BAR_GAP_FT) -> int:
    """How many bars a continental crossing of this kerb-to-kerb span carries.

    n bars with n-1 gaps between them, at most as many as fit at the nominal pitch. The
    renderer then spreads the leftover across the GAPS so the two end bars land exactly on
    the kerbs (scripts/blender/blender_crosswalks.py:_crosswalk_bars) - a whole-period pitch
    leaves up to one period unpainted, which still reads as a crossing stopping short.
    """
    period = bar_ft + gap_ft
    if period <= 0:
        return 1
    return max(int((span_ft + gap_ft) / period), 1)


def crosswalk_reach_on_leg_side_ft(leg, side: str, crossings, inner_offset_ft: float = 0.0,
                                    beyond_the_tracing: bool = False) -> float:
    """The furthest station along one side of a leg that ANY painted crossing reaches.

    Curbside paint has to stop before the crossing, and "before" has to be measured against
    where the crossings actually ARE. Two things make a single number per leg wrong:

      * Skew. A band pivots about its centre, so one end swings further along the leg than
        the other.
      * The CROSS street's crossing. A taper curving into the corner runs into the crossing
        on the intersecting leg, which has nothing to do with this leg's own offset.

    So this measures against every crossing at the junction, restricted to the strip this
    leg's curbside paint actually occupies on this side: from inner_offset_ft out to the
    kerb. A skewed band reaches further along the leg near the centerline than it does at
    the kerb, and curbside paint never goes near the centerline, so restricting matters as
    much as including.

    `beyond_the_tracing` lifts the bound at the traced kerb, and only one caller asks for it:
    paint.junction_mouths_ft, which uses this to say where the INTERSECTION ends on a kerb
    rather than where a marking should stop. That is a fact about the street, so it must not
    be cut short where nobody traced the kerb - see model.paint_stations for the same
    distinction.
    """
    import numpy as np

    from src.geometry.model import curbside_strip_polygon, station_offset_many

    if crossings is None or crossings.is_empty:
        return 0.0
    strip = curbside_strip_polygon(leg, side, inner_offset_ft, 0.0, leg.centerline.length,
                                    beyond_the_tracing=beyond_the_tracing)
    if strip is None:
        return 0.0
    region = crossings.intersection(strip)
    if region.is_empty:
        return 0.0
    points = np.asarray([xy for part in getattr(region, "geoms", [region])
                          if part.geom_type == "Polygon"
                          for xy in part.exterior.coords], dtype=float)
    if not len(points):
        return 0.0
    stations, _offsets = station_offset_many(leg.centerline, points)
    return float(stations.max())


def crosswalk_bands_ft(state, offsets: dict, skews: dict, depth_ft: float, roadway=None,
                        reaches: dict | None = None) -> dict:
    """{leg_name: band polygon} for every leg - the footprints the 2D view draws, the 3D
    render stripes, and src/checks.py validates. One definition, so a check that passes in
    one path can't be checking different geometry from the other."""
    return {name: crosswalk_band_ft(leg, offsets[name].offset_ft, depth_ft, skews.get(name, 0.0),
                                     roadway=roadway,
                                     reach=(reaches or {}).get(name))
            for name, leg in state.legs.items() if name in offsets}


def stop_bar_bands_ft(state, stop_bar_offsets: dict, skews: dict) -> dict:
    """{leg_name: stop bar polygon}, on the same shared terms as crosswalk_bands_ft."""
    bands = {}
    for name, offset_ft in stop_bar_offsets.items():
        leg = state.legs.get(name)
        if leg is None:
            continue
        # The offset comes back in the SKEWED frame the band is laid in - see
        # stop_bar_band_geometry_ft, which is also where the export reads it, so the two views
        # cannot differ about where the bar begins.
        span_ft, lateral_ft = stop_bar_band_geometry_ft(
            stop_bar_width_ft(state, name), entering_lane_width_ft(state, name) is None,
            inner_ft=divider_shift_toward_ft(state, name, Side.LEFT),
            skew_deg=skews.get(name, 0.0))
        band = crosswalk_band_ft(leg, offset_ft, STOP_BAR_PLAN_DEPTH_FT, skews.get(name, 0.0),
                                  span_ft=span_ft, lateral_offset_ft=lateral_ft)
        bands[name] = _trim_to_the_entering_side(
            leg, band, divider_shift_toward_ft(state, name, Side.LEFT))
    return bands


# How far past the centreline a straight bar's corner may fall on a CURVING leg before the bar is
# trimmed. A stop bar is a straight stripe and a centreline is not always straight, so the two
# genuinely cross by a little wherever the road bends: 0.52 ft on louellen_st_west, whose
# centerline kinks. That is a real overhang - it is drawn, and a reviewer sees a stop bar over the
# double yellow - so it is trimmed rather than tolerated.
TRIM_STOP_BAR_BEYOND_FT = 0.05


def _trim_to_the_entering_side(leg, band, divider_ft: float):
    """Cut `band` back so none of it lies past the painted centreline.

    A stop bar RESTS AGAINST that line; it never crosses it (MUTCD - the bar covers the entering
    approach only). Sizing alone cannot guarantee that: the bar is a straight quadrilateral and
    the line it stops at follows a centreline that bends, so on a curving leg the bar's inner
    corner falls the wrong side of it however carefully the span is computed. So the shape is
    trimmed to the region it is allowed to occupy, which is the one construction that cannot be
    defeated by curvature.

    `divider_ft` is where that line sits, signed toward the entering (LEFT) side - normally zero,
    and the two-way bike lane's divider offset where the travel lanes have been shifted.
    """
    if band is None or band.is_empty:
        return band
    _stations, offsets = station_offset_many(leg.centerline,
                                             np.asarray(band.exterior.coords, dtype=float))
    if float(divider_ft - offsets.min()) <= TRIM_STOP_BAR_BEYOND_FT:
        return band                      # already clear of the line
    # From the CENTRELINE, over the whole leg - NOT inset_line_ft, which clamps to the traced
    # kerb's station span. louellen_st_west's left kerb is traced only from station 60 and its
    # stop bar sits at ~50, so the clamped line stopped short of the bar, never crossed it, and
    # the trim silently did nothing. A dividing line has to span whatever it is dividing.
    line = leg.centerline
    if abs(divider_ft) > 1e-9:
        # offset_curve's sign convention depends on the line's direction, so pick the candidate
        # that actually lands on the entering side rather than assuming which one it is.
        for candidate in (leg.centerline.offset_curve(abs(divider_ft)),
                          leg.centerline.offset_curve(-abs(divider_ft))):
            if candidate.is_empty or candidate.geom_type != "LineString":
                continue
            _st, off = station_offset_many(
                leg.centerline,
                np.asarray(candidate.interpolate(0.5, normalized=True).coords, dtype=float))
            if abs(float(off[0]) - divider_ft) < abs(divider_ft) * 0.5 + 0.1:
                line = candidate
                break
    if line is None or line.is_empty:
        return band
    try:
        pieces = list(shapely.ops.split(band, line).geoms)
    except Exception:
        return band                      # not splittable - leave it for the invariant to report
    keep = []
    for piece in pieces:
        if piece.is_empty or piece.area <= 0:
            continue
        _st, off = station_offset_many(leg.centerline,
                                        np.asarray(piece.representative_point().coords,
                                                   dtype=float))
        if float(off[0]) >= divider_ft:
            keep.append(piece)
    if not keep:
        return band
    return unary_union(keep) if len(keep) > 1 else keep[0]
