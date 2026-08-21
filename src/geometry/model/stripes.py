"""The geometry of the paint itself: strips, tapers, stall lines, post rows, hatching.

Everything here takes a leg and a band of offsets and returns polygons or polylines in state-plane
feet. It decides SHAPE, never whether a marking is warranted - that is a treatment's job
(src/geometry/treatments/) - which is what lets the same taper serve a parking buffer, a bike-lane
buffer and a daylight zone."""

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from src.geometry.model.leg_frame import (Leg,STRIP_SAMPLE_FT, place_in_measured_frame,
                                          _place_no_further_in_than, point_at, _traced_curb_frame,
                                          unit_vector, curb_edge_by_station, curb_offsets_at_stations,
                                          curb_point_at_station, curb_station_span,
                                          inset_line_ft, inset_point_at_station, paint_stations,
                                          station_offset_many)



def curbside_strip_polygon(leg: "Leg", side: str, inner_offset_ft: float,
                            start_ft: float, end_ft: float | None = None,
                            beyond_the_tracing: bool = False) -> Polygon | None:
    """The strip of roadway between a leg's real curb and a line inner_offset_ft from its
    centerline, between two centerline stations.

    INVARIANT: both boundaries are sampled at the SAME stations. Never build this from
    `substring` of each line - substring measures arc length from each line's own start, so
    the two ends land at different stations and close into a wedge, not a strip.

    Returns None where there is no room - the curb comes inside inner_offset_ft (paint would
    be outside the roadway) or the span is empty.

    beyond_the_tracing is paint_stations' - see it.
    """
    stations = paint_stations(leg, side, start_ft, end_ft, beyond_the_tracing)
    if stations is None:
        return None
    lo, hi = float(stations[0]), float(stations[-1])
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    if curb_offsets is None:
        return None

    # The inner edge never crosses outside the real curb. DATUM: the traced kerb can come
    # inside the nominal half-width (broad_st_east left is traced at 22.7 ft against a
    # nominal 24.2 ft), so the nominal figure must not be taken on faith. Where the curb is
    # inside the lane edge the strip pinches to nothing.
    sign = 1.0 if side == "left" else -1.0
    inner = sign * np.minimum(inner_offset_ft, np.abs(curb_offsets))

    # The outer boundary uses the KERB'S OWN coordinates, not a resampling of them - see
    # curb_edge_by_station. The inner boundary has no such geometry to borrow and is placed
    # in the measuring frame instead.
    outer_pts = curb_edge_by_station(leg, side, lo, hi)
    if outer_pts is None:
        return None
    inner_pts = _place_no_further_in_than(leg.centerline, stations, inner)

    # OSM traces the block, not every curb kink (SKILLS.md #7) - greenwood_ave_south's right
    # kerb goes 261 ft with no vertex at all, and its centerline bends inside that gap (48
    # vertices - a real curve, not noise). A straight chord across a gap that wide, held
    # against a bending centerline, does one of two things, and both shipped a defect before
    # this check existed. Against the INNER edge it can cross it: a self-intersecting ring
    # that buffer(0) resolves into a bowtie rather than a clean strip, staying one piece
    # across a cross-street opening a later cut could not split. Short of crossing, it can
    # still bow far outside where the kerb actually runs - greenwood's LEFT side stayed
    # perfectly valid but its chord bulged 8-9 ft past the traced line at the very station an
    # opening cut was sized to reach, so the cut's outer bound never got there either and the
    # opening failed to sever the fill a second, distinct way. Both are the same mechanism
    # (a straight chord standing in for a curve), so both get the same fallback: rebuild the
    # outer edge on the SAME station grid as the inner edge, both placed in the measuring
    # frame, where outer's offset magnitude is never less than inner's (inner already clamps
    # to it above) so the two boundaries cannot cross and the outer edge cannot wander from
    # the curb line the rest of this leg's paint is measured against.
    #
    # Detected by sampling the raw chord and comparing to the smooth curve rather than by gap
    # size: gaps of 130-357 ft are normal and harmless on every other leg in this dataset (the
    # chord tracks a straight or gently-curving centerline there), so gap size alone is not
    # the signal - deviation from the smooth curve is. A survey of every curbside strip built
    # across all five sites found greenwood's two sides at 9.6-10.9 ft of deviation and every
    # other leg under 1.2 ft; STRIP_SAMPLE_FT sits well inside that margin on both sides.
    outer_line = LineString(outer_pts)
    n_samples = max(int(outer_line.length / 5.0), 2)
    sample_pts = np.array([outer_line.interpolate(t, normalized=True).coords[0]
                            for t in np.linspace(0.0, 1.0, n_samples)])
    sample_st, sample_off = station_offset_many(leg.centerline, sample_pts)
    sample_smooth = curb_offsets_at_stations(leg, side, np.clip(sample_st, lo, hi))
    chord_deviates = (sample_smooth is None
                       or np.max(np.abs(np.abs(sample_off) - np.abs(sample_smooth))) > STRIP_SAMPLE_FT)

    poly = Polygon(list(outer_pts) + list(reversed(inner_pts)))
    if not poly.is_valid or chord_deviates:
        outer_pts = place_in_measured_frame(leg.centerline, stations, curb_offsets)
        poly = Polygon(list(outer_pts) + list(reversed(inner_pts)))
        if not poly.is_valid:
            poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 1e-6) else None


def lane_narrowing_polygons_ft(leg: "Leg", stripe_width_ft: float,
                                start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                sides: tuple = ("left", "right"),
                                end_ft: float | None = None,
                                beyond_the_tracing: bool = False) -> list[Polygon]:
    """Two thin paint-only strips just inside each curb line - a visual lane narrowing done
    with paint, NOT a curb_to_curb_ft change (no pavement/curb geometry is touched). See
    src/geometry/treatments/lanes.py:LaneNarrowing.

    start_left_ft/start_right_ft trim each strip to begin exactly where its corner taper
    starts (lane_narrowing_taper_ft), independently per side; without them the strip runs on
    through the open intersection box where no paint exists.

    sides restricts which curb(s) get a strip - a marked-parking buffer
    (treatments/parking.py:MarkedParking) needs one side only.

    beyond_the_tracing is paint_stations' - see it. Only the statutory daylight zone asks."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    polys = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        poly = curbside_strip_polygon(leg, side, inner_half, start_ft, end_ft,
                                       beyond_the_tracing)
        if poly is not None:
            polys.append(poly)
    return polys


def lane_narrowing_edge_lines_ft(leg: "Leg", stripe_width_ft: float,
                                  start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                  sides: tuple = ("left", "right"),
                                  keep_inside_ft: float = 0.0,
                                  beyond_the_tracing: bool = False) -> list[LineString]:
    """The solid line marking the narrowed travel lane's outer edge on each side - the same
    inner boundary lane_narrowing_polygons_ft's buffer starts from (TARGET_LANE_WIDTH_FT in
    sites/broad_st_greenwood/scenarios.py), drawn explicitly so the lane width reads on the
    render rather than being implied by where the hatching starts.

    start_left_ft/start_right_ft/sides - see lane_narrowing_polygons_ft. Matching them keeps
    this line, the hatch fill and the corner taper starting at the same point with no gap.
    beyond_the_tracing likewise - it has to match the fill's, or the zone gets an edge line over
    part of its length and a bare hatch boundary over the rest."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    lines = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        line = inset_line_ft(leg, side, inner_half, start_ft, keep_inside_ft=keep_inside_ft,
                              beyond_the_tracing=beyond_the_tracing)
        if line is not None:
            lines.append(line)
    return lines


def _corner_bulge_normal(leg: "Leg", role: str) -> np.ndarray:
    """Unit normal pointing from a leg's curb toward where a real corner fillet's arc bulges
    - the same direction that role's own curb is already offset from centerline ('left' for
    the leg_a corner role, 'right' for leg_b; see build_corner_fillets), continuing further
    outward. The arc sits further from centerline than the curb it replaces, on the SAME
    side, not the opposite one."""
    c0, c1 = np.array(leg.centerline.coords[0]), np.array(leg.centerline.coords[1])
    u = unit_vector(c1 - c0)
    return np.array([-u[1], u[0]]) if role == "left" else np.array([u[1], -u[0]])


def _taper_arc_points(leg: "Leg", role: str, sign: int, inner_half_ft: float,
                       anchor_ft: float, target_ft: float, n_points: int) -> list[tuple] | None:
    """The taper arc on ONE side of a leg, as a list of points, or None where there is none.

    Tangent to the straight inset line at anchor_ft and passing exactly through the real curb
    at target_ft. Tangent-at-one-point + passes-through-another-point + a common circle centre
    uniquely determines the radius - solved directly, not guessed or borrowed from elsewhere:
    for chord d = target - anchor and outward unit normal n, R = |d|^2 / (2 * dot(d, n)).

    THE ONE HOME for that arc: lane_narrowing_taper_ft draws it as the line and
    lane_narrowing_taper_polygons_ft fills inside it, so they must not solve it separately.

    Held BOTH WAYS at the end, because the arc is solved in WORLD space while the lane edge it
    leaves from and the kerb it lands on are offsets in the LEG's frame - the same lines only
    while the centerline is straight. Once the alignment bends onto the carriageway
    (intersection._centre_legs_on_traced_kerbs):

      * inward, the arc can cut inside the lane edge, so the offset is floored at inner_half_ft;
      * outward, it can cross the traced kerb between its two endpoints even though both of them
        sit on it - a chord of a circle solved in world space against a kerb that curves. At 2.5x
        on louellen_st_west it stood 0.41 ft past the kerb and check_paint_over_the_curb refused
        the export, which is the check doing its job on paint sized off a nominal half-width.

    Both clamps are the move inset_line_ft makes; the arc keeps its shape everywhere it was
    already between the two.
    """
    p1 = inset_point_at_station(leg, anchor_ft, sign * inner_half_ft)
    p2 = curb_point_at_station(leg, role, target_ft)
    if p2 is None:
        return None
    normal = _corner_bulge_normal(leg, role)
    d = p2 - p1
    denom = 2 * np.dot(d, normal)
    if abs(denom) < 1e-6:
        return None     # p2 already (near enough) on the tangent line - no taper needed
    radius_ft = np.dot(d, d) / denom
    center = p1 + radius_ft * normal
    a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    a2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    angles = a1 + np.linspace(0, delta, n_points)
    arc = np.array([(center[0] + radius_ft * np.cos(t), center[1] + radius_ft * np.sin(t))
                    for t in angles])
    stations, offsets = station_offset_many(leg.centerline, arc)
    curb_offsets = curb_offsets_at_stations(leg, role, stations)
    inside = np.abs(offsets) < inner_half_ft
    outside = (np.abs(offsets) > np.abs(curb_offsets)) if curb_offsets is not None else np.zeros(
        len(offsets), bool)
    if not inside.any() and not outside.any():
        return [tuple(p) for p in arc]
    offsets[inside] = sign * inner_half_ft
    if curb_offsets is not None:
        offsets[outside] = sign * np.abs(curb_offsets)[outside]
    return place_in_measured_frame(leg.centerline, stations, offsets)


# A taper runs from the straight run's start INWARD to the curb. With target_ft further out
# than anchor_ft there is no room for one between the corner return and the crosswalk, and
# solving the arc anyway sweeps it backwards and mangles the hatching.
def _taper_fits(anchor_ft: float, target_ft: float) -> bool:
    return target_ft < anchor_ft


def lane_narrowing_taper_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                             n_points: int = 16, sides: tuple = ("left", "right")) -> list[LineString]:
    """Tapers a lane-narrowing buffer's straight edge line, both sides of the leg, from
    anchor_ft (the stop-bar/clearance point where the straight run ends) back out to the REAL
    curb at target_ft.

    A SAME-LEG taper, not a sweep around the corner to the cross leg. The sweep cannot work:
    the cross leg's own crosswalk sits at the corner by definition, so any curve reaching its
    curb ends inside the excluded zone whatever radius it uses. Terminating before this leg's
    own crosswalk avoids that.

    Tangent to the straight inset line at anchor_ft, so the buffer's edge continues with no
    seam. Arc geometry lives in _taper_arc_points - see it."""
    inner_half = max(leg.curb_to_curb_ft / 2 - stripe_width_ft, 0.5)
    if not _taper_fits(anchor_ft, target_ft):
        return []
    tapers = []
    for sign, role in ((1, "left"), (-1, "right")):
        if role not in sides:
            continue
        arc = _taper_arc_points(leg, role, sign, inner_half, anchor_ft, target_ft, n_points)
        if arc is not None:
            tapers.append(LineString(arc))
    return tapers


def lane_narrowing_taper_polygons_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                                      n_points: int = 16, sides: tuple = ("left", "right")) -> list[Polygon]:
    """The paint-only buffer's fill zone WITHIN the taper itself - the chevron paint carries
    on around the curve to the curb rather than stopping where the straight run ends.

    Bounded by the taper arc (_taper_arc_points) on one side and the real curb from target_ft
    back to anchor_ft on the other: the same curb/inset pairing lane_narrowing_polygons_ft
    uses for the straight run, curved, so hatch_lines_ft fills both with one pattern."""
    inner_half = max(leg.curb_to_curb_ft / 2 - stripe_width_ft, 0.5)
    if not _taper_fits(anchor_ft, target_ft):
        return []
    polys = []
    for sign, role in ((1, "left"), (-1, "right")):
        if role not in sides:
            continue
        # The SAME arc lane_narrowing_taper_ft draws as the line - see _taper_arc_points.
        arc_pts = _taper_arc_points(leg, role, sign, inner_half, anchor_ft, target_ft, n_points)
        if arc_pts is None:
            continue    # no taper (see lane_narrowing_taper_ft) - nothing extra to fill
        # The curb run back from target_ft to anchor_ft, sampled by STATION - never
        # `substring(curb, ...)`, which is arc length along the traced kerb from the kerb's
        # own start (see curb_point_at_station). arc_pts already ends at p2 (the curb at
        # target_ft), so that duplicate is dropped; Polygon() closes the ring back to p1.
        n_curb = max(int(np.ceil((anchor_ft - target_ft) / STRIP_SAMPLE_FT)) + 1, 2)
        curb_stations = np.linspace(target_ft, anchor_ft, n_curb)
        curb_offsets = curb_offsets_at_stations(leg, role, curb_stations)
        curb_forward = [point_at(leg.centerline, s, float(o))
                        for s, o in zip(curb_stations, curb_offsets)][1:]
        ring = arc_pts + curb_forward
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # The arc is a circle solved through two points; between them it is free to bulge
        # past the kerb, which care about the endpoints cannot prevent. Only clipping to the
        # roadway guarantees the fill stays on the road.
        roadway = curbside_strip_polygon(leg, role, 0.0, target_ft, anchor_ft)
        if roadway is not None:
            poly = poly.intersection(roadway)
        for part in getattr(poly, "geoms", [poly]):
            if part.geom_type == "Polygon" and not part.is_empty and part.area > 1e-6:
                polys.append(part)
    return polys


def bollard_points_ft(leg: "Leg", stripe_width_ft: float, start_ft: float,
                       spacing_ft: float = 10.0, sides: tuple = ("left", "right")) -> list[tuple[float, float]]:
    """Points down the center of a stripe_width_ft paint-only buffer strip next to the curb -
    one line per requested side, from start_ft (past the corner fillet curve, the same
    clearance convention as crosswalks and stop bars; see leg_clearance_ft) at spacing_ft to
    the end of the leg.

    Same inner_half math as lane_narrowing_polygons_ft, so a bollard always sits centered in
    the buffer actually painted rather than at a separately-guessed offset. Used by
    treatments/lanes.py:LaneNarrowingBollards and treatments/parking.py:ParkingBufferBollards."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    points = []
    for side, sign in (("left", 1), ("right", -1)):
        if side not in sides:
            continue
        span = curb_station_span(leg, side)
        if span is None or span[1] < start_ft:
            continue
        # Every station in one read of the kerb. Counted rather than accumulated, so the last
        # bollard's station is start + n*spacing exactly, not the sum of n additions.
        n = int((span[1] - start_ft) // spacing_ft) + 1
        stations = start_ft + np.arange(n) * spacing_ft
        curb_off = np.abs(curb_offsets_at_stations(leg, side, stations))
        # Centered between the strip's two real boundaries at EACH station, so a bollard sits
        # in the painted buffer even where the traced kerb comes inside the nominal
        # half-width (see curbside_strip_polygon).
        lateral = (curb_off + np.minimum(inner_half, curb_off)) / 2
        points.extend(tuple(point_at(leg.centerline, float(s), sign * float(o)))
                      for s, o in zip(stations, lateral))
    return points


def whole_stalls_ft(length_ft: float, stall_length_ft: float) -> int:
    """How many whole stall_length_ft stalls fit in length_ft.

    THE ONE HOME for that division, and it took three: this, metrics.stalls_in_run and
    corridor_paint.stalls_per_span each carried its own copy, so the ticks on the plan, the
    number in the summary panel and the number on the corridor sheet were three rules that
    happened to agree.

    A PARTIAL STALL IS NOT A STALL. The remainder is floored away and gets no marking, because
    a car needs the whole length: half a stall drawn at the end of a run is not a space anyone
    can use, and a driver who believes it is parks across whatever comes next.

    Floors on `length_ft / stall_length_ft + 1e-9`, not the bare ratio. A run built to hold
    exactly n stalls is itself n * stall_length_ft summed in float, so a caller that later
    recovers length_ft by subtracting that run's own (lo, hi) can land a few ULPs under the
    true multiple (21.99999999999997 instead of 22.0) - and a bare floor reads that as n-1
    stalls, silently dropping the closing tick of a stall that was already drawn to fit. The
    epsilon is 5 orders below the smallest real partial remainder this project ever compares
    (inches, not tenths of a foot), so a genuine partial stall floors the same as before.
    """
    if stall_length_ft <= 0:
        return 0
    return max(int(length_ft / stall_length_ft + 1e-9), 0)


def stall_lane_runs_ft(runs: list[tuple[float, float]], stall_length_ft: float,
                        keep_inside_ft: float = 0.0) -> list[tuple[float, float]]:
    """Each run of paintable kerb, trimmed back to the whole stalls it holds.

    THE ONE HOME for where a parking lane starts and stops, and the answer to a defect the
    renders wore for a long time: a stall lane laid out over the run where parking is LEGAL and
    then cut by everything that turns out to be in the way. Four separate things cut kerbside
    paint after a treatment has placed it - the mountable aprons, every painted crossing in the
    frame, the kerb openings, and the end of the traced kerb - and none of them was consulted
    about where a stall could go. So the ticks stopped at the last whole stall of the LEGAL run
    while the edge line ran to its end, which is the open-ended stub of parking lane at the far
    end of every run on every sheet; a tick that landed in a driveway was deleted by the opening
    rule and left the stall it was closing 44 ft long; and a stall whose two ticks straddled a
    driveway was drawn straight across it.

    Feed it the runs the paint actually reaches - PaintContext.open_runs - and every stall it
    lays is bounded at both ends by a tick of its own, with nothing in between.

    keep_inside_ft holds the two end ticks that far inside the run, the same courtesy
    inset_line_ft pays the kerb: a tick drawn exactly on the boundary of the cut that made the
    run is half over it, and half a tick is dropped by MIN_LINE_LENGTH_FT - which is the missing
    closing tick again, arrived at from the other side.
    """
    lanes = []
    for lo, hi in runs:
        lo, hi = lo + keep_inside_ft, hi - keep_inside_ft
        n_stalls = whole_stalls_ft(hi - lo, stall_length_ft)
        if n_stalls:
            lanes.append((lo, lo + n_stalls * stall_length_ft))
    return lanes


def parking_lane_edge_line_ft(leg: "Leg", side: str, depth_ft: float, start_ft: float,
                               end_ft: float | None = None, curb_offset_ft: float = 0.0) -> LineString | None:
    """The line marking the inner edge of a curbside marked-parking lane - depth_ft in from
    the curb on the given side, same start/end convention as lane_narrowing_edge_lines_ft
    (past the corner fillet curve; see leg_clearance_ft). Drawn even though real curbside
    parking is often only stall ticks, so the lane's depth reads in plan and in 3D.

    curb_offset_ft > 0 pulls the whole lane in from the curb by that much - e.g. a striped
    no-parking buffer between parking and the kerb. See treatments/parking.py:MarkedParking."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - curb_offset_ft - depth_ft, 0.5)
    # None where there is no room left on this leg to mark parking - at W Broad & Louellen's
    # acute Y, leg_clearance_ft comes out at 133 ft on a 130 ft leg. inset_line_ft says
    # "nothing here" explicitly rather than handing back an empty geometry.
    return inset_line_ft(leg, side, inner_half, start_ft, end_ft)


def parking_stall_lines_ft(leg: "Leg", side: str, depth_ft: float, stall_length_ft: float, start_ft: float,
                            end_ft: float | None = None, curb_offset_ft: float = 0.0) -> list[LineString]:
    """Perpendicular divider lines bounding each marked parallel-parking stall along one side
    of a leg - the standard MUTCD curbside-parking marking: a short tie line at each stall
    boundary, not a hatched zone (a driver parks a real vehicle inside each one). One extra
    divider closes off the last full stall, so n stalls get n+1 lines; n is
    whole_stalls_ft's.

    Each divider runs from the real curb to depth_ft in from it; curb_offset_ft > 0 (see
    parking_lane_edge_line_ft) shifts BOTH ends in, so the divider spans the parking lane
    itself rather than the no-parking buffer between it and the curb.

    start_ft..end_ft is a WHOLE NUMBER OF STALLS - stall_lane_runs_ft's job - so the closing
    tick lands on end_ft and the lane the edge line draws is exactly as long as the stalls in
    it. Handed a fractional span it still lays whole stalls and the remainder goes unmarked,
    which is the honest reading of a span that cannot hold another car."""
    half = leg.curb_to_curb_ft / 2
    sign = 1 if side == "left" else -1
    outer_off = max(half - curb_offset_ft, 0.5)
    inner_off = max(half - curb_offset_ft - depth_ft, 0.5)
    span = curb_station_span(leg, side)
    if span is None:
        return []
    end_ft = min(span[1], leg.centerline.length if end_ft is None else end_ft)
    n_stalls = whole_stalls_ft(end_ft - start_ft, stall_length_ft)
    # Station, not distance along an offset curve - see inset_line_ft. A divider is a
    # cross-section of the parking lane, so both ends must be at the same station.
    stations = start_ft + np.arange(n_stalls + 1) * stall_length_ft
    curb_off = np.abs(curb_offsets_at_stations(leg, side, stations))
    return [LineString([
                point_at(leg.centerline, float(station), sign * min(outer_off, float(off))),
                point_at(leg.centerline, float(station), sign * min(inner_off, float(off))),
            ])
            for station, off in zip(stations, curb_off)]


# ---------------------------------------------------------------------------
# Curb extensions (bulb-outs)
# ---------------------------------------------------------------------------
#
# A curb extension shortens a crossing by moving the KERB LINE laterally into the roadway
# near the junction and tapering it back out. WHY NOT THE OBVIOUS ALTERNATIVE: re-cutting the
# corner ARC at a smaller radius (set_corner_radius) leaves both curb lines where they were -
# measured on broad_st_east x greenwood_ave_north at 29.2 -> 15.0 ft, the arc shortens but
# pavement area moves 0.2 sq ft of 24,000 and every crossing span is unchanged to 0.00 ft.
# The crossings here sit 21-42 ft out, past the corner, so a radius change never reaches them.
#
# DATUM: the extension is measured from the leg's NOMINAL half-width, not from the traced kerb
# at that station, and the two are far apart at a corner. The traced kerb flares through the
# return - broad_st_east's kerbs are 39.4 and 31.6 ft off the centerline where its crossing is
# painted, against a 26.0 ft nominal half-width, so that crossing spans 65.0 ft rather than the
# 52.0 ft the cross-section suggests. Extending from the nominal replaces the flare with the
# extension's own straight face, which is what a built bulb-out does, so the crossing falls
# further than the extension alone implies: 8 ft per side takes 65.0 ft to about 36 ft.
#
# How far an extension may be pushed is bounded by the travel lane it must leave behind, so
# every caller is checked against TARGET_LANE_WIDTH_FT - see
# src/geometry/treatments/corners.py:AddCurbExtension.

# How gently the extension returns to the real kerb: feet along the leg per foot of lateral
# shift. A DESIGN CHOICE, not a measured or standard figure - flagged like
# PARKING_BUFFER_DEFAULT_FT rather than dressed up as a citation. The check that matters is
# not the rate but the total: face plus taper has to stay inside the length of kerb where
# parking is already prohibited, or the extension removes a space. See
# tests/test_curb_extensions.py:test_a_bulbout_fits_inside_the_ordinance_no_parking_length.
BULBOUT_TAPER_RATE = 5.0


def curb_extension_line(leg: "Leg", side: str, extension_ft: float, full_ft: float,
                         taper_ft: float) -> LineString | None:
    """One leg side's kerb with a curb extension built into it.

    Three stretches, in station order:

      0 -> full_ft                  the extension's face, straight, at the leg's nominal
                                    half-width less `extension_ft`
      full_ft -> full_ft + taper_ft the return to the real kerb
      beyond                        the traced kerb itself, vertex for vertex

    The taper is a raised-cosine blend between the two offsets, which is tangent to both ends
    by construction - no kink where the face meets it and none where it rejoins the tracing.
    (An arc solved through two points, which lane_narrowing_taper_ft uses for painted tapers,
    is free to bulge between them; for a KERB that bulge would be built concrete.)

    INVARIANT: the face never sits outside the traced kerb. Where the real kerb is already
    inside the nominal half-width (broad_st_east's left kerb is traced at 22.7 ft against a
    24.2 ft nominal), the TRACING wins and no extension is built there. An extension may take
    roadway, never invent it.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None or leg.curb_to_curb_ft is None:
        return None
    curb_stations, curb_offsets = frame
    sign = 1.0 if side == "left" else -1.0
    face_abs = leg.curb_to_curb_ft / 2 - extension_ft
    taper_end_ft = full_ft + taper_ft

    n = max(int(np.ceil(taper_end_ft / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(0.0, taper_end_ft, n)
    real_abs = np.abs(np.interp(stations, curb_stations, curb_offsets))
    # Raised cosine over the taper, 0 on the face, 1 once the real kerb governs again.
    ease = (1 - np.cos(np.pi * np.clip((stations - full_ft) / taper_ft, 0.0, 1.0))) / 2
    built_abs = np.minimum(face_abs, real_abs) * (1 - ease) + real_abs * ease

    points = place_in_measured_frame(leg.centerline, stations, sign * built_abs)
    # The tracing itself past the taper, not a resampling of it: beyond the extension this
    # side's kerb is still the surveyor's, vertex for vertex.
    points += [point_at(leg.centerline, float(s), float(o))
               for s, o in zip(curb_stations, curb_offsets) if s > taper_end_ft]
    return LineString(points) if len(points) >= 2 else None


# A hatch stroke shorter than this is a clipping artifact, not paint. They appear where a
# stroke grazes a corner of the polygon or crosses the needle-thin tip of a taper, and they
# render as stubs - the "sheared in half" strokes. One came out 0.0 ft long.
MIN_HATCH_STROKE_FT = 1.0


def clip_paint_clear_of(geometry, keep_clear):
    """Cut `keep_clear` out of a piece of paint, returning the surviving pieces.

    Road markings are layered by priority - a crosswalk outranks a buffer or a parking lane.
    The subtraction must be done on the GEOMETRY, never by choosing a far-enough-out start
    station: a skewed crossing reaches further along one kerb than its centre offset suggests.
    """
    if keep_clear is None or keep_clear.is_empty:
        return [geometry]
    remainder = geometry.difference(keep_clear)
    if remainder.is_empty:
        return []
    parts = getattr(remainder, "geoms", [remainder])
    return [g for g in parts if g.geom_type == geometry.geom_type and not g.is_empty]


def hatch_lines_ft(polygon: Polygon, spacing_ft: float = 2.0, angle_deg: float = 45.0,
                    phase_origin: tuple[float, float] = (0.0, 0.0)) -> list[LineString]:
    """Diagonal hatch lines filling a polygon, clipped to its boundary - paint-only
    diagonal/chevron marking, no curb or pavement geometry change.

    phase_origin fixes WHERE the family of parallel lines falls, in world coordinates. A
    buffer is not one polygon - the straight run, the corner taper and whatever survives
    being cut around a crossing are hatched separately - so ALL pieces of one treatment must
    be passed the SAME origin. Phasing each off its own bounding box instead steps the
    strokes sideways at every seam, and a stroke reads as sheared into two offset halves.
    """
    # A corner-hatch polygon built off a traced kerb can pinch to a point where the curb
    # doubles back, which is a bowtie GEOS refuses to intersect against. buffer(0) resolves
    # it without moving an edge; an empty result means there was no area to hatch.
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return []

    minx, miny, maxx, maxy = polygon.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-u[1], u[0]])

    # Which lines of the (infinite, origin-anchored) family reach this polygon: the corners'
    # distances along n, snapped outward to whole multiples of the spacing. Anchoring on
    # multiples of the spacing from a shared origin is what keeps neighbouring pieces in phase.
    origin = np.asarray(phase_origin, dtype=float)
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]) - origin
    along_n, along_u = corners @ n, corners @ u
    steps = np.arange(np.floor(along_n.min() / spacing_ft), np.ceil(along_n.max() / spacing_ft) + 1)

    # Every hatch line at once: one MultiLineString intersection rather than a GEOS call per
    # line. Each line must span the polygon's extent ALONG u as well as sit at the right
    # distance along n - the phase origin is the state-plane origin, half a million feet
    # away, so a segment merely centred on it never reaches.
    centers = origin + n * (steps * spacing_ft)[:, None]
    lo, hi = along_u.min() - diag, along_u.max() + diag
    ends = np.stack([centers + u * lo, centers + u * hi], axis=1)
    clipped = MultiLineString([tuple(map(tuple, pair)) for pair in ends]).intersection(polygon)

    if clipped.is_empty:
        return []
    pieces = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
    return [g for g in pieces
            if g.geom_type == "LineString" and g.length >= MIN_HATCH_STROKE_FT]
