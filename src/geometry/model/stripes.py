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
                                          inset_line_ft, inset_point_at_station,
                                          station_offset_many)



def curbside_strip_polygon(leg: "Leg", side: str, inner_offset_ft: float,
                            start_ft: float, end_ft: float | None = None) -> Polygon | None:
    """The strip of roadway between a leg's real curb and a line inner_offset_ft from its
    centerline, between two centerline stations.

    Both boundaries are sampled at the SAME stations, which is the whole point. The previous
    construction paired `substring(curb, start_ft, curb.length)` with
    `substring(inner, start_ft, inner.length)`, and those two substrings have nothing to do
    with each other: substring measures arc length along each line from that line's own
    start, so the curb was cut at a station 20-30 ft from where the inner line was cut, and
    the far ends differed by as much as 49 ft where the tracing ran past the leg. Closing
    that ring produced a wedge with two long diagonal ends instead of a strip, which is what
    fragmented the hatching and pushed paint outside the curb.

    Returns None where there is no room - the curb comes inside inner_offset_ft (paint would
    be outside the roadway) or the span is empty.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return None
    lo, hi = span
    lo = max(lo, start_ft)
    hi = min(hi, leg.centerline.length if end_ft is None else end_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return None

    n = max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(lo, hi, n)
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    if curb_offsets is None:
        return None

    # The inner edge never crosses outside the real curb. On several sides the traced kerb
    # comes inside the nominal half-width (broad_st_east left is traced at 22.7 ft against a
    # nominal 24.2 ft), and taking the nominal figure on faith is how paint ended up over the
    # kerb. Where the curb is inside the lane edge the strip simply pinches to nothing.
    sign = 1.0 if side == "left" else -1.0
    inner = sign * np.minimum(inner_offset_ft, np.abs(curb_offsets))

    # The outer boundary is the kerb, so it uses the KERB'S OWN coordinates rather than a
    # resampling of them - see curb_edge_by_station for why that difference is load-bearing near
    # a centerline bend. The inner boundary has no such geometry to borrow and is placed in the
    # measuring frame instead.
    outer_pts = curb_edge_by_station(leg, side, lo, hi)
    if outer_pts is None:
        return None
    inner_pts = _place_no_further_in_than(leg.centerline, stations, inner)
    poly = Polygon(list(outer_pts) + list(reversed(inner_pts)))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 1e-6) else None


def lane_narrowing_polygons_ft(leg: "Leg", stripe_width_ft: float,
                                start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                sides: tuple = ("left", "right"),
                                end_ft: float | None = None) -> list[Polygon]:
    """Two thin paint-only strips just inside each curb line - a visual lane
    narrowing treatment achieved with paint, NOT a curb_to_curb_ft change (no
    pavement/curb geometry is touched). Used by paint-only proposals - see
    src/geometry/treatments/lanes.py:LaneNarrowing.

    start_left_ft/start_right_ft trim each strip to begin past the point
    where it tapers into the corner (see lane_narrowing_taper_ft) - a real
    painted lane line doesn't stop in a straight cut at the crosswalk/
    stop-bar clearance line, it tapers back out to meet the real curb closer
    to the intersection, so this needs to line up exactly with wherever that
    taper starts on each side (which can differ between the leg's left and
    right side - each is trimmed independently). Without this, the strip's
    straight, untrimmed curb/offset lines run all the way to the
    intersection's own center point, crossing straight through the open
    intersection box where no paint actually exists.

    sides restricts which curb(s) to build a strip for - e.g. a marked-
    parking buffer (src/geometry/treatments/parking.py:MarkedParking's
    curb_offset_ft) only ever needs one side of one leg, not the symmetric
    both-sides narrowing a travel lane gets."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    polys = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        poly = curbside_strip_polygon(leg, side, inner_half, start_ft, end_ft)
        if poly is not None:
            polys.append(poly)
    return polys


def lane_narrowing_edge_lines_ft(leg: "Leg", stripe_width_ft: float,
                                  start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                  sides: tuple = ("left", "right"),
                                  keep_inside_ft: float = 0.0) -> list[LineString]:
    """The solid line marking the new, narrower travel lane's outer edge on
    each side - the same inner boundary lane_narrowing_polygons_ft's buffer
    zone starts from (11 ft from centerline for this site's proposals - see
    TARGET_LANE_WIDTH_FT in sites/broad_st_greenwood/scenarios.py) - drawn
    explicitly so the lane width actually reads on the render, rather than
    only being implied by wherever the diagonal hatching happens to start.
    start_left_ft/start_right_ft match lane_narrowing_polygons_ft's (see its
    docstring) so this line, the hatch fill, and the corner taper
    (lane_narrowing_taper_ft) all begin at the same point with no gap.
    sides - see lane_narrowing_polygons_ft's docstring."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    lines = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        line = inset_line_ft(leg, side, inner_half, start_ft, keep_inside_ft=keep_inside_ft)
        if line is not None:
            lines.append(line)
    return lines


def _corner_bulge_normal(leg: "Leg", role: str) -> np.ndarray:
    """Unit normal pointing from a leg's curb toward where a real corner
    fillet's arc bulges - the same direction that role's own curb is already
    offset from centerline ('left' for the leg_a corner role, 'right' for
    leg_b - see build_corner_fillets), just continuing further outward.
    Confirmed empirically against this project's real corner arcs (a corner
    fillet's arc sits further from centerline than the straight curb it's
    replacing, on the same side, not the opposite one)."""
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

    Extracted because lane_narrowing_taper_ft and lane_narrowing_taper_polygons_ft each carried
    a verbatim copy of it - the LINE and the FILL either side of one seam, solved twice. Two
    copies of the arc that the fill's whole purpose is to sit inside is the drift this project
    keeps paying for; they cannot disagree now.

    Held out of the travel lane at the end, because the arc is solved in WORLD space while the
    lane edge it leaves from is a fixed offset in the LEG's frame. Those are the same line only
    while the centerline is straight. Once the alignment bends onto the carriageway
    (intersection._centre_legs_on_traced_kerbs) a leg that curves toward the paint lets the arc
    cut 0.16 ft inside the 11 ft mark just after it leaves the tangent - which is real paint in
    a real travel lane, and check_paint_stays_out_of_the_travel_lane duly caught it on
    w_broad_st_northeast. Clamping the offset is the same move inset_line_ft makes against the
    kerb: the arc keeps its shape everywhere it was already outside the line.
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
    inside = np.abs(offsets) < inner_half_ft
    if not inside.any():
        return [tuple(p) for p in arc]
    offsets[inside] = sign * inner_half_ft
    return place_in_measured_frame(leg.centerline, stations, offsets)


# A taper runs from the straight run's start INWARD to the curb. When target_ft is further out
# than anchor_ft there is no room between the corner return and the crosswalk for one, and
# solving the arc anyway sweeps it backwards - which is what mangled the hatching on Princeton
# Ave's north leg (anchor 27.5 ft, target 28.6 ft) while the south leg, whose target sits
# properly inside its anchor, looked fine.
def _taper_fits(anchor_ft: float, target_ft: float) -> bool:
    return target_ft < anchor_ft


def lane_narrowing_taper_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                             n_points: int = 16, sides: tuple = ("left", "right")) -> list[LineString]:
    """Tapers a lane-narrowing buffer's straight edge line, on both sides of
    the leg, from anchor_ft (the stop-bar/clearance point where the straight
    run ends) back out to meet the REAL curb at target_ft (a point safely
    clear of the crosswalk, closer to the intersection than anchor_ft) - a
    same-leg taper, like a parking lane curving back to the curb before an
    intersection, NOT a sweep around the intersection corner to the cross
    leg. A sweep like that was tried first and doesn't work: the cross leg's
    own crosswalk sits right at the corner by definition, so any curve
    reaching all the way to the cross leg's curb inevitably cuts through it
    - there's no radius that avoids that, because the destination itself is
    inside the excluded zone. Terminating on the SAME leg, before its OWN
    crosswalk, sidesteps the problem entirely.

    The taper is tangent to the straight inset line at anchor_ft (so it
    continues the buffer's edge with no visible seam - the very thing an
    independently-computed curve, e.g. built from build_corner_fillets'
    fillet math with an unrelated radius, got wrong) and passes exactly
    through the real curb at target_ft. Tangent-at-one-point + passes-
    through-another-point + a common circle center uniquely determines the
    radius - solved directly, not guessed or borrowed from elsewhere: for
    chord d = target - anchor and outward unit normal n, R = |d|^2 / (2 *
    dot(d, n)). (For this site this R lands within ~1 ft of the real corner's
    own 20 ft radius anyway, for what it's worth - not a coincidence, just
    two ways of describing similarly-scaled curves at the same corner.)"""
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
    """The paint-only buffer's fill zone WITHIN the taper itself - same real
    source photo this whole treatment is modeled on (see lane_narrowing_taper_ft's
    docstring/PR history) shows the diagonal chevron paint continuing in the
    same pattern all the way around the curve to the curb, not stopping dead
    where the straight run ends. Bounded by the taper arc (lane_narrowing_taper_ft,
    tangent to the straight inset line at anchor_ft, passing through the real
    curb at target_ft) on one side and the real curb itself, from target_ft
    back to anchor_ft, on the other - the same curb/inset pairing
    lane_narrowing_polygons_ft uses for the straight run, just curved instead
    of straight, so hatch_lines_ft can fill it with the identical pattern and
    the two zones read as one continuous stripe with no visible seam."""
    inner_half = max(leg.curb_to_curb_ft / 2 - stripe_width_ft, 0.5)
    if not _taper_fits(anchor_ft, target_ft):
        return []
    polys = []
    for sign, role in ((1, "left"), (-1, "right")):
        if role not in sides:
            continue
        # The SAME arc lane_narrowing_taper_ft draws as the line - see _taper_arc_points. The
        # fill's entire job is to sit inside that line, so solving it twice was asking for the
        # two to disagree.
        arc_pts = _taper_arc_points(leg, role, sign, inner_half, anchor_ft, target_ft, n_points)
        if arc_pts is None:
            continue    # no taper (see lane_narrowing_taper_ft) - nothing extra to fill
        # The curb run back from target_ft to anchor_ft, sampled by STATION. `substring(curb,
        # target_ft, anchor_ft)` was arc length along the traced kerb from the kerb's own
        # start - the same confusion curb_point_at_station exists to avoid - which put this
        # edge somewhere else entirely and left the taper fill 2.3 ft over the kerb on
        # broad_st_west. arc_pts already ends at p2 (the curb at target_ft), so that
        # duplicate is dropped; Polygon() closes the ring with the segment back to p1.
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
        # past the kerb, which no amount of care about the endpoints prevents. Clipping it
        # to the roadway on this side is what actually guarantees the fill stays on the road.
        roadway = curbside_strip_polygon(leg, role, 0.0, target_ft, anchor_ft)
        if roadway is not None:
            poly = poly.intersection(roadway)
        for part in getattr(poly, "geoms", [poly]):
            if part.geom_type == "Polygon" and not part.is_empty and part.area > 1e-6:
                polys.append(part)
    return polys


def bollard_points_ft(leg: "Leg", stripe_width_ft: float, start_ft: float,
                       spacing_ft: float = 10.0, sides: tuple = ("left", "right")) -> list[tuple[float, float]]:
    """Points down the center of a paint-only buffer strip that's stripe_width_ft
    wide, next to the curb (same inner_half math as lane_narrowing_polygons_ft,
    so a bollard line always sits centered in the buffer that's actually
    painted, not a separately-guessed offset) - one line per requested side,
    starting start_ft along the centerline (past the corner fillet curve, same
    clearance convention as crosswalks/stop bars/trees - see leg_clearance_ft)
    and spaced spacing_ft apart to the end of the leg. Used by
    src/geometry/treatments/lanes.py:LaneNarrowingBollards (both sides, centered in a
    lane-narrowing buffer) and treatments/parking.py:ParkingBufferBollards (one side,
    centered in the curb_offset_ft buffer between a marked-parking lane and
    the curb - same "centered in a strip" math either way, just a different
    strip)."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    points = []
    for side, sign in (("left", 1), ("right", -1)):
        if side not in sides:
            continue
        span = curb_station_span(leg, side)
        if span is None or span[1] < start_ft:
            continue
        # Every station at once. The kerb was previously read one bollard at a time, and each
        # read re-projected the whole traced kerb into the leg frame to answer about a single
        # station. Counted rather than accumulated with +=, so the last bollard's station is
        # start + n*spacing exactly instead of the sum of n additions.
        n = int((span[1] - start_ft) // spacing_ft) + 1
        stations = start_ft + np.arange(n) * spacing_ft
        curb_off = np.abs(curb_offsets_at_stations(leg, side, stations))
        # Centered between the strip's two real boundaries at EACH station, so a bollard sits
        # in the buffer that is actually painted even where the traced kerb comes inside the
        # nominal half-width (see curbside_strip_polygon).
        lateral = (curb_off + np.minimum(inner_half, curb_off)) / 2
        points.extend(tuple(point_at(leg.centerline, float(s), sign * float(o)))
                      for s, o in zip(stations, lateral))
    return points


def parking_stall_count_ft(leg: "Leg", stall_length_ft: float, start_ft: float, end_ft: float | None = None) -> int:
    """How many full stall_length_ft stalls fit between start_ft and end_ft
    (defaults to the leg's own far end) - shared by parking_stall_lines_ft
    (which places the actual divider lines) and any caller that just wants
    the count for a label/note, so the two can never disagree."""
    end_ft = leg.centerline.length if end_ft is None else end_ft
    return max(int((end_ft - start_ft) // stall_length_ft), 0)


def parking_lane_edge_line_ft(leg: "Leg", side: str, depth_ft: float, start_ft: float,
                               end_ft: float | None = None, curb_offset_ft: float = 0.0) -> LineString | None:
    """The line marking the inner edge of a curbside marked-parking lane -
    depth_ft in from the curb (or from curb_offset_ft in from the curb, if
    the parking lane doesn't hug the curb directly - see below) on the given
    side, same start/end convention (past the corner fillet curve, see
    leg_clearance_ft) as lane_narrowing_edge_lines_ft. Real curbside parking
    doesn't always have this line painted (sometimes it's just the
    perpendicular stall ticks - see parking_stall_lines_ft), but drawing it
    makes the lane's real depth read clearly on both the plan view and the
    3D render, the same reasoning lane_narrowing_edge_lines_ft's docstring
    gives for its own edge line.

    curb_offset_ft > 0 pulls the whole parking lane in from the curb by that
    much (see src/geometry/treatments/parking.py:MarkedParking) - e.g. a striped
    no-parking buffer between the parking lane and the curb itself, so
    parking sits directly against the active travel lane instead of against
    the curb. Defaults to 0 (the lane starts right at the curb, as before)."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - curb_offset_ft - depth_ft, 0.5)
    # No room left on this leg to mark parking. Happens where the corner return eats the
    # whole leg: at W Broad & Louellen's acute Y, leg_clearance_ft comes out at 133 ft on a
    # 130 ft leg, so parking would start past the end of the road. Callers that interpolate
    # along an empty geometry fail with an unhelpful shapely type error, so inset_line_ft
    # says "nothing here" explicitly instead.
    return inset_line_ft(leg, side, inner_half, start_ft, end_ft)


def parking_stall_lines_ft(leg: "Leg", side: str, depth_ft: float, stall_length_ft: float, start_ft: float,
                            end_ft: float | None = None, curb_offset_ft: float = 0.0) -> list[LineString]:
    """Perpendicular divider lines bounding each marked parallel-parking
    stall along one side of a leg - the standard MUTCD curbside-parking
    marking: a short tie line at each stall boundary, not a filled/hatched
    zone (this is a real parking lane, not a paint-only buffer like
    lane_narrowing/corner_hatching - a driver is meant to park a real vehicle
    inside each one). One extra divider beyond the last full stall closes it
    off, so n stalls always get n+1 lines (see parking_stall_count_ft for the
    same n used elsewhere, e.g. a dimension label).

    Each divider normally runs from the real curb to depth_ft in from it;
    curb_offset_ft > 0 (see parking_lane_edge_line_ft) shifts BOTH ends in by
    that much instead, so the divider spans the parking lane itself, not the
    no-parking buffer between it and the curb."""
    half = leg.curb_to_curb_ft / 2
    sign = 1 if side == "left" else -1
    outer_off = max(half - curb_offset_ft, 0.5)
    inner_off = max(half - curb_offset_ft - depth_ft, 0.5)
    span = curb_station_span(leg, side)
    if span is None:
        return []
    end_ft = min(span[1], leg.centerline.length if end_ft is None else end_ft)
    n_stalls = parking_stall_count_ft(leg, stall_length_ft, start_ft, end_ft)
    # Station, not distance along an offset curve - see inset_line_ft. A divider is a
    # cross-section of the parking lane, so both ends have to be at the same station. All the
    # stations are read from the kerb in one pass rather than one per divider.
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
# near the junction and tapering it back out. That is the whole mechanism, and it is worth
# stating because the obvious-looking alternative does nothing: re-cutting the corner ARC at a
# smaller radius (set_corner_radius, formerly and misleadingly called bump_out) leaves both
# curb lines exactly where they were. Measured on broad_st_east x greenwood_ave_north at
# 29.2 -> 15.0 ft:
#
#     arc length     19.48 -> 3.51 ft     the arc really is re-cut
#     trimmed_a     156.19 -> 164.19 ft   the curb just extends to the new tangent point
#     pavement area  23,989.7 -> 23,989.5 sq ft      0.2 sq ft of 24,000
#     crossing spans unchanged to 0.00 ft on all four legs
#
# The crossings here sit 21-42 ft out, past the corner, so a radius change never reaches them.
#
# The extension is measured from the leg's NOMINAL half-width, not from the traced kerb at that
# station, and the difference matters. The traced kerb flares through the corner return -
# broad_st_east's kerbs are 39.4 and 31.6 ft off the centerline where its crossing is painted,
# against a 26.0 ft nominal half-width - so the crossing today spans 65.0 ft of pavement, not
# the 52.0 ft the cross-section suggests. Extending from the nominal half-width replaces that
# flare with the extension's own straight face, which is what a built bulb-out does, and it is
# why the crossing falls further than the extension alone would imply: 8 ft of extension per
# side takes broad_st_east from 65.0 ft to about 2 x (26.0 - 8) = 36 ft.
#
# How far a curb extension may be pushed is bounded by the travel lane it must leave behind,
# so every caller is checked against TARGET_LANE_WIDTH_FT - see
# src/geometry/treatments/corners.py:AddCurbExtension.

# How gently the extension returns to the real kerb: feet along the leg per foot of lateral
# shift. A DESIGN CHOICE, not a measured or standard figure - flagged like
# PARKING_BUFFER_DEFAULT_FT rather than dressed up as a citation. 5:1 is at the gentle end of
# what low-speed parking-lane transitions use, and the check that matters is not the rate but
# the total: face plus taper has to stay inside the length of kerb where parking is already
# prohibited, or the extension removes a space. See
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

    The face never sits outside the traced kerb. Where the real kerb is already inside the
    nominal half-width - which happens mid-block, broad_st_east's left kerb is traced at
    22.7 ft against a 24.2 ft nominal - the tracing wins and no extension is built there. An
    extension is only ever allowed to take roadway, never to invent it.
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
    # side's kerb is still the surveyor's, and it should stay vertex-for-vertex theirs.
    points += [point_at(leg.centerline, float(s), float(o))
               for s, o in zip(curb_stations, curb_offsets) if s > taper_end_ft]
    return LineString(points) if len(points) >= 2 else None


# A hatch stroke shorter than this is a clipping artifact, not paint. They appear where a
# stroke grazes a corner of the polygon or crosses the needle-thin tip of a taper, and they
# render as stubs - the "sheared in half" strokes. One came out 0.0 ft long.
MIN_HATCH_STROKE_FT = 1.0


def clip_paint_clear_of(geometry, keep_clear):
    """Cut `keep_clear` out of a piece of paint, returning the surviving pieces.

    Road markings are layered by priority, and a crosswalk outranks a buffer or a parking
    lane - export.py has said so in a comment since long before anything enforced it. Doing
    the subtraction on the GEOMETRY is what makes it true, rather than relying on the paint's
    start station being far enough out: a skewed crossing reaches further along one kerb than
    its centre offset suggests, which is how two hatch strokes ended up over Broad St's
    crossing while the arithmetic said they cleared it.
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
    """Diagonal hatch lines filling a polygon, clipped to its boundary - used
    to render paint-only diagonal/chevron marking (e.g. corner_hatching_polygon
    above) without any real curb/pavement geometry change.

    phase_origin fixes WHERE the family of parallel lines falls, in world coordinates. It
    matters because a buffer is not one polygon: the straight run, the taper into the
    corner, and whatever survives being cut around a crossing are separate polygons hatched
    separately. Phasing each family off its own bounding box centre - which is what this did
    - gave each piece an independent stroke position, so at every seam the strokes stepped
    sideways by some fraction of the spacing. Reading across the seam, one stroke looked
    sheared into two offset halves. Passing all the pieces of one treatment the same origin
    puts them on one continuous set of lines, and the seams disappear.
    """
    # A corner-hatch polygon built off a traced kerb can pinch to a point where the curb
    # doubles back on itself, which is a bowtie GEOS refuses to intersect against. buffer(0)
    # resolves it into the same area without moving any edge; an empty result means the
    # polygon had no area to hatch in the first place.
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return []

    minx, miny, maxx, maxy = polygon.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-u[1], u[0]])

    # Which lines of the (infinite, origin-anchored) family actually reach this polygon:
    # the range of the corners' distances along n, snapped outward to whole multiples of the
    # spacing. Anchoring on multiples of the spacing from a shared origin is what keeps
    # neighbouring pieces in phase.
    origin = np.asarray(phase_origin, dtype=float)
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]) - origin
    along_n, along_u = corners @ n, corners @ u
    steps = np.arange(np.floor(along_n.min() / spacing_ft), np.ceil(along_n.max() / spacing_ft) + 1)

    # Every hatch line at once, and one clip against the polygon instead of one GEOS call
    # per line. Endpoints are built with numpy broadcasting; the whole family goes through
    # a single MultiLineString intersection. Each line must span the polygon's extent ALONG
    # u as well as sit at the right distance along n - the phase origin is the state-plane
    # origin, half a million feet away, so a segment merely centred on it never reaches.
    centers = origin + n * (steps * spacing_ft)[:, None]
    lo, hi = along_u.min() - diag, along_u.max() + diag
    ends = np.stack([centers + u * lo, centers + u * hi], axis=1)
    clipped = MultiLineString([tuple(map(tuple, pair)) for pair in ends]).intersection(polygon)

    if clipped.is_empty:
        return []
    pieces = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
    return [g for g in pieces
            if g.geom_type == "LineString" and g.length >= MIN_HATCH_STROKE_FT]
