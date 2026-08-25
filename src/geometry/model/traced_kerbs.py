"""Turning a surveyor's traced kerb into geometry this project can measure against.

Circle fitting and the tests for whether a fit is a corner return at all, arc smoothing, and the
assignment of loose kerb points to the leg and side they belong to. This is the messiest input the
pipeline takes - OSM ways start and stop arbitrarily, cover part of a corner, or cover both sides
of a street in one way - so most of what is here is about REFUSING a bad fit rather than making
one."""

import itertools

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring
from src.geometry.model.leg_frame import (Leg, _leg_bearing, line_direction,
                                          place_in_measured_frame, placement_holds, vertex_tangents,
                                          station_offset_many)



# Gates on a circle fitted to a traced kerb, before its radius is trusted as a corner
# radius. A short arc barely constrains a circle: 8 ft of kerb spanning 36 degrees
# happened to fit well at Columbia & Princeton, but only because the tracing was careful.
MIN_KERB_ARC_SWEEP_DEG = 25.0
MAX_KERB_FIT_RESIDUAL_FT = 1.0
PLAUSIBLE_CORNER_RADIUS_FT = (5.0, 60.0)  # outside this it isn't a street corner return


def fit_circle_ft(line: LineString) -> dict | None:
    """Least-squares (Kasa) circle fit to a traced kerb line.

    Returns {"radius_ft", "center", "sweep_deg", "max_residual_ft"} or None if the fit is
    degenerate. `sweep_deg` is how much of the circle the trace actually covers and is the
    key quality signal - a wide sweep with a small residual is a trustworthy radius; a
    narrow sweep is a circle inferred from almost-straight input.
    """
    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return None
    xs, ys = coords[:, 0], coords[:, 1]
    a_matrix = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    try:
        cx, cy, c = np.linalg.lstsq(a_matrix, xs ** 2 + ys ** 2, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    inner = c + cx ** 2 + cy ** 2
    if inner <= 0:
        return None
    radius = float(np.sqrt(inner))
    residual = float(np.abs(np.hypot(xs - cx, ys - cy) - radius).max())
    angles = np.unwrap(np.sort(np.arctan2(ys - cy, xs - cx)))
    sweep = float(np.degrees(angles.max() - angles.min()))
    return {"radius_ft": radius, "center": (float(cx), float(cy)),
            "sweep_deg": sweep, "max_residual_ft": residual}


def kerb_radius_is_usable(fit: dict | None) -> bool:
    """Whether a circle fit is well-enough conditioned to use as a corner radius."""
    if fit is None:
        return False
    low, high = PLAUSIBLE_CORNER_RADIUS_FT
    return (fit["sweep_deg"] >= MIN_KERB_ARC_SWEEP_DEG
            and fit["max_residual_ft"] <= MAX_KERB_FIT_RESIDUAL_FT
            and low <= fit["radius_ft"] <= high)


def assign_kerbs_to_corners(legs: dict, kerb_lines_ft: list) -> dict:
    """{frozenset(leg_a, leg_b): [LineString, ...]} - traced kerbs grouped by the corner
    they belong to, matched by which two legs their midpoint sits closest to."""
    by_corner: dict[frozenset, list] = {}
    for line in kerb_lines_ft:
        midpoint = line.interpolate(0.5, normalized=True)
        ranked = sorted(legs.items(), key=lambda kv: kv[1].centerline.distance(midpoint))
        if len(ranked) < 2:
            continue
        by_corner.setdefault(frozenset((ranked[0][0], ranked[1][0])), []).append(line)
    return by_corner


# A traced kerb is hand-clicked, so its vertices carry the mapper's noise: used raw it
# renders as a visibly kinked corner. Smoothing replaces that noise while keeping the
# traced POSITION and endpoints, which is the whole point of using the tracing.
SMOOTHED_ARC_POINTS = 24
MAX_ARC_FIT_RESIDUAL_FT = 1.5  # beyond this the kerb isn't really circular; smooth it instead


def _chaikin(line: LineString, iterations: int = 5) -> LineString:
    """Chaikin corner-cutting. Endpoints are preserved exactly; interior vertices are
    repeatedly replaced by points 1/4 and 3/4 along each segment, which converges on a
    smooth curve. Used where a kerb isn't circular enough to fit an arc."""
    coords = np.asarray(line.coords, dtype=float)
    for _ in range(iterations):
        if len(coords) < 3:
            break
        # Both cut points for every segment in one shot; interleaved with reshape rather
        # than appended one at a time (the vertex count quadruples each iteration).
        starts, ends = coords[:-1], coords[1:]
        cuts = np.empty((2 * len(starts), 2))
        cuts[0::2] = 0.75 * starts + 0.25 * ends
        cuts[1::2] = 0.25 * starts + 0.75 * ends
        coords = np.vstack([coords[:1], cuts, coords[-1:]])
    return LineString(coords)


def smooth_traced_arc(line: LineString) -> LineString:
    """A clean curve following a traced kerb: same endpoints, same path, no click noise.

    Preferred method is to fit a circle to the traced points and redraw the arc between
    the traced ENDPOINTS along that circle. That is smooth by construction and still sits
    on the mapped kerb - unlike the old fitted fillet, which took only the radius and then
    redrew the arc off our own estimated curb lines, landing it feet away.

    Falls back to Chaikin smoothing where the kerb isn't circular enough to fit (a
    compound return, or a trace covering more than one curve).
    """
    fit = fit_circle_ft(line)
    if fit is None or fit["max_residual_ft"] > MAX_ARC_FIT_RESIDUAL_FT:
        return _chaikin(line)

    cx, cy = fit["center"]
    radius = fit["radius_ft"]
    start, end = np.array(line.coords[0]), np.array(line.coords[-1])
    mid = np.array(line.interpolate(0.5, normalized=True).coords[0])
    a0 = np.arctan2(start[1] - cy, start[0] - cx)
    a1 = np.arctan2(end[1] - cy, end[0] - cx)
    a_mid = np.arctan2(mid[1] - cy, mid[0] - cx)

    # Two ways round the circle; take the one that actually passes through the traced
    # midpoint, so a reflex return isn't silently replaced by the short way round.
    candidates = [(a1 - a0) % (2 * np.pi), (a1 - a0) % (2 * np.pi) - 2 * np.pi]
    def passes_mid(sweep):
        t = ((a_mid - a0) / sweep) if sweep else 0.0
        return 0.0 <= t <= 1.0
    sweep = next((c for c in candidates if passes_mid(c)), min(candidates, key=abs))

    angles = a0 + np.linspace(0, sweep, SMOOTHED_ARC_POINTS)
    # Deliberately NOT pinned back to the raw traced endpoints. a0/a1 are already those
    # endpoints projected onto the fitted circle, so the arc starts and ends within the
    # fit residual (<=1.5 ft) of where they were traced. Snapping the ends back to the
    # raw points put a kink at each end - it made the smoothed arc turn MORE sharply than
    # the trace it was meant to clean up. The curb lines trim to the arc's own ends, so
    # nothing downstream needs the raw endpoints.
    return LineString([(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles])


# How much of each traced curb either side of a corner is handed to the smoothing pass.
# The traced corner returns already live in the leg curbs (assign_curb_points_to_legs puts
# each return's vertices on the two sides it joins), so this only has to take the click
# noise off the join, not invent a curve.
CORNER_BLEND_FT = 8.0


def traced_corner_join(curb_a: LineString, curb_b: LineString) -> tuple[LineString, LineString, LineString]:
    """Join two traced curbs around the corner they share, smoothing the seam.

    Both curbs are the surveyor's own traced kerb, ending where the tracing ends - which for
    a mapped corner is partway around the return. So there is no corner to construct: the
    two ends are already at the corner and this walks from one to the other, taking the last
    CORNER_BLEND_FT of each, bridging whatever gap the tracing left, and Chaikin-smoothing
    the result. Returns the same (trimmed_a, arc, trimmed_b) contract as fillet_curb_corner,
    with the arc running from curb_a's side to curb_b's side.
    """
    blend_a = min(CORNER_BLEND_FT, curb_a.length / 2)
    blend_b = min(CORNER_BLEND_FT, curb_b.length / 2)
    head_a = substring(curb_a, 0, blend_a)
    head_b = substring(curb_b, 0, blend_b)
    seam = LineString(list(head_a.coords)[::-1] + list(head_b.coords))
    return substring(curb_a, blend_a, curb_a.length), _chaikin(seam), substring(curb_b, blend_b, curb_b.length)


def traced_corner_arc(kerb_lines: list, curb_a: LineString, curb_b: LineString) -> LineString | None:
    """One traced kerb, oriented to run from curb_a's side to curb_b's side.

    build_corner_fillets' contract is (trimmed_a, arc, trimmed_b) with the arc running
    from its tangent point on curb_a to the one on curb_b, and build_pavement_polygon's
    ring walk depends on that order. A traced kerb has whatever direction the mapper drew
    it in, so it is reversed if needed. Where several kerbs share a corner the longest is
    used - the others are usually short ramp segments rather than the return itself.
    """
    usable = [line for line in kerb_lines if line.length > 1.0]
    if not usable:
        return None
    line = max(usable, key=lambda l: l.length)
    start, end = Point(line.coords[0]), Point(line.coords[-1])
    if start.distance(curb_a) > end.distance(curb_a):
        line = LineString(list(line.coords)[::-1])
    return smooth_traced_arc(line)


def corner_radii_from_kerbs(legs: dict, kerb_lines_ft: list[LineString],
                             fallback_radius_ft: float) -> tuple[dict, list[str]]:
    """Per-corner radii derived from traced OSM kerb lines, plus notes on what happened.

    Returns ({frozenset(leg_a, leg_b): radius_ft}, notes). Corners with no usable traced
    kerb are simply absent - the caller uses `fallback_radius_ft` for those, so a site
    with one traced corner gets one real radius and keeps the placeholder elsewhere
    rather than having one corner's measurement spread over the whole junction.

    A kerb way is assigned to the corner between the two legs it sits closest to. Several
    traces at one corner are combined by median, which is what makes two independent
    tracings of the same return (13.6 and 13.4 ft at Columbia & Princeton) reinforce each
    other, and stops a single odd trace from deciding the answer alone.
    """
    by_corner: dict[frozenset, list[float]] = {}
    notes: list[str] = []

    for line in kerb_lines_ft:
        fit = fit_circle_ft(line)
        midpoint = line.interpolate(0.5, normalized=True)
        ranked = sorted(legs.items(), key=lambda kv: kv[1].centerline.distance(midpoint))
        if len(ranked) < 2:
            continue
        corner = frozenset((ranked[0][0], ranked[1][0]))
        if not kerb_radius_is_usable(fit):
            reason = ("degenerate fit" if fit is None else
                      f"sweep {fit['sweep_deg']:.0f} deg, residual {fit['max_residual_ft']:.2f} ft, "
                      f"radius {fit['radius_ft']:.1f} ft")
            notes.append(f"kerb trace at {'/'.join(sorted(corner))} not usable as a corner radius "
                          f"({reason}) - too short an arc, too poor a fit, or not a corner return.")
            continue
        by_corner.setdefault(corner, []).append(fit["radius_ft"])

    radii = {}
    for corner, values in by_corner.items():
        radius = float(np.median(values))
        radii[corner] = radius
        spread = f", {len(values)} traces spanning {min(values):.1f}-{max(values):.1f} ft" if len(values) > 1 else ""
        notes.append(f"corner {'/'.join(sorted(corner))}: radius {radius:.1f} ft from traced OSM kerb"
                      f"{spread} (placeholder was {fallback_radius_ft:.0f} ft).")

    # Untraced corners: prefer the median of THIS junction's own measured corners over the
    # site-wide placeholder. A generic 20 ft next to corners measured at 13.5 ft inflates
    # the modelled throat enough to swallow the real footway - which is what was dropping
    # tactile pads at Columbia & Princeton. Still an inference, but one drawn from the same
    # junction rather than from a typical-value assumption, and reported as such.
    if radii:
        local_default = float(np.median(list(radii.values())))
        untraced = [frozenset((a, b)) for a, b in _corner_pairs(legs)
                     if frozenset((a, b)) not in radii]
        if untraced and abs(local_default - fallback_radius_ft) > 1.0:
            for corner in untraced:
                radii[corner] = local_default
            names = ", ".join("/".join(sorted(c)) for c in untraced)
            notes.append(f"untraced corner(s) {names}: using {local_default:.1f} ft, the median of this "
                          f"junction's own traced corners, instead of the {fallback_radius_ft:.0f} ft "
                          f"site placeholder. Trace them to replace this.")
    return radii, notes


def _corner_pairs(legs: dict) -> list[tuple[str, str]]:
    """Angularly adjacent leg pairs - the same corners build_corner_fillets() forms."""
    usable = {name: leg for name, leg in legs.items() if leg.left_curb is not None}
    ordered = sorted(usable.items(), key=lambda kv: _leg_bearing(kv[1]))
    return [(ordered[i][0], ordered[(i + 1) % len(ordered)][0]) for i in range(len(ordered))]


# Building a leg's curb from the surveyor's traced kerb ways.
#
# EVERY traced kerb way is curb. The earlier version took only kerb=raised and only the
# single longest run per side, which threw away exactly the geometry that matters: the
# corner returns are tagged kerb=lowered (they're the ramps), so the SW corner of Broad &
# Greenwood - traced in full - was being dropped and redrawn as a fitted fillet off the
# NJDOT centerline. Raised vs lowered is a height, not a question of where the curb is.
#
# Each traced VERTEX is placed in the leg frame as (station along the centerline, signed
# offset from it), and assigned to the one leg side whose half-width it best matches. That
# splits a corner return between the two sides it joins, which is what a corner return is.
# The curb is then the traced points themselves, in station order - no offsetting, no
# fitting, no fillet. Nothing is invented except where nothing was traced.
CURB_POINT_MAX_WIDTH_RATIO = 2.6   # |offset| / half-width; corner returns flare to ~2.3x
CURB_POINT_MIN_WIDTH_RATIO = 0.45  # below this it's a median or a driveway, not this curb
CURB_POINT_BEHIND_TOLERANCE_FT = 3.0
# A vertex a little behind a leg's junction node is still claimable - a corner return's own
# geometry straddles station 0, and dropping those vertices loses the corner. But a leg must
# never outbid one the vertex lies IN FRONT of, and unpenalised it can: at E Broad & Princeton
# the two legs are 179.9 deg apart, and the vertex where East Broad's north kerb changes from
# the corner return to the straight run sits 0.8 ft ahead of e_broad_st_east and 0.8 ft BEHIND
# e_broad_st_west - on the far side of the intersection from it. The west leg's half-width
# happened to match a shade better (0.995 vs 1.010), so it took the vertex; the 58.3 ft way it
# was the near end of then had one point left, curb_line_from_points needs two, and the whole
# stretch was discarded. That left e_broad_st_east's north kerb "traced only from 59 ft out"
# and 58 ft of a surveyed no-stopping kerb unhatched.
#
# Larger than any ratio the window admits (2.6), so forward always beats behind and the ratio
# only ever breaks ties among legs that all have the vertex ahead of them.
CURB_POINT_BEHIND_PENALTY = 10.0
# Out along a leg, past its corner returns, a kerb that IS that leg's kerb runs along it.
# Offset alone can't tell the difference: at W Broad & Louellen a kerb swinging from 16 ft
# to 37 ft off Louellen's alignment over 60 ft - a driveway apron running away from the
# street - sits inside any offset window wide enough to admit the real south kerb at 34 ft,
# and claiming it measured the leg at 66 ft. A kerb 53 degrees off the street is not the
# street's edge. Inside the corner zone the test is suspended, because a corner return
# sweeps through 90 degrees by definition and is still curb.
CURB_POINT_MAX_SKEW_DEG = 30.0
# How far out the skew test is suspended at a SQUARE corner. Not a distance along the street so
# much as a statement about how far a corner return reaches, which is why it is not used raw -
# see _corner_zone_ft.
CURB_POINT_CORNER_ZONE_FT = 40.0
# The widest a sharp corner may stretch that zone. A leg meeting another at a few degrees would
# otherwise exempt its whole length from the skew test, which would readmit exactly the driveway
# aprons CURB_POINT_MAX_SKEW_DEG exists to reject.
CURB_POINT_CORNER_ZONE_MAX_SCALE = 3.0
# What a skewed vertex outside the corner zone costs, instead of being thrown away outright.
# DISQUALIFYING IT WAS THE BUG. The skew test cannot tell a corner return still sweeping from a
# driveway apron, and at a 48 degree junction the return is still sweeping at 66 ft - so at W
# Broad & Louellen it discarded 6 of the 8 traced vertices on louellen_st_west's left kerb, the
# corner was bridged with an invented line, and the hatching was drawn against that line.
# Exempting a longer zone instead let one leg outbid its neighbour for the SHARED return and cost
# w_broad_st_southwest its right kerb entirely - more data in, less data used. A penalty does
# neither: a skewed vertex is claimable, but any leg that has it unskewed takes it first. Same
# construction and the same reasoning as CURB_POINT_BEHIND_PENALTY, and larger than any ratio the
# window admits so the two orderings never interleave.
CURB_POINT_SKEW_PENALTY = 20.0


def corner_return_scale(leg, legs) -> float:
    """How much longer this leg's corner returns are than a square junction's. 1.0 at 90 degrees.

    THE FLAT 40 FT WAS FITTED TO SQUARE JUNCTIONS AND THEN APPLIED TO AN ACUTE ONE. A fillet of
    radius R tangent to two lines meeting at interior angle theta runs R/tan(theta/2) along each
    of them: at 90 degrees that is R, and at W Broad & Louellen's 48 degrees it is 2.25 R. So a
    return that is still curving at 66 ft out is ordinary geometry at that junction, and the
    skew test - which cannot tell a sweeping return from a driveway apron - was discarding it as
    an apron. `louellen_st_west` left kept 2 traced vertices of 8 and was reported "traced only
    from 60 ft out", so the corner was bridged with an invented line and the hatching drawn
    against it.

    Scaled by the SHARPEST corner this leg is part of, because the leg only needs one acute
    neighbour to have one long return. A straight-through pair (E Broad's legs are 179.9 degrees
    apart) has no corner between them and would scale to nothing, so the minimum over the other
    legs is what carries that leg's real corner with the third one.

    Floored at 1.0, so no junction gets a SHORTER reach than it has today and a widening cannot
    silently drop a vertex some site currently keeps.

    A SCALE rather than a distance because two constants encode this same fact against different
    bases - the skew test's CURB_POINT_CORNER_ZONE_FT and the width fit's
    fitting.TRACED_SECTION_START_FT - and they must stretch together. Widening only the first
    admitted return vertices to the curb LINE while the width fit still measured its
    cross-section from 35 ft, straight through the flare: louellen_st_west came out 66.5 ft wide
    against a real 42.1.
    """
    direction = line_direction(leg.centerline)
    sharpest_deg = 180.0
    for other in legs.values():
        if other is leg:
            continue
        # Both centerlines run OUTWARD from the junction, so the angle between them is the
        # interior angle at the corner - no folding, and the sign carries no information here.
        cosine = float(np.clip(direction @ line_direction(other.centerline), -1.0, 1.0))
        sharpest_deg = min(sharpest_deg, float(np.degrees(np.arccos(cosine))))
    if sharpest_deg >= 180.0:
        return 1.0
    scale = 1.0 / max(np.tan(np.radians(sharpest_deg / 2)), 1e-6)
    return float(np.clip(scale, 1.0, CURB_POINT_CORNER_ZONE_MAX_SCALE))


def assign_curb_points_to_legs(legs: dict, kerb_lines: list[LineString],
                                ratio_bounds: tuple[float, float] | None = None) -> dict:
    """{leg_name: {"left": [(station, offset), ...], "right": [...]}} from traced kerbs.

    Every vertex of every traced kerb way is considered, and goes to the single leg side
    whose half-width it sits closest to in proportional terms. One vertex can only be one
    piece of curb, so a corner return splits between the two sides that meet there rather
    than being drawn twice.

    Vectorized over vertices: each leg scores every traced vertex in one pass, and the
    winning leg per vertex is an argmin over the resulting (legs x vertices) score matrix.

    `ratio_bounds` widens (or narrows) the window a vertex has to fall in to be claimed at
    all. It exists because judging a vertex against a width the caller is only about to
    measure FROM that vertex is circular, and the circularity bites both ways at W Broad &
    Louellen: with the window at its normal width, Louellen St's south kerb - 155 ft of it,
    at a steady 34 ft offset - sat at 3.5x the half-width then assumed and was discarded, so
    the leg measured 19 ft wide off its north kerb alone; and W Broad's near kerb, 6.5 ft off
    NJDOT's badly off-centre alignment, sat at 0.43x and was discarded as a median. Opening
    the window admits both, and the proportional scoring still hands each vertex to the leg
    it best fits. See src/geometry/intersection/fitting.py:_fit_legs_to_traced_kerbs.
    """
    if not kerb_lines:
        return {}
    pts = np.concatenate([np.asarray(line.coords, dtype=float) for line in kerb_lines])
    tangents = np.concatenate([vertex_tangents(line) for line in kerb_lines])
    low, high = ratio_bounds or (CURB_POINT_MIN_WIDTH_RATIO, CURB_POINT_MAX_WIDTH_RATIO)
    min_cosine = np.cos(np.radians(CURB_POINT_MAX_SKEW_DEG))

    names, stations, offsets, ratios = [], [], [], []
    for name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        leg_stations, leg_offsets = station_offset_many(leg.centerline, pts)
        ratio = np.abs(leg_offsets) / (leg.curb_to_curb_ft / 2)
        # abs: a kerb traced against the leg's outward direction is still parallel to it.
        skewed = np.abs(tangents @ line_direction(leg.centerline)) < min_cosine
        # np.inf marks "this leg can't claim this vertex", so it never wins the argmin.
        # THE SKEW TEST HAS THREE TIERS, and collapsing any two of them broke a junction:
        #   * inside CURB_POINT_CORNER_ZONE_FT it is suspended - a return sweeps through 90
        #     degrees by definition and is still curb;
        #   * out to where THIS junction's returns actually reach (corner_return_scale) a
        #     skewed vertex is doubted, not rejected: claimable, but any leg that has it
        #     unskewed takes it first;
        #   * past that it is an apron and disqualified, as before.
        # Suspending it outright across the whole scaled reach - the obvious two-tier version -
        # let louellen_st_west claim 13 vertices at its seed width and strip
        # w_broad_st_southwest's right kerb to one, because inside a suspended zone the two legs
        # were judged on ratio alone and a badly seeded width won the contest.
        reach_ft = CURB_POINT_CORNER_ZONE_FT * corner_return_scale(leg, legs)
        doubted = skewed & (leg_stations > CURB_POINT_CORNER_ZONE_FT)
        disqualified = ((leg_stations < -CURB_POINT_BEHIND_TOLERANCE_FT)
                        | (ratio < low) | (ratio > high)
                        | (skewed & (leg_stations > reach_ft)))
        # Still claimable behind the node, but only if nobody has it in front - see
        # CURB_POINT_BEHIND_PENALTY.
        score = (ratio + np.where(leg_stations < 0, CURB_POINT_BEHIND_PENALTY, 0.0)
                 + np.where(doubted, CURB_POINT_SKEW_PENALTY, 0.0))
        names.append(name)
        stations.append(leg_stations)
        offsets.append(leg_offsets)
        ratios.append(np.where(disqualified, np.inf, score))
    if not names:
        return {}

    ratios = np.vstack(ratios)
    winner = np.argmin(ratios, axis=0)
    claimed = np.isfinite(ratios[winner, np.arange(ratios.shape[1])])

    out: dict[str, dict[str, list]] = {}
    for leg_index, name in enumerate(names):
        mine = claimed & (winner == leg_index)
        if not mine.any():
            continue
        leg_stations, leg_offsets = stations[leg_index][mine], offsets[leg_index][mine]
        for side, on_side in (("left", leg_offsets > 0), ("right", leg_offsets <= 0)):
            if on_side.any():
                out.setdefault(name, {})[side] = list(
                    zip(leg_stations[on_side].tolist(), leg_offsets[on_side].tolist()))
    return out


# Extrapolating past the end of the tracing. A curb that leaves the corner is running down
# the street, so it can diverge from the centerline by a few degrees (NJDOT's alignment
# error) but not more. Taking the slope off the last two traced vertices instead read the
# flare of a corner return - at Columbia & Princeton the south leg is traced for only 9 ft,
# all of it return, and running that slope out 100 ft crossed the two curbs into an X.
CURB_EXTRAPOLATION_MAX_SLOPE = 0.11        # ~6 degrees
CURB_EXTRAPOLATION_MIN_BASELINE_FT = 15.0  # shorter than this is corner, not street


# How finely the stretch BETWEEN two traced vertices is sampled. A traced kerb is a profile
# in the leg's frame - (station, offset) pairs - so a straight WORLD segment joining two of
# them means "constant offset" only while the centerline runs straight between them. OSM
# traces the block and not the kerb (SKILLS.md #7), so those gaps get long: greenwood_ave_south
# is traced to station 113.9 and then not again until the frame cuts the way at 387.8, and the
# alignment curves through the 274 ft in between.
#
# ONE KERB, READ TWO WAYS, IN TWO FRAMES. The pavement ring, the corner fillets and the 3D
# kerb are built from this LINE; every marking is placed against curb_offsets_at_stations,
# which interpolates the SAME vertices in the leg's frame. Across that gap the chord stood
# ~10 ft from the profile, and nothing downstream could see it, because each side was
# internally consistent - the rendered leg carried 10 ft of bare asphalt outside the hatching
# on one kerb and hatching over the sidewalk on the other. Sampling the gap in the frame the
# kerb is defined in makes the two the same line again.
#
# The step is a chord-height budget, not a taste: a sample every 10 ft departs an arc of
# radius R by 100/(8R) ft, so 0.02 ft on the ~940 ft curve above and less on anything
# straighter. Where the centerline IS straight the samples land on the chord and nothing moves,
# which is why this changes no other site.
CURB_GAP_SAMPLE_FT = 10.0

# How far the alignment has to pull away from a traced chord before the chord is read as MISSING
# THE ROAD rather than describing it.
#
# BETWEEN TWO TRACED VERTICES THE OSM WAY IS A STRAIGHT WORLD SEGMENT. That is the datum, not an
# artefact of it, so bending it toward the alignment everywhere makes a well-traced kerb WORSE:
# sampled unconditionally, louellen_st_west's right kerb left the way it was traced from by 1.31
# ft, and broad_st_east's by 0.16, where both had lain exactly on it. Only where the tracing is
# so sparse that it cannot be describing the road does the alignment know better.
#
# The two cases separate cleanly, and the survey behind stripes.py's chord_deviates is the
# measurement: greenwood_ave_south's two sides stand 9.6-10.9 ft off the smooth curve and every
# other leg-side in this dataset under 1.2 ft. So 2 ft sits in an empty gap between hand-tracing
# noise and a chord cutting a corner off the street - and a leg on the near side of it keeps
# EXACTLY the kerb it had, with its pavement ring, corner fillets, frame and goldens.
CURB_CHORD_DEVIATION_FT = 2.0


def _outward_slope(points: list[tuple[float, float]]) -> float:
    """d(offset)/d(station) for the outward end of a traced side, or 0 if the tracing is
    too short to establish one - in which case the curb continues at the width last seen."""
    end_station, end_offset = points[-1]
    for station, offset in reversed(points[:-1]):
        if end_station - station >= CURB_EXTRAPOLATION_MIN_BASELINE_FT:
            slope = (end_offset - offset) / (end_station - station)
            return float(np.clip(slope, -CURB_EXTRAPOLATION_MAX_SLOPE, CURB_EXTRAPOLATION_MAX_SLOPE))
    return 0.0


def _inward_slope(points: list[tuple[float, float]]) -> float:
    """d(offset)/d(station) for the JUNCTION end of a traced side. Mirror of _outward_slope."""
    start_station, start_offset = points[0]
    for station, offset in points[1:]:
        if station - start_station >= CURB_EXTRAPOLATION_MIN_BASELINE_FT:
            slope = (offset - start_offset) / (station - start_station)
            return float(np.clip(slope, -CURB_EXTRAPOLATION_MAX_SLOPE, CURB_EXTRAPOLATION_MAX_SLOPE))
    return 0.0


def curb_line_from_points(points: list[tuple[float, float]], leg: "Leg",
                          working_length_ft: float,
                          extend_to_junction: bool = False) -> LineString | None:
    """One leg side's curb, straight off the traced points.

    The points are the surveyor's own vertices, kept as traced and ordered along the leg.
    The outward end is extended along the bearing of the last traced stretch to reach the
    leg's working length, when the tracing stops short of it.

    The junction end is normally left exactly where the tracing ends - the corner is built
    from the traced geometry there, not by running this line on into the intersection.
    `extend_to_junction` lifts that for a side with NO CORNER RETURN, where the kerb genuinely
    runs straight through and stopping short is the fabrication. The north side of E Broad at
    Princeton is one unbroken kerb; the OSM way covering its last 20 ft before the junction has
    only two vertices, one of which the collinear leg on the far side legitimately claims, so
    the west leg's curb began 20.7 ft out and its no-stopping hatching could not be built
    inside that. Extending it in is the same extrapolation the outward end already gets, along
    a bearing the tracing establishes over 60+ ft of straight kerb - see through_street_sides
    for what licenses it.
    """
    ordered = sorted(points)
    if len(ordered) < 2:
        return None
    # One vertex per station: two traced ways can share an endpoint, and a curb that
    # doubled back in station would fold the pavement edge over itself.
    deduped = [ordered[0]]
    for station, offset in ordered[1:]:
        if station - deduped[-1][0] > 0.25:
            deduped.append((station, offset))
    if len(deduped) < 2:
        return None

    if deduped[-1][0] < working_length_ft:
        deduped.append((working_length_ft,
                        deduped[-1][1] + _outward_slope(deduped) * (working_length_ft - deduped[-1][0])))
    if extend_to_junction and deduped[0][0] > 0.0:
        station = deduped[0][0]
        deduped.insert(0, (0.0, deduped[0][1] - _inward_slope(deduped) * station))

    stations = np.array([s for s, _o in deduped])
    offsets = np.array([o for _s, o in deduped])
    return LineString(_sampled_along_the_alignment(leg.centerline, stations, offsets))


def _sampled_along_the_alignment(centerline: LineString, stations: np.ndarray,
                                  offsets: np.ndarray) -> np.ndarray:
    """The traced vertices, plus a sample every CURB_GAP_SAMPLE_FT through the gaps between them.

    Placed in the MEASURING frame rather than by point_at: these stations are imposed rather
    than derived from a surveyed point, which is the case place_in_measured_frame exists for.

    AND A GAP WHOSE CHORD IS THE TRACING IS LEFT ALONE - see CURB_CHORD_DEVIATION_FT. Sampled
    per gap, all or nothing, because the question is about the gap and not about one vertex.

    AND A SAMPLE THE FRAME CANNOT DESCRIBE IS DROPPED - placement_holds, which is the other
    half of this. Kept, the four samples inside greenwood_ave_south's fold put a phantom 1 ft
    jog in the kerb and the hatching built against it measured 0.67 ft over the kerb
    (PaintInsideTheCurb, fatal). Dropped, a chord spans the ~8 ft the frame cannot describe,
    which is the situation this function improves from 274 ft.
    """
    grid = np.union1d(stations, np.arange(stations[0], stations[-1], CURB_GAP_SAMPLE_FT))
    ask_o = np.interp(grid, stations, offsets)
    placed = np.asarray(place_in_measured_frame(centerline, grid, ask_o), dtype=float)
    # A traced vertex is kept whatever it measures - it is a fact about the street, not a sample.
    traced = np.isin(grid, stations)
    holds = placement_holds(centerline, grid, ask_o, placed)
    keep = holds | traced
    at = np.flatnonzero(traced)
    for lo, hi in itertools.pairwise(at):
        # Judged on the samples that HOLD, never on the ones a fold displaced: a fold throws a
        # point feet from where it was asked for, which read as deviation is a 9.3 ft chord
        # across a 25.7 ft gap on louellen_st_west, and sampling that gap on the strength of it
        # left the kerb 1.31 ft off the way it was traced from.
        inner = [i for i in range(lo + 1, hi) if holds[i]]
        if not inner:
            continue
        chord = LineString([placed[lo], placed[hi]])
        if max(chord.distance(Point(placed[i])) for i in inner) <= CURB_CHORD_DEVIATION_FT:
            keep[lo + 1:hi] = False        # the chord IS the tracing here - leave it alone
    return placed[keep]


def trimmed_curb_lines(legs: dict, corner_fillets: dict) -> dict[str, dict[str, LineString]]:
    """Each leg side clipped at the corner tangent point, i.e. the curb as it actually
    bounds the pavement.

    A leg's raw curb line deliberately overshoots the junction so the fillet has something
    to trim into - so drawing it raw puts curb lines straight across the middle of the
    intersection, marking a curb where there is none. The pavement polygon and the 3D
    export already use the trimmed pieces; this is how the plan view says the same thing.
    Sides whose corner failed to build keep the raw line, which is honest: that corner has
    no tangent point.
    """
    out = {name: {"left": leg.left_curb, "right": leg.right_curb} for name, leg in legs.items()}
    for (name_a, name_b), pieces in corner_fillets.items():
        if "error" in pieces:
            continue
        if name_a in out:
            out[name_a]["left"] = pieces["trimmed_a"]
        if name_b in out:
            out[name_b]["right"] = pieces["trimmed_b"]
    return out
