"""Match OSM-surveyed pedestrian crossings to intersection legs, and resolve
each leg's crosswalk offset: the real surveyed position when a crossing was
matched, else a geometric estimate (needed for hypothetical/proposed
crossings that don't exist yet). See README.md "Crosswalk styles: real data
over guessing"."""
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.geometry.model import leg_clearance_ft
from src.geometry.treatments import DesignState

# OSM crossing:markings values -> our 3 rendered styles. "lines" (two simple
# transverse boundary lines) is the least visible; FHWA/NACTO guidance treats
# continental and ladder as visibility upgrades over it - unmapped/missing
# values default to "lines" since that's the least assumption-laden guess.
OSM_MARKINGS_TO_STYLE = {
    "lines": "lines",
    "zebra": "continental",
    "ladder": "ladder",
}

CROSSWALK_CLEARANCE_FT = 5.0  # safety margin beyond a leg's resolved crosswalk offset that a lane-narrowing
                               # taper must stay clear of - comfortably more than half CROSSWALK_DEPTH_FT (below),
                               # deliberately, because a curved taper's closest approach to the intersection isn't
                               # exactly at its target_ft endpoint (the arc can bow slightly past it), so this errs
                               # generous rather than tracking the depth exactly. Shared by src/render/export.py (3D) and
                               # src/render/plan_view.py (2D) so both compute the exact same taper - if they used
                               # separately-defined copies of this constant, the two views could silently drift
                               # apart, which is exactly the kind of "why does the 3D render disagree with the 2D
                               # one" confusion this constant being centralized here is meant to prevent.

# Typical MUTCD gap between a stop line and the near edge of the crosswalk ahead of it.
STOP_BAR_TO_CROSSWALK_GAP_FT = 4.0

# Depth of a painted crosswalk - the gap between the two transverse lines, i.e. its
# dimension along the direction of travel. 6 ft is Mercer County's recommended width for
# a transverse crosswalk (Danny, 2026-08-02); the previous ~10 ft was a generic default
# and rendered visibly too wide at this scale.
#
# Authoritative in FEET because the source standard is in feet; the metres value is
# derived for the renderer. Single source of truth for BOTH views: plan_view.py draws it
# directly, and export.py writes it into the geometry JSON as `crosswalk_depth_m` for
# scripts/blender/blender_crosswalks.py, which can't import from src/ (it runs under
# Blender's bundled Python). That JSON hand-off is what keeps the 2D reconstruction and
# the 3D render from drifting apart.
CROSSWALK_DEPTH_FT = 6.0
CROSSWALK_DEPTH_M = CROSSWALK_DEPTH_FT * FT_TO_M

# Distance back toward the intersection from a leg's resolved crosswalk offset (which is
# the crosswalk's CENTRE) to its stop bar. Derived from the crosswalk depth rather than
# hardcoded, so narrowing the crosswalk moves the stop bar with it instead of leaving a
# stale literal whose own comment no longer describes it: half the depth reaches the
# crosswalk's near edge, then the MUTCD gap. An approximation - no site here is surveyed
# down to exact striping.
STOP_BAR_SETBACK_FT = CROSSWALK_DEPTH_FT / 2 + STOP_BAR_TO_CROSSWALK_GAP_FT

# A stop bar spans only the ENTERING half of the roadway - a driver stops in their own
# lanes, never across the opposing ones - unlike a crosswalk, which spans curb to curb.
# This clearance keeps it off the centerline and the curb at each end. Shared, like
# CROSSWALK_DEPTH_M: src/render/plan_view.py draws the 2D bar from it, and
# src/render/export.py writes it into the geometry JSON as `stop_bar_curb_clearance_m`
# for scripts/blender/blender_crosswalks.py:add_stop_bar. The 2D view previously drew
# the bar full curb-to-curb, disagreeing with the render it was supposed to preview.
STOP_BAR_CURB_CLEARANCE_M = 0.5

# Plan/check depth of a stop bar. The painted bar is ~1.5 ft deep; this is only used to
# give it an area to test and draw, never to size the 3D stripe.
STOP_BAR_PLAN_DEPTH_FT = 1.5


def stop_bar_band_geometry_ft(width_ft: float) -> tuple[float, float]:
    """(span_ft, lateral_offset_ft) for a stop bar on a roadway `width_ft` wide.

    The bar covers half the width minus a clearance at each end, and is centred on the
    middle of the entering half - i.e. offset a quarter of the full width off the road
    centerline, toward the leg's own 'left' side (see blender_crosswalks.add_stop_bar
    for why that is the entering driver's side under right-hand traffic).
    """
    clearance_ft = STOP_BAR_CURB_CLEARANCE_M / FT_TO_M
    half_ft = width_ft / 2
    return max(half_ft - clearance_ft, clearance_ft), half_ft / 2


def entering_lane_width_ft(state: DesignState, leg_name: str) -> float | None:
    """Real width of the entering travel lane if a treatment has narrowed that side
    (lane narrowing or marked parking), else None meaning the full curb-to-curb half.

    Lives here rather than in export.py so the plan view and the 3D export size the stop
    bar from the same rule - a bar should stop at the real lane edge, not run across a
    painted buffer or a parking lane no stopped vehicle occupies.
    """
    half_ft = state.legs[leg_name].curb_to_curb_ft / 2
    if leg_name in state.lane_narrowing and "left" in state.lane_narrowing_sides.get(leg_name, ("left", "right")):
        return half_ft - state.lane_narrowing[leg_name]
    parking_zone = state.parking_zones.get((leg_name, "left"))
    if parking_zone is not None:
        return half_ft - parking_zone["curb_offset_ft"] - parking_zone["depth_ft"]
    return None


def stop_bar_width_ft(state: DesignState, leg_name: str) -> float:
    """Full roadway width the stop bar is sized against (twice the entering lane width
    where a treatment narrowed it, else the leg's own curb-to-curb width)."""
    entering_ft = entering_lane_width_ft(state, leg_name)
    return 2 * entering_ft if entering_ft is not None else state.legs[leg_name].curb_to_curb_ft


# How square-on a crossing has to be to the leg it's credited with crossing. Measured
# across all four configured sites, candidate matches surviving the distance checks are
# sharply bimodal: the true ones run 82.3-89.8 deg (square across the road, 0.1-8.4 ft
# off its centerline), the false ones 0.6-5.9 deg (parallel to it, 17-33 ft off). The
# one value between those clusters is the W Broad & Louellen crossing at 42.2 deg - a
# real crossing that OSM has drawn oddly (78 ft end-to-end across a ~34 ft road; see
# that site's config.yaml, where the same way contradicts the width estimate).
# 30 deg sits well clear of both clusters and keeps that real-but-skewed case.
MIN_CROSSING_ANGLE_DEG = 30.0

# The most a crosswalk may be drawn off square to the road it crosses. Real skews at
# these four sites run 0.2-7.7 deg. The one value past this limit is W Broad & Louellen's
# crossing at -47.8 deg - the same OSM way that is 78 ft end-to-end across a ~33 ft road,
# i.e. loosely drawn rather than genuinely diagonal (see that site's config.yaml).
MAX_CROSSING_SKEW_DEG = 20.0


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


def _crossing_skew_deg(crossing_line: LineString, centerline: LineString) -> float:
    """Signed angle, in (-90, 90], from square-across-the-leg to the crossing's real
    direction. 0 means the surveyed crossing runs exactly perpendicular to the leg
    centerline; positive is counter-clockwise.

    Real crosswalks are not always square to the road they cross - they are painted to
    line up with the curb ramps and the sidewalks either side, which at a skewed
    junction can be several degrees off. Reconstructing the crosswalk perpendicular to
    the road centerline (as this code originally did) throws that away and draws the
    band at a visible angle to the surveyed crossing line it came from: 7.7 degrees on
    princeton_ave_south, 6.6 on broad_st_west. Carried through so both the plan view and
    the Blender render orient the crosswalk the way it was actually surveyed.
    """
    (cx0, cy0), (cx1, cy1) = crossing_line.coords[0], crossing_line.coords[-1]
    (lx0, ly0), (lx1, ly1) = centerline.coords[0], centerline.coords[-1]
    crossing_dir = math.atan2(cy1 - cy0, cx1 - cx0)
    square_dir = math.atan2(ly1 - ly0, lx1 - lx0) + math.pi / 2  # perpendicular to the leg
    # A line has no direction, so fold the difference into (-90, 90].
    diff = (math.degrees(crossing_dir - square_dir) + 90) % 180 - 90
    return diff


def _match_crossings_to_legs(legs: dict, crossings: list[dict]) -> dict:
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
            if not (0 < along < centerline.length):
                continue
            perp = centerline.interpolate(along).distance(mid)
            if perp > leg.curb_to_curb_ft / 2 + 10:  # not plausibly this leg's crossing
                continue
            if _crossing_angle_deg(line, centerline) < MIN_CROSSING_ANGLE_DEG:
                continue  # runs alongside this leg, not across it
            skew = _crossing_skew_deg(line, centerline)
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


def resolve_crosswalk_offsets(state: DesignState, crossings: list[dict]) -> dict[str, tuple[float, str]]:
    """{leg_name: (offset_ft, source)} - real OSM survey position if matched, else
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
            offset_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
            source = "geometric_estimate"
        delta_ft = state.crosswalk_offset_overrides.get(leg_name)
        if delta_ft:
            offset_ft += delta_ft
            source += f"+scenario_shift({delta_ft:+g}ft)"
        out[leg_name] = (offset_ft, source)
    return out


def resolve_crosswalk_skews(state: DesignState, crossings: list[dict]) -> dict[str, float]:
    """{leg_name: skew_deg} - how far each leg's crosswalk is rotated off square to the
    leg, taken from the surveyed crossing way (see _crossing_skew_deg).

    Only legs with a real matched crossing get an entry. A leg falling back to the
    geometric estimate has no surveyed orientation to copy, so it stays square to the
    road - inventing a skew for it would be a guess dressed up as survey data.

    Skews beyond MAX_CROSSING_SKEW_DEG are discarded rather than drawn. Genuine skews
    here run 0.2-7.7 degrees; a much larger one means the OSM way is drawn loosely
    rather than that the paint is really at that angle, and honouring it would both
    rotate the marking absurdly and inflate its span by 1/cos(skew).
    """
    matched = _match_crossings_to_legs(state.legs, crossings)
    skews = {}
    for leg_name, (_along, _style, skew, _line, _tags) in matched.items():
        if abs(skew) > MAX_CROSSING_SKEW_DEG:
            print(f"  NOTE: {leg_name}'s OSM crossing sits {skew:+.1f} deg off square, beyond the "
                  f"{MAX_CROSSING_SKEW_DEG:.0f} deg plausible limit - drawing it square instead. "
                  f"The crossing way's own geometry here is suspect.")
            continue
        skews[leg_name] = skew
    return skews


def resolve_stop_bar_offsets(state: DesignState, crosswalk_offsets: dict[str, tuple[float, str]]) -> dict[str, float]:
    """{leg_name: offset_ft} - where a signalized approach's stop bar sits,
    derived from that leg's already-resolved crosswalk offset (real or
    estimated, overrides included) minus STOP_BAR_SETBACK_FT. Clamped to
    leg_clearance_ft() so a short leg or a tight corner radius never pushes
    the stop bar back into the curb-return curve."""
    out = {}
    for leg_name, (crosswalk_offset_ft, _source) in crosswalk_offsets.items():
        min_offset_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        out[leg_name] = max(crosswalk_offset_ft - STOP_BAR_SETBACK_FT, min_offset_ft)
    return out


def crosswalk_axes(leg, offset_ft: float, skew_deg: float = 0.0):
    """(centre, along-travel axis, across-road axis, cos skew) for a crossing on this leg.

    One definition of the crossing's frame, so the band the plan view draws, the reach the
    3D export writes and the check in src/checks.py are all measured off the same axes.
    """
    (x0, y0), (x1, y1) = leg.centerline.coords[0], leg.centerline.coords[1]
    length = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    centre = (x0 + ux * offset_ft, y0 + uy * offset_ft)

    skew = np.radians(skew_deg)
    cos_s, sin_s = np.cos(skew), np.sin(skew)
    # Rotate the leg's own axes by the skew: n is the across-road axis the crosswalk
    # spans, u the along-travel axis its depth runs down.
    nx, ny = -uy, ux
    nx, ny = nx * cos_s - ny * sin_s, nx * sin_s + ny * cos_s
    ux, uy = ux * cos_s - uy * sin_s, ux * sin_s + uy * cos_s
    return centre, (ux, uy), (nx, ny), cos_s


def crosswalk_reaches_ft(state, offsets: dict, skews: dict) -> dict:
    """{leg_name: (left_ft, right_ft)} - how far each crossing runs to reach its kerbs.

    Written into the geometry JSON so the 3D render spans the same real, asymmetric width
    the plan view draws, instead of half the nominal width either side of NJDOT's
    centerline. Blender can't import from src/, so this has to travel as numbers.
    """
    reaches = {}
    for name, leg in state.legs.items():
        if name not in offsets:
            continue
        centre, _u, normal, _cos = crosswalk_axes(leg, offsets[name][0], skews.get(name, 0.0))
        reaches[name] = crosswalk_reach_to_curbs_ft(leg, centre, normal)
    return reaches


def crosswalk_band_ft(leg, offset_ft: float, depth_ft: float, skew_deg: float = 0.0,
                     span_ft: float | None = None, lateral_offset_ft: float = 0.0) -> Polygon:
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
    if span_ft is None:
        left_ft, right_ft = crosswalk_reach_to_curbs_ft(leg, (cx, cy), (nx, ny))
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


def crosswalk_reach_to_curbs_ft(leg, center, normal) -> tuple[float, float]:
    """(left, right) distance from `center` out to this leg's two curb lines along `normal`.

    A crosswalk runs kerb to kerb. It was being drawn as half the leg's NOMINAL width either
    side of the NJDOT centerline, which is the right answer only if the curbs are a symmetric
    offset of that centerline - and since they became the surveyor's traced kerbs, they are
    not. The traced kerb is asymmetric about NJDOT's alignment and flares near the corner, so
    a symmetric nominal band stops short of the kerb on one side and overshoots on the other.

    Falls back to the nominal half-width per side when a side has no curb line or the ray
    misses it, so a leg with no traced kerb behaves exactly as before.
    """
    fallback = leg.curb_to_curb_ft / 2
    origin = Point(*center)
    direction = np.asarray(normal, dtype=float)
    reach = []
    # +normal is the leg's left, matching Leg.left_curb / right_curb.
    for curb, sign in ((leg.left_curb, 1.0), (leg.right_curb, -1.0)):
        if curb is None or curb.is_empty:
            reach.append(fallback)
            continue
        far = np.asarray(center) + direction * sign * fallback * CURB_SEARCH_FACTOR
        hit = LineString([center, tuple(far)]).intersection(curb)
        if hit.is_empty:
            reach.append(fallback)
            continue
        points = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", [])]
        distances = [origin.distance(p) for p in points if p.geom_type == "Point"]
        reach.append(min(distances) if distances else fallback)
    return reach[0], reach[1]


def crosswalk_bands_ft(state, offsets: dict, skews: dict, depth_ft: float) -> dict:
    """{leg_name: band polygon} for every leg - the footprints the 2D view draws, the 3D
    render stripes, and src/checks.py validates. One definition, so a check that passes in
    one path can't be checking different geometry from the other."""
    return {name: crosswalk_band_ft(leg, offsets[name][0], depth_ft, skews.get(name, 0.0))
            for name, leg in state.legs.items() if name in offsets}


def stop_bar_bands_ft(state, stop_bar_offsets: dict, skews: dict) -> dict:
    """{leg_name: stop bar polygon}, on the same shared terms as crosswalk_bands_ft."""
    bands = {}
    for name, offset_ft in stop_bar_offsets.items():
        leg = state.legs.get(name)
        if leg is None:
            continue
        span_ft, lateral_ft = stop_bar_band_geometry_ft(stop_bar_width_ft(state, name))
        bands[name] = crosswalk_band_ft(leg, offset_ft, STOP_BAR_PLAN_DEPTH_FT, skews.get(name, 0.0),
                                         span_ft=span_ft, lateral_offset_ft=lateral_ft)
    return bands
