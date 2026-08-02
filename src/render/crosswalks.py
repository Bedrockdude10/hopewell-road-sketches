"""Match OSM-surveyed pedestrian crossings to intersection legs, and resolve
each leg's crosswalk offset: the real surveyed position when a crossing was
matched, else a geometric estimate (needed for hypothetical/proposed
crossings that don't exist yet). See README.md "Crosswalk styles: real data
over guessing"."""
import math

from shapely.geometry import LineString

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
                               # taper must stay clear of - the exact rendered crosswalk depth isn't available
                               # here (a Blender-side rendering default, see blender_crosswalks.py), and a curved
                               # taper's closest approach to the intersection isn't exactly at its target_ft
                               # endpoint (the arc can bow slightly past it), so this errs generous rather than
                               # trying to match that depth exactly. Shared by src/render/export.py (3D) and
                               # src/render/plan_view.py (2D) so both compute the exact same taper - if they used
                               # separately-defined copies of this constant, the two views could silently drift
                               # apart, which is exactly the kind of "why does the 3D render disagree with the 2D
                               # one" confusion this constant being centralized here is meant to prevent.

# Distance back toward the intersection from a leg's resolved crosswalk offset
# to its stop bar: half of the ~10 ft crosswalk depth used in
# scripts/blender/blender_crosswalks.py (so the setback starts at the crosswalk's near
# boundary, not its center) plus a typical MUTCD stop-line-to-crosswalk gap.
# An approximation (no site is surveyed down to exact striping), same category
# as src/render/props.py's STREETLIGHT_SIDEWALK_SETBACK_FT.
STOP_BAR_SETBACK_FT = 9.0

# Depth of a painted crosswalk (its dimension along the direction of travel), the
# single source of truth for BOTH views: src/render/plan_view.py draws it directly,
# and src/render/export.py writes it into the geometry JSON as `crosswalk_depth_m`
# for scripts/blender/blender_crosswalks.py to use. Blender's own module can't import
# from src/ (it runs under Blender's bundled Python), so passing the value through the
# JSON is what keeps the 2D reconstruction and the 3D render from drifting apart -
# the same reasoning as CROSSWALK_CLEARANCE_FT above. ~10 ft is a typical painted
# crosswalk depth; no site here is surveyed down to exact striping.
CROSSWALK_DEPTH_M = 3.0

# A stop bar spans only the ENTERING half of the roadway - a driver stops in their own
# lanes, never across the opposing ones - unlike a crosswalk, which spans curb to curb.
# This clearance keeps it off the centerline and the curb at each end. Shared, like
# CROSSWALK_DEPTH_M: src/render/plan_view.py draws the 2D bar from it, and
# src/render/export.py writes it into the geometry JSON as `stop_bar_curb_clearance_m`
# for scripts/blender/blender_crosswalks.py:add_stop_bar. The 2D view previously drew
# the bar full curb-to-curb, disagreeing with the render it was supposed to preview.
STOP_BAR_CURB_CLEARANCE_M = 0.5


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
