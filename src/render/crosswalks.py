"""Match OSM-surveyed pedestrian crossings to intersection legs, and resolve
each leg's crosswalk offset: the real surveyed position when a crossing was
matched, else a geometric estimate (needed for hypothetical/proposed
crossings that don't exist yet). See README.md "Crosswalk styles: real data
over guessing"."""
from shapely.geometry import LineString

from src.render.coords import wgs84_to_state_plane
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

# Distance back toward the intersection from a leg's resolved crosswalk offset
# to its stop bar: half of the ~10 ft crosswalk depth used in
# scripts/blender/blender_crosswalks.py (so the setback starts at the crosswalk's near
# boundary, not its center) plus a typical MUTCD stop-line-to-crosswalk gap.
# An approximation (no site is surveyed down to exact striping), same category
# as src/render/props.py's STREETLIGHT_SIDEWALK_SETBACK_FT.
STOP_BAR_SETBACK_FT = 9.0


def _match_crossings_to_legs(legs: dict, crossings: list[dict]) -> dict:
    """
    Match each OSM-mapped crossing way to whichever leg it actually crosses -
    real surveyed geometry beats a geometric estimate of where a crosswalk
    probably sits. A crossing is assigned to the leg whose centerline it's
    closest to (perpendicular distance), as long as its midpoint projects onto
    that leg between the intersection and its far end, and isn't absurdly far
    off to the side (i.e. it's actually this leg's crossing, not some other
    nearby crossing that happened to fall within the fetch radius).

    Returns {leg_name: (offset_ft, style)} for legs with a matched real crossing.
    """
    candidates = []
    for crossing in crossings:
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
            candidates.append((perp, leg_name, along, style))

    best_by_leg: dict[str, tuple[float, float, str]] = {}  # leg_name -> (best_perp, along, style)
    for perp, leg_name, along, style in sorted(candidates, key=lambda c: c[0]):
        if leg_name not in best_by_leg:
            best_by_leg[leg_name] = (perp, along, style)
    return {leg_name: (along, style) for leg_name, (_, along, style) in best_by_leg.items()}


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
