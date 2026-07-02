"""Match OSM-surveyed pedestrian crossings to intersection legs, and resolve
each leg's crosswalk offset: the real surveyed position when a crossing was
matched, else a geometric estimate (needed for hypothetical/proposed
crossings that don't exist yet). See README.md "Crosswalk styles: real data
over guessing"."""
from shapely.geometry import LineString

from src.coords import wgs84_to_state_plane
from src.geometry_model import leg_clearance_ft
from src.treatments import DesignState

# OSM crossing:markings values -> our 3 rendered styles. "lines" (two simple
# transverse boundary lines) is the least visible; FHWA/NACTO guidance treats
# continental and ladder as visibility upgrades over it - unmapped/missing
# values default to "lines" since that's the least assumption-laden guess.
OSM_MARKINGS_TO_STYLE = {
    "lines": "lines",
    "zebra": "continental",
    "ladder": "ladder",
}


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
