"""Serialize a DesignState to plain JSON (local meters, centered on the
intersection) so the headless Blender script can build a scene without needing
shapely/geopandas inside Blender's bundled Python."""
import json
from pathlib import Path

import pyproj
from shapely.geometry import LineString, Polygon

from src.geometry_model import NJ_STATE_PLANE_FT, WGS84, build_pavement_polygon, leg_clearance_ft
from src.intersection import IntersectionModel
from src.osm_context import fetch_buildings, fetch_crossings
from src.treatments import DesignState, build_sidewalk_pieces

FT_TO_M = 0.3048
BUILDING_CONTEXT_RADIUS_M = 130
SIDEWALK_WIDTH_FT = 6

_wgs84_to_state_plane = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)


def _ring_to_local_m(coords, center_ft) -> list[list[float]]:
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in coords]


def _wgs84_ring_to_local_m(coords_wgs84, center_ft) -> list[list[float]]:
    xs, ys = _wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in zip(xs, ys)]


def _building_footprint_ft(coords_wgs84, center_ft) -> Polygon:
    xs, ys = _wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return Polygon(zip(xs, ys))


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
        xs, ys = _wgs84_to_state_plane.transform(
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


def export_scenario(model: IntersectionModel, state: DesignState, name: str, out_path: Path,
                     buildings: list[dict] | None = None, crossings: list[dict] | None = None) -> Path:
    center_ft = model.center_ft
    pavement = build_pavement_polygon(state.corner_fillets)
    sidewalk_pieces = build_sidewalk_pieces(state, sidewalk_width_ft=SIDEWALK_WIDTH_FT)
    if buildings is None:
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if crossings is None:
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    matched = _match_crossings_to_legs(state.legs, crossings)

    # OSM building footprints are independent of (and coarser than) our SLD/field-measured
    # curb geometry - a few end up drawn overlapping the actual pavement. Drop those rather
    # than render buildings sitting in the middle of the road.
    buildings = [
        b for b in buildings
        if not _building_footprint_ft(b["coords_wgs84"], center_ft).intersects(pavement)
    ]

    data = {
        "name": name,
        "units": "meters",
        "existing_marked_crosswalks": model.config["intersection"].get("existing_marked_crosswalks", []),
        "pavement": _ring_to_local_m(pavement.exterior.coords, center_ft),
        "sidewalks": [_ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalk_pieces],
        "legs": [
            {
                "name": leg_name,
                "near_m": [(leg.centerline.coords[0][0] - center_ft.x) * FT_TO_M,
                           (leg.centerline.coords[0][1] - center_ft.y) * FT_TO_M],
                "far_m": [(leg.centerline.coords[-1][0] - center_ft.x) * FT_TO_M,
                          (leg.centerline.coords[-1][1] - center_ft.y) * FT_TO_M],
                "width_m": leg.curb_to_curb_ft * FT_TO_M,
                "confirmed": model.config["legs"][leg_name].get("confirmed", False),
                "crosswalk_offset_m": (
                    matched[leg_name][0] if leg_name in matched
                    else leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
                ) * FT_TO_M,
                "crosswalk_offset_source": "osm_survey" if leg_name in matched else "geometric_estimate",
                # A treatment (e.g. upgrade_crosswalk_markings) can override the style;
                # otherwise default to what OSM says exists today ("lines" if unmapped).
                "crosswalk_style": state.crosswalk_styles.get(
                    leg_name, matched[leg_name][1] if leg_name in matched else "lines"
                ),
            }
            for leg_name, leg in state.legs.items()
        ],
        "refuge_islands": [
            {
                "name": island_name,
                "coords": _ring_to_local_m(island["polygon"].exterior.coords, center_ft),
                "height_m": 0.15,
            }
            for island_name, island in state.refuge_islands.items()
        ],
        "raised_crossings": [
            {"name": leg_name, "coords": _ring_to_local_m(poly.exterior.coords, center_ft), "height_m": 0.10}
            for leg_name, poly in state.raised_crossings.items()
        ],
        "corner_parcels": [
            {"name": str(row["quadrant"]), "coords": _ring_to_local_m(row.geometry.exterior.coords, center_ft)}
            for _, row in model.corner_parcels.iterrows()
        ],
        "buildings": [
            {"coords": _wgs84_ring_to_local_m(b["coords_wgs84"], center_ft), "height_m": b["height_m"]}
            for b in buildings
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path
