"""Serialize a DesignState to plain JSON (local meters, centered on the
intersection) so the headless Blender script can build a scene without needing
shapely/geopandas inside Blender's bundled Python."""
import json
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry import LineString, Point, Polygon

from src.geometry_model import NJ_STATE_PLANE_FT, WGS84, build_pavement_polygon, leg_clearance_ft
from src.intersection import IntersectionModel
from src.mesh_utils import build_decimated_building_mesh
from src.osm_context import fetch_buildings, fetch_crossings
from src.treatments import DesignState, build_sidewalk_pieces

FT_TO_M = 0.3048
BUILDING_CONTEXT_RADIUS_M = 130
SIDEWALK_WIDTH_FT = 6
NEAR_ZONE_BUFFER_FT = 10  # how far past the farthest crosswalk the "near" (4k texture) pavement zone extends
TREE_SPACING_FT = 25  # typical municipal street-tree spacing (NACTO/street-design guidance), not a fabricated guess
STREETLIGHT_SIDEWALK_SETBACK_FT = 4
SIGN_SIDEWALK_SETBACK_FT = 3

_wgs84_to_state_plane = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)


def _ring_to_local_m(coords, center_ft) -> list[list[float]]:
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in coords]


def _pt_to_local_m(x, y, center_ft) -> list[float]:
    return [(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M]


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


def _resolve_crosswalk_offsets(state: DesignState, crossings: list[dict]) -> dict[str, tuple[float, str]]:
    """{leg_name: (offset_ft, source)} - real OSM survey position if matched, else
    the geometric past-the-curve estimate (needed for hypothetical/proposed crossings)."""
    matched = _match_crossings_to_legs(state.legs, crossings)
    out = {}
    for leg_name in state.legs:
        if leg_name in matched:
            out[leg_name] = (matched[leg_name][0], "osm_survey")
        else:
            out[leg_name] = (leg_clearance_ft(leg_name, state.legs, state.corner_fillets), "geometric_estimate")
    return out


def _split_near_far(polygons: list[Polygon], center_ft: Point, near_radius_ft: float):
    """
    Split a list of polygons (the pavement, the sidewalk pieces, ...) into a
    near-camera zone and everything else, by intersecting each with a circle
    around the intersection - used to texture what viewers will actually
    scrutinize (pavement/sidewalk right at the crosswalks) at a higher
    resolution than the rest. Any piece can become a MultiPolygon on either
    side of the split; always returns flat lists of simple Polygons.
    """
    circle = center_ft.buffer(near_radius_ft)
    near_polys, far_polys = [], []
    for poly in polygons:
        near = poly.intersection(circle)
        far = poly.difference(circle)
        near_polys += list(near.geoms) if near.geom_type == "MultiPolygon" else [near] if not near.is_empty else []
        far_polys += list(far.geoms) if far.geom_type == "MultiPolygon" else [far] if not far.is_empty else []
    return near_polys, far_polys


def _tree_points_along_piece(piece: Polygon, spacing_ft: float) -> list[tuple[float, float]]:
    """Sample points along a sidewalk piece's long axis at spacing_ft intervals.
    Corner wedge pieces (not meaningfully elongated) are skipped - trees belong
    along the straight runs of sidewalk, not crammed into a tiny corner fillet."""
    mrr = piece.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return []
    coords = list(mrr.exterior.coords)[:4]
    edges = [(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    lengths = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in edges]
    long_idx = max(range(4), key=lambda i: lengths[i])
    long_len, short_len = lengths[long_idx], lengths[(long_idx + 1) % 4]
    if short_len < 1e-6 or long_len / short_len < 1.8 or long_len < spacing_ft:
        return []

    a, b = edges[long_idx]
    n_trees = max(int(long_len // spacing_ft), 1)
    points = []
    for i in range(n_trees):
        t = (i + 0.5) / n_trees
        pt = Point(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if piece.buffer(1).contains(pt):
            points.append((pt.x, pt.y))
    return points


def _corner_streetlight_props(corner_fillets: dict, center_ft: Point) -> list[dict]:
    """Streetlight at each corner's fillet-arc midpoint (real geometry), pushed a
    few feet further from the intersection center onto the sidewalk/corner -
    that offset distance is a placement approximation, flagged as such."""
    props = []
    for pieces in corner_fillets.values():
        if "error" in pieces:
            continue
        mid = pieces["arc"].interpolate(0.5, normalized=True)
        outward = np.array([mid.x - center_ft.x, mid.y - center_ft.y])
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0])
        pos = (mid.x + outward[0] * STREETLIGHT_SIDEWALK_SETBACK_FT,
               mid.y + outward[1] * STREETLIGHT_SIDEWALK_SETBACK_FT)
        heading = np.degrees(np.arctan2(outward[1], outward[0]))
        props.append({
            "type": "streetlight", "position_ft": pos, "heading_deg": heading,
            "source": f"real: corner fillet arc midpoint; approximation: pushed {STREETLIGHT_SIDEWALK_SETBACK_FT} ft "
                      "outward onto the sidewalk (no surveyed pole location available)",
        })
    return props


def _leg_sign_position_ft(leg, offset_ft: float, side: str) -> tuple[tuple[float, float], float]:
    """A point offset_ft along a leg's centerline from the intersection, pushed
    laterally past the curb (left or right, per `side`) onto the sidewalk.
    Returns (position, heading_deg) with heading pointing back toward the road."""
    centerline = leg.centerline
    p = centerline.interpolate(min(offset_ft, centerline.length))
    p2 = centerline.interpolate(min(offset_ft + 1, centerline.length))
    u = np.array([p2.x - p.x, p2.y - p.y])
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]]) if side == "left" else np.array([u[1], -u[0]])
    half_w = leg.curb_to_curb_ft / 2
    pos = (p.x + n[0] * (half_w + SIGN_SIDEWALK_SETBACK_FT), p.y + n[1] * (half_w + SIGN_SIDEWALK_SETBACK_FT))
    heading = np.degrees(np.arctan2(-n[1], -n[0]))  # face back toward the road
    return pos, heading


def _stop_sign_props(state: DesignState, offsets_ft: dict) -> list[dict]:
    """One stop sign per approach, placed on the leg's 'right' curb (our own
    left/right offset convention, not a real traffic-direction analysis) just
    past where the roadway straightens out. This is a placement approximation -
    real stop sign placement depends on engineering judgment not modeled here."""
    props = []
    for leg_name, leg in state.legs.items():
        offset_ft = offsets_ft[leg_name][0]
        pos, heading = _leg_sign_position_ft(leg, offset_ft, side="right")
        props.append({
            "type": "stop_sign", "position_ft": pos, "heading_deg": heading,
            "source": "approximation: placed on the leg's near-corner curb line, arbitrary side "
                      "(not a real traffic-direction/engineering placement study)",
        })
    return props


def _extra_props_from_config(model: IntersectionModel, state: DesignState, offsets_ft: dict) -> list[dict]:
    """User-specified extra signage (e.g. a school zone sign) from the site's
    config.yaml `props.extra` list - explicitly site-specific knowledge that
    doesn't belong in the general pipeline. See sites/README.md."""
    props = []
    for entry in model.config.get("props", {}).get("extra", []):
        leg = state.legs.get(entry["leg"])
        if leg is None:
            continue
        offset_ft = entry.get("offset_ft", offsets_ft.get(entry["leg"], (10, ""))[0])
        pos, heading = _leg_sign_position_ft(leg, offset_ft, side=entry.get("side", "right"))
        props.append({
            "type": entry["type"], "position_ft": pos, "heading_deg": heading,
            "source": f"user-specified in site config.yaml (props.extra): {entry.get('note', 'no note given')}",
        })
    return props


def export_scenario(model: IntersectionModel, state: DesignState, name: str, out_path: Path,
                     buildings: list[dict] | None = None, crossings: list[dict] | None = None,
                     theme: dict | None = None) -> Path:
    center_ft = model.center_ft
    if theme is None:
        from src.theme import build_default_theme
        theme = build_default_theme()
    pavement = build_pavement_polygon(state.corner_fillets)
    sidewalk_pieces = build_sidewalk_pieces(state, sidewalk_width_ft=SIDEWALK_WIDTH_FT)
    if buildings is None:
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if crossings is None:
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crosswalk_offsets = _resolve_crosswalk_offsets(state, crossings)

    # OSM building footprints are independent of (and coarser than) our SLD/field-measured
    # curb geometry - a few end up drawn overlapping the actual pavement. Drop those rather
    # than render buildings sitting in the middle of the road.
    buildings = [
        b for b in buildings
        if not _building_footprint_ft(b["coords_wgs84"], center_ft).intersects(pavement)
    ]

    near_radius_ft = max((v[0] for v in crosswalk_offsets.values()), default=30) + NEAR_ZONE_BUFFER_FT
    pavement_near, pavement_far = _split_near_far([pavement], center_ft, near_radius_ft)
    sidewalks_near, sidewalks_far = _split_near_far(sidewalk_pieces, center_ft, near_radius_ft)

    tree_points_ft = [pt for piece in sidewalk_pieces for pt in _tree_points_along_piece(piece, TREE_SPACING_FT)]

    props = (
        _corner_streetlight_props(state.corner_fillets, center_ft)
        + _stop_sign_props(state, crosswalk_offsets)
        + _extra_props_from_config(model, state, crosswalk_offsets)
    )

    building_entries = []
    for b in buildings:
        footprint_ft = _building_footprint_ft(b["coords_wgs84"], center_ft)
        mesh = build_decimated_building_mesh(footprint_ft, b["height_m"] / FT_TO_M)
        if mesh is not None:
            verts_ft, faces = mesh
            building_entries.append({
                "mesh": True,
                "vertices_m": [_pt_to_local_m(x, y, center_ft)[:2] + [z * FT_TO_M] for x, y, z in verts_ft],
                "faces": faces,
            })
        else:
            building_entries.append({
                "mesh": False,
                "coords": _wgs84_ring_to_local_m(b["coords_wgs84"], center_ft),
                "height_m": b["height_m"],
            })

    data = {
        "name": name,
        "units": "meters",
        "theme": theme,
        "existing_marked_crosswalks": model.config["intersection"].get("existing_marked_crosswalks", []),
        "pavement_near": [_ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_near],
        "pavement_far": [_ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_far],
        "sidewalks_near": [_ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_near],
        "sidewalks_far": [_ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_far],
        "tree_points": [_pt_to_local_m(x, y, center_ft) for x, y in tree_points_ft],
        "props": [
            {**p, "position_m": _pt_to_local_m(p["position_ft"][0], p["position_ft"][1], center_ft)}
            for p in props
        ],
        "legs": [
            {
                "name": leg_name,
                "near_m": [(leg.centerline.coords[0][0] - center_ft.x) * FT_TO_M,
                           (leg.centerline.coords[0][1] - center_ft.y) * FT_TO_M],
                "far_m": [(leg.centerline.coords[-1][0] - center_ft.x) * FT_TO_M,
                          (leg.centerline.coords[-1][1] - center_ft.y) * FT_TO_M],
                "width_m": leg.curb_to_curb_ft * FT_TO_M,
                "confirmed": model.config["legs"][leg_name].get("confirmed", False),
                "crosswalk_offset_m": crosswalk_offsets[leg_name][0] * FT_TO_M,
                "crosswalk_offset_source": crosswalk_offsets[leg_name][1],
                # A treatment (e.g. upgrade_crosswalk_markings) can override the style;
                # otherwise default to what OSM says exists today ("lines" if unmapped).
                "crosswalk_style": state.crosswalk_styles.get(leg_name, "lines"),
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
        "buildings": building_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path
