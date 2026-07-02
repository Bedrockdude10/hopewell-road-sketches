"""Serialize a DesignState to plain JSON (local meters, centered on the
intersection) so the headless Blender script can build a scene without needing
shapely/geopandas inside Blender's bundled Python.

This module only orchestrates: coordinate transforms live in src/render/coords.py,
crosswalk-to-leg matching in src/render/crosswalks.py, and street-furniture placement
in src/render/props.py."""
import json
from pathlib import Path

from shapely.geometry import Point, Polygon

from src.render.coords import FT_TO_M, building_footprint_ft, pt_to_local_m, ring_to_local_m, wgs84_ring_to_local_m
from src.render.crosswalks import resolve_crosswalk_offsets, resolve_stop_bar_offsets
from src.geometry.model import (
    build_pavement_polygon, corner_overlay_polygon, hatch_lines_ft, lane_narrowing_polygons_ft, leg_clearance_ft,
)
from src.geometry.intersection import IntersectionModel
from src.render.mesh_utils import build_decimated_building_mesh
from src.sources.osm_context import fetch_buildings, fetch_crossings
from src.render.props import build_props
from src.geometry.treatments import DEFAULT_CENTERLINE_STYLE, DesignState, build_sidewalk_pieces

BUILDING_CONTEXT_RADIUS_M = 130
SIDEWALK_WIDTH_FT = 6
NEAR_ZONE_BUFFER_FT = 10  # how far past the farthest crosswalk the "near" (4k texture) pavement zone extends
TREE_SPACING_FT = 25  # typical municipal street-tree spacing (NACTO/street-design guidance), not a fabricated guess
CORNER_HATCH_SPACING_FT = 2.5  # spacing between rendered diagonal hatch lines - a rendering choice, not MUTCD-specified


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
    along the straight runs of sidewalk, not crammed into a tiny corner fillet.
    Proximity to the corner itself is filtered separately (_is_clear_of_corner)
    since that needs to know which leg a point is actually alongside, not just
    the piece's own shape."""
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


def _is_clear_of_corner(pt: tuple[float, float], legs: dict, corner_fillets: dict) -> bool:
    """A point counts as clear of the corner only if, projected onto whichever
    leg's centerline it's actually alongside, it falls past that leg's own
    real leg_clearance_ft() - the same curb-return clearance distance already
    used to place crosswalks and stop bars. Filtering by raw distance from
    the intersection CENTER point (an earlier version of this function did)
    doesn't work: a wide road's own half-width alone can already exceed a
    fixed clearance radius at every point along it, so the filter ends up
    doing nothing on a 60+ ft road while over-filtering a narrow one - the
    same Euclidean-vs-projected mistake leg_clearance_ft() itself was
    originally written to avoid (see README.md's Phase 4 general notes)."""
    point = Point(pt)
    best_leg, best_along, best_perp = None, 0.0, float("inf")
    for leg_name, leg in legs.items():
        along = leg.centerline.project(point)
        perp = leg.centerline.interpolate(along).distance(point)
        if perp < best_perp:
            best_leg, best_along, best_perp = leg_name, along, perp
    if best_leg is None:
        return True
    return best_along >= leg_clearance_ft(best_leg, legs, corner_fillets)


def export_scenario(model: IntersectionModel, state: DesignState, name: str, out_path: Path,
                     buildings: list[dict] | None = None, crossings: list[dict] | None = None,
                     theme: dict | None = None) -> Path:
    center_ft = model.center_ft
    if theme is None:
        from src.render.theme import build_default_theme
        theme = build_default_theme()
    pavement = build_pavement_polygon(state.corner_fillets)
    sidewalk_pieces = build_sidewalk_pieces(state, sidewalk_width_ft=SIDEWALK_WIDTH_FT)
    if buildings is None:
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    if crossings is None:
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crosswalk_offsets = resolve_crosswalk_offsets(state, crossings)
    # Stop bars only make sense at a signalized intersection (this site's
    # config.yaml `signals` block is what "signalized" means - see
    # src/render/props.py's _traffic_signal_props/_no_turn_on_red_props, which gate
    # the same way).
    stop_bar_offsets = resolve_stop_bar_offsets(state, crosswalk_offsets) if model.config.get("signals") else {}

    # OSM building footprints are independent of (and coarser than) our SLD/field-measured
    # curb geometry - a few end up drawn overlapping the actual pavement. Drop those rather
    # than render buildings sitting in the middle of the road.
    buildings = [b for b in buildings if not building_footprint_ft(b["coords_wgs84"]).intersects(pavement)]

    near_radius_ft = max((v[0] for v in crosswalk_offsets.values()), default=30) + NEAR_ZONE_BUFFER_FT
    pavement_near, pavement_far = _split_near_far([pavement], center_ft, near_radius_ft)
    sidewalks_near, sidewalks_far = _split_near_far(sidewalk_pieces, center_ft, near_radius_ft)

    tree_points_ft = [
        pt for piece in sidewalk_pieces
        for pt in _tree_points_along_piece(piece, TREE_SPACING_FT)
        if _is_clear_of_corner(pt, state.legs, state.corner_fillets)
    ]

    props = build_props(model, state, crosswalk_offsets, center_ft)

    # Paint-only / no-curb-change proposal treatments (see src/geometry/treatments.py:
    # add_lane_narrowing / add_corner_hatching / add_mountable_apron) - all flush
    # with the existing pavement, never touching pavement_near/far or corner_parcels.
    lane_narrowing_stripes = [
        ring_to_local_m(poly.exterior.coords, center_ft)
        for leg_name, stripe_width_ft in state.lane_narrowing.items()
        for poly in lane_narrowing_polygons_ft(state.legs[leg_name], stripe_width_ft)
    ]
    corner_hatching_lines = [
        [pt_to_local_m(x, y, center_ft) for x, y in line.coords]
        for corner, depth_ft in state.corner_hatching.items()
        if "error" not in state.corner_fillets[corner]
        for line in hatch_lines_ft(corner_overlay_polygon(state.corner_fillets[corner], center_ft, depth_ft),
                                    spacing_ft=CORNER_HATCH_SPACING_FT)
    ]
    corner_apron_polygons = [
        ring_to_local_m(corner_overlay_polygon(state.corner_fillets[corner], center_ft, extent_ft).exterior.coords,
                         center_ft)
        for corner, extent_ft in state.corner_aprons.items()
        if "error" not in state.corner_fillets[corner]
    ]

    building_entries = []
    for b in buildings:
        footprint_ft = building_footprint_ft(b["coords_wgs84"])
        mesh = build_decimated_building_mesh(footprint_ft, b["height_m"] / FT_TO_M)
        if mesh is not None:
            verts_ft, faces = mesh
            building_entries.append({
                "mesh": True,
                "vertices_m": [pt_to_local_m(x, y, center_ft)[:2] + [z * FT_TO_M] for x, y, z in verts_ft],
                "faces": faces,
            })
        else:
            building_entries.append({
                "mesh": False,
                "coords": wgs84_ring_to_local_m(b["coords_wgs84"], center_ft),
                "height_m": b["height_m"],
            })

    data = {
        "name": name,
        "units": "meters",
        "theme": theme,
        "existing_marked_crosswalks": model.config["intersection"].get("existing_marked_crosswalks", []),
        "pavement_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_near],
        "pavement_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in pavement_far],
        "sidewalks_near": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_near],
        "sidewalks_far": [ring_to_local_m(p.exterior.coords, center_ft) for p in sidewalks_far],
        "tree_points": [pt_to_local_m(x, y, center_ft) for x, y in tree_points_ft],
        "lane_narrowing_stripes": lane_narrowing_stripes,
        "corner_hatching_lines": corner_hatching_lines,
        "corner_apron_polygons": corner_apron_polygons,
        "props": [
            {
                **p,
                "position_m": pt_to_local_m(p["position_ft"][0], p["position_ft"][1], center_ft),
                **({"arm_length_m": p["arm_length_ft"] * FT_TO_M} if "arm_length_ft" in p else {}),
            }
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
                # None (not drawn) unless this site's intersection is signalized (see stop_bar_offsets above).
                "stop_bar_offset_m": stop_bar_offsets[leg_name] * FT_TO_M if leg_name in stop_bar_offsets else None,
                # Real per-leg fact from config.yaml (street-view confirmed), not an OSM tag - see
                # src/geometry/treatments.py:set_centerline_style / DEFAULT_CENTERLINE_STYLE.
                "centerline_style": state.centerline_styles.get(leg_name, DEFAULT_CENTERLINE_STYLE),
            }
            for leg_name, leg in state.legs.items()
        ],
        "refuge_islands": [
            {
                "name": island_name,
                "coords": ring_to_local_m(island["polygon"].exterior.coords, center_ft),
                "height_m": 0.15,
            }
            for island_name, island in state.refuge_islands.items()
        ],
        "raised_crossings": [
            {"name": leg_name, "coords": ring_to_local_m(poly.exterior.coords, center_ft), "height_m": 0.10}
            for leg_name, poly in state.raised_crossings.items()
        ],
        "corner_parcels": [
            {"name": str(row["quadrant"]), "coords": ring_to_local_m(row.geometry.exterior.coords, center_ft)}
            for _, row in model.corner_parcels.iterrows()
        ],
        "buildings": building_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path
