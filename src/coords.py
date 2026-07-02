"""Coordinate conversions shared by the Phase 4 export: WGS84 -> NJ State
Plane feet -> local meters centered on the intersection (what
scripts/blender_scene.py actually consumes, since Blender's bundled Python
has no shapely/geopandas/pyproj)."""
import pyproj
from shapely.geometry import Polygon

from src.geometry_model import NJ_STATE_PLANE_FT, WGS84

FT_TO_M = 0.3048

wgs84_to_state_plane = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)


def ring_to_local_m(coords, center_ft) -> list[list[float]]:
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in coords]


def pt_to_local_m(x, y, center_ft) -> list[float]:
    return [(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M]


def wgs84_ring_to_local_m(coords_wgs84, center_ft) -> list[list[float]]:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in zip(xs, ys)]


def building_footprint_ft(coords_wgs84) -> Polygon:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return Polygon(zip(xs, ys))
