"""Coordinate conversions shared by the Phase 4 export: WGS84 -> NJ State
Plane feet -> local meters centered on the intersection (what
scripts/blender/blender_scene.py actually consumes, since Blender's bundled Python
has no shapely/geopandas/pyproj)."""
import pyproj
from shapely.geometry import Polygon

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from shapely.geometry import Point

FT_TO_M = 0.3048

wgs84_to_state_plane = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)

# Rounding applied to every float in the exported JSON (units there are metres, so 1e-6 is a
# micrometre). Purpose is diff legibility, not file size: without it, repr(float) emits 17
# significant digits and float64 noise in the last few of them means any upstream change to
# operation order perturbs every vertex of a ~500,000 ft state-plane coordinate, so a changed
# line no longer means a changed shape. 6 rather than 3-4 because not every exported number is
# a length - `crosswalk_axis` is a unit vector, where absolute rounding is an ANGLE: 1e-6 is
# 0.1 mm over a 100 m leg, 1e-4 would be 1 cm. One precision for the document must be safe for
# the least forgiving field.
EXPORT_DECIMALS = 6


def round_for_export(value):
    """`value` with every float rounded to EXPORT_DECIMALS, structure otherwise untouched.

    Applied once to the whole document at serialization rather than at each of the ~40 places
    that build a coordinate list, so the guarantee is about the FILE: a marking channel added
    later gets it for free. ints stay ints (`faces` are vertex indices and must not gain a
    `.0`); bools are tested first only because `isinstance(True, int)` is true.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, EXPORT_DECIMALS)
    if isinstance(value, dict):
        return {k: round_for_export(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_for_export(v) for v in value]
    return value


def ring_to_local_m(coords, center_ft: "Point") -> list[list[float]]:
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in coords]


def pt_to_local_m(x, y, center_ft: "Point") -> list[float]:
    return [(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M]


def wgs84_ring_to_local_m(coords_wgs84, center_ft: "Point") -> list[list[float]]:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return [[(x - center_ft.x) * FT_TO_M, (y - center_ft.y) * FT_TO_M] for x, y in zip(xs, ys)]


def building_footprint_ft(coords_wgs84) -> Polygon:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords_wgs84], [c[1] for c in coords_wgs84])
    return Polygon(zip(xs, ys))
