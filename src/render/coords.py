"""Coordinate conversions shared by the Phase 4 export: WGS84 -> NJ State
Plane feet -> local meters centered on the intersection (what
scripts/blender/blender_scene.py actually consumes, since Blender's bundled Python
has no shapely/geopandas/pyproj)."""
import pyproj
from shapely.geometry import Polygon

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84

FT_TO_M = 0.3048

wgs84_to_state_plane = pyproj.Transformer.from_crs(WGS84, NJ_STATE_PLANE_FT, always_xy=True)

# HOW MANY DECIMALS SURVIVE INTO THE EXPORTED JSON. The units there are metres, so this is a
# micrometre - about eight orders of magnitude finer than anything this project can claim to
# know, and four finer than the 1 mm at which two renders would differ visibly.
#
# It exists because the alternative is repr(float), i.e. 17 significant digits, and float64
# arithmetic noise lives in the last two or three of them. A state-plane coordinate is ~500,000
# ft, so an operation order that changes anywhere upstream perturbs every vertex at ~1e-9 ft -
# invisible in the render and TOTAL in the diff. One 30-line change to src/geometry/model/
# rewrote 53,394 lines of geometry JSON that way, and a diff in which every line changed cannot
# answer the question worth asking of it: did this edit move geometry I did not mean to move?
# Truncating below the noise floor makes a changed line mean a changed shape.
#
# It is deliberately NOT tuned to file size. 6 decimals rather than 3 or 4 because a few
# exported numbers are not lengths - `crosswalk_axis` is a unit vector, where the same absolute
# rounding is an ANGLE - and 1e-6 there is 0.1 mm over a 100 m leg, while 1e-4 would be 1 cm.
# One precision for the whole document is only safe if it is safe for the least forgiving field.
EXPORT_DECIMALS = 6


def round_for_export(value):
    """`value` with every float rounded to EXPORT_DECIMALS, structure otherwise untouched.

    Applied once to the whole document at the point of serialization rather than at each of the
    ~40 places that build a coordinate list, because the guarantee wanted is about the FILE - a
    new marking channel added later gets it without anyone remembering to.

    ints pass through as ints (`faces` are vertex indices and must not gain a `.0`), and bools,
    strings and None are left alone. Note `isinstance(True, int)` is true, so bools are checked
    for first even though they are not floats and would not be rounded anyway - a reader should
    not have to know that to be sure.
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
