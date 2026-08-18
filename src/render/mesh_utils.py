"""3D mesh utilities for background/context geometry (OSM buildings) that isn't
the subject of the render and doesn't need full poly density. Not used for the
authoritative pavement/curb geometry - see src/geometry/model/ for that."""
import contextlib

from shapely.geometry import Polygon

# When a building footprint is heavy enough to be worth simplifying, and what to simplify it to.
#
# Set high on purpose. Quadric decimation does not know a building's roof is meant to be flat:
# on a short extrusion it collapses the cheapest edges, which are the vertical ones, leaving a
# crumpled tent. A straight-walled prism off an n-vertex footprint is ~4n-4 triangles, so a
# threshold of 40 fires on an ordinary house with a porch - and there is nothing to save anyway
# (all 80 buildings at Broad & Greenwood total 1,692 triangles). 400 faces is a ~100-sided
# outline, four times the most complex building at any of these junctions: the path stays ready
# for richer building data without firing on a house.
MAX_BUILDING_FACES_BEFORE_DECIMATION = 400
DECIMATE_TARGET_FACES = 200


def build_decimated_building_mesh(footprint: Polygon, height: float) -> tuple[list, list] | None:
    """Extrude a building footprint to a 3D mesh, decimating it only if complex enough to be
    worth it.

    Unit-agnostic: `height` must be in the same units as `footprint`'s coordinates (this
    project passes feet, matching the state plane CRS; export.py converts to metres after).
    Returns (vertices, faces) as JSON-serializable lists for blender_scene.py's from_pydata,
    or None if trimesh/its backends are unavailable - callers then extrude the 2D footprint
    themselves (still correct, just not decimated).
    """
    try:
        import trimesh
    except ImportError:
        return None

    try:
        mesh = trimesh.creation.extrude_polygon(footprint, height=height)
    except Exception:
        return None  # malformed footprint (self-intersecting, etc.) - let the caller fall back

    if len(mesh.faces) > MAX_BUILDING_FACES_BEFORE_DECIMATION:
        # A missing/failed decimation backend exports the un-decimated mesh rather than nothing.
        with contextlib.suppress(Exception):
            mesh = mesh.simplify_quadric_decimation(face_count=DECIMATE_TARGET_FACES)

    return mesh.vertices.tolist(), mesh.faces.tolist()
