"""3D mesh utilities for background/context geometry (OSM buildings) that isn't
the subject of the render and doesn't need full poly density. Not used for the
authoritative pavement/curb geometry - see geometry_model.py for that."""
from shapely.geometry import Polygon

MAX_BUILDING_FACES_BEFORE_DECIMATION = 40  # a straight-walled prism from a simple
# footprint is already this cheap or cheaper (a rectangular box is 12 triangles) -
# only bother decimating footprints complex enough to produce more than this.
DECIMATE_TARGET_FACES = 24


def build_decimated_building_mesh(footprint: Polygon, height: float) -> tuple[list, list] | None:
    """
    Extrude a building footprint to a 3D mesh and decimate it if it's complex
    enough to be worth it. Unit-agnostic - `height` must be in the same units
    as `footprint`'s coordinates (this project passes feet, matching the state
    plane CRS everything else is built in; export.py converts to meters after).
    Returns (vertices, faces) as plain lists (JSON-serializable) for
    blender_scene.py to build directly via from_pydata, or None if trimesh/its
    triangulation+decimation backends aren't available - callers should fall
    back to extruding the 2D footprint themselves (still correct, just not
    decimated).
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
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=DECIMATE_TARGET_FACES)
        except Exception:
            pass  # decimation backend missing/failed - export the un-decimated mesh rather than nothing

    return mesh.vertices.tolist(), mesh.faces.tolist()
