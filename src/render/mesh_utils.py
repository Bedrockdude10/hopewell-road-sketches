"""3D mesh utilities for background/context geometry (OSM buildings) that isn't
the subject of the render and doesn't need full poly density. Not used for the
authoritative pavement/curb geometry - see src/geometry/model.py for that."""
import contextlib

from shapely.geometry import Polygon

# When a building is heavy enough to be worth simplifying, and what to simplify it to.
#
# THIS WAS 40 FACES, DOWN TO 24, AND IT WAS DAMAGING THE RENDER. A straight-walled prism off an
# n-vertex footprint is about 4n-4 triangles, so 40 faces is an 11-sided building - which is not
# complex, it is an ordinary house with a porch. Measured at Broad & Greenwood: 9 of 80 buildings
# crossed that line (44, 44, 44, 44, 44, 52, 60, 68 and 100 faces) and were crushed to 24, and
# quadric decimation does not know that a building's roof is meant to be flat: it collapses
# whichever edges are cheapest, which on a short extrusion are the vertical ones, leaving a
# crumpled tent where the roof was. Four buildings rendered that way.
#
# The saving it bought: all 80 buildings undecimated total **1,692 triangles**, in a scene that
# also carries textured pavement, instanced trees and procedural signal heads. There was nothing
# to save. The threshold now sits where a footprint is genuinely heavy - 400 faces is a ~100-sided
# outline, four times the most complex building at any of these four junctions - so the path stays
# ready for richer building data (which is why it exists) without firing on a house.
MAX_BUILDING_FACES_BEFORE_DECIMATION = 400
DECIMATE_TARGET_FACES = 200


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
        # A missing/failed decimation backend exports the un-decimated mesh rather than nothing.
        with contextlib.suppress(Exception):
            mesh = mesh.simplify_quadric_decimation(face_count=DECIMATE_TARGET_FACES)

    return mesh.vertices.tolist(), mesh.faces.tolist()
