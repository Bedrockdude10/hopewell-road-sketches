"""Shared low-level Blender mesh-building helpers: extruding a 2D ring into a
solid, building a mesh directly from precomputed vertices/faces, and a
stripe/rectangle primitive used by both crosswalk markings and centerlines.
Imported by blender_scene.py - runs under Blender's bundled Python."""
import bpy
import mathutils


def apply_planar_uv(obj, tile_size_m: float = 2.0):
    """Project UVs from above at a fixed real-world tile size, so a tiled
    texture reads at a consistent physical scale across differently-sized
    pieces (pavement vs. a small sidewalk wedge) rather than stretching to
    fill each mesh. Harmless no-op-ish for flat-color materials."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=tile_size_m)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def extrude_polygon(name: str, coords_2d: list, height: float, material, uv_tile_m: float = 2.0,
                     z_base: float = 0.0):
    """z_base lifts the whole solid to sit on top of something already at that
    height (e.g. a paint marking on top of the pavement slab) instead of
    starting at z=0 and mostly overlapping it. IMPORTANT: don't set z_base to
    exactly the height of the surface it sits on - two coincident/coplanar
    faces at the exact same height z-fight (confirmed by an isolated test:
    even a flat, zero-height marking placed at z=pavement_height rendered as
    a visibly tessellated/flickering mess). Give it a small clearance gap
    above that height instead (see blender_scene.py:MARKING_CLEARANCE_M)."""
    pts = coords_2d[:-1] if coords_2d[0] == coords_2d[-1] else coords_2d
    if len(pts) < 3:
        return None
    verts = [(x, y, z_base) for x, y in pts]
    faces = [tuple(range(len(verts)))]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if not mesh.polygons:
        return None
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    if height > 0:
        bpy.ops.object.select_all(action="DESELECT")  # multi-object edit mode would otherwise re-extrude others
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, height)})
        bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(False)

    obj.data.materials.append(material)
    apply_planar_uv(obj, uv_tile_m)
    return obj


def build_mesh_from_data(name: str, vertices: list, faces: list, material):
    """Build an object directly from precomputed vertices/faces (e.g. a
    trimesh-decimated building - see src/render/mesh_utils.py) rather than extruding
    a 2D ring ourselves."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if not mesh.polygons:
        return None
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)

    # trimesh always triangulates (even a plain box becomes ~12 triangles),
    # which reads as a faceted/crystalline shape under Blender's default flat
    # shading - merge coplanar triangles back into flat faces so a simple
    # building still looks like a clean box, not a low-poly gemstone.
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_limited()
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def add_stripe_rect(name, center: mathutils.Vector, u: mathutils.Vector, n: mathutils.Vector,
                     length: float, width: float, height: float, material, z_base: float = 0.0):
    corners = [
        center + u * (length / 2) + n * (width / 2),
        center + u * (length / 2) - n * (width / 2),
        center - u * (length / 2) - n * (width / 2),
        center - u * (length / 2) + n * (width / 2),
    ]
    extrude_polygon(name, [(p.x, p.y) for p in corners], height, material, z_base=z_base)
