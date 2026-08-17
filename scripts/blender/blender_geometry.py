"""Shared low-level Blender mesh building: vertices and faces computed in Python, then handed to
Blender ONCE. Imported by blender_scene.py - runs under Blender's bundled Python.

WHY THIS MODULE LOOKS LIKE ARITHMETIC AND NOT LIKE BLENDER. Every helper here used to build its
mesh through the operator layer - `bpy.ops.object.mode_set(mode="EDIT")`, `bpy.ops.mesh.select_all`,
`bpy.ops.mesh.extrude_region_move`, back to OBJECT - and then `apply_planar_uv` did a SECOND
identical round trip for `bpy.ops.uv.cube_project`. Two edit-mode entries per polygon, and each one
bracketed by `bpy.ops.object.select_all(action="DESELECT")`.

That is three separate costs, and they compound:

  * `select_all(DESELECT)` walks EVERY object in the scene. Called about twice per object created,
    against a scene that is still growing, so building N objects costs O(N^2) - about 800,000 object
    visits for the ~900 pieces a wide render carries.
  * every `bpy.ops` call pushes a global undo snapshot and forces a depsgraph re-evaluation. ~4,000
    operator invocations for one render, each costing milliseconds regardless of how small the mesh
    is. A 4-vertex marking paid the same overhead as a building.
  * a mode switch tears down and rebuilds the object's BMesh.

None of it buys anything. A prism off a ring is closed-form: the ring at z_base, the same ring at
z_base + height, quad walls between them, a cap at each end. A planar UV is `x / tile, y / tile`.
Both are a few lines of arithmetic, and Blender only needs to hear about the result.

THE SAME LESSON THIS REPO ALREADY LEARNED IN 2D. src/render/plan_view.py:_draw groups same-styled
geometry into ONE matplotlib collection because per-collection cost grows with how many are already
on the axes: measured at 2.44 s against 0.016 s for the same geometry, 156x. `MeshBatch` below is
that argument applied to Blender - one mesh per material instead of one per piece, which takes a
wide render's object count from ~900 to about a dozen.
"""
import bpy


def prism(coords_2d, height: float, z_base: float = 0.0, first_index: int = 0):
    """(vertices, faces) for a closed 2D ring extruded to `height`, starting at vertex index n.

    `first_index` is what makes this batchable: the faces come back numbered from that offset, so a
    caller accumulating many prisms into one mesh passes the running vertex count and concatenates.

    A zero `height` gives the flat cap alone, which is what almost every marking is - paint has no
    thickness worth modelling, and the pieces that do (kerbs, refuge islands) pass a real height.

    Winding is preserved from the ring, and the top cap is reversed so both caps face outward. That
    matters under flat shading with backface culling off: a cap wound the wrong way reads as a hole.
    """
    pts = list(coords_2d)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return [], []
    n = len(pts)
    if height <= 0:
        verts = [(x, y, z_base) for x, y in pts]
        return verts, [tuple(range(first_index, first_index + n))]
    verts = [(x, y, z_base) for x, y in pts] + [(x, y, z_base + height) for x, y in pts]
    bottom = tuple(range(first_index, first_index + n))
    top = tuple(reversed(range(first_index + n, first_index + 2 * n)))
    walls = [(first_index + i, first_index + (i + 1) % n,
              first_index + n + (i + 1) % n, first_index + n + i) for i in range(n)]
    return verts, [bottom, top, *walls]


class MeshBatch:
    """Many pieces of one material, accumulated in Python and built as a single mesh.

    THE POINT IS THE OBJECT COUNT, not the vertex count. A scene's cost in Blender scales with how
    many OBJECTS it holds - depsgraph nodes, draw calls, and per-object overhead in EEVEE - far more
    than with how many triangles those objects contain. A wide render's ~900 markings are perhaps
    12,000 triangles between them, which is nothing; as 900 separate objects they are the whole cost
    of the render.

    Nothing is lost by merging. These pieces share a material, never move independently, and are
    never selected by name - they exist to be looked at. The one thing a caller gives up is a
    per-piece object name in the outliner, which no part of this pipeline reads.
    """

    def __init__(self, name: str, material):
        self.name = name
        self.material = material
        self.verts = []
        self.faces = []

    def add_prism(self, coords_2d, height: float, z_base: float = 0.0) -> None:
        verts, faces = prism(coords_2d, height, z_base, first_index=len(self.verts))
        self.verts += verts
        self.faces += faces

    def add_mesh_data(self, vertices, faces) -> None:
        """Pre-computed vertices and faces - a building prism from src/render/mesh_utils.py."""
        offset = len(self.verts)
        self.verts += [tuple(v) for v in vertices]
        self.faces += [tuple(index + offset for index in face) for face in faces]

    def build(self, uv_tile_m: float | None = None):
        """One object for everything added, or None if nothing was. UVs only where asked for."""
        if not self.faces:
            return None
        mesh = bpy.data.meshes.new(self.name)
        mesh.from_pydata(self.verts, [], self.faces)
        mesh.update()
        if not mesh.polygons:
            return None
        if uv_tile_m:
            planar_uvs(mesh, uv_tile_m)
        obj = bpy.data.objects.new(self.name, mesh)
        if self.material is not None:
            obj.data.materials.append(self.material)
        bpy.context.collection.objects.link(obj)
        return obj


def planar_uvs(mesh, tile_size_m: float = 2.0) -> None:
    """Project UVs from above at a fixed real-world tile size, computed rather than operated.

    What `bpy.ops.uv.cube_project` was being used for, and all it was doing for a top-down scene:
    every face here is either horizontal or a short vertical wall, so the projection that keeps a
    tiled texture at a consistent physical scale is just the world x,y divided by the tile size. A
    2 ft pavement tile stays 2 ft whether it is under a 500 ft slab or a 3 ft sidewalk wedge.

    Written straight into the loop layer with foreach_set, so this is one buffer copy rather than an
    edit-mode round trip per object. CALL IT ONLY FOR TEXTURED MATERIALS - the old helper's own
    docstring admitted it was "harmless no-op-ish for flat-color materials", which meant a full mode
    switch per object to compute coordinates nothing would ever sample.
    """
    uv_layer = mesh.uv_layers.new(name="UVMap")
    coords = [0.0] * (len(mesh.loops) * 2)
    positions = mesh.vertices
    for i, loop in enumerate(mesh.loops):
        x, y, _z = positions[loop.vertex_index].co
        coords[2 * i] = x / tile_size_m
        coords[2 * i + 1] = y / tile_size_m
    uv_layer.data.foreach_set("uv", coords)


def extrude_polygon(name: str, coords_2d: list, height: float, material, uv_tile_m: float = 2.0,
                     z_base: float = 0.0):
    """One ring as its own object. Kept for the pieces that really are one-of-a-kind.

    z_base lifts the solid to sit on top of something already at that height (a marking on the
    pavement slab) instead of starting at z=0 and overlapping it. IMPORTANT: do NOT set z_base to
    exactly the height of the surface it sits on - two coplanar faces at the same height z-fight
    (confirmed by an isolated test: even a flat, zero-height marking placed at z=pavement_height
    rendered as a visibly tessellated, flickering mess). Leave a small clearance gap instead - see
    blender_scene.py:MARKING_CLEARANCE_M.

    Prefer MeshBatch wherever there is more than one piece of a material. This exists for the
    handful that are genuinely singular - the ground plane, the pavement slab - where an object of
    its own costs nothing and a name is useful.
    """
    batch = MeshBatch(name, material)
    batch.add_prism(coords_2d, height, z_base)
    return batch.build(uv_tile_m=uv_tile_m)


def build_mesh_from_data(name: str, vertices: list, faces: list, material):
    """An object from precomputed vertices and faces - a building from src/render/mesh_utils.py.

    STILL USES ONE OPERATOR, deliberately, and it is the only one left in this module. trimesh
    triangulates everything it touches, so a plain box arrives as ~12 triangles, and
    `dissolve_limited` merges the coplanar ones back into flat faces. The recorded reason is that
    without it a simple building "reads as a faceted/crystalline shape under Blender's default flat
    shading".

    Removing it means changing what the EXPORTER emits - a prism with quad walls and n-gon caps,
    which needs nothing merged - and that is a separate change with its own visual risk, so it is
    not made here. Worth knowing that mesh_utils.py's own measurements say decimation never fires at
    these junctions (all 80 buildings total 1,692 triangles against a 400-face threshold), so the
    triangulate-then-merge round trip currently buys nothing but the appearance it protects.

    79 buildings is one round trip each, against the ~900 x 2 this module just removed, so it is no
    longer the thing worth fixing.
    """
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    if not mesh.polygons:
        return None
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)

    # Only the object being edited is selected, and it is deselected again after. NOT
    # `select_all(action="DESELECT")`, which walks every object in the scene and was the quadratic
    # term in the old builder.
    for other in bpy.context.selected_objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_limited()
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def add_stripe_rect(name, center, u, n, length: float, width: float, height: float, material,
                     z_base: float = 0.0):
    """A rectangle from a centre, an along-axis and an across-axis. Used by the crosswalk bars."""
    corners = [
        center + u * (length / 2) + n * (width / 2),
        center + u * (length / 2) - n * (width / 2),
        center - u * (length / 2) - n * (width / 2),
        center - u * (length / 2) + n * (width / 2),
    ]
    return extrude_polygon(name, [(p.x, p.y) for p in corners], height, material, z_base=z_base)


def stripe_rect_ring(center, u, n, length: float, width: float):
    """The four corners of add_stripe_rect's rectangle, for a caller batching many of them."""
    return [(p.x, p.y) for p in (
        center + u * (length / 2) + n * (width / 2),
        center + u * (length / 2) - n * (width / 2),
        center - u * (length / 2) - n * (width / 2),
        center - u * (length / 2) + n * (width / 2),
    )]


def line_ring(p1, p2, width_m: float):
    """The four corners of a stripe of `width_m` between two points, or None if they coincide."""
    import math

    (x1, y1), (x2, y2) = p1[:2], p2[:2]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy * width_m / 2, ux * width_m / 2
    return [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]


def polyline_rings(points, width_m: float):
    """One stripe ring per SEGMENT of a sampled polyline.

    A polyline here follows the traced kerb on a 2 ft station grid (model.STRIP_SAMPLE_FT), so a
    130 ft lane edge arrives as ~65 points. Drawn the old way - add_paint_polyline calling
    add_paint_line per segment, each building its own object - that one marking became 64 Blender
    objects, and the channel it belongs to became 1,209. Counted across a wide render the scene held
    3,753 objects where the JSON has 842 items.

    Segments rather than a single mitred ribbon on purpose: a mitred join needs the outer corner
    extended along the bisector, and on a kerb that turns sharply that overshoots into a spike. Butt
    joints overlap slightly at each vertex instead, which is invisible in flat white paint at this
    scale and cannot spike. Batched into one mesh by the caller, so the segment count no longer costs
    anything.
    """
    rings = []
    for i in range(len(points) - 1):
        ring = line_ring(points[i], points[i + 1], width_m)
        if ring is not None:
            rings.append(ring)
    return rings
