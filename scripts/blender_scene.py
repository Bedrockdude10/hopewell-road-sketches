"""
Phase 4: headless Blender scene builder + renderer for one or more geometry
exports produced by scripts/phase4_export_geometry.py (or phase4_render_3d.py).

Not run with the project's normal Python - invoke via Blender's own
interpreter, which has no network access / requests / this project's venv.
Every real asset (textures, the streetlight model) is fetched beforehand in
the venv (src/theme.py) and passed in as local file paths via the JSON -
this script only ever reads files, never fetches them. Accepts any number of
<geometry.json> <output.png> pairs, all rendered in one Blender process (each
launch has ~1-1.5s of fixed startup overhead - paying it once for N renders
instead of N times is the single biggest lever for reducing total render time):

  blender --background --python scripts/blender_scene.py -- \\
      output/geometry_existing.json output/phase4_render_existing.png \\
      output/geometry_proposed.json output/phase4_render_proposed.png
"""
import json
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils

random.seed(7)  # stable building color assignment across existing/proposed renders

BUILDING_PALETTE = [
    (0.62, 0.42, 0.35),  # brick red
    (0.82, 0.78, 0.68),  # cream siding
    (0.55, 0.55, 0.58),  # gray
    (0.70, 0.62, 0.48),  # tan
    (0.45, 0.38, 0.32),  # dark brown
]

# MUTCD-ish colors for procedurally-built signage (no CC0 traffic-sign model
# was found - see README.md "Phase 4 fidelity"). Real geometric shape/color,
# just not a downloaded asset.
STOP_SIGN_RED = (0.55, 0.03, 0.03)
SCHOOL_ZONE_YELLOW_GREEN = (0.75, 0.85, 0.05)
SIGN_POST_GRAY = (0.35, 0.35, 0.37)


def parse_args() -> list[tuple[Path, Path]]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Usage: blender --background --python blender_scene.py -- <geometry.json> <output.png> [...]")
    args = argv[argv.index("--") + 1:]
    if len(args) < 2 or len(args) % 2 != 0:
        raise SystemExit("Need pairs of <geometry.json> <output.png>")
    return [(Path(args[i]), Path(args[i + 1])) for i in range(0, len(args), 2)]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def make_material(name: str, color: tuple, roughness: float = 0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_textured_material(name: str, texture_paths: dict | None, fallback_color: tuple,
                            fallback_roughness: float = 0.9):
    """Diffuse/Roughness/Normal-mapped material from local file paths (already
    downloaded by src/theme.py in the venv - this function never fetches
    anything). Falls back to a flat color material if texture_paths is falsy
    or any image fails to load, so a missing/corrupt file never crashes the render."""
    if not texture_paths:
        return make_material(name, fallback_color, fallback_roughness)
    try:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")

        def image_node(path: str, colorspace: str):
            node = nodes.new("ShaderNodeTexImage")
            img = bpy.data.images.load(path)
            img.colorspace_settings.name = colorspace
            node.image = img
            return node

        if texture_paths.get("Diffuse"):
            links.new(image_node(texture_paths["Diffuse"], "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
        if texture_paths.get("Rough"):
            links.new(image_node(texture_paths["Rough"], "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
        if texture_paths.get("nor_gl"):
            normal_map = nodes.new("ShaderNodeNormalMap")
            links.new(image_node(texture_paths["nor_gl"], "Non-Color").outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        return mat
    except Exception as e:
        print(f"  WARNING: textured material {name!r} failed ({e}) - falling back to flat color")
        return make_material(name, fallback_color, fallback_roughness)


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------

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


def extrude_polygon(name: str, coords_2d: list, height: float, material, uv_tile_m: float = 2.0):
    pts = coords_2d[:-1] if coords_2d[0] == coords_2d[-1] else coords_2d
    if len(pts) < 3:
        return None
    verts = [(x, y, 0.0) for x, y in pts]
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
    trimesh-decimated building - see src/mesh_utils.py) rather than extruding
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
                     length: float, width: float, height: float, material):
    corners = [
        center + u * (length / 2) + n * (width / 2),
        center + u * (length / 2) - n * (width / 2),
        center - u * (length / 2) - n * (width / 2),
        center - u * (length / 2) + n * (width / 2),
    ]
    extrude_polygon(name, [(p.x, p.y) for p in corners], height, material)


# ---------------------------------------------------------------------------
# Crosswalk styles
# ---------------------------------------------------------------------------

def _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m, stripe_width_m, gap_m):
    """Parallel bars (rungs) running along travel (u), spaced across the crossing (n).
    Returns (center, span) so callers (ladder) can reuse the layout for framing rails."""
    usable_width = max(width_m - 1.5, 0.5)  # keep clear of the curb edges
    period = stripe_width_m + gap_m
    n_stripes = max(int(usable_width / period), 1)
    span = (n_stripes - 1) * period
    center = near + u * offset_m
    for i in range(n_stripes):
        lateral = -span / 2 + i * period
        add_stripe_rect(f"{name}_stripe_{i}", center + n * lateral, u, n, depth_m, stripe_width_m, 0.06, material)
    return center, span


def add_crosswalk_continental(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                               depth_m: float = 3.0, stripe_width_m: float = 0.5, gap_m: float = 0.5):
    """Continental: parallel bars only, no framing rails."""
    _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m, stripe_width_m, gap_m)


def add_crosswalk_ladder(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                          depth_m: float = 3.0, stripe_width_m: float = 0.5, gap_m: float = 0.5,
                          rail_width_m: float = 0.3):
    """Ladder: continental bars framed by two rails spanning the crossing width at
    each end of the depth - the rails are what distinguish it from bare continental."""
    center, span = _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m, stripe_width_m, gap_m)
    rail_length = span + stripe_width_m + gap_m
    for side, sign in [("near", -1), ("far", 1)]:
        rail_center = center + u * (sign * depth_m / 2)
        add_stripe_rect(f"{name}_rail_{side}", rail_center, n, u, rail_length, rail_width_m, 0.06, material)


def add_crosswalk_lines(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                         depth_m: float = 3.0, line_width_m: float = 0.3):
    """Simple/standard marking: just two transverse lines bounding the crossing, no
    bars in between - the least visible of the three styles (FHWA/NACTO recommend
    upgrading this to continental or ladder for visibility, hence it being the
    'existing conditions' style here while proposed treatments upgrade it)."""
    line_width = max(width_m - 1.0, 0.5)
    center = near + u * offset_m
    for side, sign in [("near", -1), ("far", 1)]:
        line_center = center + u * (sign * depth_m / 2)
        add_stripe_rect(f"{name}_line_{side}", line_center, n, u, line_width, line_width_m, 0.06, material)


CROSSWALK_STYLES = {
    "lines": add_crosswalk_lines,
    "continental": add_crosswalk_continental,
    "ladder": add_crosswalk_ladder,
}


def add_crosswalk(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0, style: str = "lines"):
    draw_fn = CROSSWALK_STYLES.get(style, add_crosswalk_lines)
    draw_fn(name, near, u, n, width_m, material, offset_m=offset_m)


def add_dashed_centerline(name: str, near: mathutils.Vector, far: mathutils.Vector, material,
                           start_m: float = 6.0, dash_m: float = 1.0, gap_m: float = 1.0, width_m: float = 0.15):
    direction = far - near
    length = direction.length
    if length <= start_m:
        return
    u = direction / length
    n = mathutils.Vector((-u.y, u.x, 0))
    pos = start_m
    i = 0
    while pos + dash_m < length:
        center = near + u * (pos + dash_m / 2)
        add_stripe_rect(f"{name}_dash_{i}", center, u, n, dash_m, width_m, 0.06, material)
        pos += dash_m + gap_m
        i += 1


# ---------------------------------------------------------------------------
# Props: streetlight (real glTF asset, procedural fallback), signage (procedural
# - no CC0 traffic-sign source found, see README.md), trees (geometry-nodes
# instancing of one procedural low-poly mesh)
# ---------------------------------------------------------------------------

def import_gltf_template(gltf_path: str | None, name: str):
    """Import a glTF once and return it as a hidden template object for
    add_streetlight() to make cheap linked duplicates of (shared mesh data,
    not full copies - the actual performance-relevant instancing here).
    Returns None if there's no path or the import fails."""
    if not gltf_path or not Path(gltf_path).exists():
        return None
    try:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=gltf_path)
        imported = [o for o in bpy.data.objects if o not in before]
        if not imported:
            return None
        if len(imported) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in imported:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = imported[0]
            bpy.ops.object.join()
        template = bpy.context.view_layer.objects.active
        template.name = name
        template.hide_render = True
        template.hide_set(True)
        return template
    except Exception as e:
        print(f"  WARNING: glTF import failed for {gltf_path!r} ({e}) - using a procedural fallback instead")
        return None


def add_streetlight(name: str, position: tuple, heading_deg: float, template, pole_mat, head_mat):
    if template is not None:
        obj = template.copy()
        bpy.context.collection.objects.link(obj)
        obj.name = name
        obj.location = (position[0], position[1], 0.0)
        obj.rotation_euler = (0, 0, math.radians(heading_deg))
        obj.hide_render = False
        obj.hide_set(False)
        return obj

    # Procedural fallback: a plain pole + small head, used if the Poly Haven
    # model couldn't be fetched (no network) - not what ships when online.
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=4.5, location=(x, y, 2.25))
    pole = bpy.context.active_object
    pole.name = f"{name}_pole"
    pole.data.materials.append(pole_mat)
    bpy.ops.mesh.primitive_cube_add(size=0.35, location=(x, y, 4.6))
    head = bpy.context.active_object
    head.name = f"{name}_head"
    head.scale = (1, 1, 0.5)
    head.data.materials.append(head_mat)
    return pole


def _add_post_sign(name: str, position: tuple, heading_deg: float, n_sides: int, plate_radius: float,
                    plate_color: tuple, post_mat):
    """Shared shape for procedurally-built signage: a thin post + a flat
    regular-polygon plate facing `heading_deg`. Used for stop signs (n_sides=8,
    red) and the school zone sign (n_sides=5, yellow-green) - real MUTCD shapes
    and colors, just not a downloaded model (no CC0 traffic-sign source found)."""
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=2.1, location=(x, y, 1.05))
    post = bpy.context.active_object
    post.name = f"{name}_post"
    post.data.materials.append(post_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=plate_radius, depth=0.03, vertices=n_sides, location=(x, y, 2.15))
    plate = bpy.context.active_object
    plate.name = f"{name}_plate"
    plate.rotation_euler = (math.radians(90), 0, math.radians(heading_deg))
    plate_mat = make_material(f"{name}_plate_mat", plate_color, roughness=0.35)
    plate.data.materials.append(plate_mat)
    return post


def add_stop_sign(name: str, position: tuple, heading_deg: float, post_mat):
    return _add_post_sign(name, position, heading_deg, n_sides=8, plate_radius=0.3,
                           plate_color=STOP_SIGN_RED, post_mat=post_mat)


def add_school_zone_sign(name: str, position: tuple, heading_deg: float, post_mat):
    return _add_post_sign(name, position, heading_deg, n_sides=5, plate_radius=0.35,
                           plate_color=SCHOOL_ZONE_YELLOW_GREEN, post_mat=post_mat)


def build_tree_proxy(trunk_mat, foliage_mat):
    """A single low-poly procedural tree (cone + cylinder). No CC0 source of
    genuinely low-poly stylized trees was found - Poly Haven's tree models are
    realistic photoscanned assets (multi-material, alpha-masked foliage cards)
    disproportionately heavy for background dressing instanced many times over
    at this render's scale/distance. See README.md "Phase 4 fidelity"."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=2.0, vertices=6, location=(0, 0, 1.0))
    trunk = bpy.context.active_object
    trunk.name = "tree_trunk"
    trunk.data.materials.append(trunk_mat)

    bpy.ops.mesh.primitive_cone_add(radius1=1.3, depth=3.0, vertices=8, location=(0, 0, 3.3))
    foliage = bpy.context.active_object
    foliage.name = "tree_foliage"
    foliage.data.materials.append(foliage_mat)

    bpy.ops.object.select_all(action="DESELECT")
    trunk.select_set(True)
    foliage.select_set(True)
    bpy.context.view_layer.objects.active = trunk
    bpy.ops.object.join()
    tree = bpy.context.view_layer.objects.active
    tree.name = "tree_proxy_template"
    tree.hide_render = True
    tree.hide_set(True)
    return tree


def add_tree_instances(name: str, points: list, tree_template):
    """Geometry-nodes point instancing: ONE tree mesh's data is shared across
    every point (Instance on Points), not copied per-tree - the actual
    performance requirement behind 'not individual mesh copies'."""
    if not points or tree_template is None:
        return None

    mesh = bpy.data.meshes.new(f"{name}_points")
    mesh.from_pydata([(x, y, 0.0) for x, y in points], [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    node_group = bpy.data.node_groups.new(f"{name}_GN", "GeometryNodeTree")
    node_group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = node_group.nodes
    links = node_group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = tree_template

    links.new(group_input.outputs["Geometry"], instance_on_points.inputs["Points"])
    links.new(object_info.outputs["Geometry"], instance_on_points.inputs["Instance"])
    links.new(instance_on_points.outputs["Instances"], group_output.inputs["Geometry"])

    modifier = obj.modifiers.new(name=f"{name}_GN", type="NODES")
    modifier.node_group = node_group
    return obj


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def build_scene(data: dict):
    theme = data.get("theme") or {}
    asphalt_near = make_textured_material("AsphaltNear", theme.get("asphalt_near"), (0.07, 0.07, 0.08), 0.95)
    asphalt_far = make_textured_material("AsphaltFar", theme.get("asphalt_far"), (0.07, 0.07, 0.08), 0.95)
    concrete_near = make_textured_material("ConcreteNear", theme.get("concrete_near"), (0.72, 0.71, 0.67), 0.85)
    concrete_far = make_textured_material("ConcreteFar", theme.get("concrete_far"), (0.72, 0.71, 0.67), 0.85)
    lot = make_material("Lot", (0.55, 0.6, 0.48), roughness=0.9)
    grass = make_material("Grass", (0.3, 0.48, 0.24), roughness=1.0)
    refuge_mat = make_material("Refuge", (0.22, 0.5, 0.26), roughness=0.8)
    crossing_mat = make_material("RaisedCrossing", (0.68, 0.58, 0.48), roughness=0.8)
    marking_mat = make_material("Marking", (0.9, 0.9, 0.88), roughness=0.4)
    centerline_mat = make_material("Centerline", (0.85, 0.7, 0.15), roughness=0.4)
    building_mats = [make_material(f"Building{i}", c, roughness=0.75) for i, c in enumerate(BUILDING_PALETTE)]
    pole_mat = make_material("Pole", SIGN_POST_GRAY, roughness=0.5)
    trunk_mat = make_material("TreeTrunk", (0.32, 0.22, 0.15), roughness=0.9)
    foliage_mat = make_material("TreeFoliage", (0.16, 0.4, 0.14), roughness=0.85)

    all_pavement = data.get("pavement_near", []) + data.get("pavement_far", [])
    pavement_x = [x for ring in all_pavement for x, y in ring]
    pavement_y = [y for ring in all_pavement for x, y in ring]
    cx, cy = (min(pavement_x) + max(pavement_x)) / 2, (min(pavement_y) + max(pavement_y)) / 2
    # Frame the camera on the intersection itself (the actual subject), not the
    # full building-context radius - buildings are background dressing and are
    # fine to crop at the frame edges.
    pavement_radius = max(max(pavement_x) - min(pavement_x), max(pavement_y) - min(pavement_y)) / 2
    scene_radius = pavement_radius * 1.6

    all_x = pavement_x + [x for b in data.get("buildings", []) for x, y, *_ in
                           (b["vertices_m"] if b["mesh"] else b["coords"])]
    all_y = pavement_y + [y for b in data.get("buildings", []) for x, y, *_ in
                           (b["vertices_m"] if b["mesh"] else b["coords"])]
    context_radius = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2
    ground_size = max(context_radius * 2.5, 100)
    bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(cx, cy, -0.03))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(grass)

    for parcel in data.get("corner_parcels", []):
        extrude_polygon(f"parcel_{parcel['name']}", parcel["coords"], 0.0, lot)

    for i, b in enumerate(data.get("buildings", [])):
        mat = building_mats[i % len(building_mats)]
        if b["mesh"]:
            build_mesh_from_data(f"building_{i}", b["vertices_m"], b["faces"], mat)
        else:
            extrude_polygon(f"building_{i}", b["coords"], b["height_m"], mat)

    for i, ring in enumerate(data.get("pavement_near", [])):
        extrude_polygon(f"pavement_near_{i}", ring, 0.05, asphalt_near)
    for i, ring in enumerate(data.get("pavement_far", [])):
        extrude_polygon(f"pavement_far_{i}", ring, 0.05, asphalt_far)

    for i, ring in enumerate(data.get("sidewalks_near", [])):
        extrude_polygon(f"sidewalk_near_{i}", ring, 0.03, concrete_near)
    for i, ring in enumerate(data.get("sidewalks_far", [])):
        extrude_polygon(f"sidewalk_far_{i}", ring, 0.03, concrete_far)

    for island in data.get("refuge_islands", []):
        extrude_polygon(f"refuge_{island['name']}", island["coords"], island.get("height_m", 0.15), refuge_mat)

    for crossing in data.get("raised_crossings", []):
        extrude_polygon(
            f"crossing_{crossing['name']}", crossing["coords"], crossing.get("height_m", 0.10), crossing_mat
        )

    raised_leg_names = {c["name"] for c in data.get("raised_crossings", [])}
    # Only draw a painted crosswalk where one is actually confirmed to exist
    # (config: intersection.existing_marked_crosswalks) - don't assume every
    # approach is marked just because it's a signalized 4-way.
    marked_leg_names = set(data.get("existing_marked_crosswalks", []))
    for leg in data.get("legs", []):
        near = mathutils.Vector((*leg["near_m"], 0.0))
        far = mathutils.Vector((*leg["far_m"], 0.0))
        direction = far - near
        if direction.length < 1e-3:
            continue
        u = direction / direction.length
        n = mathutils.Vector((-u.y, u.x, 0))
        offset_m = leg.get("crosswalk_offset_m", 3.0)
        if leg["name"] in marked_leg_names and leg["name"] not in raised_leg_names:
            style = leg.get("crosswalk_style", "lines")
            add_crosswalk(f"crosswalk_{leg['name']}", near, u, n, leg["width_m"], marking_mat,
                           offset_m=offset_m, style=style)
        add_dashed_centerline(f"centerline_{leg['name']}", near, far, centerline_mat, start_m=offset_m + 2)

    # Props: real streetlight model (or procedural fallback) at each corner,
    # procedural signage (no CC0 traffic-sign source available - see README).
    streetlight_template = import_gltf_template(theme.get("streetlight_gltf"), "streetlight_template")
    for i, prop in enumerate(data.get("props", [])):
        pos, heading = prop["position_m"], prop["heading_deg"]
        if prop["type"] == "streetlight":
            add_streetlight(f"streetlight_{i}", pos, heading, streetlight_template, pole_mat, pole_mat)
        elif prop["type"] == "stop_sign":
            add_stop_sign(f"stop_sign_{i}", pos, heading, pole_mat)
        elif prop["type"] == "school_zone_sign":
            add_school_zone_sign(f"school_zone_sign_{i}", pos, heading, pole_mat)

    # Trees: one shared low-poly mesh, geometry-nodes-instanced along the
    # sidewalk bands (not one mesh copy per tree).
    tree_points = data.get("tree_points", [])
    if tree_points:
        tree_template = build_tree_proxy(trunk_mat, foliage_mat)
        add_tree_instances("street_trees", tree_points, tree_template)

    return cx, cy, scene_radius


def setup_camera_and_light(cx: float, cy: float, scene_radius: float):
    dist = scene_radius * 1.6
    height = scene_radius * 2.3
    bpy.ops.object.camera_add(location=(cx, cy - dist, height))
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam
    direction = mathutils.Vector((cx, cy, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 32

    bpy.ops.object.light_add(type="SUN", location=(cx + scene_radius * 0.3, cy - scene_radius * 0.3, height))
    sun = bpy.context.active_object
    sun.data.energy = 2.2
    sun.data.angle = 0.2  # soften shadow edges slightly
    sun.rotation_euler = (0.85, 0.15, 0.75)

    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.55, 0.68, 0.82, 1.0)
        bg.inputs["Strength"].default_value = 0.6


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64  # visually indistinguishable from 128 for this flat-shaded scene, ~30% faster
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1440


def render(output_path: Path):
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    jobs = parse_args()
    configure_render()  # render settings are scene-independent - set once
    for geometry_path, output_path in jobs:
        with open(geometry_path) as f:
            data = json.load(f)

        clear_scene()
        cx, cy, scene_radius = build_scene(data)
        setup_camera_and_light(cx, cy, scene_radius)
        render(output_path)
        print(f"RENDER_DONE: {output_path}")


if __name__ == "__main__":
    main()
