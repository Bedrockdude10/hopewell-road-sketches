"""
Phase 4: headless Blender scene builder + renderer for one or more geometry
exports produced by scripts/phase4_export_geometry.py (or phase4_render_3d.py).

Not run with the project's normal Python - invoke via Blender's own
interpreter. Accepts any number of <geometry.json> <output.png> pairs, all
rendered in one Blender process (each Blender launch has ~1-1.5s of fixed
startup overhead - paying it once for N renders instead of N times is the
single biggest lever for reducing total render time):

  blender --background --python scripts/blender_scene.py -- \\
      output/geometry_existing.json output/phase4_render_existing.png \\
      output/geometry_proposed.json output/phase4_render_proposed.png
"""
import json
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
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)


def make_material(name: str, color: tuple, roughness: float = 0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def extrude_polygon(name: str, coords_2d: list, height: float, material):
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


def build_scene(data: dict):
    asphalt = make_material("Asphalt", (0.07, 0.07, 0.08), roughness=0.95)
    concrete = make_material("Concrete", (0.72, 0.71, 0.67), roughness=0.85)
    lot = make_material("Lot", (0.55, 0.6, 0.48), roughness=0.9)
    grass = make_material("Grass", (0.3, 0.48, 0.24), roughness=1.0)
    refuge_mat = make_material("Refuge", (0.22, 0.5, 0.26), roughness=0.8)
    crossing_mat = make_material("RaisedCrossing", (0.68, 0.58, 0.48), roughness=0.8)
    marking_mat = make_material("Marking", (0.9, 0.9, 0.88), roughness=0.4)
    centerline_mat = make_material("Centerline", (0.85, 0.7, 0.15), roughness=0.4)
    building_mats = [make_material(f"Building{i}", c, roughness=0.75) for i, c in enumerate(BUILDING_PALETTE)]

    pavement_x = [x for x, y in data["pavement"]]
    pavement_y = [y for x, y in data["pavement"]]
    cx, cy = (min(pavement_x) + max(pavement_x)) / 2, (min(pavement_y) + max(pavement_y)) / 2
    # Frame the camera on the intersection itself (the actual subject), not the
    # full building-context radius - buildings are background dressing and are
    # fine to crop at the frame edges.
    pavement_radius = max(max(pavement_x) - min(pavement_x), max(pavement_y) - min(pavement_y)) / 2
    scene_radius = pavement_radius * 1.6

    all_x = pavement_x + [x for b in data.get("buildings", []) for x, y in b["coords"]]
    all_y = pavement_y + [y for b in data.get("buildings", []) for x, y in b["coords"]]
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
        extrude_polygon(f"building_{i}", b["coords"], b["height_m"], mat)

    extrude_polygon("pavement", data["pavement"], 0.05, asphalt)

    for i, ring in enumerate(data.get("sidewalks", [])):
        extrude_polygon(f"sidewalk_{i}", ring, 0.03, concrete)

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
        # Skip the painted ladder crosswalk where a raised crossing treatment
        # already marks this leg's crossing distinctly - drawing both stacks
        # near-identical geometry right on top of each other.
        offset_m = leg.get("crosswalk_offset_m", 3.0)
        if leg["name"] in marked_leg_names and leg["name"] not in raised_leg_names:
            style = leg.get("crosswalk_style", "lines")
            add_crosswalk(f"crosswalk_{leg['name']}", near, u, n, leg["width_m"], marking_mat,
                           offset_m=offset_m, style=style)
        add_dashed_centerline(f"centerline_{leg['name']}", near, far, centerline_mat, start_m=offset_m + 2)

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
