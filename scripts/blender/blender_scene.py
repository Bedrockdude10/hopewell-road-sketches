"""
Phase 4: headless Blender scene builder + renderer for one or more geometry
exports produced by scripts/phase4_export_geometry.py (or phase4_render_3d.py).

Not run with the project's normal Python - invoke via Blender's own
interpreter, which has no network access / requests / this project's venv.
Every real asset (textures, the streetlight model) is fetched beforehand in
the venv (src/render/theme.py) and passed in as local file paths via the
JSON - this script only ever reads files, never fetches them. Accepts any
number of <geometry.json> <output.png> pairs, all rendered in one Blender
process (each launch has ~1-1.5s of fixed startup overhead - paying it once
for N renders instead of N times is the single biggest lever for reducing
total render time):

  blender --background --python scripts/blender/blender_scene.py -- \\
      output/geometry_existing.json output/phase4_render_existing.png \\
      output/geometry_proposed.json output/phase4_render_proposed.png

This file is the entry point + top-level scene assembly only - the actual
geometry-building code is split across sibling modules in this same
directory (plain local imports work fine under Blender's bundled Python, no
venv needed):
  blender_materials.py   flat-color and PBR-textured material builders
  blender_geometry.py    generic mesh helpers (extrude a ring, stripe rects)
  blender_crosswalks.py  the 3 painted crosswalk styles + dashed centerlines
  blender_props.py       street furniture: streetlights, signage, traffic
                          signals, trees - one builder function per prop type
"""
import json
import random
import sys
from pathlib import Path

import bpy
import mathutils

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling blender_*.py imports below

from blender_crosswalks import (
    add_crosswalk, add_dashed_centerline, add_double_yellow_centerline, add_paint_line, add_paint_polyline,
    add_stop_bar,
)
from blender_geometry import build_mesh_from_data, extrude_polygon
from blender_materials import make_material, make_textured_material
from blender_props import (
    PED_SIGNAL_HOUSING_DARK, SIGNAL_HOUSING_DARK, SIGN_POST_GRAY,
    add_prop, add_tree_instances, build_tree_proxy, import_gltf_template,
)

random.seed(7)  # stable building color assignment across existing/proposed renders

PAVEMENT_HEIGHT_M = 0.05
# crosswalks/centerlines/stop bars (add_crosswalk*/add_dashed_centerline/add_double_yellow_centerline/
# add_stop_bar) sit at blender_crosswalks.py:EXISTING_MARKING_Z_BASE (0.06) with thickness
# EXISTING_MARKING_THICKNESS_M (0.01) - this is their real top, i.e. EXISTING_MARKING_Z_BASE +
# EXISTING_MARKING_THICKNESS_M. Kept as its own constant here (rather than importing the two above)
# since this file only needs the single derived "top" value to stack the next layer above it.
EXISTING_MARKING_HEIGHT_M = 0.07
# The new paint-only overlay markings (lane narrowing, corner hatching, mountable apron) sit on top
# of EXISTING_MARKING_HEIGHT_M + this gap, NOT exactly at either that or PAVEMENT_HEIGHT_M - two
# surfaces at the exact same height are coincident/coplanar, which renders as flickering z-fighting
# (confirmed by an isolated test: a marking placed with zero gap above the pavement rendered as a
# visibly tessellated mess even as a flat, zero-height plane, ruling out "thin geometry aliasing" as
# the cause; a lane-narrowing stripe overlapping a crosswalk's footprint needed the SAME fix again
# relative to the crosswalk's own top height, not just the pavement's). ~1cm of clearance is
# imperceptible at this render's scale but enough to give the depth buffer an unambiguous answer.
#
# Separately, EXISTING_MARKING_Z_BASE=0 (the crosswalk/centerline/stop-bar layer's OLD z_base) had
# its own bug even though its 0.06 top height was never coincident with anything: z_base=0 meant its
# bottom fully overlapped the pavement's own 0-0.05 volume rather than sitting on top of it. That,
# combined with this camera's near/far clip range being far wider than the scene needed (see
# setup_camera_and_light) and so starving the depth buffer of precision at this camera's distance,
# produced a torn/tessellated look on thin, elongated shapes like a crosswalk line - confirmed by an
# isolated test. Fixed by both lifting z_base to sit flush on the pavement's top (see
# blender_crosswalks.py:EXISTING_MARKING_Z_BASE) and tightening the camera's clip range.
MARKING_CLEARANCE_M = 0.01

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
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def build_scene(data: dict):
    theme = data.get("theme") or {}
    asphalt_near = make_textured_material("AsphaltNear", theme.get("asphalt_near"), (0.07, 0.07, 0.08), 0.95)
    asphalt_far = make_textured_material("AsphaltFar", theme.get("asphalt_far"), (0.07, 0.07, 0.08), 0.95)
    concrete_near = make_textured_material("ConcreteNear", theme.get("concrete_near"), (0.72, 0.71, 0.67), 0.85)
    concrete_far = make_textured_material("ConcreteFar", theme.get("concrete_far"), (0.72, 0.71, 0.67), 0.85)
    apron_mat = make_textured_material("Apron", theme.get("apron_near"), (0.65, 0.6, 0.55), 0.8)
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
    signal_housing_mat = make_material("SignalHousing", SIGNAL_HOUSING_DARK, roughness=0.4)
    ped_signal_housing_mat = make_material("PedSignalHousing", PED_SIGNAL_HOUSING_DARK, roughness=0.4)

    all_pavement = data.get("pavement_near", []) + data.get("pavement_far", [])
    pavement_x = [x for ring in all_pavement for x, y in ring]
    pavement_y = [y for ring in all_pavement for x, y in ring]
    cx, cy = (min(pavement_x) + max(pavement_x)) / 2, (min(pavement_y) + max(pavement_y)) / 2
    # Frame the camera on the intersection itself (the actual subject), not the
    # full building-context radius - buildings are background dressing and are
    # fine to crop at the frame edges.
    pavement_radius = max(max(pavement_x) - min(pavement_x), max(pavement_y) - min(pavement_y)) / 2
    scene_radius = pavement_radius * 1.2  # tight enough to actually read paint markings/signage detail

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
        extrude_polygon(f"pavement_near_{i}", ring, PAVEMENT_HEIGHT_M, asphalt_near)
    for i, ring in enumerate(data.get("pavement_far", [])):
        extrude_polygon(f"pavement_far_{i}", ring, PAVEMENT_HEIGHT_M, asphalt_far)

    for i, ring in enumerate(data.get("sidewalks_near", [])):
        extrude_polygon(f"sidewalk_near_{i}", ring, 0.03, concrete_near)
    for i, ring in enumerate(data.get("sidewalks_far", [])):
        extrude_polygon(f"sidewalk_far_{i}", ring, 0.03, concrete_far)

    # Paint-only / no-curb-change proposal treatments (src/geometry/treatments.py:
    # add_lane_narrowing / add_corner_hatching / add_mountable_apron) - sit
    # above BOTH the pavement and the existing crosswalk/centerline markings
    # they can overlap (a stripe runs the whole leg, crossing the crosswalk),
    # with a small MARKING_CLEARANCE_M gap either way (see docstring above).
    marking_z = EXISTING_MARKING_HEIGHT_M + MARKING_CLEARANCE_M
    # A lane-narrowing buffer is a solid edge line (the new lane's real edge)
    # plus diagonal hatching filling the buffer beyond it - a real gore/chevron
    # marking, not a solid filled block of paint (which at this render's scale
    # was visually indistinguishable from a sidewalk/apron - see export.py).
    # The edge line is dead straight along the main run (a simple 2-point
    # line) but curves where it tapers into the corner (a many-point sampled
    # arc, see export.py/lane_narrowing_taper_ft) - add_paint_line only ever
    # draws a single straight chord between whatever two points it's given,
    # so a curved line needs add_paint_polyline instead (drawing every
    # consecutive segment) or it silently collapses into a straight diagonal.
    for i, line in enumerate(data.get("lane_narrowing_edge_lines", [])):
        add_paint_line(f"lane_narrowing_edge_{i}", line[0], line[-1], 0.25, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("lane_narrowing_taper_lines", [])):
        add_paint_polyline(f"lane_narrowing_taper_{i}", line, 0.15, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("lane_narrowing_hatch_lines", [])):
        add_paint_line(f"lane_narrowing_hatch_{i}", line[0], line[-1], 0.15, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("corner_hatching_lines", [])):
        add_paint_line(f"corner_hatch_{i}", line[0], line[-1], 0.15, marking_mat, z_base=marking_z)
    for i, ring in enumerate(data.get("corner_apron_polygons", [])):
        extrude_polygon(f"corner_apron_{i}", ring, 0.01, apron_mat, z_base=marking_z)
    for i, line in enumerate(data.get("parking_edge_lines", [])):
        add_paint_line(f"parking_edge_{i}", line[0], line[-1], 0.25, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("parking_stall_divider_lines", [])):
        add_paint_line(f"parking_stall_{i}", line[0], line[-1], 0.15, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("parking_buffer_edge_lines", [])):
        add_paint_line(f"parking_buffer_edge_{i}", line[0], line[-1], 0.25, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("parking_buffer_taper_lines", [])):
        add_paint_polyline(f"parking_buffer_taper_{i}", line, 0.15, marking_mat, z_base=marking_z)
    for i, line in enumerate(data.get("parking_buffer_hatch_lines", [])):
        add_paint_line(f"parking_buffer_hatch_{i}", line[0], line[-1], 0.15, marking_mat, z_base=marking_z)

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
        stop_bar_offset_m = leg.get("stop_bar_offset_m")
        if stop_bar_offset_m is not None:
            stop_bar_width_m = leg.get("stop_bar_width_m") or leg["width_m"]
            add_stop_bar(f"stop_bar_{leg['name']}", near, u, n, stop_bar_width_m, marking_mat,
                         offset_m=stop_bar_offset_m)
        # Real per-leg fact (confirmed via street-view, see src/geometry/treatments.py
        # DEFAULT_CENTERLINE_STYLE) - some legs get no centerline paint at all, so this
        # is NOT drawn unconditionally the way it used to be.
        centerline_style = leg.get("centerline_style", "single_yellow_dashed")
        if centerline_style == "double_yellow":
            add_double_yellow_centerline(f"centerline_{leg['name']}", near, far, centerline_mat,
                                          start_m=offset_m + 2)
        elif centerline_style == "single_yellow_dashed":
            add_dashed_centerline(f"centerline_{leg['name']}", near, far, centerline_mat, start_m=offset_m + 2)

    # Props: real streetlight model (or procedural fallback) at each corner,
    # procedural signage incl. traffic signals (no CC0 source available - see
    # blender_props.py / README.md). Placement is decided upstream by
    # src/render/props.py; add_prop() just dispatches each exported prop dict to its
    # builder.
    streetlight_template = import_gltf_template(theme.get("streetlight_gltf"), "streetlight_template")
    for i, prop in enumerate(data.get("props", [])):
        add_prop(f"{prop['type']}_{i}", prop, streetlight_template, pole_mat,
                 signal_housing_mat, ped_signal_housing_mat)

    # Trees: one shared low-poly mesh, geometry-nodes-instanced along the
    # sidewalk bands (not one mesh copy per tree).
    tree_points = data.get("tree_points", [])
    if tree_points:
        tree_template = build_tree_proxy(trunk_mat, foliage_mat)
        add_tree_instances("street_trees", tree_points, tree_template)

    return cx, cy, scene_radius, ground_size


def setup_camera_and_light(cx: float, cy: float, scene_radius: float, ground_size: float):
    dist = scene_radius * 1.6
    height = scene_radius * 2.3
    bpy.ops.object.camera_add(location=(cx, cy - dist, height))
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam
    direction = mathutils.Vector((cx, cy, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 32
    # Blender's default clip range (0.1 - 1000 m) is enormously wider than this
    # scene ever needs, which starves the depth buffer of precision at the
    # ~50-100 m distance this camera actually sits at - confirmed by an
    # isolated test: thin, long ground markings (crosswalk lines) rendered as
    # a torn/tessellated mess with the default clip range and perfectly solid
    # once the range was tightened to the scene's real extent, with shadow
    # settings held constant throughout (so this is a camera depth-buffer
    # precision issue, not a shadow one, despite looking similar to the
    # z-fighting/shadow-acne bugs documented elsewhere in this file/README).
    # ground_size is already the true worst-case scene extent (see build_scene) -
    # clip_end just needs to clear camera-to-farthest-ground-corner distance,
    # so dist + height + ground_size is a generous, cheap-to-compute upper bound.
    cam.data.clip_start = max(dist - scene_radius * 2, 1.0)
    cam.data.clip_end = dist + height + ground_size

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
        cx, cy, scene_radius, ground_size = build_scene(data)
        setup_camera_and_light(cx, cy, scene_radius, ground_size)
        render(output_path)
        print(f"RENDER_DONE: {output_path}")


if __name__ == "__main__":
    main()
