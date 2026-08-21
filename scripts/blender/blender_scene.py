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
import math
import os
import random
import sys
from pathlib import Path

import bpy
import mathutils

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling blender_*.py imports below
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/blender/blender_scene.py -> repo root

from blender_crosswalks import (
    add_crosswalk, add_dashed_centerline, add_double_yellow_centerline, add_paint_polyline,
    add_stop_bar,
)
from blender_geometry import (MeshBatch, build_mesh_from_data, extrude_polygon,
                              line_ring, polyline_rings)
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
# How thick a painted marking is built. Was add_paint_line's own `height_m=0.01` default, which is
# where every batched marking's thickness came from before the draw block stopped going through it -
# named here so the value is stated rather than inherited from a keyword default two modules away.
# Paint has no meaningful thickness; this exists only to give the depth buffer something to order.
PAINT_HEIGHT_M = 0.01

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


# The keys whose ABSENCE from a geometry file would produce a picture that is wrong rather than
# incomplete, so a file without them is refused instead of rendered. Every one of them is read
# below through `.get(..., [])`, which is right for a scenario that legitimately has none of
# something - and indistinguishable from a file too old to carry the key at all. That is not a
# cosmetic difference: 39 of the 65 exports committed at the time predated these four, so one drew
# a street with no kerbs, no driveways or parking aprons and no surveyed crossings, and with
# `frame` gone this script computed a camera extent of its own from the pavement, which
# src/render/export.py says in as many words it must not do. Nothing warned; the render looked
# fine. "A picture that shows a marked crosswalk as bare asphalt is not a conservative
# simplification - it is a false statement about the street, made to an audience deciding whether
# to build something" (docs/network-renderer-plan.md).
#
# NOT the whole 37-key schema, which lives in src/render/export.py and cannot be imported here
# (see .importlinter: this file runs in Blender's interpreter). These four are what this RENDERER
# needs in order not to lie. Nothing checks the committed files against that full schema: the
# exporter writes every key unconditionally, so only a file committed BEFORE a key existed can
# lack one, and this guard is what stops such a file being rendered rather than reported.
REQUIRED_KEYS = ("frame", "kerbs", "paved_surfaces", "surveyed_crossings")


def load_geometry(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise SystemExit(
            f"{path}: geometry export is stale - no {', '.join(missing)}. Rendering it would "
            f"silently draw a street without them. Re-export with scripts/build_all.py.")
    return data


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

# How far past a leg's own far end a pavement vertex may still sit and be treated as part of
# this junction. A traced kerb's last vertex lands a foot or two beyond the leg it bounds, and
# the corner fillets' trimmed curbs run to their tangent points rather than to the centerline's
# end - so this absorbs that, and nothing like the 3x overshoot a kerb drawn down the whole
# block produces. See build_scene's framing note.
LEG_REACH_TOLERANCE = 1.05

# How wide a kerb is built. A real kerb's top face is about 6 in; this only has to read as an
# edge at the camera distance, and the height is what carries the raised/lowered distinction.
KERB_WIDTH_M = 0.15

# One stripe of a centerline, matching add_double_yellow_centerline's own width_m - MUTCD's ~6 in.
# The two lines of a double yellow arrive already offset from each other, so this is the width of
# each, not of the pair.
CENTERLINE_WIDTH_M = 0.15

# WHICH PAINT CHANNELS ARE SAMPLED POLYLINES, and which are honestly two-point segments. Declared
# as data rather than left implicit in the loops below, because the distinction is load-bearing and a
# test guards it (tests/test_paint.py).
#
# A SAMPLED POLYLINE MUST BE WALKED SEGMENT BY SEGMENT. These follow the traced kerb on a 2 ft
# station grid, so drawing the chord between the first and last vertex is not the line: it deviated
# 0.7 ft on Broad St's daylight zone, which pulled the painted edge inside the 11 ft lane it marks
# and lifted it off the hatching it bounds. The value beside each is its stripe width in metres - a
# drawn-scale choice, not a standard: a solid edge line reads at 0.25 m here, a taper at 0.15 m.
SAMPLED_POLYLINE_CHANNELS = (
    ("lane_narrowing_edge_lines", 0.25),
    ("lane_narrowing_taper_lines", 0.15),
    ("parking_edge_lines", 0.25),
    ("parking_buffer_edge_lines", 0.25),
    ("parking_buffer_taper_lines", 0.15),
    ("bike_lane_edge_lines", 0.25),
)
# Two-point strokes: a hatch stroke runs edge to edge of its zone and a stall tick lies across the
# kerbside strip. Only their two ends exist, so the chord IS the line.
TWO_POINT_CHANNELS = (
    "lane_narrowing_hatch_lines", "corner_hatching_lines", "parking_stall_divider_lines",
    "parking_buffer_hatch_lines", "bike_lane_hatch_lines",
)
TWO_POINT_WIDTH_M = 0.15



def _marking_frame(leg: dict, near, u, n, prefix: str, fallback_offset_m: float):
    """(centre, u, n) for a marking, from the geometry JSON where it says.

    `u` is the leg's near->far CHORD, and stepping an offset along it is only the same point
    crosswalk_axes picks while the centerline is straight. Two of these legs are not:
    broad_st_east kinks 4.5 deg 43.1 ft out where NJDOT rounds the corner, and
    louellen_st_west 29.4 deg 15.4 ft out. On broad_st_east the chord rotated the crosswalk
    bars 4.54 deg off the plan view's and drove them through 12.6 ft of paint the plan view
    had cleared - a 2D/3D disagreement of exactly the kind the shared geometry is supposed to
    make impossible. src/render/export.py resolves the frame now; this is the fallback for a
    geometry file written before it did.

    Module level, taking the leg and its frame as arguments, rather than a closure inside
    build_scene's per-leg loop: a closure reads the loop variables as they are AT CALL TIME,
    which is only the intended leg while every call stays in the iteration that defined it.
    """
    centre_m = leg.get(f"{prefix}_centre_m")
    axis = leg.get(f"{prefix}_axis")
    if centre_m is None or axis is None:
        return near + u * fallback_offset_m, u, n
    axis_u = mathutils.Vector((axis[0], axis[1], 0.0))
    return (mathutils.Vector((*centre_m, 0.0)), axis_u,
            mathutils.Vector((-axis_u.y, axis_u.x, 0.0)))


def resolve_theme_paths(theme: dict) -> dict:
    """`theme` with every asset path resolved against THIS checkout's root.

    src/render/theme.py writes them repo-relative (`output/.textures/...`) so a geometry file
    does not carry the absolute paths of the machine that exported it - the 65 committed exports
    each held 19 of those, naming a directory that exists on one laptop. Joining them here is the
    other half of that; an ALREADY-ABSOLUTE path is passed through, which is what a geometry file
    written before that change contains.

    Missing files are still not an error - make_textured_material and import_gltf_template each
    fall back - so this only has to name the right place to look.
    """
    resolved = {}
    for key, value in theme.items():
        if isinstance(value, dict):
            resolved[key] = {k: _under_repo(v) for k, v in value.items()}
        else:
            resolved[key] = _under_repo(value)
    return resolved


def _under_repo(path):
    if not isinstance(path, str) or os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)


def build_scene(data: dict):
    theme = resolve_theme_paths(data.get("theme") or {})
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
    kerb_mat = make_material("Kerb", (0.62, 0.61, 0.58), roughness=0.8)
    # A green bike lane's surface colour. Matches plan_view.py's "mediumseagreen" closely
    # enough that the two views read as the same treatment - the plan view draws it
    # semi-transparent over grey paper, this over black asphalt, so they cannot be identical
    # numbers. Rough like the asphalt it is painted on rather than glossy like fresh stripes.
    bike_surface_mat = make_material("BikeLaneSurface", (0.13, 0.45, 0.28), roughness=0.85)
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
    # WHERE THE CAMERA POINTS IS RESOLVED IN src/render/frame.py AND CARRIED IN THE JSON, so this
    # render and the plan view frame the same ground. Computing it here as well is what let the
    # two views drift: the plan view framed a hardcoded 110 ft square on the junction node while
    # this framed the pavement's own extent, and on the four sites the two disagreed by 1.15-1.57x
    # and by 6.5-12.5 ft of centre. The block below is the fallback for a geometry file written
    # before `frame` existed, and it is also the definition src/render/frame.py implements - the
    # same pair of numbers, computed once on the side that can be tested.
    #
    # Frame the camera on the intersection itself (the actual subject), not the
    # full building-context radius - buildings are background dressing and are
    # fine to crop at the frame edges.
    #
    # MEASURED AGAINST THE MODELLED LEGS, not against every pavement vertex. The pavement ring
    # is stitched from the corner fillets' trimmed curbs, and those curbs are TRACED OSM
    # barrier=kerb ways, which do not stop where our leg does: at E Broad & Princeton,
    # e_broad_st_west's left kerb runs 425 ft from the junction off a 130 ft leg, because the
    # mapper drew one continuous kerb down the block. That single vertex made the pavement
    # bounding box 168 x 78 m, put its centre 41 m down that leg and framed the camera at a
    # 100.6 m radius against ~53 m at the other three sites - a render zoomed nearly two-fold
    # out and not even pointed at the junction.
    #
    # A leg's own far end IS the edge of what this project modelled, so anything past it is
    # kerb running on down the street rather than part of this junction. All four sites have a
    # few such vertices (1, 4, 6 and 4 of them); dropping them tightens every render and
    # centres all four, rather than special-casing the one site where it had become glaring.
    leg_reach = max((math.hypot(*leg["far_m"]) for leg in data.get("legs", [])), default=0.0)
    framed = [(x, y) for x, y in zip(pavement_x, pavement_y)
              if not leg_reach or math.hypot(x, y) <= leg_reach * LEG_REACH_TOLERANCE]
    framed_x = [x for x, _y in framed] or pavement_x
    framed_y = [y for _x, y in framed] or pavement_y
    cx, cy = (min(framed_x) + max(framed_x)) / 2, (min(framed_y) + max(framed_y)) / 2
    pavement_radius = max(max(framed_x) - min(framed_x), max(framed_y) - min(framed_y)) / 2
    scene_radius = pavement_radius * 1.2  # tight enough to actually read paint markings/signage detail
    frame = data.get("frame")
    if frame:
        cx, cy = frame["center_m"]
        scene_radius = frame["radius_m"]

    # The GROUND still covers everything, framed or not: a plane that stopped at the framed
    # extent would leave the far end of an over-long kerb standing over blank space.
    all_x = pavement_x + [x for b in data.get("buildings", []) for x, y, *_ in
                           (b["vertices_m"] if b["mesh"] else b["coords"])]
    all_y = pavement_y + [y for b in data.get("buildings", []) for x, y, *_ in
                           (b["vertices_m"] if b["mesh"] else b["coords"])]
    context_radius = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2
    # AND AT LEAST FOUR TIMES THE FRAME, because the camera can be asked to pull back further than
    # the context reaches (src/render/frame.py's HOPEWELL_FRAME_SCALE, for a picture whose subject
    # is longer than one junction). On a wide frame the ground ran out inside the shot and the
    # horizon showed the plane's own edge with sky under it - the buildings and pavement had all
    # been drawn correctly on a groundsheet too small for the view.
    ground_size = max(context_radius * 2.5, scene_radius * 4, 100)
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

    # The paved ground beside the carriageway - driveways, parking aisles and parking lots - as
    # the POLYGON src/ built, the same one the plan view fills, so the two views cannot disagree
    # about where any of it is. All one asphalt, which is what they are; `kind` names each object
    # so a scene is readable in the outliner. `driveways` is the pre-parking key, kept as the
    # fallback for a geometry file written before this one.
    # Extruded to the pavement's own height so it reads as connected paving where it meets the
    # road. A driveway running off past the modelled legs is drawn where it really is; that it
    # ends in grass is our road model stopping, not the driveway being wrong.
    for i, drive in enumerate(data.get("paved_surfaces", data.get("driveways", []))):
        coords = drive.get("coords") or []
        if len(coords) >= 3:
            extrude_polygon(f"{drive.get('kind', 'driveway')}_{i}", coords,
                            PAVEMENT_HEIGHT_M, asphalt_far)

    # The traced kerbs, at the height their OSM kerb= tag calls for (src/render/export.py:
    # KERB_HEIGHT_M). There was no kerb in this scene before - the road slab simply met the
    # concrete band - so a 6 in stood-up kerb and a driveway's dropped kerb looked the same, and
    # the kerbside markings that now BREAK over a dropped kerb had nothing visible to break for.
    #
    # add_paint_polyline, not extrude_polygon: a kerb is a band of constant width following a
    # sampled line, which is exactly what that builder makes, and drawing the chord between the
    # endpoints instead would cut every corner the tracing turns.
    for i, kerb in enumerate(data.get("kerbs", [])):
        coords = kerb.get("coords") or []
        if len(coords) < 2:
            continue
        add_paint_polyline(f"kerb_{kerb.get('kerb', 'unknown')}_{i}", coords, KERB_WIDTH_M,
                           kerb_mat, height_m=kerb.get("height_m", 0.20), z_base=0.0)

    # Paint-only / no-curb-change proposal treatments (src/geometry/treatments/:
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
    # add_paint_polyline, not add_paint_line(line[0], line[-1]): these are sampled polylines
    # that follow the traced kerb, and drawing the chord between their endpoints throws away
    # every vertex in between. It deviated 0.7 ft on Broad St's daylight zone - enough to put
    # the painted lane edge inside the 11 ft it is supposed to mark, and to pull the line off
    # the hatching it is supposed to bound. The hatch strokes and stall ticks below really are
    # two-point segments, so add_paint_line is right for those.
    # EVERY WHITE MARKING IN ONE MESH, and the yellow ones in another. What this replaced was a
    # loop per channel calling add_paint_polyline, which called add_paint_line per SEGMENT, which
    # built an object each: a 130 ft lane edge sampled every 2 ft became 64 objects, its channel
    # became 1,209, and a wide render's scene held 3,753 objects for 842 JSON items. Batched, it is
    # two. See blender_geometry.MeshBatch, and src/render/plan_view.py:_draw for the same argument
    # winning the same 156x in matplotlib.
    #
    # Grouped by MATERIAL rather than by channel, because that is the only thing a merge has to
    # respect - a mesh carries one material, and every white marking shares one. The channel
    # distinctions above it (which line is a lane edge, which a stall tick) are decided upstream in
    # src/geometry/markings.py and are already spent by the time the geometry arrives here.
    #
    # POLYLINES, NOT CHORDS. polyline_rings walks every segment, so a sampled arc still curves; the
    # old add_paint_line(line[0], line[-1]) shortcut deviated 0.7 ft on Broad St's daylight zone,
    # enough to pull the painted lane edge inside the 11 ft it marks.
    white = MeshBatch("paint_white", marking_mat)
    yellow = MeshBatch("paint_yellow", centerline_mat)
    # (channel, stripe width) - the two widths are a drawn-scale choice, not a standard: a solid
    # edge line reads at 0.25 m here and a hatch stroke at 0.15 m.
    for key, width in SAMPLED_POLYLINE_CHANNELS:
        for line in data.get(key, []):
            for ring in polyline_rings(line, width):
                white.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    # The hatch strokes and stall ticks really are two-point segments, so only their ends matter.
    for key in TWO_POINT_CHANNELS:
        for line in data.get(key, []):
            ring = line_ring(line[0], line[-1], TWO_POINT_WIDTH_M)
            if ring is not None:
                white.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    # A TWO-WAY LANE'S CENTRE STRIPE IS YELLOW, and the channel is what decides that: every
    # edge-line channel above is drawn in the white marking material, and a yellow line is not a
    # white line somewhere else. Same distinction the roadway centreline gets, for the same reason -
    # yellow means opposing directions. Already cut into dashes upstream.
    for line in data.get("bike_lane_contraflow_lines", []):
        for ring in polyline_rings(line, CENTERLINE_WIDTH_M):
            yellow.add_prism(ring, PAINT_HEIGHT_M, z_base=marking_z)
    white.build()
    yellow.build()

    for i, ring in enumerate(data.get("corner_apron_polygons", [])):
        extrude_polygon(f"corner_apron_{i}", ring, 0.01, apron_mat, z_base=marking_z)
    # The bike lane's own asphalt, painted green. UNDER the stripe layer by one clearance gap, so the
    # white edge lines sit on top of the green the way they do on a real street - and so the two never
    # end up coplanar, which is the z-fighting this file's header is mostly about. Half a clearance
    # thick, so its TOP stays below the stripe layer's base rather than landing exactly on it.
    green = MeshBatch("bike_lane_surface", bike_surface_mat)
    for ring in data.get("bike_lane_surface_polygons", []):
        green.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z - MARKING_CLEARANCE_M)
    green.build()

    # The BIKE LANE symbol, white on the green. AT the stripe layer rather than half a clearance
    # below it like the green is, because the symbol is paint applied ON the coloured surface -
    # same height as the edge lines, which is what stops it z-fighting with the green it sits on.
    symbols = MeshBatch("bike_lane_symbol", marking_mat)
    for ring in data.get("bike_lane_symbol_polygons", []):
        symbols.add_prism(ring, MARKING_CLEARANCE_M / 2, z_base=marking_z)
    symbols.build()

    # EVERY SURVEYED CROSSING IN THE PICTURE, drawn from its own traced way rather than rebuilt
    # from a leg. This is the network-renderer change (docs/network-renderer-plan.md): a crossing
    # used to reach the render only by matching one of the modelled junction's legs, so at Broad &
    # Greenwood framed 2.5x, 6 of the 10 OSM crossings inside the frame were dropped - three of
    # them tagged crossing:markings=zebra, and Blackwell & Broad rendered as bare asphalt where its
    # crosswalks are traced. A render that shows a marked crosswalk as unmarked is a false claim
    # about the street, which is the one thing these drawings cannot afford.
    #
    # Alongside `kerbs` and `paved_surfaces` rather than in the paint channels, and for the same
    # reason those two are: this is SURVEYED CONTEXT, not a treatment this project proposes. It is
    # already in the ground's coordinates, it belongs to no leg, and nothing should be cut around it.
    #
    # The style comes from the crossing's own tags upstream - zebra becomes bars, `lines` becomes
    # two transverse lines, and a crossing with nothing recorded contributes neither. Blender
    # derives nothing here, which is the rule on this side of the boundary.
    for i, crossing in enumerate(data.get("surveyed_crossings", [])):
        for j, ring in enumerate(crossing.get("bars", [])):
            extrude_polygon(f"surveyed_crossing_{i}_bar_{j}", ring, MARKING_CLEARANCE_M / 2,
                             marking_mat, z_base=marking_z)
        for j, line in enumerate(crossing.get("lines", [])):
            add_paint_polyline(f"surveyed_crossing_{i}_line_{j}", line, 0.25, marking_mat,
                                z_base=marking_z)

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
    # Depth comes from src/render/crosswalks.py:CROSSWALK_DEPTH_M via the JSON, so the 2D plan
    # view (src/render/plan_view.py) and this render draw an identically-sized crosswalk.
    crosswalk_depth_m = data.get("crosswalk_depth_m", 1.829)  # 6 ft; see blender_crosswalks.py
    stop_bar_curb_clearance_m = data.get("stop_bar_curb_clearance_m", 0.5)
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
            cw_centre, cw_u, cw_n = _marking_frame(leg, near, u, n, "crosswalk", offset_m)
            add_crosswalk(f"crosswalk_{leg['name']}", cw_centre, cw_u, cw_n, leg["width_m"],
                           marking_mat,
                           offset_m=0.0, style=style, depth_m=crosswalk_depth_m,
                           skew_deg=leg.get("crosswalk_skew_deg", 0.0),
                           reach_left_m=leg.get("crosswalk_reach_left_m"),
                           reach_right_m=leg.get("crosswalk_reach_right_m"),
                           n_stripes=leg.get("crosswalk_bar_count"))
        stop_bar_offset_m = leg.get("stop_bar_offset_m")
        if stop_bar_offset_m is not None:
            stop_bar_width_m = leg.get("stop_bar_width_m") or leg["width_m"]
            # A stop bar is painted parallel to the crosswalk ahead of it, so it takes
            # the same surveyed skew.
            sb_centre, sb_u, sb_n = _marking_frame(leg, near, u, n, "stop_bar", stop_bar_offset_m)
            add_stop_bar(f"stop_bar_{leg['name']}", sb_centre, sb_u, sb_n, stop_bar_width_m,
                         marking_mat,
                         offset_m=0.0, skew_deg=leg.get("crosswalk_skew_deg", 0.0),
                         curb_clearance_m=stop_bar_curb_clearance_m,
                         span_m=leg.get("stop_bar_span_m"),
                         lateral_offset_m=leg.get("stop_bar_lateral_offset_m"))
        # Real per-leg fact (confirmed via street-view, see src/geometry/treatments/
        # DEFAULT_CENTERLINE_STYLE) - some legs get no centerline paint at all, so this
        # is NOT drawn unconditionally the way it used to be.
        centerline_style = leg.get("centerline_style", "single_yellow_dashed")
        # Where the paint starts is decided upstream (src/render/crosswalks.py:
        # centerline_start_ft) so the "stop at the stop bar" rule is testable and can't
        # drift from the bar this same scene draws. Falls back to the old fixed gap past
        # the crosswalk for geometry written before that field existed.
        centerline_start_m = leg.get("centerline_start_m", offset_m + 2)
        # THE STRIPES COME DOWN AS GEOMETRY, following the leg's real centerline - already
        # offset into the two lines of a double yellow, already cut into the segments of a
        # dashed one (src/render/crosswalks.py:centerline_paint_ft). The two calls below build
        # a stripe between near and far instead, which is the leg's CHORD: up to 3.98 ft off
        # the centerline on broad_st_east and 7.58 ft on louellen_st_west, so the double yellow
        # missed the stop bar it is supposed to meet and the lanes either side of it came out
        # different widths. They remain as the fallback for a geometry file written before this
        # key, and they are the reason nothing may derive a marking's shape on this side of the
        # boundary: the plan view had it right the whole time and nothing could compare them.
        painted = leg.get("centerline_paint_m")
        if painted is not None:
            for i, line in enumerate(painted):
                # The raw [x, y] pairs, as every other add_paint_polyline caller passes: it
                # reaches add_paint_line, which builds its own 3D vectors with `(*p, 0.0)`.
                # Handing it mathutils.Vector((x, y, 0)) instead made that `(x, y, 0, 0.0)` and
                # Blender refused the addition - 13 scenes failed to render at all.
                add_paint_polyline(f"centerline_{leg['name']}_{i}", line,
                                    CENTERLINE_WIDTH_M, centerline_mat)
        elif centerline_style == "double_yellow":
            add_double_yellow_centerline(f"centerline_{leg['name']}", near, far, centerline_mat,
                                          start_m=centerline_start_m)
        elif centerline_style == "single_yellow_dashed":
            add_dashed_centerline(f"centerline_{leg['name']}", near, far, centerline_mat,
                                   start_m=centerline_start_m)

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


# The render's own resolution, which --dpi does NOT control: that knob is matplotlib's and
# reaches only the 2D plan views. Setting --dpi 300 and expecting sharper renders is the
# obvious mistake and somebody made it, so there is now a knob for this too - a whole-number
# multiple of the base size, from HOPEWELL_RENDER_SCALE (scripts/build_all.py --render-scale).
# A multiplier rather than a width/height pair keeps the camera framing and the 4:3 aspect
# fixed, so scale 2 is the same picture with four times the pixels, not a different crop.
BASE_RESOLUTION = (1920, 1440)
RENDER_SCALE_ENV = "HOPEWELL_RENDER_SCALE"


def render_scale() -> int:
    """The resolution multiplier, clamped to something a machine can actually finish.

    Scale 4 is 7680x5760, which is where EEVEE's memory use starts to matter alongside the
    ~11 GB a scene already costs (see phase4_render_3d.BLENDER_PEAK_RAM_GB) - past that the
    OOM killer arrives and Blender says nothing about why, so the cap is kinder than the
    crash. Anything unparseable falls back to 1 with a warning rather than failing a batch
    of renders over an environment variable.
    """
    raw = os.environ.get(RENDER_SCALE_ENV, "1")
    try:
        scale = int(raw)
    except ValueError:
        print(f"WARNING: {RENDER_SCALE_ENV}={raw!r} is not an integer - rendering at 1x.")
        return 1
    if scale < 1 or scale > 4:
        print(f"WARNING: {RENDER_SCALE_ENV}={scale} is outside 1-4 - clamping.")
    return max(1, min(scale, 4))


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64  # visually indistinguishable from 128 for this flat-shaded scene, ~30% faster
    scale = render_scale()
    scene.render.resolution_x = BASE_RESOLUTION[0] * scale
    scene.render.resolution_y = BASE_RESOLUTION[1] * scale
    if scale != 1:
        print(f"Rendering at {scene.render.resolution_x}x{scene.render.resolution_y} ({scale}x)")


def render(output_path: Path):
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def disable_undo():
    """Stop Blender snapshotting the scene after every operator.

    Undo exists for a person clicking in the UI. In a headless batch there is nobody to undo for,
    and the cost is not small: every `bpy.ops` call pushes a snapshot of the scene, so the price of
    an operator grows with the scene it is called in. This build still uses operators for the
    buildings' face merge and for the props, which run late, when the scene is at its largest.

    Set once at startup rather than per render, because it is a preference and not scene state.
    """
    bpy.context.preferences.edit.use_global_undo = False


def main():
    jobs = parse_args()
    disable_undo()
    configure_render()  # render settings are scene-independent - set once
    for geometry_path, output_path in jobs:
        data = load_geometry(geometry_path)
        clear_scene()
        cx, cy, scene_radius, ground_size = build_scene(data)
        setup_camera_and_light(cx, cy, scene_radius, ground_size)
        render(output_path)
        print(f"RENDER_DONE: {output_path}")


if __name__ == "__main__":
    main()
