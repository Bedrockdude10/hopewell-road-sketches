"""Street-furniture prop builders: streetlight (real glTF asset, procedural
fallback), signage incl. traffic signals (procedural - no CC0
traffic-sign/signal-head source found, see README.md "Phase 4 fidelity"), and
trees (geometry-nodes instancing of one procedural low-poly mesh). Placement
(position/heading/which corner gets what) is decided upstream in
src/render/props.py - this module only ever draws a prop at a given position.
Imported by blender_scene.py - runs under Blender's bundled Python."""
import math
from pathlib import Path

import bpy

from blender_materials import make_material

# MUTCD-ish colors for procedurally-built signage (no CC0 traffic-sign model
# was found - see README.md "Phase 4 fidelity"). Real geometric shape/color,
# just not a downloaded asset.
STOP_SIGN_RED = (0.55, 0.03, 0.03)
SCHOOL_ZONE_YELLOW_GREEN = (0.75, 0.85, 0.05)
SIGN_POST_GRAY = (0.35, 0.35, 0.37)
NO_TURN_ON_RED_WHITE = (0.92, 0.92, 0.9)
SIGNAL_HOUSING_DARK = (0.08, 0.08, 0.08)
PED_SIGNAL_HOUSING_DARK = (0.1, 0.1, 0.1)
VEHICLE_SIGNAL_LENS_COLORS = [
    (0.85, 0.05, 0.05),  # red (top)
    (0.85, 0.65, 0.05),  # yellow (middle)
    (0.05, 0.55, 0.15),  # green (bottom)
]
TRAFFIC_SIGNAL_POLE_HEIGHT_M = 5.5  # taller than the streetlight pole (4.5 m) - matches a real signal pole
# Real arm length is a full-width mast arm (see sites/README.md / config.yaml signals.pole_type), computed
# per-corner from real adjacent leg widths in src/render/props.py and passed in as each prop's arm_length_m. This
# constant is only a fallback for a prop dict missing that field (e.g. a site with no signals.pole_type data).
TRAFFIC_SIGNAL_ARM_LENGTH_M = 2.2
PED_SIGNAL_MOUNT_HEIGHT_M = 2.3  # typical pedestrian signal head mounting height

# RRFB (Rectangular Rapid Flashing Beacon): a MUTCD W11-2 pedestrian-crossing
# diamond sign (same fluorescent yellow-green as the school zone sign) with two
# rectangular amber beacon bars mounted just below it. No CC0 RRFB/traffic-sign
# model exists on Poly Haven - checked api.polyhaven.com/assets?type=models for
# "sign"/"traffic"/"beacon"/"light"/"post" keywords and found nothing closer
# than a concrete road barrier - so this is procedural, like the other signage.
RRFB_SIGN_YELLOW_GREEN = SCHOOL_ZONE_YELLOW_GREEN
RRFB_BEACON_AMBER = (0.95, 0.55, 0.05)
RRFB_MOUNT_HEIGHT_M = 2.3

# Plastic flex-post delineator/bollard: real MUTCD/channelizer safety orange,
# with a white reflective band near the top (both real, common features - just
# no CC0 bollard model found, same "procedural but real colors/shape" approach
# as the rest of this file's signage).
BOLLARD_SAFETY_ORANGE = (0.85, 0.28, 0.03)
BOLLARD_REFLECTIVE_WHITE = (0.92, 0.92, 0.9)
BOLLARD_HEIGHT_M = 0.9
BOLLARD_RADIUS_M = 0.05


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


def add_vehicle_signal_head(name: str, position: tuple, heading_deg: float, housing_mat):
    """Procedural 3-section vehicle signal head: a dark housing box with 3
    stacked red/yellow/green lenses on the face pointed at `heading_deg` -
    real MUTCD color/layout, not a downloaded model (no CC0 traffic-signal
    source found - same approach already used for the stop sign)."""
    x, y, z = position
    face = math.radians(heading_deg)
    fx, fy = math.cos(face), math.sin(face)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z))
    housing = bpy.context.active_object
    housing.name = f"{name}_housing"
    housing.scale = (0.32, 0.32, 0.85)  # square cross-section - housing orientation doesn't matter visually
    housing.data.materials.append(housing_mat)

    for i, color in enumerate(VEHICLE_SIGNAL_LENS_COLORS):
        lens_pos = (x + fx * 0.17, y + fy * 0.17, z + 0.24 - i * 0.24)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.09, depth=0.03, vertices=16, location=lens_pos)
        lens = bpy.context.active_object
        lens.name = f"{name}_lens_{i}"
        lens.rotation_euler = (math.radians(90), 0, face)  # same flat-disc-facing-heading trick as sign plates
        lens.data.materials.append(make_material(f"{name}_lens_{i}_mat", color, roughness=0.3))
    return housing


def add_traffic_signal_pole(name: str, position: tuple, head_facing_deg: float, pole_mat, housing_mat,
                             arm_heading_deg: float | None = None, arm_length_m: float = TRAFFIC_SIGNAL_ARM_LENGTH_M):
    """Full-width mast-arm signal - the confirmed pole type for this
    intersection (NOT a short pole-mounted rigid/davit arm, NOT span-wire; see
    sites/README.md `signals` block / config.yaml). A tall post + a
    horizontal arm + a procedural 3-section vehicle head at the arm's end.

    arm_heading_deg and head_facing_deg are DIFFERENT directions (not a fixed
    180 degrees apart): the arm extends at a right angle to the one leg it's
    built for (see src/render/props.py:_traffic_signal_props for which leg and why),
    while the head faces back down that same leg toward oncoming traffic -
    those are perpendicular axes, not opposite ends of one axis. Falls back
    to the old "arm opposite the head" behavior if arm_heading_deg isn't
    given (e.g. a prop dict from a site/version that doesn't set it).
    arm_length_m is computed upstream (src/render/props.py) from the real leg width
    this arm actually spans, not hardcoded."""
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.1, depth=TRAFFIC_SIGNAL_POLE_HEIGHT_M, location=(x, y, TRAFFIC_SIGNAL_POLE_HEIGHT_M / 2)
    )
    pole = bpy.context.active_object
    pole.name = f"{name}_pole"
    pole.data.materials.append(pole_mat)

    arm_dir = math.radians(arm_heading_deg if arm_heading_deg is not None else head_facing_deg + 180)
    dx, dy = math.cos(arm_dir), math.sin(arm_dir)
    arm_z = TRAFFIC_SIGNAL_POLE_HEIGHT_M - 0.4
    arm_center = (x + dx * arm_length_m / 2, y + dy * arm_length_m / 2, arm_z)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=arm_length_m, location=arm_center)
    arm = bpy.context.active_object
    arm.name = f"{name}_arm"
    arm.rotation_euler = (0, math.radians(90), arm_dir)  # lay the cylinder flat, then point it along arm_dir
    arm.data.materials.append(pole_mat)

    head_pos = (x + dx * arm_length_m, y + dy * arm_length_m, arm_z - 0.2)
    add_vehicle_signal_head(f"{name}_head", head_pos, head_facing_deg, housing_mat)
    return pole


def add_pedestrian_signal_head(name: str, position: tuple, heading_deg: float, own_post: bool,
                                housing_mat, post_mat):
    """Small pedestrian signal head. If `own_post`, mounted on its own short
    post (this corner is confirmed to have the ped head on a SEPARATE pole
    from the vehicle signal); otherwise just the head at typical mounting
    height, implicitly co-located with the vehicle signal pole already drawn
    at this same position (same pole - see sites/README.md `signals` block)."""
    x, y = position
    if own_post:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.05, depth=PED_SIGNAL_MOUNT_HEIGHT_M, location=(x, y, PED_SIGNAL_MOUNT_HEIGHT_M / 2)
        )
        post = bpy.context.active_object
        post.name = f"{name}_post"
        post.data.materials.append(post_mat)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, PED_SIGNAL_MOUNT_HEIGHT_M))
    head = bpy.context.active_object
    head.name = f"{name}_head"
    head.scale = (0.28, 0.28, 0.32)
    head.rotation_euler = (0, 0, math.radians(heading_deg))
    head.data.materials.append(housing_mat)
    return head


def add_no_turn_on_red_sign(name: str, position: tuple, heading_deg: float, post_mat):
    """Small rectangular NO TURN ON RED restriction sign (MUTCD R10-11 series -
    white plate; real shape/color, not a downloaded model). Shares
    _add_post_sign's post-height convention but with a rectangular plate
    instead of a regular polygon - stop/school-zone signs are octagon/pentagon,
    NTOR signs are rectangular."""
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=2.1, location=(x, y, 1.05))
    post = bpy.context.active_object
    post.name = f"{name}_post"
    post.data.materials.append(post_mat)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, 2.2))
    plate = bpy.context.active_object
    plate.name = f"{name}_plate"
    plate.scale = (0.02, 0.3, 0.2)  # thin along local X (the facing/normal axis, before the Z rotation below)
    plate.rotation_euler = (0, 0, math.radians(heading_deg))
    plate_mat = make_material(f"{name}_plate_mat", NO_TURN_ON_RED_WHITE, roughness=0.35)
    plate.data.materials.append(plate_mat)
    return post


def add_rrfb(name: str, position: tuple, heading_deg: float, post_mat):
    """Procedural Rectangular Rapid Flashing Beacon: a diamond pedestrian-
    crossing warning sign (MUTCD W11-2) with two amber beacon bars mounted
    below it. Real installations typically pair a matching unit on the
    opposite curb - only one assembly is modeled per exported prop entry (see
    src/geometry/treatments.py:add_extra_prop)."""
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=RRFB_MOUNT_HEIGHT_M, location=(x, y, RRFB_MOUNT_HEIGHT_M / 2))
    post = bpy.context.active_object
    post.name = f"{name}_post"
    post.data.materials.append(post_mat)

    # Diamond sign: a square plate, tilted 45 deg about its own facing/normal
    # axis (local X, rotated first) before the whole assembly is turned to face
    # heading_deg (local Z, rotated last) - same two-step convention as the
    # octagon/pentagon sign plates in _add_post_sign, generalized to a square.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, RRFB_MOUNT_HEIGHT_M + 0.15))
    sign = bpy.context.active_object
    sign.name = f"{name}_sign"
    sign.scale = (0.03, 0.4, 0.4)
    sign.rotation_euler = (math.radians(45), 0, math.radians(heading_deg))
    sign_mat = make_material(f"{name}_sign_mat", RRFB_SIGN_YELLOW_GREEN, roughness=0.35)
    sign.data.materials.append(sign_mat)

    face = math.radians(heading_deg)
    fx, fy = math.cos(face), math.sin(face)
    for i in range(2):
        beacon_z = RRFB_MOUNT_HEIGHT_M - 0.15 - i * 0.15
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + fx * 0.05, y + fy * 0.05, beacon_z))
        beacon = bpy.context.active_object
        beacon.name = f"{name}_beacon_{i}"
        beacon.scale = (0.03, 0.35, 0.08)
        beacon.rotation_euler = (0, 0, face)
        beacon_mat = make_material(f"{name}_beacon_{i}_mat", RRFB_BEACON_AMBER, roughness=0.3)
        beacon.data.materials.append(beacon_mat)
    return post


def add_bollard(name: str, position: tuple):
    """A single plastic flex-post delineator: a short safety-orange cylinder
    with a white reflective band near the top. Placement (which leg, spacing,
    centered in the painted buffer) is decided upstream in
    src/render/props.py:_bollard_props - heading is irrelevant for a
    rotationally-symmetric post, so unlike the other props this one takes no
    heading_deg."""
    x, y = position
    bpy.ops.mesh.primitive_cylinder_add(radius=BOLLARD_RADIUS_M, depth=BOLLARD_HEIGHT_M,
                                         location=(x, y, BOLLARD_HEIGHT_M / 2))
    post = bpy.context.active_object
    post.name = f"{name}_post"
    post.data.materials.append(make_material(f"{name}_post_mat", BOLLARD_SAFETY_ORANGE, roughness=0.5))

    band_z = BOLLARD_HEIGHT_M * 0.7
    bpy.ops.mesh.primitive_cylinder_add(radius=BOLLARD_RADIUS_M * 1.02, depth=BOLLARD_HEIGHT_M * 0.15,
                                         location=(x, y, band_z))
    band = bpy.context.active_object
    band.name = f"{name}_band"
    band.data.materials.append(make_material(f"{name}_band_mat", BOLLARD_REFLECTIVE_WHITE, roughness=0.2))
    return post


def add_prop(name: str, prop: dict, streetlight_template, pole_mat, signal_housing_mat, ped_signal_housing_mat):
    """Build the Blender geometry for one exported prop dict (placement
    decided upstream by src/render/props.py), dispatching on its "type" field to the
    matching builder above. Kept next to the builders so adding a new prop
    type never requires touching blender_scene.py."""
    pos, heading, ptype = prop["position_m"], prop["heading_deg"], prop["type"]
    if ptype == "streetlight":
        add_streetlight(name, pos, heading, streetlight_template, pole_mat, pole_mat)
    elif ptype == "stop_sign":
        add_stop_sign(name, pos, heading, pole_mat)
    elif ptype == "school_zone_sign":
        add_school_zone_sign(name, pos, heading, pole_mat)
    elif ptype == "traffic_signal_pole":
        add_traffic_signal_pole(name, pos, heading, pole_mat, signal_housing_mat,
                                 arm_heading_deg=prop.get("arm_heading_deg"),
                                 arm_length_m=prop.get("arm_length_m", TRAFFIC_SIGNAL_ARM_LENGTH_M))
    elif ptype == "pedestrian_signal_head":
        add_pedestrian_signal_head(name, pos, heading, prop.get("own_post", False), ped_signal_housing_mat, pole_mat)
    elif ptype == "no_turn_on_red_sign":
        add_no_turn_on_red_sign(name, pos, heading, pole_mat)
    elif ptype == "rrfb":
        add_rrfb(name, pos, heading, pole_mat)
    elif ptype == "bollard":
        add_bollard(name, pos)


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
