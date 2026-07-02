"""Painted crosswalk styles (lines/continental/ladder) and dashed
centerlines. Imported by blender_scene.py - runs under Blender's bundled
Python. See README.md "Crosswalk styles: real data over guessing" for how a
leg's style is decided upstream in src/render/export.py."""
import mathutils

from blender_geometry import add_stripe_rect


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


def add_stop_bar(name: str, near, u, n, width_m: float, material, offset_m: float, line_width_m: float = 0.5):
    """Stop bar: a single transverse line telling drivers where to stop for the
    signal, drawn just behind (intersection side of) the leg's crosswalk.
    Spans only the entering half of the road - `n` is the leg's own 'left'
    direction relative to its outward centerline direction (see
    src/render/props.py's left/right convention), which is the entering driver's
    right-hand side under US right-hand traffic (they travel the *opposite*
    way along the leg, so the sides swap) - a real stop bar never crosses
    into the opposing/receiving lanes, unlike a crosswalk line which spans
    the full width."""
    half_width = width_m / 2
    lane_span = max(half_width - 0.5, 0.5)  # keep clear of the centerline and the curb edge
    lane_center = near + u * offset_m + n * (half_width / 2)  # centered within the entering half only
    add_stripe_rect(f"{name}_bar", lane_center, n, u, lane_span, line_width_m, 0.06, material)


def add_paint_line(name: str, p1: tuple, p2: tuple, width_m: float, material,
                    height_m: float = 0.01, z_base: float = 0.06):
    """A single thin painted line segment between two points - used for
    corner-hatching diagonal lines (src/geometry/model.py:hatch_lines_ft) and
    any other simple paint-only marking that's just a straight stripe.
    z_base defaults just above the pavement's own top surface (0.05 m, per
    blender_scene.py's PAVEMENT_HEIGHT_M) - sitting exactly AT that height
    instead of slightly above it z-fights (see extrude_polygon's z_base
    docstring); callers that know the real pavement height should pass their
    own PAVEMENT_HEIGHT_M + MARKING_CLEARANCE_M explicitly instead."""
    p1v, p2v = mathutils.Vector((*p1, 0.0)), mathutils.Vector((*p2, 0.0))
    direction = p2v - p1v
    length = direction.length
    if length < 1e-6:
        return
    u = direction / length
    n = mathutils.Vector((-u.y, u.x, 0))
    add_stripe_rect(name, (p1v + p2v) / 2, u, n, length, width_m, height_m, material, z_base=z_base)


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
