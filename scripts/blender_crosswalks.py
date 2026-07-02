"""Painted crosswalk styles (lines/continental/ladder) and dashed
centerlines. Imported by blender_scene.py - runs under Blender's bundled
Python. See README.md "Crosswalk styles: real data over guessing" for how a
leg's style is decided upstream in src/export.py."""
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
