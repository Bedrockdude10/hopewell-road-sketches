"""Painted crosswalk styles (lines/continental/ladder) and centerline styles
(single dashed yellow, solid double yellow, or none). Imported by
blender_scene.py - runs under Blender's bundled Python. See README.md
"Crosswalk styles: real data over guessing" for how a leg's crosswalk style
is decided upstream in src/render/export.py, and
src/geometry/treatments.py:DEFAULT_CENTERLINE_STYLE for centerline style."""
import math

import mathutils

from blender_geometry import add_stripe_rect

# Existing-conditions markings (crosswalks, centerlines, stop bars) are a thin
# decal that sits flush on TOP of the pavement slab, not a thick block whose
# bottom coincides with the pavement's own bottom at z=0. The original version
# of this file used z_base=0, height=0.06 for all of these - the marking's
# bottom fully overlapped the pavement's own volume instead of sitting on top
# of it. That coincidence, combined with this render's camera having a far
# wider near/far clip range than the scene needed (see blender_scene.py:
# setup_camera_and_light), was confirmed (by an isolated test eliminating
# other candidate causes one at a time) to starve the depth buffer of enough
# precision at this camera's distance, producing a torn/tessellated look on
# thin, elongated shapes like a crosswalk line. Both fixes were needed
# together: this z_base lift, and tightening the camera's clip range.
# blender_scene.py's EXISTING_MARKING_HEIGHT_M (this layer's real top, 0.07)
# is what the newer paint-only treatments (lane narrowing, corner hatching,
# mountable aprons) stack a clearance gap above - see its own docstring.
EXISTING_MARKING_Z_BASE = 0.06  # PAVEMENT_HEIGHT_M (0.05) + one MARKING_CLEARANCE_M (0.01) gap
EXISTING_MARKING_THICKNESS_M = 0.01

# Fallback only - the real value arrives per-render in the geometry JSON as
# `crosswalk_depth_m`, from src/render/crosswalks.py:CROSSWALK_DEPTH_FT (6 ft, Mercer
# County's recommended transverse crosswalk width). Kept in step with it so a JSON
# missing the field degrades to the same number rather than silently to a stale one.
CROSSWALK_DEPTH_FALLBACK_M = 1.829  # 6 ft


def _skewed_axes(u, n, skew_deg: float):
    """Rotate a leg's (along-travel, across-road) axes by `skew_deg` about z, and return
    them plus the factor its span must grow by to still reach both curbs.

    Real crosswalks line up with the curb ramps and sidewalks either side, which at a
    skewed junction is several degrees off square to the road centerline. `skew_deg`
    comes from the surveyed OSM crossing way via the geometry JSON
    (src/render/crosswalks.py:_crossing_skew_deg), so this render and the 2D plan view
    (src/render/plan_view.py:_crosswalk_band) orient the marking identically.
    """
    angle = math.radians(skew_deg)
    cos_s, sin_s = math.cos(angle), math.sin(angle)
    u_s = mathutils.Vector((u.x * cos_s - u.y * sin_s, u.x * sin_s + u.y * cos_s, 0.0))
    n_s = mathutils.Vector((n.x * cos_s - n.y * sin_s, n.x * sin_s + n.y * cos_s, 0.0))
    return u_s, n_s, 1.0 / max(cos_s, 0.2)


def _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m, stripe_width_m, gap_m,
                     n_stripes=None):
    """Parallel bars (rungs) running along travel (u), spaced across the crossing (n).
    Returns (center, span) so callers (ladder) can reuse the layout for framing rails.

    The bars run kerb to kerb. `width_m` already IS the kerb-to-kerb span, measured to the
    surveyor's traced kerbs (src/render/crosswalks.py:crosswalk_reach_to_curbs_ft), so the
    outermost bar's OUTER EDGE is placed at that span rather than its centre - a continental
    crossing whose end bar is half a bar short of the kerb reads as stopping short.

    There used to be a flat 1.5 m "keep clear of the curb edges" inset here on top of that,
    costing ~2.5 ft at each kerb. add_crosswalk_lines had the same fudge and it was removed
    when the crossings were made to reach the kerb; this copy was missed, so the simple
    crossings reached and the continental ones - which every proposal now uses - did not.
    """
    period = stripe_width_m + gap_m
    # n bars with n-1 gaps between them must fit inside width_m. Normally the count arrives
    # from the geometry JSON as `crosswalk_bar_count`, computed by
    # src/render/crosswalks.py:continental_bar_count; the local formula is the fallback for
    # a JSON without the field, kept identical to it.
    if n_stripes is None:
        n_stripes = max(int((width_m + gap_m) / period), 1)
    # The leftover is spread across the gaps rather than left at the ends, so the two end
    # bars land exactly ON the kerbs. A whole-period pitch leaves up to one period unpainted
    # - 3.2 ft at Columbia Ave, which still reads as a crossing stopping short. Real striping
    # adjusts the spacing to fit the road; the bars keep their width, only the gaps stretch.
    span = max(width_m - stripe_width_m, 0.0)   # centre-to-centre of the outermost pair
    pitch = span / (n_stripes - 1) if n_stripes > 1 else 0.0
    center = near + u * offset_m
    for i in range(n_stripes):
        lateral = -span / 2 + i * pitch
        add_stripe_rect(f"{name}_stripe_{i}", center + n * lateral, u, n, depth_m, stripe_width_m,
                         EXISTING_MARKING_THICKNESS_M, material, z_base=EXISTING_MARKING_Z_BASE)
    return center, span


def add_crosswalk_continental(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                               depth_m: float = CROSSWALK_DEPTH_FALLBACK_M, stripe_width_m: float = 0.5,
                               gap_m: float = 0.5, n_stripes=None):
    """Continental: parallel bars only, no framing rails."""
    _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m, stripe_width_m, gap_m,
                     n_stripes)


def add_crosswalk_ladder(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                          depth_m: float = CROSSWALK_DEPTH_FALLBACK_M, stripe_width_m: float = 0.5,
                          gap_m: float = 0.5, rail_width_m: float = 0.3, n_stripes=None):
    """Ladder: continental bars framed by two rails spanning the crossing width at
    each end of the depth - the rails are what distinguish it from bare continental."""
    center, span = _crosswalk_bars(name, near, u, n, width_m, material, offset_m, depth_m,
                                    stripe_width_m, gap_m, n_stripes)
    # span is now centre-to-centre of the end bars, so adding one bar width reaches their
    # outer edges - which is the kerb-to-kerb width. The rails end where the bars do.
    rail_length = span + stripe_width_m
    for side, sign in [("near", -1), ("far", 1)]:
        rail_center = center + u * (sign * depth_m / 2)
        add_stripe_rect(f"{name}_rail_{side}", rail_center, n, u, rail_length, rail_width_m,
                         EXISTING_MARKING_THICKNESS_M, material, z_base=EXISTING_MARKING_Z_BASE)


def add_crosswalk_lines(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0,
                         depth_m: float = CROSSWALK_DEPTH_FALLBACK_M, line_width_m: float = 0.3,
                         n_stripes=None):
    """Simple/standard marking: just two transverse lines bounding the crossing, no
    bars in between - the least visible of the three styles (FHWA/NACTO recommend
    upgrading this to continental or ladder for visibility, hence it being the
    'existing conditions' style here while proposed treatments upgrade it)."""
    # Full width: a transverse crossing's two boundary lines are painted kerb to kerb.
    # They were inset by 1.0 m (~3.3 ft) total, which at this scale reads as a floating
    # box that stops short of the kerb - not what is on the street, and not what OSM's
    # crossing ways show either.
    line_width = max(width_m, 0.5)
    center = near + u * offset_m
    for side, sign in [("near", -1), ("far", 1)]:
        line_center = center + u * (sign * depth_m / 2)
        add_stripe_rect(f"{name}_line_{side}", line_center, n, u, line_width, line_width_m,
                         EXISTING_MARKING_THICKNESS_M, material, z_base=EXISTING_MARKING_Z_BASE)


CROSSWALK_STYLES = {
    "lines": add_crosswalk_lines,
    "continental": add_crosswalk_continental,
    "ladder": add_crosswalk_ladder,
}


def add_crosswalk(name: str, near, u, n, width_m: float, material, offset_m: float = 3.0, style: str = "lines",
                   depth_m: float = CROSSWALK_DEPTH_FALLBACK_M, skew_deg: float = 0.0,
                   reach_left_m: float = None, reach_right_m: float = None, n_stripes=None):
    """`depth_m` is forwarded from the geometry JSON's `crosswalk_depth_m`, which
    src/render/export.py writes from src/render/crosswalks.py:CROSSWALK_DEPTH_M - the
    same constant src/render/plan_view.py draws the 2D crosswalk from, so the plan
    view and this render can't disagree about the crosswalk's size."""
    centre = near + u * offset_m
    u_s, n_s, span_factor = _skewed_axes(u, n, skew_deg)
    draw_fn = CROSSWALK_STYLES.get(style, add_crosswalk_lines)

    # A crosswalk runs kerb to kerb, and the kerbs are the surveyor's traced ones - neither
    # symmetric about the leg centerline nor at half the nominal width. `reach_*_m` carry
    # the real distance to each kerb (src/render/crosswalks.py:crosswalk_reach_to_curbs_ft);
    # re-centring on the midpoint of the two and spanning their sum is what makes the
    # painted crossing actually meet the kerb on both sides instead of stopping short.
    if reach_left_m is not None and reach_right_m is not None:
        centre = centre + n_s * ((reach_left_m - reach_right_m) / 2)
        span_m = reach_left_m + reach_right_m
    else:
        span_m = width_m * span_factor

    # offset_m=0 because `centre` already has the offset applied along the UNSKEWED
    # axis - rotating first and then stepping out would move the crosswalk along the
    # leg as well as turning it.
    draw_fn(name, centre, u_s, n_s, span_m, material, offset_m=0.0, depth_m=depth_m,
             n_stripes=n_stripes)


def add_stop_bar(name: str, near, u, n, width_m: float, material, offset_m: float, line_width_m: float = 0.5,
                  skew_deg: float = 0.0, curb_clearance_m: float = 0.5):
    """Stop bar: a single transverse line telling drivers where to stop for the
    signal, drawn just behind (intersection side of) the leg's crosswalk.
    Spans only the entering half of the road - `n` is the leg's own 'left'
    direction relative to its outward centerline direction (see
    src/render/props.py's left/right convention), which is the entering driver's
    right-hand side under US right-hand traffic (they travel the *opposite*
    way along the leg, so the sides swap) - a real stop bar never crosses
    into the opposing/receiving lanes, unlike a crosswalk line which spans
    the full width."""
    centre = near + u * offset_m
    u_s, n_s, span_factor = _skewed_axes(u, n, skew_deg)
    half_width = width_m * span_factor / 2
    # Clearance comes from src/render/crosswalks.py:STOP_BAR_CURB_CLEARANCE_M via the JSON,
    # so src/render/plan_view.py draws an identically-sized bar.
    lane_span = max(half_width - curb_clearance_m, curb_clearance_m)
    lane_center = centre + n_s * (half_width / 2)  # centered within the entering half only
    add_stripe_rect(f"{name}_bar", lane_center, n_s, u_s, lane_span, line_width_m, EXISTING_MARKING_THICKNESS_M,
                     material, z_base=EXISTING_MARKING_Z_BASE)


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


def add_paint_polyline(name: str, points: list, width_m: float, material,
                        height_m: float = 0.01, z_base: float = 0.06):
    """A painted line through every point in sequence (e.g. a sampled curve
    like src/geometry/model.py:lane_narrowing_taper_ft's arc) - NOT
    add_paint_line(name, points[0], points[-1], ...), which would draw a
    single straight chord between the endpoints and silently discard every
    point in between, turning an actual curve into a straight diagonal
    segment (confirmed: that's exactly what was happening before this
    function existed)."""
    for i in range(len(points) - 1):
        add_paint_line(f"{name}_{i}", points[i], points[i + 1], width_m, material, height_m=height_m, z_base=z_base)


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
        add_stripe_rect(f"{name}_dash_{i}", center, u, n, dash_m, width_m, EXISTING_MARKING_THICKNESS_M, material,
                         z_base=EXISTING_MARKING_Z_BASE)
        pos += dash_m + gap_m
        i += 1


def add_double_yellow_centerline(name: str, near: mathutils.Vector, far: mathutils.Vector, material,
                                  start_m: float = 6.0, width_m: float = 0.15, line_gap_m: float = 0.1):
    """Solid double yellow: a no-passing-zone centerline - two continuous
    (not dashed) parallel lines, real MUTCD/AASHTO proportions (~6 in line
    width, ~4 in gap between them), same start_m setback past the
    intersection curve as the dashed style."""
    direction = far - near
    length = direction.length
    if length <= start_m:
        return
    u = direction / length
    n = mathutils.Vector((-u.y, u.x, 0))
    run = length - start_m
    center = near + u * (start_m + run / 2)
    lateral = line_gap_m / 2 + width_m / 2
    for side, sign in [("a", -1), ("b", 1)]:
        add_stripe_rect(f"{name}_{side}", center + n * (sign * lateral), u, n, run, width_m,
                         EXISTING_MARKING_THICKNESS_M, material, z_base=EXISTING_MARKING_Z_BASE)
