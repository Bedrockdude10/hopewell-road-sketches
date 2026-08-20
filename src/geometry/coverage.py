"""Does the drawing contain every surveyed feature inside its own frame?

The property, from docs/network-renderer-plan.md:

    Every feature the surveyor recorded inside the drawn frame is either drawn from its own
    traced geometry, or named in the notes as deliberately not drawn. Nothing is silently
    dropped.

src/checks.py cannot ask this: every invariant there asks whether the geometry we DID build is
right, and a feature that was never built has no geometry to be wrong about. Geometry is
state-plane feet throughout.

THE COMPARISON IS GEOMETRIC - "is there a drawn footprint lying along this surveyed way?" - and
must not become "which surveyed crossings did the render match to a leg". That shortcut re-derives
the very leg-gating that drops features, so it would report the same number forever and stay red
after a fix.

WHAT THIS CANNOT SEE. A layer with nothing surveyed in it reports no gap, so an empty fetch reads
as a faithful drawing. That hole is closed upstream rather than here:
osm_context.snapshot_for_site refuses a site whose window reaches outside the downloaded area, and
fetch_kerbs raises OSMDataUnavailableError rather than returning [] on an outage.
"""
import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from src.render.coords import FT_TO_M
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# How much of a surveyed way's own length has to lie under drawn paint before the drawing counts
# as containing it. Measured: a drawn crossing has 62-91% of its traced length inside its band (the
# shortfall is the ends, which run to the SIDEWALK centreline while the band stops at the kerb), and
# a dropped one measures exactly 0.00%.
#
# Not a hair above zero, because the union below is every drawn area rather than only crossing
# footprints, and a kerbside hatched zone genuinely overlaps a crossing running into it by 6-20%.
# Excluding hatch by type is not an option: a crossing arriving as a PaintPiece must still count.
CROSSING_DRAWN_FRACTION = 0.25

# How far a tactile paving pad may stand from the kerb ramp it marks and still be that ramp's pad.
# A drawn pad sits 1.5-2.7 ft from its own traced way; the nearest pad to an omitted ramp is 232 ft
# away, at another junction. Generous, because nothing lives in between.
RAMP_PAD_NEAR_FT = 15.0

# How far a control node may be from the hardware drawn for it. Big on purpose, and the size is a
# fact about the source rather than slack: OSM puts a control node ON the carriageway and the render
# draws the hardware where it physically stands - a stop sign on the kerb beside its node (15-18 ft),
# signal poles at the corners of the junction node (31-43 ft). The prop TYPE is matched too, so this
# only has to separate a node from ANOTHER JUNCTION's hardware of the same kind, 250 ft away.
CONTROL_NEAR_NODE_FT = 60.0

# Which OSM control nodes are drawn as something. highway=crossing is deliberately not one of
# them: it is a node on a crossing WAY carrying the pedestrian detail that lives on the node
# rather than the way (tactile_paving, button_operated, crossing:island - see
# osm_context.fetch_traffic_control), so it is an attribute of a feature the crossings layer already
# counts, not a second feature. Counting it would report the same crossings twice.
CONTROL_KINDS = ("stop", "give_way", "traffic_signals")

# The hardware a signalized junction is drawn with (src/render/props.py:_traffic_signal_props).
SIGNAL_HARDWARE = ("traffic_signal_pole", "pedestrian_signal_head")

# How many features a gap lists. Enough to see the pattern without turning a build log into a data
# dump; the count carries the rest.
MAX_EXAMPLES = 5
# How a drawn LINE is judged to be one of a crossing's own transverse lines rather than a lane line
# that happens to cross it - see _Drawing.has_line_along. The angle does the real work: a crosswalk's
# lines are offset curves of the crossing way and so parallel to it, while a lane edge line meets it
# at right angles. The length fraction rejects a short stub near the crossing, and the distance keeps
# a parallel line one street over from counting.
ALONG_ANGLE_TOLERANCE_DEG = 20.0
ALONG_TOLERANCE_FT = 8.0
MIN_ALONG_LENGTH_FRACTION = 0.5


def _direction(line: LineString) -> float:
    """A line's overall bearing in degrees, folded to 0-180 so direction of travel does not matter.

    From the CHORD rather than the first segment: a crossing way has 2-5 vertices and its opening
    segment can point somewhere the way as a whole does not.
    """
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0


@dataclass(frozen=True)
class Uncovered:
    """One layer's worth of surveyed features the drawing does not contain.

    `total` beside `count`: the ratio says whether a layer is broken or merely incomplete,
    and it lets a test assert gap == total - drawn so the assertion survives the fix.
    """
    layer: str
    count: int
    total: int
    examples: list[str]

    @property
    def drawn(self) -> int:
        return self.total - self.count

    def __str__(self) -> str:
        return f"{self.layer}: {self.count} of {self.total} surveyed in the frame are not drawn"


def coverage_gaps(model: "IntersectionModel", paint, frame_radius_ft: float | None = None) -> list[Uncovered]:
    """Every layer where the drawing omits a surveyed feature inside its own frame.

    `paint` is EVERYTHING THE DRAWING PUTS ON THE GROUND - PaintPieces, bare shapely footprints,
    prop dicts, any nesting of those. A check that read only SceneGeometry.build_paint() would
    report every crossing as dropped (bands travel in their own dict), and a guard that cries
    wolf on a working layer gets switched off. So a caller hands in the whole drawing::

        paint, props = scene.build_paint_and_posts(props)
        coverage_gaps(model, [*paint, *scene.crosswalk_bands.values(), *props])

    `frame_radius_ft` defaults to the frame both views draw (src/render/frame.py:junction_frame).
    Only layers WITH a gap come back; an empty list means the drawing is faithful.
    """
    # The same guard kerbs.py:kerb_openings_from_model uses, for the same reason: a model can be a
    # stand-in that carries legs and config and no OSM at all, and every geometry test in this
    # repo has to survive one of those rather than raising on it.
    if not all(hasattr(model, attr) for attr in ("center_wgs84", "center_ft", "legs")):
        return []
    radius_ft = _frame_radius_ft(model, frame_radius_ft)
    drawing = _read_drawing(paint)
    gaps = [layer(model, drawing, radius_ft) for layer in LAYERS]
    return [gap for gap in gaps if gap is not None]


def describe_coverage(gaps: list[Uncovered]) -> str:
    """One block for a build log, saying either what was dropped or that nothing was.

    NEVER THE EMPTY STRING. A check whose clean state prints nothing is indistinguishable from
    a check that did not run - the trap the OSM cache staleness fell into before cache_summary
    printed an age on every build.
    """
    if not gaps:
        return ("COVERAGE: nothing dropped - every surveyed feature inside the drawn frame is in "
                "the drawing.")
    lines = ["COVERAGE GAPS: surveyed features inside the drawn frame that the drawing omits.",
             "  A render that leaves one out states something false about the street - see "
             "src/geometry/coverage.py."]
    for gap in gaps:
        lines.append(f"  {gap}:")
        lines += [f"    - {example}" for example in gap.examples]
        remaining = gap.count - len(gap.examples)
        if remaining > 0:
            lines.append(f"    - ...and {remaining} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading the drawing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Drawing:
    """What was handed in, reduced to the two questions a layer asks of it."""
    ground: object          # union of every drawn footprint that covers ground, or None
    props: tuple            # (type, position Point) per drawn prop
    lines: tuple = ()       # every drawn LineString, for has_line_along only

    def props_of(self, types) -> list:
        return [point for kind, point in self.props if kind in types]

    def fraction_covered(self, line: LineString) -> float:
        """How much of `line`'s length lies under drawn ground, 0.0-1.0."""
        if self.ground is None or line.length <= 0:
            return 0.0
        return line.intersection(self.ground).length / line.length

    def has_line_along(self, line: LineString) -> bool:
        """Whether a drawn LINE runs along `line` - a crosswalk's own transverse lines.

        The narrow exception to keeping lines out of `ground`: a lines-style crossing is two thin
        lines with no area, so an area-only union reports every one as dropped. ALONG, not merely
        near: a lane edge line runs the length of the leg and crosses every crossing way at right
        angles, so the direction test does the work the area test cannot.
        """
        if line.length <= 0:
            return False
        want = _direction(line)
        for drawn in self.lines:
            if drawn.length < line.length * MIN_ALONG_LENGTH_FRACTION:
                continue
            if drawn.distance(line) > ALONG_TOLERANCE_FT:
                continue
            if abs(_direction(drawn) - want) <= ALONG_ANGLE_TOLERANCE_DEG:
                return True
        return False


def _read_drawing(paint) -> _Drawing:
    """Normalise whatever the caller called the drawing into one _Drawing.

    Structural rather than typed: the three renderers' notion of "a drawn thing" is already three
    things (PaintPiece, bare band Polygon, prop dict), and a check that accepted only today's
    types would report a new one as missing.
    """
    shapes, props = [], []

    def walk(item):
        if item is None:
            return
        position = item.get("position_ft") if isinstance(item, dict) else None
        if position is not None:
            props.append((item.get("type"), Point(*position)))
            return
        geometry = getattr(item, "geometry", None)
        if geometry is not None:
            shapes.append(geometry)
            return
        if hasattr(item, "geom_type"):
            shapes.append(item)
            return
        if hasattr(item, "values"):
            for value in item.values():
                walk(value)
            return
        if isinstance(item, (str, bytes)):
            return
        try:
            children = iter(item)
        except TypeError:
            return
        for child in children:
            walk(child)

    walk(paint)
    # Only what covers GROUND. A lane edge line runs the length of the leg and crosses every
    # crossing way at right angles, so buffered by any tolerance it would "cover" a fifth of each.
    areas = [shape for shape in shapes if not shape.is_empty and shape.area > 0]
    drawn_lines = [shape for shape in shapes
                   if not shape.is_empty and shape.area == 0 and shape.geom_type == "LineString"]
    return _Drawing(ground=unary_union(areas) if areas else None, props=tuple(props),
                    lines=tuple(drawn_lines))


def _frame_radius_ft(model: "IntersectionModel", given: float | None) -> float:
    """The radius of the frame both views draw, or the caller's own.

    Local import: src.render.frame reaches back into src.geometry.model, and this module is
    imported from src.geometry.
    """
    if given is not None:
        return float(given)
    from src.render.frame import junction_frame

    return junction_frame(model).radius_ft


def _in_frame(model: "IntersectionModel", radius_ft: float):
    """A test for "this is in the picture", and the distance to report it at.

    Two different centres: membership from the FRAME's centre (what is drawn), distance from
    the JUNCTION (what a reader means by "263 ft out"). Against the radius rather than the
    square, so a feature reported as dropped cannot be dismissed as one the 3D camera never
    framed.
    """
    from src.render.frame import junction_frame

    centre = junction_frame(model).center_ft

    def inside(geometry) -> bool:
        return geometry.distance(centre) <= radius_ft

    def out_ft(geometry) -> float:
        return geometry.distance(model.center_ft)

    return inside, out_ft


def _line_ft(coords_wgs84) -> LineString:
    """One fetched OSM way in state-plane feet, through the projection everything else uses."""
    from src.geometry.intersection import to_state_plane

    return LineString(to_state_plane(coords_wgs84))


def _uncovered(layer: str, features: list[tuple]) -> Uncovered | None:
    """`features` is [(is_drawn, distance_ft, description)] - the shape every layer ends in.

    Nearest first: the nearest omission is the conspicuous one.
    """
    if not features:
        return None
    missing = sorted((f for f in features if not f[0]), key=lambda f: f[1])
    if not missing:
        return None
    return Uncovered(layer=layer, count=len(missing), total=len(features),
                     examples=[f"{description} ({distance_ft:.0f} ft out)"
                               for _drawn, distance_ft, description in missing[:MAX_EXAMPLES]])


# ---------------------------------------------------------------------------
# The layers
# ---------------------------------------------------------------------------

def crossing_gaps(model: "IntersectionModel", drawing: _Drawing, radius_ft: float) -> Uncovered | None:
    """Surveyed pedestrian crossings the drawing does not draw.

    All crossings are traced WAYS carrying their own position, length and skew, so each one
    can be drawn without a leg. THE MARKINGS TAG IS REPORTED UNDER THE KEY THAT ACTUALLY
    CARRIES IT - `crossing:markings=zebra` vs `crossing=zebra` are not collapsed, because the
    difference is the whole question for a way with no tag at all.

    TWO KINDS OF NOT-DRAWN, and only one is this module's business:

      * the renderer HAD markings and did not draw them - dropped ground truth.
      * the SURVEY records no markings, so there is nothing to draw. Counting it as a gap makes
        this check permanently red for a reason no code change can fix.

    A crossing with no drawable markings is excluded from the count and named in `unrecorded`
    instead - visible, but not an accusation against the drawing. That is the "named rather than
    silently dropped" half of docs/network-renderer-plan.md's property.

    COVERAGE COUNTS PAINT OF EITHER SHAPE. Bars have area; a lines-style crossing does not.
    Lines are kept out of the general union (a lane edge line crosses every crossing way at right
    angles, and buffering it would let it "cover" a fifth of each); what counts here is a drawn
    line that runs ALONG the crossing rather than across it.
    """
    from src.sources.osm_context import fetch_crossings

    inside, out_ft = _in_frame(model, radius_ft)
    features, unrecorded = [], []
    for crossing in fetch_crossings(model.center_wgs84, radius_m=radius_ft * FT_TO_M):
        line = _line_ft(crossing["coords_wgs84"])
        if not inside(line):
            continue
        label = f"crossing way {line.length:.0f} ft long, {_markings_label(crossing.get('tags'))}"
        if not _has_drawable_markings(crossing.get("tags")):
            unrecorded.append((out_ft(line), label))
            continue
        drawn = (drawing.fraction_covered(line) >= CROSSING_DRAWN_FRACTION
                 or drawing.has_line_along(line))
        features.append((drawn, out_ft(line), label))
    gap = _uncovered("crossings", features)
    if unrecorded:
        print(f"  NOTE: {len(unrecorded)} surveyed crossing(s) in frame record no markings, so "
              f"nothing is drawn on them: "
              + "; ".join(f"{label} at {out:.0f} ft" for out, label in sorted(unrecorded)[:MAX_EXAMPLES]))
    return gap


def _has_drawable_markings(tags: dict | None) -> bool:
    """Whether the survey records paint this project knows how to draw.

    Delegates to src/geometry/surveyed.py: one decision, one place. The two must agree exactly -
    a crossing the renderer declines to draw and this check counts as dropped makes the build
    permanently red.
    """
    from src.geometry.surveyed import drawable_markings

    return drawable_markings(tags) is not None


def _markings_label(tags: dict | None) -> str:
    """What OSM says this crossing is painted with, under the key that says it.

    `crossing:markings` is current; `crossing` is the older tag carrying the same values.
    "no markings tag" is a different statement from "unmarked" - only one is true of a way
    nobody has recorded.
    """
    tags = tags or {}
    for key in ("crossing:markings", "crossing"):
        if tags.get(key):
            return f"{key}={tags[key]}"
    return "no markings tag"


def kerb_gaps(model: "IntersectionModel", drawing: _Drawing, radius_ft: float) -> Uncovered | None:
    """Traced kerb ways in the frame that the render does not draw. THE CONTROL CASE.

    Expected clean: both renderers take the same one number (drawn_kerb_radius_ft), so the
    only question is whether the drawing radius covers the frame radius.

    Compared BY WAY ID rather than geometrically, because both sides of this comparison are the
    same fetch through the same projection - a geometric test would add a tolerance to a
    comparison that has an exact answer.
    """
    from src.geometry.intersection import drawn_kerb_radius_ft, kerb_lines_with_tags_ft

    inside, out_ft = _in_frame(model, radius_ft)
    drawn_ids = {way_id for _line, _tags, way_id
                 in kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                            radius_ft=drawn_kerb_radius_ft())}
    features = [(way_id in drawn_ids, out_ft(line),
                 f"traced kerb way {way_id}, {line.length:.0f} ft long, kerb={(tags or {}).get('kerb', 'untagged')}")
                for line, tags, way_id in kerb_lines_with_tags_ft(model.center_wgs84,
                                                                  model.center_ft,
                                                                  radius_ft=radius_ft)
                if inside(line)]
    return _uncovered("kerbs", features)


def kerb_ramp_gaps(model: "IntersectionModel", drawing: _Drawing, radius_ft: float) -> Uncovered | None:
    """Surveyed kerb ramps - kerb ways tagged tactile_paving=yes - with no pad drawn on them.

    The traced KERB WAY is the ramp's geometry (OSM records a ramp as `barrier=kerb` tagged
    `kerb=lowered` + `tactile_paving=yes` + `wheelchair=yes`), so this reads the kerb layer
    rather than looking for a fetcher of its own. A crossing way's own `tactile_paving=yes` is
    NOT counted: it says a ramp exists at the crossing's ends without saying where, so the
    crossings layer already accounts for the crossing.
    """
    from src.geometry.intersection import kerb_lines_with_tags_ft

    inside, out_ft = _in_frame(model, radius_ft)
    pads = drawing.props_of(("tactile_paving_pad",))
    features = []
    for line, tags, way_id in kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                                      radius_ft=radius_ft):
        if (tags or {}).get("tactile_paving") != "yes" or not inside(line):
            continue
        near_ft = min((line.distance(pad) for pad in pads), default=float("inf"))
        features.append((near_ft <= RAMP_PAD_NEAR_FT, out_ft(line),
                         f"kerb ramp way {way_id} tagged tactile_paving=yes, nearest drawn pad "
                         + (f"{near_ft:.0f} ft away" if pads else "- no pad is drawn at all")))
    return _uncovered("kerb_ramps", features)


def traffic_control_gaps(model: "IntersectionModel", drawing: _Drawing, radius_ft: float) -> Uncovered | None:
    """Surveyed stop / give-way / signal nodes with no hardware drawn for them.

    Matched on the prop TYPE as well as the distance, so a node is only covered by hardware of
    its own kind - which is what makes CONTROL_NEAR_NODE_FT's 60 ft safe. Coarse at junction
    level: several nodes of one kind at one junction are covered by any of that junction's signs
    rather than each by its own, because a node on the carriageway carries nothing that says
    which kerb its sign stands on (see _osm_control_props), and overstating the pairing would
    be worse than reporting the junction.
    """
    from src.render.props import control_nodes_ft  # local: props imports geometry, avoid a cycle
    from src.sources.osm_context import fetch_traffic_control

    inside, out_ft = _in_frame(model, radius_ft)
    features = []
    nodes = fetch_traffic_control(model.center_wgs84, radius_m=radius_ft * FT_TO_M)
    for node in control_nodes_ft(nodes):
        kind = (node.get("tags") or {}).get("highway")
        if kind not in CONTROL_KINDS:
            continue
        point = node["point_ft"]
        if not inside(point):
            continue
        hardware = _hardware_for(kind)
        near_ft = min((point.distance(prop) for prop in drawing.props_of(hardware)),
                      default=float("inf"))
        features.append((near_ft <= CONTROL_NEAR_NODE_FT, out_ft(point),
                         f"highway={kind} node, nothing of {'/'.join(hardware)} drawn within "
                         f"{CONTROL_NEAR_NODE_FT:.0f} ft"))
    return _uncovered("traffic_control", features)


def _hardware_for(kind: str) -> tuple:
    """The prop types the render draws for one kind of control node.

    Read off OSM_CONTROL_TO_PROP rather than restated. Signals are the exception: drawn from the
    site config's `signals` block, the OSM node only attesting that the junction is signalized.
    """
    from src.render.props import OSM_CONTROL_TO_PROP

    if kind == "traffic_signals":
        return SIGNAL_HARDWARE
    drawn = OSM_CONTROL_TO_PROP.get(kind)
    return (drawn,) if drawn else ()


# Every layer, in report order. A registry rather than a chain of calls inside coverage_gaps:
# a layer that is written and never called is dead code that looks live, and this module's job
# is noticing that something is missing.
LAYERS = (crossing_gaps, kerb_gaps, kerb_ramp_gaps, traffic_control_gaps)
