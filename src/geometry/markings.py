"""Every kind of marking this project paints, and every channel a renderer reads, declared once.

The declarations live here and their relationships are checked at import, not by a test that has
to remember to look:

  * a marking's ROLE says what it is - a line, a hatched area, a drivable surface, an object -
    and every renderer decides how to draw it from that instead of from its name;
  * a marking names the CHANNEL it travels to the 3D render in, and its role has to match that
    channel's role, so a hatched zone cannot be routed to a list of lines;
  * every marking has a channel unless it is an OBJECT, which reaches the render as a prop
    (see src/render/props.py) - the one thing paint cannot carry across that boundary.

Channels are the keys in the exported JSON that scripts/blender/blender_scene.py reads by name.
Blender runs under its own bundled Python and cannot import this package, so that boundary is
data, not polymorphism: CHANNELS is the single declaration both sides are written against, and
their order here is the order they appear in the file.
"""
from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """What a marking IS, which is what decides how each renderer draws it.

    LINE     a painted stripe: a polyline in 3D, a line in the plan view.
    FILL     a hatched area: diagonal strokes in 3D (the paint that is actually applied), a
             hatch pattern with an outline in the plan view.
    SURFACE  ground that is built rather than painted - a mountable apron. Extruded in 3D.
    COLOUR   carriageway painted a solid colour rather than striped: a green bike lane. Distinct
             from FILL because it has no strokes - it travels as its polygon. Distinct from
             SURFACE because a SURFACE is built ground that every marking is cut around
             (PaintContext.seal_surfaces), and colouring a bike lane must not cut the lane's own
             edge lines out of existence.
    OBJECT   a physical thing standing on the road - a flex post. The 3D render builds objects
             only from props, so an OBJECT marking is the plan view's copy of something that
             must ALSO exist as a prop; check_bollards_are_props enforces exactly that.
    """
    LINE = "line"
    FILL = "fill"
    SURFACE = "surface"
    COLOUR = "colour"
    OBJECT = "object"


# The painted width of a stroke, in METRES, because that is the unit the 3D renderer lays it in
# and metres keep this module a leaf (it imports nothing from src, so the FT_TO_M in
# src/render/coords.py is not reachable from here). Two figures, both drawn-scale choices rather
# than standards: a solid edge line reads at 0.25 m here, a hatch stroke and a taper at 0.15 m.
#
# DECLARED HERE BECAUSE A LINE WITH NO WIDTH CANNOT BE CHECKED. These lived only in
# scripts/blender/blender_scene.py, which meant the 3D render was the sole thing in the project
# that knew paint is not infinitely thin - so an axis put half a stripe wrong was invisible to
# the plan view (a cosmetic 1.6 pt stroke about its axis) AND to checks.MarkingsDoNotCollide,
# which compared only things that cover area. The crossbike's edge lines were ruled along the
# green's own faces for exactly that long. Blender still declares its own table because it
# cannot import this package; test_blender_stroke_widths_match_the_channels pins the two.
EDGE_LINE_WIDTH_M = 0.25
NARROW_LINE_WIDTH_M = 0.15


@dataclass(frozen=True)
class Channel:
    """One list in the exported geometry JSON, read by name in scripts/blender/blender_scene.py.

    A channel carries one role, so everything in it is drawn the same way at the far end: a
    LINE channel becomes add_paint_polyline calls, a FILL channel becomes the hatch strokes
    inside the zone, a SURFACE channel becomes an extruded polygon.

    `stroke_width_m` is how wide the paint in it is actually laid. Set on every channel that is
    STROKED - the LINE channels, and the FILL channels whose hatch strokes are lines too - and
    None on the ones that travel as polygons (COLOUR, SURFACE), where the geometry already has
    its own extent. See EDGE_LINE_WIDTH_M for why it lives here.
    """
    key: str
    role: Role
    stroke_width_m: float | None = None

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True)
class PaintKind:
    """One kind of marking. Compared by identity, printed as its name.

    `channel` is None only for an OBJECT, which cannot travel as paint - see the module
    docstring. Everything else about how this marking is drawn is derived from `role`.
    """
    name: str
    role: Role
    channel: Channel | None = None

    def __str__(self) -> str:
        return self.name

    @property
    def is_line(self) -> bool:
        return self.role is Role.LINE

    @property
    def is_fill(self) -> bool:
        """Hatched paint. NOT the same question as "is this a polygon" - a bollard is stored as
        a degenerate polygon standing in for a point."""
        return self.role is Role.FILL

    @property
    def covers_area(self) -> bool:
        """Occupies ground rather than tracing a line: a hatched zone, a built surface, or a
        coloured stretch of carriageway. What MarkingsDoNotCollide compares."""
        return self.role in (Role.FILL, Role.SURFACE, Role.COLOUR)

    @property
    def is_object(self) -> bool:
        return self.role is Role.OBJECT


# --------------------------------------------------------------------------------------
# The channels, in the order they appear in the exported JSON.
# --------------------------------------------------------------------------------------
LANE_NARROWING_EDGE_LINES = Channel("lane_narrowing_edge_lines", Role.LINE, EDGE_LINE_WIDTH_M)
LANE_NARROWING_TAPER_LINES = Channel("lane_narrowing_taper_lines", Role.LINE, NARROW_LINE_WIDTH_M)
LANE_NARROWING_HATCH_LINES = Channel("lane_narrowing_hatch_lines", Role.FILL, NARROW_LINE_WIDTH_M)
CORNER_HATCHING_LINES = Channel("corner_hatching_lines", Role.FILL, NARROW_LINE_WIDTH_M)
PARKING_EDGE_LINES = Channel("parking_edge_lines", Role.LINE, EDGE_LINE_WIDTH_M)
PARKING_STALL_DIVIDER_LINES = Channel("parking_stall_divider_lines", Role.LINE, NARROW_LINE_WIDTH_M)
# The daylight zones (R.S. 39:4-138 - see src/geometry/daylighting.py) share the parking
# buffer's channels, because on a real street they are the same white hatching and the same
# white lines. The plan view distinguishes them by colour; asphalt does not.
PARKING_BUFFER_HATCH_LINES = Channel("parking_buffer_hatch_lines", Role.FILL, NARROW_LINE_WIDTH_M)
PARKING_BUFFER_EDGE_LINES = Channel("parking_buffer_edge_lines", Role.LINE, EDGE_LINE_WIDTH_M)
# Empty today, and deliberately kept: a curved line needs Blender's add_paint_polyline rather
# than add_paint_line, so tapers travel in their own channel. blender_scene.py reads the key.
PARKING_BUFFER_TAPER_LINES = Channel("parking_buffer_taper_lines", Role.LINE, NARROW_LINE_WIDTH_M)
BIKE_LANE_EDGE_LINES = Channel("bike_lane_edge_lines", Role.LINE, EDGE_LINE_WIDTH_M)
BIKE_LANE_HATCH_LINES = Channel("bike_lane_hatch_lines", Role.FILL, NARROW_LINE_WIDTH_M)
# The lane's own asphalt, painted green. Travels to the render as the polygon it is, so both
# views agree about what the proposal looks like.
BIKE_LANE_SURFACE_POLYGONS = Channel("bike_lane_surface_polygons", Role.COLOUR)
# The YELLOW centre stripe of a two-way bike lane, separating opposing riders. Its own channel
# rather than more BIKE_LANE_EDGE_LINES, because the channel decides the colour at the far end:
# blender_scene.py draws every edge-line channel in the white marking material.
BIKE_LANE_CONTRAFLOW_LINES = Channel("bike_lane_contraflow_lines", Role.LINE, NARROW_LINE_WIDTH_M)
# The BIKE LANE symbol (MUTCD Fig 9E-1). A COLOUR channel and not a LINE one, because a symbol is
# a painted AREA - it reaches the render as its footprint, reusing the coloured-polygon path.
# The footprint is a schematic arrow, not a drawn bicycle: this pipeline positions paint, it does
# not draw glyph art. The legend says so, so the drawing does not overclaim.
BIKE_LANE_SYMBOL_POLYGONS = Channel("bike_lane_symbol_polygons", Role.COLOUR)
CORNER_APRON_POLYGONS = Channel("corner_apron_polygons", Role.SURFACE)

CHANNELS: tuple[Channel, ...] = (
    LANE_NARROWING_EDGE_LINES, LANE_NARROWING_TAPER_LINES, LANE_NARROWING_HATCH_LINES,
    CORNER_HATCHING_LINES, PARKING_EDGE_LINES, PARKING_STALL_DIVIDER_LINES,
    PARKING_BUFFER_HATCH_LINES, PARKING_BUFFER_EDGE_LINES, PARKING_BUFFER_TAPER_LINES,
    BIKE_LANE_EDGE_LINES, BIKE_LANE_HATCH_LINES, BIKE_LANE_SURFACE_POLYGONS,
    BIKE_LANE_CONTRAFLOW_LINES, BIKE_LANE_SYMBOL_POLYGONS, CORNER_APRON_POLYGONS,
)


# --------------------------------------------------------------------------------------
# The markings. One line each; everything downstream is derived from it.
# --------------------------------------------------------------------------------------
_REGISTRY: dict[str, PaintKind] = {}


def _kind(name: str, role: Role, channel: Channel | None = None) -> PaintKind:
    """Declare a marking, checking the one relationship that used to be checked by hand."""
    if name in _REGISTRY:
        raise ValueError(f"paint kind {name!r} is declared twice")
    if role is Role.OBJECT:
        if channel is not None:
            raise ValueError(
                f"{name}: an object cannot travel to the 3D render as paint - the render "
                f"builds objects from props only (src/render/props.py). Leave its channel unset.")
    elif channel is None:
        raise ValueError(
            f"{name}: no channel, so nothing carries it to the 3D render and it would be "
            f"drawn in the plan view only. Give it one of markings.CHANNELS.")
    elif channel not in CHANNELS:
        raise ValueError(
            f"{name}: {channel.key} is not in markings.CHANNELS, so the export never writes it "
            f"and blender_scene.py never reads it. Add the channel there first.")
    elif channel.role is not role:
        raise ValueError(
            f"{name} is a {role.value} but {channel.key} carries {channel.role.value}s - "
            f"the render would draw it with the wrong builder.")
    kind = PaintKind(name, role, channel)
    _REGISTRY[name] = kind
    return kind


# Paint-only lane narrowing: an edge line, its hatched buffer, and the taper back to the kerb.
LANE_EDGE_LINE = _kind("lane_edge_line", Role.LINE, LANE_NARROWING_EDGE_LINES)
TAPER_LINE = _kind("taper_line", Role.LINE, LANE_NARROWING_TAPER_LINES)
LANE_NARROWING_FILL = _kind("lane_narrowing_fill", Role.FILL, LANE_NARROWING_HATCH_LINES)
TAPER_FILL = _kind("taper_fill", Role.FILL, LANE_NARROWING_HATCH_LINES)
# Diagonal hatching in a corner's gutter zone.
CORNER_HATCH_FILL = _kind("corner_hatch_fill", Role.FILL, CORNER_HATCHING_LINES)
# Marked curbside parking: the lane's edge line and its stall ticks.
PARKING_EDGE_LINE = _kind("parking_edge_line", Role.LINE, PARKING_EDGE_LINES)
STALL_DIVIDER = _kind("stall_divider", Role.LINE, PARKING_STALL_DIVIDER_LINES)
# The hatched strip between a kerbside zone and the kerb, and the lines that bound it.
BUFFER_FILL = _kind("buffer_fill", Role.FILL, PARKING_BUFFER_HATCH_LINES)
BUFFER_EDGE_LINE = _kind("buffer_edge_line", Role.LINE, PARKING_BUFFER_EDGE_LINES)
# The statutory no-parking zone at a corner (R.S. 39:4-138).
DAYLIGHT_FILL = _kind("daylight_fill", Role.FILL, PARKING_BUFFER_HATCH_LINES)
DAYLIGHT_EDGE_LINE = _kind("daylight_edge_line", Role.LINE, PARKING_BUFFER_EDGE_LINES)
# The square end of a zone with no crossing to be cut by and no room to taper.
ZONE_END_LINE = _kind("zone_end_line", Role.LINE, PARKING_BUFFER_EDGE_LINES)
# An exclusive bike lane: its own edge lines, and the hatched buffer beside it.
BIKE_LANE_EDGE_LINE = _kind("bike_lane_edge_line", Role.LINE, BIKE_LANE_EDGE_LINES)
# The same line carried across a driveway as a dotted extension - the lane does not end at an
# entrance, it is crossed there. Its own kind so a check or a reader can tell a continuous stripe
# from a broken one; same channel, since a dash is a short stripe. The dashes are in the GEOMETRY
# (see paint.py:_dashes_along), not in a line style, so the two renderers cannot disagree about
# where the gaps fall.
BIKE_LANE_DOTTED_EXTENSION = _kind("bike_lane_dotted_extension", Role.LINE, BIKE_LANE_EDGE_LINES)
BIKE_BUFFER_FILL = _kind("bike_buffer_fill", Role.FILL, BIKE_LANE_HATCH_LINES)
# The green a bike lane's asphalt is painted, between its two edge stripes - the lane itself
# rather than anything beside it.
BIKE_LANE_SURFACE = _kind("bike_lane_surface", Role.COLOUR, BIKE_LANE_SURFACE_POLYGONS)
# The centre stripe of a TWO-WAY bike lane. Yellow and broken, following MUTCD's rule for a
# two-way bikeway: yellow because it divides opposing traffic (the same meaning it carries on
# the roadway), broken because passing is permitted where sight distance allows.
BIKE_CONTRAFLOW_DIVIDER = _kind("bike_contraflow_divider", Role.LINE, BIKE_LANE_CONTRAFLOW_LINES)
#: The BIKE LANE symbol. NACTO asks for it after every driveway and intersection and at least
#: every 500 ft along a bidirectional lane; MUTCD Fig 9E-1 is the marking. See
#: src/geometry/treatments/bikeways.py:bike_symbol_stations_ft for the placement rule.
BIKE_LANE_SYMBOL = _kind("bike_lane_symbol", Role.COLOUR, BIKE_LANE_SYMBOL_POLYGONS)
# Built ground rather than paint: a flush, drivable corner surface.
APRON = _kind("apron", Role.SURFACE, CORNER_APRON_POLYGONS)
# A flex-post delineator. Paint draws the plan view's marker; the render needs a prop.
BOLLARD = _kind("bollard", Role.OBJECT)

KINDS: dict[str, PaintKind] = dict(_REGISTRY)

# --------------------------------------------------------------------------------------
# WHAT EVERY MARKING DOES WHERE A VEHICLE CROSSES THE KERB.
#
# One table, because it is one question. A driveway, a parking aisle, a side street and the
# junction this drawing is centred on are all the same event - src/geometry/kerbs.py:KerbOpening -
# and every marking has exactly one answer for each of the two KINDS of event that MUTCD
# distinguishes. Which kind a given gap is, is kerbs.OpeningSource.is_an_intersection's answer and
# nothing else's; what the marking then does is this table's and nothing else's.
# --------------------------------------------------------------------------------------

class AtAnOpening(Enum):
    """What one marking does where a vehicle crosses the kerb it runs beside.

    CARRIED   straight past, unbroken. The marking's meaning does not stop being true because
              somebody can turn across it - MUTCD 11th ed. 3B.11(09) for an edge line at a
              driveway.
    DOTTED    stops at the mouth and continues across it as a dotted extension. The lane still
              runs here, and here is where it is crossed - MUTCD 3B.11(05) and 9C.04 for a
              bicycle lane. `dotted_as` names the kind the dashes are laid in, because a broken
              line is a different instruction from the solid one it continues.
    FILLETED  the marking is part of a hatched zone, and the zone sweeps away from the mouth on
              its run-out rather than stopping square. Only ever produces a fillet at a DRIVEWAY,
              because the fillet models an apron's flare and a street mouth's flare is its corner
              return - see paint.kerb_opening_bands. At an intersection it is STOPPED by
              construction, which is why the two are one value and not two.
    STOPPED   ends square at the mouth, with nothing carried across.
    """
    CARRIED = "carried"
    DOTTED = "dotted"
    FILLETED = "filleted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class OpeningRule:
    """One row: what a marking does at a driveway, and what it does at an intersecting approach.

    Two columns and not one, because MUTCD states them a paragraph apart, in opposite directions,
    off the same definition (STANDARDS.md section 2). Section 3B.11, "Application of Pavement
    Markings through Intersections or Interchanges":

        (08) Guidance   edge line markings SHOULD BE DISCONTINUED across intersecting approaches
                        at intersections or interchanges
        (09) Guidance   driveways that DO NOT meet the definition of an intersection (see
                        Section 1C.02) SHOULD HAVE edge line markings MAINTAINED across the
                        intersecting approach of the driveway

    The reason for (09) is what the line MEANS - it marks where the running lane ends, and that
    does not stop being true because someone can turn in. The reason for (08) is the same
    sentence read the other way: at an intersection the running lane genuinely does end, because
    the ground beyond it is another street's.

    (Numbering is the 11th edition. Do not "correct" this to the 2009 Section 3B.07, which is
    "White Lane Line Markings for Non-Continuing Lanes" and says nothing about any of this.)

    `why` is the clause, per row, so somebody checking this against the manual can see which
    sentence each cell came from without reading the code that applies it.
    """
    at_a_driveway: AtAnOpening
    at_an_intersection: AtAnOpening
    dotted_as: "PaintKind | None" = None
    why: str = ""

    def __post_init__(self):
        wants_dashes = AtAnOpening.DOTTED in (self.at_a_driveway, self.at_an_intersection)
        if wants_dashes and self.dotted_as is None:
            raise ValueError("A DOTTED rule has to name the kind its dashes are laid in - a "
                             "broken line is a different instruction from the solid one it "
                             "continues, and the two must not share a kind (see "
                             "BIKE_LANE_DOTTED_EXTENSION).")
        if not wants_dashes and self.dotted_as is not None:
            raise ValueError(f"dotted_as={self.dotted_as} on a rule that never goes dotted.")
        if not self.why:
            raise ValueError("Every row cites the clause it came from.")

    def at(self, is_an_intersection: bool) -> AtAnOpening:
        return self.at_an_intersection if is_an_intersection else self.at_a_driveway


_CARRIES = (AtAnOpening.CARRIED, AtAnOpening.DOTTED)

_ZONE = OpeningRule(
    AtAnOpening.FILLETED, AtAnOpening.STOPPED,
    why="A hatched zone and the lines that bound it are one marking. MUTCD 3B.11(09) keeps an "
        "edge line across a driveway, but these lines are not only the edge of the travelled "
        "way - they are the outline of the zone behind them, and a zone that sweeps away on its "
        "run-out while its own boundary line carries straight on is the hook and the Y that "
        "paint.KerbOpenings.against describes. The line follows its zone.")

#: (upper, lower) pairs where the upper marking is LEGITIMATELY applied on top of the lower one.
#:
#: check_markings_do_not_collide is blanket over every pair of area-covering markings, and it is
#: right to be: two hatch zones over one patch means the design asserts two things about it. But
#: LAYERING is not that failure - a BIKE LANE symbol is painted white ON the green lane it marks,
#: and both renderers already draw it that way (plan_view zorder 4 over the surface's 2; separate
#: export channels).
#:
#: Declared here rather than in checks.py because it is a fact about what these markings ARE.
MAY_LIE_ON: frozenset = frozenset()      # filled in below, once the kinds exist


AT_AN_OPENING: dict[PaintKind, OpeningRule] = {
    LANE_EDGE_LINE: _ZONE,
    TAPER_LINE: _ZONE,
    LANE_NARROWING_FILL: _ZONE,
    TAPER_FILL: _ZONE,
    CORNER_HATCH_FILL: _ZONE,
    BUFFER_FILL: _ZONE,
    BUFFER_EDGE_LINE: _ZONE,
    DAYLIGHT_FILL: _ZONE,
    DAYLIGHT_EDGE_LINE: _ZONE,
    ZONE_END_LINE: _ZONE,
    BIKE_BUFFER_FILL: _ZONE,
    PARKING_EDGE_LINE: OpeningRule(
        AtAnOpening.CARRIED, AtAnOpening.STOPPED,
        why="MUTCD 3B.11(09) then (08), the pair this table exists for. Behind a parking edge "
            "line there are only stalls, which stop at a driveway because a stall there is a "
            "space you cannot park in; the line in front of them is a different statement and "
            "carries on. Across a STREET it is discontinued - it ran unbroken over the 49.7 ft "
            "mouth of Blackwell Avenue until (08) was read."),
    STALL_DIVIDER: OpeningRule(
        AtAnOpening.STOPPED, AtAnOpening.STOPPED,
        why="A stall tick belongs to the parking lane, which simply ends: there is no stall "
            "across an entrance of either kind, so there is nothing for a tick to divide."),
    BIKE_LANE_EDGE_LINE: OpeningRule(
        AtAnOpening.DOTTED, AtAnOpening.DOTTED, dotted_as=BIKE_LANE_DOTTED_EXTENSION,
        why="MUTCD 3B.11(05) and 9C.04: a bicycle lane's markings are dotted where the lane is "
            "crossed. Unlike an edge line the answer is the SAME either side of 1C.02, because "
            "the lane does not end at either - it is crossed at both, and a rider needs to see "
            "that it still runs here."),
    BIKE_LANE_SURFACE: OpeningRule(
        AtAnOpening.DOTTED, AtAnOpening.DOTTED, dotted_as=BIKE_LANE_SURFACE,
        why="The green is the lane, so it breaks where the lane's lines break and resumes as "
            "the same dashes - one marking seen three ways, which is why the phase is taken "
            "once off this surface and handed to the other two (PaintContext.dash_phase)."),
    BIKE_CONTRAFLOW_DIVIDER: OpeningRule(
        AtAnOpening.CARRIED, AtAnOpening.CARRIED,
        why="MUTCD 11th ed. 9E.04(02) and 9E.06(15), and NACTO's Urban Bikeway Design Guide for "
            "this facility: a bidirectional lane's yellow centreline continues through "
            "driveways and intersections. CARRIED and not DOTTED because it is ALREADY a "
            "broken line - its own cadence is the dotted pattern the standard asks for, and "
            "re-dashing it at an entrance would put a second, finer rhythm inside the first. "
            "It used to be cut and the inside re-laid as an exact complement, which is this "
            "row written out as two calls."),
    BIKE_LANE_SYMBOL: OpeningRule(
        AtAnOpening.CARRIED, AtAnOpening.CARRIED,
        why="A symbol is a discrete mark at a station, not a run of paint that an entrance "
            "crosses. It is never laid INSIDE a mouth - the placement rule puts it clear of one "
            "(SYMBOL_CLEAR_OF_OPENING_FT) - so there is nothing here to cut, and cutting the "
            "one that follows a driveway would delete the very reminder NACTO asks for it to be."),
    BIKE_LANE_DOTTED_EXTENSION: OpeningRule(
        AtAnOpening.CARRIED, AtAnOpening.CARRIED,
        why="This IS the extension - the marking laid inside an opening by the rules above. It "
            "is never cut against the opening it exists to cross."),
    APRON: OpeningRule(
        AtAnOpening.CARRIED, AtAnOpening.CARRIED,
        why="Built ground, not paint. A mountable apron is a surface every marking stops at "
            "(PaintContext.add_surface); it is not itself a marking to be stopped."),
    BOLLARD: OpeningRule(
        AtAnOpening.STOPPED, AtAnOpening.STOPPED,
        why="A post cannot be trimmed the way a stripe can - it is either standing in the way "
            "or it is not - so one in an entrance is dropped rather than shortened. See "
            "PaintContext.emit: paint broken over a driveway with the posts marching across it "
            "reads as a protected lane you are expected to drive through."),
}

def opening_rule(kind: PaintKind) -> OpeningRule:
    """What `kind` does at a kerb opening. Every declared marking has a row; see AT_AN_OPENING."""
    try:
        return AT_AN_OPENING[kind]
    except KeyError:
        raise KeyError(
            f"{kind} has no rule for what it does where a vehicle crosses the kerb. Add a row to "
            f"markings.AT_AN_OPENING - a driveway, a parking aisle, a side street and this "
            f"junction's own mouth are all the same event, and a marking with no answer for it "
            f"is one that will be drawn across an entrance and across an intersection with "
            f"nothing able to notice.") from None


def carries_across_an_intersection(kind: PaintKind) -> bool:
    """Whether `kind` may legitimately be drawn inside an intersecting approach's mouth.

    True for the markings the table carries or dots across one - a bicycle lane's dotted
    extension through a junction is IN the junction on purpose. Read by
    checks.NoPaintInsideTheJunction so the invariant tests the drawn geometry against the
    declaration rather than against a second list of exceptions.
    """
    return opening_rule(kind).at_an_intersection in _CARRIES


def kinds_in(channel: Channel) -> tuple[PaintKind, ...]:
    """Every marking that travels in `channel`, in declaration order."""
    return tuple(kind for kind in KINDS.values() if kind.channel is channel)


def require_every_kind(table: dict, what: str, skip: tuple = (Role.OBJECT,)) -> dict:
    """Return `table` if it covers every declared marking, or say which are missing.

    For the tables a renderer still has to write by hand - the plan view's styling is a real
    choice per marking, not something derivable - so that the omission fails at import instead
    of turning into a marking that is simply never drawn.
    """
    missing = sorted(kind.name for kind in KINDS.values()
                     if kind.role not in skip and kind not in table)
    if missing:
        raise ValueError(f"{what} is missing {missing} - every marking declared in "
                         f"src/geometry/markings.py has to be drawable, or it is invisible "
                         f"in that view with nothing to say so.")
    unknown = sorted(str(key) for key in table if key not in KINDS.values())
    if unknown:
        raise ValueError(f"{what} styles {unknown}, which src/geometry/markings.py does not "
                         f"declare - a renamed marking leaves the old entry behind.")
    return table


# A stripe painted on a coloured surface, declared rather than tolerated. checks'
# MarkingsDoNotCollide forbids a stroke from lying on a COLOUR or a SURFACE at all - a hatched
# FILL is defined BETWEEN its own bounding lines, so a line straddling its edge is the
# convention, but green asphalt under white or yellow paint is paint over paint - so anything
# that genuinely is a layer has to say so here.
MAY_LIE_ON = frozenset({
    (BIKE_LANE_SYMBOL, BIKE_LANE_SURFACE),
    # The two-way lane's yellow centre stripe runs the length of the green BY DESIGN: the green
    # is the facility and the stripe divides the two directions inside it. Every other line on
    # the green bounds it from outside, half a stripe clear - which is what makes this one worth
    # declaring rather than inferring from the fact that it currently overlaps.
    (BIKE_CONTRAFLOW_DIVIDER, BIKE_LANE_SURFACE),
})


def lies_legitimately_on(a: PaintKind, b: PaintKind) -> bool:
    """Whether these two markings are a layer rather than a collision, in either order."""
    return (a, b) in MAY_LIE_ON or (b, a) in MAY_LIE_ON


# Which zone keeps the asphalt when two would cover it - (yields, keeps), read one way only.
# checks.MarkingsDoNotCollide forbids the overlap; this says who gives way, and it is a
# STATUTORY-BEFORE-DISCRETIONARY ordering, not a drawing preference. R.S. 39:4-138 either
# prohibits parking on that ground or it does not, so a corner's daylight zone cannot be the
# thing that shrinks; a lane-narrowing buffer is this project's own proposal about width and can
# be the narrower for it. The pairs that need a rule are the ones that meet at a CORNER, where
# the two zones belong to different legs and overlap only because those legs' frames do - see
# paint.PaintContext._clear_of_the_paint_already_down.
YIELDS_THE_GROUND: frozenset = frozenset({
    (LANE_NARROWING_FILL, DAYLIGHT_FILL),
    (TAPER_FILL, DAYLIGHT_FILL),
    (BUFFER_FILL, DAYLIGHT_FILL),
    (CORNER_HATCH_FILL, DAYLIGHT_FILL),
    (BIKE_BUFFER_FILL, DAYLIGHT_FILL),
})


def yields_the_ground_to(a: PaintKind, b: PaintKind) -> bool:
    """Whether `a` gives way to `b` where both would cover one patch of road."""
    return (a, b) in YIELDS_THE_GROUND


# Every declared marking has a row, checked at import rather than by a test that has to remember
# to look. A marking with no row is one drawn across an entrance with nothing able to notice.
require_every_kind(AT_AN_OPENING, "markings.AT_AN_OPENING", skip=())
