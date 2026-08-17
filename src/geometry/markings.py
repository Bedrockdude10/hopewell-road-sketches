"""Every kind of marking this project paints, and every channel a renderer reads, declared once.

A marking used to be a bare string. That string was a key into three tables in three modules -
`plan_view.PAINT_STYLE` (how the 2D view draws it), `export.PAINT_KIND_LISTS` (which JSON list
the 3D render finds it in) and `plan_view.PAINT_FILL_EDGE` (what colour outlines it) - plus
`kind == "bollard"` and `kind.startswith("daylight")` branches in the invariants. Nothing tied
the four together, so adding a marking meant remembering four edits in four files, and
forgetting one produced a treatment that was built correctly, drawn in one view and silently
missing from the other. That happened three times: renaming `buffer_taper_*` to
`daylight_taper_*` orphaned the taper, `daylight_fill` was never wired up, and the bike lane's
bollards reached the plan view and nothing else.

So the declarations live here, and the relationship between them is checked when this module is
imported rather than by a test that has to remember to look:

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
             from FILL, which is hatching and reaches the 3D render as the diagonal strokes that
             are actually applied - a green lane has no strokes, it is the asphalt's colour, so
             it travels as its polygon and is drawn as one. Distinct from SURFACE too, and that
             distinction is load-bearing rather than pedantic: a SURFACE is built ground that
             every marking is cut around (PaintContext.seal_surfaces), and colouring a bike lane
             must not cut the lane's own edge lines out of existence.
    OBJECT   a physical thing standing on the road - a flex post. The 3D render builds objects
             only from props, so an OBJECT marking is the plan view's copy of something that
             must ALSO exist as a prop; check_bollards_are_props enforces exactly that.
    """
    LINE = "line"
    FILL = "fill"
    SURFACE = "surface"
    COLOUR = "colour"
    OBJECT = "object"


@dataclass(frozen=True)
class Channel:
    """One list in the exported geometry JSON, read by name in scripts/blender/blender_scene.py.

    A channel carries one role, so everything in it is drawn the same way at the far end: a
    LINE channel becomes add_paint_polyline calls, a FILL channel becomes the hatch strokes
    inside the zone, a SURFACE channel becomes an extruded polygon.
    """
    key: str
    role: Role

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
        a degenerate polygon standing in for a point, which is why the invariants used to carry
        `and p.kind != "bollard"` everywhere they asked about polygons."""
        return self.role is Role.FILL

    @property
    def covers_area(self) -> bool:
        """Occupies ground rather than tracing a line: a hatched zone, a built surface, or a
        coloured stretch of carriageway. What MarkingsDoNotCollide compares, so a green bike
        lane laid over a hatched buffer would be reported the way any doubled paint is."""
        return self.role in (Role.FILL, Role.SURFACE, Role.COLOUR)

    @property
    def is_object(self) -> bool:
        return self.role is Role.OBJECT


# --------------------------------------------------------------------------------------
# The channels, in the order they appear in the exported JSON.
# --------------------------------------------------------------------------------------
LANE_NARROWING_EDGE_LINES = Channel("lane_narrowing_edge_lines", Role.LINE)
LANE_NARROWING_TAPER_LINES = Channel("lane_narrowing_taper_lines", Role.LINE)
LANE_NARROWING_HATCH_LINES = Channel("lane_narrowing_hatch_lines", Role.FILL)
CORNER_HATCHING_LINES = Channel("corner_hatching_lines", Role.FILL)
PARKING_EDGE_LINES = Channel("parking_edge_lines", Role.LINE)
PARKING_STALL_DIVIDER_LINES = Channel("parking_stall_divider_lines", Role.LINE)
# The daylight zones (R.S. 39:4-138 - see src/geometry/daylighting.py) share the parking
# buffer's channels, because on a real street they are the same white hatching and the same
# white lines. The plan view distinguishes them by colour; asphalt does not.
PARKING_BUFFER_HATCH_LINES = Channel("parking_buffer_hatch_lines", Role.FILL)
PARKING_BUFFER_EDGE_LINES = Channel("parking_buffer_edge_lines", Role.LINE)
# Empty today, and deliberately kept: a curved line needs Blender's add_paint_polyline rather
# than add_paint_line, so tapers travel in their own channel. Daylight zones went square-ended
# (a keep-clear block has no taper) and nothing else needs it yet, but blender_scene.py reads
# the key and a lane-narrowing taper could be routed here.
PARKING_BUFFER_TAPER_LINES = Channel("parking_buffer_taper_lines", Role.LINE)
BIKE_LANE_EDGE_LINES = Channel("bike_lane_edge_lines", Role.LINE)
BIKE_LANE_HATCH_LINES = Channel("bike_lane_hatch_lines", Role.FILL)
# The lane's own asphalt, painted green. This used to be plan-view-only - the note here read
# "nothing in scripts/blender/ paints a coloured surface yet", so the 3D render showed the
# striping alone and the two views disagreed about what the proposal looked like. It is a real
# treatment and a widely used one, so it now travels to the render as the polygon it is.
BIKE_LANE_SURFACE_POLYGONS = Channel("bike_lane_surface_polygons", Role.COLOUR)
# The YELLOW centre stripe of a two-way bike lane, separating opposing riders. Its own channel
# rather than more BIKE_LANE_EDGE_LINES, because the channel is what decides the colour at the
# far end: blender_scene.py draws every edge-line channel in the white marking material, and a
# yellow line is not a white line that happens to be somewhere else. It is the same distinction
# the road's own centreline gets, for the same reason - yellow means opposing directions.
BIKE_LANE_CONTRAFLOW_LINES = Channel("bike_lane_contraflow_lines", Role.LINE)
CORNER_APRON_POLYGONS = Channel("corner_apron_polygons", Role.SURFACE)

CHANNELS: tuple[Channel, ...] = (
    LANE_NARROWING_EDGE_LINES, LANE_NARROWING_TAPER_LINES, LANE_NARROWING_HATCH_LINES,
    CORNER_HATCHING_LINES, PARKING_EDGE_LINES, PARKING_STALL_DIVIDER_LINES,
    PARKING_BUFFER_HATCH_LINES, PARKING_BUFFER_EDGE_LINES, PARKING_BUFFER_TAPER_LINES,
    BIKE_LANE_EDGE_LINES, BIKE_LANE_HATCH_LINES, BIKE_LANE_SURFACE_POLYGONS,
    BIKE_LANE_CONTRAFLOW_LINES, CORNER_APRON_POLYGONS,
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
# entrance, it is crossed there. Its own kind rather than more BIKE_LANE_EDGE_LINE pieces so a
# check or a reader can tell a continuous stripe from a broken one, and because a dotted line is a
# different instruction; it travels in the same channel, since a dash is a short stripe and both
# renderers already draw one. The dashes are in the GEOMETRY (see paint.py:_dashes_along), not in a
# line style, so the plan view and the 3D render cannot disagree about where the gaps fall.
BIKE_LANE_DOTTED_EXTENSION = _kind("bike_lane_dotted_extension", Role.LINE, BIKE_LANE_EDGE_LINES)
BIKE_BUFFER_FILL = _kind("bike_buffer_fill", Role.FILL, BIKE_LANE_HATCH_LINES)
# The green a bike lane's asphalt is painted, between its two edge stripes - the lane itself
# rather than anything beside it.
BIKE_LANE_SURFACE = _kind("bike_lane_surface", Role.COLOUR, BIKE_LANE_SURFACE_POLYGONS)
# The centre stripe of a TWO-WAY bike lane. Yellow and broken, following MUTCD's rule for a
# two-way bikeway: yellow because it divides opposing traffic (the same meaning it carries on
# the roadway), broken because passing is permitted where sight distance allows.
BIKE_CONTRAFLOW_DIVIDER = _kind("bike_contraflow_divider", Role.LINE, BIKE_LANE_CONTRAFLOW_LINES)
# Built ground rather than paint: a flush, drivable corner surface.
APRON = _kind("apron", Role.SURFACE, CORNER_APRON_POLYGONS)
# A flex-post delineator. Paint draws the plan view's marker; the render needs a prop.
BOLLARD = _kind("bollard", Role.OBJECT)

KINDS: dict[str, PaintKind] = dict(_REGISTRY)

# THE LINES THAT BOUND A HATCHED ZONE, as opposed to the ones that mark a lane or a parking stall.
# The distinction decides how each ends at a driveway, and it is not derivable from the role - both
# groups are LINEs. A zone's edge line is part of the zone: when the hatching sweeps away from a
# driveway mouth on its fillet (see paint.py:kerb_opening_bands) this line has to sweep with it, or
# it runs straight on to the mouth with no zone behind it and the fillet's own rim cuts diagonally
# across it - which in the render came out as a hook and a Y where the two disagreed.
#
# A bike lane's edge line is the opposite case and stays out of this set: the lane crosses the
# entrance, so its line stops at the mouth and continues across as a dotted extension. A stall
# divider likewise belongs to the parking lane, which simply ends.
ZONE_BOUNDARY_LINES: frozenset = frozenset({
    LANE_EDGE_LINE, TAPER_LINE, BUFFER_EDGE_LINE, DAYLIGHT_EDGE_LINE, ZONE_END_LINE,
})

# Markings that are the EDGE OF THE TRAVELLED WAY: they run unbroken past a DRIVEWAY, and are
# discontinued across an INTERSECTING APPROACH. Two rules, one definition between them, and the
# name says only the half that is about this set's membership - which of the two a given gap gets
# is kerbs.OpeningSource.is_an_intersection's answer, applied in paint.KerbOpenings.against.
#
# MUTCD 11th ed. Section 3B.11, "Application of Pavement Markings through Intersections or
# Interchanges" (STANDARDS.md section 2 quotes both in full):
#
#   (08) Guidance  edge line markings SHOULD BE DISCONTINUED across intersecting approaches
#   (09) Guidance  driveways that DO NOT meet the definition of an intersection (Section 1C.02)
#                  SHOULD HAVE edge line markings MAINTAINED across the intersecting approach
#
# The reason for (09) is what the line MEANS - it marks where the running lane ends, and that
# does not stop being true because someone can turn in. The reason for (08) is the same sentence
# read the other way: at an intersection the running lane genuinely does end, because the ground
# beyond it is another street's.
#
# (Cited as Section 3B.07 here until 2026-08-17, from the 2009 numbering. In the 11th edition
# 3B.07 is "White Lane Line Markings for Non-Continuing Lanes" and says nothing about any of
# this; the counterpart to 3B.11 is 3B.09(07).)
#
# Only the PARKING edge line is here, and the omissions are deliberate. LANE_EDGE_LINE and
# BUFFER_EDGE_LINE bound a hatched zone as well as the lane, and a zone that sweeps away on
# its run-out while its own boundary line carries straight on is the hook-and-Y that
# KerbOpenings.against describes - those follow their zone. Behind a parking edge line there
# are only stalls, which stop at a driveway because a stall there is a space you cannot park
# in; the line in front of them is a different statement and carries on.
LINES_UNBROKEN_BY_A_DRIVEWAY: frozenset = frozenset({PARKING_EDGE_LINE})


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
