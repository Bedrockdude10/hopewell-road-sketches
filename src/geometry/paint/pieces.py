"""WHAT A PAINTED PIECE IS: one marking, its kind, and how wide it is actually painted.

The leaf of this package - `PaintPiece` is imported in ~20 modules and both renderers draw what
they are handed here, so nothing in this file depends on anything else in it.

`stroke_width_ft` answering None is load-bearing, not a gap: a FILL has no stroke, and a caller
that treats a missing width as zero draws a line the collision check cannot see. That is how a
marking came to be checked with no width at all.
"""
from dataclasses import dataclass
from enum import StrEnum
from shapely.geometry import LineString, Polygon
from src.geometry.markings import EDGE_LINE_WIDTH_M, PaintKind
from src.render.coords import FT_TO_M

class RimCause(StrEnum):
    """What cut a zone short, for the line that closes it there.

    CROSSING a painted crossing, which the hatching runs into and is cut by - the clean diagonal
             end you see on a real street.
    OPENING  a gap in the kerb. At a DRIVEWAY that is the apron's fillet, which the hatching
             stops short of, because that arc's chord is at the hatch angle and a stroke laid
             beside it reads as a fork. At an INTERSECTING APPROACH there is no fillet - a street
             mouth has no apron (see kerb_opening_bands) - so the rim is the square end instead.
    """
    CROSSING = "crossing"
    OPENING = "opening"


@dataclass(frozen=True)
class PaintPiece:
    """One painted marking. `kind` is what it is, `leg`/`side` where it belongs.

    leg/side are None for the corner treatments, which sit at a corner between two legs and
    so belong to neither - they are the reason the curb check below skips pieces without a
    side rather than assuming one.

    `kind` is a src/geometry/markings.py:PaintKind rather than a string, so a piece carries its
    own answer to "how is this drawn, and where does it travel to the 3D render" instead of
    every consumer looking that up in a table of its own.
    """
    kind: PaintKind
    geometry: LineString | Polygon
    leg: str | None = None
    side: str | None = None
    # What cut this piece, if it is the line along a zone's CUT END rather than along its length.
    # A rim carries the same `kind` as the zone's edge line (see PaintContext.rim), so this is the
    # only thing distinguishing the two. The cause matters because the two ends differ: hatching
    # keeps half a spacing off an OPENING's fillet, whose chord runs at the hatch angle and so
    # reads as a stroke, but runs straight into a CROSSING's diagonal.
    rim: "RimCause | None" = None

    @property
    def is_fill(self) -> bool:
        """Hatched paint - asked of the MARKING, not of its geometry.

        The two answers differ: a bollard is stored as a degenerate polygon standing in for a
        point, so a geometry test would call it a fill. See markings.Role.
        """
        return self.kind.is_fill

    @property
    def covers_area(self) -> bool:
        """Occupies ground rather than tracing a line: a hatched zone or a built surface."""
        return self.kind.covers_area

# The painting order reserved for built ground - an apron. Everything else is cut around it, so
# it has to be laid before anything else is painted; see PaintContext.seal_surfaces.
SURFACE_PAINT_GROUP = 0

# The painted width of a lane-edge line, in FEET. DERIVED from the figure the channels carry
# (markings.EDGE_LINE_WIDTH_M), because that is now the single home for how wide paint is laid
# and every LINE channel is declared against it. Paint has width, and
# where it goes decides whether the lane behind it is really the width it claims: an edge line
# CENTRED on the 11 ft mark puts half its own body inside the lane, leaving 10.59 ft. So the line
# is placed OUTSIDE the mark - its inner edge lands on 11 ft - and the hatching starts outside the
# line. The width comes out of the treatment, not out of the travel lane.
LANE_EDGE_LINE_WIDTH_FT = EDGE_LINE_WIDTH_M / FT_TO_M


def stroke_width_ft(kind: PaintKind) -> float | None:
    """How wide `kind` is actually painted, in feet, or None if it is not a stroke.

    The one place metres become feet for a stripe width. A LINE and a FILL's hatch strokes are
    both laid at a real width by the 3D renderer, so both have one; a COLOUR or a SURFACE
    travels as its polygon and its extent is already in the geometry.

    This exists so a check can give a line the body it has. checks.MarkingsDoNotCollide
    compared only markings that cover area, and a line covered none - which is how the
    crossbike's edge lines came to be ruled along the green's own faces, painting 0.41 ft of
    white over colour on every mark, invisible in a plan view that strokes 1.6 pt about an axis.
    """
    if kind.channel is None or kind.channel.stroke_width_m is None:
        return None
    return kind.channel.stroke_width_m / FT_TO_M


def of_kind(pieces: list[PaintPiece], *kinds: PaintKind) -> list[PaintPiece]:
    return [p for p in pieces if p.kind in kinds]


def in_channel(pieces: list[PaintPiece], channel) -> list[PaintPiece]:
    """Every piece the 3D render will find in one of its JSON lists (markings.Channel).

    The export used to name the kinds per list itself, in a table beside the one in
    src/geometry/markings.py.
    """
    return [p for p in pieces if p.kind.channel is channel]


def _one(geometries):
    """These builders take a `sides` tuple and return a list; called per side they return at
    most one. Unpacking here keeps the caller from pretending otherwise."""
    return geometries[0] if geometries else None


def _dot(point) -> Polygon:
    """A bollard is a point, but PaintPiece holds geometry so the curb check can treat every
    piece the same way. A degenerate square is the cheapest honest polygon for one."""
    x, y = point
    return Polygon([(x, y), (x + 1e-6, y), (x + 1e-6, y + 1e-6), (x, y + 1e-6)])
