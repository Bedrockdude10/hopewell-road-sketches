"""Where a label goes, and where prose goes instead - so a panel cannot cover its own design.

A label is drawn in POINTS and the street is drawn in FEET, and nothing reconciles the two: on
an 18 in sheet at 150 dpi one point of type covers 1.9 ft of ground at 1x and 4.7 ft at
--frame-scale 2.5, so a 40-character callout at 5.2 pt spans 235 ft - five times the width of
Broad St. The boxes crowd the 1x sheet and bury the 2.5x one. Measured on Broad & Greenwood's
2.5x two-way sheet before this module existed: a fifth of the bike lane's green surface lay under
white annotation and only 52% of it was still green, so the plan view showed half a facility the
3D render showed whole - two views reading the SAME PaintPiece list and disagreeing, because one
of them wrote on top of itself.

Which treatment a label gets is decided by WHAT IT IS, never by how big it came out. A size test
would be a design decision that changes with --frame-scale (.claude/SKILLS.md 0b): the same sheet
would key a callout at 2.5x and print it on the road at 1x.

  * PROSE - a sentence about a kerb - never goes on the carriageway at all. It becomes a
    numbered note in a keyed block, which is what a plan sheet does with prose.
  * A DIMENSION - "11.0 ft", "R=25 ft" - stays where it measures, but is pushed clear of the
    paint and of the labels already placed, with a leader line back to what it measures.

Both need the label's size IN FEET, which is a fact about the axes' limits - and those are set
after everything else is drawn. So labels are QUEUED while the panel is built and flushed at the
end. Placed as they were built, they were measured against a transform that had not been set yet.
"""
import math
import textwrap
from dataclasses import dataclass, field

from matplotlib.font_manager import FontProperties
from shapely.geometry import box
from shapely.ops import unary_union

from src.checks import Violation

# Deja Vu Sans' average advance, as a fraction of the font size, and matplotlib's default line
# spacing. Only used when the renderer cannot be reached (a canvas with no `get_renderer`), and
# only ever to over-estimate: a label measured too wide is placed further out, never on the paint.
_ESTIMATED_CHAR_WIDTH = 0.62
_LINE_SPACING = 1.32

# How far a label may be pushed to find clear ground, in units of ITS OWN height - so the search
# is the same search at any frame scale. A distance in feet would be a threshold on a length the
# render frame decides (.claude/SKILLS.md 0b): 30 ft is two label-heights at 1x and most of one
# at 2.5x, and the labels would sit differently on the two sheets for no design reason.
_SEARCH_STEPS = 14
_STEP_PER_HEIGHT = 0.75
# The fan tried at each distance, in degrees off `toward`. Straight out first, so a label that
# only needs a nudge keeps the direction its caller chose.
_FAN_DEG = (0, 22, -22, 45, -45, 68, -68, 90, -90, 135, -135, 180)
# A leader line is drawn once the box has left the thing it labels - half its own height out.
_LEADER_AT_HEIGHTS = 0.5
# How many positions the notes block is offered along each edge of the panel. Along the EDGE and
# not anywhere: prose belongs in the margin of a plan sheet, so the block is confined to the
# margin and slides within it. Four corners alone were not enough - on a 1x Columbia & Princeton
# sheet a 193 x 39 ft block clipped a marking in all four of them, with 200 ft of clear edge in
# between.
_NOTES_SLOTS_PER_EDGE = 11
# The notes block's measure, in CHARACTERS. A typographic choice and deliberately not a width in
# feet: a line length is a fact about reading, so it is the same line length on every sheet,
# where a foot measure would rewrap the prose when the frame widened. Unwrapped, one note ran to
# 104 characters - 193 ft of a 303 ft frame, wider than any margin a 1x sheet has.
_NOTE_MEASURE_CHARS = 62


def ft_per_point(ax) -> float:
    """How much ground one typographic point covers, on the axes as it is framed right now.

    Zero if the axes has no extent yet, which is the caller's signal that it is too early to
    measure - see the module docstring on why labels are queued.
    """
    (x0, _y0), (x1, _y1) = ax.transData.transform([(0.0, 0.0), (1.0, 0.0)])
    px_per_ft = abs(x1 - x0)
    if px_per_ft <= 0:
        return 0.0
    return (ax.figure.dpi / 72.0) / px_per_ft


def _text_size_pt(fig, text: str, fontsize: float, family, weight) -> tuple[float, float]:
    """One label's (width, height) in points, from the font metrics where they are reachable.

    `family` and `weight` are measured rather than assumed: bold is ~5% wider than regular and
    DejaVu Sans Mono wider again, so measuring the notes block as regular sans under-reports it
    and the block is placed as though it were smaller than it is drawn.
    """
    lines = text.split("\n")
    renderer = getattr(fig.canvas, "get_renderer", None)
    if renderer is not None:
        prop = FontProperties(size=fontsize, family=family, weight=weight or "normal")
        measured = [renderer().get_text_width_height_descent(line, prop, False)[:2]
                    for line in lines]
        px_per_pt = fig.dpi / 72.0
        return (max(w for w, _h in measured) / px_per_pt,
                sum(h for _w, h in measured) / px_per_pt * _LINE_SPACING)
    return (max(len(line) for line in lines) * _ESTIMATED_CHAR_WIDTH * fontsize,
            len(lines) * fontsize * _LINE_SPACING)


def label_box_ft(ax, text: str, fontsize: float, pad_pt: float = 0.0, *,
                  family=None, weight=None) -> tuple[float, float]:
    """The ground one label covers: (width, height) in feet, box padding included."""
    w_pt, h_pt = _text_size_pt(ax.figure, text, fontsize, family, weight)
    scale = ft_per_point(ax)
    return (w_pt + 2 * pad_pt) * scale, (h_pt + 2 * pad_pt) * scale


def _rect(centre: tuple[float, float], w: float, h: float):
    return box(centre[0] - w / 2, centre[1] - h / 2, centre[0] + w / 2, centre[1] + h / 2)


def _unit(toward) -> tuple[float, float]:
    if toward is None:
        return (0.0, 1.0)
    length = math.hypot(*toward)
    return (0.0, 1.0) if length == 0 else (toward[0] / length, toward[1] / length)


@dataclass(frozen=True)
class _Dimension:
    text: str
    xy: tuple[float, float]
    toward: tuple[float, float] | None
    fontsize: float
    pad: float
    style: dict


@dataclass(frozen=True)
class _Caption:
    text: str
    at: tuple[float, float]        # axes fraction
    ha: str
    va: str
    fontsize: float
    pad: float
    style: dict


@dataclass(frozen=True)
class _Note:
    subject: str
    says: str
    xy: tuple[float, float]
    toward: tuple[float, float] | None
    colour: str


@dataclass
class LabelPlacer:
    """Every label on one panel, placed so that none of them covers the design.

    One placer per axes. `dimension` and `note` queue; `flush` measures, places and draws. What
    it may not cover is passed to `flush` rather than held here, because the paint is built after
    most labels have been asked for.
    """
    captions: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    placed: list = field(default_factory=list)          # boxes in feet, already committed
    violations: list = field(default_factory=list)

    def caption(self, text: str, at, *, ha: str = "center", va: str = "bottom",
                fontsize: float = 8.0, pad: float = 0.25, **style) -> None:
        """A line about the whole panel, pinned to the frame in AXES FRACTION.

        Queued with the rest and not simply drawn, because a caption pinned to the frame still
        stands on ground: the signalization line sits across the bottom of the panel, and the
        notes block picked that corner and printed over it. Registering the caption is what lets
        the block see it.
        """
        self.captions.append(_Caption(text, (float(at[0]), float(at[1])), ha, va,
                                      fontsize, pad, style))

    def dimension(self, text: str, xy, *, toward=None, fontsize: float = 7.0,
                  pad: float = 0.15, **style) -> None:
        """A measurement, to be drawn at `xy` or as close to it as clear ground allows."""
        self.dimensions.append(_Dimension(text, (float(xy[0]), float(xy[1])), toward,
                                          fontsize, pad, style))

    def note(self, subject: str, says: str, xy, *, toward=None, colour: str = "black") -> None:
        """A sentence about one place. Keyed to a number on the drawing; the prose goes in the
        block. Returns nothing: the number is assigned by queue order, so a panel's notes are
        numbered the way they are read."""
        self.notes.append(_Note(subject, says, (float(xy[0]), float(xy[1])), toward, colour))

    # ---------------------------------------------------------------- placing

    def _clear_of(self, rect, keep_off) -> bool:
        if keep_off is not None and not keep_off.is_empty and rect.intersects(keep_off):
            return False
        return not any(rect.intersects(other) for other in self.placed)

    def _position(self, anchor, w, h, toward, keep_off):
        """Where this box can sit: straight out from `anchor` if that is clear, else the first
        clear spot in a widening fan. The last candidate if nothing is clear, with a violation -
        a label that cannot be placed still has to be drawn, or the drawing loses the fact."""
        ux, uy = _unit(toward)
        candidate = anchor
        for step in range(_SEARCH_STEPS):
            dist = step * _STEP_PER_HEIGHT * h
            for turn in _FAN_DEG:
                if step == 0 and turn:
                    continue
                a = math.radians(turn)
                dx, dy = ux * math.cos(a) - uy * math.sin(a), ux * math.sin(a) + uy * math.cos(a)
                candidate = (anchor[0] + dx * dist, anchor[1] + dy * dist)
                rect = _rect(candidate, w, h)
                if self._clear_of(rect, keep_off):
                    return candidate, rect, True
        # Nowhere clear: drawn back AT the thing it labels rather than at the last position
        # tried. A label parked 10 heights out with a leader across the junction is worse than
        # one over the paint, and the violation is what says which it is.
        return anchor, _rect(anchor, w, h), False

    def flush(self, ax, keep_off=None) -> list:
        """Measure, place and draw every queued label. Returns the violations it could not avoid.

        Called after the axes limits are set - `ft_per_point` is meaningless before that, and a
        label measured in the wrong frame is placed in the wrong place.
        """
        self.violations = []
        if ft_per_point(ax) <= 0:
            return self.violations
        # Captions first: they are pinned, so everything else has to work around them.
        for caption in self.captions:
            self._draw_caption(ax, caption)
        for note_number, note in enumerate(self.notes, start=1):
            self._draw_one(ax, str(note_number), note.xy, note.toward, fontsize=5.8,
                           pad=0.22, keep_off=keep_off, color="white", fontweight="bold",
                           bbox=dict(boxstyle="circle,pad=0.22", fc=note.colour, ec="none",
                                     alpha=0.95))
        for dim in self.dimensions:
            self._draw_one(ax, dim.text, dim.xy, dim.toward, fontsize=dim.fontsize,
                           pad=dim.pad, keep_off=keep_off, **dim.style)
        if self.notes:
            self._draw_notes_block(ax, keep_off)
        return self.violations

    def _draw_caption(self, ax, caption) -> None:
        w, h = label_box_ft(ax, caption.text, caption.fontsize, caption.pad * caption.fontsize,
                            weight=caption.style.get("fontweight"))
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x = xmin + caption.at[0] * (xmax - xmin)
        y = ymin + caption.at[1] * (ymax - ymin)
        left = {"center": x - w / 2, "left": x, "right": x - w}[caption.ha]
        bottom = {"center": y - h / 2, "bottom": y, "top": y - h}[caption.va]
        self.placed.append(box(left, bottom, left + w, bottom + h))
        ax.annotate(caption.text, xy=caption.at, xycoords="axes fraction", ha=caption.ha,
                    va=caption.va, fontsize=caption.fontsize, zorder=9, **caption.style)

    def _draw_one(self, ax, text, anchor, toward, *, fontsize, pad, keep_off, **style) -> None:
        pad_pt = pad * fontsize
        w, h = label_box_ft(ax, text, fontsize, pad_pt, weight=style.get("fontweight"))
        at, rect, clear = self._position(anchor, w, h, toward, keep_off)
        self.placed.append(rect)
        if not clear:
            self.violations.append(Violation(
                check="label_covers_paint",
                detail=(f"no clear ground for the label {text.splitlines()[0]!r}: it is "
                        f"{w:.0f} x {h:.0f} ft on this sheet and every position tried within "
                        f"{_SEARCH_STEPS * _STEP_PER_HEIGHT:.0f} label-heights is over paint or "
                        f"another label. Shorten it, or make it a keyed note."),
                where=at, fatal=False))
        moved = math.hypot(at[0] - anchor[0], at[1] - anchor[1]) > _LEADER_AT_HEIGHTS * h
        leader = dict(arrowstyle="-", linewidth=0.5, shrinkA=0.5, shrinkB=1.0,
                      color=style.get("color", "black"), alpha=0.8) if moved else None
        ax.annotate(text, xy=anchor, xytext=at, textcoords="data", arrowprops=leader,
                    fontsize=fontsize, ha="center", va="center", zorder=9, **style)

    # ---------------------------------------------------------------- the keyed block

    def _edge_slots(self, ax, w: float, h: float):
        """Every position the notes block may take: along the four edges of the panel.

        In the panel rather than under it because every caller lays its figure out differently -
        one panel, two panels, a legend below, a change panel in the right margin - and a block
        that reaches outside the axes has to negotiate with all three. Confined to the margin
        because that is where a plan sheet's prose goes, and slid ALONG it because a corner is
        one position and an edge is many.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        inset = 0.012 * (xmax - xmin)
        x_lo, x_hi = xmin + inset, xmax - inset - w
        y_lo, y_hi = ymin + inset, ymax - inset - h
        steps = [i / (_NOTES_SLOTS_PER_EDGE - 1) for i in range(_NOTES_SLOTS_PER_EDGE)]
        along_x = [x_lo + t * (x_hi - x_lo) for t in steps]
        along_y = [y_lo + t * (y_hi - y_lo) for t in steps]
        return ([(x, y_lo) for x in along_x] + [(x, y_hi) for x in along_x]
                + [(x_lo, y) for y in along_y] + [(x_hi, y) for y in along_y])

    def _draw_notes_block(self, ax, keep_off) -> None:
        """The prose, in whichever margin slot has the least design under it.

        CHOSEN by measuring, so "least ink" is a fact about this junction rather than a guess
        about where streets usually are - the block lands top-left on Broad & Greenwood and
        bottom-right on W Broad & Louellen, and neither was decided here.
        """
        lines = []
        for i, note in enumerate(self.notes, 1):
            lines += textwrap.wrap(f"{i}  {note.subject} - {note.says}",
                                   width=_NOTE_MEASURE_CHARS, subsequent_indent="   ")
        text = "\n".join(lines)
        fontsize = 5.4
        w, h = label_box_ft(ax, text, fontsize, pad_pt=0.4 * fontsize, family="monospace")
        best = None
        for x0, y0 in self._edge_slots(ax, w, h):
            rect = box(x0, y0, x0 + w, y0 + h)
            covered = (rect.intersection(keep_off).area
                       if keep_off is not None and not keep_off.is_empty else 0.0)
            # Paint first, then the labels already placed: a slot clear of the design but sitting
            # on a dimension call-out is still the better slot, and a tuple says that without
            # inventing a weight for how many square feet a label is worth.
            score = (covered, sum(rect.intersection(o).area for o in self.placed))
            if best is None or score < best[0]:
                best = (score, x0, y0, rect)
        (covered, _overlap), x0, y0, rect = best
        self.placed.append(rect)
        if covered > 0:
            self.violations.append(Violation(
                check="label_covers_paint",
                detail=(f"the notes block covers {covered:.0f} sq ft of paint - no slot in this "
                        f"panel's margin is clear of the design. It is {w:.0f} x {h:.0f} ft on "
                        f"this sheet, against a {ax.get_xlim()[1] - ax.get_xlim()[0]:.0f} ft "
                        f"frame."),
                where=(rect.centroid.x, rect.centroid.y), fatal=False))
        ax.text(x0, y0, text, ha="left", va="bottom", family="monospace", fontsize=fontsize,
                linespacing=1.3, zorder=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#666666", alpha=0.93))

    # ---------------------------------------------------------------- what got covered anyway

    def covers(self, geometries) -> float:
        """How much of `geometries` ended up under a label. The measurement a test asserts on:
        the plan view's whole job is to show what the export exports."""
        if not self.placed:
            return 0.0
        boxes = unary_union(self.placed)
        return sum(g.intersection(boxes).area for g in geometries
                   if g is not None and not g.is_empty)
