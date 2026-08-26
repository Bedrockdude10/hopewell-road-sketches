"""A STRIP PLAN of one corridor: the whole street, straightened, on stacked panels.

    .venv/bin/python scripts/corridor_render.py --road "Broad Street"
    .venv/bin/python scripts/corridor_render.py --road "Princeton Avenue"

WHICH DESIGN IS DRAWN IS THE STREET'S DECISION, NOT THIS SCRIPT'S - `route_decision_for` on the
corridor's name. A street with a CorridorFacility gets the facility sheet: the lane, its buffer,
the far kerb's parking, and the which-kerb-carries-it table. A street with a CorridorCalming gets
the calming sheet: both travel lanes at target, every recovered foot marked or hatched, and the
refusal that decided against a facility printed on the drawing. A street this project has decided
nothing about gets the second sheet with nothing claimed as a proposal.

WHY STRAIGHTENED. Broad St through the borough is 3,693 ft long and about 47 ft wide - a 78:1
aspect. Drawn in world coordinates on one sheet the carriageway is a hairline; drawn at a
readable width it is eight feet of paper. So it is plotted in the corridor's OWN frame, station
along the page and offset across it, and cut into panels with match stations - which is how every
roadway plan set has drawn a corridor for a century, and for this reason.

What that costs, stated because a drawing may not quietly mislead: the street's curvature is
removed. Broad St bends, and on these panels it does not. Every LENGTH, WIDTH and OFFSET is true -
they are measured in the frame the drawing is plotted in - and the plan geometry is not. For the
shape of a junction, read the per-junction plan views; this drawing is for what runs BETWEEN them,
which is the thing no other drawing here shows.
"""
import argparse
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.geometry.corridor_paint import (CORRIDOR_SAMPLE_FT, JUNCTION_MOUTH,
                                         centred_on_its_kerbs, facility_side,
                                         far_kerb_lane_edge,
                                         hatch_bands, kerb_offset_ft, paint_facility,
                                         stall_bands, contraflow_centreline,
                                         green_extension_spans, stall_footprints, stall_marks,
                                         stall_room_spans, symbol_stations, travel_way_edges)
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.network import (_complement_spans, _merged_spans, corridor_facts,
                                  corridors_from_models)
from src.geometry.treatments import (BROAD_ST_TWO_WAY_BIKEWAY, CorridorFacility,
                                     TARGET_LANE_WIDTH_FT, route_decision_for)
from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT
from src.site import list_sites

ASPHALT = "#d9d9d9"
KERB = "#222222"
LANE_GREEN = "#57a773"
BUFFER_GREY = "#9a9a9a"
PAINT_WHITE = "#ffffff"
#: The travel way's own two edges. Dark, because it is the one line a reader is asked to
#: measure the hatch AGAINST - white on light asphalt is a marking to look past.
TRAVEL_EDGE = "#404040"
POST = "#e8663c"
GREEN_HOT = "#2f7d4f"
YELLOW = "#e6c000"
PARKING_BLUE = "#4b7fb5"
OPENING = "#8a5a1f"
GAP_RED = "#c1272d"
DAYLIGHT = "#e8c33c"
MOUTH_BLUE = "#3b6ea5"
# The same fill plan_view.py uses for LANE_NARROWING_FILL - "parking or hatching, never neither"
# has to look like the one hatch this project already draws, not a second visual language for
# the same claim. Gold is reused for the ROOM reason (a property of THIS DESIGN's own section);
# a distinct colour and pattern mark the LEGAL reason (a property of the STREET - a corner, a
# stop sign, a hydrant), because a reader asking "why can't this be parking" needs a different
# fix for each: a narrower section for one, nothing at all for the other.
HATCH_GOLD = "gold"
HATCH_EDGE = "goldenrod"
HATCH_LEGAL = "orchid"
HATCH_LEGAL_EDGE = "darkorchid"


def _intersect(a, b):
    """The spans in both - a stall must be legal AND have room, not one or the other."""
    out = []
    for lo_a, hi_a in a:
        for lo_b, hi_b in b:
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if hi > lo:
                out.append((lo, hi))
    return tuple(sorted(out))


def nominal_lane_edge(_station_ft: float) -> float:
    """Where the travel lane's edge sits on a street carrying no facility - the BASELINE datum.

    A stall count is only comparable with another stall count measured to the same line, and the
    line a design with no bikeway on it holds is TARGET_LANE_WIDTH_FT. The no-bikeway sheet and
    the baseline printed beside the facility's own count both measure to this, so "kept" and
    "there without it" are one question asked of two designs rather than two questions.
    """
    return TARGET_LANE_WIDTH_FT


def stall_spans(corridor, facts, side: str, edge_at):
    """Where a stall may actually be marked on one kerb, given where that kerb's lane edge sits.

    Legal room AND street room, because a length the statute permits and four feet wide holds no
    car - the distinction that took the honest figure from 108 to 32 on the south-kerb option.

    `edge_at` IS THE ONLY DIFFERENCE BETWEEN THE TWO QUESTIONS THIS ANSWERS, which is why it is a
    parameter and not a branch. `far_kerb_lane_edge(paint)` asks what the far kerb keeps once the
    section pushes the divider toward it, per run, so a constrained stretch and a standard one are
    not given the same allowance; `nominal_lane_edge` asks what a kerb holds with no facility at
    all. Everything else about a stall - the statute, the driveway mouths, the width floor, the
    walk in whole cars - is the same question either way and is asked once, here. It was written
    twice, once here against the facility and once inline in `_calming_strip` against the nominal
    lane, which is two derivations of one count and the reason the sheet could not print both.
    """
    mouths = _merged_spans([(o.start_ft, o.end_ft)
                            for opening_side, o in facts.openings if opening_side == side])
    clear = _complement_spans(mouths, 0.0, corridor.length_ft)
    room = stall_room_spans(corridor, side, edge_at)
    return _intersect(_intersect(facts.by_side("parkable", side), clear), room)


def baseline_stalls(corridor, facts) -> dict:
    """{compass: stalls} - what this corridor's two kerbs hold with no facility on either.

    THE DENOMINATOR THE PROPOSAL SHEET WAS MISSING, and it is here rather than left to the reader
    because the two parking figures this project could already produce are not comparable and
    subtracting them invents a loss. `corridor_report.py` gives 243 stalls: every 22 ft of legally
    parkable kerb over both kerbs, an upper bound counted whether or not the street's width there
    was ever measured. This sheet gives 45: stalls DRAWN on one kerb, after the width test, the
    driveway mouths and the walk in whole cars. Most of the gap between those is method, not
    bikeway - so "we lose 198 spaces" is what a reader does with them unaided, and it is wrong.

    Measured to `nominal_lane_edge` on BOTH kerbs by the same walk that draws the proposal's
    boxes, so the difference between this and `kept` is the facility and nothing else. Both kerbs
    because the one carrying the lane loses all of its parking, which is most of the answer.
    """
    return {compass: stall_marks(corridor, side,
                                 stall_spans(corridor, facts, side, nominal_lane_edge))[1]
            for compass, side in ((compass, facility_side(corridor, compass))
                                  for compass in ("north", "south"))}


def kerbside_parking(corridor, facts, side: str, edge_at):
    """(bands, marks, labels, hatch) - every foot of one kerb, drawn once and to scale.

    ONE FOOTPRINT FEEDS ALL FOUR, because the sheet is read as an area and not as a caption: a
    reader weighing 45 stalls against a bikeway is looking at how much blue there is. Shading the
    SPANS the stalls were counted out of put 3,152 ft of blue on a kerb that holds 990 ft of car -
    3.2x - because the spans are only the legal test, and the count is the legal test AND the
    width test AND the driveway mouths AND a walk in whole 22 ft steps.

    So the boxes stop where the last stall line is drawn, and every other foot is hatched with the
    reason it is not parking. Blue plus gold plus orchid plus the mouths is the whole kerb.

    `edge_at` is the line all four are measured against - `stall_spans` width-tests to it and
    every band is drawn only as deep as it leaves free, so a shape's depth on the page IS the
    spare the test found. The facility sheet passes `far_kerb_lane_edge(paint)`, the baseline and
    the no-bikeway sheet pass `nominal_lane_edge`, and nothing else about the drawing changes.
    """
    spans = stall_spans(corridor, facts, side, edge_at)
    footprints = stall_footprints(spans)
    marked = tuple((lo, hi) for lo, hi, _stalls in footprints)
    return (stall_bands(corridor, side, marked, limit_at=edge_at),
            tuple(stall_marks(corridor, side, spans)[0]),
            tuple((side, lo, hi, stalls) for lo, hi, stalls in footprints),
            hatch_bands(corridor, facts, side, marked, limit_at=edge_at))


def _band_across(corridor, side, lo_ft, hi_ft):
    """A full-depth window over one station range, for intersecting a marking out of the lane."""
    from src.geometry.model import Alignment, band_from_offsets

    if hi_ft - lo_ft <= 0:
        return None
    stations = np.linspace(lo_ft, hi_ft, max(int((hi_ft - lo_ft) / 2.0) + 2, 3))
    return band_from_offsets(Alignment(corridor.centerline), side, stations,
                             np.zeros(len(stations)), np.full(len(stations), 60.0))


def straighten(corridor, geometry):
    """One geometry's parts, each as (station, offset) in the corridor's frame.

    A LIST, because cutting the lane at every crossing and driveway mouth turns one polygon into
    a MultiPolygon - which is the whole point of the cut, and a drawing that took only the first
    part would show the lane stopping at its first break and never resuming.
    """
    if geometry is None or geometry.is_empty:
        return []
    parts = list(getattr(geometry, "geoms", [geometry]))
    out = []
    for part in parts:
        if part.is_empty:
            continue
        coords = (np.asarray(part.exterior.coords, dtype=float) if part.geom_type == "Polygon"
                  else np.asarray(part.coords, dtype=float))
        stations, offsets = station_offset_many(corridor.centerline, coords)
        out.append(np.column_stack([stations, offsets]))
    return out


def draw_panel(ax, corridor, paint, parking, openings, lo_ft, hi_ft, half_ft,
               marks=(), daylight=(), stall_labels=(), extras=None, hatch=(),
               travel_edge_ft=None):
    grid = np.arange(lo_ft, hi_ft, CORRIDOR_SAMPLE_FT)
    left = np.array([kerb_offset_ft(corridor, "left", float(s)) or np.nan for s in grid])
    right = np.array([-(kerb_offset_ft(corridor, "right", float(s)) or np.nan) for s in grid])

    # The carriageway, only where BOTH kerbs are traced - a surface drawn across an unsurveyed
    # stretch is the drawing claiming to know where the street ends.
    both = np.isfinite(left) & np.isfinite(right)
    ax.fill_between(grid, right, left, where=both, color=ASPHALT, linewidth=0, zorder=1)
    ax.plot(grid, np.where(np.isfinite(left), left, np.nan), color=KERB, lw=1.4, zorder=6)
    ax.plot(grid, np.where(np.isfinite(right), right, np.nan), color=KERB, lw=1.4, zorder=6)

    # THE TRAVEL WAY'S EDGES ON A SHEET WITH NO RUNS. Where a facility is drawn these come off the
    # runs it placed (below) and stop where the section stopped being tested. A calming places no
    # runs at all, and without this line its stalls and its hatch sit against a wide grey nothing
    # with no visible datum - when holding that line at target IS the whole design, and every foot
    # outside it is what the design gave away. Drawn only where both kerbs are traced, on the same
    # test the carriageway itself is drawn on.
    if travel_edge_ft is not None:
        for sign in (1.0, -1.0):
            ax.plot(grid, np.where(both, sign * travel_edge_ft, np.nan), color=TRAVEL_EDGE,
                    lw=0.9, linestyle=(0, (4, 3)), zorder=6)

    # Marked parking on the far kerb - legal AND wide enough AND long enough for a whole car.
    for lo, hi, band in parking:
        if hi < lo_ft or lo > hi_ft:
            continue
        for xy in straighten(corridor, band):
            ax.fill(xy[:, 0], xy[:, 1], color=PARKING_BLUE, alpha=0.45, linewidth=0, zorder=2)

    # EVERY OTHER FOOT OF THIS KERB, hatched - never left bare, which is what read as "maybe
    # parking" before this existed. Two DIFFERENT findings, two colours: gold is this DESIGN's own
    # section - too narrow once the facility holds its target lane, or too short for one whole car,
    # and a narrower facility would open it. Orchid is the STREET, independent of any design here -
    # R.S. 39:4-138's corner/stop-sign/hydrant setbacks or an OSM restriction; nothing this project
    # draws changes it. Not a driveway/side-street mouth, which is never hatched over.
    hatch_style = {"room": (HATCH_GOLD, HATCH_EDGE, "//"), "legal": (HATCH_LEGAL, HATCH_LEGAL_EDGE, "xx")}
    for lo, hi, band, reason in hatch:
        if hi < lo_ft or lo > hi_ft:
            continue
        face, edge, texture = hatch_style[reason]
        for xy in straighten(corridor, band):
            ax.fill(xy[:, 0], xy[:, 1], facecolor=face, edgecolor=edge, hatch=texture,
                   alpha=0.5, linewidth=0.6, zorder=2)

    # THE DAYLIGHTING: kerb the law keeps clear so a driver and a pedestrian can see each other.
    # Drawn under the stalls, because what a reader needs to see is that the clear zone is where
    # the stalls STOP - the treatment is the absence, and an absence has to be shown to be read.
    # A BAND ON THE KERB, not a rectangle at the kerb's mid-span offset: it is the same shape as
    # the stalls it interrupts, so the two read against each other where the kerb wanders.
    for start, end, band in daylight:
        if end < lo_ft or start > hi_ft:
            continue
        for xy in straighten(corridor, band):
            ax.fill(xy[:, 0], xy[:, 1], color=DAYLIGHT, alpha=0.35, linewidth=0, zorder=2)

    for line in marks:
        for xy in straighten(corridor, line):
            if lo_ft <= xy[:, 0].mean() <= hi_ft:
                ax.plot(xy[:, 0], xy[:, 1], color=PAINT_WHITE, lw=0.8, zorder=5)

    # EACH RUN LABELLED WITH ITS OWN COUNT, so the headline total can be found on the page rather
    # than taken on trust - and so THE LABELS SUM TO IT. That is why the test is the run's own
    # midpoint and not an overlap: a run straddling a panel edge is drawn on both panels, and
    # labelling it wherever it is visible printed its count twice and made the sheet contradict
    # its own title. Labelled once, on the panel holding its middle.
    for side, lo, hi, stalls in stall_labels:
        mid = (lo + hi) / 2
        if not lo_ft <= mid < hi_ft:
            continue
        sign = 1.0 if side == "left" else -1.0
        offset = kerb_offset_ft(corridor, side, mid)
        if offset is None:
            continue
        ax.text(mid, sign * (offset - PARKING_STALL_DEPTH_DEFAULT_FT / 2), str(stalls),
                color="#10314f", fontsize=5.5, fontweight="bold", ha="center", va="center",
                zorder=10)

    # WHERE THE TRAVEL WAY ENDS, on both sides - the line every "no room" hatch was measured
    # against, and the one thing the sheet did not show. Without it the far kerb is a wide grey
    # nothing and the hatch has no visible reason; with it a reader can see the few feet between
    # this line and the kerb for themselves. Drawn only over the runs, so where it stops is where
    # the section stopped being tested. matplotlib clips it to the panel, so it is not windowed.
    near_sign = 1.0 if paint.side == "left" else -1.0
    for stations, near, far in travel_way_edges(paint):
        for offsets, sign in ((near, near_sign), (far, -near_sign)):
            ax.plot(stations, sign * offsets, color=TRAVEL_EDGE, lw=0.9, linestyle=(0, (4, 3)),
                    zorder=6)

    for run in paint.runs:
        if run.end_ft < lo_ft or run.start_ft > hi_ft:
            continue
        for geometry, colour, z in ((run.buffer_zone, BUFFER_GREY, 2),
                                    (run.lane_surface, LANE_GREEN, 3)):
            for xy in straighten(corridor, geometry):
                ax.fill(xy[:, 0], xy[:, 1], color=colour, alpha=0.85, linewidth=0, zorder=z)
        for line in run.edge_lines:
            for xy in straighten(corridor, line):
                ax.plot(xy[:, 0], xy[:, 1], color=PAINT_WHITE, lw=1.0, zorder=4)
        posts = np.asarray(run.bollards, dtype=float)
        if len(posts):
            st, off = station_offset_many(corridor.centerline, posts)
            inside = (st >= lo_ft) & (st <= hi_ft)
            ax.plot(st[inside], off[inside], linestyle="none", marker="o", markersize=1.6,
                    color=POST, zorder=5)

    if extras:
        # Conspicuity green either side of every opening the lane crosses, laid UNDER the lane's
        # own colour so the two read as one surface with a hotter approach rather than two greens.
        for geometry in extras.get("green", ()):
            for xy in straighten(corridor, geometry):
                ax.fill(xy[:, 0], xy[:, 1], color=GREEN_HOT, alpha=0.9, linewidth=0, zorder=4)
        # The dotted YELLOW centreline - the one mark that carries across every crossbike, because
        # it is what tells a driver there are riders coming both ways.
        for line in extras.get("yellow", ()):
            for xy in straighten(corridor, line):
                if lo_ft <= xy[:, 0].mean() <= hi_ft:
                    ax.plot(xy[:, 0], xy[:, 1], color=YELLOW, lw=1.1, zorder=6)
        # BIKE LANE symbols, as a marker at each station the rule puts one.
        for station, offset in extras.get("symbols", ()):
            if lo_ft <= station <= hi_ft:
                ax.plot([station], [offset], marker="^", markersize=3.4, color=PAINT_WHITE,
                        markeredgecolor="#555555", markeredgewidth=0.3, zorder=7)

    # WHERE A VEHICLE CROSSES THE KERB. Drawn on the kerb it actually breaks, because that is the
    # asymmetry that matters here: a driveway on the north kerb interrupts the parking, and one on
    # the south interrupts the bikeway - the same feature with two different consequences.
    for side, start, end in openings:
        if end < lo_ft or start > hi_ft:
            continue
        sign = 1.0 if side == "left" else -1.0
        at = np.array([max(start, lo_ft), min(end, hi_ft)])
        offs = np.array([kerb_offset_ft(corridor, side, float(x)) or np.nan for x in at])
        if not np.isfinite(offs).all():
            continue
        ax.plot(at, sign * offs, color=OPENING, lw=2.6, solid_capstyle="butt", zorder=9)

    # Where the route breaks, and why - drawn, not left to the caption.
    # Labelled by WHAT the refusal actually was. A stretch the section could not fit and a
    # stretch nobody has traced are different findings - one is a fact about the street, the other
    # about the survey - and calling both "no room" tells a reader the street is too narrow where
    # it may not be.
    for start, end, why in ([(r.start_ft, r.end_ft,
                              "unsurveyed" if "untraced" in r.reason else "no room")
                             for r in paint.refusals]
                            + [(a, b, "junction" if why is JUNCTION_MOUTH else "unsurveyed")
                               for a, b, why in paint.untraced]):
        if end < lo_ft or start > hi_ft:
            continue
        colour = MOUTH_BLUE if why == "junction" else GAP_RED
        ax.add_patch(Rectangle((max(start, lo_ft), -half_ft), min(end, hi_ft) - max(start, lo_ft),
                               half_ft * 2, facecolor=colour, alpha=0.16, linewidth=0, zorder=7))
        ax.text((max(start, lo_ft) + min(end, hi_ft)) / 2, half_ft * 0.78, why, color=colour,
                fontsize=4.5, ha="center", va="top", rotation=-90, zorder=8)

    for junction in corridor.junctions:
        if lo_ft <= junction.node_ft <= hi_ft:
            ax.axvline(junction.node_ft, color="#3b6ea5", lw=0.8, linestyle="--", zorder=8)
            ax.text(junction.node_ft, -half_ft * 0.92, f" {junction.site}", color="#3b6ea5",
                    fontsize=5, ha="left", va="bottom", zorder=9)

    ax.set_xlim(lo_ft, hi_ft)
    ax.set_ylim(-half_ft, half_ft)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _legend(fig, facility: bool = True) -> None:
    """A real legend - colour swatches and line/marker samples with labels.

    Replaces a color key that used to live only in the title's prose ("blue = ...", "gold hatch =
    ..."), which could not show a hatch texture or a line style and grew a new clause every time a
    marking was added to the panel.

    `facility=False` DROPS THE ENTRIES THAT SHEET DOES NOT DRAW. A calming places no lane, no
    buffer, no posts and no bike symbols, and a legend naming them tells a reader to go looking
    for a bikeway on a drawing whose whole point is that there is not one - the daylighting
    swatch, which every one of these sheets does draw, was meanwhile only ever named in the
    title's prose.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lane = [
        Patch(facecolor=LANE_GREEN, alpha=0.85, label="protected bike lane"),
        Patch(facecolor=BUFFER_GREY, alpha=0.85, label="buffer / painted median"),
        Line2D([], [], color=PAINT_WHITE, markeredgecolor="#999999", lw=1.4, label="lane edge line"),
    ] if facility else [
        Patch(facecolor=DAYLIGHT, alpha=0.35, label="kerb kept clear for visibility"),
    ]
    handles = lane + [
        Line2D([], [], color=TRAVEL_EDGE, lw=0.9, linestyle=(0, (4, 3)),
               label="edge of the travel way - parking must clear it"),
        Patch(facecolor=PARKING_BLUE, alpha=0.45, label="marked parking stall"),
        Patch(facecolor=HATCH_GOLD, edgecolor=HATCH_EDGE, hatch="//", alpha=0.5,
              # ALSO THE SURPLUS BESIDE A MARKED BOX, which is why this does not say "too narrow":
              # a stall stops at a car's depth and the rest of a wide kerb is hatched, so yellow
              # covers both "no stall fits here" and "no FURTHER stall fits here".
              label="no parking - no stall fits this width or length"),
        Patch(facecolor=HATCH_LEGAL, edgecolor=HATCH_LEGAL_EDGE, hatch="xx", alpha=0.5,
              label="no parking - restricted by law or signage"),
        Line2D([], [], color=OPENING, lw=2.6, label="driveway or side-street crossing"),
    ] + ([
        Line2D([], [], color=POST, marker="o", linestyle="none", markersize=4, label="flex post"),
        Patch(facecolor=GREEN_HOT, alpha=0.9, label="conspicuity zone at a crossing"),
        Line2D([], [], color=YELLOW, lw=1.1, label="dotted centreline at a crossbike"),
        Line2D([], [], color=PAINT_WHITE, marker="^", markeredgecolor="#555555", linestyle="none",
               markersize=5, label="BIKE LANE symbol"),
    ] if facility else []) + [
        Patch(facecolor=MOUTH_BLUE, alpha=0.16, label="junction - no kerb to test"),
        Patch(facecolor=GAP_RED, alpha=0.16, label="unsurveyed - section untested"),
        Line2D([], [], color="#3b6ea5", lw=0.8, linestyle="--", label="junction node / site boundary"),
    ]
    # Clear of the note box in the bottom-left corner, which is wider on a sheet that has to
    # explain a refusal than on one that only has to justify a kerb.
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.38 if facility else 0.46, -0.10),
              ncol=3 if facility else 2, fontsize=5.5 if facility else 6, frameon=True,
              edgecolor="#3b6ea5", title="LEGEND", title_fontsize=6.5)


def _interruptions(out) -> int:
    """Everything that breaks a rider's run on this kerb: crossings cut, plus mouths crossed.

    THE ONE FIGURE THE CHOICE TURNS ON, and it is a sum because neither half decides it alone.
    The which-kerb verdict used to rank on `breaks`, which is 9 on BOTH kerbs of Broad St and
    will be on any street - the lane is cut at CROSSINGS, and a crossing is at a junction, which
    both kerbs share. So min() was returning whichever compass happened to be first in the dict
    and the sheet reported "fewer interruptions on the north kerb" off a tie. North does win, by
    28 to 35, but on the mouths - the only column that differs.
    """
    return out["breaks"] + out["openings_on_lane"]


def _verdict(outcomes) -> tuple[str, str, bool]:
    """(fewest interruptions, most parking kept, do they point the same way).

    Shared by the terminal and by the box on the sheet, so the two cannot disagree about which
    kerb this project's own measurements favour.
    """
    fewest = min(outcomes, key=lambda c: _interruptions(outcomes[c]))
    most_parking = max(outcomes, key=lambda c: outcomes[c]["kept"])
    return fewest, most_parking, fewest == most_parking


def _decision_table(fig, corridor, outcomes, drawn: str, ladder, baseline: dict) -> None:
    """The which-kerb comparison, ON THE DRAWING, as two options rather than as two rows.

    It was printed to a terminal, which is no use to anyone in a council chamber holding the
    sheet: the picture shows one kerb's proposal and the reason for choosing it lived somewhere
    the reader cannot see.

    WHAT MADE THE FIRST VERSION UNREADABLE was that it was laid out like a table of results when
    it is a comparison of two designs. Four figures in a row under bare headings - "placed",
    "breaks", "mouths", "parking kept, other kerb" - of which the first two came out IDENTICAL on
    both rows, so half the box read as a copy-paste error, the arrow marking the drawn option
    hung off the end of a parking figure it did not belong to, and the conclusion the terminal
    prints was missing entirely. Now: the options are columns, every row says in words what its
    number counts, the two rows that came out equal say why, and the box ends with the choice.

    Every row's parking figure belongs to the OTHER kerb - the lane and the parking are never on
    the same side - so the row says so and each cell names the kerb it is on.

    AND IT ENDS ON THE COST, NOT ON THE REMAINDER. "45 stalls kept" is the number a reader is
    being asked to weigh a bikeway against, and on its own it is unreadable: nobody in a council
    chamber knows whether the corridor holds 50 or 500. `baseline_stalls` is the same walk with no
    facility on either kerb, so the subtraction below is a real one - which is the whole reason it
    is done here rather than left to be done wrongly against the corridor report's 243.
    """
    order = list(outcomes)
    baseline_total = sum(baseline.values())
    fewest, most_parking, agree = _verdict(outcomes)
    col = "  ".join(f"{'the ' + c.upper():>10s}" for c in order)
    rows = [(f"bikeway placed, of {corridor.length_ft:,.0f} ft",
             [f"{outcomes[c]['paint'].placed_ft:,.0f} ft" for c in order], ""),
            ("crossings the lane is cut at",
             [f"{outcomes[c]['breaks']:d}" for c in order], ""),
            ("driveway + side-street mouths crossed",
             [f"{outcomes[c]['openings_on_lane']:d}" for c in order], ""),
            ("= interruptions a rider meets",
             [f"{_interruptions(outcomes[c]):d}" for c in order], f"fewest on the {fewest}"),
            ("parking stalls kept, on the OTHER kerb",
             [f"{outcomes[c]['kept']:d} {outcomes[c]['far_compass']}" for c in order],
             f"most on the {most_parking}"),
            (f"of {baseline_total} on the corridor with no lane at all",
             [f"{' + '.join(f'{n} {c2}' for c2, n in baseline.items())}" for _c in order],
             "same walk, both kerbs"),
            ("= parking stalls LOST to the bikeway",
             [f"{baseline_total - outcomes[c]['kept']:d}" for c in order],
             f"fewest on the {most_parking}")]
    # THE LABEL COLUMN IS AS WIDE AS ITS WIDEST LABEL, not a constant: a fixed field that one
    # label overruns shunts that row's figures out of their column, and two numbers a reader is
    # meant to compare stop lining up under each other - which is most of what made the first
    # version of this box unreadable.
    pad = max(len(label) for label, _cells, _note in rows)
    lines = ["WHICH KERB CARRIES THE TWO-WAY LANE - both options, one survey",
             f"{'':{pad + 2}s}  {'lane on':>10s}  {'lane on':>10s}",
             f"{'':{pad + 2}s}  {col}"]
    for label, cells, note in rows:
        marked = "  <- " + note if note else ""
        lines.append(f"  {label:<{pad}s}  " + "  ".join(f"{c:>10s}" for c in cells) + marked)
    # A ROW THAT CAME OUT EQUAL IS A MEASUREMENT, NOT A SLIP - said on the sheet, because two
    # identical figures side by side is exactly what a duplicated cell looks like.
    if len({round(o["paint"].placed_ft, 1) for o in outcomes.values()}) == 1:
        lines.append("")
        lines.append("  the top two rows are equal by measurement: every refusal here is")
        lines.append("  a survey gap or a junction mouth, and neither is a fact about a kerb")
    lines.append("")
    if agree:
        lines.append(f"  DRAWN: THE {drawn.upper()} KERB, which wins on both counts. "
                     f"CORRIDOR_SIDE is {ladder.side}.")
    else:
        other = next(c for c in order if c != drawn)
        lines.append(f"  DRAWN: THE {drawn.upper()} KERB. The counts DISAGREE, so this is a "
                     f"trade-off, not a")
        lines.append(f"  calculation - against the {other} it buys "
                     f"{abs(_interruptions(outcomes[drawn]) - _interruptions(outcomes[other]))} "
                     f"fewer interruptions for "
                     f"{abs(outcomes[drawn]['kept'] - outcomes[other]['kept'])} stalls.")
        lines.append(f"  CORRIDOR_SIDE is {ladder.side}.")
    fig.text(0.012, -0.004, "\n".join(lines), fontsize=6, family="monospace", va="top",
             ha="left", zorder=20,
             bbox={"facecolor": "white", "edgecolor": "#3b6ea5", "alpha": 0.92,
                   "boxstyle": "round,pad=0.5"})


def _no_facility_note(fig, corridor, outcomes, decision) -> None:
    """WHY THIS STREET IS CALMED AND NOT GIVEN A LANE, on the sheet rather than in a terminal.

    The facility sheet carries _decision_table for the choice it made; this is the same
    obligation for the choice NOT to place anything. A strip plan showing 11 ft lanes and hatch,
    with no statement of what was tried, reads as the project never having asked.
    """
    # THE SAME TWO FIGURES corridor_report.py prints, off the corridor's own methods rather than
    # off a second kerb scan here: a refusal is only honest beside the width it was made on and
    # beside how much of the street anybody has surveyed, and those two numbers may not differ
    # between the sheet and the report.
    narrowest = corridor.narrowest_width_ft()
    traced = corridor.both_traced_ft / corridor.length_ft
    lines = [f"WHY NO BIKE FACILITY ON {corridor.name.upper()} - measured, not assumed",
             f"tested: the protected two-way section declared for "
             f"{BROAD_ST_TWO_WAY_BIKEWAY.road}, on each kerb in turn"]
    for compass, out in outcomes.items():
        paint = out["paint"]
        no_room = sum(r.end_ft - r.start_ft for r in paint.refusals if "untraced" not in r.reason)
        lines.append(f"  lane on the {compass:<5s}  {paint.placed_ft:6,.0f} ft placed of "
                     f"{corridor.length_ft:,.0f} ft   refused over {no_room:5,.0f} ft of traced kerb")
    if narrowest is not None:
        lines.append(f"narrowest traced width {narrowest[0]:,.1f} ft at station "
                     f"{narrowest[1]:,.0f}, over the {traced:.0%} of the route where both kerbs "
                     f"are surveyed")
    lines.append(f"so the route decision is {type(decision).__name__}: hold the travel lanes at "
                 f"{TARGET_LANE_WIDTH_FT:.0f} ft and give the rest away")
    fig.text(0.012, -0.004, "\n".join(lines), fontsize=6, family="monospace", va="top",
             ha="left", zorder=20,
             bbox={"facecolor": "white", "edgecolor": GAP_RED, "alpha": 0.92,
                   "boxstyle": "round,pad=0.5"})


def _calming_strip(corridor, facts, args, outcomes, decision) -> int:
    """The strip plan for a street carrying no new facility: its cross-section, drawn to scale.

    ONE FUNCTION FOR TWO SHEETS BECAUSE THEY ARE ONE DESIGN - a route whose own decision is a
    CorridorCalming, and --no-bikeway on a route that has a facility, which asks what that street
    looks like if the facility is not built. Both hold every travel lane at TARGET_LANE_WIDTH_FT
    and put every recovered foot to work: marked parking where a whole car fits legally, hatch
    everywhere else with the reason it does not. Which of the two it is only changes the title,
    the filename, and whether the sheet has to explain a refusal.
    """
    quiet = io.StringIO()
    # A run-less paint object, purely to carry `untraced` into the drawing - the junction mouths
    # and unsurveyed stretches are facts about the SURVEY, so they belong on a sheet that places
    # nothing exactly as much as on one that places a lane. Its runs and refusals are dropped:
    # they are the tested facility's, and this sheet does not propose it.
    with contextlib.redirect_stdout(quiet):
        paint = paint_facility(corridor, dataclasses.replace(BROAD_ST_TWO_WAY_BIKEWAY,
                                                             side="south"), facts=facts)
    paint.runs, paint.refusals = [], []

    total, all_marks, daylight_spans, labels, hatch, parking = 0, [], [], [], [], []
    for compass in ("north", "south"):
        side = facility_side(corridor, compass)
        # PARKING OR HATCHING, NEVER NEITHER - parking.py's own rule, asked of BOTH kerbs here
        # rather than of the one kerb a facility leaves over. The kerb the boxes actually cover,
        # never the spans they were counted out of, which is a length no car can use: see
        # hatch_bands. On a no_parking street none of it is blue and the hatch is the whole
        # answer - which is the drawing saying, to scale, that the recovered width buys no stalls.
        bands, lines, footprints, side_hatch = kerbside_parking(corridor, facts, side,
                                                                nominal_lane_edge)
        all_marks += list(lines)
        labels += list(footprints)
        parking += bands
        hatch += side_hatch
        count = sum(n for _side, _lo, _hi, n in footprints)
        total += count
        # The kerb the boxes COVER, not the kerb they were counted out of - the tail of every
        # run is up to one car short and holds nothing.
        spans = stall_spans(corridor, facts, side, nominal_lane_edge)
        print(f"  {compass:5s} kerb: {count:3d} stalls DRAWN over "
              f"{sum(hi - lo for _s, lo, hi, _n in footprints):,.0f} ft of "
              f"{sum(hi - lo for lo, hi in spans):,.0f} ft available")
        # DRAWN ONLY WHERE THERE ARE STALLS FOR IT TO INTERRUPT. The daylighting is an absence,
        # and an absence is read against the thing it takes away: on a kerb the law closes end to
        # end - Princeton Ave is no_parking for its whole length - a clear-zone band covers every
        # foot the orchid hatch already covers, and the sheet states one fact twice in two
        # colours while showing nothing about visibility at the corners.
        if count:
            daylight_spans += stall_bands(corridor, side,
                                          [(z.start_ft, z.end_ft)
                                           for z in facts.by_side("no_parking", side)],
                                          limit_at=nominal_lane_edge)
    marks, daylight = tuple(all_marks), tuple(daylight_spans)
    openings = tuple((side, o.start_ft, o.end_ft) for side, o in facts.openings)

    calmed = decision is not None and not isinstance(decision, CorridorFacility)
    if calmed:
        print(f"  route calming: both travel lanes held at {TARGET_LANE_WIDTH_FT:.0f} ft, "
              f"{total} stalls marked on the corridor, every other foot hatched with its reason")
        headline = (f"{corridor.name} - ROUTE CALMING, this street's own decision: every travel "
                    f"lane held to {TARGET_LANE_WIDTH_FT:.0f} ft and every recovered foot marked "
                    f"parking or hatched against it")
        stem = "calming"
    else:
        print(f"  daylighting + crossing upgrades only: {total} stalls marked on the corridor, "
              f"counted by drawing them")
        headline = (f"{corridor.name} - daylighting and crossing upgrades ONLY, no bike facility")
        stem = "daylighting"

    half_ft = 38.0
    edges = np.linspace(0.0, corridor.length_ft, args.panels + 1)
    fig, axes = plt.subplots(args.panels, 1, figsize=(13, 2.0 * args.panels))
    for ax, lo, hi in zip(np.atleast_1d(axes), edges[:-1], edges[1:]):
        draw_panel(ax, corridor, paint, tuple(parking), openings, float(lo), float(hi), half_ft,
                   marks=marks, daylight=daylight, stall_labels=tuple(labels),
                   hatch=tuple(hatch), travel_edge_ft=TARGET_LANE_WIDTH_FT)
    axes[0].set_title(
        f"{headline}\n"
        f"{total} parking stalls marked on both kerbs, counted by drawing each one\n"
        f"straightened into the corridor's own frame - lengths and widths true, curvature "
        f"removed", fontsize=8)
    axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
    fig.tight_layout()
    if calmed:
        _no_facility_note(fig, corridor, outcomes, decision)
    _legend(fig, facility=False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip_{stem}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road", default="Broad Street")
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--out", default="output/corridor")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-bikeway", action="store_true",
                        help="draw the daylighting and crossing treatments only, with the parking "
                             "stalls actually marked and counted on both kerbs")
    parser.add_argument("--side", choices=("north", "south"), default=None,
                        help="which kerb carries the facility; default is the route's own "
                             "CORRIDOR_SIDE. Both are always MEASURED and compared.")
    args = parser.parse_args()

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        models = {site: load_intersection_model(site=site) for site in list_sites()
                  if site != "nj31_wdelaware"}
        corridors = corridors_from_models(models)
    matching = [c for c in corridors if args.road.lower() in c.name.lower()]
    if not matching:
        print(f"No corridor matching {args.road!r}. Found: "
              f"{', '.join(sorted(c.name for c in corridors))}")
        return 2
    corridor = centred_on_its_kerbs(matching[0])
    with contextlib.redirect_stdout(quiet):
        facts = corridor_facts(corridor, models)

    # WHAT IS PROPOSED HERE IS A PROPERTY OF THE STREET, NOT OF THIS SCRIPT. It used to import
    # BROAD_ST_TWO_WAY_BIKEWAY and draw it down whatever --road named, so Princeton Ave - whose
    # route decision is a calming - got a sheet headed "0 ft placed of 1,565 ft" for a design
    # nobody has proposed there, under a table asking which kerb should carry a lane that is not
    # going to exist. Looked up by the CORRIDOR'S name rather than by the argument, so a partial
    # `--road broad` still finds the decision the sites themselves apply.
    decision = route_decision_for(corridor.name)
    # The ladder the refusals below are measured with. This project declares exactly one protected
    # section, and "would THAT fit here" is the question a street without a facility has to answer
    # - so a calmed route borrows the sections to be REFUSED by them, never to be drawn with them.
    ladder = decision if isinstance(decision, CorridorFacility) else BROAD_ST_TWO_WAY_BIKEWAY
    calming = args.no_bikeway or not isinstance(decision, CorridorFacility)

    # BOTH KERBS ARE MEASURED, WHICHEVER ONE IS DRAWN. "Which side" is a route decision that has
    # to be made once for the whole borough, and bikeways.py:CORRIDOR_SIDE was settled on ONE
    # count - side streets cutting the kerb, 10 north against 7 south - before this project could
    # ask a corridor anything. It can now: the interruptions a rider actually meets include every
    # driveway mouth on their own kerb, and the parking cost is what the street can hold after the
    # travel lanes keep 11 ft. So the comparison is printed every run, and the drawn side is only
    # a choice about the picture.
    outcomes = {}
    for compass in ("north", "south"):
        facility = dataclasses.replace(ladder, side=compass)
        paint = paint_facility(corridor, facility, facts=facts)
        near = paint.side
        far = "left" if near == "right" else "right"
        far_compass = "south" if compass == "north" else "north"
        # Interruptions the RIDER meets: a mouth on their own kerb, plus every crossing. Counted
        # off paint.breaks so the number and the drawn gaps cannot disagree.
        outcomes[compass] = {
            "paint": paint, "far": far, "far_compass": far_compass,
            "breaks": len(paint.breaks),
            "openings_on_lane": sum(1 for side, opening in facts.openings if side == near),
            "kept": stall_marks(corridor, far,
                                stall_spans(corridor, facts, far,
                                            far_kerb_lane_edge(paint)))[1],
        }

    print(f"{corridor.name}: {corridor.length_ft:,.0f} ft, "
          f"{len(facts.marked_crossings)} surveyed crossings\n")
    if decision is None:
        print(f"  NOTE: this project has declared no route decision for {corridor.name}, so what "
              f"follows is the baseline - travel lanes at {TARGET_LANE_WIDTH_FT:.0f} ft and the "
              f"recovered width marked or hatched. Nothing here is a proposal.")
    if calming:
        return _calming_strip(corridor, facts, args, outcomes, decision)

    print("  WHICH KERB SHOULD CARRY THE TWO-WAY LANE - both measured, on the same OSM pull:")
    print("    Read the last column carefully: parking lands on the kerb the lane does NOT take,")
    print("    so it is the total left CORRIDOR-WIDE, not the parking on the lane's own kerb.")
    print(f"    {'lane on':>8s} {'placed':>10s} {'breaks in':>10s} {'mouths on':>10s} "
          f"{'a rider':>10s} {'parking':>8s} {'stalls left':>12s}")
    print(f"    {'':>8s} {'':>10s} {'the lane':>10s} {'the lane':>10s} "
          f"{'meets both':>10s} {'goes':>8s} {'there':>12s}")
    for compass, out in outcomes.items():
        paint = out["paint"]
        print(f"    {compass:>8s} {paint.placed_ft:7,.0f} ft {out['breaks']:10d} "
              f"{out['openings_on_lane']:10d} {_interruptions(out):10d} "
              f"{out['far_compass']:>8s} {out['kept']:9d}")
    # THE DENOMINATOR, PRINTED WITH THE OPTIONS AND NOT LEFT TO THE READER. "45 stalls left" is
    # not a cost until it is read against what this corridor holds with no lane on it, and the
    # only comparable figure is the same walk over both kerbs at the nominal lane edge. See
    # baseline_stalls for the two non-comparable numbers this replaces.
    baseline = baseline_stalls(corridor, facts)
    baseline_total = sum(baseline.values())
    print(f"\n  PARKING, LIKE FOR LIKE - one walk, both kerbs, width-tested against an "
          f"{TARGET_LANE_WIDTH_FT:.0f} ft travel lane:")
    print(f"    {'without the bikeway':<22s}{baseline_total:4d} stalls  "
          f"({', '.join(f'{n} on the {compass}' for compass, n in baseline.items())})")
    for compass, out in outcomes.items():
        lost = baseline_total - out["kept"]
        print(f"    {'lane on the ' + compass:<22s}{out['kept']:4d} stalls  "
              f"(all on the {out['far_compass']}) -> {lost} lost, {baseline[compass]} of them "
              f"the lane's own kerb and {lost - baseline[compass]} squeezed off the "
              f"{out['far_compass']}")

    fewer, more_parking, agree = _verdict(outcomes)
    # NAMED BY WHAT THE READER HAS TO DECIDE, because the first version of this table was
    # misread the obvious way: "parking kept" against the north row looks like the north kerb's
    # own parking, when it is the south kerb's - the lane and the parking are never on the same
    # side, so every row's parking figure belongs to the other one.
    if agree:
        print(f"\n    -> the {fewer} kerb wins on BOTH counts: fewer interruptions and more "
              f"parking kept. CORRIDOR_SIDE is currently {ladder.side}.")
    else:
        print(f"\n    -> THEY DISAGREE: fewer interruptions on the {fewer} kerb, more parking "
              f"kept on the {more_parking} kerb. This is a trade-off, not a calculation, and "
              f"CORRIDOR_SIDE is currently {ladder.side}.")

    drawn = args.side or ladder.side
    chosen = outcomes[drawn]
    paint = chosen["paint"]
    far_side, far_compass = chosen["far"], chosen["far_compass"]
    print(f"\n  DRAWN: the {drawn} kerb.")
    print(paint.summary(corridor.length_ft))
    # The stalls themselves, drawn as boxes over the kerb they occupy - a reader asked to weigh a
    # stall count against a bikeway cannot see that count anywhere in a shaded strip - and every
    # other foot hatched with the reason it holds no car: "parking or hatching, never neither"
    # (parking.py's own rule, asked of the whole corridor).
    parking, marks, stall_labels, hatch = kerbside_parking(corridor, facts, far_side,
                                                           far_kerb_lane_edge(paint))
    openings = tuple((side, opening.start_ft, opening.end_ft) for side, opening in facts.openings)

    # The three markings NACTO asks for besides the lane itself (STANDARDS.md, 2026-08-18).
    green = []
    for lo, hi in green_extension_spans(paint):
        for run in paint.runs:
            if run.lane_surface is None or hi < run.start_ft or lo > run.end_ft:
                continue
            window = _band_across(corridor, paint.side, max(lo, run.start_ft),
                                  min(hi, run.end_ft))
            if window is not None:
                piece = run.lane_surface.intersection(window)
                if not piece.is_empty:
                    green.append(piece)
    sign = 1.0 if paint.side == "left" else -1.0
    symbols = []
    for station in symbol_stations(paint):
        offset = kerb_offset_ft(corridor, paint.side, station)
        if offset is not None:
            symbols.append((station, sign * (offset - 5.0)))
    extras = {"green": tuple(green), "yellow": tuple(contraflow_centreline(corridor, paint)),
              "symbols": tuple(symbols)}
    print(f"  markings besides the lane: {len(extras['yellow'])} yellow centreline dashes, "
          f"{len(symbols)} BIKE LANE symbols, {len(green)} green conspicuity zones; the lane is "
          f"DOTTED across {len(paint.dotted)} openings and cut at {len(paint.breaks)} crossings")

    half_ft = 38.0
    edges = np.linspace(0.0, corridor.length_ft, args.panels + 1)
    fig, axes = plt.subplots(args.panels, 1, figsize=(13, 2.0 * args.panels))
    for ax, lo, hi in zip(np.atleast_1d(axes), edges[:-1], edges[1:]):
        draw_panel(ax, corridor, paint, parking, openings, float(lo), float(hi), half_ft,
                   marks=marks, stall_labels=stall_labels, extras=extras, hatch=hatch)
    axes[0].set_title(
        f"{corridor.name} - existing kerbs (surveyed) and the proposed two-way protected bikeway "
        # BY AREA, and the wording says so because the length claim was true while the picture
        # was not: bands were capped at a stall's depth, so 1,000 ft of the widest asphalt in the
        # borough was "marked or hatched" over 8 ft of a 27 ft kerb. See parking.allocate_kerbside.
        f"on the {paint.compass_side} kerb; every square foot of the {far_compass} kerb the "
        f"travel lane does not use is marked parking or hatched against it - see legend\n"
        f"{paint.placed_ft:,.0f} ft placed of {corridor.length_ft:,.0f} ft "
        f"({paint.placed_ft / corridor.length_ft:.0%}); straightened into the corridor's own frame "
        f"- lengths and widths true, curvature removed\n"
        f"parking, one walk over both kerbs, width-tested against an "
        f"{TARGET_LANE_WIDTH_FT:.0f} ft travel lane: {chosen['kept']} of the corridor's "
        f"{baseline_total} stalls kept, all on the {far_compass} kerb and drawn as boxes - "
        f"{baseline_total - chosen['kept']} lost, {baseline[drawn]} of them the bikeway's own "
        f"kerb; {paint.breaks and len(paint.breaks)} interruptions along the lane", fontsize=8)
    axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
    fig.tight_layout()
    _decision_table(fig, corridor, outcomes, drawn, ladder, baseline)
    _legend(fig)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip_{drawn}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
