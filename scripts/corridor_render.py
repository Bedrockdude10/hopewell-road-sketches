"""A STRIP PLAN of one corridor: the whole street, straightened, on stacked panels.

    .venv/bin/python scripts/corridor_render.py --road "Broad Street"

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
                                         centred_on_its_kerbs, far_kerb_lane_edge,
                                         hatch_bands, kerb_offset_ft, paint_facility,
                                         stall_bands, contraflow_centreline,
                                         green_extension_spans, stall_footprints, stall_marks,
                                         stall_room_spans, symbol_stations, travel_way_edges)
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.network import (_complement_spans, _merged_spans, corridor_facts,
                                  corridors_from_models)
from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY
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


def _far_kerb_stall_spans(corridor, facts, paint, far_side: str):
    """Where a stall may actually be marked on the FAR kerb once this facility is placed.

    Legal room AND street room, because a length the statute permits and four feet wide holds no
    car - the distinction that took the honest figure from 108 to 32 on the south-kerb option. The
    far kerb is measured against the divider the section pushes toward it, per run, so a
    constrained stretch and a standard one are not given the same allowance.

    Returns the SPANS, not a count, so the number in the title and the boxes on the page come out
    of one call to `stall_marks` rather than out of two counters that can drift apart.
    """
    mouths = _merged_spans([(o.start_ft, o.end_ft)
                            for side, o in facts.openings if side == far_side])
    clear = _complement_spans(mouths, 0.0, corridor.length_ft)
    room = stall_room_spans(corridor, far_side, far_kerb_lane_edge(paint))
    return _intersect(_intersect(facts.by_side("parkable", far_side), clear), room)


def far_kerb_parking(corridor, facts, paint, far_side: str):
    """(bands, marks, labels, hatch) - every foot of the far kerb, drawn once and to scale.

    ONE FOOTPRINT FEEDS ALL FOUR, because the sheet is read as an area and not as a caption: a
    reader weighing 45 stalls against a bikeway is looking at how much blue there is. Shading the
    SPANS the stalls were counted out of put 3,152 ft of blue on a kerb that holds 990 ft of car -
    3.2x - because the spans are only the legal test, and the count is the legal test AND the
    width test AND the driveway mouths AND a walk in whole 22 ft steps.

    So the boxes stop where the last stall line is drawn, and every other foot is hatched with the
    reason it is not parking. Blue plus gold plus orchid plus the mouths is the whole kerb.
    """
    # The one line all four are measured against: `_far_kerb_stall_spans` width-tests against it,
    # and every band is drawn only as deep as it leaves free, so a shape's depth on the page IS
    # the spare the test found.
    edge_at = far_kerb_lane_edge(paint)
    spans = _far_kerb_stall_spans(corridor, facts, paint, far_side)
    footprints = stall_footprints(spans)
    marked = tuple((lo, hi) for lo, hi, _stalls in footprints)
    return (stall_bands(corridor, far_side, marked, limit_at=edge_at),
            tuple(stall_marks(corridor, far_side, spans)[0]),
            tuple((far_side, lo, hi, stalls) for lo, hi, stalls in footprints),
            hatch_bands(corridor, facts, far_side, marked, limit_at=edge_at))


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
               marks=(), daylight=(), stall_labels=(), extras=None, hatch=()):
    grid = np.arange(lo_ft, hi_ft, CORRIDOR_SAMPLE_FT)
    left = np.array([kerb_offset_ft(corridor, "left", float(s)) or np.nan for s in grid])
    right = np.array([-(kerb_offset_ft(corridor, "right", float(s)) or np.nan) for s in grid])

    # The carriageway, only where BOTH kerbs are traced - a surface drawn across an unsurveyed
    # stretch is the drawing claiming to know where the street ends.
    both = np.isfinite(left) & np.isfinite(right)
    ax.fill_between(grid, right, left, where=both, color=ASPHALT, linewidth=0, zorder=1)
    ax.plot(grid, np.where(np.isfinite(left), left, np.nan), color=KERB, lw=1.4, zorder=6)
    ax.plot(grid, np.where(np.isfinite(right), right, np.nan), color=KERB, lw=1.4, zorder=6)

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


def _legend(fig) -> None:
    """A real legend - colour swatches and line/marker samples with labels.

    Replaces a color key that used to live only in the title's prose ("blue = ...", "gold hatch =
    ..."), which could not show a hatch texture or a line style and grew a new clause every time a
    marking was added to the panel.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=LANE_GREEN, alpha=0.85, label="protected bike lane"),
        Patch(facecolor=BUFFER_GREY, alpha=0.85, label="buffer / painted median"),
        Line2D([], [], color=PAINT_WHITE, markeredgecolor="#999999", lw=1.4, label="lane edge line"),
        Line2D([], [], color=TRAVEL_EDGE, lw=0.9, linestyle=(0, (4, 3)),
               label="edge of the travel way - parking must clear it"),
        Line2D([], [], color=POST, marker="o", linestyle="none", markersize=4, label="flex post"),
        Patch(facecolor=PARKING_BLUE, alpha=0.45, label="marked parking stall"),
        Patch(facecolor=HATCH_GOLD, edgecolor=HATCH_EDGE, hatch="//", alpha=0.5,
              label="no parking - too narrow / too short for this design"),
        Patch(facecolor=HATCH_LEGAL, edgecolor=HATCH_LEGAL_EDGE, hatch="xx", alpha=0.5,
              label="no parking - restricted by law or signage"),
        Line2D([], [], color=OPENING, lw=2.6, label="driveway or side-street crossing"),
        Patch(facecolor=GREEN_HOT, alpha=0.9, label="conspicuity zone at a crossing"),
        Line2D([], [], color=YELLOW, lw=1.1, label="dotted centreline at a crossbike"),
        Line2D([], [], color=PAINT_WHITE, marker="^", markeredgecolor="#555555", linestyle="none",
               markersize=5, label="BIKE LANE symbol"),
        Patch(facecolor=MOUTH_BLUE, alpha=0.16, label="junction - no kerb to test"),
        Patch(facecolor=GAP_RED, alpha=0.16, label="unsurveyed - section untested"),
        Line2D([], [], color="#3b6ea5", lw=0.8, linestyle="--", label="junction node / site boundary"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.30, -0.10), ncol=3,
              fontsize=6, frameon=True, edgecolor="#3b6ea5", title="LEGEND", title_fontsize=6.5)


def _decision_table(fig, outcomes, drawn: str) -> None:
    """The which-kerb comparison, ON THE DRAWING.

    It was printed to a terminal, which is no use to anyone in a council chamber holding the
    sheet: the picture shows one kerb's proposal and the reason for choosing it lived somewhere
    the reader cannot see. Every row's parking figure belongs to the OTHER kerb - the lane and
    the parking are never on the same side - so the column says so in words.
    """
    lines = ["WHICH KERB CARRIES THE LANE - both measured, same survey",
             f"{'lane on':>9s}  {'placed':>9s}  {'breaks':>7s}  {'mouths':>7s}  "
             f"{'parking kept, other kerb':>25s}"]
    for compass, out in outcomes.items():
        mark = "<- drawn" if compass == drawn else ""
        lines.append(f"{compass:>9s}  {out['paint'].placed_ft:6,.0f} ft  "
                     f"{out['breaks']:7d}  {out['openings_on_lane']:7d}  "
                     f"{out['kept']:6d} on the {out['far_compass']:<5s} {mark}")
    fig.text(0.012, -0.004, "\n".join(lines), fontsize=6, family="monospace", va="top",
             ha="left", zorder=20,
             bbox={"facecolor": "white", "edgecolor": "#3b6ea5", "alpha": 0.92,
                   "boxstyle": "round,pad=0.5"})


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

    # BOTH KERBS ARE MEASURED, WHICHEVER ONE IS DRAWN. "Which side" is a route decision that has
    # to be made once for the whole borough, and bikeways.py:CORRIDOR_SIDE was settled on ONE
    # count - side streets cutting the kerb, 10 north against 7 south - before this project could
    # ask a corridor anything. It can now: the interruptions a rider actually meets include every
    # driveway mouth on their own kerb, and the parking cost is what the street can hold after the
    # travel lanes keep 11 ft. So the comparison is printed every run, and the drawn side is only
    # a choice about the picture.
    outcomes = {}
    for compass in ("north", "south"):
        facility = dataclasses.replace(BROAD_ST_TWO_WAY_BIKEWAY, side=compass)
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
                                _far_kerb_stall_spans(corridor, facts, paint, far))[1],
        }

    print(f"{corridor.name}: {corridor.length_ft:,.0f} ft, "
          f"{len(facts.marked_crossings)} surveyed crossings\n")
    print("  WHICH KERB SHOULD CARRY THE TWO-WAY LANE - both measured, on the same OSM pull:")
    print("    Read the last column carefully: parking lands on the kerb the lane does NOT take,")
    print("    so it is the total left CORRIDOR-WIDE, not the parking on the lane's own kerb.")
    print(f"    {'lane on':>8s} {'placed':>10s} {'breaks in':>10s} {'mouths on':>10s} "
          f"{'parking':>8s} {'stalls left':>12s}")
    print(f"    {'':>8s} {'':>10s} {'the lane':>10s} {'the lane':>10s} "
          f"{'goes':>8s} {'there':>12s}")
    for compass, out in outcomes.items():
        paint = out["paint"]
        print(f"    {compass:>8s} {paint.placed_ft:7,.0f} ft {out['breaks']:10d} "
              f"{out['openings_on_lane']:10d} {out['far_compass']:>8s} {out['kept']:9d}")
    fewer = min(outcomes, key=lambda c: outcomes[c]["breaks"])
    more_parking = max(outcomes, key=lambda c: outcomes[c]["kept"])
    # NAMED BY WHAT THE READER HAS TO DECIDE, because the first version of this table was
    # misread the obvious way: "parking kept" against the north row looks like the north kerb's
    # own parking, when it is the south kerb's - the lane and the parking are never on the same
    # side, so every row's parking figure belongs to the other one.
    if fewer == more_parking:
        print(f"\n    -> the {fewer} kerb wins on BOTH counts: fewer interruptions and more "
              f"parking kept. CORRIDOR_SIDE is currently {BROAD_ST_TWO_WAY_BIKEWAY.side}.")
    else:
        print(f"\n    -> THEY DISAGREE: fewer interruptions on the {fewer} kerb, more parking "
              f"kept on the {more_parking} kerb. This is a trade-off, not a calculation, and "
              f"CORRIDOR_SIDE is currently {BROAD_ST_TWO_WAY_BIKEWAY.side}.")

    marks, daylight = (), ()
    if args.no_bikeway:
        # THE OTHER PROPOSAL: no facility at all, just the daylighting and the crossing upgrades,
        # with the parking MARKED so the count is a count of boxes rather than a length divided by
        # 22 ft. Both kerbs are measured against a travel lane holding 11 ft, which is what the
        # default treatment leaves them.
        empty = dataclasses.replace(BROAD_ST_TWO_WAY_BIKEWAY, side="south")
        paint = paint_facility(corridor, empty, facts=facts)
        paint.runs, paint.refusals = [], []
        total, all_marks, daylight_spans, labels = 0, [], [], []
        for side, compass in (("left", "north"), ("right", "south")):
            mouths = _merged_spans([(o.start_ft, o.end_ft)
                                   for s2, o in facts.openings if s2 == side])
            clear = _complement_spans(mouths, 0.0, corridor.length_ft)
            # An 11 ft travel lane is what the default treatment leaves this kerb, so it is the
            # line the daylighting sheet both TESTS against and draws its bands to.
            def nominal_edge(_station_ft):
                return 11.0

            room = stall_room_spans(corridor, side, nominal_edge)
            spans = _intersect(_intersect(facts.by_side("parkable", side), clear), room)
            lines, count = stall_marks(corridor, side, spans)
            all_marks += lines
            footprints = stall_footprints(spans)
            labels += [(side, lo, hi, n) for lo, hi, n in footprints]
            total += count
            # The kerb the boxes COVER, not the kerb they were counted out of - the tail of every
            # run is up to one car short and holds nothing.
            print(f"  {compass:5s} kerb: {count:3d} stalls DRAWN over "
                  f"{sum(hi - lo for lo, hi, _n in footprints):,.0f} ft of "
                  f"{sum(hi - lo for lo, hi in spans):,.0f} ft available")
            daylight_spans += stall_bands(corridor, side,
                                          [(z.start_ft, z.end_ft)
                                           for z in facts.by_side("no_parking", side)],
                                          limit_at=nominal_edge)
        print(f"  daylighting + crossing upgrades only: {total} stalls marked on the corridor, "
              f"counted by drawing them")
        marks, daylight = tuple(all_marks), tuple(daylight_spans)
        parking = ()
        openings = tuple((side, o.start_ft, o.end_ft) for side, o in facts.openings)
        drawn = "none"
        half_ft = 38.0
        edges = np.linspace(0.0, corridor.length_ft, args.panels + 1)
        fig, axes = plt.subplots(args.panels, 1, figsize=(13, 2.0 * args.panels))
        for ax, lo, hi in zip(np.atleast_1d(axes), edges[:-1], edges[1:]):
            draw_panel(ax, corridor, paint, parking, openings, float(lo), float(hi), half_ft,
                       marks=marks, daylight=daylight, stall_labels=tuple(labels))
        axes[0].set_title(
            f"{corridor.name} - daylighting and crossing upgrades ONLY, no bike facility\n"
            f"{total} parking stalls marked on both kerbs, counted by drawing each one; "
            f"yellow = kerb R.S. 39:4-138 keeps clear for visibility\n"
            f"straightened into the corridor's own frame - lengths and widths true, curvature "
            f"removed", fontsize=8)
        axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
        fig.tight_layout()
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip_daylighting.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"\nwrote {path}")
        return 0

    drawn = args.side or BROAD_ST_TWO_WAY_BIKEWAY.side
    chosen = outcomes[drawn]
    paint = chosen["paint"]
    far_side, far_compass = chosen["far"], chosen["far_compass"]
    print(f"\n  DRAWN: the {drawn} kerb.")
    print(paint.summary(corridor.length_ft))
    # The stalls themselves, drawn as boxes over the kerb they occupy - a reader asked to weigh a
    # stall count against a bikeway cannot see that count anywhere in a shaded strip - and every
    # other foot hatched with the reason it holds no car: "parking or hatching, never neither"
    # (parking.py's own rule, asked of the whole corridor).
    parking, marks, stall_labels, hatch = far_kerb_parking(corridor, facts, paint, far_side)
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
        f"on the {paint.compass_side} kerb; every foot of the {far_compass} kerb is marked parking "
        f"or hatched against it - see legend\n"
        f"{paint.placed_ft:,.0f} ft placed of {corridor.length_ft:,.0f} ft "
        f"({paint.placed_ft / corridor.length_ft:.0%}); straightened into the corridor's own frame "
        f"- lengths and widths true, curvature removed\n"
        f"parking, width-tested against an 11 ft travel lane: "
        f"{chosen['kept']} stalls kept on the {far_compass} kerb, drawn as boxes; "
        f"{paint.breaks and len(paint.breaks)} interruptions along the lane", fontsize=8)
    axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
    fig.tight_layout()
    _decision_table(fig, outcomes, drawn)
    _legend(fig)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip_{drawn}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
