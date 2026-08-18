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
                                         kerb_offset_ft, paint_facility, parking_bands,
                                         stall_marks, stall_room_spans)
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.network import (_complement_spans, _merged_spans, corridor_facts,
                                  corridors_from_models, marked_parking_capacity)
from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY
from src.site import list_sites

ASPHALT = "#d9d9d9"
KERB = "#222222"
LANE_GREEN = "#57a773"
BUFFER_GREY = "#9a9a9a"
PAINT_WHITE = "#ffffff"
POST = "#e8663c"
PARKING_BLUE = "#4b7fb5"
OPENING = "#8a5a1f"
GAP_RED = "#c1272d"
DAYLIGHT = "#e8c33c"
MOUTH_BLUE = "#3b6ea5"


def _intersect(a, b):
    """The spans in both - a stall must be legal AND have room, not one or the other."""
    out = []
    for lo_a, hi_a in a:
        for lo_b, hi_b in b:
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if hi > lo:
                out.append((lo, hi))
    return tuple(sorted(out))


def _stalls_kept(corridor, facts, paint, far_side: str) -> int:
    """Width-tested stalls left on the FAR kerb once this facility is placed on the near one.

    Legal room AND street room, because a length the statute permits and four feet wide holds no
    car - the distinction that took the honest figure from 108 to 32 on the south-kerb option. The
    far kerb is measured against the divider the section pushes toward it, per run, so a
    constrained stretch and a standard one are not given the same allowance.
    """
    mouths = _merged_spans([(o.start_ft, o.end_ft)
                            for side, o in facts.openings if side == far_side])
    clear = _complement_spans(mouths, 0.0, corridor.length_ft)
    room = stall_room_spans(corridor, far_side, far_kerb_lane_edge(paint))
    count, _over_ft = marked_parking_capacity(corridor, facts, far_side,
                                             within=_intersect(clear, room))
    return count


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
               marks=(), daylight=()):
    grid = np.arange(lo_ft, hi_ft, CORRIDOR_SAMPLE_FT)
    left = np.array([kerb_offset_ft(corridor, "left", float(s)) or np.nan for s in grid])
    right = np.array([-(kerb_offset_ft(corridor, "right", float(s)) or np.nan) for s in grid])

    # The carriageway, only where BOTH kerbs are traced - a surface drawn across an unsurveyed
    # stretch is the drawing claiming to know where the street ends.
    both = np.isfinite(left) & np.isfinite(right)
    ax.fill_between(grid, right, left, where=both, color=ASPHALT, linewidth=0, zorder=1)
    ax.plot(grid, np.where(np.isfinite(left), left, np.nan), color=KERB, lw=1.4, zorder=6)
    ax.plot(grid, np.where(np.isfinite(right), right, np.nan), color=KERB, lw=1.4, zorder=6)

    # Where the law leaves room for a stall on the far kerb. Drawn as ROOM, not as a marked lane:
    # every one of these is a length R.S. 39:4-138 does not forbid, which is a different claim
    # from a design deciding to paint it.
    for lo, hi, band in parking:
        if hi < lo_ft or lo > hi_ft:
            continue
        for xy in straighten(corridor, band):
            ax.fill(xy[:, 0], xy[:, 1], color=PARKING_BLUE, alpha=0.45, linewidth=0, zorder=2)

    # THE DAYLIGHTING: kerb the law keeps clear so a driver and a pedestrian can see each other.
    # Drawn under the stalls, because what a reader needs to see is that the clear zone is where
    # the stalls STOP - the treatment is the absence, and an absence has to be shown to be read.
    for side, start, end in daylight:
        if end < lo_ft or start > hi_ft:
            continue
        sign = 1.0 if side == "left" else -1.0
        offs = kerb_offset_ft(corridor, side, (max(start, lo_ft) + min(end, hi_ft)) / 2)
        if offs is None:
            continue
        ax.add_patch(Rectangle((max(start, lo_ft), sign * offs - (8.0 if sign > 0 else 0.0)),
                               min(end, hi_ft) - max(start, lo_ft), 8.0,
                               facecolor=DAYLIGHT, alpha=0.35, linewidth=0, zorder=2))

    for line in marks:
        for xy in straighten(corridor, line):
            if lo_ft <= xy[:, 0].mean() <= hi_ft:
                ax.plot(xy[:, 0], xy[:, 1], color=PAINT_WHITE, lw=0.8, zorder=5)

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
            "kept": _stalls_kept(corridor, facts, paint, far),
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
        total, all_marks, daylight_spans = 0, [], []
        for side, compass in (("left", "north"), ("right", "south")):
            mouths = _merged_spans([(o.start_ft, o.end_ft)
                                   for s2, o in facts.openings if s2 == side])
            clear = _complement_spans(mouths, 0.0, corridor.length_ft)
            room = stall_room_spans(corridor, side, lambda _s: 11.0)
            spans = _intersect(_intersect(facts.by_side("parkable", side), clear), room)
            lines, count = stall_marks(corridor, side, spans)
            all_marks += lines
            total += count
            print(f"  {compass:5s} kerb: {count:3d} stalls DRAWN over "
                  f"{sum(hi - lo for lo, hi in spans):,.0f} ft")
            daylight_spans += [(side, z.start_ft, z.end_ft)
                               for z in facts.by_side("no_parking", side)]
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
                       marks=marks, daylight=daylight)
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
    parking = parking_bands(corridor, facts, far_side)
    openings = tuple((side, opening.start_ft, opening.end_ft) for side, opening in facts.openings)

    half_ft = 38.0
    edges = np.linspace(0.0, corridor.length_ft, args.panels + 1)
    fig, axes = plt.subplots(args.panels, 1, figsize=(13, 2.0 * args.panels))
    for ax, lo, hi in zip(np.atleast_1d(axes), edges[:-1], edges[1:]):
        draw_panel(ax, corridor, paint, parking, openings, float(lo), float(hi), half_ft)
    axes[0].set_title(
        f"{corridor.name} - existing kerbs (surveyed) and the proposed two-way protected bikeway "
        f"on the {paint.compass_side} kerb; blue = where the law leaves parking room on the "
        f"{far_compass} kerb, brown = a driveway or side road crossing it\n"
        f"{paint.placed_ft:,.0f} ft placed of {corridor.length_ft:,.0f} ft "
        f"({paint.placed_ft / corridor.length_ft:.0%}); straightened into the corridor's own frame "
        f"- lengths and widths true, curvature removed\n"
        f"parking, width-tested against an 11 ft travel lane: "
        f"{chosen['kept']} stalls kept on the {far_compass} kerb; "
        f"{paint.breaks and len(paint.breaks)} interruptions along the lane", fontsize=8)
    axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip_{drawn}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
