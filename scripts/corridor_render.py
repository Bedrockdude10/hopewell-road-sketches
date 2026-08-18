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
                                         centred_on_its_kerbs, kerb_offset_ft, paint_facility)
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.network import corridors_from_models
from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY
from src.site import list_sites

ASPHALT = "#d9d9d9"
KERB = "#222222"
LANE_GREEN = "#57a773"
BUFFER_GREY = "#9a9a9a"
PAINT_WHITE = "#ffffff"
POST = "#e8663c"
GAP_RED = "#c1272d"
MOUTH_BLUE = "#3b6ea5"


def straighten(corridor, geometry):
    """One geometry's coordinates as (station, offset) in the corridor's frame."""
    if geometry is None or geometry.is_empty:
        return None
    coords = (np.asarray(geometry.exterior.coords, dtype=float)
              if geometry.geom_type == "Polygon" else np.asarray(geometry.coords, dtype=float))
    stations, offsets = station_offset_many(corridor.centerline, coords)
    return np.column_stack([stations, offsets])


def draw_panel(ax, corridor, paint, lo_ft, hi_ft, half_ft):
    grid = np.arange(lo_ft, hi_ft, CORRIDOR_SAMPLE_FT)
    left = np.array([kerb_offset_ft(corridor, "left", float(s)) or np.nan for s in grid])
    right = np.array([-(kerb_offset_ft(corridor, "right", float(s)) or np.nan) for s in grid])

    # The carriageway, only where BOTH kerbs are traced - a surface drawn across an unsurveyed
    # stretch is the drawing claiming to know where the street ends.
    both = np.isfinite(left) & np.isfinite(right)
    ax.fill_between(grid, right, left, where=both, color=ASPHALT, linewidth=0, zorder=1)
    ax.plot(grid, np.where(np.isfinite(left), left, np.nan), color=KERB, lw=1.4, zorder=6)
    ax.plot(grid, np.where(np.isfinite(right), right, np.nan), color=KERB, lw=1.4, zorder=6)

    for run in paint.runs:
        if run.end_ft < lo_ft or run.start_ft > hi_ft:
            continue
        for geometry, colour, z in ((run.buffer_zone, BUFFER_GREY, 2),
                                    (run.lane_surface, LANE_GREEN, 3)):
            xy = straighten(corridor, geometry)
            if xy is not None:
                ax.fill(xy[:, 0], xy[:, 1], color=colour, alpha=0.85, linewidth=0, zorder=z)
        for line in run.edge_lines:
            xy = straighten(corridor, line)
            if xy is not None:
                ax.plot(xy[:, 0], xy[:, 1], color=PAINT_WHITE, lw=1.0, zorder=4)
        posts = np.asarray(run.bollards, dtype=float)
        if len(posts):
            st, off = station_offset_many(corridor.centerline, posts)
            inside = (st >= lo_ft) & (st <= hi_ft)
            ax.plot(st[inside], off[inside], linestyle="none", marker="o", markersize=1.6,
                    color=POST, zorder=5)

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
    paint = paint_facility(corridor, BROAD_ST_TWO_WAY_BIKEWAY)
    print(paint.summary(corridor.length_ft))

    half_ft = 38.0
    edges = np.linspace(0.0, corridor.length_ft, args.panels + 1)
    fig, axes = plt.subplots(args.panels, 1, figsize=(13, 2.0 * args.panels))
    for ax, lo, hi in zip(np.atleast_1d(axes), edges[:-1], edges[1:]):
        draw_panel(ax, corridor, paint, float(lo), float(hi), half_ft)
    axes[0].set_title(
        f"{corridor.name} - existing kerbs (surveyed) and the proposed two-way protected bikeway "
        f"on the {paint.compass_side} kerb\n"
        f"{paint.placed_ft:,.0f} ft placed of {corridor.length_ft:,.0f} ft "
        f"({paint.placed_ft / corridor.length_ft:.0%}); straightened into the corridor's own frame "
        f"- lengths and widths true, curvature removed", fontsize=8)
    axes[-1].set_xlabel("station along the corridor (ft)", fontsize=7)
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.road.lower().replace(' ', '_')}_strip.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
