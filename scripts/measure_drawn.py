"""What was actually DRAWN, stationed against each leg's centreline.

    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --scenario build_two_way_bike_lane
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --leg north --kind bike_lane
    .venv/bin/python scripts/measure_drawn.py wbroad_louellen --frame-scale 2.5
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --scenario X --all

--all adds the other four questions SKILLS.md 0a says answer most complaints, so that answering
one never means writing a throwaway script or cropping a render:

    --section     what each treatment THINKS it placed, beside the room the traced kerb gives
    --limiters    all four things that decide where kerbside paint starts, not the first one
    --gaps        kerb offset minus outermost drawn offset, station by station
    --continuity  whether a facility is one piece, and how wide the holes are

Two of those exist to print a number that CANNOT BE RECONSTRUCTED from the constants you think
went into it. --section reads offsets_from_centerline_ft off the resolved treatment, because a
two-way section is measured from a centreline shifted toward the far kerb and rebuilding it from
TARGET_LANE_WIDTH_FT gives an answer that is wrong and confident. It also flags the case where
demand and room agree to the last decimal: that is one figure, not two - the section was sized to
fill the room - so a comparison between them can never fail and proves nothing.

Every serious bug in this repo has been two derivations of one number agreeing with each other
and not with the picture: "every travel lane is exactly 11.00 ft" computed from the section,
while the drawn centreline sat 2.84 ft away. So the rule is measure the drawn output, not the
arithmetic that was supposed to produce it - and the rule got skipped because invoking it meant
recalling how to build a scenario and which frame to project into. This is that, as one command.

It reads the same PaintPiece list the 3D export serialises, and reports each piece as
(station, offset) in its leg's frame via station_offset_many: station runs out along the leg
from the junction centre, offset is signed across it. Feet throughout.

A piece with no leg (the corner treatments, which belong to neither of the two legs they sit
between) is reported against the junction centre instead, as a plain distance.
"""
import argparse
import contextlib
import io
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from shapely.ops import unary_union

from scripts.build_all import scenarios_for
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.model.corners import corner_tangent_station_ft
from src.geometry.model.leg_frame import (curb_offsets_at_stations, curb_station_span,
                                          leg_clearance_ft, narrowest_half_width_ft)
from src.geometry.paint import junction_mouths_ft
from src.geometry.treatments.base import kerbside_allowance_ft
from src.render.crosswalks import crosswalk_reach_on_leg_side_ft
from src.geometry.treatments import DesignState
from src.render.export import (BUILDING_CONTEXT_RADIUS_M, KERB_RADIUS_M,
                               TRAFFIC_CONTROL_RADIUS_M)
from src.render.frame import FRAME_SCALE_ENV
from src.render.props import build_props
from src.render.scene import SceneGeometry
from src.site import list_sites, load_site_scenarios, run_scenario
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_street_furniture,
                                     fetch_traffic_control)


class Built(NamedTuple):
    """Everything the render was drawn from, so a question can be asked of any of it.

    The paint alone answers "where is it"; the state, the scene and the crossings are what
    answer "why there", and resolving them a second time here is how two derivations of one
    number get out of step. The scene in particular is not a convenience: it holds the RESOLVED
    crosswalk_bands, and junction_mouths_ft without them falls back to the corner return on
    every leg and reports that fallback as though it were the mouth.
    """
    model: object
    state: object
    scene: object
    paint: list
    crossings: object


def build(site: str, scenario: str | None) -> Built:
    """The paint one scenario implies, plus the model it was resolved against.

    Same call order as src/render/export.py:export_scenario - the props go in and come back
    because a bike lane's bollards are placed by the paint builder and a hydrant lengthens a
    daylight zone. Building the paint any other way would measure something the render is not.
    """
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        model = load_intersection_model(site=site)
        state = DesignState.from_model(model)
        if scenario:
            state = run_scenario(getattr(load_site_scenarios(site), scenario), state, model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        scene = SceneGeometry.resolve(model, state, crossings)
        props = build_props(model, state, scene.crosswalk_offsets, model.center_ft,
                            fetch_traffic_control(model.center_wgs84, radius_m=TRAFFIC_CONTROL_RADIUS_M),
                            fetch_street_furniture(model.center_wgs84,
                                                   radius_m=BUILDING_CONTEXT_RADIUS_M),
                            crossings, fetch_kerbs(model.center_wgs84, radius_m=KERB_RADIUS_M))
        paint, _props = scene.build_paint_and_posts(props)
    return Built(model, state, scene, paint, crossings)


def piece_coords(geom) -> np.ndarray:
    """Every vertex of a drawn piece, whatever shapely type it arrived as.

    A MultiPolygon has no .exterior, and a marking becomes one the moment it is clipped into
    two - so reading .exterior directly works right up until the case worth measuring.
    """
    parts = getattr(geom, "geoms", None) or [geom]
    return np.concatenate([
        np.asarray(g.exterior.coords if g.geom_type == "Polygon" else g.coords, dtype=float)
        for g in parts])


def drawn_on(built: Built, leg_name: str, side: str,
             kind_filter: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(station, |offset|) of every vertex drawn on one leg-side, in that leg's own frame."""
    leg = built.model.legs[leg_name]
    stations, offsets = [], []
    for piece in built.paint:
        if piece.leg != leg_name or piece.side != side:
            continue
        if kind_filter and kind_filter not in piece.kind.name:
            continue
        station, offset = station_offset_many(leg.centerline, piece_coords(piece.geometry))
        stations.append(station)
        offsets.append(np.abs(offset))
    if not stations:
        return None
    return np.concatenate(stations), np.concatenate(offsets)


def target_leg_sides(target) -> list[tuple[str, str | None]]:
    """The (leg, side) keys a treatment's target covers, whatever kind of target it is.

    A kerbside treatment names one; a two-way lane's AcrossTheJunction names the two ends it
    runs between, and reading only `.leg` off that one reports half of it - the half that is
    not the leg you were asked about half the time.
    """
    ends = getattr(target, "ends", None)
    if callable(ends):
        return [(leg, str(side)) for leg, side in ends()]
    leg = getattr(target, "leg", None)
    if leg is None:
        return []
    side = getattr(target, "side", None)
    return [(leg, str(side) if side is not None else None)]


def report_section(built: Built, leg_filter: str | None) -> None:
    """What each treatment THINKS it placed, beside the room the traced kerb actually gives."""
    legs = built.model.legs
    print(f"\n{'treatment':26s} {'target':22s} {'side':6s} "
          f"{'demand':>8s} {'room':>8s} {'kerbside':>9s}  section offsets ft")
    for treatment in built.state.treatments:
        section = getattr(treatment, "section", None)
        if not callable(section):
            continue
        resolved = section(built.state)
        offsets_of = getattr(resolved, "offsets_from_centerline_ft", None)
        if not callable(offsets_of):
            continue
        # A dict, keyed by which boundary each offset IS - the ordering across the road is the
        # design, so the names are the half of this worth reading.
        offsets = {k: float(v) for k, v in offsets_of().items()}
        demand = max((abs(v) for v in offsets.values()), default=float("nan"))
        written = f"{', '.join(f'{k}={v:.2f}' for k, v in offsets.items())}"
        for leg_name, side in target_leg_sides(getattr(treatment, "target", None)) or [(None, None)]:
            if leg_filter and leg_name != leg_filter:
                continue
            room = kerbside = float("nan")
            if leg_name in legs and side:
                room = narrowest_half_width_ft(legs[leg_name], side)
                kerbside = kerbside_allowance_ft(legs[leg_name], side)
            # Identity is the tell, not the confirmation: a section sized to fill the room fits
            # it by construction, so demand-vs-room can never fail and never meant anything.
            flag = "  <- IDENTITY: one figure, not two" if abs(demand - room) < 0.005 else ""
            print(f"{type(treatment).__name__:26s} {leg_name or '-':22s} {side or '-':6s} "
                  f"{demand:8.2f} {room:8.2f} {kerbside:9.2f}  {written}{flag}")
            written = "(as above)"


def report_limiters(built: Built, leg_filter: str | None, kind_filter: str | None) -> None:
    """ALL FOUR things that decide where kerbside paint starts, because they disagree.

    The answer is whichever sits furthest out, so measuring one, finding it innocent and moving
    on proves nothing - one session reported a 21.5 ft defect that did not exist that way. The
    `paint@` column is the check: it equalling the binding limiter to two decimals is a
    mechanism, and "roughly similar" is a coincidence.

    paint@ is the first station of the paint YOU SELECTED, so compare like with like: these
    limiters hold KERBSIDE paint back, and unfiltered the column reports whichever marking
    happens to start earliest - a bike lane surface that no corner return ever constrained.
    Narrow it (--kind hatch --limiters) before reading the == as agreement.
    """
    legs, fillets = built.model.legs, built.model.corner_fillets
    mouths = junction_mouths_ft(built.state, built.scene.crosswalk_bands)
    # The reach is measured against the resolved crossing BANDS, not the raw OSM crossings: the
    # cross street's band reaches along this leg too, so every band at the junction goes in.
    bands = unary_union(list(built.scene.crosswalk_bands.values()))
    print(f"\n{'leg':22s} {'side':6s} {'clearance':>10s} {'mouth':>8s} {'xwalk':>8s} "
          f"{'tangent':>8s} {'traced':>8s} {'binding':>20s} {'paint@':>8s}")
    for name, leg in sorted(legs.items()):
        if leg_filter and name != leg_filter:
            continue
        for side in ("left", "right"):
            mouth = mouths.get((name, side))
            span = curb_station_span(leg, side)
            limiters = {
                "clearance": leg_clearance_ft(name, legs, fillets, side=side),
                "mouth": mouth[1] if mouth else None,
                "xwalk": crosswalk_reach_on_leg_side_ft(leg, side, bands),
                "tangent": corner_tangent_station_ft(name, side, legs, fillets),
                "traced": span[0] if span else None,
            }
            live = {k: v for k, v in limiters.items() if v is not None}
            binding, value = max(live.items(), key=lambda kv: kv[1]) if live else ("-", 0.0)
            drawn = drawn_on(built, name, side, kind_filter)
            at = drawn[0].min() if drawn else float("nan")
            cells = "".join(f"{limiters[k]:8.2f}" if limiters[k] is not None else f"{'-':>8s}"
                            for k in ("mouth", "xwalk", "tangent", "traced"))
            match = " ==" if abs(at - value) < 0.02 else ""
            print(f"{name:22s} {side:6s} {limiters['clearance']:10.2f}{cells} "
                  f"{binding + ' ' + format(value, '.2f'):>20s} {at:8.2f}{match}")


def report_gaps(built: Built, leg_filter: str | None, kind_filter: str | None,
                bin_ft: float, threshold_ft: float) -> None:
    """Kerb offset minus outermost drawn offset, station by station.

    This is what turns "it looks janky" into "bare from station 0 to 63.7, then 1.4 ft widening
    to 2.6 ft by station 118", which names the mechanism on its own.

    Binned, and the bin is the honest unit: paint is polygons, not a function of station, so the
    outermost vertex in each bin is what "how far out does the paint reach here" can mean. The
    profile runs over the centreline's real length rather than design_length_ft, because the
    question is about what was DRAWN and the drawing is at the render frame.
    """
    print(f"\n{'leg':22s} {'side':6s}  gap profile (kerb - outermost paint, ft; "
          f"{bin_ft:.0f} ft bins, runs over {threshold_ft:.1f} ft)")
    for name, leg in sorted(built.model.legs.items()):
        if leg_filter and name != leg_filter:
            continue
        for side in ("left", "right"):
            drawn = drawn_on(built, name, side, kind_filter)
            edges = np.arange(0.0, leg.centerline.length + bin_ft, bin_ft)
            kerb = curb_offsets_at_stations(leg, side, edges)
            if drawn is None or kerb is None:
                why = "no paint on this side" if drawn is None else "no traced kerb to measure to"
                print(f"{name:22s} {side:6s}  ({why})")
                continue
            station, offset = drawn
            # An empty bin is NaN, and it has to be produced by hand: np.max(..., initial=nan)
            # folds the initial value INTO the reduction, so every bin came out NaN, every
            # comparison came out false, and the profile reported "flush the whole way" while
            # measuring nothing. A false all-clear is worse than a crash.
            reach = np.array([
                (lambda inside: offset[inside].max() if inside.any() else np.nan)(
                    (station >= lo) & (station < lo + bin_ft))
                for lo in edges])
            if not np.isfinite(reach).any():
                print(f"{name:22s} {side:6s}  (paint on this side, but none of it lands in a "
                      f"{bin_ft:.0f} ft bin along the leg - widen --bin)")
                continue
            gap = np.abs(kerb) - reach
            runs, start = [], None
            for i, value in enumerate(gap):
                over = bool(value > threshold_ft)  # NaN compares false: an empty bin ends a run
                if over and start is None:
                    start = i
                elif not over and start is not None:
                    runs.append((start, i - 1))
                    start = None
            if start is not None:
                runs.append((start, len(gap) - 1))
            if not runs:
                print(f"{name:22s} {side:6s}  within {threshold_ft:.1f} ft of the kerb "
                      f"wherever there is paint (max gap {np.nanmax(gap):.2f} ft over "
                      f"{int(np.isfinite(gap).sum())} of {len(gap)} bins)")
                continue
            described = "; ".join(
                f"{edges[a]:.1f}-{edges[b]:.1f} ft: {gap[a]:.2f}->{gap[b]:.2f} "
                f"(max {np.nanmax(gap[a:b + 1]):.2f})" for a, b in runs)
            print(f"{name:22s} {side:6s}  {described}")


def report_continuity(built: Built, kind_filter: str | None) -> None:
    """Whether a facility is ONE piece, and how wide the holes are.

    A corridor treatment that reads as continuous in a render can be two lanes 80.3 ft apart
    with nothing but dotted lines between them - 0.0 sq ft of surface. unary_union fuses
    everything that touches, so a part count above 1 means holes.

    The statistic is each part's distance to its NEAREST neighbour, and the headline is the
    largest of those. Most markings here are dashed on purpose, so a part count and a list of
    the smallest gaps says only "this is a dashed line" - a 5 ft skip line and a facility broken
    in half read identically. The most isolated part is the defect: 36 parts whose worst
    neighbour is 5.00 ft is a dash pattern, and two parts 80.3 ft apart is a severed corridor.
    """
    by_kind: dict[str, list] = {}
    for piece in built.paint:
        if kind_filter and kind_filter not in piece.kind.name:
            continue
        by_kind.setdefault(piece.kind.name, []).append(piece.geometry)
    print(f"\n{'kind':38s} {'pieces':>7s} {'parts':>6s}  gap to nearest neighbouring part, ft")
    for kind, geoms in sorted(by_kind.items()):
        fused = unary_union(geoms)
        parts = list(getattr(fused, "geoms", None) or [fused])
        if len(parts) == 1:
            holes = "one piece"
        elif len(parts) > 120:
            holes = f"{len(parts)} parts is too many to pair off; narrow with --kind"
        else:
            nearest = [float("inf")] * len(parts)
            for (i, a), (j, b) in combinations(enumerate(parts), 2):
                d = a.distance(b)
                nearest[i] = min(nearest[i], d)
                nearest[j] = min(nearest[j], d)
            worst = max(nearest)
            holes = (f"min {min(nearest):.2f}  median {sorted(nearest)[len(nearest) // 2]:.2f}  "
                     f"WORST {worst:.2f}")
            if worst > 20.0 and len(parts) > 1:
                holes += "  <- one part is stranded; check this is meant to be two facilities"
        print(f"{kind:38s} {len(geoms):7d} {len(parts):6d}  {holes}")


def report(model, paint, leg_filter: str | None, kind_filter: str | None) -> None:
    legs = model.legs
    rows = 0
    print(f"{'kind':38s} {'leg':10s} {'side':6s} {'station ft':>18s} {'offset ft':>16s}  pts")
    for piece in paint:
        name = piece.kind.name
        if kind_filter and kind_filter not in name:
            continue
        if leg_filter and piece.leg != leg_filter:
            continue
        coords = piece_coords(piece.geometry)
        if piece.leg in legs:
            station, offset = station_offset_many(legs[piece.leg].centerline, coords)
            span = f"{station.min():8.2f} {station.max():8.2f}"
            across = f"{offset.min():7.2f} {offset.max():7.2f}"
        else:
            # No leg: a corner treatment sits between two of them. Radial distance is the
            # only frame it has, and saying so beats projecting it into an arbitrary leg.
            radius = np.hypot(coords[:, 0] - model.center_ft.x, coords[:, 1] - model.center_ft.y)
            span = f"r{radius.min():7.2f} {radius.max():8.2f}"
            across = " " * 15
        print(f"{name:38s} {piece.leg or '-':10s} {piece.side or '-':6s} "
              f"{span:>18s} {across:>16s}  {len(coords)}")
        rows += 1
    if not rows:
        # An empty table means "no paint", which is a real answer for existing conditions -
        # every marking here comes from a treatment - and would otherwise read as a broken tool.
        print(f"  (no paint matched; the scenario built {len(paint)} piece(s) in total)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", choices=list_sites())
    parser.add_argument("--scenario", help="a build_* in the site's scenarios.py; "
                                           "omit for existing conditions")
    parser.add_argument("--leg", help="only this leg")
    parser.add_argument("--kind", help="substring of the PaintKind name")
    parser.add_argument("--all", action="store_true",
                        help="every report below - the whole quantitative layer in one call")
    parser.add_argument("--paint", action="store_true",
                        help="the piece-by-piece table (the default, unless you asked for one "
                             "of the reports below; 158 rows drowns the answer you wanted)")
    parser.add_argument("--section", action="store_true",
                        help="what each treatment thinks it placed, vs the room the kerb gives")
    parser.add_argument("--limiters", action="store_true",
                        help="all four things deciding where kerbside paint starts")
    parser.add_argument("--gaps", action="store_true",
                        help="kerb offset minus outermost drawn offset, station by station")
    parser.add_argument("--continuity", action="store_true",
                        help="whether a facility is one piece, and how wide the holes are")
    parser.add_argument("--bin", type=float, default=10.0, metavar="FT",
                        help="station bin for --gaps (default 10)")
    parser.add_argument("--gap-threshold", type=float, default=1.0, metavar="FT",
                        help="a gap under this is flush, for --gaps' runs (default 1.0)")
    parser.add_argument("--frame-scale", type=float, default=1.0,
                        help="measure at the frame the reader is looking at, not at 1x (default "
                             "1.0). Same flag as build_all.py: it scales leg_lengths, so a 130 ft "
                             "leg is 325 ft at 2.5 and every station reported here moves with it.")
    args = parser.parse_args()

    # Before the model is built, like build_all.py - the scale reaches the geometry, not just
    # the camera. This script exists to measure the drawn output, and measuring the 1x build
    # while the reader looks at a 2.5x sheet reports defects that are not there (and misses
    # ones that are: W Broad's narrowest half-width is 20.32 ft at 1x and 16.58 ft at 2.5x).
    os.environ[FRAME_SCALE_ENV] = str(args.frame_scale)

    if args.scenario and args.scenario not in scenarios_for(args.site):
        print(f"{args.site} has no scenario {args.scenario}; it has: "
              f"{', '.join(scenarios_for(args.site))}", file=sys.stderr)
        return 2
    built = build(args.site, args.scenario)
    reports = (args.section, args.limiters, args.gaps, args.continuity)
    if args.paint or args.all or not any(reports):
        report(built.model, built.paint, args.leg, args.kind)
    if args.all or args.section:
        report_section(built, args.leg)
    if args.all or args.limiters:
        report_limiters(built, args.leg, args.kind)
    if args.all or args.gaps:
        report_gaps(built, args.leg, args.kind, args.bin, args.gap_threshold)
    if args.all or args.continuity:
        report_continuity(built, args.kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
