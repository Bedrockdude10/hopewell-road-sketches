"""What was actually DRAWN, stationed against each leg's centreline.

    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --scenario build_two_way_bike_lane
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --leg north --kind bike_lane
    .venv/bin/python scripts/measure_drawn.py wbroad_louellen --frame-scale 2.5
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --scenario X --all

--all adds the other five questions SKILLS.md 0a says answer most complaints, so that answering
one never means writing a throwaway script or cropping a render:

    --section     what each treatment THINKS it placed, beside the room the traced kerb gives
    --limiters    all four things that decide where kerbside paint starts, not the first one
    --gaps        kerb offset minus outermost drawn offset, station by station
    --lanes       how wide the drawn travel lane is - drawn stripe to innermost drawn marking
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
from src.geometry.targets import BOTH_SIDES
from src.geometry.treatments.base import TARGET_LANE_WIDTH_FT, kerbside_allowance_ft
from src.render.crosswalks import crosswalk_reach_on_leg_side_ft
from src.geometry.treatments import DesignState
from src.render.export import (BUILDING_CONTEXT_RADIUS_M, KERB_RADIUS_M,
                               TRAFFIC_CONTROL_RADIUS_M)
from src.render.frame import FRAME_SCALE_ENV
from src.render.coords import FT_TO_M
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
    Runs over the centreline's whole length, because the question is about what was DRAWN and the
    drawing is at the render frame - measure at the reader's --frame-scale, not at 1x.
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
            # NaN compares false, so an empty bin ends a run rather than extending it.
            runs = _runs(gap > threshold_ft)
            if not runs:
                print(f"{name:22s} {side:6s}  within {threshold_ft:.1f} ft of the kerb "
                      f"wherever there is paint (max gap {np.nanmax(gap):.2f} ft over "
                      f"{int(np.isfinite(gap).sum())} of {len(gap)} bins)")
                continue
            described = "; ".join(
                f"{edges[a]:.1f}-{edges[b]:.1f} ft: {gap[a]:.2f}->{gap[b]:.2f} "
                f"(max {np.nanmax(gap[a:b + 1]):.2f})" for a, b in runs)
            print(f"{name:22s} {side:6s}  {described}")


def _bin_stat(edges: np.ndarray, bin_ft: float, station: np.ndarray, value: np.ndarray,
              reduce) -> np.ndarray:
    """`reduce` over the values landing in each bin, NaN where a bin holds nothing.

    The NaN is produced by hand because the obvious spelling does not work:
    `np.max(..., initial=np.nan)` folds the initial value INTO the reduction, so every bin comes
    out NaN, every comparison against a threshold comes out false, and a profile reports
    "flush the whole way" having measured nothing. A quantitative check that cannot see
    anything has to say so, not pass.
    """
    return np.array([
        (lambda inside: reduce(value[inside]) if inside.any() else np.nan)(
            (station >= lo) & (station < lo + bin_ft))
        for lo in edges])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The [start, end] index pairs of each contiguous True run in `mask`."""
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs + ([(start, len(mask) - 1)] if start is not None else [])


def _longitudinal_on(built: Built, leg, leg_name: str, side: str
                     ) -> tuple[np.ndarray, np.ndarray, set[str]] | None:
    """(station, |offset|) of the drawn paint that RUNS ALONG this leg-side, and its kinds.

    Longitudinal is decided by extent rather than by name: a piece counts when it reaches
    further along the leg than it does across it. A name list would have to be extended by hand
    for every new kerbside marking and would silently omit it until someone remembered - the
    hole POLYLINE_CHANNELS had. Measured, a stop bar and a stall divider (a couple of feet of
    station against several feet of offset) drop out on their own, and a taper stays in, which
    is right: where a taper runs, the taper IS the lane's edge.

    THE OFFSET RETURNED IS THE PAINT'S INNER FACE, not its axis, and only for a stroked LINE -
    a FILL or a COLOUR travels as its own polygon, so its boundary already IS its face. This
    project sizes kerbside treatments so the stripe's own width comes out of the TREATMENT and
    not out of the lane (checks.PaintClearOfTheTravelLane says so in as many words), which puts
    a stripe's axis half a width outboard of the lane it bounds. Measured axis to axis, every
    designed lane at all five sites reads 11.41 ft and compares false against an 11.00 ft
    target - one number, reported 92 times as a defect.
    """
    stations, offsets, kinds = [], [], set()
    for piece in built.paint:
        if piece.leg != leg_name or piece.side != side or piece.kind.is_object:
            continue
        station, offset = station_offset_many(leg.centerline, piece_coords(piece.geometry))
        if np.ptp(station) <= np.ptp(np.abs(offset)):
            continue
        stroke_m = piece.kind.channel.stroke_width_m if piece.kind.is_line else None
        kinds.add(piece.kind.name)
        stations.append(station)
        offsets.append(np.abs(offset) - (stroke_m or 0.0) / FT_TO_M / 2)
    if not stations:
        return None
    return np.concatenate(stations), np.concatenate(offsets), kinds


def report_lanes(built: Built, leg_filter: str | None, bin_ft: float) -> None:
    """How wide the DRAWN travel lane is on each side of each leg, station by station.

    "Every travel lane is exactly 11.00 ft" is this repo's canonical wrong answer: it was
    computed from the section while the drawn centreline sat 2.84 ft off the alignment, so one
    lane was 8.16 ft. Both bounds here are therefore read off the drawing.

    THE INNER BOUND IS THE STRIPE THE VIEWS ACTUALLY PAINT, through the same
    centerline_paint_ft call src/render/export.py and the plan view make, shift and all. Neither
    the alignment (the bug above) nor divider_shift_toward_ft, which would be the second
    derivation of one number that every serious defect here has turned out to be. A leg drawn
    with no centre stripe has no drawn inner bound at all: the row says `stripe=none` and
    measures from the alignment, because an undivided road has no lane to measure, only a
    half-road.

    THE OUTER BOUND IS THE INNERMOST LONGITUDINAL MARKING, or the traced kerb where there is
    none - and the `by` column names which, because the two answers mean opposite things. A lane
    held in by paint is this design's decision. A lane held in by the kerb is the street being
    narrow, which W Broad's north-east approach is: 7.2 ft from the alignment to its right kerb,
    and no 11 ft lane there to protect.

    From the divider's axis out to the bounding paint's INNER FACE - see _longitudinal_on for
    why the face and not the axis. That makes the figure directly comparable with
    TARGET_LANE_WIDTH_FT and with what checks.PaintClearOfTheTravelLane enforces. The asphalt a
    driver gets is narrower still by half the centre stripe, which is a constant across every
    leg here and so not what any of these rows are about.
    """
    from src.render.crosswalks import centerline_paint_ft, centerline_start_ft

    state, scene = built.state, built.scene
    print(f"\n{'leg':22s} {'side':6s} {'stripe':>7s} {'min':>7s} {'med':>7s} {'max':>7s} "
          f"{'bins':>5s}  under {TARGET_LANE_WIDTH_FT:.2f} ft")
    for leg_name, leg in sorted(built.model.legs.items()):
        if leg_filter and leg_name != leg_filter:
            continue
        style = state.centerline_style(leg_name)
        stripes = centerline_paint_ft(
            leg,
            centerline_start_ft(scene.crosswalk_offsets[leg_name].offset_ft,
                                scene.stop_bar_offsets.get(leg_name),
                                leg_name in scene.marked_crosswalks),
            style,
            *(state.travel_lane_divider_shift(leg_name) or (0.0, None)))
        edges = np.arange(0.0, leg.centerline.length + bin_ft, bin_ft)
        if stripes:
            # EACH STRIPE BINNED SEPARATELY AND THEN AVERAGED, because a double yellow is two
            # stripes 0.33 ft apart and they are separate LineStrings with vertices at their own
            # stations. Pooling the vertices and taking one mean lets an uneven count inside a bin
            # weight one stripe over the other, which wobbled a constant-offset divider by 0.04 ft
            # and put that wobble into the lane width - a measurement artefact reported as a
            # varying lane.
            # AND A BIN THAT ONE STRIPE REACHES AND THE OTHER DOES NOT IS NOT A MEASUREMENT OF THE
            # DIVIDER. nanmean read such a bin as the single stripe it could see, which put a
            # double yellow lying dead on the alignment 0.164 ft off centre: princeton_ave_north
            # reported a 10.84 ft lane against an 11.16 ft one, off two vertices a float either
            # side of one bin edge. A plain mean propagates the NaN and the interpolation below
            # then reads the drawn line through the hole, which is what the dashed case needed
            # anyway - so a bin no stripe reaches, or only half of one does, says nothing rather
            # than saying something wrong.
            divider = np.mean([
                _bin_stat(edges, bin_ft,
                          *station_offset_many(leg.centerline,
                                               np.asarray(line.coords, dtype=float)), np.mean)
                for line in stripes], axis=0)
            # A DASHED STRIPE LEAVES EMPTY BINS AND THE LINE STILL RUNS THROUGH THEM. The divider
            # is one continuous line that happens to be painted in dashes, so interpolating
            # between two dashes reads the drawn line rather than inventing it. Outside the
            # painted extent it stays NaN, because there the drawing really does show no divider:
            # the centreline stops at the stop bar and does not cross the junction.
            seen = np.flatnonzero(np.isfinite(divider))
            if seen.size:
                span = slice(seen[0], seen[-1] + 1)
                divider[span] = np.interp(edges[span], edges[seen], divider[seen])
        else:
            divider = np.zeros_like(edges)
            style = "none"
        for side in BOTH_SIDES:
            drawn = _longitudinal_on(built, leg, leg_name, str(side))
            # ONLY WHERE THE KERB WAS ACTUALLY TRACED. curb_offsets_at_stations interpolates, and
            # np.interp CLAMPS outside its range rather than refusing - so every bin beyond the
            # traced span comes back holding the kerb's last value, and a lane measured against
            # it reports a width nobody drew. Every kerb at every site starts 12-58 ft out.
            kerb = curb_offsets_at_stations(leg, str(side), edges)
            traced = curb_station_span(leg, str(side))
            if kerb is not None and traced is not None:
                kerb = np.where((edges >= traced[0]) & (edges <= traced[1]), kerb, np.nan)
            paint = (_bin_stat(edges, bin_ft, drawn[0], drawn[1], np.min) if drawn
                     else np.full_like(edges, np.nan))
            held_by_paint = np.isfinite(paint)
            bound = np.where(held_by_paint, paint, np.abs(kerb) if kerb is not None else np.nan)
            width = bound - divider * side.sign
            if not np.isfinite(width).any():
                why = ("no traced kerb and no paint running along this side" if drawn is None
                       else "nothing drawn in any bin along this side")
                print(f"{leg_name:22s} {side.value:6s} {style:>7s}  ({why})")
                continue
            under = np.isfinite(width) & (width < TARGET_LANE_WIDTH_FT - 0.01)
            if not under.any():
                verdict = (f"none - held in by "
                           f"{'paint' if held_by_paint.all() else 'kerb' if not held_by_paint.any() else 'both'}")
            else:
                worst = int(np.nanargmin(np.where(under, width, np.nan)))
                verdict = (f"{int(under.sum())} of {int(np.isfinite(width).sum())} bins, worst "
                           f"{width[worst]:.2f} ft at station {edges[worst]:.0f} "
                           f"({'paint' if held_by_paint[worst] else 'kerb'}) over "
                           + "; ".join(f"{edges[a]:.0f}-{edges[b]:.0f} ft" for a, b in _runs(under)))
            print(f"{leg_name:22s} {side.value:6s} {style:>7s} {np.nanmin(width):7.2f} "
                  f"{np.nanmedian(width):7.2f} {np.nanmax(width):7.2f} "
                  f"{int(np.isfinite(width).sum()):5d}  {verdict}")


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
    parser.add_argument("--lanes", action="store_true",
                        help="how wide the drawn travel lane is, from the drawn stripe out to "
                             "the innermost drawn marking (or the kerb)")
    parser.add_argument("--continuity", action="store_true",
                        help="whether a facility is one piece, and how wide the holes are")
    parser.add_argument("--bin", type=float, default=10.0, metavar="FT",
                        help="station bin for --gaps and --lanes (default 10)")
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
    reports = (args.section, args.limiters, args.gaps, args.lanes,
               args.continuity)
    if args.paint or args.all or not any(reports):
        report(built.model, built.paint, args.leg, args.kind)
    if args.all or args.section:
        report_section(built, args.leg)
    if args.all or args.limiters:
        report_limiters(built, args.leg, args.kind)
    if args.all or args.gaps:
        report_gaps(built, args.leg, args.kind, args.bin, args.gap_threshold)
    if args.all or args.lanes:
        report_lanes(built, args.leg, args.bin)
    if args.all or args.continuity:
        report_continuity(built, args.kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
