"""What was actually DRAWN, stationed against each leg's centreline.

    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --scenario build_two_way_bike_lane
    .venv/bin/python scripts/measure_drawn.py broad_st_greenwood --leg north --kind bike_lane

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.build_all import scenarios_for
from src.geometry.intersection import load_intersection_model
from src.geometry.model import station_offset_many
from src.geometry.treatments import DesignState
from src.render.export import (BUILDING_CONTEXT_RADIUS_M, KERB_RADIUS_M,
                               TRAFFIC_CONTROL_RADIUS_M)
from src.render.props import build_props
from src.render.scene import SceneGeometry
from src.site import list_sites, load_site_scenarios, run_scenario
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_street_furniture,
                                     fetch_traffic_control)


def build(site: str, scenario: str | None):
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
    return model, paint


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
        coords = np.asarray(piece.geometry.exterior.coords
                            if piece.geometry.geom_type == "Polygon"
                            else piece.geometry.coords, dtype=float)
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
    args = parser.parse_args()

    if args.scenario and args.scenario not in scenarios_for(args.site):
        print(f"{args.site} has no scenario {args.scenario}; it has: "
              f"{', '.join(scenarios_for(args.site))}", file=sys.stderr)
        return 2
    model, paint = build(args.site, args.scenario)
    report(model, paint, args.leg, args.kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
