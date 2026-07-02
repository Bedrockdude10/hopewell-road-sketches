"""
Phase 4 (stretch), step 1: export the Phase 2 baseline and one of a site's
Phase 3 scenarios to plain JSON (local meters) for the headless Blender render
script. Export-only, no Blender - useful for inspecting/debugging the JSON.

Usage: python scripts/phase4_export_geometry.py [--site broad_st_greenwood] [--scenario build_demo_scenario]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.render.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.geometry.intersection import load_intersection_model
from src.sources.osm_context import fetch_buildings, fetch_crossings
from src.site import add_scenario_arg, add_site_arg, load_site_scenarios, scenario_label, site_output_dir
from src.render.theme import build_default_theme
from src.geometry.treatments import DesignState


def main():
    args = add_scenario_arg(add_site_arg(argparse.ArgumentParser())).parse_args()
    out_dir = site_output_dir(args.site)
    label = scenario_label(args.scenario)

    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    build_scenario = getattr(load_site_scenarios(args.site), args.scenario)
    scenario = build_scenario(baseline)
    buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    theme = build_default_theme()

    existing_path = export_scenario(model, baseline, "Existing Conditions", out_dir / "geometry_existing.json",
                                     buildings=buildings, crossings=crossings, theme=theme)
    proposed_path = export_scenario(model, scenario, f"Proposed Treatments ({args.scenario})",
                                     out_dir / f"geometry_{label}.json",
                                     buildings=buildings, crossings=crossings, theme=theme)

    print(f"Exported existing conditions -> {existing_path}")
    print(f"Exported {args.scenario} -> {proposed_path}")


if __name__ == "__main__":
    main()
