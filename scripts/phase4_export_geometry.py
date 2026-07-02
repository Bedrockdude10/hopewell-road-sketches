"""
Phase 4 (stretch), step 1: export the Phase 2 baseline and a site's Phase 3
demo scenario to plain JSON (local meters) for the headless Blender render
script. Export-only, no Blender - useful for inspecting/debugging the JSON.

Usage: python scripts/phase4_export_geometry.py [--site broad_st_greenwood]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.intersection import load_intersection_model
from src.osm_context import fetch_buildings, fetch_crossings
from src.site import add_site_arg, load_site_scenarios, site_output_dir
from src.treatments import DesignState


def main():
    args = add_site_arg(argparse.ArgumentParser()).parse_args()
    out_dir = site_output_dir(args.site)

    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    scenario = load_site_scenarios(args.site).build_demo_scenario(baseline)
    buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)

    existing_path = export_scenario(model, baseline, "Existing Conditions", out_dir / "geometry_existing.json",
                                     buildings=buildings, crossings=crossings)
    proposed_path = export_scenario(model, scenario, "Proposed Treatments", out_dir / "geometry_proposed.json",
                                     buildings=buildings, crossings=crossings)

    print(f"Exported existing conditions -> {existing_path}")
    print(f"Exported proposed treatments -> {proposed_path}")


if __name__ == "__main__":
    main()
