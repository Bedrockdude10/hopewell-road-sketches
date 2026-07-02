"""
Phase 4 (stretch), step 1: export the Phase 2 baseline and Phase 3 treatment
scenario to plain JSON (local meters) for the headless Blender render script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.intersection import load_intersection_model
from src.osm_context import fetch_buildings, fetch_crossings
from src.scenarios import build_demo_scenario
from src.treatments import DesignState

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    model = load_intersection_model()
    baseline = DesignState.from_model(model)
    scenario = build_demo_scenario(baseline)
    buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)

    existing_path = export_scenario(model, baseline, "Existing Conditions", OUTPUT_DIR / "geometry_existing.json",
                                     buildings=buildings, crossings=crossings)
    proposed_path = export_scenario(model, scenario, "Proposed Treatments", OUTPUT_DIR / "geometry_proposed.json",
                                     buildings=buildings, crossings=crossings)

    print(f"Exported existing conditions -> {existing_path}")
    print(f"Exported proposed treatments -> {proposed_path}")


if __name__ == "__main__":
    main()
