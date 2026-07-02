"""
Phase 4 (stretch): export existing-conditions + proposed-treatment geometry, then
drive headless Blender (`blender --background --python blender_scene.py`) to
render both as presentation-ready 3D stills.

Requires Blender on PATH, or set BLENDER_BIN to the executable
(e.g. /Applications/Blender.app/Contents/MacOS/Blender on macOS).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.intersection import load_intersection_model
from src.osm_context import fetch_buildings, fetch_crossings
from src.scenarios import build_demo_scenario
from src.treatments import DesignState

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
BLENDER_SCENE_SCRIPT = Path(__file__).resolve().parent / "blender_scene.py"

DEFAULT_MAC_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def find_blender() -> str:
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    if Path(DEFAULT_MAC_BLENDER).exists():
        return DEFAULT_MAC_BLENDER
    raise RuntimeError(
        "Blender not found. Install it, add it to PATH, or set BLENDER_BIN to its executable."
    )


def render_all(blender_bin: str, jobs: list[tuple[Path, Path]]):
    """Render every (geometry.json, output.png) job in a single Blender process -
    each launch has ~1-1.5s of fixed startup overhead, not worth paying per-render."""
    args = [str(p) for pair in jobs for p in pair]
    cmd = [blender_bin, "--background", "--python", str(BLENDER_SCENE_SCRIPT), "--", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.count("RENDER_DONE") != len(jobs):
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("Blender render failed")
    for _, output_path in jobs:
        print(f"Rendered {output_path}")


def main():
    blender_bin = find_blender()
    print(f"Using Blender: {blender_bin}")

    model = load_intersection_model()
    baseline = DesignState.from_model(model)
    scenario = build_demo_scenario(baseline)

    print("Fetching OSM building context...")
    buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    print(f"  -> {len(buildings)} buildings")

    print("Fetching OSM-mapped pedestrian crossings...")
    crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    print(f"  -> {len(crossings)} crossings")

    existing_json = export_scenario(model, baseline, "Existing Conditions", OUTPUT_DIR / "geometry_existing.json",
                                     buildings=buildings, crossings=crossings)
    proposed_json = export_scenario(model, scenario, "Proposed Treatments", OUTPUT_DIR / "geometry_proposed.json",
                                     buildings=buildings, crossings=crossings)

    render_all(blender_bin, [
        (existing_json, OUTPUT_DIR / "phase4_render_existing.png"),
        (proposed_json, OUTPUT_DIR / "phase4_render_proposed.png"),
    ])


if __name__ == "__main__":
    main()
