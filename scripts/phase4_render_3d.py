"""
Phase 4 (stretch): export existing-conditions + one named proposal's geometry
for a site, then drive headless Blender (`blender --background --python
blender_scene.py`) to render both as presentation-ready 3D stills.

Usage: python scripts/phase4_render_3d.py [--site broad_st_greenwood] [--scenario build_demo_scenario]

A site can define any number of proposals in its scenarios.py (see
sites/README.md); pass the function name via --scenario to render a specific
one (e.g. --scenario build_proposal_a_paint_only). Output files are named
after it (geometry_<label>.json, phase4_render_<label>.png), except the
default scenario which keeps the original *_existing/*_proposed names.

Requires Blender on PATH, or set BLENDER_BIN to the executable
(e.g. /Applications/Blender.app/Contents/MacOS/Blender on macOS).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.intersection import load_intersection_model
from src.osm_context import fetch_buildings, fetch_crossings
from src.site import add_scenario_arg, add_site_arg, load_site_scenarios, scenario_label, site_output_dir
from src.theme import build_default_theme
from src.treatments import DesignState

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
    args = add_scenario_arg(add_site_arg(argparse.ArgumentParser())).parse_args()
    out_dir = site_output_dir(args.site)
    label = scenario_label(args.scenario)

    blender_bin = find_blender()
    print(f"Using Blender: {blender_bin}")

    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    build_scenario = getattr(load_site_scenarios(args.site), args.scenario)
    scenario = build_scenario(baseline)

    print("Fetching OSM building context...")
    buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    print(f"  -> {len(buildings)} buildings")

    print("Fetching OSM-mapped pedestrian crossings...")
    crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    print(f"  -> {len(crossings)} crossings")

    print("Fetching render theme (Poly Haven textures/models, cached under output/.textures/)...")
    theme = build_default_theme()
    missing = [k for k, v in theme.items() if v is None]
    print(f"  -> ready ({len(theme) - len(missing)}/{len(theme)} assets; missing: {missing or 'none'})")

    existing_json = export_scenario(model, baseline, "Existing Conditions", out_dir / "geometry_existing.json",
                                     buildings=buildings, crossings=crossings, theme=theme)
    proposed_json = export_scenario(model, scenario, f"Proposed Treatments ({args.scenario})",
                                     out_dir / f"geometry_{label}.json",
                                     buildings=buildings, crossings=crossings, theme=theme)

    render_all(blender_bin, [
        (existing_json, out_dir / "phase4_render_existing.png"),
        (proposed_json, out_dir / f"phase4_render_{label}.png"),
    ])


if __name__ == "__main__":
    main()
