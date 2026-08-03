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
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.render.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.geometry.intersection import load_intersection_model
from src.sources.osm_context import fetch_buildings, fetch_crossings
from src.site import add_scenario_arg, add_site_arg, load_site_scenarios, scenario_label, site_output_dir, run_scenario
from src.render.theme import build_default_theme
from src.geometry.treatments import DesignState

BLENDER_SCENE_SCRIPT = Path(__file__).resolve().parent / "blender" / "blender_scene.py"
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


# Peak resident memory of one headless Blender rendering these scenes, measured on this
# project's own output: ~11 GB, dominated by the 4k near-zone textures and the ~80-100
# building meshes. Used to decide how many can run at once - see blender_job_limit.
BLENDER_PEAK_RAM_GB = 11
# Left for the OS and whatever else is open. Without it a "safe" job count still pushes the
# machine into swap, which is slower than rendering serially.
RAM_HEADROOM_GB = 8


def blender_job_limit(requested: int | None = None) -> int:
    """How many Blender processes this machine can actually hold at once.

    Blender is not the kind of job you scale to core count. At ~11 GB peak each, four
    instances need 44 GB; on a 36 GB machine that exhausted RAM and all 7 GB of swap, and
    the OOM killer took every one of them - surfacing as `zsh: terminated` and exit 137,
    which look nothing like "out of memory" unless you already suspect it.
    """
    if requested:
        return max(1, requested)
    try:
        total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return 1  # can't tell: the safe answer is one at a time
    return max(1, int((total_gb - RAM_HEADROOM_GB) // BLENDER_PEAK_RAM_GB))


# Name of the environment variable blender_scene.py reads its resolution multiplier from.
# Defined here as well so callers (scripts/build_all.py) do not have to spell the string, and
# so grep finds both ends of it. See blender_scene.render_scale for the meaning and the cap.
RENDER_SCALE_ENV = "HOPEWELL_RENDER_SCALE"


def render_all(blender_bin: str, jobs: list[tuple[Path, Path]]):
    """Render every (geometry.json, output.png) job in a single Blender process -
    each launch has ~1-1.5s of fixed startup overhead, not worth paying per-render."""
    args = [str(p) for pair in jobs for p in pair]
    cmd = [blender_bin, "--background", "--python", str(BLENDER_SCENE_SCRIPT), "--", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    rendered = result.stdout.count("RENDER_DONE")
    if result.returncode != 0 or rendered != len(jobs):
        scenes = ", ".join(out.name for _, out in jobs)
        # A negative return code is death by signal, and -9 is the OOM killer. Saying so is
        # the difference between a five-minute fix and an afternoon: Blender's own output
        # says nothing, because it never got to run its error handling.
        if result.returncode == -signal.SIGKILL:
            raise RuntimeError(
                f"Blender was killed by the OS (SIGKILL) after {rendered}/{len(jobs)} scene(s) - "
                f"almost always out of memory. Each instance peaks around "
                f"{BLENDER_PEAK_RAM_GB} GB; lower --render-jobs. Scenes in this batch: {scenes}")
        if result.returncode < 0:
            raise RuntimeError(f"Blender was killed by signal {-result.returncode} after "
                               f"{rendered}/{len(jobs)} scene(s). Scenes: {scenes}")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender render failed after {rendered}/{len(jobs)} scene(s). "
                           f"Scenes in this batch: {scenes}")
    for _, output_path in jobs:
        print(f"Rendered {output_path}")


def main():
    parser = add_scenario_arg(add_site_arg(argparse.ArgumentParser()))
    parser.add_argument("--render-scale", type=int, default=1, choices=(1, 2, 3, 4),
                        help="render resolution as a multiple of 1920x1440 (default 1)")
    args = parser.parse_args()
    os.environ[RENDER_SCALE_ENV] = str(args.render_scale)
    out_dir = site_output_dir(args.site)
    label = scenario_label(args.scenario)

    blender_bin = find_blender()
    print(f"Using Blender: {blender_bin}")

    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    build_scenario = getattr(load_site_scenarios(args.site), args.scenario)
    scenario = run_scenario(build_scenario, baseline, model)

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
