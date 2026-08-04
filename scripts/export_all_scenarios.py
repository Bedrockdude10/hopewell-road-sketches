"""Export every site's every scenario to JSON, for diffing a refactor against itself.

`scripts/build_all.py --render-3d` writes these files too, but it also spends ~4 minutes of
Blender per site doing it, which is far too slow to run after each step of a refactor. This
does only the part that is this project's own code, and it is the right part: export_scenario
resolves the scene, builds the paint, builds the props and asserts every invariant, so a
change that moves any marking, any prop or any note shows up here.

    python scripts/export_all_scenarios.py /tmp/before
    ... make a change ...
    python scripts/export_all_scenarios.py /tmp/after
    python scripts/diff_exports.py /tmp/before /tmp/after

Offline and against the committed OSM fixture, so it is reproducible:

    HOPEWELL_OFFLINE=1 HOPEWELL_OSM_CACHE=tests/fixtures/osm_cache PYTHONPATH=. \\
        .venv/bin/python scripts/export_all_scenarios.py /tmp/before
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_all import scenarios_for
from src.geometry.intersection import load_intersection_model
from src.geometry.treatments import DesignState
from src.render.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.site import list_sites, load_site_scenarios, run_scenario, scenario_label
from src.sources.osm_context import fetch_buildings, fetch_crossings


def export_site(site: str, out_dir: Path) -> list[Path]:
    """The untreated baseline plus every build_* in this site's scenarios.py."""
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        model = load_intersection_model(site=site)
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        scenarios = load_site_scenarios(site)
        written = []
        for label, name, state in [("existing", "Existing Conditions",
                                    DesignState.from_model(model))] + [
                (scenario_label(name), name,
                 run_scenario(getattr(scenarios, name), DesignState.from_model(model), model))
                for name in scenarios_for(site, scenarios)]:
            written.append(export_scenario(
                model, state, name, out_dir / site / f"geometry_{label}.json",
                buildings=buildings, crossings=crossings, theme={}))
    return written


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1]).resolve()
    total = 0
    for site in list_sites():
        written = export_site(site, out_dir)
        total += len(written)
        print(f"  {site:22s} {len(written)} export(s)")
    print(f"{total} export(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
