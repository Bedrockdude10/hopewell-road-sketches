"""Export every site's every scenario to JSON, for diffing a refactor against itself.

`scripts/build_all.py --render-3d` writes these files too, but it also pays for Blender -
~17 s for the first scene in a process and ~5 s for each one after it, so ~1-2 minutes for
every site's every scenario. This does only the part that is this project's own code, and it
is the right part: export_scenario resolves the scene, builds the paint, builds the props and
asserts every invariant, so a change that moves any marking, any prop or any note shows up
here.

    python scripts/export_all_scenarios.py /tmp/before
    ... make a change ...
    python scripts/export_all_scenarios.py /tmp/after
    python scripts/diff_exports.py /tmp/before /tmp/after

Offline and against the committed OSM fixture, so it is reproducible:

    HOPEWELL_OFFLINE=1 HOPEWELL_OSM_CACHE=tests/fixtures/osm_cache PYTHONPATH=. \\
        .venv/bin/python scripts/export_all_scenarios.py /tmp/before

A site or scenario that will not export is reported by name at the end and the run carries
on, so one bad junction doesn't cost you the comparison for all the others; the exit status
is non-zero if anything failed. Note that diff_exports then compares two possibly-partial
trees - a scenario missing from BOTH sides is a scenario the diff cannot speak for.
"""
import argparse
import contextlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_all import scenarios_for
from scripts.jobs import MAX_BUILD_JOBS
from src.geometry.intersection import load_intersection_model
from src.geometry.treatments import DesignState
from src.render.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.site import list_sites, load_site_scenarios, run_scenario, scenario_label
from src.sources.osm_context import fetch_buildings, fetch_crossings


def export_site(site: str, out_dir: Path) -> tuple[list[Path], list[str]]:
    """The untreated baseline plus every build_* in this site's scenarios.py.

    Returns (written, failures). NOTHING RAISES OUT OF HERE. This script's whole job is a
    before/after comparison across every site at once, so an exception that escapes costs
    the reader the sites that were fine as well as the one that broke - and it writes
    nothing usable, which is worse than reporting the bad site by name. Found with the
    NJ 31 & W Delaware junction, whose pavement ring does not close: one raise inside
    build_pavement_polygon and all six sites' exports were lost.

    Two boundaries, matching scripts/build_all.py:

    - PER SITE, around loading the model, its OSM context and its scenarios.py. This work
      is shared by every scenario, so if it fails the site has no exports to attempt and
      one line says so.
    - PER SCENARIO, around running the treatment and exporting it. Scenarios are
      independent - each starts from a fresh DesignState.from_model - so a proposal whose
      geometry the corner model cannot represent must not cost this site its baseline and
      its other proposals, which are exactly what you need in order to see whether the
      refactor moved anything.
    """
    quiet = io.StringIO()
    failures: list[str] = []
    try:
        with contextlib.redirect_stdout(quiet):
            model = load_intersection_model(site=site)
            crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
            buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
            scenarios = load_site_scenarios(site)
            names = scenarios_for(site, scenarios)
    except Exception as e:
        return [], [f"{site}: could not load the site - {type(e).__name__}: {e}"]

    written = []
    for label, name in [("existing", "Existing Conditions")] + [
            (scenario_label(name), name) for name in names]:
        try:
            with contextlib.redirect_stdout(quiet):
                state = (DesignState.from_model(model) if label == "existing" else
                         run_scenario(getattr(scenarios, name), DesignState.from_model(model),
                                      model))
                written.append(export_scenario(
                    model, state, name, out_dir / site / f"geometry_{label}.json",
                    buildings=buildings, crossings=crossings, theme={}))
        except Exception as e:
            # The scenario NAME and the actual error, not just a count: "3 scenarios failed"
            # sends the reader back to reproduce them one at a time, which is the cost this
            # script exists to avoid.
            failures.append(f"{site}/{label}: export failed - {type(e).__name__}: {e}")
    return written, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="directory to write geometry_*.json under")
    parser.add_argument("--site", action="append",
                        help="limit to this site (repeatable) - a single-site before/after is "
                             "~1 s, which is the loop while editing one junction")
    parser.add_argument("--jobs", type=int, default=MAX_BUILD_JOBS,
                        help=f"parallel worker processes (default {MAX_BUILD_JOBS} - see "
                             f"scripts/jobs.py, which is a house rule about a 36 GB machine "
                             f"and not a core count)")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    sites = args.site or list_sites()
    jobs = max(1, min(args.jobs, len(sites)))

    total = 0
    failures: list[str] = []
    # Sites share nothing but a read-only cache, so they export in parallel processes, the same
    # unit scripts/build_all.py parallelises. Threads would not help: the cost is this
    # project's own geometry and it holds the GIL. pool.map keeps the sites in order, so the
    # printed report does not shuffle from run to run.
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for site, (written, site_failures) in zip(
                sites, pool.map(partial(export_site, out_dir=out_dir), sites)):
            total += len(written)
            failures += site_failures
            status = f"  {len(site_failures)} FAILED" if site_failures else ""
            print(f"  {site:22s} {len(written)} export(s){status}")
    print(f"{total} export(s) under {out_dir}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
