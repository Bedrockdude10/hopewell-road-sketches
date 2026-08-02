"""Build everything - all sites, all scenarios - with one command.

Rebuilding by hand meant ~30 separate `python scripts/phaseN_*.py` runs, each waited on in
turn. Measured, almost none of that time is this project's geometry (~0.25 s per site);
it's matplotlib rasterizing the plan views, and Blender. Both parallelise cleanly, because
sites share nothing but a read-only cache, so this runs `--jobs` of them at once. Four
sites, 27 scenarios: ~215 s serial, ~110 s at --jobs 4.

    python scripts/build_all.py                     # 2D for every site and scenario
    python scripts/build_all.py --render-3d         # ...and the Blender renders too
    python scripts/build_all.py --dpi 90            # faster pictures while iterating
    python scripts/build_all.py --refresh-osm       # re-pull OSM after tracing in the editor
    python scripts/build_all.py --site columbia_princeton --render-3d

It writes exactly the files the phase scripts write (phase2_geometry_plot.png,
phase3_before_after_*.png, geometry_*.json, phase4_render_*.png), so it replaces running
them one by one rather than producing a second set of artifacts to keep in sync.

OSM STALENESS. The Overpass cache in output/.cache is keyed by (centre, radius) and never
expires, so a kerb, crossing or tactile-paving pad traced in OSM this morning is invisible
to the build until the cache is re-pulled. That is the project's worst failure mode -
ground truth present, but never reaching the render - so every site prints how old the
layers it read are, and `--refresh-osm` re-pulls them from Overpass (one round trip per
layer per site, not per scenario). Refresh is ignored under HOPEWELL_OFFLINE, which is how
the test suite stays hermetic.

Scene invariants (src/checks.py) run on every scenario. A failure is reported per scenario
and the build carries on, so one bad junction doesn't hide the state of the other three;
the exit status is non-zero if anything failed.
"""
import argparse
import contextlib
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")   # no GUI backend: this is a batch build
import matplotlib.pyplot as plt

from src.checks import SceneInvariantError
from src.geometry.intersection import load_intersection_model
from src.geometry.treatments import DesignState
from src.render.export import BUILDING_CONTEXT_RADIUS_M, export_scenario
from src.render.plan_view import legend_handles, plot_design_state
from src.render.theme import build_default_theme
from src.site import list_sites, load_site_scenarios, scenario_label, site_output_dir
from src.sources.osm_context import REFRESH_ENV, cache_summary, fetch_buildings, fetch_crossings

# Plot resolution. 150 matches what the phase scripts write; matplotlib rasterization is
# the single biggest cost in a 2D-only build, so --dpi 90 roughly halves it when you are
# iterating on geometry and don't need a presentation-quality picture yet.
PLOT_DPI = 150

DEFAULT_SCENARIOS = ("build_proposal_1_continental",
                     "build_proposal_2_continental_parking_narrowing",
                     "build_proposal_3_continental_parking_narrowing_bulbouts")


def scenarios_for(site: str) -> list[str]:
    """Every proposal a site actually defines, in declaration order where possible."""
    module = load_site_scenarios(site)
    named = [name for name in DEFAULT_SCENARIOS if hasattr(module, name)]
    extra = [name for name in dir(module)
             if name.startswith("build_") and name not in named and callable(getattr(module, name))]
    return named + sorted(extra)


def draw_geometry_plot(model, state, out_path: Path, crossings) -> list:
    """The phase 2 single-panel plan view, byte-for-byte the same figure that script makes."""
    fig, ax = plt.subplots(figsize=(11, 11))
    violations = plot_design_state(ax, model, state, "Existing Conditions", crossings=crossings) or []
    ax.legend(handles=legend_handles(), loc="upper left", fontsize=8)
    fig.suptitle(f"{model.config['intersection']['name']} - Phase 2 geometry", fontsize=13)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return violations


def draw_before_after(model, baseline, state, scenario_name: str, out_path: Path, crossings) -> list:
    """The phase 3 before/after pair, with the same filenames the phase scripts write.

    Deliberately not a new set of artifacts: the review workflow is looking at
    phase2_geometry_plot.png and phase3_before_after_*.png, and a parallel set of
    similar-but-different pictures is how two views drift apart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    plot_design_state(axes[0], model, baseline, "Existing Conditions (Phase 2 baseline)", crossings=crossings)
    violations = plot_design_state(axes[1], model, state, f"Proposed Treatments ({scenario_name})",
                                    crossings=crossings) or []
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{model.config['intersection']['name']} - Before / After "
                 f"(NAD83 NJ State Plane, feet)", fontsize=13)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return violations


def build_site(site: str, render_3d: bool = False, dpi: int = 150,
               refresh_osm: bool = False) -> tuple[list[str], list]:
    """2D for a site's baseline and every proposal. Returns (failures, blender jobs).

    Runs in its own process (see main), so it returns its 3D jobs rather than appending to
    a shared list, and every site's Blender work is dispatched together at the end instead
    of serialising behind that site's plotting.
    """
    global PLOT_DPI
    PLOT_DPI = dpi
    if refresh_osm:
        # Set here rather than relied on from the parent: this is the process that does the
        # fetching, and on macOS the pool spawns fresh interpreters. Passing the flag through
        # the partial and setting the env var in the worker keeps the two in step whatever
        # the start method, instead of depending on how os.environ happens to be inherited.
        os.environ[REFRESH_ENV] = "1"
    blender_jobs: list = []
    failures = []
    out_dir = site_output_dir(site)
    started = time.perf_counter()

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        model = load_intersection_model(site=site)
        baseline = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)

    states = [("existing", "Existing Conditions", baseline)]
    for name in scenarios_for(site):
        with contextlib.redirect_stdout(quiet):
            states.append((scenario_label(name), name,
                            getattr(load_site_scenarios(site), name)(DesignState.from_model(model))))

    theme = buildings = None
    if render_3d:
        with contextlib.redirect_stdout(quiet):
            theme = build_default_theme()
            buildings = fetch_buildings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)

    for label, name, state in states:
        with contextlib.redirect_stdout(quiet):
            if label == "existing":
                violations = draw_geometry_plot(model, state, out_dir / "phase2_geometry_plot.png", crossings)
            else:
                suffix = "" if label == "proposed" else f"_{label}"
                violations = draw_before_after(model, baseline, state, name,
                                                out_dir / f"phase3_before_after{suffix}.png", crossings)
        for violation in violations:
            if violation.fatal:
                failures.append(f"{site}/{label}: {violation}")

        if render_3d:
            try:
                with contextlib.redirect_stdout(quiet):
                    geometry = export_scenario(model, state, name, out_dir / f"geometry_{label}.json",
                                                buildings=buildings, crossings=crossings, theme=theme)
                blender_jobs.append((geometry, out_dir / f"phase4_render_{label}.png"))
            except SceneInvariantError as e:
                failures.append(f"{site}/{label}: 3D export refused - {e}")

    status = "FAILED" if failures else "ok"
    print(f"  {site:22s} {len(states)} scenario(s)  {time.perf_counter() - started:5.1f}s  {status}\n"
          f"  {'':22s} {cache_summary()}")
    return failures, blender_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", help="limit to this site (repeatable)")
    parser.add_argument("--render-3d", action="store_true", help="also run the Blender renders")
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel worker processes, for both the 2D build and Blender (default 4)")
    parser.add_argument("--dpi", type=int, default=PLOT_DPI,
                        help=f"plan-view resolution (default {PLOT_DPI}; try 90 while iterating)")
    parser.add_argument("--refresh-osm", action="store_true",
                        help="re-pull every OSM layer from Overpass instead of serving output/.cache "
                             "- use after tracing kerbs/crossings/tactile paving in the OSM editor")
    args = parser.parse_args()

    sites = args.site or list_sites()
    started = time.perf_counter()
    print(f"Building {len(sites)} site(s){' + 3D renders' if args.render_3d else ''}"
          f"{' + re-pulling OSM' if args.refresh_osm else ''}")

    blender_jobs: list = []
    failures: list[str] = []
    # Sites share nothing but the on-disk OSM cache, so they build in parallel processes.
    # Threads would not help: the cost is matplotlib rasterization, which holds the GIL.
    # Under --refresh-osm the workers do write that cache, but each site's entries are keyed
    # by its own centre, so no two workers touch the same file.
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(sites))) as pool:
        for site_failures, site_jobs in pool.map(
                partial(build_site, render_3d=args.render_3d, dpi=args.dpi,
                        refresh_osm=args.refresh_osm), sites):
            failures += site_failures
            blender_jobs += site_jobs

    if blender_jobs:
        from scripts.phase4_render_3d import find_blender, render_all
        blender_bin = find_blender()
        print(f"Rendering {len(blender_jobs)} scene(s) via {blender_bin} ({args.jobs} at a time)")
        # Blender dominates the wall clock (~17 s per scene against ~0.25 s of geometry) and
        # each render is an independent subprocess, so this is where parallelism pays.
        chunks = [blender_jobs[i::args.jobs] for i in range(args.jobs)]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in [pool.submit(render_all, blender_bin, chunk) for chunk in chunks if chunk]:
                try:
                    future.result()
                except Exception as e:  # noqa: BLE001 - one bad render shouldn't sink the rest
                    failures.append(f"Blender: {e}")

    elapsed = time.perf_counter() - started
    if failures:
        print(f"\n{len(failures)} failure(s) in {elapsed:.1f}s:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"\nAll sites built cleanly in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
