"""
Phase 3: apply one of a site's treatment scenarios (sites/<site>/scenarios.py)
to the Phase 2 baseline geometry, and render a before/after plan-view comparison.

Usage: python scripts/phase3_treatments.py [--site broad_st_greenwood] [--scenario build_demo_scenario]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.geometry.intersection import load_intersection_model
from src.render.plan_view import legend_handles, plot_design_state
from src.site import DEFAULT_SCENARIO, add_scenario_arg, add_site_arg, load_site_scenarios, scenario_label, site_output_dir
from src.geometry.treatments import DesignState


def main():
    args = add_scenario_arg(add_site_arg(argparse.ArgumentParser())).parse_args()
    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    build_scenario = getattr(load_site_scenarios(args.site), args.scenario)
    scenario = build_scenario(baseline)

    print(f"=== Treatments applied ({args.scenario}) ===")
    for note in scenario.notes:
        print(f"  {note}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    plot_design_state(axes[0], model, baseline, "Existing Conditions (Phase 2 baseline)")
    plot_design_state(axes[1], model, scenario, f"Proposed Treatments ({args.scenario})")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{model.config['intersection']['name']} - Before / After (NAD83 NJ State Plane, feet)", fontsize=13)

    # Default scenario keeps the original filename (phase3_before_after.png, referenced
    # in README.md's Quick start) - non-default scenarios get their own labeled file
    # instead of clobbering it, so multiple proposals' renders can coexist.
    suffix = "" if args.scenario == DEFAULT_SCENARIO else f"_{scenario_label(args.scenario)}"
    out_path = site_output_dir(args.site) / f"phase3_before_after{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved before/after render to {out_path}")


if __name__ == "__main__":
    main()
