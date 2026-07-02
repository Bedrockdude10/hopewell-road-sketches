"""
Phase 3: apply a site's demo treatment scenario (sites/<site>/scenarios.py) to
the Phase 2 baseline geometry, and render a before/after plan-view comparison.

Usage: python scripts/phase3_treatments.py [--site broad_st_greenwood]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.intersection import load_intersection_model
from src.render import legend_handles, plot_design_state
from src.site import add_site_arg, load_site_scenarios, site_output_dir
from src.treatments import DesignState


def main():
    args = add_site_arg(argparse.ArgumentParser()).parse_args()
    model = load_intersection_model(site=args.site)
    baseline = DesignState.from_model(model)
    scenario = load_site_scenarios(args.site).build_demo_scenario(baseline)

    print("=== Treatments applied ===")
    for note in scenario.notes:
        print(f"  {note}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    plot_design_state(axes[0], model, baseline, "Existing Conditions (Phase 2 baseline)")
    plot_design_state(axes[1], model, scenario, "Proposed Treatments")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{model.config['intersection']['name']} - Before / After (NAD83 NJ State Plane, feet)", fontsize=13)

    out_path = site_output_dir(args.site) / "phase3_before_after.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved before/after render to {out_path}")


if __name__ == "__main__":
    main()
