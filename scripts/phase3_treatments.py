"""
Phase 3: apply composable parametric treatments (curb extensions / tightened turn
radii, a raised crossing, a pedestrian refuge island) to the Phase 2 baseline
geometry, and render a before/after plan-view comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.intersection import load_intersection_model
from src.render import legend_handles, plot_design_state
from src.scenarios import build_demo_scenario
from src.treatments import DesignState

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    model = load_intersection_model()
    baseline = DesignState.from_model(model)
    scenario = build_demo_scenario(baseline)

    print("=== Treatments applied ===")
    for note in scenario.notes:
        print(f"  {note}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    plot_design_state(axes[0], model, baseline, "Existing Conditions (Phase 2 baseline)")
    plot_design_state(axes[1], model, scenario, "Proposed Treatments")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Broad St & Greenwood Ave, Hopewell Borough, NJ - Before / After (NAD83 NJ State Plane, feet)",
                 fontsize=13)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase3_before_after.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved before/after render to {out_path}")


if __name__ == "__main__":
    main()
