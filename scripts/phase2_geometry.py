"""
Phase 2: reconcile the road network with the NJDOT SLD + ground-truth measurements
(config/intersection_config.yaml), clip parcels to establish ROW context at the
four corners, and build curb-line + rounded-corner geometry as Shapely polygons/lines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.intersection import IntersectionModel, load_intersection_model
from src.render import legend_handles, plot_design_state
from src.treatments import DesignState

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def print_leg_summary(model: IntersectionModel):
    print("\n=== Leg widths used for geometry ===")
    for name, leg in model.legs.items():
        cfg = model.config["legs"][name]
        status = "CONFIRMED" if cfg.get("confirmed") else "ESTIMATE / PLACEHOLDER"
        print(f"  {cfg['street_name']:45s} width={leg.curb_to_curb_ft:>6.1f} ft   [{status}]")
        if cfg.get("source"):
            print(f"      source: {' '.join(cfg['source'].split())[:140]}")

    radius = model.config["treatments"]["existing_corner_radius_ft"]
    print(f"\nExisting corner radius used for fillets: {radius} ft "
          f"[{'ESTIMATE' if not model.config['treatments'].get('existing_corner_radius_source', '').startswith('Confirmed') else 'CONFIRMED'}]")


def plot(model: IntersectionModel):
    fig, ax = plt.subplots(figsize=(11, 11))
    baseline = DesignState.from_model(model)
    plot_design_state(ax, model, baseline, "Broad St & Greenwood Ave, Hopewell Borough, NJ - Phase 2 geometry")
    ax.legend(handles=legend_handles(), loc="upper left", fontsize=8)
    ax.set_ylabel("Feet (EPSG:3424)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase2_geometry_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    model = load_intersection_model()
    print_leg_summary(model)

    print("\n=== Nearest parcel per quadrant (corner / ROW reference) ===")
    print(model.corner_parcels[["quadrant", "PAMS_PIN", "BLOCK", "LOT", "dist_ft"]].to_string(index=False))

    print(f"\n=== Corner fillets built: {len(model.corner_fillets)} ===")
    for (a, b), pieces in model.corner_fillets.items():
        status = "OK" if "error" not in pieces else f"FAILED: {pieces['error']}"
        print(f"  {a} <-> {b}: {status}")

    plot(model)

    unconfirmed = [name for name, cfg in model.config["legs"].items() if not cfg.get("confirmed")]
    print(f"\nNOTE: legs still using an estimate rather than a field measurement: {unconfirmed}")


if __name__ == "__main__":
    main()
