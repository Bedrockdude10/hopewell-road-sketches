"""
Phase 2: reconcile the road network with the site's SLD + ground-truth
measurements (sites/<site>/config.yaml), clip parcels to establish ROW context
at the corners, and build curb-line + rounded-corner geometry as Shapely
polygons/lines.

Usage: python scripts/phase2_geometry.py [--site broad_st_greenwood]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.intersection import IntersectionModel, load_intersection_model
from src.render import legend_handles, plot_design_state
from src.site import add_site_arg, site_output_dir
from src.treatments import DesignState


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


def plot(model: IntersectionModel, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 11))
    baseline = DesignState.from_model(model)
    plot_design_state(ax, model, baseline, f"{model.config['intersection']['name']} - Phase 2 geometry")
    ax.legend(handles=legend_handles(), loc="upper left", fontsize=8)
    ax.set_ylabel("Feet (EPSG:3424)")

    out_path = out_dir / "phase2_geometry_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    args = add_site_arg(argparse.ArgumentParser()).parse_args()
    model = load_intersection_model(site=args.site)
    print_leg_summary(model)

    print("\n=== Nearest parcel per quadrant (corner / ROW reference) ===")
    print(model.corner_parcels[["quadrant", "PAMS_PIN", "BLOCK", "LOT", "dist_ft"]].to_string(index=False))

    print(f"\n=== Corner fillets built: {len(model.corner_fillets)} ===")
    for (a, b), pieces in model.corner_fillets.items():
        status = "OK" if "error" not in pieces else f"FAILED: {pieces['error']}"
        print(f"  {a} <-> {b}: {status}")

    plot(model, site_output_dir(args.site))

    unconfirmed = [name for name, cfg in model.config["legs"].items() if not cfg.get("confirmed")]
    print(f"\nNOTE: legs still using an estimate rather than a field measurement: {unconfirmed}")


if __name__ == "__main__":
    main()
