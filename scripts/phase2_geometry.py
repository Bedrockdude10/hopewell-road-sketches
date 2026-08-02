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

from src.geometry.intersection import IntersectionModel, load_intersection_model
from src.geometry.model import build_pavement_polygon, sidewalk_span_ft
from src.provenance import ESTIMATED, FIELD_MEASURED, LABEL, OSM_DERIVED, leg_width_provenance
from src.render.plan_view import legend_handles, plot_design_state, sidewalk_lines_ft
from src.site import add_site_arg, site_output_dir
from src.render.props import data_gaps, hydrant_position_conflicts
from src.sources.osm_context import fetch_sidewalks, fetch_street_furniture, fetch_traffic_control
from src.geometry.treatments import DesignState

SIDEWALK_CONTEXT_RADIUS_M = 130  # matches src/render/plan_view.py, so both share one Overpass cache entry
MIN_CURB_TO_SIDEWALK_FT = 3.0  # half a narrow (6 ft) sidewalk built hard against the curb - the
                                # least space that can physically exist between a curb line and a
                                # sidewalk CENTERLINE. Any configured width leaving less than this
                                # per side is impossible, not merely optimistic.


def print_leg_summary(model: IntersectionModel, sidewalks: list[dict]):
    print("\n=== Leg widths used for geometry ===")
    walks = sidewalk_lines_ft(sidewalks)
    impossible = []
    for name, leg in model.legs.items():
        cfg = model.config["legs"][name]
        tier = leg_width_provenance(cfg)
        print(f"  {cfg['street_name']:45s} width={leg.curb_to_curb_ft:>6.1f} ft   [{LABEL[tier]}]")

        span = sidewalk_span_ft(leg.centerline, walks)
        if span:
            slack = (span["span_ft"] - leg.curb_to_curb_ft) / 2
            verdict = "ok" if slack >= MIN_CURB_TO_SIDEWALK_FT else "IMPOSSIBLE"
            print(f"      OSM sidewalks: {span['span_ft']:.1f} ft walk-to-walk "
                  f"-> leaves {slack:+.1f} ft/side between curb and sidewalk centerline [{verdict}]")
            if slack < MIN_CURB_TO_SIDEWALK_FT:
                impossible.append((name, tier, leg.curb_to_curb_ft, span["span_ft"], slack))
        if cfg.get("source"):
            print(f"      source: {' '.join(cfg['source'].split())[:140]}")

    for name, tier, width_ft, span_ft, slack in impossible:
        if tier == FIELD_MEASURED:
            # A field measurement is reality and is never overridden here - so when it
            # conflicts with OSM, the finding is about OSM, not about the leg.
            print(f"\n  CONFLICT on {name}: field-measured {width_ft:.1f} ft vs OSM sidewalks "
                  f"{span_ft:.1f} ft walk-to-walk ({slack:+.1f} ft/side). The measurement stands; "
                  f"OSM's sidewalk geometry here is suspect and worth fixing upstream.")
        else:
            print(f"\n  IMPLAUSIBLE WIDTH on {name}: {LABEL[tier]} {width_ft:.1f} ft, but OSM sidewalks are "
                  f"only {span_ft:.1f} ft apart ({slack:+.1f} ft/side to the sidewalk centerline). "
                  f"The curb cannot be outside the sidewalk - narrow this leg.")

    radius = model.config["treatments"]["existing_corner_radius_ft"]
    print(f"\nExisting corner radius used for fillets: {radius} ft "
          f"[{'ESTIMATE' if not model.config['treatments'].get('existing_corner_radius_source', '').startswith('Confirmed') else 'CONFIRMED'}]")


def plot(model: IntersectionModel, out_dir: Path, sidewalks: list[dict]):
    fig, ax = plt.subplots(figsize=(11, 11))
    baseline = DesignState.from_model(model)
    plot_design_state(ax, model, baseline, f"{model.config['intersection']['name']} - Phase 2 geometry",
                       sidewalks=sidewalks)
    ax.legend(handles=legend_handles(), loc="upper left", fontsize=8)
    ax.set_ylabel("Feet (EPSG:3424)")

    out_path = out_dir / "phase2_geometry_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    args = add_site_arg(argparse.ArgumentParser()).parse_args()
    model = load_intersection_model(site=args.site)
    try:
        sidewalks = fetch_sidewalks(model.center_wgs84, radius_m=SIDEWALK_CONTEXT_RADIUS_M)
    except RuntimeError as e:
        print(f"  WARNING: could not fetch OSM sidewalks ({e}).\n"
              f"  Widths will NOT be cross-checked against them this run.")
        sidewalks = []
    print_leg_summary(model, sidewalks)

    print("\n=== Nearest parcel per quadrant (corner / ROW reference) ===")
    print(model.corner_parcels[["quadrant", "PAMS_PIN", "BLOCK", "LOT", "dist_ft"]].to_string(index=False))

    print(f"\n=== Corner fillets built: {len(model.corner_fillets)} ===")
    for (a, b), pieces in model.corner_fillets.items():
        status = "OK" if "error" not in pieces else f"FAILED: {pieces['error']}"
        print(f"  {a} <-> {b}: {status}")

    plot(model, site_output_dir(args.site), sidewalks)

    print("\n=== OSM data gaps (what is being derived rather than sourced) ===")
    try:
        gaps = data_gaps(fetch_traffic_control(model.center_wgs84, radius_m=60),
                          fetch_street_furniture(model.center_wgs84, radius_m=130),
                          signalized=bool(model.config.get("signals")))
    except RuntimeError as e:
        gaps = [f"could not check - Overpass unreachable ({e})"]
    for gap in gaps or ["none - all traffic control, lighting and ADA data is sourced from OSM"]:
        print(f"  - {gap}")

    print("\n=== OSM vs. modelled geometry conflicts ===")
    try:
        pavement = build_pavement_polygon(DesignState.from_model(model).corner_fillets)
    except ValueError:
        pavement = None
    try:
        conflicts = hydrant_position_conflicts(
            fetch_street_furniture(model.center_wgs84, radius_m=SIDEWALK_CONTEXT_RADIUS_M), pavement)
    except RuntimeError as e:
        conflicts = [f"could not check - Overpass unreachable ({e})"]
    for note in conflicts or ["none"]:
        print(f"  - {note}")

    by_tier: dict[str, list[str]] = {}
    for name, cfg in model.config["legs"].items():
        by_tier.setdefault(leg_width_provenance(cfg), []).append(name)
    print("\n=== Width provenance ===")
    for tier in (FIELD_MEASURED, OSM_DERIVED, ESTIMATED):
        if by_tier.get(tier):
            print(f"  {LABEL[tier]:22s} {by_tier[tier]}")


if __name__ == "__main__":
    main()
