"""
Phase 1: resolve an intersection, load/clip/audit the road network around it,
and render a labeled plan-view sanity-check plot. This is the exploratory tool
you run ONCE per new site, before sites/<site>/config.yaml exists - its job is
to find out what NJDOT (or whatever road network file) actually recorded here,
so you know what needs supplementing with SLD/field data.

Usage:
  # A brand-new site with no config yet - pass the streets directly:
  python scripts/phase1_audit.py --street1 "Main St" --street2 "Oak Ave" \\
      --anchor "Main St, Sometown, NJ" [--road-network path/to/network.geojson]

  # An existing site - reads street1/street2/anchor_query/road_network from its config:
  python scripts/phase1_audit.py --site broad_st_greenwood
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import matplotlib.pyplot as plt

from src.sources.data_loader import DEFAULT_ROAD_NETWORK_PATH, geocode_intersection, load_road_network
from src.geometry.model import NJ_STATE_PLANE_FT, buffer_point_wgs84, clip_to_radius, reproject_to_state_plane
from src.site import list_sites, load_site_config, site_output_dir

ATTR_COLUMNS = [
    "OBJECTID", "SRI", "SRI_OLD", "MP_START", "MP_END", "DIRECTION", "SLD_NAME",
    "MEASURED_LENGTH", "PARENT_SRI", "PARENT_MP_START", "PARENT_MP_END", "ACTIVE",
    "YEAR_ACTIVE", "YEAR_RETIRED", "ROUTE_SUBTYPE", "ROAD_NUM",
]


def print_segment_audit(clipped_wgs84, center):
    """Print full attributes for the segments actually closest to the resolved
    center - for a brand-new site you don't know the SRIs in advance, so audit
    by proximity rather than a pre-known SRI list."""
    if clipped_wgs84.empty:
        print("  WARNING: nothing in the clipped set to audit.")
        return
    by_dist = clipped_wgs84.assign(_dist=clipped_wgs84.distance(center)).sort_values("_dist")
    cols = [c for c in ATTR_COLUMNS if c in by_dist.columns]
    for _, row in by_dist.head(4).iterrows():
        print(f"\n--- {row.get('SLD_NAME')} (SRI {row.get('SRI')}) ---")
        for col in cols:
            print(f"  {col:20s} = {row[col]}")

    print("\nAttributes NOT present on this layer, if blank above: lane count, road/lane width, surface type "
          "are commonly absent from NJDOT's SRI/SLD linear-referencing layer - check for a separate SLD PDF/field "
          "measurement before assuming a width.")


def plot_network(clipped_ft, center_ft, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 10))
    clipped_ft.plot(ax=ax, color="black", linewidth=2)

    for _, row in clipped_ft.iterrows():
        pt = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(
            row.get("SLD_NAME"), (pt.x, pt.y), fontsize=8, color="darkred",
            ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.scatter([center_ft.x], [center_ft.y], color="blue", zorder=5, s=40, label="Geocoded intersection")
    ax.set_title(title)
    ax.set_xlabel("Feet (EPSG:3424)")
    ax.set_ylabel("Feet (EPSG:3424)")
    ax.set_aspect("equal")
    ax.legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", help=f"Load street1/street2/anchor_query/road_network from an existing site's "
                                        f"config.yaml. Available: {', '.join(list_sites())}")
    parser.add_argument("--street1")
    parser.add_argument("--street2")
    parser.add_argument("--anchor", help="Address/place string Nominatim can geocode, to anchor the OSM search bbox")
    parser.add_argument("--road-network", default=str(DEFAULT_ROAD_NETWORK_PATH))
    parser.add_argument("--clip-radius-m", type=float, default=150)
    parser.add_argument("--out-name", default="phase1_network_plot.png")
    args = parser.parse_args()

    if args.site:
        cfg = load_site_config(args.site)["intersection"]
        street1 = args.street1 or cfg["street1"]
        street2 = args.street2 or cfg["street2"]
        anchor = args.anchor or cfg["anchor_query"]
        out_dir = site_output_dir(args.site)
    else:
        if not (args.street1 and args.street2 and args.anchor):
            parser.error("Pass --site, or all of --street1/--street2/--anchor for a brand-new site.")
        street1, street2, anchor = args.street1, args.street2, args.anchor
        out_dir = site_output_dir("_scratch")

    print(f"Resolving intersection: {street1} & {street2} (anchor query: {anchor!r})")
    center = geocode_intersection(street1, street2, anchor)
    print(f"  -> lon={center.x:.7f}, lat={center.y:.7f} (resolved via OSM way-endpoint match, not address geocoding)")
    print("  Save this as intersection.center_wgs84 in the site's config.yaml.")

    bbox = buffer_point_wgs84(center, args.clip_radius_m * 1.3)
    print("\nLoading road network (bbox-filtered read)...")
    network = load_road_network(bbox=bbox, path=args.road_network)
    print(f"  -> {len(network)} features in load bbox")

    clipped = clip_to_radius(network, center, args.clip_radius_m)
    print(f"  -> {len(clipped)} features within {args.clip_radius_m}m radius")

    print(f"\n=== Attribute audit: segments nearest {street1} & {street2} ===")
    print_segment_audit(clipped, center)

    clipped_ft = reproject_to_state_plane(clipped)
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs(NJ_STATE_PLANE_FT).iloc[0]

    title = f"{street1} & {street2}\n(clipped to {args.clip_radius_m:.0f}m radius, NAD83 NJ State Plane, feet)"
    plot_network(clipped_ft, center_ft, title, out_dir / args.out_name)


if __name__ == "__main__":
    main()
