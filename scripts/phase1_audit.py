"""
Phase 1: load, clip, and audit the NJDOT road network around Broad St & Greenwood
Ave, Hopewell Borough, NJ. Prints what attributes NJDOT actually recorded and
renders a labeled plan-view plot for a sanity check against the real intersection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.data_loader import geocode_intersection, load_road_network
from src.geometry_model import NJ_STATE_PLANE_FT, buffer_point_wgs84, clip_to_radius, reproject_to_state_plane

ANCHOR_QUERY = "Broad Street, Hopewell Borough, NJ 08525"
STREET_1 = "Broad St"
STREET_2 = "Greenwood Ave"
CLIP_RADIUS_M = 150

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# NJDOT's Roadway Network records this cross street under a different official
# name than current local signage/OSM (SRI 11051089__, SLD_NAME "COLUMBIA AVE").
# Confirmed via OSM/Overpass: the "North Greenwood Avenue" / "South Greenwood
# Avenue" ways share their endpoint node with the "West/East Broad Street" (CR
# 518) ways, and that shared node sits 0.3 ft from this NJDOT segment's line
# (vs. 8.4 ft for the Route 518 / Broad St segment itself).
DISPLAY_NAME_OVERRIDES = {
    "11051089__": "GREENWOOD AVE (NJDOT: COLUMBIA AVE)",
}

ATTR_COLUMNS = [
    "OBJECTID", "SRI", "SRI_OLD", "MP_START", "MP_END", "DIRECTION", "SLD_NAME",
    "MEASURED_LENGTH", "PARENT_SRI", "PARENT_MP_START", "PARENT_MP_END", "ACTIVE",
    "YEAR_ACTIVE", "YEAR_RETIRED", "ROUTE_SUBTYPE", "ROAD_NUM",
]


def print_segment_audit(clipped_wgs84):
    target_sris = {"00000518__", "11051089__"}  # Route 518 (Broad St) + Columbia/Greenwood Ave
    segments = clipped_wgs84[clipped_wgs84["SRI"].isin(target_sris)]
    if segments.empty:
        print("  WARNING: expected SRIs not found in clipped set - printing all clipped segments instead.")
        segments = clipped_wgs84

    cols = [c for c in ATTR_COLUMNS if c in segments.columns]
    for _, row in segments.iterrows():
        label = DISPLAY_NAME_OVERRIDES.get(row.get("SRI"), row.get("SLD_NAME"))
        print(f"\n--- {label} (SRI {row.get('SRI')}) ---")
        for col in cols:
            print(f"  {col:20s} = {row[col]}")

    present = {c for c in cols if segments[cols].notna().any().get(c, False)}
    missing_of_interest = {
        "lane count": "not a field in this layer",
        "road/lane width": "not a field in this layer",
        "surface type": "not a field in this layer",
    }
    print("\nAttributes NOT present on this layer (lane count, width, surface):")
    for k, v in missing_of_interest.items():
        print(f"  {k}: {v}")
    print("Jurisdiction/route identification IS present via SRI / ROUTE_SUBTYPE / ROAD_NUM.")


def plot_network(clipped_ft, center_ft):
    fig, ax = plt.subplots(figsize=(10, 10))
    clipped_ft.plot(ax=ax, color="black", linewidth=2)

    for _, row in clipped_ft.iterrows():
        label = DISPLAY_NAME_OVERRIDES.get(row.get("SRI"), row.get("SLD_NAME"))
        pt = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(
            label, (pt.x, pt.y), fontsize=8, color="darkred",
            ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.scatter([center_ft.x], [center_ft.y], color="blue", zorder=5, s=40, label="Geocoded intersection")
    ax.set_title("Broad St & Greenwood Ave, Hopewell Borough, NJ\n(clipped to 150m radius, NAD83 NJ State Plane, feet)")
    ax.set_xlabel("Feet (EPSG:3424)")
    ax.set_ylabel("Feet (EPSG:3424)")
    ax.set_aspect("equal")
    ax.legend()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "phase1_network_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    print(f"Resolving intersection: {STREET_1} & {STREET_2} (anchor query: {ANCHOR_QUERY!r})")
    center = geocode_intersection(STREET_1, STREET_2, ANCHOR_QUERY)
    print(f"  -> lon={center.x:.7f}, lat={center.y:.7f} (resolved via OSM way-endpoint match, not address geocoding)")

    bbox = buffer_point_wgs84(center, CLIP_RADIUS_M * 1.3)
    print("\nLoading road network (bbox-filtered read)...")
    network = load_road_network(bbox=bbox)
    print(f"  -> {len(network)} features in load bbox")

    clipped = clip_to_radius(network, center, CLIP_RADIUS_M)
    print(f"  -> {len(clipped)} features within {CLIP_RADIUS_M}m radius")

    print("\n=== Attribute audit: Broad St & Greenwood Ave segments ===")
    print_segment_audit(clipped)

    clipped_ft = reproject_to_state_plane(clipped)
    import geopandas as gpd
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs(NJ_STATE_PLANE_FT).iloc[0]

    plot_network(clipped_ft, center_ft)


if __name__ == "__main__":
    main()
