"""Data loading: NJDOT roadway network, Mercer County parcels, and intersection geocoding."""
from pathlib import Path

import geopandas as gpd
import requests
from geopy.geocoders import Nominatim
from shapely.geometry import Point, box

from src.geometry_model import NJ_STATE_PLANE_FT, WGS84, buffer_point_wgs84, reproject_to_state_plane

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Defaults only - a site's config.yaml (data_sources:) can point at different
# files entirely (e.g. a different county's parcels/road network), since
# nothing else in this module is specific to Mercer County or NJDOT's statewide file.
DEFAULT_ROAD_NETWORK_PATH = DATA_DIR / "NJ_Roadway_Network.geojson"
DEFAULT_PARCELS_PATH = DATA_DIR / "MercerCountyParcels.shp"

NOMINATIM_USER_AGENT = "hopewell-road-sketches-research/0.1 (contact: rollo.l@northeastern.edu)"
OVERPASS_USER_AGENT = NOMINATIM_USER_AGENT

# The public Overpass instances are shared/rate-limited infrastructure and
# occasionally 504 under load - try a couple of mirrors with retries before
# giving up, rather than failing the whole pipeline on a transient timeout.
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Nominatim geocodes a single street name to an arbitrary point along it (often
# a street midpoint), not to a cross-street intersection - "East Broad Street,
# Hopewell, NJ" lands ~900 ft from the actual Broad St/Greenwood Ave corner.
# Intersection sanity threshold, in degrees (~0.0007 deg ~= 230 ft at this latitude).
INTERSECTION_MATCH_TOLERANCE_DEG = 0.0007


def query_overpass(query: str, attempts_per_mirror: int = 2, timeout: int = 30) -> dict:
    """POST an Overpass QL query, retrying across mirrors on timeout/5xx errors."""
    last_error = None
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(attempts_per_mirror):
            try:
                resp = requests.post(
                    mirror, data={"data": query}, headers={"User-Agent": OVERPASS_USER_AGENT}, timeout=timeout
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                last_error = e
    raise RuntimeError(f"All Overpass mirrors failed after retries. Last error: {last_error}")


def approximate_geocode(query: str) -> Point:
    """Rough single-point geocode via Nominatim. Only precise enough to anchor a search bbox."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    location = geolocator.geocode(query, timeout=10)
    if location is None:
        raise ValueError(f"Could not geocode: {query!r}")
    return Point(location.longitude, location.latitude)


def geocode_intersection(street1: str, street2: str, anchor_query: str, search_radius_m: float = 1000) -> Point:
    """
    Resolve the real intersection point of two named streets by querying OSM/Overpass
    for way geometry within a search bbox and locating the shared endpoint node.

    This is more precise than address-string geocoding: street geocoders return a
    single point along the street (often its midpoint), not the cross-street corner.
    `anchor_query` is only used to center a search bbox via Nominatim.
    """
    anchor = approximate_geocode(anchor_query)
    west, south, east, north = buffer_point_wgs84(anchor, search_radius_m)

    query = f"""
    [out:json][timeout:25];
    (
      way["name"~"{street1}",i]({south},{west},{north},{east});
      way["name"~"{street2}",i]({south},{west},{north},{east});
    );
    out geom;
    """
    elements = query_overpass(query)["elements"]

    def endpoints(name_fragment: str) -> list[Point]:
        pts = []
        for el in elements:
            if name_fragment.lower() in el.get("tags", {}).get("name", "").lower():
                geom = el["geometry"]
                pts.append(Point(geom[0]["lon"], geom[0]["lat"]))
                pts.append(Point(geom[-1]["lon"], geom[-1]["lat"]))
        return pts

    pts1 = endpoints(street1)
    pts2 = endpoints(street2)
    if not pts1 or not pts2:
        raise ValueError(f"Could not find OSM ways matching {street1!r} and/or {street2!r} near {anchor_query!r}")

    p1, p2, dist = min(((a, b, a.distance(b)) for a in pts1 for b in pts2), key=lambda t: t[2])
    if dist > INTERSECTION_MATCH_TOLERANCE_DEG:
        raise ValueError(
            f"Closest endpoints of {street1!r} and {street2!r} are ~{dist * 364000:.0f} ft apart - "
            "does not look like a real intersection. Provide coordinates manually."
        )
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def load_road_network(
    bbox: tuple[float, float, float, float] | None = None, path: Path | str = DEFAULT_ROAD_NETWORK_PATH
) -> gpd.GeoDataFrame:
    """Load a roadway network GeoJSON (NJDOT's statewide SRI/SLD linear-referencing
    layer by default; pass `path` for a different one), optionally filtered to a
    WGS84 bbox (minx, miny, maxx, maxy)."""
    return gpd.read_file(path, bbox=bbox)


def load_parcels(
    bbox: tuple[float, float, float, float] | None = None, path: Path | str = DEFAULT_PARCELS_PATH
) -> gpd.GeoDataFrame:
    """Load a parcels/MOD-IV shapefile (Mercer County by default; pass `path` for a
    different one), optionally filtered to a bbox (in the shapefile's native CRS -
    reproject the bbox first if querying in WGS84)."""
    return gpd.read_file(path, bbox=bbox)


def load_parcels_near(
    center_wgs84: Point, radius_ft: float, path: Path | str = DEFAULT_PARCELS_PATH
) -> gpd.GeoDataFrame:
    """Load parcels within a square bbox (radius_ft) of a WGS84 point, reprojected
    to NJ State Plane. Full parcel polygons are kept (not circle-clipped) since
    partial lot fragments aren't meaningful for establishing ROW boundaries."""
    center_ft = gpd.GeoSeries([center_wgs84], crs=WGS84).to_crs(NJ_STATE_PLANE_FT).iloc[0]
    bbox_geom = box(center_ft.x - radius_ft, center_ft.y - radius_ft, center_ft.x + radius_ft, center_ft.y + radius_ft)
    # Passing a CRS-tagged GeoSeries (rather than a plain tuple) lets pyogrio resolve
    # the parcel shapefile's own (slightly different, HARN-less) NAD83 NJ State Plane CRS.
    parcels = load_parcels(bbox=gpd.GeoSeries([bbox_geom], crs=NJ_STATE_PLANE_FT), path=path)
    return reproject_to_state_plane(parcels)
