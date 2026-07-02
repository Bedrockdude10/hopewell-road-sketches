"""OSM/Overpass context data (building massing) for presentation-quality 3D
renders. This is background dressing only - never used for the authoritative
curb/pavement geometry, which comes from NJDOT SLD + field measurement (see
src/sources/data_loader.py for why OSM's own data isn't trusted for that)."""
import hashlib
import json
from pathlib import Path

from shapely.geometry import Point

from src.sources.data_loader import query_overpass
from src.geometry.model import buffer_point_wgs84

DEFAULT_BUILDING_HEIGHT_M = 7.0  # ~2 stories, typical for small-borough Main St buildings
METERS_PER_LEVEL = 3.0
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "output" / ".cache"  # src/sources/osm_context.py -> repo root


def fetch_buildings(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM building footprints within radius_m of a WGS84 point.
    Returns [{"coords_wgs84": [(lon, lat), ...], "height_m": float}, ...].

    Building footprints don't change between iterations of the same scene, and
    the public Overpass mirrors are slow/flaky - cache the raw response to disk
    keyed by (center, radius) so re-rendering doesn't re-hit the network."""
    cache_key = hashlib.sha1(f"{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"buildings_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    way["building"]({south},{west},{north},{east});
    out geom tags;
    """
    elements = query_overpass(query)["elements"]

    buildings = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 3:
            continue
        tags = el.get("tags", {})
        height_m = _estimate_height(tags)
        coords = [(pt["lon"], pt["lat"]) for pt in geom]
        buildings.append({"coords_wgs84": coords, "height_m": height_m})

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(buildings, f)
    return buildings


def fetch_crossings(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped pedestrian crossings (highway=footway/footway=crossing
    ways) within radius_m of a WGS84 point - real surveyed crosswalk lines,
    rather than a geometric estimate of where one probably is.
    Returns [{"coords_wgs84": [(lon, lat), ...], "tags": {...}}, ...]."""
    cache_key = hashlib.sha1(f"crossings,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"crossings_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    way["footway"="crossing"]({south},{west},{north},{east});
    out geom tags;
    """
    elements = query_overpass(query)["elements"]

    crossings = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        crossings.append({"coords_wgs84": [(pt["lon"], pt["lat"]) for pt in geom], "tags": el.get("tags", {})})

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(crossings, f)
    return crossings


def _estimate_height(tags: dict) -> float:
    if tags.get("height"):
        try:
            return float("".join(c for c in tags["height"] if c.isdigit() or c == "."))
        except ValueError:
            pass
    if tags.get("building:levels"):
        try:
            return float(tags["building:levels"]) * METERS_PER_LEVEL
        except ValueError:
            pass
    return DEFAULT_BUILDING_HEIGHT_M
