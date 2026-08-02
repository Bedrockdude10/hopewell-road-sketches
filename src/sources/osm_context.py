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
    out geom;
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
    # "v2": now carries node_ids, needed to detect nodes shared with kerb ways.
    cache_key = hashlib.sha1(
        f"crossings,v2,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"crossings_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    way["footway"="crossing"]({south},{west},{north},{east});
    out geom;
    """
    elements = query_overpass(query)["elements"]

    crossings = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        crossings.append({"coords_wgs84": [(pt["lon"], pt["lat"]) for pt in geom], "tags": el.get("tags", {}),
                           "node_ids": el.get("nodes", [])})

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(crossings, f)
    return crossings


def fetch_sidewalks(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped sidewalk centerlines (highway=footway/footway=sidewalk ways)
    within radius_m of a WGS84 point.
    Returns [{"coords_wgs84": [(lon, lat), ...], "tags": {...}}, ...].

    These are real surveyed geometry, and they are what OSM's crossing ways actually
    connect to - a crossing runs sidewalk-centerline to sidewalk-centerline, not
    curb to curb. Measured across this project's sites, a crossing way's length
    matches the sidewalk-to-sidewalk span to within 0.4-2.6 ft, which is what makes
    the sidewalks usable as an independent check on a leg's configured width: the
    curb line must sit INSIDE them (see src/geometry/model.py:sidewalk_span_ft).

    What they cannot do is give the width directly. Calibrated against the only two
    field-measured legs in the project, the gap between sidewalk centerline and curb
    is 11.8 ft/side on one and 4.0 ft/side on the other - on the same street, 100 ft
    apart. So this is a bound and a sanity check, not a measurement.
    """
    cache_key = hashlib.sha1(f"sidewalks,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"sidewalks_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    way["footway"="sidewalk"]({south},{west},{north},{east});
    out geom;
    """
    elements = query_overpass(query)["elements"]

    sidewalks = []
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        sidewalks.append({"coords_wgs84": [(pt["lon"], pt["lat"]) for pt in geom], "tags": el.get("tags", {})})

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(sidewalks, f)
    return sidewalks


def fetch_traffic_control(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped traffic control nodes (highway=traffic_signals / stop / give_way)
    within radius_m of a WGS84 point.
    Returns [{"lon": float, "lat": float, "tags": {...}}, ...].

    Real surveyed control, instead of guessing. src/render/props.py previously placed one
    stop sign per approach on an arbitrary side, admitting in its own docstring that this
    was "not a real traffic-direction/engineering placement study". At Columbia &
    Princeton that guess is simply wrong: OSM maps exactly two stop nodes, both on
    Columbia Ave, because Princeton Ave (CR 569) is the free-flowing through street. The
    guess put stop signs on all four approaches, including the two that don't have them.

    A stop/give_way node sits ON the road way at the approach it governs, so it gives
    both the leg and the real distance from the junction. A traffic_signals node normally
    sits at the junction itself and says only THAT the junction is signalized - not where
    the poles are - so per-corner signal hardware still comes from the site config's
    `signals` block, which is direct observation (see props.py:_traffic_signal_props).

    highway=crossing nodes are included too. They are not signs, and _osm_control_props
    ignores them, but they are where OSM records the pedestrian-facing detail that lives
    on the node rather than the crossing way: tactile_paving (ADA truncated domes),
    button_operated (a pushbutton-actuated ped phase) and crossing:island. Fetching the
    crossing WAYS alone misses all of it - which made an earlier version of data_gaps()
    report "no ADA data" at Broad/Greenwood, where all four crossings are in fact tagged
    tactile_paving=yes.
    """
    # "v2" in the key: this query grew to include highway=crossing nodes, and a cache
    # entry written by the earlier narrower query would silently under-report.
    cache_key = hashlib.sha1(
        f"traffic_control,v2,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"traffic_control_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    node["highway"~"^(traffic_signals|stop|give_way|crossing)$"]({south},{west},{north},{east});
    out geom;
    """
    nodes = [
        {"lon": el["lon"], "lat": el["lat"], "tags": el.get("tags", {})}
        for el in query_overpass(query)["elements"] if "lon" in el and "lat" in el
    ]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(nodes, f)
    return nodes


def fetch_street_furniture(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped street furniture (highway=street_lamp, emergency=fire_hydrant,
    natural=tree) within radius_m of a WGS84 point.
    Returns [{"lon": float, "lat": float, "tags": {...}}, ...].

    STREET LAMPS ARE NOT MAPPED AT ANY OF THIS PROJECT'S FOUR SITES - a survey of every
    OSM element within 80 m of each junction found zero highway=street_lamp nodes. This
    fetcher exists so that a site where they ARE mapped gets real pole positions instead
    of the derived one-per-corner placement (src/render/props.py:_corner_streetlight_props),
    and so the absence is reported rather than quietly papered over. Mapping the lamps in
    OSM is what would improve these four.
    """
    # "v2": the query grew to include natural=tree, so an entry written by the earlier
    # narrower query would silently report no trees.
    cache_key = hashlib.sha1(
        f"street_furniture,v2,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"street_furniture_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    (
      node["highway"="street_lamp"]({south},{west},{north},{east});
      node["emergency"="fire_hydrant"]({south},{west},{north},{east});
      node["natural"="tree"]({south},{west},{north},{east});
    );
    out geom;
    """
    nodes = [
        {"lon": el["lon"], "lat": el["lat"], "tags": el.get("tags", {})}
        for el in query_overpass(query)["elements"] if "lon" in el and "lat" in el
    ]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(nodes, f)
    return nodes


def fetch_kerbs(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped kerb lines and kerb nodes (barrier=kerb) within radius_m.
    Returns [{"coords_wgs84": [(lon, lat), ...] | None, "lon"/"lat" for nodes, "tags": {...}}].

    A traced kerb way is the most direct geometry this project can get for two things it
    otherwise has to guess or infer:

      * The CORNER RADIUS. Nothing else in OSM carries it - there is no radius tag, no
        area:highway coverage here, and the sidewalk ways turn at a single sharp vertex.
        A traced kerb IS the curb, so a circle fitted to it needs no verge subtraction.
      * TACTILE PAVING position. A kerb way tagged tactile_paving=yes is the real ramp,
        so pads can be placed on it directly instead of inferred from where a crossing
        line leaves our (over-wide) pavement polygon.
    """
    # "v2": now carries node_ids, so nodes shared with crossing ways can be found.
    cache_key = hashlib.sha1(
        f"kerbs,v2,{center_wgs84.x:.6f},{center_wgs84.y:.6f},{radius_m}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"kerbs_{cache_key}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    query = f"""
    [out:json][timeout:25];
    (
      way["barrier"="kerb"]({south},{west},{north},{east});
      node["barrier"="kerb"]({south},{west},{north},{east});
    );
    out geom;
    """
    kerbs = []
    for el in query_overpass(query)["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way":
            geom = el.get("geometry")
            if not geom or len(geom) < 3:
                continue  # need 3+ points to fit anything
            kerbs.append({"coords_wgs84": [(p["lon"], p["lat"]) for p in geom], "tags": tags,
                           "id": el["id"], "node_ids": el.get("nodes", [])})
        elif "lon" in el:
            kerbs.append({"coords_wgs84": None, "lon": el["lon"], "lat": el["lat"], "tags": tags,
                           "id": el["id"]})

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(kerbs, f)
    return kerbs


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
