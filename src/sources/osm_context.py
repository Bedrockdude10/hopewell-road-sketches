"""OSM/Overpass context data (building massing) for presentation-quality 3D
renders. This is background dressing only - never used for the authoritative
curb/pavement geometry, which comes from NJDOT SLD + field measurement (see
src/sources/data_loader.py for why OSM's own data isn't trusted for that)."""
import hashlib
import json
import os
import time
from pathlib import Path

from shapely.geometry import Point

from src.sources.data_loader import query_overpass
from src.geometry.model import buffer_point_wgs84

DEFAULT_BUILDING_HEIGHT_M = 7.0  # ~2 stories, typical for small-borough Main St buildings
METERS_PER_LEVEL = 3.0
# Where fetched OSM responses are cached. Overridable so the test suite can point at a
# committed fixture set and run hermetically - see tests/conftest.py and HOPEWELL_OFFLINE.
CACHE_DIR = Path(os.environ.get(
    "HOPEWELL_OSM_CACHE",
    Path(__file__).resolve().parent.parent.parent / "output" / ".cache"))

REFRESH_ENV = "HOPEWELL_REFRESH_OSM"

# Second-level cache, in memory. The disk cache already avoids the network, but a batch
# build asks for the same junction's kerbs and crossings once per scenario - 27 times over
# for the four sites - and re-reading and re-parsing the same JSON each time is pure waste.
# Keyed by the same cache key the disk layer uses, so it can never disagree with it.
_MEMO: dict[str, list] = {}

# Cache files this process fetched and wrote itself, and which a refresh therefore has no
# reason to pull again. A refresh means "one round trip per layer", not "one per call": a
# single site build asks for the same junction's kerbs and crossings once per scenario, ~27
# times over, and the public Overpass mirrors are shared, rate-limited infrastructure.
_REFRESHED: set[str] = set()

# What this process actually got each OSM layer from: cache file -> mtime it had when read,
# or None if it was pulled fresh from Overpass. Recorded at the point of use rather than
# recomputed later so the staleness report (cache_summary) describes the files that really
# fed this build - an independently derived key would drift the moment a query gains a "v3"
# and would then reassure the user about a file nobody reads.
_CACHE_READS: dict[Path, float | None] = {}

_warned: set[str] = set()


def refresh_requested() -> bool:
    """True when this process was told to ignore the cache and re-pull from Overpass.

    Read from the environment instead of threaded through every fetch signature: the six
    fetchers are called from a dozen places (src/render/export.py, src/render/plan_view.py,
    src/geometry/intersection.py, the phase scripts, tests), and a parameter that has to be
    forwarded at each of them is a parameter someone eventually forgets - which lands you
    right back at "I traced the kerb in OSM and the render didn't change".

    Refusing to refresh while HOPEWELL_OFFLINE is set is not politeness: the test suite
    runs against the committed fixture cache, and honouring a stray refresh there would
    turn every fetch into an OfflineCacheMiss.
    """
    if not os.environ.get(REFRESH_ENV):
        return False
    if os.environ.get("HOPEWELL_OFFLINE"):
        _warn_once(f"{REFRESH_ENV} ignored: HOPEWELL_OFFLINE is set, so the cached responses "
                   f"in {CACHE_DIR} are all this process is allowed to see.")
        return False
    return True


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"  {message}")


def _cache_path(kind: str, center_wgs84: Point, radius_m: float, version: str = "") -> Path:
    """Disk cache filename for one layer at one (centre, radius).

    `version` bumps a layer's key when its Overpass query widens - an entry written by the
    older, narrower query would otherwise be served forever and silently under-report.
    """
    # buildings' key carries no kind prefix: it was the first fetcher and its key predates
    # the convention. Spelling it the tidy way now would orphan every cached response and
    # every committed fixture for no gain.
    parts = [] if kind == "buildings" else [kind]
    if version:
        parts.append(version)
    signature = ",".join(parts + [f"{center_wgs84.x:.6f}", f"{center_wgs84.y:.6f}", f"{radius_m}"])
    return CACHE_DIR / f"{kind}_{hashlib.sha1(signature.encode()).hexdigest()[:16]}.json"


def _cache_hit(cache_path: Path) -> bool:
    """Whether `cache_path` may be served instead of going to Overpass, recording the read."""
    if not cache_path.exists():
        return False
    if refresh_requested() and str(cache_path) not in _REFRESHED:
        return False
    # setdefault, not assignment: once a layer has been re-pulled this run its entry is
    # None ("fresh"), and the subsequent memo-backed reads of the file we just wrote must
    # not overwrite that with an age of zero seconds.
    _CACHE_READS.setdefault(cache_path, cache_path.stat().st_mtime)
    return True


def _write_cache(cache_path: Path, data: list) -> None:
    """Persist a freshly fetched layer, keeping both cache layers in step."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    # _MEMO is keyed by path and outlives the write, so a refresh that only rewrote the
    # file would go on serving the stale parse to the rest of this process - the same
    # silent-stale bug, one layer down.
    _MEMO[str(cache_path)] = data
    _REFRESHED.add(str(cache_path))
    _CACHE_READS[cache_path] = None


def _memoized(cache_path: Path, build):
    """Return the parsed response for `cache_path`, from memory if it's already been read."""
    key = str(cache_path)
    if key not in _MEMO:
        _MEMO[key] = build()
    return _MEMO[key]


def cache_summary() -> str:
    """One line saying how old the OSM data this process just used is.

    The failure this exists for: the user traces a kerb, a crossing or a tactile-paving pad
    in OSM, re-runs the build, and sees no change - because the disk cache is keyed only by
    (centre, radius) and never expires, so the edit is invisible until someone thinks to
    delete output/.cache by hand. The ground truth was there; it just never reached the
    render, which is the class of bug this project keeps hitting. An age printed on every
    build turns a silent trap into a number you can look at.
    """
    if not _CACHE_READS:
        return "OSM cache: no OSM layers were read"
    ages = [time.time() - mtime for mtime in _CACHE_READS.values() if mtime is not None]
    fresh = sum(1 for mtime in _CACHE_READS.values() if mtime is None)
    if not ages:
        if refresh_requested():
            return f"OSM cache: re-pulled all {fresh} layer(s) from Overpass"
        return f"OSM cache: nothing was cached - pulled {fresh} layer(s) fresh from Overpass"
    line = f"OSM cache: {len(ages)} layer(s), oldest {_humanize_age(max(ages))} old"
    if fresh:
        return f"{line}; {fresh} pulled fresh from Overpass"
    return f"{line} (--refresh-osm to re-pull)"


def _humanize_age(seconds: float) -> str:
    for unit, size in (("day", 86400.0), ("hour", 3600.0), ("minute", 60.0)):
        if seconds >= size:
            count = round(seconds / size)
            return f"{count} {unit}" + ("s" if count != 1 else "")
    return "less than a minute"


def fetch_buildings(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM building footprints within radius_m of a WGS84 point.
    Returns [{"coords_wgs84": [(lon, lat), ...], "height_m": float}, ...].

    Building footprints don't change between iterations of the same scene, and
    the public Overpass mirrors are slow/flaky - cache the raw response to disk
    keyed by (center, radius) so re-rendering doesn't re-hit the network."""
    cache_path = _cache_path("buildings", center_wgs84, radius_m)

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
        _write_cache(cache_path, buildings)
    return buildings


def fetch_crossings(center_wgs84: Point, radius_m: float, use_cache: bool = True) -> list[dict]:
    """Fetch OSM-mapped pedestrian crossings (highway=footway/footway=crossing
    ways) within radius_m of a WGS84 point - real surveyed crosswalk lines,
    rather than a geometric estimate of where one probably is.
    Returns [{"coords_wgs84": [(lon, lat), ...], "tags": {...}}, ...]."""
    # "v2": now carries node_ids, needed to detect nodes shared with kerb ways.
    cache_path = _cache_path("crossings", center_wgs84, radius_m, version="v2")

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
        _write_cache(cache_path, crossings)
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
    cache_path = _cache_path("sidewalks", center_wgs84, radius_m)

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
        _write_cache(cache_path, sidewalks)
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
    cache_path = _cache_path("traffic_control", center_wgs84, radius_m, version="v2")

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
        _write_cache(cache_path, nodes)
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
    cache_path = _cache_path("street_furniture", center_wgs84, radius_m, version="v2")

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
        _write_cache(cache_path, nodes)
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
    # "v3": keeps 2-vertex ways, which v2 dropped (see below).
    cache_path = _cache_path("kerbs", center_wgs84, radius_m, version="v3")

    if use_cache and _cache_hit(cache_path):
        return _memoized(cache_path, lambda: json.loads(cache_path.read_text()))

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
            # A 2-vertex way is a straight run of kerb, and straight runs are most of what
            # gets traced along a block. Dropping them here (the old "need 3+ points to fit
            # anything" rule, which was really a circle-fitting precondition applied to the
            # wrong layer) threw away 12 of the 23 traced ways at Columbia & Princeton and
            # at E Broad & Princeton, and 5 of 12 at W Broad & Louellen - so the curb lines
            # fell back to centerline offsets on sides the surveyor had actually traced.
            # Consumers that genuinely need 3+ points (circle fitting) check for themselves.
            if not geom or len(geom) < 2:
                continue
            kerbs.append({"coords_wgs84": [(p["lon"], p["lat"]) for p in geom], "tags": tags,
                           "id": el["id"], "node_ids": el.get("nodes", [])})
        elif "lon" in el:
            kerbs.append({"coords_wgs84": None, "lon": el["lon"], "lat": el["lat"], "tags": tags,
                           "id": el["id"]})

    if use_cache:
        _write_cache(cache_path, kerbs)
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
