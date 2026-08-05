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

import requests

from src.sources.data_loader import OVERPASS_USER_AGENT, query_overpass
from src.geometry.model import buffer_point_wgs84

DEFAULT_BUILDING_HEIGHT_M = 7.0  # ~2 stories, typical for small-borough Main St buildings
METERS_PER_LEVEL = 3.0
# Where fetched OSM responses are cached. Overridable so the test suite can point at a
# committed fixture set and run hermetically - see tests/conftest.py and HOPEWELL_OFFLINE.
CACHE_DIR = Path(os.environ.get(
    "HOPEWELL_OSM_CACHE",
    Path(__file__).resolve().parent.parent.parent / "output" / ".cache"))

REFRESH_ENV = "HOPEWELL_REFRESH_OSM"

# Second-level cache, in memory: the raw borough snapshot and its parsed form. The disk cache
# already avoids the network, but a batch build asks for the same junction's kerbs and
# crossings once per scenario - 27 times over for the four sites - and re-reading and
# re-parsing the same 2.9 MB of JSON each time is pure waste. Keyed by the same cache key the
# disk layer uses, so it can never disagree with it. The per-layer VIEWS over the parsed
# snapshot are cached separately - see _LAYER_VIEWS.
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


# There is no longer a per-layer disk cache path to compute: every layer is a view over the
# one borough snapshot (see below), so the only cached file is _snapshot_path(). The old
# _cache_path() built a (kind, centre, radius) filename per layer and is gone with the 20-24
# bbox queries it keyed. Existing per-layer files in output/.cache are simply unread; the
# committed fixtures the test suite needs are the borough_*.json ones.


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


# ---------------------------------------------------------------------------
# The borough snapshot: one request, every layer.
# ---------------------------------------------------------------------------
#
# All six layers below are now VIEWS over a single download of the whole borough, rather
# than six bbox queries each. The measurements that decided it: the entire municipality is
# 2.86 MB and 1.25 s from api.openstreetmap.org, against 20-24 Overpass requests for a
# strict subset of the same data - any one of which can block for minutes when the
# volunteer mirrors are unwell, as all three were on 2026-08-02.
#
# Three things fall out of it beyond uptime:
#   * ONE CONSISTENT SNAPSHOT. Every layer and every site comes from the same read, so the
#     replication skew that mirror-pinning exists to paper over cannot happen at all.
#   * The main OSM API is the live database, not a replica - a kerb traced a minute ago is
#     there, which is the whole point of --refresh-osm.
#   * A new site inside the borough costs zero extra requests.
#
# Overpass stays as the fallback. It filters server-side, which is genuinely nicer when it
# is healthy.
OSM_API_MAP = "https://api.openstreetmap.org/api/0.6/map.json"

# Hopewell Borough, with margin for the context radii (130 m) at the edge sites. 0.000364
# sq deg against the API's 0.25 limit. Sites outside it are refused loudly rather than
# silently returning nothing - see assert_within_snapshot.
BOROUGH_BBOX = (-74.7760, 40.3830, -74.7500, 40.3970)   # west, south, east, north


class SiteOutsideSnapshotError(RuntimeError):
    """A site's context window reaches outside the downloaded borough bbox."""


def _snapshot_path() -> Path:
    key = hashlib.sha1(f"borough,v1,{BOROUGH_BBOX}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"borough_{key}.json"


def _download_snapshot() -> list[dict]:
    """The whole borough from the OSM API, falling back to Overpass."""
    if os.environ.get("HOPEWELL_OFFLINE"):
        # The Overpass path is guarded inside query_overpass, but this one calls requests
        # directly - without this the test suite would reach the network for the snapshot
        # and quietly depend on OSM's uptime and current contents.
        from src.sources.data_loader import OfflineCacheMiss
        raise OfflineCacheMiss(
            "HOPEWELL_OFFLINE is set and the borough snapshot is not in the fixture cache. "
            "Refresh it with: cp output/.cache/borough_*.json tests/fixtures/osm_cache/")

    west, south, east, north = BOROUGH_BBOX
    try:
        resp = requests.get(f"{OSM_API_MAP}?bbox={west},{south},{east},{north}",
                            headers={"User-Agent": OVERPASS_USER_AGENT}, timeout=(5, 120))
        resp.raise_for_status()
        return resp.json()["elements"]
    except requests.exceptions.RequestException as e:
        print(f"  OSM API unavailable ({type(e).__name__}); falling back to Overpass")
        # `out meta` for ways + `>` to pull their nodes: the same shape /map.json returns,
        # so everything downstream is indifferent to which source answered.
        return query_overpass(f"""
        [out:json][timeout:180];
        ( node({south},{west},{north},{east});
          way({south},{west},{north},{east}); );
        out body;
        """)["elements"]


def fetch_borough_osm(use_cache: bool = True) -> dict:
    """{"nodes": {id: element}, "ways": [element]} for the whole borough.

    Raises if any way references a node that isn't present. The OSM API completes ways
    whose nodes fall outside the bbox, so a gap means a truncated download - and half a
    kerb is worse than no kerb, because it looks like geometry.
    """
    cache_path = _snapshot_path()
    if use_cache and _cache_hit(cache_path):
        raw = _memoized(cache_path, lambda: json.loads(cache_path.read_text()))
    else:
        raw = _download_snapshot()
        if use_cache:
            _write_cache(cache_path, raw)

    key = f"parsed:{cache_path}"
    if key not in _MEMO:
        nodes = {el["id"]: el for el in raw if el["type"] == "node"}
        ways = [el for el in raw if el["type"] == "way"]
        dangling = sum(1 for w in ways for nid in w.get("nodes", []) if nid not in nodes)
        if dangling:
            raise RuntimeError(
                f"{dangling} way node reference(s) in the borough snapshot don't resolve - the "
                f"download is truncated. Delete {cache_path} and re-pull; do not build geometry "
                f"from it, the ways would come out with missing vertices.")
        _MEMO[key] = {"nodes": nodes, "ways": ways}
    return _MEMO[key]


def assert_within_snapshot(center_wgs84: Point, radius_m: float) -> None:
    """A site reaching outside the borough bbox gets NOTHING, silently, which is precisely
    how ground truth disappears in this project. Refuse instead."""
    west, south, east, north = buffer_point_wgs84(center_wgs84, radius_m)
    bw, bs, be, bn = BOROUGH_BBOX
    if west < bw or south < bs or east > be or north > bn:
        raise SiteOutsideSnapshotError(
            f"this site's {radius_m:.0f} m context window ({west:.5f},{south:.5f},{east:.5f},"
            f"{north:.5f}) reaches outside the downloaded borough bbox {BOROUGH_BBOX}. Widen "
            f"BOROUGH_BBOX in src/sources/osm_context.py and delete the cached snapshot.")


# One resolved layer per (layer, centre, radius), for as long as the snapshot it came from is
# still the one being read. Every _ways_near / _nodes_near call walks the WHOLE borough -
# 2.9 MB, thousands of elements - and the fetchers below are called repeatedly per site: the
# kerbs alone are asked for by the intersection model (twice), the plan view and the export,
# and every scenario repeats the lot.
#
# Each entry stores the snapshot it was built from and is only served while that is still the
# snapshot in hand. That, rather than remembering to clear the cache, is what makes a re-pull
# reach the render - which is this project's worst failure mode, so it should not depend on an
# invalidation call somebody could forget or a test could bypass. Holding the reference is also
# what makes the identity test sound: the object cannot be freed and its id reused while the
# entry that names it is alive.
_LAYER_VIEWS: dict[tuple, tuple] = {}


def _layer(kind: str, center_wgs84: Point, radius_m: float, build):
    snapshot = fetch_borough_osm()
    key = (kind, round(center_wgs84.x, 7), round(center_wgs84.y, 7), float(radius_m))
    cached = _LAYER_VIEWS.get(key)
    if cached is not None and cached[0] is snapshot:
        return cached[1]
    view = build()
    _LAYER_VIEWS[key] = (snapshot, view)
    return view


def _in_bbox(bbox, lon: float, lat: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def _way_coords(snapshot: dict, way: dict) -> list[tuple[float, float]]:
    nodes = snapshot["nodes"]
    return [(nodes[nid]["lon"], nodes[nid]["lat"]) for nid in way.get("nodes", [])]


def _ways_near(center_wgs84: Point, radius_m: float, predicate) -> list[tuple[dict, list]]:
    """[(way, coords)] for tagged ways with at least one vertex in the leg's bbox.

    Same rectangle-and-any-vertex rule Overpass applies to a bbox query, so switching
    source doesn't quietly change which elements a junction sees.
    """
    assert_within_snapshot(center_wgs84, radius_m)
    snapshot = fetch_borough_osm()
    bbox = buffer_point_wgs84(center_wgs84, radius_m)
    out = []
    for way in snapshot["ways"]:
        if not predicate(way.get("tags") or {}):
            continue
        coords = _way_coords(snapshot, way)
        if any(_in_bbox(bbox, lon, lat) for lon, lat in coords):
            out.append((way, coords))
    return out


def _nodes_near(center_wgs84: Point, radius_m: float, predicate) -> list[dict]:
    assert_within_snapshot(center_wgs84, radius_m)
    snapshot = fetch_borough_osm()
    bbox = buffer_point_wgs84(center_wgs84, radius_m)
    return [n for n in snapshot["nodes"].values()
            if predicate(n.get("tags") or {}) and _in_bbox(bbox, n["lon"], n["lat"])]


def fetch_buildings(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM building footprints near a point.

    Returns [{"coords_wgs84": [...], "tags": {...}, "height_m": float|None,
              "height_source": str|None}, ...], where the height is None unless a mapper
    recorded one - see height_from_tags, and src/sources/assessor.py for where the answer
    comes from when they did not, which here is almost always.
    """
    def build():
        out = []
        for way, coords in _ways_near(center_wgs84, radius_m, lambda t: "building" in t):
            if len(coords) < 3:
                continue
            tags = way.get("tags") or {}
            recorded = height_from_tags(tags)
            out.append({"coords_wgs84": coords, "tags": tags,
                        "height_m": recorded[0] if recorded else None,
                        "height_source": recorded[1] if recorded else None})
        return out
    return _layer("buildings", center_wgs84, radius_m, build)


def fetch_crossings(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM-mapped pedestrian crossings (footway=crossing ways) - real surveyed crosswalk
    lines rather than a geometric estimate of where one probably is.
    Returns [{"coords_wgs84": [...], "tags": {...}, "node_ids": [...]}, ...]."""
    def build():
        return [{"coords_wgs84": coords, "tags": way.get("tags", {}),
                 "node_ids": way.get("nodes", [])}
                for way, coords in _ways_near(center_wgs84, radius_m,
                                               lambda t: t.get("footway") == "crossing")
                if len(coords) >= 2]
    return _layer("crossings", center_wgs84, radius_m, build)


def fetch_sidewalks(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM-mapped sidewalk centerlines (footway=sidewalk ways).

    Real surveyed geometry, and what OSM's crossing ways actually connect to - a crossing
    runs sidewalk-centerline to sidewalk-centerline, not curb to curb. That makes them an
    independent bound on a leg's width (src/geometry/model.py:sidewalk_span_ft), though not
    a measurement of it: the centerline-to-curb gap measured 11.8 ft/side on one
    field-measured leg and 4.0 ft/side on another, on the same street 100 ft apart.
    """
    def build():
        return [{"coords_wgs84": coords, "tags": way.get("tags", {})}
                for way, coords in _ways_near(center_wgs84, radius_m,
                                               lambda t: t.get("footway") == "sidewalk")
                if len(coords) >= 2]
    return _layer("sidewalks", center_wgs84, radius_m, build)


def fetch_driveways(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM-mapped driveways (highway=service + service=driveway).

    The vehicle access these junctions' kerb openings exist FOR, drawn so the gap in the
    markings has something visible on the other side of it - a break in a bike lane with nothing
    leading away from it reads as a striping error rather than as an entrance.

    NOT the signal for where the markings open: that is the dropped kerb itself, which is on the
    kerb and therefore already in the leg frame, and which is tagged in places a driveway way is
    not drawn (see src/geometry/kerbs.py). Only one of the 43 driveways mapped in this borough
    reaches a kerb any of these four junctions models. So this layer is for DRAWING, and the two
    are deliberately independent - a driveway drawn with no dropped kerb tagged at its mouth is a
    survey gap worth seeing, not something to paper over by inferring one from the other.
    """
    def build():
        return [{"coords_wgs84": coords, "tags": way.get("tags", {}), "id": way["id"]}
                for way, coords in _ways_near(center_wgs84, radius_m,
                                               lambda t: t.get("highway") == "service"
                                               and t.get("service") == "driveway")
                if len(coords) >= 2]
    return _layer("driveways", center_wgs84, radius_m, build)


def fetch_traffic_control(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM traffic control nodes: highway=traffic_signals / stop / give_way / crossing.
    Returns [{"lon": float, "lat": float, "tags": {...}}, ...].

    Real surveyed control instead of guessing. At Columbia & Princeton OSM maps exactly two
    stop nodes, both on Columbia Ave, because Princeton Ave (CR 569) runs free - the old
    one-sign-per-approach guess put stop signs on two approaches that don't have them.

    highway=crossing nodes are included because that is where OSM records the
    pedestrian-facing detail that lives on the node rather than the way: tactile_paving,
    button_operated, crossing:island. Reading only the ways is what once made data_gaps()
    report "no ADA data" at Broad/Greenwood, where all four crossings are tagged
    tactile_paving=yes.
    """
    wanted = ("traffic_signals", "stop", "give_way", "crossing")

    def build():
        return [{"lon": n["lon"], "lat": n["lat"], "tags": n.get("tags", {})}
                for n in _nodes_near(center_wgs84, radius_m, lambda t: t.get("highway") in wanted)]
    return _layer("traffic_control", center_wgs84, radius_m, build)


def fetch_street_furniture(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM street furniture: highway=street_lamp, emergency=fire_hydrant, natural=tree.
    Returns [{"lon": float, "lat": float, "tags": {...}}, ...].

    STREET LAMPS ARE NOT MAPPED AT ANY OF THIS PROJECT'S FOUR SITES. This exists so a site
    where they ARE mapped gets real pole positions rather than a derived one-per-corner
    placement, and so the absence is reported (see data_gaps) rather than papered over.
    """
    def wanted(t):
        return (t.get("highway") == "street_lamp" or t.get("emergency") == "fire_hydrant"
                or t.get("natural") == "tree")

    def build():
        return [{"lon": n["lon"], "lat": n["lat"], "tags": n.get("tags", {})}
                for n in _nodes_near(center_wgs84, radius_m, wanted)]
    return _layer("street_furniture", center_wgs84, radius_m, build)


def fetch_kerbs(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM-mapped kerb lines and kerb nodes (barrier=kerb).
    Returns [{"coords_wgs84": [...] | None, "lon"/"lat" for nodes, "tags", "id", "node_ids"}].

    The most direct geometry this project can get: a traced kerb IS the curb, so it gives
    the curb line, the corner radius (nothing in OSM carries a radius tag) and the position
    of tactile paving, none of which have to be inferred from our own estimated widths.

    Two-vertex ways are kept. A straight run of kerb is two points, and dropping them (an
    old circle-fitting precondition applied at the wrong layer) threw away 12 of the 23
    traced ways at two of these sites.
    """
    def build():
        kerbs = [{"coords_wgs84": coords, "tags": way.get("tags", {}), "id": way["id"],
                  "node_ids": way.get("nodes", [])}
                 for way, coords in _ways_near(center_wgs84, radius_m,
                                                lambda t: t.get("barrier") == "kerb")
                 if len(coords) >= 2]
        kerbs += [{"coords_wgs84": None, "lon": n["lon"], "lat": n["lat"],
                   "tags": n.get("tags", {}), "id": n["id"]}
                  for n in _nodes_near(center_wgs84, radius_m,
                                        lambda t: t.get("barrier") == "kerb")]
        return kerbs
    return _layer("kerbs", center_wgs84, radius_m, build)


def height_from_tags(tags: dict) -> tuple[float, str] | None:
    """(height in metres, which tag said so) if a mapper recorded one, else None.

    None rather than DEFAULT_BUILDING_HEIGHT_M, because "nobody said" is a different answer from
    "7 m" and the caller has somewhere else to look: the assessor's storey count, in
    src/sources/assessor.py. Returning the default here is what made every building in every
    render the same height - 0 of the 1150 building ways in this borough carry `height` and 7
    carry `building:levels`, so the default WAS the model.
    """
    if tags.get("height"):
        try:
            return float("".join(c for c in tags["height"] if c.isdigit() or c == ".")), "osm_height"
        except ValueError:
            pass
    if tags.get("building:levels"):
        try:
            return float(tags["building:levels"]) * METERS_PER_LEVEL, "osm_levels"
        except ValueError:
            pass
    return None


def fetch_roads(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM highway ways near a point, with their tags and geometry.

    The road ways themselves, not the furniture on them - this is where OSM records facts
    about how the carriageway is operated rather than where things are. `overtaking=no` is
    the one currently used: it is what a double-yellow centerline MEANS, and five ways in
    Hopewell carry it (both Broad Streets, both Greenwood Avenues, Princeton Avenue).
    Returns [{"coords_wgs84": [...], "tags": {...}, "id": int}, ...].
    """
    def build():
        return [{"coords_wgs84": coords, "tags": way.get("tags", {}), "id": way["id"]}
                for way, coords in _ways_near(center_wgs84, radius_m, lambda t: "highway" in t)
                if len(coords) >= 2]
    return _layer("roads", center_wgs84, radius_m, build)


def fetch_stop_lines(center_wgs84: Point, radius_m: float) -> list[dict]:
    """OSM-mapped stop bars (road_marking=stop_line ways) near a point.

    A surveyed stop bar gives all three things this project was previously deriving: how far
    back from the junction it sits, how wide it is, and which half of the roadway it covers.
    The derived version could only ever place it a fixed setback behind the crosswalk.
    Returns [{"coords_wgs84": [...], "tags": {...}, "id": int}, ...].
    """
    def build():
        return [{"coords_wgs84": coords, "tags": way.get("tags", {}), "id": way["id"]}
                for way, coords in _ways_near(center_wgs84, radius_m,
                                               lambda t: t.get("road_marking") == "stop_line")
                if len(coords) >= 2]
    return _layer("stop_lines", center_wgs84, radius_m, build)
