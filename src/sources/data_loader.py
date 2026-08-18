"""Data loading: NJDOT roadway network, Mercer County parcels, and intersection geocoding."""
import os
from pathlib import Path

import geopandas as gpd
import requests
from geopy.geocoders import Nominatim
from shapely.geometry import MultiLineString, MultiPolygon, Point, box

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84, buffer_point_wgs84, reproject_to_state_plane
from src.sources.schemas import ParcelsSchema, RoadNetworkSchema, validate_layer


class OfflineCacheMiss(RuntimeError):
    """HOPEWELL_OFFLINE is set and a fetch wasn't satisfied from the fixture cache."""

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"  # src/sources/data_loader.py -> repo root
# Defaults only - a site's config.yaml (data_sources:) can point at different
# files entirely (e.g. a different county's parcels/road network), since
# nothing else in this module is specific to Mercer County or NJDOT's statewide file.
DEFAULT_ROAD_NETWORK_PATH = DATA_DIR / "NJ_Roadway_Network.geojson"
DEFAULT_PARCELS_PATH = DATA_DIR / "MercerCountyParcels.shp"

# GeoJSON has no spatial index, so a bbox-filtered read still parses the whole
# file: pulling the 9 segments around one intersection out of NJDOT's 170 MB
# statewide layer costs ~2.2 s, versus ~2.5 s to read all 105,838 features - the
# bbox saves almost nothing. An indexed format makes the same read ~0.002 s.
# scripts/convert_road_network.py writes that sibling; if it exists, it's used
# automatically. See _resolve_indexed_path.
INDEXED_SUFFIXES = (".fgb", ".gpkg")
_announced_indexed: set[Path] = set()

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

# Nominatim geocodes a single street name to an arbitrary point along it, not to a
# cross-street intersection. This threshold catches the gap (~230 ft at this latitude).
INTERSECTION_MATCH_TOLERANCE_DEG = 0.0007


_pinned_mirror: list[str] = []

# How long to wait for the TCP handshake, separately from the read. Overpass queries can
# legitimately take tens of seconds to ANSWER, but connecting is either quick or the mirror
# is not there.
CONNECT_TIMEOUT_S = 5

# Mirrors that have already failed their whole retry budget this run.
_dead_mirrors: set[str] = set()


def query_overpass(query: str, attempts_per_mirror: int = 4, timeout: int = 30) -> dict:
    """POST an Overpass QL query, retrying across mirrors on timeout/5xx errors.

    Once a mirror answers, it is PINNED for the rest of the process, so every fetch sees
    the same snapshot. Retries are per-mirror before failover.
    """
    if os.environ.get("HOPEWELL_OFFLINE"):
        # The test suite runs against a committed fixture cache. If something reaches this
        # far it means the fixture is missing, and the honest outcome is a loud failure -
        # not a silent network call that makes the tests depend on Overpass's uptime and
        # current replication state.
        raise OfflineCacheMiss(
            "HOPEWELL_OFFLINE is set and this query is not in the fixture cache. Add the "
            "response to tests/fixtures/osm_cache (see tests/conftest.py) rather than "
            f"letting a test reach the network. Query was:\n{query.strip()[:400]}")

    ordered = ([m for m in _pinned_mirror if m in OVERPASS_MIRRORS]
               + [m for m in OVERPASS_MIRRORS if m not in _pinned_mirror])
    last_error = None
    for mirror in ordered:
        host = mirror.split("//")[1].split("/")[0]
        if host in _dead_mirrors:
            continue  # already proved unreachable this run; don't pay the timeout again
        for attempt in range(attempts_per_mirror):
            try:
                resp = requests.post(
                    mirror, data={"data": query}, headers={"User-Agent": OVERPASS_USER_AGENT},
                    # (connect, read). A mirror that is blackholing packets never completes
                    # the TCP handshake, and a single 30 s number spent the whole budget
                    # there: one editing session sat 10+ minutes in SYN_SENT against
                    # overpass.kumi.systems with nothing printed. A connect is either fast
                    # or not happening; a read legitimately takes a while.
                    timeout=(CONNECT_TIMEOUT_S, timeout),
                )
                resp.raise_for_status()
                if not _pinned_mirror:
                    _pinned_mirror.append(mirror)
                    print(f"  Overpass: using {host} "
                          f"(pinned for this run so every fetch sees one replication state)")
                return resp.json()
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                last_error = e
                # Say so as it happens. A retry loop that prints nothing is indistinguishable
                # from a hang, which is exactly how this failure presented.
                print(f"  Overpass: {host} attempt {attempt + 1}/{attempts_per_mirror} failed "
                      f"({type(e).__name__}); {'retrying' if attempt + 1 < attempts_per_mirror else 'giving up on it'}",
                      flush=True)
                if isinstance(e, requests.exceptions.ConnectTimeout):
                    break  # unreachable, not busy - further attempts just burn the budget
        _dead_mirrors.add(host)
    raise RuntimeError(f"All Overpass mirrors failed after retries. Last error: {last_error}")


def approximate_geocode(query: str) -> Point:
    """Rough single-point geocode via Nominatim. Only precise enough to anchor a search bbox."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    location = geolocator.geocode(query, timeout=10)
    if location is None:
        raise ValueError(f"Could not geocode: {query!r}")
    return Point(location.longitude, location.latitude)


def geocode_intersection(street1: str, street2: str, anchor_query: str, search_radius_m: float = 1000) -> Point:
    """Resolve the real intersection point of two named streets by querying OSM/Overpass
    for way geometry and locating the node the two streets share.

    More precise than address-string geocoding: street geocoders return a single point along
    the street, not the cross-street corner. `anchor_query` centers a search bbox via
    Nominatim. Matching is on ANY shared node, not just shared way endpoints - a way only
    ends at a junction when OSM happens to split it there.

    Where several nodes are shared, the one nearest the anchor is returned.
    """
    anchor = approximate_geocode(anchor_query)
    west, south, east, north = buffer_point_wgs84(anchor, search_radius_m)

    # BOTH `name` AND `ref`, because a state highway routinely has no name at all. Every
    # way at NJ 31 & W Delaware Ave carries `ref=NJ 31` and no `name`, so a name-only match
    # found nothing there and reported it as "could not find OSM ways matching 'NJ 31'" -
    # which reads like a misspelling rather than a tag that does not exist on this class of
    # road. A local street is named and unreffed, an arterial is often both, and which of the
    # two a caller passes is not something they should have to know in advance.
    #
    # `out geom` gives coordinates but not node IDs; `out geom` + the `nodes` array
    # (included for ways by default in Overpass JSON) gives both, positionally paired.
    clauses = "\n      ".join(
        f'way["highway"]["{tag}"~"{street}",i]({south},{west},{north},{east});'
        for street in (street1, street2) for tag in ("name", "ref")
    )
    query = f"""
    [out:json][timeout:25];
    (
      {clauses}
    );
    out geom;
    """
    elements = query_overpass(query)["elements"]

    def nodes_of(street: str) -> dict[int, Point]:
        """Map node id -> position for every node on every way matching this street.

        Matched against `name` and `ref` alike - the same pair the query asked for, so a way
        the query returned cannot then be discarded here for carrying the wrong one of the two.
        """
        found: dict[int, Point] = {}
        for el in elements:
            tags = el.get("tags", {})
            if not any(street.lower() in (tags.get(tag) or "").lower() for tag in ("name", "ref")):
                continue
            for node_id, coords in zip(el.get("nodes", []), el.get("geometry", [])):
                found[node_id] = Point(coords["lon"], coords["lat"])
        return found

    nodes1 = nodes_of(street1)
    nodes2 = nodes_of(street2)
    # Name the street that actually failed. "street1 and/or street2" sends you to check both,
    # and the answer is nearly always one of them - a typo, or a name OSM does not use here.
    unmatched = [s for s, found in ((street1, nodes1), (street2, nodes2)) if not found]
    if unmatched:
        raise ValueError(
            f"Could not find OSM ways matching {' or '.join(repr(s) for s in unmatched)} near "
            f"{anchor_query!r} - searched both the `name` and `ref` tags. Check the spelling "
            "against what OSM actually tags this road with; a state highway is often reffed "
            "(`ref=NJ 31`) and unnamed, and a county route carries both."
        )

    shared = set(nodes1) & set(nodes2)
    if shared:
        return nodes1[min(shared, key=lambda nid: nodes1[nid].distance(anchor))]

    # No shared node: the streets may genuinely not meet, or OSM may have them
    # meeting at two coincident-but-unmerged nodes. Fall back to the closest pair
    # of nodes, and only accept it if they're close enough to be one junction.
    p1, p2, dist = min(
        ((a, b, a.distance(b)) for a in nodes1.values() for b in nodes2.values()), key=lambda t: t[2]
    )
    if dist > INTERSECTION_MATCH_TOLERANCE_DEG:
        raise ValueError(
            f"{street1!r} and {street2!r} share no OSM node near {anchor_query!r}, and their closest "
            f"nodes are ~{dist * 364000:.0f} ft apart - does not look like a real intersection. "
            "Provide coordinates manually."
        )
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def _resolve_indexed_path(path: Path | str) -> Path:
    """Return an indexed sibling of `path` (same stem, .fgb/.gpkg) if one exists and
    is at least as new as `path`, else `path` unchanged.

    Pure format/index swap: the converter writes every feature and attribute through
    untouched. A stale sibling - older than the source - is ignored.
    """
    path = Path(path)
    if path.suffix.lower() in INDEXED_SUFFIXES:
        return path
    for suffix in INDEXED_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if not candidate.exists():
            continue
        if path.exists() and candidate.stat().st_mtime < path.stat().st_mtime:
            print(f"  NOTE: {candidate.name} is older than {path.name} - ignoring it as stale. "
                  f"Rerun scripts/convert_road_network.py to refresh it.")
            continue
        if candidate not in _announced_indexed:
            _announced_indexed.add(candidate)
            print(f"  Using spatially-indexed {candidate.name} instead of {path.name} (identical data, ~1000x faster).")
        return candidate
    return path


def _unpack_single_part(geometry):
    """Collapse single-part Multi* geometries back to their simple counterpart.

    Indexed formats store one geometry type per layer, so a mixed LineString/MultiLineString
    layer promotes everything to MultiLineString. Genuinely multi-part geometries are left
    alone.
    """
    if isinstance(geometry, (MultiLineString, MultiPolygon)) and len(geometry.geoms) == 1:
        return geometry.geoms[0]
    return geometry


def load_road_network(
    bbox: tuple[float, float, float, float] | None = None, path: Path | str = DEFAULT_ROAD_NETWORK_PATH
) -> gpd.GeoDataFrame:
    """Load a roadway network file (NJDOT's statewide SRI/SLD linear-referencing
    layer by default; pass `path` for a different one), optionally filtered to a
    WGS84 bbox (minx, miny, maxx, maxy).

    Transparently prefers a spatially-indexed sibling of `path` if one has been
    built (see _resolve_indexed_path / scripts/convert_road_network.py) - same data,
    dramatically faster bbox reads.
    """
    resolved = _resolve_indexed_path(path)
    network = gpd.read_file(resolved, bbox=bbox)
    if not network.empty:
        network = network.set_geometry(network.geometry.map(_unpack_single_part))
    # Validated at the boundary, once - see src/sources/schemas.py. A renamed SRI column is
    # otherwise read downstream as "this leg matched no road", which is drawn, not raised.
    return validate_layer(network, RoadNetworkSchema, resolved, expect_crs=WGS84)


def load_parcels(
    bbox: tuple[float, float, float, float] | None = None, path: Path | str = DEFAULT_PARCELS_PATH
) -> gpd.GeoDataFrame:
    """Load a parcels/MOD-IV shapefile (Mercer County by default; pass `path` for a
    different one), optionally filtered to a bbox (in the shapefile's native CRS -
    reproject the bbox first if querying in WGS84)."""
    # The CRS check here is the one that has actually bitten: a WGS84 bbox against this
    # State-Plane shapefile returns zero rows, which reads as "no parcels here".
    return validate_layer(gpd.read_file(path, bbox=bbox), ParcelsSchema, path,
                           expect_crs=NJ_STATE_PLANE_FT)


def load_parcels_near(
    center_wgs84: Point, radius_ft: float, path: Path | str = DEFAULT_PARCELS_PATH
) -> gpd.GeoDataFrame:
    """Load parcels within a square bbox (radius_ft) of a WGS84 point, reprojected
    to NJ State Plane. Full parcel polygons are kept, not circle-clipped.
    """
    center_ft = gpd.GeoSeries([center_wgs84], crs=WGS84).to_crs(NJ_STATE_PLANE_FT).iloc[0]
    bbox_geom = box(center_ft.x - radius_ft, center_ft.y - radius_ft, center_ft.x + radius_ft, center_ft.y + radius_ft)
    # Passing a CRS-tagged GeoSeries (rather than a plain tuple) lets pyogrio resolve
    # the parcel shapefile's own (slightly different, HARN-less) NAD83 NJ State Plane CRS.
    parcels = load_parcels(bbox=gpd.GeoSeries([bbox_geom], crs=NJ_STATE_PLANE_FT), path=path)
    return reproject_to_state_plane(parcels)
