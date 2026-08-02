"""What we keep, and what we throw away, from what the surveyor traced.

The most expensive class of bug in this project isn't a wrong calculation - it's ground
truth being silently discarded on the way in, so everything downstream computes a careful
answer from a guess. These tests guard the intake.
"""
import json

import pytest
from shapely.geometry import Point

from src.sources import osm_context
from src.sources.data_loader import OfflineCacheMiss, query_overpass


def a_way(way_id, coords, tags=None):
    return {"type": "way", "id": way_id, "geometry": [{"lon": x, "lat": y} for x, y in coords],
            "tags": tags or {"barrier": "kerb"}, "nodes": list(range(len(coords)))}


def test_two_vertex_kerb_ways_are_kept(monkeypatch, tmp_path):
    """A straight run of kerb is two points, and straight runs are most of what gets traced.

    The old `len(geom) < 3` rule - really a circle-fitting precondition applied at the wrong
    layer - dropped 12 of the 23 traced ways at Columbia & Princeton and at E Broad &
    Princeton, and 5 of 12 at W Broad & Louellen. Those sides then fell back to centerline
    offsets on legs the surveyor had actually traced.
    """
    payload = {"elements": [
        a_way(1, [(-74.76, 40.39), (-74.7599, 40.3901)]),                     # straight, 2 points
        a_way(2, [(-74.76, 40.39), (-74.7599, 40.3901), (-74.7598, 40.3902)]),
    ]}
    monkeypatch.setattr(osm_context, "query_overpass", lambda *a, **k: payload)
    monkeypatch.setattr(osm_context, "CACHE_DIR", tmp_path)

    kerbs = osm_context.fetch_kerbs(Point(-74.76, 40.39), radius_m=120)
    assert len(kerbs) == 2, "the 2-vertex way must survive"
    assert min(len(k["coords_wgs84"]) for k in kerbs) == 2


def test_a_one_vertex_way_is_still_dropped(monkeypatch, tmp_path):
    """One point is not a line - there is no kerb to follow."""
    monkeypatch.setattr(osm_context, "query_overpass",
                        lambda *a, **k: {"elements": [a_way(1, [(-74.76, 40.39)])]})
    monkeypatch.setattr(osm_context, "CACHE_DIR", tmp_path)
    assert osm_context.fetch_kerbs(Point(-74.76, 40.39), radius_m=120) == []


def test_kerb_node_ids_are_kept(monkeypatch, tmp_path):
    """Node ids are how a kerb is matched to the crossing it serves - one lowered kerb
    serving two crossings is distinguishable from two separate ramps only through these."""
    monkeypatch.setattr(osm_context, "query_overpass",
                        lambda *a, **k: {"elements": [a_way(7, [(-74.76, 40.39), (-74.7599, 40.3901)])]})
    monkeypatch.setattr(osm_context, "CACHE_DIR", tmp_path)
    kerbs = osm_context.fetch_kerbs(Point(-74.76, 40.39), radius_m=120)
    assert kerbs[0]["node_ids"], "node ids must survive the fetch"


def test_kerb_tags_are_kept_whatever_their_value(monkeypatch, tmp_path):
    """kerb=lowered is a corner RAMP - the corner return itself. Filtering to kerb=raised
    dropped whole traced corners in favour of a fitted guess."""
    monkeypatch.setattr(osm_context, "query_overpass", lambda *a, **k: {"elements": [
        a_way(1, [(-74.76, 40.39), (-74.7599, 40.3901)], {"barrier": "kerb", "kerb": "raised"}),
        a_way(2, [(-74.7599, 40.3901), (-74.7598, 40.3902)],
              {"barrier": "kerb", "kerb": "lowered", "tactile_paving": "yes"}),
    ]})
    monkeypatch.setattr(osm_context, "CACHE_DIR", tmp_path)
    kerbs = osm_context.fetch_kerbs(Point(-74.76, 40.39), radius_m=120)
    assert {k["tags"].get("kerb") for k in kerbs} == {"raised", "lowered"}


def test_the_test_suite_cannot_reach_the_network():
    """conftest sets HOPEWELL_OFFLINE. A cache miss must fail loudly, not fetch.

    A test that silently depends on Overpass depends on its uptime AND its current
    replication state - two consecutive live fetches of one junction returned 4 tactile
    pads and then 0 during a single editing session.
    """
    with pytest.raises(OfflineCacheMiss):
        query_overpass("[out:json];node(1);out;")


def test_the_fixture_cache_is_present_and_readable():
    """The snapshot the rest of the suite runs against."""
    files = list(osm_context.CACHE_DIR.glob("*.json"))
    assert files, f"no OSM fixtures in {osm_context.CACHE_DIR}"
    for path in files:
        json.loads(path.read_text())
