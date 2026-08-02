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


def a_snapshot(ways=(), nodes=()):
    """A fake borough snapshot in the shape fetch_borough_osm returns.

    Node coordinates sit inside the bbox the tests query, so the bbox filter keeps them and
    the assertions are about intake rules rather than geography.
    """
    node_table, way_list = {}, []
    next_id = 100
    for tags in nodes:
        node_table[next_id] = {"type": "node", "id": next_id, "lon": LON, "lat": LAT, "tags": tags}
        next_id += 1
    for way_id, vertex_count, tags in ways:
        refs = []
        for i in range(vertex_count):
            node_table[next_id] = {"type": "node", "id": next_id,
                                    "lon": LON + i * 1e-5, "lat": LAT + i * 1e-5, "tags": {}}
            refs.append(next_id)
            next_id += 1
        way_list.append({"type": "way", "id": way_id, "nodes": refs, "tags": tags})
    return {"nodes": node_table, "ways": way_list}


LON, LAT = -74.7600, 40.3890   # inside BOROUGH_BBOX
CENTRE = Point(LON, LAT)


def test_two_vertex_kerb_ways_are_kept(monkeypatch):
    """A straight run of kerb is two points, and straight runs are most of what gets traced.

    The old `len(geom) < 3` rule - really a circle-fitting precondition applied at the wrong
    layer - dropped 12 of the 23 traced ways at Columbia & Princeton and at E Broad &
    Princeton, and 5 of 12 at W Broad & Louellen. Those sides then fell back to centerline
    offsets on legs the surveyor had actually traced.
    """
    snapshot = a_snapshot(ways=[(1, 2, {"barrier": "kerb"}), (2, 3, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    kerbs = osm_context.fetch_kerbs(CENTRE, radius_m=120)
    assert len(kerbs) == 2, "the 2-vertex way must survive"
    assert min(len(k["coords_wgs84"]) for k in kerbs) == 2


def test_a_one_vertex_way_is_still_dropped(monkeypatch):
    """One point is not a line - there is no kerb to follow."""
    snapshot = a_snapshot(ways=[(1, 1, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    assert osm_context.fetch_kerbs(CENTRE, radius_m=120) == []


def test_kerb_node_ids_are_kept(monkeypatch):
    """Node ids are how a kerb is matched to the crossing it serves - one lowered kerb
    serving two crossings is distinguishable from two separate ramps only through these."""
    snapshot = a_snapshot(ways=[(7, 2, {"barrier": "kerb"})])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    assert osm_context.fetch_kerbs(CENTRE, radius_m=120)[0]["node_ids"]


def test_kerb_tags_are_kept_whatever_their_value(monkeypatch):
    """kerb=lowered is a corner RAMP - the corner return itself. Filtering to kerb=raised
    dropped whole traced corners in favour of a fitted guess."""
    snapshot = a_snapshot(ways=[
        (1, 2, {"barrier": "kerb", "kerb": "raised"}),
        (2, 2, {"barrier": "kerb", "kerb": "lowered", "tactile_paving": "yes"}),
    ])
    monkeypatch.setattr(osm_context, "fetch_borough_osm", lambda *a, **k: snapshot)
    kerbs = osm_context.fetch_kerbs(CENTRE, radius_m=120)
    assert {k["tags"].get("kerb") for k in kerbs} == {"raised", "lowered"}


def test_a_dangling_node_reference_is_refused(monkeypatch):
    """A truncated download must not become geometry with missing vertices.

    The OSM API completes ways whose nodes fall outside the bbox, so an unresolvable
    reference means the snapshot is incomplete - and half a kerb looks like a real kerb.
    """
    raw = [{"type": "node", "id": 1, "lon": LON, "lat": LAT, "tags": {}},
           {"type": "way", "id": 9, "nodes": [1, 999], "tags": {"barrier": "kerb"}}]
    monkeypatch.setattr(osm_context, "_cache_hit", lambda p: False)
    monkeypatch.setattr(osm_context, "_download_snapshot", lambda: raw)
    monkeypatch.setattr(osm_context, "_write_cache", lambda p, d: None)
    osm_context._MEMO.clear()
    with pytest.raises(RuntimeError, match="don't resolve"):
        osm_context.fetch_borough_osm()
    osm_context._MEMO.clear()


def test_a_site_outside_the_snapshot_is_refused():
    """Reaching outside the downloaded bbox returns NOTHING, silently - which is exactly
    how ground truth disappears in this project. It must raise instead."""
    with pytest.raises(osm_context.SiteOutsideSnapshotError):
        osm_context.assert_within_snapshot(Point(-75.5, 40.0), radius_m=130)


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
