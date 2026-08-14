"""Which downloaded OSM snapshot a site is served from.

The snapshot was ONE bbox covering Hopewell Borough, and a site outside it was refused with
"widen BOROUGH_BBOX and delete the cached snapshot". Taking that advice for a site in the
next town along would have re-keyed the only snapshot there is: every existing site would
re-download, the committed fixture in tests/fixtures/osm_cache/ would no longer match the
cache key, and 236 tests would start missing it. It would also pull the several miles of
farmland between the two boroughs to reach a junction 3 miles away.

So the areas are plural, each cached and keyed separately. What these tests hold down is
that adding one cannot disturb the others - particularly Hopewell's cache key, which the
committed fixture is named after.
"""
import pytest
from shapely.geometry import Point

from src.sources.osm_context import (SNAPSHOT_AREAS, SiteOutsideSnapshotError,
                                      _area_for, _snapshot_path, assert_within_snapshot)

HOPEWELL_BBOX = (-74.7760, 40.3830, -74.7500, 40.3970)
BROAD_AND_GREENWOOD = Point(-74.7614, 40.3893)
NJ31_AND_W_DELAWARE = Point(-74.7989728, 40.3272925)


def test_hopewells_cache_key_is_unchanged():
    """The committed fixture is named for this hash. If it moves, the offline suite stops
    finding the snapshot and every junction test skips - silently green, testing nothing."""
    assert SNAPSHOT_AREAS["hopewell_borough"] == HOPEWELL_BBOX
    assert _snapshot_path(HOPEWELL_BBOX).name == "borough_33409013af7cbb1a.json"


def test_each_borough_is_served_from_its_own_area():
    assert _area_for(BROAD_AND_GREENWOOD, 130) == SNAPSHOT_AREAS["hopewell_borough"]
    assert _area_for(NJ31_AND_W_DELAWARE, 130) == SNAPSHOT_AREAS["pennington_borough"]


def test_the_two_areas_have_different_cache_keys():
    paths = {_snapshot_path(bbox).name for bbox in SNAPSHOT_AREAS.values()}
    assert len(paths) == len(SNAPSHOT_AREAS), "two areas sharing a cache file would overwrite"


def test_a_site_in_no_area_is_refused_and_told_which_areas_exist():
    """Trenton - a real place, and not one this project has a snapshot of."""
    with pytest.raises(SiteOutsideSnapshotError) as raised:
        assert_within_snapshot(Point(-74.7429, 40.2206), 130)
    message = str(raised.value)
    assert "hopewell_borough" in message and "pennington_borough" in message, message
    assert "SNAPSHOT_AREAS" in message, "the message must name what to edit"


def test_a_window_straddling_an_edge_is_refused_not_half_served():
    """The failure this whole guard exists for: a context window partly outside its area
    returns the elements that happen to be inside and NOTHING for the rest, which looks
    like geometry rather than like an error. A point just inside Hopewell's west edge with
    a radius that reaches past it must raise, not quietly return half a junction."""
    west_edge = Point(HOPEWELL_BBOX[0] + 0.0002, 40.3900)
    with pytest.raises(SiteOutsideSnapshotError):
        assert_within_snapshot(west_edge, 130)


def test_a_site_well_inside_an_area_passes():
    assert_within_snapshot(BROAD_AND_GREENWOOD, 130)
    assert_within_snapshot(NJ31_AND_W_DELAWARE, 250)
