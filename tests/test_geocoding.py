"""Resolving a junction from two street names, when one of the streets has no name.

`geocode_intersection` is how every site's `center_wgs84` is resolved, and how
sites/README.md promises re-resolving it later is one command. It matched OSM's `name`
tag and nothing else, which quietly excluded a whole class of road: a state highway is
routinely mapped with a `ref` and no `name` at all. NJ 31 is - every way at NJ 31 & W
Delaware Ave in Pennington carries `ref=NJ 31` and no name - so the junction could not be
resolved by the documented command, and the failure said "could not find OSM ways
matching 'New Jersey 31'", which reads like a spelling problem rather than a tag that
does not exist.
"""
import pytest
from shapely.geometry import Point

from src.sources import data_loader
from src.sources.data_loader import geocode_intersection


def a_way(way_id: int, tags: dict, coords: list[tuple[float, float]], node_ids: list[int]):
    """One Overpass `out geom` way: tags, geometry and the positionally-paired node ids."""
    return {"type": "way", "id": way_id, "tags": tags,
            "nodes": node_ids,
            "geometry": [{"lon": lon, "lat": lat} for lon, lat in coords]}


ANCHOR = Point(-74.7989, 40.3272)
JUNCTION = (-74.7989728, 40.3272925)

# NJ 31 through the junction: a ref and no name, exactly as OSM has it. W Delaware Ave
# crosses it with a name and no ref. They share node 104089792, the real junction node.
A_NAMELESS_HIGHWAY = [
    a_way(1375777324, {"highway": "trunk", "ref": "NJ 31"},
          [JUNCTION, (-74.7989, 40.3280)], [104089792, 2]),
    a_way(61043008, {"highway": "tertiary", "name": "West Delaware Avenue"},
          [JUNCTION, (-74.8000, 40.3272)], [104089792, 3]),
]


@pytest.fixture
def offline_overpass(monkeypatch):
    """Answer both network calls from the fixture above - no Overpass, no Nominatim."""
    def serve(elements):
        monkeypatch.setattr(data_loader, "approximate_geocode", lambda q: ANCHOR)
        monkeypatch.setattr(data_loader, "query_overpass", lambda q: {"elements": elements})
    return serve


def test_a_street_matched_by_ref_resolves_the_junction(offline_overpass):
    """The regression: 'NJ 31' has to match `ref` because there is no `name` to match.

    Against the pre-change code this raises ValueError - `nodes_of` reads `tags["name"]`
    only, so the highway contributes no nodes and the two streets share none.
    """
    offline_overpass(A_NAMELESS_HIGHWAY)
    point = geocode_intersection("NJ 31", "West Delaware Avenue",
                                 "West Delaware Avenue, Pennington, NJ 08534")
    assert point.x == pytest.approx(JUNCTION[0])
    assert point.y == pytest.approx(JUNCTION[1])


def test_matching_by_name_still_wins_where_both_tags_exist(offline_overpass):
    """A ref match must not broaden a name that was already resolving correctly.

    Ingleside Avenue carries `ref=CR 631`, and a bare substring search over both tags
    would let the query for "631" pick up anything. The name is what was asked for, so
    the way that carries it is the one that answers.
    """
    ways = [
        a_way(1, {"highway": "tertiary", "name": "Ingleside Avenue", "ref": "CR 631"},
              [JUNCTION, (-74.7989, 40.3280)], [50, 2]),
        a_way(2, {"highway": "residential", "name": "North Main Street", "ref": "CR 640"},
              [JUNCTION, (-74.8000, 40.3272)], [50, 3]),
    ]
    offline_overpass(ways)
    point = geocode_intersection("Ingleside Avenue", "North Main Street", "Pennington, NJ")
    assert point.x == pytest.approx(JUNCTION[0])


def test_a_street_that_matches_nothing_still_names_itself(offline_overpass):
    """The error has to say which street failed, whichever tag it was looked for in."""
    offline_overpass(A_NAMELESS_HIGHWAY)
    with pytest.raises(ValueError, match="Yellow Brick Road"):
        geocode_intersection("Yellow Brick Road", "West Delaware Avenue", "Pennington, NJ")
