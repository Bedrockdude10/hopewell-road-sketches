"""Matching a road network's centerline pieces to the legs a config declares.

An SRI is split at the junction and each piece is matched to a configured leg by bearing.
When the config declares FEWER legs on an SRI than the network splits it into, the road
runs through the junction but the config only says one side of it exists - which is a real
and easy mistake: NJDOT carries the whole of Delaware Ave in Pennington under one SRI named
"E DELAWARE AVE", so a config that reads the NJDOT name literally puts only the east leg on
it and the west leg on `PENNINGTON-TITUSVILLE RD`, whose segment actually starts 334 ft away
and never reaches this junction.

That mistake used to surface as `ValueError: min() iterable argument is empty` from inside
the matcher, which names neither the SRI, nor the leg, nor the config - the same class of
failure sites/README.md's schema validation exists to prevent.
"""
import re

import pytest
from shapely.geometry import LineString, Point

from src.geometry.intersection import _assign_leg_pieces

CENTRE = Point(0.0, 0.0)
NORTH = LineString([(0, 0), (0, 130)])
SOUTH = LineString([(0, 0), (0, -130)])
EAST = LineString([(0, 0), (130, 0)])

LEGS_CFG = {
    "main_north": {"bearing_deg": 0.0},
    "main_south": {"bearing_deg": 180.0},
    "cross_east": {"bearing_deg": 90.0},
}


def test_pieces_are_matched_to_legs_by_bearing():
    assigned = _assign_leg_pieces([NORTH, SOUTH], ["main_north", "main_south"], LEGS_CFG, CENTRE)
    assert assigned["main_north"] is NORTH
    assert assigned["main_south"] is SOUTH


def test_a_stub_leg_takes_the_only_piece():
    """One piece, one name - a dead-end approach is not an error."""
    assigned = _assign_leg_pieces([EAST], ["cross_east"], LEGS_CFG, CENTRE)
    assert assigned["cross_east"] is EAST


def test_more_pieces_than_legs_names_the_sri_and_the_bearing():
    """The through-road-declared-as-a-stub case, which is what crashed.

    The message has to carry enough to fix the config without reading this source: which
    SRI, how many pieces against how many legs, which legs were declared, and the bearing
    of the piece left over - that bearing IS the `bearing_deg` of the missing leg.
    """
    with pytest.raises(ValueError) as raised:
        _assign_leg_pieces([EAST, LineString([(0, 0), (-130, 0)])], ["cross_east"],
                           LEGS_CFG, CENTRE, sri="11081029__")
    message = str(raised.value)
    assert "11081029__" in message
    assert "cross_east" in message
    assert "2 piece" in message
    # The leftover piece runs due west, so the config is missing a leg at 270 degrees.
    assert re.search(r"\b270(\.\d)?\b", message), message


def test_more_legs_than_pieces_is_also_reported():
    """The opposite mistake: a leg declared on an SRI the network does not split there.

    Left unreported it is worse than the crash, because the leg simply never appears - no
    centerline, no curb, no crossing, and nothing said.
    """
    with pytest.raises(ValueError) as raised:
        _assign_leg_pieces([NORTH], ["main_north", "main_south"], LEGS_CFG, CENTRE,
                           sri="00000031__")
    message = str(raised.value)
    assert "00000031__" in message
    assert "main_south" in message, message
