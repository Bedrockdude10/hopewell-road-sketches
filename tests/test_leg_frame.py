"""The leg frame: (station along the centerline, signed offset from it).

Every curb, sign, pad and band position in this project is ultimately expressed in this
frame, so a defect here doesn't stay local - it becomes a curb drawn through the junction
or a pad in the road. Two of those shipped and both trace back to a property asserted here.
"""
import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.model import _point_at, station_offset, station_offset_many

STRAIGHT = LineString([(0, 0), (100, 0)])
BENT = LineString([(0, 0), (50, 3), (110, -2), (160, 6)])   # an old street, not quite straight


def test_offset_sign_is_left_positive():
    """Positive offset is to the left of travel, matching Leg.left_curb."""
    _station, offset = station_offset(STRAIGHT, (50, 10))
    assert offset > 0
    _station, offset = station_offset(STRAIGHT, (50, -10))
    assert offset < 0


def test_station_is_negative_behind_the_junction():
    """The bug that let one leg claim the opposite leg's curb.

    LineString.project() clamps to [0, length], so a point behind the junction came back at
    station 0 with a small offset - indistinguishable from a point right at the corner. The
    leg then adopted the opposite leg's curb and drew it back through the intersection.
    """
    station, _offset = station_offset(STRAIGHT, (-40, 12))
    assert station < 0, "a point behind the junction must have a negative station"


def test_station_continues_past_the_far_end():
    """Past the end, stations must keep increasing rather than piling up on the last vertex."""
    near, _ = station_offset(STRAIGHT, (120, 5))
    far, _ = station_offset(STRAIGHT, (180, 5))
    assert far > near > STRAIGHT.length


@pytest.mark.parametrize("station,offset", [(10, 12.5), (0, -8), (-5, 20), (140, -30), (200, 9), (-40, 15)])
def test_round_trip_is_exact(station, offset):
    """_point_at and station_offset must be exact inverses over the whole line.

    They weren't when the forward direction used segment tangents and the inverse estimated
    one from a +/-2 ft window: a curb rebuilt from its own traced points came back shifted.
    """
    x, y = _point_at(BENT, station, offset)
    back_station, back_offset = station_offset(BENT, (x, y))
    assert back_station == pytest.approx(station, abs=1e-6)
    assert back_offset == pytest.approx(offset, abs=1e-6)


def test_vectorized_matches_scalar():
    """The many-point path is the one used in anger; it must agree with the single-point one."""
    points = np.random.default_rng(0).uniform(-60, 240, size=(200, 2))
    stations, offsets = station_offset_many(BENT, points)
    for i in range(0, len(points), 13):
        s, o = station_offset(BENT, tuple(points[i]))
        assert stations[i] == pytest.approx(s, abs=1e-9)
        assert offsets[i] == pytest.approx(o, abs=1e-9)


def test_offset_equals_perpendicular_distance_on_a_straight_leg():
    stations, offsets = station_offset_many(STRAIGHT, np.array([[25.0, 7.0], [75.0, -3.5]]))
    assert stations == pytest.approx([25.0, 75.0])
    assert offsets == pytest.approx([7.0, -3.5])
