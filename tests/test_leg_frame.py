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


# --------------------------------------------------------------------------
# The crossing frame is part of the leg frame
# --------------------------------------------------------------------------

def a_leg(centerline, width_ft=42.0):
    from src.geometry.model import Leg

    return Leg(name="louellen", centerline=centerline, curb_to_curb_ft=width_ft)


def test_a_crossing_is_square_to_the_street_at_its_own_station():
    """crosswalk_axes used to extrapolate the leg's FIRST SEGMENT, which is exact only for a
    straight centerline. Ten of this project's twelve legs are straight 2-vertex lines, so
    the shortcut survived; louellen_st_west is not.

    Its centerline leaves the junction at 239.2 deg for 15.4 ft and then runs 268.6 - NJDOT
    rounding the corner where CR 518 turns off W Broad onto Louellen - a 29.4 deg bend. The
    crossing at station 31.5 came out 29.4 deg off square to the street it crosses, with its
    centre ~8 ft to the side of the real carriageway centre.
    """
    from src.render.crosswalks import crosswalk_axes

    # 240 deg for 15 ft, then due west - Louellen's shape, in round numbers.
    bend = LineString([(0, 0), (-13.0, -7.5), (-130.0, -7.5)])
    leg = a_leg(bend)

    centre, _u, across, _cos = crosswalk_axes(leg, 31.5)
    # On the centerline, not beside it.
    station, offset = station_offset(bend, centre)
    assert station == pytest.approx(31.5, abs=0.01)
    assert offset == pytest.approx(0.0, abs=0.01)
    # And spanning across the street it is actually crossing: due west street, so the bars
    # run north-south.
    assert abs(across[1]) == pytest.approx(1.0, abs=0.01), f"bars not square to the street: {across}"
    assert abs(across[0]) == pytest.approx(0.0, abs=0.01)


def test_a_crossing_inside_the_first_segment_is_where_it_always_was():
    """The complement, and what bounds the change: reading the frame at the station is
    identical to extrapolating segment one for any station that falls inside segment one.
    broad_st_east bends 4.5 deg but does so 43.1 ft out, past its crossing, so nothing about
    that crossing moved."""
    from src.render.crosswalks import crosswalk_axes

    bend = LineString([(0, 0), (43.1, 0), (150.0, 8.4)])
    centre, u, across, _cos = crosswalk_axes(a_leg(bend), 21.3)
    assert centre == pytest.approx((21.3, 0.0), abs=1e-9)
    assert u == pytest.approx((1.0, 0.0), abs=1e-9)
    assert across == pytest.approx((0.0, 1.0), abs=1e-9)


def test_the_crossing_centre_round_trips_through_the_leg_frame():
    """Whatever the shape of the leg: the station a crossing is BUILT at has to be the
    station everything else MEASURES it at, or the reach, the band and the paint that keeps
    clear of it are each working from a different crossing."""
    from src.render.crosswalks import crosswalk_axes

    for station in (5.0, 31.5, 60.0, 129.0):
        centre, _u, _n, _cos = crosswalk_axes(a_leg(BENT), station)
        back, offset = station_offset(BENT, centre)
        assert back == pytest.approx(station, abs=1e-6)
        assert offset == pytest.approx(0.0, abs=1e-6)
