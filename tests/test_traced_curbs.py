"""Building curb lines out of the surveyor's traced OSM kerbs.

Every test here is a bug that shipped. The theme running through all of them is the same:
the traced kerb IS the ground truth, and the failure mode is code quietly preferring a
derived guess to it - by dropping the tracing, by using only part of it, or by
extrapolating past it on a bearing taken from the wrong place.
"""
import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.model import (
    CURB_EXTRAPOLATION_MAX_SLOPE,
    Leg,
    assign_curb_points_to_legs,
    curb_line_from_points,
    station_offset_many,
)


def a_leg(name="east", width_ft=30.0, length_ft=120.0):
    return Leg(name=name, centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)


def offsets_of(curb, leg):
    _stations, offsets = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    return offsets


# --------------------------------------------------------------------------
# Which traced points belong to which leg side
# --------------------------------------------------------------------------

def test_a_traced_kerb_is_assigned_to_the_side_it_lies_on():
    leg = a_leg()
    north = LineString([(20, 15), (100, 15)])
    south = LineString([(20, -15), (100, -15)])
    assigned = assign_curb_points_to_legs({"east": leg}, [north, south])
    assert set(assigned["east"]) == {"left", "right"}
    assert all(o > 0 for _s, o in assigned["east"]["left"])
    assert all(o < 0 for _s, o in assigned["east"]["right"])


def test_a_kerb_behind_the_junction_is_not_claimed():
    """The opposite leg's curb must not be adopted and drawn back through the junction."""
    leg = a_leg()
    behind = LineString([(-100, 15), (-20, 15)])
    assert assign_curb_points_to_legs({"east": leg}, [behind]) == {}


def test_a_kerb_far_from_the_leg_is_not_claimed():
    """A parallel street a block away is not this leg's curb."""
    leg = a_leg()
    far = LineString([(20, 200), (100, 200)])
    assert assign_curb_points_to_legs({"east": leg}, [far]) == {}


def test_each_vertex_goes_to_exactly_one_leg_side():
    """A corner return splits between the two sides it joins - it is not drawn twice."""
    east = a_leg("east")
    north = Leg(name="north", centerline=LineString([(0, 0), (0, 120)]), curb_to_curb_ft=30.0)
    corner = LineString([(40, 15), (20, 16), (16, 20), (15, 40)])
    assigned = assign_curb_points_to_legs({"east": east, "north": north}, [corner])
    claimed = sum(len(points) for sides in assigned.values() for points in sides.values())
    assert claimed == len(corner.coords), "every vertex claimed exactly once"
    assert assigned["east"] and assigned["north"], "the return is shared between both legs"


# --------------------------------------------------------------------------
# Turning traced points into a curb line
# --------------------------------------------------------------------------

def test_the_curb_follows_the_traced_points():
    """No offsetting, no fitting - the traced position is the answer."""
    leg = a_leg()
    points = [(20.0, 15.0), (60.0, 14.0), (100.0, 13.5)]
    curb = curb_line_from_points(points, leg, working_length_ft=120.0)
    for station, offset in points:
        x, y = leg.centerline.interpolate(station).coords[0]
        assert curb.distance(LineString([(x, y + offset), (x, y + offset)]).centroid) < 1e-6


def test_extrapolation_beyond_the_tracing_is_bounded():
    """The Columbia & Princeton X.

    That leg was traced for 9 ft, all of it corner return. Taking the bearing from the last
    two vertices ran the return's flare out 100 ft and crossed the two curbs into an X, so
    the extrapolation is clamped to a few degrees off the street.
    """
    leg = a_leg()
    flaring = [(13.0, 15.0), (18.0, 22.0), (22.0, 28.0)]   # a corner return, steeply flared
    curb = curb_line_from_points(flaring, leg, working_length_ft=120.0)
    offsets = offsets_of(curb, leg)
    grew_by = abs(offsets[-1] - flaring[-1][1])
    ran_for = 120.0 - flaring[-1][0]
    assert grew_by <= ran_for * CURB_EXTRAPOLATION_MAX_SLOPE + 1e-6


def test_extrapolation_holds_the_last_width_when_the_tracing_is_short():
    """Too little traced to establish a bearing: continue at the width last seen."""
    leg = a_leg()
    curb = curb_line_from_points([(20.0, 15.0), (24.0, 16.0)], leg, working_length_ft=120.0)
    assert offsets_of(curb, leg)[-1] == pytest.approx(16.0, abs=1e-6)


def test_the_curb_reaches_the_leg_working_length():
    leg = a_leg()
    curb = curb_line_from_points([(20.0, 15.0), (60.0, 15.0)], leg, working_length_ft=120.0)
    stations, _ = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    assert stations.max() == pytest.approx(120.0, abs=1e-6)


def test_the_curb_does_not_run_back_into_the_junction():
    """It ends where the tracing ends; the corner is built from traced geometry, not by
    running this line on into the intersection."""
    leg = a_leg()
    curb = curb_line_from_points([(20.0, 15.0), (60.0, 15.0)], leg, working_length_ft=120.0)
    stations, _ = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    assert stations.min() >= 20.0 - 1e-6


def test_a_single_traced_point_is_not_a_curb():
    assert curb_line_from_points([(20.0, 15.0)], a_leg(), working_length_ft=120.0) is None


def test_points_are_ordered_along_the_leg():
    """Out-of-order input must not fold the curb back on itself."""
    leg = a_leg()
    curb = curb_line_from_points([(80.0, 15.0), (20.0, 15.0), (50.0, 15.0)], leg, working_length_ft=120.0)
    stations, _ = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    assert list(stations) == sorted(stations)
