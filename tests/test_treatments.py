"""Treatment functions: what they refuse to build, and what they say when they refuse.

A treatment that crashes with a ZeroDivisionError has told the reader nothing about the
junction. Both band treatments did: refuge_island and raise_crossing each interpolated two
stations along a leg and divided by the distance between them without checking it was
non-zero, and both stations clamp to the leg's far end whenever the treatment is asked for
past the end of the leg. That is reachable, not theoretical - leg_clearance_ft returns 133 ft
on W Broad & Louellen's 130 ft southwest leg, because its acute Y makes the corner return
consume the whole leg, and raise_crossing places itself at exactly that clearance.
"""
import pytest
from shapely.geometry import LineString

from src.geometry.model import Leg
from src.geometry.treatments import (DesignState, NACTO_MIN_REFUGE_ISLAND_WIDTH_FT,
                                     raise_crossing, refuge_island)


def a_state(length_ft=130.0, width_ft=30.0, corner_fillets=None):
    leg = Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]),
              curb_to_curb_ft=width_ft)
    return DesignState(legs={"east": leg}, corner_fillets=corner_fillets or {})


# --------------------------------------------------------------------------
# The bands: a shape with no extent along the road is not a shape.
# --------------------------------------------------------------------------

def test_a_raised_crossing_past_the_end_of_the_leg_says_so():
    """The W Broad & Louellen case: the corner return eats the leg, so the crossing's own
    start station is already past its far end and both ends clamp to the same point.

    raise_crossing takes its start from leg_clearance_ft rather than an argument, so the way
    to reach this is a corner whose tangent point sits beyond the leg's far end - which is
    exactly how the real junction reaches it (133 ft of clearance on a 130 ft leg).
    """
    state = a_state(length_ft=100.0, corner_fillets={
        ("east", "north"): {
            "trimmed_a": LineString([(200, 15), (250, 15)]),   # tangent point 100 ft past the leg
            "trimmed_b": LineString([(200, -15), (250, -15)]),
            "arc": LineString([(200, 15), (200, -15)]),
            "radius_ft": 20.0,
        }
    })
    with pytest.raises(ValueError) as caught:
        raise_crossing(state, "east", crossing_width_ft=10.0)
    assert "raised crossing" in str(caught.value)
    assert "same point" in str(caught.value)


def test_a_refuge_island_on_a_zero_length_span_says_so():
    with pytest.raises(ValueError) as caught:
        refuge_island(a_state(), "east", offset_ft=60.0,
                      width_ft=NACTO_MIN_REFUGE_ISLAND_WIDTH_FT, along_road_ft=0.0)
    assert "refuge island" in str(caught.value)
    assert "no extent along the road" in str(caught.value)


def test_a_refuge_island_below_the_nacto_minimum_is_refused():
    with pytest.raises(ValueError, match="NACTO minimum"):
        refuge_island(a_state(), "east", offset_ft=60.0,
                      width_ft=NACTO_MIN_REFUGE_ISLAND_WIDTH_FT - 1)


# --------------------------------------------------------------------------
# ...and the ordinary case still builds the shape it says it does.
# --------------------------------------------------------------------------

def test_a_refuge_island_spans_its_stated_width_across_the_road():
    state = refuge_island(a_state(), "east", offset_ft=60.0, width_ft=8.0, along_road_ft=20.0)
    island = next(iter(state.refuge_islands.values()))
    minx, miny, maxx, maxy = island["polygon"].bounds
    assert (maxy - miny) == pytest.approx(8.0)    # across the road: the stated width
    assert (maxx - minx) == pytest.approx(20.0)   # along the road: along_road_ft
    assert island["width_ft"] == 8.0


def test_a_raised_crossing_spans_curb_to_curb():
    state = raise_crossing(a_state(width_ft=34.0), "east", crossing_width_ft=12.0)
    minx, miny, maxx, maxy = state.raised_crossings["east"].bounds
    assert (maxy - miny) == pytest.approx(34.0)    # the full roadway
    assert (maxx - minx) == pytest.approx(12.0)


def test_a_leg_with_no_width_cannot_take_a_crossing():
    state = DesignState(legs={"east": Leg(name="east", centerline=LineString([(0, 0), (100, 0)]))},
                        corner_fillets={})
    with pytest.raises(ValueError, match="no curb lines"):
        raise_crossing(state, "east")
