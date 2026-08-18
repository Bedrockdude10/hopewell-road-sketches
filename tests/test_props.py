"""Sign placement along a leg, at the one station that has no direction: the far end.

A leg is only as long as its kerb is traced (sites/README.md, working_length_ft), and an OSM
stop node is placed where the stop line really is - so the two disagree whenever a junction is
traced less far than it is signed. At Princeton Ave & E Prospect St the East Prospect legs are
drawn 30 and 35 ft while the stop nodes sit 31 and 33 ft out, which is the first time in this
project a sign has been asked for at or past a centerline's end.
"""
import math

import pytest
from shapely.geometry import LineString

from src.geometry.model import Leg
from src.render.props import _leg_sign_position_ft
from tests.conftest import needs_source_data


def a_leg(length_ft=30.0):
    """A straight 30 ft leg running due east, with no traced kerb - the shape of an East
    Prospect St approach at princeton_eprospect."""
    return Leg(name="short_leg", centerline=LineString([(0, 0), (length_ft, 0)]),
               curb_to_curb_ft=30.5, traced_sides=set())


@pytest.mark.parametrize("offset_ft", [30.0, 31.0, 33.0, 100.0])
def test_a_sign_at_or_past_the_end_of_a_leg_still_has_a_heading(offset_ft):
    """The bug: interpolate() clamps to the endpoint, so a sign at or past the end took its
    direction from a point minus itself - a zero vector, normalised to NaN. The prop was then
    emitted with a NaN heading, which is not caught by any scene invariant (they check
    position, not rotation) and reaches Blender as an unrenderable rotation.
    """
    pos, heading = _leg_sign_position_ft(a_leg(), offset_ft, side="left")
    assert pos is not None
    assert not math.isnan(heading), f"heading is NaN for a sign {offset_ft} ft along a 30 ft leg"
    assert all(not math.isnan(c) for c in pos), f"position is NaN at {offset_ft} ft"


def test_a_sign_past_the_end_faces_the_same_way_as_one_just_inside():
    """A leg does not change direction at its last foot, so neither should the sign."""
    _, h_inside = _leg_sign_position_ft(a_leg(), 20.0, side="left")
    _, h_end = _leg_sign_position_ft(a_leg(), 30.0, side="left")
    assert h_inside == pytest.approx(h_end, abs=1e-6)


@needs_source_data
def test_the_school_service_road_opens_the_kerb_it_meets():
    """A vehicle entrance 46 ft from the junction centre is an opening, not part of the junction.

    Hopewell Elementary's service road (OSM way 845227293, highway=service, maxspeed=5 mph) meets
    Princeton Ave 46 ft south of the junction centre. It was excluded by cross_streets.py's
    JUNCTION_OWN_REACH_FT test, which asked how far along the LEG the meeting point was rather
    than whether the WAY is one of this junction's own arms - so the school's entrance was
    swallowed by a filter meant for the four legs that meet at the middle, and the lane-narrowing
    hatching was painted straight across the drive children are dropped off in.

    Its dropped kerb cannot rescue it either: the two kerb=lowered ways at its mouth carry no
    `wheelchair` tag, which src/geometry/kerbs.py:opens_the_kerb deliberately reads as
    "unspecified, does not open".
    """
    from src.geometry.intersection import load_intersection_model
    from src.geometry.treatments import DesignState

    model = load_intersection_model(site="princeton_eprospect")
    state = DesignState.from_model(model)
    south = [o for (leg, _side), openings in state.kerb_openings.items()
             for o in openings if leg == "princeton_ave_south"]
    away_from_the_junction = [o for o in south if o.start_ft > 30]
    assert away_from_the_junction, (
        "no kerb opening on princeton_ave_south beyond the junction mouth - the school service "
        f"road produced none. Openings found: {south}")
