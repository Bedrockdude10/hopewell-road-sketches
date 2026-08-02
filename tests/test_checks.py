"""Scene invariants: each one must fire on the failure it was written for.

A check that never fires is worse than no check, because it reads as coverage. So every
invariant here is tested twice - once on geometry that violates it, once on geometry that
doesn't - against a synthetic junction rather than a site, so these run in milliseconds and
can't be broken by re-tracing a kerb in OSM.
"""
import pytest
from shapely.geometry import LineString, Polygon

from src.checks import (
    SceneInvariantError,
    Violation,
    assert_scene_valid,
    check_crosswalks_cross_the_roadway,
    check_curbs_clear_of_junction,
    check_curbs_do_not_cross,
    check_furniture_off_roadway,
    check_pads_against_a_curb,
    check_stop_bars_on_entering_half,
)
from src.geometry.model import Leg

# A plain crossroads: a 30 ft wide east-west street, roadway from y=-15 to y=+15.
ROADWAY = Polygon([(-120, -15), (120, -15), (120, 15), (-120, 15)])


def a_leg(name="east", width_ft=30.0):
    leg = Leg(name=name, centerline=LineString([(0, 0), (120, 0)]), curb_to_curb_ft=width_ft)
    return leg


def prop(kind, position, **extra):
    return {"type": kind, "position_ft": position, "heading_deg": 0.0, **extra}


# --------------------------------------------------------------------------
# Nothing that belongs on the footway may be in the street. The headline case.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [
    "stop_sign", "yield_sign", "no_turn_on_red_sign", "traffic_signal_pole",
    "pedestrian_signal_head", "pedestrian_pushbutton", "streetlight",
])
def test_a_sign_in_the_street_is_a_violation(kind):
    violations = check_furniture_off_roadway([prop(kind, (60.0, 0.0))], ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "furniture_in_roadway"
    assert violations[0].fatal


@pytest.mark.parametrize("kind", ["stop_sign", "traffic_signal_pole", "pedestrian_pushbutton"])
def test_the_same_sign_on_the_footway_is_fine(kind):
    assert check_furniture_off_roadway([prop(kind, (60.0, 22.0))], ROADWAY) == []


def test_tactile_paving_in_the_street_is_a_violation():
    """The one that shipped twice: a detectable warning surface drawn in the carriageway."""
    violations = check_furniture_off_roadway([prop("tactile_paving_pad", (60.0, 0.0))], ROADWAY)
    assert len(violations) == 1
    assert "tactile paving pad" in violations[0].detail
    assert violations[0].fatal


def test_tactile_paving_on_the_footway_is_fine():
    assert check_furniture_off_roadway([prop("tactile_paving_pad", (60.0, 18.0))], ROADWAY) == []


def test_tactile_paving_grazing_the_kerb_is_tolerated():
    """A pad butts up against the kerb by design; polygon tolerance must not fail it.

    The pad is TACTILE_PAD_WIDTH_FT (3 ft) across the kerb, so a pad sitting exactly at the
    kerb is centred 1.5 ft behind it - here y = 15 + 1.5. A hair over the line is tolerance,
    not a pad in the road.
    """
    assert check_furniture_off_roadway([prop("tactile_paving_pad", (60.0, 16.5))], ROADWAY) == []
    assert check_furniture_off_roadway([prop("tactile_paving_pad", (60.0, 16.49))], ROADWAY) == []


def test_a_pad_half_in_the_road_is_still_caught():
    """The original bug was pads CENTRED on the kerb line - half in the carriageway."""
    violations = check_furniture_off_roadway([prop("tactile_paving_pad", (60.0, 15.0))], ROADWAY)
    assert len(violations) == 1
    assert "50%" in violations[0].detail


def test_bollards_are_allowed_in_the_roadway():
    """Bollards and delineators are placed in the carriageway deliberately."""
    assert check_furniture_off_roadway([prop("bollard", (60.0, 0.0))], ROADWAY) == []


def test_an_unknown_prop_type_is_checked_by_default():
    """A new prop type must be covered without anyone remembering to add it."""
    violations = check_furniture_off_roadway([prop("school_zone_sign", (60.0, 0.0))], ROADWAY)
    assert len(violations) == 1


def test_a_surveyed_position_in_the_roadway_is_reported_but_not_fatal():
    """An OSM node we can't move that lands in our roadway is a source conflict.

    Worth saying every run - one of the two sources is wrong - but no edit to this repo
    fixes it, so it must not block the site from rendering forever.
    """
    violations = check_furniture_off_roadway(
        [prop("fire_hydrant", (60.0, 0.0), surveyed_position=True)], ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "surveyed_furniture_in_roadway"
    assert not violations[0].fatal


# --------------------------------------------------------------------------
# A pad marks a ramp, so it belongs at a kerb
# --------------------------------------------------------------------------

def test_a_pad_far_from_any_kerb_is_a_violation():
    legs = {"east": a_leg()}
    violations = check_pads_against_a_curb([prop("tactile_paving_pad", (60.0, 60.0))], legs, {})
    assert len(violations) == 1
    assert violations[0].check == "pad_off_the_kerb"


def test_a_pad_at_the_kerb_is_fine():
    legs = {"east": a_leg()}
    assert check_pads_against_a_curb([prop("tactile_paving_pad", (60.0, 16.0))], legs, {}) == []


# --------------------------------------------------------------------------
# Curbs
# --------------------------------------------------------------------------

def test_a_curb_running_back_through_the_junction_is_a_violation():
    """The curb-across-the-middle-of-the-intersection bug, in its simplest form."""
    leg = a_leg()
    leg.left_curb = LineString([(-60, 15), (120, 15)])    # starts 60 ft behind the junction
    leg.right_curb = LineString([(0, -15), (120, -15)])
    violations = check_curbs_clear_of_junction({"east": leg})
    assert len(violations) == 1
    assert violations[0].check == "curb_through_junction"


def test_a_curb_starting_at_the_junction_is_fine():
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, 15)])
    leg.right_curb = LineString([(0, -15), (120, -15)])
    assert check_curbs_clear_of_junction({"east": leg}) == []


def test_curbs_that_cross_each_other_are_a_violation():
    """Extrapolating a curb from a corner return's flare closed the roadway to nothing."""
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, -15)])     # converging
    leg.right_curb = LineString([(0, -15), (120, 15)])
    violations = check_curbs_do_not_cross({"east": leg})
    assert len(violations) == 1
    assert violations[0].check == "curbs_cross"


def test_parallel_curbs_do_not_cross():
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, 15)])
    leg.right_curb = LineString([(0, -15), (120, -15)])
    assert check_curbs_do_not_cross({"east": leg}) == []


# --------------------------------------------------------------------------
# Crosswalks and stop bars
# --------------------------------------------------------------------------

def test_a_crosswalk_outside_the_roadway_is_a_violation():
    stranded = Polygon([(60, 40), (66, 40), (66, 70), (60, 70)])
    violations = check_crosswalks_cross_the_roadway({"east": stranded}, ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "crosswalk_off_the_roadway"


def test_a_crosswalk_across_the_roadway_is_fine():
    band = Polygon([(57, -16), (63, -16), (63, 16), (57, 16)])
    assert check_crosswalks_cross_the_roadway({"east": band}, ROADWAY) == []


def test_a_stop_bar_across_both_directions_is_a_violation():
    """A driver stops in their own lanes; the bar covers the entering half only."""
    leg = a_leg()
    full_width = Polygon([(58, -15), (60, -15), (60, 15), (58, 15)])
    violations = check_stop_bars_on_entering_half({"east": full_width}, {"east": leg})
    assert len(violations) == 1
    assert violations[0].check == "stop_bar_crosses_centerline"


def test_a_stop_bar_on_one_half_is_fine():
    leg = a_leg()
    half = Polygon([(58, 0.5), (60, 0.5), (60, 15), (58, 15)])
    assert check_stop_bars_on_entering_half({"east": half}, {"east": leg}) == []


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_all_violations_are_reported_together():
    """Failing on the first violation turns one bad junction into N edit-run cycles."""
    class FakeState:
        legs = {}
        corner_fillets = {}

    class FakeModel:
        config = {"intersection": {"name": "Test Junction"}}

    props = [prop("stop_sign", (60.0, 0.0)), prop("tactile_paving_pad", (40.0, 0.0)),
             prop("streetlight", (20.0, 0.0))]
    with pytest.raises(SceneInvariantError) as excinfo:
        assert_scene_valid(FakeModel(), FakeState(), props, ROADWAY)
    message = str(excinfo.value)
    assert "3 scene invariant(s) failed" in message
    for kind in ("stop_sign", "tactile paving pad", "streetlight"):
        assert kind in message


def test_a_violation_carries_its_coordinates():
    """The plan view draws these, so the message and the picture agree on where to look."""
    violations = check_furniture_off_roadway([prop("stop_sign", (60.0, 0.0))], ROADWAY)
    assert violations[0].where == (60.0, 0.0)
    assert "(60.0, 0.0)" in str(violations[0])


def test_non_fatal_violations_alone_do_not_raise():
    class FakeState:
        legs = {}
        corner_fillets = {}

    class FakeModel:
        config = {"intersection": {"name": "Test Junction"}}

    assert_scene_valid(FakeModel(), FakeState(),
                        [prop("fire_hydrant", (60.0, 0.0), surveyed_position=True)], ROADWAY)


def test_violation_str_is_readable_without_coordinates():
    assert str(Violation("some_check", "something is wrong")) == "[some_check] something is wrong"
