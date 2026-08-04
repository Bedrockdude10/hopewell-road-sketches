"""Scene invariants: each one must fire on the failure it was written for.

A check that never fires is worse than no check, because it reads as coverage. So every
invariant here is tested twice - once on geometry that violates it, once on geometry that
doesn't - against a synthetic junction rather than a site, so these run in milliseconds and
can't be broken by re-tracing a kerb in OSM.
"""
import pytest
from shapely.geometry import LineString, Polygon

from src.checks import (
    CrosswalksCrossTheRoadway,
    CurbsClearOfJunction,
    CurbsDoNotCross,
    FurnitureOffRoadway,
    PadsAgainstACurb,
    SceneContext,
    SceneInvariantError,
    StopBarsOnEnteringHalf,
    Violation,
    assert_scene_valid,
)
from src.geometry.model import Leg
from src.geometry.treatments import DesignState


def run(check, **fields):
    """One invariant against a scene built from only the fields it reads.

    Every check takes the same SceneContext now, so a test says which parts of a scene it is
    describing instead of matching a positional signature - see src/checks.py:SceneContext for
    why the per-check argument list had to go.
    """
    return check.run(SceneContext(**fields))


def a_state(legs=None, corner_fillets=None):
    """A DesignState with nothing applied to it. The checks that used to take a raw dict of
    legs now read them off the state, so the fixture is the real type."""
    return DesignState(legs=legs or {}, corner_fillets=corner_fillets or {})

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
    violations = run(FurnitureOffRoadway(), props=[prop(kind, (60.0, 0.0))], pavement=ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "furniture_in_roadway"
    assert violations[0].fatal


@pytest.mark.parametrize("kind", ["stop_sign", "traffic_signal_pole", "pedestrian_pushbutton"])
def test_the_same_sign_on_the_footway_is_fine(kind):
    assert run(FurnitureOffRoadway(), props=[prop(kind, (60.0, 22.0))], pavement=ROADWAY) == []


def test_tactile_paving_in_the_street_is_a_violation():
    """The one that shipped twice: a detectable warning surface drawn in the carriageway."""
    violations = run(FurnitureOffRoadway(), props=[prop("tactile_paving_pad", (60.0, 0.0))], pavement=ROADWAY)
    assert len(violations) == 1
    assert "tactile paving pad" in violations[0].detail
    assert violations[0].fatal


def test_tactile_paving_on_the_footway_is_fine():
    assert run(FurnitureOffRoadway(), props=[prop("tactile_paving_pad", (60.0, 18.0))], pavement=ROADWAY) == []


def test_tactile_paving_grazing_the_kerb_is_tolerated():
    """A pad butts up against the kerb by design; polygon tolerance must not fail it.

    The pad is TACTILE_PAD_WIDTH_FT (3 ft) across the kerb, so a pad sitting exactly at the
    kerb is centred 1.5 ft behind it - here y = 15 + 1.5. A hair over the line is tolerance,
    not a pad in the road.
    """
    assert run(FurnitureOffRoadway(), props=[prop("tactile_paving_pad", (60.0, 16.5))], pavement=ROADWAY) == []
    assert run(FurnitureOffRoadway(), props=[prop("tactile_paving_pad", (60.0, 16.49))], pavement=ROADWAY) == []


def test_a_pad_half_in_the_road_is_still_caught():
    """The original bug was pads CENTRED on the kerb line - half in the carriageway."""
    violations = run(FurnitureOffRoadway(), props=[prop("tactile_paving_pad", (60.0, 15.0))], pavement=ROADWAY)
    assert len(violations) == 1
    assert "50%" in violations[0].detail


def test_bollards_are_allowed_in_the_roadway():
    """Bollards and delineators are placed in the carriageway deliberately."""
    assert run(FurnitureOffRoadway(), props=[prop("bollard", (60.0, 0.0))], pavement=ROADWAY) == []


def test_an_unknown_prop_type_is_checked_by_default():
    """A new prop type must be covered without anyone remembering to add it."""
    violations = run(FurnitureOffRoadway(), props=[prop("school_zone_sign", (60.0, 0.0))], pavement=ROADWAY)
    assert len(violations) == 1


def test_a_surveyed_position_in_the_roadway_is_reported_but_not_fatal():
    """An OSM node we can't move that lands in our roadway is a source conflict.

    Worth saying every run - one of the two sources is wrong - but no edit to this repo
    fixes it, so it must not block the site from rendering forever.
    """
    violations = run(FurnitureOffRoadway(),
        props=[prop("fire_hydrant", (60.0, 0.0), surveyed_position=True)], pavement=ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "surveyed_furniture_in_roadway"
    assert not violations[0].fatal


# --------------------------------------------------------------------------
# A pad marks a ramp, so it belongs at a kerb
# --------------------------------------------------------------------------

def test_a_pad_far_from_any_kerb_is_a_violation():
    legs = {"east": a_leg()}
    violations = run(PadsAgainstACurb(), props=[prop("tactile_paving_pad", (60.0, 60.0))], state=a_state(legs))
    assert len(violations) == 1
    assert violations[0].check == "pad_off_the_kerb"


def test_a_pad_at_the_kerb_is_fine():
    legs = {"east": a_leg()}
    assert run(PadsAgainstACurb(), props=[prop("tactile_paving_pad", (60.0, 16.0))], state=a_state(legs)) == []


# --------------------------------------------------------------------------
# Curbs
# --------------------------------------------------------------------------

def test_a_curb_running_back_through_the_junction_is_a_violation():
    """The curb-across-the-middle-of-the-intersection bug, in its simplest form."""
    leg = a_leg()
    leg.left_curb = LineString([(-60, 15), (120, 15)])    # starts 60 ft behind the junction
    leg.right_curb = LineString([(0, -15), (120, -15)])
    violations = run(CurbsClearOfJunction(), state=a_state({"east": leg}))
    assert len(violations) == 1
    assert violations[0].check == "curb_through_junction"


def test_a_curb_starting_at_the_junction_is_fine():
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, 15)])
    leg.right_curb = LineString([(0, -15), (120, -15)])
    assert run(CurbsClearOfJunction(), state=a_state({"east": leg})) == []


def test_curbs_that_cross_each_other_are_a_violation():
    """Extrapolating a curb from a corner return's flare closed the roadway to nothing."""
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, -15)])     # converging
    leg.right_curb = LineString([(0, -15), (120, 15)])
    violations = run(CurbsDoNotCross(), state=a_state({"east": leg}))
    assert len(violations) == 1
    assert violations[0].check == "curbs_cross"


def test_parallel_curbs_do_not_cross():
    leg = a_leg()
    leg.left_curb = LineString([(0, 15), (120, 15)])
    leg.right_curb = LineString([(0, -15), (120, -15)])
    assert run(CurbsDoNotCross(), state=a_state({"east": leg})) == []


# --------------------------------------------------------------------------
# Crosswalks and stop bars
# --------------------------------------------------------------------------

def test_a_crosswalk_outside_the_roadway_is_a_violation():
    stranded = Polygon([(60, 40), (66, 40), (66, 70), (60, 70)])
    violations = run(CrosswalksCrossTheRoadway(), crosswalk_bands={"east": stranded}, pavement=ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "crosswalk_off_the_roadway"


def test_a_crosswalk_meeting_both_kerbs_is_fine():
    """Kerb to kerb, and no further. The roadway here is +/-15 ft."""
    band = Polygon([(57, -15), (63, -15), (63, 15), (57, 15)])
    assert run(CrosswalksCrossTheRoadway(), crosswalk_bands={"east": band}, pavement=ROADWAY) == []


def test_a_crosswalk_overhanging_the_kerb_is_a_violation():
    """This band overhangs by 1 ft at each kerb - 6% of its area on the footway.

    It used to pass: the tolerance was 55% inside, from when a crossing was drawn as half
    the leg's NOMINAL width either side of the centerline and routinely overshot the traced
    kerb. That slack hid the real thing it was named for - a skewed crossing's ray running
    diagonally up a corner return and painting the end bars 12 ft onto the sidewalk.
    """
    band = Polygon([(57, -16), (63, -16), (63, 16), (57, 16)])
    violations = run(CrosswalksCrossTheRoadway(), crosswalk_bands={"east": band}, pavement=ROADWAY)
    assert len(violations) == 1
    assert violations[0].check == "crosswalk_off_the_roadway"


def test_a_stop_bar_across_both_directions_is_a_violation():
    """A driver stops in their own lanes; the bar covers the entering half only."""
    leg = a_leg()
    full_width = Polygon([(58, -15), (60, -15), (60, 15), (58, 15)])
    violations = run(StopBarsOnEnteringHalf(), stop_bars={"east": full_width}, state=a_state({"east": leg}))
    assert len(violations) == 1
    assert violations[0].check == "stop_bar_crosses_centerline"


def test_a_stop_bar_on_one_half_is_fine():
    leg = a_leg()
    half = Polygon([(58, 0.5), (60, 0.5), (60, 15), (58, 15)])
    assert run(StopBarsOnEnteringHalf(), stop_bars={"east": half}, state=a_state({"east": leg})) == []


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
        assert_scene_valid(SceneContext(model=FakeModel(), state=FakeState(),
                                         props=props, pavement=ROADWAY))
    message = str(excinfo.value)
    assert "3 scene invariant(s) failed" in message
    for kind in ("stop_sign", "tactile paving pad", "streetlight"):
        assert kind in message


def test_a_violation_carries_its_coordinates():
    """The plan view draws these, so the message and the picture agree on where to look."""
    violations = run(FurnitureOffRoadway(), props=[prop("stop_sign", (60.0, 0.0))], pavement=ROADWAY)
    assert violations[0].where == (60.0, 0.0)
    assert "(60.0, 0.0)" in str(violations[0])


def test_non_fatal_violations_alone_do_not_raise():
    class FakeState:
        legs = {}
        corner_fillets = {}

    class FakeModel:
        config = {"intersection": {"name": "Test Junction"}}

    assert_scene_valid(SceneContext(
        model=FakeModel(), state=FakeState(),
        props=[prop("fire_hydrant", (60.0, 0.0), surveyed_position=True)], pavement=ROADWAY))


def test_violation_str_is_readable_without_coordinates():
    assert str(Violation("some_check", "something is wrong")) == "[some_check] something is wrong"


# --------------------------------------------------------------------------
# Travel lane width
# --------------------------------------------------------------------------

def a_state_with_paint(width_ft, hatch_ft=None, parking_ft=None):
    from src.geometry.treatments import DesignState

    leg = a_leg(width_ft=width_ft)
    state = DesignState(legs={"east": leg}, corner_fillets={})
    if hatch_ft is not None:
        state.lane_narrowing["east"] = hatch_ft
        state.lane_narrowing_sides["east"] = ("left",)
    if parking_ft is not None:
        state.parking_zones[("east", "right")] = {"depth_ft": parking_ft, "stall_length_ft": 22,
                                                   "curb_offset_ft": 0.0}
    return state


def test_paint_that_leaves_a_narrow_lane_is_a_violation():
    """The 1.7 ft lanes: fixed-width paint applied without checking what the road can spare."""
    from src.checks import TravelLanesKeepTheirWidth

    # 30 ft road, half is 15 ft; 8 ft of parking leaves a 7 ft lane.
    violations = run(TravelLanesKeepTheirWidth(), state=a_state_with_paint(30.0, parking_ft=8.0))
    assert len(violations) == 1
    assert violations[0].check == "travel_lane_too_narrow"


def test_paint_sized_to_leave_the_target_is_fine():
    from src.checks import TravelLanesKeepTheirWidth
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT

    # 30 ft road: 4 ft of paint leaves exactly 11 ft.
    allowance = 15.0 - TARGET_LANE_WIDTH_FT
    assert run(TravelLanesKeepTheirWidth(), state=a_state_with_paint(30.0, parking_ft=allowance)) == []
    assert run(TravelLanesKeepTheirWidth(), state=a_state_with_paint(30.0, hatch_ft=allowance)) == []


def test_a_naturally_narrow_street_is_not_our_error():
    """Louellen Street is 19.3 ft curb to curb. Its lanes are under target because the
    street is, not because a treatment did it - and no check widens a road."""
    from src.checks import TravelLanesKeepTheirWidth

    assert run(TravelLanesKeepTheirWidth(), state=a_state_with_paint(19.3)) == []


# --------------------------------------------------------------------------
# Paint layering and hatch quality
# --------------------------------------------------------------------------

def test_degenerate_hatch_strokes_are_dropped():
    """Clipping produces stubs where a stroke grazes a corner or a taper's thin tip.

    One came out 0.0 ft long. They render as strokes sheared off mid-buffer.
    """
    from src.geometry.model import MIN_HATCH_STROKE_FT, hatch_lines_ft

    # A wedge: strokes near the point are arbitrarily short.
    wedge = Polygon([(0, 0), (60, 0), (60, 12)])
    strokes = hatch_lines_ft(wedge, spacing_ft=1.0, angle_deg=45)
    assert strokes, "the wedge should still be hatched"
    assert min(s.length for s in strokes) >= MIN_HATCH_STROKE_FT


def test_paint_is_cut_around_a_keep_clear_area():
    """A crosswalk outranks a buffer, so the buffer is cut around it geometrically.

    Relying on the paint's start station instead let two strokes land on Broad St's crossing:
    a SKEWED crossing reaches further along one kerb than its centre offset implies.
    """
    from src.geometry.model import clip_paint_clear_of

    buffer_strip = Polygon([(0, 0), (100, 0), (100, 6), (0, 6)])
    crossing = Polygon([(40, -2), (50, -2), (50, 8), (40, 8)])
    pieces = clip_paint_clear_of(buffer_strip, crossing)

    assert len(pieces) == 2, "the strip should be split either side of the crossing"
    assert all(p.geom_type == "Polygon" for p in pieces)
    assert not any(p.intersection(crossing).area > 1e-9 for p in pieces)
    assert sum(p.area for p in pieces) == pytest.approx(buffer_strip.area - 60.0)


def test_clipping_against_nothing_leaves_the_paint_alone():
    from src.geometry.model import clip_paint_clear_of

    strip = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    assert clip_paint_clear_of(strip, None) == [strip]


def test_paint_entirely_inside_the_keep_clear_area_disappears():
    from src.geometry.model import clip_paint_clear_of

    strip = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert clip_paint_clear_of(strip, big) == []


# --------------------------------------------------------------------------
# The registry itself. A check that is never called is not a check.
# --------------------------------------------------------------------------

def test_defining_a_check_registers_it():
    """The reason SceneCheck exists.

    check_scene used to be a `+` chain naming thirteen functions, so a check could be written,
    tested in isolation, and never actually run against a scene - dead code that reads as
    coverage. Subclassing is now what puts it in CHECKS, and check_scene is a loop over that.
    """
    from src.checks import CHECKS, SceneCheck, check_scene

    before = len(CHECKS)

    class NoStopSignsAtAll(SceneCheck):
        def run(self, scene):
            return [Violation("no_stop_signs_at_all", "a stop sign exists")
                    for prop in scene.props if prop["type"] == "stop_sign"]

    try:
        assert len(CHECKS) == before + 1, "defining a check did not register it"
        found = check_scene(SceneContext(state=a_state(), pavement=ROADWAY,
                                         props=[prop("stop_sign", (9.0, 9.0))]))
        assert "no_stop_signs_at_all" in [v.check for v in found], \
            "check_scene did not run a check that had just been defined"
    finally:
        # Registration is global by design, so a check defined inside a test has to be taken
        # back out or every later test in this session runs it too.
        CHECKS[:] = [c for c in CHECKS if type(c).__name__ != "NoStopSignsAtAll"]


def test_a_check_reading_a_field_the_caller_left_out_gets_a_default():
    """SceneContext defaults everything, so a test describes only the part of the scene it means.

    This is what replaced thirteen positional signatures. Under those, omitting an argument was
    a TypeError at best and a differently-built version of the same geometry at worst - one
    check was handed crossing bands built with the mutual-exclusion reaches and another bands
    built without them, 15 sq ft apart at W Broad & Louellen.
    """
    from src.checks import CHECKS, check_scene

    empty = SceneContext()
    # Every check runs, and the only thing any of them has to say is that there is no roadway -
    # which is true, and is a claim PavementRingCloses is right to make.
    assert [v.check for v in check_scene(empty)] == ["pavement_ring"]
    for check in CHECKS:
        check.run(empty)      # must not raise: the fields it reads are all defaulted
