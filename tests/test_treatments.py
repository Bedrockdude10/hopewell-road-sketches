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
from src.geometry.targets import Corner, LegSide, LegTarget, Side
from src.geometry.treatments import (AASHTO_MIN_BIKE_LANE_FT, AddBikeLane, AddBikeLaneBollards,
                                     DesignState, LaneNarrowing,
                                     NACTO_MIN_REFUGE_ISLAND_WIDTH_FT, RaiseCrossing,
                                     RefugeIsland, Treatment)


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
        state.apply(RaiseCrossing(LegTarget("east"), crossing_width_ft=10.0))
    assert "raised crossing" in str(caught.value)
    assert "same point" in str(caught.value)


def test_a_refuge_island_on_a_zero_length_span_says_so():
    with pytest.raises(ValueError) as caught:
        a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=60.0, width_ft=NACTO_MIN_REFUGE_ISLAND_WIDTH_FT, along_road_ft=0.0))
    assert "refuge island" in str(caught.value)
    assert "no extent along the road" in str(caught.value)


def test_a_refuge_island_below_the_nacto_minimum_is_refused():
    with pytest.raises(ValueError, match="NACTO minimum"):
        a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=60.0, width_ft=NACTO_MIN_REFUGE_ISLAND_WIDTH_FT - 1))


# --------------------------------------------------------------------------
# ...and the ordinary case still builds the shape it says it does.
# --------------------------------------------------------------------------

def test_a_refuge_island_spans_its_stated_width_across_the_road():
    state = a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=60.0, width_ft=8.0, along_road_ft=20.0))
    island = next(iter(state.refuge_islands.values()))
    minx, miny, maxx, maxy = island["polygon"].bounds
    assert (maxy - miny) == pytest.approx(8.0)    # across the road: the stated width
    assert (maxx - minx) == pytest.approx(20.0)   # along the road: along_road_ft
    assert island["width_ft"] == 8.0


def test_a_raised_crossing_spans_curb_to_curb():
    state = a_state(width_ft=34.0).apply(RaiseCrossing(LegTarget("east"), crossing_width_ft=12.0))
    minx, miny, maxx, maxy = state.raised_crossings["east"].bounds
    assert (maxy - miny) == pytest.approx(34.0)    # the full roadway
    assert (maxx - minx) == pytest.approx(12.0)


def test_a_leg_with_no_width_cannot_take_a_crossing():
    state = DesignState(legs={"east": Leg(name="east", centerline=LineString([(0, 0), (100, 0)]))},
                        corner_fillets={})
    with pytest.raises(ValueError, match="no curb lines"):
        state.apply(RaiseCrossing(LegTarget("east")))


# --------------------------------------------------------------------------
# The funnel: everything that used to be each treatment function's own business
# --------------------------------------------------------------------------

def test_a_treatment_aimed_at_a_leg_that_does_not_exist_is_refused():
    """This used to write a dict key nothing ever read.

    A treatment function checked `if leg_name not in state.legs` if whoever wrote it thought
    of it - several did not - so a mistyped leg name produced a design with the treatment
    recorded, no paint, no props, no error, and a render that looked deliberate. The target
    is now checked once, for every treatment, in DesignState.apply.
    """

    state = a_state()
    with pytest.raises(KeyError, match="no leg 'wsst'"):
        state.apply(LaneNarrowing(LegTarget("wsst"), stripe_width_ft=3.0))


def test_the_error_says_which_legs_the_junction_actually_has():
    """A refusal that names the alternatives is one round trip; one that doesn't is several."""

    with pytest.raises(KeyError, match=r"\['east'\]"):
        a_state().apply(LaneNarrowing(LegTarget("wsst"), stripe_width_ft=3.0))


def test_a_treatment_that_needs_the_model_and_has_none_is_refused():
    """The phase4 bug, made impossible.

    phase4_export_geometry dropped the model argument it was passing to a scenario builder, so
    every treatment that reads the model quietly did nothing and E Broad exported with no
    treatments at all. It rendered plausibly, which is why it took a measurement to find.
    """

    class NeedsTheModel(Treatment):
        needs_model = True

        def describe(self):
            return "needs the model"

        def apply_to(self, state, model=None):
            state.notes.append(f"read {model}")

    with pytest.raises(ValueError, match="needs the IntersectionModel"):
        a_state().apply(NeedsTheModel(LegTarget("east")))


def test_applying_a_treatment_records_it_and_leaves_the_original_alone():
    """Provenance by construction. Every treatment function used to append its own note, and
    the ones that forgot were simply absent from the notes a render ships with."""

    state = a_state()
    after = state.apply(LaneNarrowing(LegTarget("east"), stripe_width_ft=3.0))
    assert state.notes == [] and state.treatments == [], "apply mutated the design it was given"
    assert len(after.treatments) == 1 and after.notes[0].startswith("LaneNarrowing(")
    assert after.lane_narrowing["east"] == 3.0


def test_treatments_chain_in_one_call_or_several():

    state = a_state()
    one_call = state.apply(LaneNarrowing(LegTarget("east"), 3.0), LaneNarrowing(LegTarget("east"), 4.0))
    chained = state.apply(LaneNarrowing(LegTarget("east"), 3.0)).apply(LaneNarrowing(LegTarget("east"), 4.0))
    assert one_call.lane_narrowing == chained.lane_narrowing == {"east": 4.0}
    assert len(one_call.treatments) == len(chained.treatments) == 2


def test_a_lane_narrowing_with_no_width_is_refused():
    """Validation this treatment never had. As a function it checked the leg existed and
    nothing else, so a zero stripe painted a buffer with no width."""

    with pytest.raises(ValueError, match="needs a width"):
        LaneNarrowing(LegTarget("east"), stripe_width_ft=0.0)


def test_a_treatment_is_refused_before_it_touches_the_design():
    """Constructed, therefore valid - the point of putting the checks in __post_init__."""

    with pytest.raises(ValueError):
        AddBikeLane(LegSide("east", "left"), width_ft=AASHTO_MIN_BIKE_LANE_FT - 1)


def test_bollards_still_refuse_a_lane_with_no_buffer_through_the_funnel():
    """A precondition on another treatment rather than on the street, so it is checked when the
    treatment meets the design - see AddBikeLaneBollards."""

    state = a_state(width_ft=40.0)
    with_lane = state.apply(AddBikeLane(LegSide("east", "left"), width_ft=6.0))
    with pytest.raises(ValueError, match="no buffer"):
        with_lane.apply(AddBikeLaneBollards(LegSide("east", "left")))


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def test_a_side_is_left_or_right_and_nothing_else():
    """`state.bike_lanes[("east", "north")]` was a perfectly good expression that matched
    nothing. Side is a StrEnum, so it still equals and hashes like the string it replaces."""

    assert Side("left") is Side.LEFT and Side.LEFT == "left"
    assert {("east", Side.LEFT): 1}[("east", "left")] == 1, "must key the existing state dicts"
    with pytest.raises(ValueError):
        Side("north")


def test_a_leg_side_coerces_the_string_form():
    """Scenarios say "left"; the treatment gets the enum, and a typo is refused at the target
    rather than becoming a key nothing reads."""

    assert LegSide("east", "right").side is Side.RIGHT
    assert LegSide("east", "right").key == ("east", "right")
    with pytest.raises(ValueError):
        LegSide("east", "middle")


def test_a_side_knows_its_own_sign():
    """`1 if side == "left" else -1` was written out in ten places, and an invariant that
    forgot the sign passed anything on the right-hand side of a leg."""

    assert Side.LEFT.sign == 1.0 and Side.RIGHT.sign == -1.0
    assert Side.LEFT.other is Side.RIGHT
    assert Side.RIGHT.curb_attr == "right_curb"


def test_a_corner_is_ordered():
    """A corner is (leg_a's left kerb, leg_b's right kerb), so the two orderings are two
    different corners of the junction - see fillet_curb_corner."""

    assert Corner("a", "b") != Corner("b", "a")
    state = a_state(corner_fillets={("a", "b"): {}})
    assert Corner("a", "b").missing_from(state) is None
    assert "the order matters" in Corner("b", "a").missing_from(state)


def test_a_treatment_asks_about_another_treatment_not_about_a_dict():
    """A precondition on another treatment is a question about a DECISION.

    LaneNarrowingBollards used to check `leg in state.lane_narrowing`, which is a question about
    state anything could have written - including a test poking the dict. It asks
    state.treatment_for now, so the answer is "did someone apply a lane narrowing here", and a
    design assembled by hand rather than by applying treatments correctly refuses.
    """
    from src.geometry.targets import LegTarget
    from src.geometry.treatments import LaneNarrowing, LaneNarrowingBollards

    state = a_state()
    # The dict says there is a buffer here; no treatment does.
    state.lane_narrowing["east"] = 3.0
    with pytest.raises(KeyError, match="no lane-narrowing buffer"):
        state.apply(LaneNarrowingBollards(LegTarget("east")))
    # Applied properly, it is accepted - and the posts take that buffer's own width.
    narrowed = a_state().apply(LaneNarrowing(LegTarget("east"), stripe_width_ft=3.0))
    assert narrowed.apply(LaneNarrowingBollards(LegTarget("east"))).bollard_lines == {"east": 10.0}


def test_the_last_treatment_on_a_target_is_the_one_that_counts():
    """A design is a sequence of decisions and the later one is the decision.

    It matters for painting: two MarkedParking treatments on one kerb are one marked lane, not
    two painted on top of each other, which markings_collide would report.
    """
    from src.geometry.targets import LegSide
    from src.geometry.treatments import MarkedParking

    state = a_state(width_ft=40.0).apply(
        MarkedParking(LegSide("east", "left"), depth_ft=8.0),
        MarkedParking(LegSide("east", "left"), depth_ft=7.0))
    assert state.treatment_for(MarkedParking, LegSide("east", "left")).depth_ft == 7.0
    assert len(state.treatments_of(MarkedParking)) == 1, "one kerb, one marked lane"
