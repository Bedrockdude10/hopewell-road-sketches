"""Scene invariants: each one must fire on the failure it was written for.

A check that never fires is worse than no check, because it reads as coverage. So every
invariant here is tested twice - once on geometry that violates it, once on geometry that
doesn't - against a synthetic junction rather than a site, so these run in milliseconds and
can't be broken by re-tracing a kerb in OSM.
"""
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from src.checks import (
    BikewayReachesTheEndOfItsKerb,
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
    """A design with kerbside paint on it, applied rather than written in.

    It used to write state.lane_narrowing and state.parking_zones directly. Those dicts are
    gone: the check reads the treatments now, so the fixture has to apply them, which also
    means it can no longer describe a design no scenario could produce.
    """
    from src.geometry.targets import LegSide, LegTarget, Side
    from src.geometry.treatments import DesignState, LaneNarrowing, MarkedParking

    state = DesignState(legs={"east": a_leg(width_ft=width_ft)}, corner_fillets={})
    if hatch_ft is not None:
        state = state.apply(LaneNarrowing(LegTarget("east"), stripe_width_ft=hatch_ft,
                                           sides=(Side.LEFT,)))
    if parking_ft is not None:
        state = state.apply(MarkedParking(LegSide("east", Side.RIGHT), depth_ft=parking_ft))
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


def test_a_zone_that_arrives_in_two_pieces_survives_a_clip_that_misses_it():
    """The parts of a MultiPolygon are Polygons, so a filter keyed off the CONTAINER's type
    discards every one of them.

    Not a corner case: a kerbside zone is built between a constant inner edge and the traced
    kerb, so wherever the kerb comes inside that edge the strip pinches to nothing and the zone
    arrives here in two pieces. That is the DESIGN - curbside_strip_polygon says so - and it is
    what a zone sized per station rather than off one minimum will do far more often.

    The filter exists to drop debris of the WRONG DIMENSION, which a difference can leave
    behind; that is a question about each part, never about the container it travelled in.
    Asked the wrong way it cost 1617.8 sq ft of hatching on w_broad_st_southwest's south kerb
    against a keep-clear area that did not touch the zone at all, and 65.1 sq ft on
    broad_st_greenwood in two shipped scenarios.
    """
    from src.geometry.model import clip_paint_clear_of

    near = Polygon([(0, 0), (100, 0), (100, 6), (0, 6)])
    far = Polygon([(140, 0), (240, 0), (240, 6), (140, 6)])
    zone = MultiPolygon([near, far])
    elsewhere = Polygon([(500, 500), (510, 500), (510, 510), (500, 510)])

    survives = clip_paint_clear_of(zone, elsewhere)

    assert sum(p.area for p in survives) == pytest.approx(zone.area), \
        "a clip that touches nothing must remove nothing"
    assert all(p.geom_type == "Polygon" for p in survives)


def test_a_line_that_arrives_in_two_pieces_survives_a_clip_that_misses_it():
    """The same confusion on the other dimension, and it takes an already-broken line to see it.

    A plain LineString is safe here by luck: cutting one yields a MultiLineString whose parts
    are LineStrings, which is what the container comparison happens to be asking for. It is a
    line that arrives ALREADY in two pieces - an edge line cut by a crossing, a dashed run, a
    lane line broken at a driveway - that the filter throws away entire.
    """
    from src.geometry.model import clip_paint_clear_of

    line = MultiLineString([[(0, 0), (100, 0)], [(140, 0), (240, 0)]])
    elsewhere = Polygon([(500, 500), (510, 500), (510, 510), (500, 510)])

    survives = clip_paint_clear_of(line, elsewhere)

    assert sum(p.length for p in survives) == pytest.approx(line.length), \
        "a clip that touches nothing must remove nothing"
    assert all(p.geom_type == "LineString" for p in survives)


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


# --------------------------------------------------------------------------
# A bikeway runs the whole kerb it was placed on - unless the design SAID
# why it does not.
# --------------------------------------------------------------------------

def a_bikeway_kerb(reaches_ft, kerb_to_ft=120.0):
    """A leg with a traced left kerb, carrying a bike lane surface up to `reaches_ft`."""
    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.paint import PaintPiece

    leg = a_leg(name="east")
    leg.left_curb = LineString([(0, 15), (kerb_to_ft, 15)])
    leg.right_curb = LineString([(0, -15), (kerb_to_ft, -15)])
    surface = Polygon([(0, 5), (reaches_ft, 5), (reaches_ft, 13), (0, 13)])
    piece = PaintPiece(kind=BIKE_LANE_SURFACE, geometry=surface, leg="east", side="left")
    return a_state(legs={"east": leg}), [piece]


def test_a_bikeway_that_stops_short_of_its_kerb_with_no_reason_is_reported():
    """The headline case: 60 ft of kerb with no facility on it and nothing saying why."""
    state, paint = a_bikeway_kerb(reaches_ft=60.0)
    violations = run(BikewayReachesTheEndOfItsKerb(), state=state, paint=paint)
    assert len(violations) == 1, f"expected one report, got {violations}"
    assert "60.0" in violations[0].detail and "120.0" in violations[0].detail


def test_a_refused_tail_is_not_a_shortfall():
    """A stretch the design MEASURED and declined is a drawing saying what it is doing.

    This is the whole reason a refusal is a record on DesignState rather than a line on stdout:
    the check's own complaint is "no note saying why", so it has to be able to read the note. A
    facility that stops at 60 ft on a kerb traced to 120 with the 60-120 ft span refused - by name,
    carrying the measurement that stopped it - is the honest half of TwoWayBikeway._reach_on.
    """
    from src.geometry.treatments.state import FacilityRefusal

    state, paint = a_bikeway_kerb(reaches_ft=60.0)
    state.refuse("east", "left", FacilityRefusal(60.0, 120.0, "the street is 28.4 ft here", 28.4))
    assert not run(BikewayReachesTheEndOfItsKerb(), state=state, paint=paint)


def test_a_refusal_only_excuses_the_ground_it_covers():
    """Refuse 20 ft and stop 60 ft short, and the other 40 ft is still a defect.

    A refusal that excused a whole kerb because it named part of one would be the relax-the-check
    move SKILLS 4 is about - and the failure it would hide is the original: paint trimmed at the
    first station it stopped fitting, with the posts and the centre stripe carrying on regardless.
    """
    from src.geometry.treatments.state import FacilityRefusal

    state, paint = a_bikeway_kerb(reaches_ft=60.0)
    state.refuse("east", "left", FacilityRefusal(100.0, 120.0, "the street is 28.4 ft here", 28.4))
    violations = run(BikewayReachesTheEndOfItsKerb(), state=state, paint=paint)
    assert len(violations) == 1, f"expected the un-refused 40 ft to report, got {violations}"
    assert "100.0" in violations[0].detail, (
        f"the message should measure against the refusal's start, not the kerb's end: "
        f"{violations[0].detail}")


def test_a_refusal_nowhere_near_the_facilitys_end_excuses_nothing():
    """A refused span in the MIDDLE of a kerb says nothing about where the paint stopped.

    THE SPANS ARE THE POINT, WHICH IS WHY THE CHEAP VERSION IS WRONG. Subtracting the LENGTH of
    every refusal from the kerb would pass this: 70 ft refused against a 60 ft shortfall comes out
    negative and reports nothing, while 10 ft at the end of the kerb was never declined at all.
    Only a refusal reaching the end of the kerb, or a contiguous chain of them reaching it, can
    explain a facility that ends before the kerb does.
    """
    from src.geometry.treatments.state import FacilityRefusal

    # Refused: 40-110 ft (70 ft of it). Painted to 60, kerb traced to 120 - so 60 ft short, with
    # the last 10 ft of kerb outside any refusal and outside BIKEWAY_SHORTFALL_TOLERANCE_FT of one.
    state, paint = a_bikeway_kerb(reaches_ft=60.0)
    state.refuse("east", "left", FacilityRefusal(40.0, 110.0, "the street is 28.4 ft here", 28.4))
    violations = run(BikewayReachesTheEndOfItsKerb(), state=state, paint=paint)
    assert len(violations) == 1, (
        f"a refusal 5 ft shy of the end of the kerb excused a facility that stopped 60 ft shy of "
        f"it, so the refused SPAN is being read as a refused LENGTH: {violations}")
    assert "120.0" in violations[0].detail, (
        f"nothing adjacent to the paint's end was refused, so the whole kerb is what it fell "
        f"short of: {violations[0].detail}")
