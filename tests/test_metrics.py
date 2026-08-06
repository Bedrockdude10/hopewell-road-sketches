"""What a design ACHIEVES, as numbers - and that the numbers come off the drawn geometry.

Against a synthetic junction rather than a site, for the reason tests/test_checks.py gives:
these run in milliseconds and cannot be broken by re-tracing a kerb in OSM. It also means
they run without data/, which the whole-site tests need and skip without.

The thing being pinned throughout is that a metric is MEASURED off what the scenario drew -
the two-pass crossing reaches, the paint pieces actually put down - and not recomputed from
the nominal inputs. A number recomputed from the config would agree with the drawing only by
coincidence, and would keep agreeing after a treatment moved the kerb.
"""
import pytest
from shapely.geometry import LineString

from src.geometry.markings import PARKING_EDGE_LINE, STALL_DIVIDER
from src.geometry.model import Leg
from src.geometry.paint import PaintPiece
from src.geometry.targets import Corner, LegSide, LegTarget
from src.geometry.treatments import DesignState, MarkedParking, RefugeIsland
from src.metrics import (MUTCD_WALKING_SPEED_FT_S, SLOW_WALKING_SPEED_FT_S, Comparison,
                         SceneMetrics, leg_label, stalls_in_run, turn_speed_mph)
from src.render.crosswalks import CrosswalkOffset

# A 30 ft wide east-west street, curbs at y = +/-15, crossing at station 20.
CROSSING_AT_FT = 20.0


def a_leg(name="east", width_ft=30.0, length_ft=120.0):
    return Leg(name=name, centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)


def a_state(legs=None, corner_fillets=None):
    legs = legs or {"east": a_leg()}
    return DesignState(legs=legs, corner_fillets=corner_fillets or {})


def metrics(state, reaches=None, offsets=None, paint=(), marked=None):
    """SceneMetrics of a design, defaulting the crossing to a square 15/15 reach.

    Deliberately the same arguments SceneGeometry hands it (src/render/scene.py), so a test
    describes a scene rather than a call signature.
    """
    reaches = {"east": (15.0, 15.0)} if reaches is None else reaches
    offsets = ({name: CrosswalkOffset(CROSSING_AT_FT, "osm_survey") for name in reaches}
               if offsets is None else offsets)
    return SceneMetrics.of(state, reaches=reaches, offsets=offsets, skews={},
                           paint=list(paint), marked=marked)


def parking_run(length_ft, leg="east", side="left", start_ft=30.0):
    """One painted parking edge line, `length_ft` long, as the paint builder emits it."""
    return PaintPiece(kind=PARKING_EDGE_LINE, leg=leg, side=side,
                       geometry=LineString([(start_ft, 11), (start_ft + length_ft, 11)]))


# --------------------------------------------------------------------------
# Crossing distance: the headline number, measured to the real kerbs.
# --------------------------------------------------------------------------

def test_a_crossing_is_as_long_as_its_two_reaches():
    m = metrics(a_state())
    assert m.crossing("east").distance_ft == pytest.approx(30.0)


def test_an_asymmetric_crossing_measures_the_real_kerbs_not_the_nominal_width():
    """The reaches are asymmetric because the traced kerbs are (see crosswalk_reach_to_curbs_ft).

    Half the nominal width either side would say 30 ft here. The crossing is 32 ft long, and
    it is the 32 that a person walks.
    """
    m = metrics(a_state(), reaches={"east": (12.0, 20.0)})
    assert m.crossing("east").distance_ft == pytest.approx(32.0)


def test_a_crossing_carries_the_provenance_of_its_position():
    surveyed = metrics(a_state()).crossing("east")
    estimated = metrics(a_state(),
                         offsets={"east": CrosswalkOffset(CROSSING_AT_FT, "geometric_estimate")}
                         ).crossing("east")
    assert surveyed.is_surveyed
    assert not estimated.is_surveyed


def test_only_legs_with_a_crossing_are_measured():
    """Every leg gets a resolved offset; only a marked one has a crossing to measure.

    Mirrors the gate the plan view and blender_scene.py draw by - a leg with no marked
    crossing is drawn as a thin outline, and reporting a crossing distance for it would be
    reporting a number for something that is not there.
    """
    state = a_state(legs={"east": a_leg("east"), "west": a_leg("west")})
    reaches = {"east": (15.0, 15.0), "west": (15.0, 15.0)}
    m = metrics(state, reaches=reaches, marked={"east"})
    assert [c.leg for c in m.crossings] == ["east"]


# --------------------------------------------------------------------------
# Staging: a refuge island is the difference between 30 ft and two 12 ft walks.
# --------------------------------------------------------------------------

def test_a_refuge_island_splits_a_crossing_into_two_stages():
    state = a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=CROSSING_AT_FT,
                                          width_ft=6, along_road_ft=20))
    crossing = metrics(state).crossing("east")
    assert crossing.stages_ft == pytest.approx((12.0, 12.0))
    assert crossing.distance_ft == pytest.approx(24.0)
    assert crossing.longest_stage_ft == pytest.approx(12.0)
    assert crossing.is_staged


def test_an_island_further_down_the_leg_does_not_split_the_crossing():
    """An island is a band across the road at ITS OWN station. One 60 ft away shelters nothing.

    Without this the metric would credit any refuge anywhere on the leg to every crossing on
    it, which is the easiest way for a summary panel to claim an improvement that is not there.
    """
    state = a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=80.0,
                                          width_ft=6, along_road_ft=20))
    crossing = metrics(state).crossing("east")
    assert crossing.stages_ft == pytest.approx((30.0,))
    assert not crossing.is_staged


def test_exposure_is_the_longest_stage_not_the_whole_walk():
    """Exposure is time in front of moving traffic. A refuge is somewhere to stand, so a
    staged crossing exposes a person for one stage at a time - and the honest number for a
    two-stage crossing is the worse of the two stages, not their sum."""
    state = a_state().apply(RefugeIsland(LegTarget("east"), offset_ft=CROSSING_AT_FT,
                                          width_ft=6, along_road_ft=20))
    crossing = metrics(state).crossing("east")
    assert crossing.exposure_s() == pytest.approx(12.0 / MUTCD_WALKING_SPEED_FT_S)
    assert crossing.crossing_time_s() == pytest.approx(24.0 / MUTCD_WALKING_SPEED_FT_S)


def test_exposure_is_reported_at_the_walking_speed_it_was_asked_for():
    crossing = metrics(a_state()).crossing("east")
    assert crossing.exposure_s() == pytest.approx(30.0 / 3.5)
    assert crossing.exposure_s(SLOW_WALKING_SPEED_FT_S) == pytest.approx(30.0 / 3.0)
    assert crossing.exposure_s(SLOW_WALKING_SPEED_FT_S) > crossing.exposure_s()


# --------------------------------------------------------------------------
# Parking: counted off the paint, because that is what is actually marked.
# --------------------------------------------------------------------------

def test_stalls_are_counted_from_the_paint_that_was_put_down():
    state = a_state().apply(MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22))
    m = metrics(state, paint=[parking_run(100.0)])
    assert m.total_stalls == 4          # 100 // 22
    assert [run.stalls for run in m.parking] == [4]


def test_each_run_of_stalls_is_counted_separately():
    """A hydrant or a driveway splits a kerb into two runs, and the paint builder emits one
    edge line per run. Counting the sum of the lengths would claim a stall that straddles the
    gap between them."""
    state = a_state().apply(MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22))
    m = metrics(state, paint=[parking_run(40.0), parking_run(40.0, start_ft=90.0)])
    assert m.total_stalls == 2          # 1 + 1, not (40 + 40) // 22 == 3


def test_a_run_too_short_for_one_stall_holds_none():
    state = a_state().apply(MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22))
    m = metrics(state, paint=[parking_run(15.0)])
    assert m.total_stalls == 0


def test_paint_that_is_not_a_parking_line_is_not_counted_as_parking():
    state = a_state().apply(MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22))
    divider = PaintPiece(kind=STALL_DIVIDER, leg="east", side="left",
                         geometry=LineString([(30, 11), (30, 15)]))
    m = metrics(state, paint=[parking_run(100.0), divider])
    assert m.total_stalls == 4


def test_stalls_in_run_is_the_rule_the_plan_view_labels_with():
    """One rule, so the label beside a run and the total in the panel cannot disagree.
    src/render/plan_view.py:_label_paint calls this."""
    assert stalls_in_run(100.0, 22.0) == 4
    assert stalls_in_run(21.9, 22.0) == 0
    assert stalls_in_run(0.0, 22.0) == 0


# --------------------------------------------------------------------------
# Corners.
# --------------------------------------------------------------------------

def test_a_tighter_corner_implies_a_slower_turn():
    assert turn_speed_mph(15) < turn_speed_mph(20) < turn_speed_mph(35)


def test_the_turn_speed_is_the_aashto_side_friction_relation():
    """v = sqrt(15 * R * (e + f)), flat (e = 0) at the low-speed side friction factor."""
    assert turn_speed_mph(20) == pytest.approx((15 * 20 * 0.30) ** 0.5)


def test_a_corner_that_did_not_solve_is_not_reported():
    """build_corner_fillets records an "error" where it could not cut an arc. Reading a radius
    off that entry is what the plan view already guards against; a metric must skip it too,
    rather than report a turn speed for a corner that has no geometry."""
    fillets = {("east", "north"): {"error": "curbs do not meet"},
               ("north", "west"): {"radius_ft": 20.0}}
    m = metrics(a_state(corner_fillets=fillets))
    assert [c.corner for c in m.corners] == [Corner("north", "west")]


def test_a_tightened_corner_reads_as_a_slower_turn_in_the_metrics():
    """R=35 to R=15 is the change an argument about a curb extension turns on, and the radius
    is the only form of it currently on the drawing."""
    before = metrics(a_state(corner_fillets={("east", "north"): {"radius_ft": 35.0}}))
    after = metrics(a_state(corner_fillets={("east", "north"): {"radius_ft": 15.0}}))
    assert after.corners[0].turn_speed_mph < before.corners[0].turn_speed_mph


# --------------------------------------------------------------------------
# The comparison: what a before/after figure is actually about.
# --------------------------------------------------------------------------

def a_pair():
    """A 30 ft crossing with 4 stalls, against a 22 ft staged crossing with 3."""
    before_state = a_state().apply(
        MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22))
    before = metrics(before_state, paint=[parking_run(100.0)])
    after_state = a_state().apply(
        MarkedParking(LegSide("east", "left"), depth_ft=8, stall_length_ft=22),
        RefugeIsland(LegTarget("east"), offset_ft=CROSSING_AT_FT, width_ft=6, along_road_ft=20))
    after = metrics(after_state, reaches={"east": (11.0, 11.0)}, paint=[parking_run(75.0)])
    return before, after


def test_the_comparison_reports_the_crossing_it_shortened():
    before, after = a_pair()
    change = Comparison.of(before, after).crossing("east")
    assert change.before_ft == pytest.approx(30.0)
    assert change.after_ft == pytest.approx(16.0)     # 22 ft of roadway, staged around a 6 ft island
    assert change.saved_ft == pytest.approx(14.0)
    assert change.saved_s() > 0


def test_the_comparison_reports_the_parking_it_removed():
    before, after = a_pair()
    comparison = Comparison.of(before, after)
    assert comparison.stalls_before == 4
    assert comparison.stalls_after == 3
    assert comparison.stalls_delta == -1


def test_a_design_that_changes_nothing_reports_no_change():
    """The panel has to be able to say "nothing moved". A summary that only ever shows
    improvements is a poster, not a reconstruction."""
    before, _ = a_pair()
    comparison = Comparison.of(before, before)
    assert comparison.crossing("east").saved_ft == pytest.approx(0.0)
    assert comparison.stalls_delta == 0


def test_a_crossing_that_only_exists_afterwards_is_reported_as_new():
    """A proposal that marks a leg with no crossing today has nothing to compare against, and
    saying "0 ft saved" would be false in both directions."""
    before = metrics(a_state(), marked=set())
    after = metrics(a_state(), marked={"east"})
    change = Comparison.of(before, after).crossing("east")
    assert change.before_ft is None
    assert change.saved_ft is None
    assert change.is_new


# --------------------------------------------------------------------------
# The panel itself: what a reader sees.
# --------------------------------------------------------------------------

def test_the_panel_leads_with_the_crossing_and_the_parking():
    before, after = a_pair()
    text = Comparison.of(before, after).panel_text()
    assert "30.0" in text and "16.0" in text
    assert "-14.0 ft" in text
    assert "4 -> 3" in text or "4 → 3" in text


def test_the_panel_says_a_crossing_is_staged_rather_than_hiding_it():
    """16 ft after against 30 before is only true because a person now stops halfway. A panel
    that reports the number without the staging is claiming a 22 ft street is 16 ft wide."""
    before, after = a_pair()
    text = Comparison.of(before, after).panel_text()
    assert "staged" in text.lower()


def test_the_panel_names_the_walking_speed_it_used():
    """A time in seconds is meaningless without it, and 3.5 ft/s is an assumption about who is
    crossing - a slower walker is the person the treatment is for."""
    before, after = a_pair()
    assert "3.5 ft/s" in Comparison.of(before, after).panel_text()


def test_the_panel_marks_an_estimated_crossing_position_as_estimated():
    state = a_state()
    before = metrics(state, offsets={"east": CrosswalkOffset(CROSSING_AT_FT, "geometric_estimate")})
    after = metrics(state, reaches={"east": (11.0, 11.0)},
                     offsets={"east": CrosswalkOffset(CROSSING_AT_FT, "geometric_estimate")})
    assert "est." in Comparison.of(before, after).panel_text()


def test_the_panel_actually_renders_onto_a_figure():
    """The one part of this that a site test would otherwise be the first to exercise.

    src/render/plan_view.py:draw_change_panel is only reached from a full before/after build,
    which needs data/ and skips without it. Drawing the block onto a bare figure and forcing a
    render is enough to catch a bad kwarg or a glyph matplotlib cannot draw, and costs
    milliseconds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import io as _io

    import matplotlib.pyplot as plt

    from src.render.plan_view import draw_change_panel

    before, after = a_pair()
    fig = plt.figure(figsize=(6, 4))
    try:
        comparison = draw_change_panel(fig, before, after)
        fig.savefig(_io.BytesIO(), format="png", bbox_inches="tight")
    finally:
        plt.close(fig)
    assert comparison.stalls_delta == -1


def test_a_leg_reads_as_a_street_name_not_a_dict_key():
    assert leg_label("broad_st_east") == "Broad St East"
    assert leg_label("greenwood_ave_north") == "Greenwood Ave North"


def test_exposure_is_measured_across_the_travel_lanes_not_curb_to_curb():
    """A person in a bike lane or a parking lane is not standing in front of a car.

    Exposure was the crossing distance divided by a walking speed, which made the panel's two
    rows the same measurement under two headings - and meant no paint-only proposal could ever
    move either. A bike lane takes 18 ft of Broad St out of the part a car drives on; the number
    did not budge. Curb-to-curb is still the whole walk, which is a different question.
    """
    from src.geometry.targets import LegSide
    from src.geometry.treatments import MarkedParking

    state = a_state()
    bare = metrics(state, marked={"east"}).crossing("east")
    assert bare.distance_ft == pytest.approx(30.0)
    assert bare.exposure_s() == pytest.approx(30.0 / 3.5)

    # 8 ft of stalls against each kerb of a 30 ft street, plus a 1 ft buffer: 9 ft either side
    # is no longer ground a car drives on, so 30 ft of walking is 12 ft of exposure.
    parked = state
    for side in ("left", "right"):
        parked = parked.apply(MarkedParking(LegSide("east", side), depth_ft=8.0,
                                             curb_offset_ft=1.0))
    after = metrics(parked, marked={"east"}).crossing("east")
    assert after.distance_ft == pytest.approx(30.0), "the walk itself did not get shorter"
    assert after.motor_distance_ft == pytest.approx(12.0, abs=0.5)
    assert after.exposure_s() == pytest.approx(12.0 / 3.5, abs=0.2)
    assert after.exposure_s() < bare.exposure_s()


def test_the_panel_says_motor_traffic_because_that_is_what_it_measured():
    """A bike lane is a real conflict this number does not count. The heading has to say so."""
    before = metrics(a_state(), marked={"east"})
    assert "MOTOR TRAFFIC" in Comparison.of(before, before).panel_text()
