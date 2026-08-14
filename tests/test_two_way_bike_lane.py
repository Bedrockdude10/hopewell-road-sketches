"""A two-way bike lane on ONE side of a street, and the asymmetry that forces.

Every other treatment here is symmetric about the leg centerline, because NJDOT's alignment
is the datum every offset, stop bar and crossing frame is measured from. A two-way lane
breaks that: it takes 12-15 ft out of one kerbside and nothing out of the other, so the two
travel lanes no longer straddle the alignment - they sit off it, and the double yellow
between them moves with them.

The main README said this was "a real design, but not one this pipeline can draw". What
makes it drawable without moving the datum is that the datum does not have to be the middle
of the travel lanes: the alignment stays exactly where it is, every station and crossing
stays where it is, and the cross-section is simply described asymmetrically about it.
"""
import pytest

from src.geometry.treatments import (TARGET_LANE_WIDTH_FT, BikeLane, TwoWayBikeLane,
                                      far_kerb_surplus_ft, travel_lane_divider_shift_ft)

# Broad & Greenwood's east leg, measured: 21.59 ft to the north kerb, 21.67 to the south.
NORTH_HALF_FT, SOUTH_HALF_FT = 21.59, 21.67


def test_a_one_way_lane_still_starts_at_the_travel_lane_width():
    """The default must not move. Every existing scenario depends on the section beginning
    at TARGET_LANE_WIDTH_FT from the alignment."""
    lane = BikeLane(width_ft=5.0, buffer_ft=2.0)
    assert lane.offsets_from_centerline_ft()["travel_lane_edge_ft"] == TARGET_LANE_WIDTH_FT


def test_a_two_way_section_is_measured_from_its_own_inner_edge():
    """The two-way lane's inner edge is NOT 11 ft from the alignment - it is wherever the
    shifted travel lanes end. Hard-coding TARGET_LANE_WIDTH_FT there would draw the lane
    overlapping the travel lane it is supposed to sit beside."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
    bounds = section.offsets_from_centerline_ft()
    # The section is the lane, its buffer AND the outer stripe that bounds it against the
    # hatching - 12 + 3 + one 0.82 ft line - because every width in BikeLane is between paint
    # faces and the stripes come out of the section rather than out of the travel lane.
    assert section.section_ft == pytest.approx(15.82, abs=0.01)
    assert bounds["travel_lane_edge_ft"] == pytest.approx(SOUTH_HALF_FT - section.section_ft,
                                                          abs=0.01)
    assert bounds["bike_outer_ft"] <= SOUTH_HALF_FT + 0.01, "the lane must not cross the kerb"


def test_the_travel_lanes_hold_the_target_width_and_the_far_kerb_keeps_the_surplus():
    """An equal split is the obvious rule and it is wrong on a wide street: Broad St's west leg
    gave two 18.35 ft lanes that way, and an 18 ft lane invites the speed this project exists to
    reduce. Spare width beside a travel lane is parking or hatching, never lane."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=26.24, far_half_ft=26.29)
    shift_ft = travel_lane_divider_shift_ft(section)
    inner_edge_ft = 26.24 - section.section_ft
    # The near travel lane runs from the section's inner edge to the divider.
    assert shift_ft + inner_edge_ft == pytest.approx(TARGET_LANE_WIDTH_FT, abs=0.01)
    # And the surplus lands against the far kerb, where parking can use it.
    assert far_kerb_surplus_ft(section) == pytest.approx(
        26.29 + inner_edge_ft - 2 * TARGET_LANE_WIDTH_FT, abs=0.01)
    assert far_kerb_surplus_ft(section) > 8.0, "this leg should free a stall's worth and more"


def test_a_leg_too_narrow_for_two_target_lanes_splits_what_it_has():
    """E Broad's east leg cannot hold two 11 ft lanes beside the section, so the shortfall is the
    street's and there is nothing to allocate - it splits equally and reports the width."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=18.04, far_half_ft=17.86)
    shift_ft = travel_lane_divider_shift_ft(section)
    travel_way_ft = 18.04 + 17.86 - section.section_ft
    assert travel_way_ft < 2 * TARGET_LANE_WIDTH_FT
    assert 17.86 - shift_ft == pytest.approx(travel_way_ft / 2, abs=0.01)
    assert far_kerb_surplus_ft(section) < 0


def test_the_divider_shifts_toward_the_far_kerb():
    """Sanity of sign: taking width out of the south kerbside pushes the traffic north."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
    assert travel_lane_divider_shift_ft(section) > 0


def test_a_section_that_leaves_no_room_for_two_travel_lanes_is_refused():
    """W Broad at Louellen: 32.10 ft of roadway. A 12 ft lane and a 3 ft buffer would leave
    17.1 ft for two lanes - 8.55 ft each, under any standard. Refused rather than drawn."""
    with pytest.raises(ValueError, match="travel lane"):
        TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0, near_half_ft=17.26, far_half_ft=14.84)


def test_the_divider_shift_reaches_both_views(site_models):
    """The contraflow stripe and the shifted double yellow are the SAME decision reaching two
    renderers, which is the seam every marking in this project has shipped a bug at.

    Asserted through the design rather than by calling the paint helpers: what matters is that
    a scenario applying the treatment produces a shift both views can read, because a shift the
    plan view honours and the export does not is a render whose lanes are different widths.
    """
    from src.geometry.targets import LegSide
    from src.geometry.treatments import AddTwoWayBikeLane, DesignState

    model = site_models["broad_st_greenwood"]
    state = DesignState.from_model(model)
    lane = state.legs["broad_st_east"]
    south = "left" if lane.centerline.coords[-1][1] < lane.centerline.coords[0][1] else "right"
    state = state.apply(AddTwoWayBikeLane(LegSide("broad_st_east", south), width_ft=12.0,
                                          buffer_ft=3.0))
    shift = state.travel_lane_divider_shift("broad_st_east")
    assert shift is not None, "a two-way lane must record a divider shift"
    shift_ft, shift_side = shift
    assert shift_ft > 0
    assert shift_side != south, "the divider shifts AWAY from the side carrying the lane"
    # A leg with no two-way lane keeps the alignment as its divider - nothing else moves.
    assert state.travel_lane_divider_shift("greenwood_ave_north") is None


def test_a_lane_under_the_two_way_floor_is_refused():
    """A two-way lane carries opposing traffic, so it has its own floor - a 5 ft one-way
    width is not a two-way lane however much the arithmetic fits."""
    with pytest.raises(ValueError, match="two-way"):
        TwoWayBikeLane(width_ft=6.0, buffer_ft=3.0,
                       near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)


def test_the_south_side_is_resolved_per_leg(site_models):
    """A corridor decision ("the south kerb") is not a leg decision ("left"). The same real kerb
    is left on one approach and right on the other, and translating it by hand is how a corridor
    treatment lands on the north kerb of one leg and the south kerb of the next."""
    from src.geometry.model import side_facing

    state_legs = site_models["broad_st_greenwood"].legs
    east, west = state_legs["broad_st_east"], state_legs["broad_st_west"]
    # Opposite approaches of one street: the same ground is the other hand on each.
    assert side_facing(east, "south") != side_facing(west, "south")
    assert side_facing(east, "south") != side_facing(east, "north")


def test_a_north_south_leg_has_no_compass_side(site_models):
    """Greenwood Ave runs north-south. It has an east and a west kerb, and answering "which is
    the south side" with whichever way its lean falls would be a guess presented as a fact."""
    from src.geometry.model import side_facing

    greenwood = site_models["broad_st_greenwood"].legs["greenwood_ave_north"]
    with pytest.raises(ValueError, match="north-south"):
        side_facing(greenwood, "south")
