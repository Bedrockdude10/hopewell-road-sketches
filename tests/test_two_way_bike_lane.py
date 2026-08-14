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
                                      travel_lane_divider_shift_ft)

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


def test_the_two_travel_lanes_come_out_equal():
    """The whole point of the shift. Total travel way is what is left after the section, and
    the divider sits in the middle of THAT, not in the middle of the road."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
    shift_ft = travel_lane_divider_shift_ft(section)
    travel_way_ft = NORTH_HALF_FT + SOUTH_HALF_FT - section.section_ft
    # Distance from the divider to each kerb-side edge of the travel way.
    to_far_kerb = NORTH_HALF_FT - shift_ft
    to_section = shift_ft + (SOUTH_HALF_FT - section.section_ft)
    assert to_far_kerb == pytest.approx(travel_way_ft / 2, abs=0.01)
    assert to_section == pytest.approx(travel_way_ft / 2, abs=0.01)


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


def test_a_lane_under_the_two_way_floor_is_refused():
    """A two-way lane carries opposing traffic, so it has its own floor - a 5 ft one-way
    width is not a two-way lane however much the arithmetic fits."""
    with pytest.raises(ValueError, match="two-way"):
        TwoWayBikeLane(width_ft=6.0, buffer_ft=3.0,
                       near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
