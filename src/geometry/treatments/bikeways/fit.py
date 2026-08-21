"""WHETHER A SECTION FITS THIS KERB, and what it leaves for everything else.

Pure arithmetic on a section and a traced kerb: no treatment, no design state being mutated, so a
scenario can ask these before committing to anything. THE ANSWERS ARE ORDERED OUTWARD FROM THE
ALIGNMENT, which is the order widths are given up in - the travel lane is fixed, the buffer is
fixed because it is what a post stands in, and the bike lane takes what is left.

Go through `bike_lane_spare_ft` rather than subtracting widths by hand; a caller doing its own
accounting misses the lane LINE, which is 0.82 ft and decides whether e_broad_st_east is buildable.
"""
from src.geometry.model import narrowest_half_width_ft
from src.geometry.treatments.base import LANE_WIDTH_SLACK_FT, TARGET_LANE_WIDTH_FT
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.sections import (BIKE_LANE_BUFFER_FT, BIKE_LANE_WIDTH_FT, BikeLane,
                                                       MIN_BIKE_LANE_FT, TwoWayBikeLane)

def travel_lane_divider_shift_ft(section: TwoWayBikeLane) -> float:
    """How far the painted divider between the travel lanes sits off the alignment.

    Positive TOWARD THE FAR KERB - away from the side carrying the lane, which is the direction
    traffic is pushed by taking width out of one kerbside.

    THE TRAVEL LANES HOLD TARGET_LANE_WIDTH_FT AND THE FAR KERB KEEPS THE SURPLUS. Placing the
    divider mid-way through whatever the section leaves is the obvious rule and it is the wrong
    one on a wide street: Broad St's west leg has 52.5 ft between kerbs, so an equal split gave
    two 18.35 ft travel lanes, and an 18 ft lane invites exactly the speed this whole project
    exists to reduce. Spare width beside a travel lane is not the travel lane's - it is parking,
    or it is hatched - which is the same accounting a bike lane and an 8 ft stall already get.

    Where the leg cannot hold two target-width lanes beside the section, it falls back to an
    equal split, because then the shortfall is the street's and there is nothing to allocate.
    E Broad's east leg is that case at 10.04 ft a lane.

    BOTH RULES ARE ONE EXPRESSION - the divider sits one lane width in from the near travel edge -
    and they were written as two branches, which is how the equal-split case came to be missing
    from everything downstream. `far_half - travel_way/2` and `divided_lane_width_ft - inner_edge`
    are the same figure to the last decimal, so this is a rewrite and not a change: the branch was
    hiding that the answer is always the lane width, and callers that reconstructed it as
    `TARGET_LANE_WIDTH_FT + shift` were wrong by 0.92 ft on any leg taking the equal split. See
    divider.travel_lane_edge_ft, which is now the only place that sum is written.
    """
    return divided_lane_width_ft(section) - (section.near_half_ft - section.section_ft)


def divided_lane_width_ft(section: TwoWayBikeLane) -> float:
    """How wide EACH travel lane is built beside this section, whichever kerb it is against.

    TARGET_LANE_WIDTH_FT where the leg can hold two of them and half the travel way where it
    cannot - the equal split of travel_lane_divider_shift_ft, hoisted out of it because the
    divider is not the only thing that needs the figure. Anything asking "where does the travel
    lane end" needs the lane's WIDTH, and reaching for the target instead is right on five of the
    six corridor legs and 0.92 ft wrong on w_broad_st_northeast, which takes the split at 10.08 ft
    a lane. That 0.92 ft is what left a 0.10 ft ribbon of buffer hatching standing across every
    driveway on that leg, outlined by a 49.68 ft edge line straight over a 9.5 ft driveway.
    """
    travel_way_ft = section.near_half_ft + section.far_half_ft - section.section_ft
    return min(TARGET_LANE_WIDTH_FT, travel_way_ft / 2)


def far_kerb_surplus_ft(section: TwoWayBikeLane) -> float:
    """Width left against the FAR kerb once the section and two target-width lanes are placed.

    What a two-way lane on one side frees up on the other, and the reason the pair belongs in one
    proposal: the kerb losing its parking to the bike lane is not the kerb that gains this. Zero
    or negative where the leg has nothing spare.
    """
    inner_edge_ft = section.near_half_ft - section.section_ft
    return section.far_half_ft + inner_edge_ft - 2 * TARGET_LANE_WIDTH_FT


def bike_lane_spare_ft(state: DesignState, leg_name: str, side: str, width_ft: float,
                        buffer_ft: float = 0.0, parking_ft: float = 0.0) -> float:
    """Room left over on this kerb after a bike lane cross-section, at its narrowest point.

    What a caller sizing a shy distance needs, and it goes through BikeLane's own accounting
    rather than being re-derived: a caller subtracting the travel lane and the lane width by
    hand misses the lane LINE, which is 0.82 ft and the difference between a section that fits
    e_broad_st_east and one that is refused for being 0.70 ft too wide.
    """
    lane = BikeLane(width_ft=width_ft, buffer_ft=buffer_ft, parking_ft=parking_ft)
    return narrowest_half_width_ft(state.legs[leg_name], side) - lane.total_ft


def widest_protected_lane_ft(state: DesignState, leg_name: str, side: str) -> float | None:
    """The widest PROTECTED bike lane this kerb can hold, or None if that is under the floor.

    THE BUFFER IS KEPT AND THE LANE GIVES, which is the opposite of what this project did first.
    The earlier rule held the lane at a nominal 5 ft and dropped the 2 ft buffer whenever the last
    few inches did not fit, so a kerb 0.51 ft short lost its flex posts entirely and got a
    conventional lane instead - trading all of the protection for 6 in of paint. A rider is better
    served by a 4.49 ft lane with a post beside it than by a 5 ft lane with a moving truck beside
    it, and 4 ft is a width AASHTO recognises (MIN_BIKE_LANE_FT).

    Ordered outward from the centerline, which is the order the widths are given up in: the 11 ft
    travel lane is fixed (TravelLanesKeepTheirWidth), the 2 ft buffer is fixed because it is what a
    post stands in, and the bike lane takes what is left - capped at the 5 ft design width, since
    spare beyond that is hatched rather than spent on a lane wider than the standard.

    Measured, this is the difference between one protected kerb and two on E Broad's east leg:
    +0.01 and +0.14 ft spare on the west leg (5 ft either side), -0.51 on the east right (4.49 ft,
    protected) and -1.20 on the east left (3.80 ft, under the floor - see the caller for what
    happens then).
    """
    spare_ft = bike_lane_spare_ft(state, leg_name, side, width_ft=BIKE_LANE_WIDTH_FT,
                                   buffer_ft=BIKE_LANE_BUFFER_FT)
    fitted_ft = min(BIKE_LANE_WIDTH_FT, BIKE_LANE_WIDTH_FT + spare_ft)
    return fitted_ft if fitted_ft >= MIN_BIKE_LANE_FT - LANE_WIDTH_SLACK_FT else None
