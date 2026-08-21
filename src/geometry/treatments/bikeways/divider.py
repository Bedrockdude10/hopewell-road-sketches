"""WHERE THE TRAVEL-LANE DIVIDER SITS once a two-way lane has taken one kerbside.

ONE DEFINITION, because four things need it and they must agree: the two travel-lane checks in
src/checks.py, the plan view's lane dimension label, and the centreline paint both views draw. The
label was the one that got it wrong - it measured the lane from the ALIGNMENT and printed
"lane 9.6 ft" beside a lane the geometry had built at 11.00 ft.

Its own file because it is a question about the RESOLVED design rather than about a section: it
reads the treatments a state ended up with, so it sits above them and not beside them.
"""
from src.geometry.treatments.base import TARGET_LANE_WIDTH_FT
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.fit import divided_lane_width_ft, travel_lane_divider_shift_ft
from src.geometry.treatments.bikeways.place import AddTwoWayBikeLane

def divider_shift_toward_ft(state: DesignState, leg_name: str, side: str) -> float:
    """How far the travel-lane divider sits off the alignment, measured TOWARD `side`.

    Zero on every leg whose travel lanes straddle the alignment, which is all of them until a
    two-way bike lane takes width out of one kerbside. Signed, because the two sides of a leg see
    the same shift in opposite directions, and anything that ignores the sign is wrong on exactly
    one of them.

    ONE DEFINITION, because four things need it and they must agree: the two travel-lane checks in
    src/checks.py, the plan view's lane dimension label, and the centreline paint both views draw.
    The label was the one that got it wrong - it measured the lane from the ALIGNMENT and printed
    "lane 9.6 ft" beside a lane the geometry had built at 11.00 ft. A wrong number on a correct
    drawing is worse than a wrong drawing, because it is the number a reviewer takes away, and an
    11 ft lane is not negotiable with a county engineer.
    """
    for treatment in state.treatments_of(AddTwoWayBikeLane):
        if treatment.target.leg != leg_name:
            continue
        shift_ft = travel_lane_divider_shift_ft(treatment.section(state))
        # The shift is defined as positive AWAY from the side carrying the lane.
        return -shift_ft if str(treatment.target.side) == str(side) else shift_ft
    return 0.0


def travel_lane_edge_ft(state: DesignState, leg_name: str, side: str) -> float:
    """How far from the alignment the travel lane REACHES on this side - where kerbside starts.

    The one home for a sum that had three, and the two wrong copies did not agree with the drawing.
    Everything that has to know where the travel way ends asks here: the kerb-opening cutter, the
    paint checks, and anything sizing a kerbside zone.

    TARGET_LANE_WIDTH_FT from the alignment on every leg whose two travel lanes straddle it, which
    is every leg of every scenario until a two-way bike lane takes one kerbside. Then the sum is
    the DIVIDER's offset plus the lane's own width, and both terms move:

      * the divider is off the alignment by divider_shift_toward_ft, signed, so the two sides of
        one leg see it in opposite directions;
      * the lane is divided_lane_width_ft wide, which is the target only where the leg can hold
        two of them. w_broad_st_northeast cannot: it splits the travel way at 10.08 ft a lane.

    WRITING IT AS `TARGET_LANE_WIDTH_FT + shift` IS WRONG ON EXACTLY THE LEGS THAT SPLIT, and by
    the whole 0.92 ft shortfall. It reads as innocent because it is right on five of the six
    corridor legs, and on the sixth it put the kerb-opening cutter 0.92 ft outboard of the buffer
    it had to cut: the cut landed inside the zone, took all of it but a 0.10 ft ribbon along its
    inner face, and PaintContext.rim then outlined that ribbon as a 49.68 ft solid edge line
    straight across a 9.5 ft driveway. The neat swept fillet the other legs get comes from the cut
    starting INSIDE the zone; a tenth of a foot of sign error is the whole difference.

    Equivalent to the section's own `travel_lane_edge_ft` on the side carrying the lane, and this
    is the general form - checks.ZonesGiveWayAtAnOpening asks about both sides.
    """
    for treatment in state.treatments_of(AddTwoWayBikeLane):
        if treatment.target.leg != leg_name:
            continue
        return (divider_shift_toward_ft(state, leg_name, side)
                + divided_lane_width_ft(treatment.section(state)))
    return TARGET_LANE_WIDTH_FT


def travel_lane_width_ft(state: DesignState, leg_name: str, side: str, painted_ft: float) -> float:
    """The real width of the travel lane on this side, given how much kerbside paint it has.

    From the DIVIDER to the paint, not from the alignment to the paint - those are the same thing
    only while the two lanes straddle the alignment. Everything that reports or checks a lane
    width goes through here.
    """
    half_ft = state.legs[leg_name].curb_to_curb_ft / 2
    return half_ft - painted_ft - divider_shift_toward_ft(state, leg_name, side)
