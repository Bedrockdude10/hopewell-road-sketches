"""WHERE THE TRAVEL-LANE DIVIDER SITS once a two-way lane has taken one kerbside.

ONE DEFINITION, because four things need it and they must agree: the two travel-lane checks in
src/checks.py, the plan view's lane dimension label, and the centreline paint both views draw. The
label was the one that got it wrong - it measured the lane from the ALIGNMENT and printed
"lane 9.6 ft" beside a lane the geometry had built at 11.00 ft.

Its own file because it is a question about the RESOLVED design rather than about a section: it
reads the treatments a state ended up with, so it sits above them and not beside them.
"""
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.fit import travel_lane_divider_shift_ft
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


def travel_lane_width_ft(state: DesignState, leg_name: str, side: str, painted_ft: float) -> float:
    """The real width of the travel lane on this side, given how much kerbside paint it has.

    From the DIVIDER to the paint, not from the alignment to the paint - those are the same thing
    only while the two lanes straddle the alignment. Everything that reports or checks a lane
    width goes through here.
    """
    half_ft = state.legs[leg_name].curb_to_curb_ft / 2
    return half_ft - painted_ft - divider_shift_toward_ft(state, leg_name, side)
