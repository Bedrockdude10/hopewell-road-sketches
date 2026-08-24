"""hold_travel_lane_at_target: does the kerbside leftover actually reach the traced kerb."""
import contextlib
import io

import pytest

from tests.conftest import needs_source_data


@needs_source_data
def test_a_kerb_that_narrows_only_at_its_tail_still_keeps_the_lane_over_the_reach():
    """A whole-leg minimum let one narrow tail veto a kerb that has room almost everywhere.

    w_broad_st_southwest's left kerb, at HOPEWELL_FRAME_SCALE=3.0 - the scale the corridor's own
    checked-in renders actually use, confirmed by matching output/wbroad_louellen's committed
    frame.radius_m against every candidate scale rather than assuming it (tests/conftest.py's
    WIDE_FRAME_SCALE=2.5 is the suite's own invariant-sweep convention and is a DIFFERENT number)
    - holds an 11 ft travel lane for 336 of its 390 traced feet and pinches inside it only over
    the last stretch. Judged by narrowest_half_width_ft over the WHOLE leg - what
    hold_travel_lane_at_target used before _lane_target_reach_ft existed - that one pinch reads
    back as "the street has nothing spare" and NOTHING is drawn on the other 336 ft either, which
    is exactly the bug SKILLS 0a and 0b describe for the two-way section: a single station
    deciding a whole approach.

    Asked per station instead, the lane (or marking) has to reach as far as the kerb actually
    holds it and refuse the tail by name - not run the whole leg (which would draw past where
    the kerb narrows) and not silently draw nothing (which is what shipped and drew no hatching
    on this kerb at all).
    """
    from src.geometry.intersection import load_intersection_model
    from src.geometry.model import curb_station_span, side_facing
    from src.geometry.targets import LegSide, LegTarget
    from src.geometry.treatments.corridor import BROAD_ST_TWO_WAY_BIKEWAY
    from src.geometry.treatments.lanes import LaneNarrowing
    from src.geometry.treatments.parking import MarkedParking
    from src.geometry.treatments.state import DesignState
    from src.render.frame import FRAME_SCALE_ENV

    LEG, SIDE = "w_broad_st_southwest", "left"
    CORRIDOR_RENDER_SCALE = "3.0"     # see the docstring: this is not WIDE_FRAME_SCALE

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(FRAME_SCALE_ENV, CORRIDOR_RENDER_SCALE)
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_intersection_model(site="wbroad_louellen")
    finally:
        monkey.undo()

    span = curb_station_span(model.legs[LEG], SIDE)
    assert span is not None, f"{LEG} {SIDE} has no traced kerb - this test is pinning nothing"

    # NOT a bare call: hold_travel_lane_at_target runs on the FAR kerb, after the corridor's
    # own AddTwoWayBikeLane has already gone on the near one (side_facing(leg, "north"), here
    # "right") and set the divider shift the far kerb's own room is measured against. Calling
    # it on a treatment-free state skips that shift and does not reproduce the real defect.
    near_side = side_facing(model.legs[LEG], BROAD_ST_TWO_WAY_BIKEWAY.side)
    assert near_side != SIDE, (
        f"{LEG}'s {BROAD_ST_TWO_WAY_BIKEWAY.side}-facing kerb is {SIDE!r}, same as the far "
        f"kerb under test - the fixture assumption behind this test no longer holds")
    with contextlib.redirect_stdout(io.StringIO()):
        state = BROAD_ST_TWO_WAY_BIKEWAY._place_on(
            DesignState.from_model(model), LEG, near_side, quiet=True)

    lane = state.treatment_for(LaneNarrowing, LegTarget(LEG))
    parking = state.treatment_for(MarkedParking, LegSide(LEG, SIDE))
    treatment = lane if lane is not None else parking
    assert treatment is not None, (
        f"nothing at all was marked on {LEG} {SIDE} - the whole-leg minimum is still vetoing "
        f"most of this kerb that has room")

    assert treatment.end_ft is not None, (
        f"the kerb narrows inside this leg's own tail (traced {span[0]:.1f}-{span[1]:.1f} ft), "
        f"so the treatment should stop short of the leg's end and refuse the rest rather than "
        f"either running the whole leg or (as it did before this fix) refusing all of it")
    assert float(span[0]) < treatment.end_ft < float(span[1]), (
        f"end_ft={treatment.end_ft} is not inside the traced kerb "
        f"{span[0]:.1f}-{span[1]:.1f} ft")

    refusals = state.refusals_on(LEG, SIDE)
    assert len(refusals) == 1, f"expected exactly one tail refusal, got {refusals}"
    tail = refusals[0]
    assert tail.start_ft == pytest.approx(treatment.end_ft), (
        f"the refusal starts at {tail.start_ft:.1f} ft but the paint stopped at "
        f"{treatment.end_ft:.1f} ft - a gap between them is kerb with neither paint nor a "
        f"reason on it")
    assert tail.end_ft >= float(span[1]) - 1e-6, (
        f"the refusal covers to {tail.end_ft:.1f} ft but the kerb is traced to {span[1]:.1f} ft")
    assert tail.narrowest_ft is not None and tail.narrowest_ft > 0
    assert f"{tail.narrowest_ft:.2f}" in tail.reason, (
        f"the reason should quote the width that stopped it: {tail.reason}")
