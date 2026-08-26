"""WHETHER A SECTION FITS THIS KERB, and what it leaves for everything else.

Pure arithmetic on a section and a traced kerb: no treatment, no design state being mutated, so a
scenario can ask these before committing to anything. THE ANSWERS ARE ORDERED OUTWARD FROM THE
ALIGNMENT, which is the order widths are given up in - the travel lane is fixed, the buffer is
fixed because it is what a post stands in, and the bike lane takes what is left.

Go through `bike_lane_spare_ft` rather than subtracting widths by hand; a caller doing its own
accounting misses the lane LINE, which is 0.82 ft and decides whether e_broad_st_east is buildable.
"""
import numpy as np

from src.geometry.model import narrowest_half_width_ft
from src.geometry.treatments.base import (LANE_WIDTH_SLACK_FT, PARKING_STALL_DEPTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT)
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


#: A stretch shorter than this is not a facility, it is a gap between two of them. A rider cannot
#: use 30 ft of protected lane, and drawing one invites the reader to count it as coverage.
MIN_FACILITY_RUN_FT = 100.0


def section_at(facility, near_half_ft: float, far_half_ft: float):
    """The best rung of the facility's ladder that fits this cross-section, or None.

    THE CLASS IS THE PREDICATE. TwoWayBikeLane.__post_init__ already refuses a section that
    leaves the travel lanes under NACTO's floor, with the measurement in the message; writing a
    second "does it fit" test here would be a second definition of the rule the whole facility
    turns on, and the two would drift the first time the floor changed. So a rung is tried by
    CONSTRUCTING it, and the ValueError it raises is the refusal, quoted verbatim.

    HERE RATHER THAN IN corridor_paint, WHICH IS WHERE IT WAS, because both renderers ask it. The
    between-junction strip asked it per station and the junction pieces did not ask it at all -
    they took two whole-leg minima and applied one section to the whole approach - and that is
    how a quarter-inch shortfall at one station of W Broad's southwest approach denied a protected
    bikeway over the 270 ft where the FULL rung fits. Two renderers, one fit rule.

    THE NARROWEST RUNG THAT FITS IS THE DEFAULT, not the widest, mirroring
    `CorridorFacility._place_on`'s policy rather than the mere geometric fit this used to stop at.
    `facility.sections` is declared widest to narrowest, so the last rung to construct without
    raising is the floor. A wider rung is only taken where it costs the far kerb nothing: this
    compares `far_kerb_surplus_ft` against the same `MIN_USABLE_STALL_FT`/
    `PARKING_STALL_DEPTH_DEFAULT_FT` cap `allocate_kerbside` applies, so the corridor strip agrees
    with the per-junction sheets on which rung is drawn even though it never resolves a
    `DesignState` to check. Before this, the strip kept the old first-fit-wins order and picked
    the widest geometrically-possible rung regardless: narrowing the junction sheets' bikeway
    left the between-junction strip's stall count exactly where it started.

    `facility` is anything carrying a `.sections` ladder, which is CorridorFacility; typed loosely
    on purpose, since that class is layered above this module.
    """
    from src.geometry.treatments.parking import MIN_USABLE_STALL_FT

    candidates = []
    refusal = None
    for rung in facility.sections:
        try:
            candidates.append(TwoWayBikeLane(width_ft=rung.width_ft, buffer_ft=rung.buffer_ft,
                                             constrained=rung.constrained, near_half_ft=near_half_ft,
                                             far_half_ft=far_half_ft))
        except ValueError as too_narrow:
            refusal = str(too_narrow)
    if not candidates:
        return None, refusal

    def stall_depth_ft(section: TwoWayBikeLane) -> float:
        surplus_ft = far_kerb_surplus_ft(section)
        return min(surplus_ft, PARKING_STALL_DEPTH_DEFAULT_FT) if surplus_ft >= MIN_USABLE_STALL_FT else 0.0

    chosen = candidates[-1]
    narrow_depth = stall_depth_ft(chosen)
    for wider in candidates[:-1]:
        if stall_depth_ft(wider) >= narrow_depth:
            chosen = wider
            break
    return chosen, None


def travel_way_profile(leg, side: str, from_ft: float = 0.0, to_ft: float | None = None):
    """(stations, near half-widths, far half-widths) over the stretch where BOTH kerbs are traced.

    The measurement a two-way section is judged on, station by station. Clipped to the
    intersection of the two traced spans, because outside it one of the two numbers is
    extrapolation - curb_offsets_at_stations interpolates and will happily flat-extend a kerb
    tens of feet past where the surveyor stopped, which reads as room that was never measured.

    BOTH SIDES ON ONE GRID, which is the same invariant curbside_strip_polygon states for a
    strip's two boundaries: sampled at different stations, near and far are a cross-section of
    nothing. None where there is no such stretch to measure.
    """
    from src.geometry.model import curb_offsets_at_stations, curb_station_span, half_width_profile

    other = "right" if side == "left" else "left"
    near_span, far_span = curb_station_span(leg, side), curb_station_span(leg, other)
    if near_span is None or far_span is None:
        return None
    lo = max(near_span[0], far_span[0], from_ft)
    hi = min(near_span[1], far_span[1],
             leg.centerline.length if to_ft is None else to_ft)
    profile = half_width_profile(leg, side, lo, hi)
    if profile is None:
        return None
    stations, near_ft = profile
    far_ft = curb_offsets_at_stations(leg, other, stations)
    if far_ft is None:
        return None
    return stations, near_ft, np.abs(far_ft)


def governing_half_widths_ft(leg, side: str, from_ft: float = 0.0, to_ft: float | None = None
                              ) -> tuple[float, float]:
    """The two half-widths a section promised over this stretch has to fit between.

    TWO QUESTIONS, AND THEY DO NOT BIND AT THE SAME STATION. Returning one kerb pair papers over
    that, so which pair is a real choice and both obvious answers are wrong.

    WHERE THE NEAR-SIDE PAINT GOES is the near kerb's OWN minimum. `near_half_ft` is the datum
    every kerbside mark is placed off - `travel_edge_ft = near_half_ft - section_ft` in
    TwoWayBikeLane.offsets_from_centerline_ft - so a section built on anything wider overruns the
    near kerb wherever it pinches. This function returned the min-SUM station's own near half for
    exactly one session, and the receipt is worth keeping: on w_broad_st_northeast at 3x that
    station reads 17.05 ft while the same kerb comes in to 16.12 ft at station 35.8, which put the
    outer edge line 0.84 ft INSIDE its own floor and 4.9 sq ft of white stripe on top of the green.
    markings_collide, fatal. The floor in AddBikeLane.paint holds each mark to its designed offset
    and the green to a band off the kerb, and those two rules only agree while the section fits.

    WHAT IS LEFT FOR THE TRAVEL LANES is the FAR kerb's OWN minimum, and the smallest SUM is the
    trap. min(near + far) is what the STREET measures at its narrowest cross-section, and on
    w_broad_st_northeast that is 2.44 ft more than the travel way ever gets: the section's inner
    edge is held on the alignment at `min(near) - section_ft`, because the travel lane's edge
    always comes off the alignment so the lane holds its width whatever the kerb does (place.py,
    lane_edge_line - only the lane's OWN two edges hug the kerb). So where the near kerb runs
    wider than its own minimum the surplus is drawn as bike-lane hatching on THAT kerb and never
    reaches the travel way. The drawn travel way is `min(near) - section_ft + far(s)`, whose
    minimum over the leg is exactly the two independent minima.

    THIS RETURNED min(near + far) - min(near) FOR ONE SESSION and the receipt is the point. It
    credited that approach with a travel way it does not have, promoted it from the constrained
    rung to the full one, and put the divider 9.45 ft toward a far kerb that pinches to 17.30 ft
    at station 48 - a 7.88 ft travel lane on a rural arterial carrying trucks, at 49 of 67
    stations. Nothing in the repo could see it: all three travel-lane checks either need kerbside
    PAINT to fire or clamp what a lane is entitled to AT the traced kerb, so a lane squeezed by a
    bare kerb could not fail one. ShiftedTravelLanesClearTheirKerb now closes that, and this
    arithmetic is what stops needing it.

    A near pinch at station 40 and a far pinch at station 300 DO describe a cross-section that
    exists nowhere - the objection that motivated the sum was right about the street and wrong
    about the drawing. With a constant-width section on one datum and a straight divider, neither
    kerb's slack is reachable from the other, so here the conservative pair is also the exact one.

    AND `to_ft` BOUNDS THE NEAR SIDE ONLY, because the two kerbs bound two different drawings.
    The facility stops where the reach stops; the DIVIDER it implies does not - a centre stripe is
    one offset per approach and it is drawn to the end of the leg whatever the green does. So a far
    minimum taken over the run measures the wrong stretch of kerb, and the tail pays: on a 3x
    w_broad_st_southwest the run stopped at 371.5 ft where the far kerb holds 15.95 ft, the leg
    runs to 389.9 where it comes in to 15.65, and the 0.30 ft went straight out of a travel lane
    already at its floor - a 9.65 ft lane under a design claiming 9.95, at 8 of 196 stations, with
    nothing in the arithmetic that had looked. Measuring the far kerb over the whole leg costs
    33.9 ft of that approach's protected lane and buys a divider that holds everywhere it is
    drawn. It changes nothing at 1x or 2.5x, where every corridor approach reaches the end of its
    kerb and the two stretches are the same stretch.

    Falls back to each side's own minimum where there is nothing traced to measure, which is
    narrowest_half_width_ft's nominal answer - no measurement, so no reason to prefer a pairing.
    """
    profile = travel_way_profile(leg, side, from_ft, to_ft)
    if profile is None:
        other = "right" if side == "left" else "left"
        return (narrowest_half_width_ft(leg, side, from_ft, to_ft),
                narrowest_half_width_ft(leg, other, from_ft, to_ft))
    _stations, near_ft, _far_over_the_run = profile
    whole_leg = travel_way_profile(leg, side)
    far_ft = _far_over_the_run if whole_leg is None else whole_leg[2]
    return float(near_ft.min()), float(far_ft.min())
