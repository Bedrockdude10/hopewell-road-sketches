"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.treatments import (BIKE_LANE_BUFFER_FT, MIN_TWO_WAY_BIKE_LANE_FT,
    MIN_BIKE_LANE_FT, widest_protected_lane_ft,
    TARGET_LANE_WIDTH_FT, AddBikeLane, AddBikeLaneBollards, DesignState, LaneNarrowing,
    MarkedParking, ProtectDaylightZone, all_crosswalks_continental, apply_osm_parking,
    complete_centerlines)

GREENWOOD_LEGS = ("greenwood_ave_north", "greenwood_ave_south")

TIGHTENED_RADIUS_FT = 10


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given.

    The named proposals were cleared for re-audit, so this is no longer a proposal: it just
    paints each kerb the way OSM says it is used - crossed hatching where parking is
    restricted, marked stalls where it isn't. Every mark here is derived from surveyed data,
    so nothing in it is a design choice waiting to be reviewed.

    Needs the model for the OSM tags, so it falls back to the untouched baseline when called
    with a state alone (the older single-argument convention).
    """
    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model)
    state = complete_centerlines(state)
    return all_crosswalks_continental(state)


def _protect_every_daylight_zone(state: DesignState, kind: str) -> DesignState:
    """Stand `kind` in every daylight zone this design created.

    Only kerbs that got marked parking have a daylight zone worth protecting - a kerb hatched
    end to end is already no-parking for its whole length, and objects along all of it would
    be street furniture, not a corner treatment.
    """
    for parking in state.treatments_of(MarkedParking):
        state = state.apply(ProtectDaylightZone(parking.target, kind=kind))
    return state


def build_proposal_daylight_bollards(baseline: DesignState, model=None) -> DesignState:
    """The default proposal, with flex-post bollards standing in each daylight zone.

    Identical geometry to build_demo_scenario - same lanes, same stalls, same hatching. The
    posts make the statutory setback self-enforcing instead of merely painted. They are NOT
    a curb extension under R.S. 39:4-138(e) (a flex-post bends flat under a tyre), so the
    25 ft setback stands and the parking is unchanged.
    """
    if model is None:
        return baseline
    return _protect_every_daylight_zone(build_demo_scenario(baseline, model), "bollards")


PARKING_SIDES = ("left", "right")  # both Broad St legs now mark parking on BOTH sides - see
                                     # _add_broad_st_both_side_parking's docstring for why this replaced the
                                     # one-side-parking + other-side-plain-line design from the previous iteration
PARKING_BUFFER_DEFAULT_FT = 4.0  # striped no-parking buffer width between the parking lane and curb (Proposals B/C)
                                  # - a rendering/design choice (a real "shy distance" off the curb), not from a
                                  # specific MUTCD/AASHTO figure the way PARKING_STALL_DEPTH/LENGTH_DEFAULT_FT are.


def _add_broad_st_both_side_parking(state: DesignState, curb_offset_ft: float = 0.0) -> DesignState:
    """Marked curbside parking on BOTH sides of both Broad St legs, starting
    at whichever is farther from the intersection out of the physical
    past-the-corner-curve point and the real legal minimum distance from the
    actual crosswalk (LEGAL_PARKING_SETBACK_FT, NJSA 39:4-138 - handled
    inside add_marked_parking's consumers, src/render/export.py and
    plan_view.py). Previously only the "right" side got parking, with the
    "left" (opposite/entering-traffic) side getting a plain 11 ft
    lane-narrowing line instead (add_lane_narrowing, line_only=True) - now
    that side gets real marked parking too, up to the same legal limit,
    replacing that plain line entirely (parking's own edge line already
    marks the travel lane boundary, just at parking's own depth_ft instead
    of an arbitrary 11 ft target)."""
    for leg_name in BROAD_ST_LEGS:
        for side in PARKING_SIDES:
            state = state.apply(MarkedParking(LegSide(leg_name, side), curb_offset_ft=curb_offset_ft))
    return state


# --- Broad St road-diet series: three escalating treatments for the two
# confirmed, over-wide Broad St legs (55.5/68 ft curb-to-curb vs. two travel
# lanes' worth of actual need), independent of the Greenwood-focused PBSAC
# proposals above. Each is a distinct scenario, not a stack, so they can be
# compared side by side.
BROAD_ST_LEGS = ("broad_st_west", "broad_st_east")
# TARGET_LANE_WIDTH_FT is imported from src, not redeclared here. It is a standard
# (NACTO/AASHTO urban minimum travel lane), not a per-site choice, and four sites each
# holding their own copy is what src/geometry/treatments/'s own comment on it warns
# about - a leg could then be narrowed to one number and checked against another.


def _narrow_broad_st_to_11ft_lanes(state: DesignState, line_only: bool = False) -> DesignState:
    """Paint-only lane narrowing on both Broad St legs: stripe each side's
    buffer so the real remaining travel lane is 11 ft, filling everything
    from the outside of that lane to the leg's own (config.yaml-confirmed)
    curb with paint. stripe_width_ft is derived per leg from its real width,
    not a fixed guess - broad_st_west (55.5 ft) and broad_st_east (68 ft) get
    different stripe widths (16.75 ft / 23 ft) because they're different
    widths in reality.

    The buffer's edge line doesn't stop in a straight cut where the
    crosswalk/stop-bar clearance zone begins - src/render/export.py
    automatically continues it curving into every corner a narrowed leg
    touches (see src/geometry/model/stripes.py:lane_narrowing_taper_ft) until it
    meets the real curb, reading as a soft, paint-only bulb-out - no separate
    treatment call needed here for that. line_only=True (see
    build_proposal_h_broad_st_line_only) skips the chevron fill in both the
    straight run and the taper, leaving just that edge/taper line - see
    add_lane_narrowing's own line_only param."""
    for leg_name in BROAD_ST_LEGS:
        half_width_ft = state.legs[leg_name].curb_to_curb_ft / 2
        stripe_width_ft = half_width_ft - TARGET_LANE_WIDTH_FT
        state = state.apply(LaneNarrowing(LegTarget(leg_name), stripe_width_ft, line_only=line_only))
    return state


# --- Bike lanes ----------------------------------------------------------------------------
#
# Buffered, not parking-protected, and the reason is worth keeping in the file.
#
# The parking-protected section - 8 parking + 3 buffer + 6 bike + 11 + 11 + 6 bike + 3 buffer
# - totals 48 ft, which does fit inside 52.0 and 55.5 ft of roadway. But the total is not the
# constraint. Everything in this project is measured as an offset from the leg centerline (the
# NJDOT alignment), and the PARKING SIDE alone needs 28.0 ft of it. broad_st_east has 26.0 ft
# nominal and 22.8 at its narrowest traced point; broad_st_west has 27.8 and 25.9. Both short.
#
# Fitting it would mean shifting the travel lanes off the alignment, which is a real design
# and not one this pipeline can draw: the alignment is the datum every offset, stop bar and
# crossing frame is measured from. Rather than draw 48 ft of paint across a leg that narrows
# to 46.5 ft, the proposal uses the buffered section, which fits at both legs' NARROWEST
# traced cross-section and not merely at their nominal one.
#
# The parking-protected form is also less useful here than it looks: Schedule I bans parking
# for 100 ft from the junction and Schedule III's 2-hour parking on the south side only starts
# about 114 ft out, so there is almost no legal parking inside the area these renders cover to
# protect a lane with.
# The section comes from src (BIKE_LANE_WIDTH_FT, BIKE_LANE_BUFFER_FT): a 5 ft lane with a 2 ft
# buffer. This site asked for 6 ft + 3 ft on the reasoning that Broad St can afford it, and it
# can - all four kerbs here have 21.3-26.6 ft to the alignment against the 18.8 the standard
# section needs. It is standardised anyway, because a lane width is a standard and not a thing to
# spend spare width on: what the spare width buys here is hatching, which is what says the road
# is narrower than it looks.
BIKE_LANE_BOLLARD_SPACING_FT = 8.0  # same flex-post pitch a daylight zone uses - reads as a
                                     # continuous delineator rather than a row of dots


def _one_way_bike_lanes_reference(baseline: DesignState, model=None) -> DesignState:
    """NOT RENDERED - kept as the reference one-way section the tests pin.

    Removed from the render set 2026-08-15: measured against the corridor, one-way lanes do not
    fit ANY continuous run of Broad St. The binding kerb has 15.13 ft and the narrowest one-way
    section - a 4 ft lane with no buffer, already below AASHTO's floor - needs 15.82 ft. So this
    was a proposal that could only ever be built in fragments, and a 500 yard bike lane is not a
    bike lane. The two-way section needs 13.82 ft on ONE kerb and does fit the whole length.

    The TREATMENT is still valid and still used elsewhere; it is this corridor that cannot take
    it. Kept under a non-build_ name so tests/test_curb_extensions.py can still pin the one-way
    rules (the buffer-is-kept-and-the-lane-gives fallback, the green surface bounds) without the
    scenario appearing in any render.

    Buffered bike lanes both sides of both Broad St legs. Greenwood Ave gets none.

    Per side, outward from the centerline: 11 ft travel lane, 3 ft painted buffer with flex-post
    delineators down it, 5 ft bike lane, then the leftover asphalt HATCHED to the kerb. That last
    part matters: a bike lane is a standard width and the street's spare width is not part of it,
    the same accounting an 8 ft parking stall gets when the remainder becomes a hatched kerb
    buffer. Without the outer stripe and that hatching the lane read as running all the way to
    the kerb and looked far wider than the 6 ft it is.

    PROTECTED, not just painted. The delineators stand in the buffer on the TRAFFIC side of the
    lane, which is the side a rider needs protecting from - posts in the kerb-side hatching would
    protect nothing. The 2 ft buffer is what makes that possible, and it is why E Broad's east leg's
    lanes
    (see that site's scenarios.py) cannot be protected: 17.6 ft to its nearest kerb is fully spent
    on an 11 ft lane, a 5 ft lane and their two stripes.

    The cross-section takes 18.8 ft of the 21.3 ft broad_st_east has where its kerbs come closest
    to the alignment, and of the 25.9 ft broad_st_west has - so it holds for the whole traced
    length of both legs rather than only where they are widest. broad_st_east is the binding case,
    and its kerb hatching pinches from about 5 ft down to half a foot at that narrow point.

    GREENWOOD AVE IS NOT PROPOSED FOR ONE. It has 2.3 ft (north) and 4.6 ft (south) per side
    spare beside an 11 ft lane, both under AASHTO's 5 ft minimum for an exclusive lane. A
    narrower lane is not a bike lane, and drawing one would be proposing something that fails
    the standard it is meant to meet - the sort of thing that gets waved through because the
    picture looks plausible. add_bike_lane refuses it; this is why.

    Greenwood's kerbs keep the OSM-derived markings the other proposals give them, so the
    junction still reads as a whole street rather than two treated legs floating in it.
    """
    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=GREENWOOD_LEGS)
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    for leg_name in BROAD_ST_LEGS:
        for side in ("left", "right"):
            # The same rule E Broad uses (widest_protected_lane_ft): the travel lane and the
            # buffer are fixed and the bike lane takes what is left, down to the floor. All four
            # of these kerbs have 2.51-7.77 ft to spare, so all four get the full design width -
            # but the rule lives in src rather than being a local assumption that happens to hold.
            lane_ft = widest_protected_lane_ft(state, leg_name, side)
            if lane_ft is None:
                print(f"  NOTE: no protected lane on {leg_name} {side} - under the "
                      f"{MIN_BIKE_LANE_FT:.0f} ft floor once the travel lane and buffer are taken.")
                continue
            state = state.apply(AddBikeLane(LegSide(leg_name, side), width_ft=lane_ft,
                                             buffer_ft=BIKE_LANE_BUFFER_FT))
            state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                    spacing_ft=BIKE_LANE_BOLLARD_SPACING_FT))
    return state


# --- The borough two-way corridor -----------------------------------------------------------
#
# A single two-way protected bike lane on ONE side of Broad St, running the length of Hopewell
# Borough - 6,871 ft of W Broad + E Broad - rather than a pair of one-way lanes on each leg.
#
# THE SIDE IS A CORRIDOR DECISION AND IT IS THE SOUTH KERB. Measured over the whole borough
# length from OSM, 2026-08-13:
#
#   * side streets cutting the kerb   north 10, SOUTH 7. Five crossings cut both kerbs whichever
#     side is chosen (Eaton/Ege, Lanning, Greenwood, Maple, Elm); the difference is one-sided
#     T-junctions - Windsor Way, Louellen, Mercer, Blackwell and Hamilton on the north against
#     Seminary and Princeton on the south.
#   * parking capacity lost           north 246 stalls, SOUTH 241. A 2% difference, and derived
#     from geometry rather than counted: OSM carries no parking:* tag anywhere on this corridor,
#     and the borough's Schedule I restrictions are not loaded as a data source. Treat it as a
#     tie, not as a finding.
#   * mapped driveways                north 20, south 21 - and NOT usable either way. OSM has a
#     driveway for 29% of the parcels fronting Broad St, so both figures are roughly threefold
#     undercounts and the undercount rate is the same on both sides.
#
# So the crossings decided it, because that is the count OSM records completely. Junctions are
# also the hazard that matters most for this treatment specifically: a two-way lane puts
# contraflow riders at every one of them, arriving from the direction a turning driver does not
# check.
#
# The side is chosen ONCE for the route and then translated per leg by side_facing() - a leg's
# left/right is in its own frame, so the same real kerb is "left" on the east approach and
# "right" on the west, and hand-translating it is how a corridor treatment ends up on the north
# kerb of one leg and the south kerb of the next.
CORRIDOR_SIDE = "south"

# TEN FEET, NOT THE 12 FT DESIGN WIDTH, AND PARKING IS WHY. Hopewell Borough is car-dependent;
# a corridor plan that removes a kerb of parking and returns none is not viable here whatever it
# does for riders. broad_st_east has 43.26 ft between its traced kerbs, and 12 ft of lane plus a
# 3 ft buffer plus two 11 ft travel lanes leaves 5.44 ft against the far kerb - under a stall, so
# the whole leg came out with no parking at all. At 10 ft the section leaves 7.44 ft, which is a
# usable stall.
#
# 10 ft is NACTO's MINIMUM for a two-way lane (12 ft desirable): two riders can pass, but an
# oncoming pair is tight. That is the cost, it is real, and it is the one being paid deliberately
# to keep the parking. The alternative on the table was narrowing the travel lanes to 10 ft
# instead, which would have kept the lane at 12 - not taken, so the travel lanes hold 11 ft.
CORRIDOR_LANE_WIDTH_FT = MIN_TWO_WAY_BIKE_LANE_FT
# The narrowest parallel stall worth marking. Below 7 ft a car cannot sit clear of the travel
# lane, so it is not a stall; src's MIN_MARKED_PARKING_DEPTH_FT (8 ft) is the width to mark when
# the street can spare it, not the floor for whether parking exists at all.
MIN_USABLE_STALL_FT = 7.0


def build_proposal_two_way_bike_lane(baseline: DesignState, model=None) -> DesignState:
    """A 12 ft two-way protected bike lane along the south kerb of both Broad St legs.

    Across the road from the NORTH kerb: travel lane, the double yellow, travel lane, a 3 ft
    buffer with flex posts in it, the 12 ft two-way lane with its yellow contraflow stripe, and
    the leftover hatched to the south kerb.

    THE ALIGNMENT DOES NOT MOVE, and that is what makes this drawable at all - see
    TwoWayBikeLane. The travel lanes shift north because 15.8 ft comes out of the south
    kerbside, so the double yellow between them shifts with them; every station, crossing frame
    and stop bar is still measured from the NJDOT alignment exactly as before.

    Greenwood Ave gets none, for the same reason it gets no one-way lane: 2.3 and 4.6 ft spare
    per side beside an 11 ft lane. Its kerbs keep the OSM-derived markings.

    The 3 ft buffer is NACTO's figure for a two-way lane beside moving traffic, wider than the
    2 ft a one-way lane gets, because a head-on error here is a closing speed rather than an
    overtaking one.
    """
    from src.geometry.model import side_facing
    from src.geometry.treatments import (TWO_WAY_BIKE_LANE_BUFFER_FT, AddTwoWayBikeLane,
                                          hold_travel_lane_at_target)

    if model is None:
        return baseline
    # Greenwood keeps its OSM-derived kerb paint; the Broad legs are re-striped entirely.
    state = apply_osm_parking(baseline, model, legs=GREENWOOD_LEGS)
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    for leg_name in ("broad_st_east", "broad_st_west"):
        side = side_facing(state.legs[leg_name], CORRIDOR_SIDE)
        lane = AddTwoWayBikeLane(LegSide(leg_name, side), width_ft=CORRIDOR_LANE_WIDTH_FT,
                                  buffer_ft=TWO_WAY_BIKE_LANE_BUFFER_FT)
        try:
            state = state.apply(lane)
        except ValueError as too_narrow:
            # Reported, not silently narrowed or dropped: which legs of the corridor can carry
            # the section IS the finding this scenario exists to produce.
            print(f"  NOTE: no two-way lane on {leg_name} {side} ({CORRIDOR_SIDE} kerb) - "
                  f"{too_narrow}")
            continue
        state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                 spacing_ft=BIKE_LANE_BOLLARD_SPACING_FT))
        # THE FAR KERB GETS THE SURPLUS, and this is why the two belong in one proposal
        # rather than two: the kerb that loses its parking to the bike lane is not the kerb that
        # gains this. hold_travel_lane_at_target holds the lane at 11 ft and spends what is left
        # on parking where the kerb may legally hold it, hatching where it may not - the same
        # rule every other kerb in the project gets, in src rather than written out here. It
        # WAS written out here, and sites/ebroad_princeton/scenarios.py did not have it, which
        # is how the same corridor treatment left E Broad with 11.68 ft lanes.
        state = hold_travel_lane_at_target(state, leg_name, str(Side(side).other))
    return state
