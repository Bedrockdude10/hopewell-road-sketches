"""Treatment proposals for this site.

Three escalating options, all sharing the same crosswalk upgrade:
  1. continental crosswalks
  2. + 11 ft travel lanes with the recovered width marked as parking
  3. + mountable pedestrian bulb-outs (drivable by trucks/EMS)

Existing conditions come from the July 2026 Hopewell Crosswalk Inventory and OSM;
see this site config.yaml. Every width here is osm_derived or estimated, NOT field-
measured, so treat the lane/parking dimensions below as a design study rather than a
construction drawing.
"""
from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.treatments import (
    MIN_BIKE_LANE_FT, MIN_TWO_WAY_BIKE_LANE_FT, widest_protected_lane_ft,BIKE_LANE_BUFFER_FT, BIKE_LANE_WIDTH_FT,
    LANE_WIDTH_SLACK_FT, TARGET_LANE_WIDTH_FT, AddBikeLane, AddBikeLaneBollards, DesignState,
    LaneNarrowing, MarkedParking, ProtectDaylightZone, UpgradeCrosswalkMarkings,
    all_crosswalks_continental, apply_osm_parking, bike_lane_spare_ft, complete_centerlines)

# TARGET_LANE_WIDTH_FT is imported from src, not redeclared here. It is a standard
# (NACTO/AASHTO urban minimum travel lane), not a per-site choice, and four sites each
# holding their own copy is what src/geometry/treatments.py's own comment on it warns
# about - a leg could then be narrowed to one number and checked against another.
PARKING_DEPTH_FT = 8.0        # a standard marked parallel stall
MIN_PARKING_DEPTH_FT = 7.0    # below this it isn't a usable stall, so none is marked


def _continental_everywhere(state: DesignState) -> DesignState:
    """Repaint every leg's crosswalk to continental. Applied to every leg, not just the
    ones marked today - upgrading a marking a leg doesn't have would be a new crossing,
    which is a different proposal, but src/render/export.py only paints legs listed in
    the config's existing_marked_crosswalks, so unmarked legs stay unmarked in the render."""
    for leg_name in state.legs:
        state = state.apply(UpgradeCrosswalkMarkings(LegTarget(leg_name), "continental"))
    return state


def _parking_and_narrowing(state: DesignState) -> DesignState:
    """Narrow every travel lane to TARGET_LANE_WIDTH_FT and put the recovered width to
    work as marked parallel parking.

    The two treatments have to be sized together, not stacked: an 11 ft lane plus an 8 ft
    stall needs 19 ft per side, and most legs here are 33-39 ft curb-to-curb (16.5-19.5 ft
    per side). So the recovered width - half the roadway minus 11 ft - is what's available,
    and each leg gets whichever of these fits:

      * >= PARKING_DEPTH_FT + 1: an 8 ft stall, with the remainder as a striped buffer
        between the stall and the kerb (add_marked_parking's curb_offset_ft).
      * >= MIN_PARKING_DEPTH_FT: a single stall taking the whole recovered width.
      * less than that: no parking - paint-only lane narrowing, since a 6 ft stall isn't
        a stall. The leg still gets its 11 ft lanes.

    Printed per leg so the trade-off is visible rather than buried in geometry.
    """
    for leg_name, leg in state.legs.items():
        recovered_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        if recovered_ft < MIN_PARKING_DEPTH_FT:
            state = state.apply(LaneNarrowing(LegTarget(leg_name), max(recovered_ft, 0.5)))
            print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) recovers only "
                  f"{recovered_ft:.1f} ft per side at {TARGET_LANE_WIDTH_FT:.0f} ft lanes - too narrow "
                  f"for a stall, so paint-only narrowing here, no parking.")
            continue
        depth_ft = min(recovered_ft, PARKING_DEPTH_FT)
        buffer_ft = max(recovered_ft - depth_ft, 0.0)
        for side in ("left", "right"):
            state = state.apply(MarkedParking(LegSide(leg_name, side), depth_ft=depth_ft, curb_offset_ft=buffer_ft))
        print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) -> {TARGET_LANE_WIDTH_FT:.0f} ft lanes + "
              f"{depth_ft:.1f} ft parking both sides"
              + (f" + {buffer_ft:.1f} ft striped buffer" if buffer_ft > 0.1 else "") + ".")
    return state


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



def build_proposal_daylight_bollards(baseline: DesignState, model=None) -> DesignState:
    """The default proposal, with flex-post bollards standing in each daylight zone.

    Identical geometry to build_demo_scenario - same lanes, same hatching, same crossings.
    The posts change nothing that is painted; they make the statutory setback in
    R.S. 39:4-138 self-enforcing instead of merely marked, which is the whole difference
    between a drawing of the law and a corner that stays clear.

    Every kerb here is HATCHED rather than marked for parking (no leg has a stall's worth of
    spare width beside an 11 ft lane), so unlike Broad & Greenwood the zones to protect are
    keyed off the narrowed sides, not off parking. That does not make them longer: a daylight
    zone is bounded by the statute at the corner however the rest of the kerb is painted, so
    the posts stand at the corners and nowhere else. Flex-posts are NOT a curb extension
    under 39:4-138(e), so the 25 ft setback is unchanged.
    """
    if model is None:
        return baseline
    state = build_demo_scenario(baseline, model)
    treated = {parking.target for parking in state.treatments_of(MarkedParking)}
    for narrowing in state.treatments_of(LaneNarrowing):
        treated.update(LegSide(narrowing.target.leg, side) for side in narrowing.sides)
    for kerb in sorted(treated):
        state = state.apply(ProtectDaylightZone(kerb, kind="bollards"))
    return state


# --- Bike lanes ----------------------------------------------------------------------------
#
# Not parking-protected: there is no parking here to protect a lane with. Both sides of
# e_broad_st_east are tagged no_stopping in OSM, and e_broad_st_west is no_stopping too - so the
# width a bike lane would use is width nobody is allowed to stand a vehicle in today.
#
# BOLLARD-PROTECTED WHERE THE WIDTH ALLOWS, WHICH IS ONE LEG OF THE TWO. Flex posts protecting a
# bike lane belong in a buffer on the TRAFFIC side of it, so a protected lane needs the standard
# 2 ft buffer (src: BIKE_LANE_BUFFER_FT). Measured to each kerb's nearest approach to the
# alignment, against the 18.82 ft a full section spends (11 travel + 2 buffer + 5 lane + the
# 0.82 ft outer stripe):
#
#     e_broad_st_east  left  17.62 ft   short 1.20 ft   conventional
#     e_broad_st_east  right 18.31 ft   short 0.51 ft   conventional
#     e_broad_st_west  left  18.83 ft   fits            protected
#     e_broad_st_west  right 18.96 ft   fits            protected
#
# So e_broad_st_west's lanes are buffered and posted and e_broad_st_east's stay conventional, at
# 17.64 ft of unbuffered section against 17.62 - its lane runs hard against the kerb with no
# hatched margin at all. An earlier version of this file made both legs conventional on the
# reasoning that a mixed pair reads worse than a consistent one. That was a presentation
# judgement overriding a real safety treatment on the leg that can carry it, and it is the wrong
# way round: the finding to report is that this junction can protect half its bike network and
# exactly why not the other half. Broad & Greenwood's four kerbs all take the full section.
#
# The buffer is the STANDARD width or nothing - not whatever a kerb can spare. Sizing it from the
# spare gave 2.01 and 2.14 ft buffers, which is a lane sized by the noisiest input in the model
# rather than to a standard. Spare width's job here is to be hatched.
BIKE_LANE_BOLLARD_SPACING_FT = 8.0  # matches Broad & Greenwood's pitch - reads as a continuous
                                     # delineator rather than a row of dots
E_BROAD_LEGS = ("e_broad_st_east", "e_broad_st_west")


def build_proposal_bike_lanes(baseline: DesignState, model=None) -> DesignState:
    """Bike lanes both sides of both E Broad St legs - protected on the leg that can hold a
    buffer. Princeton Ave gets none.

    Per side, outward from the centerline: 11 ft travel lane, its edge stripe, a buffer where the
    leg can spare one, a 5 ft bike lane, its outer stripe, and whatever asphalt is left hatched to
    the kerb. On e_broad_st_east that leftover is essentially nothing - the cross-section spends
    17.64 of the 17.62 ft its narrowest kerb offers - so its lane runs hard against the kerb rather
    than with the hatched margin Broad & Greenwood's lanes get, and there is no room for the buffer
    a flex post would stand in. That is the leg's width, not a drawing choice; see the note above
    for the measurements and for why the two legs are treated differently.

    PRINCETON AVE IS NOT PROPOSED FOR ONE. It has 4.1 ft per side spare beside an 11 ft lane,
    under AASHTO's 5 ft minimum, so there is no lane to draw - only a narrower stripe that would
    read as one. Its kerbs keep the OSM-derived markings the other proposals give them.

    Both E Broad legs are already no_stopping, so this displaces no parking. Where a leg turns
    out not to have the room after all, it is reported and left alone rather than given a lane
    that does not fit - the point of the exercise is to find out which legs can take one.
    """
    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=("princeton_ave_south",))
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    for leg_name in E_BROAD_LEGS:
        for side in ("left", "right"):
            # PROTECTION FIRST, and the lane gives to keep it. The buffer is what a flex post
            # stands in, so a kerb that is a few inches short narrows its lane to the floor rather
            # than spending the whole buffer to hold a nominal 5 ft - see widest_protected_lane_ft.
            lane_ft = widest_protected_lane_ft(state, leg_name, side)
            if lane_ft is not None:
                state = state.apply(AddBikeLane(LegSide(leg_name, side), width_ft=lane_ft,
                                                 buffer_ft=BIKE_LANE_BUFFER_FT))
                state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                        spacing_ft=BIKE_LANE_BOLLARD_SPACING_FT))
                if lane_ft < BIKE_LANE_WIDTH_FT - LANE_WIDTH_SLACK_FT:
                    print(f"  NOTE: {leg_name} {side}'s protected lane is {lane_ft:.2f} ft, under "
                          f"the {BIKE_LANE_WIDTH_FT:.0f} ft design width - the kerb has room for "
                          f"the 11 ft travel lane, the {BIKE_LANE_BUFFER_FT:.0f} ft buffer and this "
                          f"much lane, and the buffer is kept because it is what the posts stand "
                          f"in. Above the {MIN_BIKE_LANE_FT:.0f} ft floor.")
                continue
            # Under the floor with a buffer. Fall back to the conventional lane this kerb CAN hold
            # rather than dropping it: a 5 ft unprotected lane is still a lane, and losing it
            # entirely would be a worse answer than the one being flagged.
            try:
                state = state.apply(AddBikeLane(LegSide(leg_name, side),
                                                 width_ft=BIKE_LANE_WIDTH_FT, buffer_ft=0.0))
            except ValueError as too_narrow:
                print(f"  NOTE: no bike lane on {leg_name} {side} - {too_narrow}")
                continue
            spare_ft = bike_lane_spare_ft(state, leg_name, side, width_ft=BIKE_LANE_WIDTH_FT,
                                           buffer_ft=BIKE_LANE_BUFFER_FT)
            print(f"  NOTE: {leg_name} {side} is CONVENTIONAL, not protected - keeping the "
                  f"{BIKE_LANE_BUFFER_FT:.0f} ft buffer would leave "
                  f"{BIKE_LANE_WIDTH_FT + spare_ft:.2f} ft of lane, under the "
                  f"{MIN_BIKE_LANE_FT:.0f} ft floor. Widening this kerb by "
                  f"{MIN_BIKE_LANE_FT - (BIKE_LANE_WIDTH_FT + spare_ft):.2f} ft would buy a "
                  f"protected lane.")
    return state


# --- The borough two-way corridor -----------------------------------------------------------
#
# The same route as sites/broad_st_greenwood/scenarios.py - one two-way lane along the SOUTH
# kerb for the whole borough length of Broad St. See that file for the measurements the side was
# chosen on (10 side streets cutting the north kerb against 7 on the south, over a corridor
# where the parking difference is 2% and the driveway data is 29% complete).
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


def build_proposal_two_way_bike_lane(baseline: DesignState, model=None) -> DesignState:
    """A 12 ft two-way protected bike lane along the south kerb of both E Broad St legs.

    E Broad is the corridor's narrow end - 36.0 and 37.9 ft between traced kerbs against Broad &
    Greenwood's 43.3 and 52.5 - so this is where the section is most likely not to fit, and the
    refusal carries the measurement when it does not. Both legs are already no_stopping on both
    sides, so unlike the rest of the corridor this stretch displaces no parking at all.

    Princeton Ave gets none: it has 4.1 ft per side spare beside an 11 ft lane, under AASHTO's
    minimum for even a one-way lane.
    """
    from src.geometry.model import side_facing
    from src.geometry.treatments import (TWO_WAY_BIKE_LANE_BUFFER_FT, AddTwoWayBikeLane,
                                          hold_travel_lane_at_target)

    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=("princeton_ave_south",))
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    for leg_name in E_BROAD_LEGS:
        side = side_facing(state.legs[leg_name], CORRIDOR_SIDE)
        try:
            state = state.apply(AddTwoWayBikeLane(LegSide(leg_name, side),
                                                  width_ft=CORRIDOR_LANE_WIDTH_FT,
                                                  buffer_ft=TWO_WAY_BIKE_LANE_BUFFER_FT))
        except ValueError as too_narrow:
            print(f"  NOTE: no two-way lane on {leg_name} {side} ({CORRIDOR_SIDE} kerb) - "
                  f"{too_narrow}")
            continue
        state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                 spacing_ft=BIKE_LANE_BOLLARD_SPACING_FT))
        # And hold the OPPOSITE kerb's travel lane at 11 ft, spending the surplus on parking or
        # hatching. Missing here while broad_st_greenwood had it inline is exactly what left this
        # site with 11.68 ft and 13.21 ft lanes; TravelLanesHoldTheTarget now fails the build for
        # it, and the rule lives in src so there is one of it.
        state = hold_travel_lane_at_target(state, leg_name, str(Side(side).other))
    return state
