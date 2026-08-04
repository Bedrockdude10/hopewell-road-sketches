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
from src.geometry.treatments import (
    add_bike_lane, add_lane_narrowing, add_marked_parking, all_crosswalks_continental,
    apply_osm_parking, complete_centerlines, DesignState, protect_daylight_zone,
    TARGET_LANE_WIDTH_FT, upgrade_crosswalk_markings,
)

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
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
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
            state = add_lane_narrowing(state, leg_name, max(recovered_ft, 0.5))
            print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) recovers only "
                  f"{recovered_ft:.1f} ft per side at {TARGET_LANE_WIDTH_FT:.0f} ft lanes - too narrow "
                  f"for a stall, so paint-only narrowing here, no parking.")
            continue
        depth_ft = min(recovered_ft, PARKING_DEPTH_FT)
        buffer_ft = max(recovered_ft - depth_ft, 0.0)
        for side in ("left", "right"):
            state = add_marked_parking(state, leg_name, side=side, depth_ft=depth_ft,
                                        curb_offset_ft=buffer_ft)
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
    treated = set(state.parking_zones)
    for leg_name in state.lane_narrowing:
        for side in state.lane_narrowing_sides.get(leg_name, ("left", "right")):
            treated.add((leg_name, side))
    for leg_name, side in sorted(treated):
        state = protect_daylight_zone(state, leg_name, side, kind="bollards")
    return state


# --- Bike lanes ----------------------------------------------------------------------------
#
# Conventional, not parking-protected: there is no parking here to protect a lane with. Both
# sides of e_broad_st_east are tagged no_stopping in OSM, and e_broad_st_west is no_stopping too -
# so the width a bike lane would use is width nobody is allowed to stand a vehicle in today.
#
# AND NOT BOLLARD-PROTECTED EITHER, which is a width finding rather than a choice. Flex posts
# protecting a bike lane belong in a buffer on the TRAFFIC side of it, and E Broad has no room
# for one. Measured to each kerb's nearest approach to the alignment:
#
#     e_broad_st_east  left  17.62 ft    right 18.31 ft
#     e_broad_st_west  left  18.83 ft    right 18.96 ft
#
# An 11 ft travel lane, a 5 ft bike lane and the two 0.82 ft stripes bounding them already come
# to 17.64 ft. Adding even a 2 ft buffer needs 18.82, which fits neither side of
# e_broad_st_east. e_broad_st_west alone could take one, and a protected lane that loses its
# posts at the junction is worse than a consistently conventional pair - so both legs stay
# conventional and this says why. Broad & Greenwood's lanes ARE protected; it has the width.
BIKE_LANE_WIDTH_FT = 5.0   # AASHTO's minimum for an exclusive lane - the floor, and all that fits
E_BROAD_LEGS = ("e_broad_st_east", "e_broad_st_west")


def build_proposal_bike_lanes(baseline: DesignState, model=None) -> DesignState:
    """Conventional bike lanes both sides of both E Broad St legs. Princeton Ave gets none.

    Per side, outward from the centerline: 11 ft travel lane, its edge stripe, a 5 ft bike lane,
    its outer stripe, and whatever asphalt is left hatched to the kerb. On e_broad_st_east that
    leftover is essentially nothing - the cross-section spends 17.64 of the 17.62 ft its narrowest
    kerb offers - so its lane runs hard against the kerb rather than with the hatched margin
    Broad & Greenwood's lanes get. That is the leg's width, not a drawing choice, and it is the
    reason these lanes are unbuffered and unprotected: see the note above.

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
            try:
                state = add_bike_lane(state, leg_name, side, width_ft=BIKE_LANE_WIDTH_FT)
            except ValueError as too_narrow:
                print(f"  NOTE: no bike lane on {leg_name} {side} - {too_narrow}")
    return state
