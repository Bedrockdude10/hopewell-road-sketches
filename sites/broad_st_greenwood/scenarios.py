"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.geometry.targets import LegSide, LegTarget
from src.geometry.treatments import (AddBikeLane, AddBikeLaneBollards, DesignState,
    LaneNarrowing, MarkedParking, ProtectDaylightZone, TARGET_LANE_WIDTH_FT,
    all_crosswalks_continental, apply_osm_parking, bulb_out_corner_pair, complete_centerlines,
    resolved_crossing_stations)

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
# holding their own copy is what src/geometry/treatments.py's own comment on it warns
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
    touches (see src/geometry/model.py:lane_narrowing_taper_ft) until it
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


# --- Curb extensions with mountable aprons -------------------------------------------------
#
# The two Broad St legs only. Greenwood Ave is 26.6 and 31.2 ft curb to curb, so it has 2.3
# and 4.6 ft per side to give beside an 11 ft travel lane and cannot hold a bulb-out at all;
# add_curb_extension refuses one rather than quietly building a shallower thing than this
# docstring describes. That asymmetry is real: a corner treated on the Broad side only still
# shortens the Broad crossing, which is the one 65 ft long.
BULBOUT_EXTENSION_FT = 8.0
BROAD_ST_BULBOUT_LEGS = ("broad_st_east", "broad_st_west")


def build_proposal_apron_bulbouts(baseline: DesignState, model=None) -> DesignState:
    """Curb extensions at all four corners of Broad St, each backed by a mountable apron.

    WHAT IT DOES. Both kerbs of both Broad St legs move 8 ft into the roadway for the length
    of the crossing plus the statutory setback, then taper back. Measured, not asserted - the
    crossing is re-measured against the moved kerb the way both renderers measure it:

        broad_st_east    65.0 ft -> 35.5 ft
        broad_st_west    69.5 ft -> 39.0 ft

    Note which numbers those are. The 52.0 and 55.5 ft in config.yaml are mid-block
    cross-sections; the crossings are painted where the traced kerbs have already flared
    through the corner returns, 39.4 and 31.6 ft off the centerline on broad_st_east against
    a 26.0 ft nominal half-width. So a person crossing Broad St today walks 65 ft of asphalt,
    not 52, and the extension takes nearly 30 ft off that rather than the 16 ft the
    cross-section arithmetic suggests. The pavement polygon loses about 1,090 sq ft.

    IT REMOVES NO PARKING. Schedule I of the borough code prohibits parking 100 ft each way
    on both sides of both Broad St legs. Each extension's whole footprint - the straight face
    plus the taper - is 74 ft, so it occupies kerb that is already legally not-parking. A
    curb extension normally trades spaces for safety; this one does not, and that is the
    strongest thing that can be said for it to a Borough council.
    tests/test_curb_extensions.py pins the footprint against the 100 ft, so the claim cannot
    quietly stop being true.

    THE APRON IS NOT OPTIONAL. Broad St is CR 518, a rural arterial carrying buses and
    trucks. Each extension presents a 15 ft face to a passenger car, and the annulus from
    that out to the corner's OWN measured radius is laid as mountable apron - flush, drivable,
    read as corner rather than carriageway. The four corners here are traced at 29.2, 24.6,
    29.0 and 22.9 ft and each apron is built to its own, read off the baseline fillet rather
    than repeated here as a literal, so re-tracing a kerb in OSM flows straight through. The
    swept path a bus has today is preserved by construction, not by assertion.

    AND IT BUYS BACK KERB. A constructed extension triggers the second clause of
    R.S. 39:4-138(e), which cuts the no-parking setback from 25 ft to 10 ft. The four treated
    kerbs are declared as `curb_extension` daylight devices, and each one's no-parking zone
    shortens by exactly that 15 ft: 63.7 -> 48.7, 63.0 -> 48.0, 66.6 -> 51.6 and 59.2 -> 44.2 ft.
    Sixty feet of kerb across the junction returned to potential parking, on top of the zero it
    costs. Nothing is drawn differently to claim it; the statute applies because the thing it
    names has been built.

    It arrives through the SIDE LINE arm, not the crosswalk arm, and the difference is worth
    knowing before anyone quotes it. The clause reads "within 25 feet of the nearest crosswalk
    OR side line", further wins, and src/geometry/daylighting.py:sideline_station_ft takes the
    corner fillet's tangent point as a deliberately conservative stand-in for the side line -
    34-42 ft out at these corners against a crossing at 21 ft. So the side line binds. That
    makes these zones longer than the statute strictly requires, which is the safe direction to
    err, and it is the same corner-tangent proxy the known-open leg_clearance_ft thread is
    about: tightening it would return more kerb still.

    BROAD ST'S KERBS ARE HATCHED, NOT MARKED FOR PARKING, and that is a correction rather than
    a style choice. apply_osm_parking reads the OSM tags, and neither Broad St leg carries a
    parking:*:restriction at all - so from OSM alone the kerb looks parkable and the first
    version of this proposal duly marked 8 ft stalls starting about 50 ft out. Schedule I
    prohibits parking there for 100 ft. The drawing was asserting something the ordinance
    forbids, in the same proposal whose central claim is that Schedule I already bans parking on
    this kerb - so the ordinance is used as the source it is, and the spare width is hatched the
    way any restricted kerb here is hatched.

    That gap is not this junction's alone: the borough ordinance (Schedules I-IV) is
    under-tagged in OSM at all four sites, and until it is tagged, a scenario that derives
    kerbside parking from OSM tags will over-mark exactly where the ordinance is strictest.
    Greenwood Ave IS tagged (no_parking on three of its four kerbs) so it keeps its OSM-derived
    markings, and the junction still reads as a whole street.
    """
    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=GREENWOOD_LEGS)
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    # Schedule I, not OSM: no parking for 100 ft either way, so the spare asphalt beside an
    # 11 ft lane is hatched rather than marked. Sized per leg from its own measured width.
    for leg_name in BROAD_ST_BULBOUT_LEGS:
        spare_ft = state.legs[leg_name].curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        state = state.apply(LaneNarrowing(LegTarget(leg_name), stripe_width_ft=spare_ft))
    crossing_at = resolved_crossing_stations(model, baseline)
    for leg_name in BROAD_ST_BULBOUT_LEGS:
        state = bulb_out_corner_pair(state, leg_name, extension_ft=BULBOUT_EXTENSION_FT,
                                      crossing_ft=crossing_at[leg_name])
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
BIKE_LANE_WIDTH_FT = 6.0     # AASHTO's 5 ft minimum plus a foot; Broad St can afford it
BIKE_LANE_BUFFER_FT = 3.0    # painted separation from an 11 ft travel lane on an arterial
BIKE_LANE_BOLLARD_SPACING_FT = 8.0  # same flex-post pitch a daylight zone uses - reads as a
                                     # continuous delineator rather than a row of dots


def build_proposal_bike_lanes(baseline: DesignState, model=None) -> DesignState:
    """Buffered bike lanes both sides of both Broad St legs. Greenwood Ave gets none.

    Per side, outward from the centerline: 11 ft travel lane, 3 ft painted buffer with flex-post
    delineators down it, 6 ft bike lane, then the leftover asphalt HATCHED to the kerb. That last
    part matters: a bike lane is a standard width and the street's spare width is not part of it,
    the same accounting an 8 ft parking stall gets when the remainder becomes a hatched kerb
    buffer. Without the outer stripe and that hatching the lane read as running all the way to
    the kerb and looked far wider than the 6 ft it is.

    PROTECTED, not just painted. The delineators stand in the buffer on the TRAFFIC side of the
    lane, which is the side a rider needs protecting from - posts in the kerb-side hatching would
    protect nothing. The 3 ft buffer is what makes that possible, and it is why E Broad's lanes
    (see that site's scenarios.py) cannot be protected: 17.6 ft to its nearest kerb is fully spent
    on an 11 ft lane, a 5 ft lane and their two stripes.

    The cross-section takes 20.8 ft of the 21.3 ft broad_st_east has where its kerbs come closest
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
            state = state.apply(AddBikeLane(LegSide(leg_name, side), width_ft=BIKE_LANE_WIDTH_FT, buffer_ft=BIKE_LANE_BUFFER_FT))
            state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side), spacing_ft=BIKE_LANE_BOLLARD_SPACING_FT))
    return state
