"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.geometry.treatments import (
    protect_daylight_zone, all_crosswalks_continental, complete_centerlines, DesignState, add_lane_narrowing, add_marked_parking, apply_osm_parking,
)

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
    for leg_name, side in sorted(state.parking_zones):
        state = protect_daylight_zone(state, leg_name, side, kind=kind)
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
            state = add_marked_parking(state, leg_name, side=side, curb_offset_ft=curb_offset_ft)
    return state


# --- Broad St road-diet series: three escalating treatments for the two
# confirmed, over-wide Broad St legs (55.5/68 ft curb-to-curb vs. two travel
# lanes' worth of actual need), independent of the Greenwood-focused PBSAC
# proposals above. Each is a distinct scenario, not a stack, so they can be
# compared side by side.
BROAD_ST_LEGS = ("broad_st_west", "broad_st_east")
TARGET_LANE_WIDTH_FT = 11  # NACTO/AASHTO urban minor-arterial minimum travel lane width


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
        state = add_lane_narrowing(state, leg_name, stripe_width_ft, line_only=line_only)
    return state
