"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.geometry.treatments import (
    DesignState, add_bollards, add_lane_narrowing, add_marked_parking, add_mountable_apron,
    add_parking_buffer_bollards, bump_out, raise_crossing, refuge_island, upgrade_crosswalk_markings,
)

TIGHTENED_RADIUS_FT = 10


def build_demo_scenario(baseline: DesignState) -> DesignState:
    """Tighten the two corners touching the confirmed West Broad St leg, add a
    refuge island mid-crossing on West Broad St, raise the Greenwood Ave (south)
    crossing, and upgrade the remaining painted crosswalks from the existing
    "lines" marking to continental (a real, low-cost visibility treatment on
    its own - the raised crossing already replaces greenwood_ave_south's)."""
    state = baseline
    for corner in list(state.corner_fillets):
        leg_a, leg_b = corner
        if "broad_st_west" in (leg_a, leg_b):
            state = bump_out(state, corner, TIGHTENED_RADIUS_FT)

    state = refuge_island(state, "broad_st_west", offset_ft=35, width_ft=8, along_road_ft=22)
    state = raise_crossing(state, "greenwood_ave_south", crossing_width_ft=10)
    for leg_name in ("broad_st_west", "broad_st_east", "greenwood_ave_north"):
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
    return state


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


def build_proposal_a_crosswalks_and_parking(baseline: DesignState) -> DesignState:
    """Proposal A - continental crosswalks + marked parking: the lowest-cost
    option in this series. Two independent, real, low-cost changes stacked
    together: (1) repaint all four approaches' crosswalks from the existing
    "lines" marking to continental (a real FHWA/NACTO visibility upgrade on
    its own), and (2) mark real curbside parallel parking along BOTH sides of
    both Broad St legs (add_marked_parking, hugging the curb - curb_offset_ft=0,
    the default) in what's otherwise unused shoulder width per the NJDOT
    SLD's own "Shoulder" column for this route, up to the real legal minimum
    distance from each crosswalk (LEGAL_PARKING_SETBACK_FT). No curb/
    pavement geometry change anywhere."""
    state = baseline
    for leg_name in state.legs:
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
    state = _add_broad_st_both_side_parking(state)
    return state


def build_proposal_b_parking_with_striped_buffer(baseline: DesignState) -> DesignState:
    """Proposal B - Proposal A + striped parking buffer: everything in
    Proposal A, but the marked parking lanes (both sides) are pulled
    curb_offset_ft off the curb (add_marked_parking's curb_offset_ft) so a
    striped no-parking buffer - the same chevron paint treatment as a
    lane-narrowing buffer - sits between each parking lane and the curb
    instead of between the parking lane and the travel lane. Net effect:
    parking now sits directly against the active travel lane on both sides,
    with the striped buffer (and whatever it's protecting against - a fire
    hydrant, a sight line, drainage) on the curb side instead."""
    state = baseline
    for leg_name in state.legs:
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
    state = _add_broad_st_both_side_parking(state, curb_offset_ft=PARKING_BUFFER_DEFAULT_FT)
    return state


def build_proposal_c_parking_buffer_bollards(baseline: DesignState) -> DesignState:
    """Proposal C - Proposal B + bollards on the outside of parking:
    everything in Proposal B, plus plastic flex-post bollards
    (add_parking_buffer_bollards) centered in the striped buffer between each
    parking lane and the curb - i.e. on the curb side of parking, not the
    travel-lane side add_bollards uses for a lane-narrowing buffer. Applied
    to both sides of both Broad St legs. The most protected (but still fully
    paint-and-delineator, no curb/pavement change) option in this series."""
    state = build_proposal_b_parking_with_striped_buffer(baseline)
    for leg_name in BROAD_ST_LEGS:
        for side in PARKING_SIDES:
            state = add_parking_buffer_bollards(state, leg_name, side=side)
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


def build_proposal_c_broad_st_paint_only(baseline: DesignState) -> DesignState:
    """Broad St road diet, Proposal C - paint only: two real 11 ft travel
    lanes (one each direction) on West and East Broad St, striped paint
    filling the gap between the outside of each lane and the existing curb.
    Zero curb/pavement geometry change, fully reversible - the lowest-cost
    option in this series."""
    return _narrow_broad_st_to_11ft_lanes(baseline)


def build_proposal_h_broad_st_line_only(baseline: DesignState) -> DesignState:
    """Broad St road diet, Proposal H - line only (debug/comparison): the
    same real 11 ft lane geometry as Proposal C, but with NO chevron fill in
    the buffer - just the solid line (straight run + corner taper) marking
    the outside edge of the travel lane. Not primarily a PBSAC option; this
    exists so the exact same lane-width geometry can be checked, uncluttered
    by hatch density, directly against the Phase 3 plan view (which now
    renders this same edge/taper line - see src/render/plan_view.py) instead
    of trying to eyeball it off the 3D render alone."""
    return _narrow_broad_st_to_11ft_lanes(baseline, line_only=True)


def build_proposal_i_broad_st_marked_parking(baseline: DesignState) -> DesignState:
    """Broad St road diet, Proposal I - marked curbside parking: instead of
    a paint-only no-parking buffer (Proposals C/D/H), mark the same real
    shoulder width on BOTH sides of both Broad St legs as actual parallel
    parking lanes (add_marked_parking) - a lane-edge line plus perpendicular
    stall dividers, AASHTO/NACTO default 8 ft depth / 22 ft stalls, up to the
    real legal minimum distance from each crosswalk (LEGAL_PARKING_SETBACK_FT).
    Distinct from (and independent of) add_lane_narrowing - this doesn't
    narrow the travel lane itself, it designates real curbside parking in
    what's otherwise unused shoulder width, per the NJDOT SLD's own
    "Shoulder" column for this route."""
    return _add_broad_st_both_side_parking(baseline)


def build_proposal_d_broad_st_paint_and_bollards(baseline: DesignState) -> DesignState:
    """Broad St road diet, Proposal D - paint + bollards: the same 11 ft
    paint-only lane narrowing as Proposal C, escalated with a line of plastic
    flex-post bollards down the center of each painted buffer. Still no
    curb/pavement change - the bollards are a physically firmer (but still
    mountable/replaceable, not poured) edge than paint alone."""
    state = _narrow_broad_st_to_11ft_lanes(baseline)
    for leg_name in BROAD_ST_LEGS:
        state = add_bollards(state, leg_name)
    return state


def build_proposal_f_continental_crosswalks_only(baseline: DesignState) -> DesignState:
    """Broad St / Greenwood Ave, Proposal F - continental crosswalks only:
    every other real-world condition stays exactly as surveyed (no lane
    narrowing, no curb/pavement change, no signage change) - the ONLY
    difference from Existing Conditions is repainting all four approaches'
    crosswalks from the existing "lines" marking to continental. Isolates
    that one change so it can be evaluated (and rendered) independent of
    the Broad St road-diet series below."""
    state = baseline
    for leg_name in state.legs:
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
    return state


def build_proposal_g_continental_crosswalks_and_broad_st_paint(baseline: DesignState) -> DesignState:
    """Broad St / Greenwood Ave, Proposal G - continental crosswalks + Broad
    St paint-only road diet: stacks build_proposal_f_continental_crosswalks_only's
    crosswalk upgrade (all four legs) on top of
    build_proposal_c_broad_st_paint_only's 11 ft paint-only lane narrowing
    (Broad St West/East only) - the two lowest-cost treatments in this
    project combined into one buildable proposal."""
    state = _narrow_broad_st_to_11ft_lanes(baseline)
    for leg_name in state.legs:
        state = upgrade_crosswalk_markings(state, leg_name, "continental")
    return state


def build_proposal_e_broad_st_mountable_bulbouts(baseline: DesignState) -> DesignState:
    """Broad St road diet, Proposal E - mountable bulb-outs: a textured,
    flush-with-grade curb extension (add_mountable_apron) at all four
    corners. Visually and physically narrows each corner for pedestrians
    (shorter crossing distance) while remaining fully drivable - e.g. by a
    fire apparatus's rear wheels - since no elevation or hard curb change is
    introduced. The most substantial treatment in this series, but still
    reversible/mountable rather than a permanent poured bump-out."""
    state = baseline
    for corner in list(state.corner_fillets):
        state = add_mountable_apron(state, corner)
    return state
