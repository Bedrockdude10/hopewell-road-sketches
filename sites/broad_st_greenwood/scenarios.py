"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.coords import FT_TO_M
from src.treatments import (
    DesignState, add_corner_hatching, add_extra_prop, add_lane_narrowing, add_mountable_apron, bump_out,
    find_corner, raise_crossing, refuge_island, shift_crosswalk_offset, upgrade_crosswalk_markings,
)

EXISTING_RADIUS_FT = 20  # matches sites/broad_st_greenwood/config.yaml treatments.existing_corner_radius_ft
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


def build_proposal_a_paint_only(baseline: DesignState) -> DesignState:
    """PBSAC Proposal A - Paint-Only Narrowing: the lowest-cost, fully
    reversible option. No curb/pavement geometry changes anywhere (this
    function never calls bump_out or otherwise touches corner_fillets) - all
    calming and crossing improvements are paint, signage, and delineators
    only, so fire apparatus turning paths at the two Columbia-route corners
    (SW/SE) are unaffected by construction, not just by careful design."""
    state = baseline
    for corner in list(state.corner_fillets):  # all 4 corners, incl. the 2 fire-route corners
        state = add_corner_hatching(state, corner)

    for leg_name in ("greenwood_ave_north", "greenwood_ave_south"):
        state = add_lane_narrowing(state, leg_name)
        state = upgrade_crosswalk_markings(state, leg_name, "continental")

    # offset_ft omitted deliberately - both belong right at the real (OSM-surveyed)
    # crosswalk, not a guessed distance (see add_extra_prop's docstring for why a
    # hardcoded number here previously landed these signs inside the curb-return
    # curve, in the roadway, instead of on the sidewalk).
    state = add_extra_prop(state, "greenwood_ave_south", "rrfb",
                            note="Pedestrian-initiated RRFB, lower priority per PBSAC report.")
    state = add_extra_prop(state, "greenwood_ave_south", "school_zone_sign", side="left",
                            note="Relocated/duplicated so the existing school zone sign (currently on Broad "
                                 "St West) is also visible from the Columbia approach.")
    return state


BULB_OUT_RADIUS_REDUCTION_M = 1.5  # PBSAC Proposal B spec
APRON_EXTENT_M = 1.5
CROSSWALK_SETBACK_INCREASE_M = 0.5


def build_proposal_b_mountable_apron_hybrid(baseline: DesignState) -> DesignState:
    """PBSAC Proposal B - Mountable Apron Hybrid: real hard curb extensions
    (bulb-outs, via bump_out()) at the NW/NE corners, which are NOT on the
    fire apparatus route from the Columbia Ave station; a flush, texturally
    distinct mountable apron (no elevation/curb change, so a fire engine's
    rear wheels can still track over it) at the SW/SE Columbia-route corners
    instead. Corner-anchored props (traffic signals, streetlights, stop
    signs) need no manual repositioning here - they're always recomputed from
    state.corner_fillets, so bump_out()'ing NW/NE automatically moves them to
    the new fillet arc midpoint.

    NOTE: verifying the NW/NE bulb-out radius and the SW/SE apron extent
    against a real WB-40/NJ-pumper turning template is a human engineering
    step this pipeline doesn't perform - see README.md "Known gaps" (this
    project explicitly does not simulate vehicle turning paths)."""
    state = baseline
    reduced_radius_ft = EXISTING_RADIUS_FT - BULB_OUT_RADIUS_REDUCTION_M / FT_TO_M
    state = bump_out(state, find_corner(state, "broad_st_west", "greenwood_ave_north"), reduced_radius_ft)
    state = bump_out(state, find_corner(state, "broad_st_east", "greenwood_ave_north"), reduced_radius_ft)

    apron_extent_ft = APRON_EXTENT_M / FT_TO_M
    state = add_mountable_apron(state, find_corner(state, "broad_st_west", "greenwood_ave_south"), apron_extent_ft)
    state = add_mountable_apron(state, find_corner(state, "broad_st_east", "greenwood_ave_south"), apron_extent_ft)

    state = shift_crosswalk_offset(state, "greenwood_ave_south", CROSSWALK_SETBACK_INCREASE_M / FT_TO_M)
    for leg_name in ("greenwood_ave_north", "greenwood_ave_south"):
        state = upgrade_crosswalk_markings(state, leg_name, "continental")

    state = add_extra_prop(state, "greenwood_ave_south", "rrfb",
                            note="Pedestrian-initiated RRFB.")
    state = add_extra_prop(state, "greenwood_ave_south", "school_zone_sign", side="left",
                            note="Relocated/duplicated for visibility from the Columbia approach.")
    return state
