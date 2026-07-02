"""Example treatment scenarios, shared by the Phase 3 plan-view render and the
Phase 4 3D export so both phases show the exact same design."""
from src.treatments import DesignState, bump_out, raise_crossing, refuge_island, upgrade_crosswalk_markings

EXISTING_RADIUS_FT = 20
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
