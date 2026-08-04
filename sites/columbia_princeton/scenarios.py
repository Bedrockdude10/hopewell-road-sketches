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
from src.geometry.targets import LegSide, LegTarget
from src.geometry.treatments import (DesignState, LaneNarrowing, MarkedParking,
    TARGET_LANE_WIDTH_FT, UpgradeCrosswalkMarkings, all_crosswalks_continental,
    apply_osm_parking, complete_centerlines)

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


# Only Princeton Ave is treated here. Columbia Ave measures 26.4 and 26.9 ft curb to curb
# between its traced kerbs, which leaves 2.2-2.4 ft per side beside an 11 ft lane - a strip
# too thin to read as anything, and hatching it drew two slivers of paint down a street that
# does not need calming. Princeton Ave is the leg that does: 30.5 and 31.2 ft, 4.3-4.6 ft of
# spare width per side, and it is the through movement at this junction.
CALMED_LEGS = ("princeton_ave_north", "princeton_ave_south")


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given.

    Traffic calming on Princeton Ave only: its lanes narrowed to TARGET_LANE_WIDTH_FT with
    the recovered width marked according to what OSM says about parking there - crossed
    hatching where it is restricted, marked stalls where it isn't. Columbia Ave is left as it
    is; see CALMED_LEGS for why. The crosswalk upgrade and the centerlines still apply to
    every leg, because those are about the junction, not about one street's cross-section.

    Needs the model for the OSM tags, so it falls back to the untouched baseline when called
    with a state alone (the older single-argument convention).
    """
    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=CALMED_LEGS)
    state = complete_centerlines(state)
    return all_crosswalks_continental(state)

