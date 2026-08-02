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
    DesignState, add_lane_narrowing, add_marked_parking, add_mountable_apron,
    upgrade_crosswalk_markings,
)

TARGET_LANE_WIDTH_FT = 11.0   # NACTO/AASHTO urban minimum travel lane - the width the road diet aims at
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


def build_proposal_1_continental(baseline: DesignState) -> DesignState:
    """Proposal 1 - continental crosswalks only.

    The lowest-cost option: repaint the existing crosswalks from their current parallel-
    line markings to continental (FHWA/NACTO treat this as a visibility upgrade). No curb,
    pavement or lane geometry changes at all, so nothing here needs survey work first.
    """
    return _continental_everywhere(baseline)


def build_proposal_2_continental_parking_narrowing(baseline: DesignState) -> DesignState:
    """Proposal 2 - continental crosswalks + 11 ft lanes + marked parking.

    Proposal 1, plus a paint-only road diet: travel lanes narrowed to 11 ft and the
    recovered width marked as parallel parking (see _parking_and_narrowing for how legs
    too narrow for a stall are handled). Still entirely paint - no curb is moved - so it
    stays reversible and cheap, but it narrows the visual carriageway, which is the part
    that actually slows traffic.
    """
    return _parking_and_narrowing(_continental_everywhere(baseline))


def build_proposal_3_continental_parking_narrowing_bulbouts(baseline: DesignState) -> DesignState:
    """Proposal 3 - Proposal 2 + mountable pedestrian bulb-outs at every corner.

    add_mountable_apron builds a textured curb extension that is FLUSH WITH GRADE rather
    than a poured kerb, so a fire apparatus or a turning truck can simply drive over it
    while it still visually narrows the corner and shortens the pedestrian crossing. That
    was the explicit requirement here: bulb-outs emergency vehicles can mount.

    It's the most substantial of the three, but still reversible - no drainage work, no
    kerb reconstruction. Corners whose fillet failed to build are skipped by
    add_mountable_apron's consumers (src/render/export.py, plan_view.py).
    """
    state = build_proposal_2_continental_parking_narrowing(baseline)
    for corner in list(state.corner_fillets):
        if "error" in state.corner_fillets[corner]:
            continue
        state = add_mountable_apron(state, corner)
    return state


def build_demo_scenario(baseline: DesignState) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given: Proposal 3, the
    full set. The individual proposals are available by name via --scenario."""
    return build_proposal_3_continental_parking_narrowing_bulbouts(baseline)
