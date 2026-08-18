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
from src.geometry.treatments import (all_crosswalks_continental, apply_osm_parking,
    complete_centerlines, DesignState, osm_derived_baseline)

# NOTHING IN THIS FILE MAY RESTATE A STANDARD. Lane widths, stall depths, post spacing, which
# kerb the corridor runs along - all of them are the same answer at the next junction, so they
# are imported from src/geometry/treatments/ rather than declared here. A site file is for what
# is true of THIS street: its widths, which legs it treats, what its proposals are called.
#
# Enforced, not merely asked for: tests/test_sites.py:test_no_site_redeclares_what_src_already_defines
# fails the build on a local copy of anything src exports, and
# test_no_rule_is_written_out_in_more_than_one_site fails on a rule copied between two sites.
# Both were written after six constants and four whole functions were found duplicated across
# these files - see README, "A site is not a place to keep a standard".


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given.

    The named proposals were cleared for re-audit, so this is no longer a proposal: it just
    paints each kerb the way OSM says it is used - crossed hatching where parking is
    restricted, marked stalls where it isn't. Every mark here is derived from surveyed data,
    so nothing in it is a design choice waiting to be reviewed.

    Needs the model for the OSM tags, so it falls back to the untouched baseline when called
    with a state alone (the older single-argument convention).
    """
    return osm_derived_baseline(baseline, model)


# --- The borough two-way corridor -----------------------------------------------------------
#
# The same route as the other two Broad St sites - one two-way protected lane along the SOUTH
# kerb for the whole borough length. See sites/broad_st_greenwood/scenarios.py for the
# measurements the side was chosen on.
#
# THIS IS THE CORRIDOR'S PINCH POINT, and the scenario exists to show it rather than to hide it.
# W Broad's north-east approach has 15.13 ft to its nearest traced kerb - the narrowest of the
# twelve Broad St kerbs - and 32.35 ft between kerbs in total. The standard section needs 13.82
# ft of that, which leaves 18.53 ft for traffic: 9.27 ft a lane, under the 10 ft floor. So the
# section fits the KERB here and fails on the TRAVEL LANES, which is a different constraint from
# the one that stops it elsewhere and worth reading in the output.


def build_proposal_two_way_bike_lane(baseline: DesignState, model=None) -> DesignState:
    """The corridor's two-way lane, attempted along W Broad's south kerb.

    Expect refusals. This junction is where the borough-length facility runs out of street, and
    the render's job is to say so precisely - a corridor plan that quietly stops at its hardest
    point is the plan nobody costed.

    Tries the standard 10 ft + 3 ft section first, then the unbuffered fallback, and reports
    what each one costs. An unbuffered two-way lane is not a protected lane: there is nowhere to
    stand a flex post that is not in the bike lane or the travel lane, so it would be paint
    beside 30 mph traffic and the note says as much.
    """
    from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY

    if model is None:
        return baseline
    state = apply_osm_parking(baseline, model, legs=("louellen_st_west",))
    state = complete_centerlines(state)
    state = all_crosswalks_continental(state)
    return BROAD_ST_TWO_WAY_BIKEWAY.apply_to(state, model)
