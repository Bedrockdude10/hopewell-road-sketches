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
from src.geometry.treatments import (DesignState, PRINCETON_AVE_CALMING)

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


# WHICH LEGS ARE CALMED IS A ROUTE DECISION, so it is PRINCETON_AVE_CALMING's and not this
# file's - the same two leg names were written out here and at princeton_eprospect, read off by
# eye from the street name the config already states.
#
# WHAT IS THIS JUNCTION'S OWN, and the reason Columbia Ave is not calmed as well: Columbia
# measures 26.4 and 26.9 ft curb to curb between its traced kerbs, which leaves 2.2-2.4 ft per
# side beside an 11 ft lane - a strip too thin to read as anything, and hatching it drew two
# slivers of paint down a street that does not need calming. Princeton Ave is the leg that does:
# 30.5 and 31.2 ft, 4.3-4.6 ft of spare width per side, and it is the through movement here.


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given.

    The Princeton Ave route's calming, applied at this junction: its lanes narrowed to
    TARGET_LANE_WIDTH_FT with the recovered width marked according to what OSM says about
    parking there - crossed hatching where it is restricted, marked stalls where it isn't.
    Columbia Ave keeps its cross-section; see the note above for the width that decides that.
    The crosswalk upgrade and the centerlines still apply to every leg, because those are about
    the junction, not about one street's cross-section.

    Needs the model for the OSM tags, so it falls back to the untouched baseline when called
    with a state alone (the older single-argument convention).
    """
    return PRINCETON_AVE_CALMING.apply_to(baseline, model)

