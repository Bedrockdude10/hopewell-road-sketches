"""Treatment proposals for the elementary school junction.

Traffic calming on Princeton Ave (CR 569) only, sharing the crosswalk upgrade every proposal
in this project starts from. Existing conditions come from OSM and from Danny's 2026-08-18
kerb trace; see this site's config.yaml. NO width here is field-measured, and the two East
Prospect St legs are not even osm_derived, so treat what follows as a design study rather
than a construction drawing.

WHY THE PROPOSAL IS THIS SMALL. Princeton Ave is a designated state truck route past an
elementary school (hgv=designated - see config.yaml) with no on-street parking anywhere on
it, so the two levers this project uses elsewhere are both unavailable: the travel lanes
cannot go below the 11 ft floor, and there is no parking lane to reclaim. What is left is
the width beside an 11 ft lane, which at 30.5 ft curb-to-curb is 4.25 ft per side.
"""
from src.geometry.treatments import DesignState, osm_derived_baseline

# NOTHING IN THIS FILE MAY RESTATE A STANDARD - see the same note in the other site files, and
# tests/test_sites.py:test_no_site_redeclares_what_src_already_defines, which enforces it.

# Only Princeton Ave is treated. The East Prospect St legs are drawn 30 and 35 ft long,
# because that is as far as the kerb is traced (config.yaml), and a treatment applied over
# that distance is a corner return with paint on it, not a cross-section. Princeton Ave is
# also the through movement here and the street the school fronts.
CALMED_LEGS = ("princeton_ave_north", "princeton_ave_south")


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    """Default scenario for phase3/phase4 when no --scenario is given.

    Princeton Ave narrowed to TARGET_LANE_WIDTH_FT with the recovered width marked according
    to what OSM says about parking there - which on this street is `no_parking` for its whole
    length, so the recovered width hatches rather than becoming stalls. The crosswalk upgrade
    and the centerlines apply to every leg, because those are about the junction rather than
    about one street's cross-section.

    Needs the model for the OSM tags, so it falls back to the untouched baseline when called
    with a state alone (the older single-argument convention).
    """
    return osm_derived_baseline(baseline, model, legs=CALMED_LEGS)
