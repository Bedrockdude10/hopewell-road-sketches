"""Raised versus lowered kerbs, and the openings a lowered one puts in the markings.

Every barrier=kerb way mapped at these four junctions is tagged raised or lowered, and the
distinction was reaching the geometry and going no further - the plan view drew one black line
for all of them, the 3D render drew none, and the kerbside paint ran unbroken past every
driveway. These pin the tags reaching the markings, because "the surveyor mapped it and the
render ignored it" is the failure this project exists to prevent and it is silent by nature.
"""
import contextlib
import io

import numpy as np

from src.geometry.kerbs import KerbType, opens_the_kerb
from src.geometry.markings import BIKE_LANE_SURFACE
from src.geometry.model import station_offset_many
from src.geometry.treatments import DesignState
from src.site import load_site_scenarios, run_scenario

from tests.conftest import needs_source_data
from tests.test_sites import resolved_scene, scene_props


def test_a_kerb_with_no_tag_is_unknown_and_not_assumed_raised():
    """"Nobody said" is not "raised", and a raised kerb is the one thing it must not become -
    that would silently turn an unmapped kerb into a claim that a vehicle cannot cross it."""
    assert KerbType.from_tags({"barrier": "kerb"}) is KerbType.UNKNOWN
    assert KerbType.from_tags({}) is KerbType.UNKNOWN
    assert KerbType.from_tags({"kerb": "something_new"}) is KerbType.UNKNOWN
    assert KerbType.from_tags({"kerb": "raised"}) is KerbType.RAISED
    assert KerbType.from_tags({"kerb": "lowered"}) is KerbType.LOWERED


def test_a_pedestrian_ramp_does_not_open_the_paint():
    """A crossing's kerb ramp is lowered for a wheelchair, not for a car.

    Both are kerb=lowered, so tactile_paving is what separates them. Opening the paint at a
    ramp would put a gap in a bike lane at the crosswalk, where the crossing band already cuts
    it - a double break, and a claim that a vehicle crosses there.
    """
    assert opens_the_kerb({"kerb": "lowered"})
    assert opens_the_kerb({"kerb": "lowered", "tactile_paving": "no"})
    assert opens_the_kerb({"kerb": "flush"})
    assert not opens_the_kerb({"kerb": "lowered", "tactile_paving": "yes"})
    assert not opens_the_kerb({"kerb": "raised"})
    assert not opens_the_kerb({"barrier": "kerb"})


@needs_source_data
def test_the_markings_break_over_a_dropped_kerb(site_models):
    """E Broad's driveway, which is the one this was written for.

    e_broad_st_east's left kerb is tagged kerb=lowered over stations 59-96 (OSM way 1546804848),
    and driveway way 772378207 runs up to it. Before this, the bike lane proposal painted both
    edge lines and the green surface straight across the driveway mouth, stations 0-130 unbroken.
    """
    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios("ebroad_princeton").build_proposal_bike_lanes,
                             DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    openings = state.kerb_openings[("e_broad_st_east", "left")]
    assert openings, "the dropped kerb on this leg was not seeded onto the design"
    opening = openings[0]
    assert opening.kerb is KerbType.LOWERED
    assert opening.way_id is not None, "an opening has to name the kerb way that caused it"

    leg = state.legs["e_broad_st_east"]
    green = [p for p in paint if p.kind is BIKE_LANE_SURFACE
             and p.leg == "e_broad_st_east" and p.side == "left"]
    assert len(green) >= 2, (
        f"the green surface is in {len(green)} piece(s) - it should be broken by the "
        f"{opening.length_ft:.0f} ft dropped kerb at {opening.start_ft:.0f}-{opening.end_ft:.0f} ft")
    # Nothing may be painted inside the opening. Checked on the pieces' own stations rather than
    # by counting them, because two pieces could still both overlap it.
    for piece in green:
        stations, _offsets = station_offset_many(
            leg.centerline, np.asarray(piece.geometry.exterior.coords, dtype=float))
        inside = (stations > opening.start_ft + 1.0) & (stations < opening.end_ft - 1.0)
        assert not inside.any(), (
            f"green surface still painted inside the {opening.start_ft:.0f}-"
            f"{opening.end_ft:.0f} ft driveway opening")


@needs_source_data
def test_every_opening_names_the_kerb_that_caused_it(site_models):
    """A gap in a marking is a claim about the street, so it has to be auditable against OSM.

    Same reasoning as ParkingRestriction.citation: "the paint stops at 59 ft" is unreviewable,
    "OSM kerb=lowered on way 1546804848" can be checked by someone who is not reading this code.
    """
    from src.geometry.kerbs import describe_kerb_openings

    found = 0
    for site, model in site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            state = DesignState.from_model(model)
        for (leg_name, side), openings in state.kerb_openings.items():
            assert leg_name in state.legs, f"{site}: opening on a leg that does not exist"
            assert side in ("left", "right")
            for opening in openings:
                found += 1
                assert opening.end_ft > opening.start_ft
                assert "kerb=" in opening.citation and str(opening.way_id) in opening.citation
        for line in describe_kerb_openings(state):
            assert "OSM kerb=" in line, f"{site}: an opening reported without its source"
    assert found, "no dropped kerbs found at any site - the tags are not being read"
