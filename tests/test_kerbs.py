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
import pytest

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


def test_only_the_surveyors_driveway_tagging_opens_the_paint():
    """A driveway is tagged wheelchair=no AND tactile_paving=no; a ramp is yes and yes.

    That is the surveyor's own convention, and reading it as a POSITIVE test rather than as
    "lowered and not obviously a ramp" is what makes the two cases below come out right. Both
    were wrong under the looser rule:

      * a bare kerb=lowered is a kerb recorded as dropped without saying what for, and breaking
        a bike lane over it invents a driveway. One exists in the borough (way 1546755075);
      * tactile paving present with wheelchair=no is two tags disagreeing, and the safe reading
        of a disagreement is the one that does NOT put a gap in a marking.

    wheelchair=no is not sufficient on its own either - all 67 raised kerbs here carry it.
    """
    assert opens_the_kerb({"kerb": "lowered", "tactile_paving": "no", "wheelchair": "no"})
    assert opens_the_kerb({"kerb": "flush", "tactile_paving": "no", "wheelchair": "no"})
    # A pedestrian ramp: dropped for a wheelchair, not for a car. The crossing band already cuts
    # the markings there, so opening them again would be a double break.
    assert not opens_the_kerb({"kerb": "lowered", "tactile_paving": "yes", "wheelchair": "yes"})
    # Dropped, but nobody said what for.
    assert not opens_the_kerb({"kerb": "lowered"})
    assert not opens_the_kerb({"kerb": "lowered", "tactile_paving": "no"})
    # Contradictory: tactile paving means a pedestrian facility whatever else is on the way.
    assert not opens_the_kerb({"kerb": "lowered", "tactile_paving": "yes", "wheelchair": "no"})
    # Not dropped at all - and this is the combination that shows wheelchair=no is not the signal.
    assert not opens_the_kerb({"kerb": "raised", "tactile_paving": "no", "wheelchair": "no"})
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


@needs_source_data
def test_the_opening_is_trimmed_back_and_rounded(site_models):
    """A driveway gap is slightly wider than the dropped kerb, with rounded corners.

    A square-ended gap exactly the width of the kerb tag both reads as punched out of the
    markings and gives a turning vehicle nothing to aim at. The trim is deliberately small -
    OPENING_TRIM_FT, a foot and a half - because every foot of it is bike lane or hatched buffer
    given up, and pedestrians and cyclists have priority here. This pins that it is applied AND
    that it stays small, so "a little more room" cannot quietly become a swept-path design.
    """
    from src.geometry.paint import OPENING_TRIM_FT

    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios("ebroad_princeton").build_proposal_bike_lanes,
                             DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    assert 0 < OPENING_TRIM_FT <= 3.0, (
        f"the opening trim is {OPENING_TRIM_FT} ft - past about 3 ft this stops being a rounded "
        f"edge for cohesion and starts being a design decision about vehicle turning radii, "
        f"which is not what it was asked to be")

    leg = state.legs["e_broad_st_east"]
    opening = state.kerb_openings[("e_broad_st_east", "left")][0]
    green = sorted((p for p in paint if p.kind is BIKE_LANE_SURFACE
                    and p.leg == "e_broad_st_east" and p.side == "left"),
                   key=lambda p: p.geometry.centroid.distance(
                       leg.centerline.interpolate(0.0)))
    ends = []
    for piece in green:
        stations, _offsets = station_offset_many(
            leg.centerline, np.asarray(piece.geometry.exterior.coords, dtype=float))
        ends.append((float(stations.min()), float(stations.max())))
    before = max(hi for lo, hi in ends if hi <= opening.start_ft + 0.5)
    after = min(lo for lo, hi in ends if lo >= opening.end_ft - 0.5)
    assert before < opening.start_ft, (
        f"the paint stops at {before:.1f} ft, not trimmed back from the {opening.start_ft:.1f} ft "
        f"where the dropped kerb starts")
    assert after > opening.end_ft, (
        f"the paint resumes at {after:.1f} ft, not trimmed back from the {opening.end_ft:.1f} ft "
        f"where the dropped kerb ends")
    # Trimmed by about the stated amount at each end, not by an arbitrary margin.
    assert opening.start_ft - before == pytest.approx(OPENING_TRIM_FT, abs=0.6)
    assert after - opening.end_ft == pytest.approx(OPENING_TRIM_FT, abs=0.6)


@needs_source_data
@pytest.mark.parametrize("site", ["ebroad_princeton", "broad_st_greenwood"])
def test_no_flex_post_stands_in_a_driveway(site, site_models):
    """An opening gaps the BOLLARDS as well as the paint.

    PaintContext.emit deliberately skips clipping, because a post is a point and cannot be
    trimmed the way a stripe can - so the posts marched straight across every driveway while the
    paint broke around it. That is worse than not breaking the paint at all: it draws a protected
    lane whose protection you are expected to drive through. Seven of E Broad's 26 posts sat in a
    driveway before this.

    Checked on BOTH lists, because they come from different places and have disagreed before:
    the painted markers the plan view draws, and the props the 3D render builds.
    """
    from shapely.geometry import Point

    from src.geometry.markings import BOLLARD
    from src.geometry.paint import kerb_opening_bands

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios(site).build_proposal_bike_lanes,
                             DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint, props = scene.build_paint_and_posts(scene_props(model, state, scene))

    bands = kerb_opening_bands(state)
    if bands is None:
        pytest.skip(f"{site} has no dropped kerbs along its legs")
    posted = [p for p in paint if p.kind is BOLLARD]
    assert posted, "the proposal draws no posts at all, so this proves nothing"
    standing = [p for p in posted if bands.intersects(p.geometry)]
    assert not standing, f"{len(standing)} painted post(s) stand inside a driveway opening"
    prop_posts = [p for p in props if p.get("type") == "bollard"]
    in_a_drive = [p for p in prop_posts if bands.intersects(Point(p["position_ft"]))]
    assert not in_a_drive, f"{len(in_a_drive)} post prop(s) stand inside a driveway opening"
