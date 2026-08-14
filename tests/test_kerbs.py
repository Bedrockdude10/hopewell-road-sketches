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
from src.geometry.model import curb_offsets_at_stations, station_offset_many
from src.geometry.treatments import DesignState
from src.site import load_site_scenarios, run_scenario

from tests.conftest import needs_source_data
from tests.test_sites import resolved_scene, scene_props
import itertools


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

    from src.geometry.kerbs import OpeningSource

    openings = state.kerb_openings[("e_broad_st_east", "left")]
    assert openings, "the dropped kerb on this leg was not seeded onto the design"
    # By SOURCE, not by index: this leg's mouth is now found by both signals - the tagged kerb and
    # the driveway way that runs up to it - and this test is about the surveyed one.
    kerbs = [o for o in openings if o.source is OpeningSource.DROPPED_KERB]
    assert kerbs, "the tagged dropped kerb produced no opening"
    opening = kerbs[0]
    assert opening.kerb is KerbType.LOWERED
    assert opening.way_id is not None, "an opening has to name the kerb way that caused it"

    from src.geometry.paint import DOTTED_GAP_FT, DOTTED_MARK_FT

    leg = state.legs["e_broad_st_east"]
    green = [p for p in paint if p.kind is BIKE_LANE_SURFACE
             and p.leg == "e_broad_st_east" and p.side == "left"]
    assert len(green) >= 2, (
        f"the green surface is in {len(green)} piece(s) - it should be broken by the "
        f"{opening.length_ft:.0f} ft dropped kerb at {opening.start_ft:.0f}-{opening.end_ft:.0f} ft")

    # The green BREAKS over the opening - into marks, not into nothing. This asserted that no green
    # at all survived inside, which was the behaviour until Danny asked for the surface to carry the
    # dotted extension too: a lane is crossed at a driveway, not ended, and the green saying so
    # matters more than the gap did. What must not survive is a continuous slab.
    runs = []
    for piece in green:
        stations, _offsets = station_offset_many(
            leg.centerline, np.asarray(piece.geometry.exterior.coords, dtype=float))
        lo, hi = max(float(stations.min()), opening.start_ft), min(float(stations.max()),
                                                                   opening.end_ft)
        if hi - lo > 0.05:
            runs.append((lo, hi))
    assert runs, (
        f"no green at all inside the {opening.start_ft:.0f}-{opening.end_ft:.0f} ft opening - the "
        f"lane stops at the driveway instead of being dotted across it")
    for lo, hi in runs:
        assert hi - lo <= DOTTED_MARK_FT + 0.5, (
            f"a {hi - lo:.1f} ft run of green inside the opening, against a {DOTTED_MARK_FT} ft "
            f"mark - that is a slab painted through the driveway, not a dotted extension")
    runs.sort()
    for (_lo, hi), (next_lo, _next_hi) in itertools.pairwise(runs):
        assert next_lo - hi >= DOTTED_GAP_FT - 0.5, (
            f"only {next_lo - hi:.1f} ft between two marks, against a {DOTTED_GAP_FT} ft gap")


@needs_source_data
def test_every_opening_names_the_osm_object_that_caused_it(site_models):
    """A gap in a marking is a claim about the street, so it has to be auditable against OSM.

    Same reasoning as ParkingRestriction.citation: "the paint stops at 59 ft" is unreviewable,
    "OSM kerb=lowered on way 1546804848" can be checked by someone who is not reading this code.

    Both sources have to say which they are, because they are not equally good evidence: a dropped
    kerb's extent is surveyed, a driveway's mouth is assumed. An opening that cited neither, or
    that cited a driveway as though its width were measured, would be the sort of quiet
    over-claiming this project's provenance strings exist to prevent.
    """
    from src.geometry.kerbs import OpeningSource, describe_kerb_openings

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
                assert str(opening.way_id) in opening.citation
                if opening.source is OpeningSource.DROPPED_KERB:
                    assert "kerb=" in opening.citation
                    assert opening.is_surveyed_width
                elif opening.source is OpeningSource.DRIVEWAY:
                    assert "service=driveway" in opening.citation
                    assert not opening.is_surveyed_width, (
                        "a driveway centreline carries no width, so its mouth must not be "
                        "reported as surveyed")
                else:
                    # A cross street opens the kerb over its own carriageway, which OSM records
                    # for almost none of them - so the mouth is this repo's assumption about a
                    # highway class, exactly as a driveway's is, and must not claim otherwise.
                    assert opening.source is OpeningSource.CROSS_STREET
                    assert "intersecting street" in opening.citation
                    assert not opening.is_surveyed_width
        for line in describe_kerb_openings(state):
            assert "OSM " in line, f"{site}: an opening reported without its source"
    assert found, "no openings found at any site - neither signal is being read"


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

    from src.geometry.kerbs import OpeningSource

    leg = state.legs["e_broad_st_east"]
    opening = next(o for o in state.kerb_openings[("e_broad_st_east", "left")]
                   if o.source is OpeningSource.DROPPED_KERB)
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

    openings = kerb_opening_bands(state)
    if not openings:
        pytest.skip(f"{site} has no dropped kerbs along its legs")
    # Against the ENTRANCE (KerbOpenings.driven), not the hatching's run-out: a post is in the way
    # if it stands where a car crosses, and the extra few feet the hatching gives up to taper off
    # is paint ending gracefully rather than roadway anything drives on.
    posted = [p for p in paint if p.kind is BOLLARD]
    assert posted, "the proposal draws no posts at all, so this proves nothing"
    standing = [p for p in posted if openings.driven.intersects(p.geometry)]
    assert not standing, f"{len(standing)} painted post(s) stand inside a driveway opening"
    prop_posts = [p for p in props if p.get("type") == "bollard"]
    in_a_drive = [p for p in prop_posts if openings.driven.intersects(Point(p["position_ft"]))]
    assert not in_a_drive, f"{len(in_a_drive)} post prop(s) stand inside a driveway opening"


@needs_source_data
def test_a_lane_line_goes_dotted_across_a_driveway_rather_than_stopping(site_models):
    """A driveway does not end a bike lane, so its lines do not end at one either.

    The gap on its own said the lane stops here and starts again 37 ft later, which is not what a
    driveway does to a lane - and it was defended in the code as "the honest version of the paint
    not continuing", which was really an admission that the dotted extension had not been built.
    MUTCD's dotted lane extension is what a striper paints through a conflict area.

    Pinned per dash rather than in aggregate: the dashes have to lie ON the line they extend (a row
    of marks a foot off the lane edge is worse than a gap), inside the entrance, and be the stated
    length - not one long stripe straight across, which is the failure mode that would look right
    at 2D scale and paint over the driveway in 3D.
    """
    from src.geometry.markings import BIKE_LANE_DOTTED_EXTENSION, BIKE_LANE_EDGE_LINE
    from src.geometry.paint import DOTTED_MARK_FT, kerb_opening_bands

    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios("ebroad_princeton").build_proposal_bike_lanes,
                             DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    openings = kerb_opening_bands(state)
    dashes = [p for p in paint if p.kind is BIKE_LANE_DOTTED_EXTENSION]
    assert dashes, ("no dotted extension anywhere, so every bike lane still just stops at its "
                    "driveways")

    def offset_ft(piece):
        """How far off its leg's centerline a piece lies, which is what says it is ON a lane line.
        Not distance to the surviving stripe: a dash is centred in the gap, so it is legitimately
        several feet from the nearest piece of the line it continues."""
        _stations, offsets = station_offset_many(
            state.legs[piece.leg].centerline,
            np.asarray(piece.geometry.coords, dtype=float))
        return float(np.abs(offsets).mean())

    lane_lines = {}
    for piece in paint:
        if piece.kind is BIKE_LANE_EDGE_LINE:
            lane_lines.setdefault((piece.leg, piece.side), set()).add(round(offset_ft(piece), 2))
    for dash in dashes:
        assert dash.geometry.length == pytest.approx(DOTTED_MARK_FT, abs=0.05), (
            f"a dash is {dash.geometry.length:.2f} ft, not the {DOTTED_MARK_FT} ft MUTCD's dotted "
            f"extension asks for")
        assert openings.driven.covers(dash.geometry), (
            "a dash lies outside the opening it is supposed to be crossing")
        on_this_side = lane_lines.get((dash.leg, dash.side), set())
        assert on_this_side, f"a dash on {dash.leg} {dash.side} where no lane line was painted"
        assert min(abs(offset_ft(dash) - o) for o in on_this_side) < 0.25, (
            f"a dash sits {offset_ft(dash):.2f} ft off the centerline where this side's lane lines "
            f"run at {sorted(on_this_side)} - it has to continue one of them, not run beside it")


@needs_source_data
def test_a_hatched_zone_tapers_off_at_an_opening_where_a_lane_line_stops_dead(site_models):
    """The two ways paint ends at a driveway, and they are not the same way.

    A no-travel zone that stops square reads as a rectangle punched out of the hatching; the same
    zone at a crossing ends on the crossing's own clean diagonal, which is what a striper paints.
    So a FILL is cut against the openings' rounded run-out and everything else against the
    entrance itself - measured here at the E Broad opening, where the removed ground has to be
    wider at the travel lane's edge than at the kerb and the entrance itself must NOT be.

    AND THE RUN-OUT IS A FILLET, tangent to the zone's edge line at the travel lane. That is the
    property, not just "wider at the lane than at the kerb": the first version was an arc the other
    way round - tangent to the TRANSVERSE direction at the lane edge, so it was flat exactly where
    the eye follows the line and curved only in the last foot at the kerb. It measured as a taper
    (2.5 ft of sweep) and still read as a blunt cut at every drawing scale. Pinned by probing the
    profile across the strip: most of the sweep has to happen in the half nearest the LANE.

    An earlier version was wrong more crudely: profiled across the nominal width the band was
    requested at (25.9 ft) rather than the kerb it was clamped to (7.6 ft), every step came out
    within 3% of the full run - a square gap 4 ft wider at each end with no taper in it at all.
    """
    from shapely.geometry import Point

    from src.geometry.kerbs import OpeningSource
    from src.geometry.model import inset_point_at_station
    from src.geometry.paint import kerb_opening_bands
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT

    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios("ebroad_princeton").build_proposal_bike_lanes,
                             DesignState.from_model(model), model)
    openings = kerb_opening_bands(state)
    leg = state.legs["e_broad_st_east"]
    opening = next(o for o in state.kerb_openings[("e_broad_st_east", "left")]
                   if o.source is OpeningSource.DROPPED_KERB)
    # The traced kerb at this opening, which is both the strip's depth and the fillet's radius.
    kerb_ft = float(np.abs(curb_offsets_at_stations(
        leg, "left", np.array([(opening.start_ft + opening.end_ft) / 2]))).max())

    def gap_ft(shape, offset_ft):
        """How much station `shape` removes at one offset from the centerline, by probing it."""
        inside = [s / 8 for s in range(int(8 * (opening.start_ft - 40)),
                                       int(8 * (opening.end_ft + 40)))
                  if shape.contains(Point(inset_point_at_station(leg, s / 8, offset_ft)))]
        return max(inside) - min(inside) if inside else 0.0

    depth_ft = kerb_ft - TARGET_LANE_WIDTH_FT
    probes = [TARGET_LANE_WIDTH_FT + 1.0, TARGET_LANE_WIDTH_FT + depth_ft / 2, kerb_ft - 0.4]
    entrance = [gap_ft(openings.driven, off) for off in probes]
    hatching = [gap_ft(openings.tapered, off) for off in probes]

    assert entrance[0] == pytest.approx(entrance[2], abs=0.3), (
        f"the entrance is {entrance[0]:.2f} ft at the lane edge and {entrance[2]:.2f} ft at the "
        f"kerb - it is the surveyed dropped kerb and must not taper; only the hatching does")
    # The PROFILE, not the arc's own formula. The mouth's 1.5 ft rounded trim is buffered onto the
    # union afterwards, and a round buffer smears the arc's steep end sideways - measured, it puts
    # ~2 ft more station into the probe 1 ft out from the lane edge than the bare arc has there.
    # So what is pinned is the shape the fillet gives the zone, which is what was wrong before.
    assert hatching[2] == pytest.approx(entrance[2], abs=0.6), (
        f"the run-out still removes {hatching[2] - entrance[2]:.2f} ft more than the entrance at "
        f"the kerb - the fillet is supposed to arrive exactly at the surveyed mouth")
    assert hatching[0] > hatching[1] > hatching[2], (
        f"the run-out is not monotonic across the strip: {[round(h, 2) for h in hatching]}")
    assert hatching[0] - entrance[0] >= depth_ft, (
        f"the sweep spends {hatching[0] - entrance[0]:.2f} ft at the lane edge on a strip "
        f"{depth_ft:.2f} ft deep - too short to read as a taper at drawing scale")
    # TANGENCY, as a profile: an arc that leaves the edge line tangentially closes most of its run
    # in the half nearest the LANE. The first version's arc was tangent to the transverse direction
    # instead, and closed most of it near the KERB - flat exactly where the eye follows the line.
    #
    # Which assertion catches that, measured rather than assumed: the wrong-way arc fails on the
    # KERB ARRIVAL above (9.0 ft of gap still open at the kerb, where the fillet closes to the
    # surveyed mouth), because a bulge keeps its run all the way out. Even given the same radius
    # and the same two endpoints it fails there first. This one pins the front-loading that makes
    # the sweep read as one stroke rather than as a wide gap with rounded corners.
    near_lane_half = hatching[0] - hatching[1]
    near_kerb_half = hatching[1] - hatching[2]
    assert near_lane_half > near_kerb_half, (
        f"the run-out closes {near_lane_half:.2f} ft over the lane-side half of the strip and "
        f"{near_kerb_half:.2f} ft over the kerb-side half, so the arc is tangent to the wrong "
        f"direction - flat where the eye follows the edge line, which is the blunt end again")
