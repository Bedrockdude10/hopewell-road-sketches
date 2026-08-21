"""A crossing the surveyor traced inside the frame is drawn, and drawn from its own geometry.

Every test here is one half of the defect docs/network-renderer-plan.md measures at Broad &
Greenwood with HOPEWELL_FRAME_SCALE=2.5 - 10 traced crossings inside a 431.2 ft frame, 4 of them
drawn, the 4 that happen to match this junction's modelled legs:

  * WHAT IS COLLECTED. The frame decides, not the legs. At 1x Greenwood's frame contains exactly
    the 4 that are drawn today, which is why nothing looked wrong until the frame was widened.
  * WHAT IS DRAWN FROM. The traced way itself, so a crossing at a junction with no legs is still
    drawable, and one with legs still lands where the per-leg code puts it (0.2-2.7 ft across the
    four sites) rather than being rebuilt off a station and a skew.
  * WHAT IS NOT DRAWN. A crossing with no `crossing:markings` tag gets no paint. The existing
    per-leg path reads that tag through `OSM_MARKINGS_TO_STYLE.get(tag, "lines")`, so an unmarked
    crossing comes out wearing two transverse lines nobody surveyed - a drawing that makes a claim
    about the street on the strength of a missing tag.

The numbers below are the committed OSM snapshot's (tests/fixtures/osm_cache), so a change in them
is a change in the survey and should be read as one.
"""
import contextlib
import io
import itertools

import pytest
from shapely.geometry import LineString, Point

from src.geometry.surveyed import (crossing_bars_ft, crossing_lines_ft,
                                   surveyed_crossings_in_frame)
from src.render.coords import wgs84_to_state_plane
from src.render.crosswalks import CROSSWALK_DEPTH_FT, crosswalk_axes
from src.render.frame import FRAME_SCALE_ENV, junction_frame
from tests.conftest import SITES, WIDE_FRAME_SCALE, needs_source_data

GREENWOOD = "broad_st_greenwood"

# The frame docs/network-renderer-plan.md measures the defect at: 431.2 ft, 2.5x the 172.5 ft
# Greenwood is drawn at by default - and the scale this module's counts were measured at (10
# crossings in Greenwood's frame, 6 of them off-leg). NO LONGER A SECOND NUMBER: it was written
# here as its own "2.5" while conftest's WIDE_FRAME_SCALE said 2.2, so this module and the
# wide_site_models fixture it uses were on different sheets. One constant, and it is the render's.
MEASURED_FRAME_SCALE = str(WIDE_FRAME_SCALE)
WIDE_FRAME_RADIUS_FT = 431.2

# Inside that frame: 10 traced crossings, 4 of them this junction's own legs', 6 belonging to
# neighbouring junctions 263-420 ft away. Inside the 1x frame: exactly the 4.
CROSSINGS_IN_THE_WIDE_FRAME = 10
CROSSINGS_ON_A_MODELLED_LEG = 4

# How far a band built from the traced way may sit from where the per-leg code puts the same
# crossing's band. Measured, the four legs at Greenwood come out 1.44, 1.63, 1.98 and 2.73 ft
# apart, and the four sites together 0.20-2.73 ft. That is not error either way: the per-leg centre
# is the crossing's midpoint PROJECTED onto the modelled centreline, so the gap is the crossing's
# own lateral offset from it - the part of the survey the projection discards.
BAND_CENTRE_TOLERANCE_FT = 3.0


def _crossings_layer(model):
    """The fetched OSM crossing layer, quietly - loading a site prints its phase notes."""
    from src.sources.osm_context import fetch_crossings
    from src.geometry.treatments import CROSSING_CONTEXT_RADIUS_M
    from src.render.frame import context_radius_m

    with contextlib.redirect_stdout(io.StringIO()):
        return fetch_crossings(model.center_wgs84,
                               radius_m=context_radius_m(CROSSING_CONTEXT_RADIUS_M))


def _traced_lines_ft(crossings):
    """Every fetched record as a state-plane line, independently of src/geometry/surveyed.py.

    A test that counted what the module returned against the module's own idea of what it should
    have returned would pass on any filter at all. This is the second opinion.
    """
    lines = []
    for record in crossings:
        coords = record["coords_wgs84"]
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        lines.append(LineString(zip(xs, ys)))
    return lines


@needs_source_data
def test_every_traced_crossing_inside_the_frame_is_returned(site_models, monkeypatch):
    """All 10, counted independently off the fetched layer - not 4, and not one of them dropped.

    The count is recomputed here from the raw OSM records rather than taken on trust, because the
    bug being fixed is a SILENT omission: the previous path returned a subset and said nothing, and
    a test asserting only "10" would keep passing if the module and the test agreed on the wrong
    10. `distance_ft` is asserted against the same measure to make sure the number reported beside
    a crossing is the one that decided it is in the picture.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    model = site_models[GREENWOOD]
    frame = junction_frame(model)
    assert frame.radius_ft == pytest.approx(WIDE_FRAME_RADIUS_FT, abs=0.1)

    crossings = _crossings_layer(model)
    inside = [line for line in _traced_lines_ft(crossings)
              if model.center_ft.distance(line) <= frame.radius_ft]
    assert len(inside) == CROSSINGS_IN_THE_WIDE_FRAME

    found = surveyed_crossings_in_frame(model, crossings)
    assert len(found) == len(inside)
    assert {crossing.geometry.wkt for crossing in found} == {line.wkt for line in inside}
    for crossing in found:
        assert crossing.distance_ft == pytest.approx(
            model.center_ft.distance(crossing.geometry), abs=1e-6)
        assert crossing.distance_ft <= frame.radius_ft
    assert [crossing.distance_ft for crossing in found] == sorted(
        crossing.distance_ft for crossing in found)


@needs_source_data
def test_the_frame_decides_membership_and_the_legs_do_not(site_models, monkeypatch):
    """Widening the frame finds 6 more crossings; the 4 on legs are unchanged by it.

    The bug in one assertion. At 1x the frame holds exactly the crossings the per-leg path can
    draw, so leg-gating and frame-gating agreed and the loss was invisible; at 2.5x the frame holds
    10 and the leg match still finds 4. A module that cached the frame, or that read the leg match
    before the frame, would pass the first half of this and fail the second.
    """
    model = site_models[GREENWOOD]

    monkeypatch.setenv(FRAME_SCALE_ENV, "1")
    at_1x = surveyed_crossings_in_frame(model)
    assert len(at_1x) == CROSSINGS_ON_A_MODELLED_LEG
    assert all(crossing.leg is not None for crossing in at_1x)

    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    at_2_5x = surveyed_crossings_in_frame(model)
    assert len(at_2_5x) == CROSSINGS_IN_THE_WIDE_FRAME
    on_a_leg = [crossing for crossing in at_2_5x if crossing.leg is not None]
    assert len(on_a_leg) == CROSSINGS_ON_A_MODELLED_LEG
    assert {crossing.leg for crossing in on_a_leg} == set(model.legs)
    # The extra 6 are not defective, they are other junctions' crossings - three of them marked,
    # 263-420 ft out. Being on no leg of THIS junction is the ordinary state of a surveyed feature.
    off_leg = [crossing for crossing in at_2_5x if crossing.leg is None]
    assert len(off_leg) == 6
    assert min(crossing.distance_ft for crossing in off_leg) > 200


@needs_source_data
def test_a_leg_matched_crossing_lands_where_the_per_leg_code_puts_it(site_models, monkeypatch):
    """Within 3 ft of the existing band centre, on all four of Greenwood's legs.

    The two constructions have to agree where they overlap or "drawn as traced" is a different
    drawing rather than the same one sourced better. Compared against the centre
    crosswalk_axes derives from `crosswalk_offsets`, which is the point the per-leg band is built
    on - NOT against that band polygon's centroid, which slides toward whichever kerb is further
    away because the two reaches are asymmetric (3.73 ft on broad_st_east, against 2.73 ft for the
    centre it was built from). The centroid's extra offset is a fact about our kerb model, and
    measuring against it would be testing the traced crossing for agreement with that.
    """
    from src.geometry.treatments import DesignState
    from src.render.scene import SceneGeometry

    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    model = site_models[GREENWOOD]
    crossings = _crossings_layer(model)
    with contextlib.redirect_stdout(io.StringIO()):
        scene = SceneGeometry.resolve(model, DesignState.from_model(model), crossings)

    on_a_leg = [c for c in surveyed_crossings_in_frame(model, crossings) if c.leg is not None]
    assert len(on_a_leg) == CROSSINGS_ON_A_MODELLED_LEG
    for crossing in on_a_leg:
        offset = scene.crosswalk_offsets[crossing.leg]
        # Both halves matter: a leg whose offset came from the geometric estimate has nothing
        # surveyed to agree with, so the comparison below would be meaningless there.
        assert offset.is_surveyed
        centre, _along, _across, _cos = crosswalk_axes(
            model.legs[crossing.leg], offset.offset_ft, scene.crosswalk_skews.get(crossing.leg, 0))
        assert Point(centre).distance(crossing.centre) < BAND_CENTRE_TOLERANCE_FT


@needs_source_data
def test_a_crossing_with_no_markings_tag_is_not_drawn_as_marked(site_models, monkeypatch):
    """No bars and no lines - the fidelity rule, not an edge case.

    2 of Greenwood's 10 record no `crossing:markings`, and the existing per-leg path would draw
    both of them with two transverse lines, because it reads the tag as
    `OSM_MARKINGS_TO_STYLE.get(tag, "lines")`. Painting a crossing nobody recorded paint on is the
    same class of error as dropping one that was recorded, in the other direction.

    THE DEPRECATED TAG IS NOW READ, which is the change this assertion was pinned to force somebody
    to make on purpose. 12 of the 30 ways carry `crossing=zebra` with no `crossing:markings`, and one
    of them is inside this frame 419.6 ft out; reporting it as unrecorded rendered a marked crossing
    as bare asphalt, which is the exact failure this module exists to end. See
    surveyed.LEGACY_CROSSING_MARKINGS. So Greenwood's unrecorded count is 2 -> 1 and its zebra count
    2 -> 3, and no crossing in this frame is left with a legacy tag unread.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    model = site_models[GREENWOOD]

    unrecorded = [c for c in surveyed_crossings_in_frame(model) if c.markings is None]
    assert len(unrecorded) == 1
    for crossing in unrecorded:
        assert not crossing.is_marked
        assert crossing_bars_ft(crossing) == []
        assert crossing_lines_ft(crossing) == []
        # The band is still there: the ground a crossing occupies is surveyed even where the paint
        # on it is not, and that is what a coverage check compares against the drawing.
        assert crossing.band_ft.area > 0

    # Nothing left unrecorded still carries a legacy value we could have read. This is the
    # assertion that would catch a THIRD spelling of the same fact appearing in the data.
    assert not [c for c in unrecorded if c.tags.get("crossing")], (
        f"a crossing reported as unrecorded still has a legacy crossing= tag: "
        f"{[c.tags for c in unrecorded]}")


@needs_source_data
def test_a_zebra_crossing_is_striped_along_its_own_traced_way(site_models, monkeypatch):
    """Continental bars, inside the footprint of the way they came from, reaching both its ends.

    All THREE of Greenwood's zebras belong to no modelled leg, so today none is drawn - they are
    three of the marked crossings docs/network-renderer-plan.md counts as dropped. Two are tagged
    `crossing:markings=zebra` (15 and 18 bars over 48.9 and 59.4 ft); the third carries only the
    legacy `crossing=zebra`, which is why the count here is 3 and not 2.

    The containment assertion is the whole claim of this stream stated geometrically: bars built
    from the way cannot leave the way's own footprint. It is what a mitred buffer join would break
    at a traced bend (9 of the 10 ways here have 3-5 vertices), and what any construction that went
    back to a leg's frame for its axes would break at a skewed junction.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    model = site_models[GREENWOOD]

    zebras = [c for c in surveyed_crossings_in_frame(model) if c.markings == "zebra"]
    assert len(zebras) == 3
    for crossing in zebras:
        assert crossing.leg is None
        bars = crossing_bars_ft(crossing)
        assert len(bars) > 1
        footprint = crossing.geometry.buffer(CROSSWALK_DEPTH_FT)
        assert all(footprint.contains(bar) for bar in bars)
        # The end bars land ON the ends of the traced way. A whole-period pitch leaves up to one
        # period unpainted at one end (3.2 ft), which reads as a crossing stopping short - see
        # src/render/crosswalks.py:continental_bar_count.
        assert bars[0].intersects(Point(crossing.geometry.coords[0]))
        assert bars[-1].intersects(Point(crossing.geometry.coords[-1]))
        # And they land on the ends without running past them or into each other. A bar buffered
        # with round caps instead of flat ones grows 3 ft at each end against a 1.64 ft gap, so
        # every bar overlaps its neighbours and the end pair overhangs the way - paint drawn where
        # nothing was traced, which is the same error as the drop this stream exists to fix.
        #
        # NO SHARED AREA rather than disjoint: the pitch is along the arc, so where the way bends
        # between two bars their ends converge on the inside of the turn, and surveyed.py:_kept_apart
        # clips the later one back. Clipped bars TOUCH along that edge, which paints no ground twice
        # and is what this is about; requiring disjointness failed on the one zebra here that bends.
        assert all(before.intersection(after).area == pytest.approx(0.0, abs=1e-6)
                   for before, after in itertools.pairwise(bars))
        # Continental is bars only. The two transverse lines are the other style, not a frame
        # around this one - that would be a ladder, and no way in this snapshot is tagged one.
        assert crossing_lines_ft(crossing) == []


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_site_draws_only_the_markings_its_own_tags_record(site, site_models, monkeypatch):
    """Across all four junctions: paint only where a tag says so, always inside the traced way.

    Greenwood is where the defect was measured, but the rule has to hold on the other three, and
    each of them carries a case Greenwood does not: Columbia & Princeton's four crossings have no
    `crossing:markings` tag at all (against a config that lists all four as marked - see
    SurveyedCrossing.is_marked), and W Broad & Louellen has the snapshot's one
    `crossing:markings=no`, which is a surveyed ABSENCE of paint rather than a missing record. Both
    draw nothing, and the difference between them is why is_marked exists.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, MEASURED_FRAME_SCALE)
    model = site_models[site]
    frame = junction_frame(model)

    found = surveyed_crossings_in_frame(model)
    assert found, f"{site} has no traced crossing inside a {frame.radius_ft:.0f} ft frame"
    for crossing in found:
        assert crossing.distance_ft <= frame.radius_ft
        bars, lines = crossing_bars_ft(crossing), crossing_lines_ft(crossing)
        if crossing.markings in (None, "no"):
            assert not bars and not lines
            assert not crossing.is_marked
        else:
            assert crossing.is_marked
            assert bars or lines
        footprint = crossing.geometry.buffer(CROSSWALK_DEPTH_FT)
        assert all(footprint.contains(piece) for piece in bars + lines)


# --------------------------------------------------------------------------
# A crossing at a junction this site does not model still outranks the paint
# --------------------------------------------------------------------------

@needs_source_data
def test_no_paint_is_laid_over_a_crossing_at_an_unmodelled_junction(wide_site_models,
                                                                    monkeypatch):
    """Drawing a crosswalk and painting over it is two claims about one piece of asphalt.

    This module's whole premise is that a surveyed crossing inside the frame gets drawn from its
    own traced way. It does - and nothing was getting out of its way, because `keep_clear` was
    built from `crosswalk_bands`, which is keyed by leg and so cannot contain a crossing that
    belongs to no leg here. Measured before the fix: 164 sq ft of bike lane green, buffer
    hatching and lane fill, plus 48 ft of edge line and contraflow stripe, over Blackwell
    Avenue's two zebras in the corridor proposal - and 44 sq ft at W Broad & Louellen, 1 at
    E Broad & Princeton.

    Run over every scenario each site publishes, not just the default, because the worst case was
    the proposal rather than the baseline: the more paint a scenario lays down, the more of it
    landed on a crossing.
    """
    import contextlib
    import io

    from shapely.ops import unary_union

    from src.geometry.treatments import DesignState
    from src.site import load_site_scenarios, run_scenario
    from tests.test_sites import resolved_scene, scene_props

    # The models were BUILT wide; the frame is read again at draw time, and the fixture has
    # already put the env var back. Without this the scene resolves a 1x frame around models with
    # wide legs, finds no crossing at any other junction, and the test passes having checked
    # nothing - which is what it did first time round.
    monkeypatch.setenv(FRAME_SCALE_ENV, str(WIDE_FRAME_SCALE))
    checked = 0
    for site, model in wide_site_models.items():
        scenarios = load_site_scenarios(site)
        for name in sorted(n for n in dir(scenarios) if n.startswith("build_")):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
                scene = resolved_scene(model, state)
                paint, _posts = scene.build_paint_and_posts(scene_props(model, state, scene))
            bands = scene.unmodelled_crossing_bands
            if not bands:
                continue
            checked += 1
            crossings = unary_union(list(bands))
            for piece in paint:
                if piece.kind.is_object:
                    # A post is a point, so it is dropped rather than trimmed - but it must be
                    # dropped, and a flex post planted in a marked crosswalk is worse than paint
                    # over one. Two stood in Blackwell Avenue's zebras before PaintContext.emit
                    # learned to test the crossings as well as the openings.
                    assert not crossings.intersects(piece.geometry), (
                        f"{site}/{name}: a {piece.kind} stands in a surveyed crosswalk at "
                        f"another junction in the frame ({piece.leg} {piece.side})")
                    continue
                shared = crossings.intersection(piece.geometry)
                if shared.is_empty:
                    continue
                amount = shared.area if piece.covers_area else shared.length
                assert amount < 1.0, (
                    f"{site}/{name}: {piece.kind} is painted over "
                    f"{amount:.1f}{' sq ft' if piece.covers_area else ' ft'} of a surveyed "
                    f"crosswalk at another junction in the frame "
                    f"({piece.leg} {piece.side}) - the drawing shows a crossing and paints on it")
    assert checked, ("no site/scenario had a marked crossing at an unmodelled junction, so this "
                     "asserted nothing - check wide_site_models is wide enough to reach one")


@needs_source_data
def test_the_scene_invariant_reports_paint_over_such_a_crossing(wide_site_models,
                                                                monkeypatch):
    """And the invariant that keeps it fixed is wired to the same set the paint was cut against.

    The test above would keep passing if CrossingsAreNotPaintedOver were handed an empty tuple
    by SceneGeometry.context - a check that cannot see anything cannot fail. So this feeds it
    paint it must reject: the crossings themselves, as if a marking had been laid along one.
    """
    import contextlib
    import io

    from src.checks import CrossingsAreNotPaintedOver
    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.paint import PaintPiece
    from src.geometry.treatments import DesignState
    from tests.test_sites import resolved_scene, scene_props

    monkeypatch.setenv(FRAME_SCALE_ENV, str(WIDE_FRAME_SCALE))
    for site, model in wide_site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            state = DesignState.from_model(model)
            scene = resolved_scene(model, state)
            props = scene_props(model, state, scene)
        bands = scene.unmodelled_crossing_bands
        if not bands:
            continue
        over_the_crossing = [PaintPiece(BIKE_LANE_SURFACE, bands[0], "a_leg", "left")]
        context = scene.context(props, over_the_crossing)
        assert context.unmodelled_crossing_bands, (
            f"{site}: SceneGeometry.context did not pass the crossings to the invariants, so "
            f"CrossingsAreNotPaintedOver can never see anything")
        violations = CrossingsAreNotPaintedOver().run(context)
        assert violations, f"{site}: green laid over a whole crosswalk was not reported"
        assert violations[0].check == "paint_over_a_crossing"
        return
    pytest.fail("no site had a marked crossing at an unmodelled junction")


# --------------------------------------------------------------------------
# A marking POLICY is about the frame, not about this junction's four legs
# --------------------------------------------------------------------------

@needs_source_data
def test_a_proposal_restyles_every_marked_crossing_in_the_frame(wide_site_models, monkeypatch):
    """"All crosswalks continental" has to mean all of them.

    all_crosswalks_continental looped over `state.legs` - this junction's four approaches - and a
    2.5x frame holds ten surveyed crossings. The other six belong to Blackwell, Model and
    Seminary Avenue and were drawn from their own OSM tag whatever the proposal said, so two
    tagged `crossing:markings=lines` rendered as two parallel lines 260 ft from four that had
    been repainted. Same shape as the statutory setback that only applied at the modelled
    junction, one layer up in the markings.

    It could not be fixed in one place either: THREE consumers built these markings themselves off
    the raw drawers, so styling one of them changed neither picture. That is why this asserts
    against SceneGeometry, which both renderers now read.
    """
    import contextlib
    import io

    from src.geometry.surveyed import crossing_style_in
    from src.geometry.treatments import DesignState
    from src.site import load_site_scenarios, run_scenario
    from tests.test_sites import resolved_scene

    monkeypatch.setenv(FRAME_SCALE_ENV, str(WIDE_FRAME_SCALE))
    checked = 0
    for site, model in wide_site_models.items():
        scenarios = load_site_scenarios(site)
        for name in sorted(n for n in dir(scenarios) if n.startswith("build_")):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
                scene = resolved_scene(model, state)
            if not any(c.is_marked for c in scene.unmodelled_crossings):
                continue
            checked += 1
            for crossing in scene.unmodelled_crossings:
                if not crossing.is_marked:
                    continue
                assert crossing_style_in(state, crossing) == "continental", (
                    f"{site}/{name}: the crossing {crossing.distance_ft:.0f} ft out is drawn as "
                    f"{crossing_style_in(state, crossing)!r} while this junction's own legs are "
                    f"continental - a proposal's crosswalk policy stops at the modelled legs")
    assert checked, "no scenario had a marked crossing at an unmodelled junction"


@needs_source_data
def test_a_policy_never_paints_a_crossing_nobody_marked(wide_site_models, monkeypatch):
    """The other half of the rule, and the one that must not be traded away for the first.

    A crossing recorded as unpainted (`crossing:markings=no`), or with nothing recorded at all,
    draws NOTHING however loudly a proposal says "all crosswalks continental". Painting one would
    be a new crossing at an uncontrolled approach - MUTCD 3C.02(04) wants an engineering study
    with pedestrian counts this project does not hold (STANDARDS.md section 2) - and it would be
    this repo's signature failure inverted: inventing survey data instead of dropping it.

    One such crossing is in Broad & Greenwood's frame, 375 ft out, with no crossing:markings tag.
    """
    import contextlib
    import io

    from src.geometry.surveyed import crossing_style_in
    from src.geometry.treatments import DesignState
    from src.site import load_site_scenarios, run_scenario
    from tests.test_sites import resolved_scene

    monkeypatch.setenv(FRAME_SCALE_ENV, str(WIDE_FRAME_SCALE))
    checked = 0
    for site, model in wide_site_models.items():
        scenarios = load_site_scenarios(site)
        for name in sorted(n for n in dir(scenarios) if n.startswith("build_")):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
                scene = resolved_scene(model, state)
            drawn = {id(c) for c, _bars, _lines in scene.surveyed_crossing_markings()}
            for crossing in scene.unmodelled_crossings:
                if crossing.is_marked:
                    continue
                checked += 1
                assert crossing_style_in(state, crossing) is None, (
                    f"{site}/{name}: a policy assigned a style to an unmarked crossing")
                assert id(crossing) not in drawn, (
                    f"{site}/{name}: paint was drawn on the crossing {crossing.distance_ft:.0f} ft "
                    f"out, which records no markings - that is a NEW crossing, not a repaint")
    assert checked, "no unmarked crossing in any frame, so this asserted nothing"


@needs_source_data
def test_existing_conditions_still_draw_each_crossing_as_surveyed(wide_site_models, monkeypatch):
    """And with no policy applied, the survey stands - which is what a baseline must show."""
    import contextlib
    import io

    from src.geometry.surveyed import crossing_style_in, drawable_markings
    from src.geometry.treatments import DesignState
    from tests.test_sites import resolved_scene

    monkeypatch.setenv(FRAME_SCALE_ENV, str(WIDE_FRAME_SCALE))
    checked = 0
    for site, model in wide_site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = DesignState.from_model(model)
            scene = resolved_scene(model, baseline)
        for crossing in scene.unmodelled_crossings:
            checked += 1
            assert crossing_style_in(baseline, crossing) == drawable_markings(crossing.tags), (
                f"{site}: existing conditions redrew a crossing in something other than the style "
                f"it was surveyed in")
    assert checked, "no surveyed crossing in any frame"


def test_a_skewed_stop_bar_starts_on_the_centreline_in_both_views():
    """The bar's near end lands on the line it is measured to, at any skew, from one function.

    THE TWO NUMBERS ARE IN DIFFERENT FRAMES, deliberately: the span is perpendicular to the leg
    and every renderer stretches it by 1/cos(skew) itself, the lateral offset is already along the
    rotated across-axis because neither renderer stretches that one. Splitting that stretch across
    the callers is what went wrong - the plan view applied its own 1/cos to the offset and
    src/render/export.py exported the raw figure, so on louellen_st_west's -44 deg crossing the 3D
    bar stood 2.15 ft the wrong side of the centreline, straight through the opposing lanes, while
    the plan view was correct. Nothing compared the two, which is why this test exists and not
    just the fix.

    Checked as geometry rather than as arithmetic: project the near end back onto the alignment's
    own perpendicular and it must be exactly `inner_ft` out. That holds in both views because both
    now read the same function.
    """
    import math

    from src.render.crosswalks import stop_bar_band_geometry_ft
    from src.render.export import _stop_bar_span_m
    from src.render.coords import FT_TO_M

    for skew_deg in (0.0, -44.03, 12.5, 60.0):
        for inner_ft in (0.0, 2.4):
            span_ft, lateral_ft = stop_bar_band_geometry_ft(
                34.0, edge_is_kerb=True, inner_ft=inner_ft, skew_deg=skew_deg)
            # As a renderer builds it: the span is the figure that gets stretched by 1/cos,
            # then the whole thing is projected back onto the leg's own perpendicular.
            cos_s = math.cos(math.radians(abs(skew_deg)))
            near_ft = (lateral_ft - span_ft / (2 * cos_s)) * cos_s
            assert near_ft == pytest.approx(inner_ft, abs=1e-9), (
                f"at {skew_deg:+.1f} deg skew the bar's near end sits {near_ft - inner_ft:+.2f} ft "
                f"off the line it is measured to")

    # And the 3D path is the same figure, not a parallel derivation - the whole defect was two
    # callers stretching differently. Compared in metres, which is the frame the export writes.
    class _Fake:
        """Only what _stop_bar_span_m reads, so this stays a unit test."""

    span_ft, lateral_ft = stop_bar_band_geometry_ft(34.0, edge_is_kerb=True, inner_ft=0.0,
                                                   skew_deg=-44.03)
    from unittest.mock import patch
    with patch("src.render.export.stop_bar_width_ft", return_value=34.0), \
         patch("src.render.export.entering_lane_width_ft", return_value=None), \
         patch("src.render.export.divider_shift_toward_ft", return_value=0.0):
        exported = _stop_bar_span_m(None, "leg", True, skew_deg=-44.03)
    assert exported["stop_bar_lateral_offset_m"] == pytest.approx(lateral_ft * FT_TO_M)
    assert exported["stop_bar_span_m"] == pytest.approx(span_ft * FT_TO_M)
