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
from tests.conftest import SITES, needs_source_data

GREENWOOD = "broad_st_greenwood"

# The frame docs/network-renderer-plan.md measures the defect at: 431.2 ft, 2.5x the 172.5 ft
# Greenwood is drawn at by default.
WIDE_FRAME_SCALE = "2.5"
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
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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

    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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

    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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
        assert all(before.disjoint(after) for before, after in itertools.pairwise(bars))
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
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME_SCALE)
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
