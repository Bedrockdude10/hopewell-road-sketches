"""The guard that says a render dropped something the surveyor recorded.

Every test here is about one property: what is in the frame is in the drawing. The reason it
needs its own guard is that no other check in this repo can fail on it - src/checks.py asks
whether the geometry we built is right, and a feature that was never built has no geometry to be
wrong about, so a render that omits six of Broad & Greenwood's ten mapped crossings passes every
invariant in the suite and looks finished.

Two of these tests are deliberately CONTROL CASES rather than bug reports. A checker that reports
every layer as broken is worth nothing, so the kerbs layer - already fixed in `cb9c8b6`, drawn at
the drawing radius - has to come out clean, and the kerb ramps layer has to come out clean at 1x
and dirty at 2.5x on the same code. If either of those ever goes the other way, the number in the
failing test is the least of the problems.

And the crossings assertions are split on purpose. One asserts the RELATIONSHIP - the gap is the
surveyed crossings the drawing does not contain, measured independently here - and is written to
hold unchanged once stream A of docs/network-renderer-plan.md draws all ten. The other pins
today's 6-of-10 and says in its own name that it is expected to fail.
"""
import contextlib
import io

import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.geometry.coverage import (CONTROL_NEAR_NODE_FT, CROSSING_DRAWN_FRACTION, Uncovered,
                                   _markings_label, _read_drawing, coverage_gaps,
                                   describe_coverage)
from src.geometry.treatments import DesignState
from src.render.coords import FT_TO_M
from src.render.frame import FRAME_SCALE_ENV, junction_frame
from src.render.scene import SceneGeometry
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_street_furniture,
                                     fetch_traffic_control)
from tests.conftest import WIDE_FRAME_SCALE, needs_source_data

# The frame docs/network-renderer-plan.md measures the gap at, and the frame output/ is drawn at.
# At 1x every leg ends inside the 130-170 ft its width was measured over, so the neighbouring
# junctions whose features are dropped are not in shot at all - which is why the same code is
# faithful at one scale and not at the other. conftest's WIDE_FRAME_SCALE is that scale; this
# module used to restate it as its own "2.5" beside a conftest that said 2.2.
WIDE_FRAME = str(WIDE_FRAME_SCALE)
GREENWOOD_WIDE_RADIUS_FT = 431.2


def a_drawing(model, radius_ft: float):
    """(scene, everything the drawing puts on the ground) for one site's existing conditions.

    Paint AND crossing bands AND props, because all three are the drawing and none of them alone
    is: a crossing band is not a PaintPiece (it lives in its own dict on SceneGeometry and its own
    key in the exported JSON) and a tactile pad is not paint at all. See coverage_gaps, which
    accepts the lot for exactly this reason.

    `props` is empty here - only the posts the paint itself implies come back from
    build_paint_and_posts. a_full_drawing adds the OSM-sourced furniture, and the difference
    between the two is itself a coverage difference, which is the point of keeping both.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        state = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=radius_ft * FT_TO_M)
        scene = SceneGeometry.resolve(model, state, crossings)
        paint, props = scene.build_paint_and_posts([])
    return scene, [*paint, *scene.crosswalk_bands.values(), *props,
                   *scene.surveyed_crossing_paint()]


def a_full_drawing(model, radius_ft: float):
    """The same, plus every prop the renderers build - the drawing a reader actually sees.

    src/render/props.py:build_props is what both views call, so the tactile pads and control
    hardware here are the ones in the picture rather than a re-derivation. Needed by any layer
    whose features are drawn as objects instead of as paint.
    """
    radius_m = radius_ft * FT_TO_M
    with contextlib.redirect_stdout(io.StringIO()):
        from src.render.props import build_props

        state = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=radius_m)
        scene = SceneGeometry.resolve(model, state, crossings)
        props = build_props(model, state, scene.crosswalk_offsets, model.center_ft,
                            fetch_traffic_control(model.center_wgs84, radius_m=radius_m),
                            fetch_street_furniture(model.center_wgs84, radius_m=radius_m),
                            crossings, fetch_kerbs(model.center_wgs84, radius_m=radius_m))
        paint, props = scene.build_paint_and_posts(props)
    return scene, [*paint, *scene.crosswalk_bands.values(), *props,
                   *scene.surveyed_crossing_paint()]


def gap_for(gaps: list[Uncovered], layer: str) -> Uncovered | None:
    return next((gap for gap in gaps if gap.layer == layer), None)


def drawn_ground(drawing: list):
    """The union of every footprint in `drawing`, derived here rather than imported.

    Four lines of duplication on purpose: the tests below use it to count what the drawing
    contains INDEPENDENTLY of the module doing the same, so importing the module's own reader
    would make the comparison circular. It also keeps working when a future stream changes what
    a drawn crossing is - it reads whatever is in the list.
    """
    shapes = [getattr(piece, "geometry", piece) for piece in drawing
              if not isinstance(piece, dict)]
    areas = [shape for shape in shapes if not shape.is_empty and shape.area > 0]
    return unary_union(areas) if areas else None


def surveyed_crossings_ft(model, radius_ft: float) -> list[LineString]:
    """Every OSM crossing way whose geometry reaches inside the drawn frame, in feet."""
    from src.geometry.intersection import to_state_plane

    centre = junction_frame(model).center_ft
    with contextlib.redirect_stdout(io.StringIO()):
        fetched = fetch_crossings(model.center_wgs84, radius_m=radius_ft * FT_TO_M)
    lines = [LineString(to_state_plane(c["coords_wgs84"])) for c in fetched]
    return [line for line in lines if line.distance(centre) <= radius_ft]


# --------------------------------------------------------------------------
# Crossings: the gap this module was written for
# --------------------------------------------------------------------------

@needs_source_data
def test_the_crossings_gap_is_what_the_drawing_does_not_contain(site_models, monkeypatch):
    """The reported gap equals the surveyed crossings in frame minus the ones drawn.

    WRITTEN TO SURVIVE THE FIX. Nothing here is a remembered count: the total comes from OSM and
    the drawn count is measured off the drawing that was handed in, by a DIFFERENT test from the
    module's own (is the way's midpoint under drawn ground, rather than is a quarter of its length)
    - so the two agreeing is worth something. When stream A draws all ten, `drawn` becomes 10, no
    crossings gap comes back, and the first branch holds without an edit.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME)
    model = site_models["broad_st_greenwood"]
    radius_ft = junction_frame(model).radius_ft
    _scene, drawing = a_drawing(model, radius_ft)

    # DRAWABLE ONLY. A crossing whose survey records no markings has nothing to draw, so it is
    # excluded from the count and named in a note instead (coverage.crossing_gaps). One of
    # Greenwood's ten is like that, so the denominator here is 9, and comparing against all ten
    # would demand paint nobody surveyed.
    from src.geometry.surveyed import drawable_markings, surveyed_crossings_in_frame
    surveyed = [c.geometry for c in surveyed_crossings_in_frame(model)
                if drawable_markings(c.tags) is not None]
    # PAINT OF EITHER SHAPE. Bars have area; a `lines`-style crossing is two thin lines and has
    # none, so an area-only measure calls all four of Greenwood's line-style crossings undrawn even
    # though they are on the page. Half of this is no longer independent of the module - the
    # has_line_along call is the module's own - and that is worth saying rather than hiding: the
    # AREA half is still measured differently (is the midpoint under drawn ground, against
    # coverage's is a quarter of the length), which is where the cross-check still earns its keep.
    # PROXIMITY, not containment. A leg-matched crossing's band is rebuilt from the leg's frame and
    # its centre sits 1.44-2.73 ft off the traced way's own centre (measured across the four sites),
    # against a band only CROSSWALK_DEPTH_FT = 6 ft deep - so asking whether the traced midpoint is
    # inside the drawn band fails on two of Greenwood's four for a reason that is about our kerb
    # model rather than about whether anything was drawn. Distance to the nearest drawn geometry is
    # robust to that, and still a different question from coverage's own (a fraction of the way's
    # length under drawn ground), which is where this cross-check earns its keep.
    from shapely.ops import unary_union
    shapes = [g for g in (getattr(item, "geometry", item) for item in drawing)
              if hasattr(g, "geom_type") and not g.is_empty]
    everything = unary_union(shapes) if shapes else None
    drawn = sum(1 for line in surveyed
                if everything is not None
                and everything.distance(line.interpolate(0.5, normalized=True)) <= 3.0)

    gap = gap_for(coverage_gaps(model, drawing), "crossings")
    if gap is None:
        assert drawn == len(surveyed), (
            f"no crossings gap was reported, but only {drawn} of {len(surveyed)} surveyed "
            f"crossings in the frame have anything drawn on them")
    else:
        assert gap.total == len(surveyed)
        assert gap.count == len(surveyed) - drawn, (
            f"reported {gap.count} of {gap.total} dropped; measured {len(surveyed) - drawn}")
        assert gap.drawn == drawn


@needs_source_data
def test_greenwoods_wide_frame_now_draws_every_crossing_it_records(site_models, monkeypatch):
    """The measured state AFTER the surveyed-crossings path landed: no crossings gap at all.

    This replaces a test that pinned 6 of 10 dropped and said it should fail when the fix landed. It
    did. The history is worth keeping in one place, because the numbers are the argument:

        before   10 in frame, 4 drawn, 6 dropped 263-420 ft out, three of them a zebra
        after    10 in frame, 9 with recorded markings and all 9 drawn from their own traced ways;
                 the tenth records no markings and is named in a note rather than counted

    Across all four sites the crossings layer went from 12 dropped to 0. What remains dirty at 2.5x
    is kerb_ramps and traffic_control, which are PROPS placed per leg and so have the same structural
    problem crossings had - see the module docstring.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME)
    model = site_models["broad_st_greenwood"]
    radius_ft = junction_frame(model).radius_ft
    assert radius_ft == pytest.approx(GREENWOOD_WIDE_RADIUS_FT, abs=0.5), (
        "the frame moved - every number in this file is measured at this radius")

    _scene, drawing = a_drawing(model, radius_ft)
    assert gap_for(coverage_gaps(model, drawing), "crossings") is None, (
        "a crossing inside the frame has nothing drawn on it - the whole point of "
        "src/geometry/surveyed.py is that this cannot happen for a crossing whose markings are "
        "recorded")


@needs_source_data
def test_a_narrower_frame_drops_fewer_crossings(site_models, monkeypatch):
    """At 1x the neighbouring junctions are not in shot, so there is little or nothing to drop.

    The same code, the same OSM, a different frame: 4 of 4 crossings drawn at 172 ft against 4 of
    10 at 431 ft. That is what says the gap is the render's leg-gating meeting a wider picture,
    rather than a crossing this project has always been failing to draw.
    """
    model = site_models["broad_st_greenwood"]

    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME)
    wide_radius_ft = junction_frame(model).radius_ft
    _scene, wide = a_drawing(model, wide_radius_ft)
    wide_gap = gap_for(coverage_gaps(model, wide), "crossings")

    monkeypatch.delenv(FRAME_SCALE_ENV, raising=False)
    near_radius_ft = junction_frame(model).radius_ft
    _scene, near = a_drawing(model, near_radius_ft)
    near_gap = gap_for(coverage_gaps(model, near), "crossings")

    assert near_radius_ft < wide_radius_ft
    assert near_gap is None or near_gap.count < wide_gap.count, (
        f"the narrower frame drops {near_gap and near_gap.count} crossings and the wider one "
        f"{wide_gap and wide_gap.count} - a smaller picture cannot contain more surveyed ground")


# --------------------------------------------------------------------------
# The control cases: layers that must come out clean
# --------------------------------------------------------------------------

@needs_source_data
@pytest.mark.parametrize("scale", [None, WIDE_FRAME])
def test_the_kerbs_layer_reports_no_gap(site_models, monkeypatch, scale):
    """Every traced kerb inside the frame is drawn - the layer that was already fixed.

    THE PROOF THAT THIS CHECK IS NOT JUST ALWAYS RED. 29 of 29 at Broad & Greenwood's 431 ft
    frame, and clean at 1x too. Both renderers take kerb_lines_with_tags_ft at
    drawn_kerb_radius_ft(), which scales with the frame and comes to 984 ft at 2.5x, so the set
    that is drawn cannot be narrower than the set that is in shot. Before `cb9c8b6` they took the
    default NEAR set - within 80 ft of the junction centre, a test written for a corner-radius
    circle fit - and 8,938 ft of traced kerb along the corridor was dropped.
    """
    if scale is None:
        monkeypatch.delenv(FRAME_SCALE_ENV, raising=False)
    else:
        monkeypatch.setenv(FRAME_SCALE_ENV, scale)
    model = site_models["broad_st_greenwood"]
    _scene, drawing = a_drawing(model, junction_frame(model).radius_ft)

    gaps = coverage_gaps(model, drawing)
    assert gap_for(gaps, "kerbs") is None, (
        f"kerbs are collected by the DRAWING radius since cb9c8b6, so this layer is the control "
        f"case for the whole module: {gap_for(gaps, 'kerbs')}")


@needs_source_data
def test_the_kerb_ramps_layer_is_clean_at_1x_and_dirty_at_2_5x(site_models, monkeypatch):
    """A second control case, on a layer drawn as PROPS rather than as paint.

    Measured against the drawing both renderers actually build (build_props): a drawn pad sits
    1.5-2.7 ft from its own traced ramp way, and the nearest pad to a ramp the render omits is
    288-309 ft off, at another junction. So the same code is faithful at 1x - all of Greenwood's
    own ramps drawn - and drops 7 of the 11 in the 2.5x frame, which are the ramps of the same
    neighbouring junctions whose crossings go missing. One cause, two layers.

    The counts are facts about the committed OSM snapshot and moved with it (3 of 7 -> 7 of 11 when
    the fixture was refreshed onto tracing that added ramps); the PROPERTY - clean at 1x, dirty at
    2.5x, for one reason - is what this pins.
    """
    model = site_models["broad_st_greenwood"]

    monkeypatch.delenv(FRAME_SCALE_ENV, raising=False)
    _scene, near = a_full_drawing(model, junction_frame(model).radius_ft)
    assert gap_for(coverage_gaps(model, near), "kerb_ramps") is None, (
        "every ramp in the 1x frame belongs to this junction and is drawn")

    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME)
    _scene, wide = a_full_drawing(model, junction_frame(model).radius_ft)
    gap = gap_for(coverage_gaps(model, wide), "kerb_ramps")
    assert gap is not None and (gap.count, gap.total) == (7, 11)
    assert all("tactile_paving=yes" in example for example in gap.examples), (
        "an example has to name the tag that says the ramp is there")


@needs_source_data
def test_the_traffic_control_layer_finds_the_stop_nodes_nobody_draws(site_models, monkeypatch):
    """Two stop nodes 298 and 409 ft out, in the picture and with no sign drawn.

    The same leg-gating in another disguise: src/render/props.py matches a control node to a leg
    and refuses past STOP_NODE_MAX_ALONG_FT (100 ft), which is the right rule for "does this node
    govern THIS junction" and the wrong one for "is it in the drawing". So the wide render shows
    two priority junctions with no control on them.

    The signals node at the junction itself is NOT reported, which is the part that makes the
    layer trustworthy: its poles stand 43 ft away at the corners and are matched on prop type
    within CONTROL_NEAR_NODE_FT, so a node whose hardware IS drawn comes out covered.
    """
    monkeypatch.setenv(FRAME_SCALE_ENV, WIDE_FRAME)
    model = site_models["broad_st_greenwood"]
    _scene, drawing = a_full_drawing(model, junction_frame(model).radius_ft)

    gap = gap_for(coverage_gaps(model, drawing), "traffic_control")
    assert gap is not None and (gap.count, gap.total) == (2, 3)
    assert all("highway=stop" in example for example in gap.examples), (
        f"the signals node's hardware is drawn {CONTROL_NEAR_NODE_FT:.0f} ft away and must count "
        f"as covered: {gap.examples}")


# --------------------------------------------------------------------------
# What counts as drawing a crossing
# --------------------------------------------------------------------------

def test_a_line_crossing_a_crossing_way_does_not_count_as_drawing_it():
    """Only ground counts. A lane edge line meets every crossing on its leg at right angles.

    This is the false positive that would make the check useless in the direction that matters:
    report a dropped crossing as drawn and the guard is worse than absent, because it has
    certified the omission. A zero-area line contributes nothing to the union whatever tolerance
    it is given, which is why _read_drawing keeps footprints and discards lines.
    """
    crossing = LineString([(0, 0), (40, 0)])
    lane_edge_line = LineString([(20, -100), (20, 100)])
    drawing = _read_drawing([lane_edge_line])
    assert drawing.ground is None
    assert drawing.fraction_covered(crossing) == 0.0


def test_a_band_over_a_crossing_counts_as_drawing_it():
    """And the footprint that IS a crossing covers most of its way's length.

    5 ft either side of the way, 10 ft short of each end - the shape a band has, since the band
    stops at the kerb while the traced way runs on to the sidewalk centreline. 50% here against
    the 62-91% measured at the four sites, and CROSSING_DRAWN_FRACTION is a quarter.
    """
    crossing = LineString([(0, 0), (40, 0)])
    band = Polygon([(10, -5), (30, -5), (30, 5), (10, 5)])
    drawing = _read_drawing([band])
    assert drawing.fraction_covered(crossing) == pytest.approx(0.5)
    assert drawing.fraction_covered(crossing) >= CROSSING_DRAWN_FRACTION


def test_the_drawing_is_read_out_of_whatever_shape_it_arrives_in():
    """PaintPieces, bare footprints, prop dicts, and dicts and lists of those.

    Three renderers already disagree about what a drawn thing IS, and stream A adds a fourth. A
    reader that accepted only today's types would report the new one as missing from the very
    drawing that draws it - so this pins the breadth rather than leaving it to a docstring.
    """
    class APaintPiece:
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    drawing = _read_drawing([
        APaintPiece(),                                        # something with .geometry
        Polygon([(5, 0), (6, 0), (6, 1), (5, 1)]),            # a bare footprint
        {"east": Polygon([(9, 0), (10, 0), (10, 1), (9, 1)])},  # a bands dict, unwrapped
        [{"type": "tactile_paving_pad", "position_ft": (2.0, 3.0)}],   # a prop, nested
        None,
    ])
    assert drawing.ground.area == pytest.approx(3.0)
    assert len(drawing.props_of(("tactile_paving_pad",))) == 1
    assert drawing.props_of(("stop_sign",)) == []


def test_the_markings_are_reported_under_the_tag_that_carries_them():
    """`crossing:markings` and the older `crossing` both hold these values, and absent is neither.

    Both tags are in use in this borough on ways 250 ft apart - two of Greenwood's three dropped
    zebras are crossing:markings=zebra and the third is crossing=zebra - so a report that
    normalised them into one key would assert a tag the way does not have. And a way with no
    markings tag must not be labelled "unmarked": "nobody recorded it" is the statement that
    decides whether the render owes it paint at all.
    """
    assert _markings_label({"crossing:markings": "zebra"}) == "crossing:markings=zebra"
    assert _markings_label({"crossing": "zebra"}) == "crossing=zebra"
    assert _markings_label({"crossing:markings": "lines", "crossing": "uncontrolled"}) == (
        "crossing:markings=lines")
    assert _markings_label({"footway": "crossing"}) == "no markings tag"
    assert _markings_label(None) == "no markings tag"


def test_a_model_with_no_osm_reports_nothing():
    """A stand-in model is vacuously covered, rather than raising.

    The same guard kerbs.py:kerb_openings_from_model carries, for the same reason: plenty of tests
    build a model-shaped object with legs and no OSM at all, and a check that raised on one would
    have to be skipped in exactly the places geometry is easiest to get wrong.
    """
    class NotReallyAModel:
        legs = {}

    assert coverage_gaps(NotReallyAModel(), []) == []


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def test_describe_coverage_says_so_when_nothing_was_dropped():
    """Never the empty string. A clean check that prints nothing looks like a check that did not
    run, which is how the stale OSM cache went unnoticed until cache_summary printed an age every
    build (src/sources/osm_context.py)."""
    report = describe_coverage([])
    assert report.strip()
    assert "nothing dropped" in report.lower()


def test_describe_coverage_names_the_layer_the_count_and_the_examples():
    """A build log line has to be actionable without opening the code: which layer, how many out
    of how many, and enough of them to go and look."""
    gap = Uncovered(layer="crossings", count=6, total=10,
                    examples=[f"crossing way 49 ft long, crossing:markings=zebra ({d} ft out)"
                              for d in (263, 277, 314, 375, 387)])
    report = describe_coverage([gap])
    assert "crossings: 6 of 10" in report
    assert "crossing:markings=zebra" in report
    assert "...and 1 more" in report, "five of six shown, so the sixth has to be accounted for"
