"""A plan sheet may not cover its own design with its own annotation.

The three representations of one proposal - the 2D sheet, the exported geojson, the 3D render -
are one PaintPiece list read three times, and markings.require_every_kind already makes it
impossible to export a marking the plan view has no style for. What nothing pinned was whether
the marking the plan view drew was still THERE afterwards: on Broad & Greenwood's 2.5x two-way
sheet a fifth of the bike lane's green surface lay under white call-out boxes, so the plan view
showed half a facility the render showed whole. Drawn is not the same as visible.

The mechanism is a units mismatch nothing in matplotlib reconciles - type is sized in points,
the street in feet - so the same annotation covers 1.9 ft of ground per point at 1x and 4.7 ft
at 2.5x. That is why the whole-site test runs at BOTH scales: at 1x these labels fit.
"""
import contextlib
import io

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import box

from src.render.labels import LabelPlacer, ft_per_point, label_box_ft
from tests.conftest import SITES, WIDE_FRAME_SCALE, needs_source_data


def _axes(span_ft=400.0, size_in=9.0):
    """One panel framed on `span_ft` of ground - the transform the placer measures against."""
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(0, span_ft)
    ax.set_ylim(0, span_ft)
    return fig, ax


def _drawn_boxes(placer):
    return placer.placed


# --------------------------------------------------------------------- the units mismatch


def test_a_label_is_measured_in_the_frame_the_reader_is_looking_at():
    """The same words cover more ground on a wider sheet, and the placer has to know it.

    This is .claude/SKILLS.md 0b in the label layer: --frame-scale changes what a point is worth
    on the ground, so a placement rule written in feet would place differently on the two sheets
    for no design reason.
    """
    sizes = {}
    for span_ft in (400.0, 1000.0):
        fig, ax = _axes(span_ft)
        try:
            sizes[span_ft] = label_box_ft(ax, "lane 11.0 ft", 6.5)
        finally:
            plt.close(fig)
    narrow, wide = sizes[400.0], sizes[1000.0]
    assert wide[0] == pytest.approx(narrow[0] * 2.5, rel=0.02), (
        f"one label measured {narrow[0]:.1f} ft on a 400 ft frame and {wide[0]:.1f} ft on a "
        f"1000 ft one; it should be exactly 2.5x, because the type did not change size")
    assert wide[1] == pytest.approx(narrow[1] * 2.5, rel=0.02)


def test_ft_per_point_is_zero_before_the_axes_is_framed():
    """The signal that it is too early to place - which is why labels are queued, not drawn."""
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_axes([0, 0, 0, 0])          # no extent at all
    try:
        assert ft_per_point(ax) == 0.0
    finally:
        plt.close(fig)


def test_a_placer_flushed_too_early_draws_nothing_rather_than_guessing():
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_axes([0, 0, 0, 0])
    try:
        placer = LabelPlacer()
        placer.dimension("11.0 ft", (0, 0))
        assert placer.flush(ax) == []
        assert _drawn_boxes(placer) == []
    finally:
        plt.close(fig)


# --------------------------------------------------------------------- pushing clear


def test_a_dimension_over_paint_is_pushed_off_it():
    fig, ax = _axes()
    try:
        paint = box(150, 150, 250, 250)
        placer = LabelPlacer()
        placer.dimension("lane 11.0 ft", (200, 200), toward=(0, 1), fontsize=6.5)
        assert placer.flush(ax, paint) == []
        assert len(_drawn_boxes(placer)) == 1
        assert not _drawn_boxes(placer)[0].intersects(paint), (
            "the label was left standing on the paint it was supposed to be pushed off")
        # Pushed the way it was told to go, not somewhere arbitrary.
        assert _drawn_boxes(placer)[0].centroid.y > 250
    finally:
        plt.close(fig)


def test_two_dimensions_wanting_one_spot_do_not_stack():
    fig, ax = _axes()
    try:
        placer = LabelPlacer()
        placer.dimension("11.0 ft", (200, 200), toward=(0, 1))
        placer.dimension("12.0 ft", (200, 200), toward=(0, 1))
        assert placer.flush(ax) == []
        first, second = _drawn_boxes(placer)
        assert not first.intersects(second), (
            "two labels asked for the same point and both took it")
    finally:
        plt.close(fig)


def test_a_label_with_nowhere_to_go_is_reported_and_still_drawn():
    """Paint everywhere. The label goes back where it belongs and says so.

    Reported rather than dropped, and drawn rather than parked ten heights out with a leader
    across the junction: a label that is not on the drawing has lost the fact it carried, which
    is worse than one the reader can see is crowded.
    """
    fig, ax = _axes()
    try:
        placer = LabelPlacer()
        placer.dimension("lane 11.0 ft", (200, 200), toward=(0, 1))
        violations = placer.flush(ax, box(-1e4, -1e4, 1e4, 1e4))
        assert [v.check for v in violations] == ["label_covers_paint"]
        assert not violations[0].fatal, (
            "a crowded label must not block the 3D export - it is a drafting problem, not "
            "geometry that is wrong")
        assert violations[0].where == (200, 200)
        assert _drawn_boxes(placer)[0].contains(
            box(199.9, 199.9, 200.1, 200.1).centroid), "drawn away from what it labels"
    finally:
        plt.close(fig)


# --------------------------------------------------------------------- prose is keyed, not placed


def test_prose_is_keyed_so_its_size_never_decides_where_the_design_shows():
    """A note leaves a marker at the place and puts the sentence in the block.

    The sentence's length is the whole problem - 40 characters of it covered 235 ft of Broad St -
    so what stands on the ground is the NUMBER, and the number is the same size whatever the
    sentence says.
    """
    fig, ax = _axes()
    try:
        short, long = LabelPlacer(), LabelPlacer()
        short.note("a kerb", "no_parking", (200, 200))
        long.note("a kerb", "OSM parking: no_parking over 0-130 ft -> stalls beyond it, and "
                             "then a great deal more prose about the same kerb", (200, 200))
        for placer in (short, long):
            assert placer.flush(ax) == []
        key_short, key_long = _drawn_boxes(short)[0], _drawn_boxes(long)[0]
        assert key_short.equals(key_long), (
            f"the marker on the ground grew with the prose: {key_short.bounds} vs "
            f"{key_long.bounds}")
    finally:
        plt.close(fig)


def test_the_notes_block_takes_the_corner_with_least_design_under_it():
    fig, ax = _axes()
    try:
        placer = LabelPlacer()
        placer.note("a kerb", "no_parking -> hatched", (200, 200))
        # Three corners paved over; only the top-right is clear.
        paint = box(0, 0, 400, 200).union(box(0, 0, 200, 400))
        assert placer.flush(ax, paint) == []
        block = _drawn_boxes(placer)[-1]
        assert not block.intersects(paint), f"the block landed on paint at {block.bounds}"
        assert block.centroid.x > 200 and block.centroid.y > 200
    finally:
        plt.close(fig)


def test_a_pinned_caption_is_ground_the_notes_block_has_to_avoid():
    """The caption is in axes fraction, so it moves with the frame and not with the street -
    but it still occupies a strip of the panel, and the block printed over it until the caption
    was registered like everything else."""
    fig, ax = _axes()
    try:
        placer = LabelPlacer()
        placer.caption("SIGNALIZED - 4 signal pole(s)", (0.5, 0.005), fontsize=8)
        placer.note("a kerb", "no_parking -> hatched", (200, 380))
        assert placer.flush(ax) == []
        caption, block = _drawn_boxes(placer)[0], _drawn_boxes(placer)[-1]
        assert not block.intersects(caption), (
            f"the notes block {block.bounds} printed over the caption {caption.bounds}")
    finally:
        plt.close(fig)


# --------------------------------------------------------------------- the whole sheet


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_label_covers_the_paint_on_any_sheet(site, request):
    """Every scenario of every site, at 1x and at the scale output/ is actually drawn at.

    BOTH scales, because this defect is invisible at 1x: the labels that buried Greenwood's bike
    lane fit beside it on the narrow sheet. A test at one scale would have passed throughout.
    """
    from src.geometry.treatments import DesignState
    from src.render.frame import FRAME_SCALE_ENV
    from src.render.plan_view import plot_design_state
    from src.site import load_site_scenarios, run_scenario
    from src.geometry.intersection import load_intersection_model

    import os

    quiet = io.StringIO()
    previous = os.environ.get(FRAME_SCALE_ENV)
    try:
        for scale in (1.0, WIDE_FRAME_SCALE):
            os.environ[FRAME_SCALE_ENV] = str(scale)
            with contextlib.redirect_stdout(quiet):
                model = load_intersection_model(site=site)
                scenarios = load_site_scenarios(site)
            names = ["existing", *sorted(n for n in dir(scenarios) if n.startswith("build_"))]
            for name in names:
                with contextlib.redirect_stdout(quiet):
                    state = DesignState.from_model(model)
                    if name != "existing":
                        state = run_scenario(getattr(scenarios, name), state, model)
                    fig, ax = plt.subplots(figsize=(9, 10))
                    try:
                        result = plot_design_state(ax, model, state, name)
                    finally:
                        plt.close(fig)
                covered = [v for v in result.violations if v.check == "label_covers_paint"]
                assert not covered, (
                    f"{site} / {name} at {scale}x: {len(covered)} label(s) with nowhere clear "
                    f"to go, so the sheet hides paint the geojson and the 3D render carry:\n  "
                    + "\n  ".join(v.detail for v in covered))
    finally:
        if previous is None:
            os.environ.pop(FRAME_SCALE_ENV, None)
        else:
            os.environ[FRAME_SCALE_ENV] = previous
