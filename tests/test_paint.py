"""Curbside paint: strips, stall ticks, tapers, and the invariant that keeps them on the road.

Every test here is a bug that shipped, and they share one cause with the traced-curb bugs in
test_traced_curbs.py: code that was correct while a curb was a symmetric offset of the
centerline, and was never revisited once curbs became traced kerbs. A traced kerb starts
13-47 ft out, runs at its own bearing, sometimes carries on 78 ft past the end of the leg,
and is sometimes closer to the centerline than the leg's nominal half-width. Any code that
addresses it by ARC LENGTH, or that trusts the nominal half-width, is wrong on real geometry
and looks fine on a synthetic straight leg - so the legs below are built to have those
properties.
"""
import contextlib
import io

import numpy as np
import pytest
from shapely.geometry import LineString

from src.checks import (PAINT_PAST_CURB_TOLERANCE_FT, PaintInsideTheCurb, SceneContext)
from src.geometry.model import (curb_offsets_at_stations, curb_station_span,
                                curbside_strip_polygon, inset_line_ft,
                                lane_narrowing_polygons_ft, parking_stall_lines_ft,
                                station_offset_many)
from src.geometry.targets import LegSide
from src.geometry.treatments import MarkedParking
from src.geometry.markings import (BUFFER_EDGE_LINE, BUFFER_FILL, CORNER_HATCH_FILL,
                                   DAYLIGHT_EDGE_LINE, DAYLIGHT_FILL,
                                   LANE_EDGE_LINE, LANE_NARROWING_FILL, PARKING_EDGE_LINE,
                                   STALL_DIVIDER, TAPER_LINE)
from src.geometry.paint import PaintPiece
from src.geometry.treatments import DesignState
from src.render.crosswalks import CrosswalkOffset


def crossing_at(station_ft, source="geometric_estimate"):
    """The resolved-crosswalk-offsets dict for a one-leg fixture.

    The real type rather than a bare tuple, so these tests exercise what
    resolve_crosswalk_offsets actually hands the paint builder.
    """
    return {"east": CrosswalkOffset(station_ft, source)}


def a_leg(length_ft=130.0, width_ft=30.0):
    from src.geometry.model import Leg
    return Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)


def traced(leg, side, points):
    """Attach a traced kerb, given (station, offset) pairs in the leg's own frame."""
    sign = 1 if side == "left" else -1
    setattr(leg, f"{side}_curb", LineString([(s, sign * abs(o)) for s, o in points]))
    return leg


def stations_of(geometry, leg):
    coords = (geometry.exterior.coords if geometry.geom_type == "Polygon" else geometry.coords)
    stations, offsets = station_offset_many(leg.centerline, np.asarray(coords, dtype=float))
    return stations, offsets


# --------------------------------------------------------------------------
# The strip itself
# --------------------------------------------------------------------------

def test_a_strip_starts_where_it_is_asked_to():
    """Both boundaries, at the same station.

    The old construction paired substring(curb, start, curb.length) with
    substring(inner, start, inner.length). Those measure arc length along two different
    lines from each line's own start, so the curb edge was cut 20-30 ft from where the inner
    edge was cut and the ring closed with a long diagonal.
    """
    leg = traced(a_leg(), "left", [(20, 15), (60, 15), (130, 15)])
    strip = curbside_strip_polygon(leg, "left", inner_offset_ft=11.0, start_ft=45.0)
    stations, _ = stations_of(strip, leg)
    assert stations.min() == pytest.approx(45.0, abs=0.01)


def test_a_strip_stops_at_the_end_of_the_leg():
    """Several real kerbs are traced 11-78 ft past the 130 ft leg, because the tracing
    carries on down the block. Paint must not follow it out there."""
    leg = traced(a_leg(length_ft=130.0), "left", [(20, 15), (208, 15)])
    strip = curbside_strip_polygon(leg, "left", inner_offset_ft=11.0, start_ft=45.0)
    stations, _ = stations_of(strip, leg)
    assert stations.max() == pytest.approx(130.0, abs=0.01)


def test_a_strip_is_a_strip_not_a_wedge():
    """Its two ends are cross-sections of the leg, not diagonals across it.

    The wedge is the visible failure: hatch lines clipped against a triangular tail come out
    as short fragments fanning around, which is what "sheared in half" looked like.
    """
    leg = traced(a_leg(), "left", [(20, 15), (75, 16), (130, 15)])
    strip = curbside_strip_polygon(leg, "left", inner_offset_ft=11.0, start_ft=40.0)
    stations, _ = stations_of(strip, leg)
    stations = stations[:-1]   # exterior rings repeat their first vertex to close
    # Exactly two vertices at each end - one per boundary - and nothing beyond them.
    assert sum(abs(stations - 40.0) < 0.01) == 2
    assert sum(abs(stations - 130.0) < 0.01) == 2
    assert stations.min() == pytest.approx(40.0) and stations.max() == pytest.approx(130.0)


def test_paint_never_crosses_a_kerb_traced_inside_the_nominal_width():
    """broad_st_east's left kerb is traced at 22.7 ft against a nominal half-width of 24.2.

    Anything sized off the nominal figure sits 1.5 ft over it. The nominal width is a
    summary of the street; the tracing is where the kerb is.
    """
    leg = traced(a_leg(width_ft=30.0), "left", [(20, 8.0), (130, 8.0)])   # nominal half 15
    strip = curbside_strip_polygon(leg, "left", inner_offset_ft=11.0, start_ft=40.0)
    if strip is not None:
        _stations, offsets = stations_of(strip, leg)
        assert offsets.max() <= 8.0 + 1e-6


def test_a_strip_follows_a_kerb_that_is_not_parallel_to_the_centerline():
    """The outer edge is the traced kerb, at whatever offset it actually has."""
    leg = traced(a_leg(), "left", [(20, 15), (75, 19), (130, 17)])
    strip = curbside_strip_polygon(leg, "left", inner_offset_ft=11.0, start_ft=40.0)
    stations, offsets = stations_of(strip, leg)
    at_75 = offsets[np.abs(stations - 75.0) < 1.5]
    assert at_75.max() == pytest.approx(19.0, abs=0.2)


def test_no_strip_where_the_side_was_never_traced():
    leg = a_leg()
    leg.left_curb = None
    assert curbside_strip_polygon(leg, "left", 11.0, 40.0) is None
    assert curb_station_span(leg, "left") is None
    assert inset_line_ft(leg, "left", 11.0, 40.0) is None


def test_no_strip_where_the_start_is_past_the_end_of_the_leg():
    """W Broad & Louellen's acute Y: leg_clearance_ft comes out at 133 ft on a 130 ft leg."""
    leg = traced(a_leg(length_ft=130.0), "left", [(20, 15), (130, 15)])
    assert curbside_strip_polygon(leg, "left", 11.0, start_ft=133.0) is None


# --------------------------------------------------------------------------
# Stations, not arc length
# --------------------------------------------------------------------------

def test_stall_dividers_land_on_their_stations():
    """A curved leg's offset curve has a different arc length from the centerline, so
    offset_curve(x).interpolate(d) is not station d - the ticks drifted along the leg."""
    from src.geometry.model import Leg

    curve = LineString([(x, 0.12 * x) for x in np.linspace(0, 130, 40)])
    leg = Leg(name="bend", centerline=curve, curb_to_curb_ft=30.0)
    for side in ("left", "right"):
        sign = 1 if side == "left" else -1
        from src.geometry.model import _point_at
        setattr(leg, f"{side}_curb",
                LineString([_point_at(curve, s, sign * 15.0) for s in np.linspace(5, 130, 30)]))

    dividers = parking_stall_lines_ft(leg, "left", depth_ft=8.0, stall_length_ft=22.0,
                                       start_ft=40.0, curb_offset_ft=0.0)
    assert dividers
    for i, divider in enumerate(dividers):
        stations, _ = stations_of(divider, leg)
        expected = 40.0 + i * 22.0
        assert stations.min() == pytest.approx(expected, abs=0.05)
        assert stations.max() == pytest.approx(expected, abs=0.05), "a divider is one cross-section"


def test_a_stall_divider_does_not_reach_past_the_kerb():
    leg = traced(a_leg(width_ft=30.0), "left", [(20, 9.0), (130, 9.0)])   # nominal half 15
    for divider in parking_stall_lines_ft(leg, "left", depth_ft=8.0, stall_length_ft=22.0,
                                           start_ft=40.0, curb_offset_ft=0.0):
        _stations, offsets = stations_of(divider, leg)
        assert offsets.max() <= 9.0 + 1e-6


def test_curb_offsets_are_read_at_the_station_asked_for():
    leg = traced(a_leg(), "left", [(20, 15), (70, 20), (130, 15)])
    got = curb_offsets_at_stations(leg, "left", np.array([20.0, 70.0, 130.0]))
    assert list(np.round(got, 6)) == [15.0, 20.0, 15.0]


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------

def a_state(legs):
    """A DesignState over these legs, with every treatment field at its default.

    The real dataclass, not a stub of it. There was a FakeState here and each test then
    assigned the six or seven treatment dicts curbside_paint_ft happens to read - which meant
    adding a field to DesignState broke these tests with an AttributeError from inside the
    builder instead of a failure about behaviour, and a test could silently stop covering a
    field nobody remembered to add. DesignState defaults them all.
    """
    return DesignState(legs=legs, corner_fillets={})


def test_paint_over_the_curb_is_a_violation():
    leg = traced(a_leg(), "left", [(20, 15), (130, 15)])
    over = LineString([(40, 18), (120, 18)])            # 3 ft outside the kerb
    violations = PaintInsideTheCurb().run(SceneContext(state=a_state({"east": leg}),
                                              paint=[PaintPiece(LANE_EDGE_LINE, over, "east", "left")]))
    assert len(violations) == 1
    assert violations[0].check == "paint_over_the_curb"
    assert "3.0 ft past" in violations[0].detail


def test_paint_meeting_the_curb_is_fine():
    """A curbside marking touches the kerb by definition - this is not a clearance check."""
    leg = traced(a_leg(), "left", [(20, 15), (130, 15)])
    at_the_kerb = LineString([(40, 15), (120, 15)])
    assert not PaintInsideTheCurb().run(SceneContext(state=a_state({"east": leg}),
                                            paint=[PaintPiece(LANE_EDGE_LINE, at_the_kerb, "east", "left")]))


def test_the_violation_is_measured_against_the_traced_kerb_not_the_nominal_width():
    """The whole point. Nominal half-width 15 ft, kerb actually traced at 9 ft: paint at
    12 ft is inside the nominal road and 3 ft up on the footway."""
    leg = traced(a_leg(width_ft=30.0), "left", [(20, 9.0), (130, 9.0)])
    paint = LineString([(40, 12), (120, 12)])
    violations = PaintInsideTheCurb().run(SceneContext(state=a_state({"east": leg}),
                                              paint=[PaintPiece(LANE_EDGE_LINE, paint, "east", "left")]))
    assert violations, "measured against the nominal half-width this passes, and it should not"


def test_a_corner_treatment_is_not_measured_against_one_leg():
    """corner_hatch_fill/apron span the corner between two legs, so neither side applies."""
    leg = traced(a_leg(), "left", [(20, 15), (130, 15)])
    way_outside = LineString([(40, 60), (120, 60)])
    assert not PaintInsideTheCurb().run(SceneContext(state=a_state({"east": leg}),
                                            paint=[PaintPiece(CORNER_HATCH_FILL, way_outside)]))


def test_the_tolerance_allows_sampling_noise_and_nothing_more():
    leg = traced(a_leg(), "left", [(20, 15), (130, 15)])
    just_inside = LineString([(40, 15 + PAINT_PAST_CURB_TOLERANCE_FT / 2), (120, 15)])
    just_outside = LineString([(40, 15 + PAINT_PAST_CURB_TOLERANCE_FT * 3), (120, 15)])
    state = a_state({"east": leg})
    assert not PaintInsideTheCurb().run(SceneContext(state=state, paint=[PaintPiece(BUFFER_FILL, just_inside, "east", "left")]))
    assert PaintInsideTheCurb().run(SceneContext(state=state, paint=[PaintPiece(BUFFER_FILL, just_outside, "east", "left")]))


def test_the_right_side_is_measured_against_the_right_kerb():
    """Offsets are signed, so comparing them without taking absolute values passes anything
    on the right-hand side of the leg."""
    leg = traced(a_leg(), "right", [(20, 15), (130, 15)])
    over = LineString([(40, -18), (120, -18)])
    assert PaintInsideTheCurb().run(SceneContext(state=a_state({"east": leg}),
                                        paint=[PaintPiece(LANE_EDGE_LINE, over, "east", "right")]))


# --------------------------------------------------------------------------
# The strip builders go through the same code
# --------------------------------------------------------------------------

def test_lane_narrowing_strips_stay_inside_the_kerb():
    leg = traced(a_leg(width_ft=30.0), "left", [(20, 12.0), (75, 13.0), (130, 12.0)])
    leg = traced(leg, "right", [(20, 16.0), (130, 16.0)])
    state = a_state({"east": leg})
    pieces = [PaintPiece(LANE_NARROWING_FILL, poly, "east", side)
              for side in ("left", "right")
              for poly in lane_narrowing_polygons_ft(leg, 4.0, start_left_ft=40.0,
                                                      start_right_ft=40.0, sides=(side,))]
    assert len(pieces) == 2
    assert not PaintInsideTheCurb().run(SceneContext(state=state, paint=pieces))


# --------------------------------------------------------------------------
# Hatch phase: why strokes looked sheared at a seam
# --------------------------------------------------------------------------

def test_adjacent_pieces_hatch_in_phase():
    """A buffer is not one polygon - the straight run, the taper, and the offcuts left by
    clipping around a crossing are hatched separately.

    Phasing each family off its own bounding-box centre gave every piece an independent
    stroke position, so at each seam the strokes stepped sideways by a fraction of the
    spacing and read as one stroke sheared into two offset halves.
    """
    from shapely.geometry import box

    from src.geometry.model import hatch_lines_ft

    left, right = box(0, 0, 40, 20), box(40, 0, 90, 20)   # share the edge at x=40
    origin = (7.3, 11.9)                                   # deliberately not on the grid
    def offsets(strokes, n):
        # Where each stroke sits along the hatch normal, relative to the shared origin.
        return sorted(round(float(np.dot(np.asarray(s.coords[0]) - origin, n)), 6) for s in strokes)

    theta = np.radians(45.0)
    n = np.array([-np.sin(theta), np.cos(theta)])
    a = offsets(hatch_lines_ft(left, 5.0, 45.0, phase_origin=origin), n)
    b = offsets(hatch_lines_ft(right, 5.0, 45.0, phase_origin=origin), n)
    for value in a + b:
        assert abs(value % 5.0) < 1e-6 or abs(value % 5.0 - 5.0) < 1e-6, \
            "every stroke sits on a whole multiple of the spacing from the shared origin"


def test_hatching_reaches_a_polygon_far_from_the_phase_origin():
    """State-plane feet put these junctions ~419,000 ft east and ~567,000 ft north of the
    origin. A hatch family anchored at the origin has to be EXTENDED to the polygon, not
    merely positioned at the right distance along the normal - centring each segment on the
    origin produced zero strokes at every real site while every synthetic test still passed.
    """
    from shapely.geometry import box

    from src.geometry.model import hatch_lines_ft

    far = box(419100, 566700, 419160, 566760)
    assert hatch_lines_ft(far, 8.0, 45.0, phase_origin=(419130.0, 566730.0))
    assert hatch_lines_ft(far, 8.0, 45.0, phase_origin=(0.0, 0.0)), \
        "a distant phase origin must still produce paint"


# --------------------------------------------------------------------------
# Paint colliding with other paint
# --------------------------------------------------------------------------

def test_two_fills_over_the_same_ground_is_a_collision():
    """The daylighting bug: a hydrant's no-parking zone (18.9-38.9 ft on broad_st_west) sat
    entirely inside the junction's (0-45.7 ft) and both got hatched - 98 sq ft painted twice.

    Nothing caught it, because every other invariant checks paint against the STREET (the
    kerb, the roadway, the crosswalk) and none checked paint against other paint.
    """
    from shapely.geometry import box

    from src.checks import MarkingsDoNotCollide, SceneContext

    outer = PaintPiece(DAYLIGHT_FILL, box(0, 0, 50, 10), "east", "left")
    inner = PaintPiece(DAYLIGHT_FILL, box(19, 0, 39, 10), "east", "left")
    violations = MarkingsDoNotCollide().run(SceneContext(paint=[outer, inner]))
    assert len(violations) == 1
    assert violations[0].check == "markings_collide"
    assert "200 sq ft" in violations[0].detail


def test_fills_that_merely_abut_are_fine():
    """The junction zone ends exactly where the stalls begin - by design, not by accident."""
    from shapely.geometry import box

    from src.checks import MarkingsDoNotCollide, SceneContext

    a = PaintPiece(DAYLIGHT_FILL, box(0, 0, 50, 10), "east", "left")
    b = PaintPiece(BUFFER_FILL, box(50, 0, 90, 10), "east", "left")
    assert not MarkingsDoNotCollide().run(SceneContext(paint=[a, b]))


def test_two_lines_down_the_same_stretch_is_a_collision():
    """daylight_edge_line and parking_edge_line sit at the SAME offset - the lane edge - and
    are kept apart only by their station ranges. If a range ever overlaps, both get painted."""
    from src.checks import MarkingsDoNotCollide, SceneContext

    a = PaintPiece(DAYLIGHT_EDGE_LINE, LineString([(0, 11), (60, 11)]), "east", "left")
    b = PaintPiece(PARKING_EDGE_LINE, LineString([(40, 11), (100, 11)]), "east", "left")
    violations = MarkingsDoNotCollide().run(SceneContext(paint=[a, b]))
    assert violations and "run along each other" in violations[0].detail


def test_a_stall_divider_crossing_the_lane_edge_is_not_a_collision():
    """A divider meets the lane edge at right angles - that is what a divider does. A check
    that flagged every touch would fire on every correct drawing and get switched off."""
    from src.checks import MarkingsDoNotCollide, SceneContext

    edge = PaintPiece(PARKING_EDGE_LINE, LineString([(0, 11), (100, 11)]), "east", "left")
    divider = PaintPiece(STALL_DIVIDER, LineString([(40, 11), (40, 19)]), "east", "left")
    assert not MarkingsDoNotCollide().run(SceneContext(paint=[edge, divider]))


def test_a_hatch_stroke_ending_on_its_own_boundary_line_is_not_a_collision():
    """Measured on the real geometry: a buffer stroke ends at offset -19.000000000 and the
    buffer's edge line is at -19.0. It touches its own boundary, which is correct."""
    from src.checks import MarkingsDoNotCollide, SceneContext

    edge = PaintPiece(BUFFER_EDGE_LINE, LineString([(0, 19), (100, 19)]), "east", "left")
    stroke = PaintPiece(TAPER_LINE, LineString([(45, 22.2), (48.9, 19.0)]), "east", "left")
    assert not MarkingsDoNotCollide().run(SceneContext(paint=[edge, stroke]))


# --------------------------------------------------------------------------
# Stopping before the crossing, rather than being cut off by it
# --------------------------------------------------------------------------

def test_the_taper_aims_past_a_skewed_crossing_on_the_side_it_reaches_furthest():
    """A band pivots about its centre, so skew swings one end further along the leg.

    broad_st_west's crossing, skewed 7.1 degrees, reaches station 28.6 on the left kerb and
    19.2 on the right. Aiming both sides at the centre offset plus a fixed 5 ft put the left
    taper 2.9 ft inside the crossing, where the backstop clip chopped it off square - which
    is what "the hatching is conflicting with the crosswalk" looked like.
    """
    from shapely.geometry import Polygon

    from src.geometry.paint import leg_anchors
    from src.render.crosswalks import CROSSWALK_CLEARANCE_FT

    leg = traced(a_leg(width_ft=30.0), "left", [(10, 15), (130, 15)])
    leg = traced(leg, "right", [(10, 15), (130, 15)])
    state = a_state({"east": leg})
    # Skewed: its far edge runs from (30, 15) at the left kerb to (20, -15) at the right.
    skewed = Polygon([(24, 15), (30, 15), (20, -15), (14, -15)])

    left = leg_anchors(state, "east", "left", crossing_at(25.0), skewed, inner_offset_ft=11.0)
    right = leg_anchors(state, "east", "right", crossing_at(25.0), skewed, inner_offset_ft=11.0)
    assert left.target_ft == pytest.approx(30.0 + CROSSWALK_CLEARANCE_FT, abs=0.5)
    assert right.target_ft == pytest.approx(21.3 + CROSSWALK_CLEARANCE_FT, abs=0.5)
    assert left.target_ft > right.target_ft + 8, "the two sides cannot share one target"


def test_the_reach_is_measured_in_the_strip_the_paint_occupies():
    """A skewed band reaches further along the leg near the CENTRELINE than at the kerb, and
    curbside paint never goes near the centreline. Measuring across the whole half-road made
    the target 6-8 ft too conservative and opened a visible gap before the crossing."""
    from shapely.geometry import Polygon

    from src.render.crosswalks import crosswalk_reach_on_leg_side_ft

    leg = traced(a_leg(width_ft=30.0), "left", [(10, 15), (130, 15)])
    # Far edge slants from station 40 at the centreline back to 25 at the kerb.
    skewed = Polygon([(15, 0), (40, 0), (25, 15), (15, 15)])
    whole_half = crosswalk_reach_on_leg_side_ft(leg, "left", skewed, inner_offset_ft=0.0)
    paint_strip = crosswalk_reach_on_leg_side_ft(leg, "left", skewed, inner_offset_ft=11.0)
    assert whole_half == pytest.approx(40.0, abs=0.5)
    assert paint_strip == pytest.approx(29.0, abs=1.0)
    assert paint_strip < whole_half


def test_the_taper_also_clears_the_cross_streets_crossing():
    """A taper curving into the corner runs into the INTERSECTING leg's crossing, which has
    nothing to do with this leg's own offset. That was the last 30 sq ft of overlap at
    broad_st_west, and no per-leg figure finds it."""
    from shapely.geometry import Polygon

    from src.geometry.paint import leg_anchors

    leg = traced(a_leg(width_ft=30.0), "left", [(10, 15), (130, 15)])
    state = a_state({"east": leg})
    state.corner_fillets = {}
    own = Polygon([(22, 15), (28, 15), (28, -15), (22, -15)])
    # The cross street's crossing, lying across this leg's left side further out.
    cross = Polygon([(35, 8), (45, 8), (45, 15), (35, 15)])

    own_only = leg_anchors(state, "east", "left", crossing_at(25.0), own, inner_offset_ft=4.0)
    both = leg_anchors(state, "east", "left", crossing_at(25.0), own.union(cross),
                        inner_offset_ft=4.0)
    assert both.target_ft > own_only.target_ft
    assert both.target_ft >= 45.0, "it has to clear the far edge of the cross crossing"


def test_no_crossing_geometry_falls_back_to_the_offset():
    from src.geometry.paint import leg_anchors
    from src.render.crosswalks import CROSSWALK_CLEARANCE_FT

    leg = traced(a_leg(), "left", [(10, 15), (130, 15)])
    state = a_state({"east": leg})
    state.corner_fillets = {}
    at = leg_anchors(state, "east", "left", crossing_at(25.0), None)
    assert at.target_ft == pytest.approx(25.0 + CROSSWALK_CLEARANCE_FT)


# --------------------------------------------------------------------------
# Continental crossings reaching the kerb
# --------------------------------------------------------------------------

def test_a_continental_crossing_reaches_the_kerb():
    """Its outermost bar's OUTER EDGE lands on the kerb-to-kerb span, not its centre.

    The renderer inset a flat 1.5 m before laying the bars out, losing ~2.5 ft at each kerb.
    add_crosswalk_lines carried the same fudge and lost it when crossings were made to reach
    the kerb; the bar layout - which every continental crossing in every proposal uses - was
    missed, so the simple crossings reached the kerb and the continental ones did not.
    """
    from src.render.crosswalks import (CONTINENTAL_BAR_GAP_FT, CONTINENTAL_BAR_WIDTH_FT,
                                        continental_bar_count)

    period = CONTINENTAL_BAR_WIDTH_FT + CONTINENTAL_BAR_GAP_FT
    for span_ft in (18.0, 25.1, 30.0, 48.4, 55.5, 75.7):
        n = continental_bar_count(span_ft)
        # The renderer spreads the leftover across the gaps, so the outermost bars' outer
        # edges land exactly on the span. This is that layout, in feet.
        centre_to_centre = span_ft - CONTINENTAL_BAR_WIDTH_FT
        pitch = centre_to_centre / (n - 1)
        painted = centre_to_centre + CONTINENTAL_BAR_WIDTH_FT
        assert painted == pytest.approx(span_ft), f"{span_ft} ft: does not reach the kerb"
        assert n > 1, f"{span_ft} ft: a road crossing needs more than one bar"
        assert pitch >= period - 1e-9, f"{span_ft} ft: gaps were squeezed below the nominal pitch"
        assert pitch < 2 * period, f"{span_ft} ft: gaps stretched so far the bars read as sparse"


def test_the_old_inset_really_did_fall_short():
    """Guards the regression rather than just the fix: the previous formula is reproduced
    here so the test fails if someone reinstates it."""
    from src.render.coords import FT_TO_M
    from src.render.crosswalks import (CONTINENTAL_BAR_GAP_FT, CONTINENTAL_BAR_WIDTH_FT,
                                        continental_bar_count)

    span_ft = 48.4
    period = CONTINENTAL_BAR_WIDTH_FT + CONTINENTAL_BAR_GAP_FT
    old_n = max(int(max(span_ft - 1.5 / FT_TO_M, 0.5) / period), 1)
    old_painted = (old_n - 1) * period + CONTINENTAL_BAR_WIDTH_FT
    assert continental_bar_count(span_ft) > old_n
    assert span_ft > old_painted + 3.0, "the fix has to actually widen the crossing"


def test_a_narrow_crossing_still_gets_one_bar():
    from src.render.crosswalks import continental_bar_count

    assert continental_bar_count(0.5) == 1
    assert continental_bar_count(0.0) == 1


def test_the_reach_does_not_walk_up_a_corner_return():
    """The bug that put crossings on the sidewalk.

    A traced kerb includes the corner return, which flares away from the road. Casting a ray
    from the crossing's centre until it crosses that LINE runs diagonally into the flare and
    stops far outside the carriageway - 39.8 ft on a street 27.8 ft wide at broad_st_west, so
    the end bars were painted 12 ft up the corner. The test is instead "is this point still
    inside the roadway", asked per-station in the leg's frame, which never walks up a return
    because the return is at a different station from the point beside it.
    """
    from src.render.crosswalks import crosswalk_reach_to_curbs_ft

    leg = a_leg(width_ft=30.0, length_ft=130.0)
    # A straight 15 ft kerb that flares out to 40 ft as it turns the corner behind us.
    leg.left_curb = LineString([(2, 40), (8, 22), (14, 15), (60, 15), (130, 15)])
    leg.right_curb = LineString([(14, -15), (130, -15)])

    centre = (25.0, 0.0)
    square = crosswalk_reach_to_curbs_ft(leg, centre, (0.0, 1.0), (1.0, 0.0), 6.0)
    assert square[0] == pytest.approx(15.0, abs=0.3), "square-on: straight out to the kerb"

    # Skewed 15 degrees, so the ray leans back toward the corner flare.
    import math
    a = math.radians(15.0)
    normal = (-math.sin(a), math.cos(a))
    along = (math.cos(a), math.sin(a))
    skewed = crosswalk_reach_to_curbs_ft(leg, centre, normal, along, 6.0)
    assert skewed[0] < 20.0, f"the ray walked up the corner return to {skewed[0]:.1f} ft"


def test_the_reach_stops_inside_the_roadway_not_on_its_boundary():
    """Off-by-one worth a test: returning the first step OUTSIDE puts the paint's outer edge
    exactly on the boundary, which the kerb test reads as inside and the pavement test reads
    as outside. The bars kept landing a hair over the kerb."""
    from shapely.geometry import box

    from src.render.crosswalks import crosswalk_reach_to_curbs_ft

    leg = a_leg(width_ft=30.0)
    leg.left_curb = LineString([(0, 15), (130, 15)])
    leg.right_curb = LineString([(0, -15), (130, -15)])
    roadway = box(0, -15, 130, 15)
    left, _right = crosswalk_reach_to_curbs_ft(leg, (60.0, 0.0), (0.0, 1.0), (1.0, 0.0), 6.0,
                                                roadway)
    assert left < 15.0, "must stop inside the pavement, which excludes its own boundary"
    assert left > 14.5, "but not by more than the sampling step"


# --------------------------------------------------------------------------
# Square ends vs curved tapers
# --------------------------------------------------------------------------

def test_a_gentle_taper_curves_and_a_hairpin_does_not():
    """Measured at Broad & Greenwood. Greenwood's lane-narrowing buffers run 1.5 ft of depth
    across 8-11 ft of station and read well; Broad St's parking buffers had to swing 13-17 ft
    across 0-5.6 ft, which is a hairpin, not a taper."""
    from src.geometry.paint import LegAnchors, tapers_cleanly

    greenwood = LegAnchors(anchor_ft=50.6, target_ft=41.5)      # 9.1 ft of run
    assert tapers_cleanly(1.5, greenwood)

    broad_east = LegAnchors(anchor_ft=31.6, target_ft=29.5)     # 2.1 ft of run
    assert not tapers_cleanly(13.2, broad_east)

    broad_west = LegAnchors(anchor_ft=33.6, target_ft=33.6)     # no run at all
    assert not tapers_cleanly(16.8, broad_west)


def test_a_daylight_zone_is_square_ended():
    """Not a fallback: a keep-clear block is painted square on a real street, whatever the
    geometry would allow. A curve is a claim about a lane transition, which this is not.

    Square means its ends are cross-sections of the leg - exactly two vertices at each end
    station, and nothing beyond them.
    """
    from src.geometry.paint import curbside_paint_ft

    leg = traced(a_leg(width_ft=40.0, length_ft=130.0), "left", [(10, 20), (130, 20)])
    leg = traced(leg, "right", [(10, 20), (130, 20)])
    state = a_state({"east": leg})
    # Through apply: the paint builder dispatches to the treatments a design records
    # (Treatment.paint), and since the collapse there is no longer any parking-zone dict to poke
    # instead - a marked lane exists exactly when someone applied a MarkedParking.
    state = state.apply(MarkedParking(LegSide("east", "left"), depth_ft=8.0, stall_length_ft=22.0,
                                       curb_offset_ft=1.0))

    paint = curbside_paint_ft(state, crossing_at(20.0), None)
    fills = [p for p in paint if p.kind is DAYLIGHT_FILL]
    assert fills, "the daylight zone has to be drawn at all"
    assert not [p for p in paint if "taper" in p.kind.name], "a keep-clear zone has no taper"

    for piece in fills:
        stations, _ = stations_of(piece.geometry, leg)
        stations = stations[:-1]          # the ring repeats its first vertex to close
        lo, hi = stations.min(), stations.max()
        assert sum(abs(stations - lo) < 0.01) == 2
        assert sum(abs(stations - hi) < 0.01) == 2


def _offset_span_ft(piece) -> float:
    """How far across the kerbside strip a line reaches, on these synthetic straight legs.

    The legs a_leg builds run along x with their offsets in y, so a line drawn ALONG the kerb has
    no y-extent and one drawn ACROSS it (a rim, a stall divider) does. That is what identifies a
    rim now that it is painted in the same kind as the zone's own edge line."""
    ys = [y for _x, y in piece.geometry.coords]
    return max(ys) - min(ys)


def test_a_fill_cut_by_a_crossing_gets_a_line_along_the_cut():
    """A hatched zone is outlined, and the outline carries on around the end where the
    crossing cuts it - the diagonal that finishes the zone off against the crossing on a real
    street. Without it the zone just stopped, with hatch strokes ending in mid-air.
    """
    from shapely.geometry import box

    from src.geometry.paint import curbside_paint_ft

    leg = traced(a_leg(width_ft=40.0, length_ft=130.0), "left", [(10, 20), (130, 20)])
    leg = traced(leg, "right", [(10, 20), (130, 20)])
    state = a_state({"east": leg})
    # Through apply: the paint builder dispatches to the treatments a design records
    # (Treatment.paint), and since the collapse there is no longer any parking-zone dict to poke
    # instead - a marked lane exists exactly when someone applied a MarkedParking.
    state = state.apply(MarkedParking(LegSide("east", "left"), depth_ft=8.0, stall_length_ft=22.0,
                                       curb_offset_ft=1.0))
    band = box(18, -20, 24, 20)

    paint = curbside_paint_ft(state, crossing_at(21.0), None, {"east": band},
                               marked_crosswalks={"east"})
    # A rim is found by SHAPE, not by a kind of its own: it runs across the zone's depth where
    # every other line on this kerb runs along it. That is the point of the change that removed
    # `crossing_rim_line` - the rim is the zone's own edge line continued around the cut, so it
    # cannot be identified by colour, and a test that asked for a dedicated kind was really
    # asking for the drawing to be assembled out of differently-coloured pieces.
    rims = [p for p in paint if p.kind.is_line and p.kind is not STALL_DIVIDER
            and _offset_span_ft(p) > 5.0]
    assert rims, "no line drawn where the zone meets the crossing"
    for r in rims:
        assert r.geometry.length > 5.0
        assert r.geometry.distance(band) < 1.5
        assert r.kind is DAYLIGHT_EDGE_LINE, (
            f"the rim is drawn as {r.kind}, not as the edge line of the zone it closes - it has "
            f"to be the same paint continued or the outline reads as two different markings")


def test_no_rim_where_there_is_no_crossing_to_cut_against():
    from src.geometry.paint import curbside_paint_ft

    leg = traced(a_leg(width_ft=40.0, length_ft=130.0), "left", [(10, 20), (130, 20)])
    leg = traced(leg, "right", [(10, 20), (130, 20)])
    state = a_state({"east": leg})
    # Through apply: the paint builder dispatches to the treatments a design records
    # (Treatment.paint), and since the collapse there is no longer any parking-zone dict to poke
    # instead - a marked lane exists exactly when someone applied a MarkedParking.
    state = state.apply(MarkedParking(LegSide("east", "left"), depth_ft=8.0, stall_length_ft=22.0,
                                       curb_offset_ft=1.0))
    paint = curbside_paint_ft(state, crossing_at(21.0), None)
    assert not [p for p in paint if p.kind.is_line and p.kind is not STALL_DIVIDER
                and _offset_span_ft(p) > 5.0], (
        "a line runs across the zone's depth with nothing there to have cut it")


def test_sampled_polylines_are_rendered_as_polylines_not_chords():
    """add_paint_line(line[0], line[-1]) draws the straight chord between a polyline's
    endpoints and silently discards every vertex between them.

    The lane-edge lines follow the traced kerb and are sampled every 2 ft, so the chord is
    not the line: it deviated 0.7 ft on Broad St's daylight zone, which both pulled the
    painted edge inside the 11 ft lane it is supposed to mark and lifted it off the hatching
    it is supposed to bound. add_paint_polyline exists precisely for this and its own
    docstring warns about it, so this is a static guard rather than a comment.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "blender" / "blender_scene.py").read_text()
    # Lists whose entries come from inset_line_ft / taper arcs - many vertices apiece.
    for name in ("lane_narrowing_edge", "lane_narrowing_taper", "parking_edge",
                  "parking_buffer_edge", "parking_buffer_taper"):
        chord = re.search(rf'add_paint_line\(f"{name}_\{{i\}}", line\[0\], line\[-1\]', source)
        assert chord is None, f"{name} is drawn as a chord; it is a sampled polyline"
        assert f'add_paint_polyline(f"{name}_{{i}}"' in source, f"{name} is not drawn at all"


# --------------------------------------------------------------------------
# Paint has width, and it comes out of the treatment
# --------------------------------------------------------------------------

def test_the_lane_edge_line_sits_outside_the_lane_it_marks():
    """An edge line CENTRED on the 11 ft mark puts half its own body inside the lane.

    Every approach at every site measured 10.59 ft of clear asphalt against an 11.0 ft
    target, and the design arithmetic said 11.0 the whole time - the numbers were right and
    the paint was in the wrong place.
    """
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT, lane_edge_stripes

    line_ft, fill_ft = lane_edge_stripes(5.0)
    assert line_ft == pytest.approx(5.0 - LANE_EDGE_LINE_WIDTH_FT / 2)
    assert fill_ft == pytest.approx(5.0 - LANE_EDGE_LINE_WIDTH_FT)
    assert fill_ft < line_ft, "the hatching starts outside the line, not under it"


def test_a_treatment_thinner_than_its_own_line_collapses_rather_than_going_negative():
    from src.geometry.paint import lane_edge_stripes

    assert lane_edge_stripes(0.1) == (0.0, 0.0)


def test_a_clamped_line_stays_inside_the_kerb_instead_of_straddling_it():
    """Where the road is narrower than the offset asked for, the line clamps to the kerb.
    Clamping its AXIS there hangs half the paint over the kerb - measured at W Broad's
    north-east approach, whose right kerb comes to 7.2 ft of the NJDOT alignment."""
    leg = traced(a_leg(width_ft=30.0), "left", [(10, 7.2), (130, 7.2)])
    line = inset_line_ft(leg, "left", 11.0, 20.0, keep_inside_ft=0.41)
    _stations, offsets = stations_of(line, leg)
    assert offsets.max() == pytest.approx(7.2 - 0.41, abs=1e-6)

    flush = inset_line_ft(leg, "left", 11.0, 20.0)
    assert stations_of(flush, leg)[1].max() == pytest.approx(7.2, abs=1e-6)


def test_the_travel_lane_check_measures_against_the_real_kerb():
    """W Broad's north-east approach has the alignment 7.2 ft from its right kerb and 25-31 ft
    from its left. There is no 11 ft lane to protect on that side, so paint clamped to the
    kerb is correct - measuring against the NOMINAL half-width called it a violation."""
    from src.checks import PaintClearOfTheTravelLane, SceneContext

    leg = traced(a_leg(width_ft=30.0), "left", [(10, 7.2), (130, 7.2)])   # nominal half 15
    state = a_state({"east": leg})
    at_the_kerb = PaintPiece(LANE_EDGE_LINE, LineString([(30, 6.79), (120, 6.79)]),
                              "east", "left")
    assert not PaintClearOfTheTravelLane().run(SceneContext(state=state, paint=[at_the_kerb]))

    roomy = traced(a_leg(width_ft=30.0), "left", [(10, 15.0), (130, 15.0)])
    intruding = PaintPiece(LANE_EDGE_LINE, LineString([(30, 11.0), (120, 11.0)]),
                            "east", "left")
    violations = PaintClearOfTheTravelLane().run(SceneContext(state=a_state({"east": roomy}), paint=[intruding]))
    assert violations and violations[0].check == "paint_in_the_travel_lane"


# --------------------------------------------------------------------------
# Centring the design on the road rather than on the route alignment
# --------------------------------------------------------------------------

def _centred(leg, name="east"):
    """Run the model's width-and-centre fit on one leg and hand back the result.

    Calls _resize_and_centre_from_traced_kerbs directly rather than through
    load_intersection_model: this is about the arithmetic on a known cross-section, and a
    real junction supplies neither a known one nor a fast one.
    """
    from src.geometry.intersection import _resize_and_centre_from_traced_kerbs

    # `traced` attaches the kerb geometry; the fit also wants the leg to SAY which sides are
    # traced, which is what _apply_traced_curb_lines does for it on the real path.
    leg.traced_sides = {side for side in ("left", "right")
                        if getattr(leg, f"{side}_curb", None) is not None}
    legs = {name: leg}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        _resize_and_centre_from_traced_kerbs(legs, {})
    return legs[name], out.getvalue()


def test_recentring_splits_the_leftover_evenly_between_the_kerbs():
    """The leg centerline is NJDOT's ROUTE alignment, which says where the route goes, not
    where the middle of the carriageway is. Greenwood Ave south's kerbs sit 12.6 and 18.2 ft
    off it - two lanes each exactly at target, on a road visibly not symmetrical about its
    own centre line.
    """
    import numpy as np

    from src.geometry.model import curb_offsets_at_stations

    leg = a_leg(width_ft=30.0, length_ft=130.0)
    leg = traced(leg, "left", [(10, 12.6), (130, 12.6)])
    leg = traced(leg, "right", [(10, 18.2), (130, 18.2)])

    out, _log = _centred(leg)
    stations = np.linspace(40, 130, 20)
    left = np.abs(curb_offsets_at_stations(out, "left", stations)).min()
    right = np.abs(curb_offsets_at_stations(out, "right", stations)).min()
    assert abs(left - right) < 0.1, f"still lopsided: {left:.1f} vs {right:.1f}"
    assert left == pytest.approx((12.6 + 18.2) / 2, abs=0.1)


def test_the_width_is_the_distance_between_the_two_kerbs_not_double_either_one():
    """The bug this replaced: the width came from the NEAREST kerb, doubled. On a leg whose
    alignment is off centre that is neither kerb-to-kerb distance - 12.6 and 18.2 ft apart
    is a 30.8 ft street, not the 25.2 ft doubling the near one gives. Every leg at every one
    of the four junctions was reported too narrow this way, by 1-6 ft."""
    leg = traced(traced(a_leg(width_ft=25.2, length_ft=130.0),
                         "left", [(10, 12.6), (130, 12.6)]),
                  "right", [(10, 18.2), (130, 18.2)])
    out, _log = _centred(leg)
    assert out.curb_to_curb_ft == pytest.approx(30.8, abs=0.1)


def test_a_leg_already_centred_is_left_alone():
    leg = traced(traced(a_leg(width_ft=30.0), "left", [(10, 15), (130, 15)]),
                  "right", [(10, 15), (130, 15)])
    before = list(leg.centerline.coords)
    out, _log = _centred(leg)
    assert list(out.centerline.coords) == before


def test_a_midpoint_that_wanders_is_reported_rather_than_shifted():
    """A single constant shift describes a PARALLEL offset between the alignment and the
    street. Where the kerbs' midpoint swings along the leg the alignment is bending relative
    to the street instead, no one number centres it, and moving the paint on that evidence
    would be worse than leaving it."""
    leg = traced(traced(a_leg(width_ft=30.0, length_ft=130.0),
                         "left", [(10, 7.2), (130, 7.2)]),
                  "right", [(10, 28.0), (130, 60.0)])
    before = list(leg.centerline.coords)
    out, log = _centred(leg)
    assert list(out.centerline.coords) == before
    assert "wanders" in log and "left as surveyed" in log, log


# --------------------------------------------------------------------------
# The flex-post delineator, whose whole job is being seen
# --------------------------------------------------------------------------

def blender_props_module():
    """scripts/blender/blender_props.py, importable outside Blender.

    It runs under Blender's bundled Python and imports bpy at module level, so the venv
    cannot import it as-is. Stubbing bpy in is enough to reach the pure arithmetic - the band
    layout is just constants, and the alternative (asserting against the source TEXT, the way
    test_sampled_polylines_are_rendered_as_polylines_not_chords has to) cannot check that the
    numbers come out right, only that certain characters are present.
    """
    import sys
    import types
    from pathlib import Path

    blender_dir = Path(__file__).resolve().parent.parent / "scripts" / "blender"
    stubs = {"bpy": types.ModuleType("bpy"), "mathutils": types.ModuleType("mathutils")}
    stubs["bpy"].ops = types.SimpleNamespace()
    stubs["bpy"].data = types.SimpleNamespace()
    stubs["bpy"].context = types.SimpleNamespace()
    stubs["mathutils"].Vector = tuple
    materials = types.ModuleType("blender_materials")
    materials.make_material = lambda *a, **k: None
    materials.make_retroreflective_material = lambda *a, **k: None
    stubs["blender_materials"] = materials

    saved = {name: sys.modules.get(name)
             for name in (*stubs, "blender_props", "blender_crosswalks")}
    sys.modules.update(stubs)
    sys.path.insert(0, str(blender_dir))
    try:
        import importlib

        sys.modules.pop("blender_props", None)
        sys.modules.pop("blender_crosswalks", None)
        return importlib.import_module("blender_props")
    finally:
        sys.path.remove(str(blender_dir))
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


IN_TO_M = 0.0254


def test_the_delineator_post_is_42_inches_tall():
    """Specified, not derived - so the test states the number rather than a formula that
    would agree with whatever the constant happens to be."""
    props = blender_props_module()
    assert props.BOLLARD_HEIGHT_M / IN_TO_M == pytest.approx(42.0, abs=0.01)


def test_the_post_carries_several_hi_vis_bands_near_its_top():
    """One band 0.6 m up was invisible at this render's camera distance, and the docstring
    claimed a band the code had put in only once.

    Checks the banding pattern rather than the literal heights: at least two bands, the top
    one close to the top of the post, none of them wider apart than the pattern allows, and
    all of them above ground. That is what makes a post read; the exact stack can move.
    """
    props = blender_props_module()
    centres = props.bollard_band_centres_m()
    band = props.BOLLARD_BAND_HEIGHT_M

    assert len(centres) >= 2, f"a single band does not read as a delineator: {centres}"
    assert centres == sorted(centres, reverse=True), "expected top band first"
    top_gap = props.BOLLARD_HEIGHT_M - (centres[0] + band / 2)
    assert top_gap == pytest.approx(props.BOLLARD_TOP_TO_FIRST_BAND_M, abs=1e-9)
    assert top_gap / IN_TO_M <= 2.01, f"top band sits {top_gap / IN_TO_M:.1f} in below the top"
    for upper, lower in zip(centres, centres[1:]):
        gap = (upper - band / 2) - (lower + band / 2)
        assert 0 < gap / IN_TO_M <= 6.01, f"bands {gap / IN_TO_M:.1f} in apart"
    assert min(centres) - band / 2 > 0, "a band is buried in the asphalt"


def test_a_shorter_post_drops_bands_instead_of_burying_them(monkeypatch):
    """The stack is measured down from the top, so a short post has to lose its lowest band
    rather than push it underground."""
    props = blender_props_module()
    monkeypatch.setattr(props, "BOLLARD_HEIGHT_M", 12 * IN_TO_M)
    centres = props.bollard_band_centres_m()
    assert centres, "a 12 in post should still carry its top band"
    assert all(z - props.BOLLARD_BAND_HEIGHT_M / 2 > 0 for z in centres)
    assert len(centres) < props.BOLLARD_BAND_COUNT
