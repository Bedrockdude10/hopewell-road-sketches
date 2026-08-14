"""Property-based tests for the station/offset frame (src/geometry/model.py).

Every measurement this project makes goes through this frame: a leg's centerline defines
"distance along" and "signed distance from", and every kerb vertex, crosswalk bar, stop line
and parking stall is placed and re-measured in it. It is also the piece with the most edge
cases per line, and the four sites exercise a handful of real centerlines - two straight, two
with one kink each. The bugs its own docstrings record (a point behind the junction claiming
the opposite leg's kerb; a vertex placed at station 44.0 reading back at 41.59 across a 4.5 deg
bend) are all shape-dependent, and the shapes that produce them are the shapes nobody typed in.

So these tests state the frame's contracts and let hypothesis look for the shape that breaks
them, rather than asserting numbers off the four centerlines that happen to exist. They are
about the transform in isolation - no site data, no OSM, no config - which is why this file
runs in milliseconds and needs none of the fixtures the rest of the suite does.

The contracts, in the order they are tested:

  1. A point ON the centerline measures offset 0, at its own arc length.
  2. Stations increase monotonically along the line, INCLUDING past both ends - the property
     that keeps a station behind the junction negative.
  3. Offset is signed and antisymmetric about the centerline: left is positive.
  4. On a straight centerline the transform and its inverse round-trip exactly, at any offset.
  5. On a bent one they do not, and _place_in_measured_frame's whole contract is that it is
     never WORSE than the naive placement it corrects. That is the invariant that holds even
     inside the fold, where a round-trip flatly does not.
  6. Away from a kink, that same placement measures back to within the tolerance a painted
     dimension is held to.
  7. AT a kink - the case the placement code exists for - it is still within that tolerance at
     the two angles the real centerlines bend by. Fixed inputs, not sampled ones.

(5) and (6)/(7) are deliberately a pair: (5) is the contract that holds everywhere including
inside a fold, and (7) is the one with teeth. These were checked by mutation - breaking the
correction (removing it, flipping its sign) is caught only by (7), because the sampled cases in
(6) are scoped away from the discontinuity where the correction does its work, and because a
bad correction is not necessarily WORSE than no correction, which is all (5) asks.

One known gap, stated rather than papered over: replacing "keep whichever trial measures
closest" with "keep the last trial" is NOT caught by anything here. With the current two
correction passes the last trial is as good as the best in every case these tests reach, so the
guard is currently unobservable from outside. It matters at pass counts above two.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from shapely.geometry import LineString

from src.geometry.model import _place_in_measured_frame, _point_at, station_offset_many

# Feet. Every tolerance here is a claim about float arithmetic, not about geometry: the frame
# is exact maths, so anything above this is a real disagreement, not accumulated error.
EXACT_FT = 1e-6

# Centerlines are built at this scale - a junction leg is 100-200 ft and its kerbs sit within
# ~40 ft of it. Keeping the generated geometry in the same range as the real thing keeps the
# failures it finds relevant, and keeps float precision comparable to production's.
COORD = st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False)
OFFSET = st.floats(min_value=-60, max_value=60, allow_nan=False, allow_infinity=False)

# Hypothesis's default deadline flags the first call of a numpy path as a failure because it
# includes one-time setup; these are pure array maths with no I/O, so time is not the signal.
SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def straight_lines(draw) -> LineString:
    """A two-point centerline, long enough that direction is well defined."""
    x0, y0 = draw(COORD), draw(COORD)
    angle = draw(st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False))
    length = draw(st.floats(min_value=10, max_value=400, allow_nan=False))
    return LineString([(x0, y0), (x0 + length * np.cos(angle), y0 + length * np.sin(angle))])


@st.composite
def polylines(draw, max_turn_deg: float = 60) -> LineString:
    """A 2-5 vertex centerline with no doubled vertices and no reversals.

    `max_turn_deg` bounds how sharply consecutive segments meet. The real centerlines here kink
    by 4.5 and 29.4 degrees, so 30 is "shaped like a road this project actually renders" and the
    60 default is a deliberate overshoot for the invariants that should hold on any line. A
    frame is not meaningful on a line that doubles back on itself, and testing one would be
    inventing a failure the caller cannot produce.
    """
    turn = np.radians(max_turn_deg)
    n = draw(st.integers(min_value=2, max_value=5))
    x, y = draw(COORD), draw(COORD)
    heading = draw(st.floats(min_value=0, max_value=2 * np.pi, allow_nan=False))
    points = [(x, y)]
    for _ in range(n - 1):
        heading += draw(st.floats(min_value=-turn, max_value=turn, allow_nan=False))
        length = draw(st.floats(min_value=10, max_value=200, allow_nan=False))
        x, y = x + length * np.cos(heading), y + length * np.sin(heading)
        points.append((x, y))
    line = LineString(points)
    assume(line.is_simple)      # a self-crossing centerline has no single frame to be right in
    return line


@given(line=polylines(), fraction=st.floats(min_value=0, max_value=1, allow_nan=False))
@SETTINGS
def test_a_point_on_the_centerline_has_no_offset(line, fraction):
    """The defining property: the centerline is offset zero, at its own arc length."""
    point = line.interpolate(fraction * line.length)
    stations, offsets = station_offset_many(line, np.array([[point.x, point.y]]))
    assert offsets[0] == pytest.approx(0.0, abs=1e-6 * max(1.0, line.length))
    assert stations[0] == pytest.approx(fraction * line.length, abs=1e-6 * max(1.0, line.length))


@given(line=polylines(), fractions=st.lists(st.floats(min_value=-0.5, max_value=1.5,
                                                      allow_nan=False),
                                            min_size=2, max_size=8))
@SETTINGS
def test_stations_increase_along_the_line_including_past_both_ends(line, fractions):
    """Station order is the whole basis of "before" and "after" on a leg, and it has to survive
    leaving the line at either end.

    Past the far end this is what stops every point beyond the working length collapsing onto
    one station. Behind the junction - a NEGATIVE station - it is what stops a leg claiming the
    kerb of the leg opposite it, which is a bug this frame has actually shipped.

    STRICTLY increasing, and the fractions are thinned to a real separation first. Asserting
    only "not decreasing" is what that shipped bug looks like from here: clamping every point
    behind the junction to station 0 puts them all EQUAL, which a >= test waves through.
    """
    ordered = _separated(fractions, minimum=0.01)
    assume(len(ordered) >= 2)
    verts, _, _, _ = _frame_arrays(line)
    points = np.array([_extrapolated_point(line, verts, f) for f in ordered])
    stations, _offsets = station_offset_many(line, points)
    assert np.all(np.diff(stations) > 0), (
        f"stations not strictly increasing for fractions {ordered}: {stations}")


@given(line=straight_lines(), offset=OFFSET,
       fraction=st.floats(min_value=0, max_value=1, allow_nan=False))
@SETTINGS
def test_offset_is_signed_and_antisymmetric(line, offset, fraction):
    """Left is positive, right is negative, and the same distance either side measures the same
    magnitude. Leg.left_curb / right_curb depend on this sign; getting it wrong mirrors a
    street rather than failing."""
    station = fraction * line.length
    left = _point_at(line, station, offset)
    right = _point_at(line, station, -offset)
    _s, offsets = station_offset_many(line, np.array([left, right]))
    assert offsets[0] == pytest.approx(offset, abs=EXACT_FT * max(1.0, abs(offset)))
    assert offsets[1] == pytest.approx(-offset, abs=EXACT_FT * max(1.0, abs(offset)))


@given(line=straight_lines(), offset=OFFSET,
       fraction=st.floats(min_value=-0.5, max_value=1.5, allow_nan=False))
@SETTINGS
def test_the_transform_round_trips_exactly_on_a_straight_line(line, offset, fraction):
    """_point_at and station_offset_many are exact inverses where there is no bend - the claim
    _polyline_frame's docstring makes, and the reason both directions read one frame."""
    station = fraction * line.length
    x, y = _point_at(line, station, offset)
    stations, offsets = station_offset_many(line, np.array([[x, y]]))
    scale = max(1.0, line.length, abs(offset))
    assert stations[0] == pytest.approx(station, abs=EXACT_FT * scale)
    assert offsets[0] == pytest.approx(offset, abs=EXACT_FT * scale)


@given(line=polylines(),
       stations=st.lists(st.floats(min_value=-50, max_value=250, allow_nan=False),
                          min_size=1, max_size=6),
       offset=OFFSET)
@SETTINGS
def test_placing_in_the_measured_frame_is_never_worse_than_not(line, stations, offset):
    """_place_in_measured_frame's actual contract, and the only one that survives a fold.

    Around a bend the two directions of the transform disagree (they resolve the wedge outside
    the corner against different segments), and past the radius of curvature the offset curve
    FOLDS - station order reverses, and no placement can measure back as what was asked. So the
    function does not promise a round-trip. It promises that correcting never makes the
    residual bigger than the naive placement it started from, which is exactly what "keeps
    whichever estimate measures closest to what was asked" means, and it is checkable
    everywhere including inside the fold.
    """
    target_s = np.array(stations, dtype=float)
    target_o = np.full(len(stations), offset, dtype=float)

    naive = np.array([_point_at(line, s, offset) for s in target_s])
    corrected = np.array(_place_in_measured_frame(line, target_s, target_o), dtype=float)

    naive_error = _frame_residual(line, naive, target_s, target_o)
    corrected_error = _frame_residual(line, corrected, target_s, target_o)

    # Per point, not in aggregate: the correction is decided per point, so an average could
    # hide one point made worse by another made better.
    worse = corrected_error > naive_error + 1e-9
    assert not worse.any(), (
        f"{worse.sum()} point(s) measured back FURTHER from the requested frame after "
        f"correction:\n  requested stations {target_s[worse]} at offset {offset}\n"
        f"  naive residual     {naive_error[worse]}\n  corrected residual {corrected_error[worse]}")


# The project's own tolerance for a painted dimension (src/checks.py:LANE_WIDTH_TOLERANCE_FT).
# Not an arbitrary epsilon: a placement wrong by more than this is wrong by more than the
# invariants downstream are willing to absorb, and shows up as a reported violation.
PLACEMENT_TOLERANCE_FT = 0.05


@given(line=polylines(max_turn_deg=30),
       stations=st.lists(st.floats(min_value=0, max_value=150, allow_nan=False),
                          min_size=1, max_size=6),
       offset=st.floats(min_value=-19, max_value=19, allow_nan=False))
@SETTINGS
def test_a_placed_point_measures_back_as_what_was_asked_for(line, stations, offset):
    """The function's headline promise: "world points that MEASURE BACK as (station, offset)".

    Held to a real tolerance, unlike the never-worse property above, which a correction that
    silently stopped correcting would still satisfy.

    Scoped to WELL-CONDITIONED asks - away from the discontinuity at a kink, where a tolerance
    is not a meaningful thing to assert at all (see _well_conditioned, which says exactly why,
    and which was written twice because hypothesis rejected the first version's reasoning). The
    kink itself is not left untested; it is pinned by name in the test below this one, at the
    two angles the real centerlines actually bend by, where the input is exact.

    Bends are capped at 30 deg here because that is the shape of a road: the sharpest of the
    four sites' centerlines kinks by 29.4.
    """
    assume(all(_well_conditioned(line, s, offset) for s in stations))
    target_s = np.array(stations, dtype=float)
    target_o = np.full(len(stations), offset, dtype=float)
    placed = np.array(_place_in_measured_frame(line, target_s, target_o), dtype=float)
    residual = _frame_residual(line, placed, target_s, target_o)
    assert np.all(residual < PLACEMENT_TOLERANCE_FT), (
        f"placed points measure back {residual} ft from the frame they were asked for "
        f"(stations {target_s}, offset {offset}) - over the {PLACEMENT_TOLERANCE_FT} ft a "
        f"painted dimension is allowed to be out by")


@pytest.mark.parametrize("turn_deg, offset_ft, worst_ft", [
    (4.5, 19.0, 0.001),     # broad_st_east's kink, at a kerb offset
    (29.4, 19.0, 0.05),     # louellen_st_west's, the sharpest of the four sites
])
def test_the_worst_place_on_a_real_centerline_is_still_within_tolerance(turn_deg, offset_ft,
                                                                        worst_ft):
    """The one case the property search above will not reliably find, pinned by hand.

    The placement is least accurate at exactly one spot - the vertex station, on the OUTSIDE of
    the bend - and that is a single point in a space hypothesis samples broadly, so it lands
    there rarely. It is also the case that matters: it is where a kerbside marking is placed,
    and being 0.05 ft short of the offset asked for is a reported violation
    (src/checks.py:PaintStaysOutOfTheTravelLane).

    The numbers are what the correction buys, measured: at 29.4 deg and a 19 ft offset the
    uncorrected placement measures back 2.45 ft from what was asked, and the corrected one
    0.04 ft. A regression that quietly stopped correcting would still satisfy every property
    above; it fails here by a factor of 50.
    """
    turn = np.radians(turn_deg)
    line = LineString([(0, 0), (100, 0), (100 + 80 * np.cos(turn), 80 * np.sin(turn))])
    _verts, _dirs, _lens, cumulative = _frame_arrays(line)
    at_vertex = np.array([cumulative[1]])
    # Negative offset is the outside of this left-hand bend. The inside at this offset is
    # inside the fold and unreachable by anything - see _reachable.
    outside = np.array([-offset_ft])

    placed = np.array(_place_in_measured_frame(line, at_vertex, outside), dtype=float)
    residual = _frame_residual(line, placed, at_vertex, outside)
    assert residual[0] < worst_ft, (
        f"a {turn_deg} deg kink at {offset_ft} ft offset places {residual[0]:.4f} ft from the "
        f"frame it was asked for, over the {worst_ft} ft budget")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separated(values, minimum: float) -> list[float]:
    """`values` sorted, keeping only entries at least `minimum` past the one before.

    So that "strictly increasing" is asserted of points that are genuinely at different places
    along the line, rather than of two floats a rounding error apart.
    """
    out = []
    for value in sorted(values):
        if not out or value - out[-1] >= minimum:
            out.append(value)
    return out


def _well_conditioned(line: LineString, station: float, offset: float, margin: float = 1.5) -> bool:
    """Is (station, offset) an ask with ONE clear answer, away from a kink's discontinuity?

    Excludes a band of width |offset| * tan(turn/2) either side of every interior vertex. Both
    sides, for different reasons, and both found the hard way:

      * INSIDE the bend, that band is genuinely unreachable. A polyline vertex is a corner of
        zero radius, so the inner offset lines CROSS there, and stations between the crossing
        and the vertex lie behind it - no world point measures back to them at any offset.
      * OUTSIDE, the ask is reachable but AMBIGUOUS. The offset lines leave a wedge instead of
        crossing, every point in it is perpendicular-nearest to the vertex itself, so a whole
        region collapses onto one station. The placement is iterating toward a discontinuity,
        and how close it gets swings on the last digits of the input: the same case rounded to
        3 decimal places for a failure report converged to 0.00000 ft while the unrounded one
        sat at 0.817. A tolerance is not a meaningful thing to assert there.

    That second class is not waved away - it is the case the placement code exists for, so it
    is pinned by name at the real centerlines' kink angles in
    test_the_worst_place_on_a_real_centerline_is_still_within_tolerance, where the input is
    exact and the answer is a fixed number rather than a sample.
    """
    if offset == 0:
        return True
    verts, seg_dir, _seg_len, cumulative = _frame_arrays(line)
    for i in range(1, len(verts) - 1):
        before, after = seg_dir[i - 1], seg_dir[i]
        cross = before[0] * after[1] - before[1] * after[0]
        turn = abs(np.arctan2(cross, float(np.dot(before, after))))
        if abs(station - cumulative[i]) < abs(offset) * np.tan(turn / 2) * margin:
            return False
    return True


def _frame_arrays(line: LineString):
    from src.geometry.model import _polyline_frame
    return _polyline_frame(line)


def _extrapolated_point(line: LineString, verts, fraction: float):
    """A point at `fraction` of the way along, allowing fraction outside [0, 1] by running on
    along the end tangent - which is how a real leg reaches behind the junction."""
    if 0 <= fraction <= 1:
        point = line.interpolate(fraction * line.length)
        return (point.x, point.y)
    if fraction < 0:
        direction = verts[1] - verts[0]
        return tuple(verts[0] + direction / np.linalg.norm(direction) * fraction * line.length)
    direction = verts[-1] - verts[-2]
    return tuple(verts[-1] + direction / np.linalg.norm(direction) * (fraction - 1) * line.length)


def _frame_residual(line: LineString, points: np.ndarray, target_s, target_o) -> np.ndarray:
    """How far each placed point measures back from the (station, offset) it was asked for."""
    got_s, got_o = station_offset_many(line, points)
    return np.hypot(got_s - target_s, got_o - target_o)
