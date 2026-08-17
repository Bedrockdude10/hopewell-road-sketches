"""The leg frame: (station along the centerline, signed offset from it).

Every curb, sign, pad and band position in this project is ultimately expressed in this
frame, so a defect here doesn't stay local - it becomes a curb drawn through the junction
or a pad in the road. Two of those shipped and both trace back to a property asserted here.
"""
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from src.geometry.model import point_at, station_offset, station_offset_many

STRAIGHT = LineString([(0, 0), (100, 0)])
BENT = LineString([(0, 0), (50, 3), (110, -2), (160, 6)])   # an old street, not quite straight


def test_offset_sign_is_left_positive():
    """Positive offset is to the left of travel, matching Leg.left_curb."""
    _station, offset = station_offset(STRAIGHT, (50, 10))
    assert offset > 0
    _station, offset = station_offset(STRAIGHT, (50, -10))
    assert offset < 0


def test_station_is_negative_behind_the_junction():
    """The bug that let one leg claim the opposite leg's curb.

    LineString.project() clamps to [0, length], so a point behind the junction came back at
    station 0 with a small offset - indistinguishable from a point right at the corner. The
    leg then adopted the opposite leg's curb and drew it back through the intersection.
    """
    station, _offset = station_offset(STRAIGHT, (-40, 12))
    assert station < 0, "a point behind the junction must have a negative station"


def test_station_continues_past_the_far_end():
    """Past the end, stations must keep increasing rather than piling up on the last vertex."""
    near, _ = station_offset(STRAIGHT, (120, 5))
    far, _ = station_offset(STRAIGHT, (180, 5))
    assert far > near > STRAIGHT.length


@pytest.mark.parametrize("station,offset", [(10, 12.5), (0, -8), (-5, 20), (140, -30), (200, 9), (-40, 15)])
def test_round_trip_is_exact(station, offset):
    """point_at and station_offset must be exact inverses over the whole line.

    They weren't when the forward direction used segment tangents and the inverse estimated
    one from a +/-2 ft window: a curb rebuilt from its own traced points came back shifted.
    """
    x, y = point_at(BENT, station, offset)
    back_station, back_offset = station_offset(BENT, (x, y))
    assert back_station == pytest.approx(station, abs=1e-6)
    assert back_offset == pytest.approx(offset, abs=1e-6)


def test_vectorized_matches_scalar():
    """The many-point path is the one used in anger; it must agree with the single-point one."""
    points = np.random.default_rng(0).uniform(-60, 240, size=(200, 2))
    stations, offsets = station_offset_many(BENT, points)
    for i in range(0, len(points), 13):
        s, o = station_offset(BENT, tuple(points[i]))
        assert stations[i] == pytest.approx(s, abs=1e-9)
        assert offsets[i] == pytest.approx(o, abs=1e-9)


def test_offset_equals_perpendicular_distance_on_a_straight_leg():
    stations, offsets = station_offset_many(STRAIGHT, np.array([[25.0, 7.0], [75.0, -3.5]]))
    assert stations == pytest.approx([25.0, 75.0])
    assert offsets == pytest.approx([7.0, -3.5])


# --------------------------------------------------------------------------
# The crossing frame is part of the leg frame
# --------------------------------------------------------------------------

def a_leg(centerline, width_ft=42.0):
    from src.geometry.model import Leg

    return Leg(name="louellen", centerline=centerline, curb_to_curb_ft=width_ft)


def test_a_crossing_is_square_to_the_street_at_its_own_station():
    """crosswalk_axes used to extrapolate the leg's FIRST SEGMENT, which is exact only for a
    straight centerline. Ten of this project's twelve legs are straight 2-vertex lines, so
    the shortcut survived; louellen_st_west is not.

    Its centerline leaves the junction at 239.2 deg for 15.4 ft and then runs 268.6 - NJDOT
    rounding the corner where CR 518 turns off W Broad onto Louellen - a 29.4 deg bend. The
    crossing at station 31.5 came out 29.4 deg off square to the street it crosses, with its
    centre ~8 ft to the side of the real carriageway centre.
    """
    from src.render.crosswalks import crosswalk_axes

    # 240 deg for 15 ft, then due west - Louellen's shape, in round numbers.
    bend = LineString([(0, 0), (-13.0, -7.5), (-130.0, -7.5)])
    leg = a_leg(bend)

    centre, _u, across, _cos = crosswalk_axes(leg, 31.5)
    # On the centerline, not beside it.
    station, offset = station_offset(bend, centre)
    assert station == pytest.approx(31.5, abs=0.01)
    assert offset == pytest.approx(0.0, abs=0.01)
    # And spanning across the street it is actually crossing: due west street, so the bars
    # run north-south.
    assert abs(across[1]) == pytest.approx(1.0, abs=0.01), f"bars not square to the street: {across}"
    assert abs(across[0]) == pytest.approx(0.0, abs=0.01)


def test_a_crossing_inside_the_first_segment_is_where_it_always_was():
    """The complement, and what bounds the change: reading the frame at the station is
    identical to extrapolating segment one for any station that falls inside segment one.
    broad_st_east bends 4.5 deg but does so 43.1 ft out, past its crossing, so nothing about
    that crossing moved."""
    from src.render.crosswalks import crosswalk_axes

    bend = LineString([(0, 0), (43.1, 0), (150.0, 8.4)])
    centre, u, across, _cos = crosswalk_axes(a_leg(bend), 21.3)
    assert centre == pytest.approx((21.3, 0.0), abs=1e-9)
    assert u == pytest.approx((1.0, 0.0), abs=1e-9)
    assert across == pytest.approx((0.0, 1.0), abs=1e-9)


def test_the_crossing_centre_round_trips_through_the_leg_frame():
    """Whatever the shape of the leg: the station a crossing is BUILT at has to be the
    station everything else MEASURES it at, or the reach, the band and the paint that keeps
    clear of it are each working from a different crossing."""
    from src.render.crosswalks import crosswalk_axes

    for station in (5.0, 31.5, 60.0, 129.0):
        centre, _u, _n, _cos = crosswalk_axes(a_leg(BENT), station)
        back, offset = station_offset(BENT, centre)
        assert back == pytest.approx(station, abs=1e-6)
        assert offset == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# A through street has no corner in it
# --------------------------------------------------------------------------

def two_legs(bearing_b_deg, width_ft=40.0):
    """Leg 'a' heading due east, leg 'b' at `bearing_b_deg` (math degrees), both outward."""
    import numpy as np

    from src.geometry.model import Leg

    legs = {}
    for name, deg in (("a", 0.0), ("b", bearing_b_deg)):
        d = np.radians(deg)
        line = LineString([(0, 0), (130 * np.cos(d), 130 * np.sin(d))])
        legs[name] = Leg(name=name, centerline=line, curb_to_curb_ft=width_ft)
    return legs


def test_two_collinear_legs_get_no_corner_return():
    """They are one street running through the junction. The curb between them does not
    curve, so it cannot constrain where a crossing or a hatched zone may start.

    e_broad_st_east and e_broad_st_west are 179.9 deg apart - the continuous north edge of
    E Broad St, opposite the stem of the T. traced_corner_join drew a diagonal between the two
    curbs whose start lay 67.1 ft up the leg, that became the leg's corner-return tangent
    point, and the kerbside hatching was held 75 ft out from a junction whose surveyed stop bar
    is at 52.9 ft. fillet_curb_corner has rejected this case since it was the only path; the
    traced branches return before reaching it.
    """
    from src.geometry.model import build_corner_fillets, leg_clearance_ft

    legs = two_legs(179.9)
    fillets = build_corner_fillets(legs, radius_ft=15.0)
    assert any(p.get("through_street") for p in fillets.values()), \
        "a 179.9 deg pair was treated as a corner"
    for name in legs:
        assert leg_clearance_ft(name, legs, fillets) == pytest.approx(3.0), \
            "a through-street join is constraining the leg's clearance"


def test_a_real_corner_still_constrains_its_legs():
    """The complement - without it the test above passes by disabling corners entirely."""
    from src.geometry.model import build_corner_fillets, leg_clearance_ft

    legs = two_legs(90.0)
    fillets = build_corner_fillets(legs, radius_ft=15.0)
    assert not any(p.get("through_street") for p in fillets.values())
    assert leg_clearance_ft("a", legs, fillets) > 10.0


def test_the_through_street_test_reads_the_leg_not_its_first_stub():
    """louellen_st_west leaves its junction on a 15 ft stub bearing 239 deg before settling
    onto 269. Measured off that stub it is 178.6 deg from w_broad_st_northeast and reads as a
    through street; measured off the leg it is 149.2, which is the truth - the route turns
    there and the traced kerbs show a real 14 ft return."""
    import numpy as np

    from src.geometry.model import Leg, is_through_street

    ne = Leg(name="ne", centerline=LineString([(0, 0), (130 * np.cos(np.radians(32.2)),
                                                        130 * np.sin(np.radians(32.2)))]),
             curb_to_curb_ft=35.0)
    stub = np.radians(-119.2)   # 239 deg compass
    onward = np.radians(-178.6)
    kink = np.array([15 * np.cos(stub), 15 * np.sin(stub)])
    louellen = Leg(name="louellen",
                   centerline=LineString([(0, 0), tuple(kink),
                                          tuple(kink + 115 * np.array([np.cos(onward),
                                                                       np.sin(onward)]))]),
                   curb_to_curb_ft=42.0)
    assert not is_through_street(ne, louellen), \
        "the stub, not the leg, decided this - and it decided wrong"


def test_a_kerb_vertex_goes_to_the_leg_it_lies_in_front_of():
    """Two collinear legs both see a vertex at the junction node. The one it is AHEAD of wins.

    CURB_POINT_BEHIND_TOLERANCE_FT lets a leg claim a vertex up to 3 ft behind its own node,
    which a corner return needs. Unpenalised, that let a leg outbid one the vertex lies in
    front of, purely on whose half-width matched a shade better. At E Broad & Princeton the
    two legs are 179.9 deg apart and 36.9 / 38.2 ft wide: the vertex where East Broad's north
    kerb changes from corner return to straight run sits 0.8 ft ahead of the east leg and
    0.8 ft behind the west one, and the west leg took it on a ratio of 0.995 against 1.010.
    That vertex was the near end of a 58.3 ft traced way; with one point left,
    curb_line_from_points needs two, and the whole stretch was dropped - so 58 ft of a kerb
    the surveyor had tagged no_stopping went unhatched and was reported as untraced.
    """
    from src.geometry.model import Leg, point_at, assign_curb_points_to_legs

    east = Leg(name="east", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=36.9)
    west = Leg(name="west", centerline=LineString([(0, 0), (-130, 0)]), curb_to_curb_ft=38.2)
    legs = {"east": east, "west": west}

    # One continuous kerb along the north side, straddling the junction node: a vertex 0.8 ft
    # onto the east leg, then out along it.
    kerb = LineString([point_at(east.centerline, s, 18.65) for s in (0.8, 59.2, 96.1)])
    assigned = assign_curb_points_to_legs(legs, [kerb])

    east_left = assigned.get("east", {}).get("left", [])
    assert len(east_left) == 3, (
        f"the east leg got {len(east_left)} of the 3 vertices in front of it; "
        f"west got {len(assigned.get('west', {}).get('right', []))}")
    assert min(s for s, _o in east_left) == pytest.approx(0.8, abs=0.1)


def test_a_vertex_behind_every_leg_is_still_claimed():
    """The complement: the penalty must not turn into a hard rejection. A corner return's own
    geometry straddles station 0, and dropping those vertices loses the corner."""
    from src.geometry.model import Leg, point_at, assign_curb_points_to_legs

    east = Leg(name="east", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=36.9)
    legs = {"east": east}
    kerb = LineString([point_at(east.centerline, s, 18.65) for s in (-1.5, 20.0, 60.0)])
    assigned = assign_curb_points_to_legs(legs, [kerb])
    stations = [s for s, _o in assigned["east"]["left"]]
    assert min(stations) == pytest.approx(-1.5, abs=0.1), \
        f"the vertex behind the node was dropped: {sorted(stations)}"


def test_the_along_a_leg_kerb_test_accepts_the_outer_half_of_a_leg():
    """_runs_along_a_leg is the correct relevance test for a curb LINE, and is measured here
    even though it is not wired in yet (see kerb_lines_with_tags_ft's docstring for why).

    A kerb at station 100 on a 130 ft leg is 100 ft from the junction centre, so the
    KERB_NEAR_JUNCTION_FT radius that is right for fitting a corner radius throws it away.
    14 traced ways across the four junctions are in that position.
    """
    from src.geometry.intersection import _runs_along_a_leg
    from src.geometry.model import Leg, point_at

    leg = Leg(name="east", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=31.0)
    legs = {"east": leg}

    outer = LineString([point_at(leg.centerline, s, 15.5) for s in (90.0, 126.0)])
    assert _runs_along_a_leg(outer, legs), "the outer half of the leg's own kerb was rejected"

    # A kerb out in a field, or one belonging to a street 300 ft away, is not this leg's.
    elsewhere = LineString([(400.0, 400.0), (460.0, 400.0)])
    assert not _runs_along_a_leg(elsewhere, legs)
    behind = LineString([point_at(leg.centerline, s, 15.5) for s in (-90.0, -40.0)])
    assert not _runs_along_a_leg(behind, legs), "a kerb behind the junction is another leg's"


def test_the_width_fit_never_ends_up_using_less_traced_kerb_than_it_found(monkeypatch):
    """The fit's monotonicity guard, with a regressing round injected.

    A width feeds the window deciding which vertices the NEXT round may claim, so a round can
    talk itself out of a kerb it was already using - and the loss compounds into a runaway. At
    W Broad & Louellen the leg went 2 traced kerbs -> 1 -> a width guessed by doubling one
    offset into an 80 ft "street" -> at 80 ft its own kerb fell under
    CURB_POINT_MIN_WIDTH_RATIO and could never be recovered. Every step was defensible.

    No input in this repo currently makes a round regress, so the regression is INJECTED
    rather than hoped for: the second resize widens the leg to 200 ft, which puts both its
    real kerbs far outside the ratio window. Without the guard the fit ends there. This is
    defensive code, and defensive code that nothing exercises is decoration.
    """
    import contextlib
    import io

    import src.geometry.intersection as intersection
    from src.geometry.model import Leg, point_at

    leg = Leg(name="east", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=12.0)
    legs = {"east": leg}
    ways = [(LineString([point_at(leg.centerline, s, sign * 15.5) for s in (20.0, 70.0, 120.0)]),
             {"barrier": "kerb"})
            for sign in (1, -1)]

    real_resize = intersection._resize_and_centre_from_traced_kerbs
    calls = {"n": 0}

    def resize_then_sabotage(target_legs, legs_cfg, quiet=False):
        changed = real_resize(target_legs, legs_cfg, quiet=quiet)
        calls["n"] += 1
        if calls["n"] == 2:            # a round that throws the leg's own kerbs out of range
            target_legs["east"] = Leg(name="east", centerline=target_legs["east"].centerline,
                                       curb_to_curb_ft=200.0)
            return True
        return changed

    monkeypatch.setattr(intersection, "_resize_and_centre_from_traced_kerbs", resize_then_sabotage)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        intersection._fit_legs_to_traced_kerbs(legs, ways, Point(0, 0), {})

    assert intersection._traced_side_count(legs) == 2, (
        f"the fit ended using {intersection._traced_side_count(legs)} of 2 traced kerb sides; "
        f"the sabotaged round was allowed to stand")
    assert legs["east"].curb_to_curb_ft == pytest.approx(31.0, abs=0.5), (
        f"width came out {legs['east'].curb_to_curb_ft:.1f} ft, not the 31 ft between the kerbs")
    assert "fewer leg side" in out.getvalue(), "the rollback happened silently"


def test_the_width_fit_measures_a_badly_configured_leg_from_its_kerbs():
    """The ordinary case the guard must not interfere with: a leg configured at 12 ft whose
    two traced kerbs are 31 ft apart ends up 31 ft, with both sides traced."""
    import contextlib
    import io

    from src.geometry.intersection import _fit_legs_to_traced_kerbs, _traced_side_count
    from src.geometry.model import Leg, point_at

    leg = Leg(name="east", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=12.0)
    legs = {"east": leg}
    ways = [(LineString([point_at(leg.centerline, s, sign * 15.5) for s in (20.0, 70.0, 120.0)]),
             {"barrier": "kerb"})
            for sign in (1, -1)]
    with contextlib.redirect_stdout(io.StringIO()):
        _fit_legs_to_traced_kerbs(legs, ways, Point(0, 0), {})
    assert _traced_side_count(legs) == 2
    assert legs["east"].curb_to_curb_ft == pytest.approx(31.0, abs=0.5)
