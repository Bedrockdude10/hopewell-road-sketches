"""Where a car may legally park, per R.S. 39:4-138.

Each distance here is a citation, not a preference, so each test names the clause it is
checking. If one of these fails the proposal is drawing something illegal - which matters
more than it looking wrong, because the drawing is what someone would build from.
"""
import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.daylighting import (CROSSWALK_SETBACK_FT, CROSSWALK_SETBACK_WITH_BULBOUT_FT,
                                       FIRE_HYDRANT_SETBACK_FT, SIDELINE_SETBACK_FT,
                                       STOP_SIGN_SETBACK_FT, legal_parking_start_ft,
                                       no_parking_zones_ft, parkable_runs_ft)
from src.geometry.model import Leg
from tests.conftest import needs_source_data
from src.geometry.targets import LegSide
from src.geometry.treatments import (DesignState, ProtectDaylightZone)

# The real DesignState, not a stub of it. There was a hand-rolled FakeState here mirroring the
# four fields these rules read, and it had already drifted: curb_extensions was added to
# DesignState and the stub did not have it, so a rule that consults it raised AttributeError
# from inside a test rather than failing on the behaviour. DesignState needs nothing these tests
# do not already build.


def a_state(length_ft=200.0, width_ft=30.0):
    leg = Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)
    return DesignState(legs={"east": leg}, corner_fillets={})


def prop(kind, x, y):
    return {"type": kind, "position_ft": (x, y)}


def with_device(state, kind, spacing_ft=8.0):
    """A design with this daylight device recorded, without going through apply.

    DesignState.apply refuses a curb-extension-class device with no AddCurbExtension under it -
    a real invariant, and the one
    test_claiming_a_curb_extension_setback_without_building_one_is_refused covers. The two tests
    below are about WHICH SETBACK the clause resolves to, so they record the device directly
    rather than also building a kerb extension, which would make them tests of AddCurbExtension
    as well.
    """
    state.treatments.append(ProtectDaylightZone(LegSide("east", "left"), kind=kind,
                                                spacing_ft=spacing_ft))
    return state


# --------------------------------------------------------------------------
# 39:4-138(e) - the daylighting distance
# --------------------------------------------------------------------------

def test_parking_starts_25_feet_past_the_crosswalk():
    state = a_state()
    start = legal_parking_start_ft(state, "east", "left", {"east": (30.0,)})
    assert start == pytest.approx(30.0 + CROSSWALK_SETBACK_FT)


def test_a_curb_extension_reduces_both_setbacks_to_ten_feet(monkeypatch):
    """39:4-138(e), second clause. The reduction applies to the crosswalk arm AND the side
    line arm - the clause reads "within 10 feet of the nearest crosswalk or side line ... if
    a curb extension or bulbout has been constructed". Cutting only the crosswalk arm leaves
    the side line binding at 25 ft, so the extension buys nothing, which is not what it says.

    The device is monkeypatched in rather than using the real `curb_extension`, so this stays a
    test of the CLAUSE and not of the set's current contents. Pinning it to whatever happens to
    be in CURB_EXTENSION_DEVICES today would make it re-fail every time that set changes, which
    is what test_only_a_built_curb_extension_buys_back_the_setback is for.
    """
    import src.geometry.treatments as treatments
    from src.geometry.treatments import corners
    from src.geometry.daylighting import SIDELINE_SETBACK_WITH_BULBOUT_FT

    # BOTH BINDINGS, because the two readers reach these constants differently and patching one
    # place stopped being enough when treatments became a package. ProtectDaylightZone resolves
    # them as globals of the module it is DEFINED in (treatments.corners), while
    # daylighting.legal_parking_start_ft imports CURB_EXTENSION_DEVICES from the package inside
    # the function, so it reads the re-export. Patching only the package left the constructor
    # refusing the very kind this test installs.
    for module in (corners, treatments):
        monkeypatch.setattr(module, "CURB_EXTENSION_DEVICES", frozenset({"built_bulbout"}))
        # The kind has to be constructible as well as recognised: a Treatment validates itself,
        # so "built_bulbout" would be refused by ProtectDaylightZone's own constructor otherwise.
        monkeypatch.setattr(module, "VALID_DAYLIGHT_DEVICES",
                            (*corners.VALID_DAYLIGHT_DEVICES, "built_bulbout"))
    state = with_device(a_state(), "built_bulbout", spacing_ft=5.0)
    start = legal_parking_start_ft(state, "east", "left", {"east": (30.0,)})
    assert start == pytest.approx(30.0 + CROSSWALK_SETBACK_WITH_BULBOUT_FT)
    assert CROSSWALK_SETBACK_WITH_BULBOUT_FT < CROSSWALK_SETBACK_FT
    assert SIDELINE_SETBACK_WITH_BULBOUT_FT < SIDELINE_SETBACK_FT

    # A flex-post line is not a constructed curb extension.
    assert legal_parking_start_ft(with_device(a_state(), "bollards"), "east", "left",
                                  {"east": (30.0,)}) == pytest.approx(30.0 + CROSSWALK_SETBACK_FT)


def test_only_a_built_curb_extension_buys_back_the_setback():
    """The complement, and the one that guards the live behaviour.

    This used to assert CURB_EXTENSION_DEVICES was EMPTY - true while nothing in the repo could
    build the thing the statute names. add_curb_extension now can, so `curb_extension` is in the
    set and the assertion had to become the narrower one that was always the real point:
    everything OTHER than a constructed extension leaves the 25 ft setback intact. Painting or
    posting a setback is not constructing a bulbout, and parking at 10 ft is only lawful where
    one has actually been built.
    """
    from src.geometry.treatments import CURB_EXTENSION_DEVICES, VALID_DAYLIGHT_DEVICES

    assert {"curb_extension"} == CURB_EXTENSION_DEVICES
    for kind in VALID_DAYLIGHT_DEVICES:
        state = with_device(a_state(), kind)
        start = legal_parking_start_ft(state, "east", "left", {"east": (30.0,)})
        expected = (CROSSWALK_SETBACK_WITH_BULBOUT_FT if kind in CURB_EXTENSION_DEVICES
                    else CROSSWALK_SETBACK_FT)
        assert start == pytest.approx(30.0 + expected), f"{kind} resolved the wrong setback"


def test_claiming_a_curb_extension_setback_without_building_one_is_refused():
    """The statutory reduction is for an extension that EXISTS.

    protect_daylight_zone(kind="curb_extension") is only a declaration - add_curb_extension is
    what moves the kerb. Letting the declaration stand alone would mark parking 15 ft closer to
    a crossing than R.S. 39:4-138(e) allows, on the strength of a bulbout nobody drew.
    """
    with pytest.raises(ValueError, match="no curb extension has been built"):
        a_state().apply(ProtectDaylightZone(LegSide("east", "left"), kind="curb_extension"))


def test_the_side_line_governs_a_leg_with_no_marked_crossing():
    """The statute says "nearest crosswalk OR side line". Only the crosswalk arm was applied
    here, so a leg with no marked crossing had no junction setback at all."""
    state = a_state()
    zones = no_parking_zones_ft(state, "east", "left", {})   # no crossing on this leg
    assert zones[0].end_ft == pytest.approx(SIDELINE_SETBACK_FT)
    assert "side line" in zones[0].reason


def test_the_further_of_the_two_arms_wins():
    state = a_state()
    zones = no_parking_zones_ft(state, "east", "left", {"east": (40.0,)})
    assert zones[0].end_ft == pytest.approx(40.0 + CROSSWALK_SETBACK_FT)
    assert "crosswalk" in zones[0].reason


# --------------------------------------------------------------------------
# 39:4-138(h) and (i) - radii, not shifted starts
# --------------------------------------------------------------------------

def test_a_hydrant_forbids_parking_ten_feet_either_side_of_itself():
    """The bug this replaced: a point setback was applied as "parking starts after this",
    so a hydrant 209 ft past the end of a 130 ft leg pushed every stall off the leg. It is a
    RADIUS - it makes a gap and leaves the kerb beyond it parkable.
    """
    state = a_state(length_ft=200.0)
    zones = no_parking_zones_ft(state, "east", "left", {"east": (20.0,)},
                                 [prop("fire_hydrant", 120.0, 16.0)])
    hydrant = [z for z in zones if "hydrant" in z.reason]
    assert len(hydrant) == 1
    assert hydrant[0].start_ft == pytest.approx(120.0 - FIRE_HYDRANT_SETBACK_FT)
    assert hydrant[0].end_ft == pytest.approx(120.0 + FIRE_HYDRANT_SETBACK_FT)

    runs = parkable_runs_ft(state, "east", "left", {"east": (20.0,)},
                             [prop("fire_hydrant", 120.0, 16.0)], min_run_ft=5.0)
    assert len(runs) == 2, "the hydrant splits the kerb, it does not end it"
    assert runs[0][1] == pytest.approx(110.0)
    assert runs[1][0] == pytest.approx(130.0)


def test_a_stop_sign_forbids_parking_fifty_feet_either_side():
    state = a_state(length_ft=300.0)
    zones = no_parking_zones_ft(state, "east", "left", {"east": (20.0,)},
                                 [prop("stop_sign", 150.0, 16.0)])
    sign = [z for z in zones if "stop sign" in z.reason]
    assert sign[0].start_ft == pytest.approx(150.0 - STOP_SIGN_SETBACK_FT)
    assert sign[0].end_ft == pytest.approx(150.0 + STOP_SIGN_SETBACK_FT)


def test_a_feature_far_down_the_next_block_does_not_govern_this_leg():
    """The exact failure: a hydrant at station 338.9 on a 130 ft leg. Its zone has to
    actually reach the leg."""
    state = a_state(length_ft=130.0)
    zones = no_parking_zones_ft(state, "east", "left", {"east": (20.0,)},
                                 [prop("fire_hydrant", 338.9, 8.0)])
    assert not [z for z in zones if "hydrant" in z.reason]


def test_a_feature_just_past_the_end_of_the_leg_still_governs_it():
    """The complement: 5 ft past a 130 ft leg, a hydrant still forbids parking at 125-130."""
    state = a_state(length_ft=130.0)
    zones = no_parking_zones_ft(state, "east", "left", {"east": (20.0,)},
                                 [prop("fire_hydrant", 135.0, 16.0)])
    assert [z for z in zones if "hydrant" in z.reason]


def test_a_feature_on_the_other_kerb_does_not_govern_this_side():
    state = a_state()
    on_the_right = [prop("fire_hydrant", 120.0, -16.0)]
    assert not [z for z in no_parking_zones_ft(state, "east", "left", {"east": (20.0,)}, on_the_right)
                if "hydrant" in z.reason]
    assert [z for z in no_parking_zones_ft(state, "east", "right", {"east": (20.0,)}, on_the_right)
            if "hydrant" in z.reason]


def test_a_feature_out_in_a_field_does_not_govern_this_kerb():
    """A hydrant belongs on the footway just behind the kerb; one 60 ft off to the side
    belongs to the cross street or a property."""
    state = a_state()
    far = [prop("fire_hydrant", 120.0, 60.0)]
    assert not [z for z in no_parking_zones_ft(state, "east", "left", {"east": (20.0,)}, far)
                if "hydrant" in z.reason]


def test_a_hydrant_on_the_footway_behind_the_kerb_does_govern():
    """It has to. The test that says "is it in the roadway" excludes every real hydrant."""
    state = a_state(width_ft=30.0)                     # kerb at 15 ft
    behind_the_kerb = [prop("fire_hydrant", 120.0, 18.0)]
    assert [z for z in no_parking_zones_ft(state, "east", "left", {"east": (20.0,)}, behind_the_kerb)
            if "hydrant" in z.reason]


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def test_a_run_too_short_for_one_stall_is_not_marked():
    state = a_state(length_ft=200.0)
    props = [prop("fire_hydrant", 120.0, 16.0), prop("fire_hydrant", 145.0, 16.0)]
    runs = parkable_runs_ft(state, "east", "left", {"east": (20.0,)}, props, min_run_ft=22.0)
    assert all(end - start >= 22.0 for start, end in runs)
    assert not any(start >= 130.0 and end <= 135.0 for start, end in runs), \
        "the 5 ft gap between the two hydrants is not a parking space"


def test_no_room_at_all_gives_no_runs():
    state = a_state(length_ft=60.0)
    assert parkable_runs_ft(state, "east", "left", {"east": (30.0,)},
                             physical_clearance_ft=0.0, min_run_ft=22.0) == []
    assert legal_parking_start_ft(state, "east", "left", {"east": (30.0,)},
                                   min_run_ft=22.0) is None


def test_the_physical_clearance_can_bind_even_where_the_law_does_not():
    """Not a legal limit, but you cannot paint a stall on the corner return's curve."""
    state = a_state()
    runs = parkable_runs_ft(state, "east", "left", {"east": (5.0,)}, physical_clearance_ft=90.0)
    assert runs[0][0] == pytest.approx(90.0)


def test_runs_never_start_behind_the_junction():
    state = a_state()
    runs = parkable_runs_ft(state, "east", "left", {"east": (10.0,)})
    assert all(start >= 0 for start, _end in runs)


# --------------------------------------------------------------------------
# On the real junctions
# --------------------------------------------------------------------------

def test_the_binding_rule_is_reported():
    """A proposal has to be able to say WHY a stall starts where it does - "parking starts
    at 61 ft" is unreviewable, a statutory citation is not."""
    state = a_state()
    zones = no_parking_zones_ft(state, "east", "left", {"east": (30.0,)})
    assert "39:4-138" in zones[0].reason


def test_a_zone_reports_its_own_length():
    state = a_state()
    zone = no_parking_zones_ft(state, "east", "left", {"east": (30.0,)})[0]
    assert zone.length_ft == pytest.approx(zone.end_ft - zone.start_ft)
    assert zone.length_ft > 0


def test_stations_are_measured_in_the_leg_frame_not_world_coordinates():
    """A leg that does not run along +x. The setbacks are distances along the street."""
    diagonal = Leg(name="ne", centerline=LineString([(0, 0), (100 / np.sqrt(2), 100 / np.sqrt(2))]),
                    curb_to_curb_ft=30.0)
    state = DesignState(legs={"ne": diagonal}, corner_fillets={})
    hydrant_at_station_50 = prop("fire_hydrant", 50 / np.sqrt(2) - 10 / np.sqrt(2),
                                  50 / np.sqrt(2) + 10 / np.sqrt(2))
    zones = [z for z in no_parking_zones_ft(state, "ne", "left", {"ne": (10.0,)},
                                             [hydrant_at_station_50]) if "hydrant" in z.reason]
    assert zones and zones[0].start_ft == pytest.approx(50.0 - FIRE_HYDRANT_SETBACK_FT, abs=0.1)


def _crossing_at(x_ft: float, half_width_ft: float = 20.0) -> dict:
    """One OSM crossing way, square across a leg running east along y=0, at station `x_ft`.

    In WGS84 because that is what the matcher takes, so the coordinates are converted back
    from the state plane the synthetic leg lives in.
    """
    from src.render.coords import wgs84_to_state_plane

    import pyproj
    back = pyproj.Transformer.from_crs(wgs84_to_state_plane.target_crs,
                                        wgs84_to_state_plane.source_crs, always_xy=True)
    ends = [back.transform(x_ft, -half_width_ft), back.transform(x_ft, half_width_ft)]
    return {"coords_wgs84": [list(e) for e in ends], "tags": {}, "id": 1}


def test_a_crossing_down_the_block_is_not_this_junctions_crossing():
    """A leg drawn further must not adopt the NEXT junction's crosswalk.

    The only longitudinal test used to be "projects between the junction and the leg's far
    end", so leg_working_length_ft - a drawing-extent setting - decided junction membership.
    Lengthening broad_st_east from 170 ft to 374 ft made it adopt a crossing at station 264,
    reported as osm_survey; daylighting then took its 25 ft setback from THAT and blanked
    289 ft of kerb, moving the statutory zone from the corner into the middle of the block.

    The same crossing, against a short leg and a long one: neither may match it.
    """
    from src.render.crosswalks import CROSSING_NEAR_JUNCTION_FT, _matched_crossings

    far_ft = CROSSING_NEAR_JUNCTION_FT + 100
    crossing = [_crossing_at(far_ft)]
    for length_ft in (CROSSING_NEAR_JUNCTION_FT + 50, far_ft + 200):
        legs = {"east": Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]),
                             curb_to_curb_ft=40.0)}
        assert "east" not in _matched_crossings(legs, crossing), (
            f"a crossing {far_ft:.0f} ft out was adopted by a {length_ft:.0f} ft leg")


def test_this_junctions_own_crossing_still_matches_however_far_the_leg_is_drawn():
    """The other half: the bound must not cost a real crossing when a leg is drawn long.

    Every genuine match across the four sites sits at 19.5-41.7 ft, so a crossing at 30 ft is
    squarely inside the range this has to keep.
    """
    from src.render.crosswalks import _matched_crossings

    crossing = [_crossing_at(30.0)]
    for length_ft in (130.0, 400.0):
        legs = {"east": Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]),
                             curb_to_curb_ft=40.0)}
        matched = _matched_crossings(legs, crossing)
        assert "east" in matched, f"a real crossing at 30 ft was dropped on a {length_ft:.0f} ft leg"
        assert matched["east"][0] == pytest.approx(30.0, abs=1.0)


@needs_source_data
def test_no_marked_parking_within_25_ft_of_any_intersecting_street(site_models):
    """R.S. 39:4-138(e) applies at EVERY intersection, not the one the drawing is centred on.

    Legs are carried out with the frame now, so Broad St runs 374 ft - past Blackwell Avenue and
    the rest of the block - and the markings did not know: stalls were painted straight across
    the mouth of a side street, and the statutory setback existed only at this junction's own
    corners. The data was already fetched; fetch_roads had been read for `overtaking=no` and its
    geometry thrown away.

    Asserted against the PAINT rather than the zone list, because a zone nobody subtracts is not
    a setback. Every marked stall on every leg, against every street that leg crosses.
    """
    import contextlib
    import io

    from src.geometry.daylighting import SIDELINE_SETBACK_FT, parkable_runs_ft
    from src.render.scene import SceneGeometry
    from src.sources.osm_context import fetch_crossings

    checked = 0
    for site, model in site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            state = DesignState.from_model(model)
            scene = SceneGeometry.resolve(
                model, state, crossings=fetch_crossings(model.center_wgs84, radius_m=130))
        crossings = model.cross_streets
        for leg_name, streets in crossings.items():
            for side in ("left", "right"):
                runs = parkable_runs_ft(state, leg_name, side, scene.crosswalk_offsets, [])
                for cross in streets:
                    if side not in cross.sides:
                        continue
                    checked += 1
                    near_ft, far_ft = cross.mouth_ft
                    forbidden = (near_ft - SIDELINE_SETBACK_FT, far_ft + SIDELINE_SETBACK_FT)
                    for start_ft, end_ft in runs:
                        assert end_ft <= forbidden[0] or start_ft >= forbidden[1], (
                            f"{site}: parking run {start_ft:.0f}-{end_ft:.0f} ft on "
                            f"{leg_name}/{side} overlaps the {SIDELINE_SETBACK_FT:.0f} ft "
                            f"setback {forbidden[0]:.0f}-{forbidden[1]:.0f} ft around "
                            f"{cross.citation}")
    assert checked, "no leg crossed another street at any site, so this asserted nothing"


@needs_source_data
def test_a_cross_street_breaks_the_kerb_it_joins_and_not_the_one_opposite(site_models):
    """A T-junction opens one kerb. Opening both would break paint that is really there."""
    import contextlib
    import io

    from src.geometry.kerbs import OpeningSource

    seen = 0
    for site, model in site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            state = DesignState.from_model(model)
        for leg_name, streets in model.cross_streets.items():
            for cross in streets:
                assert cross.sides, "a cross street that leaves on neither side is not one"
                seen += 1
                for side in ("left", "right"):
                    openings = [o for o in state.kerb_openings.get((leg_name, side), [])
                                if o.source is OpeningSource.CROSS_STREET
                                and o.way_id == cross.way_id]
                    if side in cross.sides:
                        assert openings, (
                            f"{site}: {cross.citation} joins {leg_name}/{side} but does not "
                            f"break its kerb - paint runs across the mouth")
                    else:
                        assert not openings, (
                            f"{site}: {cross.citation} broke {leg_name}/{side}, the kerb "
                            f"OPPOSITE the one it joins")
    assert seen, "no cross street found at any site"


# --------------------------------------------------------------------------
# 39:4-138(e), the CROSSWALK arm, at a cross street - and N.J.S.A. 39:1-1
# --------------------------------------------------------------------------

def _a_cross_street(station_ft=100.0, half_width_ft=13.0, sides=("left",), crosswalks=()):
    from src.geometry.cross_streets import CrossStreet

    return CrossStreet(leg="east", station_ft=station_ft, half_width_ft=half_width_ft,
                        sides=frozenset(sides), name="Blackwell Avenue", way_id=1,
                        crosswalks=tuple(crosswalks))


def test_a_cross_street_gets_the_crosswalk_arm_of_e_and_not_only_the_side_line():
    """R.S. 39:4-138(e) has two arms and the further one binds - at EVERY intersection.

    Only the side line arm was applied away from the modelled junction, which is the same
    half-a-rule this module's docstring records at the junction end, run backwards: a cross
    street with a marked zebra across our own street got the setback owed to one with nothing.
    """
    from src.geometry.cross_streets import CrossStreetCrosswalk

    # A crosswalk 10 ft outside each end of the mouth - further out than the side line, so it is
    # the arm that binds, which is the whole point of measuring both.
    cross = _a_cross_street(crosswalks=(CrossStreetCrosswalk(77.0, is_surveyed=True,
                                                              node_ids=(4242,)),
                                        CrossStreetCrosswalk(123.0, is_surveyed=True,
                                                              node_ids=(4243,))))
    state = a_state(length_ft=300.0)
    state.cross_streets = {"east": [cross]}

    zones = no_parking_zones_ft(state, "east", "left", {"east": (30.0,)})
    from_crosswalk = [z for z in zones if "crosswalk" in z.reason and "side line" not in z.reason
                      and z.start_ft > 40]
    assert len(from_crosswalk) == 2, (
        f"expected one zone per approach crosswalk, got {[z.reason for z in zones]}")
    assert min(z.start_ft for z in from_crosswalk) == pytest.approx(77.0 - CROSSWALK_SETBACK_FT)
    assert max(z.end_ft for z in from_crosswalk) == pytest.approx(123.0 + CROSSWALK_SETBACK_FT)

    # And it must actually reach the parking, not merely be listed: a zone nobody subtracts is
    # not a setback.
    runs = parkable_runs_ft(state, "east", "left", {"east": (30.0,)})
    for start_ft, end_ft in runs:
        assert end_ft <= 77.0 - CROSSWALK_SETBACK_FT or start_ft >= 123.0 + CROSSWALK_SETBACK_FT

    assert "4242" in " ".join(z.reason for z in from_crosswalk), (
        "a surveyed crosswalk's setback has to cite the way it was read off")


def test_an_intersection_with_no_paint_still_carries_the_crosswalk_setback():
    """N.J.S.A. 39:1-1: a crosswalk exists "either marked or unmarked … at each approach of
    every roadway intersection".

    So the setback is not a function of whether a surveyor traced a zebra. Making it one would
    report the SURVEY's coverage as if it were the LAW's reach - and OSM has crossings traced at
    Blackwell and none at Model Avenue, two intersections 130 ft apart on the same street.
    """
    from src.geometry.cross_streets import CrossStreetCrosswalk

    cross = _a_cross_street(crosswalks=(CrossStreetCrosswalk(80.0, is_surveyed=False),
                                        CrossStreetCrosswalk(120.0, is_surveyed=False)))
    state = a_state(length_ft=300.0)
    state.cross_streets = {"east": [cross]}

    zones = no_parking_zones_ft(state, "east", "left", {"east": (30.0,)})
    placed = [z for z in zones if "unmarked crosswalk" in z.reason]
    assert len(placed) == 2, f"got {[z.reason for z in zones]}"
    assert min(z.start_ft for z in placed) == pytest.approx(80.0 - CROSSWALK_SETBACK_FT)
    for zone in placed:
        assert "position estimated" in zone.reason, (
            "an estimated crosswalk must say so - the setback is the law's, the position is ours")


def test_the_crosswalk_arm_only_daylights_the_kerb_the_street_joins():
    """A T-junction does not daylight the kerb opposite it, and that is true of both arms.

    The side line arm already honoured `sides`; adding the crosswalk arm beside it is exactly
    the kind of change that reintroduces a bug one line below the one it fixes.
    """
    from src.geometry.cross_streets import CrossStreetCrosswalk

    cross = _a_cross_street(sides=("left",),
                            crosswalks=(CrossStreetCrosswalk(80.0, is_surveyed=False),
                                        CrossStreetCrosswalk(120.0, is_surveyed=False)))
    state = a_state(length_ft=300.0)
    state.cross_streets = {"east": [cross]}

    right = no_parking_zones_ft(state, "east", "right", {"east": (30.0,)})
    assert not [z for z in right if "Blackwell" in z.reason], (
        "the kerb opposite a T-junction got a setback from it")


# --------------------------------------------------------------------------
# The hatching has to REACH the crossing, or it is not daylighting
# --------------------------------------------------------------------------

@needs_source_data
def test_the_daylight_hatching_reaches_the_crossing_it_daylights(site_models):
    """A hatched zone that stops short of the crosswalk does not daylight it.

    Daylighting works by keeping the approach to a crossing clear of parked cars, and the
    stretch that matters most is the one immediately beside the crossing - that is the car that
    blocks the sight line. Hatching that stops 7 ft short leaves exactly that space unmarked, and
    an unmarked setback beside marked hatching reads as permission to park in it. The treatment
    is then worse than nothing: it has drawn a boundary in the wrong place.

    It stopped short for a reason that had nothing to do with the statute. Every kerbside
    marking is built on the stations where the kerb is TRACED, which is right for a design
    choice and wrong for a statement of law: W Broad & Louellen's south kerb is traced only from
    station 60.3 against a statutory zone of 0-93.3, so 92% of the zone was hatched and the
    missing 8% was the part against the crossing. See leg_frame.paint_stations.

    Measured as the DISTANCE from the hatching to the crossing band, which is the thing a person
    looks at, and allowed to be the striper's gap and no more.
    """
    import contextlib
    import io

    from src.geometry.markings import DAYLIGHT_FILL
    from src.geometry.paint import PAINT_TO_CROSSWALK_GAP_FT
    from src.site import load_site_scenarios, run_scenario
    from tests.test_sites import resolved_scene, scene_props

    # A hair over the striper's gap: the fill is cut against the crossing band buffered by
    # exactly that, so anything further out is the zone failing to reach.
    allowed_ft = PAINT_TO_CROSSWALK_GAP_FT + 0.25
    # NO EXEMPTION FOR THE JUNCTION'S OWN MOUTH, and there was one here for about an hour. When
    # the junction first became a kerb opening its mouth ended at the CORNER RETURN, which on two
    # of these kerbs sits well outside the crossing - 15.3 ft on W Broad & Louellen's south kerb,
    # 13.2 ft on Greenwood Ave north's - so the hatching was cut back to the corner and this test
    # was widened to accept "reaches the mouth" instead. That was the test being talked out of the
    # property it exists for. The mouth now ends AT the crossing where one is painted
    # (paint.junction_mouths_ft), because filling the corner outside the crosswalk is the whole
    # point of a painted curb extension - so the original assertion holds again, unqualified.

    checked = 0
    for site, model in site_models.items():
        scenarios = load_site_scenarios(site)
        for name in sorted(n for n in dir(scenarios) if n.startswith("build_")):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
                scene = resolved_scene(model, state)
                paint = scene.build_paint(scene_props(model, state, scene))

            for leg_name in sorted(scene.marked_crosswalks):
                band = scene.crosswalk_bands.get(leg_name)
                if band is None or band.is_empty:
                    continue
                for side in ("left", "right"):
                    fills = [p.geometry for p in paint if p.kind is DAYLIGHT_FILL
                             and p.leg == leg_name and p.side == side]
                    if not fills:
                        continue        # no marked parking on this kerb, so no zone to draw
                    checked += 1
                    nearest_ft = min(f.distance(band) for f in fills)
                    assert nearest_ft <= allowed_ft, (
                        f"{site}/{name}: the daylight hatching on {leg_name}/{side} stops "
                        f"{nearest_ft:.2f} ft short of the crossing it is there to daylight "
                        f"(the striper's gap is {PAINT_TO_CROSSWALK_GAP_FT:.1f} ft) - the bare "
                        f"stretch beside a crossing is the parking space daylighting exists to "
                        f"remove")
    assert checked, "no marked crossing had a daylight zone beside it, so this asserted nothing"


@needs_source_data
def test_kerbside_hatching_reaches_the_crossing_on_an_UNMARKED_leg_too(site_models):
    """The sibling of the test above, for the legs it does not look at.

    That one sweeps `scene.marked_crosswalks`, which is every leg whose crossing is PAINTED - so
    the two W Broad legs at Louellen, both unmarked, were outside every assertion in the suite.
    Those are the legs the arrows landed on. An unmarked approach still has a crosswalk (N.J.S.A.
    39:1-1) and still has a statutory setback measured from it (R.S. 39:4-138(e), which
    no_parking_zones_ft cites by that name here), so the hatching beside it has the same job and
    the same failure mode: a bare stretch beside a crossing is where a car parks and blocks the
    sight line.

    MEASURED FROM max(crossing reach, where the kerb is TRACED), and the second term is not a
    let-off. leg_frame.paint_stations refuses to draw a design-choice marking on ground nobody
    surveyed, deliberately and with its reasons; every kerb at all five sites is traced only from
    station 12-58, because OSM traces the block and not the corner. So this pins the part of the
    rule the code owns - the corner return must not hold the hatching back - and reports the
    tracing separately rather than blaming the geometry for missing data.

    Before the mouth was resolved for unpainted crossings, `w_broad_st_southwest right` started
    its hatching at 63.71 ft: the corner clearance to the foot, 31.7 ft outside a crossing
    reaching 32.06, on a kerb the statute closes from 0 to 85.7.
    """
    import contextlib
    import io

    import numpy as np

    from src.geometry.model import curb_station_span, station_offset_many
    from src.geometry.paint import PAINT_TO_CROSSWALK_GAP_FT
    from src.render.crosswalks import crosswalk_reach_on_leg_side_ft
    from src.site import load_site_scenarios, run_scenario
    from tests.test_sites import resolved_scene, scene_props

    # The striper's gap, plus the sample step the kerbside strip is built on: a zone cut against
    # the mouth lands on that grid, so a fraction of a step is the cut and not a shortfall.
    allowed_ft = PAINT_TO_CROSSWALK_GAP_FT + 2.5
    checked, unmarked_seen = 0, 0
    for site, model in site_models.items():
        scenarios = load_site_scenarios(site)
        for name in sorted(n for n in dir(scenarios) if n.startswith("build_")):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
                scene = resolved_scene(model, state)
                paint = scene.build_paint(scene_props(model, state, scene))

            for leg_name, leg in sorted(model.legs.items()):
                if leg_name in scene.marked_crosswalks:
                    continue            # the test above owns these
                band = scene.crosswalk_bands.get(leg_name)
                if band is None or band.is_empty:
                    continue
                unmarked_seen += 1
                for side in ("left", "right"):
                    reach_ft = crosswalk_reach_on_leg_side_ft(leg, side, band,
                                                               beyond_the_tracing=True)
                    if not reach_ft:
                        continue
                    span = curb_station_span(leg, side)
                    # Where a marking may begin at all: past the crossing, and not on ground
                    # nobody traced.
                    floor_ft = max(reach_ft, span[0] if span else 0.0)
                    sign = 1 if side == "left" else -1
                    starts = []
                    for piece in paint:
                        if piece.leg != leg_name or str(piece.side) != side:
                            continue
                        if not piece.covers_area:
                            continue
                        geometry = piece.geometry
                        coords = np.asarray(
                            geometry.exterior.coords if geometry.geom_type == "Polygon"
                            else geometry.coords, dtype=float)
                        stations, offsets = station_offset_many(leg.centerline, coords)
                        on_this_side = (offsets * sign) > 0
                        if on_this_side.any():
                            starts.append(float(stations[on_this_side].min()))
                    if not starts:
                        continue        # nothing hatched on this kerb in this scenario
                    checked += 1
                    bare_ft = min(starts) - floor_ft
                    assert bare_ft <= allowed_ft, (
                        f"{site}/{name}: the kerbside hatching on {leg_name}/{side} starts "
                        f"{min(starts):.2f} ft out, leaving {bare_ft:.2f} ft bare past the "
                        f"furthest thing entitled to hold it back (crossing reach "
                        f"{reach_ft:.2f}, tracing starts "
                        f"{span[0]:.2f} ft" f") - the corner return is not one of them")
    assert unmarked_seen, "no unmarked approach was reached, so this asserted nothing"
    assert checked, "no unmarked approach had hatching beside it, so this asserted nothing"
