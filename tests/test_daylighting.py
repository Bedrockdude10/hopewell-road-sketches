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
    from src.geometry.daylighting import SIDELINE_SETBACK_WITH_BULBOUT_FT

    monkeypatch.setattr(treatments, "CURB_EXTENSION_DEVICES", frozenset({"built_bulbout"}))
    # The kind has to be constructible as well as recognised: a Treatment validates itself, so
    # "built_bulbout" would be refused by ProtectDaylightZone's own constructor otherwise.
    monkeypatch.setattr(treatments, "VALID_DAYLIGHT_DEVICES",
                        (*treatments.VALID_DAYLIGHT_DEVICES, "built_bulbout"))
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
