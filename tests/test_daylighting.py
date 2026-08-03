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


class FakeState:
    def __init__(self, legs, corner_fillets=None, parking_zones=None):
        self.legs = legs
        self.corner_fillets = corner_fillets or {}
        self.parking_zones = parking_zones or {}


def a_state(length_ft=200.0, width_ft=30.0):
    leg = Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)
    return FakeState({"east": leg})


def prop(kind, x, y):
    return {"type": kind, "position_ft": (x, y)}


# --------------------------------------------------------------------------
# 39:4-138(e) - the daylighting distance
# --------------------------------------------------------------------------

def test_parking_starts_25_feet_past_the_crosswalk():
    state = a_state()
    start = legal_parking_start_ft(state, "east", "left", {"east": (30.0,)})
    assert start == pytest.approx(30.0 + CROSSWALK_SETBACK_FT)


def test_a_curb_extension_reduces_the_setback_to_ten_feet():
    """39:4-138(e), second clause: a bulbout has already taken the parking lane out of the
    sight line, so the statute lets parking resume 10 ft from the crossing."""
    state = a_state()
    state.curb_extensions = {("east", "left"): 6.0}
    start = legal_parking_start_ft(state, "east", "left", {"east": (30.0,)})
    assert start == pytest.approx(30.0 + CROSSWALK_SETBACK_WITH_BULBOUT_FT)
    assert CROSSWALK_SETBACK_WITH_BULBOUT_FT < CROSSWALK_SETBACK_FT


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
    state = FakeState({"ne": diagonal})
    hydrant_at_station_50 = prop("fire_hydrant", 50 / np.sqrt(2) - 10 / np.sqrt(2),
                                  50 / np.sqrt(2) + 10 / np.sqrt(2))
    zones = [z for z in no_parking_zones_ft(state, "ne", "left", {"ne": (10.0,)},
                                             [hydrant_at_station_50]) if "hydrant" in z.reason]
    assert zones and zones[0].start_ft == pytest.approx(50.0 - FIRE_HYDRANT_SETBACK_FT, abs=0.1)
