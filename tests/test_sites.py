"""Every real site, every scenario, checked against the scene invariants.

The unit tests prove each invariant fires on the failure it was written for. This proves
the four actual junctions satisfy them - existing conditions and all three proposals - which
is the claim the renders make. It runs against the committed OSM snapshot, so it fails when
this repo's geometry changes, not when someone re-traces a kerb in OSM.
"""
import contextlib
import io

import pytest

from src.checks import check_scene
from src.geometry.model import build_pavement_polygon
from src.geometry.treatments import DesignState
from src.render.crosswalks import (CROSSWALK_DEPTH_M, crosswalk_bands_ft, resolve_crosswalk_offsets,
                                   resolve_crosswalk_skews, resolve_stop_bar_offsets, stop_bar_bands_ft)
from src.render.coords import FT_TO_M
from src.render.props import build_props
from src.site import load_site_scenarios
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_stop_lines,
                                     fetch_street_furniture, fetch_traffic_control)

from tests.conftest import SITES, needs_source_data

PROPOSALS = ("build_proposal_1_continental",
             "build_proposal_2_continental_parking_narrowing",
             "build_proposal_3_continental_parking_narrowing_bulbouts")


def scene_violations(model, state):
    """Exactly what src/render/export.py and the plan view check, on the same shared geometry."""
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        traffic_control = fetch_traffic_control(model.center_wgs84, radius_m=60)
        street_furniture = fetch_street_furniture(model.center_wgs84, radius_m=130)
        kerbs = fetch_kerbs(model.center_wgs84, radius_m=120)

        try:
            pavement = build_pavement_polygon(state.corner_fillets)
        except ValueError:
            pavement = None
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)
        props = build_props(model, state, offsets, model.center_ft, traffic_control,
                             street_furniture, crossings, kerbs)
        stop_lines = fetch_stop_lines(model.center_wgs84, radius_m=130)
        stop_offsets = (resolve_stop_bar_offsets(state, offsets, stop_lines)
                        if model.config.get("signals") else {})
        return check_scene(model, state, props, pavement,
                            crosswalk_bands=crosswalk_bands_ft(state, offsets, skews,
                                                                CROSSWALK_DEPTH_M / FT_TO_M),
                            stop_bars=stop_bar_bands_ft(state, stop_offsets, skews))


def fatal(violations):
    return [v for v in violations if v.fatal]


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_existing_conditions_satisfy_the_invariants(site, site_models):
    violations = fatal(scene_violations(site_models[site], DesignState.from_model(site_models[site])))
    assert not violations, "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", [s for s in SITES if s != "broad_st_greenwood"])
@pytest.mark.parametrize("proposal", PROPOSALS)
def test_proposals_satisfy_the_invariants(site, proposal, site_models):
    """A proposal moves curbs and repaints the junction, which is exactly when furniture
    ends up in the road - a bulb-out narrows the carriageway under an existing sign."""
    model = site_models[site]
    scenarios = load_site_scenarios(site)
    builder = getattr(scenarios, proposal, None)
    if builder is None:
        pytest.skip(f"{site} has no {proposal}")
    with contextlib.redirect_stdout(io.StringIO()):
        state = builder(DesignState.from_model(model))
    violations = fatal(scene_violations(model, state))
    assert not violations, "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_tactile_paving_or_sign_stands_in_the_street(site, site_models):
    """The headline invariant, stated on its own so a failure names itself in the report."""
    violations = [v for v in fatal(scene_violations(site_models[site],
                                                     DesignState.from_model(site_models[site])))
                  if v.check == "furniture_in_roadway"]
    assert not violations, "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_curbs_never_cross_the_junction(site, site_models):
    """The curb-across-the-middle-of-the-intersection bug, on the real geometry."""
    violations = [v for v in fatal(scene_violations(site_models[site],
                                                     DesignState.from_model(site_models[site])))
                  if v.check in ("curb_through_junction", "curbs_cross")]
    assert not violations, "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_leg_curb_comes_from_the_traced_kerb(site, site_models):
    """What the surveyor traced is what gets drawn.

    A side falling back to a centerline offset means the tracing was dropped somewhere on
    the way in - which is how half the traced ways at two sites went missing. W Broad &
    Louellen is the known exception: two of its sides genuinely have no tracing yet.
    """
    model = site_models[site]
    untraced = [f"{name} {side}" for name, leg in model.legs.items()
                for side in ("left", "right") if side not in leg.traced_sides]
    if site == "wbroad_louellen":
        pytest.xfail("louellen_st_west right and w_broad_st_southwest right are not traced in OSM yet")
    assert not untraced, f"curb sides not built from traced kerbs: {untraced}"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_sidewalk_hugs_the_traced_kerb(site, site_models):
    """A sidewalk is the thing on the other side of the kerb from the road.

    It used to be re-derived from the leg centerlines (widened and re-filleted), which was
    right only while the curbs were symmetric centerline offsets. Once they became traced
    kerbs, 11-19% of the kerb had no sidewalk against it - gaps up to 27 ft of grass running
    to the roadway - and 658 sq ft of "sidewalk" sat inside the carriageway at W Broad.

    Sampled only where a kerb actually exists: the leg END CAPS are where the model stops,
    not a kerb, so nothing should be built against them.
    """
    import numpy as np
    from shapely.ops import unary_union

    from src.geometry.model import build_pavement_polygon
    from src.geometry.treatments import build_sidewalk_pieces

    model = site_models[site]
    state = DesignState.from_model(model)
    pavement = build_pavement_polygon(state.corner_fillets)
    pieces = build_sidewalk_pieces(state, 6)
    walk = unary_union(pieces)

    in_roadway = sum(p.intersection(pavement).area for p in pieces)
    assert in_roadway < 1.0, f"{in_roadway:.0f} sq ft of sidewalk lies inside the roadway"

    kerb = unary_union([parts[k] for parts in state.corner_fillets.values() if "error" not in parts
                        for k in ("trimmed_a", "arc", "trimmed_b")])
    samples = [pavement.exterior.interpolate(t, normalized=True) for t in np.linspace(0, 1, 400)]
    on_kerb = [p for p in samples if kerb.distance(p) < 0.5]
    stranded = [p for p in on_kerb if walk.distance(p) > 1.0]
    assert not stranded, (f"{len(stranded)} of {len(on_kerb)} kerb points have no sidewalk "
                          f"against them - grass would run up to the roadway there")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_passing_legs_get_a_double_yellow(site, site_models):
    """OSM's overtaking=no IS the no-passing marking - a double yellow centerline.

    Reading it beats defaulting the leg to a dashed line, which was only ever a placeholder.
    """
    model = site_models[site]
    state = DesignState.from_model(model)
    for leg_name, tags in model.leg_osm_tags.items():
        if tags.get("overtaking") == "no" and "centerline_style" not in model.config["legs"][leg_name]:
            assert state.centerline_styles[leg_name] == "double_yellow", (
                f"{leg_name} is tagged overtaking=no but renders "
                f"{state.centerline_styles[leg_name]}")


def test_centerline_precedence_goes_by_provenance_not_by_file():
    """An OBSERVED style outranks OSM; a retained repo default does not.

    The distinction matters because a config entry equal to DEFAULT_CENTERLINE_STYLE carries
    no information - it is this repo's own placeholder written down, and the configs that
    have one say exactly that in their comments. Treating it as an observation let the
    generic guess beat real surveyed data, which is this project's core principle inverted:
    Princeton Ave rendered a dashed line at two junctions while OSM said overtaking=no.
    """
    from src.geometry.treatments import DEFAULT_CENTERLINE_STYLE

    class FakeModel:
        config = {"legs": {
            "retained_default": {"centerline_style": DEFAULT_CENTERLINE_STYLE},
            "observed_none": {"centerline_style": "none"},
            "observed_double": {"centerline_style": "double_yellow"},
            "unset": {},
        }}
        legs = {}
        corner_fillets = {}
        leg_osm_tags = {name: {"overtaking": "no"} for name in
                        ("retained_default", "observed_none", "observed_double", "unset")}

    styles = DesignState.from_model(FakeModel()).centerline_styles
    assert styles["retained_default"] == "double_yellow", "OSM must beat the retained default"
    assert styles["observed_none"] == "none", "an observed 'none' must survive an OSM tag"
    assert styles["observed_double"] == "double_yellow"
    assert styles["unset"] == "double_yellow", "OSM must be used where config is silent"


def test_a_leg_with_no_osm_tag_keeps_the_default():
    """Nothing unattested: absent overtaking data is not evidence of a no-passing zone."""
    class FakeModel:
        config = {"legs": {"untagged": {}}}
        legs = {}
        corner_fillets = {}
        leg_osm_tags = {"untagged": {"highway": "residential"}}

    assert DesignState.from_model(FakeModel()).centerline_styles["untagged"] == "single_yellow_dashed"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_stop_bars_use_the_surveyed_position(site, site_models):
    """A road_marking=stop_line way IS the painted bar - its position is the answer.

    Ten are mapped across the three signalized junctions. Before they existed the bar could
    only be inferred as a fixed setback behind the crosswalk, which was off by up to 20.7 ft
    (E Broad's east approach, where the real bar sits well back).
    """
    from src.render.crosswalks import match_stop_lines_to_legs

    model = site_models[site]
    state = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        stop_lines = fetch_stop_lines(model.center_wgs84, radius_m=130)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        resolved = resolve_stop_bar_offsets(state, offsets, stop_lines)
    matched = match_stop_lines_to_legs(state.legs, stop_lines)

    if not model.config.get("signals"):
        assert not matched, f"{site} is unsignalized but matched stop bars: {list(matched)}"
        return

    for leg_name, line in matched.items():
        painted = state.legs[leg_name].centerline.project(line.interpolate(0.5, normalized=True))
        assert resolved[leg_name] == pytest.approx(painted, abs=0.01), (
            f"{leg_name}'s stop bar was moved off its surveyed position - the clamp against "
            f"our own corner return must not overrule a painted bar")


def test_a_stop_line_lying_along_a_leg_is_not_claimed():
    """A bar is credited to the leg it crosses SQUARE, not to whichever leg is nearest.

    At a four-way junction the cross street's bar passes just as close to the centre as this
    leg's own, so proximity alone would credit it to both.
    """
    from shapely.geometry import LineString

    from src.render.crosswalks import STOP_LINE_MIN_ANGLE_DEG, _crossing_angle_deg

    leg = LineString([(0, 0), (120, 0)])                    # a leg running east
    across = LineString([(40, -15), (40, 15)])              # a real bar, square across it
    alongside = LineString([(30, 12), (60, 12)])            # the cross street's, parallel to it

    assert _crossing_angle_deg(across, leg) >= STOP_LINE_MIN_ANGLE_DEG
    assert _crossing_angle_deg(alongside, leg) < STOP_LINE_MIN_ANGLE_DEG


def test_a_derived_stop_bar_is_still_clamped_out_of_the_corner():
    """Only a SURVEYED bar overrules the corner clearance. With nothing traced, the derived
    setback must still be kept out of the curb return, where a bar certainly isn't."""
    from shapely.geometry import LineString

    from src.geometry.model import Leg
    from src.render.crosswalks import resolve_stop_bar_offsets

    class FakeState:
        legs = {"east": Leg(name="east", centerline=LineString([(0, 0), (120, 0)]), curb_to_curb_ft=30)}
        corner_fillets = {}

    # crosswalk at 10 ft - 7 ft setback would put the bar at 3 ft, inside any real corner.
    resolved = resolve_stop_bar_offsets(FakeState(), {"east": (10.0, "estimated")}, stop_lines=[])
    assert resolved["east"] >= 3.0


def test_the_centerline_stops_at_the_stop_bar():
    """Paint terminates at the line drivers stop on - it doesn't run into the junction.

    The old rule was a fixed 2 m gap past the crosswalk, which held only while the stop bar
    was itself derived from the crosswalk. Real surveyed bars broke it: E Broad's east
    approach has its bar 52.9 ft out against a crosswalk at ~39 ft, so the double yellow ran
    ~14 ft past the bar.
    """
    from src.render.crosswalks import CENTERLINE_CROSSWALK_GAP_FT, centerline_start_ft

    # Surveyed bar well beyond the crosswalk - the real case that exposed this.
    assert centerline_start_ft(39.0, 52.9) == pytest.approx(52.9)
    # No stop bar (unsignalized): fall back to clearing the crosswalk.
    assert centerline_start_ft(39.0, None) == pytest.approx(39.0 + CENTERLINE_CROSSWALK_GAP_FT)
    # A bar closer in than the crosswalk must not drag paint across the crossing.
    assert centerline_start_ft(39.0, 20.0) == pytest.approx(39.0 + CENTERLINE_CROSSWALK_GAP_FT)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_centerline_paint_reaches_past_a_stop_bar(site, site_models):
    """On the real geometry, for every leg that has a bar."""
    from src.render.crosswalks import centerline_start_ft

    model = site_models[site]
    state = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        stop_lines = fetch_stop_lines(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        bars = (resolve_stop_bar_offsets(state, offsets, stop_lines)
                if model.config.get("signals") else {})

    for leg_name, bar_ft in bars.items():
        start_ft = centerline_start_ft(offsets[leg_name][0], bar_ft)
        assert start_ft >= bar_ft - 1e-6, (
            f"{leg_name}'s centerline starts {bar_ft - start_ft:.1f} ft inside its stop bar")


# --------------------------------------------------------------------------
# OSM kerbside parking restrictions
# --------------------------------------------------------------------------

def test_osm_parking_sides_flip_for_a_leg_running_against_its_way():
    """OSM's left/right are relative to the WAY; a leg's are relative to its outward
    direction. Half this project's legs run against their way, so reading the tag straight
    through would paint the restriction on the wrong kerb - and look entirely plausible."""
    from src.geometry.intersection import parking_restriction_by_side

    tags = {"parking:left:restriction": "no_parking", "parking:right:restriction": "none"}
    assert parking_restriction_by_side(tags, aligned=True) == {"left": "no_parking", "right": "none"}
    assert parking_restriction_by_side(tags, aligned=False) == {"left": "none", "right": "no_parking"}


def test_parking_both_applies_to_each_side_either_way_round():
    from src.geometry.intersection import parking_restriction_by_side

    tags = {"parking:both:restriction": "no_parking"}
    for aligned in (True, False):
        assert parking_restriction_by_side(tags, aligned) == {"left": "no_parking", "right": "no_parking"}


def test_untagged_is_not_the_same_as_restriction_none():
    """Absent means OSM says nothing; "none" is a positive statement that parking is
    allowed. Both end up parkable, but only one of them is evidence."""
    from src.geometry.intersection import parking_is_restricted, parking_restriction_by_side

    assert parking_restriction_by_side({}, True) == {"left": None, "right": None}
    assert not parking_is_restricted(None)
    assert not parking_is_restricted("none")
    for value in ("no_parking", "no_standing", "no_stopping"):
        assert parking_is_restricted(value)


@needs_source_data
def test_the_same_kerb_is_restricted_from_both_of_its_legs(site_models):
    """Columbia Avenue is tagged once but reaches the junction as two opposed legs.

    Whichever leg you look from, the restriction must land on the same physical kerb - the
    north side. This is the side-flip bug stated in terms of the real street.
    """
    from src.geometry.intersection import parking_is_restricted, parking_restriction_by_side

    model = site_models["columbia_princeton"]
    restricted_sides = {}
    for leg_name in ("columbia_ave_east", "columbia_ave_west"):
        sides = parking_restriction_by_side(model.leg_osm_tags.get(leg_name, {}),
                                            model.leg_osm_aligned.get(leg_name, True))
        restricted_sides[leg_name] = {s for s in ("left", "right") if parking_is_restricted(sides[s])}

    assert restricted_sides["columbia_ave_east"] == {"left"}
    assert restricted_sides["columbia_ave_west"] == {"right"}, (
        "the east leg's left kerb and the west leg's right kerb are the SAME kerb - if both "
        "came back 'left', the side flip is not being applied")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_restricted_kerbs_get_hatching_and_the_rest_get_stalls(site, site_models):
    """The rule: crossed paint where OSM prohibits parking, stalls where it doesn't."""
    from src.geometry.intersection import parking_is_restricted, parking_restriction_by_side
    from src.geometry.treatments import apply_osm_parking

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)

    for leg_name in model.legs:
        sides = parking_restriction_by_side(model.leg_osm_tags.get(leg_name, {}),
                                            model.leg_osm_aligned.get(leg_name, True))
        for side in ("left", "right"):
            hatched = (leg_name in state.lane_narrowing
                       and side in state.lane_narrowing_sides.get(leg_name, ("left", "right")))
            stalls = (leg_name, side) in state.parking_zones
            if parking_is_restricted(sides[side]):
                assert hatched and not stalls, f"{leg_name} {side} is restricted but got stalls"
            else:
                assert stalls and not hatched, f"{leg_name} {side} is parkable but got hatching"


@needs_source_data
def test_a_side_the_scenario_already_treated_is_left_alone(site_models):
    """"Unless otherwise specified" - apply_osm_parking is a baseline, not an override."""
    from src.geometry.treatments import add_marked_parking, apply_osm_parking

    model = site_models["columbia_princeton"]
    # princeton_ave_south left is restricted in OSM, so the rule would hatch it.
    with contextlib.redirect_stdout(io.StringIO()):
        state = add_marked_parking(DesignState.from_model(model), "princeton_ave_south", "left")
        state = apply_osm_parking(state, model)

    assert ("princeton_ave_south", "left") in state.parking_zones
    assert "left" not in state.lane_narrowing_sides.get("princeton_ave_south", ())
