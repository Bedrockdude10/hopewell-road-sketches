"""Every real site, every scenario, checked against the scene invariants.

The unit tests prove each invariant fires on the failure it was written for. This proves
the four actual junctions satisfy them - existing conditions and all three proposals - which
is the claim the renders make. It runs against the committed OSM snapshot, so it fails when
this repo's geometry changes, not when someone re-traces a kerb in OSM.
"""
import contextlib
import io

import numpy as np
import pytest

from src.checks import check_scene
from src.geometry.model import build_pavement_polygon
from src.geometry.treatments import DesignState
from src.render.crosswalks import (CROSSWALK_DEPTH_M, crosswalk_bands_ft, resolve_crosswalk_offsets,
                                   resolve_crosswalk_skews, resolve_stop_bar_offsets, stop_bar_bands_ft)
from src.render.coords import FT_TO_M
from src.render.props import build_props
from src.geometry.paint import curbside_paint_ft
from src.site import load_site_scenarios, run_scenario
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_stop_lines,
                                     fetch_street_furniture, fetch_traffic_control)

from tests.conftest import SITES, needs_source_data

# Whatever each site's scenarios.py actually defines. Naming the scenarios here instead
# meant that when the proposals were cleared out for re-auditing, nine tests started
# skipping and nothing said so - the demo scenario, the one every render in the repo shows,
# went unchecked.
def scenario_builders(site):
    scenarios = load_site_scenarios(site)
    return {name: getattr(scenarios, name) for name in dir(scenarios)
            if name.startswith("build_") and name != "build_baseline"
            and callable(getattr(scenarios, name))}


def marked_crosswalks(model):
    """Legs that actually carry a painted crossing - not merely a resolved offset."""
    return set(model.config["intersection"].get("existing_marked_crosswalks", []))


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
        bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement)
        # props and offsets both go in: without them the paint is built with no knowledge of
        # the stop signs and hydrants, and check_parking_is_legal then has nothing to check
        # against - a test that passes by being handed nothing.
        paint = curbside_paint_ft(state, offsets, model.center_ft, bands, props,
                                   marked_crosswalks=marked_crosswalks(model))
        return check_scene(model, state, props, pavement, crosswalk_bands=bands,
                            stop_bars=stop_bar_bands_ft(state, stop_offsets, skews),
                            paint=paint, crosswalk_offsets=offsets)


def fatal(violations):
    return [v for v in violations if v.fatal]


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_existing_conditions_satisfy_the_invariants(site, site_models):
    violations = fatal(scene_violations(site_models[site], DesignState.from_model(site_models[site])))
    assert not violations, "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_scenario_satisfies_the_invariants(site, site_models):
    """A proposal repaints the junction, which is exactly when paint ends up somewhere it
    should not be - over a crossing, or over the kerb onto the footway."""
    model = site_models[site]
    builders = scenario_builders(site)
    assert builders, f"{site} defines no scenarios - this test would silently check nothing"
    for name, builder in sorted(builders.items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
        violations = fatal(scene_violations(model, state))
        assert not violations, f"{site}/{name}:\n" + "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_paint_is_drawn_over_a_kerb(site, site_models):
    """Stated on its own so a failure names itself, like the furniture check.

    A marking meets the kerb; it never crosses it. Curbside strips used to be built by
    pairing two substrings taken along different lines, which cut their two boundaries at
    unrelated stations and pushed the paint onto the footway.
    """
    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
        violations = [v for v in fatal(scene_violations(model, state))
                      if v.check == "paint_over_the_curb"]
        assert not violations, f"{site}/{name}:\n" + "\n".join(str(v) for v in violations)


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
    the way in - which is how half the traced ways at two sites went missing.

    W Broad & Louellen carried an xfail here for "two sides not traced in OSM yet". They
    were traced; the width fit was throwing them away, because it judged each traced vertex
    against a half-width it had derived from the vertices it had already kept. All six
    sides pass now. See src/geometry/intersection.py:_fit_legs_to_traced_kerbs.
    """
    model = site_models[site]
    untraced = [f"{name} {side}" for name, leg in model.legs.items()
                for side in ("left", "right") if side not in leg.traced_sides]
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
def test_each_kerb_gets_the_paint_its_restriction_and_width_allow(site, site_models):
    """The full rule, which has three outcomes rather than two.

    Restricted -> crossed hatching. Unrestricted with room -> stalls. Unrestricted WITHOUT
    room -> nothing: hatching it would read as "no parking", the opposite of what OSM
    records, and a 1.1 ft parking lane is not a parking lane. A leg too narrow for two
    target lanes gets no kerbside paint at all.
    """
    from src.geometry.intersection import parking_is_restricted, parking_restriction_by_side
    from src.geometry.treatments import (MIN_MARKED_PARKING_DEPTH_FT,
                                          PARKING_STALL_DEPTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT, apply_osm_parking)

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)

    for leg_name, leg in model.legs.items():
        allowance_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        sides = parking_restriction_by_side(model.leg_osm_tags.get(leg_name, {}),
                                            model.leg_osm_aligned.get(leg_name, True))
        for side in ("left", "right"):
            hatched = (leg_name in state.lane_narrowing
                       and side in state.lane_narrowing_sides.get(leg_name, ("left", "right")))
            stalls = (leg_name, side) in state.parking_zones

            if allowance_ft <= 0:
                assert not hatched and not stalls, (
                    f"{leg_name} is {leg.curb_to_curb_ft:.1f} ft - too narrow for two "
                    f"{TARGET_LANE_WIDTH_FT:.0f} ft lanes, so it must get no kerbside paint")
            elif parking_is_restricted(sides[side]):
                assert hatched and not stalls, f"{leg_name} {side} is restricted but got stalls"
            elif allowance_ft >= MIN_MARKED_PARKING_DEPTH_FT:
                assert stalls and not hatched, f"{leg_name} {side} is parkable but got hatching"
                zone = state.parking_zones[(leg_name, side)]
                assert zone["depth_ft"] == pytest.approx(PARKING_STALL_DEPTH_DEFAULT_FT), (
                    "a stall is a standard width - the leftover goes to the kerb buffer, it "
                    "does not make the stall wider")
                assert zone["curb_offset_ft"] == pytest.approx(
                    allowance_ft - PARKING_STALL_DEPTH_DEFAULT_FT, abs=0.01)
            else:
                assert hatched and not stalls, (
                    f"{leg_name} {side} has only {allowance_ft:.1f} ft spare - too little for a "
                    f"stall, so it must be hatched as buffer to hold the lane at target")


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


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_osm_parking_never_narrows_a_lane_below_target(site, site_models):
    """On the real geometry: every kerb this paints must leave an 11 ft lane beside it."""
    from src.checks import check_travel_lanes
    from src.geometry.treatments import apply_osm_parking

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)
    violations = check_travel_lanes(state)
    assert not violations, "\n".join(str(v) for v in violations)


def test_a_street_too_narrow_for_two_target_lanes_gets_no_paint():
    """Painting a 19 ft street down to 11 ft lanes is impossible; marking parking there
    anyway is what produced 1.7 ft lanes. It must decline instead.

    On a built leg, not a real one: this used to run against louellen_st_west, which was
    "19.3 ft wide" only because its south kerb had been discarded by the width fit
    (src/geometry/intersection.py:_fit_legs_to_traced_kerbs). Measuring it properly made it
    42 ft, the test passed vacuously, and the rule it guards went unchecked - no leg at any
    of the four junctions is under 22 ft. A width is the input to this rule, so the test
    supplies one.
    """
    from shapely.geometry import LineString

    from src.geometry.model import Leg
    from src.geometry.treatments import apply_osm_parking

    narrow = Leg(name="narrow", centerline=LineString([(0, 0), (130, 0)]), curb_to_curb_ft=19.3)
    state = DesignState(legs={"narrow": narrow}, corner_fillets={})

    class NoTags:
        leg_osm_tags: dict = {}
        leg_osm_aligned: dict = {}

    with contextlib.redirect_stdout(io.StringIO()) as out:
        state = apply_osm_parking(state, NoTags())
    assert "narrow" not in state.lane_narrowing
    assert not state.parking_zones
    assert "too narrow for two 11 ft lanes" in out.getvalue()


@needs_source_data
def test_an_unrestricted_kerb_too_narrow_to_park_is_hatched_not_widened(site_models):
    """Leaving it bare abandons the target: E Broad's 36 ft legs went to 18 ft lanes.

    Hatching beside a travel lane reads as buffer/shoulder - the same thing the strip between
    a parking lane and the kerb already is - so it holds the lane at target without claiming
    a parking restriction OSM doesn't record.
    """
    from src.geometry.treatments import (MIN_MARKED_PARKING_DEPTH_FT, TARGET_LANE_WIDTH_FT,
                                          apply_osm_parking)

    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)

    leg = "e_broad_st_east"
    spare_ft = model.legs[leg].curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
    assert 0 < spare_ft < MIN_MARKED_PARKING_DEPTH_FT, (
        f"{leg} is {model.legs[leg].curb_to_curb_ft:.1f} ft, which leaves {spare_ft:.1f} ft spare - "
        f"that is no longer the case this test is about. Pick a leg that is.")
    assert "left" in state.lane_narrowing_sides.get(leg, ())
    assert (leg, "left") not in state.parking_zones
    lane_ft = model.legs[leg].curb_to_curb_ft / 2 - state.lane_narrowing[leg]
    assert lane_ft == pytest.approx(TARGET_LANE_WIDTH_FT, abs=0.05)


def test_completing_centerlines_only_fills_real_gaps():
    """A leg with NO centerline gets one. A leg that already has markings is left alone -
    upgrading a dashed line to a no-passing double is a sight-line judgement, not a gap."""
    from src.geometry.treatments import complete_centerlines

    state = DesignState(legs={}, corner_fillets={}, centerline_styles={
        "unmarked": "none", "dashed": "single_yellow_dashed", "double": "double_yellow"})
    completed = complete_centerlines(state)
    assert completed.centerline_styles["unmarked"] == "double_yellow"
    assert completed.centerline_styles["dashed"] == "single_yellow_dashed"
    assert completed.centerline_styles["double"] == "double_yellow"


@needs_source_data
def test_the_proposal_adds_the_missing_greenwood_centerline(site_models):
    """Greenwood Ave south of Broad has no centerline paint today - confirmed by street-view
    review - so EXISTING must show none and the proposal must add one."""
    model = site_models["broad_st_greenwood"]
    baseline = DesignState.from_model(model)
    assert baseline.centerline_styles["greenwood_ave_south"] == "none"

    with contextlib.redirect_stdout(io.StringIO()):
        proposed = run_scenario(load_site_scenarios("broad_st_greenwood").build_demo_scenario,
                                 baseline, model)
    assert proposed.centerline_styles["greenwood_ave_south"] == "double_yellow"


def test_a_taper_is_refused_when_there_is_no_room_for_one():
    """A taper runs from the straight run's start INWARD to the curb.

    When the crosswalk sits further out than the corner return, target overtakes anchor and
    there is nothing to taper across - solving the arc anyway sweeps it backwards, which is
    what mangled the hatching on Princeton Ave's north leg (anchor 27.5 ft, target 28.6 ft)
    while the south leg, whose target sits inside its anchor, looked correct.
    """
    from shapely.geometry import LineString

    from src.geometry.model import (Leg, lane_narrowing_taper_ft,
                                     lane_narrowing_taper_polygons_ft)

    leg = Leg(name="east", centerline=LineString([(0, 0), (120, 0)]), curb_to_curb_ft=30.0)
    assert lane_narrowing_taper_ft(leg, 4.0, anchor_ft=27.5, target_ft=28.6) == []
    assert lane_narrowing_taper_polygons_ft(leg, 4.0, anchor_ft=27.5, target_ft=28.6) == []
    # ...and the ordinary case still produces one.
    assert lane_narrowing_taper_ft(leg, 4.0, anchor_ft=30.0, target_ft=20.0)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_lane_narrowing_starts_clear_of_the_crosswalk(site, site_models):
    """The straight run has to clear BOTH the corner return and the crossing.

    Anchoring on the corner clearance alone ran the paint to within 3.9 ft of Princeton Ave
    north's crossing where CROSSWALK_CLEARANCE_FT of room was intended.
    """
    from src.geometry.model import leg_clearance_ft
    from src.render.crosswalks import CROSSWALK_CLEARANCE_FT

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario_for(site, model)
        offsets = resolve_crosswalk_offsets(state, fetch_crossings(model.center_wgs84, radius_m=130))

    for leg_name in state.lane_narrowing:
        target_ft = offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
        anchor_ft = max(leg_clearance_ft(leg_name, state.legs, state.corner_fillets), target_ft)
        assert anchor_ft >= target_ft - 1e-9, (
            f"{leg_name}'s narrowing starts inside the crosswalk clearance")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_proposed_crosswalk_is_continental(site, site_models):
    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario_for(site, model)
    assert state.crosswalk_styles, "the proposal should set a style on every leg"
    assert set(state.crosswalk_styles.values()) == {"continental"}
    for leg_name in model.legs:
        assert state.crosswalk_styles.get(leg_name) == "continental", f"{leg_name} was missed"


def run_scenario_for(site, model):
    """The site's default proposal, built from its baseline."""
    return run_scenario(load_site_scenarios(site).build_demo_scenario,
                         DesignState.from_model(model), model)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_painted_marking_overlaps_a_crosswalk(site, site_models):
    """Crosswalks outrank every other marking, on the real geometry of every site."""
    import tempfile
    from pathlib import Path
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from src.render.coords import FT_TO_M
    from src.render.export import export_scenario

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario_for(site, model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        out = Path(tempfile.mkdtemp()) / "geometry.json"
        export_scenario(model, state, "proposed", out, crossings=crossings)
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)

    import json

    from src.render.crosswalks import crosswalk_reaches_ft

    exported = json.loads(out.read_text())
    # Built exactly as export_scenario builds them - bounded by the pavement, and with the
    # two-pass reaches that keep adjoining crossings off each other. Reconstructing them from
    # the bare offsets gives LARGER bands than the render actually uses, so the test would be
    # grading geometry nothing draws.
    with contextlib.redirect_stdout(io.StringIO()):
        pavement = build_pavement_polygon(state.corner_fillets)
    marked = marked_crosswalks(model)
    bands = unary_union(list(crosswalk_bands_ft(
        state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement,
        crosswalk_reaches_ft(state, offsets, skews, pavement, marked)).values()))

    def to_ft(points):
        return LineString([(model.center_ft.x + x / FT_TO_M, model.center_ft.y + y / FT_TO_M)
                           for x, y, *_ in points])

    for key in ("parking_buffer_hatch_lines", "lane_narrowing_hatch_lines"):
        for points in exported.get(key, []):
            stroke = to_ft(points)
            assert not stroke.intersects(bands), f"a {key} stroke lies on a crosswalk"
            assert stroke.length >= 1.0, f"a {key} stroke is {stroke.length:.2f} ft - a clipping stub"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_proposal_marks_illegal_parking(site, site_models):
    """R.S. 39:4-138 on the real junctions, stated on its own so a failure names itself.

    A stall painted within 25 ft of a crossing, 50 ft of a stop sign or 10 ft of a hydrant is
    a drawing of something that cannot lawfully be built. See src/geometry/daylighting.py.
    """
    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
        violations = [v for v in fatal(scene_violations(model, state))
                      if v.check == "parking_inside_a_legal_setback"]
        assert not violations, f"{site}/{name}:\n" + "\n".join(str(v) for v in violations)


@needs_source_data
def test_the_proposal_marks_the_daylight_zone(site_models):
    """Daylighting is the POINT of the treatment, so the zone has to actually be painted.

    The setback was already law and already respected - it was just left as bare asphalt
    beside a marked stall, which reads as more stall. If this ever returns nothing, the
    proposals have quietly stopped daylighting anything.
    """
    import contextlib as _contextlib

    model = site_models["broad_st_greenwood"]
    with _contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario_for("broad_st_greenwood", model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)
        props = build_props(model, state, offsets, model.center_ft,
                             fetch_traffic_control(model.center_wgs84, radius_m=60),
                             fetch_street_furniture(model.center_wgs84, radius_m=130),
                             crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
        pavement = build_pavement_polygon(state.corner_fillets)
        bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement)
        paint = curbside_paint_ft(state, offsets, model.center_ft, bands, props,
                                   marked_crosswalks=marked_crosswalks(model))

    daylight = [p for p in paint if p.kind.startswith("daylight")]
    assert daylight, "no daylighting is marked anywhere at Broad & Greenwood"
    assert any(p.is_fill and p.geometry.area > 50 for p in daylight), \
        "the daylight zones are all slivers - the treatment is not actually being drawn"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_kind_of_paint_reaches_the_3d_render(site, site_models):
    """A marking built by paint.py that no export list claims is invisible in 3D.

    It happened twice in one sitting: renaming buffer_taper_* to daylight_taper_* orphaned
    the taper, and adding daylight_fill never wired it up. Both were built correctly, both
    appeared in the plan view, neither reached the render, and nothing raised. The plan view
    is supposed to show what the render will show, so a silent divergence is the worst
    possible failure here.
    """
    import contextlib as _contextlib

    from src.render.export import PAINT_KIND_LISTS, PAINT_KINDS_NOT_IN_LISTS

    rendered = set(PAINT_KINDS_NOT_IN_LISTS).union(*PAINT_KIND_LISTS.values())
    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with _contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
            crossings = fetch_crossings(model.center_wgs84, radius_m=130)
            offsets = resolve_crosswalk_offsets(state, crossings)
            skews = resolve_crosswalk_skews(state, crossings)
            props = build_props(model, state, offsets, model.center_ft,
                                 fetch_traffic_control(model.center_wgs84, radius_m=60),
                                 fetch_street_furniture(model.center_wgs84, radius_m=130),
                                 crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
            try:
                pavement = build_pavement_polygon(state.corner_fillets)
            except ValueError:
                pavement = None
            bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement)
            paint = curbside_paint_ft(state, offsets, model.center_ft, bands, props,
                                   marked_crosswalks=marked_crosswalks(model))
        orphaned = {p.kind for p in paint} - rendered
        assert not orphaned, (f"{site}/{name}: paint kinds built but never rendered in 3D: "
                              f"{sorted(orphaned)} - add them to export.PAINT_KIND_LISTS")


def test_no_export_list_names_a_kind_paint_never_builds():
    """The other direction: a stale name in the table renders nothing and hides a typo."""
    import inspect

    from src.geometry import paint as paint_module
    from src.render.export import PAINT_KIND_LISTS

    source = inspect.getsource(paint_module)
    for list_name, kinds in PAINT_KIND_LISTS.items():
        for kind in kinds:
            assert f'"{kind}"' in source, (
                f"export.PAINT_KIND_LISTS[{list_name!r}] names {kind!r}, which "
                f"src/geometry/paint.py never produces")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_two_markings_are_painted_over_each_other(site, site_models):
    """Real paint is opaque and applied once. Stated on its own so a failure names itself."""
    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
        violations = [v for v in fatal(scene_violations(model, state))
                      if v.check == "markings_collide"]
        assert not violations, f"{site}/{name}:\n" + "\n".join(str(v) for v in violations)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_curbside_paint_ends_against_its_crossing(site, site_models):
    """Where a leg has a painted crossing, the hatching runs up to it and is cut by it.

    That cut IS the design: the crossing trims the zone along its own edge, which on a
    skewed crossing is a diagonal, and the diagonal meeting the straight lane-edge line is
    the right-angled rim you see on a real street. So this asserts both halves - the paint
    gets there rather than stopping short, and it does not cross the line.

    An earlier version asserted the opposite: that the backstop clip must never have
    anything to do. That was right while a taper was supposed to resolve itself back to the
    kerb BEFORE the crossing, and wrong once the crossing became the thing to end against.
    """
    from src.geometry.paint import PAINT_TO_CROSSWALK_GAP_FT

    model = site_models[site]
    marked = marked_crosswalks(model)
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
            paint, bands = paint_and_bands(model, state)

        for leg_name in sorted(marked):
            band = bands.get(leg_name)
            if band is None or band.is_empty:
                continue
            near = [p for p in paint if p.leg == leg_name and p.is_fill]
            if not near:
                continue
            on_it = [p.kind for p in near if p.geometry.intersection(band).area > 0.5]
            assert not on_it, f"{site}/{name}/{leg_name}: {on_it} painted on the crossing"
            gap_ft = min(p.geometry.distance(band) for p in near)
            assert gap_ft >= PAINT_TO_CROSSWALK_GAP_FT - 0.3, \
                f"{site}/{name}/{leg_name}: paint {gap_ft:.2f} ft from the crossing"
            assert gap_ft <= PAINT_TO_CROSSWALK_GAP_FT + 2.0, \
                (f"{site}/{name}/{leg_name}: hatching stops {gap_ft:.1f} ft short of the "
                 f"crossing instead of ending against it")



def paint_and_bands(model, state):
    """The paint and the crossing bands for one state, exactly as the renderers build them."""
    crossings = fetch_crossings(model.center_wgs84, radius_m=130)
    offsets = resolve_crosswalk_offsets(state, crossings)
    skews = resolve_crosswalk_skews(state, crossings)
    props = build_props(model, state, offsets, model.center_ft,
                         fetch_traffic_control(model.center_wgs84, radius_m=60),
                         fetch_street_furniture(model.center_wgs84, radius_m=130),
                         crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
    try:
        pavement = build_pavement_polygon(state.corner_fillets)
    except ValueError:
        pavement = None
    bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement)
    paint = curbside_paint_ft(state, offsets, model.center_ft, bands, props,
                               marked_crosswalks=marked_crosswalks(model))
    return paint, bands


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_adjoining_crossings_do_not_paint_over_each_other(site, site_models):
    """At a shared corner two crossings reach for the same kerb.

    Each was measured on its own, so Greenwood north's bars and Broad east's overlapped by
    2.07 sq ft of doubled paint - invisible to markings_collide, which only inspects the
    curbside paint list, and to every crossing check, which looks at one band at a time.
    """
    from src.render.crosswalks import crosswalk_reaches_ft

    model = site_models[site]
    marked = marked_crosswalks(model)
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario_for(site, model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)
        try:
            pavement = build_pavement_polygon(state.corner_fillets)
        except ValueError:
            pytest.skip(f"{site} has no closed pavement ring")
        bands = crosswalk_bands_ft(
            state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement,
            crosswalk_reaches_ft(state, offsets, skews, pavement, marked))

    painted = [(name, band) for name, band in bands.items()
               if name in marked and band is not None and not band.is_empty]
    for i, (name_a, a) in enumerate(painted):
        for name_b, b in painted[i + 1:]:
            overlap = a.intersection(b).area
            assert overlap < 0.5, (f"{site}: {name_a} and {name_b} crossings overlap by "
                                   f"{overlap:.2f} sq ft")


@needs_source_data
@pytest.mark.parametrize("site", ["broad_st_greenwood", "ebroad_princeton"])
def test_the_bollard_proposals_show_their_bollards_in_the_plan_view(site, site_models):
    """A proposal whose whole point is the posts has to draw the posts, in BOTH views.

    It didn't, and it failed twice over. The plan view skipped every prop of type "bollard"
    on the reasoning that the treatment layer already drew them from state.bollard_lines -
    true for the ones standing in a parking buffer, false for the daylight-zone posts, which
    exist only as props. Untagging those got them as far as the dispatch chain, where there
    was no branch for them either, so they fell through to the generic "extras" case and came
    out as goldenrod TRIANGLES: in the picture, but not as the thing the legend says.

    So this counts markers on a real Axes. The first version asserted only that the props
    existed and were untagged, which the second bug would have sailed straight through.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from src.render.plan_view import BOLLARD_PLAN_COLOR, _draw_props

    model = site_models[site]
    builder = scenario_builders(site).get("build_proposal_daylight_bollards")
    assert builder is not None, f"{site} has no bollard proposal to check"
    with contextlib.redirect_stdout(io.StringIO()):
        state = builder(DesignState.from_model(model), model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        fig, ax = plt.subplots()
        props = _draw_props(ax, model, state, offsets,
                             fetch_traffic_control(model.center_wgs84, radius_m=60),
                             fetch_street_furniture(model.center_wgs84, radius_m=130),
                             crossings, False)

    expected = sum(1 for prop in props if prop["type"] == "bollard")
    assert expected, "the bollard proposal produced no bollards at all"
    wanted = to_rgba(BOLLARD_PLAN_COLOR)
    drawn = 0
    for collection in ax.collections:
        face = collection.get_facecolor()
        if len(face) and np.allclose(face[0], wanted, atol=0.01):
            drawn += len(collection.get_offsets())
    plt.close(fig)
    assert drawn == expected, (
        f"{expected} bollard props but {drawn} bollard markers in the plan view - they are "
        f"either being skipped or drawn as something else")
