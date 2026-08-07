"""Every real site, every scenario, checked against the scene invariants.

The unit tests prove each invariant fires on the failure it was written for. This proves
the four actual junctions satisfy them - existing conditions and all three proposals - which
is the claim the renders make. It runs against the committed OSM snapshot, so it fails when
this repo's geometry changes, not when someone re-traces a kerb in OSM.
"""
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.geometry.daylighting import no_parking_zones_ft
from src.geometry.model import (build_pavement_polygon, narrowest_half_width_ft,
                                station_offset_many)
from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.treatments import (AddCurbExtension, DesignState, LaneNarrowing, MarkedParking,
                                     UpgradeCrosswalkMarkings)
from src.render.crosswalks import (CROSSWALK_DEPTH_M, STOP_BAR_CURB_CLEARANCE_M,
                                   crosswalk_bands_ft, resolve_crosswalk_offsets,
                                   resolve_crosswalk_skews, resolve_stop_bar_offsets)
from src.render.coords import FT_TO_M
from src.render.props import build_props
from src.geometry.markings import (DAYLIGHT_EDGE_LINE, DAYLIGHT_FILL, PARKING_EDGE_LINE,
                                   STALL_DIVIDER)
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


def resolved_scene(model, state):
    """The scene geometry both renderers resolve, via the same code path they use.

    Through SceneGeometry.resolve rather than open-coded here, which is the whole point: this
    helper used to rebuild the crossing bands WITHOUT the two-pass mutual-exclusion reaches
    while claiming in its docstring to check "exactly what export.py and the plan view check".
    At W Broad & Louellen that was a 15 sq ft difference, so the test guarding every invariant
    at every site was guarding geometry no renderer built.
    """
    from src.render.scene import SceneGeometry

    return SceneGeometry.resolve(model, state, fetch_crossings(model.center_wgs84, radius_m=130))


def scene_props(model, state, scene):
    """The street furniture both renderers place, from the same fetched OSM layers."""
    return build_props(model, state, scene.crosswalk_offsets, model.center_ft,
                        fetch_traffic_control(model.center_wgs84, radius_m=60),
                        fetch_street_furniture(model.center_wgs84, radius_m=130),
                        fetch_crossings(model.center_wgs84, radius_m=130),
                        fetch_kerbs(model.center_wgs84, radius_m=120))


def scene_violations(model, state):
    """Exactly what src/render/export.py and the plan view check, on the same shared geometry."""
    with contextlib.redirect_stdout(io.StringIO()):
        scene = resolved_scene(model, state)
        # props go in: without them the paint is built with no knowledge of the stop signs and
        # hydrants, and check_parking_is_legal then has nothing to check against - a test that
        # passes by being handed nothing.
        props = scene_props(model, state, scene)
        # build_paint_and_posts, not build_paint: the props come back extended with the
        # bollards the paint places, which is what both renderers hand the check.
        paint, props = scene.build_paint_and_posts(props)
        return scene.check(props, paint)


def fatal(violations):
    return [v for v in violations if v.fatal]


def demo_paint(site):
    """(model, state, paint) for one site's default scenario, as both renderers build it."""
    from src.geometry.intersection import load_intersection_model

    with contextlib.redirect_stdout(io.StringIO()):
        model = load_intersection_model(site=site)
        builder = load_site_scenarios(site).build_demo_scenario
        state = run_scenario(builder, DesignState.from_model(model), model)
        paint, _bands = paint_and_bands(model, state)
    return model, state, paint


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
            assert state.centerline_style(leg_name) == "double_yellow", (
                f"{leg_name} is tagged overtaking=no but renders "
                f"{state.centerline_style(leg_name)}")


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

    styles = DesignState.from_model(FakeModel()).existing_centerline_styles
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

    assert DesignState.from_model(FakeModel()).centerline_style("untagged") == "single_yellow_dashed"


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
    from src.geometry.treatments import (MIN_MARKED_PARKING_DEPTH_FT,
                                          PARKING_STALL_DEPTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT, _restriction_summary,
                                          apply_osm_parking, kerbside_allowance_ft)

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)

    for leg_name, leg in model.legs.items():
        narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
        for side in ("left", "right"):
            # The MEASURED room, from the one function the implementation decides on. This
            # test used to restate the rule as `curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT`,
            # so it agreed with the nominal figure that marked 8 ft stalls on kerbs holding
            # 3.4 - a test that re-derives the rule can only ever confirm the arithmetic it
            # copied. Asking the same question the code asks is the point of having one.
            allowance_ft = kerbside_allowance_ft(leg, side)
            # Per STRETCH of kerb, not one value for the leg. Reading the whole-leg tag here
            # asked a question the street does not answer: broad_st_east's dominant way says
            # "none" while its first 79.5 ft are tagged no_parking, so this test used to agree
            # that stalls belonged on a kerb OSM forbids them on.
            at = _restriction_summary(state, leg_name, side, leg.centerline.length)
            hatched = narrowing is not None and side in narrowing.sides
            parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
            stalls = parking is not None

            if allowance_ft <= 0:
                assert not hatched and not stalls, (
                    f"{leg_name} is {leg.curb_to_curb_ft:.1f} ft - too narrow for two "
                    f"{TARGET_LANE_WIDTH_FT:.0f} ft lanes, so it must get no kerbside paint")
            elif at.restricted_throughout or (at.restricted_in_part and at.holds_no_stall):
                assert hatched and not stalls, (
                    f"{leg_name} {side}: {at.describe()}, so it must not get stalls")
            elif at.restricted_in_part and allowance_ft >= MIN_MARKED_PARKING_DEPTH_FT:
                # Marked for parking, with the restricted stretch carved back out as a
                # no-parking zone rather than the whole kerb given up.
                assert stalls and not hatched, (
                    f"{leg_name} {side}: {at.describe()}, and {at.open_ft:.0f} ft is open - the "
                    f"open part should be marked and the restriction carved out of it")
                zones = [z for z in no_parking_zones_ft(state, leg_name, side, {}) if "OSM" in z.reason]
                assert zones, "the restricted stretch must become a no-parking zone"
            elif allowance_ft >= MIN_MARKED_PARKING_DEPTH_FT:
                assert stalls and not hatched, f"{leg_name} {side} is parkable but got hatching"
                assert parking.depth_ft == pytest.approx(PARKING_STALL_DEPTH_DEFAULT_FT), (
                    "a stall is a standard width - the leftover goes to the kerb buffer, it "
                    "does not make the stall wider")
                # The kerb buffer is a COORDINATE - it and depth_ft are subtracted from the
                # nominal half to land the parking lane's inner edge on TARGET_LANE_WIDTH_FT,
                # whatever the traced kerb does out there (the stall's outer edge is the kerb
                # itself). Measured room decides WHETHER to mark; the nominal datum decides
                # WHERE. See apply_osm_parking.
                assert parking.curb_offset_ft == pytest.approx(
                    leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
                    - PARKING_STALL_DEPTH_DEFAULT_FT, abs=0.01)
            else:
                assert hatched and not stalls, (
                    f"{leg_name} {side} has only {allowance_ft:.1f} ft spare - too little for a "
                    f"stall, so it must be hatched as buffer to hold the lane at target")


@needs_source_data
def test_a_side_the_scenario_already_treated_is_left_alone(site_models):
    """"Unless otherwise specified" - apply_osm_parking is a baseline, not an override."""
    from src.geometry.treatments import apply_osm_parking

    model = site_models["columbia_princeton"]
    # princeton_ave_south left is restricted in OSM, so the rule would hatch it.
    with contextlib.redirect_stdout(io.StringIO()):
        state = DesignState.from_model(model).apply(MarkedParking(LegSide("princeton_ave_south", "left")))
        state = apply_osm_parking(state, model)

    kerb = LegSide("princeton_ave_south", "left")
    assert state.treatment_for(MarkedParking, kerb) is not None
    narrowing = state.treatment_for(LaneNarrowing, kerb.leg_target)
    assert narrowing is None or Side.LEFT not in narrowing.sides


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_osm_parking_never_narrows_a_lane_below_target(site, site_models):
    """On the real geometry: every kerb this paints must leave an 11 ft lane beside it."""
    from src.checks import SceneContext, TravelLanesKeepTheirWidth
    from src.geometry.treatments import apply_osm_parking

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)
    violations = TravelLanesKeepTheirWidth().run(SceneContext(state=state))
    assert not violations, "\n".join(str(v) for v in violations)


# The frame the committed outputs are drawn at (scripts/build_all.py --frame-scale; see the
# commit "Regenerate the outputs: the corridor, drawn to the frame at 2.2x"). The session
# fixture builds at 1x, where every leg ends inside the 130 ft its width and centre were
# measured over - so a promise that only breaks further out cannot show up there at all.
WIDE_FRAME_SCALE = 2.2


@pytest.fixture(scope="module")
def wide_site_models():
    """{site: IntersectionModel} built at the frame scale output/ is drawn at."""
    from src.geometry.intersection import load_intersection_model
    from src.render.frame import FRAME_SCALE_ENV

    previous = os.environ.get(FRAME_SCALE_ENV)
    os.environ[FRAME_SCALE_ENV] = str(WIDE_FRAME_SCALE)
    try:
        models = {}
        for site in SITES:
            with contextlib.redirect_stdout(io.StringIO()):
                models[site] = load_intersection_model(site=site)
        return models
    finally:
        if previous is None:
            os.environ.pop(FRAME_SCALE_ENV, None)
        else:
            os.environ[FRAME_SCALE_ENV] = previous


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_two_halves_of_a_through_street_line_up(site, site_models):
    """One street through a junction is one alignment, not two that nearly agree.

    Each half is modelled as its own leg off its own NJDOT route line, and nothing required
    them to meet. At W Broad & Louellen they did not - the halves are two different routes
    (CR 518 turns onto Louellen, CR 654 carries on) and their junction ends sat 3.1 ft apart,
    which drew as a dog-leg with the centreline paint kinking at the node and the kerbside
    hatching fanning off it. Greenwood Ave's two halves were 3.5 ft apart for the same reason.

    Measured sideways, which is the part that shows: a longitudinal difference between the two
    legs' origins is invisible, since each paints outward from its own start and the junction
    box sits between them.
    """
    from src.geometry.intersection import _through_leg_pairs

    model = site_models[site]
    pairs = _through_leg_pairs(model.legs)
    assert pairs, f"{site} has no through street - this test would check nothing"
    for name_a, name_b in pairs:
        start_b = np.asarray([model.legs[name_b].centerline.coords[0]], dtype=float)
        _stations, offsets = station_offset_many(model.legs[name_a].centerline, start_b)
        assert abs(float(offsets[0])) <= 0.5, (
            f"{site}: {name_b} starts {abs(float(offsets[0])):.2f} ft to the side of "
            f"{name_a}'s alignment - they are the same street and the paint runs through")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_cross_street_mouth_ends_where_the_traced_kerb_does(site, wide_site_models):
    """The treatment stops with the kerb it is drawn against, not with an assumed width.

    A cross street's mouth was taken as its own carriageway width every time - a fact about
    that street, not a measurement of THIS kerb. At E Broad & Hamilton the tracing runs out at
    station 143.4 and picks up at 173.4, while the assumed mouth was 141.7-167.7: the kerbside
    hatching restarted 5.8 ft before the kerb did and ran on over ground with no kerb beside
    it. Where the survey shows where the kerb stops, that is the mouth - the same "surveyed
    width wins" rule kerb_openings_from_model already applied to driveways.

    Checked against the tracing rather than against a remembered number, so it still holds when
    somebody re-traces that corner.
    """
    from src.geometry.intersection import kerb_lines_with_tags_ft
    from src.geometry.kerbs import OpeningSource, _place_on_a_leg_side, opens_the_kerb

    model = wide_site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = DesignState.from_model(model)
        ways = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft, model.legs)

    covered = {}
    for line, tags, _way_id in ways:
        if opens_the_kerb(tags):
            continue
        placed = _place_on_a_leg_side(line, model.legs)
        if placed is None:
            continue
        leg_name, side, start_ft, end_ft = placed
        covered.setdefault((leg_name, side), []).append((start_ft, end_ft))

    checked = 0
    for key, openings in (getattr(state, "kerb_openings", {}) or {}).items():
        spans = covered.get(key, [])
        for opening in openings:
            if opening.source is not OpeningSource.CROSS_STREET:
                continue
            # Only where the tracing actually brackets the mouth - elsewhere the assumed width
            # is the best answer there is, and this says nothing about it.
            before = [end for _s, end in spans if end <= opening.start_ft + 0.01]
            after = [start for start, _e in spans if start >= opening.end_ft - 0.01]
            if not (before and after):
                continue
            checked += 1
            assert opening.start_ft == pytest.approx(max(before), abs=0.05), (
                f"{site}/{key}: the mouth starts at {opening.start_ft:.2f} ft but the traced "
                f"kerb runs to {max(before):.2f} - the treatment must stop with the kerb")
            assert opening.end_ft == pytest.approx(min(after), abs=0.05), (
                f"{site}/{key}: the mouth ends at {opening.end_ft:.2f} ft but the traced kerb "
                f"resumes at {min(after):.2f} - the treatment must restart with the kerb")
    if not checked:
        pytest.skip(f"{site} has no cross-street mouth bracketed by traced kerb")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_edge_line_runs_unbroken_past_a_driveway(site, wide_site_models):
    """MUTCD 3B.07: an edge line is maintained across a driveway, interrupted at intersections.

    A driveway breaks what a car drives over on its way in. It does not break the line marking
    where the running lane ends, because that does not stop being true because someone can turn
    in. kerb_opening_bands said exactly this and the code did the opposite - a parking edge line
    was cut against the entrance like a stall, so at the driveway 178-204 ft along
    broad_st_east's south kerb the stalls stopped, both lines stopped, and 26 ft of kerb was
    left with nothing drawn on it.

    The stalls still stop, which is right: a stall across a driveway is a space nobody can park
    in. It is the line in front of them that carries on.
    """
    from src.geometry.markings import LINES_UNBROKEN_BY_A_DRIVEWAY

    model = wide_site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(load_site_scenarios(site).build_demo_scenario,
                             DesignState.from_model(model), model)
        paint = resolved_scene(model, state).build_paint()

    openings = {key: spans for key, spans in getattr(state, "kerb_openings", {}).items() if spans}
    if not openings:
        pytest.skip(f"{site} has no traced kerb openings")

    checked = 0
    for (leg_name, side), spans in openings.items():
        leg = model.legs.get(leg_name)
        if leg is None:
            continue
        # Every piece of this kind on this kerb, as station ranges. Asked of the SET and not of
        # each piece: a broken line is exactly a set of pieces that each stop short of the
        # driveway, so a per-piece test skips the very case it exists for - which this one did
        # until it was checked against the defect.
        runs = []
        for piece in paint:
            if (piece.leg, piece.side) != (leg_name, side):
                continue
            if piece.kind not in LINES_UNBROKEN_BY_A_DRIVEWAY:
                continue
            stations, _offsets = station_offset_many(
                leg.centerline, np.asarray(piece.geometry.coords, dtype=float))
            runs.append((float(stations.min()), float(stations.max())))
        if not runs:
            continue
        # A cross street's mouth is a kerb opening too, and MUTCD interrupts an edge line
        # THERE - the distinction it draws is a driveway "that does not meet the definition of
        # an intersection". Blackwell Avenue's mouth (278-304 ft on broad_st_east's north kerb)
        # is the case, and the code already gets it right: the line stops. Demanding continuity
        # across it was the test being wrong about the standard, not the code.
        mouths = [cross.mouth_ft for cross in getattr(model, "cross_streets", {}).get(leg_name, [])
                  if side in cross.sides]
        for opening in spans:
            if any(near <= opening.end_ft and far >= opening.start_ft for near, far in mouths):
                continue
            # Only an opening the line reaches on both sides - past the end of the marked run
            # there is nothing to carry through.
            if not (min(lo for lo, _hi in runs) < opening.start_ft
                    and max(hi for _lo, hi in runs) > opening.end_ft):
                continue
            checked += 1
            assert any(lo <= opening.start_ft and hi >= opening.end_ft for lo, hi in runs), (
                f"{site}/{leg_name} {side}: the edge of the travelled way is broken across the "
                f"driveway at {opening.start_ft:.0f}-{opening.end_ft:.0f} ft - pieces run "
                + ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in sorted(runs))
                + ". MUTCD 3B.07 maintains an edge line across a driveway")
    if not checked:
        pytest.skip(f"{site} has no driveway inside a marked run")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_marked_stall_fits_the_kerb_it_is_drawn_against(site, wide_site_models):
    """The promise and the drawing have to be the same number.

    apply_osm_parking decided an 8 ft stall fitted from the leg's NOMINAL half-width - a
    figure field-measured at the intersection - and the paint was then clipped to wherever
    the traced kerb actually ran. On broad_st_east that meant committing to 8 ft stalls off
    a 26.0 ft half-width and drawing them 4.6 ft deep against a kerb 16.0 ft out, which is
    what "the parking spaces look unusable" turned out to be. Nothing checked that the
    promise survived contact with the survey.

    So: wherever a stall is marked, the measured room must hold the lane AND the stall.
    """
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT, apply_osm_parking

    model = wide_site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)

    for leg_name, leg in model.legs.items():
        for side in ("left", "right"):
            parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
            if parking is None:
                continue
            promised_ft = TARGET_LANE_WIDTH_FT + parking.depth_ft
            measured_ft = narrowest_half_width_ft(leg, side)
            assert measured_ft >= promised_ft - 0.05, (
                f"{site}/{leg_name} {side}: marked an {parking.depth_ft:.0f} ft stall beside "
                f"an {TARGET_LANE_WIDTH_FT:.0f} ft lane, needing {promised_ft:.1f} ft, but the "
                f"traced kerb comes within {measured_ft:.1f} ft of the alignment - the stall is "
                f"drawn {measured_ft - TARGET_LANE_WIDTH_FT:.1f} ft deep where it was promised "
                f"{parking.depth_ft:.0f}")


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
    assert not state.treatments_of(LaneNarrowing)
    assert not state.treatments_of(MarkedParking)
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
    narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg))
    assert narrowing is not None and Side.LEFT in narrowing.sides
    assert state.treatment_for(MarkedParking, LegSide(leg, "left")) is None
    lane_ft = model.legs[leg].curb_to_curb_ft / 2 - narrowing.stripe_width_ft
    assert lane_ft == pytest.approx(TARGET_LANE_WIDTH_FT, abs=0.05)


def test_completing_centerlines_only_fills_real_gaps():
    """A leg with NO centerline gets one. A leg that already has markings is left alone -
    upgrading a dashed line to a no-passing double is a sight-line judgement, not a gap."""
    from shapely.geometry import LineString as _LineString

    from src.geometry.model import Leg
    from src.geometry.treatments import complete_centerlines

    # The legs have to exist, because the treatment's target is checked against the design now.
    # This state used to carry centerline styles for three legs the junction did not have, which
    # DesignState.from_model cannot produce - it seeds the styles FROM the config's legs.
    named = ("unmarked", "dashed", "double")
    state = DesignState(
        legs={name: Leg(name=name, centerline=_LineString([(0, 0), (100, 0)]), curb_to_curb_ft=30.0)
              for name in named},
        corner_fillets={},
        existing_centerline_styles={"unmarked": "none", "dashed": "single_yellow_dashed",
                                     "double": "double_yellow"})
    completed = complete_centerlines(state)
    assert completed.centerline_style("unmarked") == "double_yellow"
    assert completed.centerline_style("dashed") == "single_yellow_dashed"
    assert completed.centerline_style("double") == "double_yellow"


@needs_source_data
def test_the_proposal_adds_the_missing_greenwood_centerline(site_models):
    """Greenwood Ave south of Broad has no centerline paint today - confirmed by street-view
    review - so EXISTING must show none and the proposal must add one."""
    model = site_models["broad_st_greenwood"]
    baseline = DesignState.from_model(model)
    assert baseline.centerline_style("greenwood_ave_south") == "none"

    with contextlib.redirect_stdout(io.StringIO()):
        proposed = run_scenario(load_site_scenarios("broad_st_greenwood").build_demo_scenario,
                                 baseline, model)
    assert proposed.centerline_style("greenwood_ave_south") == "double_yellow"


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

    for narrowing in state.treatments_of(LaneNarrowing):
        leg_name = narrowing.target.leg
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
    from src.render.crosswalks import resolve_crosswalk_style

    assert state.treatments_of(UpgradeCrosswalkMarkings), "the proposal should restyle every leg"
    for leg_name in model.legs:
        assert resolve_crosswalk_style(state, leg_name) == "continental", f"{leg_name} was missed"


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
    # Only the bands of legs that actually CARRY a painted crossing. The rest are the
    # footprint a crossing would occupy if one were ever added, and every leg has one because
    # every leg needs a resolved station for a hypothetical - but reserving room around a
    # crossing that is not painted is what held the north side of E Broad's hatching 37 ft
    # out from a kerb with no corner on it. curbside_paint_ft has always clipped against the
    # marked set only; this check was grading against the full set, so it passed by accident
    # while the anchors were conservative and failed the moment they stopped being. The check
    # against the bars Blender really draws is
    # test_no_rendered_paint_runs_through_a_rendered_crosswalk.
    all_bands = crosswalk_bands_ft(
        state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M, pavement,
        crosswalk_reaches_ft(state, offsets, skews, pavement, marked))
    bands = unary_union([band for name, band in all_bands.items()
                         if name in marked and band is not None and not band.is_empty])

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

    daylight = [p for p in paint if p.kind in (DAYLIGHT_FILL, DAYLIGHT_EDGE_LINE)]
    assert daylight, "no daylighting is marked anywhere at Broad & Greenwood"
    assert any(p.is_fill and p.geometry.area > 50 for p in daylight), \
        "the daylight zones are all slivers - the treatment is not actually being drawn"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_marking_a_scenario_builds_is_declared(site, site_models):
    """Whatever the real proposals paint, src/geometry/markings.py knows about it.

    The rest of what this used to check is now a type. A marking used to be a bare string in
    three hand-synced tables, and this test compared what paint.py built against what each
    table listed - because renaming a kind orphaned it silently. It happened twice in one
    sitting: buffer_taper_* became daylight_taper_* and the taper stopped rendering, and
    daylight_fill was added and never wired up. Both were built, both drawn in the plan view,
    neither in the render, nothing raised.

    Now a PaintKind carries its own channel and its own role, so "built but not rendered in
    3D" and "built but not drawn in 2D" are both refused when this package is imported - see
    markings._kind and markings.require_every_kind, and the tests of those below. What is left
    for a live scenario to prove is that nothing reaches the paint list from outside the
    registry.
    """
    import contextlib as _contextlib

    from src.geometry.markings import KINDS

    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with _contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
            paint, _bands = paint_and_bands(model, state)
        for piece in paint:
            assert KINDS.get(piece.kind.name) is piece.kind, (
                f"{site}/{name}: {piece.kind} is not the marking of that name declared in "
                f"src/geometry/markings.py, so nothing routed it to either renderer")


def test_a_marking_with_no_channel_is_refused():
    """The 3D half of what test_every_kind_of_paint_reaches_both_renders used to catch.

    A marking with no channel is drawn in the plan view and absent from the render. That is now
    a ValueError at declaration rather than something a test has to notice afterwards.
    """
    from src.geometry.markings import Role, _kind

    with pytest.raises(ValueError, match="no channel"):
        _kind("invented_line", Role.LINE)


def test_a_marking_cannot_be_routed_to_a_channel_that_draws_something_else():
    """A fill in a channel of lines would be drawn by the wrong builder at the far end -
    add_paint_polyline over a polygon's ring instead of the hatch strokes inside it."""
    from src.geometry.markings import LANE_NARROWING_EDGE_LINES, Role, _kind

    with pytest.raises(ValueError, match="carries lines"):
        _kind("invented_fill", Role.FILL, LANE_NARROWING_EDGE_LINES)


def test_a_marking_the_plan_view_has_no_style_for_is_refused():
    """The 2D half. A marking absent from PAINT_STYLE simply is not drawn, and the plan view
    is supposed to show what the render shows."""
    from src.geometry.markings import BUFFER_FILL, require_every_kind

    with pytest.raises(ValueError, match="is missing"):
        require_every_kind({BUFFER_FILL: dict(color="gold")}, "a table with one entry")


def test_every_declared_marking_is_something_paint_can_build():
    """The other direction: a declaration nothing emits renders nothing and hides a typo.

    Read off the source rather than off a scenario, because several markings are real but
    conditional - a taper is only built where a zone tapers, and no current proposal has one at
    these four junctions.

    Both modules, because markings are moving out of paint.py's per-treatment blocks and onto the
    treatments themselves (Treatment.paint). This test caught that move the first time a marking
    left: CORNER_HATCH_FILL is emitted by CornerHatching now, and searching paint.py alone called
    it stale.
    """
    import inspect

    from src.geometry import paint as paint_module
    from src.geometry import treatments as treatments_module
    from src.geometry.markings import KINDS

    source = inspect.getsource(paint_module) + inspect.getsource(treatments_module)
    for name, kind in KINDS.items():
        constant = name.upper()
        assert constant in source, (
            f"src/geometry/markings.py declares {name!r} but nothing in src/geometry/paint.py or "
            f"src/geometry/treatments.py emits {constant} - a marking nothing builds is a stale "
            f"declaration")


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
            # is_fill AND not a bollard: a bollard's geometry is a degenerate 1e-6 ft square
            # standing in for a point (paint.py:_dot), so it is a Polygon by type but is not a
            # zone that can run up to a crossing and be cut by one. check_markings_do_not_collide
            # excludes it for the same reason.
            near = [p for p in paint
                    if p.leg == leg_name and p.covers_area]
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
    """The paint and the crossing bands for one state, exactly as the renderers build them.

    Both come off SceneGeometry, so "exactly" is now structural rather than a claim - see
    resolved_scene.
    """
    scene = resolved_scene(model, state)
    return scene.build_paint(scene_props(model, state, scene)), scene.crosswalk_bands


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


# --------------------------------------------------------------------------
# The 2D and the 3D have to agree about where a marking IS
# --------------------------------------------------------------------------

def crosswalk_bars_as_blender_draws_them(leg_json: dict, depth_m: float):
    """The crosswalk bar rectangles scripts/blender/blender_crosswalks.py will build.

    Replicated from the geometry JSON rather than driven through Blender, because the test
    suite cannot run Blender - and the thing being checked is precisely whether the numbers
    in that file put the bars where the 2D said they were.
    """
    import math

    from shapely.geometry import Polygon

    stripe_m = 0.5              # blender_crosswalks: CONTINENTAL_BAR_WIDTH
    axis, centre_m = leg_json.get("crosswalk_axis"), leg_json.get("crosswalk_centre_m")
    if axis is None or centre_m is None:
        pytest.fail("the geometry JSON carries no resolved crosswalk frame, so Blender falls "
                    "back to the near->far chord - see src/render/export.py:_marking_frame_m")
    u = np.asarray(axis, dtype=float)
    n = np.asarray([-u[1], u[0]])
    centre = np.asarray(centre_m, dtype=float)

    skew = math.radians(leg_json.get("crosswalk_skew_deg", 0.0))
    cos_s, sin_s = math.cos(skew), math.sin(skew)
    u_s = np.asarray([u[0] * cos_s - u[1] * sin_s, u[0] * sin_s + u[1] * cos_s])
    n_s = np.asarray([n[0] * cos_s - n[1] * sin_s, n[0] * sin_s + n[1] * cos_s])

    left_m, right_m = leg_json["crosswalk_reach_left_m"], leg_json["crosswalk_reach_right_m"]
    centre = centre + n_s * ((left_m - right_m) / 2)
    span_m = left_m + right_m
    count = leg_json["crosswalk_bar_count"]
    span = max(span_m - stripe_m, 0.0)
    pitch = span / (count - 1) if count > 1 else 0.0
    bars = []
    for i in range(count):
        c = centre + n_s * (-span / 2 + i * pitch)
        bars.append(Polygon([c + u_s * (depth_m / 2) + n_s * (stripe_m / 2),
                             c + u_s * (depth_m / 2) - n_s * (stripe_m / 2),
                             c - u_s * (depth_m / 2) - n_s * (stripe_m / 2),
                             c - u_s * (depth_m / 2) + n_s * (stripe_m / 2)]))
    return bars


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_rendered_paint_runs_through_a_rendered_crosswalk(site, site_models, tmp_path):
    """Checked on the EXPORTED numbers, which is the only place the two views can drift.

    curbside_paint_ft clears its markings of the crosswalk bands the plan view draws, so the
    2D is self-consistent by construction and a 2D check cannot catch this. blender_scene.py
    was rebuilding the crossing's frame from the leg's near->far CHORD instead of reading the
    one src/render/crosswalks.py resolved - identical while a centerline is straight, 4.54 deg
    out on broad_st_east, which kinks 4.5 deg 43.1 ft from the junction where NJDOT rounds the
    corner. That rotated the bars off the cleared footprint and drove them through 11.5 ft of
    lane-edge line and 1.1 ft of hatching at the NE corner: correct in the plan view, wrong in
    the render, no check anywhere between them.
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from src.render.export import PAINT_KIND_LISTS, export_scenario

    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run_scenario(builder, DesignState.from_model(model), model)
            path = export_scenario(model, state, name, tmp_path / f"{site}_{name}.json",
                                   buildings=[], crossings=fetch_crossings(model.center_wgs84,
                                                                           radius_m=130))
        data = json.loads(Path(path).read_text())

        marked = set(data.get("existing_marked_crosswalks", []))
        bars = [bar for leg in data["legs"] if leg["name"] in marked
                for bar in crosswalk_bars_as_blender_draws_them(leg, data["crosswalk_depth_m"])]
        if not bars:
            continue
        crossings = unary_union(bars)

        worst = []
        for key in PAINT_KIND_LISTS:
            for line in data.get(key, []):
                hit = LineString([(p[0], p[1]) for p in line]).intersection(crossings)
                if not hit.is_empty and hit.length / FT_TO_M > 0.1:
                    worst.append(f"{hit.length / FT_TO_M:.2f} ft of {key}")
        assert not worst, (f"{site}/{name}: rendered paint runs through the rendered "
                           f"crosswalk bars:\n  " + "\n  ".join(sorted(worst, reverse=True)))


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_drawn_crosswalk_is_parallel_to_the_surveyed_one(site, site_models):
    """The whole point of carrying the skew is that the marking lines up with the way OSM
    traced. It has to be measured in the frame it gets applied in.

    _crossing_skew_deg took "square" from the leg's whole-length chord while crosswalk_axes
    applies it against the local segment at the crossing's own station. Identical on a
    straight centerline; 4.54 deg apart on broad_st_east, whose alignment kinks 4.5 deg where
    NJDOT rounds the corner 43.1 ft out - so on the one leg where the skew mattered most it
    cancelled out exactly as much as it recovered.

    EVERY matched crossing is checked, including louellen_st_west's -44 deg one. That used to
    be gated off as "not a depiction of the paint", which was the squareness assumption
    excusing itself from the one junction that falsifies it - a 48 deg Y whose kerb ramps are
    not opposite each other. If a surveyed way is good enough to place the crosswalk it is
    good enough to orient it, and this test is what holds those two together.
    """
    from src.render.crosswalks import (_match_crossings_to_legs, crosswalk_axes,
                                       resolve_crosswalk_offsets, resolve_crosswalk_skews)

    model = site_models[site]
    state = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        matched = _match_crossings_to_legs(state.legs, crossings)
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)

    assert matched, f"{site}: no OSM crossing matched any leg at all"
    assert set(skews) == set(matched), (
        f"{site}: {sorted(set(matched) - set(skews))} matched a surveyed crossing but carried "
        f"no skew - a surveyed orientation is being dropped somewhere")

    checked = 0
    for leg_name, (_along, _style, _skew, line, _tags) in sorted(matched.items()):
        _c, _u, across, _cos = crosswalk_axes(state.legs[leg_name], offsets[leg_name][0],
                                               skews[leg_name])
        surveyed = np.asarray(line.coords[-1], dtype=float) - np.asarray(line.coords[0], dtype=float)
        surveyed /= np.linalg.norm(surveyed)
        cosine = abs(float(np.clip(np.dot(surveyed, np.asarray(across, dtype=float)), -1, 1)))
        off_deg = np.degrees(np.arccos(cosine))
        assert off_deg < 0.01, (
            f"{site}/{leg_name}: the crosswalk is drawn {off_deg:.2f} deg off the OSM way it "
            f"took its skew from")
        checked += 1
    assert checked == len(matched), (
        f"{site}: only {checked} of {len(matched)} matched crossings were checked")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_centreline_runs_up_to_the_stop_bar(site, site_models):
    """A double yellow stops at the bar drivers stop on - it does not stop short of it.

    centerline_start_ft holds the paint back behind whichever is further out, the bar or the
    crosswalk. On a leg with no marked crossing that second term is the geometric estimate,
    which at e_broad_st_east is this junction's modelled 70.1 ft corner return - a number the
    phase output already reports as contradicted by the surveyed stop bar 17 ft inside it. The
    yellow stopped 23.8 ft short of the bar to clear a crossing that is not painted.
    """
    from src.render.crosswalks import (centerline_start_ft, resolve_crosswalk_offsets,
                                       resolve_stop_bar_offsets)
    from src.sources.osm_context import fetch_stop_lines

    model = site_models[site]
    if not model.config.get("signals"):
        pytest.skip(f"{site} is unsignalized - no surveyed stop bars")
    state = DesignState.from_model(model)
    marked = marked_crosswalks(model)
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        stop_offsets = resolve_stop_bar_offsets(
            state, offsets, fetch_stop_lines(model.center_wgs84, radius_m=130))

    checked = 0
    for leg_name, bar_ft in sorted(stop_offsets.items()):
        start_ft = centerline_start_ft(offsets[leg_name][0], bar_ft, leg_name in marked)
        assert start_ft <= bar_ft + 0.01, (
            f"{site}/{leg_name}: centreline paint starts {start_ft - bar_ft:.1f} ft beyond its "
            f"own stop bar, leaving a gap where the road has no centreline at all")
        checked += 1
    assert checked, f"{site}: no leg had a surveyed stop bar, so this test checked nothing"


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_leg_is_held_back_by_a_through_street_join(site, site_models):
    """On the real junctions: dropping the through-street joins changes no leg's clearance.

    leg_clearance_ft is what decides how close to the junction a crossing, a hatched zone or a
    stall may start, so a join that is not a corner return must not appear in it. The one pair
    this fires on is e_broad_st_east/e_broad_st_west at 179.9 deg, where it was worth 38 ft of
    clearance on e_broad_st_east - enough to hold that leg's hatching 22 ft short of its own
    surveyed stop bar.
    """
    from src.geometry.model import leg_clearance_ft

    model = site_models[site]
    state = DesignState.from_model(model)
    real_corners = {key: pieces for key, pieces in state.corner_fillets.items()
                    if not pieces.get("through_street")}
    for leg_name in state.legs:
        with_joins = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        without = leg_clearance_ft(leg_name, state.legs, real_corners)
        assert with_joins == pytest.approx(without), (
            f"{site}/{leg_name}: a through-street join adds "
            f"{with_joins - without:.1f} ft of corner clearance it has no business adding")


@needs_source_data
def test_e_broad_east_hatching_reaches_its_stop_bar():
    """The leg the through-street join was holding back, by its own numbers.

    Its right kerb is traced from 23 ft out, so there IS curb to build a strip against inside
    the stop bar at 52.9 ft - and the hatching now starts at 37 ft, 16 ft past the bar. Its
    LEFT kerb is only traced from 59 ft, so that side still starts at 59: a gap in the OSM
    tracing, which the phase output reports by name, not something geometry can recover.
    """
    from src.geometry.model import curb_station_span, station_offset_many
    from src.render.crosswalks import resolve_crosswalk_offsets, resolve_stop_bar_offsets
    from src.sources.osm_context import fetch_stop_lines

    model, state, paint = demo_paint("ebroad_princeton")
    leg_name = "e_broad_st_east"
    leg = state.legs[leg_name]
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        bars = resolve_stop_bar_offsets(state, resolve_crosswalk_offsets(state, crossings),
                                        fetch_stop_lines(model.center_wgs84, radius_m=130))
    bar_ft = bars[leg_name]

    for side in ("left", "right"):
        fills = [p for p in paint if p.leg == leg_name and p.side == side and p.is_fill]
        assert fills, f"{leg_name} {side} has no hatched zone at all"
        start_ft = min(station_offset_many(leg.centerline,
                                           np.asarray(f.geometry.exterior.coords, dtype=float))[0].min()
                       for f in fills)
        traced_from_ft = curb_station_span(leg, side)[0]
        if traced_from_ft > bar_ft:
            assert start_ft == pytest.approx(traced_from_ft, abs=1.0), (
                f"{side}: kerb traced only from {traced_from_ft:.0f} ft, so the zone should "
                f"begin there, not at {start_ft:.0f}")
            continue
        assert start_ft <= bar_ft, (
            f"{side}: hatching starts {start_ft - bar_ft:.1f} ft short of the stop bar at "
            f"{bar_ft:.0f} ft, leaving bare full-width asphalt where the lane most needs "
            f"narrowing")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_stop_bar_reaches_the_centreline_and_the_lane_edge(site, site_models):
    """It spans the approach lane: centerline to lane edge, nothing standing off either.

    stop_bar_band_geometry_ft subtracted the kerb clearance from the SPAN while centring the
    bar on the middle of the entering half, so half the clearance landed at the centerline
    end - leaving the bar 0.7-0.8 ft off the centerline with nothing on the far side of the
    gap. And where a treatment had narrowed the lane, the "kerb" clearance was being applied
    against a painted edge line 1.6 ft away, so the bar stopped short at that end too.
    MUTCD's stop line runs across the approach lanes; both ends meet what they run to.
    """
    from src.geometry.model import station_offset_many
    import math

    from src.render.crosswalks import (STOP_BAR_PLAN_DEPTH_FT, entering_lane_width_ft,
                                       resolve_crosswalk_offsets, resolve_crosswalk_skews,
                                       resolve_stop_bar_offsets, stop_bar_bands_ft)
    from src.sources.osm_context import fetch_stop_lines

    model = site_models[site]
    if not model.config.get("signals"):
        pytest.skip(f"{site} is unsignalized - no surveyed stop bars")
    _m, state, _paint = demo_paint(site)
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        offsets = resolve_crosswalk_offsets(state, crossings)
        bars_at = resolve_stop_bar_offsets(
            state, offsets, fetch_stop_lines(model.center_wgs84, radius_m=130))
        skews = resolve_crosswalk_skews(state, crossings)
        bands = stop_bar_bands_ft(state, bars_at, skews)

    assert bands, f"{site} is signalized but drew no stop bars"
    for leg_name, band in sorted(bands.items()):
        leg = state.legs[leg_name]
        _st, off = station_offset_many(leg.centerline,
                                       np.asarray(band.exterior.coords, dtype=float))
        entering_ft = entering_lane_width_ft(state, leg_name)
        edge_ft = entering_ft if entering_ft is not None else leg.curb_to_curb_ft / 2
        inner = min(abs(off.min()), abs(off.max()))
        outer = max(abs(off.min()), abs(off.max()))
        # A SKEWED bar is a rotated rectangle, so its two centerline-side corners straddle the
        # centerline by half the depth's rotated projection - one inboard, one outboard, and
        # no placement puts both on it. That is the bar meeting the centerline correctly, not
        # a gap. Louellen's -44 deg crossing leaves 1.5 * sin(44) / 2 = 0.52 ft. Square bars
        # get the flat 0.25 ft this always used, because sin(0) is 0.
        skew_slack = STOP_BAR_PLAN_DEPTH_FT * abs(math.sin(math.radians(skews.get(leg_name, 0.0)))) / 2
        assert inner < 0.25 + skew_slack, (
            f"{site}/{leg_name}: the stop bar stands {inner:.2f} ft off the road centerline, "
            f"which is a gap with nothing on the other side of it")
        # Where the lane was narrowed the bar meets its own edge line; where the far end is
        # the kerb it is held back deliberately, so allow the clearance there.
        allowed = 0.25 if entering_ft is not None else STOP_BAR_CURB_CLEARANCE_M / FT_TO_M + 0.25
        allowed += skew_slack
        assert edge_ft - outer < allowed, (
            f"{site}/{leg_name}: the stop bar stops {edge_ft - outer:.2f} ft short of the "
            f"{'lane edge line' if entering_ft is not None else 'kerb'} at {edge_ft:.1f} ft")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_plan_view_draws_without_raising(site, site_models):
    """Actually draw it, for existing conditions and every scenario.

    Nothing in this suite drew the plan view before, and that is how a crash reached the
    user: the through-street join carried "radius_ft": None, plot_design_state labels a
    corner's radius wherever that key is PRESENT, and `f"{None:.0f}"` is a TypeError. Every
    other check passed, the 3D render was verified, and the 2D build died on the one site
    with a through-street pair.

    A smoke test, deliberately: it asserts no exception and that something was drawn, not what
    it looks like. The geometry itself is checked by the invariants; what was missing was
    anyone running the drawing code at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.render.plan_view import legend_handles, plot_design_state

    model = site_models[site]
    states = {"existing": DesignState.from_model(model)}
    for name, builder in sorted(scenario_builders(site).items()):
        with contextlib.redirect_stdout(io.StringIO()):
            states[name] = run_scenario(builder, DesignState.from_model(model), model)

    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        for label, state in states.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            try:
                plot_design_state(ax, model, state, f"{site} {label}", crossings=crossings)
                assert ax.collections or ax.lines, f"{site}/{label}: nothing was drawn"
            finally:
                plt.close(fig)
    assert legend_handles(), "the legend is empty"


def test_every_marking_the_plan_view_draws_is_in_its_legend():
    """A marking drawn with nothing to say what it is, is one the reader has to guess at.

    PAINT_STYLE is already guarded: require_every_kind raises on import for a declared marking
    with no style. The LEGEND was not, and it is the same class of omission one step further
    along - adding the green bike lane surface drew it in every plan view of a bike lane
    proposal and explained it nowhere, which is how this test came to exist.

    Checked by APPEARANCE rather than per marking, because the legend groups deliberately and
    should: one "Lane narrowing / corner hatching" swatch covers three gold hatched kinds, and
    splitting it into three identical rows would be worse for the reader. So the rule is that
    every way a marking can LOOK has an entry, not that every marking has its own.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    from src.render.plan_view import PAINT_STYLE, legend_handles

    handles = legend_handles()
    areas = {(to_rgba(h.get_facecolor())[:3], h.get_hatch())
             for h in handles if isinstance(h, Patch)}
    # A line's colour may be explained by a swatch's OUTLINE rather than by a line of its own,
    # and for two of them it is: the orangered daylight edge and the crossing rim are the
    # outline of the daylighting patch, which is what PAINT_FILL_EDGE pairs them with. That is
    # the legend reading correctly, not a gap, so an edgecolor counts.
    lines = ({to_rgba(h.get_color())[:3] for h in handles if not isinstance(h, Patch)}
             | {to_rgba(h.get_edgecolor())[:3] for h in handles if isinstance(h, Patch)})
    for kind, style in sorted(PAINT_STYLE.items(), key=lambda kv: str(kv[0])):
        rgb, hatch = to_rgba(style["color"])[:3], style.get("hatch")
        if kind.covers_area:
            assert (rgb, hatch) in areas, (
                f"{kind} is drawn as a {style['color']} area"
                + (f" hatched {hatch!r}" if hatch else " with no hatch")
                + " and no legend swatch looks like that, so the plan view draws it with "
                  "nothing to say what it is. Add a Patch to legend_handles().")
        else:
            assert rgb in lines, (
                f"{kind} is drawn as a {style['color']} line and no legend entry uses that "
                f"colour. Add a Line2D to legend_handles().")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_each_leg_reads_its_tags_off_a_carriageway(site, site_models):
    """A leg's operational tags have to come from the street, not from something parked on it.

    Geometry alone cannot tell them apart. East of Princeton Ave, OSM has a
    `highway=service, service=parking_aisle` way (772378208) running 0.5 ft from East Broad
    Street's centerline at 0.2 deg to it - closer on neither count, and it won the
    nearest-way tie. So e_broad_st_east read its restrictions off a parking aisle, which has
    none, and East Broad Street's own `parking:both:restriction=no_stopping` (way 1546878992)
    was never seen. The kerb still came out hatched, for having 7.5 ft spare rather than for
    being no-stopping, and the plan view reported it as untagged.
    """
    from src.geometry.intersection import ROAD_MATCH_HIGHWAY_CLASSES

    model = site_models[site]
    for leg_name in sorted(model.legs):
        tags = model.leg_osm_tags.get(leg_name)
        if tags is None:
            continue        # no match at all is reported and defaults are used - see the matcher
        assert tags.get("highway") in ROAD_MATCH_HIGHWAY_CLASSES, (
            f"{site}/{leg_name} took its tags from a highway={tags.get('highway')!r} "
            f"(service={tags.get('service')!r}, name={tags.get('name')!r}) - not a carriageway")
        assert "service" not in tags, (
            f"{site}/{leg_name} matched a service way: {tags.get('service')!r}")


@needs_source_data
def test_east_broad_reads_the_no_stopping_the_surveyor_tagged():
    """The specific restriction the parking aisle was masking, on the leg it was masked on."""
    from src.geometry.intersection import parking_restriction_by_side

    model, _state, _paint = demo_paint("ebroad_princeton")
    tags = model.leg_osm_tags["e_broad_st_east"]
    assert tags.get("name") == "East Broad Street", f"matched {tags.get('name')!r} instead"
    sides = parking_restriction_by_side(tags, model.leg_osm_aligned["e_broad_st_east"])
    assert sides["left"] == "no_stopping" and sides["right"] == "no_stopping", (
        f"East Broad east is tagged no_stopping on both sides in OSM; this read {sides}")


# --------------------------------------------------------------------------
# Data accounting: fetched source data must be USED, or accounted for
# --------------------------------------------------------------------------

@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_leg_side_is_built_from_traced_kerb(site, site_models):
    """All 24 leg sides across the four junctions come from OSM tracing, not an offset.

    The strongest single statement of "we are using what we have", and the one that three
    separate discard bugs each violated: the width fit judging vertices against a width it
    was about to measure from them, the parallelism gap, and a leg claiming a vertex from
    behind its own junction node. Every one of them showed up here first as a side quietly
    falling back to a centerline offset.
    """
    model = site_models[site]
    fallen_back = [f"{name} {side}" for name, leg in sorted(model.legs.items())
                   for side in ("left", "right") if side not in leg.traced_sides]
    assert not fallen_back, (
        f"{site}: these sides are drawn as centerline offsets, not from the traced kerb: "
        f"{fallen_back}")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_no_traced_kerb_vertex_is_silently_unclaimed(site, site_models):
    """Every vertex of every kerb way this junction accepts must be claimable by some leg.

    An unclaimable vertex is either a real exclusion - a median, a driveway apron, a
    neighbouring street - or ground truth going in the bin. The tolerated count is stated per
    site so that a NEW one fails here rather than disappearing into a total. W Broad &
    Louellen's five are its two stub ways behind the junction node running across it, not
    along any leg.
    """
    import numpy as np

    from src.geometry.intersection import kerb_lines_with_tags_ft
    from src.geometry.model import (CURB_POINT_BEHIND_TOLERANCE_FT, CURB_POINT_CORNER_ZONE_FT,
                                    CURB_POINT_MAX_SKEW_DEG, CURB_POINT_MAX_WIDTH_RATIO,
                                    CURB_POINT_MIN_WIDTH_RATIO, _line_direction,
                                    _vertex_tangents, station_offset_many)

    TOLERATED = {"broad_st_greenwood": 0, "ebroad_princeton": 0,
                 "columbia_princeton": 0, "wbroad_louellen": 5}

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        ways = [line for line, *_ in kerb_lines_with_tags_ft(model.center_wgs84,
                                                                 model.center_ft)]
    points = np.concatenate([np.asarray(w.coords, dtype=float) for w in ways])
    tangents = np.concatenate([_vertex_tangents(w) for w in ways])
    min_cosine = np.cos(np.radians(CURB_POINT_MAX_SKEW_DEG))

    unclaimable = 0
    for i, point in enumerate(points):
        for leg in model.legs.values():
            stations, offsets = station_offset_many(leg.centerline, point[None, :])
            ratio = abs(offsets[0]) / (leg.curb_to_curb_ft / 2)
            skewed = abs(float(tangents[i] @ _line_direction(leg.centerline))) < min_cosine
            if (stations[0] >= -CURB_POINT_BEHIND_TOLERANCE_FT
                    and CURB_POINT_MIN_WIDTH_RATIO <= ratio <= CURB_POINT_MAX_WIDTH_RATIO
                    and not (skewed and stations[0] > CURB_POINT_CORNER_ZONE_FT)):
                break
        else:
            unclaimable += 1
    assert unclaimable <= TOLERATED[site], (
        f"{site}: {unclaimable} traced kerb vertices can be claimed by no leg "
        f"({TOLERATED[site]} known). A new one means kerb the surveyor drew is being discarded")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_matched_crossing_and_stop_bar_is_used(site, site_models):
    """A crossing or stop bar that matched a leg has to reach the drawing.

    Not "was fetched" - the fetch radius deliberately pulls in neighbouring junctions. Once
    the matcher has credited one to a leg, though, dropping it is a discard.
    """
    from src.render.crosswalks import (_match_crossings_to_legs, resolve_crosswalk_offsets,
                                       resolve_stop_bar_offsets)
    from src.sources.osm_context import fetch_stop_lines

    model = site_models[site]
    state = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        matched = _match_crossings_to_legs(state.legs, crossings)
        offsets = resolve_crosswalk_offsets(state, crossings)
        bars = resolve_stop_bar_offsets(
            state, offsets, fetch_stop_lines(model.center_wgs84, radius_m=130))

    for leg_name in matched:
        assert offsets[leg_name][1].startswith("osm_survey"), (
            f"{site}/{leg_name}: a matched OSM crossing was not used for the crossing's "
            f"position - source says {offsets[leg_name][1]!r}")
    # A stop bar the matcher credited to a leg must be drawn on it, signalized or not.
    for leg_name, station_ft in bars.items():
        assert station_ft > 0, f"{site}/{leg_name}: stop bar resolved to {station_ft}"


@needs_source_data
def test_a_leg_can_be_carried_further_than_its_neighbours(site_models):
    """legs.<name>.working_length_ft overrides the site default for that leg ALONE.

    Broad & Greenwood needs it: Schedule I of the borough parking code bans parking for
    100 ft east of Greenwood's curb, and Schedule III's 2 hr zone starts exactly where that
    ends, so at the site default of 130 ft the render shows the prohibition and 16 ft of the
    parking - under one 22 ft stall, reading as "remove all the parking". East Broad is the
    only leg here traced far enough to carry honestly (173.8 ft left, 179.1 ft right), so it
    goes to 170 and the rest stay at 130.

    Fails against a single site-wide working length in either direction: shared 130 makes
    every leg 130, shared 170 lengthens the three legs whose kerbs run out well before it.
    """
    legs = site_models["broad_st_greenwood"].legs
    assert legs["broad_st_east"].centerline.length == pytest.approx(170.0, abs=0.5)
    for name in ("broad_st_west", "greenwood_ave_north", "greenwood_ave_south"):
        assert legs[name].centerline.length == pytest.approx(130.0, abs=0.5), (
            f"{name} was carried to its neighbour's length - the override is not per-leg")
    # ...and the lengthened leg is still drawn from tracing for its whole run, which is the
    # only reason 170 is allowed. An extrapolated curb here would defeat the point.
    assert legs["broad_st_east"].traced_sides == {"left", "right"}


@needs_source_data
def test_how_far_a_leg_is_drawn_does_not_change_how_wide_it_is_measured(site_models):
    """A presentation choice may not move a measurement.

    The cross-section window used to run to the far end of the traced curb line, and a curb
    line is drawn to the leg's working length - so lengthening a leg to show more of it
    silently re-measured its width. Carrying broad_st_east from 130 to 170 ft moved it
    52.0 -> 49.9 ft, because East Broad narrows leaving the junction and the extra 40 ft of
    narrower street pulled the median down. Every dimension in the proposal is an offset
    from that width.

    Fails without TRACED_SECTION_END_FT: the two widths below come out 2.1 ft apart.
    """
    import src.geometry.intersection as I

    measured = {}
    for cap in (I.TRACED_SECTION_END_FT, 1e9):
        saved = I.TRACED_SECTION_END_FT
        try:
            I.TRACED_SECTION_END_FT = cap
            with contextlib.redirect_stdout(io.StringIO()):
                model = I.load_intersection_model(site="broad_st_greenwood")
            measured[cap] = model.legs["broad_st_east"].curb_to_curb_ft
        finally:
            I.TRACED_SECTION_END_FT = saved

    capped, uncapped = measured[I.TRACED_SECTION_END_FT], measured[1e9]
    assert capped == pytest.approx(52.0, abs=0.2), (
        f"broad_st_east measures {capped:.1f} ft over the fixed approach window; the value "
        f"every other 130 ft leg is measured against is 52.0")
    assert uncapped < capped - 1.0, (
        "this test is not testing anything: with the window free to follow the 170 ft curb "
        "line the width should drop by ~2 ft, and it did not")


@needs_source_data
def test_the_crosswalk_estimate_reproduces_the_surveyed_crossings(site_models):
    """The estimator has to predict the crossings we DIDN'T give it.

    Eleven of the fourteen legs across the four sites have an OSM-surveyed crossing. Those are
    the only ground truth there is for where a crosswalk belongs, so a rule for the other
    three is worth exactly what it scores against them. Two earlier candidates failed here and
    were dropped: the fillet tangent point (leg_clearance_ft) scattered -31.5 to +41.7 ft, and
    projecting the cross street's kerb lines onto the leg centerline scattered -38.0 to -2.3
    and returned 119.7 ft for w_broad_st_northeast.

    Held to the spread that justified the constant. A change that widens it is a worse rule
    however reasonable it looks, and CROSSWALK_SETBACK_FT stops being a measurement.
    """
    from src.geometry.model import crosswalk_estimate_ft
    from src.render.crosswalks import resolve_crosswalk_offsets
    from src.sources.osm_context import fetch_crossings

    errors = {}
    for site, model in sorted(site_models.items()):
        state = DesignState.from_model(model)
        with contextlib.redirect_stdout(io.StringIO()):
            offsets = resolve_crosswalk_offsets(
                state, fetch_crossings(model.center_wgs84, radius_m=130))
        for leg_name, (surveyed_ft, source) in offsets.items():
            if source != "osm_survey":
                continue
            errors[f"{site}/{leg_name}"] = (
                crosswalk_estimate_ft(leg_name, model.legs) - surveyed_ft)

    assert len(errors) == 11, f"expected 11 surveyed crossings to score against, got {len(errors)}"
    spread = max(errors.values()) - min(errors.values())
    worst = max(errors, key=lambda k: abs(errors[k]))
    assert spread <= 10.0, (
        f"the estimate's error spread across the surveyed crossings is {spread:.1f} ft "
        f"(worst {worst} {errors[worst]:+.1f}) - it was 8.8 ft when CROSSWALK_SETBACK_FT was "
        f"fitted. {errors}")
    assert abs(np.mean(list(errors.values()))) <= 1.0, (
        f"the estimate is biased {np.mean(list(errors.values())):+.1f} ft against the surveyed "
        f"crossings - refit CROSSWALK_SETBACK_FT")


@needs_source_data
def test_no_crosswalk_is_estimated_outside_the_junction(site_models):
    """An estimated crossing has to land where a real one plausibly could.

    The rule this replaced put w_broad_st_southwest's crossing 67.8 ft from the node - past
    the cross street's far kerb, out in the middle of the block - and w_broad_st_northeast's
    at 11.5 ft, inside a corner return still 25.4 ft off the centerline against a 17.6 ft
    half-width. Both at the same junction, from the same rule, in opposite directions. The
    bound is the surveyed range (19.5-41.7 ft) with a little room either side.
    """
    from src.render.crosswalks import resolve_crosswalk_offsets
    from src.sources.osm_context import fetch_crossings

    for site, model in sorted(site_models.items()):
        state = DesignState.from_model(model)
        with contextlib.redirect_stdout(io.StringIO()):
            offsets = resolve_crosswalk_offsets(
                state, fetch_crossings(model.center_wgs84, radius_m=130))
        for leg_name, (offset_ft, source) in sorted(offsets.items()):
            if source == "osm_survey":
                continue
            assert 15.0 <= offset_ft <= 50.0, (
                f"{site}/{leg_name}: estimated crosswalk at {offset_ft:.1f} ft, outside the "
                f"15-50 ft band every surveyed crossing at these four junctions falls in")


# --------------------------------------------------------------------------
# The proposals as shipped
#
# The primitives are tested in tests/test_curb_extensions.py. These test the scenarios a
# reviewer would actually be shown, which is a different thing: a proposal is a set of claims,
# and a claim that stops being true because a treatment was reordered or a leg re-measured is
# exactly the failure that reaches a Borough council rather than a test run.
# --------------------------------------------------------------------------

@needs_source_data
@pytest.mark.parametrize("site,treated,untreated", [
    ("broad_st_greenwood", ("broad_st_east", "broad_st_west"),
     ("greenwood_ave_north", "greenwood_ave_south")),
    ("ebroad_princeton", ("e_broad_st_east", "e_broad_st_west"), ("princeton_ave_south",)),
])
def test_the_bike_lane_proposal_treats_only_the_legs_wide_enough(site_models, site, treated,
                                                                 untreated):
    """Which legs can take a bike lane is the finding, so it is the thing pinned.

    Greenwood Ave and Princeton Ave would be left 1.0-1.7 ft of lane once the 11 ft travel lane
    and the 2 ft buffer are taken, which is under the 4 ft floor by any reading. A narrower stripe
    would read as a bike lane in the render while failing the standard it is meant to meet, which
    is the sort of thing that gets waved through because the picture looks plausible.

    The width asserted is the FLOOR, not the design width, since the rule changed: a kerb that
    cannot hold the full 5 ft narrows the lane rather than giving up the buffer, so E Broad's east
    kerb now carries a 4.49 ft protected lane. What must never appear is a lane under 4 ft.
    """
    from src.geometry.treatments import MIN_BIKE_LANE_FT, AddBikeLane

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios(site).build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)

    for leg_name in treated:
        for side in ("left", "right"):
            treatment = state.treatment_for(AddBikeLane, LegSide(leg_name, side))
            assert treatment is not None, f"{leg_name} {side} has the width but got no lane"
            lane = treatment.lane
            assert lane.width_ft >= MIN_BIKE_LANE_FT
            assert lane.total_ft <= narrowest_half_width_ft(state.legs[leg_name], side) + 0.05, (
                f"{leg_name} {side}'s section does not fit where the leg is narrowest")
    for leg_name in untreated:
        assert not [t for t in state.treatments_of(AddBikeLane) if t.target.leg == leg_name], (
            f"{leg_name} was given a bike lane it has no room for")


# --------------------------------------------------------------------------
# A restriction that covers PART of a leg
#
# OSM records a fact that changes part way along a street by splitting the way. That is how
# "no parking for the first 100 ft from the junction" is expressed, and reading one way per leg
# discarded it - silently, because a pipeline that read one way and found no restriction is
# indistinguishable from one that read the restriction and threw it away.
# --------------------------------------------------------------------------

@needs_source_data
def test_a_split_leg_keeps_every_way_that_covers_it(site_models):
    """East Broad at Greenwood is two OSM ways, and both have to be read.

    Way 1547092834 carries parking:both:restriction=no_parking over the first 79.5 ft; way
    11647647 carries restriction=none for the rest. The matcher used to keep whichever way was
    nearest the leg's MIDPOINT - station 85, past the split - so the restricted way lost by
    1.9 ft against 5.8 and the render marked parking on a kerb a mapper had just tagged as
    having none.
    """
    model = site_models["broad_st_greenwood"]
    spans = model.leg_road_spans["broad_st_east"]
    assert len(spans) >= 2, f"only {len(spans)} way(s) matched a leg OSM has split"

    covered = sorted((s.start_ft, s.end_ft) for s in spans)
    assert covered[0][0] == pytest.approx(0.0, abs=1.0), "no way covers the junction end"
    assert covered[-1][1] >= model.legs["broad_st_east"].centerline.length - 1.0, (
        "the spans stop short of the far end of the leg")
    # Contiguous: OSM splits a way at a node, so one span ends where the next begins.
    for (_lo, hi), (next_lo, _next_hi) in zip(covered, covered[1:]):
        assert next_lo == pytest.approx(hi, abs=1.0), f"gap in coverage at {hi:.1f} ft"

    restricted = [s for s in spans if s.tags.get("parking:both:restriction") == "no_parking"]
    assert restricted, "the no_parking way is not among the matched spans"
    assert restricted[0].end_ft < model.legs["broad_st_east"].centerline.length, (
        "the restriction is supposed to cover only the approach, not the whole leg")


@needs_source_data
def test_a_restriction_over_part_of_a_kerb_reaches_the_paint(site_models):
    """End to end: the tag becomes a no-parking zone, hatching, and no stall inside it.

    The whole point of reading the split. East Broad's first ~80 ft must be hatched and carry no
    stall divider; beyond it, stalls. Asserted on the geometry the renderers actually draw, not
    on the state, because "the restriction is in the state" was true of the old code too - it was
    the paint that disagreed.
    """
    from src.geometry.treatments import apply_osm_parking

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        state = apply_osm_parking(DesignState.from_model(model), model)
        paint, _bands = paint_and_bands(model, state)
        scene = resolved_scene(model, state)

    leg = state.legs["broad_st_east"]
    restricted_to_ft = max(r.end_ft for r in state.parking_restrictions[("broad_st_east", "left")]
                           if r.prohibits)
    assert 60.0 < restricted_to_ft < 110.0, f"expected the first ~80 ft, got {restricted_to_ft:.1f}"

    # No stall marking may start inside the restricted stretch.
    for piece in paint:
        if piece.leg != "broad_st_east" or piece.kind not in ("stall_divider", "parking_edge_line"):
            continue
        stations, _offsets = station_offset_many(
            leg.centerline, np.asarray(piece.geometry.coords, dtype=float))
        assert stations.min() >= restricted_to_ft - 1.0, (
            f"{piece.kind} on broad_st_east {piece.side} reaches station {stations.min():.1f} ft, "
            f"inside the {restricted_to_ft:.0f} ft OSM tags as no_parking")

    # ...and the stretch is hatched rather than left blank.
    fills = [p for p in paint if p.leg == "broad_st_east" and p.kind is DAYLIGHT_FILL]
    assert fills, "the restricted stretch is not hatched"

    # The statutory zone and the OSM zone are both reported, each with its own citation.
    reasons = {z.reason for z in no_parking_zones_ft(state, "broad_st_east", "left",
                                                      scene.crosswalk_offsets)}
    assert any("OSM parking restriction" in r for r in reasons), reasons
    assert any("39:4-138" in r for r in reasons), reasons


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_replaying_a_designs_treatments_rebuilds_it(site, site_models):
    """state.treatments is a complete account of what a scenario did.

    This is the property the dict collapse rested on. Every treatment used to write one of
    DesignState's twenty dicts and every renderer read those dicts, so the treatment objects
    could in principle have been a partial record - a policy that edited the state directly
    rather than applying a treatment would leave the list short, and nothing would say so.
    complete_centerlines did exactly that until it was migrated.

    So: take the design a real scenario produces, apply its recorded treatments to a fresh
    baseline in order, and require the result to be the same design.

    WHAT "THE SAME DESIGN" MEANS is narrower after the collapse, and that narrowing is the
    result rather than a weakening. The dicts this used to compare are gone, so the parameters a
    renderer reads ARE the treatment list - there is no second store left to disagree with it,
    and the failure this test was written to catch is now unconstructible rather than merely
    unobserved. Comparing the paint would be tautological for the same reason: paint is a pure
    function of the treatments, the modelled street and the OSM facts from_model seeds.

    What is left is the MODELLED STREET, which is the one thing a treatment still writes:
    AddCurbExtension moves a kerb line and re-cuts the corner that kerb feeds, and
    SetCornerRadius re-cuts a corner. A policy that moved a kerb itself instead of applying one
    of those would leave a design replay cannot rebuild, and that is what this fails on.

    state.notes is deliberately NOT compared. A policy that emits several treatments explains
    itself in a note of its own - apply_osm_parking says which OSM tag produced each kerb's
    markings - and that sentence belongs to the policy, not to any one treatment, so replaying
    the treatments alone legitimately loses it.
    """
    import contextlib as _contextlib

    model = site_models[site]
    for name, builder in sorted(scenario_builders(site).items()):
        with _contextlib.redirect_stdout(io.StringIO()):
            built = run_scenario(builder, DesignState.from_model(model), model)
            replayed = DesignState.from_model(model).apply(*built.treatments, model=model)

        assert built.treatments, f"{site}/{name} recorded no treatments at all"
        assert _street_signature(replayed) == _street_signature(built), (
            f"{site}/{name}: replaying the recorded treatments produced a different modelled "
            f"street - so something moved a kerb or a corner without being recorded as a "
            f"treatment, and state.treatments is not the whole story")


def _street_signature(state) -> dict:
    """The modelled street a design ended up with: every kerb line and every corner fillet."""
    signature = {}
    for leg_name, leg in sorted(state.legs.items()):
        for side in ("left", "right"):
            curb = getattr(leg, f"{side}_curb")
            signature[f"{leg_name}.{side}_curb"] = None if curb is None else curb.wkt
    for corner, pieces in sorted(state.corner_fillets.items()):
        signature[f"{corner}"] = {key: round(value.length, 6)
                                  for key, value in sorted(pieces.items())
                                  if hasattr(value, "length")}
    return signature
