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
from src.sources.osm_context import fetch_crossings, fetch_kerbs, fetch_street_furniture, fetch_traffic_control

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
        stop_offsets = resolve_stop_bar_offsets(state, offsets) if model.config.get("signals") else {}
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


def test_an_explicit_centerline_style_beats_the_osm_tag():
    """Direct observation outranks OSM (src/provenance.py). A config that says
    single_yellow_dashed must not be overridden by a way tagged overtaking=no."""
    class FakeModel:
        config = {"legs": {"a": {"centerline_style": "single_yellow_dashed"}, "b": {}}}
        legs = {}
        corner_fillets = {}
        leg_osm_tags = {"a": {"overtaking": "no"}, "b": {"overtaking": "no"}}

    styles = DesignState.from_model(FakeModel()).centerline_styles
    assert styles["a"] == "single_yellow_dashed", "config must win over OSM"
    assert styles["b"] == "double_yellow", "OSM must be used where config is silent"
