"""Curb extensions: the treatment that shortens a crossing, and the one that only looks like it.

The distinction is the whole reason this file exists. `bump_out` claimed in its docstring that
"the curb physically extends into the corner" and had no test and no scenario, so the claim was
never checked. It re-cuts the corner ARC between two curb lines that stay exactly where they
are, which at these junctions moves no crossing at all - the crossings sit 21-42 ft out, past
the corner. Both facts are pinned below, because the failure mode is a proposal that renders
beautifully and shortens nothing.
"""
import contextlib
import io

import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.model import (BULBOUT_TAPER_RATE, Leg, build_pavement_polygon,
                                corner_apron_annulus, curb_extension_line,
                                curb_offsets_at_stations, narrowest_half_width_ft)
from src.geometry.targets import Corner, LegSide, LegTarget
from src.geometry.treatments import (AASHTO_MIN_BIKE_LANE_FT, AddBikeLane, AddBikeLaneBollards,
                                   AddCurbExtension, BikeLane, CornerApron,
                                   DesignState, LaneNarrowing, LaneNarrowingBollards,
                                   SetCornerRadius, TARGET_LANE_WIDTH_FT, find_corner)
from src.geometry.markings import BOLLARD, BUFFER_EDGE_LINE, BUFFER_FILL
from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT
from src.render.scene import SceneGeometry
from src.site import load_site_scenarios, run_scenario
from src.sources.osm_context import fetch_crossings

from tests.conftest import needs_source_data
from tests.test_sites import resolved_scene, scene_props

# Schedule I of the borough code prohibits parking 100 ft each way on both sides of both Broad
# St legs. A curb extension whose whole footprint fits inside that occupies kerb that is
# already legally not-parking, so it removes no space - which is the strongest thing that can
# be said for a bulb-out and the reason this number is a test rather than a comment.
SCHEDULE_I_NO_PARKING_FT = 100.0

# The two Broad St legs, their measured curb-to-curb widths, and the crossing span each one
# has TODAY - measured, not nominal. The nominal widths (52.0 / 55.5) are mid-block
# cross-sections; the crossings are painted where the traced kerbs have flared through the
# corner returns, 39.4 and 31.6 ft off the centerline on broad_st_east against a 26.0 ft
# nominal half-width. Being clear about which number is which is the point: an 8 ft extension
# per side reads as "52 -> 36" on the cross-section and is really 65.0 -> 35.5 on the ground.
BROAD_ST_TODAY = {"broad_st_east": 65.00, "broad_st_west": 69.50}
BROAD_ST_AFTER_8FT = {"broad_st_east": 36.0, "broad_st_west": 39.5}


def a_leg(length_ft=170.0, width_ft=52.0, half_traced_ft=26.0):
    """A straight leg with both kerbs traced at a constant offset, from station 15 outward."""
    leg = Leg(name="east", centerline=LineString([(0, 0), (length_ft, 0)]), curb_to_curb_ft=width_ft)
    for side, sign in (("left", 1), ("right", -1)):
        setattr(leg, f"{side}_curb",
                LineString([(15.0, sign * half_traced_ft), (length_ft, sign * half_traced_ft)]))
    return leg


def a_state(**kwargs):
    return DesignState(legs={"east": a_leg(**kwargs)}, corner_fillets={})


# --------------------------------------------------------------------------
# The blocker: what set_corner_radius does and does not do
# --------------------------------------------------------------------------

@needs_source_data
def test_set_corner_radius_alone_does_not_shorten_a_crossing(site_models):
    """Pinned so nobody "fixes" a bulb-out by tightening a radius again.

    Measured on broad_st_east x greenwood_ave_north, 29.2 -> 15.0 ft: the arc is genuinely
    re-cut (19.5 -> 3.5 ft) and the curb line simply runs on to the new tangent point
    (156.2 -> 164.2 ft), so the pavement moves 0.2 sq ft out of 24,000 and no crossing moves at
    all. The arithmetic was never wrong; the claim that it extended the curb was.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        before = SceneGeometry.resolve(model, base, crossings)
        corner = find_corner(base, "broad_st_east", "greenwood_ave_north")
        tightened = base.apply(SetCornerRadius(Corner(*corner), 15.0))
        after = SceneGeometry.resolve(model, tightened, crossings)

    assert tightened.corner_fillets[corner]["arc"].length < base.corner_fillets[corner]["arc"].length
    areas = (build_pavement_polygon(base.corner_fillets).area,
             build_pavement_polygon(tightened.corner_fillets).area)
    assert abs(areas[1] - areas[0]) < 1.0, f"pavement moved {areas[1] - areas[0]:.1f} sq ft"
    for leg_name in base.legs:
        assert sum(after.crosswalk_reaches[leg_name]) == pytest.approx(
            sum(before.crosswalk_reaches[leg_name]), abs=0.01), (
            f"{leg_name}'s crossing moved, which a radius change cannot do here")


# --------------------------------------------------------------------------
# ...and what a curb extension does
# --------------------------------------------------------------------------

@needs_source_data
def test_a_curb_extension_shortens_the_crossing_it_daylights(site_models):
    """The treatment doing the thing the other one only claimed.

    Both sides of both Broad St legs, 8 ft each. The reach is measured against the real kerb
    and the real pavement, so nothing here is asserted - the crossing is re-measured after the
    kerb moves, exactly as the renderers and the invariants measure it.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        before = SceneGeometry.resolve(model, base, crossings)
        state = _bulb_out_broad_st(base, before)
        after = SceneGeometry.resolve(model, state, crossings)

    for leg_name, today_ft in BROAD_ST_TODAY.items():
        assert sum(before.crosswalk_reaches[leg_name]) == pytest.approx(today_ft, abs=0.05), (
            "the starting span is measured, not nominal - if this moved, re-read the docstring "
            "before adjusting the expectation")
        got = sum(after.crosswalk_reaches[leg_name])
        # Within a foot of nominal-half-minus-extension doubled. Not exact: the reach is also
        # bounded by the pavement's own corner arc and stepped in 0.25 ft increments.
        assert got == pytest.approx(BROAD_ST_AFTER_8FT[leg_name], abs=1.0), (
            f"{leg_name} crosses {got:.1f} ft, wanted about {BROAD_ST_AFTER_8FT[leg_name]:.1f}")
        assert got < today_ft - 25.0, f"{leg_name} barely changed: {today_ft:.1f} -> {got:.1f}"

    # Greenwood keeps its crossings: it cannot spare the width for an extension, so none was
    # built on it, and a corner treated on one side only must not move the other street.
    for leg_name in ("greenwood_ave_north", "greenwood_ave_south"):
        assert sum(after.crosswalk_reaches[leg_name]) == pytest.approx(
            sum(before.crosswalk_reaches[leg_name]), abs=0.01)


@needs_source_data
def test_a_curb_extension_takes_real_ground_out_of_the_roadway(site_models):
    """A bulb-out is built, so the pavement polygon has to lose the area it occupies.

    set_corner_radius moves 0.2 sq ft; this moves about 1,090. That difference IS the
    treatment, and the pavement polygon is what the sidewalk band, the texture split and the
    furniture placement are all measured against.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        state = _bulb_out_broad_st(base, SceneGeometry.resolve(model, base, crossings))
        before_area = build_pavement_polygon(base.corner_fillets).area
        after_area = build_pavement_polygon(state.corner_fillets).area
    assert after_area < before_area - 500.0, (
        f"pavement only fell {before_area - after_area:.1f} sq ft - a bulb-out that takes no "
        f"ground is not a bulb-out")


@needs_source_data
def test_a_bulbout_fits_inside_the_ordinance_no_parking_length(site_models):
    """The claim that these extensions remove zero parking spaces.

    Schedule I already prohibits parking 100 ft each way on both sides of both Broad St legs.
    The extension's whole footprint - straight face plus taper - has to fit inside that, or it
    is occupying kerb somebody could otherwise legally park at, and the proposal has a cost it
    is not admitting to.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        state = _bulb_out_broad_st(base, SceneGeometry.resolve(model, base, crossings))

    assert state.treatments_of(AddCurbExtension), "nothing was built"
    for extension in state.treatments_of(AddCurbExtension):
        leg_name, side = extension.target.leg, extension.target.side
        assert extension.footprint_ft <= SCHEDULE_I_NO_PARKING_FT, (
            f"{leg_name} {side}'s extension runs {extension.footprint_ft:.1f} ft, past the "
            f"{SCHEDULE_I_NO_PARKING_FT:.0f} ft Schedule I already prohibits - it would remove "
            f"a legal parking space and the proposal claims it removes none")


# --------------------------------------------------------------------------
# What it refuses to build
# --------------------------------------------------------------------------

@needs_source_data
@pytest.mark.parametrize("leg_name", ["greenwood_ave_north", "greenwood_ave_south"])
def test_greenwood_cannot_take_an_eight_foot_extension(site_models, leg_name):
    """The finding, not an obstacle.

    Greenwood Ave is 26.6 and 31.2 ft curb to curb, so it has 2.3 and 4.6 ft per side to give
    beside an 11 ft lane. It cannot hold a bulb-out and two travel lanes at once, and the
    treatment says so instead of quietly building a shallower one - a drawing that no longer
    matches its own description is worse than a refusal.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
    with pytest.raises(ValueError, match="under the .* ft target"):
        base.apply(AddCurbExtension(LegSide(leg_name, "left"), extension_ft=8.0, crossing_ft=30.0))


def test_an_extension_that_would_eat_the_travel_lane_is_refused():
    state = a_state(width_ft=30.0)          # 15 ft half-width, so 4 ft to give
    with pytest.raises(ValueError, match="travel lane"):
        state.apply(AddCurbExtension(LegSide("east", "left"), extension_ft=6.0, crossing_ft=20.0))


def test_the_widest_extension_a_leg_can_take_leaves_exactly_the_target_lane():
    """The boundary is the target lane width, and it is inclusive - a leg with exactly enough
    room may use all of it."""
    state = a_state(width_ft=30.0)
    spare_ft = 15.0 - TARGET_LANE_WIDTH_FT
    built = state.apply(AddCurbExtension(LegSide("east", "left"), extension_ft=spare_ft, crossing_ft=20.0))
    offsets = curb_offsets_at_stations(built.legs["east"], "left", np.array([5.0]))
    assert abs(float(offsets[0])) == pytest.approx(TARGET_LANE_WIDTH_FT, abs=0.01)


def test_a_side_with_no_traced_kerb_cannot_be_extended():
    """An extension is measured from the kerb that is there. With nothing mapped there is
    nothing to move, and inventing one would be fabricating the baseline, not proposing a
    change to it."""
    state = a_state()
    state.legs["east"].left_curb = None
    with pytest.raises(ValueError, match="no traced kerb"):
        state.apply(AddCurbExtension(LegSide("east", "left"), extension_ft=4.0, crossing_ft=20.0))


# --------------------------------------------------------------------------
# The kerb geometry itself
# --------------------------------------------------------------------------

def test_the_face_sits_at_the_nominal_half_width_less_the_extension():
    leg = a_leg(width_ft=52.0, half_traced_ft=26.0)
    built = curb_extension_line(leg, "left", extension_ft=8.0, full_ft=34.0, taper_ft=40.0)
    leg.left_curb = built
    on_the_face = curb_offsets_at_stations(leg, "left", np.array([0.0, 10.0, 25.0, 33.0]))
    assert np.allclose(on_the_face, 18.0, atol=0.05), f"face at {on_the_face}"


def test_the_taper_returns_to_the_traced_kerb_and_stays_there():
    leg = a_leg(width_ft=52.0, half_traced_ft=26.0)
    leg.left_curb = curb_extension_line(leg, "left", 8.0, full_ft=34.0, taper_ft=40.0)
    past_the_taper = curb_offsets_at_stations(leg, "left", np.array([74.0, 100.0, 165.0]))
    assert np.allclose(past_the_taper, 26.0, atol=0.05), f"tail at {past_the_taper}"


def test_the_taper_is_monotonic_so_the_kerb_never_doubles_back():
    """A built kerb that wanders back and forth is a kerb no contractor can pour. The raised
    cosine is tangent at both ends, so the offset moves one way only."""
    leg = a_leg(width_ft=52.0, half_traced_ft=26.0)
    leg.left_curb = curb_extension_line(leg, "left", 8.0, full_ft=34.0, taper_ft=40.0)
    offsets = curb_offsets_at_stations(leg, "left", np.linspace(0.0, 90.0, 200))
    assert np.all(np.diff(offsets) >= -1e-9), "the extension's kerb reverses direction"


def test_the_face_never_sits_outside_the_traced_kerb():
    """Where the real kerb is already inside the nominal half-width - broad_st_east's left kerb
    is traced at 22.7 ft against 24.2 nominal - the tracing wins. An extension may take
    roadway; it may never invent it."""
    leg = a_leg(width_ft=52.0, half_traced_ft=14.0)     # kerb well inside the 26 ft nominal
    leg.left_curb = curb_extension_line(leg, "left", extension_ft=2.0, full_ft=34.0, taper_ft=10.0)
    offsets = curb_offsets_at_stations(leg, "left", np.linspace(16.0, 160.0, 80))
    assert offsets.max() <= 14.0 + 1e-6, f"kerb pushed out to {offsets.max():.2f} ft"


# --------------------------------------------------------------------------
# The apron - the swept path claim
# --------------------------------------------------------------------------

def test_the_apron_annulus_spans_the_face_radius_to_the_swept_radius():
    """The claim is that a bus keeps the corner it has today. That is only true if the mountable
    ground actually reaches the corner's own measured radius, so the annulus is built from both
    radii off the same two curb lines rather than offset off the drawn arc by a chosen depth."""
    curb_a = LineString([(0.0, 20.0), (200.0, 20.0)])
    curb_b = LineString([(20.0, 0.0), (20.0, -200.0)])
    annulus = corner_apron_annulus(curb_a, curb_b, face_radius_ft=15.0, swept_radius_ft=29.2)
    assert annulus is not None and annulus.area > 10.0

    # The claim stated directly: a vehicle tracking the corner's ORIGINAL arc stays on mountable
    # ground for the whole sweep. Asserted by covering that arc rather than by measuring a depth,
    # because the depth needed varies along the arc and only the covering is the thing promised.
    from src.geometry.model import fillet_curb_corner
    _a, swept_arc, _b = fillet_curb_corner(curb_a, curb_b, 29.2)
    assert annulus.buffer(1e-6).covers(swept_arc), "a bus on the original arc leaves the apron"

    # And it does not reach past the tightened face into the new sidewalk: the closest the apron
    # comes to the corner vertex is the face arc's own midpoint. A fillet of radius R puts that
    # R*(1/sin(theta/2) - 1) from the vertex, which at 90 degrees is 0.4142 R.
    corner = np.array([20.0, 20.0])
    reach = np.hypot(*(np.asarray(annulus.exterior.coords) - corner).T)
    assert reach.min() == pytest.approx(0.4142 * 15.0, abs=0.3), "the apron crosses the face"
    assert reach.max() == pytest.approx(29.2, abs=0.3), "outer edge is not the swept tangent point"


def test_no_apron_where_the_corner_was_not_tightened():
    curb_a = LineString([(0.0, 20.0), (200.0, 20.0)])
    curb_b = LineString([(20.0, 0.0), (20.0, -200.0)])
    assert corner_apron_annulus(curb_a, curb_b, face_radius_ft=29.2, swept_radius_ft=29.2) is None
    assert corner_apron_annulus(curb_a, curb_b, face_radius_ft=30.0, swept_radius_ft=29.2) is None


def test_an_apron_is_one_shape_or_the_other_never_both_and_never_neither():
    """A fixed depth inward from the arc and an annulus out to a swept radius are different
    shapes for different reasons; a record carrying both, or neither, describes nothing."""
    with pytest.raises(ValueError, match="exactly one"):
        CornerApron()
    with pytest.raises(ValueError, match="exactly one"):
        CornerApron(depth_ft=5.0, swept_radius_ft=29.2)
    with pytest.raises(ValueError, match="face_radius_ft"):
        CornerApron(swept_radius_ft=29.2)


@needs_source_data
def test_every_bulbout_corner_gets_an_apron_out_to_its_own_measured_radius(site_models):
    """Per corner, not one figure for the junction. The four corners at Broad & Greenwood are
    traced at 29.2, 24.6, 29.0 and 22.9 ft, and an apron built to an average would hand two of
    them less swept path than they have today."""
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        measured = {corner: pieces["radius_ft"]
                    for corner, pieces in base.corner_fillets.items()}
        state = _bulb_out_broad_st(base, SceneGeometry.resolve(model, base, crossings))

    aprons = {t.apron_corner(state): t.apron for t in state.treatments_of(AddCurbExtension)}
    assert len(aprons) == 4, "every treated corner needs its swept path back"
    for corner, apron in aprons.items():
        assert apron.swept_radius_ft == pytest.approx(measured[corner]), (
            f"{'/'.join(sorted(corner))}'s apron reaches {apron.swept_radius_ft} ft, but its "
            f"kerb is traced at {measured[corner]:.1f} ft")
        assert apron.face_radius_ft < apron.swept_radius_ft


# --------------------------------------------------------------------------
# Bike lanes
# --------------------------------------------------------------------------

def test_a_bike_lane_under_the_aashto_minimum_is_refused():
    with pytest.raises(ValueError, match="under AASHTO"):
        BikeLane(width_ft=AASHTO_MIN_BIKE_LANE_FT - 0.5)


@needs_source_data
@pytest.mark.parametrize("site,leg_name", [("broad_st_greenwood", "greenwood_ave_north"),
                                            ("broad_st_greenwood", "greenwood_ave_south"),
                                            ("ebroad_princeton", "princeton_ave_south")])
def test_the_narrow_legs_cannot_hold_a_bike_lane(site_models, site, leg_name):
    """The DO-NOT-PROPOSE finding, pinned.

    Greenwood Ave has 2.3 and 4.6 ft spare per side and Princeton Ave 4.1, all under AASHTO's
    5 ft minimum for an exclusive lane. Proposing one anyway would be proposing something that
    fails the standard it is meant to meet, and it is the kind of thing that gets waved through
    because the drawing looks plausible.
    """
    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
    spare_ft = base.legs[leg_name].curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
    assert spare_ft < AASHTO_MIN_BIKE_LANE_FT, (
        f"{leg_name} has {spare_ft:.1f} ft spare - it could take a lane after all, so this "
        f"test and the proposal's exclusion both need revisiting")
    with pytest.raises(ValueError):
        base.apply(AddBikeLane(LegSide(leg_name, "left"), width_ft=AASHTO_MIN_BIKE_LANE_FT))


@needs_source_data
def test_the_parking_protected_section_does_not_fit_broad_st(site_models):
    """The finding that changed the proposal, pinned so it cannot be quietly re-promised.

    The parking-protected cross-section - 8 parking + 3 buffer + 6 bike + 11 + 11 + 6 bike +
    3 buffer - totals 48 ft, which does fit inside 52.0 and 55.5 ft of roadway. But the total is
    not the constraint: everything here is measured as an offset from the leg centerline, and the
    PARKING SIDE alone needs 28.0 ft of that. broad_st_east has 26.01 ft nominal and 22.84 at its
    narrowest traced point; broad_st_west has 27.75 and 25.94. Both are short, by 5.2 and 2.1 ft
    against the tracing.

    Fitting it would mean shifting the travel lanes off the NJDOT alignment - a real design, but
    one this pipeline cannot draw: the alignment is the datum every offset, stop bar and crossing
    frame is measured from. So the proposal uses the buffered section, which does fit, and says
    so rather than drawing 48 ft of paint across a 46.5 ft narrow point.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
    protected = BikeLane(width_ft=6.0, buffer_ft=3.0, parking_ft=8.0)
    buffered = BikeLane(width_ft=6.0, buffer_ft=3.0)
    # 8 parking + 3 buffer + 6 bike + 11 travel per side: the spec's 48 ft, plus the outer stripe
    # each side needs - between the bike lane and the parking on one, between the bike lane and
    # its kerb hatching on the other.
    assert protected.total_ft + buffered.total_ft == pytest.approx(48.0 + 2 * LANE_EDGE_LINE_WIDTH_FT)

    for leg_name in ("broad_st_east", "broad_st_west"):
        leg = base.legs[leg_name]
        assert protected.total_ft > leg.curb_to_curb_ft / 2, (
            f"{leg_name} now has room for the parking side - re-check the proposal")
        with pytest.raises(ValueError, match="Short by"):
            base.apply(AddBikeLane(LegSide(leg_name, "right"), width_ft=6.0, buffer_ft=3.0, parking_ft=8.0))
        # ...and the buffered section fits at the leg's NARROWEST traced point, not just nominal.
        assert buffered.total_ft <= narrowest_half_width_ft(leg, "left")
        assert buffered.total_ft <= narrowest_half_width_ft(leg, "right")
        base.apply(AddBikeLane(LegSide(leg_name, "left"), width_ft=6.0, buffer_ft=3.0))


@needs_source_data
def test_a_bike_lane_is_bounded_by_the_tracing_not_the_nominal_width(site_models):
    """broad_st_east is 52.0 ft nominal - 26.0 per side - and its kerbs come within 22.84 ft of
    the alignment somewhere along the traced run. A cross-section between those two figures has
    to be refused, or it is drawn over the kerb at the narrow point."""
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        base = DesignState.from_model(model)
    leg = base.legs["broad_st_east"]
    nominal_ft = leg.curb_to_curb_ft / 2
    narrowest_ft = narrowest_half_width_ft(leg, "left")
    assert narrowest_ft < nominal_ft - 2.0, "this leg no longer narrows - the test needs a new one"

    # A section that fits the nominal half but not the tracing: bike lane sized to land between.
    between_ft = (nominal_ft + narrowest_ft) / 2 - TARGET_LANE_WIDTH_FT
    with pytest.raises(ValueError, match="narrowest traced point"):
        base.apply(AddBikeLane(LegSide("broad_st_east", "left"), width_ft=between_ft))


def test_a_cross_section_wider_than_the_leg_is_refused_with_the_shortfall():
    state = a_state(width_ft=40.0)      # 20 ft half-width
    with pytest.raises(ValueError, match="Short by"):
        state.apply(AddBikeLane(LegSide("east", "left"), width_ft=6.0, buffer_ft=3.0, parking_ft=8.0))


def test_the_bike_lane_boundaries_read_outward_from_the_centerline():
    """The ordering across the road IS the design: travel lane, buffer, bike lane, then parking
    OUTSIDE it. Parking outside the lane is what makes it parking-protected rather than a
    conventional lane with cars opening doors into it."""
    lane = BikeLane(width_ft=6.0, buffer_ft=3.0, parking_ft=8.0)
    at = lane.offsets_from_centerline_ft()
    assert at["travel_lane_edge_ft"] == pytest.approx(TARGET_LANE_WIDTH_FT)
    assert at["bike_inner_ft"] == pytest.approx(TARGET_LANE_WIDTH_FT + 3.0)
    assert at["bike_outer_ft"] == pytest.approx(TARGET_LANE_WIDTH_FT + 9.0)
    assert (at["travel_lane_edge_ft"] < at["bike_inner_ft"] < at["bike_outer_ft"]
            < at["parking_outer_ft"])


def test_every_bike_lane_stripe_lies_outside_the_width_it_protects():
    """The accounting that check_paint_clear_of_the_travel_lane exists to enforce.

    Paint has width. An 0.82 ft edge line centred on the 11 ft mark leaves a 10.59 ft travel
    lane, which is how the first version of this cross-section reported four violations at each
    of two sites. Every stripe centre therefore sits half a stripe OUTSIDE the face it marks,
    and the buffer between the two lanes is what pays for it.
    """
    half = LANE_EDGE_LINE_WIDTH_FT / 2
    for lane in (BikeLane(width_ft=6.0, buffer_ft=3.0),
                 BikeLane(width_ft=5.0, shy_ft=0.5),
                 BikeLane(width_ft=6.0, buffer_ft=3.0, parking_ft=8.0)):
        at = lane.offsets_from_centerline_ft()
        # The stripe against the travel lane: its inner face lands exactly on the 11 ft mark.
        assert at["inner_line_ft"] - half == pytest.approx(TARGET_LANE_WIDTH_FT)
        # ...and the bike lane keeps its own full width between faces.
        assert at["bike_outer_ft"] - at["bike_inner_ft"] == pytest.approx(lane.width_ft)
        if lane.buffer_ft:
            # Both bounding stripes come out of the buffer, not out of either lane.
            assert at["buffer_outer_line_ft"] + half == pytest.approx(at["bike_inner_ft"])
            assert at["buffer_outer_line_ft"] > at["inner_line_ft"]
        else:
            assert at["buffer_outer_line_ft"] is None, "no buffer means one stripe, not two"
        # The outer stripe is ALWAYS drawn: a bike lane is a standard width and the leftover to
        # the kerb is hatched, so the kerb is never the lane's own boundary. Without it the lane
        # read as reaching the kerb and looked far wider than specified.
        assert at["outer_line_ft"] is not None
        assert at["outer_line_ft"] - half == pytest.approx(at["bike_outer_ft"])


# --------------------------------------------------------------------------
# The recorded numbers
# --------------------------------------------------------------------------

def test_the_footprint_is_the_face_plus_the_taper():
    """And every number in it is derived from the treatment's own arguments.

    full_ft is the crossing plus half a crossing's depth plus the 10 ft the extension itself
    buys under R.S. 39:4-138(e), so this is asked of an AddCurbExtension rather than of a
    separate record built inside apply_to - there is nowhere for the two to disagree now.
    """
    extension = AddCurbExtension(LegSide("east", "left"), extension_ft=8.0, crossing_ft=21.0,
                                  taper_ft=40.0)
    # 21 ft to the crossing, + 3 ft of half a crossing's depth, + the 10 ft of R.S. 39:4-138(e).
    assert extension.full_ft == pytest.approx(34.0)
    assert extension.footprint_ft == pytest.approx(74.0)


def test_the_taper_length_follows_the_stated_rate():
    """BULBOUT_TAPER_RATE is a design choice and says so; this pins that the choice is actually
    applied, so the number in the docstring is the number in the geometry."""
    state = a_state()
    built = state.apply(AddCurbExtension(LegSide("east", "left"), extension_ft=8.0, crossing_ft=21.0))
    assert (built.treatment_for(AddCurbExtension, LegSide("east", "left")).resolved_taper_ft
            == pytest.approx(8.0 * BULBOUT_TAPER_RATE))


def _bulb_out_broad_st(base: DesignState, scene: SceneGeometry) -> DesignState:
    """Both sides of both Broad St legs, 8 ft, each corner's apron out to its OWN traced radius.

    The same construction sites/broad_st_greenwood/scenarios.py uses, reading the measured
    radius off the baseline fillet rather than repeating it as a literal - so a re-traced kerb
    flows through to the apron instead of leaving a stale number here.
    """
    state = base
    for leg_name in ("broad_st_east", "broad_st_west"):
        for side in ("left", "right"):
            corner = next(c for c in state.corner_fillets
                          if c[0 if side == "left" else 1] == leg_name)
            state = state.apply(AddCurbExtension(LegSide(leg_name, side), extension_ft=8.0, crossing_ft=scene.crosswalk_offsets[leg_name].offset_ft, swept_radius_ft=base.corner_fillets[corner]["radius_ft"]))
    return state


# --------------------------------------------------------------------------
# The bike lane cross-section, drawn
# --------------------------------------------------------------------------

@needs_source_data
def test_a_stop_bar_stops_where_the_bike_lane_starts(site_models):
    """A stopping car has no business in a bike lane, a buffer, or the kerb hatching.

    entering_lane_width_ft knew about lane narrowing and marked parking but not about bike lanes,
    so on a bike-lane leg it returned None - "use the full curb-to-curb half" - and the bar was
    drawn straight across the lane. Measured in the leg's own frame, against the travel lane edge
    the lane records.
    """
    from src.render.crosswalks import entering_lane_width_ft
    from src.geometry.model import station_offset_many

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)

    checked = 0
    for leg_name, band in sorted(scene.stop_bar_bands.items()):
        treatment = state.treatment_for(AddBikeLane, LegSide(leg_name, "left"))
        if treatment is None or band is None or band.is_empty:
            continue
        lane = treatment.lane
        checked += 1
        assert entering_lane_width_ft(state, leg_name) == pytest.approx(TARGET_LANE_WIDTH_FT), (
            f"{leg_name}'s bar is being sized against something other than the travel lane")
        bounds = lane.offsets_from_centerline_ft()
        _stations, offsets = station_offset_many(
            state.legs[leg_name].centerline, np.asarray(band.exterior.coords, dtype=float))
        # Never into the bike lane itself. It may reach into its own edge stripe, and a skewed
        # bar reaches a fraction further in the leg's frame because the whole band is rotated -
        # which is why this is bounded by the lane's inner FACE and not by the bare 11 ft mark.
        assert np.abs(offsets).max() <= bounds["bike_inner_ft"] - 0.05, (
            f"{leg_name}'s stop bar reaches {np.abs(offsets).max():.2f} ft, into a bike lane whose "
            f"inner face is at {bounds['bike_inner_ft']:.2f} ft")
    assert checked == 2, f"expected both Broad St legs to have been checked, got {checked}"


@needs_source_data
def test_bike_lane_bollards_stand_in_the_buffer_on_the_traffic_side(site_models):
    """Posts protecting a bike lane go between it and the moving traffic.

    Not in the kerb-side hatching, where they would protect nothing, and not in either lane. So
    every post has to land between the travel lane's edge and the bike lane's inner face - which
    is the buffer, and is why add_bike_lane_bollards refuses a lane that has no buffer.
    """
    from src.geometry.model import station_offset_many

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    assert state.treatments_of(AddBikeLaneBollards), "the proposal is supposed to protect its lanes"
    posts = [p for p in paint if p.kind is BOLLARD]
    assert posts, "no delineators were drawn"
    for piece in posts:
        lane = state.treatment_for(AddBikeLane, LegSide(piece.leg, piece.side)).lane
        bounds = lane.offsets_from_centerline_ft()
        _stations, offsets = station_offset_many(
            state.legs[piece.leg].centerline,
            np.asarray(piece.geometry.exterior.coords, dtype=float))
        at_ft = float(np.abs(offsets).mean())
        assert bounds["travel_lane_edge_ft"] <= at_ft <= bounds["bike_inner_ft"], (
            f"a post on {piece.leg} {piece.side} stands {at_ft:.2f} ft out, outside the buffer "
            f"({bounds['travel_lane_edge_ft']:.2f}-{bounds['bike_inner_ft']:.2f} ft)")


def test_bollards_are_refused_on_a_bike_lane_with_no_buffer():
    """E Broad's case. A lane with no buffer has nowhere to put a post that is not in a travel
    lane or in the bike lane, and improvising one would draw protection that cannot be built."""
    state = a_state(width_ft=40.0).apply(AddBikeLane(LegSide("east", "left"), width_ft=6.0))
    with pytest.raises(ValueError, match="no buffer"):
        state.apply(AddBikeLaneBollards(LegSide("east", "left")))


def test_a_bike_lane_holds_its_width_and_hatches_the_rest_to_the_kerb():
    """The parking-stall rule, applied along the other axis.

    An 8 ft stall stays 8 ft and the leftover becomes a hatched kerb buffer; a bike lane is no
    different. Without the outer stripe and that hatching the lane read as running to the kerb,
    which is what made a 6 ft lane look far wider than 6 ft.
    """
    lane = BikeLane(width_ft=6.0, buffer_ft=3.0)
    at = lane.offsets_from_centerline_ft()
    assert at["bike_outer_ft"] - at["bike_inner_ft"] == pytest.approx(6.0)
    # Whatever the street has beyond the outer stripe is hatching, and it absorbs ALL the
    # variation - so the lane is 6 ft on a 21 ft half-width and 6 ft on a 26 ft one.
    assert lane.kerb_hatch_ft(21.3) == pytest.approx(21.3 - at["outer_ft"])
    assert lane.kerb_hatch_ft(26.0) == pytest.approx(26.0 - at["outer_ft"])
    assert lane.kerb_hatch_ft(at["outer_ft"] - 1.0) == 0.0, "hatching pinches out, never negative"


@needs_source_data
@pytest.mark.parametrize("site", ["broad_st_greenwood", "ebroad_princeton"])
def test_the_green_surface_covers_the_bike_lane_and_nothing_else(site, site_models):
    """A green bike lane is green over the LANE - between its two edge stripes, not past them.

    The width is the whole content of the marking: green asphalt is how a rider is told which
    part of the road is theirs, so green reaching 6.6 ft past the outer stripe (which the first
    construction here did, on broad_st_west's right lane, wherever the traced kerb is unmapped)
    claims ground the proposal is not offering. Nothing else caught it: that ground carries no
    other paint to collide with and no traced kerb to be outside of, so both
    MarkingsDoNotCollide and PaintInsideTheCurb were silent and correct to be.

    Bounded here by the lane's own offsets, which is also what the marking is now built from
    (src/geometry/model.py:offset_band_polygon).
    """
    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.model import station_offset_many

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios(site).build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    lanes = state.treatments_of(AddBikeLane)
    assert lanes, f"{site}'s bike lane proposal built no lanes"
    for treatment in lanes:
        leg_name, side = treatment.target.leg, str(treatment.target.side)
        bounds = treatment.lane.offsets_from_centerline_ft()
        green = [p for p in paint if p.kind is BIKE_LANE_SURFACE
                 and p.leg == leg_name and p.side == side]
        assert green, f"{leg_name} {side}'s bike lane has no green surface painted on it"
        for piece in green:
            _stations, offsets = station_offset_many(
                state.legs[leg_name].centerline,
                np.asarray(piece.geometry.exterior.coords, dtype=float))
            reach_ft = float(np.abs(offsets).max())
            # A tenth of a foot of slack: the band is placed in the leg's MEASURED frame and
            # read back here by projection, and on a centerline that kinks (broad_st_east bends
            # 4.5 deg 43 ft out) those two frames differ by about half an inch.
            assert reach_ft <= bounds["bike_outer_ft"] + 0.1, (
                f"{leg_name} {side}'s green reaches {reach_ft:.2f} ft from the centerline, past "
                f"its own outer stripe at {bounds['bike_outer_ft']:.2f} ft - it is painting "
                f"asphalt that is not the bike lane")
            assert float(np.abs(offsets).min()) >= bounds["bike_inner_ft"] - 0.1, (
                f"{leg_name} {side}'s green reaches inside its inner stripe, into the buffer "
                f"or the travel lane")


@needs_source_data
def test_the_kerb_hatching_beside_a_bike_lane_is_actually_drawn(site_models):
    """...and reaches the kerb rather than stopping at the lane's outer stripe."""
    from src.geometry.model import curb_offsets_at_stations, station_offset_many

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint = scene.build_paint(scene_props(model, state, scene))

    for treatment in state.treatments_of(AddBikeLane):
        leg_name, side, lane = treatment.target.leg, str(treatment.target.side), treatment.lane
        fills = [p for p in paint if p.leg == leg_name and p.side == side
                 and p.kind is BUFFER_FILL]
        assert fills, f"{leg_name} {side} has no hatching between its bike lane and the kerb"
        outer_ft = lane.offsets_from_centerline_ft()["outer_ft"]
        leg = state.legs[leg_name]
        for piece in fills:
            points = np.asarray(piece.geometry.exterior.coords, dtype=float)
            stations, offsets = station_offset_many(leg.centerline, points)
            # Inside the lane's outer stripe is the lane, not hatching.
            assert np.abs(offsets).max() >= outer_ft - 0.1, (
                f"{leg_name} {side}'s hatching only reaches {np.abs(offsets).max():.2f} ft")
            kerb = curb_offsets_at_stations(leg, side, stations)
            assert np.all(np.abs(offsets) <= np.abs(kerb) + 0.25), "hatching crosses the kerb"


@needs_source_data
def test_the_bike_lanes_bollards_reach_the_3d_render(site_models):
    """A post in the plan view and no post in the render is not a rendering difference.

    The two views take posts from different places: the plan view draws them off the paint,
    and the 3D render builds objects, which it only ever does from props. So a bollard that
    exists only as a PaintPiece is drawn in 2D and simply absent in 3D - which is what
    shipped. Broad St's lanes were drawn with 61 flex posts protecting them and exported
    with none, and no check compared the two.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint, props = scene.build_paint_and_posts(scene_props(model, state, scene))

    painted = [p.geometry.centroid for p in paint if p.kind is BOLLARD]
    assert painted, "the proposal is supposed to protect its lanes"
    placed = np.array([p["position_ft"] for p in props if p["type"] == "bollard"], dtype=float)
    assert len(placed), "the export ships no bollard props at all - the 3D render draws none"
    for point in painted:
        assert np.hypot(placed[:, 0] - point.x, placed[:, 1] - point.y).min() <= 0.1, (
            f"a post drawn at ({point.x:.1f}, {point.y:.1f}) in the plan view has no prop, so "
            f"the 3D render builds nothing there")
    # And the invariant that says so, on the geometry both renderers actually check.
    assert [v for v in scene.check(props, paint) if v.check == "post_not_in_the_render"] == []


@needs_source_data
def test_a_post_drawn_only_in_paint_is_an_invariant_failure(site_models):
    """The check above, checked: drop one prop and the scene must stop being valid.

    Without this, check_bollards_are_props could be vacuous - it passes trivially on the
    three scenarios that paint no posts at all.
    """
    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint, props = scene.build_paint_and_posts(scene_props(model, state, scene))

    kept = [p for p in props if p["type"] != "bollard"]
    violations = [v for v in scene.check(kept, paint) if v.check == "post_not_in_the_render"]
    assert len(violations) == sum(1 for p in paint if p.kind is BOLLARD), (
        "removing every bollard prop should report every painted post as missing from the render")


@needs_source_data
def test_the_kerb_hatching_beside_a_bike_lane_is_trimmed_where_the_crossing_cuts_it(site_models):
    """A hatched zone is outlined, and the outline carries on around its cut end.

    The plan view outlines a fill polygon for free, so this zone looked finished in 2D while
    the 3D render - which is handed the hatch strokes and the lines actually painted, and
    nothing else - had its strokes stopping in mid-air at the crossing. Every other hatched
    zone here goes through paint.py's rim(); this one did not.
    """
    from src.geometry.model import station_offset_many

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        builder = load_site_scenarios("broad_st_greenwood").build_proposal_bike_lanes
        state = run_scenario(builder, DesignState.from_model(model), model)
        scene = resolved_scene(model, state)
        paint, _props = scene.build_paint_and_posts(scene_props(model, state, scene))

    for treatment in state.treatments_of(AddBikeLane):
        leg_name, side, lane = treatment.target.leg, str(treatment.target.side), treatment.lane
        hatching = [p for p in paint if p.leg == leg_name and p.side == side
                    and p.kind is BUFFER_FILL]
        assert hatching, f"{leg_name} {side} has no kerb hatching to trim"
        # A rim is the zone's own edge line continued around the cut (no dedicated kind any
        # more - see PaintContext.rim), so it is identified by running ACROSS the strip where
        # the longitudinal edge lines run along it.
        rims = [p for p in paint if p.leg == leg_name and p.side == side
                and p.kind is BUFFER_EDGE_LINE
                and np.ptp(station_offset_many(
                    state.legs[leg_name].centerline,
                    np.asarray(p.geometry.coords, dtype=float))[1]) > 1.0]
        outer_ft = lane.offsets_from_centerline_ft()["outer_ft"]
        # The buffer between the lane and the traffic has a rim of its own, and it is the
        # INNER one - so this looks for a rim out where the kerb hatching is, not just any.
        reaches = [np.abs(station_offset_many(
            state.legs[leg_name].centerline,
            np.asarray(piece.geometry.coords, dtype=float))[1]).max() for piece in rims]
        assert any(reach >= outer_ft for reach in reaches), (
            f"{leg_name} {side}: the kerb hatching's cut end has no line finishing it off - "
            f"rims reach {sorted(round(r, 2) for r in reaches)}, hatching starts at "
            f"{outer_ft:.2f} ft")


def test_bollards_only_stand_in_the_buffer_that_is_painted():
    """A leg narrowed on one kerb gets posts on that kerb.

    props.py used to take bollard_points_ft's both-sides default, which stood a row of posts
    down a buffer that is not painted on the other kerb: present in the 3D render, absent
    from the plan view, which draws them from the paint.
    """
    from src.geometry.model import station_offset_many
    from src.render.props import _bollard_props

    state = a_state(width_ft=40.0).apply(LaneNarrowing(LegTarget("east"), stripe_width_ft=3.0, sides=("left",))).apply(LaneNarrowingBollards(LegTarget("east"), spacing_ft=10.0))
    posts = _bollard_props(state)
    assert posts, "no posts were placed at all"
    _stations, offsets = station_offset_many(
        state.legs["east"].centerline,
        np.array([p["position_ft"] for p in posts], dtype=float))
    assert np.all(offsets > 0), (
        "posts were placed on the right kerb of a leg narrowed only on its left")
