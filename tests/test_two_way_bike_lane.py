"""A two-way bike lane on ONE side of a street, and the asymmetry that forces.

Every other treatment here is symmetric about the leg centerline, because NJDOT's alignment
is the datum every offset, stop bar and crossing frame is measured from. A two-way lane
breaks that: it takes 12-15 ft out of one kerbside and nothing out of the other, so the two
travel lanes no longer straddle the alignment - they sit off it, and the double yellow
between them moves with them.

The main README said this was "a real design, but not one this pipeline can draw". What
makes it drawable without moving the datum is that the datum does not have to be the middle
of the travel lanes: the alignment stays exactly where it is, every station and crossing
stays where it is, and the cross-section is simply described asymmetrically about it.
"""
import pytest

from src.geometry.treatments import (TARGET_LANE_WIDTH_FT, BikeLane, TwoWayBikeLane,
                                      far_kerb_surplus_ft, travel_lane_divider_shift_ft)
from tests.conftest import needs_source_data

# Broad & Greenwood's east leg, measured: 21.59 ft to the north kerb, 21.67 to the south.
NORTH_HALF_FT, SOUTH_HALF_FT = 21.59, 21.67


def test_a_one_way_lane_still_starts_at_the_travel_lane_width():
    """The default must not move. Every existing scenario depends on the section beginning
    at TARGET_LANE_WIDTH_FT from the alignment."""
    lane = BikeLane(width_ft=5.0, buffer_ft=2.0)
    assert lane.offsets_from_centerline_ft()["travel_lane_edge_ft"] == TARGET_LANE_WIDTH_FT


def test_a_two_way_section_is_measured_from_its_own_inner_edge():
    """The two-way lane's inner edge is NOT 11 ft from the alignment - it is wherever the
    shifted travel lanes end. Hard-coding TARGET_LANE_WIDTH_FT there would draw the lane
    overlapping the travel lane it is supposed to sit beside."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
    bounds = section.offsets_from_centerline_ft()
    # The section is the lane, its buffer AND the outer stripe that bounds it against the
    # hatching - 12 + 3 + one 0.82 ft line - because every width in BikeLane is between paint
    # faces and the stripes come out of the section rather than out of the travel lane.
    assert section.section_ft == pytest.approx(15.82, abs=0.01)
    assert bounds["travel_lane_edge_ft"] == pytest.approx(SOUTH_HALF_FT - section.section_ft,
                                                          abs=0.01)
    assert bounds["bike_outer_ft"] <= SOUTH_HALF_FT + 0.01, "the lane must not cross the kerb"


def test_the_travel_lanes_hold_the_target_width_and_the_far_kerb_keeps_the_surplus():
    """An equal split is the obvious rule and it is wrong on a wide street: Broad St's west leg
    gave two 18.35 ft lanes that way, and an 18 ft lane invites the speed this project exists to
    reduce. Spare width beside a travel lane is parking or hatching, never lane."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=26.24, far_half_ft=26.29)
    shift_ft = travel_lane_divider_shift_ft(section)
    inner_edge_ft = 26.24 - section.section_ft
    # The near travel lane runs from the section's inner edge to the divider.
    assert shift_ft + inner_edge_ft == pytest.approx(TARGET_LANE_WIDTH_FT, abs=0.01)
    # And the surplus lands against the far kerb, where parking can use it.
    assert far_kerb_surplus_ft(section) == pytest.approx(
        26.29 + inner_edge_ft - 2 * TARGET_LANE_WIDTH_FT, abs=0.01)
    assert far_kerb_surplus_ft(section) > 8.0, "this leg should free a stall's worth and more"


def test_a_leg_too_narrow_for_two_target_lanes_splits_what_it_has():
    """E Broad's east leg cannot hold two 11 ft lanes beside the section, so the shortfall is the
    street's and there is nothing to allocate - it splits equally and reports the width."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=18.04, far_half_ft=17.86)
    shift_ft = travel_lane_divider_shift_ft(section)
    travel_way_ft = 18.04 + 17.86 - section.section_ft
    assert travel_way_ft < 2 * TARGET_LANE_WIDTH_FT
    assert 17.86 - shift_ft == pytest.approx(travel_way_ft / 2, abs=0.01)
    assert far_kerb_surplus_ft(section) < 0


def test_the_divider_shifts_toward_the_far_kerb():
    """Sanity of sign: taking width out of the south kerbside pushes the traffic north."""
    section = TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0,
                             near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)
    assert travel_lane_divider_shift_ft(section) > 0


def test_a_section_that_leaves_no_room_for_two_travel_lanes_is_refused():
    """W Broad at Louellen: 32.10 ft of roadway. A 12 ft lane and a 3 ft buffer would leave
    17.1 ft for two lanes - 8.55 ft each, under any standard. Refused rather than drawn."""
    with pytest.raises(ValueError, match="travel lane"):
        TwoWayBikeLane(width_ft=12.0, buffer_ft=3.0, near_half_ft=17.26, far_half_ft=14.84)


@needs_source_data
def test_the_divider_shift_reaches_both_views(site_models):
    """The contraflow stripe and the shifted double yellow are the SAME decision reaching two
    renderers, which is the seam every marking in this project has shipped a bug at.

    Asserted through the design rather than by calling the paint helpers: what matters is that
    a scenario applying the treatment produces a shift both views can read, because a shift the
    plan view honours and the export does not is a render whose lanes are different widths.
    """
    from src.geometry.targets import LegSide
    from src.geometry.treatments import AddTwoWayBikeLane, DesignState

    model = site_models["broad_st_greenwood"]
    state = DesignState.from_model(model)
    lane = state.legs["broad_st_east"]
    south = "left" if lane.centerline.coords[-1][1] < lane.centerline.coords[0][1] else "right"
    state = state.apply(AddTwoWayBikeLane(LegSide("broad_st_east", south), width_ft=12.0,
                                          buffer_ft=3.0))
    shift = state.travel_lane_divider_shift("broad_st_east")
    assert shift is not None, "a two-way lane must record a divider shift"
    shift_ft, shift_side = shift
    assert shift_ft > 0
    assert shift_side != south, "the divider shifts AWAY from the side carrying the lane"
    # A leg with no two-way lane keeps the alignment as its divider - nothing else moves.
    assert state.travel_lane_divider_shift("greenwood_ave_north") is None


def test_a_lane_under_the_two_way_floor_is_refused():
    """A two-way lane carries opposing traffic, so it has its own floor - a 5 ft one-way
    width is not a two-way lane however much the arithmetic fits."""
    with pytest.raises(ValueError, match="two-way"):
        TwoWayBikeLane(width_ft=6.0, buffer_ft=3.0,
                       near_half_ft=SOUTH_HALF_FT, far_half_ft=NORTH_HALF_FT)


@needs_source_data
def test_the_south_side_is_resolved_per_leg(site_models):
    """A corridor decision ("the south kerb") is not a leg decision ("left"). The same real kerb
    is left on one approach and right on the other, and translating it by hand is how a corridor
    treatment lands on the north kerb of one leg and the south kerb of the next."""
    from src.geometry.model import side_facing

    state_legs = site_models["broad_st_greenwood"].legs
    east, west = state_legs["broad_st_east"], state_legs["broad_st_west"]
    # Opposite approaches of one street: the same ground is the other hand on each.
    assert side_facing(east, "south") != side_facing(west, "south")
    assert side_facing(east, "south") != side_facing(east, "north")


def test_a_north_south_leg_has_no_compass_side():
    """A leg running nearly due north-south has east and west sides, not north and south ones,
    and answering anyway would return whichever way its survey lean happened to fall.

    Built synthetically rather than borrowed from a site, because the site leg this used to
    assert on turned out not to be north-south at all: greenwood_ave_north runs 30 deg off due
    north (|dx|/len = 0.504), which has a perfectly clear compass side. The old guard refused it
    only because it used a |dx| < |dy| cut - a hard 45 deg threshold - and that same cut refused
    w_broad_st_southwest at 222.3 deg and silently dropped the corridor bike lane from one of the
    two Broad St legs at Louellen. The test agreed with the bug because it was written from it.
    """
    from types import SimpleNamespace

    from shapely.geometry import LineString

    from src.geometry.model import side_facing

    due_north = SimpleNamespace(name="synthetic_north_south",
                                 centerline=LineString([(0.0, 0.0), (0.0, 100.0)]))
    with pytest.raises(ValueError, match="north-south"):
        side_facing(due_north, "south")

    # And a diagonal DOES have one - 45 deg is not the boundary, near-due-north-south is.
    diagonal = SimpleNamespace(name="synthetic_diagonal",
                                centerline=LineString([(0.0, 0.0), (50.0, 100.0)]))
    assert side_facing(diagonal, "south") in ("left", "right")


def _two_way_scene(site_models, site="broad_st_greenwood"):
    """Build the two-way corridor scenario and hand back (model, state, paint pieces)."""
    import contextlib
    import io

    from src.geometry.treatments import DesignState
    from src.render.scene import SceneGeometry
    from src.site import load_site_scenarios, run_scenario
    from src.sources.osm_context import fetch_crossings

    model = site_models[site]
    builder = load_site_scenarios(site).build_proposal_two_way_bike_lane
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_scenario(builder, DesignState.from_model(model), model)
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        scene = SceneGeometry.resolve(model, state, crossings)
        return model, state, scene.build_paint()


@needs_source_data
def test_no_flex_post_stands_in_the_bike_lane(site_models):
    """The invariant Danny asked for, asserted on the real scenario.

    A post inside the lane is worse than no post: it removes ridable width and puts an obstacle
    where a rider belongs, while the drawing still reads as protected. Thirty of them were drawn
    down the middle of broad_st_east's lane and nothing failed - post_not_in_the_render compared
    the paint against the props, and both came off the same wrong cross-section, so they agreed.
    """
    from shapely.ops import unary_union

    from src.geometry.markings import BIKE_LANE_SURFACE

    _model, _state, paint = _two_way_scene(site_models)
    posts = [p for p in paint if p.kind.is_object]
    surfaces = [p.geometry for p in paint if p.kind is BIKE_LANE_SURFACE]
    assert posts, "this scenario is supposed to place flex posts - nothing to check otherwise"
    assert surfaces, "and to paint a two-way lane surface"
    lane = unary_union(surfaces)
    inside = [p for p in posts if lane.contains(p.geometry.centroid)]
    assert not inside, (
        f"{len(inside)} of {len(posts)} flex posts stand inside the bike lane surface rather than "
        f"in the buffer beside it")


@needs_source_data
def test_the_far_kerb_keeps_its_parking(site_models):
    """Hopewell Borough is car-dependent, so a corridor plan that returns no parking is not
    viable here however good it is for riders. This pins that the plan returns some.

    It also pins the bug that made it return none: a restriction over PART of a kerb was read as
    closing all of it, which hatched 90.4 ft of explicitly `restriction=none` kerb on
    broad_st_east.
    """
    from src.geometry.treatments import MarkedParking

    _model, state, _paint = _two_way_scene(site_models)
    parking = state.treatments_of(MarkedParking)
    on_broad = [p for p in parking if "broad_st" in p.target.leg]
    assert on_broad, (
        "the two-way corridor scenario marks no parking on either Broad St leg - the freed width "
        "on the far kerb is the whole reason the pair of treatments belongs in one proposal")


@needs_source_data
def test_the_drawn_centreline_sits_on_the_divider(site_models):
    """The painted centreline must be WHERE THE DIVIDER IS, and this is checked against the
    drawn geometry rather than against the arithmetic that was supposed to produce it.

    Everything else validated the intention: PaintClearOfTheTravelLane and
    TravelLanesKeepTheirWidth both measure against divider_shift_toward_ft, the stop bar rests
    against divider_shift_toward_ft, and all of them agreed. Nothing asked whether the line the
    renderer actually drew landed there. It did not on broad_st_west - the shift is NEGATIVE
    there, centerline_paint_ft took abs() of it, and the double yellow was drawn 1.42 ft on the
    WRONG side of the alignment, 2.84 ft from the stop bar it is supposed to meet. The travel
    lanes either side of it came out 13.84 ft and 8.16 ft against a reported 11.00.

    That is this project's signature failure - two derivations of one fact, agreeing with each
    other and not with the picture - and it is why the render is checked and not just the model.
    """
    import numpy as np

    from src.geometry.model import station_offset_many
    from src.geometry.treatments import divider_shift_toward_ft
    from src.geometry.targets import Side
    from src.render.crosswalks import centerline_paint_ft

    _model, state, _paint = _two_way_scene(site_models)
    for leg_name in ("broad_st_east", "broad_st_west"):
        leg = state.legs[leg_name]
        want_ft = divider_shift_toward_ft(state, leg_name, Side.LEFT)
        shift = state.travel_lane_divider_shift(leg_name)
        shift_ft, shift_side = shift if shift else (0.0, None)
        stripes = centerline_paint_ft(leg, 60.0, state.centerline_style(leg_name),
                                       shift_ft, shift_side)
        assert stripes, f"{leg_name} should have centreline paint"
        offsets = []
        for stripe in stripes:
            _st, off = station_offset_many(leg.centerline, np.asarray(stripe.coords, dtype=float))
            offsets.append(float(off.mean()))
        drawn_ft = sum(offsets) / len(offsets)      # midway between a double yellow's two lines
        assert drawn_ft == pytest.approx(want_ft, abs=0.15), (
            f"{leg_name}: the divider belongs {want_ft:+.2f} ft from the alignment (+ = left) and "
            f"the centreline is drawn at {drawn_ft:+.2f} ft - {abs(drawn_ft - want_ft):.2f} ft "
            f"away from it, so the two travel lanes are not the widths the design says")


@needs_source_data
def test_every_restriped_lane_holds_the_target_everywhere(site_models):
    """An 11 ft lane is the easy win, so it has to be the rule and not a per-scenario habit.

    Swept across every site and every scenario rather than the two-way one, because that is the
    claim: wherever a design has restriped a leg, both its travel lanes come out at the target
    and the surplus is parking or hatching. It was NOT true when written - E Broad's two-way
    scenario left 11.68 ft and 13.21 ft lanes, because the far-kerb rule was written inline in
    one site's scenarios.py and never reached the other's.

    The bound comes from treatments.lane_surplus_that_cannot_be_striped_ft, the same call
    checks.TravelLanesHoldTheTarget makes - a surplus too narrow to stripe cannot be taken off the
    lane, and demanding it be taken anyway is what put a 0.5 ft stripe over 0.37 ft of spare at E
    Broad. Imported rather than restated: this test and that check briefly held two copies.
    """
    import contextlib
    import io

    from scripts.build_all import scenarios_for
    from src.geometry.targets import LegSide, LegTarget
    from src.geometry.model import narrowest_half_width_ft
    from src.geometry.treatments import (TARGET_LANE_WIDTH_FT, AddBikeLane, DesignState,
                                          LaneNarrowing, MarkedParking, divider_shift_toward_ft,
                                          lane_surplus_that_cannot_be_striped_ft,
                                          travel_lane_width_ft)
    from src.site import load_site_scenarios, run_scenario

    over, checked = [], 0
    for site, model in site_models.items():
        with contextlib.redirect_stdout(io.StringIO()):
            scenarios = load_site_scenarios(site)
        for name in scenarios_for(site, scenarios):
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_scenario(getattr(scenarios, name),
                                      DesignState.from_model(model), model)
            for leg_name, leg in sorted(state.legs.items()):
                if leg.curb_to_curb_ft is None:
                    continue
                narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))

                def painted(side, narrowing=narrowing, leg_name=leg_name, state=state):
                    parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
                    if parking is not None:
                        return parking.depth_ft + parking.curb_offset_ft
                    return (narrowing.stripe_width_ft
                            if narrowing and side in narrowing.sides else 0.0)

                sides = ("left", "right")
                if not any(painted(s) > 0
                           or state.treatment_for(AddBikeLane, LegSide(leg_name, s))
                           for s in sides):
                    continue        # untouched leg - the street as it is, not a design
                for side in sides:
                    if state.treatment_for(AddBikeLane, LegSide(leg_name, side)):
                        continue
                    checked += 1
                    # Same traced-kerb bound TravelLanesHoldTheTarget applies: on an unpainted
                    # side the lane ends at the KERB, and the traced kerb is not the nominal one.
                    lane_ft = travel_lane_width_ft(state, leg_name, side, painted(side))
                    if painted(side) <= 0:
                        lane_ft = min(lane_ft,
                                       narrowest_half_width_ft(leg, side)
                                       - divider_shift_toward_ft(state, leg_name, side))
                    if lane_ft > (TARGET_LANE_WIDTH_FT
                                  + lane_surplus_that_cannot_be_striped_ft()):
                        over.append(f"{site}/{name} {leg_name} {side}: {lane_ft:.2f} ft")
    assert checked > 20, f"only {checked} lanes swept - the sweep stopped finding scenarios"
    assert not over, "travel lanes left over target on legs the design restriped:\n  " + \
                     "\n  ".join(over)


@needs_source_data
def test_a_two_way_design_carries_the_njdot_objection(site_models):
    """NJDOT's own guidance calls this facility unacceptable, so every design that uses it says so
    in its provenance.

    An invariant about DOCUMENTS rather than geometry, and the reason it is a test: a render that
    omits the objection claims more consensus than exists, and this is the state's published
    guidance for the state the project is in. The counter-argument - 1996 predates separated
    bikeways, MUTCD 11th ed. 9E.06 now marks them - belongs in the submission, not in silence.
    """
    from src.geometry.treatments import NJDOT_TWO_WAY_OBJECTION

    _model, state, _paint = _two_way_scene(site_models)
    two_way_notes = [n for n in state.notes if "AddTwoWayBikeLane" in n]
    assert two_way_notes, "this scenario should record a two-way lane"
    for note in two_way_notes:
        assert NJDOT_TWO_WAY_OBJECTION in note, (
            f"a two-way lane's note omits the NJDOT objection, so the render ships without it:\n"
            f"  {note}")


@needs_source_data
def test_a_two_way_lane_stays_against_the_kerb_that_protects_it(site_models):
    """The gap Danny pointed at, swept over every leg that carries a two-way lane.

    A protected lane's whole claim is the kerb beside it, and the lane used to be pinned to the
    leg's NARROWEST traced half-width - so wherever the street widened, the lane stayed put and the
    kerb walked away from it. On w_broad_st_southwest that left the lane 4.8 ft off the kerb at
    station 67 and 8.4 ft off at station 222: bare pavement down the outside of the bikeway for
    270 ft, on a drawing captioned "protected".

    THE MEDIAN, not the worst vertex, because the complaint was "for almost the whole length" and
    because one local excursion is the taper limiter working: broad_st_east's kerb flares 4.3 ft
    out over 20 ft at the corner and a lane is right not to follow that. What must not happen is
    the gap tracking the street's width down the whole leg. Measured on the DRAWN surface rather
    than on the section arithmetic, because the section was right both times and the picture
    was not.
    """
    import numpy as np
    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.model import curb_offsets_at_stations, station_offset_many

    WORST_TYPICAL_FT = 1.5     # the lane's own outer stripe is 0.82 ft wide

    by_side, far = {}, []
    for site in ("wbroad_louellen", "broad_st_greenwood", "ebroad_princeton"):
        model, _state, paint = _two_way_scene(site_models, site)
        lanes = [p for p in paint if p.kind is BIKE_LANE_SURFACE and p.leg in model.legs]
        assert lanes, f"{site} should paint a two-way lane surface"
        for piece in lanes:
            leg = model.legs[piece.leg]
            coords = np.asarray(piece.geometry.exterior.coords, dtype=float)
            stations, offsets = station_offset_many(leg.centerline, coords)
            kerb = curb_offsets_at_stations(leg, piece.side, stations)
            if kerb is None:
                continue
            gap = np.abs(kerb) - np.abs(offsets)
            # Only the vertices on the OUTER edge: the inner edge is a lane width further in, and
            # how far THAT sits from the kerb is not this test's question.
            outer = gap < gap.max() * 0.5 + 1.0
            if outer.any():
                by_side.setdefault(f"{site} {piece.leg} {piece.side}", []).append(gap[outer])

    assert len(by_side) >= 6, f"only {len(by_side)} lane sides swept - the sweep lost scenarios"
    for where, pieces in sorted(by_side.items()):
        typical = float(np.median(np.concatenate(pieces)))
        if typical > WORST_TYPICAL_FT:
            far.append(f"{where}: typically {typical:.2f} ft off the kerb")
    assert not far, ("a two-way lane is drawn away from the kerb that is supposed to protect it:\n"
                     "  " + "\n  ".join(far))
