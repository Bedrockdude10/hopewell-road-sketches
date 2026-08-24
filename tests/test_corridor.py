"""THE CHECKPOINT FOR STEP 2 of docs/network-model.md: a road that runs the length of the borough.

tests/test_network.py established that a road through ONE junction reproduces the per-leg widths.
This file asks the same question of a road that spans several junctions and carries on past them
along NJDOT's alignment, plus the two things a corridor has to get right that a 300 ft road cannot
get wrong:

  * it must not SKIP a junction. Broad St resolves as one road across all three modelled ones, and
    the axis has to run one way over all 2,800+ ft of it, not just through each node.
  * every figure taken off it must carry the coverage it was measured over. That is asserted
    STRUCTURALLY - Coverage has no defaults and Figure cannot be built without one - rather than by
    matching strings in the output, because the failure being guarded against is a future question
    added without a denominator, and a string test would pass for that until someone read it.

The numbers here are deliberately PROPERTIES, not values. The committed OSM snapshot is a moment in
a survey that is still being traced (the live cache already has ~50% more kerb along Broad St than
the fixture does), so a test that pinned "1,227 ft traced" would fail on the next refresh for the
best possible reason. What must not change is that the road is one road, that its widths still
agree with the legs', and that nothing is counted without saying over how much.
"""
import contextlib
import dataclasses
import io

import numpy as np
import pytest
from shapely.geometry import Point

from scripts.corridor_report import Coverage, Figure, corridor_report
from src.geometry.corridor_paint import hatch_bands
from src.geometry.network import (KERB_FROM_TRACING, corridor_facts, corridors_from_models,
                                 marked_parking_capacity, osm_window_spans)
from tests.conftest import WIDE_FRAME_SCALE, needs_source_data
from tests.test_network import AGREEMENT_TOL_FT, _leg_width_at

# The three sites this project models on Broad St / CR 518. Columbia & Princeton is deliberately in
# the model set too: it is on neither Broad St nor its SRI, so it is what proves the chaining
# discriminates rather than merely joining everything it is given.
BROAD_ST_SITES = ("wbroad_louellen", "broad_st_greenwood", "ebroad_princeton")

# A single junction's Road is 260-300 ft (tests/test_network.py). A corridor across three of them
# has to be several times that, and this is the floor: the 2,435 ft between the outer two junction
# nodes, which is a measured distance and not a guess about how far the extensions reach.
MIN_BROAD_ST_LENGTH_FT = 2435.0

# Broad St resolves to 4526.0 ft at 1x and 4527.4 ft at 3x - the same survey, re-chained through
# junction pieces of a different length, so the joins land on slightly different vertices. That
# 1.6 ft is re-assembly noise; anything above it is the sheet adding street.
CORRIDOR_EXTENT_TOL_FT = 2.0

# Traced coverage carries the same re-assembly noise, as a FRACTION: re-chaining moves where the
# kerb stations against the corridor, so a run's measured span shifts a little. Measured worst case
# is Princeton Ave, 31.2 ft of 2065.0 (1.5%) between 1x and 3x. Fractional rather than absolute
# because the noise is per-seam and Broad St has the most seams and the most coverage.
DESIGN_SPAN_TOL = 0.02


@pytest.fixture(scope="session")
def corridors(site_models):
    return corridors_from_models(site_models)


@pytest.fixture(scope="session")
def broad_st(corridors):
    matches = [road for road in corridors if set(BROAD_ST_SITES) <= set(road.sites)]
    assert len(matches) == 1, (
        f"expected exactly one road across all of {BROAD_ST_SITES}, got "
        f"{[(road.name, road.sites) for road in corridors]}")
    return matches[0]


@needs_source_data
def test_broad_st_is_one_road_across_all_three_junctions(broad_st):
    """THE HEADLINE. All three modelled junctions land on a single road, in order along it.

    The failure this is written against is a chain that SKIPS the middle junction: E Broad &
    Princeton faces W Broad & Louellen at 2,417 ft on the same SRI with opposed bearings and passes
    every geometric test, so a matcher that took the first plausible link would produce a road with
    a 2,400 ft hole and Broad & Greenwood sitting on a road of its own.
    """
    assert set(broad_st.sites) == set(BROAD_ST_SITES)
    nodes = [junction.node_ft for junction in broad_st.junctions]
    assert nodes == sorted(nodes), (
        f"{broad_st.name}'s junctions are at stations {nodes} - a chain out of order means the "
        f"pieces were assembled in one order and stationed in another")
    assert broad_st.length_ft > MIN_BROAD_ST_LENGTH_FT, (
        f"{broad_st.name} is only {broad_st.length_ft:.0f} ft long; the outer two junction nodes "
        f"alone are {MIN_BROAD_ST_LENGTH_FT:.0f} ft apart, so it cannot be spanning them")
    assert nodes[-1] - nodes[0] > MIN_BROAD_ST_LENGTH_FT
    assert "broad" in broad_st.name.lower(), (
        f"the corridor across Broad St is called {broad_st.name!r} - the compass halves OSM and "
        f"NJDOT split this street into should collapse to one name")


@needs_source_data
def test_broad_st_carries_two_njdot_routes(broad_st):
    """CR 518 turns west onto Louellen St, so this street is two SRIs and must say so.

    Not bookkeeping. A corridor report that called the whole thing SRI 00000518__ would be wrong
    about everything southwest of Louellen, which is CR 654 - and the mistake is invisible in a
    render. The road is a STREET; an SRI is a route reference that happens to leave it.
    """
    sris = [sri for _lo, _hi, sri in broad_st.sri_spans]
    assert len(sris) == len(set(sris)) and len(sris) >= 2, (
        f"{broad_st.name} reports routes {sris}; CR 518 (00000518__) turns off it at Louellen St "
        f"and CR 654 (11000654__) carries on, so it has at least two")
    assert broad_st.sri_spans[0][0] == pytest.approx(0.0)
    assert broad_st.sri_spans[-1][1] == pytest.approx(broad_st.length_ft)
    for (_lo, hi, _sri), (next_lo, _next_hi, _next) in zip(broad_st.sri_spans,
                                                           broad_st.sri_spans[1:]):
        assert hi == pytest.approx(next_lo), "the route spans must tile the road with no gap"


@needs_source_data
def test_the_borough_length_road_still_reproduces_every_leg_width(corridors, site_models):
    """THE GUARD ON THE DATUM: extending the road did not move the widths it reports.

    Same measurement as tests/test_network.py's checkpoint and the same tolerance, asked of the
    long road instead of the per-junction one. It is the whole reason the corridor is built by
    CHAINING the junction Roads rather than re-reading the traced kerbs against NJDOT's raw
    alignment: measured that way the two agree to 0.03-0.36 ft out along the street and disagree by
    up to 2.8 ft in a junction throat, where the kerb flares 0.68 ft per foot of station.
    """
    disagreements, compared = [], 0
    for corridor in corridors:
        for junction in corridor.junctions:
            for leg_name, _sign in junction.legs:
                leg = site_models[junction.site].legs[leg_name]
                for leg_station in np.linspace(5.0, max(leg.centerline.length - 5.0, 6.0), 8):
                    leg_width = _leg_width_at(leg, float(leg_station))
                    width = corridor.width_at_ft(corridor.station_of(leg_name, float(leg_station)))
                    if leg_width is None or width is None:
                        continue
                    compared += 1
                    if abs(width - leg_width) > AGREEMENT_TOL_FT:
                        disagreements.append(
                            f"{corridor.name} @ {corridor.station_of(leg_name, leg_station):.1f} "
                            f"({junction.site}/{leg_name} @ {leg_station:.1f}): road says "
                            f"{width:.2f} ft, leg says {leg_width:.2f} ft")
    assert compared >= 24, (
        f"only {compared} station(s) could be compared across {len(corridors)} corridor(s) - the "
        f"guard is not exercising the roads it is meant to")
    assert not disagreements, (
        f"{len(disagreements)} of {compared} station(s) disagree by more than {AGREEMENT_TOL_FT} "
        f"ft:\n  " + "\n  ".join(disagreements[:8]))


@needs_source_data
def test_the_corridor_axis_runs_one_way_over_its_whole_length(corridors):
    """Stations increase monotonically along every corridor, bridges and extensions included.

    tests/test_network.py pins this per junction, where the only way it can fail is a leg that was
    not reversed. A corridor has three more ways to fail it, and all three are in the assembly: a
    bridge laid in the alignment's direction rather than the road's, an extension pointing back
    into the road instead of away from it, and a lateral correction steep enough to fold the frame.
    Every station-based clip downstream assumes this does not happen.
    """
    for corridor in corridors:
        coords = np.asarray(corridor.centerline.coords, dtype=float)
        stations = np.asarray([corridor.centerline.project(Point(point)) for point in coords])
        backwards = [(i, stations[i - 1], stations[i]) for i in range(1, len(stations))
                     if stations[i] < stations[i - 1] - 1e-6]
        assert not backwards, (
            f"{corridor.name}'s axis doubles back at vertex {backwards[0][0]} of {len(coords)}: "
            f"{backwards[0][1]:.2f} -> {backwards[0][2]:.2f} ft")
        # ...and the arc length has to agree with the axis, or a station means two places.
        assert stations[-1] == pytest.approx(corridor.length_ft, abs=0.5)


@needs_source_data
def test_a_leg_station_of_zero_is_its_own_junction_node(corridors):
    """Each of a junction's legs starts where it actually meets the corridor.

    Both start at the node in the model; the two NJDOT alignments behind them do not meet, and
    JunctionOnRoad.leg_joint_ft carries what the blend could not close. So a leg maps to the node
    plus its own joint - zero for one leg, the open gap for the other (2.79 ft at W Broad &
    Louellen) - and the network-level twin of this test,
    tests/test_network.py:test_a_leg_station_maps_to_the_node_at_zero, says the same thing about
    the per-junction Road. Asserting a flat node_ft for both is what put every far-leg station up
    the street.
    """
    for corridor in corridors:
        for junction in corridor.junctions:
            joints = dict(junction.leg_joint_ft)
            for leg_name, sign in junction.legs:
                assert corridor.station_of(leg_name, 0.0) == pytest.approx(
                    junction.node_ft + sign * joints.get(leg_name, 0.0), abs=1e-6)
            assert junction.start_ft < junction.node_ft < junction.end_ft


@needs_source_data
def test_asking_for_a_leg_that_is_not_on_the_road_raises(corridors, site_models):
    """A leg on another street is not on this road, and a quiet answer would put a corridor
    figure on the wrong street."""
    for corridor in corridors:
        on_it = set(corridor.leg_names)
        others = {name for site in corridor.sites for name in site_models[site].legs} - on_it
        for other in sorted(others):
            with pytest.raises(KeyError, match=other):
                corridor.station_of(other, 10.0)


@needs_source_data
def test_the_road_refuses_a_width_where_the_kerb_is_not_traced(broad_st):
    """No width is reported across a stretch nobody traced - the honesty requirement, as a test.

    This is the one property that makes every width figure in the report trustworthy. np.interp
    will happily hold the last traced offset flat forever, so without the span test the 1,126 ft of
    W Broad between Greenwood Ave and Louellen St that has no `barrier=kerb` on either side in the
    committed snapshot would come back as a confident cross-section straight across it.
    """
    gaps = broad_st.unmeasurable_gaps_ft(min_ft=20.0)
    if not gaps:
        pytest.skip("every stretch of this road now has a kerb on both sides - nothing to refuse")
    for lo, hi in gaps:
        middle = (lo + hi) / 2
        assert broad_st.width_at_ft(middle) is None, (
            f"{broad_st.name} reports a width at station {middle:.0f}, inside the "
            f"{hi - lo:.0f} ft stretch {lo:.0f}-{hi:.0f} where it has no kerb to read")
    # ...and the surveyed-coverage gaps are at least as big, because a modelled junction's kerb
    # line is not a survey: no width figure may be counted over one.
    assert (sum(hi - lo for lo, hi in broad_st.untraced_gaps_ft())
            >= sum(hi - lo for lo, hi in gaps) - 1e-6)


@needs_source_data
def test_traced_coverage_is_counted_over_the_surveyors_own_kerb_only(broad_st):
    """Coverage comes from the TRACED runs, never from a modelled junction's assembled kerb line.

    The distinction is the point of KerbRun.source. A junction's kerb line has been extended to its
    leg's working length by curb_line_from_points, so counting it as coverage would report the
    project's own extrapolation as somebody's survey - which is the reporting failure this whole
    stream exists to remove, pointed the other way.
    """
    for side in ("left", "right"):
        traced = broad_st.traced_ft(side)
        runs = [run for run in broad_st.kerb_runs if run.side == side]
        assert any(run.source == KERB_FROM_TRACING for run in runs)
        assert any(run.source != KERB_FROM_TRACING for run in runs), (
            "this road spans modelled junctions, so some of its kerb must come from them")
        from_tracing = sum(run.length_ft for run in runs if run.source == KERB_FROM_TRACING)
        assert traced <= from_tracing + 1e-6, (
            f"{side} coverage of {traced:.0f} ft exceeds the {from_tracing:.0f} ft of traced runs "
            f"it is supposed to be measured over - a modelled kerb is being counted as survey")
        assert traced <= broad_st.length_ft + 1e-6
    assert broad_st.both_traced_ft <= min(broad_st.traced_ft("left"),
                                          broad_st.traced_ft("right")) + 1e-6


@needs_source_data
def test_every_figure_the_report_emits_carries_a_coverage_denominator(corridors, site_models):
    """THE STRUCTURAL GUARANTEE, asserted three ways rather than by reading the printed text.

    A count over the part of a corridor that happens to be surveyed, printed as a fact about the
    corridor, is what produced all three wrong answers recorded in docs/network-model.md. So the
    denominator is not a formatting convention here - it is a field with no default, on a type that
    cannot be constructed without it, checked on every figure of every road.
    """
    coverage_field = next(f for f in dataclasses.fields(Figure) if f.name == "coverage")
    assert coverage_field.default is dataclasses.MISSING, (
        "Figure.coverage has a default, so a figure can be emitted without one")
    assert coverage_field.default_factory is dataclasses.MISSING
    for field in dataclasses.fields(Coverage):
        assert field.default is dataclasses.MISSING, (
            f"Coverage.{field.name} has a default - an omitted denominator is the failure this "
            f"guards against, so none of its fields may be optional")

    seen = 0
    for corridor in corridors:
        report = corridor_report(corridor, corridor_facts(corridor, site_models), site_models)
        assert report.figures, f"{corridor.name} produced no figures at all"
        for figure in report.figures:
            seen += 1
            assert figure.coverage is not None, f"{corridor.name}/{figure.label} has no coverage"
            assert figure.coverage.total_ft > 0, (
                f"{corridor.name}/{figure.label} has a zero denominator, which reads as 0% "
                f"coverage of everything rather than as the length it was measured against")
            assert 0.0 <= figure.coverage.measured_ft <= figure.coverage.total_ft + 1e-6, (
                f"{corridor.name}/{figure.label} claims {figure.coverage.measured_ft:.0f} ft "
                f"measured out of {figure.coverage.total_ft:.0f} ft")
            assert figure.coverage.basis.strip(), (
                f"{corridor.name}/{figure.label}'s coverage does not say what it measured")
            assert str(figure.coverage) in figure.line(), (
                f"{corridor.name}/{figure.label} prints its value without its coverage")
    assert seen >= 4 * len(corridors), "every road should answer every corridor question"


@needs_source_data
def test_the_report_covers_the_questions_the_plan_names(broad_st, site_models):
    """Length, narrowest width, driveway openings, street crossings, parking - all five present.

    Named individually rather than counted, because the point of docs/network-renderer-plan.md's
    stream C is that these particular questions become answerable, and a report that quietly
    dropped one would still pass a "figures exist" test.
    """
    report = corridor_report(broad_st, corridor_facts(broad_st, site_models), site_models)
    labels = " | ".join(figure.label for figure in report.figures)
    for wanted in ("length", "narrowest", "driveway openings", "streets crossing",
                   "marked parking"):
        assert wanted in labels, f"no figure for {wanted!r}; the report has {labels}"


@needs_source_data
def test_parking_capacity_never_exceeds_the_kerb_it_is_counted_over(broad_st, site_models):
    """A stall is 22 ft, so a count times 22 cannot exceed the kerb it was counted over.

    The arithmetic sanity check the scratch scripts did not have. It catches the specific mistake
    of counting a run twice - once per side, or once per overlapping no-parking zone - which is how
    a corridor grows more parking than it has kerb.
    """
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT

    facts = corridor_facts(broad_st, site_models)
    for side in ("left", "right"):
        stalls, measured_ft = marked_parking_capacity(broad_st, facts, side)
        assert stalls * PARKING_STALL_LENGTH_DEFAULT_FT <= measured_ft + 1e-6
        assert measured_ft <= broad_st.length_ft + 1e-6, (
            f"{side} has {measured_ft:.0f} ft of parkable kerb on a {broad_st.length_ft:.0f} ft "
            f"road - the runs overlap")
        tested, tested_ft = marked_parking_capacity(broad_st, facts, side,
                                                   within=broad_st.both_traced_spans())
        assert tested <= stalls and tested_ft <= measured_ft + 1e-6, (
            "the width-tested count is a subset of the length-only count by construction")


@needs_source_data
def test_hatch_bands_are_not_dropped_by_a_few_untraced_samples(broad_st, site_models):
    """A span that is mostly traced must not vanish because a handful of its samples are not.

    `_kerb_band_over` used to require every sampled offset in a span to be finite and drop the
    WHOLE span otherwise - so a kerb-topology seam a few feet wide, inside an otherwise-traced
    500 ft run, discarded the run. With nothing marked, hatch_bands's spans must cover essentially
    all of the kerb outside its openings; the pre-fix code covered only 2,199 of 3,526 ft here.
    """
    facts = corridor_facts(broad_st, site_models)
    for side in ("left", "right"):
        hatch = hatch_bands(broad_st, facts, side, ())
        hatch_ft = sum(hi - lo for lo, hi, _, _ in hatch)
        mouths_ft = sum(opening.end_ft - opening.start_ft
                        for opening_side, opening in facts.openings if opening_side == side)
        expected_ft = broad_st.length_ft - mouths_ft
        # ~2% of Broad St's kerb is genuinely untraced end-to-end (OSM traces the block, not the
        # kerb) and correctly drops out; anything below 90% is the whole-span-discard bug back.
        assert hatch_ft > expected_ft * 0.9, (
            f"{side} kerb: hatch_bands covered {hatch_ft:.0f} of {expected_ft:.0f} ft with "
            f"nothing marked - a partially-traced span is being dropped instead of trimmed")


@needs_source_data
def test_everything_counted_from_osm_is_inside_a_fetch_window(broad_st, site_models):
    """Nothing is counted where nothing was fetched - "unmapped" and "unfetched" are not the same.

    The measurement that made CORRIDOR_KERB_RADIUS_M 400 m rather than the junction fetch's 120 m:
    at 120 m the three circles on Broad St leave 173 m of the Greenwood-to-Louellen block outside
    every window, and an opening count over that road would have covered 80% of it while reading as
    a total. Asserted as a property so a future radius change cannot quietly reintroduce it.
    """
    window = osm_window_spans(broad_st, site_models)
    covered = sum(hi - lo for lo, hi in window)
    assert covered == pytest.approx(broad_st.length_ft, abs=broad_st.length_ft * 0.02), (
        f"only {covered:.0f} of {broad_st.length_ft:.0f} ft of {broad_st.name} is inside an OSM "
        f"fetch window, so its OSM-derived counts describe part of it")
    facts = corridor_facts(broad_st, site_models)
    for _side, opening in facts.openings:
        assert opening.start_ft >= 0.0 and opening.end_ft <= broad_st.length_ft + 1e-6
    for cross in facts.crossings:
        assert 0.0 <= cross.station_ft <= broad_st.length_ft + 1e-6


@needs_source_data
@pytest.mark.parametrize("scale", [None, 2.5])
def test_the_facility_covers_the_street_it_is_drawn_on(scale):
    """A bikeway runs the whole kerb it is placed on, at whatever width the sheet is.

    THIS IS THE PROPERTY THE FRAME-INVARIANT RUNG WAS TRADED FOR, and it is worth more. Sections
    used to be sized over a configured span while being drawn over HOPEWELL_FRAME_SCALE times it,
    so which rung a leg took could not move with the sheet - and the 195 ft nobody had measured
    had its paint trimmed off at the first station the section stopped fitting. broad_st_east
    carried green over 180 ft of a 425 ft leg, under 42 flex posts and a centre stripe that both
    ran the full length, because those are drawn by paths that never consulted the design span.

    Measured off the DRAWN pieces against the traced kerb, which is the comparison
    BikewayReachesTheEndOfItsKerb makes in the pipeline; asserted here at two scales because the
    fixtures build at one, and a single scale is exactly what hid this.
    """
    import numpy as np
    from src.checks import BIKEWAY_SHORTFALL_TOLERANCE_FT
    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.model import curb_station_span, station_offset_many
    from src.render.frame import FRAME_SCALE_ENV
    from scripts.measure_drawn import build

    with contextlib.ExitStack() as stack:
        monkey = pytest.MonkeyPatch()
        stack.callback(monkey.undo)
        if scale is None:
            monkey.delenv(FRAME_SCALE_ENV, raising=False)
        else:
            monkey.setenv(FRAME_SCALE_ENV, str(scale))
        with contextlib.redirect_stdout(io.StringIO()):
            built = build("wbroad_louellen", "build_proposal_two_way_bike_lane")

    reached: dict[tuple[str, str], float] = {}
    for piece in built.paint:
        if piece.kind is not BIKE_LANE_SURFACE or piece.leg is None:
            continue
        leg = built.state.legs[piece.leg]
        coords = np.asarray(piece.geometry.exterior.coords, dtype=float)
        stations, _offsets = station_offset_many(leg.centerline, coords)
        key = (piece.leg, piece.side)
        reached[key] = max(reached.get(key, float("-inf")), float(stations.max()))

    assert reached, "no bikeway surface was drawn, so nothing was measured"
    short = {}
    for (leg_name, side), reached_ft in reached.items():
        span = curb_station_span(built.state.legs[leg_name], side)
        if span is None:
            continue
        if float(span[1]) - reached_ft > BIKEWAY_SHORTFALL_TOLERANCE_FT:
            short[f"{leg_name}/{side}"] = (reached_ft, float(span[1]))
    assert not short, (
        "a bikeway stops before the kerb it was placed on does: "
        + ", ".join(f"{k} reached {a:.1f} of {b:.1f} ft" for k, (a, b) in short.items()))


@needs_source_data
def test_the_design_span_is_the_surveyed_one_not_the_sheets(site_models, wide_site_models):
    """The amount of SURVEYED street a corridor covers is a fact about the survey, not the sheet.

    This is the bug class in one assertion. A facility's rung is chosen over a span
    (corridor_paint._collect takes the governing cross-section of each run), so if the span moves
    with HOPEWELL_FRAME_SCALE then the render viewport is voting on the design - which is how W
    Broad's southwest approach carried a protected lane at 2.5x and nothing at all at 3x.

    The leak was that a corridor's EXTENSIONS were measured from the end of a junction piece, and a
    piece is a frame-cut leg. Both the search window and the fetch-radius cap in _traced_end_ft
    were relative to that moving seam, and the junction centre defining the cap circle was chosen
    by proximity to it - so a wider sheet slid the window outward and discovered street a narrower
    sheet had not looked for. Columbia Ave's traced coverage moved 369 ft between sheets. Anchoring
    on the junction NODE, a surveyed point, makes it 0.7.

    Measured as TRACED coverage rather than raw length, because that is what the design reads. The
    raw length still moves, and deliberately has its own pin below.
    """
    def coverage(models):
        return {road.name: sum(run.end_ft - run.start_ft for run in road.kerb_runs
                               if run.source == KERB_FROM_TRACING)
                for road in corridors_from_models(models)}

    narrow, wide = coverage(site_models), coverage(wide_site_models)
    assert narrow, "no corridor resolved, so nothing was compared"
    assert set(narrow) == set(wide), (
        f"a street is a corridor on one sheet and not the other: "
        f"{sorted(set(narrow) ^ set(wide))}")
    moved = {name: (narrow[name], wide[name]) for name in narrow
             if abs(wide[name] - narrow[name]) > DESIGN_SPAN_TOL * max(narrow[name], 1.0)}
    assert not moved, (
        "the surveyed street a corridor covers moved with the frame scale, so the sheet is "
        "voting on how much street the design is asked about: "
        + ", ".join(f"{name} {a:.1f} -> {b:.1f} ft ({b - a:+.1f})"
                    for name, (a, b) in sorted(moved.items())))


@needs_source_data
@pytest.mark.xfail(strict=True, reason=(
    "A junction piece is a frame-cut leg (intersection/load.py multiplies the surveyed leg length "
    "by frame_scale()), so a wider sheet can push a piece PAST the last traced kerb and the "
    "corridor claims street nothing was surveyed on: Greenwood 1564.9 -> 1698.4 ft, Columbia "
    "1502.4 -> 1698.3 ft. No paint comes of it - corridor_paint refuses an untraced span by name - "
    "but a Coverage denominator that moves with the sheet is still a figure measured over a "
    "viewport. Retires itself: this XPASSes, and so FAILS, the moment leg extent stops being a "
    "render parameter."))
def test_a_corridor_never_claims_more_street_than_the_survey(site_models, wide_site_models):
    """The extent a corridor REPORTS, as opposed to the part of it that is surveyed."""
    narrow = {road.name: road.length_ft for road in corridors_from_models(site_models)}
    wide = {road.name: road.length_ft for road in corridors_from_models(wide_site_models)}
    grew = {name: (narrow[name], wide[name]) for name in narrow
            if abs(wide[name] - narrow[name]) > CORRIDOR_EXTENT_TOL_FT}
    assert not grew, (
        "a corridor's length moved with the frame scale: "
        + ", ".join(f"{name} {a:.1f} -> {b:.1f} ft ({b - a:+.1f})"
                    for name, (a, b) in sorted(grew.items())))


@needs_source_data
def test_a_narrower_rung_never_costs_an_approach_its_protection():
    """What a wider sheet may and may not change about a section.

    MAY change the width. A treatment applies to the street in the drawing, so a sheet that shows
    more street is asking the question of more street: W Broad's southwest approach measures
    20.32 ft over 130 ft and 16.58 ft over 325 ft, off a real pinch 318 ft out, and takes NACTO's
    constrained rung on the sheet that shows it. Both drawings are honest about what they depict -
    the alternative was a section sized on the short span and drawn over the long one, which is
    how the same leg came to carry 63.6 ft of amputated paint.

    MAY NOT change whether it is protected. The buffer is the rung that never gives on this
    facility (see BROAD_ST_TWO_WAY_BIKEWAY), and it is what the flex posts stand in - so a
    painted lane in one picture and a protected one in another is a difference in the PROPOSAL,
    not in the sheet. That is the half of the old frame-invariance assertion worth keeping.

    MAY NOT change whether the approach carries one AT ALL, which is the assertion below that
    two sheets were not enough to catch. This swept 1x and 2.5x, passed both, and the southwest
    approach still went bare at 3x - so the property was pinned at the two scales that happened
    to agree. Three sheets is not a principle either; what makes it enough here is that the third
    is the one the reader asked for, and the mechanism is now measured per station rather than
    per sheet (see AddTwoWayBikeLane.to_ft and TwoWayBikeway._reach_on).

    The 3x failure is worth keeping as a number, because nothing about it looked like a rounding
    problem from the outside: 168 of that leg's 169 sampled stations held the section, station
    363.6 measured 31.813 ft between kerbs against the 31.820 the rung wanted, and the 0.0036 ft
    that left each travel lane short of MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT denied a protected
    bikeway over all 335 traced feet. That is a fortieth of an inch deciding a corridor.
    """
    from src.geometry.intersection import load_intersection_model
    from src.geometry.treatments import (AddBikeLane, BROAD_ST_TWO_WAY_BIKEWAY, DesignState)
    from src.geometry.targets import LegSide
    from src.render.frame import FRAME_SCALE_ENV

    def sections(scale):
        with contextlib.ExitStack() as stack:
            monkey = pytest.MonkeyPatch()
            stack.callback(monkey.undo)
            if scale is None:
                monkey.delenv(FRAME_SCALE_ENV, raising=False)
            else:
                monkey.setenv(FRAME_SCALE_ENV, str(scale))
            with contextlib.redirect_stdout(io.StringIO()):
                model = load_intersection_model(site="wbroad_louellen")
                state = BROAD_ST_TWO_WAY_BIKEWAY.apply_to(DesignState.from_model(model), model,
                                                          quiet=True)
        placed = {}
        for leg_name in BROAD_ST_TWO_WAY_BIKEWAY.legs_on(model):
            for side in ("left", "right"):
                lane = state.treatment_for(AddBikeLane, LegSide(leg_name, side))
                if lane is not None:
                    placed[(leg_name, side)] = (round(lane.width_ft, 2),
                                                round(lane.buffer_ft, 2))
        return placed

    placed_at = {"1x": sections(None), "2.5x": sections(2.5), "3x": sections(3.0)}
    assert all(placed_at.values()), (
        f"a sheet carried the facility on no approach at all, so nothing was compared: "
        f"{ {name: len(placed) for name, placed in placed_at.items()} }")
    carried = {name: set(placed) for name, placed in placed_at.items()}
    assert len(set(map(frozenset, carried.values()))) == 1, (
        "an approach carries the facility on one sheet and not another, so the render viewport "
        "is deciding whether a leg is protected: "
        + "; ".join(f"{name} {sorted(f'{leg}/{side}' for leg, side in legs)}"
                    for name, legs in carried.items()))
    for scale_name, placed in placed_at.items():
        assert all(buffer_ft > 0 for _width, buffer_ft in placed.values()), (
            f"at {scale_name} an approach carries an UNBUFFERED lane, which takes no flex posts - "
            f"the buffer is the rung that never gives on this facility: {placed}")


@needs_source_data
def test_a_station_that_refuses_the_section_costs_the_tail_and_not_the_approach():
    """One station that cannot hold the facility ends it THERE, with the span named.

    THIS IS THE BUG CLASS, AND THE SHAPE OF IT IS THE POINT. A fit judged once for a whole
    approach lets any single station veto every foot of it: 168 of w_broad_st_southwest's 169
    sampled stations held the section on the 3x sheet, station 363.6 was 0.0036 ft per travel lane
    short, and the approach went bare - so the reader got a corridor that arrives at the junction
    and stops, and nothing in the drawing said why. Asked per station instead, the same leg carries
    a protected lane for 283 of its 335 traced feet and REFUSES the tail by name.

    A REFUSAL IS AN OUTPUT, NOT A PRINT STATEMENT, which is the second half of this. The span, the
    reason and the measurement go on the design (DesignState.refuse) because
    BikewayReachesTheEndOfItsKerb is fatal and has to tell a stretch that was measured and
    declined from one the paint quietly stopped on - and it cannot read stdout.

    Driven with a ONE-RUNG ladder, because the real ladder's job is to make this rare: the
    step-down buys length rather than trading it, and on this sheet the full rung stops at 285.8 ft
    where the constrained one carries to 337.6 of the kerb's 54.4-389.9.

    THE LAST ASSERTION HERE USED TO BE THAT THE LADDER COVERED THE WHOLE LEG, which held only
    while the travel way was measured as min(near + far) - a figure that credits the lanes with
    room the drawing spends on bike-lane hatching at the OTHER kerb, see
    governing_half_widths_ft. Measured as drawn, the last 50.2 ft of this 3x leg is 0.06 ft per
    lane under the floor, so the ladder refuses a tail here too - 16.3 ft of that once fitted and
    lost to the same function's second rule, that the far kerb is measured over the whole leg
    because the divider is DRAWN over the whole leg. Nothing in the drawing refuses anything:
    all six corridor approaches take the ladder over their whole kerb at 1x and at 2.5x, pinned in
    test_no_approach_in_the_drawing_carries_a_shortened_facility.
    """
    from src.geometry.model import curb_station_span
    from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY, DesignState
    from src.geometry.treatments.bikeways import (MIN_FACILITY_RUN_FT, MIN_TWO_WAY_BIKE_LANE_FT,
                                                  TWO_WAY_BIKE_LANE_BUFFER_FT)
    from src.geometry.treatments.corridor import CorridorFacility, Section
    from src.geometry.intersection import load_intersection_model
    from src.render.frame import FRAME_SCALE_ENV

    LEG, SIDE = "w_broad_st_southwest", "right"

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv(FRAME_SCALE_ENV, "3.0")
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_intersection_model(site="wbroad_louellen")
    finally:
        monkey.undo()

    span = curb_station_span(model.legs[LEG], SIDE)
    assert span is not None, f"{LEG} {SIDE} has no traced kerb, so there is nothing to reach along"

    # The full rung with no fallback under it, so where the street pinches there is nowhere to go.
    one_rung = CorridorFacility(
        road=BROAD_ST_TWO_WAY_BIKEWAY.road, side=BROAD_ST_TWO_WAY_BIKEWAY.side,
        sections=(Section(MIN_TWO_WAY_BIKE_LANE_FT, TWO_WAY_BIKE_LANE_BUFFER_FT),))
    state = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        reach_ft, refused = one_rung._reach_on(state, LEG, SIDE)

    assert refused is None, f"the whole approach was given up rather than part of it: {refused}"
    assert reach_ft is not None, (
        "the section fits every station of this leg, so no split was measured - this test is "
        "pinning nothing on this data")
    assert float(span[0]) < reach_ft < float(span[1]), (
        f"a reach of {reach_ft} ft is not inside the {span[0]:.1f}-{span[1]:.1f} ft traced kerb")
    assert reach_ft - float(span[0]) >= MIN_FACILITY_RUN_FT, (
        f"a {reach_ft - float(span[0]):.0f} ft facility should have been refused outright, not "
        f"drawn - MIN_FACILITY_RUN_FT is {MIN_FACILITY_RUN_FT:.0f} ft")

    refusals = state.refusals_on(LEG, SIDE)
    assert len(refusals) == 1, f"expected the tail to be refused once, got {refusals}"
    tail = refusals[0]
    assert reach_ft <= tail.start_ft <= float(span[1]), (
        f"the refused span starts at {tail.start_ft:.1f} ft, which is not where the facility "
        f"stopped ({reach_ft:.1f} ft) - a gap between them is ground with neither paint nor a "
        f"reason on it")
    assert tail.end_ft >= float(span[1]) - 1e-6, (
        f"the refusal covers to {tail.end_ft:.1f} ft but the kerb is traced to {span[1]:.1f} ft")
    assert tail.narrowest_ft is not None and tail.narrowest_ft > 0, (
        f"the refusal carries no measurement, so it is an opinion: {tail}")
    assert f"{tail.narrowest_ft:.2f}" in tail.reason, (
        f"the reason should quote the width that stopped it: {tail.reason}")

    # AND THE LADDER BUYS LENGTH, which is what a rung below is for: a narrower section covers at
    # least as much street as the wider one it stands in for, never less. Anything it still cannot
    # reach is refused by name, on the same terms as the tail above.
    laddered = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        ladder_reach, ladder_refused = BROAD_ST_TWO_WAY_BIKEWAY._reach_on(laddered, LEG, SIDE)
    assert ladder_refused is None, (
        f"the ladder gave up the whole approach where a single rung carried {reach_ft:.0f} ft of "
        f"it, so stepping down cost this leg its facility instead of extending it: {ladder_refused}")
    assert ladder_reach is None or ladder_reach > reach_ft, (
        f"the ladder reached {ladder_reach:.1f} ft against the single rung's {reach_ft:.1f} - a "
        f"narrower section cannot cover less street than a wider one, so the rung here was chosen "
        f"by something other than what fits")
    ladder_refusals = laddered.refusals_on(LEG, SIDE)
    if ladder_reach is None:
        assert not ladder_refusals, (
            f"the ladder covered the whole kerb and refused part of it anyway: {ladder_refusals}")
    else:
        assert (len(ladder_refusals) == 1
                and ladder_refusals[0].end_ft >= float(span[1]) - 1e-6), (
            f"the ladder stopped at {ladder_reach:.1f} ft of a kerb traced to {span[1]:.1f} ft and "
            f"the bare tail is not covered by one refusal: {ladder_refusals}")


@needs_source_data
def test_no_approach_in_the_drawing_carries_a_shortened_facility(site_models, wide_site_models):
    """A tail refusal is a legal output, and on the sheets the drawing uses there are none.

    The reach is a measurement, so it CAN come in short, and TwoWayBikeway._reach_on records the
    span, the reason and the width when it does - because BikewayReachesTheEndOfItsKerb is fatal
    and has to tell a stretch that was measured and declined from one the paint quietly stopped
    on. That is the escape valve; this is what stops it becoming a licence. An extent that moves
    is the bug class (SKILLS 0b), so the published claim is that every approach carrying the
    facility carries it to the end of its traced kerb, asserted per approach rather than as one
    number for the borough.

    HAS TEETH ON REAL DATA: the same sweep at 3x reports w_broad_st_southwest reaching 371.5 ft of
    a kerb traced to 389.9, the pinch
    test_a_station_that_refuses_the_section_costs_the_tail_and_not_the_approach is built on. So
    this is a property of these two sheets, not of arithmetic that cannot fail.

    BOTH FIXTURES, because a leg is 2.5x longer on the wide one and that is 2.5x as much street to
    find a pinch in - W Broad's southwest approach reaches its 16.58 ft pinch only there.
    """
    from src.geometry.model import curb_station_span, side_facing
    from src.geometry.treatments import BROAD_ST_TWO_WAY_BIKEWAY, DesignState

    carrying, swept, short = 0, 0, []
    for sheet, models in (("1x", site_models), (f"{WIDE_FRAME_SCALE}x", wide_site_models)):
        for site, model in sorted(models.items()):
            for leg_name in BROAD_ST_TWO_WAY_BIKEWAY.legs_on(model):
                try:
                    side = side_facing(model.legs[leg_name], BROAD_ST_TWO_WAY_BIKEWAY.side)
                except ValueError:
                    continue        # a stem meeting the route square on has no kerb on this side
                swept += 1
                state = DesignState.from_model(model)
                with contextlib.redirect_stdout(io.StringIO()):
                    reach_ft, refused = BROAD_ST_TWO_WAY_BIKEWAY._reach_on(state, leg_name, side)
                if refused is not None:
                    continue        # carries nothing here, and says why - a different property
                carrying += 1
                if reach_ft is None:
                    continue        # the whole traced kerb, which is the answer being asserted
                span = curb_station_span(model.legs[leg_name], side)
                end_ft = reach_ft if span is None else float(span[1])
                short.append(f"{sheet} {site} {leg_name} {side}: reaches {reach_ft:.1f} ft of a "
                             f"kerb traced to {end_ft:.1f} ft")
    assert carrying >= 12, (
        f"only {carrying} of {swept} corridor approaches carried the facility at all, so "
        f"'none of them stops short' is measuring almost nothing")
    assert not short, ("an approach in the drawing carries a facility that stops short of its own "
                       "kerb:\n  " + "\n  ".join(short))
