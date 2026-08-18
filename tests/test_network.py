"""THE CHECKPOINT for docs/network-model.md: does a road reproduce the per-leg model?

Nothing renders from src/geometry/network.py yet. Its only purpose is to answer, cheaply and before
14k lines are rewritten, whether reading the traced kerbs as ONE CONTINUOUS ROAD gives the same
geometry the two legs give separately. If it does, the frame can move onto roads. If it does not,
the disagreement is the finding, and the migration should stop until it is understood.

So these are not tests of a feature. They are the measurement the plan is gated on, written down so
it re-runs on every change to either model rather than being taken on faith from one session.
"""
import numpy as np
import pytest

from src.geometry.model import (curb_offsets_at_stations, curb_station_span,
                                narrowest_half_width_ft)
from src.geometry.network import Road, road_station_of_leg_station, roads_from_model
from tests.conftest import SITES, needs_source_data

# Feet. The road resamples the same traced kerbs the leg does, so the two should agree to well
# inside a stripe's width; this is loose enough to absorb resampling and tight enough that a real
# disagreement - a mispaired side, a station axis that runs the wrong way - cannot hide under it.
AGREEMENT_TOL_FT = 0.5


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_every_through_street_becomes_one_road(site, site_models):
    """A junction's through streets each collapse to a single road with the node inside it.

    W Broad & Louellen and E Broad & Princeton are T-junctions with one through street; Broad &
    Greenwood and Columbia & Princeton are four-way and have two.
    """
    model = site_models[site]
    roads = roads_from_model(model)
    assert roads, f"{site} has no through street, which no site in this project should"
    for road in roads:
        assert isinstance(road, Road)
        assert road.near_leg != road.far_leg
        assert 0 < road.node_ft < road.length_ft, (
            f"{road.name}: the junction sits at station {road.node_ft:.1f} of a "
            f"{road.length_ft:.1f} ft road - it has to be INSIDE it, or the two legs were joined "
            f"end-to-end instead of head-to-head")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_road_reproduces_each_legs_measured_width(site, site_models):
    """THE CHECKPOINT ITSELF. Read as one road, the traced kerbs give each leg's width back.

    Sampled at stations translated through road_station_of_leg_station, so this also exercises the
    axis translation the whole migration turns on: a station that means one place on the leg has to
    mean the same place on the road.

    Legs whose kerbs are only partly traced return None for the width rather than a guess, and are
    skipped here - the point is agreement where both models can answer, not coverage.
    """
    model = site_models[site]
    disagreements = []
    compared = 0
    for road in roads_from_model(model):
        for leg_name in (road.near_leg, road.far_leg):
            leg = model.legs[leg_name]
            if leg.curb_to_curb_ft is None:
                continue
            for leg_station in np.linspace(5.0, max(leg.centerline.length - 5.0, 6.0), 8):
                station = road_station_of_leg_station(road, leg_name, float(leg_station))
                width = road.width_at_ft(station)
                if width is None:
                    continue
                compared += 1
                # Against the leg's own traced cross-section at that station, not against
                # curb_to_curb_ft: that figure is ONE number for the whole leg (the narrowest, or a
                # field measurement), and the two models are expected to differ from it in the same
                # way. What must agree is the reading at a given station.
                leg_width = _leg_width_at(leg, float(leg_station))
                if leg_width is None:
                    continue
                if abs(width - leg_width) > AGREEMENT_TOL_FT:
                    disagreements.append(
                        f"{road.name} @ road station {station:.1f} ({leg_name} @ "
                        f"{leg_station:.1f}): road says {width:.2f} ft, leg says {leg_width:.2f} ft")
    assert compared, f"{site}: nothing could be compared - both models declined every station"
    assert not disagreements, (
        f"{len(disagreements)} of {compared} station(s) disagree by more than "
        f"{AGREEMENT_TOL_FT} ft:\n  " + "\n  ".join(disagreements[:8]))


def _leg_width_at(leg, station_ft: float) -> float | None:
    """The leg's own kerb-to-kerb reading at one station, or None where either side is untraced."""
    from src.geometry.model import curb_offsets_at_stations, curb_station_span

    total = 0.0
    for side in ("left", "right"):
        span = curb_station_span(leg, side)
        if span is None or not (span[0] <= station_ft <= span[1]):
            return None
        at = curb_offsets_at_stations(leg, side, np.array([station_ft]))
        if at is None or at[0] is None or not np.isfinite(at[0]):
            return None
        total += abs(float(at[0]))
    return total


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_road_axis_runs_one_way_through_the_junction(site, site_models):
    """Stations increase monotonically along the joined centreline.

    The two legs are joined head-to-head, so a sign error would make the axis fold back on itself at
    the node - and every station-based clip downstream assumes it does not. Same property
    tests/test_leg_frame.py pins for a placed stripe, asserted here for the road's own axis.
    """
    from shapely.geometry import Point

    for road in roads_from_model(site_models[site]):
        stations = [road.centerline.project(Point(c)) for c in road.centerline.coords]
        backwards = [(stations[i - 1], stations[i]) for i in range(1, len(stations))
                     if stations[i] < stations[i - 1] - 1e-6]
        assert not backwards, (
            f"{road.name}'s axis doubles back, e.g. {backwards[0][0]:.2f} -> "
            f"{backwards[0][1]:.2f} ft - the near leg was probably not reversed")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_leg_station_maps_to_the_node_at_zero(site, site_models):
    """Both legs' station 0 is the junction, so both map to the road's node station."""
    for road in roads_from_model(site_models[site]):
        for leg_name in (road.near_leg, road.far_leg):
            assert road_station_of_leg_station(road, leg_name, 0.0) == pytest.approx(road.node_ft)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_asking_for_a_leg_that_is_not_on_the_road_raises(site, site_models):
    """A stem leg is not on the through road, and quietly returning a station for it would put a
    marking on the wrong street."""
    model = site_models[site]
    for road in roads_from_model(model):
        others = set(model.legs) - {road.near_leg, road.far_leg}
        for other in others:
            with pytest.raises(KeyError, match=other):
                road_station_of_leg_station(road, other, 10.0)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_width_is_refused_outside_the_traced_span_rather_than_extrapolated(site, site_models):
    """No width is reported past the end of the tracing, however far past it you ask.

    The property `Corridor` depends on and `Road` has always had, pinned here because the two now
    share one implementation (network._kerb_offset_at). np.interp holds the first and last traced
    offset flat forever, so without the span test a road would answer a mile out with the width it
    last measured - and on a corridor that is what turns 1,126 ft of unmapped street into a
    confident cross-section.
    """
    for road in roads_from_model(site_models[site]):
        for station in (-500.0, -1.0e4, road.length_ft + 500.0, road.length_ft + 1.0e4):
            assert road.width_at_ft(station) is None, (
                f"{road.name} reports a width at station {station:.0f} on a "
                f"{road.length_ft:.0f} ft road, which is outside anything anybody traced")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_road_goes_through_the_frame_functions_unmodified(site, site_models):
    """The frame takes a Road with no shim, no adapter and no second code path.

    STEP 4 of docs/network-model.md turns on this: `curb_offsets_at_stations` and friends read
    `.centerline` and one of `.left_curb`/`.right_curb` and nothing else, so they were never about
    legs. If a Road has to be wrapped to get through them, then moving the datum means rewriting
    the frame; if it does not, it means changing the callers, which is a far smaller and far more
    reversible thing.

    Called positionally and through the package's own public names, exactly as every leg caller
    does - the point is that these are THE SAME functions, not road-shaped copies of them.
    """
    for road in roads_from_model(site_models[site]):
        for side in ("left", "right"):
            if getattr(road, f"{side}_curb") is None:
                continue
            span = curb_station_span(road, side)
            assert span is not None, f"{road.name} {side}: a traced kerb with no station span"
            mid = (span[0] + span[1]) / 2
            offsets = curb_offsets_at_stations(road, side, np.array([mid]))
            assert offsets is not None and np.isfinite(offsets[0]), (
                f"{road.name} {side}: the frame could not read the kerb it just gave a span for")
            # Signed by side, the same convention a leg's kerb reads in: left is +offset.
            assert (offsets[0] > 0) == (side == "left")
            assert narrowest_half_width_ft(road, side, span[0], span[1]) > 0
