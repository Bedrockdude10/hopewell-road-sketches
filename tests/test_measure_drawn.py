"""What scripts/measure_drawn.py measures, which is what this repo diagnoses from.

The tool is the quantitative layer SKILLS.md 0a says answer every geometry complaint at, so a
wrong number here is worse than a wrong number in a render: it is a wrong number that arrives
wearing the authority of a measurement. `--gaps` spent its life reporting up to 17.54 ft of
separation between paint and kerb on paint drawn flush, because it reduced over a polygon's
VERTICES and a fill inherits its outer edge - and only its outer edge's sparse vertices - from
the traced kerb. These pin the measurement, not the printing.
"""
import contextlib
import io

import numpy as np
import pytest

from conftest import needs_source_data

CORNER_END_FT = 50.0
"""Past the corner return on both Greenwood kerbs, where the hatch runs beside its kerb.

The return itself IS bare - the kerb flares to 21.12 ft off the alignment by station 33 on the
east while the hatch starts at 44.78 - so a profile that reported nothing anywhere would have
stopped measuring rather than started being right.
"""


@pytest.fixture(scope="module")
def greenwood_two_way():
    from scripts.measure_drawn import build

    with contextlib.redirect_stdout(io.StringIO()):
        return build("broad_st_greenwood", "build_proposal_two_way_bike_lane")


@needs_source_data
def test_the_gap_profile_reads_flush_where_the_hatch_follows_its_kerb(greenwood_two_way):
    """Both Greenwood kerbs, past the corner, measured 0.04 ft from their hatch and reported 4.06.

    A LaneNarrowing fill is offset from the traced kerb, so its outer edge carries the kerb's
    vertices and nothing between them - 9 over 85 ft against 44 on the inner edge, which is a
    plain offset from the alignment. Every 10 ft bin from station 54.5 out therefore held inner
    vertices and no outer one, and the reduction returned the INNER edge at 11.82 ft: station 120
    reported 15.88 - 11.82 = 4.06 ft of separation on paint that touches.
    """
    from scripts.measure_drawn import gap_profile
    from src.geometry.targets import BOTH_SIDES

    for side in BOTH_SIDES:
        edges, gap, why = gap_profile(greenwood_two_way, "greenwood_ave_south", side, 10.0)
        assert why is None, f"{side.value}: {why}"
        along = np.isfinite(gap) & (edges >= CORNER_END_FT)
        assert along.any(), f"{side.value}: nothing measured past the corner"
        worst = int(np.nanargmax(np.where(along, gap, -np.inf)))
        assert gap[worst] <= 0.25, (
            f"greenwood_ave_south {side.value}: the hatch reads {gap[worst]:.2f} ft off its kerb "
            f"at station {edges[worst]:.0f}, and a perpendicular cut through the drawn fill "
            f"there puts it flush")


@needs_source_data
def test_the_gap_profile_still_sees_the_bare_corner_return(greenwood_two_way):
    """And the fix did not buy that by blinding the measurement.

    A check that cannot fail proves nothing, and one that reports nothing anywhere is the same
    thing wearing an all-clear. The corner return is real bare pavement - the kerb curves out to
    21.12 ft off the alignment by station 33 while the hatch is held back to 44.78 by the corner
    clearance - so the profile has to still say so.
    """
    from scripts.measure_drawn import gap_profile
    from src.geometry.targets import Side

    edges, gap, why = gap_profile(greenwood_two_way, "greenwood_ave_south", Side("left"), 10.0)
    assert why is None
    corner = np.isfinite(gap) & (edges < CORNER_END_FT)
    assert np.nanmax(gap[corner]) > 1.0, (
        "the corner return reads flush, so the profile has stopped measuring rather than "
        "started being right")


@needs_source_data
def test_the_reach_is_the_paint_and_not_its_nearest_vertex(greenwood_two_way):
    """surface_reach_at, station by station, against the vertex reduction it replaced.

    The direct form of the defect: between two outer-edge vertices the vertex maximum falls back
    onto the fill's inner edge, a fixed 11.82 ft from the alignment, while the surface is out at
    the kerb. Asserted as a spread rather than a single station because which bins are starved
    depends on where the OSM trace happens to carry a node.
    """
    from scripts.measure_drawn import piece_coords, surface_reach_at
    from src.geometry.model import station_offset_many
    from src.geometry.targets import Side

    leg = greenwood_two_way.model.legs["greenwood_ave_south"]
    fills = [p for p in greenwood_two_way.paint
             if p.leg == "greenwood_ave_south" and str(p.side) == "left"
             and p.kind.name == "lane_narrowing_fill"]
    assert fills, "no hatch on this kerb to measure"
    _stations, offsets = station_offset_many(leg.centerline, piece_coords(fills[0].geometry))
    outer = np.abs(offsets) > np.abs(offsets).mean()
    assert outer.sum() * 3 < (~outer).sum(), (
        f"the outer edge carries {int(outer.sum())} of {len(offsets)} vertices, so this leg no "
        f"longer exhibits the sparsity the reduction fell through")

    sample = np.arange(60.0, 130.0, 5.0)
    reach = surface_reach_at(greenwood_two_way, "greenwood_ave_south", Side("left"), sample)
    assert np.isfinite(reach).all(), "the cut missed a fill it crosses"
    assert reach.min() > np.abs(offsets[~outer]).max() + 1.0, (
        f"the reach bottoms out at {reach.min():.2f} ft, at or inside the fill's inner edge - "
        f"the measurement is still reading the wrong edge")
