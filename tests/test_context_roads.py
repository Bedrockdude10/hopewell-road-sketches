"""The streets around the junction, and how much of each one was actually measured.

The bug these are written against: a wide render showed a cross of asphalt floating on grass,
because the exported kerbs were the corner-fit's NEAR set (within 80 ft of the centre) and
8,938 ft of traced kerb within 600 m of Broad & Greenwood went on the floor. So the two things
pinned here are that tracing REACHES the drawing, and that a street reports honestly whether
its edges were traced or guessed.

Synthetic geometry throughout, apart from the marked site tests - a straight street with kerbs
put exactly where the test wants them is the only way to assert a measured width is the width
that was traced.
"""
import numpy as np
import pytest
from shapely.geometry import LineString

from src.geometry.context_roads import (MIN_TRACED_FRACTION, ROADWAY_DEFAULT_WIDTH_FT,
                                        assign_kerbs_to_roads, assumed_width_ft, is_carriageway,
                                        kerb_points, roadway_surface)
from tests.conftest import SITES, needs_source_data


def straight_street(length_ft=400.0):
    return LineString([(0.0, 0.0), (length_ft, 0.0)])


def kerb_along(y_ft, x0=0.0, x1=400.0):
    """One traced kerb, parallel to `straight_street` at `y_ft`, as a TWO-VERTEX way.

    Two vertices on purpose. That is how a straight kerb is really mapped - the borough's ways
    near Broad & Greenwood carry about three apiece - and reading vertices instead of resampling
    the line is what made a fully traced street report as unmapped.
    """
    return LineString([(x0, y_ft), (x1, y_ft)])


def measure(street, kerbs, tags=None):
    stations, offsets = assign_kerbs_to_roads([street], kerb_points(kerbs))[0]
    return roadway_surface(street, stations, offsets, tags or {"highway": "residential"})


def test_a_street_traced_on_both_sides_is_as_wide_as_the_tracing():
    """The number that matters: 44 ft of traced street measures 44 ft, not the class assumption.

    Deliberately nothing like ROADWAY_DRAWN_WIDTH_FT["residential"] (26 ft), so a surface built
    from the assumption instead of the kerbs cannot pass by coincidence.
    """
    surface, traced, _assumed = measure(straight_street(), [kerb_along(30.0), kerb_along(-14.0)])
    assert traced == {"left", "right"}
    assert surface.area / 400.0 == pytest.approx(44.0, abs=1.0)


def test_a_two_vertex_kerb_still_measures_the_whole_run():
    """A straight kerb is mapped with two vertices; sampling THOSE finds nothing in between.

    This is the bug that made West Broad Street - traced end to end - come back untraced: the
    kerb was read at its corners, so 28 of 82 stations found a vertex nearby and the street was
    reported unmapped. kerb_points resamples the line, so coverage is the run, not the corners.
    """
    kerb = kerb_along(20.0)
    assert len(kerb.coords) == 2
    assert len(kerb_points([kerb])) > 20, "a 400 ft kerb resampled to fewer than 20 points"
    _surface, traced, _assumed = measure(straight_street(), [kerb, kerb_along(-20.0)])
    assert traced == {"left", "right"}


def test_a_street_nobody_traced_falls_back_to_its_highway_class():
    surface, traced, assumed = measure(straight_street(), [], tags={"highway": "residential"})
    assert traced == set()
    assert assumed == 26.0
    assert surface.area / 400.0 == pytest.approx(26.0, abs=0.5)


def test_one_traced_kerb_places_the_far_edge_off_the_kerb_not_off_the_centreline():
    """With one side mapped, the traced kerb is used where it is and the width is assumed.

    Measured off the TRACED EDGE rather than as a half width either side of the alignment,
    because an OSM centreline is not a carriageway centre - the legs that do have both kerbs
    traced sit 0.2-10.3 ft off. Here the kerb is 40 ft out on a 26 ft class: a symmetric guess
    would put the street between -13 and +13 and miss the one edge that was actually surveyed.
    """
    surface, traced, _assumed = measure(straight_street(), [kerb_along(40.0)],
                                        tags={"highway": "residential"})
    assert traced == {"left"}
    ymin, ymax = surface.bounds[1], surface.bounds[3]
    assert ymax == pytest.approx(40.0, abs=1.0), "the traced kerb moved"
    assert ymin == pytest.approx(14.0, abs=1.0), "the far edge is not one assumed width off it"


def test_partial_tracing_is_used_but_not_called_surveyed():
    """A third of a street is a measurement worth keeping and not a survey. Both halves matter.

    Discarding it would throw away the only evidence of how wide the street is; calling it
    surveyed would claim the untraced two-thirds were measured. So the offsets are used and the
    flag says no - which is what decides solid against dashed in the plan view.
    """
    part = 400.0 * (MIN_TRACED_FRACTION - 0.35)
    surface, traced, _assumed = measure(straight_street(),
                                        [kerb_along(21.0, 0.0, part), kerb_along(-21.0, 0.0, part)])
    assert traced == set(), "a third of a street should not report as surveyed"
    assert surface.area / 400.0 == pytest.approx(42.0, abs=2.0), (
        "the traced offsets were discarded instead of carried across the untraced stretch")


def test_a_kerb_goes_to_the_nearest_street_not_to_every_street_in_reach():
    """Two parallel streets 60 ft apart must not both claim the kerb between them.

    Without this, each widens to swallow the other's edge and they meet in the middle - the same
    failure _runs_along_a_leg exists to prevent one layer down.
    """
    near, far = straight_street(), LineString([(0.0, 60.0), (400.0, 60.0)])
    shared = kerb_along(18.0)
    (near_st, _near_off), (far_st, _far_off) = assign_kerbs_to_roads([near, far], kerb_points([shared]))
    assert len(near_st) > 0, "the nearer street did not claim the kerb beside it"
    assert len(far_st) == 0, "a street 42 ft away also claimed it"


def test_a_kerb_beyond_a_streets_own_ends_is_not_its_kerb():
    """Station outside [0, length] means the vertex is past what this way covers.

    Perpendicular distance alone would let a way claim kerb off the end of itself, which at a
    junction is the cross street's return.
    """
    street = LineString([(0.0, 0.0), (100.0, 0.0)])
    beyond = LineString([(400.0, 5.0), (500.0, 5.0)])
    (stations, _offsets), = assign_kerbs_to_roads([street], kerb_points([beyond]))
    assert len(stations) == 0


def test_a_footway_is_not_a_carriageway_and_a_driveway_is_not_drawn_twice():
    """A sidewalk has its own layer, and a driveway is already a PavedSurface.

    Drawing either here puts two coplanar asphalt polygons at the same height, which in Blender
    is z-fighting rather than redundancy.
    """
    assert is_carriageway({"highway": "residential"})
    assert not is_carriageway({"highway": "footway"})
    assert not is_carriageway({"highway": "service", "service": "driveway"})
    assert not is_carriageway({"highway": "service", "service": "parking_aisle"})
    assert is_carriageway({"highway": "service"}), "an ordinary service road IS carriageway"


def test_a_mappers_width_tag_beats_the_class_assumption():
    """Somebody who recorded a width measured something; a class table did not."""
    assert assumed_width_ft({"highway": "residential", "width": "9"}) == pytest.approx(29.5, abs=0.5)
    assert assumed_width_ft({"highway": "residential", "width": "nonsense"}) == 26.0
    assert assumed_width_ft({"highway": "primary_link"}) == 40.0, "a link takes its parent class"
    assert assumed_width_ft({"highway": "unheard_of"}) == ROADWAY_DEFAULT_WIDTH_FT


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_traced_corridor_reaches_the_drawing(site, site_models):
    """The regression itself: kerb past the junction has to survive into what gets rendered.

    Before this, the renderers took kerb_lines_with_tags_ft's default - the near set, 80 ft of
    the centre - and a corridor traced for hundreds of feet arrived as four corner returns. The
    assertion is deliberately about REACH rather than a count: what was wrong was not how many
    kerbs there were but how far out they went.
    """
    from src.geometry.intersection import (KERB_NEAR_JUNCTION_FT, drawn_kerb_radius_ft,
                                            kerb_lines_with_tags_ft)

    model = site_models[site]
    radius_ft = drawn_kerb_radius_ft()
    drawn = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft, radius_ft=radius_ft)
    near = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft)
    assert len(drawn) > len(near), "the drawing test found no kerb the corner fit was dropping"
    reach = max((line.distance(model.center_ft) for line, _t, _w in drawn), default=0.0)
    assert reach > KERB_NEAR_JUNCTION_FT, (
        f"{site}: every drawn kerb is still inside the {KERB_NEAR_JUNCTION_FT} ft near set, so "
        f"the render has nothing past the junction to draw")
    assert all(line.distance(model.center_ft) <= radius_ft for line, _t, _w in drawn), (
        "a kerb beyond the fetched radius is being drawn")


@needs_source_data
def test_a_surveyed_corridor_keeps_its_kerbs_the_whole_way(site_models):
    """A street reported as traced on both sides has kerb beside it along its whole length.

    The failure this is written against: the export clipped kerbs to the FRAME radius while the
    roads were built to the context radius, so the 3D render drew street out to 938 ft and kerb
    to 379 - the corridor lost its edges partway along and nothing said so. Asserted as coverage
    ALONG the street rather than as two reach numbers, because a ribbon extends half a width past
    its own clipped centreline and comparing extents just measures that overhang.
    """
    from shapely.ops import unary_union

    from src.geometry.context_roads import MAX_HALF_WIDTH_FT
    from src.geometry.intersection import drawn_kerb_radius_ft, kerb_lines_with_tags_ft

    checked = 0
    for site, model in site_models.items():
        drawn = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                         radius_ft=drawn_kerb_radius_ft())
        surveyed = [p for p in model.paved_surfaces
                    if str(p.kind) == "roadway" and p.extent_is_surveyed and p.line.length > 100]
        if not surveyed or not drawn:
            continue
        kerbs = unary_union([line for line, _t, _w in drawn])
        for road in surveyed:
            checked += 1
            n = max(int(road.line.length // 25), 2)
            bare = [s for s in np.linspace(0, road.line.length, n + 1)
                    if road.line.interpolate(float(s)).distance(kerbs) > MAX_HALF_WIDTH_FT]
            assert not bare, (
                f"{site}: {road.tags.get('name', '?')} is reported traced on both sides, but "
                f"{len(bare)} of {n + 1} stations along it have no kerb within "
                f"{MAX_HALF_WIDTH_FT:.0f} ft - the drawn street outruns its own edges")
    assert checked, "no surveyed corridor at any site, so this asserted nothing"
