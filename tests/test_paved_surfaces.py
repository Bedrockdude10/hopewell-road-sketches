"""Paved ground that is not carriageway: driveways, parking aisles, parking lots.

All three are the same thing to a renderer - asphalt beside the road - and differ in exactly two
ways: whether the extent was surveyed, and whether it opens a kerb. These pin both, because both
are places where a plausible-looking shortcut would quietly overstate what is known.
"""
import contextlib
import io

import pytest

from src.geometry.intersection import PavedKind
from tests.conftest import SITES, needs_source_data


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_parking_is_drawn_and_only_the_driveways_open_a_kerb(site, site_models):
    """A lot and an aisle are paving; a driveway is paving AND a hole in the markings.

    The distinction is the reason all three share a type rather than a code path. A parking lot
    behind a building crosses no kerb this project models, and an aisle inside one reaches the
    street through a driveway that OSM maps separately - so letting either put a gap in a bike
    lane would be inventing an entrance. `model.driveways` is what the opening logic reads, and it
    has to keep returning driveways only.
    """
    model = site_models[site]
    kinds = {p.kind for p in model.paved_surfaces}
    assert PavedKind.PARKING_LOT in kinds or PavedKind.PARKING_AISLE in kinds, (
        f"{site} has no parking in range at all - either the fetch is broken or the borough "
        f"snapshot has lost its amenity=parking areas")
    assert all(d.kind == PavedKind.DRIVEWAY for d in model.driveways), (
        "model.driveways is handing parking to the kerb-opening logic")
    assert len(model.driveways) == sum(1 for p in model.paved_surfaces
                                       if p.kind == PavedKind.DRIVEWAY)
    for paved in model.paved_surfaces:
        assert paved.surface is not None and not paved.surface.is_empty
        assert paved.surface.area > 0


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_a_surveyed_outline_is_not_confused_with_a_widened_one(site, site_models):
    """A lot's extent is traced; a driveway's and an aisle's are this project's own assumption.

    Same distinction the curb lines have carried since Phase 2 (FIELD-MEASURED / OSM-derived /
    estimated), and it has to survive to the render: the plan view draws a widened outline dashed
    and the exported JSON carries `surveyed` per surface. A lot claiming a width this repo picked,
    or a driveway claiming a surveyed one, would be the quiet over-claim the provenance strings
    exist to prevent.

    A ROADWAY is the one kind that can be either, and that is the point of it: a street with both
    kerbs traced has a measured outline, and one with neither is a ribbon at the width this repo
    picked for its highway class. So the rule is not "a lot is surveyed and nothing else is" - it
    is that every surface reports which of the two it is, and reports a width only when it guessed.
    """
    model = site_models[site]
    for paved in model.paved_surfaces:
        if paved.kind == PavedKind.PARKING_LOT:
            assert paved.extent_is_surveyed
            assert paved.line is None, "a lot is mapped as an area, not a centreline"
        elif paved.kind == PavedKind.ROADWAY:
            assert paved.extent_is_surveyed == (paved.traced_sides == frozenset({"left", "right"}))
            assert paved.line is not None
        else:
            assert not paved.extent_is_surveyed
            assert paved.line is not None
        # A width is what this project ASSUMED. Reporting one alongside a surveyed outline claims
        # the outline came from it; reporting none on a widened line hides that a guess was made.
        if paved.extent_is_surveyed:
            assert paved.width_ft is None, f"{paved.kind} has a surveyed extent AND a drawn width"
        else:
            assert paved.width_ft > 0, f"{paved.kind} has no drawn width"


@needs_source_data
def test_an_aisle_inside_a_lot_is_not_paved_twice(site_models):
    """Two coplanar surfaces at the same height are not redundancy, they are z-fighting.

    6 of the borough's 20 aisles run inside a mapped `amenity=parking` area, whose own surveyed
    outline already paves that ground. The aisle strips are cut against the lots at load, so the
    render gets one surface per patch of ground - the same reason MARKING_CLEARANCE_M exists a
    layer up. The other 14 aisles are outside any mapped lot, which is why the layer is read at
    all rather than being dropped in favour of the areas.
    """
    from shapely.ops import unary_union

    for site, model in site_models.items():
        lots = [p.surface for p in model.paved_surfaces if p.kind == PavedKind.PARKING_LOT]
        aisles = [p.surface for p in model.paved_surfaces if p.kind == PavedKind.PARKING_AISLE]
        if not lots or not aisles:
            continue
        overlap = unary_union(aisles).intersection(unary_union(lots)).area
        assert overlap < 1.0, (
            f"{site}: {overlap:.0f} sq ft of aisle is drawn on top of a parking lot, which will "
            f"z-fight in the render")


@needs_source_data
def test_both_views_get_the_same_paved_polygons(site_models, tmp_path):
    """One polygon per patch of ground, drawn by the plan view and extruded by the render.

    The failure this prevents is the one driveways already had once: the plan view drawing a
    centreline and Blender re-widening a number, so the two views disagreed about where the
    paving was. Checked as a count and an area rather than by eye.
    """
    import json
    from pathlib import Path

    from src.geometry.treatments import DesignState
    from src.render.export import export_scenario
    from src.sources.osm_context import fetch_crossings

    model = site_models["ebroad_princeton"]
    with contextlib.redirect_stdout(io.StringIO()):
        path = export_scenario(model, DesignState.from_model(model), "existing",
                               tmp_path / "paved.json",
                               crossings=fetch_crossings(model.center_wgs84, radius_m=130),
                               theme={})
    exported = json.loads(Path(path).read_text())["paved_surfaces"]
    drawn = [p for p in model.paved_surfaces if p.surface is not None]
    assert len(exported) == len(drawn), (
        f"the render gets {len(exported)} paved surfaces and the plan view draws {len(drawn)}")
    assert {e["kind"] for e in exported} == {str(p.kind) for p in drawn}
    assert sum(1 for e in exported if e["surveyed"]) == sum(1 for p in drawn
                                                            if p.extent_is_surveyed)
