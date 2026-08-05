"""Building heights from the assessor's storey count, rather than one number for the whole town.

OSM gave this project real building outlines and no heights at all - 0 of the borough's 1150
building ways carry `height`, 7 carry `building:levels` - so every building was extruded to the
same 7 m default. The MOD-IV tax list has been in data/ the whole time with a storey count per
parcel, described in the README as "joinable by PIN, not currently used".
"""
import contextlib
import io

import pytest

from src.sources.assessor import (SOURCE_ASSESSOR, SOURCE_ASSUMED, storeys_from_description)
from tests.conftest import SITES, needs_source_data


def test_the_assessors_shorthand_is_read_as_the_assessor_writes_it():
    """Real BLDG_DESC codes out of Hopewell's own records.

    The storey count is anchored on the S because the letters after it are construction type and
    outbuildings: `2SF` is a two-storey frame, `1.5SF 1G` a storey-and-a-half with a one-car
    garage, `B2S` a two-storey over a basement. A code with no S describes no storeyed building -
    `2G` is a detached two-car garage, and reading its 2 as two storeys would put a house on a
    parcel that has none.
    """
    assert storeys_from_description("2SF") == 2.0
    assert storeys_from_description("2SF 1G") == 2.0
    assert storeys_from_description("1.5SF 1G") == 1.5
    assert storeys_from_description("B2S") == 2.0
    assert storeys_from_description("2SFWUG") == 2.0
    assert storeys_from_description("2.5SF") == 2.5
    # Outbuildings only, and the empties.
    assert storeys_from_description("2G") is None
    assert storeys_from_description("1CG") is None
    assert storeys_from_description("F") is None
    assert storeys_from_description("") is None
    assert storeys_from_description(None) is None


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_buildings_are_as_tall_as_the_records_say(site, site_models, tmp_path):
    """Most buildings get a real height, and the heights differ from each other.

    Both halves matter. The first is the join working at all; the second is the point of it - the
    previous behaviour passed any "most buildings have a height" test trivially, because every
    building had the same one. Measured here: 283 of 307 buildings across the four sites take the
    assessor's storey count, and a street of 1, 1.5, 2, 2.5 and 3 storey houses stops rendering as
    one row of identical boxes.
    """
    import json
    from pathlib import Path

    from src.geometry.treatments import DesignState
    from src.render.export import export_scenario
    from src.sources.osm_context import DEFAULT_BUILDING_HEIGHT_M, fetch_crossings

    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        path = export_scenario(model, DesignState.from_model(model), "existing",
                               tmp_path / f"{site}.json",
                               crossings=fetch_crossings(model.center_wgs84, radius_m=130),
                               theme={})
    buildings = json.loads(Path(path).read_text())["buildings"]
    assert buildings, f"{site} exported no buildings"

    from_a_record = [b for b in buildings if b["height_source"] != SOURCE_ASSUMED]
    assert len(from_a_record) > 0.75 * len(buildings), (
        f"{site}: only {len(from_a_record)}/{len(buildings)} buildings have a height anybody "
        f"recorded - the parcel join is not finding the tax rows")
    assert any(b["height_source"] == SOURCE_ASSESSOR for b in buildings), (
        f"{site}: no height came from the assessor, so the MOD-IV join is doing nothing")

    tops = {round(max(v[2] for v in b["vertices_m"]) if b["mesh"] else b["height_m"], 2)
            for b in buildings}
    assert len(tops) >= 3, (
        f"{site}: every building is one of {sorted(tops)} m tall. A borough of 1, 1.5 and 2 "
        f"storey houses does not have {len(tops)} height(s) in it - this is the uniform "
        f"{DEFAULT_BUILDING_HEIGHT_M} m default again, wearing a different name")


@needs_source_data
def test_a_height_nobody_recorded_says_so(site_models, tmp_path):
    """The buildings that fall through are flagged, not quietly averaged into the rest.

    A footprint in no parcel, a parcel in no tax row, a description with no storey in it: all
    three keep the default and export `height_source: assumed`, the same contract
    crosswalk_offset_source has. Roughly one building in ten here, and a reader is entitled to
    know which.
    """
    import json
    from pathlib import Path

    from src.geometry.treatments import DesignState
    from src.render.export import export_scenario
    from src.sources.osm_context import DEFAULT_BUILDING_HEIGHT_M, fetch_crossings

    model = site_models["broad_st_greenwood"]
    with contextlib.redirect_stdout(io.StringIO()):
        path = export_scenario(model, DesignState.from_model(model), "existing",
                               tmp_path / "flagged.json",
                               crossings=fetch_crossings(model.center_wgs84, radius_m=130),
                               theme={})
    buildings = json.loads(Path(path).read_text())["buildings"]
    assumed = [b for b in buildings if b["height_source"] == SOURCE_ASSUMED]
    assert assumed, "nothing was flagged as assumed, which would mean every parcel had a record"
    for b in assumed:
        top = max(v[2] for v in b["vertices_m"]) if b["mesh"] else b["height_m"]
        assert top == pytest.approx(DEFAULT_BUILDING_HEIGHT_M, abs=0.01), (
            f"a building flagged 'assumed' is {top:.2f} m tall, not the "
            f"{DEFAULT_BUILDING_HEIGHT_M} m default it claims to be")
