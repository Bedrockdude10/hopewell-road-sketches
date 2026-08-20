"""The external data layers are validated at the load boundary, and the validators agree.

Two things here. First, that a renamed or missing column FAILS rather than being read as an
absent fact - which is the whole point of src/sources/schemas.py. Second, that the four
validation mechanisms this project now uses do not contradict each other:

    pydantic          src/site_schema.py     one config.yaml, per-field
    pandera           src/sources/schemas.py one external data layer, per-column
    import-linter     .importlinter          the import graph
    golden geometry   tests/test_geometry_regression/  the drawn result

They validate different things on purpose and none replaces another, but they share constants -
a CRS string, a column name - and a second copy of a shared constant is this repo's most
repeated bug. So the overlap is asserted rather than assumed.

On the columns this project READS, the pandera schemas are a contract: not-null where the real
file is never null, a domain where the domain is known, and a shared format on the join key. Only
the QUESTION OF WHICH COLUMNS MAY EXIST is left open, and src/sources/schemas.py argues that one.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point
from tests.conftest import needs_source_data

from src.sources.schemas import (DataLayerError, ParcelsSchema, RoadNetworkSchema, TaxListSchema,
                                 validate_layer)


def _road_frame(**overrides) -> gpd.GeoDataFrame:
    data = {"SRI": ["00000518__"], "SLD_NAME": ["BROAD ST"], "ROUTE_SUBTYPE": [3],
            "geometry": [LineString([(0, 0), (1, 1)])]}
    data.update(overrides)
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def test_a_valid_layer_passes_through_unchanged():
    frame = _road_frame()
    assert validate_layer(frame, RoadNetworkSchema, "x.geojson", expect_crs="EPSG:4326") is frame


def test_a_renamed_column_fails_and_names_the_file():
    """The failure this module exists for. NJDOT renaming SRI would otherwise read downstream as
    'this leg matched no road' - drawn, not raised."""
    frame = _road_frame().rename(columns={"SRI": "SRI_ID"})
    with pytest.raises(DataLayerError, match="NJ_Roadway_Network"):
        validate_layer(frame, RoadNetworkSchema, "NJ_Roadway_Network.geojson")


def test_the_wrong_projection_fails_rather_than_returning_nothing():
    """The one that has actually bitten, twice in one session: a bbox built in the wrong CRS
    returns ZERO ROWS, which looks exactly like 'nothing mapped here'."""
    frame = _road_frame().to_crs("EPSG:3424")
    with pytest.raises(DataLayerError, match="ZERO ROWS"):
        validate_layer(frame, RoadNetworkSchema, "parcels.shp", expect_crs="EPSG:4326")


def test_a_missing_crs_fails():
    frame = gpd.GeoDataFrame({"PAMS_PIN": ["1106_1_1"], "MUN": ["1106"],
                              "geometry": [Point(0, 0)]}, crs=None)
    with pytest.raises(DataLayerError, match="no CRS"):
        validate_layer(frame, ParcelsSchema, "parcels.shp", expect_crs="EPSG:3424")


def test_an_empty_result_still_has_its_crs_checked():
    """An empty frame is not automatically fine - it is the SYMPTOM of the CRS bug, so the CRS
    check has to run on it rather than being skipped as 'nothing to validate'."""
    empty = gpd.GeoDataFrame({"PAMS_PIN": pd.Series([], dtype=str),
                              "MUN": pd.Series([], dtype=str),
                              "geometry": []}, crs="EPSG:4326")
    with pytest.raises(DataLayerError, match="ZERO ROWS"):
        validate_layer(empty, ParcelsSchema, "parcels.shp", expect_crs="EPSG:3424")


def test_an_all_null_key_column_fails():
    """The case the first version of this schema LET THROUGH, and the reason 'a floor' was the
    wrong framing for the columns we depend on.

    Every field was nullable=True with no constraint, so a column of all nulls validated - and an
    all-null SRI then matches no road on any leg, which is exactly the silent failure the module
    was written to stop. Measured against the real file SRI is never null, so it says so.
    """
    frame = _road_frame(SRI=[None])
    with pytest.raises(DataLayerError, match="SRI"):
        validate_layer(frame, RoadNetworkSchema, "NJ_Roadway_Network.geojson")


def test_a_broken_join_key_format_fails_on_both_sides():
    """PAMS_PIN and GIS_PIN are the same municipality_block_lot key in two different files, and
    nothing else in the pipeline compares them. If one side's format changes the join matches
    nothing, every building takes one default height, and the render is a field of identical
    boxes - visible only as a slightly different picture."""
    parcels = gpd.GeoDataFrame({"PAMS_PIN": ["1106-18-14"], "MUN": ["1106"],
                                "geometry": [Point(0, 0)]}, crs="EPSG:3424")
    with pytest.raises(DataLayerError, match="PAMS_PIN"):
        validate_layer(parcels, ParcelsSchema, "MercerCountyParcels.shp")
    tax = pd.DataFrame({"GIS_PIN": ["not-a-pin"], "BLDG_DESC": ["2SF"]})
    with pytest.raises(DataLayerError, match="GIS_PIN"):
        validate_layer(tax, TaxListSchema, "MercerTaxList.dbf")


def test_the_join_keys_share_one_pattern():
    """Two regexes for one key could drift, which would defeat the only reason to check it."""
    from src.sources.schemas import PIN_PATTERN

    parcels_check = ParcelsSchema.to_schema().columns["PAMS_PIN"].checks
    tax_check = TaxListSchema.to_schema().columns["GIS_PIN"].checks
    assert [c.name for c in parcels_check] == [c.name for c in tax_check]
    assert PIN_PATTERN in str(parcels_check) and PIN_PATTERN in str(tax_check)


def test_a_genuinely_absent_value_is_still_allowed():
    """BLDG_DESC is nullable because 12.4% of county rows have none - vacant land has no
    building. Tightening that one would fail on the real file, which is the difference between a
    contract and a guess."""
    frame = pd.DataFrame({"GIS_PIN": ["1106_1_1"], "BLDG_DESC": [None]})
    assert validate_layer(frame, TaxListSchema, "MercerTaxList.dbf") is frame


def test_an_out_of_domain_route_subtype_fails():
    """NJDOT's functional class runs 2-8 here. A 0 or a 99 means the column is not what we think
    it is - which matters because ROUTE_SUBTYPE decides whether an SLD sheet exists for a leg."""
    with pytest.raises(DataLayerError, match="ROUTE_SUBTYPE"):
        validate_layer(_road_frame(ROUTE_SUBTYPE=[99]), RoadNetworkSchema, "x.geojson")


def test_unknown_columns_are_allowed():
    """strict=False is deliberate, and it is the ONE thing here that is a floor rather than a
    contract. These are third-party files on somebody else's release schedule - NJDOT ships 16
    attributes and this project reads 3 - so an additive upstream change is not a defect, and
    failing the build for one would be a false positive that blocks every site. A validator that
    cries wolf on a benign change gets deleted; this module's own CRS check did exactly that."""
    frame = _road_frame(SOME_NEW_NJDOT_FIELD=["whatever"])
    assert validate_layer(frame, RoadNetworkSchema, "x.geojson") is frame


def test_the_tax_list_needs_no_crs():
    """It is a table, not a layer - the join key and the storey string are all this reads."""
    frame = pd.DataFrame({"GIS_PIN": ["1106_1_1"], "BLDG_DESC": ["2SF"]})
    assert validate_layer(frame, TaxListSchema, "MercerTaxList.dbf") is frame


# --- the validators agree ------------------------------------------------------------------

def test_the_crs_strings_have_one_definition():
    """pandera's callers pass the datum from src/geometry/model/crs.py rather than a second copy.

    schemas.py deliberately declares NO projection constants of its own; if someone adds them,
    this fails and points at the one place they belong.
    """
    import src.sources.schemas as schemas

    duplicated = [name for name in dir(schemas)
                  if isinstance(getattr(schemas, name), str)
                  and getattr(schemas, name).startswith("EPSG:")]
    assert not duplicated, (
        f"src/sources/schemas.py declares its own CRS constant(s) {duplicated} - "
        f"src/geometry/model/crs.py owns WGS84 and NJ_STATE_PLANE_FT, and two definitions of one "
        f"datum is the bug this project keeps finding")


def test_pandera_and_pydantic_cover_different_things():
    """They are not redundant and must not drift into overlapping. pydantic validates one site's
    config.yaml (a record); pandera validates an external layer (a table). If a name appears in
    both, one of them is restating a fact the other owns."""
    from src.sources import schemas as pandera_schemas

    pandera_columns = set()
    for model in (RoadNetworkSchema, ParcelsSchema, TaxListSchema):
        pandera_columns |= set(model.to_schema().columns)
    # The config schema's field names, flattened one level - enough to catch a real collision.
    from src import site_schema

    config_fields = set()
    for name in dir(site_schema):
        model = getattr(site_schema, name)
        fields = getattr(model, "model_fields", None)
        if fields:
            config_fields |= set(fields)
    assert not (pandera_columns & config_fields), (
        f"{pandera_columns & config_fields} is declared in BOTH the pydantic config schema and a "
        f"pandera data-layer schema - a fact should be owned by whichever source actually "
        f"supplies it, not asserted twice")
    assert pandera_schemas.TaxListSchema is TaxListSchema      # module exports what it declares


@needs_source_data
def test_a_compound_crs_with_the_right_horizontal_datum_is_accepted():
    """MercerCountyParcels.shp is NAD83 / New Jersey (ftUS) PLUS an NAVD88 vertical axis, so its
    string form is a 20-line WKT matching no EPSG code while its horizontal projection is exactly
    the one every offset here is measured in.

    Pinned because the first version of this check compared string forms and rejected the correct
    file - a validator failing on the thing it exists to accept, which is worse than no validator
    because the obvious fix is to delete it.
    """
    from pyproj import CRS

    from src.sources.data_loader import resolve_data_path
    from src.sources.schemas import _same_horizontal_crs

    # Through resolve_data_path, so this reads whatever the run is reading - the committed clip
    # keeps the county's .prj bytes verbatim precisely so this check means the same thing there.
    compound = CRS.from_wkt(gpd.read_file(
        resolve_data_path("data/MercerCountyParcels.shp"), rows=1).crs.to_wkt())
    assert compound.is_compound, "this test is about the compound case; the file changed"
    assert _same_horizontal_crs(compound, "EPSG:3424")
    # And it still rejects a genuinely different projection.
    assert not _same_horizontal_crs(compound, "EPSG:4326")
