"""What this project requires of each external data layer, stated as pandera schemas.

Every layer here comes from somebody else - NJDOT's statewide roadway network, Mercer County's
parcels, the county's MOD-IV tax list - and none of them is versioned, pinned, or promised to
this repo. A column gets renamed, a shapefile is re-exported with a different CRS, a download
comes back truncated, and the pipeline does not stop: it reads a missing column as an absent
FACT and carries on drawing.

That is the failure this module exists to make impossible, and it is not hypothetical:

  * `storeys_by_pin` asks for `GIS_PIN` and `BLDG_DESC`. Where the join misses, a building
    keeps a 7 m default and the render says "8 with no record" - which is indistinguishable
    from 8 genuinely unrecorded buildings and from a renamed key that matched nothing.
  * reading the parcels with a WGS84 bbox against a NAD83-State-Plane shapefile returns
    ZERO ROWS rather than an error. That looks exactly like "no parcels here", and it cost two
    round trips in one session.

So each layer is validated AT THE LOAD BOUNDARY, once, and a violation names the file, the
column and what was expected. `pandera` rather than more `pydantic`: pydantic models a record
and already owns `config.yaml` (src/site_schema.py), while these are tabular and geospatial -
thousands of rows, a CRS, a geometry type - which is what a DataFrame schema is for. The two do
not overlap and neither replaces the other.

DELIBERATELY A FLOOR, NOT A CONTRACT. Only the columns this repo actually reads are declared,
and unknown ones pass: NJDOT ships 16 attributes and we use 6, so pinning all 16 would fail the
build the next time NJDOT adds one. `strict=False` is the point, not an omission.
"""
from pathlib import Path

import pandera.pandas as pa

# NO CRS CONSTANTS HERE. src/geometry/model.py already owns WGS84 and NJ_STATE_PLANE_FT, and
# `validate_layer` takes the expected one as an argument, so each caller passes the datum it
# already works in. A schema module holding its own second copy of a projection string is the
# exact class of bug this repo keeps finding - two definitions of one datum that can drift.


class RoadNetworkSchema(pa.DataFrameModel):
    """NJDOT's SRI/SLD linear-referencing roadway layer.

    SRI is the join key every site's config.yaml names its legs by, and SLD_NAME is what the
    phase-1 audit prints to tell you which road NJDOT thinks it is. ROUTE_SUBTYPE distinguishes
    a state route from a county road from a municipal street, which is what decides whether an
    SLD sheet exists for a leg at all.

    NOT declared: lane count, width, surface. They are genuinely absent from this layer, which
    is the whole reason config.yaml carries field measurements - see the main README.
    """
    SRI: pa.typing.Series[str] = pa.Field(nullable=True)
    SLD_NAME: pa.typing.Series[str] = pa.Field(nullable=True)
    ROUTE_SUBTYPE: pa.typing.Series[int] = pa.Field(nullable=True, coerce=True)

    class Config:
        strict = False          # NJDOT ships more columns than we read; that is fine
        coerce = True


class ParcelsSchema(pa.DataFrameModel):
    """Mercer County's parcel polygons.

    PAMS_PIN is the key the assessor's storey count joins through (src/sources/assessor.py), and
    MUN is what tells one municipality's parcels from another's - this project spans Hopewell
    Borough, Hopewell Township and Pennington Borough, and the NJ 31 junction sits on a boundary.
    """
    PAMS_PIN: pa.typing.Series[str] = pa.Field(nullable=True)
    MUN: pa.typing.Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class TaxListSchema(pa.DataFrameModel):
    """The MOD-IV tax list, which is where building HEIGHTS come from.

    Both columns are load-bearing and neither is obviously so from a render: GIS_PIN is the join
    key and BLDG_DESC is the string a storey count is parsed out of. If either is renamed the
    join matches nothing, every building falls back to one default height, and the render is a
    field of identical boxes - which is exactly the thing assessor.py was written to fix.
    """
    GIS_PIN: pa.typing.Series[str] = pa.Field(nullable=True)
    BLDG_DESC: pa.typing.Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


def _horizontal(crs):
    """The 2D part of a CRS.

    MercerCountyParcels.shp is a COMPOUND CRS - NAD83 / New Jersey (ftUS) plus an NAVD88 vertical
    axis - so its string form is a 20-line WKT that equals no EPSG code, while its horizontal
    projection is exactly the EPSG:3424 every offset in this project is measured in. Comparing
    string forms rejected the correct file, which is a check failing on the thing it was meant to
    accept. What matters here is only the horizontal datum, because that is what a bbox is built
    in and what a reprojection would silently break.
    """
    if getattr(crs, "is_compound", False) and getattr(crs, "sub_crs_list", None):
        return crs.sub_crs_list[0]
    return crs


def _crs_label(crs) -> str:
    """A CRS in one line, for an error message. The full WKT of a compound CRS is unreadable."""
    horizontal = _horizontal(crs)
    epsg = horizontal.to_epsg()
    name = getattr(horizontal, "name", None) or "unknown"
    return f"EPSG:{epsg}" if epsg else name


def _same_horizontal_crs(crs, expect_crs: str) -> bool:
    """Whether `crs` projects horizontally the same way as `expect_crs`.

    By EPSG code and then by pyproj equality, never by string comparison - see _horizontal.
    """
    from pyproj import CRS

    want = _horizontal(CRS.from_user_input(expect_crs))
    have = _horizontal(crs if isinstance(crs, CRS) else CRS.from_user_input(crs))
    if have.to_epsg() is not None and want.to_epsg() is not None:
        return have.to_epsg() == want.to_epsg()
    return have.equals(want)


class DataLayerError(ValueError):
    """An external data layer is not the shape this project reads it as."""


def validate_layer(frame, schema, path: Path | str, expect_crs: str | None = None):
    """Check `frame` against `schema`, and its CRS, naming the file on failure.

    Returns the frame so this can wrap a read in one line. Empty frames are checked too - a
    schema violation on an empty result is still a violation, and the CRS check is precisely
    what turns "zero rows" back into the error it is.

    A geometry column is NOT declared in the schemas above: pandera's geopandas support varies
    by version, and the CRS is the part that actually goes wrong here. So it is asserted
    directly, which also keeps these schemas usable against a plain DataFrame (the tax list has
    no geometry at all).
    """
    if expect_crs is not None:
        crs = getattr(frame, "crs", None)
        if crs is None:
            raise DataLayerError(
                f"{Path(path).name} has no CRS. Every bbox in this project is built in the "
                f"layer's own coordinates, so a missing CRS reads downstream as an empty result "
                f"rather than as an error. Expected {expect_crs}.")
        if not _same_horizontal_crs(crs, expect_crs):
            raise DataLayerError(
                f"{Path(path).name} is in {_crs_label(crs)}, not {expect_crs}. A bbox built in "
                f"the wrong projection returns ZERO ROWS instead of failing, which looks exactly "
                f"like 'nothing mapped here' - reproject the file or the bbox, do not ignore this.")
    try:
        schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as failures:
        raise DataLayerError(
            f"{Path(path).name} is not the shape {schema.__name__} describes. This project reads "
            f"the columns below as facts about the street; a missing or renamed one is read as an "
            f"absent fact and drawn as though it were true.\n{failures.failure_cases}") from failures
    return frame
