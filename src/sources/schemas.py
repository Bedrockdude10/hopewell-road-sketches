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

TWO SEPARATE QUESTIONS, AND ONLY ONE OF THEM IS "FLOOR VS CONTRACT".

  1. WHICH COLUMNS MAY EXIST. `strict=False`: a column this repo does not read is allowed to
     appear. These are third-party files on somebody else's release schedule - NJDOT ships 16
     attributes and we read 3 - so an additive upstream change is not a defect here.

  2. WHAT THE COLUMNS WE READ MUST CONTAIN. Here it IS a contract. An all-null SRI passes
     validation and then matches no road on any leg - which is precisely the silent failure
     this module was written to stop. None of these columns is ever null except BLDG_DESC
     (12.4%, genuinely - vacant land has no building), so that is what they now declare.

The join keys carry a format check for the same reason. PAMS_PIN and GIS_PIN are the same
municipality_block_lot identifier in two files, and nothing else in the pipeline compares them:
if one side's format changes the join simply matches nothing and every building falls back to
one height. A regex on both is the cheapest place to catch a break that is otherwise invisible.
"""
from pathlib import Path

import pandera.pandas as pa

# THE PARCEL KEY, in both files that carry it. `1106_18_14` and `1106_18_14_Q0009` - municipality,
# block, lot, and an optional qualifier. Declared once and used by both schemas, because the whole
# value of checking it is that the two sides of the join agree; two regexes that could drift would
# defeat the point of writing one at all.
PIN_PATTERN = r"^\d{4}_[^_]+_[^_]+(_.+)?$"
# A four-digit NJ municipality code (1106 Hopewell Twp, 1108 Pennington Boro, and neighbours).
MUN_PATTERN = r"^\d{4}$"
# NJDOT's Standard Route Identifier: 10 or 17 characters, no spaces.
SRI_PATTERN = r"^\S{10}(\S{7})?$"

# NO CRS CONSTANTS HERE. src/geometry/model/crs.py already owns WGS84 and NJ_STATE_PLANE_FT, and
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
    # Never null in the real file (0.0% across a 5,000-row sample), and an all-null SRI would
    # match no road on any leg while passing a nullable check.
    SRI: pa.typing.Series[str] = pa.Field(nullable=False, str_matches=SRI_PATTERN)
    SLD_NAME: pa.typing.Series[str] = pa.Field(nullable=False)
    # NJDOT's functional class. 2-8 in the statewide file; 1 is allowed as headroom since the
    # domain is theirs, but a 0 or a 99 means the column is not what we think it is.
    ROUTE_SUBTYPE: pa.typing.Series[int] = pa.Field(nullable=False, coerce=True,
                                                     in_range={"min_value": 1, "max_value": 8})

    class Config:
        strict = False          # NJDOT ships more columns than we read; that is fine
        coerce = True


class ParcelsSchema(pa.DataFrameModel):
    """Mercer County's parcel polygons.

    PAMS_PIN is the key the assessor's storey count joins through (src/sources/assessor.py), and
    MUN is what tells one municipality's parcels from another's - this project spans Hopewell
    Borough, Hopewell Township and Pennington Borough, and the NJ 31 junction sits on a boundary.
    """
    PAMS_PIN: pa.typing.Series[str] = pa.Field(nullable=False, str_matches=PIN_PATTERN)
    MUN: pa.typing.Series[str] = pa.Field(nullable=False, str_matches=MUN_PATTERN)

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
    GIS_PIN: pa.typing.Series[str] = pa.Field(nullable=False, str_matches=PIN_PATTERN)
    # THE ONE GENUINELY NULLABLE FIELD HERE, and it is measured rather than assumed: 12.4% of
    # rows have no building description, because vacant land has no building. That is a fact
    # about the county, not slack in the schema.
    BLDG_DESC: pa.typing.Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


def _horizontal(crs):
    """The 2D part of a CRS.

    MercerCountyParcels.shp is a COMPOUND CRS - NAD83 / New Jersey (ftUS) plus an NAVD88
    vertical axis - so its string form is a 20-line WKT that equals no EPSG code, while
    its horizontal projection is exactly EPSG:3424. Comparing string forms rejected the
    correct file.
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

    Returns the frame so this can wrap a read in one line. Empty frames are checked too. A
    geometry column is NOT declared in the schemas: pandera's geopandas support varies by
    version, and the CRS is the part that actually goes wrong here.
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
