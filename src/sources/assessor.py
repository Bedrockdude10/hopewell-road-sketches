"""What the tax assessor says is standing on each parcel - MOD-IV, joined to OSM's footprints.

OSM gives this project real building OUTLINES and almost nothing about the third dimension.
So every building was extruded to one DEFAULT_BUILDING_HEIGHT_M and a render of a place where
half the houses are a storey and a half came out as a field of identical 7 m boxes.

The fact is on disk. New Jersey's MOD-IV property tax records carry a BLDG_DESC per parcel in
the assessor's own shorthand - `2SF` is a two-storey frame house, `1.5SF 1G` a storey-and-a-
half with a one-car garage, `B2S` a two-storey with a basement - and `data/MercerTaxList.dbf`
has been in this repo since before any of the rendering work, listed in the README as "joinable
by PIN, not currently used".

THE JOIN IS GEOMETRIC, then by PIN: an OSM footprint sits in a parcel (`PAMS_PIN`), and that
parcel's PIN keys the tax row (`GIS_PIN`). The rule is LARGEST OVERLAP WINS, which is
unambiguous here (the median building's best parcel covers 100% of it).

WHAT IT DOES NOT KNOW is said out loud rather than smoothed over. A building with no parcel, a
parcel with no tax row, or a description with no storey in it (`2G` is a detached garage) keeps
the default height and is exported as such, the same way a crosswalk says
`crosswalk_offset_source: geometric_estimate`. See height_source.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from src.sources.schemas import TaxListSchema, validate_layer

# The assessor counts storeys; this turns them into a height. Same figure OSM's building:levels is
# read with (osm_context.METERS_PER_LEVEL), because they are counting the same thing and a house
# should not change height depending on which source described it.
from src.sources.osm_context import DEFAULT_BUILDING_HEIGHT_M, METERS_PER_LEVEL

# How far out to load parcels for the join. Matches export.BUILDING_CONTEXT_RADIUS_M (130 m), the
# radius buildings themselves are fetched at, so every building has a parcel to be found in.
BUILDING_JOIN_RADIUS_FT = 130 / 0.3048

# Where the number came from, exported per building. Ordered best to worst, which is also the
# order they are tried.
SOURCE_OSM_HEIGHT = "osm_height"          # the mapper measured it
SOURCE_OSM_LEVELS = "osm_levels"          # the mapper counted floors
SOURCE_ASSESSOR = "assessor_storeys"      # MOD-IV BLDG_DESC, this module
SOURCE_ASSUMED = "assumed"                # nobody said; DEFAULT_BUILDING_HEIGHT_M

# A storey count at the front of a MOD-IV building description: "2SF", "1.5SF 1G", "B2S", "2SFWUG".
# The digits before the S are the storeys and the letters after it are construction and outbuildings,
# which this does not use. Anchored on the S so "1G" (a one-car garage, no dwelling) does NOT parse
# as one storey - 15 of Hopewell's 697 descriptions are outbuildings only.
_STOREYS = re.compile(r"(\d(?:\.\d)?)\s*S", re.IGNORECASE)

_PARCEL_PIN_FIELDS = ("PAMS_PIN", "PIN", "GIS_PIN")


@dataclass(frozen=True)
class BuildingHeight:
    """How tall a building is, and who said so."""
    height_m: float
    source: str

    @property
    def is_assumed(self) -> bool:
        return self.source == SOURCE_ASSUMED


def storeys_from_description(description) -> float | None:
    """Storeys from a MOD-IV BLDG_DESC, or None if it does not describe a storeyed building."""
    if not description:
        return None
    match = _STOREYS.search(str(description))
    return float(match.group(1)) if match else None


def storeys_by_pin(tax_list_path: str | Path) -> dict[str, float]:
    """{PIN: storeys} for every parcel the assessor describes as having a storeyed building."""
    path = Path(tax_list_path)
    if not path.exists():
        return {}
    rows = gpd.read_file(path, columns=["GIS_PIN", "BLDG_DESC"])
    # Validated here rather than trusted, because the failure is invisible in a render: if either
    # column is renamed the join matches nothing, every building falls back to one default height,
    # and the result is the field of identical boxes this module was written to fix. No CRS check -
    # the tax list is a table, not a layer. See src/sources/schemas.py.
    validate_layer(rows, TaxListSchema, path)
    out = {}
    for pin, description in zip(rows["GIS_PIN"], rows["BLDG_DESC"]):
        storeys = storeys_from_description(description)
        if storeys and pin:
            out[str(pin).strip()] = storeys
    return out


def _pin_field(parcels) -> str | None:
    return next((f for f in _PARCEL_PIN_FIELDS if f in parcels.columns), None)


def height_of(footprint, parcels, storeys: dict[str, float], osm_height=None) -> BuildingHeight:
    """How tall to build `footprint`, preferring what somebody actually recorded.

    OSM first where a mapper gave a height or floor count, then the assessor, then the
    default, flagged. LARGEST OVERLAP WINS among the parcels a footprint touches: a building
    on a lot line intersects its neighbour's parcel by a sliver, and that sliver must not
    decide its height.
    """
    if osm_height is not None:
        return osm_height
    pin_field = _pin_field(parcels) if parcels is not None and len(parcels) else None
    if pin_field and storeys:
        best_pin, best_area = None, 0.0
        for pin, geometry in zip(parcels[pin_field], parcels.geometry):
            if geometry is None or geometry.is_empty or not geometry.intersects(footprint):
                continue
            shared = geometry.intersection(footprint).area
            if shared > best_area:
                best_pin, best_area = str(pin).strip(), shared
        if best_pin in storeys:
            return BuildingHeight(storeys[best_pin] * METERS_PER_LEVEL, SOURCE_ASSESSOR)
    return BuildingHeight(DEFAULT_BUILDING_HEIGHT_M, SOURCE_ASSUMED)


def assessor_path(model) -> Path:
    """Where this site's MOD-IV tax list is, from its own config.

    A path per site like the road network and the parcels, because a site in another county
    points at another county's records - and one with none at all gets {} and says so, rather
    than failing to build.
    """
    from src.geometry.intersection import ROOT_DIR

    configured = (model.config.get("data_sources") or {}).get("tax_list")
    return ROOT_DIR / configured if configured else Path("")


def parcels_near_buildings(model, radius_ft: float = BUILDING_JOIN_RADIUS_FT):
    """The parcels an OSM footprint could sit in, out to the radius buildings are fetched at.

    Not model.parcels: those are loaded to 300 ft for the corner/ROW context, and buildings come
    from a 130 m (427 ft) circle, so the outer ring of them would find no parcel and silently take
    the default height. The shapefile is spatially indexed - this read is too fast to measure.
    """
    from src.geometry.intersection import ROOT_DIR
    from src.sources.data_loader import load_parcels_near

    configured = (model.config.get("data_sources") or {}).get("parcels")
    if not configured:
        return None
    return load_parcels_near(model.center_wgs84, radius_ft=radius_ft, path=ROOT_DIR / configured)


def describe_building_heights(heights: list[BuildingHeight]) -> str:
    """One line for the phase output: how many buildings anyone actually described."""
    if not heights:
        return "no buildings in range"
    counted = {}
    for height in heights:
        counted[height.source] = counted.get(height.source, 0) + 1
    recorded = {s: n for s, n in sorted(counted.items()) if s != SOURCE_ASSUMED}
    assumed = counted.get(SOURCE_ASSUMED, 0)
    parts = ", ".join(f"{n} from {source}" for source, n in recorded.items())
    return (f"{sum(recorded.values())}/{len(heights)} building heights are recorded "
            f"({parts})" + (f"; {assumed} with no record kept at "
                            f"{DEFAULT_BUILDING_HEIGHT_M:.0f} m" if assumed else ""))
