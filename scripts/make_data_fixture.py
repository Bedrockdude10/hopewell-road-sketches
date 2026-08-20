"""Clip data/ down to the few features the sites actually read, as a committed test fixture.

    .venv/bin/python scripts/make_data_fixture.py            # rebuild tests/fixtures/data
    .venv/bin/python scripts/make_data_fixture.py --pad-ft 3000 --out /tmp/clip

WHY. data/ is a 391 MB third-party download kept out of git, so every test that builds a real
junction - 333 of 707, including every geometry golden - skips in CI and in any checkout without
it. But a site reads 9 road segments out of NJDOT's 105,838 and the parcels within a few hundred
feet: the union of all six sites, padded a quarter-mile, is under a megabyte. tests/fixtures/
osm_cache already established the pattern for Overpass; this is the same trick for the two
GIS layers and the tax list.

THE CLIP IS READ THROUGH THE PRODUCTION LOADERS. HOPEWELL_DATA_DIR re-roots `data/...` paths
(src/sources/data_loader.resolve_data_path) and nothing else changes: same pandera schemas, same
CRS checks, same indexed-sibling swap. A separate "test data" code path would be a second datum
for the same fact, which is the most expensive defect class this repo has.

TWO THINGS KEEP IT HONEST, because a clip's failure mode is a read that SUCCEEDS with fewer
features - a wrong measurement that looks right:

  * Every layer written here is read back and compared to the same bbox read of the source,
    attribute by attribute and geometry WKB by WKB (verify_identical, borrowed from
    convert_road_network.py). A clip that is not a byte-faithful subset is deleted, not kept.
  * FIXTURE.json records how far the clip reaches, and load_road_network/load_parcels raise
    FixtureExtentExceeded on a read that runs off the edge of it (or on an unbounded read).
    Note the read bbox comes from each site's `clip_radius_m`, NOT from HOPEWELL_FRAME_SCALE -
    the frame scales leg lengths and the OSM context radius, not these two GIS reads - so
    editing clip_radius_m or adding a site is what makes the guard fire.

tests/test_data_fixture.py closes the loop from the other side: it builds every site from the
clip AND from the real data/ and compares the resolved models. That test needs the download, so
it runs locally and skips in CI - the guard above is the part that works where data/ is absent.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
import pyogrio

from src.geometry.model import NJ_STATE_PLANE_FT, WGS84, buffer_point_wgs84
from src.site import list_sites, load_site_config
from src.sources.assessor import BUILDING_JOIN_RADIUS_FT
from src.sources.data_loader import (DATA_DIR, FIXTURE_MANIFEST_NAME, _resolve_indexed_path,
                                     _unpack_single_part)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT_DIR / "tests" / "fixtures" / "data"

# The parcel reads are load.py's 300 ft context ring and assessor.py's building-join radius
# (130 m); take the larger so the clip covers both.
PARCEL_READ_RADIUS_FT = max(300.0, BUILDING_JOIN_RADIUS_FT)

# A quarter mile past the furthest read any site makes today. Costs kilobytes and absorbs a
# clip_radius_m edit or a new site a few streets over without a regenerate.
DEFAULT_PAD_FT = 1320.0

# Columns the parcel clip carries. NOT "all of them": LASTUPDATE is a shapefile Date, and OGR
# cannot round-trip a date through a shapefile as anything but a string ("created as String
# field, though DateTime requested"), so a clip that kept it would differ from the county file in
# a column nothing reads. Dropping it is what lets verify_identical stay an exact comparison
# rather than one with an exception carved into it. PAMS_PIN and MUN are what ParcelsSchema
# declares; BLOCK/LOT/QCODE are the assessor's own identifiers, kept because they are free.
PARCEL_COLUMNS = ("PAMS_PIN", "MUN", "BLOCK", "LOT", "QCODE")
# Same idea, and additionally a privacy one - see clip_tax_list.
TAX_COLUMNS = ("GIS_PIN", "BLDG_DESC")
FT_PER_DEG_LAT = 364000.0   # good to ~1% at this latitude; padding, not geometry


def site_extents(sites: list[str]) -> tuple[tuple, tuple, dict]:
    """(wgs84 roads bbox, NJ-plane parcels bbox, per-site detail) covering every site's reads.

    The bboxes are built with the SAME calls the loaders make - buffer_point_wgs84 at
    clip_radius_m * 1.3 for roads, a square of PARCEL_READ_RADIUS_FT for parcels - rather than
    re-derived here, because a fixture sized by a second copy of that arithmetic is exactly the
    kind of agreeing-but-wrong pair this repo keeps getting bitten by.
    """
    road_boxes, parcel_boxes, detail = [], [], {}
    for site in sites:
        config = load_site_config(site)
        lon, lat = config["intersection"]["center_wgs84"]
        centre = gpd.points_from_xy([lon], [lat], crs=WGS84)[0]
        clip_radius_m = config["intersection"]["clip_radius_m"]
        roads = buffer_point_wgs84(centre, clip_radius_m * 1.3)
        centre_ft = gpd.GeoSeries([centre], crs=WGS84).to_crs(NJ_STATE_PLANE_FT).iloc[0]
        parcels = (centre_ft.x - PARCEL_READ_RADIUS_FT, centre_ft.y - PARCEL_READ_RADIUS_FT,
                   centre_ft.x + PARCEL_READ_RADIUS_FT, centre_ft.y + PARCEL_READ_RADIUS_FT)
        road_boxes.append(tuple(float(v) for v in roads))
        parcel_boxes.append(parcels)
        detail[site] = {"clip_radius_m": clip_radius_m, "roads_wgs84": road_boxes[-1],
                        "parcels_nj_ft": parcels}
    return _union(road_boxes), _union(parcel_boxes), detail


def _union(boxes: list[tuple]) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _pad(bbox: tuple, pad_ft: float, degrees: bool) -> tuple:
    """Grow a bbox by pad_ft on all four sides.

    In degrees, both axes are padded by the LATITUDE figure. A longitude degree is shorter than
    a latitude one at 40.4 N (cos ~ 0.76), so that over-covers east-west - the safe direction,
    for a number whose whole job is slack.
    """
    pad = pad_ft / FT_PER_DEG_LAT if degrees else pad_ft
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def verify_identical(source_path: Path, bbox, written: Path, columns: list[str] | None = None) -> bool:
    """Compare the written clip against the same bbox read of the source, exactly.

    Attributes and geometry WKB, after _unpack_single_part on both sides - which is what the
    loaders apply on every read, so this checks the data as it will actually be consumed.
    """
    want = gpd.read_file(source_path, bbox=bbox, columns=columns)
    got = gpd.read_file(written, bbox=None, columns=columns)
    if len(want) != len(got):
        print(f"  MISMATCH: {len(want)} features in the source bbox, {len(got)} written.")
        return False
    shared = [c for c in want.columns if c != "geometry"]
    if sorted(shared) != sorted(c for c in got.columns if c != "geometry"):
        print(f"  MISMATCH: columns differ - {sorted(shared)} vs "
              f"{sorted(c for c in got.columns if c != 'geometry')}")
        return False
    key = next((c for c in ("OBJECTID", "PAMS_PIN", "GIS_PIN") if c in shared), None)
    if key:
        want, got = want.sort_values(key).reset_index(drop=True), got.sort_values(key).reset_index(drop=True)
    for column in shared:
        if not want[column].astype("string").fillna("").equals(got[column].astype("string").fillna("")):
            print(f"  MISMATCH: column {column} differs.")
            return False
    if "geometry" in want.columns and want.geometry.notna().any():
        if want.crs is not None and got.crs is not None and want.crs != got.crs:
            print(f"  MISMATCH: CRS {want.crs.name} vs {got.crs.name}")
            return False
        a = want.geometry.map(_unpack_single_part).to_wkb()
        b = got.geometry.map(_unpack_single_part).to_wkb()
        if not (a == b).all():
            print("  MISMATCH: geometry WKB differs.")
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_geoseries(bbox: tuple):
    """A CRS-tagged box, which is how load_parcels queries the shapefile's own NAD83 variant."""
    return gpd.GeoSeries([gpd.GeoSeries.from_wkt([
        f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, "
        f"{bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"]).iloc[0]], crs=NJ_STATE_PLANE_FT)


def clip_roads(out: Path, bbox: tuple) -> tuple[Path, Path, int]:
    """The roadway network, as FlatGeobuf - the format the loaders already prefer."""
    source = _resolve_indexed_path(DATA_DIR / "NJ_Roadway_Network.geojson")
    target = out / "NJ_Roadway_Network.fgb"
    roads = gpd.read_file(source, bbox=bbox)
    roads.to_file(target, driver="FlatGeobuf")
    return source, target, len(roads)


def clip_parcels(out: Path, bbox: tuple) -> tuple[Path, Path, int]:
    """The parcels, as a shapefile with the source .prj copied over byte for byte.

    MercerCountyParcels.shp is a COMPOUND CRS whose WKT matches no EPSG code (see
    src/sources/schemas._horizontal), and a rewritten .prj is a different string for the same
    ground. Copying it keeps the fixture's CRS textually identical to the county's, so the
    boundary check tests the same thing here as it does against the download.
    """
    source = DATA_DIR / "MercerCountyParcels.shp"
    target = out / "MercerCountyParcels.shp"
    parcels = gpd.read_file(source, bbox=_as_geoseries(bbox), columns=list(PARCEL_COLUMNS))
    parcels.to_file(target, driver="ESRI Shapefile")
    shutil.copyfile(source.with_suffix(".prj"), target.with_suffix(".prj"))
    return source, target, len(parcels)


def clip_tax_list(out: Path, pins: set[str]) -> tuple[Path, Path, int]:
    """The MOD-IV rows for the clipped parcels, as a standalone .dbf.

    Only GIS_PIN and BLDG_DESC: those are the two columns assessor.py reads and TaxListSchema
    declares, and the other 100-odd carry owner names and sale prices that have no business in
    a public fixture. The schemas are strict=False on the column SET by design.
    """
    source = DATA_DIR / "MercerTaxList.dbf"
    target = out / "MercerTaxList.dbf"
    rows = gpd.read_file(source, columns=list(TAX_COLUMNS))
    kept = pd.DataFrame(rows.drop(columns="geometry", errors="ignore"))
    kept = kept[kept["GIS_PIN"].astype("string").str.strip().isin(pins)]
    pyogrio.write_dataframe(kept, target, driver="ESRI Shapefile")
    return source, target, len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where to write the clip (default {DEFAULT_OUT.relative_to(ROOT_DIR)})")
    parser.add_argument("--pad-ft", type=float, default=DEFAULT_PAD_FT,
                        help=f"padding past the furthest read any site makes (default {DEFAULT_PAD_FT:.0f})")
    parser.add_argument("--site", action="append", choices=list_sites(),
                        help="only these sites (default: all of them)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"data/ is not here ({DATA_DIR}), and this script clips FROM it - it is how the "
              f"fixture gets built, not how it gets used. See README, section \"Data\".",
              file=sys.stderr)
        return 2

    sites = args.site or list_sites()
    roads_bbox, parcels_bbox, detail = site_extents(sites)
    roads_bbox = _pad(roads_bbox, args.pad_ft, degrees=True)
    parcels_bbox = _pad(parcels_bbox, args.pad_ft, degrees=False)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Clipping {len(sites)} site(s) out of data/, padded {args.pad_ft:.0f} ft -> {out}")

    started = time.perf_counter()
    written: dict[str, dict] = {}
    road_source, road_target, n_roads = clip_roads(out, roads_bbox)
    print(f"  roads:   {n_roads} features")
    parcel_source, parcel_target, n_parcels = clip_parcels(out, parcels_bbox)
    print(f"  parcels: {n_parcels} polygons")
    pins = {str(p).strip() for p in gpd.read_file(parcel_target)["PAMS_PIN"].dropna()}
    tax_source, tax_target, n_tax = clip_tax_list(out, pins)
    print(f"  tax:     {n_tax} rows for {len(pins)} parcels")

    print("Verifying each clip is a faithful subset of the source...")
    checks = [
        ("roads", road_source, roads_bbox, road_target, None),
        ("parcels", parcel_source, _as_geoseries(parcels_bbox), parcel_target, list(PARCEL_COLUMNS)),
        ("tax", tax_source, None, tax_target, list(TAX_COLUMNS)),
    ]
    for name, source, bbox, target, columns in checks:
        if name == "tax":
            continue      # a PIN filter, not a bbox clip - checked by the join test instead
        if not verify_identical(source, bbox, target, columns):
            for stem in target.parent.glob(target.stem + ".*"):
                stem.unlink()
            print(f"Verification FAILED for {name} - deleted {target.name}. data/ is untouched.",
                  file=sys.stderr)
            return 1
        print(f"  {name}: identical (attributes equal, geometry WKB equal).")

    for name, source, target in (("roads", road_source, road_target),
                                 ("parcels", parcel_source, parcel_target),
                                 ("tax", tax_source, tax_target)):
        written[name] = {"file": target.name, "clipped_from": source.name,
                         "source_sha256": _sha256(source),
                         "columns": {"parcels": list(PARCEL_COLUMNS),
                                     "tax": list(TAX_COLUMNS)}.get(name, "all")}
    manifest = {
        "generated_by": "scripts/make_data_fixture.py",
        "why": "data/ is a large third-party download kept out of git; these are the features "
               "the configured sites actually read. Read through HOPEWELL_DATA_DIR.",
        "pad_ft": args.pad_ft,
        "sites": sites,
        "extents": {"roads": list(roads_bbox), "parcels": list(parcels_bbox)},
        "extent_crs": {"roads": "EPSG:4326", "parcels": NJ_STATE_PLANE_FT},
        "per_site_reads": detail,
        "layers": written,
    }
    (out / FIXTURE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"Wrote {total / 1e6:.2f} MB in {time.perf_counter() - started:.1f}s "
          f"(data/ is {sum(f.stat().st_size for f in DATA_DIR.iterdir() if f.is_file()) / 1e6:.0f} MB)")
    print(f"Use it with: HOPEWELL_DATA_DIR={out} .venv/bin/python -m pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
