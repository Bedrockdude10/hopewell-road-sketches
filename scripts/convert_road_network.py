"""
Build a spatially-indexed copy of a roadway network file, then VERIFY the copy is
byte-for-byte equivalent to the original before leaving it in place.

Why: GeoJSON carries no spatial index, so a bbox-filtered read still parses the
entire file. Pulling the ~9 segments around one intersection out of NJDOT's 170 MB
statewide layer costs ~2.2 s - versus ~2.5 s to read all 105,838 features, i.e. the
bbox filter saves almost nothing. FlatGeobuf stores a packed Hilbert R-tree, so the
same bbox read drops to ~0.002 s. Every phase script pays this cost at least once,
and Phase 3/4 are separate processes that each pay it again.

This is a one-off per data file. src/sources/data_loader.py picks the sibling up
automatically once it exists (see _resolve_indexed_path) - no config change needed,
and sites/*/config.yaml keeps pointing at the original as the canonical source.

Usage:
  python scripts/convert_road_network.py [path/to/network.geojson] [--format fgb|gpkg]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src.sources.data_loader import DEFAULT_ROAD_NETWORK_PATH, _unpack_single_part

DRIVERS = {"fgb": "FlatGeobuf", "gpkg": "GPKG"}


def verify_identical(source: Path, converted: Path) -> bool:
    """Read both back and compare attributes and geometry WKB exactly.

    The comparison applies _unpack_single_part to the converted side, because that
    is what load_road_network does on every read - so this checks the data as it
    will actually be consumed, not the raw file. Both are sorted by OBJECTID first,
    since neither format promises to preserve feature order.
    """
    a = gpd.read_file(source)
    b = gpd.read_file(converted)
    if len(a) != len(b) or list(a.columns) != list(b.columns) or a.crs != b.crs:
        print(f"  MISMATCH: rows {len(a)} vs {len(b)}, or columns/CRS differ.")
        return False

    sort_col = "OBJECTID" if "OBJECTID" in a.columns else None
    if sort_col:
        a = a.sort_values(sort_col).reset_index(drop=True)
        b = b.sort_values(sort_col).reset_index(drop=True)
    b = b.set_geometry(b.geometry.map(_unpack_single_part))

    attrs = [c for c in a.columns if c != a.geometry.name]
    if not a[attrs].equals(b[attrs]):
        print("  MISMATCH: attribute tables differ.")
        return False
    if not (a.geometry.geom_type == b.geometry.geom_type).all():
        print("  MISMATCH: geometry types differ even after unpacking single-part multis.")
        return False
    if not (a.geometry.to_wkb() == b.geometry.to_wkb()).all():
        print("  MISMATCH: geometry WKB differs.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=str(DEFAULT_ROAD_NETWORK_PATH))
    parser.add_argument("--format", choices=DRIVERS, default="fgb",
                        help="fgb (FlatGeobuf, default - smallest and fastest) or gpkg (GeoPackage)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        parser.error(f"No such file: {source}")
    target = source.with_suffix(f".{args.format}")

    print(f"Reading {source.name} ({source.stat().st_size / 1e6:.0f} MB)...")
    start = time.time()
    network = gpd.read_file(source)
    print(f"  -> {len(network)} features in {time.time() - start:.1f}s")

    print(f"Writing {target.name} ({DRIVERS[args.format]})...")
    start = time.time()
    network.to_file(target, driver=DRIVERS[args.format])
    print(f"  -> {target.stat().st_size / 1e6:.0f} MB in {time.time() - start:.1f}s")

    print("Verifying the converted copy is identical to the original...")
    if not verify_identical(source, target):
        target.unlink()
        raise SystemExit(f"Verification FAILED - deleted {target.name}. The original is untouched.")
    print("  -> identical (attributes equal, geometry WKB equal).")

    print(f"\nDone. {source.name} is still the canonical source and is unchanged; "
          f"data_loader.py will now read {target.name} automatically.")


if __name__ == "__main__":
    main()
