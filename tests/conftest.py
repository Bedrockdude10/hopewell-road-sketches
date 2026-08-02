"""Test configuration: hermetic by default.

Two things make the suite reproducible and fast:

  * The OSM cache is pointed at tests/fixtures/osm_cache, a committed snapshot of the
    Overpass responses for the four sites. Tests therefore see a fixed OSM state.
  * HOPEWELL_OFFLINE is set, so any fetch that ISN'T satisfied from that snapshot raises
    OfflineCacheMiss instead of quietly reaching the network. A test that silently depends
    on Overpass's uptime and current replication state is worse than no test - during one
    editing session two consecutive live fetches of the same junction returned 4 tactile
    pads and then 0.

Refresh the snapshot with:  cp output/.cache/*.json tests/fixtures/osm_cache/
"""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CACHE = REPO_ROOT / "tests" / "fixtures" / "osm_cache"

# Set before any src module is imported: osm_context reads HOPEWELL_OSM_CACHE at import time.
os.environ.setdefault("HOPEWELL_OSM_CACHE", str(FIXTURE_CACHE))
os.environ.setdefault("HOPEWELL_OFFLINE", "1")

SITES = ("broad_st_greenwood", "ebroad_princeton", "columbia_princeton", "wbroad_louellen")

# The NJDOT road network and county parcels are large, licensed downloads kept out of git
# (see .gitignore). Everything that only needs geometry primitives or the OSM snapshot runs
# without them; the whole-site integration tests skip rather than fail when they're absent.
needs_source_data = pytest.mark.skipif(
    not (REPO_ROOT / "data").exists(),
    reason="data/ (NJDOT road network + Mercer County parcels) not present - see README",
)


@pytest.fixture(scope="session")
def site_models():
    """{site: IntersectionModel} built once for the whole session.

    Building a model is the expensive part of these tests (parcels, road network, the OSM
    snapshot), and it is read-only afterwards - so it is done once rather than per test.
    """
    import contextlib
    import io

    from src.geometry.intersection import load_intersection_model

    models = {}
    for site in SITES:
        with contextlib.redirect_stdout(io.StringIO()):   # phase notes are noise here
            models[site] = load_intersection_model(site=site)
    return models
