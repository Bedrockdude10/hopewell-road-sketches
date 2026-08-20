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
FIXTURE_DATA = REPO_ROOT / "tests" / "fixtures" / "data"

# Set before any src module is imported: osm_context reads HOPEWELL_OSM_CACHE at import time.
os.environ.setdefault("HOPEWELL_OSM_CACHE", str(FIXTURE_CACHE))
os.environ.setdefault("HOPEWELL_OFFLINE", "1")
# The GIS layers come from the committed clip, EVEN WHEN data/ IS PRESENT, so that a local run
# and CI are checking the same bytes. Otherwise the goldens would be pinned against the full
# county here and against the clip there, and a divergence would show up as a mystery failure on
# whichever machine was the minority. tests/test_data_fixture.py is the bridge: it builds every
# site both ways and compares, and it is the one thing here that does need data/.
os.environ.setdefault("HOPEWELL_DATA_DIR", str(FIXTURE_DATA))

SITES = ("broad_st_greenwood", "ebroad_princeton", "columbia_princeton", "wbroad_louellen")

# Whole-site tests need the GIS layers, which now means the committed clip rather than the 391 MB
# download - so this skips essentially never, and 333 of 707 tests that used to sit out every CI
# run (every geometry golden among them) now run. Kept as a marker rather than deleted: it still
# names the dependency at each test, and it still fires if the clip is missing - a checkout with
# no LFS, a partial clone, or someone regenerating the fixture into the wrong directory.
needs_source_data = pytest.mark.skipif(
    not FIXTURE_DATA.exists() and not (REPO_ROOT / "data").exists(),
    reason=f"no GIS layers: neither {FIXTURE_DATA.relative_to(REPO_ROOT)} (scripts/"
           f"make_data_fixture.py) nor data/ is present - see README",
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


# The frame scale output/ is actually drawn at. Here rather than in one test module because three
# of them need it now: anything about a feature PAST the modelled junction - a cross street, its
# kerb opening, its crosswalks - is invisible at 1x, so a test that forgets to widen the frame
# passes having checked nothing.
WIDE_FRAME_SCALE = 2.2


@pytest.fixture(scope="session")
def wide_site_models():
    """{site: IntersectionModel} built at the frame scale output/ is drawn at.

    NOTE the env var is restored when this returns. The frame is read again at DRAW time, so a
    test that resolves a scene from these models must set it back itself (FRAME_SCALE_ENV,
    WIDE_FRAME_SCALE) or it will draw a 1x frame around 2.2x legs.
    """
    import contextlib
    import io

    from src.geometry.intersection import load_intersection_model
    from src.render.frame import FRAME_SCALE_ENV

    previous = os.environ.get(FRAME_SCALE_ENV)
    os.environ[FRAME_SCALE_ENV] = str(WIDE_FRAME_SCALE)
    try:
        models = {}
        for site in SITES:
            with contextlib.redirect_stdout(io.StringIO()):
                models[site] = load_intersection_model(site=site)
        return models
    finally:
        if previous is None:
            os.environ.pop(FRAME_SCALE_ENV, None)
        else:
            os.environ[FRAME_SCALE_ENV] = previous
