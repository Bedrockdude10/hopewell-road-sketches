"""The texture and model paths in a geometry export name files, and a name has to work on the
machine that READS it.

Every one of the 65 committed exports carried 19 absolute paths from one laptop -
`/Users/danny/.../output/.textures/asphalt_01/4k/asphalt_01_Diffuse_4k.jpg` - pointing into a
gitignored directory that is re-fetched on demand. On any other checkout they named nothing, and
nothing said so: blender_materials.make_textured_material falls back to a flat colour per
unreadable image and import_gltf_template returns None, so a clone rendered untextured asphalt
and reported success. That is the same failure shape as a stale export - a plausible wrong
picture - and the fix is the same, which is to make the artifact not depend on where it was made.
"""
from pathlib import Path

import pytest

from src.render.assets import CACHE_DIR, REPO_ROOT
from src.render.theme import _portable, build_default_theme

# output/.textures is gitignored, so a fresh clone has no assets to resolve and
# build_default_theme returns None for every entry (correctly - it never fetches under
# HOPEWELL_OFFLINE). The _portable tests below need nothing on disk and always run.
needs_textures = pytest.mark.skipif(not CACHE_DIR.exists(),
                                    reason=f"no {CACHE_DIR.relative_to(REPO_ROOT)} - "
                                           "fetched on demand by src/render/theme.py")


def test_a_path_inside_the_checkout_is_written_relative_to_it():
    """What goes in the JSON. `output/.textures/...`, not `/Users/somebody/.../output/...`."""
    assert _portable(CACHE_DIR / "asphalt_01" / "4k" / "x.jpg") == "output/.textures/asphalt_01/4k/x.jpg"


def test_the_consumer_can_get_the_file_back():
    """The contract's other half, which lives in scripts/blender/blender_scene.py:
    resolve_theme_paths joins these onto its own REPO_ROOT. Asserted here as a round trip because
    that module imports bpy and cannot be imported by this suite - so what is pinned is the
    property both sides rely on, in the one place a test can reach it."""
    original = (CACHE_DIR / "models" / "street_lamp_01_1k" / "street_lamp_01_1k.gltf").resolve()
    assert Path(REPO_ROOT) / _portable(original) == original


def test_a_path_outside_the_checkout_stays_absolute():
    """There is no portable spelling of /opt/textures, and a relative path computed from the wrong
    root would resolve to a file that does not exist while LOOKING portable. An honest
    machine-specific path is the better wrong answer: blender_scene passes an absolute path
    through untouched, so it still loads here and still degrades to a flat colour elsewhere."""
    outside = Path("/opt/shared-assets/asphalt.jpg")
    assert _portable(outside) == str(outside)


@needs_textures
def test_no_resolved_theme_path_is_absolute():
    """The regression itself, over the real theme rather than a constructed path: 19 entries, and
    it only took one of them to tie a geometry file to a machine."""
    theme = build_default_theme()
    paths = [v for entry in theme.values()
             for v in (entry.values() if isinstance(entry, dict) else [entry]) if v]
    assert paths, "no assets resolved - CACHE_DIR exists but is empty?"
    assert [p for p in paths if Path(p).is_absolute()] == []


@needs_textures
def test_every_resolved_theme_path_names_a_file_that_exists():
    """Relative and wrong is worse than absolute and wrong, because it looks portable. This is
    the check that the relative spelling is anchored where the reader will anchor it."""
    theme = build_default_theme()
    paths = [v for entry in theme.values()
             for v in (entry.values() if isinstance(entry, dict) else [entry]) if v]
    assert [p for p in paths if not (REPO_ROOT / p).exists()] == []
