"""Resolve the set of real texture/model files a render should use, fetching
them (via src/render/assets.py) in the project's normal Python environment. Blender's
own bundled Python has no network access / requests / this project's venv, so
blender_scene.py never fetches anything itself - it only reads the paths this
module resolves, REPO-RELATIVE so the exported JSON is not tied to the machine
that wrote it (see _portable), with every entry allowed to be None (asset
unavailable -> blender_scene.py falls back to a flat color / procedural shape)."""
from functools import lru_cache
from pathlib import Path

from src.render.assets import REPO_ROOT, fetch_polyhaven_model, fetch_polyhaven_texture

# Poly Haven slugs. See README.md "Phase 4 fidelity" for why these specific ones, and why
# signage/trees are procedural instead (no CC0 source - flagged, not hidden).
ASPHALT_SLUG = "asphalt_01"
CONCRETE_SLUG = "pavement_02"
STREETLIGHT_SLUG = "street_lamp_01"
# Mountable-apron surface (Proposal B: "stamped/colored concrete, distinct texture from
# travel lane"). Has real Diffuse/Rough/nor_gl maps at 2k/4k, like the other two.
APRON_SLUG = "patterned_concrete_pavers"

NEAR_RESOLUTION = "4k"
FAR_RESOLUTION = "2k"


def _portable(path) -> str:
    """An asset path as it goes into the geometry JSON: RELATIVE TO THE REPO ROOT, forward
    slashes, e.g. `output/.textures/asphalt_01/4k/asphalt_01_Diffuse_4k.jpg`.

    Absolute is what fetch_* returns and what this exported for a year, which put 19 paths from
    ONE MACHINE into every one of the 65 committed geometry files - all of them under
    output/.textures/, which is gitignored and re-fetched on demand, so on any other checkout
    they named nothing. Nothing failed loudly: blender_scene.py falls back to a flat colour per
    unreadable texture, so a clone rendered untextured asphalt and said so nowhere.

    Falls back to the absolute path if the asset somehow sits outside the checkout, because
    THERE IS NO PORTABLE SPELLING OF THAT and a relative path computed from the wrong root would
    be a worse answer than an honest machine-specific one.
    """
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        return str(resolved)
    return resolved.relative_to(REPO_ROOT).as_posix()


def _texture_paths(slug: str, resolution: str) -> dict[str, str] | None:
    paths = fetch_polyhaven_texture(slug, resolution=resolution)
    if paths is None:
        return None
    return {k: _portable(v) for k, v in paths.items()}


@lru_cache(maxsize=1)
def build_default_theme() -> dict:
    """{"asphalt_near": {...} | None, "asphalt_far", "concrete_near", "concrete_far",
    "apron_near", "apron_far", "streetlight_gltf": str | None}.

    Every path is REPO-RELATIVE (see _portable) because these are serialized into the geometry
    JSON and read back by another interpreter on possibly another machine; blender_scene.py's
    resolve_theme_paths joins them onto its own repo root.

    Cached for the life of the process: the assets vary by neither site nor scenario, so one
    resolution serves a whole multi-site build. Callers treat the dict as read-only (it is
    serialized into the geometry JSON, never mutated).
    """
    return {
        "asphalt_near": _texture_paths(ASPHALT_SLUG, NEAR_RESOLUTION),
        "asphalt_far": _texture_paths(ASPHALT_SLUG, FAR_RESOLUTION),
        "concrete_near": _texture_paths(CONCRETE_SLUG, NEAR_RESOLUTION),
        "concrete_far": _texture_paths(CONCRETE_SLUG, FAR_RESOLUTION),
        "apron_near": _texture_paths(APRON_SLUG, NEAR_RESOLUTION),
        "apron_far": _texture_paths(APRON_SLUG, FAR_RESOLUTION),
        "streetlight_gltf": (lambda p: _portable(p) if p else None)(fetch_polyhaven_model(STREETLIGHT_SLUG)),
    }
