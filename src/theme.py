"""Resolve the set of real texture/model files a render should use, fetching
them (via src/assets.py) in the project's normal Python environment. Blender's
own bundled Python has no network access / requests / this project's venv, so
blender_scene.py never fetches anything itself - it only reads the local paths
this module resolves, with every entry allowed to be None (asset unavailable ->
blender_scene.py falls back to a flat color / procedural shape)."""
from src.assets import fetch_polyhaven_model, fetch_polyhaven_texture

# Poly Haven slugs chosen by browsing api.polyhaven.com/assets?t=textures - see
# README.md "Phase 4 fidelity" for why these specific ones and why no CC0 source
# was used for signage/trees (built procedurally instead - flagged, not hidden).
ASPHALT_SLUG = "asphalt_01"
CONCRETE_SLUG = "pavement_02"
STREETLIGHT_SLUG = "street_lamp_01"

NEAR_RESOLUTION = "4k"
FAR_RESOLUTION = "2k"


def _texture_paths(slug: str, resolution: str) -> dict[str, str] | None:
    paths = fetch_polyhaven_texture(slug, resolution=resolution)
    if paths is None:
        return None
    return {k: str(v) for k, v in paths.items()}


def build_default_theme() -> dict:
    """{"asphalt_near": {...} | None, "asphalt_far": ..., "concrete_near": ...,
    "concrete_far": ..., "streetlight_gltf": str | None}. Fetched once and
    shared across every scenario export for a render (the assets don't vary
    per-scenario) - see scripts/phase4_render_3d.py."""
    return {
        "asphalt_near": _texture_paths(ASPHALT_SLUG, NEAR_RESOLUTION),
        "asphalt_far": _texture_paths(ASPHALT_SLUG, FAR_RESOLUTION),
        "concrete_near": _texture_paths(CONCRETE_SLUG, NEAR_RESOLUTION),
        "concrete_far": _texture_paths(CONCRETE_SLUG, FAR_RESOLUTION),
        "streetlight_gltf": (lambda p: str(p) if p else None)(fetch_polyhaven_model(STREETLIGHT_SLUG)),
    }
