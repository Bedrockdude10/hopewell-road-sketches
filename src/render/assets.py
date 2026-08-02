"""Fetch + disk-cache CC0 assets from Poly Haven's public API (api.polyhaven.com)
for Phase 4 render fidelity. Poly Haven's ToS asks for a unique User-Agent per
application - reuses the same one as Overpass/Nominatim (src/sources/data_loader.py).

Every fetch function returns None on failure rather than raising - a missing
texture/model must never hard-fail scripts/phase4_render_3d.py when there's no
network access. Callers (blender_scene.py) fall back to flat colors / procedural
geometry - see README.md "Phase 4 fidelity" section for what's real vs. procedural."""
import json
from pathlib import Path

import requests

from src.sources.data_loader import NOMINATIM_USER_AGENT

POLYHAVEN_API = "https://api.polyhaven.com"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "output" / ".textures"  # src/render/assets.py -> repo root
HEADERS = {"User-Agent": NOMINATIM_USER_AGENT}
TIMEOUT = 30


def _get_json(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


# One manifest per slug, per process. Poly Haven's /files/<slug> response covers every
# resolution of an asset, but this module asks for one resolution at a time - so the same
# manifest was being fetched once per resolution (3 of every 7 requests were exact
# duplicates within a single build_default_theme call).
_manifests: dict[str, dict | None] = {}


def _manifest(slug: str) -> dict | None:
    if slug not in _manifests:
        _manifests[slug] = _get_json(f"{POLYHAVEN_API}/files/{slug}")
    return _manifests[slug]


def _texture_dest(slug: str, resolution: str, map_name: str) -> Path:
    """Where a texture map lands on disk. Fully determined by (slug, resolution, map) - no
    manifest needed - which is what lets a warm cache skip the network entirely."""
    return CACHE_DIR / slug / resolution / f"{slug}_{map_name}_{resolution}.jpg"


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except requests.exceptions.RequestException:
        return False


def fetch_polyhaven_texture(slug: str, resolution: str = "2k",
                             maps: tuple[str, ...] = ("Diffuse", "Rough", "nor_gl")) -> dict[str, Path] | None:
    """Download (or reuse cached) Diffuse/Roughness/Normal jpgs for a Poly Haven
    texture at the given resolution. Returns {"Diffuse": path, "Rough": path,
    "nor_gl": path} or None if the asset/network is unavailable.

    Checks the disk BEFORE the network. The manifest fetch used to come first, so a fully
    cached theme still made 7 HTTPS round trips per call - asking the API to describe files
    already sitting in output/.textures - and 3D rendering couldn't work offline at all.
    """
    dests = {map_name: _texture_dest(slug, resolution, map_name) for map_name in maps}
    if all(dest.exists() for dest in dests.values()):
        return dests

    manifest = _manifest(slug)
    if manifest is None:
        return None

    out = {}
    for map_name, dest in dests.items():
        try:
            file_info = manifest[map_name][resolution]["jpg"]
        except KeyError:
            return None
        if not _download(file_info["url"], dest):
            return None
        out[map_name] = dest
    return out


def fetch_polyhaven_model(slug: str, resolution: str = "1k") -> Path | None:
    """Download (or reuse cached) a Poly Haven model as a glTF bundle (the .gltf
    JSON + its .bin + referenced textures, preserving the relative folder layout
    the glTF expects). Returns the local path to the .gltf file, or None."""
    model_dir = CACHE_DIR / "models" / f"{slug}_{resolution}"
    gltf_path = model_dir / f"{slug}_{resolution}.gltf"
    manifest_path = model_dir / "_manifest.json"  # marks a fully-downloaded bundle

    # The completed-bundle marker is checked first now. This test already existed but sat
    # BELOW the manifest fetch, so it saved the file downloads and nothing else.
    if manifest_path.exists():
        return gltf_path

    manifest = _manifest(slug)
    if manifest is None:
        return None
    try:
        gltf_entry = manifest["gltf"][resolution]["gltf"]
    except KeyError:
        return None

    if not _download(gltf_entry["url"], gltf_path):
        return None
    for rel_path, file_info in gltf_entry.get("include", {}).items():
        if not _download(file_info["url"], model_dir / rel_path):
            return None

    manifest_path.write_text(json.dumps({"slug": slug, "resolution": resolution}))
    return gltf_path
