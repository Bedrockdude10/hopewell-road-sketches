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
    "nor_gl": path} or None if the asset/network is unavailable."""
    manifest = _get_json(f"{POLYHAVEN_API}/files/{slug}")
    if manifest is None:
        return None

    out = {}
    for map_name in maps:
        try:
            file_info = manifest[map_name][resolution]["jpg"]
        except KeyError:
            return None
        dest = CACHE_DIR / slug / resolution / f"{slug}_{map_name}_{resolution}.jpg"
        if not _download(file_info["url"], dest):
            return None
        out[map_name] = dest
    return out


def fetch_polyhaven_model(slug: str, resolution: str = "1k") -> Path | None:
    """Download (or reuse cached) a Poly Haven model as a glTF bundle (the .gltf
    JSON + its .bin + referenced textures, preserving the relative folder layout
    the glTF expects). Returns the local path to the .gltf file, or None."""
    manifest = _get_json(f"{POLYHAVEN_API}/files/{slug}")
    if manifest is None:
        return None
    try:
        gltf_entry = manifest["gltf"][resolution]["gltf"]
    except KeyError:
        return None

    model_dir = CACHE_DIR / "models" / f"{slug}_{resolution}"
    gltf_path = model_dir / f"{slug}_{resolution}.gltf"
    manifest_path = model_dir / "_manifest.json"  # marks a fully-downloaded bundle

    if manifest_path.exists():
        return gltf_path

    if not _download(gltf_entry["url"], gltf_path):
        return None
    for rel_path, file_info in gltf_entry.get("include", {}).items():
        if not _download(file_info["url"], model_dir / rel_path):
            return None

    manifest_path.write_text(json.dumps({"slug": slug, "resolution": resolution}))
    return gltf_path
