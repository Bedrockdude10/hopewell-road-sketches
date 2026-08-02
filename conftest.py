"""Root conftest: catch "ran pytest with the wrong interpreter" before collection.

This exists because of one specific, repeatable mistake. Running `python -m pytest` from
the repo root WITHOUT `source .venv/bin/activate` picks up whatever `python` is on PATH -
here a pyenv 3.13.0 that happens to have pytest installed but none of this project's
scientific stack. Collection then fails five times over, once per test module, with a wall
of `ModuleNotFoundError: No module named 'geopandas'` tracebacks pointing into
src/geometry/model.py. Nothing in that output says "wrong interpreter"; it reads like the
repo is broken, and the obvious next move (debugging the imports) is the wrong one.

So: probe for the project's third-party imports here, in the root conftest, which pytest
loads before it imports any test module. If they're missing AND a working .venv sits at the
repo root, stop with a single message naming both interpreters and the exact command.

Deliberately narrow so this never fires in a valid environment:
  * deps present  -> no-op, whatever the interpreter (covers CI, activated venv, a global
                     install with no .venv at all).
  * deps missing and no usable .venv -> no-op, fall through to pytest's normal errors. A
                     fresh clone that has never run `pip install -r requirements.txt` has a
                     genuinely missing dependency, not an interpreter mix-up, and inventing
                     a venv path that doesn't exist would send them somewhere useless.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Top-level third-party imports that src/ does at module scope, so a missing one is fatal at
# collection rather than at some later test. geopandas is the one that actually blows up
# first (src/geometry/model.py line 5), the rest are here so the message lists everything
# that's wrong instead of dripping them out one re-run at a time.
REQUIRED = ("geopandas", "shapely", "matplotlib", "numpy", "pyproj", "yaml", "trimesh")


def _missing(finder) -> list[str]:
    out = []
    for name in REQUIRED:
        try:
            if finder(name) is None:
                out.append(name)
        except (ImportError, ValueError):
            out.append(name)
    return out


def _venv_has(name: str) -> bool:
    """Is `name` importable by the venv interpreter, judged from the filesystem?

    Checked by looking for it in the venv's site-packages rather than by shelling out to
    `.venv/bin/python -c "import ..."` - that would cost a subprocess (and a geopandas
    import, which is not cheap) on every single pytest run, in the common case where none of
    this matters.
    """
    for site_packages in (REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"):
        if (site_packages / name).exists() or list(site_packages.glob(f"{name}*.py")):
            return True
        if (site_packages / f"{name}.py").exists():
            return True
    return False


def pytest_configure(config):
    """Raised from a hook, not at module scope.

    A bare `raise` at import time also stops the run, but pytest reports it as "ImportError
    while loading conftest" wrapped around a traceback frame - which reintroduces exactly the
    "something is broken in this repo" reading we're trying to remove. From pytest_configure
    a UsageError prints as a plain `ERROR:` block and nothing else, and it still runs before
    any test module (or tests/conftest.py) is imported, so the five tracebacks never happen.
    """
    missing = _missing(importlib.util.find_spec)
    if not missing:
        return
    if not (VENV_PYTHON.exists() and _venv_has("geopandas")):
        return

    raise pytest.UsageError(
        "Wrong Python interpreter - this one has pytest but not the project's dependencies.\n"
        "\n"
        f"  running under : {sys.executable}\n"
        f"  missing       : {', '.join(missing)}\n"
        f"  project venv  : {VENV_PYTHON}\n"
        "\n"
        "Run the tests with the venv's interpreter instead:\n"
        "\n"
        "  ./scripts/test.sh\n"
        "\n"
        "or, equivalently:\n"
        "\n"
        "  .venv/bin/python -m pytest\n"
        "\n"
        "(`source .venv/bin/activate` first also works, and then plain `python -m pytest`.)"
    )
