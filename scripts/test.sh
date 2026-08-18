#!/usr/bin/env bash
# Run the test suite with the project venv's interpreter, activated or not.
#
# The habit this exists for is typing `python -m pytest` at the repo root. Without
# `source .venv/bin/activate` that resolves to whatever python is on PATH; if that one
# happens to have pytest but not geopandas, collection dies in a way that looks like a
# broken repo rather than a wrong interpreter (see conftest.py at the repo root, which
# catches the case and says so). This script sidesteps it entirely - it never consults PATH.
#
# Takes the same arguments as pytest:  ./scripts/test.sh -k traced_curbs -x
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No venv at $REPO_ROOT/.venv - create one first:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# cd so pytest.ini / rootdir resolve the same way no matter where this was invoked from.
cd "$REPO_ROOT"

# -n auto: the suite is real geometry and matplotlib work, not startup - 128 s serially, 31 s
# across 8 workers with identical results. It goes before "$@" so a caller can override it
# (`./scripts/test.sh -n 0` to debug with a single process, which -x and pdb need).
exec "$VENV_PYTHON" -m pytest -n auto "$@"
