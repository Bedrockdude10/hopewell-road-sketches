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

    # Check the interpreter that command would USE, because below 3.12 it cannot succeed and
    # says so in terms of the wrong thing: numpy 2.5.0 ships no wheel under 3.12, so pip
    # reports "Could not find a version that satisfies the requirement numpy==2.5.0" over a
    # version list stopping at 2.4.6. That reads as a bad pin in requirements.txt, and the
    # next move - editing the pin - is wrong. Same failure shape as the wrong-interpreter case
    # in conftest.py, so it gets the same treatment: name the interpreter, not the symptom.
    if command -v python3 >/dev/null 2>&1 \
       && ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        FOUND="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "unknown")"
        echo >&2
        echo "  ...but python3 on PATH is $FOUND, and requirements.txt needs >= 3.12 (numpy 2.5.0" >&2
        echo "  publishes no wheel below it). The numpy pin is correct - do not edit it." >&2

        # Name a working interpreter rather than leaving the caller to hunt for one: a
        # container with python3 -> 3.11 often still ships 3.12/3.13 alongside it.
        for V in 3.13 3.12; do
            if command -v "python$V" >/dev/null 2>&1; then
                echo >&2
                echo "  python$V is on PATH here - use it instead:" >&2
                echo "    python$V -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
                break
            fi
        done
    fi
    exit 1
fi

# cd so pytest.ini / rootdir resolve the same way no matter where this was invoked from.
cd "$REPO_ROOT"

# -n auto: the suite is real geometry and matplotlib work, not startup - 128 s serially, 31 s
# across 8 workers with identical results. It goes before "$@" so a caller can override it
# (`./scripts/test.sh -n 0` to debug with a single process, which -x and pdb need).
exec "$VENV_PYTHON" -m pytest -n auto "$@"
