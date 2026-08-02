"""Static checks for the errors that only surface on a path nobody ran.

This exists because of a specific miss. Collapsing a multi-line import in
scripts/build_all.py dropped `fetch_buildings`, which is referenced only under
`--render-3d`. Every test passed, a full 2D build passed, and the break surfaced minutes
later in a worker process - as a NameError wrapped in a ProcessPoolExecutor traceback,
which is about the least legible way an error can arrive.

Nothing dynamic was needed to catch it: pyflakes reports an undefined name in under a
second, without importing anything or running Blender. That is the right shape of guard for
a repo with slow, rarely-exercised branches (3D rendering, the network paths, one scenario
per site that only one site defines).
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["src", "scripts", "tests", "conftest.py"]

# pyflakes reports real errors and a few stylistic notes. Only the first kind is worth
# failing a build over here; the rest are noise in a codebase that deliberately imports for
# re-export (src/render/props.py) and shadows names in comprehensions.
FATAL = ("undefined name", "may be undefined")


def _pyflakes() -> list[str]:
    result = subprocess.run([sys.executable, "-m", "pyflakes", *TARGETS],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):   # 1 just means "findings"; anything else is broken
        pytest.skip(f"pyflakes unavailable: {result.stderr.strip()[:200]}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_no_undefined_names():
    """An undefined name is a guaranteed runtime crash on whatever path reaches it."""
    findings = [line for line in _pyflakes() if any(f in line for f in FATAL)]
    assert not findings, (
        "undefined name(s) - these WILL crash when that branch runs:\n  "
        + "\n  ".join(findings))


def test_no_syntax_errors_anywhere():
    """Every module parses, including ones no test imports."""
    import ast

    broken = []
    for target in TARGETS:
        path = REPO_ROOT / target
        files = path.rglob("*.py") if path.is_dir() else [path]
        for source_file in files:
            try:
                ast.parse(source_file.read_text())
            except SyntaxError as e:
                broken.append(f"{source_file.relative_to(REPO_ROOT)}:{e.lineno}: {e.msg}")
    assert not broken, "\n  ".join(["syntax errors:"] + broken)
