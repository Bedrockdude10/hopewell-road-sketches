"""Static checks for the errors that only surface on a path nobody ran.

This exists because of a specific miss. Collapsing a multi-line import in
scripts/build_all.py dropped `fetch_buildings`, which is referenced only under
`--render-3d`. Every test passed, a full 2D build passed, and the break surfaced minutes
later in a worker process - as a NameError wrapped in a ProcessPoolExecutor traceback,
which is about the least legible way an error can arrive.

Nothing dynamic was needed to catch it: a static pass reports an undefined name in under a
second, without importing anything or running Blender. That is the right shape of guard for
a repo with slow, rarely-exercised branches (3D rendering, the network paths, one scenario
per site that only one site defines).

Two checkers run here, both configured at the repo root and both treated the same way - if the
tool cannot run, that is a failure, not a pass:

  * RUFF (ruff.toml), over every .py file, for what one file gets wrong on its own.
  * IMPORT-LINTER (.importlinter), over the import graph, for what no single file can show you -
    a blender script reaching into the venv, a config read dragging in shapely, geometry
    importing the thing that draws it.

Two things changed when ruff replaced a `python -m pyflakes` subprocess:

  * FINDINGS ARE RULE CODES, NOT SUBSTRINGS. The old version decided what was fatal by
    matching "undefined name" against stdout, so a reworded message would have silently
    retired the guard.
  * A MISSING CHECKER IS A FAILURE, NOT A PASS. This is the bug that mattered. pyflakes was
    never in requirements.txt - it happened to be in the working venv. On any other machine
    `python -m pyflakes` exited 1 with its complaint on STDERR; the old code read exit 1 as
    "ran fine, had findings", found an empty stdout, and reported success. The guard against
    silent breakage was itself silently broken. So: the checker is pinned in requirements.txt
    - the same one every other dependency is in, deliberately not a separate dev file - and if
    it cannot be run this file FAILS and says how to install it.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["src", "scripts", "tests", "conftest.py"]
RUFF = Path(sys.executable).parent / "ruff"
LINT_IMPORTS = Path(sys.executable).parent / "lint-imports"

# The subset that is a guaranteed crash on whatever path reaches it, as opposed to the rest
# of ruff.toml's selection, which is code that works but shouldn't be written that way.
CRASHING = {
    "F821",   # undefined name
    "F822",   # undefined name in __all__
    "F823",   # local variable referenced before assignment
    "E999",   # syntax error - the file does not parse at all
}


def _ruff(*args) -> list[dict]:
    """Every ruff finding over TARGETS, as parsed JSON.

    Raises rather than returns on anything that isn't ruff working normally - see the module
    docstring for why a checker that can't run must not read as a clean checkout.
    """
    if not RUFF.exists():
        raise AssertionError(
            f"ruff is not installed in this interpreter's environment ({sys.executable}).\n"
            f"  expected: {RUFF}\n\n"
            "This test cannot pass without it - a lint guard that skips itself when the "
            "linter is missing is how the bug it exists to catch got shipped. Install the "
            "dev tooling:\n\n"
            "  .venv/bin/pip install -r requirements.txt"
        )
    result = subprocess.run([str(RUFF), "check", "--output-format=json", *args, *TARGETS],
                             cwd=REPO_ROOT, capture_output=True, text=True)
    # 0 = clean, 1 = findings. Anything else (2 = bad config/arguments) means the result says
    # nothing about the code, so it must not be read as an absence of findings.
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"ruff could not run (exit {result.returncode}) - this is a tooling failure, not a "
            f"clean checkout:\n{result.stderr.strip()[:1000]}"
        )
    return json.loads(result.stdout or "[]")


def _format(findings: list[dict]) -> str:
    return "\n  ".join(
        f"{Path(f['filename']).relative_to(REPO_ROOT)}:{(f.get('location') or {}).get('row', '?')}: "
        f"{f['code'] or 'E999'} {f['message']}"
        for f in findings
    )


def test_no_undefined_names():
    """An undefined name is a guaranteed runtime crash on whatever path reaches it."""
    findings = [f for f in _ruff() if (f["code"] or "E999") in CRASHING]
    assert not findings, (
        "undefined name(s) - these WILL crash when that branch runs:\n  " + _format(findings))


def test_lint_clean():
    """Everything else ruff.toml selects. Each ignore in that file is an argued exception, so
    a finding here is a rule the project decided it wanted, firing on new code."""
    findings = [f for f in _ruff() if (f["code"] or "E999") not in CRASHING]
    assert not findings, (
        f"{len(findings)} lint finding(s) - fix, or argue the rule down in ruff.toml:\n  "
        + _format(findings))


def test_import_contracts_hold():
    """The architectural rules in .importlinter - which one file's imports cannot show you.

    Each of the three is a rule the README already stated in prose, and each has the same shape:
    an import added in the wrong place works perfectly for whoever added it and breaks on a path
    they were not running. The blender one is the sharpest - `scripts/blender/*.py` execute in
    Blender's own interpreter, so importing anything from the venv is an error that first appears
    minutes into a 3D render, inside a subprocess.

    Runs the checker as a subprocess for the same reason the ruff tests do, and fails the same
    way if it cannot run: a contract nobody checked is not a contract that held.
    """
    if not LINT_IMPORTS.exists():
        raise AssertionError(
            f"import-linter is not installed in this interpreter's environment ({sys.executable}).\n"
            f"  expected: {LINT_IMPORTS}\n\n"
            "Install it with everything else:\n\n"
            "  .venv/bin/pip install -r requirements.txt"
        )
    result = subprocess.run([str(LINT_IMPORTS)], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        return
    # Its own report is already the readable thing - contract by contract, with the offending
    # import chain and line numbers under each - so it is passed through rather than reformatted.
    report = result.stdout.strip() or result.stderr.strip()
    raise AssertionError(
        "import contract(s) broken - see .importlinter for what each rule is for:\n\n"
        + "\n".join(line for line in report.splitlines() if not line.startswith(("╔", "╚", "║", " ║", "  └", "      ╚")))
    )
