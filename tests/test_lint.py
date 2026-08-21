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
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# `sites` was missing until 2026-08-17, and it is the directory where a rule is most
# likely to be quietly re-invented - the consolidation that removed six duplicated
# constants and four duplicated functions from it left nine unused imports behind,
# and nothing said so. A linter that does not read a directory is not linting it.
TARGETS = ["src", "scripts", "sites", "tests", "conftest.py"]
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


def test_no_readme_section_shadows_a_module():
    """A README section titled with a src/ path duplicates the module's own docstring.

    The module docstring is the home — it lives beside the code someone is editing.
    A README section retelling the same story drifts from it and costs tokens on every
    file read.  This is the exact regression the prose cut was meant to prevent.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    shadowed = [
        line.strip()
        for line in readme.splitlines()
        if line.startswith("#") and "src/" in line
    ]
    assert not shadowed, (
        "README section title(s) contain a src/ path — the module docstring is the home:\n  "
        + "\n  ".join(shadowed)
    )


# The roots the README's tree claims to cover. Asserted rather than assumed: a tree that
# quietly stopped covering a directory would pass a comparison scoped to what it lists.
TREE_ROOTS = ("src/", "scripts/", "sites/")
# `<site>/` entries in the tree are a SCHEMA, not paths - every site has these two files and
# listing six copies would say nothing. Everything else in the tree is a real path.
TREE_PLACEHOLDER_SUFFIXES = ("/config.yaml", "/scenarios.py")
# A DIRECTORY MAY BE BARE - `scripts/` and `sites/` carry no description, and requiring one
# silently reparented every script under src/ the first time this ran.
TREE_ENTRY = re.compile(r"^(\s*)([A-Za-z_][\w.\-]*(?:/|\.\w+))(?:\s\s+(\S.*))?$")


def _readme_tree() -> tuple[dict[str, str], set[str]]:
    """{path: one-line description} and the set of directory prefixes, from README's tree.

    Indentation carries the path, so the prefix is rebuilt from a stack rather than read off
    each line. TREE_ENTRY also does the work of skipping the wrapped second line of a long
    description - `--all adds the section, limiter...` under measure_drawn.py parses as an
    entry named `--all` under any looser pattern, and then the tree appears to list a file
    that cannot exist.
    """
    lines = (REPO_ROOT / "README.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Repo structure"))
    block, inside = [], False
    for line in lines[start:]:
        if line.strip() == "```":
            if inside:
                break
            inside = True
            continue
        if inside:
            block.append(line)
    files, dirs, stack = {}, set(), []
    for line in block:
        match = TREE_ENTRY.match(line)
        if match is None:
            continue
        indent, name, description = len(match[1]), match[2], match[3] or ""
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = (stack[-1][1] if stack else "") + name
        if name.endswith("/"):
            stack.append((indent, full))
            dirs.add(full)
        else:
            files[full] = description.strip()
    return files, dirs


def test_the_readme_tree_is_the_tree_on_disk():
    """Every module under src/ and scripts/ is in README's tree, and nothing else is.

    THE TREE IS THE ONLY INDEX OF THIS REPO, and an index nobody checks is worse than none -
    it is believed. Two failures, both from one session: `treatments/bikeways.py` was split
    into a package and the old 1552-line file stayed on disk, dead and shadowed, listed in the
    tree as though it were the live one; and `scripts/whatis.py` - a tool written to answer
    "what is this symbol" - was never added to the tree at all, so the answer to "how do I
    find my way around this code" was itself unfindable. Nine other modules were missing.

    WHAT THIS DOES NOT CHECK IS THE PROSE. The one-liners are editorial - they are the
    docstring's first line cut to fit a tree, and no generator writes them well. So the
    structure is derived and enforced, the description is written by whoever adds the file,
    and the enforcement is what makes them write it: a new module cannot land without a line.
    """
    listed, listed_dirs = _readme_tree()
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    on_disk = {p for p in tracked
               if p.startswith(TREE_ROOTS) and p.endswith((".py", ".sh"))
               and not p.endswith(TREE_PLACEHOLDER_SUFFIXES)}
    assert on_disk, "found no modules to check - is this a git checkout?"

    def covered(path: str) -> bool:
        # A directory entry stands for its own __init__.py: the package's docstring is the
        # thing the tree's line about the directory is already summarising.
        package = path.rsplit("/", 1)[0] + "/"
        return path in listed or (path.endswith("/__init__.py") and package in listed_dirs)

    bare = sorted(path for path, description in listed.items() if not description)
    assert not bare, (
        "tree entr(ies) with no description - the path alone is what the filesystem already "
        f"says, and the line is there for the other half: {bare}")

    missing = sorted(p for p in on_disk if not covered(p))
    # Ghosts are checked against the filesystem, not against `on_disk`: the tree lists a few
    # things that are neither .py nor .sh (sites/README.md) and they are not ghosts.
    ghosts = sorted(p for p in listed
                    if not (REPO_ROOT / p).exists()
                    and not p.endswith(TREE_PLACEHOLDER_SUFFIXES))
    assert not (missing or ghosts), (
        "README.md's `## Repo structure` tree disagrees with the checkout. It is the only "
        "index of this repo, so a wrong one costs the next reader a session:\n"
        + "".join(f"\n  ON DISK, NOT IN THE TREE: {p}" for p in missing)
        + "".join(f"\n  IN THE TREE, NOT ON DISK: {p}" for p in ghosts)
        + "\n\nAdd the line beside its neighbours, or delete it. One line, cut from the "
          "module's own docstring - `.venv/bin/python scripts/whatis.py <module>` prints it."
    )


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
