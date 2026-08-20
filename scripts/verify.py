"""The whole verification loop in one command, so a change costs one round trip.

Run by hand the loop is eight commands - export the before side, export the after side, diff
them, run the suite - and, worse, eight judgements, the last of which is always "which of
these failures were already here?". This runs all of it at once and prints one verdict.

    scripts/verify.py --no-tests            # THE LOOP: six sites, both sides, ~22 s
    scripts/verify.py --no-tests --site broad_st_greenwood    # one junction, ~6 s
    scripts/verify.py --site X -k traced_curbs   # ...and the tests that could see it, ~20 s
    scripts/verify.py                       # everything, serially: ~5 min
    scripts/verify.py --base main           # against another revision
    scripts/verify.py --record              # re-record the known-failure baseline

ONE STEP AT A TIME, AND WHY. Every step here is a subprocess, and running the exports beside
the suite overlapped nicely on paper: 72 s against 76 s serial. It is off anyway. This is a
36 GB machine with Blender, an editor and more than one agent session on it, and the operator
was watching it hit OOM while the per-worker measurements (0.21 GB an export worker, 0.27 GB a
pytest worker) said there was room to spare. A four-second saving is not worth a run that
cannot be trusted to finish, so the default is serial everywhere - `--jobs` buys the
parallelism back for the suite when you know the machine is quiet.

WHY THE BASELINE. A suite you did not turn red costs more than a slow one: every run you have
to work out again which failures are yours. So the failing set is recorded (in
output/.verify/baseline.json) and every run reports NEW, KNOWN and FIXED against it. NEW is
the only number that says anything about the change in front of you. Re-record with --record
once you have decided the red you inherited is not yours to fix - and note that a baseline is
a claim about somebody else's mess, so it is stamped with the revision and time it was taken.

Both halves of that comparison are taken against the tests that ACTUALLY RAN, never against
the whole baseline, because -k deselects most of the suite and a skip runs nothing either.
Compared against the whole baseline, `-k traced_curbs` congratulated itself with FIXED 9 for
nine goldens it had not so much as executed. They are reported as NOT RUN instead: the honest
answer, which is that this run says nothing about them.

WHY THE INPUTS ARE WIRED IN TWO WAYS. The before side runs in a git worktree at --base, and
README's loop warns you to symlink the gitignored data/ into it or every run dies in 0.6 s and
you read that as a result. This does both: HOPEWELL_DATA_DIR and HOPEWELL_OSM_CACHE are passed
as absolute paths out of THIS checkout, AND data/ is symlinked into the worktree - pointing at
the committed clip, not at the 391 MB download. Belt and braces because the env var is only
honoured by revisions that know about it: run this against a base that predates
src/sources/data_loader.py's HOPEWELL_DATA_DIR and it reads plain data/, finds nothing, and
reports a crash where you were looking for a geometry diff. Either route lands on the same
bytes, which is the only way the diff means "my code moved this" and not "the inputs moved".

The worktree lives in the system temp directory, NOT under the repo: git supports a worktree
nested inside its own checkout, but a second full copy of src/ inside the project is a copy
every editor, indexer and file watcher will find. It is reused between runs (`git checkout
--detach` on the second call), so only the first run pays for creating it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.jobs import MAX_BUILD_JOBS

VERIFY_DIR = REPO_ROOT / "output" / ".verify"
BASELINE = VERIFY_DIR / "baseline.json"
# One reused worktree, not one per revision: two copies of this repo on disk is already
# generous, and switching an existing worktree to another rev is faster than making a new one.
WORKTREE = Path(tempfile.gettempdir()) / "hopewell-verify-base"
# Copied into the worktree so both sides are measured by the same instrument: the export driver
# and the modules under scripts/ it imports that hold no design decisions. EVERY ONE OF THEM,
# not just the driver - jobs.py is untracked, so the worktree had no copy, and the before side
# died on `No module named scripts.jobs` and reported all 15 exports as changed.
HARNESS_FILES = ("export_all_scenarios.py", "jobs.py")


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True)


def hermetic_env() -> dict:
    """The inputs held constant across both sides of the comparison.

    Absolute, and out of THIS checkout - see the module docstring. HOPEWELL_OFFLINE matters as
    much as the rest: an Overpass fetch on one side and a cache hit on the other is a diff that
    is about the internet.
    """
    return {
        "HOPEWELL_OFFLINE": "1",
        "HOPEWELL_OSM_CACHE": str(REPO_ROOT / "tests" / "fixtures" / "osm_cache"),
        "HOPEWELL_DATA_DIR": str(REPO_ROOT / "tests" / "fixtures" / "data"),
    }


def prepare_worktree(base: str) -> tuple[Path, str]:
    """A checkout of `base` to export the before side from. Returns (path, short sha)."""
    sha = run(["git", "rev-parse", base], REPO_ROOT).stdout.strip()
    if not sha:
        raise SystemExit(f"verify: no such revision: {base}")
    short = sha[:7]
    if (WORKTREE / ".git").exists():
        checkout = run(["git", "-C", str(WORKTREE), "checkout", "--detach", sha], REPO_ROOT)
        if checkout.returncode:
            # A worktree left half-switched is worse than none: delete and remake rather than
            # export from a revision nobody asked for.
            shutil.rmtree(WORKTREE, ignore_errors=True)
            run(["git", "worktree", "prune"], REPO_ROOT)
    if not (WORKTREE / ".git").exists():
        made = run(["git", "worktree", "add", "--detach", str(WORKTREE), sha], REPO_ROOT)
        if made.returncode:
            raise SystemExit(f"verify: could not create the worktree:\n{made.stderr}")
    # The MEASURING INSTRUMENT is held constant along with the data; only src/ differs between
    # the two sides. Otherwise the before side runs whatever driver existed at `base`, and the
    # first thing that breaks is a flag this tool passes that the old one has never heard of -
    # which reads as "the before side failed" rather than "you compared two harnesses".
    for name in HARNESS_FILES:
        shutil.copy2(REPO_ROOT / "scripts" / name, WORKTREE / "scripts" / name)
    link_inputs(WORKTREE)
    return WORKTREE, short


def link_inputs(tree: Path) -> None:
    """Point the worktree's gitignored inputs at this checkout's.

    data/ is aimed at the COMMITTED CLIP where there is one, so an old revision reading plain
    data/ and a new one reading HOPEWELL_DATA_DIR see the same bytes; the clip mirrors data/'s
    layout exactly, which is what makes that substitution invisible. Without a clip it falls
    back to the real download. Symlinks, not copies: 391 MB, and they are read-only here.
    """
    fixture = REPO_ROOT / "tests" / "fixtures" / "data"
    for link, target in ((tree / "data", fixture if fixture.exists() else REPO_ROOT / "data"),
                         (tree / "tests" / "fixtures" / "data", fixture)):
        if not target.exists() or link.exists() or link.is_symlink():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)


def export(tree: Path, out_dir: Path, sites: list[str] | None, jobs: int) -> tuple[bool, str]:
    """Every scenario of every site, from the code in `tree`, into `out_dir`.

    `jobs` is passed in rather than defaulted, because the caller knows how many other sides
    of the comparison are running at the same time - see scripts/jobs.py.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [str(REPO_ROOT / ".venv" / "bin" / "python"),
           str(tree / "scripts" / "export_all_scenarios.py"), str(out_dir), "--jobs", str(jobs)]
    for site in sites or []:
        cmd += ["--site", site]
    done = run(cmd, cwd=tree, env=hermetic_env())
    return done.returncode == 0, done.stdout + done.stderr


def failing_tests(k: str | None, jobs: str) -> tuple[set[str], set[str], str, bool]:
    """(failing node ids, node ids that RAN, the one-line count summary, whether pytest ran).

    The set that ran matters as much as the set that failed. Under -k most of the suite is
    deselected, and a baseline failure that was never executed is not a failure you fixed -
    reported as FIXED, which is what this did first, it credits the change with nine repairs it
    did not make.

    Deliberately -p no:randomly. The comparison against the baseline is by node id so order
    does not affect it, but a random order makes a genuinely order-dependent failure surface as
    NEW on one run and vanish on the next, and this tool's whole value is that NEW means
    something. Run ./scripts/test.sh for the shuffled order.
    """
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    xml = VERIFY_DIR / "last-run.xml"
    xml.unlink(missing_ok=True)
    # NO -q: pytest.ini's addopts already carries one, and a second makes it -qq, which drops
    # the "N failed, M passed" line entirely. That read as "pytest printed no summary".
    cmd = [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "-n", jobs,
           "-p", "no:randomly", "--tb=no", f"--junit-xml={xml}"]
    if k:
        cmd += ["-k", k]
    done = run(cmd, cwd=REPO_ROOT)
    tail = [line for line in done.stdout.splitlines() if " passed" in line or " failed" in line
            or " error" in line]
    summary = tail[-1].strip() if tail else "(pytest printed no summary)"
    if not xml.exists():
        return set(), set(), f"pytest did not run: {done.stdout[-500:]}{done.stderr[-500:]}", False
    failed, ran = set(), set()
    for case in ET.parse(xml).getroot().iter("testcase"):
        # `file` + `name` reconstruct the node id you can paste back into pytest; classname is
        # dotted and loses the difference between a package and a module.
        where = case.get("file") or (case.get("classname") or "").replace(".", "/") + ".py"
        node = f"{where}::{case.get('name')}"
        # A skip is not a run: it tells you nothing about whether the test would pass, so a
        # baseline failure that skipped today has not been fixed either.
        if case.find("skipped") is not None:
            continue
        ran.add(node)
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(node)
    return failed, ran, summary, True


def load_baseline() -> dict:
    if not BASELINE.exists():
        return {}
    try:
        return json.loads(BASELINE.read_text())
    except json.JSONDecodeError:
        return {}


def save_baseline(failed: set[str]) -> dict:
    head = run(["git", "rev-parse", "--short", "HEAD"], REPO_ROOT).stdout.strip()
    dirty = bool(run(["git", "status", "--porcelain"], REPO_ROOT).stdout.strip())
    record = {
        "failures": sorted(failed),
        "recorded": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rev": head + (" + uncommitted changes" if dirty else ""),
    }
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(record, indent=2) + "\n")
    return record


def rule(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD",
                        help="revision to compare the working tree against (default HEAD)")
    parser.add_argument("--site", action="append",
                        help="limit the EXPORT comparison to this site (repeatable)")
    parser.add_argument("-k", dest="k", help="pass through to pytest, to narrow the suite")
    parser.add_argument("--jobs", default="0",
                        help="pytest -n value (default 0, meaning one process - see the "
                             "docstring on why nothing here runs in parallel by default). "
                             "`auto` takes the suite from ~5 min to ~60 s across 12 workers "
                             "at a measured 0.27 GB each, when the machine is quiet enough.")
    parser.add_argument("--no-tests", action="store_true", help="exports only")
    parser.add_argument("--no-exports", action="store_true", help="suite only")
    parser.add_argument("--record", action="store_true",
                        help="record this run's failures as the known baseline")
    args = parser.parse_args()

    started = time.perf_counter()
    before_dir, after_dir = VERIFY_DIR / "before", VERIFY_DIR / "after"
    short = ""

    # One at a time, cheapest first - see the module docstring. Exports before tests because
    # they are 17 s against minutes and they run export_scenario, so a change that breaks an
    # invariant outright shows up here before you have waited for the suite to say so.
    export_results = None
    if not args.no_exports:
        tree, short = prepare_worktree(args.base)
        export_results = [export(tree, before_dir, args.site, MAX_BUILD_JOBS),
                          export(REPO_ROOT, after_dir, args.site, MAX_BUILD_JOBS)]
    test_results = None if args.no_tests else failing_tests(args.k, args.jobs)
    moved = 0

    # VOID means a side produced nothing to compare. NOT "a side reported a failure": one bad
    # junction failing identically on both sides is a standing condition of this repo, and
    # export_all_scenarios tolerates it on purpose so it does not cost you the other five.
    void = export_results is not None and not all(
        any(d.rglob("*.json")) for d in (before_dir, after_dir))
    partial = export_results is not None and not all(ok for ok, _ in export_results)
    if export_results:
        rule(f"exports: working tree vs {args.base} ({short})")
        for label, (ok, output) in zip(("before", "after"), export_results):
            counts = [line for line in output.splitlines() if "export(s) under" in line]
            print(f"  {label:7s} {counts[-1] if counts else 'no exports written'}")
            if not ok:
                # Whole tail, unfiltered. A `before` side that wrote nothing is the failure mode
                # this tool exists to make loud: filtering it to the lines that matched a
                # keyword once printed "1 FAILED" and not one word about the cause.
                print(f"    {label} FAILED:")
                print("\n".join(f"      {line}" for line in output.strip().splitlines()[-12:]))
        diff = run([str(REPO_ROOT / ".venv" / "bin" / "python"),
                    str(REPO_ROOT / "scripts" / "diff_exports.py"),
                    str(before_dir), str(after_dir)], cwd=REPO_ROOT)
        print("\n".join(f"  {line}" for line in diff.stdout.strip().splitlines()))
        moved = diff.returncode

    new_failures: set[str] = set()
    if test_results:
        failed, ran_nodes, summary, ran = test_results
        rule("tests")
        print(f"  {summary}")
        if not ran:
            return 2
        baseline = load_baseline()
        if not baseline and not args.record:
            save_baseline(failed)
            print(f"  no baseline existed - recorded these {len(failed)} failure(s) as known.\n"
                  f"  THIS RUN CANNOT TELL YOU WHAT YOU BROKE. Re-run to compare against it.")
        else:
            known = set(baseline.get("failures", []))
            new_failures = failed - known
            # FIXED and NOT RUN are both computed against `ran_nodes`, never against the whole
            # baseline - see failing_tests.
            print(f"  NEW    {len(new_failures):3d}" +
                  ("  <- the only number about your change" if new_failures else ""))
            print(f"  KNOWN  {len(failed & known):3d}  (baseline {baseline.get('rev', '?')}, "
                  f"taken {baseline.get('recorded', '?')})")
            print(f"  FIXED  {len((known & ran_nodes) - failed):3d}")
            if known - ran_nodes:
                print(f"  NOT RUN{len(known - ran_nodes):4d}  of the baseline's failures - "
                      f"deselected or skipped, so this run says nothing about them")
            for node in sorted(new_failures)[:15]:
                print(f"    NEW  {node}")
            if len(new_failures) > 15:
                print(f"    ... and {len(new_failures) - 15} more")
        if args.record:
            record = save_baseline(failed)
            print(f"  recorded {len(record['failures'])} failure(s) as the baseline")

    rule("verdict")
    if void:
        # Loudly, and NOT as a diff. A side that wrote nothing makes every export on the other
        # side read as "only in after", which prints as "15 of 15 export(s) differ" - the most
        # alarming possible way to say "this tool did not run".
        print("  THE EXPORT COMPARISON IS VOID: one side wrote nothing (see its traceback\n"
              "  above). Every 'only in after' line above is that failure, not a change.")
        print(f"  {time.perf_counter() - started:.0f}s")
        return 2
    if partial:
        print("  one or more sites failed to export ON BOTH SIDES - see the tracebacks above. "
              "The\n  sites that did export are still compared; those that did not are not "
              "spoken for.")
    if new_failures:
        print(f"  {len(new_failures)} NEW test failure(s) - yours. See the list above.")
    elif test_results:
        print("  no new test failures.")
    if moved:
        print("  exports moved. A moved number is not automatically a bug: read every line "
              "above\n  and confirm each one is a number you meant to move.")
    elif export_results:
        print("  no export moved: no marking, prop or note changed anywhere.")
    hint = "  (--no-tests is the ~22 s version of this)" if not (
        args.no_tests or args.no_exports or args.k) else ""
    print(f"  {time.perf_counter() - started:.0f}s{hint}")
    return 1 if new_failures else 0


if __name__ == "__main__":
    sys.exit(main())
