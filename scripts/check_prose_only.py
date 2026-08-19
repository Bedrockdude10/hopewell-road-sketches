#!/usr/bin/env python3
"""Prove a commit changed only prose - comments and docstrings - and no code.

WHY THIS EXISTS. Cutting prose across 23,742 lines of src/ is safe in principle: this repo
enables no doctests, ruff selects no pydocstyle rules, and no test asserts on docstring text.
But "in principle" is not a thing several agents editing in parallel can rely on, and the
2-minute suite is too slow to run after every paragraph. So this compares the CODE of two
revisions directly: it parses both, strips every docstring, discards comments (the tokenizer
never emits them into the AST), and dumps the normalised tree. Identical dumps mean the only
thing that changed was prose, whatever the diff looks like.

    scripts/check_prose_only.py                # working tree vs HEAD
    scripts/check_prose_only.py --base main    # working tree vs where this branch left main

--base RESOLVES THROUGH THE MERGE BASE. `--base main` compares against the commit this
branch forked from, not main's current tip, so commits someone else landed on main after the
fork are not reported as this branch's code changes. For the default HEAD the merge base IS
HEAD, so plain uncommitted-vs-HEAD is unchanged.

EXIT 0 means prose-only: no behaviour can have changed, and a full test run is not required
to establish that. EXIT 1 names the files whose code moved, and those DO need the suite.

THE ONE EXCEPTION IT REPORTS SEPARATELY. Five scripts pass their module docstring to argparse
or print it as help (build_all, corridor_render, corridor_report, export_all_scenarios,
diff_exports), so for those the docstring IS user-facing output rather than commentary.
corridor_report.py reads `__doc__.splitlines()[0]` and so depends on its FIRST LINE. Edits
there are flagged as needing a `--help` eyeball, because no AST check can see that.
"""
from __future__ import annotations
import argparse, ast, subprocess, sys, pathlib

DOC_AS_HELP = {
    "scripts/build_all.py", "scripts/corridor_render.py", "scripts/corridor_report.py",
    "scripts/export_all_scenarios.py", "scripts/diff_exports.py",
}

class StripDocstrings(ast.NodeTransformer):
    """Remove docstring statements; leave every other statement in place."""
    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:] or [ast.Pass()]
        node.body = body
        return node
    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip

def code_shape(source: str) -> str:
    """The AST with all docstrings gone, dumped without positions."""
    tree = StripDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)

def fork_point(ref: str) -> str:
    """The commit `ref` and HEAD diverged at, or `ref` itself if they do not share one.

    Comparing to a ref's TIP charges this branch with every commit that landed on that ref
    since the fork - which for a day-old branch off main is most of the report. The merge
    base is the question actually being asked: what did THIS work change.
    """
    r = subprocess.run(["git", "merge-base", ref, "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ref


def same_commit(a: str, b: str) -> bool:
    """Whether two refs name the same commit, compared by resolved sha rather than by string."""
    def sha(ref):
        r = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ref
    return sha(a) == sha(b)


def git_show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="HEAD", help="ref to compare against (default HEAD)")
    args = ap.parse_args()
    base = fork_point(args.base)
    # Say so only when the merge base is a DIFFERENT commit from the ref asked for - for the
    # default HEAD it always resolves to HEAD, and announcing that is noise.
    if not same_commit(base, args.base):
        print(f"comparing against merge base of {args.base} and HEAD: {base[:12]}")

    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "*.py"],
        capture_output=True, text=True, check=True).stdout.split()
    if not changed:
        print("no changed .py files vs", args.base); return 0

    code_moved, prose_only, unparsable, help_text = [], [], [], []
    for path in changed:
        p = pathlib.Path(path)
        after = p.read_text(encoding="utf-8") if p.exists() else None
        before = git_show(base, path)
        if before is None or after is None:
            code_moved.append(f"{path} (added or deleted)"); continue
        try:
            same = code_shape(before) == code_shape(after)
        except SyntaxError as exc:
            unparsable.append(f"{path} ({exc})"); continue
        (prose_only if same else code_moved).append(path)
        if same and path in DOC_AS_HELP:
            help_text.append(path)

    for path in prose_only: print(f"  prose only   {path}")
    for path in code_moved: print(f"  CODE MOVED   {path}")
    for path in unparsable: print(f"  UNPARSABLE   {path}")
    if help_text:
        print("\nNOTE: these pass __doc__ to argparse/help - check `--help` still reads well:")
        for path in help_text: print(f"  {path}")
    if unparsable:
        print("\nFAIL: file(s) do not parse."); return 1
    if code_moved:
        print(f"\nNot prose-only: {len(code_moved)} file(s) changed code. Run ./scripts/test.sh.")
        return 1
    print(f"\nPROSE ONLY: {len(prose_only)} file(s), no code changed. Suite not required.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
