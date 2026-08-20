#!/usr/bin/env python3
"""What is this symbol? Signature, docstring first line, and how callers actually use it.

    scripts/whatis.py junction_mouths_ft treatments_of

The signature is the contract, and in this repo it is buried: docstrings here run to 40 lines
because the WHY is the valuable part, so `grep -n "def name"` lands you in the middle of a wall
of prose and the return type is off the top of the screen. Worse, every name in this repo recurs
inside those docstrings and in note text, so grep for a symbol returns mostly prose.

WHAT CALLERS DO IS PART OF THE ANSWER, and it is the half no signature carries. A bare `-> list`
says nothing; sixteen call sites all writing `for line, tags, way_id in ...` say it is a 3-tuple -
and say it more reliably than the docstring, which on that function claims a 2-tuple and is
wrong. So the unpack shapes are reported next to the signature, and where they disagree with the
docstring, believe them.
"""
from __future__ import annotations

import argparse
import ast
import copy
import pathlib
import sys
import textwrap

ROOTS = ("src", "scripts", "tests", "sites")
ELLIPSIS = [ast.Expr(ast.Constant(...))]


def repo_root() -> pathlib.Path:
    """The checkout, found by marker rather than by depth, so this runs from anywhere."""
    for start in (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "src").is_dir() and (candidate / "pytest.ini").is_file():
                return candidate
    raise SystemExit("not inside the hopewell-road-sketches checkout")


def _trees(repo: pathlib.Path):
    for root in ROOTS:
        for path in sorted((repo / root).rglob("*.py")):
            try:
                yield path, ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue


def _signature(node) -> str:
    """The def line as written, without the body - decorators included, they are contract too."""
    shown = copy.deepcopy(node)
    shown.body = ELLIPSIS
    if isinstance(shown, ast.ClassDef):
        shown.body = ELLIPSIS
    text = ast.unparse(shown)
    return text.rsplit("\n", 1)[0] if text.endswith("...") else text


def _first_doc_line(node) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().split("\n", 1)[0]


def _mentions(node, name: str) -> bool:
    """A bare mention of the name - a constant read, or an attribute of that name."""
    if isinstance(node, ast.Name):
        return node.id == name and isinstance(node.ctx, ast.Load)
    return isinstance(node, ast.Attribute) and node.attr == name


def _called_name(call: ast.Call) -> str | None:
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    return fn.attr if isinstance(fn, ast.Attribute) else None


def definitions(repo: pathlib.Path, name: str):
    """Every def/class/module-constant with this name, and the class it sits in."""
    found = []
    for path, tree in _trees(repo):
        # id() rather than a recursive walk with a parent argument: ast.walk is the cheap way
        # round, and `.body` is not a statement list on every node that has one (JoinedStr).
        owners = {id(child): parent.name
                  for parent in ast.walk(tree) if isinstance(parent, ast.ClassDef)
                  for child in parent.body}
        module_level = {id(n) for n in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    found.append((path, node, owners.get(id(node))))
            elif (isinstance(node, ast.Assign) and id(node) in module_level
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
                found.append((path, node, None))
    return found


def usages(repo: pathlib.Path, name: str):
    """(call sites, references, {unpack shape: [where]}) - what the callers do with it.

    References are counted separately because a CONSTANT is never called: reporting only call
    sites says "0 uses" for TARGET_LANE_WIDTH_FT, which is the opposite of true.
    """
    sites, refs, shapes = [], [], {}

    def note(target, path, lineno):
        shapes.setdefault(ast.unparse(target), []).append(f"{path}:{lineno}")

    for path, tree in _trees(repo):
        rel = path.relative_to(repo)
        # ast.walk sees a call AND the Name inside it, so a plain call would be counted twice -
        # once as a call site, once as a bare reference. Skip the func node of every match.
        called = {id(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and _called_name(n) == name}
        for node in ast.walk(tree):
            if id(node) in called:
                continue
            if isinstance(node, ast.Call) and _called_name(node) == name:
                sites.append(f"{rel}:{node.lineno}")
            elif _mentions(node, name):
                refs.append(f"{rel}:{node.lineno}")
            # a, b = f(...)  /  for a, b in f(...)  /  [x for a, b in f(...)]
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and _called_name(node.value) == name:
                for t in node.targets:
                    if isinstance(t, (ast.Tuple, ast.List)):
                        note(t, rel, node.lineno)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                    and _called_name(node.iter) == name:
                note(node.target, rel, node.lineno)
            if isinstance(node, ast.comprehension) and isinstance(node.iter, ast.Call) \
                    and _called_name(node.iter) == name:
                note(node.target, rel, node.iter.lineno)
    return sites, refs, shapes


def report(repo: pathlib.Path, name: str) -> bool:
    defs = definitions(repo, name)
    sites, refs, shapes = usages(repo, name)
    if not defs:
        print(f"{name}: no def, class or module constant with that name under {'/, '.join(ROOTS)}/")
        if sites or refs:
            print(f"  but {len(sites) or len(refs)} use(s) - imported from a dependency, or a method on "
                  f"an object: {', '.join((sites or refs)[:4])}")
        return False

    for path, node, owner in defs:
        rel = path.relative_to(repo)
        where = f"{rel}:{node.lineno}"
        if isinstance(node, ast.Assign):
            print(f"{where}  {ast.unparse(node)}")
            continue
        qualifier = f"  (on {owner})" if owner else ""
        print(f"{where}{qualifier}")
        print(textwrap.indent(_signature(node), "    "))
        doc = _first_doc_line(node)
        if doc:
            print(f"    └─ {doc}")
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith("_")]
            if methods:
                print(f"    methods: {', '.join(methods)}")
    counted = f"{len(sites)} call site(s)" if sites else f"{len(refs)} reference(s)"
    if sites and refs:
        counted += f", {len(refs)} other reference(s)"
    print(f"  {counted}")
    # ARITY is the finding, so tuple shapes are itemised; a plain loop variable carries no
    # arity, only a hint of the element type, so those collapse to one line of names.
    tuples = {k: v for k, v in shapes.items() if not k.isidentifier()}
    names = {k: v for k, v in shapes.items() if k.isidentifier()}
    if tuples:
        print("  callers unpack it as:")
        for shape, where in sorted(tuples.items(), key=lambda kv: -len(kv[1])):
            print(f"    {len(where):3d}x  {shape:40s} e.g. {where[0]}")
    if names:
        ranked = sorted(names, key=lambda k: -len(names[k]))
        print(f"  iterated one at a time as: {', '.join(ranked[:8])}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+", help="function, class, method or module constant")
    args = ap.parse_args(argv)
    repo = repo_root()
    ok = True
    for i, symbol in enumerate(args.symbols):
        if i:
            print()
        ok = report(repo, symbol) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
