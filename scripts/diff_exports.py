"""Diff two directories of exported geometry JSON, key by key.

The answer a refactor needs is "which keys changed", not which bytes - a paint channel is a
list of sampled polylines and a float that moved in its 14th decimal place is not a finding.
So this walks the parsed structure and reports one line per key that moved.

Two list shapes get identity rather than position, because the alternative hides the finding:

  * legs, refuge islands and raised crossings are keyed by their `name`;
  * props by (type, position), since the props array is order-sensitive in the render and
    "the same 25 props in a different order" is a completely different finding from "a prop
    moved". Both are reported, separately.

    python scripts/diff_exports.py /tmp/before /tmp/after
"""
import json
import sys
from pathlib import Path

# Below this a difference is float noise from a different order of the same arithmetic, not a
# marking that moved. Coordinates are metres, so this is a tenth of a millimetre.
TOL_M = 1e-4


def numbers(value) -> list[float]:
    """Every number anywhere inside `value`, in traversal order."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [n for item in value for n in numbers(item)]
    if isinstance(value, dict):
        return [n for _k, v in sorted(value.items()) for n in numbers(v)]
    return []


def shape(value) -> str:
    if isinstance(value, list):
        return f"{len(value)} item(s), {len(numbers(value))} number(s)"
    return repr(value)[:120]


def identity(item):
    """A stable name for one item of a list, or None if it has none."""
    if not isinstance(item, dict):
        return None
    if "name" in item:
        return str(item["name"])
    if "type" in item and "position_ft" in item:
        x, y = item["position_ft"][:2]
        return f"{item['type']}@{x:.3f},{y:.3f}"
    return None


def keyed(items) -> dict | None:
    """`items` as a dict keyed by identity, or None if any item has none."""
    keys = [identity(item) for item in items]
    if any(key is None for key in keys):
        return None
    seen: dict[str, int] = {}
    out = {}
    for key, item in zip(keys, items):
        seen[key] = seen.get(key, 0) + 1
        out[key if seen[key] == 1 else f"{key}#{seen[key]}"] = item
    return out


def compare(before, after, path: str, out: list) -> None:
    if type(before) is not type(after) and not (
            isinstance(before, (int, float)) and isinstance(after, (int, float))):
        out.append(f"{path}: type {type(before).__name__} -> {type(after).__name__}")
        return
    if isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                out.append(f"{path}.{key}: ADDED ({shape(after[key])})")
            elif key not in after:
                out.append(f"{path}.{key}: REMOVED ({shape(before[key])})")
            else:
                compare(before[key], after[key], f"{path}.{key}", out)
        return
    if isinstance(before, list):
        by_name = keyed(before) if before and after else None
        if by_name is not None and (by_after := keyed(after)) is not None:
            compare(by_name, by_after, path, out)
            if list(by_name) != list(by_after) and set(by_name) == set(by_after):
                out.append(f"{path}: same {len(by_name)} item(s), ORDER changed")
            return
        if all(isinstance(i, str) for i in before + after):
            for line in [s for s in before if s not in after]:
                out.append(f"{path}: REMOVED {line!r}")
            for line in [s for s in after if s not in before]:
                out.append(f"{path}: ADDED {line!r}")
            if before != after and sorted(before) == sorted(after):
                out.append(f"{path}: same {len(before)} line(s), ORDER changed")
            return
        if len(before) != len(after):
            out.append(f"{path}: {len(before)} item(s) -> {len(after)} item(s)")
            return
        for index, (b, a) in enumerate(zip(before, after)):
            compare(b, a, f"{path}[{index}]", out)
        return
    if isinstance(before, (int, float)) and not isinstance(before, bool):
        if abs(before - after) > TOL_M:
            out.append(f"{path}: {before:.4f} -> {after:.4f}")
        return
    if before != after:
        out.append(f"{path}: {before!r} -> {after!r}")


def summarize(lines: list[str], limit: int = 20) -> list[str]:
    """Collapse a run of per-coordinate reports into one line per key."""
    by_key: dict[str, list[str]] = {}
    for line in lines:
        # Everything up to the first index bracket: one paint channel, one prop, one leg key.
        by_key.setdefault(line.split("[")[0], []).append(line)
    out = [group[0] if len(group) == 1
           else f"{key}: {len(group)} difference(s), e.g. {group[0]}"
           for key, group in by_key.items()]
    return out[:limit] + ([f"... and {len(out) - limit} more key(s)"] if len(out) > limit else [])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    before_dir, after_dir = (Path(p).resolve() for p in sys.argv[1:3])
    before_files = {p.relative_to(before_dir): p for p in sorted(before_dir.rglob("*.json"))}
    after_files = {p.relative_to(after_dir): p for p in sorted(after_dir.rglob("*.json"))}
    differing = 0
    for rel in sorted(set(before_files) | set(after_files)):
        if rel not in before_files or rel not in after_files:
            print(f"{rel}: only in {'after' if rel in after_files else 'before'}")
            differing += 1
            continue
        out: list[str] = []
        compare(json.loads(before_files[rel].read_text()),
                json.loads(after_files[rel].read_text()), "", out)
        if out:
            differing += 1
            print(f"\n{rel}  ({len(out)} difference(s))")
            for line in summarize(out):
                print(f"  {line}")
    total = len(set(before_files) | set(after_files))
    print(f"\n{differing} of {total} export(s) differ")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
