---
name: verify-change
description: Verify a change to this repo's geometry, markings, renders or numbers before calling it done. Use after editing anything under src/ or scripts/, when a golden test fails, or when asked whether a change is correct. Runs the export/diff/test/render/measure loop in cost order.
---

# Verifying a change here

Run these **in order**. Each is cheaper than the one after it, and each catches a class the
earlier ones cannot.

## 0. Is it prose only?

If the diff touches only comments and docstrings:

```bash
.venv/bin/python scripts/check_prose_only.py --base <your-branch-point>
```

**Exit 0 means no behaviour changed and steps 1-5 are not required.** Exit 1 names the files
whose code moved — verify those normally.

Two exceptions the checker flags: five scripts use `__doc__` as CLI help (`build_all.py`,
`corridor_render.py`, `corridor_report.py`, `export_all_scenarios.py`, `diff_exports.py`), and
`corridor_report.py` reads `__doc__.splitlines()[0]`. Run `--help` and read it.

## 1. Write the test, and confirm it fails first

`git stash` the change, watch the test report, restore. A check that has never fired pins
nothing.

## 2. Diff the exports

```bash
python scripts/export_all_scenarios.py /tmp/before
```

change → `/tmp/after` → `python scripts/diff_exports.py /tmp/before /tmp/after`. This resolves
the scene, builds the paint and props and asserts every invariant, without spending Blender.

## 3. The suite

```bash
./scripts/test.sh
```

Never bare `pytest`. Includes lint, import contracts and goldens. A golden failure is not
automatically a bug — read the diff, confirm every moved number is one you meant to move, then
`--force-regen` and commit the goldens **in the same commit as the cause**.

## 4. Render, and open the PNGs

```bash
python scripts/build_all.py --render-3d
```

Actually look at them. Rendering without opening them is not step 4.

## 5. Measure the drawn geometry

```bash
.venv/bin/python scripts/measure_drawn.py <site> --scenario <build_*> --kind <substring>
```

Prints each drawn piece as (station, offset) in its leg's frame. **This is the rule that would
have prevented most of this repo's bugs**: a check that reads the same function the renderer
reads is not a check. Every serious defect here has been two derivations of one number agreeing
with each other and disagreeing with the picture.
