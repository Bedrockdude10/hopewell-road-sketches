# CLAUDE.md

A router, not a summary. README.md is the architecture, STANDARDS.md is the figures register,
and `.claude/SKILLS.md` (imported below) is the list of what agents here actually get wrong.

## Answer at the quantitative layer

**Diagnose from numbers; render to confirm, and only at the end.** Everything this project
draws exists first as numbers, and both of the tools below print them in seconds. The renders
are cheap now too (~17 s a scene), so waiting on one is no longer the cost — believing it is.
A PNG is where you NOTICE a problem and never where you diagnose one: one session cropped
renders to three wrong diagnoses in a row on a single complaint, and two of the three were
about the wrong leg. See §0a of SKILLS.md, which is the same rule with the receipts.

So the shape of a turn here is: **measure → change → measure → verify → render once.** A
render inside the loop is a round trip that cannot answer the question you asked it.

## Running things

- **`scripts/measure_drawn.py <site> --scenario <s> --leg <leg> --all`, ~2 s.** The whole
  quantitative layer: what is drawn stationed against the centreline, plus `--section` (what
  the treatment thinks it placed vs the room the kerb gives), `--limiters` (all four things
  deciding where kerbside paint starts), `--gaps` (kerb minus outermost paint, station by
  station) and `--continuity` (is the facility one piece, how wide are the holes). Narrow with
  `--leg`/`--kind`; measure at the reader's `--frame-scale`, not at 1x.
- **`scripts/verify.py` before you report done.** One command for the whole loop: exports the
  working tree and `--base` side by side, diffs them, runs the suite, and reports failures as
  NEW / KNOWN / FIXED against a recorded baseline. `--no-tests` is ~22 s, `--no-tests --site
  <site>` ~6 s, everything ~5 min. **NEW is the only number that says anything about your
  change** - this repo's suite is often red from work in flight, and re-deriving whose red it
  is by hand was costing more than the run.
- **`./scripts/test.sh`, never bare `pytest`.** The script pins the venv interpreter, so a
  wrong `python` on PATH cannot masquerade as a broken repo. It runs `-n auto`; pass `-n 0`
  when you need `-x`, `pdb`, or readable ordering.
- `scripts/whatis.py <symbol>`, ~1 s, before you write a second copy of anything: signature,
  the docstring's first line, and every call site. §1 of SKILLS.md is the list of facts that
  already have a home; this is how you find the ones that are not on it.
- `scripts/check_prose_only.py --base <rev>` proves a diff changed only comments and
  docstrings. Exit 0 means no behaviour moved and the suite is not required.

## What CI cannot see

`data/` is a 391 MB licensed download kept out of git, and every golden and whole-site test is
marked `needs_source_data` — those **skip**, not fail, when it is absent. A green tick on
`main` is green over the subset that does not need it, so geometry changes have to be verified
locally.

## Prose

One home per fact: keep the trap, the datum, the invariant and why not the obvious alternative;
cut the discovery story, the session archaeology, and anything the code already says. Published
figures live as rows in STANDARDS.md, not as inline comments.

@.claude/SKILLS.md
