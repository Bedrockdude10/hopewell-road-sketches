# CLAUDE.md

A router, not a summary. README.md is the architecture, STANDARDS.md is the figures register,
and `.claude/SKILLS.md` (imported below) is the list of what agents here actually get wrong.

## Running things

- **`./scripts/test.sh`, never bare `pytest`.** The script pins the venv interpreter, so a
  wrong `python` on PATH cannot masquerade as a broken repo. It runs `-n auto`; pass `-n 0`
  when you need `-x`, `pdb`, or readable ordering.
- `scripts/measure_drawn.py <site> <scenario>` prints drawn coordinates stationed against the
  leg centreline. That is the check §0 of SKILLS.md is about; it is one command now.
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
