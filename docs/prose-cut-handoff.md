# Handoff: cut the prose, speed up the loop

**Status: executed, 2026-08-20.** Kept as the record of why the prose contract and the agent
guidance are shaped the way they are; it is not a live work queue. Measured against the plan:

| | planned | now |
|---|---|---|
| Phase 0 (§6) | 0.1-0.6 | **all landed** - `-n auto` in `scripts/test.sh`, `CLAUDE.md` @-importing `.claude/SKILLS.md`, `.claude/skills/verify-change/SKILL.md`, `measure_drawn.py`, `check_prose_only.py` |
| README | under ~6k words | 6,291 words, and **0** sections titled with a `src/` path (Lane R's queue, §7) |
| `src/` prose:code | 0.6:1 | **0.81:1**, from 1.13:1 |
| suite | `-n auto` | 4.4 s here, 333 of 707 skipped with `data/` absent |

So the structural work is done and only the `src/` ratio fell short of its target. That last row
is the one thing here still worth acting on - but act on it **by the rules in §5, never to the
number**, which is what §5 says in its own first line. The lane table in §7 is spent: it names an
agent's uncommitted files and a blocked Lane X that have long since landed.

A plan for several agents to execute in parallel. Measured 2026-08-18; every number came from a
command in this repo. **One document on purpose** — an earlier draft split the diagnosis from the
execution and immediately restated 13 of 17 figures across the two halves, which is the defect
this plan exists to remove.

---

## 1. What was measured

**Feedback loop**

| | |
|---|---|
| full suite, serial | **128.55 s** (670 passed, 7 skipped) |
| full suite, `-n 8` | **31.44 s** — identical results |
| import cost / collection alone | 0.77 s / 1.24 s |

The two minutes is real geometry and matplotlib work, not startup, and it parallelises **4.1x** for
one dependency. `build_all.py` already says sites "share nothing but a read-only cache" and runs
`--jobs` of them at once; the suite never got the same treatment.

**Prose weight in `src/`**

| | lines | share |
|---|---|---|
| total | 23,742 | |
| code | 9,540 | **40.2%** |
| comments | 3,319 | 14.0% |
| docstrings | 7,480 | 31.5% |
| blank | 3,403 | |

**1.13 prose lines per code line** — 123,335 prose words, 10,823 prose lines. Well-documented Python
normally runs 0.2–0.4:1. Tests are 0.63:1 and scripts 0.52:1, i.e. already fine.

Token cost: README.md is 29,811; all of `src/` is 364,075. A task touching `paint.py`, `network.py`,
`bikeways.py` and the README costs **~102,600 tokens**, ~46k of it prose.

**Duplication**

| | |
|---|---|
| verbatim 9-word runs shared between docs and code prose | **4.6%** of doc prose |
| verbatim runs repeated across ≥2 `src/` files | 0.6% |
| `src/` public symbols (>6 chars) / named in README+STANDARDS | 881 / **203 (23%)** |
| docstring words `src/` spends on those same 203 symbols | 15,878 |

## 2. The diagnosis

**The duplication is real but it is not copy-paste** — only 4.6% verbatim, so `grep` will not find it.
It is *semantic*: the same narrative retold. README.md is organised as per-module narration — 44
sections, many titled with the module path they describe (`## Treatments
(src/geometry/treatments/)`, `### What the design achieves (src/metrics.py)`, `### "Is this kerb
ours" has three answers (src/geometry/context_roads.py)`) — so each shadows a docstring telling the
same story.

That last pair is the clearest case. Section and module docstring both narrate: the wide render
showing a cross of asphalt floating on grass, `leg_working_length_ft` as the wrong suspect, the 80 ft
near-set, the 8,938 ft of traced kerb dropped, the three relevance tests, the bimodal
surveyed/assumed coverage. Two full tellings, near-zero verbatim overlap, one subject — both
maintained, nothing detecting drift.

**So is the README to blame? Partly, and it is the smaller half.** Deleting it outright removes 17k of
~174k prose words — about 12%. The weight is in `src/`, where reading any file costs roughly 2.5x the
tokens of the code inside it.

**But the prose is not waste, and mass deletion would be a mistake.** It has prevented real defects —
the two-datum trap, `inset_line_ft` never `offset_curve`, "measure the drawn output, not the
arithmetic." Split by **half-life**:

- **Load-bearing** — the trap, the invariant, the datum, why-not-the-obvious-thing. True as long as
  the code is. Belongs beside the code it guards.
- **Narrative history** — "the first wide render showed…", "the obvious suspect was X and it was
  wrong." True once, then ballast. A commit message living in a source file.

Roughly a third of `src/` prose is the second kind, and it is that third the README retells.

**One finding is nearly free to fix.** `.claude/` contains exactly one file, `SKILLS.md`. Claude Code
auto-loads `CLAUDE.md` (plus `@`-imports) and discovers skills at `.claude/skills/<name>/SKILL.md` —
that path is neither, and nothing references it. The most valuable document here, the list of what an
agent in this repo actually gets wrong, never enters context. Meanwhile 123k words of narrative
history load on every file read. **The prose budget is spent almost exactly backwards.**

**Goal:** `src/` prose:code near **0.6:1** with every load-bearing sentence kept, README under ~6k
words, suite at `-n auto`, and the guidance actually loaded.

## 3. Not doing

**An MCP server.** MCP earns its place when a tool must reach something Bash cannot — an
authenticated service, a remote API, a live application. The one case here that qualifies is Blender,
which already has a server. Everything else is `pytest`, `python scripts/*.py`, `grep`. A new server
would add tool schemas to every request and fix none of this.

**A prose purge.** 1.13:1 is too high, but these comments encode facts about a physical street the
code cannot express. The goal is one home per fact, not less knowledge.

---

## 4. The safety property that makes this parallelisable

No doctests are enabled, `ruff.toml` selects **no pydocstyle (`D`) rules** (only `F, E9, B, SIM, C4,
UP, RUF`), and no test asserts on docstring text. So removing a comment or docstring cannot change
behaviour — and that is now mechanically checkable:

```bash
.venv/bin/python scripts/check_prose_only.py --base <your-branch-point>
```

Written and self-tested: it parses both revisions, strips docstrings, compares normalised ASTs.
**Exit 0 means no behaviour changed and the suite is not required.** Exit 1 names files whose code
moved.

**This is the speed unlock** — a prose lane's inner loop is a ~1 s AST check, not a 31 s suite. Run
the suite once per lane at the end, and once on the merged result.

### Four exceptions — do not cut blind

1. **Five scripts use `__doc__` as user-facing CLI help**: `build_all.py:235`,
   `corridor_render.py:242`, `corridor_report.py:272`, plus `export_all_scenarios.py:56` and
   `diff_exports.py:126` which `print(__doc__)`. **`corridor_report.py` reads
   `__doc__.splitlines()[0]`, so it depends on its first line.** The verifier flags these; run
   `--help` and read it. That docstring is product copy, not commentary.
2. **`ruff.toml` states a policy** about long aligned lines and long comments. That is line *length*,
   not prose *volume* — leave `line-length`/`E501` alone, and if you touch the comment keep it true.
3. **Test docstrings cite docs by name** — `docs/network-model.md`, `sites/README.md`, `STANDARDS.md
   section 2`, named README sections (`test_corridor.py`, `test_coverage.py`,
   `test_surveyed_crossings.py`, `test_geometry_regression.py:96`). Prose references, so nothing
   fails when a section disappears; it goes stale silently. **Lane R owns fixing them.**
4. **`tests/` and `scripts/` are already reasonable.** Do not cut them. The target is `src/` only.

## 5. The prose contract — same rules for every lane

Cut **by rule, not to a quota.** The ~45% reduction is the *expected consequence* of these rules, not
a number to hit by deleting something load-bearing. A lane landing far off should say so, not force it.

**KEEP** — the trap ("never `offset_curve` for stationed work"); the datum (which of two measurements
a number comes from — the traced-kerb vs nominal-half-width split, 25 ft apart on `broad_st_east`,
has caused three bugs); the invariant and why it cannot be relaxed; why not the obvious alternative;
units, CRS, coordinate frame; a one-line summary of what the function is for; the pointer to a
constant's single home.

**CUT** — the discovery story; session archaeology ("this cost two round trips", "it drifted
immediately"); before/after tables of fixed bugs; restatements of what the code plainly says; a
concept re-explained in another file (611 nine-word runs repeat across ≥2 `src/` files — keep one
home, point at it); published figures with a provenance claim, which belong in `STANDARDS.md` — move
the row, cite it, delete the inline copy. **Never delete a figure with no STANDARDS row; add the row
first.**

**TRANSFORM** — the dominant move. `WHY THIS EXISTS. [400 words of narrative]` becomes *[one
sentence: the rule this enforces and the trap it prevents]*. The story is not lost; `git log --follow`
has it. If it is not in a commit message yet, **put it there in the commit that removes it.**

---

## 6. Phase 0 — serial, ONE agent, lands on `main` before any lane starts

Shared files, and everything downstream depends on it. Do not parallelise.

| # | Task | Files |
|---|---|---|
| 0.1 | Add `pytest-xdist` to `requirements.txt`; add `-n auto` to `scripts/test.sh`'s exec line. Verify 670 pass. `pytest-randomly` is installed — if worker-split flakiness appears, pin order in CI, don't drop xdist. | `requirements.txt`, `scripts/test.sh` |
| 0.2 | Create `CLAUDE.md` (auto-loaded). **Short, a router, not a summary:** use `./scripts/test.sh` never bare `pytest`; goldens are local-only and CI cannot see them (`data/` is a 391 MB licensed download, 236 tests skip there); the prose contract in one line; then `@.claude/SKILLS.md`. **It must not restate README.md.** | `CLAUDE.md` |
| 0.3 | Make the failure-mode guide load, via 0.2's `@`-import. Its content is good — re-home it, do not rewrite it. | `.claude/SKILLS.md` |
| 0.4 | Create `.claude/skills/verify-change/SKILL.md` from §8 of `SKILLS.md` — export → diff → test → render → measure is a procedure, which is what skills are for. Add: prose-only changes verify with `check_prose_only.py` and skip the suite. | new |
| 0.5 | Write `scripts/measure_drawn.py` — given a site/scenario, print drawn coordinates stationed against the leg centreline (`station_offset_many`). §0's "measure the drawn output" is the most-skipped rule *because invoking it costs recall*. Make it one command. | new |
| 0.6 | Commit `scripts/check_prose_only.py` (untracked at handoff). | — |

**Exit:** suite green in ~31 s, `CLAUDE.md` loads the guide, both scripts run. **Tag the commit** —
every lane branches from it.

## 7. Phase 1 — parallel. Lanes own disjoint files.

`git worktree add ../hw-<lane> -b prose/<lane> <phase0-tag>`. Prose lanes are file-disjoint, so
merges are trivial. **Never edit a file outside your lane** — that is the only way this collides.

| Lane | Owns | Prose words |
|---|---|---|
| **A1** | `src/geometry/paint.py`, `network.py`, `markings.py` | 19,325 |
| **A2** | `src/geometry/` rest of top level: `kerbs.py`, `coverage.py`, `surveyed.py`, `corridor_paint.py`, `cross_streets.py`, `daylighting.py`, … | 20,307 |
| **B** | `src/render/` (11 files) | 22,211 |
| **C** | `src/geometry/treatments/` **except `bikeways.py`** | 12,235 |
| **D** | `src/geometry/model/` **except `leg_frame.py`** | 8,771 |
| **E** | `src/geometry/intersection/` + `src/sources/` | 15,888 |
| **F** | `src/*.py` top level: `checks.py`, `metrics.py`, `site_schema.py`, … | 8,832 |
| **R** | `README.md`, `STANDARDS.md`, `docs/*.md`, `sites/README.md` — no `.py` at all | — |
| **S** | `scripts/export_all_scenarios.py` — robustness, not prose | — |
| **X** | `src/geometry/treatments/bikeways.py`, `src/geometry/model/leg_frame.py` | 15,766 |

`surveyed.py` (Lane A2) is the worst ratio in the repo at 3.69:1.

**Lane X is blocked.** Another agent holds `leg_frame.py`, `bikeways.py` and
`tests/test_two_way_bike_lane.py` (modified, uncommitted). **Do not touch those three.** Start X after
that work lands, then rebase onto `main`. C and D proceed on everything else.

**Lane R in detail.** The pattern is structural, not textual, so there is nothing to grep for:
1. Find every section whose title contains a `src/` path. The **module docstring is the home** —
   replace the section with a one- or two-line entry saying what the module is for and pointing at it.
2. The 203 symbols documented in both places are the work queue.
3. Fix the stale doc references in `tests/` (§4 exception 3) in the same commit.
4. **Target README under ~6k words**, still a complete map.
5. Leave `STANDARDS.md` largely intact — it is a figures register, and lanes will be *adding* rows.
   If a prose lane needs a row it adds it and says so.

**Lane S in detail.** `export_all_scenarios.py` dies on a self-intersecting pavement ring
(`build_pavement_polygon`, `src/geometry/model/corners.py:299`), killing the whole before/after
comparison. Collect per-site failures, report them, export the rest, exit non-zero — as
`build_all.py` already does. **Do not fix the underlying geometry**; that is a separate change with
golden consequences.

**Per-lane loop:** `check_prose_only.py --base <phase0-tag>` must exit 0. Commit in small batches, one
file or one related group each, message naming the rule applied — and carrying any discovery story
worth keeping. Run `./scripts/test.sh` **once** before handing the lane back, not per commit.

## 8. Phase 2 — after Phase 1 merges

| # | Task |
|---|---|
| 2.1 | Merge lanes in any order (disjoint files). Run the full suite **plus** `scripts/build_all.py --render-3d` and **open the PNGs**. The AST check proves each lane; this confirms the merge. |
| 2.2 | Add a guard in `tests/test_lint.py`: **no README section title may contain a `src/` path** — the exact shape of the regression. Verify it fails before Lane R and passes after. |
| 2.3 | Re-run the measurements and record the new ratios in §1 of this file. Expected: prose:code ≈ 0.6:1, README ≈ 6k words, suite ≈ 31 s. |
| 2.4 | Lane X once unblocked, then re-run 2.1. |

## 9. Parallelism summary

Phase 0 serial (1 agent) → **Phase 1 nine agents at once** (A1, A2, B, C, D, E, F, R, S; X waits) →
Phase 2 serial (1 agent). Critical path is Phase 0, the longest lane (~22k prose words), then Phase 2.
The prose lanes verify in ~1 s instead of 31 s, which is what makes nine of them practical.

**Two rules keep this from going wrong:** never edit a file another lane owns, and never delete a
sentence you cannot replace with a shorter true one.

---

## Appendix — noticed, not in scope

Three tests fail on `353ece1` with a clean tree and pass on `11b7f21`:
`test_the_two_way_corridor_geometry_is_unchanged[broad_st_greenwood, wbroad_louellen]` and
`test_sampled_polylines_are_rendered_as_polylines_not_chords`. Attributed to the in-flight agent
holding Lane X's files. Worth recording that **CI cannot see this class of break** — `data/` is
absent there, so 236 tests skip including every golden. A green tick on `main` is green over the
wrong subset, which is why a slow local loop is expensive.
