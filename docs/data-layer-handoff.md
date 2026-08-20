# Handoff: the exported geometry layer

For an agent running on the machine that **has `data/`** (the 391 MB licensed download) and can
run Blender. Everything below was measured in a container without `data/`, which is why the
regeneration half could not be done there and is the whole reason this document exists.

Branch `claude/repo-cleanliness-t67yzl`, commit `05b3aa4`, is already pushed and unrelated to this
work (it fixes the Python 3.12 install floor, the scrambled `requirements.txt` header, duplicated
`.gitignore` rules, and stamps `docs/prose-cut-handoff.md` as executed). Branch from it or from
`main`; nothing here conflicts.

**Read `.claude/SKILLS.md` first.** §0a in particular: every number below came from querying the
data, and if you disagree with one, re-measure it rather than reasoning from the constants.

---

## The state of `output/`, measured

| | |
|---|---|
| tracked `geometry_*.json` | 65 files, **33.8 MB** |
| tracked PNGs | 37 files, 28.1 MB |
| tracked `output/` total | **61.9 MB**, against a 120 MB `.git` |
| of one file's bytes, actual JSON content | **38%** — the other 62% is pretty-print whitespace |
| files whose top-level schema is NOT current | **39 of 65** |

Byte share within a file (`wbroad_louellen/geometry_existing.json`, 426 KiB): `paved_surfaces`
32.5%, `buildings` 28.4%, `sidewalks_far` 8.8%, `kerbs` 5.9%. So the weight is real geometry, not
metadata — which is why the fix is how it is written, not what is in it.

---

## 1. Land the writer change BEFORE regenerating anything

`json.dump(round_for_export(data), f, indent=2)` (`src/render/export.py:513`) puts **every single
number on its own line**, so one coordinate pair spans four lines:

```json
"coords": [
  [
    -27.680958,
    -3.780609
  ],
```

Emitting the innermost numeric list inline instead — `[-27.680958, -3.780609]` — keeping `indent=2`
for all the structure above it:

| | before | after |
|---|---|---|
| `wbroad_louellen/geometry_existing.json` | 426 KiB / 29,679 lines | **223 KiB / 8,092 lines** |
| all 65 files | 32.9 MiB | **19.0 MiB** (42% smaller, 73% fewer lines) |

I verified `json.loads(new) == json.loads(old)` for all 65 files, so this moves no data.

**This makes diffs better, not worse, which is the point.** The existing format spreads a moved
vertex across two changed lines with brackets between them and never shows x beside y; one line per
vertex is the unit a reader actually wants. That matters because diff legibility is the stated
reason the export is rounded at all — see the next paragraph.

**DO NOT touch `EXPORT_DECIMALS`.** `src/render/coords.py:14-22` already argues 6 decimals: the
purpose is diff legibility rather than file size, and 6 rather than 3–4 because *not every exported
number is a length* — `crosswalk_axis` is a unit vector, where absolute rounding is an angle, and
1e-4 there would be 1 cm over a 100 m leg. I measured what dropping to 3 dp would buy: **6.6% of
the file**, i.e. a sixth of what formatting buys, in exchange for breaking the least forgiving
field. One precision for the document is the correct design. Leave it.

Where the writer lives is your call, but `round_for_export`'s docstring makes the case for the
whole-document-at-serialization layer, so a `dumps_for_export(data)` beside it in
`src/render/coords.py` is the consistent home. Pin it with a round-trip test — parse the emitted
string back and assert equality with the input — because that is the property that makes the
reformat safe.

## 2. Make the texture paths portable, in the same pass

Every one of the 65 files embeds **19 absolute paths from one machine**:

```
"/Users/danny/ProductiveProjects/hopewell-road-sketches/output/.textures/asphalt_01/4k/asphalt_01_Diffuse_4k.jpg"
```

They point into `output/.textures/`, which is gitignored and re-downloaded on demand, so on any
other machine these resolve to nothing. Store them repo-relative (`output/.textures/...`) and
resolve against the repo root in the consumer. The consumer side is
`scripts/blender/blender_scene.py`, which runs inside Blender's own interpreter — check how it
reaches the repo root there before assuming `__file__` works the way you expect.

Do this **before** the regeneration too, so one rewrite of 33 MB fixes both this and the
formatting. Regenerating first and reformatting after churns those files twice in the history.

## 3. Then regenerate, because 39 of 65 committed exports are stale

This is the actual defect; the size questions above are cosmetics next to it. Three distinct
schemas are committed:

| files | keys | missing vs current |
|---|---|---|
| 26 | 37 | — (current) |
| **35** | 28 | `frame`, `kerbs`, `paved_surfaces`, `surveyed_crossings`, `bike_lane_edge_lines`, `bike_lane_hatch_lines`, `bike_lane_surface_polygons`, `bike_lane_symbol_polygons`, `bike_lane_contraflow_lines` |
| 4 | 35 | `surveyed_crossings`, `bike_lane_symbol_polygons` |

Stale by group: `broad_st_greenwood` 11/15, `columbia_princeton` 3/5, `ebroad_princeton` 4/8,
`wbroad_louellen` 3/6, `princeton_eprospect` 0/2, `wide2.5x` 18/29.

It is staleness and not legitimately-empty scenarios: `src/render/export.py` builds one flat dict
literal with every key unconditional (`frame` at :325, `kerbs` at :462, `paved_surfaces` at :475,
`surveyed_crossings` at :495), and the exporter emits empty lists for keys that have no content —
`tree_points` and `existing_marked_crosswalks` are present-and-empty in the current files. The
stale files were last written 2026-08-16 in `27cb75d`, the same commit that *added* those keys to
the exporter; the current ones were rewritten 2026-08-19.

**Why this is worse than untidy.** The consumer reads all four with a default:

```
blender_scene.py:248   frame = data.get("frame")
blender_scene.py:299   data.get("paved_surfaces", data.get("driveways", []))   # legacy key fallback
blender_scene.py:313   data.get("kerbs", [])
blender_scene.py:415   data.get("surveyed_crossings", [])
```

So rendering a stale file **silently** produces a street with no kerbs, no driveways or parking
aprons, and no surveyed crossings — and with `frame` absent, Blender computes an extent of its own
from the pavement, which the comment at `export.py:325` says in as many words it must not do. That
is the thing `docs/network-renderer-plan.md` states the project exists to prevent: "A picture that
shows a marked crosswalk as bare asphalt is not a conservative simplification — it is a false
statement about the street, made to an audience deciding whether to build something." Nothing warns
you; the `data.get("driveways", ...)` fallback is itself a compatibility shim for a still older key
name, which is part of how this went unnoticed.

Regenerate every scenario at both frame scales with `scripts/export_all_scenarios.py` /
`scripts/build_all.py`, and follow `.claude/skills/verify-change/SKILL.md` — export to `/tmp/before`
first, regenerate, `scripts/diff_exports.py`, and read the diff rather than trusting it. Expect the
39 stale files to gain whole keys, and **expect the other 26 to change too**, from the formatting in
§1. Those two causes are hard to read in one diff, so land §1 and §2 as their own commit with the
files untouched, then regenerate in a second commit whose diff is *only* the content change. Do not
combine them.

## 4. Add the guard that would have caught it

Nothing in `tests/` reads the committed `geometry_*.json` at all — `export_scenario` is tested
(`tests/test_sites.py`, `tests/test_geometry_regression.py`) but only ever freshly called, so no
check compares what is on disk against what the current exporter produces. That is the hole.

A schema check needs no `data/` — it only reads JSON — so it can run in CI, where the geometry
goldens cannot. Assert every committed export carries the current top-level key set. Note the
ordering trap: **write it, watch it report the 39, and only then regenerate**, per SKILLS.md §4
("verify a new check fails first" — a check that has never fired pins nothing). It cannot land green
before §3, so either land it in the same commit as the regeneration or keep it a script until then.

Consider also asserting the absent-key case is *loud* rather than silent, since the `.get(..., [])`
defaults are what converted stale data into a plausible wrong picture.

## 5. Stop committing `output/wide2.5x/`

Decided: gitignore it. 29 files, **26.6 MB — 43% of all tracked `output/`**, and 18 of the 29 are
stale. It is regenerable from the same configs with `HOPEWELL_FRAME_SCALE=2.5`.

Follow the convention the rest of `.gitignore` uses: every rule there argues *why* in a comment
above it. State what is lost — the wide sheet is no longer reviewable from a clone — and that
SKILLS.md §0b makes that sheet the one that exposes frame-dependent design bugs, so the rule is
trading review convenience for 43% of the tracked bytes. `git rm --cached` the tree in the same
commit as the rule, and check whether anything reads those paths
(`tests/conftest.py` builds at `WIDE_FRAME_SCALE = 2.2`, which is a *third* scale — worth
understanding before you assume 2.5 is unused).

---

## What I deliberately did not conclude

**Whether the JSON should be committed at all.** The `.gitignore` rationale is that the geometry is
"the reconstruction preserved in git" while the 3D renders are not. That holds up better than it
looks: these files are *not* regenerable without the 391 MB licensed `data/` download, so for
anyone without it the committed JSON is the only copy of the reconstruction. Committing it is
defensible. Keeping 39 of 65 stale is what breaks the argument — a stale export preserves an older
program, and nothing tells you which. Fix currency before revisiting the policy.

**Whether `paved_surfaces` and `buildings` (61% of bytes between them) could be leaner.** Buildings
are decimated meshes (`build_decimated_building_mesh`) and I did not check how hard. There may be a
real win in the decimation target, but it is a geometry decision with a visible consequence in the
render, so it wants the measure-the-drawn-output loop and not a size argument.
