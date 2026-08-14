# hopewell-road-sketches

Parametric pedestrian-safety visualization for real intersections — real-world geometry (NJDOT + field measurements + OSM), not hand-drawn sketches. Produces to-scale 2D plan-view before/after comparisons and presentation-quality 3D renders. Currently has one site configured: **Broad St (CR 518) & Greenwood Ave, Hopewell Borough, NJ 08525**.

**[STANDARDS.md](STANDARDS.md)** indexes every published figure the geometry relies on — the R.S. 39:4-138 setbacks, the MUTCD and AASHTO numbers, which constant encodes each one, and which have actually been checked against their source rather than written from memory.

## Quick start

```bash
source .venv/bin/activate   # venv already has geopandas/shapely/trimesh/etc. - see requirements.txt
python scripts/phase1_audit.py --site broad_st_greenwood       # load/clip/audit the road network (one-time, per new site)
python scripts/phase2_geometry.py --site broad_st_greenwood    # curb-line + corner-fillet geometry from the site's config.yaml
python scripts/phase3_treatments.py --site broad_st_greenwood  # apply treatments, render before/after plan view
python scripts/phase4_render_3d.py --site broad_st_greenwood   # export geometry + render both scenarios in Blender
```

`--site` defaults to `broad_st_greenwood` if omitted. Outputs land in `output/<site>/`: `phase1_network_plot.png`, `phase2_geometry_plot.png`, `phase3_before_after.png`, `phase4_render_existing.png`, `phase4_render_proposed.png`. Phase 3 also prints the change summary it draws beside the two panels — crossing distance, exposure, parking, turn speed (see "What the design *achieves*" below).

To rebuild **everything** — all sites, all proposals — in one command instead of ~30:

```bash
python scripts/build_all.py                  # 2D for every site and scenario (~9s)
python scripts/build_all.py --render-3d      # ...and the Blender renders
python scripts/build_all.py --dpi 90         # faster pictures while iterating on geometry
python scripts/build_all.py --refresh-osm    # re-pull OSM after tracing kerbs/crossings
```

It writes the same files the phase scripts do, runs sites in parallel (`--jobs`), and checks every scenario against the scene invariants.

**If you just traced something in OSM, use `--refresh-osm`.** The cached borough snapshot in `output/.cache/borough_*.json` never expires, so a kerb, crossing or `tactile_paving=yes` pad you mapped this morning stays invisible to the build until it's re-pulled — ground truth present, but never reaching the render, which is exactly the failure this project keeps guarding against. So every build prints how old the layers it read are:

```
  columbia_princeton     2 scenario(s)    3.4s  ok
                         OSM cache: 1 layer(s), oldest 3 days old (--refresh-osm to re-pull)
```

`--refresh-osm` re-pulls the whole borough in **one** request, in the parent process before the worker pool starts, and rewrites the cache; the line then reports what was pulled fresh. Every layer at every site is a view over that one download, so it is one visible round trip rather than 20-24 fanned out across four concurrent workers against shared volunteer infrastructure. It is ignored under `HOPEWELL_OFFLINE`, so it can never make the test suite reach the network.

## Tests

```bash
./scripts/test.sh          # takes pytest's arguments:  ./scripts/test.sh -k traced_curbs -x
```

The test tooling (`ruff`, `pytest-regressions`, `hypothesis`) is in `requirements.txt` alongside everything else rather than in a separate dev file, because `tests/test_lint.py` *fails* rather than skips when its linter is missing — deliberately, since its predecessor skipped and so reported success on every machine that hadn't installed pyflakes by hand. A guard that strict has to be installed by the same one command as everything else, or the split just recreates the bug it was meant to prevent. (`pytest` itself was missing from `requirements.txt` too, so the documented install could never run the suite.)

This runs `.venv/bin/python -m pytest` and works whether or not the venv is active. Plain `python -m pytest` only works once you've run `source .venv/bin/activate` — without it, `python` is whatever is on your PATH, and if that interpreter happens to have pytest but not geopandas the suite fails at collection. The root `conftest.py` detects that case and prints one message telling you which interpreter you're on and what to run instead, rather than five `ModuleNotFoundError` tracebacks.

526 tests, ~85 s, **no network**: they run against a committed snapshot of the OSM responses in `tests/fixtures/osm_cache/`, and `HOPEWELL_OFFLINE=1` makes any un-snapshotted fetch fail loudly rather than reach Overpass. Refresh the snapshot with `cp output/.cache/borough_*.json tests/fixtures/osm_cache/` — it does NOT update itself when you re-pull, so after editing OSM you have to do both (see "Kerbside parking varies ALONG a leg" below).

Three of them are not ordinary example-based tests, and are worth knowing about before a failure surprises you:

- **`tests/test_lint.py`** runs `ruff` over `src/ scripts/ tests/ conftest.py`, configured by `ruff.toml`. Undefined names are reported separately from everything else, because they are a guaranteed crash on whatever branch reaches them — which is the whole reason the file exists (an import collapse dropped `fetch_buildings`, referenced only under `--render-3d`, and it surfaced as a `NameError` inside a worker process). Every `ignore` in `ruff.toml` carries the argument for it.
- **`tests/test_geometry_regression.py`** is a golden-file test: it exports every site's baseline and demo scenario and compares a digest against committed files in `tests/test_geometry_regression/`. It fails on any unexplained change to the drawn geometry. **A failure is not automatically a bug** — read the diff, confirm every moved number is one you meant to move, then `./scripts/test.sh tests/test_geometry_regression.py --force-regen` and commit the regenerated goldens *in the same commit as the change that moved them*.
- **`tests/test_frame_properties.py`** is property-based (hypothesis) over the station/offset frame in `src/geometry/model.py`. It asserts the transform's contracts on generated centerlines rather than on the four that exist, so a failure will name a shape nobody wrote down. It has already been useful for what it *disproved*: it established that a polyline kink makes a band of stations around the vertex genuinely unreachable at any offset, which is a fact about the geometry that the placement code works around rather than a defect.

`tests/test_kerbs.py` covers the raised/lowered tagging and the openings it puts in the markings, `tests/test_checks.py` covers the scene invariants (see below), `tests/test_traced_curbs.py` covers building curb lines from traced OSM kerbs, `tests/test_curb_extensions.py` covers curb extensions and bike lanes (and pins what a corner-radius change does *not* do), and `tests/test_sites.py` asserts all four real junctions and every proposal satisfy the invariants.

## What is a type here, and why

Four things in this codebase used to be conventions spread across several modules, and each one generated the same class of bug: something built correctly, drawn in one view, silently missing from the other. They are now types, and the checks that used to catch the mistake afterwards happen when the package is imported.

| Was | Is | What that makes impossible |
|---|---|---|
| A paint kind: a bare string keyed into `PAINT_STYLE`, `PAINT_KIND_LISTS`, `PAINT_FILL_EDGE` and `kind == "bollard"` branches | `src/geometry/markings.py` — a `PaintKind` with a role (line/fill/surface/colour/object) and the channel it travels to the 3D render in | Declaring a marking that reaches no renderer; routing a hatched zone to a channel of lines; a plan-view style table missing an entry. All three raise on import |
| An invariant: a function you had to remember to add to a `+` chain, with a hand-picked argument list | `src/checks.py` — a `SceneCheck` subclass, registered by being defined, reading one `SceneContext` | Writing a check that never runs; handing two checks differently-built versions of the same geometry (that shipped: 15 sq ft apart at W Broad & Louellen) |
| A kerb: a traced OSM line, drawn as one black stroke whatever `kerb=raised`/`kerb=lowered` said about it | `src/geometry/kerbs.py` — a `KerbType`, and a `KerbOpening` where the kerb is dropped for a vehicle | A surveyor's raised/lowered tagging reaching the geometry and no renderer; kerbside paint and flex posts running unbroken across a driveway |
| A treatment: a function writing one of 20 dicts on `DesignState` that five other modules read back, validating whatever its author remembered | `src/geometry/treatments.py` — a `Treatment` frozen dataclass with a typed target from `src/geometry/targets.py`, applied through `DesignState.apply`, and the **only** record of what a design asked for | An unvalidated treatment existing at all; a treatment aimed at a leg the junction doesn't have; a treatment that needs the model being silently skipped; a missing provenance note; a parameter a renderer reads disagreeing with the decision that set it |

`Side` is a `StrEnum`, so it matches OSM's `parking:left` tag keys and the traced kerb attributes (`leg.left_curb`) unchanged, but `Side("north")` raises and the `1 if side == "left" else -1` that appeared in ten places has one home.

A treatment owns its data, its validation, its provenance **and its markings**: `Treatment.paint(ctx)` puts them down through a `PaintContext` holding what is shared (the crossing bands everything is cut around, the apron surfaces everything stops at, `add`/`rim`/`anchors`). `curbside_paint_ft` is a dispatcher over the treatments a design recorded, ordered by a class-level `paint_group`/`paint_rank`, and went from ~350 lines to 126.

That separation is the point. The bike lane's kerb hatching was `add(...)` where every other hatched zone was `rim(add(...))` — one missing call in a 350-line function, invisible in the plan view because matplotlib outlines a fill for free, and visible in the render as hatch strokes ending in mid-air. A marking now sits beside the treatment that calls for it.

The aprons went last, and they are the one case where the order is load-bearing rather than cosmetic. An apron is built ground, so every marking is cut around it — which means the union of them has to be **complete** before anything else paints. That is `paint_group = 0` plus `PaintContext.seal_surfaces()`, and the pass is ordered by the corner the ground lands at rather than by the treatment's own target, because a curb extension is aimed at a leg-side and lays its apron at the corner that kerb feeds. Sealing after the markings instead of before is caught immediately: `markings_collide` reports the apron overlapping a lane-narrowing buffer by 1–4 sq ft at Broad & Greenwood.

Each apron treatment paints its own, from its own fields. There was a `state.corner_aprons` holding one entry per corner, and reading from that would have let two treatments which each asked for an apron there paint one apron between them — a corner with two aprons specified is a design error the collision invariant should report.

### The design is the list of treatments

There were **twenty dicts** on `DesignState` — `lane_narrowing`, `parking_zones`, `bike_lanes`, `daylight_devices`, `corner_aprons` and the rest. Each treatment's `apply_to` wrote one and five modules outside the treatment layer read them back. That is two records of one decision, and they can only agree by convention:

- a leg-name typo wrote a key nothing read, so the treatment silently did nothing (which is why a target is now a type);
- `state.bike_lanes[("east", "north")]` was a perfectly good expression that never matched anything;
- a dict is last-write-wins, so nothing could tell one marked parking lane from two painted on top of each other;
- a test could poke a dict and get a design no scenario could produce.

They are gone. Every parameter every renderer, invariant and policy reads now comes off `state.treatments` through three accessors, and which one you want is a statement about the treatment:

| | | |
|---|---|---|
| `state.treatment_for(kind, target)` | the one at that target, last applied wins | "is there a buffered bike lane on this kerb" |
| `state.treatments_of(kind)` | one per target, **sorted by target** | every marked parking lane in the design |
| `state.every_treatment(kind)` | all of them, in application order | the two treatments that ACCUMULATE |

`treatments_of` sorts because the `props` array in the exported JSON is order-sensitive and the prop builders read the treatments — application order would make that file depend on the order of `BROAD_ST_LEGS` in a site's `scenarios.py`, which is `("broad_st_west", "broad_st_east")`, west first. `every_treatment` exists for `ShiftCrosswalk` and `ExtraProp`, the two that wrote something cumulative (a `+=` and a list append) where the rest wrote a key: two 5 ft shifts move a crossing 10 ft and two RRFBs on one leg are two signs, so collapsing them per target would silently lose one of each.

What is left on `DesignState` is six fields, and each is something the treatment list genuinely does not carry:

| | |
|---|---|
| `legs`, `corner_fillets` | the modelled street — `AddCurbExtension` moves a kerb and `SetCornerRadius` re-cuts a corner, and those writes are the only bodies `apply_to` still has |
| `existing_centerline_styles`, `parking_restrictions` | **observed facts**, seeded by `from_model`: what is painted down each leg today (`config.yaml`, street-view confirmed, or OSM's `overtaking=no`) and what OSM's `parking:*:restriction` says per stretch of each kerb. Neither is a treatment's parameter, and a proposal's own choice is read through `state.centerline_style(leg)`, which lets a `SetCenterlineStyle` outrank the observation |
| `treatments` | the design |
| `notes` | the provenance every export ships |

`Treatment.apply_to` is therefore no longer abstract: for most treatments there is nothing to do there, because being recorded **is** the change. What is left for a subclass is to refuse something only the design can refuse — `AddBikeLane`'s cross-section against the leg's narrowest traced width, `AddBikeLaneBollards`' requirement of a buffer, `ProtectDaylightZone`'s requirement that a `curb_extension` device have an extension under it — or to move the kerb.

`RefugeIsland` and `RaiseCrossing` produce derived **geometry** rather than parameters, and they build it on demand (`polygon(state)`) rather than at apply time. For the raised crossing that is a fix: its start station comes from `leg_clearance_ft`, which reads the corner fillets, and `AddCurbExtension` re-cuts them — so applied *before* an extension on the same leg it used to keep the corner it happened to be measured against while every other marking followed the kerb that moved, putting the same two decisions in the other order 8 ft apart. A design is a set of decisions, not a sequence of snapshots. Their `apply_to` still builds the band and throws it away, for the refusals: a leg with no traced kerbs, or one whose corner return consumes its whole length.

The whole collapse changed **nothing** in the 13 exported scenes — verified key by key with `scripts/export_all_scenarios.py` + `scripts/diff_exports.py`, which is the cheap way to check this (it runs `export_scenario`, so it resolves the scene, builds the paint and the props and asserts every invariant, in ~3 s rather than the ~16 minutes of Blender `build_all.py --render-3d` needs to write the same files).

## Scene invariants

`src/checks.py` holds the things that must be true of every render, checked on **both** the 2D plan view and the 3D export — a check that guards only one lets the two drift, and the entire premise here is that the 2D reconstruction shows what the 3D render will show. Each is a `SceneCheck` subclass; defining one registers it, and `check_scene` is a loop over the registry.

| Invariant | Catches |
|---|---|
| `furniture_in_roadway` | a sign, signal, pushbutton or tactile paving pad standing in the street |
| `pad_off_the_kerb` | a tactile pad nowhere near a curb — it marks a ramp, so it belongs at one |
| `curb_through_junction` | a curb line drawn across the middle of the intersection |
| `curbs_cross` | a leg's two curb lines crossing, closing the roadway to zero width |
| `pavement_ring` | a pavement polygon that isn't simple |
| `crosswalk_off_the_roadway` | a crosswalk floating outside the roadway it crosses |
| `stop_bar_crosses_centerline` | a stop bar painted across the opposing lanes |
| `post_not_in_the_render` | a bollard drawn in the plan view with no prop behind it, so the 3D render builds no post there — the two views take posts from different places, and Broad St's bike lanes shipped with 61 in 2D and none in 3D |

All violations are collected and reported together rather than failing on the first, and each carries coordinates so the plan view can ring them in red where they happen. A violation at a *surveyed* OSM position (an `emergency=fire_hydrant` node inside our modelled roadway) is reported as a source conflict rather than a failure: one of the two sources is wrong, but no edit to this repo fixes it.

### One resolution, three consumers (`src/render/scene.py`)

Checking both paths is necessary but not sufficient: the two paths also have to be looking at the **same geometry**, and for a while they weren't. `SceneGeometry.resolve(model, state, crossings)` now resolves the pavement, the crosswalk offsets/skews/reaches/bands and the stop bars once, and the plan view, the 3D export and the invariants all read that one object. Before it, each of them assembled the sequence itself, and they had already diverged three ways:

- The plan view resolved offsets, skews and stop bars **three times per figure** and built the crossing bands twice with different arguments — with the two-pass mutual-exclusion reaches for the paint it drew, without them for the invariants it checked. At W Broad & Louellen those two bands differ by **15 sq ft**, so the 2D check was validating a crossing neither the 2D view nor the 3D render used.
- The plan view's stop bar was built without the skew stretch factor `stop_bar_bands_ft` applies, drawing it **3.8 ft** from the checked one on Louellen's -44° crossing.
- `tests/test_sites.py`'s helper made the same bands-without-reaches substitution while its docstring claimed to check "exactly what `export.py` and the plan view check".

None of the three was visible from any one call site — each looked locally reasonable, and they only disagreed side by side. That is the argument for resolving once rather than agreeing to follow a convention in four places.

### What the design *achieves* (`src/metrics.py`)

Every dimension on the drawing is an **input**: the 55.5 ft street, the 8 ft stall, `R=20` at the corner. They say what is built. None of them says what it accomplished, and "the crossing is 14 ft shorter and a person is in the road for 4 fewer seconds" is the sentence a proposal is actually argued over — so the before/after figure was two plan views and a forty-row legend, with the reader left to diff them by eye.

`SceneMetrics.of(...)` measures the outcome off the **resolved scene** (`SceneGeometry.metrics(paint)`), never off the config that scene was built from, and `draw_change_panel` puts it beside the panels. Four numbers, and the reason each is measured rather than derived:

- **Crossing distance, curb to curb.** The sum of the two-pass reaches, not the leg's nominal width. `crosswalk_reach_to_curbs_ft` measures out to the *traced* kerbs and the answer is asymmetric (12 ft one way, 20 the other on a 30 ft street); a curb extension changes it. Re-deriving it from `leg.curb_to_curb_ft` would agree with the drawing on a straight symmetric leg, disagree quietly everywhere else, and keep reporting the old width after a treatment moved the kerb.
- **Time exposed to motor traffic** — the **longest stage**, at MUTCD's 3.5 ft/s, measured across the **travel lanes** rather than curb to curb. This was the crossing distance divided by a walking speed, which made the panel's two rows the same measurement under two headings and meant no paint-only proposal could ever move either: a bike lane takes 18 ft of Broad St out of the part a car drives on and the number did not budge. A person standing in a bike lane or a parking lane is not standing in front of a car. The travel lane's edge comes from `travel_lane_edge_ft`, which is the same rule the **stop bar** already stops at — one definition, so the bar and this number cannot disagree about where the lane ends. It is taken from the leg's *allocation* rather than from paint at the crossing, because every treatment is held back from the crossing by the daylight setback: on `broad_st_east` the bike lane is painted from station 26.4 and the crossing sits at 21.3, so sampling polygons there finds bare asphalt on every leg. A refuge island still splits the walk into stages, and the longest one is what counts — summing them would credit the island with nothing. Two limits, both stated rather than buried: bicycle traffic is a real conflict this does not count, which is why the row says *motor* traffic; and **hatching is paint, not a kerb**, so this is the exposure the design intends rather than one anything physically enforces — putting a flex post in the buffer is what makes it true on the ground.
- **Marked parking**, counted off the `PARKING_EDGE_LINE` pieces the paint builder emitted, one run at a time — a hydrant or a driveway splits a kerb into two runs, and a daylight zone shortens a run rather than the leg. `stalls_in_run` is shared with the plan view's per-run label, so the number beside a run and the total in the panel cannot be two different arithmetics.
- **Turn speed at a tightened corner**, from AASHTO's `V = sqrt(15·R·(e+f))` at the low-speed side friction factor. `R=20` against `R=15` means nothing to a reader; ~9.5 mph against ~8.2 does. Labelled *modelled, not measured*, on the same terms an estimated width is dashed rather than solid.

A crossing carries its `CrosswalkOffset` provenance through to the panel, so a distance measured at an estimated position says `est. position`; a leg a proposal marks but that has nothing today is reported as `new` rather than as "0 ft saved", which would be false in both directions. `tests/test_metrics.py` pins all of it against a synthetic junction, so it runs without `data/`.

### Both views frame the same ground (`src/render/frame.py`)

The same failure one level up, in the most literal form available: the plan view framed a hardcoded 110 ft square on the junction node, and the 3D camera framed the pavement's own extent clipped to the modelled legs. Measured on the four sites the 3D frame was **1.15x–1.57x** the 2D frame and centred **6.5–12.5 ft** away from it, so the plan view cropped a third of Broad St's modelled legs — and with them the far ends of the bike lanes the proposal paints — while the render showed all of it. Nobody had chosen that; one number was computed from the geometry and the other was a constant, so the disagreement varied per site.

`junction_frame(model)` now resolves it once: the modelled pavement's extent, clipped at the legs' reach, plus a 20% margin. The plan view sets its axes from it, `export_scenario` writes it into the JSON as `frame`, and `blender_scene.py` points its camera at that rather than recomputing an extent of its own. Two deliberate details: it is measured from the **model** rather than from a `DesignState`, because a curb extension moves the kerb and a before/after pair whose two panels frame differently is exactly what makes two pictures incomparable; and vertices past a leg's far end are dropped, because a traced kerb runs on down the block (425 ft off a 130 ft leg at E Broad) and that is street, not junction. `tests/test_frame.py` compares the plan view's own axis limits against the exported frame, per site.

### "Is this kerb ours" has three answers (`src/geometry/context_roads.py`)

The first wide render — `--frame-scale 2.2` — showed a cross of asphalt floating on grass. The buildings and driveways ran out to 200 m and filled the frame; the street stopped dead at 52 m with sharply cut ends. The obvious suspect was `leg_working_length_ft: 130`, and it was wrong. The cause was one default argument:

```python
for line, tags, _way_id in kerb_lines_with_tags_ft(model.center_wgs84, center_ft)
```

No `legs`, so that returns the **near set** — everything within `KERB_NEAR_JUNCTION_FT` (80 ft) of the junction centre. Its own docstring says that test is for *fitting a corner radius and measuring a width*, and that anything looser "drags in the neighbouring junctions' returns". Correct, for a circle fit. But **8,938 ft of kerb is traced within 600 m of Broad & Greenwood** — both sides of the corridor, Louellen to Princeton — and an 80 ft filter meant for the fit was silently deciding what a drawing contains. The model already knew: `broad_st_west left curb follows traced kerb out to 244 ft`, `e_broad_st_west ... out to 425 ft`. It printed that and then drew none of it.

So there are three relevance tests, and they are not interchangeable: **near the centre** (the radius fit), **along a leg** (a curb line), and **in the picture** (`radius_ft`, for anything being rendered). The third did not exist.

`context_roads.py` then widens each OSM highway way into asphalt, taking the width **from the traced kerbs wherever they exist**, per side and per station. The measured coverage is sharply bimodal, which is why the surveyed/assumed threshold is not a knob anyone has to tune:

| | left | right | |
|---|---|---|---|
| West Broad Street | 100% | 100% | measured both edges |
| East Broad Street | 95% | 100% | measured both edges |
| North Greenwood Avenue | 44% | 59% | its own median, flagged assumed |
| Blackwell Avenue | 38% | 38% | " |
| Model Ave, Railroad Pl, Front St | 0% | 0% | class-assumed ribbon |

Three details worth stating. A side below the threshold still **uses every kerb that was traced** — the fraction sets the provenance flag only, because discarding real measurements for being sparse is the same over-correction as trusting a third of a street and calling it surveyed. A kerb vertex goes to the **nearest** centreline, not to every road within reach, or two parallel streets each claim the kerb between them and widen to meet in the middle. And kerbs are **resampled, not read at their vertices**: a straight kerb is mapped with as few vertices as it takes to be straight (~3 per way here), so reading vertices found kerb at 28 of 82 stations along West Broad and reported a fully traced street as unmapped.

`PavedKind.ROADWAY` joins driveway/aisle/lot rather than becoming a parallel field with its own fetch and its own branch in each renderer — which `PavedSurface`'s docstring already argues against. It is the one kind whose extent can be either surveyed or assumed, so `extent_is_surveyed` decides solid vs dashed in the plan view exactly as it does for a lot vs a driveway.

### A drawing-extent setting was deciding junction membership (`CROSSING_NEAR_JUNCTION_FT`)

Wiring `leg_working_length_ft` to the frame scale — so a treatment runs the length of the drawn street — exposed a bug that was already there. `_matched_crossings` had two guards on whether a mapped crossing belongs to a leg (lateral distance, and orientation) and, longitudinally, only this:

```python
if not (0 < along < centerline.length):
```

So **how far a leg is drawn** decided **which crossing is ours**. Lengthening `broad_st_east` from 170 ft to 374 ft made it adopt the *next junction's* crossing at station 264 — and report it as `osm_survey`. Everything downstream was then correct arithmetic on the wrong crossing:

| | leg 170 ft | leg 374 ft (before the fix) |
|---|---|---|
| `crosswalk_offset` | 21.3 ft `osm_survey` | **264.0 ft** `osm_survey` |
| daylight zone | `0–48.4` — 25 ft from the side line | **`0–289.0`** — 25 ft from *the crosswalk* |
| parkable runs | `[(79.5, 170.0)]` | `[(289.0, 373.9)]` |

A daylight zone at station 268 is not a daylight zone — R.S. 39:4-138 is a setback *from the intersection*. The visible symptom was that lengthening a leg didn't extend the parking, it **slid the whole assembly down the block**, leaving 268 ft of kerb bare.

This is the third instance of one shape of bug in this repo: a feature matched by *"anywhere along the leg"* rather than *"belongs to this junction"* — after the way-nearest-the-midpoint parking matcher, and the near-set kerbs above. The fix is the missing third guard, at the same 80 ft as `KERB_NEAR_JUNCTION_FT` and for the same reason. Measured, the 11 genuine matches across the four sites run **19.5–41.7 ft**, so it is nearly twice the observed worst case and cannot exclude a real crossing, while a neighbour's is hundreds of feet away.

Worth knowing even if you never touch `--frame-scale`: **`working_length_ft` in `config.yaml` was silently a placement setting.** Changing `broad_st_east` from 170 to 200 would have moved the statutory daylight zone 30 ft down the block. It is a drawing-extent knob again.

Because treatments now follow the frame, `SceneMetrics` splits **measured** from **projected**: stalls past the length the site configured are counted separately, and `Comparison.panel_text()` says so. The measured figure is stable across scales (8 stalls on Broad St at both 1x and 2.2x); only the projected part grows.

### The statute applies at every intersection (`src/geometry/cross_streets.py`)

Once legs run 374 ft they cross other streets, and the markings did not know. Stalls were painted straight across the mouth of Blackwell Avenue, and R.S. 39:4-138(e)'s 25 ft setback existed only at the junction the drawing is centred on — which is not what the statute says.

The fact was already fetched and being thrown away. `fetch_roads` pulls every `highway=*` way in range and had been read for exactly one tag, `overtaking=no`; where those ways actually meet ours was discarded. So cross streets now feed the two mechanisms that already exist for this shape of thing — a **no-parking zone** either side (the same statutory rule the junction end gets) and a **kerb opening** across the mouth (what a driveway already gets; a cross street is a driveway a whole street drives out of).

```
zone  253.1..329.1   R.S. 39:4-138(e), 25 ft from the side line of Blackwell Avenue on way 11643011
runs  [(79.5, 253.1), (329.1, 373.9)]
```

Nothing in any marking builder was touched to make this work, and that is the point. Every marking goes through `PaintContext.emit`, which cuts it against `KerbOpenings.against(kind)`, and each kind declares its own behaviour in `markings.py` — `is_fill`, `ZONE_BOUNDARY_LINES`, `dashes_through_openings`. So adding cross streets as a *source* of `KerbOpening` propagated to all of them at once. The bike lane, unprompted:

```
bike_lane_edge_line         77.7..276.6      stops before the mouth
bike_lane_dotted_extension 278.1..280.1      dotted extension ACROSS the conflict area
                           ...302.1..304.1   (7 dashes)
bike_lane_edge_line        305.6..373.9      resumes after
bike_buffer_fill            87.5..268.5 / 313.8..373.9
```

That is the MUTCD treatment for a bike lane through a conflict area, and no code knew a cross street existed. If a new interruption needs adding — a rail crossing, a bus pad — it is a new `OpeningSource` and every marking already handles it.

Three details that are not obvious:

- **It is not a geometric intersection.** A side street's OSM way stops on OSM's centreline for the main road; our leg is the NJDOT alignment, a few feet away. Requiring a true crossing found 2 ways at Broad & Greenwood and *both were Broad Street itself*. The test is approach — a street that reaches inside our own carriageway is meeting us, whoever drew which centreline where.
- **An angle test is required**, because this leg's own OSM way runs alongside it for its whole length. Without it every leg reports itself as its own cross street. Same discriminator `_matched_crossings` uses.
- **One side, not both.** A T-junction opens the kerb it joins; opening the kerb opposite would break paint that is really there. The side comes from the way's own vertices — both sides only for a genuine crossroads.

The setback is measured from the **side line** (the edge of the cross street's carriageway), not its centreline, matching `sideline_station_ft` at the junction end. That width is OSM's `width` tag where a mapper recorded one, else the highway class — an assumption, and `KerbOpening.citation` says so.

If you edit `sites/<site>/config.yaml` (widths, corner radius, crosswalks, treatments, props), rerun from Phase 2 onward — Phase 1 doesn't depend on it.

Phase 4 shells out to Blender (not the project venv — `blender_scene.py` runs under Blender's own bundled Python, with no network access and none of this project's packages). Needs Blender on `PATH`, or set `BLENDER_BIN` — defaults to `/Applications/Blender.app/Contents/MacOS/Blender` on Mac if nothing else is found.

## Adding a new site (intersection)

Everything specific to one intersection lives under `sites/<name>/` - `src/` is a general-purpose library with no hardcoded site data. To add one:

1. `python scripts/phase1_audit.py --street1 "Main St" --street2 "Oak Ave" --anchor "Main St, Sometown, NJ"` - resolves the intersection point via OSM and prints what the road network actually records there.
2. Create `sites/<name>/config.yaml` (copy `sites/broad_st_greenwood/config.yaml` as a template) - fill in `center_wgs84` from step 1, `data_sources` (which road network/parcels files to use - they don't have to be the ones already in `data/`), and one `legs` entry per approach with a `bearing_deg` (compass bearing, 0=N/90=E/clockwise, from the intersection outward along that leg - this is the ONLY thing that has to be geometrically accurate for `src/geometry/intersection.py` to tell the legs apart; nothing else assumes 4 legs or perpendicular roads, so 3-way/5-way/non-perpendicular intersections all work the same way).
3. Create `sites/<name>/scenarios.py` exposing `build_demo_scenario(baseline) -> DesignState` (copy `sites/broad_st_greenwood/scenarios.py` as a template).
4. Run the Quick start commands with `--site <name>`.

## Adding something new to the street

Kerbs, driveways, bike lanes, green surfacing, flex posts and raised crossings all went in the same way, and each of them shipped the same handful of bugs on the way. They are not coincidences — they are the seams of this pipeline, and every one of them is a place where two pieces of code have to agree about the same shape. The checklists below exist so the next element pays for those mistakes once rather than again.

### First decide which of three things it is

| It is… | so it lives… | seeded/applied by |
|---|---|---|
| **a fact about the street as it exists** (a kerb, a driveway, a parking prohibition, a crossing) | on `IntersectionModel`, resolved **once** at load | `load_intersection_model()`, then read onto the state by `DesignState.from_model()` |
| **a decision a proposal makes** (narrow this lane, protect this kerb) | as a `Treatment` subclass, with its derived geometry as a *method* | `state.apply(...)` |
| **a way of drawing something already decided** (green asphalt, a dashed kerb, a taller kerb) | as a `PaintKind` in `src/geometry/markings.py`, or a prop | the treatment's `paint`, or `src/render/props.py` |

Getting this wrong is the single most expensive mistake available here. A fact modelled as a treatment cannot be read by anything that has only a model; a decision materialised at apply time freezes a snapshot the rest of the design then moves out from under (see "The design is the list of treatments"); and geometry built inside a renderer is geometry the *other* renderer will build differently.

### A real-world fact belongs on the model, fetched once

Driveways were fetched and projected in **three** places — the plan view, the export, and the opening logic — each with its own radius constant. That is precisely the divergence `SceneGeometry` was built to stop, committed again one layer down, and it is invisible from any one call site because each fetch looks locally reasonable. If a new element comes from OSM:

1. Add the fetcher to `src/sources/osm_context.py` (it will be a view over the same cached borough snapshot — no new network call).
2. Project it to feet **in `src/geometry/intersection.py`**, store it as a frozen dataclass on the model, and give that dataclass the derived geometry every consumer wants (`Driveway.surface`, not `driveway_width_m` for each renderer to re-widen).
3. If `DesignState.from_model()` reads it, guard the attribute — the test doubles are deliberately partial models. `kerbs.py:kerb_openings_from_model` checks `hasattr(model, …)` for exactly this reason.
4. **Refresh the test fixture separately from the cache.** `output/.cache/borough_*.json` is what a build reads; `tests/fixtures/osm_cache/` is what the suite reads, and it does not update itself.

### A new marking touches six places, and five of them are checked

| Where | What | What happens if you forget |
|---|---|---|
| `src/geometry/markings.py` | `_kind(name, Role, Channel)` — the declaration | nothing else works; this is the registry |
| `plan_view.PAINT_STYLE` | how the 2D view draws it | `require_every_kind` raises **at import** |
| `plan_view.legend_handles()` | its swatch | `test_every_marking_the_plan_view_draws_is_in_its_legend` fails |
| `src/render/export.py` | serialisation | automatic from `CHANNELS` — *unless* the marking needs a new `Role`, which needs a new branch |
| `scripts/blender/blender_scene.py` | how the 3D render draws that channel | **nothing catches this.** Blender runs under its own Python and cannot import `src`, so `markings.CHANNELS` is a data contract with no type behind it |
| `src/render/props.py` | only for `Role.OBJECT` | `post_not_in_the_render` fails the build |

That fifth row is the one unguarded seam in the project, and it is where the bike lane's bollards shipped visible in 2D and absent in 3D. **After adding a channel, look at the render**, not just the JSON.

And after touching anything in `scripts/blender/`, put one scene through Blender *before* committing:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/blender/blender_scene.py -- output/broad_st_greenwood/geometry_proposed.json /tmp/probe.png
```

It takes about five seconds and it is the only check that exists over there. The failure mode is not a wrong picture, it is **no picture**: handing `add_paint_polyline` a `mathutils.Vector((x, y, 0))` where every other caller passes a raw `[x, y]` pair made the point four-dimensional inside `add_paint_line`, and all 13 scenes failed to render. The suite was green throughout, because none of this is importable from the test process.

### Build derived geometry once, from the thing that bounds it

The green bike-lane surface was first built by differencing two kerbside strips. Where the traced kerb is unmapped that difference **overshot its own outer stripe by 6.6 ft**, and neither `MarkingsDoNotCollide` nor `PaintInsideTheCurb` fired — correctly, since there was no other paint there and no traced kerb to be outside of. `model.offset_band_polygon(leg, side, inner, outer, start, end)` builds a band from the two offsets that actually define it, on the same station grid and with the same kerb clamping `inset_line_ft` uses, so a band and the stripes drawn at its edges cannot disagree.

The corollary is a habit, not an API: **measure the geometry you just built.** Print its extent, its area, its distance to the thing it is supposed to touch. Every geometry bug in this repo's history was found by measuring and missed by looking.

### Anything drivable has to break where the kerb opens

`PaintContext.add` clips new paint against surfaces, keep-clear zones and kerb openings, in that order. `PaintContext.emit` deliberately does **not** — a post is a point, not a stripe — so a new element routed through `emit`, or built directly in a prop builder, will happily stand in a driveway. Seven of E Broad's 26 flex posts did. That is worse than not breaking the paint at all: it draws a protected lane whose protection you are expected to drive through. Any new object placed along a kerb needs `stands_in_an_opening(state.kerb_openings, geometry)` applied to it.

### Keep surveyed and assumed numbers separate, and name the assumption

A driveway's mouth is surveyed (the extent of the `kerb=lowered` way); a driveway's own *width* is not — not one of the 43 mapped here carries a `width` tag. Both are "how wide the driveway is" in English and they are 10 ft vs 37 ft on E Broad. So the assumed one is named for what it is (`DRIVEWAY_DRAWN_WIDTH_FT`), documented on `Driveway.width_ft` as the only assumed number in the driveway path, labelled in the legend ("width DRAWN is assumed"), and **loses** to the surveyed extent wherever both exist. When a new element mixes the two, make the constant's name say which it is.

### A new element also has to be *distinguishable*

Driveways were drawn from the day they were modelled, as a thin dashed brown-grey centreline — on a drawing that already carries parcel lines, sidewalk centrelines and leg centrelines at similar weights. They were indistinguishable from three other things, which for review purposes is the same as absent. A new element that is one more line among the lines needs a fill, a weight or a hatch, not just another colour.

### The verification loop

In this order, because each step is cheaper than the next:

1. **Write the test first, and confirm it fails against the pre-change code** (`git stash`, or a worktree — if you use a worktree, symlink the gitignored `data/` into it, or every run crashes in 0.6 s and you will read that as a result).
2. `scripts/export_all_scenarios.py /tmp/before` → change → `/tmp/after` → `scripts/diff_exports.py`. Thirteen scenes, ~2 s, key by key. This runs `export_scenario`, so it resolves the scene, builds the paint and the props and asserts every invariant.
3. `scripts/test.sh` — which now includes the lint pass (`ruff`, via `tests/test_lint.py`) and the golden geometry comparison, so there is no separate linting step to remember. `.venv/bin/ruff check src scripts tests conftest.py` on its own is the sub-second version while you are still editing.
4. `scripts/build_all.py --render-3d`, and then **look at the PNGs** — the only check on the Blender seam.

### Small gotchas that cost real time

- **geopandas rejects dash-tuple linestyles** (`ValueError: inhomogeneous shape`). Use named linestyles (`"--"`, `":"`, `"-."`) in any style dict a GeoSeries plot will see.
- **Place an OSM way as a whole, not vertex by vertex.** Filtering a kerb way's vertices by offset collapsed 4 of 6 openings to 0.0 ft, because a dropped kerb across a driveway mouth is drawn *across* the leg, not along it. `kerbs.py:_place_on_a_leg_side` takes the median offset and station of the way, then measures its span from all of its vertices.
- **Height ordering in 3D is manual.** Pavement extrudes 0.05 m and anything on top of it must be taller. Nothing checks this; sidewalks are currently 0.03 and therefore sit *below* the road.

## The core design principle

**Never trust a generic/geometric guess when real, sourced data exists.** This project repeatedly found that "obvious" defaults (a road network's own width attribute, a geocoder's address match, an assumed corner radius, OSM's map style tag, a CC0 asset's fitness for a specific use) were wrong, missing, or the wrong tool for the job, and the fix was always to go find the authoritative source instead. Concretely:

| What | Source of truth | NOT this |
|---|---|---|
| Intersection location | OSM way-endpoint match (§ below), cross-checked against NJDOT SLD milepost | Nominatim address geocoding (off by ~900 ft) |
| Road widths | NJDOT SLD + Danny's field measurements (`sites/<site>/config.yaml`) | Any width field on the road network file (there isn't one) |
| Crosswalk position | Real OSM-surveyed `highway=footway`+`footway=crossing` geometry, matched to legs | A geometric estimate from corner-fillet tangent points |
| Crosswalk style | Danny's direct confirmation (existing = "lines", matches OSM's `crossing:markings` tag) | Assumption ("probably ladder") |
| Building footprints | Real OSM building outlines | Placeholder boxes |
| Parking lot extent | The `amenity=parking` area as mapped | A width assumed around an aisle centreline |
| Building heights | The assessor's own storey count (MOD-IV `BLDG_DESC`), joined to the footprint through its parcel | One default height for every building in town |
| Pavement/sidewalk material | Real Poly Haven CC0 PBR textures (asphalt_01, pavement_02) | Flat colors |
| Streetlight model | Real Poly Haven CC0 model (street_lamp_01) | Flat colors / a guessed asset URL |
| Traffic signage, trees | **Procedural** geometry, explicitly - no viable CC0 source found (see "Phase 4 fidelity" below) | Silently faking a "real asset" that isn't one |
| Traffic signal pole type, ped-head pole config, NTOR legs | Danny's direct street-view photo confirmation (`sites/<site>/config.yaml` `signals:` block) - real/observed, not surveyed | Assuming a generic signalized-intersection layout |

When no real source exists (e.g. Greenwood Ave has no NJDOT SLD at all — it's a local road; no CC0 traffic-sign model exists on Poly Haven), the code falls back to the best available estimate/procedural geometry and **flags it explicitly** (`confirmed: false` in config, `crosswalk_offset_source: "geometric_estimate"` / prop `"source"` strings in exported JSON, distinct dashed/red styling in 2D renders). Never silently guess, and never silently substitute a worse asset for a real one without saying so.

## Data (`data/`, gitignored — large binaries, not in git)

- `NJ_Roadway_Network.geojson` (170MB) — NJDOT's **statewide** SRI/SLD linear-referencing roadway layer (despite living in a "Hopewell" folder, it's not pre-clipped). Has jurisdiction/route-ID fields (SRI, ROUTE_SUBTYPE, ROAD_NUM) but **no lane count, width, or surface type** — that's why Phase 2 needs the SLD PDF + field measurements.

  **Convert this once, before doing anything else:**

  ```bash
  python scripts/convert_road_network.py
  ```

  GeoJSON carries no spatial index, so a bbox-filtered read still parses the whole file — pulling the ~9 segments around one intersection out of the statewide layer costs **~2.2 s**, versus ~2.5 s to read all 105,838 features (i.e. the bbox filter saves almost nothing). The script writes a FlatGeobuf sibling with a packed Hilbert R-tree, dropping the same read to **~0.002 s** and `load_intersection_model()` from ~2.5 s to ~0.10 s. Every phase script pays that cost at least once, and Phase 3/4 are separate processes that each pay it again.

  It's safe: the script verifies the copy is WKB-identical to the original before keeping it (and deletes it if not), the `.geojson` stays the canonical source and is never modified, and `sites/*/config.yaml` keeps pointing at the `.geojson`. `src/sources/data_loader.py` picks up the sibling automatically (`_resolve_indexed_path`) and ignores it if it's older than the source. The `.fgb` is gitignored — rebuild it, don't commit it. Skipping this step changes nothing but speed.
- `00000518__8.000-11.000.pdf` — NJDOT Straight Line Diagram for Route 518 (Broad St), milepost 8.000–11.000. Our intersection is **MP 10.30**, a signalized crossing inside NJDOT's "West Broad Street" segment. Read this by rendering locally at high DPI (`pdftoppm -r 400 file.pdf page`) and cropping — the pdf-viewer MCP tool's own screenshot is too low-res to read the tick labels.
- `MercerCountyParcels.*` (shapefile) — Mercer County parcel polygons, used for ROW/corner context and to estimate Greenwood Ave's width (see below). `MUN=1105` is Hopewell Borough.
- `MercerTaxList.dbf` — MOD-IV property tax records, joined by PIN to the parcels above. `BLDG_DESC` is the assessor's shorthand for what stands on the lot (`2SF` two-storey frame, `1.5SF 1G` storey-and-a-half with a one-car garage, `B2S` two-storey over a basement), and it is where **building heights** come from — see "Buildings are as tall as the records say" below. Reading the two columns needed out of all 131,631 county rows takes 0.2 s.

A different site can point `data_sources:` at entirely different files (different county's parcels, a different state's road network) - see "Adding a new site" above.

## Key findings worth knowing before you touch anything

- **NJDOT's SLD naming is inconsistent with reality.** The Greenwood Ave cross-street is recorded in NJDOT's system as **SRI `11051089__`, name `COLUMBIA AVE`** — not Greenwood. Confirmed via OSM: the OSM ways for N/S Greenwood Ave and the NJDOT "Columbia Ave" line share the same physical location (within 0.3 ft). If you're looking up SLD/GIS records for this street, search under "Columbia Ave."
- **Geocoding this intersection by address fails.** Nominatim single-string geocoding lands ~900 ft off (returns an arbitrary point along a street, not the corner). `src/sources/data_loader.py:geocode_intersection()` instead finds the two named OSM ways and locates their shared endpoint node — verified against the NJDOT SLD milepost.
- **NJDOT's own West/East Broad St naming split is NOT at Greenwood Ave** — it's further east at the Route 569/Hamilton Ave signal. OSM's naming splits West/East right at Greenwood. Both of our "west"/"east" legs are technically "West Broad Street" per NJDOT.
- **SLD segment-average widths aren't corner-specific.** The SLD says 48 ft nominal pavement for the whole corridor segment; the actual field-measured width at this specific corner is 55.5 ft (west) and 68 ft (east) — real local widening (turn lanes / parking) that a corridor-level SLD entry can't capture. Always prefer a corner-specific field measurement over a segment average when you have one.
- **Real intersection widths, per `sites/broad_st_greenwood/config.yaml`** (as of this writing): West Broad 55.5 ft (confirmed), East Broad 68 ft (confirmed), Greenwood N/S 34 ft each (**estimate** — derived from a measured ~50 ft parcel-to-parcel ROW gap minus an assumed 8 ft/side sidewalk allowance; no NJDOT SLD exists for this local road at all). Existing corner radius: 20 ft (**estimate** — no survey; parcel lot lines are plain straight lines with no chamfer to read a real radius from).

## Repo structure

```
sites/
  README.md                      Config schema every site's config.yaml must follow
  broad_st_greenwood/
    config.yaml                  Per-leg widths/bearings, corner radius, crosswalks, extra signage - see above
    scenarios.py                 build_demo_scenario() - this site's example treatment package
src/                              General-purpose library - no data specific to any one intersection
  site.py            Site discovery/loading (config.yaml + dynamic import of scenarios.py) - see src/site.py
  config.py          Generic YAML loader (no knowledge of sites)
  site_schema.py     What a site's config.yaml must contain, as pydantic models - validated on
                      every load, so a misspelled key or a leg name matching nothing fails
                      immediately instead of drawing something wrong. sites/README.md is its prose.
  geometry/                       Core domain model - pure geometry, no I/O
    model.py         CRS/clipping utilities, Leg dataclass, corner fillets, pavement polygon, leg_clearance_ft
    intersection.py  load_intersection_model() - THE entry point every phase script uses
    kerbs.py         KerbType + KerbOpening: what OSM says a kerb is, and where vehicles cross it
    treatments.py    DesignState + composable treatment functions (see below)
  sources/                        External data fetching - real-world inputs, nothing rendering-specific
    data_loader.py   Road network/parcel loading (paths passed in, not hardcoded), Overpass retry, geocoding
    osm_context.py   OSM building + real crosswalk fetching, disk-cached to output/.cache/
  render/                         Everything that turns a DesignState into a picture (2D plan view or
                                   Phase 4's Blender-ready JSON) - add a new prop/texture/marking type here
    plan_view.py     matplotlib plan-view rendering (Phase 2/3)
    coords.py        WGS84/state-plane/local-meter coordinate conversions (Phase 4 export helper)
    crosswalks.py    Matches real OSM crossings to legs, resolves each leg's crosswalk/stop-bar offset
    props.py         Street-furniture placement: streetlights, signs, traffic signals - WHERE + WHY, not drawing
    assets.py        Poly Haven texture/model fetch + disk-cache to output/.textures/ (Phase 4 fidelity)
    theme.py         Resolves the specific texture/model slugs this project uses into local file paths
    mesh_utils.py    trimesh-based building mesh decimation (Phase 4 fidelity)
    scene.py         SceneGeometry: every marking position a DesignState implies, resolved ONCE and
                       shared by the plan view, the export and the invariants - see below
    export.py        Orchestrator: assembles a DesignState + theme into local-meters JSON for Blender
scripts/
  phase1_audit.py         Load/clip/audit the road network for a site (or a brand-new one via --street1/2/--anchor)
  phase2_geometry.py      Build + plot curb-line/corner geometry
  phase3_treatments.py    Apply demo scenario, plot before/after
  phase4_export_geometry.py  Export-only (no Blender) - useful for debugging the JSON
  phase4_render_3d.py    Full Phase 4 pipeline: fetch theme + export + shell out to Blender
  export_all_scenarios.py  Every site's every scenario through export_scenario, into a directory
  diff_exports.py         Diff two such directories key by key - says WHAT changed, in full detail,
                           once tests/test_geometry_regression.py has told you THAT something did
  test.sh                 Run the suite under .venv/bin/python, activated or not
  blender/                Runs INSIDE Blender's own Python (no network, no venv) - add a new prop/marking
                           DRAWING function here (its placement lives in src/render/ above)
    blender_scene.py       Entry point + top-level scene assembly; imports the 4 sibling modules below
    blender_materials.py   Flat-color and PBR-textured material builders
    blender_geometry.py    Generic mesh helpers: extrude a 2D ring, build from precomputed verts/faces, stripe rects
    blender_crosswalks.py  The 3 painted crosswalk styles (lines/continental/ladder) + dashed centerlines/stop bars
    blender_props.py       Street furniture DRAWING: one builder function per prop type, dispatched by add_prop()
```

## The geometry model

`IntersectionModel` (`src/geometry/intersection.py`) has one `Leg` per approach, each with a `centerline` (from the road network, clipped and simplified) and a `curb_to_curb_ft` from config, from which `left_curb`/`right_curb` offset lines are derived automatically. Legs sharing a road (SRI) are told apart by matching each split centerline piece's compass bearing to the closest `bearing_deg` among that SRI's configured leg entries (`src/geometry/intersection.py:_assign_leg_pieces`) - this is what lets the same code handle any number of legs at any angles, not just a neat 4-way. `build_corner_fillets()` rounds each corner with an analytic tangent-arc fillet (`fillet_curb_corner()`), and `build_pavement_polygon()` stitches every corner's trimmed curbs + arcs into one continuous filled "plus" shape.

`leg_clearance_ft()` computes how far along a leg's centerline you have to go before the roadway is straight (i.e. past the corner curve) — **always project onto the centerline, never use raw Euclidean distance** from a point on the (laterally-offset) curb line, or wide legs will wildly overshoot (a 68 ft leg has a 34 ft half-width, which alone dominates a naive distance calc). Used to place crosswalks, raised-crossing treatments, and props outside the curve, not inside it, whenever real survey/geometric-basis data isn't available.

## Treatments (`src/geometry/treatments.py`)

A treatment is an object, and `state.apply(...)` is the only way one enters a design:

```python
state = baseline.apply(
    AddCurbExtension(LegSide("broad_st_east", Side.LEFT), extension_ft=8.0, crossing_ft=41.5),
    ProtectDaylightZone(LegSide("broad_st_east", Side.LEFT), kind="curb_extension"),
)
```

`DesignState` is immutable-by-clone, so `apply` returns a new design and chains. Constructing a treatment validates it, so an invalid one cannot exist; `apply` checks its target exists at this junction, refuses a treatment that needs the `IntersectionModel` without one, records it in `state.treatments`, and writes the note. All four of those used to be each treatment function's own business, and the ones that forgot were silent: a mistyped leg name wrote a dict key nothing read, and a treatment that skipped its note was absent from the provenance a render ships with.

Four things stay functions, because they are **policies that emit several treatments** rather than one treatment: `apply_osm_parking` (reads OSM's per-stretch parking tags and marks or hatches each kerb accordingly), `complete_centerlines`, `all_crosswalks_continental`, and `bulb_out_corner_pair`.

- `AddCurbExtension(LegSide(leg, side), extension_ft, crossing_ft, ...)` — **the one treatment here that moves a kerb** rather than painting on it. It shifts the kerb line laterally into the roadway near the junction and tapers it back, so everything downstream that measures against the kerb follows without being told: the crossing shortens, the pavement polygon loses the corner, the kerbside paint rebuilds against the new edge, and the invariants check the result. Refused rather than clamped when the leg cannot spare the width beside a `TARGET_LANE_WIDTH_FT` lane.
- `SetCornerRadius(Corner(leg_a, leg_b), radius_ft)` — re-cuts one corner's fillet. **This was called `bump_out` and its docstring claimed "the curb physically extends into the corner". It does not.** See below.
- `AddBikeLane(LegSide(leg, side), width_ft, buffer_ft, parking_ft, shy_ft)` — an exclusive lane with its own edge lines, optionally buffered or parking-protected. `LaneNarrowing` cannot express this: it paints a *buffer*, saying nothing belongs in the strip, where a bike lane says a specific vehicle does. Every width is between paint faces and the stripes' own bodies come out of the buffer, not out of either lane. Refused below AASHTO's 5 ft minimum, and bounded by the leg's **narrowest traced** cross-section rather than its nominal half-width. The cross-section reads *travel lane → buffer → bike lane → hatching → kerb*: a lane is a standard width and the street's spare asphalt is hatched, the same accounting an 8 ft parking stall gets when its remainder becomes a kerb buffer. Drawn without that outer stripe, a 6 ft lane read as reaching the kerb and looked far wider than it is.
- `AddBikeLaneBollards(LegSide(leg, side), spacing_ft)` — flex posts down the buffer on the **traffic** side, which is what makes a lane protected; posts in the kerb-side hatching would protect nothing. Requires a buffer, and refuses without one rather than improvising: E Broad has 17.6 ft to its nearest kerb and an 11 ft lane, a 5 ft lane and their two stripes already spend 17.6 of it, so its lanes are conventional and the proposal says so. A post is an *object*, so it has to reach the 3D render as a **prop** — paint alone draws it in the plan view and nowhere else, which is how these shipped visible in 2D and absent in 3D. Its props are read off the paint (`props.bollard_props_from_paint`) rather than recomputed, because where the row starts depends on how far the crossing reaches on that side, and `post_not_in_the_render` now fails the build if a painted post has no prop.
- `MountableApron(Corner(leg_a, leg_b), extent_ft)` — a flush, drivable corner surface at a fixed depth. Where an apron exists to preserve a large vehicle's swept path around a *tightened* corner its depth is not free, so `AddCurbExtension` records the swept radius instead and the apron is built as the annulus between the two radii (`corner_apron_annulus`).
- `RefugeIsland(LegTarget(leg), offset_ft, width_ft, along_road_ft)` — NACTO 6 ft minimum width enforced. Its footprint is `island.polygon(state)`, built on demand.
- `RaiseCrossing(LegTarget(leg), crossing_width_ft)` — marks a crossing as a raised speed table; placed via `leg_clearance_ft()`, and likewise `raised.polygon(state)` rather than a polygon frozen when it was applied — see "The design is the list of treatments" above for the 8 ft that cost.
- `UpgradeCrosswalkMarkings(LegTarget(leg), style)` — repaints a crosswalk to a more visible style (`"lines"` → `"continental"` → `"ladder"`, FHWA/NACTO visibility ranking). A real, standalone low-cost treatment, not just cosmetic.
- `SetCenterlineStyle(LegTarget(leg), style)` — changes what's painted down a leg's middle (`"single_yellow_dashed"` / `"double_yellow"` / `"none"`). Not a visibility ranking like crosswalk style — just this proposal's choice, over what is actually there today (`DesignState.from_model()` seeds `existing_centerline_styles` per leg from config.yaml's `centerline_style`, since there's no OSM tag for it). `state.centerline_style(leg)` resolves the pair.
- `ShiftCrosswalk(LegTarget(leg), delta_ft)` — moves a leg's crossing further from or closer to the junction, on top of the resolved offset. The one treatment here that is not idempotent: two shifts of a leg add up, which is why it is read with `every_treatment`.
- `ExtraProp(LegTarget(leg), prop_type, offset_ft, side, note)` — one scenario-specific prop (an RRFB, a relocated sign). Also accumulating: two on one leg are two signs.
- `CornerHatching(Corner(leg_a, leg_b), depth_ft)` / `LaneNarrowing(LegTarget(leg), stripe_width_ft, line_only, sides)` / `MarkedParking(LegSide(leg, side), depth_ft, stall_length_ft, curb_offset_ft)` / `LaneNarrowingBollards` / `ParkingBufferBollards` / `ProtectDaylightZone(LegSide(leg, side), kind, spacing_ft)` — the paint-only kerbside and corner treatments, each painting its own markings through `Treatment.paint`.
- `build_sidewalk_pieces(state, sidewalk_width_ft)` — reuses the *same* fillet pipeline at a wider offset to get a sidewalk band that hugs the pavement exactly (12 pieces: 4 leg strips × 2 sides + 4 corner wedges).

### A corner radius is not a curb extension

`bump_out` had no test and no scenario, and its claim was false. Measured on `broad_st_east × greenwood_ave_north`, 29.2 → 15.0 ft:

| | before | after |
|---|---|---|
| arc length | 19.48 ft | 3.51 ft |
| `trimmed_a` | 156.19 ft | 164.19 ft |
| pavement area | 23,989.7 sq ft | 23,989.5 sq ft |
| crossing spans | — | **unchanged to 0.00 ft on all four legs** |

The arithmetic was never wrong; re-cutting an arc between two curb lines leaves both curb lines where they are, and the crossings here sit 21–42 ft out, past the corner. It is now called `SetCornerRadius`, which is what it does, and `AddCurbExtension` is the treatment that shortens a crossing. Both facts are pinned in `tests/test_curb_extensions.py` so a future "fix" cannot quietly go back to tightening radii.

**Nominal width is not crossing length.** The 52.0 and 55.5 ft in `config.yaml` are mid-block cross-sections. The crossings are painted where the traced kerbs have already flared through the corner returns — 39.4 and 31.6 ft off the centerline on `broad_st_east` against a 26.0 ft nominal half-width — so a person crossing Broad St today walks **65.0 ft** of asphalt, not 52. An 8 ft extension per side reads as "52 → 36" on the cross-section and is really **65.0 → 35.5 ft** on the ground.

### Proposals

| Site | Scenario | What it does |
|---|---|---|
| broad_st_greenwood | `build_proposal_bike_lanes` | The standard section — a 5 ft lane with a 2 ft buffer (`BIKE_LANE_WIDTH_FT` / `BIKE_LANE_BUFFER_FT` in `src`, since it is a standard rather than a per-site choice) — flex posts down each buffer, both sides of both Broad legs, asphalt painted green. All four kerbs have 21.3–26.6 ft against the 18.8 the section needs, and the surplus is hatched rather than spent on a wider lane. **Not** parking-protected — see below. Greenwood Ave gets none (2.3 / 4.6 ft spare per side, under AASHTO's 5 ft). |
| ebroad_princeton | `build_proposal_bike_lanes` | The same section, asphalt painted green, both sides of both E Broad legs. **Protected on three of the four kerbs**, and the fourth is the finding: the travel lane and the 2 ft buffer are fixed and the bike lane takes what is left, down to a 4 ft floor - so `e_broad_st_east` right carries a **4.49 ft protected lane** where the old rule (hold 5 ft, drop the buffer) gave it a conventional one, and `e_broad_st_east` left comes to 3.80 ft, under the floor, so it falls back to a conventional 5 ft lane and says so. Widening that kerb by **0.20 ft** would buy the fourth. Princeton Ave gets no lane (1.0 ft of lane would be left). No parking displaced - both E Broad legs are already `no_stopping`/`no_parking`. |

**The buffer is kept and the lane gives.** Which way round that goes is a design decision, and this project had it the other way first: it held the lane at a nominal 5 ft and dropped the 2 ft buffer whenever the section did not quite fit, so a kerb **0.51 ft** short lost every flex post to hold six inches of paint. A rider is better served by a 4.49 ft lane with a post beside it than a 5 ft lane with a truck beside it, and 4 ft is a width AASHTO recognises. Hence two constants rather than one: `AASHTO_MIN_BIKE_LANE_FT` (5 ft, the width to design to, and what AASHTO asks where a curb face and gutter pan eat into the lane) and `MIN_BIKE_LANE_FT` (4 ft, the floor below which it is not a bike lane). `widest_protected_lane_ft` applies it, both sites use it, and below the floor the caller falls back to the conventional lane the kerb *can* hold rather than to nothing.

**Parking-protected does not fit Broad St, and the reason is worth knowing.** The 48 ft section (8 parking + 3 buffer + 6 bike + 11 + 11 + 6 bike + 3 buffer) does fit inside 52.0 and 55.5 ft of roadway, but the total is not the constraint: every offset here is measured from the leg centerline, and the *parking side alone* needs 28.0 ft of it against 26.0 / 27.8 nominal and 22.8 / 25.9 at the narrowest traced point. Fitting it would mean shifting the travel lanes off the NJDOT alignment — a real design, but not one this pipeline can draw, since the alignment is the datum every offset, stop bar and crossing frame is measured from.

**Where the ordinance is not tagged in OSM, nothing currently says so — and this is an open gap.** Schedule I of the borough code bans parking 100 ft each way on both Broad St legs, and OSM carries no `parking:*:restriction` for either. `apply_osm_parking` reads the tags, finds none, and marks stalls: `build_demo_scenario` and `build_proposal_daylight_bollards` both paint parking on `broad_st_east` from station **79.5 ft**, so ~20 ft of each run sits inside the 100 ft the ordinance prohibits. The one scenario that handled it hatched that width from the ordinance instead, and it was deleted with the bulb-out proposal. Either tag the restriction in OSM — which is the fix that helps every consumer, not just this repo — or carry the borough schedules as a data source beside the OSM layers. Inferring parking from an absent tag is the one place this project still guesses where it could source.

## Where the kerb is dropped, the markings open

OSM says whether each kerb is `raised` or `lowered`, and this project read the geometry and threw the tag away: the plan view drew one black line for all 95 mapped ways, the 3D render drew **no kerb at all** (the road slab simply met the concrete band), and the kerbside paint ran unbroken past every driveway.

**Both signals are read.** A driveway is mapped twice over — as a `service=driveway` way running up to the road, and as the stretch of kerb it crosses being tagged `kerb=lowered` — and each opens the markings. The kerb is the better evidence, for one specific reason: its extent is *surveyed*, where a driveway centreline carries no width at all and its mouth has to be assumed. Every opening records which source produced it so the citation says so. Reading only the kerb (which an earlier version did, arguing the mismatch would be visible — nothing was comparing them) meant a driveway drawn without its kerb tagged produced no opening and the markings ran across it.

**Parking is paving too.** `amenity=parking` areas and `service=parking_aisle` ways are read alongside the driveways and drawn as the same asphalt, because to a renderer they are the same thing: paved ground that is not carriageway. They differ from a driveway in exactly two ways, and both are modelled rather than glossed:

- **A lot's extent is surveyed** — it is mapped as an *area*, so its outline has the standing of a building footprint or a traced kerb, and nothing about its size is assumed. A driveway and an aisle are centrelines with no `width` on them (0 of the borough's 43 driveways, 0 of its 20 aisles), so their strips are as wide as this project says. The plan view draws a widened outline **dashed** and a surveyed one solid, and the exported JSON carries `surveyed` per surface.
- **Neither opens a kerb.** A lot behind a building crosses no kerb this project models, and an aisle reaches the street through a driveway OSM maps separately, so `model.driveways` — what the opening logic reads — stays driveways only.

An aisle inside a mapped lot is **cut against it**, because two coplanar surfaces at the same height are not redundancy, they are z-fighting. 6 of the 20 aisles are inside one; the other 14 are why the aisle layer is read at all rather than dropped in favour of the areas. Across the four junctions this took the paved ground drawn from 3,290–9,066 sq ft of driveway to 17,993–58,663 sq ft — at E Broad, 44,347 sq ft of parking that had been rendering as grass.

**A driveway is street geometry, so it lives on the model.** `IntersectionModel.driveways`, resolved once at load beside `corner_parcels` and `leg_road_spans`. It was previously fetched and projected three separate times — by the plan view, by the export, and by the opening logic, each with its own radius constant — which is exactly the divergence `SceneGeometry` was built to stop, committed again one layer down.

**The surveyor's convention is read, not inferred.** A driveway here is tagged `wheelchair=no` *and* `tactile_paving=no`; a pedestrian ramp is `yes` and `yes`. Borough-wide that separates them with no overlap:

| tags on a `barrier=kerb` way | n | |
|---|---|---|
| `kerb=lowered` `tactile_paving=no` `wheelchair=no` | 12 | driveways — the markings open |
| `kerb=lowered` `tactile_paving=yes` `wheelchair=yes` | 14 | pedestrian ramps — the crossing band already cuts the paint |
| `kerb=lowered`, neither tag | 1 | unspecified — does **not** open |
| `kerb=lowered` `tactile_paving=yes` `wheelchair=no` | 1 | contradictory — does **not** open |
| `kerb=raised` `tactile_paving=no` `wheelchair=no` | 67 | not an opening at all |

So the test is positive ("the mapper said this is not a pedestrian crossing point") rather than "lowered and not obviously a ramp". `wheelchair=no` alone is not the signal — all 67 raised kerbs carry it too; it only means anything once the kerb is known to be dropped.

What opens is what a vehicle drives over: the band runs from the travel lane's edge out to the real kerb, so the green surface, the outer lines, the hatching and the stalls break while the travel-lane edge line runs straight past. The ends are trimmed back and rounded by 1.5 ft so a turning vehicle has a little room and the gap reads as an entrance — kept small on purpose and pinned by a test, since every foot is a foot of bike lane given up.

**Paint does not all end the same way, so the openings are two shapes** (`paint.KerbOpenings`). One shape for what a car drives over and one termination per marking was the whole reason a driveway looked punched out rather than painted:

| At an opening | ends how | cut against |
|---|---|---|
| a bike lane — **both edge lines and the green between them** | **dotted extension straight across** — 2 ft marks, 2 ft gaps, MUTCD's dotted lane extension | `driven`, then the marks are put back |
| a hatched no-travel zone | **sweeps away on a fillet** — an arc of the strip's own depth, tangent to the zone's edge line at the travel lane, arriving at the mouth at the kerb | `tapered` |
| a parking stall, a flex post | stops at the entrance | `driven` |

The dotted extension is the correction of an argument this file used to make — that a plain gap was "the honest version of the paint not continuing", since the marking did not exist. It was an admission, not a principle: a driveway does not end a bike lane, it crosses one, and a gap says the lane stops and restarts 37 ft later. The marks live in the **geometry**, not in a line style, so the plan view draws short stripes and the render extrudes the same ones; they travel in the bike lane's existing channels, which is why the 3D side needed no new code at all.

**The green is dotted with the lines, from one set of stations.** The lane's two edge stripes and the colour between them are one marking seen three ways, so they are broken at the same places: `PaintContext.opening_dash_spans` measures the crossing once off the lane's own footprint — the surface, since the lines are its edges — and everything is built from those spans. Dashing each along its own arc length instead puts them out of phase, by little on a straight leg and visibly on a curved one, where the inner and outer stripes have different lengths through the same mouth. Measured at E Broad: all 10 green marks land on a white dash span exactly.

**The sweep is a fillet, and which way the arc curves is the whole thing.** On a real street the white line beside the travel lane runs straight, peels away in one continuous stroke around the driveway apron, and comes back — no corner anywhere in it. That is an arc *tangent to the edge line* at the lane, arriving at the surveyed mouth at the kerb, with a radius equal to the strip's own depth (so there is no constant to tune, and the arc uses the whole cross-section). `PaintContext.rim` then paints that curve, so the zone's outline follows it in the 3D render too and not only in the plan view, where matplotlib outlines a polygon for free.

**A zone's edge line goes with its zone** (`markings.ZONE_BOUNDARY_LINES`). Cut at the mouth while the hatching swept away on its fillet, the line ran on with nothing behind it and the fillet's rim cut diagonally across it — a hook and a Y at every driveway in the 3D render. The distinction is not derivable from the role, since both groups are `LINE`s: a zone's edge line belongs to the zone and sweeps with it, while a bike lane's edge line belongs to the lane, which *crosses* the entrance, so it stops at the mouth and continues as the dotted extension. It is therefore declared.

Three versions of this were wrong in ways worth keeping:

- **The profile was measured across the width the band is *requested* at (25.9 ft) rather than the traced kerb it gets *clamped* to (7.6 ft).** Every step then came out within 3% of the full run: a gap 4 ft wider at each end with no taper in it at all. Nothing could have caught it; probing the profile at five offsets showed a constant 47.75 ft immediately.
- **Then the arc curved the wrong way** — tangent to the *transverse* direction at the lane edge, so it was flat exactly where the eye follows the line and turned only in the last foot at the kerb. It measured as a taper (2.5 ft of sweep across 14 ft of depth at Broad St) and still read as a blunt cut at every drawing scale. The test that pins the fix probes the profile at three depths; given the same radius and the same two endpoints, the wrong-way arc fails on **arriving at the mouth** — a bulge keeps 9.0 ft of gap open at the kerb where a fillet closes to the surveyed width.
- **And then the arc was sliced by equal depth**, which put a visible ~1 ft ledge in the sweep. The arc is vertical in (depth, run) at the lane edge — an infinitesimal change of depth there is a large change of station — so uniform depth slices spend all their resolution where the curve is flat and none where it turns. Sliced by equal **run** instead, every step is at most radius/32 (0.44 ft on Broad St), under the trim that smooths it.

**The trim belongs to the mouth, and the rim to the line's own locus.** Two seams came out of getting those wrong. Buffering the fillet along with the mouth grew the sweep by 1.5 ft in every direction *including along its own tangent*, so the curve left the edge line 1.5 ft wide instead of at a point — a bulge exactly where it should be seamless; the fillet is now built as the arc itself, unbuffered, growing from the trimmed mouth. And a rim traced on the fill's boundary sits half a stripe to the side of the line it continues (`lane_edge_stripes` puts the line's centre outside the hatching, with its body filling the space between), which near the tangent stretched into a **1.78 ft break** in the line. The rim is now drawn on the line's locus, held inside the traced kerb — `PaintInsideTheCurb` reported it 0.4 ft over the kerb on all four of Columbia & Princeton's kerbs when it was not.

Rimming the openings also turned up a latent defect one layer down, which is what the invariants are for. A `lane_narrowing_fill` of **0.0 sq ft with a 12.0 ft perimeter** — a hairline left by differencing polygons that share an edge — had been surviving all along, drawn in the plan view as an outline with nothing inside it. Harmless until its boundary started generating rim segments, at which point `MarkingsDoNotCollide` reported 1.7 ft of doubled lane edge line. A zone with no area is not paint, whatever it is drawn around (`MIN_ZONE_AREA_SQ_FT`).

**The posts gap too.** `PaintContext.emit` skips clipping deliberately (a post is a point, not a stripe), so the paint broke over each driveway while the flex posts marched across it — 7 of E Broad's 26. That is worse than not breaking the paint: it draws a protected lane whose protection you are expected to drive through.

Two openings overlap their leg's crossing band and are **reported rather than filtered** (`describe_kerb_openings` names the way behind every one). Overriding a surveyed tag with a geometric guess about what belongs near a corner is what this repo's core principle rules out.

## Kerbside parking varies ALONG a leg

OSM records a fact that changes part way along a street by **splitting the way**, and that is how "no parking for the first 100 ft from the junction" is expressed. This project read **one way per leg** — whichever was nearest the leg's midpoint — and dropped the rest.

At Broad & Greenwood, East Broad is two ways:

| way | covers | says | distance to leg midpoint |
|---|---|---|---|
| `1547092834` | stations 0 – 79.5 ft | `parking:both:restriction=no_parking` | 5.8 ft — **dropped** |
| `11647647` | stations 79.5 – 170 ft | `restriction=none` | 1.9 ft — **chosen** |

The leg's midpoint is station 85, past the split, so the restricted way lost and the render marked parking on a kerb a mapper had just tagged as having none. **Nothing reported a problem**, which is the point: a pipeline that read one way and found no restriction is indistinguishable from one that read the restriction and threw it away.

`IntersectionModel.leg_road_spans` now keeps **every** way lying along a leg with the stretch it covers (`RoadSpan`), and `DesignState.parking_restrictions` carries `ParkingRestriction(start_ft, end_ft, value, way_id)` per kerb. A mapped prohibition then becomes a `NoParkingZone` in `src/geometry/daylighting.py` exactly like a statutory one, so the existing machinery hatches it and excludes stalls from it with no special-casing. Three outcomes instead of two:

- **restricted throughout** → hatched end to end, no stalls
- **restricted in part** → marked for parking, with the restricted stretch carved back out
- **restricted in part, but what's left is under one 22 ft stall** → hatched end to end (`e_broad_st_west` is `no_stopping` over 114.5 of its 130 ft; 15.5 ft is not a parking space)

Whole-leg tags like `overtaking=no` still come from a single way — now the one covering **most** of the leg, rather than the one nearest its midpoint, which has no claim to describing the leg as a whole.

**If you have just edited OSM, the test fixture is separate from the cache.** `output/.cache/borough_*.json` is what a build reads (`--refresh-osm` re-pulls it); `tests/fixtures/osm_cache/` is the committed snapshot the suite runs against, and it does not update itself. Refresh it with `cp output/.cache/borough_*.json tests/fixtures/osm_cache/` — otherwise the tests keep asserting against the street as it was.

## Crosswalk styles: real data over guessing

OSM actually has surveyed crosswalk geometry at this intersection (`highway=footway` + `footway=crossing` ways). `src/sources/osm_context.py:fetch_crossings()` pulls it; `src/render/crosswalks.py:_match_crossings_to_legs()` matches each real crossing to a leg by projecting its midpoint onto every leg's centerline and picking the closest plausible match. When a match exists, its real position is used (`crosswalk_offset_source: "osm_survey"` in the exported JSON) and its `crossing:markings` OSM tag maps to one of our 3 render styles (`lines`/`zebra`→`continental`/`ladder`). No match → fall back to `leg_clearance_ft()` geometric estimate (needed for hypothetical/proposed crossings that don't exist yet).

All 4 real crossings here are tagged `crossing:markings=lines` — confirmed correct by Danny (simple 2-line marking, not ladder or continental). The proposed scenario upgrades 3 of them to continental via `UpgradeCrosswalkMarkings`; the 4th becomes a raised crossing instead.

`blender_crosswalks.py` implements all 3 styles: `add_crosswalk_lines` (2 transverse boundary lines only), `add_crosswalk_continental` (parallel bars, no rails), `add_crosswalk_ladder` (bars + 2 framing rails) — dispatched via `CROSSWALK_STYLES` by the `crosswalk_style` field per leg in the exported JSON.

### The centerline follows the road (`crosswalks.centerline_paint_ft`)

The third instance of the same failure, found by looking at a render: the plan view offset the leg's real centerline, and the 3D render was handed the leg's `near` and `far` points and drew a straight stripe between them — the **chord**. Ten of this project's twelve legs are straight 2-vertex lines, so it looked right nearly everywhere. On the two that are not, the stripe it drew ran **0.16 to 4.14 ft** from where the paint belongs on `broad_st_east` and **0.16 to 5.49 ft** on `louellen_st_west`, the error growing with distance from the junction. Four feet is most of a lane's width of asphalt moved from one side of the road to the other: the double yellow missed the stop bar it is supposed to meet, and the lanes either side of it came out different widths.

Both views now call one function, which returns the stripes themselves — already offset into the two lines of a double yellow, already cut into segments for a dashed one, so the two views break the line in the same places. `export_scenario` writes them as `centerline_paint_m` and `blender_scene.py` draws them; the chord functions remain only as the fallback for a geometry file written before that key. Same lesson as `crosswalk_axes` learned for the crossing frame, in the marking next to it: nothing on the Blender side of the boundary may derive a marking's shape, because nothing over there can be compared against the plan view.

## Centerline styles: another real-vs-assumed fact, no OSM equivalent

Unlike crosswalks, OSM has no tag for what's painted down the middle of a road, so this can't be resolved from real survey data at export time the way crosswalk style is - it has to be a per-leg fact recorded directly in `config.yaml` (`legs.<name>.centerline_style`, confirmed via street-view photo review, same sourcing category as the `signals` block). `DesignState.from_model()` seeds `state.existing_centerline_styles` from it, an observed fact about the street rather than any treatment's parameter; a `SetCenterlineStyle` treatment lets a proposal change what is drawn, and `state.centerline_style(leg)` resolves the two (the proposal wins). This replaced an earlier version of the pipeline that just drew a single dashed yellow line down every leg unconditionally, which happened to be wrong here: **West Broad St, East Broad St, and North Greenwood Ave all have a solid double yellow (no-passing zone) centerline; South Greenwood Ave has no centerline paint at all.** `blender_crosswalks.py:add_double_yellow_centerline` draws two continuous parallel lines (real MUTCD/AASHTO proportions - 6 in lines, 4 in gap) for `"double_yellow"`; `"none"` draws nothing.

## Buildings are as tall as the records say (`src/sources/assessor.py`)

The footprints were always real OSM outlines. The **heights** were one number for the whole town, and that is what made a borough of storey-and-a-half houses render as a field of identical boxes. OSM cannot help here — of the 1150 building ways in the snapshot, `height` appears **0** times, `building:levels` 7 times, and `roof:shape` and `building:material` not at all — so the default *was* the model.

New Jersey's MOD-IV property tax records have the answer, and `data/MercerTaxList.dbf` had been in this repo since before any of the rendering work, described in this file as "joinable by PIN, not currently used". `BLDG_DESC` is the assessor's shorthand for what stands on each lot, and for Hopewell Borough it parses to a storey count on **682 of the 697** parcels that carry a description: 2 storeys (435), 1 (140), 1½ (94), 2½ (8), 3 (5).

The join is geometric and then by PIN: an OSM footprint sits in a parcel (`PAMS_PIN`), whose PIN keys the tax row (`GIS_PIN`). **Largest overlap wins** among the parcels a footprint touches, because a building on a lot line clips its neighbour's parcel by a sliver and that sliver must not decide its height — measured at Broad & Greenwood, 59 of 80 footprints sit in exactly one parcel, 20 straddle two to four, one lands in none, and the median building's best parcel covers 100% of it.

| | before | after |
|---|---|---|
| heights from a record | 2 of 307 (both OSM `building:levels`) | **283 of 307** |
| distinct heights per site | 1–2 | 5–6 |

Ordered by who looked at what: a mapper's `height` or `building:levels` first (they looked at the building; the assessor's record is about the parcel), then the assessor's storeys, then the default. A footprint in no parcel, a parcel in no tax row, or a description with no storey in it (`2G` is a detached garage — 15 of Hopewell's 697) keeps the default and exports `height_source: assumed`, the same contract `crosswalk_offset_source` has.

Roofs stay flat. Nothing in OSM or MOD-IV says what shape they are, so a pitched roof would be invention rather than reconstruction — and it would be **load-bearing** invention, since the ridge is the tallest thing in the render. The consequence is worth stating: a one-storey house is extruded to its 3 m eaves and reads squat, because the roof above it is a thing this project does not know.

## Phase 4 fidelity (textures, props, trees, mesh optimization)

**Textures.** `src/render/assets.py` fetches real CC0 PBR textures from Poly Haven's public API (`asphalt_01` for pavement, `pavement_02` for sidewalks - Diffuse/Roughness/OpenGL-normal maps), caching to `output/.textures/`. Anything within the "near zone" (past the farthest crosswalk + a buffer, computed per-intersection in `src/render/export.py:_split_near_far`) gets the 4k version; everything else gets 2k - this applies to both pavement and sidewalks, split by intersecting with a circle so a piece can straddle the boundary (`pavement_near`/`pavement_far`/`sidewalks_near`/`sidewalks_far` in the exported JSON). `blender_materials.py:make_textured_material()` wires Diffuse→Base Color, Roughness→Roughness, normal→Normal Map, and falls back to a flat color if a texture path is missing or fails to load - Phase 4 must never hard-fail without network access. Each extruded piece gets a real-world-scaled planar UV projection (`blender_geometry.py:apply_planar_uv`, `bpy.ops.uv.cube_project`) so the tiling reads consistently across differently-sized pieces.

**Streetlights.** A real Poly Haven model (`street_lamp_01`, glTF bundle at 1k texture resolution - the 8k default would be enormous for a background prop instanced 4 times) is fetched once and imported as a hidden template; each corner gets a cheap linked duplicate (`obj.copy()`, sharing mesh data) positioned at that corner's fillet-arc midpoint (real geometry) pushed a few feet onto the sidewalk (a placement approximation, flagged in the exported JSON's prop `"source"` field). Falls back to a procedural pole+box if the model can't be fetched.

**Signage.** No CC0 stop-sign or school-zone-sign model was found on Poly Haven (their catalog has no traffic signage) or reliably fetchable from Kenney.nl (no stable public API/URLs to fetch from without guessing - which this project's own principle rules out). Built procedurally instead: correct MUTCD shape/color (octagon/red for stop signs, pentagon/yellow-green for school zone), a real post + flat plate mesh. One stop sign per approach (`src/render/props.py:_stop_sign_props`, placed near the leg's near-corner curb - an approximation, not a real traffic-engineering placement study) plus whatever's listed in a site's `config.yaml` under `props.extra` (e.g. the school zone sign on Broad St West here - genuinely site-specific, unverified-against-a-real-inventory knowledge that belongs in the site config, not the general pipeline).

**Traffic signals.** Replaces the old `signalized: true` flag (nothing downstream ever read it) with modeled geometry driven by a site's `signals:` config block (see `sites/README.md`) - pole type, per-corner pedestrian-head pole configuration, and no-turn-on-red legs, all confirmed via **direct street-view photo review** (not a field survey, but a real observed fact rather than a geometric placeholder). `src/render/props.py:_traffic_signal_props` places a `traffic_signal_pole` at each configured corner's fillet-arc midpoint (the same real geometry `_corner_streetlight_props` uses) plus a `pedestrian_signal_head`, either co-located with the pole (`pedestrian_head: same_pole`) or offset a few feet along the sidewalk onto its own post (`separate_pole` - a flagged placement approximation, same pattern as `STREETLIGHT_SIDEWALK_SETBACK_FT`). `_no_turn_on_red_props` places a small rectangular NTOR sign on each restricted approach, positioned like the automatic stop signs. `blender_props.py` builds all of it procedurally - a full-width mast arm (confirmed NOT a short pole-mounted rigid/davit arm or span-wire; the arm's reach is derived per-corner from real adjacent leg widths, `src/render/props.py:_traffic_signal_props`, not a fixed constant) with a MUTCD-accurate stacked red/yellow/green 3-section vehicle head (`add_traffic_signal_pole`/`add_vehicle_signal_head`), a small pedestrian head (`add_pedestrian_signal_head`), and a white rectangular NTOR plate (`add_no_turn_on_red_sign`), all dispatched by `add_prop()` off each exported prop's `"type"` field. Each signalized approach also gets a **stop bar** - a single transverse marking placed just behind (intersection side of) its crosswalk (`src/render/crosswalks.py:resolve_stop_bar_offsets`, `scripts/blender/blender_crosswalks.py:add_stop_bar`), only drawn when the site config has a `signals` block. **Exactly which approach each signal head visually aims at is a render-fidelity simplification, not a signal-timing-accurate model** - real per-arm aiming/phasing isn't modeled.

**Trees.** One low-poly procedural tree (cone + cylinder - no CC0 source of genuinely low-poly stylized trees was found; Poly Haven's tree models are realistic multi-material photoscans, disproportionately heavy for background dressing instanced many times at this render's scale) is instanced along each sidewalk piece via **Blender geometry nodes** (`GeometryNodeInstanceOnPoints`, `src/render/export.py:_tree_points_along_piece` samples points along a piece's long axis at 25 ft spacing - standard municipal street-tree spacing, not a fabricated number; corner wedge pieces are skipped as not meaningfully elongated). Geometry-node instancing shares one mesh's data across every instance rather than creating N copies - the actual performance property requested, not just a style choice.

**Building mesh optimization.** OSM buildings are background context, not the render's subject. `src/render/mesh_utils.py` extrudes each footprint with `trimesh` and applies quadric decimation (`fast_simplification` backend) to anything genuinely heavy. This file used to say the threshold (40 faces) made decimation "a no-op today", and that was wrong in a way that showed: a prism off an n-vertex footprint is about 4n-4 triangles, so 40 faces is an 11-sided building - an ordinary house with a porch. **Nine of Broad & Greenwood's 80 buildings crossed it** (44 to 100 faces) and were crushed to 24, and quadric decimation does not know a roof is meant to be flat - it collapses the cheapest edges, which on a short extrusion are the vertical ones, leaving a crumpled tent. Four buildings rendered that way, and the assessor-derived heights made it worse (3 -> 4) because a shorter extrusion is cheaper to collapse. All 80 buildings undecimated come to **1,692 triangles**, in a scene carrying textured pavement, instanced trees and procedural signal heads: there was nothing to save. The threshold is now 400 faces (a ~100-sided outline, four times the most complex building at any of these junctions), and `test_a_building_keeps_its_flat_roof` pins the shape rather than the number. **Gotcha hit:** `trimesh` always triangulates, which reads as a faceted/crystalline shape under Blender's default flat shading even for an undecimated simple box - fixed with `bpy.ops.mesh.dissolve_limited()` after building the mesh, merging coplanar triangles back into flat faces.

## Phase 4 (3D) general notes

- Render engine: `BLENDER_EEVEE_NEXT` (the only one in Blender 4.3 — old EEVEE identifier is gone).
- Blender's own Python has no network access, no `requests`, and no access to this project's venv - all fetching (`src/render/assets.py`, `src/sources/osm_context.py`) happens beforehand in the normal venv-based scripts, which pass only local file paths / already-fetched data into the exported JSON. None of the `blender_*.py` scripts call out to the network.
- **Blender does NOT put a `--python` script's own directory on `sys.path`** (unlike plain `python script.py`) - confirmed empirically, not assumed. `blender_scene.py` inserts it manually before importing its sibling `blender_materials`/`blender_geometry`/`blender_crosswalks`/`blender_props` modules, or those imports fail with `ModuleNotFoundError` regardless of cwd.
- **Marking height must exceed pavement height** — pavement is extruded 0.05 m; anything meant to be visible on top of it (crosswalks, centerlines) must be taller (0.06 m used) or it renders buried inside the solid pavement block with zero visible effect.
- **Blender's multi-object edit mode re-extrudes every *selected* mesh, not just the active one.** Always `bpy.ops.object.select_all(action='DESELECT')` before entering edit mode on a single object, or previously-created objects (e.g. the ground plane) silently accumulate extra height every time something else gets extruded.
- OSM building footprints don't reconcile with our precise curb geometry — a few end up overlapping the pavement. `export.py` filters any building whose footprint intersects the pavement polygon. (Buildings that just look close to the road in the render are legitimate — small-town buildings really do sit near the curb; verify with a numeric intersects check before assuming a rendering bug.)
- `scripts/blender/blender_scene.py` accepts any number of `<geometry.json> <output.png>` pairs and renders them all in **one Blender process** — each launch has ~1–1.5s fixed startup overhead, not worth paying per-render. `phase4_render_3d.py` uses this to do both scenarios in one shot.
- Every OSM layer is a view over one cached borough snapshot in `output/.cache/` (see "The borough snapshot" in `src/sources/osm_context.py`); `assets.py`'s texture/model fetches cache to `output/.textures/`, keyed by (slug, resolution). Delete the relevant directory to force a refetch. Each layer view is also memoized in-process against the snapshot it came from, so the eight fetchers stop rescanning 2.9 MB of OSM on every call - and a re-pull cannot be served a view of the old snapshot, because an entry is only used while the snapshot object it was built from is still the one in hand.
- Overpass's public instances are flaky (504s are common) — `src/sources/data_loader.py:query_overpass()` retries across 3 mirrors (`overpass-api.de`, `kumi.systems`, `openstreetmap.ru`) before giving up.
- EEVEE samples: 64 (dropped from 128 - visually indistinguishable for this flat-shaded scene, ~30% faster). Current full render (both scenarios, all fidelity features, warm caches): ~13s total.

## Known gaps / next steps

- Greenwood Ave (N & S) widths and the existing corner radius are still estimates — need field measurement or survey/aerial confirmation. Once available, update `sites/broad_st_greenwood/config.yaml` and rerun from Phase 2.
- East Broad St's "54 ft active roadway" vs "68 ft total" distinction isn't used anywhere yet (only the 68 ft total is) — could matter if a future treatment needs lane-level detail.
- ~~Asset-library question (Poly Haven PBR textures, free low-poly prop packs)~~ **Resolved** - real Poly Haven textures (asphalt/concrete) and a real streetlight model are wired in; procedural fallbacks (flagged, not hidden) cover what has no viable CC0 source (signage, low-poly trees). See "Phase 4 fidelity" above.
- Only one demo treatment scenario exists per site (`build_demo_scenario`). Additional scenarios would just be new functions in that site's `scenarios.py` composing the same treatment primitives.
- Prop placement (streetlights, stop signs, the school zone sign) is grounded in real corner/leg geometry but the *exact* setback/offset distances are approximations, not a surveyed signage inventory - flagged via each prop's `"source"` field in the exported JSON.
- **A driveway strip is not clipped at the kerb.** An OSM driveway way runs to the road's *centreline*, so widening it into a strip paints over the carriageway. Measured across all four sites (ten driveway/site pairs, eight distinct ways — the two at E Broad fall inside Columbia & Princeton's context radius as well), exactly one does this: way `772378207` at E Broad, 187 of its 1,577 sq ft (12%) inside the modelled roadway, visible in the plan view as a tan band over the travel lane. Every other pair overlaps 0%, because they connect beyond where the modelled legs stop.
- **Only one mapped driveway reaches a modelled kerb** (`772378207`). The rest are 21.7–352 ft from one, so they render as strips ending in grass — the legs stop at ~130 ft, not the driveways being wrong. Conversely most dropped kerbs near these junctions have no driveway way mapped at all (nearest 200–500 ft), so most openings are surveyed kerb with nothing visible behind them.
- Sidewalks extrude 0.03 m against pavement's 0.05, so a footway sits *below* the road surface in the 3D render.
- The bike lane's 2 ft buffer, with this project's 0.82 ft (10 in) edge stripes, leaves 0.36 ft of visible asphalt between them, so the buffer's diagonal hatching no longer reads in 3D. Three ways out: widen the buffer, narrow `LANE_EDGE_LINE_WIDTH_FT` toward a real 6 in, or stop hatching a buffer that narrow. Not chosen yet.
- Building mesh decimation now fires only on genuinely complex footprints (400+ faces), which none of these four junctions has - it'll matter once/if a site uses richer building data. Until it was measured it was firing on nine ordinary houses per junction and mangling four of them; see "Building mesh optimization" above.
