# hopewell-road-sketches

Parametric pedestrian-safety visualization for real intersections — real-world geometry (NJDOT + field measurements + OSM), not hand-drawn sketches. Produces to-scale 2D plan-view before/after comparisons and presentation-quality 3D renders. Four junctions are configured; `broad_st_greenwood` (Broad St / CR 518 & Greenwood Ave, Hopewell Borough, NJ 08525) is the default.

**Where the answers live.** This README is the map: what exists, where it lives, how to run it, and the facts about *this* street that no module owns. The reasoning behind a module's design lives in that module's own docstring, because that is where someone editing the code will see it. Three files sit beside this one:

- **[.claude/SKILLS.md](.claude/SKILLS.md)** — read before changing geometry. What people (and agents) actually get wrong here: the two datums 25 ft apart, the constants that already exist, and the rule that matters most — measure the DRAWN output, not the arithmetic that was supposed to produce it.
- **[STANDARDS.md](STANDARDS.md)** — every published figure the geometry relies on (R.S. 39:4-138, MUTCD, AASHTO, NACTO, NJDOT), which constant encodes it, and which have been checked against their source rather than written from memory.
- **[docs/](docs/)** — `network-model.md` (replacing legs with a road network, with its checkpoints) and `network-renderer-plan.md` (the corridor/coverage work, with what each stream owes).

## Quick start

```bash
source .venv/bin/activate   # venv already has geopandas/shapely/trimesh/etc. - see requirements.txt
python scripts/phase1_audit.py --site broad_st_greenwood       # load/clip/audit the road network (one-time, per new site)
python scripts/phase2_geometry.py --site broad_st_greenwood    # curb-line + corner-fillet geometry from the site's config.yaml
python scripts/phase3_treatments.py --site broad_st_greenwood  # apply treatments, render before/after plan view
python scripts/phase4_render_3d.py --site broad_st_greenwood   # export geometry + render both scenarios in Blender
```

`--site` defaults to `broad_st_greenwood`. Outputs land in `output/<site>/`: `phase1_network_plot.png`, `phase2_geometry_plot.png`, `phase3_before_after.png`, `phase4_render_existing.png`, `phase4_render_proposed.png`. Phase 3 also prints the change summary it draws beside the panels — crossing distance, exposure, parking, turn speed (`src/metrics.py`).

To rebuild **everything** — all sites, all proposals — in one command instead of ~30:

```bash
python scripts/build_all.py                  # 2D for every site and scenario (~9s)
python scripts/build_all.py --render-3d      # ...and the Blender renders
python scripts/build_all.py --dpi 90         # faster pictures while iterating on geometry
python scripts/build_all.py --refresh-osm    # re-pull OSM after tracing kerbs/crossings
```

It writes the same files the phase scripts do, runs sites in parallel (`--jobs`), and checks every scenario against the scene invariants.

**If you just traced something in OSM, use `--refresh-osm`.** The cached borough snapshot in `output/.cache/borough_*.json` never expires, so a kerb, crossing or `tactile_paving=yes` pad you mapped this morning stays invisible to the build until it is re-pulled — ground truth present, but never reaching the render, which is exactly the failure this project keeps guarding against. Every build therefore prints how old the layers it read are:

```
  columbia_princeton     2 scenario(s)    3.4s  ok
                         OSM cache: 1 layer(s), oldest 3 days old (--refresh-osm to re-pull)
```

`--refresh-osm` re-pulls the whole borough in **one** request before the worker pool starts, rather than 20-24 fanned out across four workers against shared volunteer infrastructure. It is ignored under `HOPEWELL_OFFLINE`, so it can never make the test suite reach the network.

Two more scripts read a whole corridor rather than one junction: `scripts/corridor_report.py` answers the corridor questions with the coverage of each answer beside it, and `scripts/corridor_render.py` draws a straightened strip plan of a whole street on stacked panels.

## Tests

```bash
./scripts/test.sh          # takes pytest's arguments:  ./scripts/test.sh -k traced_curbs -x
```

This runs `.venv/bin/python -m pytest` and works whether or not the venv is active. Plain `python -m pytest` only works after `source .venv/bin/activate`; the root `conftest.py` detects the wrong-interpreter case and prints one message rather than five `ModuleNotFoundError` tracebacks.

The test tooling (`ruff`, `pytest-regressions`, `hypothesis`) is in `requirements.txt` rather than a separate dev file, because `tests/test_lint.py` *fails* rather than skips when its linter is missing — deliberately, since its predecessor skipped and so reported success on every machine that had not installed pyflakes by hand.

Facts worth knowing before a failure surprises you:

- **Order is shuffled every run** (pytest-randomly), which is how you find out whether the session-scoped `site_models` fixture is as read-only as it claims. The header prints `Using --randomly-seed=N` and `./scripts/test.sh --randomly-seed=N` replays it. (`pytest.ini` sets `-q`, which hides that header — CI runs without it for this reason.)
- **No network.** The suite runs against a committed snapshot of the OSM responses in `tests/fixtures/osm_cache/`, and `HOPEWELL_OFFLINE=1` makes any un-snapshotted fetch fail loudly. That snapshot is **separate from the build cache and does not update itself**: after editing OSM, `cp output/.cache/borough_*.json tests/fixtures/osm_cache/` as well as `--refresh-osm`.
- **CI has no `data/`**, so every test that builds a real junction skips there — *including the golden geometry comparison*. A green tick means the code is sound, not that the renders are. The goldens are a local guard.
- **`tests/test_lint.py`** runs `ruff` (per `ruff.toml`) over every `.py` file, reporting undefined names separately because they are a guaranteed crash, and `import-linter` (`.importlinter`) over the import *graph* for three rules no single file can show you: `scripts/blender/*` must not import this project or its venv; reading a `config.yaml` must not drag in shapely; and geometry must not import the output stages. Every `ignore` and every contract carries the argument for it.
- **`tests/test_geometry_regression.py`** is a golden-file test over every site's scenarios. **A failure is not automatically a bug** — read the diff, confirm every moved number is one you meant to move, then `./scripts/test.sh tests/test_geometry_regression.py --force-regen` and commit the regenerated goldens *in the same commit as the change that moved them*.
- **`tests/test_frame_properties.py`** is property-based (hypothesis) over the leg frame in `src/geometry/model/`, so a failure will name a shape nobody wrote down.

## The four types, and why they are types

Four things here used to be conventions spread across modules, and each generated the same class of bug: something built correctly, drawn in one view, silently missing from the other. They are now types, and the checks that used to catch the mistake afterwards happen at import.

| Was | Is | What that makes impossible |
|---|---|---|
| A paint kind: a bare string keyed into several tables and `kind == "bollard"` branches | `src/geometry/markings.py` — a `PaintKind` with a role (line/fill/surface/colour/object) and the channel it travels to the 3D render in | Declaring a marking that reaches no renderer; a plan-view style table missing an entry. Both raise on import |
| An invariant: a function you had to remember to add to a `+` chain | `src/checks.py` — a `SceneCheck` subclass, registered by being defined, reading one `SceneContext` | Writing a check that never runs; handing two checks differently-built versions of the same geometry |
| A kerb: a traced OSM line, drawn as one black stroke whatever `kerb=raised`/`lowered` said | `src/geometry/kerbs.py` — a `KerbType`, and a `KerbOpening` where the kerb is dropped for a vehicle | A surveyor's tagging reaching the geometry and no renderer; kerbside paint running unbroken across a driveway |
| A treatment: a function writing one of 20 dicts on `DesignState` that five other modules read back | `src/geometry/treatments/` — a `Treatment` frozen dataclass with a typed target from `src/geometry/targets.py`, applied through `DesignState.apply`, and the **only** record of what a design asked for | An unvalidated treatment; one aimed at a leg the junction lacks; a missing provenance note; a renderer's parameter disagreeing with the decision that set it |

`Side` is a `StrEnum`, so it matches OSM's `parking:left` tag keys and the traced kerb attributes unchanged, but `Side("north")` raises.

**The design is the list of treatments.** Every parameter every renderer, invariant and policy reads comes off `state.treatments` through three accessors — `treatment_for(kind, target)`, `treatments_of(kind)` (one per target, sorted by target) and `every_treatment(kind)` (all, in application order, for the two treatments that accumulate). What is left on `DesignState` is the modelled street, two observed facts (`existing_centerline_styles`, `parking_restrictions`), the treatment list and the provenance notes. `src/geometry/treatments/state.py` documents each accessor and why the sort order is load-bearing.

**A treatment owns its markings too.** `Treatment.paint(ctx)` puts them down through a `PaintContext`, and `src/geometry/paint.py:curbside_paint_ft` is a dispatcher over the treatments a design recorded, ordered by a class-level `paint_group`/`paint_rank`. Aprons paint first because everything else is cut around them; `src/geometry/paint.py` documents the ordering and what breaks without it.

## Scene invariants

`src/checks.py` holds the things that must be true of every render, checked on **both** the 2D plan view and the 3D export — a check guarding only one lets the two drift, and the entire premise here is that the 2D reconstruction shows what the 3D render will show. Each is a `SceneCheck` subclass; defining one registers it.

| Invariant | Catches |
|---|---|
| `furniture_in_roadway` | a sign, signal, pushbutton or tactile pad standing in the street |
| `pad_off_the_kerb` | a tactile pad nowhere near a curb — it marks a ramp, so it belongs at one |
| `curb_through_junction` | a curb line drawn across the middle of the intersection |
| `curbs_cross` | a leg's two curb lines crossing, closing the roadway to zero width |
| `pavement_ring` | a pavement polygon that isn't simple |
| `crosswalk_off_the_roadway` | a crosswalk floating outside the roadway it crosses |
| `stop_bar_crosses_centerline` | a stop bar painted across the opposing lanes |
| `post_not_in_the_render` | a bollard drawn in 2D with no prop behind it, so the 3D render builds no post |
| `markings_collide` / `paint_inside_the_curb` | paint doubled over itself, or outside the traced kerb |
| `crossings_are_not_painted_over` | paint laid across a surveyed crossing, including one at a junction this site does not model |
| `coverage` | a surveyed feature inside the drawn frame that the drawing omits (`src/geometry/coverage.py`) |

All violations are collected and reported together, each carrying coordinates so the plan view can ring them in red. A violation at a *surveyed* OSM position (a fire hydrant inside our modelled roadway) is reported as a source conflict rather than a failure: one of the two sources is wrong, but no edit to this repo fixes it.

## Repo structure

Every module's own docstring says what it is for and why it is shaped that way. This tree is the index.

```
sites/
  README.md                Config schema every site's config.yaml must follow
  <site>/config.yaml       Per-leg widths/bearings, corner radius, crosswalks, signals, extra props
  <site>/scenarios.py      This site's baseline + proposal builders
src/                       General-purpose library - NO data specific to any one intersection
  site.py                  Site discovery/loading (config.yaml + dynamic import of scenarios.py)
  site_schema.py           What a config.yaml must contain, as pydantic models - sites/README.md is its prose
  config.py                Generic YAML loader (no knowledge of sites)
  checks.py                Scene invariants, checked on both the 2D and 3D paths
  metrics.py               What a design ACHIEVES, measured off the geometry it drew
  provenance.py            How well-sourced a number is, and which source wins when two disagree
  geometry/                Pure geometry, no I/O
    intersection/          Building an IntersectionModel from config + OSM + NJDOT
      load.py              load_intersection_model() - THE entry point every phase script uses
      junction.py          What a junction IS once built: the model every phase reads
      fitting.py           Fitting the legs to the traced kerbs - the heaviest thing here
      kerb_sources.py      Traced kerb out of OSM and into state-plane feet
      osm_roads.py         Tying our legs to NJDOT SRI centrelines and OSM ways
      paved.py             Driveways, parking aisles and lots
    model/                 The measurement primitives
      crs.py               Projections, and the operations only valid in one
      leg_frame.py         (station along the centreline, lateral offset from it)
      corners.py           Corner fillets and the pavement polygon they close
      traced_kerbs.py      A surveyor's traced kerb, as geometry to measure against
      stripes.py           The geometry of paint itself: strips, tapers, stall lines, post rows, hatching
      context.py           Measurements about the surroundings rather than the carriageway
    treatments/            What a proposal DECIDES - one module per family
      state.py             DesignState: the thing every treatment transforms
      base.py              The Treatment ABC, shared value objects, shared constants
      corners.py           Corner radius, curb extensions, aprons, corner hatching
      bikeways.py          Bike-lane cross-sections, the treatments that place them, their bollards
      crossings.py         Refuge islands, raised crossings, crosswalk markings, crosswalk shifts
      parking.py           Marked stalls, their buffer, and the borough's parking tags
      lanes.py             Lane narrowing and the flex posts that hold it
      corridor.py          A facility declared once for a route and applied at every junction on it
      extras.py            Scenario-specific props, and the sidewalk band
    targets.py             What a treatment is applied TO: a leg, one kerb of a leg, a corner
    markings.py            Every marking kind and every renderer channel, declared once
    paint.py               Every piece of curbside paint a DesignState calls for, built once
    kerbs.py               Whether a kerb is raised, and where it is dropped for a vehicle
    daylighting.py         Where a car may legally park near these junctions; the rest is marked clear
    cross_streets.py       Where a leg crosses ANOTHER street, and what that costs the kerb
    context_roads.py       The streets around the junction, built from the kerb actually traced
    surveyed.py            Every crossing the surveyor traced inside the frame, drawn as traced
    coverage.py            Does the drawing contain every surveyed feature inside its own frame?
    network.py             A ROAD: one street with continuous stationing, through a borough
    corridor_paint.py      The facility painted along a ROAD, not along a leg
  sources/                 External data - real-world inputs, nothing rendering-specific
    data_loader.py         NJDOT network + parcels, Overpass retry, intersection geocoding
    osm_context.py         The cached borough snapshot every OSM layer is a view over
    assessor.py            MOD-IV tax records joined to OSM footprints (building heights)
    schemas.py             What this project requires of each external layer, as pandera schemas
  render/                  Everything that turns a DesignState into a picture
    scene.py               SceneGeometry: every marking position, resolved ONCE and shared
    frame.py               The one piece of ground both views are pointed at
    plan_view.py           matplotlib plan-view rendering (Phase 2/3)
    crosswalks.py          Matches OSM crossings to legs; resolves crosswalk/stop-bar/centreline paint
    props.py               Street-furniture placement: WHERE + WHY, not drawing
    export.py              Orchestrator: DesignState + theme -> local-meters JSON for Blender
    coords.py              WGS84 / state-plane / local-meter conversions
    assets.py              Poly Haven texture/model fetch + disk cache
    theme.py               Which texture/model slugs this project uses
    mesh_utils.py          trimesh building-mesh decimation
scripts/
  phase1_audit.py          Resolve + audit the road network for a site (or a new one via --street1/2/--anchor)
  phase2_geometry.py       Build + plot curb-line/corner geometry
  phase3_treatments.py     Apply a scenario, plot before/after
  phase4_export_geometry.py  Export-only (no Blender) - useful for debugging the JSON
  phase4_render_3d.py      Fetch theme + export + shell out to Blender
  build_all.py             Every site, every scenario, in parallel
  export_all_scenarios.py  Every scenario through export_scenario, into a directory
  diff_exports.py          Diff two such directories key by key - says WHAT changed
  measure_drawn.py         What was actually DRAWN, stationed against each leg's centreline
  corridor_report.py       The corridor questions, with the coverage of every answer beside them
  corridor_render.py       A straightened strip plan of one corridor, on stacked panels
  convert_road_network.py  Build + verify a spatially-indexed copy of a roadway network file
  check_prose_only.py      Prove a commit changed only comments and docstrings
  test.sh                  Run the suite under .venv/bin/python, activated or not
  blender/                 Runs INSIDE Blender's own Python (no network, no venv)
    blender_scene.py       Entry point + scene assembly; imports the four siblings below
    blender_materials.py   Flat-color and PBR-textured materials
    blender_geometry.py    Generic mesh helpers: extrude a ring, build from verts/faces, stripe rects
    blender_crosswalks.py  The painted crosswalk styles + centrelines and stop bars
    blender_props.py       Street furniture DRAWING, dispatched by add_prop()
```

## Adding a new site (intersection)

Everything specific to one intersection lives under `sites/<name>/`; `src/` has no hardcoded site data. To add one:

1. `python scripts/phase1_audit.py --street1 "Main St" --street2 "Oak Ave" --anchor "Main St, Sometown, NJ"` — resolves the intersection point via OSM and prints what the road network records there.
2. Create `sites/<name>/config.yaml` (copy `sites/broad_st_greenwood/config.yaml`) — `center_wgs84` from step 1, `data_sources`, and one `legs` entry per approach with a `bearing_deg` (compass, 0=N/90=E/clockwise, from the intersection outward). That bearing is the **only** thing that has to be geometrically accurate for `src/geometry/intersection/` to tell the legs apart; nothing assumes 4 legs or perpendicular roads, so 3-way/5-way/skewed junctions all work the same way. `sites/README.md` documents every key.
3. Create `sites/<name>/scenarios.py` exposing `build_demo_scenario(baseline) -> DesignState`.
4. Run the Quick start commands with `--site <name>`.

Editing a `config.yaml` means rerunning from Phase 2 onward; Phase 1 does not depend on it. Phase 4 shells out to Blender (its own bundled Python, no network, none of this project's packages) — needs Blender on `PATH`, or set `BLENDER_BIN`; defaults to `/Applications/Blender.app/Contents/MacOS/Blender` on Mac.

**A site is not a place to keep a standard.** `src/` contains no site data, and the converse holds too: two tests fail the build if a site re-declares something shared — `test_no_site_redeclares_what_src_already_defines` catches a shared **number**, `test_no_rule_is_written_out_in_more_than_one_site` compares normalised ASTs to catch a shared **rule**. Both exist because a consolidation found eight rules copied across three or four site files (a corridor-wide bike-lane side among them, whose own comment called it "a corridor decision, not a per-junction one"). `sites/` is linted along with `src`, `scripts` and `tests`.

## Adding something new to the street

Kerbs, driveways, bike lanes, green surfacing, flex posts and raised crossings all went in the same way, and each shipped the same handful of bugs. They are the seams of this pipeline — every one is a place where two pieces of code have to agree about the same shape.

### First decide which of three things it is

| It is… | so it lives… | seeded/applied by |
|---|---|---|
| **a fact about the street as it exists** (a kerb, a driveway, a parking prohibition, a crossing) | on `IntersectionModel`, resolved **once** at load | `load_intersection_model()`, then `DesignState.from_model()` |
| **a decision a proposal makes** (narrow this lane, protect this kerb) | as a `Treatment` subclass, with its derived geometry as a *method* | `state.apply(...)` |
| **a way of drawing something already decided** (green asphalt, a dashed kerb) | as a `PaintKind` in `src/geometry/markings.py`, or a prop | the treatment's `paint`, or `src/render/props.py` |

Getting this wrong is the most expensive mistake available here. A fact modelled as a treatment cannot be read by anything holding only a model; a decision materialised at apply time freezes a snapshot the rest of the design then moves out from under; and geometry built inside a renderer is geometry the *other* renderer will build differently.

### A real-world fact belongs on the model, fetched once

Driveways were once fetched and projected in **three** places, each with its own radius constant. If a new element comes from OSM:

1. Add the fetcher to `src/sources/osm_context.py` (it will be a view over the same cached borough snapshot — no new network call).
2. Project it to feet **in `src/geometry/intersection/`**, store it as a frozen dataclass on the model, and give that dataclass the derived geometry every consumer wants (`Driveway.surface`, not a width for each renderer to re-widen).
3. If `DesignState.from_model()` reads it, guard the attribute — the test doubles are deliberately partial models.
4. Refresh the test fixture separately from the cache (see Tests).

### A new marking touches six places, and five of them are checked

| Where | What | If you forget |
|---|---|---|
| `src/geometry/markings.py` | the declaration | nothing else works; this is the registry |
| `plan_view.PAINT_STYLE` | how the 2D view draws it | `require_every_kind` raises **at import** |
| `plan_view.legend_handles()` | its swatch | `test_every_marking_the_plan_view_draws_is_in_its_legend` fails |
| `src/render/export.py` | serialisation | automatic from `CHANNELS` — *unless* it needs a new `Role` |
| `scripts/blender/blender_scene.py` | how the 3D render draws that channel | **nothing catches this** |
| `src/render/props.py` | only for `Role.OBJECT` | `post_not_in_the_render` fails the build |

That fifth row is the one unguarded seam in the project — Blender runs under its own Python and cannot import `src`, so `markings.CHANNELS` is a data contract with no type behind it. It is where the bike lane's bollards shipped visible in 2D and absent in 3D. **After adding a channel, look at the render.** And after touching anything in `scripts/blender/`, put one scene through Blender before committing:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/blender/blender_scene.py -- output/broad_st_greenwood/geometry_proposed.json /tmp/probe.png
```

Five seconds, and the only check that exists over there. The failure mode is not a wrong picture, it is **no picture**.

### Four habits the seams demand

- **Build derived geometry once, from the thing that bounds it.** `model.offset_band_polygon(...)` builds a band from the two offsets that define it, on the same station grid and with the same kerb clamping the stripes use, so a band and the stripes at its edges cannot disagree. Differencing two strips instead overshot its own outer stripe by 6.6 ft and no invariant could have fired.
- **Anything drivable has to break where the kerb opens.** `PaintContext.add` clips against surfaces, keep-clear zones and kerb openings; `PaintContext.emit` deliberately does not, because a post is a point. Any new object placed along a kerb needs `stands_in_an_opening(state.kerb_openings, geometry)`. Seven of E Broad's 26 flex posts stood in driveways — worse than not breaking the paint, because it draws a protected lane you are expected to drive through.
- **Keep surveyed and assumed numbers separate, and name the assumption.** A driveway's mouth is surveyed (the `kerb=lowered` extent); its width is not — 10 ft vs 37 ft on E Broad. So the assumed one is named `DRIVEWAY_DRAWN_WIDTH_FT`, labelled in the legend, and **loses** to the surveyed extent wherever both exist.
- **A new element also has to be *distinguishable*.** Driveways were drawn as a thin dashed line on a drawing that already carries parcel lines, sidewalk centrelines and leg centrelines at similar weights. Indistinguishable is, for review purposes, the same as absent.

### The verification loop

In this order, because each step is cheaper than the next:

1. **Write the test first, and confirm it fails against the pre-change code** (`git stash`, or a worktree — if a worktree, symlink the gitignored `data/` in, or every run crashes in 0.6 s and you will read that as a result).
2. `scripts/export_all_scenarios.py /tmp/before` → change → `/tmp/after` → `scripts/diff_exports.py`. Thirteen scenes, ~2 s, key by key. This runs `export_scenario`, so it resolves the scene, builds the paint and props and asserts every invariant.
3. `scripts/test.sh` — includes the lint pass and the golden comparison, so there is no separate linting step. `.venv/bin/ruff check src scripts tests conftest.py` is the sub-second version while editing.
4. `scripts/build_all.py --render-3d`, then **look at the PNGs** — the only check on the Blender seam.

And the habit under all of it: **measure the geometry you just built.** Print its extent, its area, its distance to the thing it should touch. Every geometry bug in this repo's history was found by measuring and missed by looking.

### Small gotchas that cost real time

- **geopandas rejects dash-tuple linestyles** (`ValueError: inhomogeneous shape`). Use named linestyles (`"--"`, `":"`, `"-."`) in any style dict a GeoSeries plot sees.
- **Place an OSM way as a whole, not vertex by vertex.** Filtering a kerb way's vertices by offset collapsed 4 of 6 openings to 0.0 ft, because a dropped kerb across a driveway mouth runs *across* the leg.
- **Height ordering in 3D is manual.** Pavement extrudes 0.05 m and anything on top must be taller. Nothing checks this; sidewalks are 0.03 and therefore sit *below* the road.

## The core design principle

**Never trust a generic or geometric guess when real, sourced data exists.** This project repeatedly found that "obvious" defaults — a road network's own width attribute, a geocoder's address match, an assumed corner radius, a CC0 asset's fitness for a specific use — were wrong, missing, or the wrong tool, and the fix was always to go find the authoritative source.

| What | Source of truth | NOT this |
|---|---|---|
| Intersection location | OSM way-endpoint match, cross-checked against NJDOT SLD milepost | Nominatim address geocoding (off by ~900 ft) |
| Road widths | NJDOT SLD + field measurements (`sites/<site>/config.yaml`) | Any width field on the road network file (there isn't one) |
| Crosswalk position | Real OSM-surveyed crossing geometry, matched to legs | A geometric estimate from corner-fillet tangent points |
| Crosswalk style | Direct street-view confirmation, matching OSM's `crossing:markings` | Assumption ("probably ladder") |
| Roadway extent and width | The traced kerb, per side and per station | The highway class's nominal ribbon |
| Building footprints / heights | OSM outlines; the assessor's storey count (MOD-IV `BLDG_DESC`) via the parcel | Placeholder boxes; one default height per town |
| Parking lot extent | The `amenity=parking` area as mapped | A width assumed around an aisle centreline |
| Pavement material, streetlight | Real Poly Haven CC0 textures and model | Flat colors, or a guessed asset URL |
| Traffic signage, trees | **Procedural**, explicitly — no viable CC0 source found | Silently faking a "real asset" |
| Signal pole type, ped-head config, NTOR legs | Direct street-view photo confirmation (`signals:` block) | A generic signalized-intersection layout |

Where no real source exists (Greenwood Ave has no NJDOT SLD at all; no CC0 traffic-sign model exists), the code falls back to the best estimate and **flags it explicitly** — `confirmed: false`, `crosswalk_offset_source: "geometric_estimate"`, prop `"source"` strings, distinct dashed/red styling in 2D. Never silently guess; never silently substitute a worse asset without saying so.

The corollary that keeps recurring is a bug shape: a feature matched by *"anywhere along the leg"* rather than *"belongs to this junction"*. It has landed three times — the way-nearest-the-midpoint parking matcher, the near-set kerbs, and a crossing at the next junction adopted as ours because a drawing-extent setting was deciding junction membership. Each fix is a distance guard, documented where it lives (`src/geometry/context_roads.py`, `src/render/crosswalks.py`).

## Data (`data/`, gitignored — large binaries, not in git)

- `NJ_Roadway_Network.geojson` (170 MB) — NJDOT's **statewide** SRI/SLD linear-referencing roadway layer (despite the folder name, not pre-clipped). Has jurisdiction/route-ID fields but **no lane count, width, or surface type** — which is why Phase 2 needs the SLD PDF plus field measurements.

  **Convert this once, before anything else:** `python scripts/convert_road_network.py`. GeoJSON carries no spatial index, so a bbox-filtered read still parses the whole file (~2.2 s versus ~2.5 s for all 105,838 features). The script writes a FlatGeobuf sibling with a packed Hilbert R-tree, dropping that read to ~0.002 s and `load_intersection_model()` from ~2.5 s to ~0.10 s — and every phase script pays that cost at least once. It verifies the copy is WKB-identical before keeping it, the `.geojson` stays canonical, and `src/sources/data_loader.py` picks up the sibling automatically and ignores it if stale. The `.fgb` is gitignored: rebuild it, don't commit it.
- `00000518__8.000-11.000.pdf` — NJDOT Straight Line Diagram for Route 518 (Broad St), MP 8.000–11.000. Our intersection is **MP 10.30**. Read it by rendering locally at high DPI (`pdftoppm -r 400 file.pdf page`) and cropping; the pdf-viewer tool's screenshot is too low-res for the tick labels.
- `MercerCountyParcels.*` — Mercer County parcel polygons, for ROW/corner context and Greenwood Ave's width estimate. `MUN=1105` is Hopewell Borough.
- `MercerTaxList.dbf` — MOD-IV property tax records, joined by PIN to those parcels. `BLDG_DESC` is the assessor's shorthand for what stands on the lot and is where **building heights** come from.

A different site can point `data_sources:` at entirely different files (another county's parcels, another state's network).

## Facts about this street

These are not in any module, because they are about Hopewell rather than about geometry.

- **NJDOT's SLD naming is inconsistent with reality.** The Greenwood Ave cross-street is recorded as SRI `11051089__`, name **`COLUMBIA AVE`**. Confirmed via OSM: the two lines share the same physical location within 0.3 ft. Search SLD/GIS records under "Columbia Ave."
- **Geocoding this intersection by address fails.** Nominatim single-string geocoding lands ~900 ft off. `data_loader.geocode_intersection()` instead finds the two named OSM ways and locates their shared endpoint node.
- **NJDOT's West/East Broad St split is NOT at Greenwood Ave** — it is further east at the Route 569/Hamilton Ave signal. OSM splits it right at Greenwood. Both of our "west"/"east" legs are technically "West Broad Street" per NJDOT.
- **SLD segment-average widths aren't corner-specific.** The SLD says 48 ft nominal for the whole corridor; the field-measured width at this corner is 55.5 ft (west) and 68 ft (east) — real local widening a corridor-level entry cannot capture.
- **Real widths per `sites/broad_st_greenwood/config.yaml`:** West Broad 55.5 ft (confirmed), East Broad 68 ft (confirmed), Greenwood N/S 34 ft each (**estimate** — a measured ~50 ft parcel-to-parcel ROW gap minus an assumed 8 ft/side sidewalk allowance; no NJDOT SLD exists for this local road). Existing corner radius 20 ft (**estimate** — parcel lot lines are straight, with no chamfer to read a radius from).
- **Nominal width is not crossing length.** Crossings are painted where the traced kerbs have already flared through the corner returns, so a person crossing Broad St today walks **65.0 ft** of asphalt, not the 52 ft cross-section. An 8 ft extension per side reads as "52 → 36" on paper and is really 65.0 → 35.5 ft on the ground.
- **All four real crossings here are tagged `crossing:markings=lines`** — confirmed directly, not inferred.
- **West Broad, East Broad and North Greenwood all have a solid double yellow; South Greenwood has no centreline paint at all.** OSM has no tag for this, so it is a per-leg `centerline_style` in `config.yaml`, street-view confirmed — the same sourcing category as the `signals:` block.
- **Where the borough ordinance is not tagged in OSM, nothing currently says so — an open gap.** Schedule I bans parking 100 ft each way on both Broad St legs, and OSM carries no `parking:*:restriction` for either, so scenarios paint stalls from station 79.5 ft and ~20 ft of each run sits inside the prohibition. Either tag it in OSM (which helps every consumer) or carry the borough schedules as a data source. Inferring parking from an absent tag is the one place this project still guesses where it could source.

## Proposals

| Site | Scenario | What it does |
|---|---|---|
| broad_st_greenwood | `build_proposal_bike_lanes` | The standard section — 5 ft lane, 2 ft buffer — flex posts down each buffer, both sides of both Broad legs, asphalt green. All four kerbs have 21.3–26.6 ft against the 18.8 the section needs; the surplus is hatched rather than spent on a wider lane. **Not** parking-protected. Greenwood Ave gets none (2.3 / 4.6 ft spare, under AASHTO's 5 ft). |
| ebroad_princeton | `build_proposal_bike_lanes` | The same section, both sides of both E Broad legs, **protected on three of four kerbs**. The fourth is the finding: `e_broad_st_east` right carries a **4.49 ft protected lane**; its left comes to 3.80 ft, under the 4 ft floor, so it falls back to a conventional 5 ft lane and says so. Widening that kerb by **0.20 ft** would buy the fourth. No parking displaced. |

**The buffer is kept and the lane gives.** This project had it the other way first — hold a nominal 5 ft lane and drop the 2 ft buffer whenever the section did not quite fit — so a kerb 0.51 ft short lost every flex post to hold six inches of paint. A rider is better served by a 4.49 ft lane with a post beside it than a 5 ft lane with a truck beside it. Hence two constants: `AASHTO_MIN_BIKE_LANE_FT` (5 ft, the width to design to) and `MIN_BIKE_LANE_FT` (4 ft, the floor below which it is not a bike lane).

**Parking-protected does not fit Broad St.** The 48 ft section fits inside 52.0 and 55.5 ft of roadway, but the total is not the constraint: every offset here is measured from the leg centreline, and the *parking side alone* needs 28.0 ft against 26.0 / 27.8 nominal. Fitting it would mean shifting the travel lanes off the NJDOT alignment — a real design, but not one this pipeline can draw, since that alignment is the datum every offset, stop bar and crossing frame is measured from.

The treatment catalogue itself — every `Treatment` subclass, its parameters, what it refuses and why — lives in `src/geometry/treatments/`, one module per family (see the tree above). `state.apply(...)` is the only way one enters a design:

```python
state = baseline.apply(
    AddCurbExtension(LegSide("broad_st_east", Side.LEFT), extension_ft=8.0, crossing_ft=41.5),
    ProtectDaylightZone(LegSide("broad_st_east", Side.LEFT), kind="curb_extension"),
)
```

`DesignState` is immutable-by-clone, so `apply` returns a new design and chains. Constructing a treatment validates it; `apply` checks its target exists at this junction, refuses one that needs the model without one, records it, and writes the provenance note. Four policies stay plain functions because they emit *several* treatments: `apply_osm_parking`, `complete_centerlines`, `all_crosswalks_continental`, `bulb_out_corner_pair`.

## Phase 4 (3D) notes

The Blender side cannot import `src`, so its own reasoning cannot live in a module docstring this project's tests can reach. What follows is therefore kept here.

**Textures.** `src/render/assets.py` fetches real CC0 PBR textures from Poly Haven (`asphalt_01` for pavement, `pavement_02` for sidewalks), caching to `output/.textures/`. Anything within the "near zone" (past the farthest crosswalk plus a buffer, `export.py:_split_near_far`) gets 4k, everything else 2k, split by intersecting with a circle so a piece can straddle the boundary. `blender_materials.py:make_textured_material()` falls back to a flat color if a texture is missing — Phase 4 must never hard-fail without network access. Each piece gets a real-world-scaled planar UV projection so tiling reads consistently across differently-sized pieces.

**Streetlights.** A real Poly Haven model (`street_lamp_01`, glTF at 1k — the 8k default would be enormous for a background prop) is fetched once as a hidden template; each corner gets a linked duplicate at that corner's fillet-arc midpoint (real geometry) pushed a few feet onto the sidewalk (an approximation, flagged in the prop's `"source"`). Falls back to a procedural pole+box.

**Signage.** No CC0 stop-sign or school-zone model exists on Poly Haven, and nothing on Kenney.nl is fetchable without guessing a URL — which this project's own principle rules out. Built procedurally with correct MUTCD shape and color instead. One stop sign per approach, plus whatever a site's `config.yaml` lists under `props.extra`.

**Traffic signals.** Driven by a site's `signals:` block (see `sites/README.md`) — pole type, per-corner pedestrian-head configuration, and no-turn-on-red legs, all street-view confirmed. `props.py:_traffic_signal_props` places a pole at each configured corner's fillet-arc midpoint plus a pedestrian head, co-located or on its own post. The mast arm's reach is derived per-corner from real adjacent leg widths, not a fixed constant. Each signalized approach also gets a stop bar just behind its crosswalk, only when the config has a `signals` block. **Which approach each head visually aims at is a render-fidelity simplification** — real per-arm aiming and phasing are not modelled.

**Trees.** One low-poly procedural tree, instanced along each sidewalk piece via Blender geometry nodes at 25 ft spacing (standard municipal street-tree spacing). Geometry-node instancing shares one mesh across every instance — the actual performance property wanted, not a style choice. Poly Haven's trees are realistic photoscans, disproportionately heavy for background dressing.

**Building meshes.** `src/render/mesh_utils.py` extrudes each footprint with `trimesh` and decimates anything genuinely heavy. The threshold used to be 40 faces, which is an 11-sided building — an ordinary house with a porch — and nine of Broad & Greenwood's 80 crossed it and were crushed to 24 faces, leaving four rendered as crumpled tents (quadric decimation collapses the cheapest edges, which on a short extrusion are the vertical ones). All 80 undecimated come to **1,692 triangles**: there was nothing to save. The threshold is now 400 faces and `test_a_building_keeps_its_flat_roof` pins the shape rather than the number. Roofs stay flat, because nothing in OSM or MOD-IV says what shape they are and the ridge would be the tallest thing in the render.

**General:**

- Render engine `BLENDER_EEVEE_NEXT` (the only one in Blender 4.3). Samples 64, dropped from 128 — visually indistinguishable here, ~30% faster. Full render, both scenarios, warm caches: ~13 s.
- Blender's Python has no network access and no access to this venv. All fetching happens beforehand; only local file paths reach the exported JSON.
- **Blender does NOT put a `--python` script's own directory on `sys.path`** (unlike plain `python script.py`) — confirmed empirically. `blender_scene.py` inserts it manually before importing its siblings.
- **Marking height must exceed pavement height** — pavement extrudes 0.05 m, markings 0.06, or they render buried inside it.
- **Blender's multi-object edit mode re-extrudes every *selected* mesh**, not just the active one. Always `select_all(action='DESELECT')` first, or the ground plane silently gains height every time something else is extruded.
- OSM footprints don't reconcile with our precise curb geometry, so `export.py` filters any building intersecting the pavement polygon. Buildings that merely look close are legitimate — verify with a numeric intersects check before assuming a bug.
- `blender_scene.py` accepts any number of `<geometry.json> <output.png>` pairs and renders them in **one** process; each launch costs ~1–1.5 s of fixed startup.
- Overpass's public instances are flaky (504s common) — `data_loader.query_overpass()` retries across three mirrors.

## Known gaps / next steps

- Greenwood Ave (N & S) widths and the existing corner radius are still estimates — need field measurement or aerial confirmation.
- East Broad St's "54 ft active roadway" vs "68 ft total" distinction is not used anywhere yet.
- Only one demo scenario exists per site beyond the proposals above; more would be new functions in that site's `scenarios.py`.
- Prop placement setbacks are approximations, not a surveyed signage inventory — flagged per prop.
- **A driveway strip is not clipped at the kerb.** An OSM driveway runs to the road's centreline, so widening it paints over the carriageway. Measured across all four sites, exactly one does: way `772378207` at E Broad, 187 of its 1,577 sq ft (12%) inside the modelled roadway.
- **Only one mapped driveway reaches a modelled kerb** (`772378207`); the rest are 21.7–352 ft away, so they render as strips ending in grass. Conversely most dropped kerbs near these junctions have no driveway way mapped at all, so most openings are surveyed kerb with nothing visible behind them.
- Sidewalks extrude 0.03 m against pavement's 0.05, so a footway sits *below* the road surface in 3D.
- The bike lane's 2 ft buffer, with 0.82 ft (10 in) edge stripes, leaves 0.36 ft of visible asphalt between them, so the buffer's hatching no longer reads in 3D. Widen the buffer, narrow `LANE_EDGE_LINE_WIDTH_FT` toward a real 6 in, or stop hatching a buffer that narrow. Not chosen.
- Two kerb openings overlap their leg's crossing band and are **reported rather than filtered** (`describe_kerb_openings`). Overriding a surveyed tag with a geometric guess is what the core principle rules out.
- The network/corridor work is in progress — see `docs/network-model.md` and `docs/network-renderer-plan.md` for the endpoint, the checkpoints already measured, and what remains.
