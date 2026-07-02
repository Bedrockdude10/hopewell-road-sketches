# hopewell-road-sketches

Parametric pedestrian-safety visualization for **Broad St (CR 518) & Greenwood Ave, Hopewell Borough, NJ 08525** — real-world geometry (NJDOT + field measurements + OSM), not hand-drawn sketches. Produces to-scale 2D plan-view before/after comparisons and presentation-quality 3D renders.

## Quick start

```bash
source .venv/bin/activate   # venv already has geopandas/shapely/etc. - see requirements.txt
python scripts/phase1_audit.py       # load/clip/audit the NJDOT road network
python scripts/phase2_geometry.py    # curb-line + corner-fillet geometry from config/intersection_config.yaml
python scripts/phase3_treatments.py  # apply treatments, render before/after plan view
python scripts/phase4_render_3d.py   # export geometry + render both scenarios in Blender
```

Outputs land in `output/`: `phase1_network_plot.png`, `phase2_geometry_plot.png`, `phase3_before_after.png`, `phase4_render_existing.png`, `phase4_render_proposed.png`.

If you edit `config/intersection_config.yaml` (widths, corner radius, crosswalks, treatments), rerun from Phase 2 onward — Phase 1 doesn't depend on it.

Phase 4 shells out to Blender (not the project venv — `blender_scene.py` runs under Blender's own bundled Python). Needs Blender on `PATH`, or set `BLENDER_BIN` — defaults to `/Applications/Blender.app/Contents/MacOS/Blender` on Mac if nothing else is found.

## The core design principle

**Never trust a generic/geometric guess when real, sourced data exists.** This project repeatedly found that "obvious" defaults (a road network's own width attribute, a geocoder's address match, an assumed corner radius, OSM's map style tag) were wrong or missing, and the fix was always to go find the authoritative source instead. Concretely:

| What | Source of truth | NOT this |
|---|---|---|
| Intersection location | OSM way-endpoint match (§ below), cross-checked against NJDOT SLD milepost | Nominatim address geocoding (off by ~900 ft) |
| Road widths | NJDOT SLD + Danny's field measurements (`config/intersection_config.yaml`) | Any width field on the road network file (there isn't one) |
| Crosswalk position | Real OSM-surveyed `highway=footway`+`footway=crossing` geometry, matched to legs | A geometric estimate from corner-fillet tangent points |
| Crosswalk style | Danny's direct confirmation (existing = "lines", matches OSM's `crossing:markings` tag) | Assumption ("probably ladder") |
| Building massing | Real OSM building footprints | Placeholder boxes |

When no real source exists (e.g. Greenwood Ave has no NJDOT SLD at all — it's a local road), the code falls back to the best available geometric estimate and **flags it explicitly** (`confirmed: false` in config, `crosswalk_offset_source: "geometric_estimate"` in exported JSON, distinct dashed/red styling in renders). Never silently guess.

## Data (`data/`, gitignored — large binaries, not in git)

- `NJ_Roadway_Network.geojson` (170MB) — NJDOT's **statewide** SRI/SLD linear-referencing roadway layer (despite living in a "Hopewell" folder, it's not pre-clipped). Has jurisdiction/route-ID fields (SRI, ROUTE_SUBTYPE, ROAD_NUM) but **no lane count, width, or surface type** — that's why Phase 2 needs the SLD PDF + field measurements.
- `00000518__8.000-11.000.pdf` — NJDOT Straight Line Diagram for Route 518 (Broad St), milepost 8.000–11.000. Our intersection is **MP 10.30**, a signalized crossing inside NJDOT's "West Broad Street" segment. Read this by rendering locally at high DPI (`pdftoppm -r 400 file.pdf page`) and cropping — the pdf-viewer MCP tool's own screenshot is too low-res to read the tick labels.
- `MercerCountyParcels.*` (shapefile) — Mercer County parcel polygons, used for ROW/corner context and to estimate Greenwood Ave's width (see below). `MUN=1105` is Hopewell Borough.
- `MercerTaxList.dbf` — MOD-IV tax attributes, joinable by PIN, not currently used.

## Key findings worth knowing before you touch anything

- **NJDOT's SLD naming is inconsistent with reality.** The Greenwood Ave cross-street is recorded in NJDOT's system as **SRI `11051089__`, name `COLUMBIA AVE`** — not Greenwood. Confirmed via OSM: the OSM ways for N/S Greenwood Ave and the NJDOT "Columbia Ave" line share the same physical location (within 0.3 ft). If you're looking up SLD/GIS records for this street, search under "Columbia Ave."
- **Geocoding this intersection by address fails.** Nominatim single-string geocoding lands ~900 ft off (returns an arbitrary point along a street, not the corner). `src/data_loader.py:geocode_intersection()` instead finds the two named OSM ways and locates their shared endpoint node — verified against the NJDOT SLD milepost.
- **NJDOT's own West/East Broad St naming split is NOT at Greenwood Ave** — it's further east at the Route 569/Hamilton Ave signal. OSM's naming splits West/East right at Greenwood. Both of our "west"/"east" legs are technically "West Broad Street" per NJDOT.
- **SLD segment-average widths aren't corner-specific.** The SLD says 48 ft nominal pavement for the whole corridor segment; the actual field-measured width at this specific corner is 55.5 ft (west) and 68 ft (east) — real local widening (turn lanes / parking) that a corridor-level SLD entry can't capture. Always prefer a corner-specific field measurement over a segment average when you have one.
- **Real intersection widths, per `config/intersection_config.yaml`** (as of this writing): West Broad 55.5 ft (confirmed), East Broad 68 ft (confirmed), Greenwood N/S 34 ft each (**estimate** — derived from a measured ~50 ft parcel-to-parcel ROW gap minus an assumed 8 ft/side sidewalk allowance; no NJDOT SLD exists for this local road at all). Existing corner radius: 20 ft (**estimate** — no survey; parcel lot lines are plain straight lines with no chamfer to read a real radius from).

## Repo structure

```
config/intersection_config.yaml   Authoritative per-leg widths, corner radius, existing crosswalks - see above
src/
  data_loader.py     NJDOT/parcel loading, Overpass query helper w/ mirror retry, intersection geocoding
  geometry_model.py  CRS/clipping utilities, Leg dataclass, corner fillets, pavement polygon, leg_clearance_ft
  intersection.py    load_intersection_model() - THE entry point every phase script uses
  config.py          YAML loader
  treatments.py      DesignState + composable treatment functions (see below)
  scenarios.py       build_demo_scenario() - the example treatment package used by Phase 3 & 4
  render.py          matplotlib plan-view rendering (Phase 2/3)
  osm_context.py     OSM building + real crosswalk fetching (Phase 4 fidelity), disk-cached
  export.py          Serializes a DesignState to local-meters JSON for Blender
scripts/
  phase1_audit.py         Load/clip/audit the road network
  phase2_geometry.py      Build + plot curb-line/corner geometry
  phase3_treatments.py    Apply demo scenario, plot before/after
  phase4_export_geometry.py  Export-only (no Blender) - useful for debugging the JSON
  phase4_render_3d.py    Full Phase 4 pipeline: export + shell out to Blender
  blender_scene.py       Runs INSIDE Blender's own Python - builds + renders the 3D scene
```

## The geometry model

`IntersectionModel` (`src/intersection.py`) has 4 `Leg`s (one per approach), each with a `centerline` (from the NJDOT network, clipped and simplified) and a `curb_to_curb_ft` from config, from which `left_curb`/`right_curb` offset lines are derived automatically. `build_corner_fillets()` rounds each of the 4 corners with an analytic tangent-arc fillet (`fillet_curb_corner()`), and `build_pavement_polygon()` stitches all 4 corners' trimmed curbs + arcs into one continuous filled "plus" shape.

`leg_clearance_ft()` computes how far along a leg's centerline you have to go before the roadway is straight (i.e. past the corner curve) — **always project onto the centerline, never use raw Euclidean distance** from a point on the (laterally-offset) curb line, or wide legs will wildly overshoot (a 68 ft leg has a 34 ft half-width, which alone dominates a naive distance calc). This function is used to place crosswalks and raised-crossing treatments outside the curve, not inside it, whenever real survey data isn't available.

## Treatments (`src/treatments.py`)

`DesignState` is immutable-by-clone — every treatment function takes a state and returns a *new* one, so scenarios compose by chaining: `state = bump_out(state, ...)`.

- `bump_out(state, corner, radius_ft)` — rebuilds one corner's fillet at a new radius. A curb extension and a tightened turn radius are **the same geometric operation** (shrinking the corner radius does both: shortens the crossing and slows the vehicle turn).
- `refuge_island(state, leg_name, offset_ft, width_ft, along_road_ft)` — NACTO 6 ft minimum width enforced.
- `raise_crossing(state, leg_name, crossing_width_ft)` — marks a crossing as a raised speed table; placed via `leg_clearance_ft()`.
- `upgrade_crosswalk_markings(state, leg_name, style)` — repaints a crosswalk to a more visible style (`"lines"` → `"continental"` → `"ladder"`, FHWA/NACTO visibility ranking). A real, standalone low-cost treatment, not just cosmetic.
- `build_sidewalk_pieces(state, sidewalk_width_ft)` — reuses the *same* fillet pipeline at a wider offset to get a sidewalk band that hugs the pavement exactly (12 pieces: 4 leg strips × 2 sides + 4 corner wedges).

The demo scenario (`src/scenarios.py:build_demo_scenario`) tightens the two corners on the confirmed West Broad St leg (20→10 ft), adds a refuge island, raises the Greenwood-south crossing, and upgrades the other 3 crosswalks to continental.

## Crosswalk styles: real data over guessing

OSM actually has surveyed crosswalk geometry at this intersection (`highway=footway` + `footway=crossing` ways). `src/osm_context.py:fetch_crossings()` pulls it; `src/export.py:_match_crossings_to_legs()` matches each real crossing to a leg by projecting its midpoint onto every leg's centerline and picking the closest plausible match. When a match exists, its real position is used (`crosswalk_offset_source: "osm_survey"` in the exported JSON) and its `crossing:markings` OSM tag maps to one of our 3 render styles (`lines`/`zebra`→`continental`/`ladder`). No match → fall back to `leg_clearance_ft()` geometric estimate (needed for hypothetical/proposed crossings that don't exist yet).

All 4 real crossings here are tagged `crossing:markings=lines` — confirmed correct by Danny (simple 2-line marking, not ladder or continental). The proposed scenario upgrades 3 of them to continental via `upgrade_crosswalk_markings`; the 4th becomes a raised crossing instead.

`blender_scene.py` implements all 3 styles: `add_crosswalk_lines` (2 transverse boundary lines only), `add_crosswalk_continental` (parallel bars, no rails), `add_crosswalk_ladder` (bars + 2 framing rails) — dispatched via `CROSSWALK_STYLES` by the `crosswalk_style` field per leg in the exported JSON.

## Phase 4 (3D) notes

- Render engine: `BLENDER_EEVEE_NEXT` (the only one in Blender 4.3 — old EEVEE identifier is gone).
- **Marking height must exceed pavement height** — pavement is extruded 0.05 m; anything meant to be visible on top of it (crosswalks, centerlines) must be taller (0.06 m used) or it renders buried inside the solid pavement block with zero visible effect.
- **Blender's multi-object edit mode re-extrudes every *selected* mesh, not just the active one.** Always `bpy.ops.object.select_all(action='DESELECT')` before entering edit mode on a single object, or previously-created objects (e.g. the ground plane) silently accumulate extra height every time something else gets extruded.
- OSM building footprints don't reconcile with our precise curb geometry — a few end up overlapping the pavement. `export.py` filters any building whose footprint intersects the pavement polygon. (Buildings that just look close to the road in the render are legitimate — small-town buildings really do sit near the curb; verify with a numeric intersects check before assuming a rendering bug.)
- `scripts/blender_scene.py` accepts any number of `<geometry.json> <output.png>` pairs and renders them all in **one Blender process** — each launch has ~1–1.5s fixed startup overhead, not worth paying per-render. `phase4_render_3d.py` uses this to do both scenarios in one shot.
- `fetch_buildings()` and `fetch_crossings()` cache their raw Overpass response to `output/.cache/`, keyed by (center, radius). Delete that directory to force a refetch (e.g. after changing `BUILDING_CONTEXT_RADIUS_M` in `src/export.py`).
- Overpass's public instances are flaky (504s are common) — `src/data_loader.py:query_overpass()` retries across 3 mirrors (`overpass-api.de`, `kumi.systems`, `openstreetmap.ru`) before giving up.
- EEVEE samples: 64 (dropped from 128 - visually indistinguishable for this flat-shaded scene, ~30% faster).

## Known gaps / next steps

- Greenwood Ave (N & S) widths and the existing corner radius are still estimates — need field measurement or survey/aerial confirmation. Once available, update `config/intersection_config.yaml` and rerun from Phase 2.
- East Broad St's "54 ft active roadway" vs "68 ft total" distinction isn't used anywhere yet (only the 68 ft total is) — could matter if a future treatment needs lane-level detail.
- Asset-library question (Poly Haven PBR textures for asphalt/grass/concrete, free low-poly prop packs for trees/streetlights) was raised but not yet implemented - flat colors work but real textures would likely be the next highest-leverage fidelity improvement.
- Only one demo treatment scenario exists (`build_demo_scenario`). Additional scenarios would just be new functions in `src/scenarios.py` composing the same treatment primitives.
