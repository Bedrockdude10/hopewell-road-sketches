# hopewell-road-sketches

Parametric pedestrian-safety visualization for real intersections — real-world geometry (NJDOT + field measurements + OSM), not hand-drawn sketches. Produces to-scale 2D plan-view before/after comparisons and presentation-quality 3D renders. Currently has one site configured: **Broad St (CR 518) & Greenwood Ave, Hopewell Borough, NJ 08525**.

## Quick start

```bash
source .venv/bin/activate   # venv already has geopandas/shapely/trimesh/etc. - see requirements.txt
python scripts/phase1_audit.py --site broad_st_greenwood       # load/clip/audit the road network (one-time, per new site)
python scripts/phase2_geometry.py --site broad_st_greenwood    # curb-line + corner-fillet geometry from the site's config.yaml
python scripts/phase3_treatments.py --site broad_st_greenwood  # apply treatments, render before/after plan view
python scripts/phase4_render_3d.py --site broad_st_greenwood   # export geometry + render both scenarios in Blender
```

`--site` defaults to `broad_st_greenwood` if omitted. Outputs land in `output/<site>/`: `phase1_network_plot.png`, `phase2_geometry_plot.png`, `phase3_before_after.png`, `phase4_render_existing.png`, `phase4_render_proposed.png`.

If you edit `sites/<site>/config.yaml` (widths, corner radius, crosswalks, treatments, props), rerun from Phase 2 onward — Phase 1 doesn't depend on it.

Phase 4 shells out to Blender (not the project venv — `blender_scene.py` runs under Blender's own bundled Python, with no network access and none of this project's packages). Needs Blender on `PATH`, or set `BLENDER_BIN` — defaults to `/Applications/Blender.app/Contents/MacOS/Blender` on Mac if nothing else is found.

## Adding a new site (intersection)

Everything specific to one intersection lives under `sites/<name>/` - `src/` is a general-purpose library with no hardcoded site data. To add one:

1. `python scripts/phase1_audit.py --street1 "Main St" --street2 "Oak Ave" --anchor "Main St, Sometown, NJ"` - resolves the intersection point via OSM and prints what the road network actually records there.
2. Create `sites/<name>/config.yaml` (copy `sites/broad_st_greenwood/config.yaml` as a template) - fill in `center_wgs84` from step 1, `data_sources` (which road network/parcels files to use - they don't have to be the ones already in `data/`), and one `legs` entry per approach with a `bearing_deg` (compass bearing, 0=N/90=E/clockwise, from the intersection outward along that leg - this is the ONLY thing that has to be geometrically accurate for `src/intersection.py` to tell the legs apart; nothing else assumes 4 legs or perpendicular roads, so 3-way/5-way/non-perpendicular intersections all work the same way).
3. Create `sites/<name>/scenarios.py` exposing `build_demo_scenario(baseline) -> DesignState` (copy `sites/broad_st_greenwood/scenarios.py` as a template).
4. Run the Quick start commands with `--site <name>`.

## The core design principle

**Never trust a generic/geometric guess when real, sourced data exists.** This project repeatedly found that "obvious" defaults (a road network's own width attribute, a geocoder's address match, an assumed corner radius, OSM's map style tag, a CC0 asset's fitness for a specific use) were wrong, missing, or the wrong tool for the job, and the fix was always to go find the authoritative source instead. Concretely:

| What | Source of truth | NOT this |
|---|---|---|
| Intersection location | OSM way-endpoint match (§ below), cross-checked against NJDOT SLD milepost | Nominatim address geocoding (off by ~900 ft) |
| Road widths | NJDOT SLD + Danny's field measurements (`sites/<site>/config.yaml`) | Any width field on the road network file (there isn't one) |
| Crosswalk position | Real OSM-surveyed `highway=footway`+`footway=crossing` geometry, matched to legs | A geometric estimate from corner-fillet tangent points |
| Crosswalk style | Danny's direct confirmation (existing = "lines", matches OSM's `crossing:markings` tag) | Assumption ("probably ladder") |
| Building massing | Real OSM building footprints | Placeholder boxes |
| Pavement/sidewalk material | Real Poly Haven CC0 PBR textures (asphalt_01, pavement_02) | Flat colors |
| Streetlight model | Real Poly Haven CC0 model (street_lamp_01) | Flat colors / a guessed asset URL |
| Traffic signage, trees | **Procedural** geometry, explicitly - no viable CC0 source found (see "Phase 4 fidelity" below) | Silently faking a "real asset" that isn't one |

When no real source exists (e.g. Greenwood Ave has no NJDOT SLD at all — it's a local road; no CC0 traffic-sign model exists on Poly Haven), the code falls back to the best available estimate/procedural geometry and **flags it explicitly** (`confirmed: false` in config, `crosswalk_offset_source: "geometric_estimate"` / prop `"source"` strings in exported JSON, distinct dashed/red styling in 2D renders). Never silently guess, and never silently substitute a worse asset for a real one without saying so.

## Data (`data/`, gitignored — large binaries, not in git)

- `NJ_Roadway_Network.geojson` (170MB) — NJDOT's **statewide** SRI/SLD linear-referencing roadway layer (despite living in a "Hopewell" folder, it's not pre-clipped). Has jurisdiction/route-ID fields (SRI, ROUTE_SUBTYPE, ROAD_NUM) but **no lane count, width, or surface type** — that's why Phase 2 needs the SLD PDF + field measurements.
- `00000518__8.000-11.000.pdf` — NJDOT Straight Line Diagram for Route 518 (Broad St), milepost 8.000–11.000. Our intersection is **MP 10.30**, a signalized crossing inside NJDOT's "West Broad Street" segment. Read this by rendering locally at high DPI (`pdftoppm -r 400 file.pdf page`) and cropping — the pdf-viewer MCP tool's own screenshot is too low-res to read the tick labels.
- `MercerCountyParcels.*` (shapefile) — Mercer County parcel polygons, used for ROW/corner context and to estimate Greenwood Ave's width (see below). `MUN=1105` is Hopewell Borough.
- `MercerTaxList.dbf` — MOD-IV tax attributes, joinable by PIN, not currently used.

A different site can point `data_sources:` at entirely different files (different county's parcels, a different state's road network) - see "Adding a new site" above.

## Key findings worth knowing before you touch anything

- **NJDOT's SLD naming is inconsistent with reality.** The Greenwood Ave cross-street is recorded in NJDOT's system as **SRI `11051089__`, name `COLUMBIA AVE`** — not Greenwood. Confirmed via OSM: the OSM ways for N/S Greenwood Ave and the NJDOT "Columbia Ave" line share the same physical location (within 0.3 ft). If you're looking up SLD/GIS records for this street, search under "Columbia Ave."
- **Geocoding this intersection by address fails.** Nominatim single-string geocoding lands ~900 ft off (returns an arbitrary point along a street, not the corner). `src/data_loader.py:geocode_intersection()` instead finds the two named OSM ways and locates their shared endpoint node — verified against the NJDOT SLD milepost.
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
  data_loader.py     Road network/parcel loading (paths passed in, not hardcoded), Overpass retry, geocoding
  geometry_model.py  CRS/clipping utilities, Leg dataclass, corner fillets, pavement polygon, leg_clearance_ft
  intersection.py    load_intersection_model() - THE entry point every phase script uses
  config.py          Generic YAML loader (no knowledge of sites)
  treatments.py      DesignState + composable treatment functions (see below)
  render.py          matplotlib plan-view rendering (Phase 2/3)
  osm_context.py     OSM building + real crosswalk fetching, disk-cached to output/.cache/
  assets.py          Poly Haven texture/model fetch + disk-cache to output/.textures/ (Phase 4 fidelity)
  theme.py           Resolves the specific texture/model slugs this project uses into local file paths
  mesh_utils.py      trimesh-based building mesh decimation (Phase 4 fidelity)
  export.py          Serializes a DesignState + theme to local-meters JSON for Blender
scripts/
  phase1_audit.py         Load/clip/audit the road network for a site (or a brand-new one via --street1/2/--anchor)
  phase2_geometry.py      Build + plot curb-line/corner geometry
  phase3_treatments.py    Apply demo scenario, plot before/after
  phase4_export_geometry.py  Export-only (no Blender) - useful for debugging the JSON
  phase4_render_3d.py    Full Phase 4 pipeline: fetch theme + export + shell out to Blender
  blender_scene.py       Runs INSIDE Blender's own Python (no network, no venv) - builds + renders the 3D scene
```

## The geometry model

`IntersectionModel` (`src/intersection.py`) has one `Leg` per approach, each with a `centerline` (from the road network, clipped and simplified) and a `curb_to_curb_ft` from config, from which `left_curb`/`right_curb` offset lines are derived automatically. Legs sharing a road (SRI) are told apart by matching each split centerline piece's compass bearing to the closest `bearing_deg` among that SRI's configured leg entries (`src/intersection.py:_assign_leg_pieces`) - this is what lets the same code handle any number of legs at any angles, not just a neat 4-way. `build_corner_fillets()` rounds each corner with an analytic tangent-arc fillet (`fillet_curb_corner()`), and `build_pavement_polygon()` stitches every corner's trimmed curbs + arcs into one continuous filled "plus" shape.

`leg_clearance_ft()` computes how far along a leg's centerline you have to go before the roadway is straight (i.e. past the corner curve) — **always project onto the centerline, never use raw Euclidean distance** from a point on the (laterally-offset) curb line, or wide legs will wildly overshoot (a 68 ft leg has a 34 ft half-width, which alone dominates a naive distance calc). Used to place crosswalks, raised-crossing treatments, and props outside the curve, not inside it, whenever real survey/geometric-basis data isn't available.

## Treatments (`src/treatments.py`)

`DesignState` is immutable-by-clone — every treatment function takes a state and returns a *new* one, so scenarios compose by chaining: `state = bump_out(state, ...)`.

- `bump_out(state, corner, radius_ft)` — rebuilds one corner's fillet at a new radius. A curb extension and a tightened turn radius are **the same geometric operation** (shrinking the corner radius does both: shortens the crossing and slows the vehicle turn).
- `refuge_island(state, leg_name, offset_ft, width_ft, along_road_ft)` — NACTO 6 ft minimum width enforced.
- `raise_crossing(state, leg_name, crossing_width_ft)` — marks a crossing as a raised speed table; placed via `leg_clearance_ft()`.
- `upgrade_crosswalk_markings(state, leg_name, style)` — repaints a crosswalk to a more visible style (`"lines"` → `"continental"` → `"ladder"`, FHWA/NACTO visibility ranking). A real, standalone low-cost treatment, not just cosmetic.
- `build_sidewalk_pieces(state, sidewalk_width_ft)` — reuses the *same* fillet pipeline at a wider offset to get a sidewalk band that hugs the pavement exactly (12 pieces: 4 leg strips × 2 sides + 4 corner wedges).

The demo scenario (`sites/broad_st_greenwood/scenarios.py:build_demo_scenario`) tightens the two corners on the confirmed West Broad St leg (20→10 ft), adds a refuge island, raises the Greenwood-south crossing, and upgrades the other 3 crosswalks to continental.

## Crosswalk styles: real data over guessing

OSM actually has surveyed crosswalk geometry at this intersection (`highway=footway` + `footway=crossing` ways). `src/osm_context.py:fetch_crossings()` pulls it; `src/export.py:_match_crossings_to_legs()` matches each real crossing to a leg by projecting its midpoint onto every leg's centerline and picking the closest plausible match. When a match exists, its real position is used (`crosswalk_offset_source: "osm_survey"` in the exported JSON) and its `crossing:markings` OSM tag maps to one of our 3 render styles (`lines`/`zebra`→`continental`/`ladder`). No match → fall back to `leg_clearance_ft()` geometric estimate (needed for hypothetical/proposed crossings that don't exist yet).

All 4 real crossings here are tagged `crossing:markings=lines` — confirmed correct by Danny (simple 2-line marking, not ladder or continental). The proposed scenario upgrades 3 of them to continental via `upgrade_crosswalk_markings`; the 4th becomes a raised crossing instead.

`blender_scene.py` implements all 3 styles: `add_crosswalk_lines` (2 transverse boundary lines only), `add_crosswalk_continental` (parallel bars, no rails), `add_crosswalk_ladder` (bars + 2 framing rails) — dispatched via `CROSSWALK_STYLES` by the `crosswalk_style` field per leg in the exported JSON.

## Phase 4 fidelity (textures, props, trees, mesh optimization)

**Textures.** `src/assets.py` fetches real CC0 PBR textures from Poly Haven's public API (`asphalt_01` for pavement, `pavement_02` for sidewalks - Diffuse/Roughness/OpenGL-normal maps), caching to `output/.textures/`. Anything within the "near zone" (past the farthest crosswalk + a buffer, computed per-intersection in `src/export.py:_split_near_far`) gets the 4k version; everything else gets 2k - this applies to both pavement and sidewalks, split by intersecting with a circle so a piece can straddle the boundary (`pavement_near`/`pavement_far`/`sidewalks_near`/`sidewalks_far` in the exported JSON). `blender_scene.py:make_textured_material()` wires Diffuse→Base Color, Roughness→Roughness, normal→Normal Map, and falls back to a flat color if a texture path is missing or fails to load - Phase 4 must never hard-fail without network access. Each extruded piece gets a real-world-scaled planar UV projection (`apply_planar_uv`, `bpy.ops.uv.cube_project`) so the tiling reads consistently across differently-sized pieces.

**Streetlights.** A real Poly Haven model (`street_lamp_01`, glTF bundle at 1k texture resolution - the 8k default would be enormous for a background prop instanced 4 times) is fetched once and imported as a hidden template; each corner gets a cheap linked duplicate (`obj.copy()`, sharing mesh data) positioned at that corner's fillet-arc midpoint (real geometry) pushed a few feet onto the sidewalk (a placement approximation, flagged in the exported JSON's prop `"source"` field). Falls back to a procedural pole+box if the model can't be fetched.

**Signage.** No CC0 stop-sign or school-zone-sign model was found on Poly Haven (their catalog has no traffic signage) or reliably fetchable from Kenney.nl (no stable public API/URLs to fetch from without guessing - which this project's own principle rules out). Built procedurally instead: correct MUTCD shape/color (octagon/red for stop signs, pentagon/yellow-green for school zone), a real post + flat plate mesh. One stop sign per approach (`src/export.py:_stop_sign_props`, placed near the leg's near-corner curb - an approximation, not a real traffic-engineering placement study) plus whatever's listed in a site's `config.yaml` under `props.extra` (e.g. the school zone sign on Broad St West here - genuinely site-specific, unverified-against-a-real-inventory knowledge that belongs in the site config, not the general pipeline).

**Trees.** One low-poly procedural tree (cone + cylinder - no CC0 source of genuinely low-poly stylized trees was found; Poly Haven's tree models are realistic multi-material photoscans, disproportionately heavy for background dressing instanced many times at this render's scale) is instanced along each sidewalk piece via **Blender geometry nodes** (`GeometryNodeInstanceOnPoints`, `src/export.py:_tree_points_along_piece` samples points along a piece's long axis at 25 ft spacing - standard municipal street-tree spacing, not a fabricated number; corner wedge pieces are skipped as not meaningfully elongated). Geometry-node instancing shares one mesh's data across every instance rather than creating N copies - the actual performance property requested, not just a style choice.

**Building mesh optimization.** OSM buildings are background context, not the render's subject. `src/mesh_utils.py` extrudes each footprint with `trimesh` and applies quadric decimation (`fast_simplification` backend) if the raw mesh exceeds 40 faces - in practice this intersection's buildings (5-13 vertex footprints) mostly don't, so decimation is a no-op today but the path is exercised and ready for heavier building data later. **Gotcha hit:** `trimesh` always triangulates, which reads as a faceted/crystalline shape under Blender's default flat shading even for an undecimated simple box - fixed with `bpy.ops.mesh.dissolve_limited()` after building the mesh, merging coplanar triangles back into flat faces.

## Phase 4 (3D) general notes

- Render engine: `BLENDER_EEVEE_NEXT` (the only one in Blender 4.3 — old EEVEE identifier is gone).
- Blender's own Python has no network access, no `requests`, and no access to this project's venv - all fetching (`src/assets.py`, `src/osm_context.py`) happens beforehand in the normal venv-based scripts, which pass only local file paths / already-fetched data into the exported JSON. `blender_scene.py` never calls out to the network itself.
- **Marking height must exceed pavement height** — pavement is extruded 0.05 m; anything meant to be visible on top of it (crosswalks, centerlines) must be taller (0.06 m used) or it renders buried inside the solid pavement block with zero visible effect.
- **Blender's multi-object edit mode re-extrudes every *selected* mesh, not just the active one.** Always `bpy.ops.object.select_all(action='DESELECT')` before entering edit mode on a single object, or previously-created objects (e.g. the ground plane) silently accumulate extra height every time something else gets extruded.
- OSM building footprints don't reconcile with our precise curb geometry — a few end up overlapping the pavement. `export.py` filters any building whose footprint intersects the pavement polygon. (Buildings that just look close to the road in the render are legitimate — small-town buildings really do sit near the curb; verify with a numeric intersects check before assuming a rendering bug.)
- `scripts/blender_scene.py` accepts any number of `<geometry.json> <output.png>` pairs and renders them all in **one Blender process** — each launch has ~1–1.5s fixed startup overhead, not worth paying per-render. `phase4_render_3d.py` uses this to do both scenarios in one shot.
- `fetch_buildings()`/`fetch_crossings()` cache to `output/.cache/`, `assets.py`'s texture/model fetches cache to `output/.textures/` - both keyed by (center, radius) or (slug, resolution) respectively. Delete the relevant directory to force a refetch.
- Overpass's public instances are flaky (504s are common) — `src/data_loader.py:query_overpass()` retries across 3 mirrors (`overpass-api.de`, `kumi.systems`, `openstreetmap.ru`) before giving up.
- EEVEE samples: 64 (dropped from 128 - visually indistinguishable for this flat-shaded scene, ~30% faster). Current full render (both scenarios, all fidelity features, warm caches): ~13s total.

## Known gaps / next steps

- Greenwood Ave (N & S) widths and the existing corner radius are still estimates — need field measurement or survey/aerial confirmation. Once available, update `sites/broad_st_greenwood/config.yaml` and rerun from Phase 2.
- East Broad St's "54 ft active roadway" vs "68 ft total" distinction isn't used anywhere yet (only the 68 ft total is) — could matter if a future treatment needs lane-level detail.
- ~~Asset-library question (Poly Haven PBR textures, free low-poly prop packs)~~ **Resolved** - real Poly Haven textures (asphalt/concrete) and a real streetlight model are wired in; procedural fallbacks (flagged, not hidden) cover what has no viable CC0 source (signage, low-poly trees). See "Phase 4 fidelity" above.
- Only one demo treatment scenario exists per site (`build_demo_scenario`). Additional scenarios would just be new functions in that site's `scenarios.py` composing the same treatment primitives.
- Prop placement (streetlights, stop signs, the school zone sign) is grounded in real corner/leg geometry but the *exact* setback/offset distances are approximations, not a surveyed signage inventory - flagged via each prop's `"source"` field in the exported JSON.
- Building mesh decimation is implemented but currently a no-op for this site (its OSM footprints are all simple enough to stay under the 40-face threshold) - it'll matter once/if a site uses richer building data.
