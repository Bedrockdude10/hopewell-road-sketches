# Replacing legs with a road network

**Status: proposal, nothing built.** Written 2026-08-15 after a session in which the leg
decomposition produced five bugs of one kind and made every corridor question unanswerable
inside the pipeline.

## The endpoint

A **Road** is the primary object: one continuous centreline, one traced kerb line per side,
continuous stationing, with facts positioned along it — driveway openings, crossings, parking
restrictions, field measurements, speed limits.

A **Junction** is a *node on roads*, not a thing that owns four stubs. Corner fillets are computed
between two roads' kerb lines at a node. A crossing is a station and an orientation on a road.
Signal heads face an approach, which is a road direction at a node.

**There is no `Leg`.** Not "a leg becomes a view" — the object goes away, and the things that
currently ask a leg a question ask a road-and-a-station instead.

## Why, in one number

`curb_to_curb_ft` says broad_st_east is 68 ft. Its traced kerbs are 21.97–34.55 ft from the
centreline. **That contradiction is the entire reason for five bugs in one session:**

| bug | what disagreed |
|---|---|
| driveway break covered only part of the lane | region started at a fixed 11 ft, not the section edge |
| 30 flex posts inside the bike lane | declared vs resolved cross-section |
| far-kerb paint 20 ft wide | sized off traced surplus, drawn in nominal offsets |
| `TravelLanesHoldTheTarget` false positive | unpainted side measured nominally, kerb 1.66 ft inside |
| 8 of 9 driveways produced no opening | kerb matched against nominal half-width |

Every one is "config says X, the kerb says Y." A road with one kerb line cannot hold that
contradiction, so the whole family stops being expressible. That is the case for doing this —
not tidiness.

The second reason: **the pipeline cannot answer a corridor question.** Every corridor number this
session came from a scratch script on raw OSM, and three of them were wrong, because
`narrowest_half_width_ft`, `opens_the_kerb` and the daylighting rules are only reachable through a
leg and a corridor has no legs.

## What the config becomes

Ten per-leg fields today: `sri`, `bearing_deg`, `street_name`, `curb_to_curb_ft`, `confirmed`,
`width_provenance`, `width_measured_at`, `centerline_style`, `working_length_ft`, `source`.

Under roads:

| field | becomes |
|---|---|
| `bearing_deg` | **gone.** It exists only to decide which piece of an SRI is which leg — a problem created by cutting the road up |
| `curb_to_curb_ft`, `confirmed`, `width_provenance`, `width_measured_at`, `source` | a **station-ranged measurement** on a road: "55 ft 6 in, field-measured, stations 0–40 at the Greenwood node". A field measurement is still the best evidence *where it was taken* — it stops being a whole-leg constant |
| `working_length_ft` | **gone.** Replaced by the drawn extent |
| `centerline_style`, `sri`, `street_name` | station-ranged attributes on a road |

The provenance machinery survives intact and gets *better*: today a measurement taken "immediately
east of the intersection" is applied to a whole 170 ft leg. As a station range it applies where it
was taken and the tracing governs elsewhere — which is what `src/provenance.py` already argues for
and cannot currently express.

## Blast radius, measured

```
22   files in src/ and scripts/ reference legs
373  references to leg_name
104  accesses to .legs
26   LegSide,  21 LegTarget
28   test files touch legs (of 25 — some multiple times, i.e. effectively all)
10   per-leg config fields, across 5 sites
~14k lines in geometry/ + render/ + checks.py
```

Most of the 373 are **consumers** — they take a leg and ask for `.centerline`,
`.curb_to_curb_ft`, `.left_curb`. Those are mechanical. The genuinely hard parts are the few
places that *define* the frame:

- `intersection.py:load_intersection_model` — builds legs from config + SRI matching + kerb fitting
- `model.py` — `station_offset_many`, `curb_offsets_at_stations`, `inset_line_ft`,
  `offset_band_polygon`: the leg frame itself, ~470 references to `centerline`
- `_build_corners` / `build_pavement_polygon` — the corner-fillet model, which is also what fails
  at Pennington and in Louellen's 190–220 ft band

## What breaks

- **Every golden.** Stationing changes when the datum becomes a continuous road, so all 16
  regression files regenerate. That is the one guard we would be flying without during the
  migration, and it argues for doing it in a branch with the old and new exports diffed by
  `scripts/diff_exports.py` rather than by golden equality.
- **All four site configs**, plus `site_schema.py` (pydantic) and `sites/README.md`.
- **`ExtraProp`, signal blocks, `no_turn_on_red_legs`** — all leg-keyed in config.
- **The 3D export format**, which is per-leg (`legs: [...]` with `near_m`/`far_m`), and
  `blender_scene.py` reads it. That is the seam with no test behind it.

## Sequencing, with a revertible checkpoint

1. **Build `Road` alongside legs, changing nothing.** One road per SRI across a whole borough
   snapshot, continuous kerb lines, station-ranged facts. Assert the new model reproduces each
   existing leg's width and kerb within tolerance. **Nothing renders from it yet** — this is the
   checkpoint where the work is still free to abandon, and where a real answer arrives: does the
   traced kerb alone reproduce the field measurements?
2. **Move the corridor questions onto it first** — the parking baseline, the driveway counts, the
   side comparison. These have no goldens and are currently wrong in scratch scripts, so the new
   model is strictly better than the status quo and gets exercised on real questions.
3. **Re-express openings, crossings and restrictions as road facts.** Already half-done: kerb
   openings now collect by drawing radius rather than leg membership.
4. **Move the frame** — `station_offset_many` and friends take a road and a station range. This is
   the large mechanical pass and where the goldens all move.
5. **Delete `Leg`,** and with it `bearing_deg`, `working_length_ft`, and the nominal half-width.

## What this collapses

Three open tasks are the same problem and stop existing: **#7** (corner-return window scaled to
the junction), **#12** (per-leg `working_length_ft`), **#13** (corridor parking baseline). The
Pennington failure and Louellen's 190–220 ft dead band are both the corner-fillet model applied to
leg stubs, so step 4 is where they get addressed rather than worked around.

## The honest risk

Step 4 touches ~14k lines with every golden regenerating at once, and the 3D export seam has no
automated check. If the work stalls midway the repo is worse than either endpoint. Steps 1–3 are
independently valuable and revertible; **step 4 is the commitment**, and it should not start until
1–3 have shown the road model reproduces the current geometry.
