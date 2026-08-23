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

## The third reason, and the one you can see: the drawing discards surveyed ground truth

Measured 2026-08-17 at Broad & Greenwood, rendered at `--frame-scale 2.5` (431 ft frame radius):

```
10  OSM crossings inside the frame
 4  drawn - exactly the four matched to this junction's legs
 6  DISCARDED, at 263-411 ft, three of them tagged crossing:markings=zebra
```

Blackwell & Broad is in that picture with its crosswalks traced, and the render shows bare asphalt
there. This is not a filter that can be widened: a drawn crossing needs a STATION, an ORIENTATION
and a REACH TO BOTH KERBS, and all three are properties of a leg. A junction the site does not
model has no legs, so its surveyed crossings are unreachable however far the fetch radius goes.

It is the same fault as the kerbs, which were per-leg until `cb9c8b6` moved them to the drawing
radius on the grounds that *what a drawing contains is a question about the drawing*. Crossings
cannot follow without somewhere to hang a station and a kerb pair - which is the Road.

This one matters more than the five bugs above, because those were wrong numbers a reader could not
see. This is a render that shows a marked crosswalk as unmarked asphalt, to an audience deciding
whether to build something. The wider the frame - and a corridor argument needs a wide frame - the
more of the surveyed borough it silently drops.

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

## What the checkpoints measured, 2026-08-17

Steps 1–3 have run. The road model reproduces the current geometry, so the gate above is passed —
but two of the things measured on the way change what step 4 costs, and both were found by writing
the checkpoint rather than by reading the code.

**The frame was never leg-shaped, so step 4 is smaller than 14k lines suggests.** Every function in
`model/leg_frame.py` is annotated `leg: "Leg"` and not one of them touches a leg: they read
`.centerline` and one of `.left_curb`/`.right_curb`. Under those names a `Road` goes through
`curb_station_span`, `curb_offsets_at_stations` and `narrowest_half_width_ft` unmodified. Moving
the datum is a change of CALLER, not a rewrite of the frame — which is why `Alignment` and
`Approach` landed without a single exported coordinate moving (0 of 13 exports differ, twice).

**Steps 4 and 5 are one step, not two.** The plan has the config migration last, after the frame
moves. It cannot be: the loader cannot fit one continuous centreline while each half carries a
single `curb_to_curb_ft`. W Broad is 55.5 ft west of Greenwood and a different width east of it, so
a Road needs the station-ranged width *on the first day of the inversion*. The config FORMAT can
stay put — the ranges seed from today's two per-leg entries — but the internal model cannot. Plan
step 4 and the width half of step 5 as one commit.

**The through-join is broken in a way that only a road can fix, and it is worse than recorded.**
`fitting.py:_join_through_legs` says it gives the two halves of a street a shared junction point.
`_blend_onto` applies that point as a LATERAL offset profile — it takes `station_offset_many`'s
offset and discards its station — so it can slide a leg's end sideways onto the joint but never
along the street. At W Broad & Louellen 2.74 ft of the 3.1 ft gap is longitudinal and survives,
under a NOTE announcing that the halves were joined. Splitting the difference makes it worse, not
better: putting the road's node at the midpoint of the two ends moves the error onto the leg that
had none, and cost 0.75 ft of width at `w_broad_st_northeast` station 5, where the kerb flares 0.68
ft per foot. There is no node station that reproduces two datums 2.74 ft apart. Only building the
line once removes the gap.

And that is harder at Louellen than anywhere else, which is worth knowing before starting: the two
halves come off DIFFERENT SRIs (CR 518 turns west onto Louellen, CR 654 carries on southwest), so
there is no single source alignment to build the road from. The seam is real and has to be eased,
the way `network/corridor.py:_eased_alignment` already eases a corridor across one.

**The corridor's EXTENT was a render parameter, and that was the bug class itself.** Extensions were
carried out from the end of a junction piece - a frame-cut leg - so the search window and the
fetch-radius cap in what is now `_traced_end_ft` were both relative to a seam that moves with
`HOPEWELL_FRAME_SCALE`, and the junction centre defining the cap circle was picked by proximity to
that seam. A wider sheet slid the window outward and discovered street the narrower sheet never
looked for: Columbia Ave's traced coverage moved 369 ft between sheets, Greenwood's 196 ft. Since a
facility's rung is chosen over a span, the viewport was voting on the design. Anchoring on the
junction NODE - a surveyed point - takes Columbia's movement to 0.7 ft, and moved **0 of 15
exports**, because nothing renders from the corridor yet.

**What that leaves is one coupling, and it is the one step 5 deletes.** `intersection/load.py`
multiplies the surveyed leg length by `frame_scale()`, so a junction piece can still be pushed past
the last traced kerb and the corridor then claims street nothing was surveyed on (Greenwood 1564.9
-> 1698.4 ft). No paint comes of it - `corridor_paint` refuses an untraced span by name - but a
Coverage denominator measured over a viewport is not a figure. `test_corridor.py` pins both halves:
the surveyed span invariant now, and the raw extent as a `strict` xfail that fails the moment leg
extent stops being a render parameter.

**`nj31_wdelaware` does not build at all** on current OSM — `build_all` reports 4 of 5 sites ok and
that one as "Pavement ring is self-intersecting". It is the Pennington failure this document names,
and it is invisible to the test suite, because the committed fixture cache carries no borough
snapshot for Pennington and every test that would build it skips. Step 4 should be measured against
it: a junction whose corner-fillet model fails on leg stubs is the case the road model exists for.
