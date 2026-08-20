# Working in this repo without reinventing it

Read this before changing geometry, adding a marking, or "fixing" a number. It is not a summary
of README.md — it is the list of things an agent in this repo gets wrong, written from the ones
that actually happened, with the place each answer already lives.

README.md explains how the pipeline works. STANDARDS.md records every published figure and
whether it has been checked. **This file is about the failure modes**, because knowing the
architecture did not stop any of the mistakes below.

---

## 0. The one rule that would have prevented most of this

**Measure the drawn output, not the arithmetic that was supposed to produce it.**

Every serious bug in this repo's history has the same shape: two derivations of one fact, in
agreement with each other and not with the picture. In one session:

| what I checked | what was true |
|---|---|
| "every travel lane is exactly 11.00 ft" — computed from the section | the drawn centreline was 2.84 ft away, so a lane was 8.16 ft |
| flex posts sit in the buffer — computed from the declared cross-section | 30 posts were drawn inside the bike lane |
| the label says 9.6 ft so the geometry is wrong | the geometry was right; the *label* used the wrong datum |

So: after any change to geometry, **build the thing and measure the drawn coordinates** —
`station_offset_many(leg.centerline, np.asarray(drawn.coords))` — and open the PNG. A check that
reads the same function the renderer reads is not a check.

### 0a. Diagnose from the quantitative layer, not from the picture

The PNG is where you NOTICE a problem. It is never where you diagnose one. Everything this project
draws exists first as numbers — the resolved `DesignState`, the `PaintPiece` list, the traced kerb,
`output/*/geometry_*.json` — and that is the layer the whole design is kept in so a defect can be
*measured* instead of guessed at.

Cropping and squinting at renders cost most of one session and produced **three wrong diagnoses in
a row** on a single complaint ("the bike lane doesn't reach the kerb on W Broad"):

| guessed from the picture | what the numbers said |
|---|---|
| the hatch spills over the kerb | it reaches the kerb exactly; the kerb wanders 3 ft |
| the wrong channel removed the hatching | both channels hatch; only the colour changed |
| the unbuffered rung skips the kerbside zone | the zone is built and added, on the *other* leg |
| — | the two lanes' green is **80.3 ft apart** and the extension between them has **0.0 sq ft** of surface: dotted lines only |

Only the last line is the defect, and it took one query to find once the question was asked of the
data. The reader also cannot tell you which leg they mean from a render, and neither can you: two of
those wrong answers were about `w_broad_st_northeast` when the complaint was about
`w_broad_st_southwest`.

**So: before proposing a cause, print the numbers.** The four that answer most questions:

```python
# what is actually drawn on this leg-side, stationed in its own frame
[(p.kind.name, *np.abs(station_offset_many(leg.centerline, coords)[1])[[0, -1]]) for p in paint]
# where the traced kerb is, at the same stations
curb_offsets_at_stations(leg, side, stations)      # and curb_station_span(leg, side)
# what the treatment THINKS it placed
treatment.section(state).offsets_from_centerline_ft()
# whether a facility is continuous: distance between the pieces that should meet
unary_union(a).distance(unary_union(b))
```

**AND NEVER REBUILD A SECTION'S ARITHMETIC FROM THE CONSTANTS YOU THINK IT USED.** The third query
above exists because that reconstruction is always wrong here. A two-way section is measured from a
centreline that has been SHIFTED toward the far kerb, so the near-side travel edge is 3.8–6.5 ft on
Broad St and not `TARGET_LANE_WIDTH_FT`; assuming the 11 ft produced a confident "a protected lane
needs 22.0 ft and the kerb gives 20.32, so it does not fit", and a four-option design decision put
to Danny on the strength of it. Nothing about it was true.

The tell was in the numbers and it is worth learning as a shape: printing demand against room gave
**identity on all six corridor legs** — 14.71/14.71, 20.32/20.32, 26.29/26.29. Two figures that
agree to the last decimal in every case are not a confirmation, **they are one figure**: the
section is sized to fill the room at the leg's narrowest point, so it fits there by construction and
the comparison can never fail. When a check cannot fail, the thing it was going to prove is not a
finding — go and find which span, which datum, or which frame differs instead. (That is also how
`section_holds_to_ft` came to cut a lane to a 6 ft stub: searching from station 0, the first station
where the kerb is "inside the section" is the narrowest point the section was sized on.)

A gap profile — kerb offset minus outermost drawn offset, station by station — turns "it looks
janky" into "bare from station 0 to 63.7, then 1.4 ft widening to 2.6 ft by station 118", which
names the mechanism on its own. `scripts/measure_drawn.py` is where that belongs; extend it rather
than writing another throwaway script, and NEVER answer a geometry question by cropping a PNG.

**AND WHEN A MARKING DOES NOT REACH, PRINT EVERY LIMITER, NOT THE FIRST ONE.** Four separate
things decide where a kerbside marking starts, they disagree, and the answer is whichever sits
furthest out — so measuring one, finding it innocent, and moving on proves nothing:

| limiter | where it lives |
|---|---|
| the corner return | `leg_clearance_ft` |
| this junction's mouth | `junction_mouths_ft`, with `corner_tangent_station_ft` as its fallback |
| the crossing band | `crosswalk_reach_on_leg_side_ft` — the **BAND**, never `crosswalk_offsets[leg].offset_ft`, which is the crossing's CENTRE and reads 32.3 where the skewed band reaches 53.8 |
| whether the kerb is traced there | `curb_station_span` — every kerb at all five sites starts 12–58 ft out, because OSM traces the block and not the corner |

One session found each of those binding on a different leg of ONE junction, and reported a 21.5 ft
defect that did not exist by measuring the third one wrong. The tell that you have the right
limiter is arithmetic identity: `paint@` equalling `clearance_ft` to two decimals on five
leg-sides at four sites is a mechanism, and "roughly similar" is a coincidence.

Note also that a length derived from the corner fillet **diverges as a corner sharpens** —
a tangent point sits `R/tan(θ/2)` back, so 1.0 R at a square junction and 2.5 R at a 44° Y. That
is §0b's lesson in a second disguise: the quantity is local, but it is not the quantity you meant.

### 0b. MEASURE AT THE FRAME THE READER IS LOOKING AT — legs are longer on a wide sheet

`--frame-scale` does not just crop wider. It scales `leg_lengths`, so a 130 ft leg is 325 ft on a
2.5× sheet and every function that measures "over the leg" gets a different answer. Measuring the 1×
build while the reader is looking at a 2.5× sheet reports a defect fixed that is not:

```bash
HOPEWELL_FRAME_SCALE=2.5 .venv/bin/python scripts/measure_drawn.py wbroad_louellen --scenario ...
```

Worse than a measuring mistake, it is a *design* mistake waiting to happen. A rule sized off a
TOTAL over the leg — total kerb swing, total anything — silently becomes a function of the render
frame, so the drawing changes when you widen the picture. The bikeway followed its kerb at 1× and
stood 8.4 ft clear of it at 2.5×, from one `<= 6.0 ft` threshold. **Anything compared against a
threshold has to be a rate, a curvature, or something else local**; if it is an amount accumulated
over a leg, the leg's length is an input and the leg's length is a rendering decision. `git grep`
for a new constant's units before you add it.

**AND A MINIMUM OVER A LEG IS A TOTAL TOO.** This bit twice, the second time through
`narrowest_half_width_ft`, which took the least half-width from the tracing's start to
`leg.centerline.length` — a longer leg simply reaches further and finds a narrower pinch, so the
street was judged *less* able to hold a facility on a wider sheet. W Broad's southwest approach:
20.32 ft at 1×, 16.58 ft at 2.5×, off a pinch 318 ft out that the 1× sheet does not show. It cost
that leg its 3 ft buffer and its flex posts on the wide render only — a protected bikeway in one
picture and a painted one in the other. The fix is a span that is a design decision:
`Leg.design_length_ft`, the configured length before `frame_scale()` multiplies it. So the test is
not just "is this a rate" — it is **"could this answer have been different on a wider sheet"**, and
min/max/any/all over a leg all fail it.

Note also that `tests/conftest.py` builds `site_models` at `WIDE_FRAME_SCALE = 2.2`, so the fixture
is *already* not 1× — a test can agree with the goldens and disagree with every render.

---

## 1. Before you write a number, look for it

There is exactly one home for each of these. Adding a second copy is the most common defect here.

| you need | it already exists as | in |
|---|---|---|
| travel lane target width | `TARGET_LANE_WIDTH_FT` | `src/geometry/treatments/` |
| a CRS string | `WGS84`, `NJ_STATE_PLANE_FT` | `src/geometry/model/` |
| how far the divider sits off the alignment | `divider_shift_toward_ft(state, leg, side)` | `src/geometry/treatments/` |
| a travel lane's real width | `travel_lane_width_ft(state, leg, side, painted_ft)` | `src/geometry/treatments/` |
| hold a lane at target, spend the surplus | `hold_travel_lane_at_target(state, leg, side)` | `src/geometry/treatments/` |
| may this kerb hold parking? | `kerb_may_hold_parking(state, leg, side)` | `src/geometry/treatments/` |
| which side of a leg faces north/south | `side_facing(leg, "south")` | `src/geometry/model/` |
| room beside a lane, measured on the traced kerb | `narrowest_half_width_ft`, `kerbside_allowance_ft` | `src/geometry/model/`, `treatments.py` |
| the centreline stripes both views draw | `centerline_paint_ft` | `src/render/crosswalks.py` |
| a band between two lateral offsets | `offset_band_polygon` | `src/geometry/model/` |
| a line N ft to one side | `inset_line_ft` — **never** `offset_curve` for stationed work | `src/geometry/model/` |

**Search before writing.** `grep -rn "SOME_CONSTANT" src/` costs nothing. Writing the second copy
costs a session: the far-kerb rule was written inline in one site's `scenarios.py`, the sibling
site never got it, and two legs shipped with 11.68 ft and 13.21 ft lanes.

---

## 2. The two datums, which are 25 ft apart

This has caused three separate bugs. Learn it once.

- **WHETHER there is room** is a measurement of the **traced kerb** —
  `narrowest_half_width_ft(leg, side)`.
- **WHERE the paint goes** is an offset from the **nominal half-width** (`leg.curb_to_curb_ft/2`),
  because that is the datum `MarkedParking` and `LaneNarrowing` subtract their own widths from.

On `broad_st_east` those differ by 25 ft (config says 68.0; the traced kerbs are 43.26 apart).
Size paint off the traced number and you draw a 20 ft hatch. `apply_osm_parking` spells this out
in a comment; read it rather than rediscovering it.

---

## 3. Adding a marking touches six places, and the seventh is the golden

README.md has the six-place table. Two additions to it:

1. **The channel decides the colour.** `blender_scene.py` draws every edge-line channel in the
   white `marking_mat` and only the centreline channel in yellow. A yellow marking routed through
   an edge-line channel renders white in 3D and yellow in 2D, with nothing to catch it.
2. **`POLYLINE_CHANNELS` in `tests/test_geometry_regression.py` is derived from
   `markings.CHANNELS`** — so a new channel gets a golden automatically. It was a hardcoded list
   and it drifted immediately: a marking with 30 segments, drawn in both views, had no golden.

**Pin extent, not just position.** The digest pinned `stop_bar_centre_m` and `stop_bar_axis` but
not the span, so moving where the bar starts changed nothing in any golden. Same hole existed for
the crosswalk reach and for `centerline_paint_m`. If you add a marking, pin where it is *and how
far it goes*.

---

## 4. Invariants: re-express, never relax

When a check fires on a design you believe is correct, the check is usually right about the
property and wrong about the frame. Fix the frame.

`PaintClearOfTheTravelLane` measured the lane from the alignment, which only holds while the two
travel lanes straddle it. The two-way lane breaks that. The fix was to read the shifted lane edge
off the design — **not** to exempt those legs, which would have dropped the check on the design
most likely to get the arithmetic wrong.

Also: **verify a new check fails first.** `git stash` the fix, watch it report, restore. A check
that has never fired pins nothing. Both new checks in this session were verified that way (18
violations before, 0 after).

And note which checks are **fatal** — most are, so they block the 3D export. A violation printed
by `phase3_treatments.py` still saved a picture; that does not mean it passed.

---

## 5. Where facts live, and what that decides

| it is… | it lives… |
|---|---|
| a fact about the street as it exists | on `IntersectionModel`, resolved once at load |
| a decision a proposal makes | a `Treatment` subclass, geometry as a *method* |
| a way of drawing something decided | a `PaintKind`, or a prop |
| a published figure | a row in **STANDARDS.md**, with its provenance tier |

**The last row is not optional and I skipped it.** Six standards constants went into code comments
from memory in one session — the exact failure STANDARDS.md's preamble describes. If you write a
number and cite NACTO/AASHTO/MUTCD for it, add the row in the same commit, and mark it *as cited*
unless you actually opened the document.

---

## 6. Data comes from somebody else, so validate at the boundary

`src/sources/schemas.py` holds a pandera schema per external layer, validated in
`load_road_network`, `load_parcels` and `assessor.py`. Two things to know:

- **A wrong CRS returns ZERO ROWS, not an error** — indistinguishable from "nothing mapped here".
  It cost two round trips in one session. Compare CRS by **horizontal identity**, never by string:
  `MercerCountyParcels.shp` is a compound CRS whose WKT matches no EPSG code.
- **`strict=False` is deliberate** for the column *set* (third-party files add columns), but the
  columns we read are a contract — not-null where the real file is never null, with a domain.
  `nullable=True` everywhere is not a lax contract, it is the absence of one.

---

## 7. OSM is partial, and partial in a specific way

- **A restriction over PART of a kerb does not close it.** OSM splits the way. `broad_st_east` is
  `no_parking` for its first 79.6 ft and explicitly `restriction=none` for the 90.4 ft beyond;
  reading any prohibiting span as closing the kerb hatched 90 ft of legal parking. Use
  `kerb_may_hold_parking`.
- **Driveway coverage is ~29%** of parcels fronting Broad St. State the coverage fraction beside
  any count. Parcel frontage is *not* a proxy — plenty of lots have no driveway.
- **A branch's side must be probed ~25 ft along it.** A driveway joins at a shared node *on* the
  centreline, so the northing there carries no side information and comes out as float noise.
- **The cache and the test fixture are separate.** `--refresh-osm` re-pulls
  `output/.cache/borough_*.json`; `tests/fixtures/osm_cache/` does not update itself.

---

## 8. The verification loop, in cost order

1. Write the test; **confirm it fails** against the pre-change code.
2. `scripts/export_all_scenarios.py /tmp/before` → change → `/tmp/after` → `scripts/diff_exports.py`.
3. `./scripts/test.sh` (includes lint, import contracts, goldens).
4. `scripts/build_all.py --render-3d`, **and open the PNGs**.
5. Measure the drawn geometry numerically (see §0).

A golden failure is not automatically a bug. Read the diff, confirm every moved number is one you
meant to move, then `--force-regen` and commit the goldens **in the same commit as the cause**.
