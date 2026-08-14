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

---

## 1. Before you write a number, look for it

There is exactly one home for each of these. Adding a second copy is the most common defect here.

| you need | it already exists as | in |
|---|---|---|
| travel lane target width | `TARGET_LANE_WIDTH_FT` | `src/geometry/treatments.py` |
| a CRS string | `WGS84`, `NJ_STATE_PLANE_FT` | `src/geometry/model.py` |
| how far the divider sits off the alignment | `divider_shift_toward_ft(state, leg, side)` | `src/geometry/treatments.py` |
| a travel lane's real width | `travel_lane_width_ft(state, leg, side, painted_ft)` | `src/geometry/treatments.py` |
| hold a lane at target, spend the surplus | `hold_travel_lane_at_target(state, leg, side)` | `src/geometry/treatments.py` |
| may this kerb hold parking? | `kerb_may_hold_parking(state, leg, side)` | `src/geometry/treatments.py` |
| which side of a leg faces north/south | `side_facing(leg, "south")` | `src/geometry/model.py` |
| room beside a lane, measured on the traced kerb | `narrowest_half_width_ft`, `kerbside_allowance_ft` | `src/geometry/model.py`, `treatments.py` |
| the centreline stripes both views draw | `centerline_paint_ft` | `src/render/crosswalks.py` |
| a band between two lateral offsets | `offset_band_polygon` | `src/geometry/model.py` |
| a line N ft to one side | `inset_line_ft` — **never** `offset_curve` for stationed work | `src/geometry/model.py` |

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
