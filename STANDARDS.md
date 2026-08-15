# Traffic standards used in this project

Every published figure this repo relies on, what it says, the constant that encodes it, and
where that constant lives. The point is to stop re-deriving the same lookups: if a number in
the geometry looks arbitrary, it should be findable here in one pass.

**This is an index, not an authority.** Where a row matters to a decision, open the source.
Several of these came into the repo as a code comment written from memory, and a comment that
has never been checked against the document it cites is a plausible-looking number, not a
standard.

### Provenance key

| | meaning |
|---|---|
| **Verified** | checked against the published source during this project; the source is linked |
| **As cited** | stated in the repo's own comments and not independently checked here |
| **Local** | a figure supplied for this project rather than read from a document |
| **Modelled** | our own choice, calibrated against site measurements — not a standard at all |

---

## 1. New Jersey statute — parking prohibitions (daylighting)

The governing law for every no-parking setback drawn here. NJ has no statute called
"daylighting"; it gets there through the ordinary parking prohibitions.

Encoded in [`src/geometry/daylighting.py`](src/geometry/daylighting.py), whose module docstring
is the long-form version of this section.

**R.S. 39:4-138**, as amended by P.L. 2009 c.257 — parking prohibited: *(as cited)*

| clause | prohibition | constant | value |
|---|---|---|---|
| (a) | within an intersection | — | — |
| (e) | within 25 ft of the nearest crosswalk | `CROSSWALK_SETBACK_FT` | 25.0 ft |
| (e) | …reduced where a curb extension is built | `CROSSWALK_SETBACK_WITH_BULBOUT_FT` | 10.0 ft |
| (e) | within 25 ft of the **side line** of an intersecting street | `SIDELINE_SETBACK_FT` | 25.0 ft |
| (e) | …reduced where a curb extension is built | `SIDELINE_SETBACK_WITH_BULBOUT_FT` | 10.0 ft |
| (h) | within 50 ft of a stop sign | `STOP_SIGN_SETBACK_FT` | 50.0 ft |
| (i) | within 10 ft of a fire hydrant | `FIRE_HYDRANT_SETBACK_FT` | 10.0 ft |

All four are floors. The binding one is whichever sits furthest from the junction.

Two things worth remembering, because both were bugs:

- **(e) has two arms.** The crosswalk arm and the side-line arm. Only the crosswalk arm was
  applied at first, so legs with no marked crossing got no setback at all.
- **The statute is about *an* intersection, not *this* one.** A leg drawn 374 ft out crosses
  Blackwell Avenue and Model Avenue too, and each gets its own (e) setback. See
  `src/geometry/cross_streets.py`.

**R.S. 39:4-138.6** — municipal authority. *(as cited)* Hopewell Borough may set its own
permissible distances by ordinance, but may **not** permit parking within 25 ft of a crosswalk
or side line, nor within 50 ft of a stop sign in a school zone while school is in session.

> **Open item.** The Borough's traffic chapter has not been read — ecode360 refuses automated
> requests. The figures above are the **state** defaults, which is the correct default absent a
> local ordinance, and for (e) is a floor an ordinance cannot lower. If the Borough has adopted
> something stricter these numbers go **up**; none of them may go down without checking
> 39:4-138.6 first.

---

## 2. MUTCD

### Edge lines at driveways and intersections — **Verified**

**MUTCD §3B.07.** An edge line is **maintained across the intersecting approach of a driveway
that does not meet the definition of an intersection**, and is **interrupted at an actual
intersection**, where a dotted extension may carry it through instead.

This is the one rule in this document checked against the published source during the project
(2026-08-07). It settled the question of whether a driveway should be hatched or the shoulder
line simply carried past: **carried past.**

Encoded as `LINES_UNBROKEN_BY_A_DRIVEWAY` in
[`src/geometry/markings.py`](src/geometry/markings.py), consumed by `KerbOpenings.against()` in
[`src/geometry/paint.py`](src/geometry/paint.py). The intersection half was already correct —
the line stops at Blackwell Avenue's mouth and carries across the driveways.

### Other MUTCD figures — *as cited*

| figure | value | constant | file |
|---|---|---|---|
| Normal walking speed, pedestrian clearance | 3.5 ft/s | `MUTCD_WALKING_SPEED_FT_S` | `src/metrics.py` |
| Slower walker (the person a treatment is usually for) | 3.0 ft/s | `SLOW_WALKING_SPEED_FT_S` | `src/metrics.py` |
| Stop line to near edge of crosswalk | 4.0 ft | `STOP_BAR_TO_CROSSWALK_GAP_FT` | `src/render/crosswalks.py` |
| Longitudinal line width | 4–6 in | — | see the caveat below |
| Dotted lane extension across a conflict area | 2 ft mark / 2 ft gap | `DOTTED_MARK_FT`, `DOTTED_GAP_FT` | `src/geometry/paint.py` |
| Tubular marker banding | at least two retroreflective bands | — | `scripts/blender/blender_props.py` |
| Sign codes used | R10-11 (no turn on red), W11-2 (ped crossing) | — | `scripts/blender/blender_props.py` |

> **Known divergence — line width.** `LANE_EDGE_LINE_WIDTH_FT` is `0.25 / FT_TO_M` = **0.82 ft
> (9.8 in)**, chosen for legibility at render scale, against MUTCD's 4–6 in. This is not a
> rounding error: it is 1.64 ft of a cross-section once both stripes are counted, and it is the
> difference between E Broad's north kerb carrying a conventional 5 ft bike lane and carrying
> none. See the comment at `treatments.py:BIKE_LANE_WIDTH_FT`, which lists the three ways out.

---

## 3. AASHTO — *as cited*

| figure | value | constant | file |
|---|---|---|---|
| Bike lane design width (and the width to design to) | 5 ft | `AASHTO_MIN_BIKE_LANE_FT` | `src/geometry/treatments.py` |
| Bike lane hard floor, no curb face | 4 ft | `MIN_BIKE_LANE_FT` | `src/geometry/treatments.py` |
| Parallel parking lane depth | 8 ft | `PARKING_STALL_DEPTH_DEFAULT_FT` | `src/geometry/treatments.py` |
| Parallel parking stall length | 22 ft | `PARKING_STALL_LENGTH_DEFAULT_FT` | `src/geometry/treatments.py` |
| Low-speed side friction factor | 0.30 | `TURN_SIDE_FRICTION` | `src/metrics.py` |
| Minimum-radius relation | `V = sqrt(15·R·(e+f))` | `TURN_SUPERELEVATION = 0.0` | `src/metrics.py` |

Two of these carry project decisions worth knowing:

- **5 ft is a minimum, not a target.** The project proposes the narrowest lane the standard
  permits and puts the rest into a 2 ft buffer, on the reasoning that the buffer is what a flex
  post stands in.
- **4 ft is a real floor and the buffer outranks the lane.** Where a kerb is a few inches short,
  the lane narrows toward 4 ft rather than the buffer being spent — a 4.5 ft lane with a post
  beside it beats a 5 ft lane with a truck beside it. See `widest_protected_lane_ft`.
- **Turn speed is labelled *modelled, not measured* wherever it is shown.** It is a comfort model
  of a vehicle tracking the curb face, not a design speed.

---

## 4. NACTO — *as cited*

| figure | value | constant | file |
|---|---|---|---|
| Minimum pedestrian refuge island width | 6 ft | `NACTO_MIN_REFUGE_ISLAND_WIDTH_FT` | `src/geometry/treatments.py` |
| Typical low-cost paint buffer / shoulder stripe | 5 ft | `LANE_NARROWING_DEFAULT_STRIPE_FT` | `src/geometry/treatments.py` |
| Urban minimum travel lane (with AASHTO) | 10 ft | — | see `TARGET_LANE_WIDTH_FT` below |
| …**plus** NJDOT's truck-route allowance (§6) | **11 ft** | `TARGET_LANE_WIDTH_FT` | `src/geometry/treatments.py` |
| Crosswalk visibility ranking | continental > ladder > transverse | — | `src/render/crosswalks.py` |

### Two-way (bidirectional) bikeway — *as cited, and NONE OF IT CHECKED*

Added 2026-08-14 for the Broad St corridor. **Every figure below was written into a code comment
from memory and never opened against the Urban Bikeway Design Guide**, which is exactly the
failure this file's preamble describes. They are plausible and they are load-bearing - the 10 ft
row is what the corridor's lane width was reduced TO in order to free parking - so they need
checking before any of this goes to a county engineer.

| figure | value | constant | file |
|---|---|---|---|
| Two-way lane width, desirable | 12 ft | `TWO_WAY_BIKE_LANE_WIDTH_FT` | `src/geometry/treatments.py` |
| Two-way lane width, minimum | 10 ft | `MIN_TWO_WAY_BIKE_LANE_FT` | `src/geometry/treatments.py` |
| Buffer beside moving traffic, with vertical elements | 3 ft | `TWO_WAY_BIKE_LANE_BUFFER_FT` | `src/geometry/treatments.py` |
| Travel lane floor beside a two-way lane | 10 ft | `MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT` | `src/geometry/treatments.py` |

### Driveways across a protected bike lane — **Verified 2026-08-14**

**The lane continues through the driveway. Nobody loses a driveway.** That is the standard
condition for this facility, not a compromise it tolerates, and it is worth stating plainly
because the opposite assumption kills the proposal politically before it is ever drawn.

**MUTCD 11th ed. Part 9E** ([source](https://www.roundabout.tech/mutcd/11r1/part-9e-markings/)):

| section | force | wording |
|---|---|---|
| §9E.03(07) | **Standard** | "Extensions of bicycle lanes through intersections **shall** use dotted line patterns." |
| §9E.04(02) | Option | "Bicycle lanes **may** be continued through a driveway using solid or dotted longitudinal lines." |
| §9E.04(03) | Option | Bicycle symbol, arrow, or word markings **may** be used in bicycle lane extensions through driveways. |
| §9E.06(15) | **Guidance** | "Lane extension markings **should** be used to extend a buffer-separated bicycle lane across intersections and driveways." |

So a driveway is NOT an intersection for §9E.03 purposes - the Standard there is about
intersections, and driveways fall under §9E.04's Option. But §9E.06's Guidance is the operative
one for what this project draws, because ours is buffer-separated: extension markings *should*
carry across driveways.

**NACTO Urban Bikeway Design Guide**, contraflow and bidirectional protected lanes
([source](https://nacto.org/publication/urban-bikeway-design-guide/designing-bikeways-for-all-ages-and-abilities/protected-bike-lanes/designing-protected-bike-lanes/)
- page returns 403 to automated fetches; figures below are from NACTO's own indexed summary, so
treat as *as cited* until someone opens the guide):

- bidirectional protected lanes **must continue through intersections and driveways**
- **dotted yellow centrelines** along bidirectional lanes and through the associated crossbikes
- BIKE LANE symbol or marking **after driveways**, after intersections, and at least every 500 ft
- crossbikes through all crossings including driveways; cities *may* apply them at busier driveways

What this repo now draws, and where each answer comes from:

| marking | at a driveway | authority |
|---|---|---|
| edge lines | break, continue as a dotted extension | §9E.06(15) Guidance |
| yellow contraflow centre stripe | **carries through as its own dashes** | NACTO dotted yellow centreline; §9E.06(15) |
| green surface | continues across | our choice — **Modelled**; colour is not specified |

The centre stripe used to stop dead at each driveway - 22 dashes on a kerb with two of them
against 30 on a kerb with none - while the edge lines continued and the green carried across.
Three answers to one conflict point, and that one belonged to nobody. Fixed 2026-08-14.

> **Still missing: the BIKE LANE symbol after each driveway**, which NACTO asks for and
> §9E.04(03) permits. Nothing in this repo draws a pavement word or bike symbol at all, so it is
> a new marking rather than a parameter - see the "new marking touches six places" checklist in
> README.md.

### NJDOT says this facility is unacceptable — **Verified 2026-08-14**

*Bicycle Compatible Roadways and Bikeways: Planning and Design Guidelines*, NJDOT, 1996
([source](https://nj.gov/transportation/about/publicat/pdf/BikeComp/introtofac.pdf)), read
2026-08-14. **It rules out the corridor treatment this repo draws**, in terms:

> "Bicycle lanes should always be one-way facilities and carry traffic in the same direction as
> adjacent motor vehicle traffic. **Two-way bicycle lanes on one side of the roadway are
> unacceptable** because they promote riding against the flow of motor vehicle traffic.
> Wrong-way riding is a major cause of bicycle accidents and violates the Rules of the Road
> stated in the Uniform Vehicle code."

And separately, on the adjacent-path form of the same idea: *"Two-way bicycle paths located
immediately adjacent to a roadway are not generally recommended."*

**This is not a technicality and it must be stated in any submission.** It is the state DOT's
published guidance for the state the project is in, and a county engineer may cite it directly.

What can honestly be said against it:

- **It is from 1996** and predates the modern separated-bikeway evidence base entirely. Its
  vocabulary has no "protected", "buffered" or "separated" bike lane — it addresses shared lanes,
  paved shoulders and painted one-way bike lanes only. The facility it calls unacceptable is a
  *painted contraflow lane*, not a vertically separated two-way lane with its own signal phasing.
- **Federal guidance has since moved.** MUTCD 11th ed. Part 9E provides markings for
  buffer-separated and separated bike lanes (§9E.06), and NACTO's Urban Bikeway Design Guide
  treats bidirectional protected lanes as a standard facility.
- **NJ adopts the federal MUTCD**, so §9E governs the *markings* regardless. The 1996 document is
  guidance on facility *selection*, and that is where the conflict lives.

None of that makes the objection go away. It means the corridor proposal has to argue the case
explicitly rather than assume it: **cite the 1996 guidance, say why it is being departed from,
and expect that to be the first question asked.** Recorded here so the argument is made once and
found again, rather than rediscovered under scrutiny.

### Skewed intersections, and what a bikeway does at one — **Verified 2026-08-15**

W Broad & Louellen is not a plain T, and treating it as one is why its render looks wrong.
Measured from the site config:

| | |
|---|---|
| Louellen St west × W Broad south-west | **43.6°** |
| Louellen St west × W Broad north-east | 145.5° |
| W Broad north-east × south-west | 170.9° |

And the route TURNS here: `louellen_st_west` and `w_broad_st_northeast` are both SRI
`00000518__`, while the south-west leg is `11000654__`. So CR 518 arrives on W Broad from the
north-east and turns west onto Louellen, bending **34.5°**, and CR 654 joins at 43.6°. This is a
**skewed (oblique) junction where the numbered route turns** - not a crossroads, and not a
symmetric T.

**AASHTO** *(as cited, via [intersection geometric design summary](https://www.cedengineering.com/userfiles/C04-033%20-%20Intersection%20Geometric%20Design%20-%20US.pdf))*: roadways
should ideally cross at 90° and **not less than 75°**; **skew beyond 60° is to be avoided**.
Severely skewed intersections have restricted sight distance - worse for vans and trucks, and
worse when skewed to the left. At 43.6° this junction is well outside that guidance before any
treatment is drawn.

**Channelization principle** *(as cited)*: channelization separates and defines points of
conflict so that **"bicyclists, pedestrians and motorists are exposed to only one conflict, or
confronted with one decision, at a time."** That is the sentence this project's Louellen
treatment currently fails: a two-way lane through a 43.6° skew where the county route turns asks
a rider to resolve several conflicts at once.

**TxDOT Roadway Design Manual §18.5**, bicycle facilities at intersections
*(Verified 2026-08-15, [source](https://www.txdot.gov/manuals/des/rdw/chapter-18-bicycle-facilities-/18-5-intersections-and-crossings.html))* - the operative guidance for what to
DRAW where a two-way lane cannot continue:

- crossings should meet the roadway **at 90°** to minimise crossing distance and maximise sight
  distance;
- transitioning from a two-way to a one-way separated lane: **"the use of directional, tapered
  islands can provide positive direction for bicyclists to follow the desired transition route"**;
- **corner islands** position waiting riders **in front of** stopped motorists, improving
  visibility, and create queuing space for a two-stage turn clear of through riders;
- where width is insufficient, designers should consider **transitions to protected
  intersections, to sidewalk via bike ramps, or to shared lanes**.

> **What this means for our Louellen render, which is currently wrong.** The scenario draws
> NOTHING there and prints the refusal to the console. But "the section does not fit" is not the
> same as "no treatment" - the standards describe a specific thing to draw: a **transition**,
> channelized with directional tapered islands, squaring the crossing toward 90°, with a corner
> island for visibility and two-stage turns, ending in a marked shared-lane transition where the
> separated facility genuinely runs out. That is what a designer would put on the sheet, and it
> is what makes the corridor read as continuous-with-a-transition rather than as a gap.

### NJDOT figures actually usable here — **Verified 2026-08-14**

From the same document:

| figure | value | note |
|---|---|---|
| Unpaved driveway/street paved back from the ROW or curb line | **10 ft (3.0 m)** | §6 "Intersections and Driveways" — the concern is debris drawn onto the bicyclist's path, not markings |
| Edge line warranted when total lane width ≥ | **15 ft (4.5 m)** | |
| NJDOT minimum shoulder width on state highways | **8 ft (2.4 m)** | relevant to NJ 31, not to borough streets |
| Assumed parking lane width | **8 ft (2.4 m)** | agrees with `PARKING_STALL_DEPTH_DEFAULT_FT` |
| Width increase where trucks exceed 15% of the mix | **+1 ft (0.3 m) minimum** | **already applied** — see below |

The driveway row is the only one bearing on the driveway question, and it is about **surface**,
not striping: the markings question is settled by MUTCD §9E above.

**The truck row is already satisfied, and this is easy to get wrong** (Danny, 2026-08-14).
`TARGET_LANE_WIDTH_FT` is 11 ft = the 10 ft NACTO/AASHTO urban minimum **plus** this 1 ft. Broad
St is CR 518; E Broad and NJ 31 both carry `hgv=designated`, NJ 31 on the state truck network. So
the allowance is inside the number rather than outstanding on top of it — reading "11 ft urban
minimum" beside "+1 ft on truck routes" leads straight to proposing 12 ft lanes on a corridor
whose whole purpose is to stop being over-wide. Corollary: **narrowing any lane here to 10 ft
would be spending the truck allowance**, not trimming fat.

`CONTRAFLOW_DASH_FT` (3 ft) and `CONTRAFLOW_GAP_FT` (5 ft) in the same file are **Modelled** -
chosen to read at this drawing's scale, not taken from any document. See §7.

`TARGET_LANE_WIDTH_FT` is the single most load-bearing number in the repo — every kerbside
treatment is measured as "what is left beside an 11 ft lane". It lives in `src/` rather than in
each site's `scenarios.py` precisely because four copies is how nothing ends up enforcing it.

---

## 5. Mercer County / local — **Local**

| figure | value | constant | file | source |
|---|---|---|---|---|
| Transverse crosswalk depth | 6 ft | `CROSSWALK_DEPTH_FT` | `src/render/crosswalks.py` | Danny, 2026-08-02 |

> **Open item.** No Mercer County standard *sheet* has been located. A search for a
> county-specific driveway / shoulder-line marking detail turned up nothing, and county road
> markings generally defer to the MUTCD. If a county standard detail exists, it should replace
> the national defaults in §2 and this table should say so. The 6 ft crosswalk depth above is
> the only county figure in the repo and it came in verbally.

---

## 6. NJDOT — data source, not a design standard

Recorded here because it is routinely mistaken for one.

- **SRI / SLD linear-referencing roadway layer** is the *alignment* source
  (`data/NJ_Roadway_Network.*`, `scripts/convert_road_network.py`). An SRI line is a
  linear-referencing reference, **not a surveyed carriageway centre** — it sits off centre and
  it bends relative to the street. Everything in `_centre_legs_on_traced_kerbs` and
  `_join_through_legs` exists because of that one fact.
- **Straight Line Diagrams** supply nominal pavement widths where no field measurement exists.
  Per-leg provenance is recorded in each site's `config.yaml` and rendered into the drawing's
  line styling (solid = field-measured, dash-dot = OSM-derived, dashed = estimate).
- **[NJDOT Roadway Design Manual (2015)](https://dot.nj.gov/transportation/eng/documents/RDM/documents/2015RoadwayDesignManual_20231226.pdf)**
  covers shoulders and markings. Consulted 2026-08-07; no driveway-specific edge-line rule was
  found in it, which is why §2 falls back to the national MUTCD.

---

## 7. Numbers that are ours, not anyone's — **Modelled**

Listed so nobody goes looking for a standard behind them.

| constant | value | file | what it really is |
|---|---|---|---|
| `MIN_MARKED_PARKING_DEPTH_FT` | 8 ft | `treatments.py` | "a standard stall or nothing" policy |
| `CORNER_HATCHING_DEFAULT_DEPTH_FT` | 6 ft | `treatments.py` | paint-only zone, sized like a modest bulbout |
| `MIN_CROSSING_ANGLE_DEG` | 30° | `crosswalks.py` | calibrated against the four sites' bimodal match data |
| `REPORT_CROSSING_SKEW_DEG` | 20° | `crosswalks.py` | reporting threshold, deliberately not a discard |
| `OPENING_TRIM_FT` | 1.5 ft | `paint.py` | cohesion at a driveway mouth, not a swept-path design |
| `THROUGH_JOIN_BLEND_FT` | 60 ft | `intersection.py` | the run over which a striper swings a centreline |
| `TRACED_SECTION_START/END_FT` | 35 / 130 ft | `intersection.py` | the window a leg's *width* is a fact about |
| `CROSSWALK_SETBACK_FT` | 8.3 ft | `model.py` | **not the statute** — see the name clash below |

> **Name clash — `CROSSWALK_SETBACK_FT` means two different things.**
>
> | file | value | meaning |
> |---|---|---|
> | `src/geometry/daylighting.py` | **25.0 ft** | R.S. 39:4-138(e) — how far from a crosswalk parking is forbidden |
> | `src/geometry/model.py` | **8.3 ft** | measured — how far beyond a cross street's kerb line a crosswalk actually sits, fitted to the 11 surveyed crossings (σ 2.4 ft, range 5.1–13.9) |
>
> Different modules, so nothing is shadowed and there is no bug today. But one grep for the
> name returns both, and only one of them is a legal figure. Renaming the measured one
> (`CROSSWALK_OFFSET_FROM_KERB_FT` or similar) would end it.

---

## What to check before trusting a row

1. Anything marked **as cited** has not been opened during this project.
2. The **line width** divergence in §2 changes design outcomes, not just appearance.
3. The **Hopewell Borough ordinance** in §1 is unread and can only make the setbacks stricter.
4. No **Mercer County standard sheet** has been found; §5 is one verbal figure.
5. `CROSSWALK_SETBACK_FT` resolves to two different numbers depending on the module — §7.
