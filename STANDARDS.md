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

Three things worth remembering, because all three were bugs, and the third is the first two
combined:

- **(e) has two arms.** The crosswalk arm and the side-line arm. Only the crosswalk arm was
  applied at first, so legs with no marked crossing got no setback at all.
- **The statute is about *an* intersection, not *this* one.** A leg drawn 425 ft out crosses
  Blackwell Avenue and Model Avenue too, and each gets its own (e) setback. See
  `src/geometry/cross_streets.py`.
- **Both arms, at every intersection** *(fixed 2026-08-17)*. When (e) was extended past the
  modelled junction only the **side-line** arm went with it — the same half-a-rule as the first
  bullet, run backwards. A cross street with a marked zebra across our own street got the
  setback owed to one with nothing. Measured on `broad_st_east` at Blackwell Avenue, the zone
  was `253.3–329.3` where the two surveyed crossings put it at `239.1–340.6`: **28 ft of kerb
  the drawing marked as parkable and the statute does not.**

**And the crosswalk arm binds even where nothing is painted**, which is the part that is easy to
get backwards. **N.J.S.A. 39:1-1** defines a crosswalk as one *"either marked or unmarked
existing at each approach of every roadway intersection"* (quoted in full in §2). So every cross
street contributes two of them whether or not a surveyor traced any paint. Making the setback
conditional on a traced zebra would report **the survey's coverage as if it were the law's
reach** — and OSM has crossings traced at Blackwell and none at Model Avenue, two intersections
130 ft apart on the same street.

Positions come from the surveyed way where one is traced and from the measured
`CROSSWALK_OFFSET_FROM_KERB_FT` (8.3 ft beyond the kerb line, §7) where none is;
`NoParkingZone.reason` says which, so an estimated position is never presented as a survey. Since
a crosswalk sits *outside* the side line by that offset, the crosswalk arm is the binding one at
essentially every intersection — that is not a surprise to be tuned away, it is what the statute
says, and it is already how the junction end behaves.

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

### A DRIVEWAY IS NOT AN INTERSECTION, and which one a gap is decides the markings — **Verified 2026-08-17**

This is the definition the whole of the rest of this section hangs on, and the repo had none:
every gap in a kerb was one kind of thing. It is not our judgement to make — both the manual
and the statute define it, and they agree.

**MUTCD 11th ed. §1C.02(113), "Intersection"**
([source](https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/part1.pdf), read 2026-08-17):

> (a) The area embraced within the prolongation or connection of the lateral curb lines, or if
> none, the lateral boundary lines of the roadways of two highways that join one another at, or
> approximately at, right angles, or the area within which vehicles traveling on different
> highways that join at any other angle might come into conflict.
>
> (b) **The junction of an alley, driveway, or site roadway with a public roadway or highway
> shall not constitute an intersection, unless the public roadway or highway at said junction is
> controlled by a traffic control device.**

§1C.02(63) defines a **Driveway** as *"an access from a roadway to a building, site, or abutting
property."* And §9E.04(01) sends you straight back here — *"The definition of an 'Intersection'
in Section 1C.02 contains information to determine if a driveway can be considered an
intersection"* — so the manual itself treats this as the load-bearing test, not a technicality.

**N.J.S.A. 39:1-1** agrees, in the state whose law governs the parking rules in §1
([source](https://codes.findlaw.com/nj/title-39-motor-vehicles-and-traffic-regulation/nj-st-sect-39-1-1/),
read 2026-08-17):

> **Intersection** — "the area embraced within the prolongation of the lateral curb lines or, if
> none, the lateral boundary lines of two or more **highways** which join one another at an
> angle, whether or not one such highway crosses another."
>
> **Private road or driveway** — "every road or driveway **not open to the use of the public**
> for purposes of vehicular travel."

A driveway is not a highway, so a driveway junction is not an intersection under either
authority. Encoded as `OpeningSource.is_an_intersection` in
[`src/geometry/kerbs.py`](src/geometry/kerbs.py) — one property, read by every marking rule
below, so a new source of kerb opening (a rail crossing, a bus pad) answers the question once
rather than in each rule.

**§1C.02(113)(b)'s list is OSM's `service=*` list** — *"an alley, driveway, or site roadway"*
against `service=alley`, `service=driveway`, `service=parking_aisle`. So the negative arm is read
straight off the tag rather than restated in our own words, and each value is its own
`OpeningSource` so the citation names what the mapper actually wrote. A parking aisle is a site
roadway and opens a kerb; until 2026-08-17 `IntersectionModel` filtered aisles out before they
could, on reasoning that holds only for the 6 of the borough's 20 aisles that sit inside a mapped
lot.

**And the junction the drawing is centred on is an intersection under (a)** — two highways joining
— which sounds too obvious to write down and was the one intersection with no rule at all. Its
mouth is now `OpeningSource.JUNCTION`, running from the junction node to the corner return's
tangent point (`src/geometry/model/corners.py:junction_mouth_ft`). It is the only opening in this
project with **no OSM object behind it**, and that is a real gap in the data rather than a
shortcut: OSM maps no intersection *area* — no way or relation whose extent is the ground inside a
junction — so the corner return is the measurement, and it is the same point R.S. 39:4-138(e) is
read from in §1. Before this, that mouth was handled by hand in three `Treatment.paint` methods,
and 164 sq ft of daylight hatching was drawn inside the intersection at W Broad & Louellen with
every check passing.

> **The clause we do NOT implement, said out loud:** §1C.02(113)(b)'s *"unless … controlled by a
> traffic control device"*. A driveway with a signal or a STOP/YIELD sign on the **public
> roadway** at its mouth IS an intersection. The data to answer it is already fetched —
> `src/render/props.py:fetch_traffic_control` pulls every `highway=traffic_signals` node in the
> frame — so this is a branch away, and the reason it is still unwritten is that **no driveway at
> any of these five sites is signalised**, so the branch would never run. A rule that has never
> fired pins nothing. Write it when there is a junction to fire it on; it is a tag read in
> `kerb_openings_from_model`, not a new mechanism.

### Edge lines at driveways and intersections — **Verified 2026-08-17**

**MUTCD 11th ed. §3B.11, "Application of Pavement Markings through Intersections or
Interchanges"** ([source](https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/part3.pdf)):

| ¶ | force | wording |
|---|---|---|
| 07 | **Standard** | "Solid lines **shall not** be used to extend edge lines into or through intersections **except through that part of an intersection with no intersecting approach (such as at the far side of a T-intersection)**." |
| 08 | Guidance | "Edge line markings **should be discontinued** across intersecting approaches at intersections or interchanges." |
| 09 | Guidance | "**Driveways that do not meet the definition of an intersection** (see Section 1C.02) **should have edge line markings maintained** across the intersecting approach of the driveway." |
| 10 | Option | "Dotted edge line extensions **may** be placed through intersections." |

§3B.09(07) says the same thing from the other end: *"Edge line markings should not be continued
through intersections, except for … dotted edge line extensions … or through that part of an
intersection with no intersecting approach."*

**Two rules, opposite directions, one definition between them.** Encoded as **one row per
marking** in `markings.AT_AN_OPENING` ([`src/geometry/markings.py`](src/geometry/markings.py)) —
what it does at a driveway, what it does at an intersecting approach, and the clause each cell
came from — consumed by `KerbOpenings.against()` in
[`src/geometry/paint/`](src/geometry/paint/).

> **Why a table and not two sets.** Until 2026-08-17 this was `LINES_UNBROKEN_BY_A_DRIVEWAY` plus
> `ZONE_BOUNDARY_LINES`, and a third rule — whether a marking carries a **dotted extension** —
> was not declared anywhere at all: it was whether a treatment remembered to call
> `emit_across_opening`, which only the bikeways did. Four places, three of them the absence of a
> declaration. A marking now has a row or `markings.require_every_kind` refuses to import.

The ¶07 exception — the far side of a T — needs no code of its own, and that is worth knowing
because it looks like it should. An opening is only ever made on the kerb the approach actually
leaves on: `src/geometry/cross_streets.py` reads which side of our centreline the cross street's
own vertices fall on, and `junction_mouth_ft` returns **None** where the two legs are one street
running through (`through_street_sides`), because a kerb with no corner in it has no mouth. So a
T's far kerb is never opened by either, and the ¶07 exception falls out of the geometry rather
than being written as a rule.

> **A citation error this file exists to catch.** Until 2026-08-17 the row above cited **MUTCD
> §3B.07**, from a 2026-08-07 reading. In the 11th edition §3B.07 is *White Lane Line Markings
> for Non-Continuing Lanes* and says nothing about driveways; the rule is §3B.11(08)–(09), with
> §3B.09(07) as its counterpart. The substance was right and the pointer was wrong, which is the
> harder kind to notice — nothing downstream misbehaves, and the next person to open the manual
> finds lane drops.

> **And a claim that had gone stale.** The same row used to say "the intersection half was
> already correct — the line stops at Blackwell Avenue's mouth and carries across the driveways."
> That was true when written and false by 2026-08-17: once `cross_streets.py` began producing
> `KerbOpening`s, an intersecting approach became indistinguishable from a driveway to
> `KerbOpenings.against()`, so the parking edge line was carried **across Blackwell Avenue** on
> exactly the Guidance that says it should be discontinued there. A rule stated in prose and
> enforced nowhere lasted one refactor.

### Crosswalks at an intersection nobody signalized — **Verified 2026-08-17**

Broad St crosses Blackwell Avenue, Model Avenue, Seminary Avenue and more inside the drawn
frame. None of them is this site's modelled junction; all of them are intersections, and three
of the crossings there are traced in OSM as a zebra. The rules that apply do not care which
junction a drawing is centred on.

**N.J.S.A. 39:1-1, "Crosswalk"** — and this is the sentence that matters most:

> "that part of a highway at an intersection, **either marked or unmarked existing at each
> approach of every roadway intersection**, included within the connections of the lateral lines
> of the sidewalks on opposite sides of the highway measured from the curbs or, in the absence of
> curbs, from the edges of the shoulder, or, if none, from the edges of the roadway"

So **a crosswalk exists at every approach of every intersection whether or not anyone painted
one**, and R.S. 39:4-138(e)'s "within 25 feet of the nearest crosswalk" (§1 above) therefore
binds at Blackwell exactly as it binds at Greenwood. See
[`src/geometry/daylighting.py`](src/geometry/daylighting.py); the crossing's position comes from
the surveyed way where one is traced and from `CROSSWALK_OFFSET_FROM_KERB_FT` where none is.

**A SECOND CONSUMER of that same unmarked crosswalk, added 2026-08-19:**
[`paint.junction_mouths_ft`](src/geometry/paint/) ends this junction's own mouth at the leg's
crosswalk *painted or not*, on the strength of the sentence above — the mouth is where the
intersection stops on a kerb, and the statute says the crosswalk is there either way. It matters
because the alternative was the corner fillet's tangent point, which is `R/tan(theta/2)` back
from the corner and so **diverges as a corner sharpens**: 1.0 R at a square junction, 2.5 R at
W Broad & Louellen's 44° Y. That held the hatching 63.7 ft out on a kerb whose crossing reaches
32.1 ft and whose statutory no-parking zone runs 0–85.7 ft.

**MUTCD 11th ed. §1C.02(50)** draws the same two cases: (a) the unmarked connection of the
sidewalk lines at an intersection, (b) "any portion of a roadway at an intersection **or
elsewhere** distinctly indicated as a pedestrian crossing by pavement marking lines."

**MUTCD 11th ed. §3C.02, application of crosswalk markings**, for whether to draw one that is
not there today:

| ¶ | force | wording |
|---|---|---|
| 01 | Guidance | "At locations controlled by traffic control signals, crosswalk markings **should** be installed." |
| 03 | Guidance | "On approaches controlled by STOP or YIELD signs, crosswalk markings **should** be installed where engineering judgment indicates they are needed…" |
| 04 | Guidance | "**At uncontrolled approaches, an engineering study should be performed before a marked crosswalk is installed**," against fourteen listed criteria (lanes, median, ADT, speed, sight lines, transit stops, …). |
| 05 | **Standard** | "Crosswalk markings **shall** be provided at legally established crosswalks at non-intersection locations." |

> **This project does not propose new crosswalks at uncontrolled approaches, and ¶04 is why.**
> The study it asks for needs pedestrian counts and ADT that this repo does not hold. What the
> repo does instead is draw every crossing the surveyor **did** record, keep paint off it, and
> apply the statutory setback around it. Marking a new one is a recommendation with a study
> behind it, not a geometry change.

§3C.01(03) is worth keeping beside that: *"At non-intersection locations, crosswalk markings
legally establish the crosswalk."* At an intersection the crosswalk is already there in law and
the paint only makes it visible; mid-block, the paint is what creates it. That is the whole
difference between the two, and it is why an unmarked intersection approach still carries a
setback and an unmarked mid-block stretch does not.

### Other MUTCD figures — *as cited*

| figure | value | constant | file |
|---|---|---|---|
| Normal walking speed, pedestrian clearance | 3.5 ft/s | `MUTCD_WALKING_SPEED_FT_S` | `src/metrics.py` |
| Slower walker (the person a treatment is usually for) | 3.0 ft/s | `SLOW_WALKING_SPEED_FT_S` | `src/metrics.py` |
| Stop line to near edge of crosswalk | 4.0 ft | `STOP_BAR_TO_CROSSWALK_GAP_FT` | `src/render/crosswalks.py` |
| Longitudinal line width | 4–6 in | — | see the caveat below |
| Dotted lane extension across a conflict area | 2 ft mark / 2 ft gap | `DOTTED_MARK_FT`, `DOTTED_GAP_FT` | `src/geometry/paint/` |
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
| Bike lane design width (and the width to design to) | 5 ft | `AASHTO_MIN_BIKE_LANE_FT` | `src/geometry/treatments/` |
| Bike lane hard floor, no curb face | 4 ft | `MIN_BIKE_LANE_FT` | `src/geometry/treatments/` |
| Parallel parking lane depth | 8 ft | `PARKING_STALL_DEPTH_DEFAULT_FT` | `src/geometry/treatments/` |
| Parallel parking stall length | 22 ft | `PARKING_STALL_LENGTH_DEFAULT_FT` | `src/geometry/treatments/` |
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
| Minimum pedestrian refuge island width | 6 ft | `NACTO_MIN_REFUGE_ISLAND_WIDTH_FT` | `src/geometry/treatments/` |
| Typical low-cost paint buffer / shoulder stripe | 5 ft | `LANE_NARROWING_DEFAULT_STRIPE_FT` | `src/geometry/treatments/` |
| Urban minimum travel lane (with AASHTO) | 10 ft | — | see `TARGET_LANE_WIDTH_FT` below |
| …**plus** NJDOT's truck-route allowance (§6) | **11 ft** | `TARGET_LANE_WIDTH_FT` | `src/geometry/treatments/` |
| Crosswalk visibility ranking | continental > ladder > transverse | — | `src/render/crosswalks.py` |

### Two-way (bidirectional) bikeway — **Verified 2026-08-18**

Source: NACTO *Urban Bikeway Design Guide*, Third Edition (Island Press), *Designing Protected
Bike Lanes* — "Contraflow and Two-Way Protected Bike Lanes". Text supplied by Danny 2026-08-18
after nacto.org returned 403 to automated fetch. **This section replaces four figures that had been
written from memory since 2026-08-14, and TWO OF THEM WERE WRONG.**

| figure | NACTO | what this project had | constant |
|---|---|---|---|
| Two-way width, should be at least | **13 ft** | 12 ft "desirable" | `TWO_WAY_BIKE_LANE_WIDTH_FT` |
| Two-way width, absolute minimum | **8 ft**, "avoided except on short street segments" | 10 ft "minimum" | `MIN_TWO_WAY_BIKE_LANE_FT` |
| Buffer, parking-protected (door swing) | 3 ft | 3 ft ✓ | `TWO_WAY_BIKE_LANE_BUFFER_FT` |
| General travel lane, ≤35 mph, supports trucks | **10 ft** | 10 ft floor ✓ | `MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT` |

**THE CORRIDOR'S 10 FT LANE HAS NO STANDING IN THIS GUIDE.** It is neither the ≥13 ft NACTO asks
for nor the 8 ft absolute floor; it was chosen to free parking and recorded as "NACTO's MINIMUM",
which it is not. The 8 ft constrained rung IS the absolute minimum and NACTO says to avoid it
except on short segments - W Broad & Louellen is a short pinch, so using it there is defensible,
but it has to be described as the absolute minimum being spent, not as a NACTO width.

**THE LADDER SPENDS THE LANE'S WIDTH AND NEVER THE BUFFER — reversed 2026-08-19 on the client's
instruction: "the bike lane should stay protected no matter what", 5 ft per direction where the
street allows it and 4 ft where it does not.** There is no unbuffered rung, so every approach that
carries the facility carries flex posts. NACTO requires a solid white line along both buffer edges
and diagonals in any buffer of 3 ft or more, which every rung now has.

The rung below the standard one is reached at `w_broad_st_northeast`, where 10 ft + 3 ft leaves
9.09 ft travel lanes, under NACTO's 10 ft truck floor. It takes 8 ft + 3 ft instead — **the absolute
minimum being spent, not a NACTO width** — which leaves 10.09 ft travel lanes, 0.91 ft under the
11 ft target and still over the floor.

**WHICH OTHER APPROACHES REACH IT DEPENDS ON HOW MUCH STREET THE SHEET SHOWS — measured
2026-08-21.** A section is sized on the narrowest half-width anywhere along the leg that is drawn,
and `HOPEWELL_FRAME_SCALE` decides how far that is, so a longer sheet can reach a pinch the short
one never shows. `w_broad_st_southwest` has one 318 ft out:

| sheet | leg drawn | `w_broad_st_northeast` | `w_broad_st_southwest` |
|---|---|---|---|
| 1× | 130 ft | 8 + 3 | 10 + 3 |
| 2.5× | 325 ft | 8 + 3 | **8 + 3** — the pinch is inside the frame |

This is deliberate, on the client's instruction of 2026-08-20 — *"We should be apply all treatments
to whatever's rendered. Roads are a network"* — so a treatment applies to the street in the drawing and the
question of what the street can hold is asked over exactly the street the reader is looking at. What
does NOT move with the sheet is the facility's EXTENT or its protection — every rung keeps the full
3 ft buffer, and `BikewayReachesTheEndOfItsKerb` is fatal if a bikeway stops short of its kerb.

**AND THE 10 FT RUNG'S PARKING JUSTIFICATION DOES NOT SURVIVE THE LONGER SHEET — measured
2026-08-21.** The 10 ft width was chosen over 12 ft to leave a parkable far kerb on
`broad_st_east`, and over that leg's first 170 ft it does: 43.26 ft between traced kerbs, 7.44 ft
spare, above `MIN_USABLE_STALL_FT` (7 ft). Over the whole 425 ft drawn at 2.5× the same leg
measures 39.16 ft and leaves 3.34 ft, so it is hatched rather than parked on that sheet whichever
rung it takes (12 ft would leave 1.34 ft). `state.notes` says which: `MarkedParking(depth_ft=7.44)`
at 1×, `LaneNarrowing(stripe_width_ft=9.56)` at 2.5×. The parking this corridor actually keeps is
`broad_st_west`'s, 8 ft deep, which the section never threatened. **So 12 ft — 1 ft off NACTO's
ask — is open again on any sheet longer than 170 ft, and the 10 ft rung should not be defended on
parking grounds without restating the span.**

**The ordering this replaced put an unbuffered 10 ft rung ahead of the constrained 8 ft one**, on
the earlier instruction that 5 ft per direction was the requirement. What it produced at W Broad &
Louellen was 5 ft per direction and NO PROTECTION on both W Broad approaches — a painted lane
beside live traffic at the one junction on the corridor where the street is tightest.

**Travel lanes: NACTO says 10 ft supports trucks at ≤35 mph** and asks that motor lanes be
narrowed to widen the bikeway. Broad St is posted 25. This is evidence AGAINST holding 11 ft, and
it is recorded rather than acted on: 11 ft is Danny's standing requirement for Broad St as a rural
arterial with truck traffic, and NJDOT §6's truck-route allowance is the local basis. A local
authority may be stricter than NACTO; it may not be stricter and pretend NACTO required it.

#### Markings a two-way bikeway needs BESIDES the lane — **Verified 2026-08-18**

| marking | rule | drawn? |
|---|---|---|
| **Crossbikes** through **all** intersections **and driveways** - alleys, minor and major cross-streets alike | required; the lane *"must continue through intersections and driveways"* and cannot merge into traffic | **yes** - `corridor_paint.opening_spans`, dotted per `markings.AT_AN_OPENING` |
| Dotted **yellow** centreline along the lane **and in the crossbikes** | required | **yes** - `contraflow_centreline`, carried across every opening |
| BIKE LANE symbol / word marking (MUTCD Fig 9E-1) | after every driveway and intersection, and at least every 500 ft | **yes** - `symbol_stations`, both rules |
| Green surfacing | optional full length; or 20-50 ft approaching/departing each intersection and driveway | **yes** - `green_extension_spans`, 20 ft |
| Solid white line along **both** buffer edges | required for street-level lanes | yes |
| Diagonals or chevrons in the buffer | **required** at ≥3 ft buffer; diagonals slant away from adjacent traffic's direction | partly |
| Warning signage for two-way bike traffic (modified MUTCD W6-3) | at any T intersection or major driveway | n/a (signs) |

**BUFFER MARKING SPACING IS LEFT ALONE, DELIBERATELY.** The guide's sentence gives it as "10 ft
(2 m)", and those two figures disagree - 2 m is 6.6 ft - so one of them is a typo and it is not
knowable from this text which. Our hatching is at 2 ft. Danny's read is that the 10 ft figure
describes something other than stroke pitch, and on an internally inconsistent sentence that is the
safer conclusion: nothing changes until the printed guide is checked. Recorded rather than dropped,
because a 5x discrepancy in a required marking is worth someone's eyes.

**THE CROSSBIKE ROW IS A DEFECT IN OUR DRAWING, not a missing nicety.** `081e990` broke the lane at
all 33 crossings and driveway mouths, on the reasoning that a crossing outranks what runs along the
kerb. That is right for a pedestrian crosswalk and WRONG for a driveway or side street: NACTO
requires the bikeway to carry through, marked as a crossbike with the dotted yellow centreline
continuing. A drawing that shows the lane stopping at each mouth depicts a facility NACTO
explicitly rejects - riders merging into traffic - and it understates the design's own continuity.

#### Turning riders out of a two-way lane — **Verified 2026-08-18**

NACTO: bidirectional lanes *"introduce additional conflict points at intersections and driveways"*
and additional measures to separate vehicular turns from the bikeway are often necessary, because
other street users are unlikely to expect contraflow bike traffic. Named tools: signal phase
separation, corner islands, turn wedges, medians to slow turning vehicles, and visibility zones at
turn conflicts. **All three of Broad St's modelled junctions are signalized** (Greenwood, E Broad &
Princeton, and Louellen - corrected 2026-08-18), so phase separation is available at each.

#### How closely a kerbside marking follows the kerb — **Modelled, checked against cited tapers, 2026-08-24**

| figure | value | constant | where |
|---|---|---|---|
| Steepest lateral shift a marking will follow a traced kerb through | 1:10 | `MAX_KERB_FOLLOW_TAPER` | `src/geometry/model/` |
| MUTCD shifting taper at 25 mph, for comparison | 1:5.2 | — | *as cited* |
| NACTO lateral shift floor, bidirectional bikeway, for comparison | 1:5 | — | *as cited* |

**Modelled, and now with two cited figures to hold it against.** No guide consulted gives a taper
for tracking a kerb that *wanders*, so the value is still ours. What it is calibrated on: across the
three corridor junctions, the legs where the traced kerb records the street genuinely bending move
at 1:6 or gentler, and the two whose tracing takes in a corner flare kink at 1:2. At 1:10 a marking
gives up a mean 0.11–0.28 ft of the drift on the legs that drift and refuses up to 12.2 ft of the
flare on `broad_st_east`.

**Why it now needs citing at all.** `corridor_paint._build_run` places the whole two-way bikeway
section against this profile, and the travel-lane divider hangs off it — so the rate is no longer a
cosmetic choice about paint but a lateral shift a driver is steered through, and it has to answer to
the published tapers. MUTCD's merging taper is `L = W·S²/60` below 45 mph and a *shifting* taper is
about half of it; at Broad St's posted 25 mph (recorded below) that is 1:5.2. NACTO's floor for a
bidirectional bikeway's lateral shift is 1:5. 1:10 is roughly twice as gentle as either, so a figure
chosen as a tracing filter turns out to be conservative as a design taper too, and no new constant
is needed. Both citations are marked *as cited* — the values are from the guides' rules as recorded
here, not re-read out of the documents for this row.

**Being 2x conservative costs no parking, measured.** The obvious objection to 1:10 is that it
leaves room on the table when the guides permit 1:5 — it does not. Swept through the corridor's own
pipeline, 1:10 → 1:5 moves the far kerb from 1,694 to 1,714 ft of room, 72 to 73 stalls of room, and
**45 to 45 stalls actually drawn**; removing the limit entirely is worth one drawn stall. The
mechanism is that the taper costs 0.36 ft of lateral placement on average, and 0.36 ft cannot move a
`MIN_USABLE_STALL_FT` threshold of 7.0 except at the margin. What binds instead is street width
through `divided_lane_width_ft` — on a run too narrow for two 11 ft lanes plus the section, the
travel way takes everything and the kerb gets nothing at the pinch by construction. Do not spend the
taper's conservatism looking for stalls; there are none there.

**The speed is an input, and this constant hides that.** `S` enters as 1/S², so a street posted 45
mph wants 1:17 and this constant would be too steep for it. Any corridor materially faster than
Broad St needs the rate derived from its own posted speed rather than read from
`MAX_KERB_FOLLOW_TAPER`.

**Why it is a rate and not an amount.** The first version of this rule compared the kerb's TOTAL
swing over a leg against a 6 ft threshold, which made the answer depend on leg length — and leg
length is set by the render frame. The same `w_broad_st_southwest` kerb swings 5.4 ft over the 130 ft
leg of a 1× sheet and 9.0 ft over the 325 ft leg of a 2.5× one, so the bikeway followed its kerb on
one drawing and stood up to 8.4 ft clear of it on the other. A rate limit cannot do that: it is a
local property of the kerb. `tests/test_leg_frame.py` pins the invariance directly.

### WHICH KERB a two-way bikeway belongs on — **NOT ESTABLISHED, 2026-08-18**

The `CORRIDOR_SIDE` decision (south) was made on one count - side streets cutting each kerb, 10
north against 7 south - and never against a published guide. Asked for the accepted practice, here
is what an hour of looking actually established.

**FHWA, *Separated Bike Lane Planning and Design Guide* (2015) — OPENED, and it does NOT answer
this.** [Page 3](https://www.fhwa.dot.gov/environment/bicycle_pedestrian/publications/separated_bikelane_pdg/page03.cfm),
read 2026-08-18. It carries no side-of-street guidance, no driveway-conflict minimisation rule and
nothing on contraflow riders being unexpected to turning drivers. Its own Table 1 defers
two-way/contraflow facilities to NACTO's Urban Bikeway Design Guide, and the text says the practice
is still evolving. **This is a verified negative and worth keeping:** the obvious federal source
does not settle it, so nobody should cite it as though it does.

**NACTO — NOT OPENED. 403 to automated fetch.** A web search reports NACTO as saying a
bidirectional lane may be safer where one side has significantly fewer intersections and
driveways, that bidirectional lanes add conflict points at intersections and driveways needing
extra measures to separate turns from the bikeway, and a desirable 30 ft no-parking area either
side of a crossing. **None of that has been read at the source**, so none of it may be quoted to
the borough or to a county engineer. It is recorded here as a lead, not a citation, exactly as the
width table below is.

**What this means for the corridor decision.** Measured on OSM: the north kerb carries 19 driveway
mouths and the south 26, so the lane's conflict count is 29 breaks north against 36 south, while
the parking mirrors it - 26 stalls kept with a north-side lane, 39 with a south-side one, because
parking lands on whichever kerb the lane does not take. The trade is real and this project cannot
currently point to a standard that resolves it. `src/geometry/corridor_paint.py` measures both and
refuses to pick.

**To close this**, someone has to open the Urban Bikeway Design Guide - a copy, not a search
result - and either confirm the fewer-conflicts rule or find that it too is silent. Until then the
honest statement to the borough is that the side choice is a local trade-off between conflict
count and parking, not a standards requirement.

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

| marking | at a driveway | at an intersecting approach | authority |
|---|---|---|---|
| edge lines | break, continue as a dotted extension | same | §9E.06(15) Guidance; §9E.04(02) Option |
| yellow contraflow centre stripe | **carries through unbroken** | same | NACTO dotted yellow centreline; §9E.06(15) |
| green surface | breaks and resumes as the same dashes | same | our choice — **Modelled**; colour is not specified |
| green surface, across the JUNCTION'S OWN box | n/a — a driveway has no box | dotted marks ruled between the two lane ends | §9E.03(07) Standard; §9E.06(15) Guidance; NACTO crossbike |

The last row is the one added on 2026-08-18, and it is not a cell in `AT_AN_OPENING` like the
others — see "The lane extension ACROSS the junction" below for why it could not be.

Every one of those cells is now a row in `markings.AT_AN_OPENING` rather than a paragraph here
and a call site there — see §2. The contraflow stripe is the one whose cell changed on 2026-08-17:
it reads **CARRIED**, not dotted, because it is *already* a broken line and its own cadence is the
dotted pattern the standard asks for. It used to be cut at each entrance and the part inside
re-laid as an exact complement, which is this row written out as two calls that had to agree.

The centre stripe used to stop dead at each driveway - 22 dashes on a kerb with two of them
against 30 on a kerb with none - while the edge lines continued and the green carried across.
Three answers to one conflict point, and that one belonged to nobody. Fixed 2026-08-14.

> **Still missing: the BIKE LANE symbol after each driveway**, which NACTO asks for and
> §9E.04(03) permits. Nothing in this repo draws a pavement word or bike symbol at all, so it is
> a new marking rather than a parameter - see the "new marking touches six places" checklist in
> README.md.

**The lane extension ACROSS the junction — added 2026-08-18.** §9E.06(15) asks for it in the same
sentence as the driveway case, and until now this section was honest about the driveway column and
silent about the other one — for a geometric reason, not a standards one. A lane is built leg by
leg, so at the junction the drawing is centred on there is no single marking spanning the mouth for
an extension to be the continuation OF; each leg's green ends at its own corner return. That is why
it could not be a row in `AT_AN_OPENING`: the dash machinery clips a *parent* marking to the
opening, and here there was no parent. `PaintContext._dash_spans_along` is explicit about refusing
the case — "a lane that simply ENDS at an opening has nothing to extend" — and that guard is right
and unchanged. What was missing was the marking it should have been extending.

`treatments.ExtendBikeLaneThroughJunction` now builds it, applied by `CorridorFacility` because it
belongs to the route rather than to either approach:

| | before | after |
|---|---|---|
| Broad & Greenwood, north kerb | 49 ft of nothing | 10 marks, 186 sq ft; longest bare run **8.0 ft** |
| W Broad & Louellen, north kerb | 81 ft of nothing | 21 marks, 386 sq ft; longest bare run **2.0 ft** |
| E Broad & Princeton, north kerb | already continuous | **refused, and reported** — see below |

The bare runs are what a rider meets, measured down the middle of the lane by
`test_the_two_way_lane_carries_across_the_junction`. Louellen's 2.0 ft is one dotted gap: the
pattern is unbroken. Greenwood's 8.0 ft is two crosswalks being crossed — a marked crossing
outranks the lane and cuts it, exactly as it does everywhere else on the kerb.

At **E Broad & Princeton** the stem is on the far side, so the corridor's own kerb is never opened
(§3B.11(07)'s T-intersection case), the two legs' green already runs through the node overlapped,
and an extension would be a second lane laid on the join. The treatment refuses, and
`CorridorFacility` prints the refusal — a junction that silently draws nothing here is
indistinguishable from one that silently failed to draw something.

**The extension is drawn straight**, ruled between the two end cross-sections rather than curved to
follow the kerb between them. At a junction there *is* no kerb between them — that is what the
mouth means — and a crossing is drawn straight across the ground it crosses. Louellen's corridor
legs are 170.9° apart, so the chord runs about 3 ft off where a curve through the node would, well
inside the lane's own width.

> **Still missing: the rest of the crossbike.** NACTO carries the lane's dotted edge lines and its
> dotted yellow centreline through the box alongside the green; only the green is drawn. Both are
> the same construction against a different pair of offsets — `ExtendBikeLaneThroughJunction.paint`
> already has the two end cross-sections in hand. Recorded here rather than left to be inferred
> from a render that now looks continuous.

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

**The STREET goes through; the ROUTE NUMBER turns.** Those are different facts and conflating
them was an error (corrected 2026-08-15):

- **Geometrically the north-east and south-west legs are 170.9° apart** - a 9.1° bend, which is
  a straight-through movement by any practical reading. Whatever it is called, traffic runs
  through this junction.
- **Louellen St west is the skewed third leg**, 43.6° off the south-west leg. That is the skew,
  and it is a property of the geometry, independent of any naming.
- **The CR 518 designation does turn.** `louellen_st_west` and `w_broad_st_northeast` are both
  SRI `00000518__`; the south-west leg is `11000654__`, and OSM near the junction carries only
  *Louellen Street (CR 518)* and *West Broad Street (CR 654)*. So the county route follows the
  north-east leg and turns west onto Louellen, while the street continues south-west as CR 654.

**UNRESOLVED, and it should be settled on the ground rather than from a desk.** The repo's config
names the north-east leg `w_broad_st_northeast`; OSM names the CR 518 through-way *Louellen
Street*; Danny reports W Broad St signed on both sides. All three can only be reconciled by
looking at the signs. Nothing in the treatment depends on the answer - the skew and the widths
are measured, not named - but the drawing's labels do, and a plan sheet that calls a leg by the
wrong street name is the kind of error a reviewer stops reading at.

So: a **skewed (oblique) T** - a through movement with a 43.6° third leg - not a crossroads and
not a symmetric T.

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

**WHAT ACTUALLY GOVERNS HERE, in order.** Broad St is **CR 518 / CR 654 — a MERCER COUNTY
route**, in Hopewell Borough, New Jersey. The authorities are, in descending order of standing:

| | who | standing on this street |
|---|---|---|
| 1 | **MUTCD 11th ed.** | federally adopted; NJ adopts it. **Governs the markings**, and Part 9E is already cited above |
| 2 | **NJDOT** — [Roadway Design Manual (2015)](https://dot.nj.gov/transportation/eng/documents/RDM/documents/2015RoadwayDesignManual_20231226.pdf), *Bicycle Compatible Roadways* | state DOT; NJ agencies design to it. §1 above records what its bike guidance says |
| 3 | **AASHTO** — Green Book; *Guide for the Development of Bicycle Facilities* (2012) | **NJDOT's own guidance defers to it by name** |
| 4 | **Mercer County engineering** | **owns this road.** No county standard sheet has been located — see §5, which is an open item and matters more than it looked |
| 5 | NACTO | not a jurisdiction; widely referenced, and NJDOT's wording ("or other best practices") leaves room for it |

> **A citation error worth keeping, because it is the kind this file exists to stop.** An earlier
> version of this section quoted **TxDOT's Roadway Design Manual §18.5** for the transition
> treatment. TxDOT has **no jurisdiction in New Jersey**. It was reached for because a search
> returned it and it had usable specifics - which is the same failure as writing a NACTO figure
> from memory, in a document whose entire purpose is recording what actually governs. Its
> substance (directional tapered islands, corner islands, squaring the crossing) is ordinary
> practice and probably right, but it is **not authority here** and must not be cited as such.
> The equivalent NJ/AASHTO provisions have NOT yet been read.

#### NJDOT Roadway Design Manual §6, at-grade intersections — **Verified 2026-08-15**

Read, from the [2015 manual](https://dot.nj.gov/transportation/eng/documents/RDM/documents/2015RoadwayDesignManual_20231226.pdf)
(§6.2 alignment, §6.3.2 bicycle sight distance). **This is the governing document for Louellen
and it is far more specific than anything cited before it:**

> "Roads intersecting at acute angles require extensive turning roadway areas. **Intersection
> angles less than 75 degrees normally warrant realignment closer to 90 degrees.** At skewed
> intersections where the approach leg to the left intersects the driver's approach leg at an
> angle of less than 75 degrees, **a right-turn-on-red (RTOR) prohibition is desirable.** When
> realignment cannot be obtained, extensive application of appropriate signing and signal
> control is recommended. **Roundabouts should also be considered** at locations where
> intersection skew is severe and realignment cannot be obtained."

> "**Older drivers in particular have difficulty with skewed intersections**, due to restricted
> range of motion and diminished reaction time. Refer to the FHWA *Handbook for Designing
> Roadways for the Aging Population*."

**Applied to W Broad & Louellen at 43.6°**, NJDOT's own manual gives four things in order:

| | treatment | why it lands here |
|---|---|---|
| 1 | **Realign toward 90°** | "normally warrant[ed]" below 75°, and we are at 43.6° |
| 2 | **Prohibit right-turn-on-red** | "desirable" at exactly this geometry - and the 2023-11-08 injury here was a driver **turning right** from W Broad onto Louellen into a cyclist in the crosswalk. The manual names the movement that hurt someone |
| 3 | **Signing and signal control** | where realignment cannot be obtained |
| 4 | **Consider a roundabout** | explicitly, where skew is severe and realignment is unavailable |

None of that is a bike-lane treatment, and that is the finding: **NJDOT's answer to this junction
is to fix the junction, not to stripe through it.** The two-way lane's refusal here is a symptom
of a geometry the state's own manual says warrants realignment.

**§6.3.2, bicycle sight distance** — *"In general the sight distance required to see a bicycle is
no greater than that to see a vehicle... At locations where a **separated bicycle facility
crosses the roadway**, or elsewhere where cyclists may enter or cross the roadway independent of
vehicles, appropriate sight distance should be provided."* The same section restates
**R.S. 39:4-138**'s 25 ft / 50 ft setbacks as a *sight distance* measure and confirms curb
extensions may reduce them - which is what `src/geometry/daylighting.py` already encodes, now
with the state manual behind it rather than the statute alone.

#### NJDOT RDM on lane width — **Verified 2026-08-15**

Direct on `TARGET_LANE_WIDTH_FT` and on the 10 ft floor this project falls back to:

> "While lane widths of 12 feet are desirable on land service highways... **Lane widths of 11
> feet in urban areas are acceptable.** Existing lane widths of 10 feet have been provided in
> certain locations where right of way and existing development became stringent controls and
> where truck volumes were limited. However, **new or reconstructed 10 foot wide lanes would not
> be proposed today, except in traffic calming areas.**"

Two consequences, both load-bearing:

1. **11 ft is affirmatively acceptable here** - Broad St is urban land service highway. The
   target is not a NACTO borrowing this project has to defend; the state manual states it.
2. **`MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT` (10 ft) is defensible ONLY as traffic calming.** NJDOT
   would not propose a new 10 ft lane otherwise. Every scenario here IS a traffic calming
   proposal, so the exception applies - but it has to be *claimed* in the submission, not
   assumed, and a 10 ft lane drawn without that framing is outside current NJDOT practice.

And on the outside lane, which is what a bikeway changes:

> "**where it is not practical to provide a shoulder adjacent to the outside lane** (design
> exception required), **the outside lane width shall be 15 feet to accommodate bicyclists.**
> Where alternate bike access is provided, the outside lane width **should be 1 foot wider than
> the adjacent through lane width.** The designer should strive to accommodate the bicyclist and
> pedestrian on all projects."

> **This does NOT apply once a protected lane is built, and reading it as "+1 ft on top of our
> 11" was an error (corrected 2026-08-15).** The whole provision is about carrying cyclists **in
> the outside travel lane** where there is no shoulder: 15 ft is a *shared* lane wide enough for
> a car to pass a bike within it. "Where alternate bike access is provided" the requirement
> **relaxes** from 15 ft to one foot over the through lane - it is a reduction, not an addition,
> and it is conditioned on cyclists having somewhere else to be.
>
> **A protected bike lane IS that alternate access.** Riders are not in the travel lane at all,
> so the provision is satisfied by the facility rather than by widening the carriageway. And
> widening a travel lane beside a protected bike lane would be actively counterproductive: wider
> lanes carry higher speeds, Broad St is posted **25 mph its whole length through the borough**
> (OSM `maxspeed=25 mph` on all three segments, CR 518 and CR 654), and every scenario here is a
> traffic-calming proposal. Adding a foot in the name of accommodating bicyclists, on a street
> that has just been given a separated bikeway, gets the intent exactly backwards.
>
> So 11 ft stands, and the only +1 ft inside it is the truck allowance in §6.

> **Still unread:** AASHTO's *Guide for the Development of Bicycle Facilities* on separated-lane
> transitions, and whatever Mercer County holds. The county owns CR 518, but NJDOT §6 above is
> the design authority NJ agencies work to, so the county call is no longer a precondition for
> drawing Louellen - it is a check on the result.

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
| `OPENING_TRIM_FT` | 1.5 ft | `paint/openings.py` | cohesion at a driveway mouth, not a swept-path design |
| `OPENING_PAST_THE_KERB_FT` | 0.15 ft | `paint/openings.py` | not a design figure — the margin that stops an opening's cut from *being* a description of the kerb |
| `THROUGH_JOIN_BLEND_FT` | 60 ft | `intersection.py` | the run over which a striper swings a centreline |
| `TRACED_SECTION_START/END_FT` | 35 / 130 ft | `intersection.py` | the window a leg's *width* is a fact about |
| `CROSSWALK_OFFSET_FROM_KERB_FT` | 8.3 ft | `model/context.py` | **not the statute** — measured, see below |
| `MAX_CROSSWALK_FROM_MOUTH_FT` | 25 ft | `cross_streets.py` | how far outside a mouth a traced crossing is still that junction's |

> **Name clash — resolved 2026-08-17.** `CROSSWALK_SETBACK_FT` used to mean two things:
>
> | file | value | meaning |
> |---|---|---|
> | `src/geometry/daylighting.py` | **25.0 ft** | R.S. 39:4-138(e) — how far from a crosswalk parking is forbidden |
> | `src/geometry/model/context.py` | **8.3 ft** | measured — how far beyond a cross street's kerb line a crosswalk actually sits, fitted to the 11 surveyed crossings (σ 2.4 ft, range 5.1–13.9) |
>
> Different modules, so nothing was ever shadowed. It stopped being merely untidy when
> `cross_streets.py` began placing the unmarked crosswalk at the measured 8.3 ft and handing it
> to `daylighting.py` to measure the statutory 25 ft from — two constants of the same name one
> call apart. The measured one is now `CROSSWALK_OFFSET_FROM_KERB_FT`; only the legal figure
> keeps the old name.

---

## What to check before trusting a row

1. Anything marked **as cited** has not been opened during this project.
2. The **line width** divergence in §2 changes design outcomes, not just appearance.
3. The **Hopewell Borough ordinance** in §1 is unread and can only make the setbacks stricter.
4. No **Mercer County standard sheet** has been found; §5 is one verbal figure.
5. The **traffic-control-device arm** of MUTCD 1C.02(113)(b) is not implemented — a driveway
   whose public roadway carries a signal or a STOP/YIELD sign at the mouth IS an intersection,
   and this project would draw it as a driveway. The OSM data to answer it is already fetched
   (`fetch_traffic_control`); what is missing is a signalised driveway at any of these five sites
   to exercise the branch on, and an unexercised rule pins nothing. See §2.
6. **New crosswalks at uncontrolled approaches are not proposed** — MUTCD 3C.02(04) wants an
   engineering study this repo has no counts for. Existing ones are drawn and deferred to; §2.
