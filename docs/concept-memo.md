# Broad Street and Princeton Avenue — bikeway and traffic-calming concept

| | |
|---|---|
| **To** | Mercer County Engineering |
| **Through** | Borough of Hopewell |
| **Date** | 25 August 2026 |
| **Re** | Concept geometry for a two-way protected bikeway on Broad Street (CR 518) and traffic calming on Princeton Avenue (CR 569) |
| **Status** | **Concept — not for construction.** Not survey-grade. See §6. |

---

## 1. What is proposed

Three things, drawn to measured geometry rather than sketched:

1. **A two-way protected bikeway on the north kerb of Broad Street**, running 3,986 ft of the
   4,524 ft between the borough lines — 88% of the corridor, in 11 runs separated by junction
   mouths and crossings.
2. **Traffic calming on Princeton Avenue**, 1,565 ft: both travel lanes held at 11 ft and every
   remaining foot of carriageway hatched, with the reason for each hatched stretch recorded.
3. **Junction treatments at five intersections** — daylighting to the R.S. 39:4-138 setbacks,
   crossing upgrades, and the bikeway's transition through each junction.

The corridor termini are **jurisdictional, not a design choice**: the drawings stop at the borough
line because nothing beyond it is Hopewell's to propose.

## 2. Jurisdiction and approval path

**Neither street is borough-owned.**

| street | route | functional class | posted | jurisdiction |
|---|---|---|---|---|
| Broad Street | CR 518 (NJDOT SRI `00000518__`) | Urban Minor Arterial | 25 mph | **Mercer County** |
| Princeton Avenue | CR 569 (NJDOT SRI `00000569__`) | Urban Minor Arterial | 25 mph | **Mercer County** |

The Borough of Hopewell cannot restripe either road on its own authority. This package is
therefore addressed to the County as the roadway authority, with the Borough as the requesting
party. Cross streets touched at the junctions (Greenwood/Columbia Ave, Louellen St, E Prospect
St) are borough local roads.

## 3. Drawings in this package

| drawing | what it shows |
|---|---|
| `output/corridor/broad_street_strip_north.png` | Broad St strip plan, stationed, with the proposed bikeway and the far kerb's parking and hatching |
| `output/corridor/broad_street_strip_daylighting.png` | Broad St with daylighting and crossing upgrades **only**, no bike facility — the do-less option |
| `output/corridor/princeton_avenue_strip_calming.png` | Princeton Ave strip plan, calming |
| `output/<junction>/phase3_before_after*.png` | Before/after plan views, five junctions |
| `output/<junction>/phase4_render_*.png` | 3D views of the same |
| `output/<junction>/geometry_*.json` | Every marking as coordinates, for import |

Strip plans are drawn in the corridor's **own frame** — station along the page, offset across it.
Every length, width and offset is true; the street's curvature is removed. For junction shape,
read the per-junction plan views.

## 4. Quantities

Order-of-magnitude, measured off the drawn geometry. Not a bid schedule.

**Broad Street bikeway**

| item | quantity |
|---|---|
| Bikeway placed | 3,986 ft of 4,524 (88%), 11 runs |
| Section — 10 ft lane + 3 ft buffer | 3,370 ft |
| Section — 8 ft lane + 3 ft buffer (constrained) | 617 ft |
| Bikeway surface | 37,903 sq ft |
| Buffer / painted separation | 11,732 sq ft |
| Flexible delineator posts | 505 |
| Bikeway edge lines | 33 runs |
| Centreline dashes | 498 |
| BIKE LANE pavement symbols | 32 |
| Green conspicuity zones at crossings | 29 |
| Openings the lane is dotted across | 19 |
| Crossings the lane is cut at | 9 |

**Broad Street south kerb** (the kerb the bikeway does not take)

| item | quantity |
|---|---|
| Marked stalls | 45 (7,878 sq ft) |
| Hatched — restricted by law or signage | 14 bands |
| Hatched — no stall fits this width or length | 41 bands |
| Total hatching | 19,540 sq ft |

Every square foot of the south kerb the travel lane does not use is either a marked stall or
hatched. The hatch sits between the stalls and the kerb; the stalls sit against the travel lane.

**Princeton Avenue:** both travel lanes held at 11 ft over 1,565 ft; no kerb on the corridor is
legally parkable, so no stalls are marked and the recovered width is hatched with its reason.

## 5. Parking — the trade-off, stated plainly

This is the cost of the proposal and it is not small.

| | stalls |
|---|---|
| Today, both kerbs, width-tested against an 11 ft travel lane | **188** (89 north, 99 south) |
| With the bikeway on the north kerb | **45**, all on the south |
| **Lost** | **143** — 89 of them the bikeway's own kerb, 54 squeezed off the south |

The alternative was measured on the same survey, not estimated:

| bikeway on | stalls kept | interruptions a rider meets |
|---|---|---|
| **north kerb** (drawn) | 45 | **28** |
| south kerb | **47** | 35 |

**The two measures disagree**, so this is a trade-off and not a calculation: the north kerb buys 7
fewer interruptions for 2 stalls. The north kerb is drawn because Broad Street's south frontage
carries more driveways, and every driveway is a conflict point a rider meets at speed. The County
may reasonably prefer the other answer; both are drawn from one survey and either can be produced.

## 6. Basis of design, and what this is not

**Sources**

- **Kerb geometry** — OpenStreetMap kerb lines traced for this project against aerial imagery.
- **Carriageway widths** — cross-checked against the NJDOT Straight Line Diagram, SRI
  `00000518__`, MP 8.000–11.000, **inventoried October 2012**.
- **Right of way** — Mercer County parcel boundaries.
- **Parking prohibitions** — R.S. 39:4-138 setbacks (crosswalk, side line, stop sign, hydrant),
  plus OSM parking restriction tags where a kerb is signed.
- Every published standard the design relies on is indexed in `STANDARDS.md` with its provenance.
  **18 rows were checked against the source during this project; 14 are cited from the repository's
  own notes and have not been independently opened** — including the AASHTO and NACTO sections that
  give the bikeway width and buffer.

**Limitations — please read before relying on any dimension**

1. **Not survey-grade.** Kerb positions are traced from imagery, not surveyed. Adequate to decide
   *whether* a section fits; not adequate to build from.
2. **Driveway coverage is approximately 29%** of parcels fronting Broad Street. The 19 openings the
   lane is dotted across are the ones in the data, not all of them. A driveway inventory is needed.
3. **26 ft of the corridor could not be tested** — stations 0–12 and 63–77, where one kerb is
   untraced. Shown on the strip plan in pink.
4. **The Hopewell Borough parking ordinance has not been read.** A municipal ordinance can only
   make the setbacks *stricter*, so the 45-stall figure is a **ceiling**, not a count.
5. **No traffic counts and no signal analysis.** No new crosswalk is proposed anywhere:
   MUTCD 3C.02(04) wants an engineering study this work has no counts for. Existing crossings are
   drawn and deferred to. Signal phasing at the signalised approaches has not been examined.
6. **No utility, drainage or structural review.** Inlets, castings and utility covers falling
   inside the buffer or bikeway have not been located.
7. **Sight distance, turning templates and truck off-tracking have not been checked.** Broad Street
   is a truck route.

## 7. To advance this

In the order the answers are needed:

1. **County concurrence in principle** on the corridor concept, and on which kerb carries the lane.
2. **Topographic survey** of the corridor, to replace the traced kerb.
3. **Driveway inventory** — the 29% coverage above is the binding data gap for the bikeway.
4. **Traffic counts and signal analysis**, which also unlocks the crossing question at §6.5.
5. **Borough ordinance review** against the R.S. 39:4-138 setbacks, to firm the stall count.
6. **Utility and drainage locates** within the proposed buffer.

Items 2–6 are the difference between this concept and a preliminary design. None of them changes
the question in §5, which the County and Borough can decide now on what is drawn here.

---

*Geometry, drawings and quantities in this memo are generated from a reproducible model; every
figure above can be re-derived from the source data. Contact the Borough for the underlying files.*
