# Making this a network renderer

**Status: in progress, 2026-08-17.** Follows `docs/network-model.md`, whose step 1 landed in
`f528281`. That document argues *why*; this one is the work breakdown and what each piece must
prove.

## The goal, stated as a property

> Every feature the surveyor recorded inside the drawn frame is either **drawn from its own traced
> geometry**, or **named in the notes as deliberately not drawn**. Nothing is silently dropped.

That is what makes a render usable as a public-action resource. A picture that shows a marked
crosswalk as bare asphalt is not a conservative simplification — it is a false statement about the
street, made to an audience deciding whether to build something.

## What is wrong now, measured

Broad & Greenwood at `--frame-scale 2.5`, 431 ft frame radius:

```
17  OSM crossings fetched, ALL of them traced ways (2-5 points, 38-77 ft long)
10  inside the frame
 4  drawn - exactly those matched to this junction's four legs
 6  dropped, 263-411 ft out, three tagged crossing:markings=zebra
```

The cause is not the fetch radius. A drawn crossing is currently rebuilt from a **leg**: match the
traced way to a leg, reduce it to `(station, skew)`, then re-derive the band from the leg's frame
(`crosswalk_axes`). Two consequences:

1. **Leg-gated.** A junction the site does not model has no legs, so its crossings are unreachable
   at any radius.
2. **Lossy even when it works.** The traced way has 2–5 points and its own length; the rebuilt band
   is a rectangle from a station, a skew angle and a reach computed off the kerbs. The survey is
   used as a *hint* to reconstruct geometry we already have.

For fidelity, a surveyed crossing should be drawn **as traced**. Derivation is for things nobody
surveyed — a proposal's stop bar, an estimated crossing on a leg with none.

## The distinction the whole design rests on

| | positioned by | who decides it exists |
|---|---|---|
| **surveyed** — kerbs, crossings, driveways, tactile pads, signals | its own traced geometry | the surveyor |
| **derived** — treatments, proposals, estimated crossings, stop bars | a road + a station range | this project |

Surveyed features need no leg and no road: they are already in state-plane feet. They need only to
be *in the frame*. Derived features need a datum, and that datum becomes the `Road` from step 1
rather than a `Leg`.

`cb9c8b6` already moved kerbs to this model — collected by drawing radius, drawn as traced — on the
grounds that *what a drawing contains is a question about the drawing*. This generalises that.

## Work breakdown

Sliced to be **file-disjoint**, so the three streams can run in parallel without clobbering each
other. The render-path integration is deliberately not parallelised: `export.py`, `plan_view.py`,
`scene.py` and `checks.py` are the shared core and one hand should change them.

### A — surveyed crossings, drawn as traced
*New: `src/geometry/surveyed.py`, `tests/test_surveyed_crossings.py`.*

Every OSM crossing inside the frame, as a drawable band built from its own traced way, styled from
its own `crossing:markings` (`zebra` → continental bars, `lines` → two transverse lines, absent →
not drawn as marked). Carries whether it belongs to a modelled leg, so the render can prefer the
treated version where a proposal restyles one.

Must prove: all 10 in Greenwood's 2.5× frame are produced; the 4 leg-matched ones land within a
tolerance of where the existing per-leg code puts them; a crossing with no markings tag is not
reported as marked.

### B — a coverage check that fails the build
*New: `src/geometry/coverage.py`, `tests/test_coverage.py`.*

Compare surveyed features in the frame against pieces actually drawn, and return the difference.
This is the guard that turns "we might be dropping ground truth" into a build failure, and it is
the piece that makes the rest trustworthy rather than merely better.

Must prove: it reports the 6 dropped crossings on today's code, and reports zero once A is wired in.

### C — the corridor, on roads
*Extends `src/geometry/network.py`; new `scripts/corridor_report.py`, `tests/test_corridor.py`.*

A Road that runs the length of the borough rather than stopping at one junction's legs, and the
corridor questions asked of it: stalls, driveway openings, crossings, narrowest width. Replaces the
scratch scripts whose three wrong answers are recorded in `network-model.md`.

Must prove: Broad St resolves as one road across all three modelled junctions; every count states
its own traced coverage beside it.

### D — integration (not parallel)
`export.py`, `plan_view.py`, `scene.py`, `checks.py`, then regenerate goldens and render.

## What must not happen

- **No golden regeneration by a parallel stream.** Goldens move once, in D, with the diff explained.
- **No changes to the leg frame.** That is step 4 of `network-model.md` and is not in this plan.
- **No new second definition.** If a stream needs a constant that exists, it imports it.
