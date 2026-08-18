"""THE FACILITY PAINTED ALONG A ROAD, not along a leg.

A treatment targets one kerb of one approach, and an approach is a 130-170 ft stub either side of
a node. Broad Street is 3,693 ft with 91% of both kerbs traced and no untraced gap over 50 ft, and
the three modelled junctions' Broad St legs cover 820 ft of it - 22%. So the survey is four times
ahead of what can be drawn, and a corridor drawing made from leg paint shows a protected bikeway
appearing and vanishing three times along a street it is supposed to run the length of.

WHY NOT JUST CARRY THE LEGS FURTHER. Measured, because it is the obvious cheap answer: legs were
capped at 130-170 ft because the tracing stopped there, and it no longer does. At
HOPEWELL_FRAME_SCALE=2.5 Broad & Greenwood's legs reach 325-425 ft and the junction still builds;
at 4 and 5 it DOES NOT BUILD AT ALL - `build_pavement_polygon` raises "Pavement ring is
self-intersecting", the same failure that stops nj31_wdelaware from building on today's OSM. Four
legs each carrying their own full-width envelope out of one node have envelopes that cross once
they are long enough and the street bends. The corner-fillet model is the ceiling, and it is a
ceiling at roughly a tenth of the corridor.

A road has no such ceiling, because away from a node there is no fillet: there is a centreline, two
traced kerbs, and a cross-section between them.

WHAT IS NOT REDEFINED HERE. The cross-section is `treatments.bikeways.BikeLane` and its subclass,
exactly the object the per-leg treatment uses - `offsets_from_kerb_ft()` and `section_ft` are asked
of the same class, so the corridor and the junctions cannot come to different answers about where
the stripes go. What this module owns is only WHERE the section is placed: which stations it fits
at, and what to do at the ones where it does not. The leg treatment's own `paint` method is not
reachable from here - it needs a PaintContext, crossing bands and a junction's anchors - and
duplicating it would be the second definition this project keeps paying for. It is deliberately
NOT duplicated: the junction detail in a corridor drawing comes from the junction models, and this
supplies the run between them.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon

from src.geometry.model import (Alignment, band_from_offsets, line_from_offsets,
                                place_in_measured_frame, side_facing)
from src.geometry.treatments.bikeways import TwoWayBikeLane

#: How finely the section is sampled along the corridor. The kerb is traced, so its offset moves
#: with every vertex the surveyor placed; 5 ft follows a real kerb's course without turning every
#: click into a corner in the paint. Coarser than the junction paint's STRIP_SAMPLE_FT because
#: this runs 3,693 ft rather than 130 and nothing here is measured against a corner return.
CORRIDOR_SAMPLE_FT = 5.0

#: A stretch shorter than this is not a facility, it is a gap between two of them. A rider cannot
#: use 30 ft of protected lane, and drawing one invites the reader to count it as coverage.
MIN_FACILITY_RUN_FT = 100.0


@dataclass(frozen=True)
class FacilityRun:
    """One continuous stretch where the section fits, with the paint that goes on it."""
    start_ft: float
    end_ft: float
    lane_surface: Polygon | None = None
    buffer_zone: Polygon | None = None
    edge_lines: tuple = ()
    bollards: tuple = ()

    @property
    def length_ft(self) -> float:
        return self.end_ft - self.start_ft


@dataclass(frozen=True)
class FacilityRefusal:
    """One stretch where the section does NOT fit, and the measurement that says so.

    A refusal is an output, not an error. A corridor plan that quietly stops at its hardest point
    is the plan nobody costed, so every stretch the facility cannot cross is carried out of here
    with the width that refused it and drawn as a gap in the route.
    """
    start_ft: float
    end_ft: float
    reason: str
    narrowest_ft: float | None = None

    @property
    def length_ft(self) -> float:
        return self.end_ft - self.start_ft


@dataclass
class CorridorFacilityPaint:
    """Everything the facility puts on this corridor, and everywhere it could not."""
    road: str
    side: str                       # the corridor's own left/right
    compass_side: str               # the kerb as a person standing on the street names it
    section_ft: float
    runs: list = field(default_factory=list)
    refusals: list = field(default_factory=list)
    untraced: list = field(default_factory=list)

    @property
    def placed_ft(self) -> float:
        return sum(run.length_ft for run in self.runs)

    def summary(self, corridor_length_ft: float) -> str:
        lines = [f"{self.road}: two-way protected lane on the {self.compass_side} kerb",
                 f"  placed        {self.placed_ft:8,.0f} ft of {corridor_length_ft:,.0f} "
                 f"({self.placed_ft / corridor_length_ft:.0%}) in {len(self.runs)} run(s)"]
        for refusal in self.refusals:
            narrow = "" if refusal.narrowest_ft is None else f", narrowest {refusal.narrowest_ft:.1f} ft"
            lines.append(f"  REFUSED       {refusal.start_ft:6,.0f}-{refusal.end_ft:<6,.0f} "
                         f"({refusal.length_ft:5,.0f} ft) {refusal.reason}{narrow}")
        for lo, hi in self.untraced:
            lines.append(f"  NO SURVEY     {lo:6,.0f}-{hi:<6,.0f} ({hi - lo:5,.0f} ft) "
                         f"one or both kerbs untraced - the section cannot be tested here")
        return "\n".join(lines)


def facility_side(corridor, compass: str) -> str:
    """Which of the corridor's own sides faces `compass`.

    Through the SAME side_facing every leg uses, on the corridor's centreline. A corridor runs one
    way along its whole length by construction, so unlike a leg pair there is one answer and it
    holds end to end - which is the property that stops a route swapping kerbs at a junction.
    """
    return side_facing(Alignment(corridor.centerline), compass)


def kerb_offset_ft(corridor, side: str, station_ft: float) -> float | None:
    """How far out the traced kerb sits on one side, unsigned, or None where it is not traced."""
    from src.geometry.network import _kerb_offset_at

    run = corridor.kerb_run_at(side, station_ft)
    if run is None:
        return None
    return _kerb_offset_at(corridor.centerline, run.line, side, station_ft)


def section_at(facility, near_half_ft: float, far_half_ft: float):
    """The best rung of the facility's ladder that fits this cross-section, or None.

    THE CLASS IS THE PREDICATE. TwoWayBikeLane.__post_init__ already refuses a section that
    leaves the travel lanes under NACTO's floor, with the measurement in the message; writing a
    second "does it fit" test here would be a second definition of the rule the whole facility
    turns on, and the two would drift the first time the floor changed. So a rung is tried by
    CONSTRUCTING it, and the ValueError it raises is the refusal, quoted verbatim.
    """
    refusal = None
    for rung in facility.sections:
        try:
            return TwoWayBikeLane(width_ft=rung.width_ft, buffer_ft=rung.buffer_ft,
                                  constrained=rung.constrained, near_half_ft=near_half_ft,
                                  far_half_ft=far_half_ft), None
        except ValueError as too_narrow:
            refusal = str(too_narrow)
    return None, refusal


def paint_facility(corridor, facility) -> CorridorFacilityPaint:
    """Place `facility` along `corridor`, wherever the street can carry it.

    Walks the stations at which BOTH kerbs are traced - the only stations at which a width is a
    measurement rather than an interpolation - and asks the section itself whether it fits at
    each. Everything else comes back as a refusal or as an unsurveyed gap, and both are drawn: a
    corridor plan that quietly stops at its hardest point is the plan nobody costed.
    """
    side = facility_side(corridor, facility.side)
    other = "right" if side == "left" else "left"
    paint = CorridorFacilityPaint(road=corridor.name, side=side, compass_side=facility.side,
                                  section_ft=0.0)
    # EVERY hole, not only the ones longer than a usable facility. A 41 ft gap where the kerb is
    # untraced across a side street's mouth is still a stretch the drawing cannot speak for, and
    # left unnamed it reads as a hole in the road. MIN_FACILITY_RUN_FT is about whether a RUN is
    # worth calling a facility; it has nothing to say about whether a GAP is worth admitting to.
    paint.untraced = [(lo, hi) for lo, hi in corridor.untraced_gaps_ft(min_ft=0.0)]

    for span_lo, span_hi in corridor.both_traced_spans():
        if span_hi - span_lo < MIN_FACILITY_RUN_FT:
            continue
        stations = np.append(np.arange(span_lo, span_hi, CORRIDOR_SAMPLE_FT), span_hi)
        near = np.array([kerb_offset_ft(corridor, side, float(s)) or np.nan for s in stations])
        far = np.array([kerb_offset_ft(corridor, other, float(s)) or np.nan for s in stations])
        fits = np.array([bool(np.isfinite(n) and np.isfinite(f)
                              and section_at(facility, n, f)[0] is not None)
                         for n, f in zip(near, far)])
        _collect(paint, corridor, facility, side, stations, near, far, fits)
    paint.runs = [run for run in paint.runs if run.length_ft >= MIN_FACILITY_RUN_FT]
    return paint


def _collect(paint, corridor, facility, side, stations, near, far, fits):
    """Split one traced span into the stretches that fit and the ones that do not.

    The section for a run is rebuilt at the run's NARROWEST cross-section, not at each station:
    a facility drawn along a stretch of kerb is a promise about the whole of that stretch, which
    is the same reason AddBikeLane sizes against a leg's narrowest traced point.
    """
    for lo_i, hi_i, ok in _blocks(fits):
        lo, hi = float(stations[lo_i]), float(stations[hi_i])
        if hi - lo < CORRIDOR_SAMPLE_FT:
            continue
        block_near, block_far = near[lo_i:hi_i + 1], far[lo_i:hi_i + 1]
        total = block_near + block_far
        if not np.isfinite(total).any():
            paint.refusals.append(FacilityRefusal(lo, hi, "one or both kerbs untraced here"))
            continue
        # THE GOVERNING CROSS-SECTION is the narrowest station's OWN two half-widths, not the
        # smallest half-width on each side taken separately. Those are different numbers and the
        # second one is wrong: a street that is narrow at one end and off-centre at the other has
        # no station where both minima occur, so combining them invents a cross-section that is
        # nowhere on the road. It refused 464 ft of Broad St for a 29.57 ft width whose narrowest
        # real section is 31.9 ft - and refused it after every station in it had individually
        # passed, which is how the disagreement showed up.
        at = int(np.nanargmin(total))
        narrowest = float(total[at])
        section, refusal = section_at(facility, float(block_near[at]), float(block_far[at]))
        if not ok or section is None:
            paint.refusals.append(FacilityRefusal(
                lo, hi, refusal or "one or both kerbs untraced here", narrowest))
            continue
        paint.section_ft = section.section_ft
        run = _build_run(corridor, side, section, stations[lo_i:hi_i + 1], block_near)
        if run is not None:
            paint.runs.append(run)


def _blocks(flags: np.ndarray):
    """[(start index, end index, value)] over runs of equal value."""
    if not len(flags):
        return []
    edges = np.flatnonzero(np.diff(flags.astype(int))) + 1
    bounds = [0, *edges.tolist(), len(flags) - 1]
    return [(bounds[i], bounds[i + 1], bool(flags[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _build_run(corridor, side: str, section, stations, offs) -> FacilityRun | None:
    """The paint itself, from the SAME section accounting the per-leg treatment uses.

    The lane hugs the kerb (TwoWayBikeLane.hugs_kerb), so its three boundaries are insets from the
    traced kerb and follow it; the travel lane's edge is measured from the alignment so it holds
    its target whatever the kerb does; the buffer between them absorbs the difference. That split
    is bikeways.BikeLane.offsets_from_kerb_ft's, quoted rather than re-derived.
    """
    kerb = section.offsets_from_kerb_ft()
    on = Alignment(corridor.centerline)
    if not np.isfinite(offs).all():
        return None
    lane_outer = offs - kerb["bike_outer_ft"]
    lane_inner = offs - kerb["bike_inner_ft"]
    travel_edge = lane_inner - section.buffer_ft
    if (travel_edge <= 0).any():
        return None

    lane = band_from_offsets(on, side, stations, lane_inner, lane_outer)
    buffer_zone = band_from_offsets(on, side, stations, travel_edge, lane_inner)
    edges = [line_from_offsets(on, side, stations, off)
             for off in (lane_outer, lane_inner, travel_edge)]
    return FacilityRun(start_ft=float(stations[0]), end_ft=float(stations[-1]),
                       lane_surface=lane, buffer_zone=buffer_zone,
                       edge_lines=tuple(e for e in edges if e is not None),
                       bollards=_bollards(on, side, stations, (travel_edge + lane_inner) / 2))


def _bollards(on: Alignment, side: str, stations, offsets) -> tuple:
    """Flex posts down the middle of the buffer, at the facility's own pitch.

    Placed on the corridor's station grid rather than on a fresh one, so a post never lands
    between two samples of the buffer it is supposed to stand in.
    """
    from src.geometry.treatments.bikeways import BIKE_LANE_BOLLARD_SPACING_FT

    sign = 1.0 if side == "left" else -1.0
    want = np.arange(float(stations[0]), float(stations[-1]), BIKE_LANE_BOLLARD_SPACING_FT)
    at = np.interp(want, stations, offsets)
    return tuple(place_in_measured_frame(on.centerline, want, sign * at))


def centred_on_its_kerbs(corridor, sample_ft: float = 10.0, smooth_ft: float = 60.0):
    """The corridor with its centreline moved onto the midpoint of its two traced kerbs.

    WITHOUT THIS THE CORRIDOR CANNOT BE PAINTED, and the measurement is stark. Between the
    modelled junctions a corridor's centreline is NJDOT's raw SRI alignment, which is a
    linear-referencing reference and not a surveyed carriageway centre. Sampled over Broad St
    stations 921-1959, the south kerb sits a median 15.1 ft out and the north kerb 31.2 ft - a 16
    ft asymmetry on a 47 ft street. Every offset in a section is measured from this line, so a
    two-way lane placed against the south kerb had its travel-lane edge land at a NEGATIVE offset
    at 21 of 209 stations: paint on the far side of the line it is measured from.

    Inside each modelled junction the alignment is already centred - fitting.py's
    _centre_legs_on_traced_kerbs bends each leg onto its own carriageway centre over the whole leg
    - and that pass simply does not reach the 78% of the corridor no junction models. This is the
    same correction, applied to the road instead of to a leg, which is what
    docs/network-model.md's step 4 says the centring becomes.

    Held FLAT where only one kerb is traced, rather than extrapolated: with one kerb there is no
    midpoint, only a distance to one edge, and inventing the other half of a cross-section is the
    one thing this project must not do. Smoothed over `smooth_ft` for the reason the per-leg pass
    gives - the result has to be a line a striper would lay, following the street's bend and not
    every wobble in the tracing.
    """
    import dataclasses

    from src.geometry.intersection.fitting import (MIN_CENTRE_VERTEX_GAP_FT, _smoothed,
                                                   _thinned)
    from src.geometry.model import station_offset_many

    grid = np.append(np.arange(0.0, corridor.length_ft, sample_ft), corridor.length_ft)
    left = np.array([kerb_offset_ft(corridor, "left", float(s)) or np.nan for s in grid])
    right = np.array([kerb_offset_ft(corridor, "right", float(s)) or np.nan for s in grid])
    both = np.isfinite(left) & np.isfinite(right)
    if both.sum() < 2:
        return corridor
    # +left, -right, so the midpoint's own offset is half their difference of magnitudes.
    midpoint = (left - right) / 2
    filled = np.interp(grid, grid[both], midpoint[both])      # flat beyond the traced ends
    corrections = _smoothed(filled, sample_ft, smooth_ft)

    own, _offsets = station_offset_many(corridor.centerline,
                                        np.asarray(corridor.centerline.coords, dtype=float))
    stations = _thinned(np.unique(np.concatenate([grid, np.clip(own, 0.0, corridor.length_ft)])),
                        MIN_CENTRE_VERTEX_GAP_FT)
    moved = LineString(place_in_measured_frame(corridor.centerline, stations,
                                               np.interp(stations, grid, corrections)))
    return dataclasses.replace(corridor, centerline=moved)
