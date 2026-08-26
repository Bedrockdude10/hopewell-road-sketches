"""THE FACILITY PAINTED ALONG A ROAD, not along a leg.

A treatment targets one kerb of one approach, and an approach is a 130-170 ft stub. Broad Street
is 3,693 ft with 91% of both kerbs traced - the survey is four times ahead of what can be drawn,
and a corridor drawing made from leg paint shows a protected bikeway appearing and vanishing
three times along a street it is supposed to run the length of.

WHY NOT JUST CARRY THE LEGS FURTHER. At HOPEWELL_FRAME_SCALE=2.5 the legs reach 325-425 ft and
the junction still builds; at 4 and 5 it DOES NOT BUILD AT ALL. Four legs each carrying their
own full-width envelope out of one node have envelopes that cross once they are long enough and
the street bends. The corner-fillet model is the ceiling, at roughly a tenth of the corridor.

A road has no such ceiling, because away from a node there is no fillet: there is a centreline,
two traced kerbs, and a cross-section between them.

WHAT IS NOT REDEFINED HERE. The cross-section is `treatments.bikeways.BikeLane` and its subclass -
`offsets_from_kerb_ft()` and `section_ft` are asked of the same class, so the corridor and the
junctions cannot come to different answers about where the stripes go. This module owns only
WHERE the section is placed. The junction detail in a corridor drawing comes from the junction
models; this supplies the run between them.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon

from src.geometry.model import (MAX_KERB_FOLLOW_TAPER, Alignment, band_from_offsets,
                                line_from_offsets, place_in_measured_frame, side_facing,
                                taper_limited, whole_stalls_ft)
from src.geometry.treatments.bikeways import (MIN_FACILITY_RUN_FT, section_at)
from src.geometry.treatments.state import FacilityRefusal
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.network import Corridor

#: How finely the section is sampled along the corridor. The kerb is traced, so its offset moves
#: with every vertex the surveyor placed; 5 ft follows a real kerb's course. Coarser than the
#: junction paint's STRIP_SAMPLE_FT because this runs 3,693 ft rather than 130.
CORRIDOR_SAMPLE_FT = 5.0

@dataclass(frozen=True)
class FacilityRun:
    """One continuous stretch where the section fits, with the paint that goes on it."""
    start_ft: float
    end_ft: float
    lane_surface: Polygon | None = None
    buffer_zone: Polygon | None = None
    edge_lines: tuple = ()
    bollards: tuple = ()
    #: (stations, offsets) of the travel lane's edge AS DRAWN - the profile _build_run actually
    #: placed, not the scalar it was sized from. Kept because the divider hangs one lane width off
    #: this line, so far_kerb_lane_edge has to read the drawn profile or the far kerb's parking is
    #: measured against a stripe that is not where the stripe is.
    travel_edge_ft: tuple = ()
    #: The rung this run landed on. Kept because the divider shift - and therefore how much room
    #: is left on the FAR kerb for parking - is a property of the section, and it differs between
    #: a standard run and a constrained one.
    section: object = None

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
    #: (lo, hi) where the lane is CUT - pedestrian crossings, which outrank it.
    breaks: tuple = ()
    #: (lo, hi) where the lane CONTINUES but dotted - a driveway or side street on its own kerb.
    #: The crossbike. See opening_spans for why these are not the same list.
    dotted: tuple = ()

    @property
    def placed_ft(self) -> float:
        return sum(run.length_ft for run in self.runs)

    def summary(self, corridor_length_ft: float) -> str:
        lines = [f"{self.road}: two-way protected lane on the {self.compass_side} kerb",
                 f"  placed        {self.placed_ft:8,.0f} ft of {corridor_length_ft:,.0f} "
                 f"({self.placed_ft / corridor_length_ft:.0%}) in {len(self.runs)} run(s)"]
        # HOW FAR THE SECTION SLID, because a reader who is not told assumes it did not. A run's
        # rung is chosen at its narrowest cross-section and the section then follows its
        # rate-limited kerb out from there, so this is the width one narrow spot is NOT costing the
        # rest of the run - and it lands on the other kerb, which is where the parking count moves.
        slide = [max(run.travel_edge_ft[1]) - min(run.travel_edge_ft[1])
                 for run in self.runs if run.travel_edge_ft]
        if slide:
            lines.append(f"  resectioned   {max(slide):8.1f} ft of lateral give at the most "
                         f"generous run, {sum(slide) / len(slide):.1f} ft mean, all of it inside "
                         f"the traced kerb and inside 1:{1 / MAX_KERB_FOLLOW_TAPER:.0f}")
        for refusal in self.refusals:
            narrow = "" if refusal.narrowest_ft is None else f", narrowest {refusal.narrowest_ft:.1f} ft"
            lines.append(f"  REFUSED       {refusal.start_ft:6,.0f}-{refusal.end_ft:<6,.0f} "
                         f"({refusal.length_ft:5,.0f} ft) {refusal.reason}{narrow}")
        for lo, hi, why in self.untraced:
            label = "JUNCTION" if why == JUNCTION_MOUTH else "NO SURVEY"
            lines.append(f"  {label:13s} {lo:6,.0f}-{hi:<6,.0f} ({hi - lo:5,.0f} ft) {why}")
        return "\n".join(lines)


#: Why a stretch of corridor has no kerb line on one or both sides. The distinction is the whole
#: point: one is a fact about the STREET and the other about the SURVEY, and a drawing that calls
#: them both "unsurveyed" tells a reader to go and trace something that is already correct - and
#: understates its own coverage while doing it.
JUNCTION_MOUTH = "a cross street's mouth - there is no kerb across an intersection"
NOT_TRACED = "one or both kerbs untraced - the section cannot be tested here"

#: How near a cross street a gap has to start to be that street's mouth rather than missing
#: tracing. The corner-return zone, for the same reason it is that: past the return the kerb is
#: the road's own again, so a hole out there is a hole in the survey.
MOUTH_REACH_FT = 45.0


def _why_no_kerb(corridor: "Corridor", lo_ft: float, hi_ft: float) -> str:
    """Whether this hole is an intersection or a stretch nobody has traced."""
    mid = (lo_ft + hi_ft) / 2
    for junction in corridor.junctions:
        if abs(junction.node_ft - mid) <= MOUTH_REACH_FT:
            return JUNCTION_MOUTH
    for cross in corridor.cross_street_ft:
        if abs(cross - mid) <= MOUTH_REACH_FT:
            return JUNCTION_MOUTH
    return NOT_TRACED


def facility_side(corridor: "Corridor", compass: str) -> str:
    """Which of the corridor's own sides faces `compass`.

    Through the SAME side_facing every leg uses, on the corridor's centreline. A corridor runs one
    way along its whole length by construction, so unlike a leg pair there is one answer and it
    holds end to end - which is the property that stops a route swapping kerbs at a junction.
    """
    return side_facing(Alignment(corridor.centerline), compass)


def kerb_offset_ft(corridor: "Corridor", side: str, station_ft: float) -> float | None:
    """How far out the traced kerb sits on one side, unsigned, or None where it is not traced."""
    from src.geometry.network import _kerb_offset_at

    run = corridor.kerb_run_at(side, station_ft)
    if run is None:
        return None
    return _kerb_offset_at(corridor.centerline, run.line, side, station_ft)


#: How wide a surveyed crossing's break in the lane is taken to be, where the corridor knows the
#: crossing only as a station. The marked crossings on Broad St are traced ways 38-77 ft long
#: ACROSS the street; what is needed here is their extent ALONG it, which is the crosswalk's own
#: width, and MUTCD 3C.02 puts a marked crosswalk at 6 ft minimum with 10 ft usual on a road like
#: this. Taken as a constant and labelled an assumption, because reading it off each traced way
#: needs the way's own bearing and that belongs with surveyed.py rather than here.
CROSSING_BREAK_FT = 10.0


def opening_spans(corridor: "Corridor", facts, side: str) -> tuple:
    """Driveway and side-street mouths on the facility's OWN kerb - where the lane goes DOTTED.

    IT DOES NOT BREAK HERE. NACTO requires a bidirectional lane to continue through every
    intersection and driveway as a crossbike (STANDARDS.md, verified 2026-08-18), explicitly
    rejecting the alternative because merging riders into traffic would send the contraflow
    direction against its flow. The spans come back so the drawing can dot the lane over them;
    nothing is cut.

    FROM THIS KERB ONLY. A mouth on the far kerb does not touch this lane.
    """
    return tuple(sorted((max(o.start_ft, 0.0), min(o.end_ft, corridor.length_ft))
                        for opening_side, o in facts.openings
                        if opening_side == side and o.end_ft > 0
                        and o.start_ft < corridor.length_ft))


def crossing_spans(corridor: "Corridor", facts) -> tuple:
    """Pedestrian crossings - the ONE thing that does cut the lane.

    A crossing outranks whatever runs along the kerb.
    """
    return tuple(sorted((max(station_ft - CROSSING_BREAK_FT / 2, 0.0),
                         min(station_ft + CROSSING_BREAK_FT / 2, corridor.length_ft))
                        for station_ft, _markings in facts.marked_crossings))


def _cut_around(geometry, corridor: "Corridor", side: str, spans, reach_ft: float = 60.0):
    """`geometry` with the break spans taken out of it.

    Cut as full-depth bands across the kerbside rather than as station ranges of the polygon,
    because the lane, its buffer and its edge lines all have to break at the SAME stations.
    """
    from shapely.ops import unary_union

    if geometry is None or geometry.is_empty or not spans:
        return geometry
    on = Alignment(corridor.centerline)
    cutters = []
    for lo, hi in spans:
        if hi - lo <= 0:
            continue
        stations = np.linspace(lo, hi, max(int((hi - lo) / 2.0) + 2, 3))
        band = band_from_offsets(on, side, stations, np.zeros(len(stations)),
                                 np.full(len(stations), reach_ft))
        if band is not None and not band.is_empty:
            cutters.append(band)
    if not cutters:
        return geometry
    cut = geometry.difference(unary_union(cutters))
    return None if cut.is_empty else cut


def paint_facility(corridor: "Corridor", facility, facts=None) -> CorridorFacilityPaint:
    """Place `facility` along `corridor`, wherever the street can carry it.

    Walks the stations at which BOTH kerbs are traced - the only stations where a width is a
    measurement - and asks the section itself whether it fits. Everything else comes back as a
    refusal or an unsurveyed gap.
    """
    side = facility_side(corridor, facility.side)
    other = "right" if side == "left" else "left"
    paint = CorridorFacilityPaint(road=corridor.name, side=side, compass_side=facility.side,
                                  section_ft=0.0)
    # EVERY hole, not only the ones longer than a usable facility. A 41 ft gap where the kerb is
    # untraced across a side street's mouth is still a stretch the drawing cannot speak for, and
    # left unnamed it reads as a hole in the road. MIN_FACILITY_RUN_FT is about whether a RUN is
    # worth calling a facility; it has nothing to say about whether a GAP is worth admitting to.
    paint.untraced = [(lo, hi, _why_no_kerb(corridor, lo, hi))
                      for lo, hi in corridor.untraced_gaps_ft(min_ft=0.0)]
    if facts is not None:
        paint.breaks = crossing_spans(corridor, facts)          # cut
        paint.dotted = opening_spans(corridor, facts, side)     # carried across, dotted

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
    return paint


def _collect(paint, corridor: "Corridor", facility, side: str, stations: np.ndarray, near, far, fits):
    """Split one traced span into the stretches that fit and the ones that do not.

    The section for a run is rebuilt at the run's NARROWEST cross-section: a facility drawn
    along a stretch of kerb is a promise about the whole of that stretch.
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
        # smallest half-width on each side taken separately - those are different numbers and the
        # second one is wrong.
        at = int(np.nanargmin(total))
        narrowest = float(total[at])
        section, refusal = section_at(facility, float(block_near[at]), float(block_far[at]))
        if not ok or section is None:
            paint.refusals.append(FacilityRefusal(
                lo, hi, refusal or "one or both kerbs untraced here", narrowest))
            continue
        paint.section_ft = section.section_ft
        run, why = _build_run(corridor, side, section, stations[lo_i:hi_i + 1], block_near,
                              paint.breaks)
        if run is None:
            paint.refusals.append(FacilityRefusal(lo, hi, why, narrowest))
        elif run.length_ft < MIN_FACILITY_RUN_FT:
            # A stretch too short to be a facility is still a stretch the reader can see bare
            # asphalt on. Dropped silently it is an unexplained hole in the drawing, which is the
            # same failure as calling a junction mouth unsurveyed.
            paint.refusals.append(FacilityRefusal(
                lo, hi, f"only {run.length_ft:.0f} ft of continuous room here, under the "
                        f"{MIN_FACILITY_RUN_FT:.0f} ft a usable facility needs", narrowest))
        else:
            paint.runs.append(run)


def _blocks(flags: np.ndarray):
    """[(start index, end index, value)] over runs of equal value."""
    if not len(flags):
        return []
    edges = np.flatnonzero(np.diff(flags.astype(int))) + 1
    bounds = [0, *edges.tolist(), len(flags) - 1]
    return [(bounds[i], bounds[i + 1], bool(flags[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _build_run(corridor: "Corridor", side: str, section, stations: np.ndarray, offs, breaks=()):
    """The paint itself, from the SAME section accounting the per-leg treatment uses.

    The lane hugs the kerb (TwoWayBikeLane.hugs_kerb), so THE WHOLE SECTION SLIDES AS ONE PIECE
    and every offset comes off a single placement line. Written per-offset instead, the section
    stretched: the buffer's inner edge followed the raw tracing while the divider downstream of it
    (far_kerb_lane_edge) stayed frozen at the governing station's arithmetic, and on the 1,050 ft
    run through station 1400 those two datums ended 7.3 ft apart.

    The placement line is the kerb RATE-LIMITED, not the kerb - tapered_curb_offsets' rule, which
    the corridor was the only kerb-following paint in the repo not to follow. It matters more here
    than on a leg: the divider is derived from this line, so a flare in the tracing would steer a
    driver through it. MAX_KERB_FOLLOW_TAPER is NACTO's 1:5; see it for why that is the figure.

    NO FLOOR AT THE GOVERNING HALF-WIDTH, and it costs nothing to leave it out. The section does
    slide inside where it was sized - `_collect` governs off the min-SUM station, whose near kerb
    can be 2.95 ft wider than the narrowest near kerb on the run - but the divider hangs off THIS
    line (far_kerb_lane_edge), so both travel lanes stay at `divided_lane_width_ft` everywhere and
    the whole deficit lands on the far kerb as parking room. Measured over 815 stations of Broad
    St that room never goes negative. A floor would instead hold paint 2.13 ft OUT over the traced
    kerb, measured, which is what taper_limited's at-or-inside promise exists to rule out.
    """
    kerb = section.offsets_from_kerb_ft()
    on = Alignment(corridor.centerline)
    if not np.isfinite(offs).all():
        return None, "the kerb is not readable at every station of this stretch"
    place = taper_limited(stations, offs)
    lane_outer = place - kerb["bike_outer_ft"]
    lane_inner = place - kerb["bike_inner_ft"]
    travel_edge = place - section.section_ft
    if (travel_edge <= 0).any():
        # The section is wider than the distance from the alignment to its own kerb, so the travel
        # lane's edge lands on the far side of the line it is measured from. That is a real
        # finding - the carriageway is not centred on the alignment here, or the kerb swings in -
        # and it must be reported rather than dropped, which is what it was.
        worst = float(np.min(travel_edge))
        return None, (f"the section reaches {abs(worst):.1f} ft PAST the alignment - the kerb "
                      f"comes within {float(np.min(offs)):.1f} ft of it here, less than the "
                      f"{section.section_ft:.1f} ft the section needs on this side")

    mine = tuple((lo, hi) for lo, hi in breaks
                 if hi > float(stations[0]) and lo < float(stations[-1]))
    lane = _cut_around(band_from_offsets(on, side, stations, lane_inner, lane_outer),
                       corridor, side, mine)
    buffer_zone = _cut_around(band_from_offsets(on, side, stations, travel_edge, lane_inner),
                              corridor, side, mine)
    edges = [line_from_offsets(on, side, stations, off)
             for off in (lane_outer, lane_inner, travel_edge)]
    return FacilityRun(start_ft=float(stations[0]), end_ft=float(stations[-1]),
                       lane_surface=lane, buffer_zone=buffer_zone,
                       edge_lines=tuple(e for e in edges if e is not None),
                       bollards=_bollards(on, side, stations, (travel_edge + lane_inner) / 2),
                       travel_edge_ft=(tuple(stations.tolist()), tuple(travel_edge.tolist())),
                       section=section), None


def _bollards(on: Alignment, side: str, stations: np.ndarray, offsets) -> tuple:
    """Flex posts down the middle of the buffer, at the facility's own pitch.

    Placed on the corridor's station grid rather than on a fresh one, so a post never lands
    between two samples of the buffer it is supposed to stand in.
    """
    from src.geometry.treatments.bikeways import BIKE_LANE_BOLLARD_SPACING_FT

    sign = 1.0 if side == "left" else -1.0
    want = np.arange(float(stations[0]), float(stations[-1]), BIKE_LANE_BOLLARD_SPACING_FT)
    at = np.interp(want, stations, offsets)
    return tuple(place_in_measured_frame(on.centerline, want, sign * at))


def centred_on_its_kerbs(corridor: "Corridor", sample_ft: float = 10.0, smooth_ft: float = 60.0):
    """The corridor with its centreline moved onto the midpoint of its two traced kerbs.

    WITHOUT THIS THE CORRIDOR CANNOT BE PAINTED. Between modelled junctions the centreline is
    NJDOT's raw SRI alignment - a linear-referencing reference, not a carriageway centre. An
    asymmetric street (south kerb 15.1 ft out, north kerb 31.2 ft) puts the travel-lane edge at
    a NEGATIVE offset at some stations: paint on the far side of the line it is measured from.

    Held FLAT where only one kerb is traced (no midpoint, only a distance to one edge), smoothed
    over `smooth_ft` to follow the street's bend rather than every wobble in the tracing.
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
    return dataclasses.replace(corridor, centerline=moved,
                               kerb_runs=tuple(_restationed(run, moved) for run in corridor.kerb_runs))


def _restationed(run, centerline: LineString):
    """One kerb run's station span, re-measured against a centreline that has moved.

    MOVING THE CENTRELINE MOVES EVERY STATION ON IT. A KerbRun carries its own start_ft and
    end_ft; left alone they describe where the run sat on the OLD line, and a station inside
    the declared span can fall outside the real one. Re-measured rather than shifted by a
    constant: the correction varies along the road.
    """
    import dataclasses

    from src.geometry.model import curb_station_span

    span = curb_station_span(Alignment.one_sided(centerline, run.side, run.line), run.side)
    if span is None:
        return run
    return dataclasses.replace(run, start_ft=float(span[0]), end_ft=float(span[1]))


def _kerb_band_over(corridor: "Corridor", on: "Alignment", side: str, spans, depth_ft: float,
                    limit_at=None, piece: str = "stall", may_park: bool = True):
    """[(lo_ft, hi_ft, polygon)] - one band against the traced kerb, per span.

    Shared by `stall_bands` and `hatch_bands` so a marked-stall band and a hatched one are the
    same shape drawn over different spans, not two independently-tuned rectangles that could
    drift apart depth-wise.

    AS DEEP AS THE KERB IS FREE, not a flat `depth_ft`. `limit_at(station) -> offset` is the
    innermost the band may reach - the travel way's own edge. Drawn at a flat stall depth, Broad
    St's "no room" hatch stood 8 ft off a kerb with a median of 4.01 ft free and overlapped the
    travel lane at 346 of 381 samples; the restricted hatch has 1.60 ft free and overlapped by a
    median of 6.40 ft. Both then LOOK like a stall's worth of kerb, which is the one thing the
    reader is being asked to judge - the sheet was answering "could a car park here" with a
    rectangle the width of a car, everywhere.

    Pass the same line the spans were width-tested against, so a drawn depth is the tested spare
    and not a second opinion about it.

    HOW DEEP IS `parking.allocate_kerbside`'S CALL AND NOT THIS FUNCTION'S. It used to be
    `min(depth_ft, free)` written here, which is the junction rule with its remainder term
    deleted: past 8 ft of free kerb the stall stopped growing, correctly, and nothing took the
    rest. On Broad St's south kerb - 48 to 58 ft between kerbs approaching Greenwood, where the
    design hands every foot it does not use to that one side - it left 5,748 sq ft over 1,000 ft
    unpainted, on the widest asphalt in the borough. `piece` picks which half of the allocation
    to draw: "stall" the marked box against the kerb, "stripe" the hatched remainder inboard of
    it. A hatch span passes `may_park=False`, which allocates the whole zone to the stripe.
    """
    from src.geometry.treatments.parking import allocate_kerbside
    out = []
    for lo, hi in spans:
        if hi - lo < CORRIDOR_SAMPLE_FT:
            continue
        stations = np.append(np.arange(lo, hi, CORRIDOR_SAMPLE_FT), hi)
        offs = np.array([kerb_offset_ft(corridor, side, float(s)) or np.nan for s in stations])
        # A handful of untraced samples inside an otherwise-traced span (a kerb-topology seam, not
        # a real gap) used to drop the WHOLE span - losing 1,326 of 3,526 ft of Broad St's far-kerb
        # hatch to four samples out of 272. Build the band from what IS traced instead.
        finite = np.isfinite(offs)
        if finite.sum() < 2:
            continue
        # Never past the kerb itself either: where the travel way reaches the kerb there is no
        # kerbside strip to draw, and a band that inverts would hatch the carriageway.
        limits = (np.full_like(offs, np.nan) if limit_at is None
                  else np.array([limit_at(float(s)) for s in stations], dtype=float))
        zone = np.where(np.isfinite(limits), offs - limits, depth_ft)
        alloc = [allocate_kerbside(float(z), float(z), may_park) if np.isfinite(z)
                 else None for z in zone]
        stall = np.array([0.0 if a is None else a.stall_depth_ft for a in alloc])
        total = np.array([0.0 if a is None else a.total_ft for a in alloc])
        # THE BOX SITS AGAINST THE TRAVEL LANE AND THE HATCH AGAINST THE KERB, which is the
        # junction's order (MarkedParking.curb_offset_ft: "parking sits directly against the
        # active travel lane instead of against the curb") and was this view's the other way
        # round. Nothing showed it until the remainder started being drawn at all - with no hatch
        # beside a box there was no order for the two views to disagree about. Measured on
        # broad_st_east's right kerb the junction draws the stall over 14.64-22.49 ft and its
        # buffer over 22.49-26.54, so the hatch is the outboard piece.
        base = offs - total
        outer = base + (stall if piece == "stall" else total)
        inner = base + (0.0 if piece == "stall" else stall)
        band = band_from_offsets(on, side, stations[finite], inner[finite], outer[finite])
        if band is not None and not band.is_empty:
            out.append((float(lo), float(hi), band))
    return out


def stall_bands(corridor: "Corridor", side: str, spans, depth_ft: float | None = None,
                limit_at=None):
    """[(lo_ft, hi_ft, polygon)] - a stall-deep band against the traced kerb over the spans given.

    IT TAKES THE SPANS AND DOES NOT CHOOSE THEM. This was `parking_bands`, which picked
    `CorridorFacts.parkable` itself - the LEGAL test alone - and so shaded 3,152 ft of Broad St's
    far kerb as parking where 990 ft of it holds a car: no width test, no driveway mouths, and the
    unusable tail of every run. Where a stall may be marked is three independent tests (legal,
    room, clear) and then a whole-car walk, and a band builder is not the place any of that gets
    decided. Pass `stall_footprints`, so what is shaded is what was counted.

    Stationed against the KERB and not the alignment - the kerb wanders, and a kerbside zone that
    ignored it would drift off the street. WHERE IN THAT ZONE the box sits is a separate question
    and the answer is NOT "against the kerb": it sits against the travel lane, with the surplus
    hatched outboard of it. See _kerb_band_over, and MarkedParking.curb_offset_ft for the junction
    saying the same thing first.
    """
    from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT

    depth_ft = PARKING_STALL_DEPTH_DEFAULT_FT if depth_ft is None else depth_ft
    return _kerb_band_over(corridor, Alignment(corridor.centerline), side, spans, depth_ft,
                           limit_at, piece="stall", may_park=True)


def hatch_bands(corridor: "Corridor", facts, side: str, marked_spans,
                depth_ft: float | None = None, limit_at=None):
    """[(lo_ft, hi_ft, polygon, reason)] where this kerb carries neither a marked stall nor a
    vehicle crossing - `parking.py`'s "parking or hatching, never neither" rule, asked of a road.

    `reason` is `"legal"` where R.S. 39:4-138, a sign, or an OSM restriction closes the kerb
    outright (`CorridorFacts.no_parking`), and `"room"` where the law allows parking but no stall
    fits - too narrow once the facility holds its target lane, or too short for one whole car
    (`CorridorFacts.parkable` minus what actually got marked). These are drawn in different colours
    because they are different findings about different things: one is a fact about the STREET
    (a corner, a hydrant, a stop sign) and the other is a fact about this DESIGN's own section - a
    narrower facility or a shorter divider shift would open kerb that is already legally clear.

    `marked_spans` is the kerb the boxes ACTUALLY cover - `stall_footprints`, not the spans they
    were counted out of, and not re-derived from `stall_room_spans` here. Both of the other two
    have been passed and both are wrong the same way: they hand this function a length no car can
    use as though it were parking, and it is then hatched nowhere. A run's own tail is the case
    this function exists to catch, so it is the one thing the argument must not smuggle back in.
    Excludes driveway/side-street mouths - a hatch is a paint decision, and nothing is painted
    over a place a vehicle actually crosses the kerb.
    """
    from src.geometry.network import _complement_spans, _intersect_spans, _merged_spans
    from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT

    depth_ft = PARKING_STALL_DEPTH_DEFAULT_FT if depth_ft is None else depth_ft
    mouths = _merged_spans([(opening.start_ft, opening.end_ft)
                            for opening_side, opening in facts.openings if opening_side == side])
    clear = _complement_spans(mouths, 0.0, corridor.length_ft)
    unmarked = _complement_spans(marked_spans, 0.0, corridor.length_ft)
    candidate = _intersect_spans(clear, unmarked)

    # no_parking zones can overlap each other (stacked statutory setbacks) so are merged before
    # `_intersect_spans`, which assumes disjoint input; `parkable` is already the merged complement
    # of `no_parking`, computed once in corridor_facts, so the two partition `candidate` exactly.
    restricted = _merged_spans([(zone.start_ft, zone.end_ft)
                                for zone in facts.by_side("no_parking", side)])
    legal = _intersect_spans(candidate, restricted)
    room = _intersect_spans(candidate, facts.by_side("parkable", side))

    on = Alignment(corridor.centerline)

    def bands(spans, reason, may_park):
        return [(lo, hi, band, reason)
                for lo, hi, band in _kerb_band_over(corridor, on, side, spans, depth_ft, limit_at,
                                                    piece="stripe", may_park=may_park)]

    # THE THIRD SET IS THE REMAINDER BESIDE A MARKED STALL, and leaving it out is what made this
    # function's own headline false. A stall stops at PARKING_STALL_DEPTH_DEFAULT_FT because that
    # is how deep a car is; where the kerb spares more - 48 to 58 ft between kerbs on Broad St
    # approaching Greenwood - the rest is neither a stall nor, until now, a hatch. `marked_spans`
    # carries a stall, so `may_park=True` allocates the box first and this draws what is left.
    return (bands(legal, "legal", False) + bands(room, "room", False)
            + bands(_intersect_spans(_merged_spans(marked_spans), clear), "room", True))


def stall_room_spans(corridor: "Corridor", side: str, lane_edge_at, sample_ft: float = CORRIDOR_SAMPLE_FT):
    """Where this kerb has room for a usable stall once the travel lane holds its target.

    THE OTHER HALF OF A STALL COUNT. `CorridorFacts.parkable` says where the LAW leaves room;
    this says where the STREET does. Exactly parking.py:hold_travel_lane_at_target's arithmetic,
    per station: the surplus is whatever the traced kerb leaves beyond the travel lane's edge, and
    MIN_USABLE_STALL_FT is the floor below which it is hatched rather than marked.
    """
    from src.geometry.treatments.parking import MIN_USABLE_STALL_FT

    grid = np.append(np.arange(0.0, corridor.length_ft, sample_ft), corridor.length_ft)
    offs = np.array([kerb_offset_ft(corridor, side, float(s)) or np.nan for s in grid])
    edges = np.array([lane_edge_at(float(s)) for s in grid])
    fits = np.isfinite(offs) & ((offs - edges) >= MIN_USABLE_STALL_FT)
    out = []
    for lo_i, hi_i, ok in _blocks(fits):
        if ok and grid[hi_i] - grid[lo_i] >= sample_ft:
            out.append((float(grid[lo_i]), float(grid[hi_i])))
    return tuple(out)


def far_kerb_lane_edge(paint: CorridorFacilityPaint, default_ft: float | None = None):
    """station -> where the FAR kerb's travel lane edge sits, given what the facility placed.

    Taking width out of one kerbside pushes the divider toward the other, and the divider sits one
    lane width in from the near travel edge - so the far kerb's lane edge is two lane widths in
    from the DRAWN near travel edge. `divided_lane_width_ft`, NOT `TARGET_LANE_WIDTH_FT`: they
    agree only on a leg wide enough to hold two target-width lanes, and
    `divider.travel_lane_edge_ft`'s docstring names the leg that doesn't (w_broad_st_northeast, an
    0.92 ft error). Per run, because a constrained rung has a different lane width.

    A RE-EXPRESSION OF travel_lane_divider_shift_ft, NOT A RELAXATION OF IT (SKILLS.md section 4):
    that shift is `lane_w - (near_half_ft - section_ft)`, so `shift + lane_w` is identically
    `2*lane_w - travel_edge` at the one station the section was sized from. What changes is that
    the near travel edge is a PROFILE - the section follows its rate-limited kerb - so a run whose
    near kerb opens out 4 ft past its own narrowest point hands those 4 ft to the far kerb instead
    of pinning 1,050 ft of street to one cross-section. Reading the scalar while _build_run drew
    the profile put the two 7.3 ft apart through station 1400.
    """
    from src.geometry.treatments.base import TARGET_LANE_WIDTH_FT
    from src.geometry.treatments.bikeways import divided_lane_width_ft

    default_ft = TARGET_LANE_WIDTH_FT if default_ft is None else default_ft
    placed = [(run.start_ft, run.end_ft, 2.0 * divided_lane_width_ft(run.section),
               np.asarray(run.travel_edge_ft[0]), np.asarray(run.travel_edge_ft[1]))
              for run in paint.runs if run.section is not None and run.travel_edge_ft]

    def at(station_ft: float) -> float:
        for lo, hi, two_lanes_ft, stations, edges in placed:
            if lo <= station_ft <= hi:
                return two_lanes_ft - float(np.interp(station_ft, stations, edges))
        return default_ft

    return at


def travel_way_edges(paint: CorridorFacilityPaint, sample_ft: float = CORRIDOR_SAMPLE_FT):
    """((stations, near_ft, far_ft), ...) - one per run: the travel way's own two outside edges.

    `near_ft` is how far the travel way's edge sits from the alignment on the FACILITY's side and
    `far_ft` the same on the other. Both are magnitudes, like `kerb_offset_ft`, so a caller signs
    them the way it signs its own kerbs.

    THIS IS THE LINE THE PARKING IS TESTED AGAINST, and until it was drawn the corridor sheet did
    not show it anywhere: a reader saw a wide grey carriageway and 1,904 ft of kerb hatched "no
    room" against nothing visible, which reads as the drawing being wrong rather than the street
    being narrow. It is not - 89% of that hatch has under 7 ft between this line and the kerb, a
    median of 4.0 ft - but a sheet that cannot be checked by eye is asking to be taken on trust.

    ONE PER RUN, AND NOWHERE ELSE. `far_kerb_lane_edge` answers every station, falling back to
    TARGET_LANE_WIDTH_FT where no run was placed - the right answer for a width test and the wrong
    one for a drawing, because a line drawn across a junction mouth claims a lane edge the section
    was never tested against. A run is where the section actually resolved, so the line stops
    where the testing stops, and the 370 ft of Broad St's far kerb decided against that default
    can be seen to have been decided against nothing drawn.
    """
    edge_at = far_kerb_lane_edge(paint)
    out = []
    for run in paint.runs:
        if run.section is None or not run.travel_edge_ft:
            continue
        stations = np.append(np.arange(run.start_ft, run.end_ft, sample_ft), run.end_ft)
        near = np.interp(stations, run.travel_edge_ft[0], run.travel_edge_ft[1])
        # Off `far_kerb_lane_edge` and not off `2 * lane_w - near` rebuilt here: the far edge has
        # one home, and every station asked for is inside a run, so its fallback cannot fire.
        far = np.array([edge_at(float(station)) for station in stations])
        out.append((stations, near, far))
    return tuple(out)


def _stall_ft(stall_ft: float | None) -> float:
    """The stall length, defaulted once - `stalls_per_span`, `stall_footprints` and `stall_marks`
    all walk a run in the same steps, and three separate defaults are three chances to disagree.
    """
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT

    return PARKING_STALL_LENGTH_DEFAULT_FT if stall_ft is None else stall_ft


def stalls_per_span(spans, stall_ft: float | None = None):
    """((lo, hi, stalls), ...) - how many whole cars each span holds, and where.

    The per-span split exists so a drawing can LABEL each run with its own number: a boro reading
    a total has to be able to find that total on the page, run by run, or it is being asked to
    take it on trust. `stall_marks` counts through this, so the labels sum to the headline.
    """
    stall_ft = _stall_ft(stall_ft)
    # A span shorter than one car holds none, and gets no label pretending otherwise.
    return tuple((lo, hi, whole_stalls_ft(hi - lo, stall_ft)) for lo, hi in spans
                 if whole_stalls_ft(hi - lo, stall_ft) >= 1)


def stall_footprints(spans, stall_ft: float | None = None):
    """((lo, lo + stalls * stall_ft, stalls), ...) - the kerb the marked stalls ACTUALLY occupy.

    `stalls_per_span` says how many cars a span holds; this says how much of the span they cover,
    and the difference is the tail - up to one car short of a whole stall at the end of every run,
    plus the whole of any run too short to hold one at all.

    THE TAIL IS NOT PARKING AND IT IS NOT NOTHING. Drawn as a band it read as parking, and passed
    to `hatch_bands` as if it were marked it was excluded from the hatch as well: 176 ft of Broad
    St's far kerb was shaded blue AND left out of the "no room" gold, so a length too short to
    hold a car counted as parking twice. Whatever a caller shades, hatches or labels a run with
    comes off this, so the boxes, the leftover and the count are one walk.

    Off the same `lo + index * stall_ft` walk `stall_marks` draws its boundary lines on, so a
    footprint's ends are exactly the first and the last mark on that run.
    """
    return tuple((lo, lo + stalls * _stall_ft(stall_ft), stalls)
                 for lo, _hi, stalls in stalls_per_span(spans, stall_ft))


def stall_marks(corridor: "Corridor", side: str, spans, depth_ft: float | None = None,
                stall_ft: float | None = None):
    """(divider lines, stall count) - the stalls DRAWN, one mark per boundary between two cars.

    COUNTED BY DRAWING rather than by division: a quotient cannot see that a 30 ft stretch holds
    one car and wastes 8 ft. Here a stall exists when there is room to draw it, and the number
    returned is the number of boxes on the page.

    Marks run from the kerb inward by `depth_ft`, following the kerb rather than standing off the
    narrowest point - a parked car sits where the kerb is.
    """
    from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT

    depth_ft = PARKING_STALL_DEPTH_DEFAULT_FT if depth_ft is None else depth_ft
    stall_ft = _stall_ft(stall_ft)
    sign = 1.0 if side == "left" else -1.0
    marks, stalls = [], 0
    for lo, _hi, whole in stall_footprints(spans, stall_ft):
        stalls += whole
        for index in range(whole + 1):
            station = lo + index * stall_ft
            offset = kerb_offset_ft(corridor, side, station)
            if offset is None:
                continue
            inner = place_in_measured_frame(corridor.centerline, np.array([station]),
                                           np.array([sign * (offset - depth_ft)]))
            outer = place_in_measured_frame(corridor.centerline, np.array([station]),
                                           np.array([sign * offset]))
            marks.append(LineString([tuple(inner[0]), tuple(outer[0])]))
    return marks, stalls


#: BIKE LANE symbol placement (MUTCD Fig 9E-1). NACTO: after every driveway and intersection, and
#: at least every 500 ft along the lane. STANDARDS.md, verified 2026-08-18.
SYMBOL_INTERVAL_FT = 500.0
#: How far past an opening the reminder symbol sits - clear of the mouth itself, near enough to
#: read as belonging to it.
SYMBOL_AFTER_OPENING_FT = 15.0
#: Green conspicuity surfacing approaching and departing an intersection or driveway. NACTO gives
#: 20-50 ft; the low end is taken, because it is the figure that fits between the closely spaced
#: mouths on this corridor without the extensions merging into one continuous green.
GREEN_EXTENSION_FT = 20.0
#: The contraflow centreline's cadence, imported rather than restated - NACTO requires a dotted
#: YELLOW centreline along a bidirectional lane and in its crossbikes.
def _contraflow_cadence():
    from src.geometry.treatments.bikeways import CONTRAFLOW_DASH_FT, CONTRAFLOW_GAP_FT

    return CONTRAFLOW_DASH_FT, CONTRAFLOW_GAP_FT


def symbol_stations(paint: CorridorFacilityPaint) -> tuple:
    """Stations for the BIKE LANE symbol: after each opening, and every SYMBOL_INTERVAL_FT.

    Both rules, not either - NACTO asks for the reminder after every mouth AND a floor on the
    interval.
    """
    at = []
    for run in paint.runs:
        at.append(run.start_ft + SYMBOL_AFTER_OPENING_FT)
        station = run.start_ft + SYMBOL_INTERVAL_FT
        while station < run.end_ft:
            at.append(station)
            station += SYMBOL_INTERVAL_FT
        for _lo, hi in paint.dotted:
            if run.start_ft < hi < run.end_ft:
                at.append(hi + SYMBOL_AFTER_OPENING_FT)
    return tuple(sorted(s for s in at
                        if any(r.start_ft <= s <= r.end_ft for r in paint.runs)))


def green_extension_spans(paint: CorridorFacilityPaint, reach_ft: float | None = None) -> tuple:
    """Where the lane gets conspicuity green: reach_ft either side of every opening it crosses."""
    reach_ft = GREEN_EXTENSION_FT if reach_ft is None else reach_ft
    spans = [(lo - reach_ft, hi + reach_ft) for lo, hi in paint.dotted]
    spans += [(lo - reach_ft, hi + reach_ft) for lo, hi in paint.breaks]
    return tuple(sorted(spans))


def contraflow_centreline(corridor: "Corridor", paint: CorridorFacilityPaint):
    """The dotted yellow centreline down each run, CONTINUING across every opening.

    NACTO asks for the dotted yellow both along a bidirectional lane and in its crossbikes,
    so the mark that tells a driver two directions are present must not stop where they cross.
    """
    dash_ft, gap_ft = _contraflow_cadence()
    on = Alignment(corridor.centerline)
    sign = 1.0 if paint.side == "left" else -1.0
    out = []
    for run in paint.runs:
        if run.section is None:
            continue
        kerb = run.section.offsets_from_kerb_ft()
        middle = (kerb["bike_outer_ft"] + kerb["bike_inner_ft"]) / 2
        station = run.start_ft
        while station + dash_ft <= run.end_ft:
            offs = [kerb_offset_ft(corridor, paint.side, s) for s in (station, station + dash_ft)]
            if all(o is not None for o in offs):
                pts = place_in_measured_frame(
                    on.centerline, np.array([station, station + dash_ft]),
                    np.array([sign * (offs[0] - middle), sign * (offs[1] - middle)]))
                out.append(LineString([tuple(pts[0]), tuple(pts[1])]))
            station += dash_ft + gap_ft
    return out
