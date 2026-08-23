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

from src.geometry.model import (Alignment, band_from_offsets, line_from_offsets,
                                place_in_measured_frame, side_facing, whole_stalls_ft)
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

    The lane hugs the kerb (TwoWayBikeLane.hugs_kerb); the travel lane's edge is measured from
    the alignment; the buffer between them absorbs the difference. That split is
    bikeways.BikeLane.offsets_from_kerb_ft's, quoted rather than re-derived.
    """
    kerb = section.offsets_from_kerb_ft()
    on = Alignment(corridor.centerline)
    if not np.isfinite(offs).all():
        return None, "the kerb is not readable at every station of this stretch"
    lane_outer = offs - kerb["bike_outer_ft"]
    lane_inner = offs - kerb["bike_inner_ft"]
    travel_edge = lane_inner - section.buffer_ft
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


def parking_bands(corridor: "Corridor", facts, side: str, depth_ft: float | None = None):
    """[(lo_ft, hi_ft, polygon)] where a stall may legally be marked along one kerb.

    The spans are `CorridorFacts.parkable` - R.S. 39:4-138 applied along the whole road rather
    than to one junction's legs. This only gives them a footprint: a band `depth_ft` deep
    against the traced kerb, drawn against the KERB and not the alignment (a parked car sits
    against the kerb wherever the kerb happens to be).
    """
    from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT

    depth_ft = PARKING_STALL_DEPTH_DEFAULT_FT if depth_ft is None else depth_ft
    on = Alignment(corridor.centerline)
    out = []
    for lo, hi in facts.by_side("parkable", side):
        if hi - lo < CORRIDOR_SAMPLE_FT:
            continue
        stations = np.append(np.arange(lo, hi, CORRIDOR_SAMPLE_FT), hi)
        offs = np.array([kerb_offset_ft(corridor, side, float(s)) or np.nan for s in stations])
        if not np.isfinite(offs).all():
            continue
        band = band_from_offsets(on, side, stations, offs - depth_ft, offs)
        if band is not None and not band.is_empty:
            out.append((float(lo), float(hi), band))
    return out


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

    Taking width out of one kerbside pushes the divider toward the other, so the far kerb's
    lane edge is `travel_lane_divider_shift_ft` plus the lane's own width wherever the section
    is actually down - `divided_lane_width_ft`, NOT `TARGET_LANE_WIDTH_FT`: they agree only on a
    leg wide enough to hold two target-width lanes, and `divider.travel_lane_edge_ft`'s docstring
    names the leg that doesn't (w_broad_st_northeast, an 0.92 ft error) - the same reconstruction,
    repeated here, undercounted far-kerb room on every run that takes the equal split. Per run,
    because a constrained rung shifts the divider by a different amount.
    """
    from src.geometry.treatments.base import TARGET_LANE_WIDTH_FT
    from src.geometry.treatments.bikeways import divided_lane_width_ft, travel_lane_divider_shift_ft

    default_ft = TARGET_LANE_WIDTH_FT if default_ft is None else default_ft
    placed = [(run.start_ft, run.end_ft,
               travel_lane_divider_shift_ft(run.section) + divided_lane_width_ft(run.section))
              for run in paint.runs if run.section is not None]

    def at(station_ft: float) -> float:
        for lo, hi, edge_ft in placed:
            if lo <= station_ft <= hi:
                return edge_ft
        return default_ft

    return at


def stalls_per_span(spans, stall_ft: float | None = None):
    """((lo, hi, stalls), ...) - how many whole cars each span holds, and where.

    The per-span split exists so a drawing can LABEL each run with its own number: a boro reading
    a total has to be able to find that total on the page, run by run, or it is being asked to
    take it on trust. `stall_marks` counts through this, so the labels sum to the headline.
    """
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT

    stall_ft = PARKING_STALL_LENGTH_DEFAULT_FT if stall_ft is None else stall_ft
    # A span shorter than one car holds none, and gets no label pretending otherwise.
    return tuple((lo, hi, whole_stalls_ft(hi - lo, stall_ft)) for lo, hi in spans
                 if whole_stalls_ft(hi - lo, stall_ft) >= 1)


def stall_marks(corridor: "Corridor", side: str, spans, depth_ft: float | None = None,
                stall_ft: float | None = None):
    """(divider lines, stall count) - the stalls DRAWN, one mark per boundary between two cars.

    COUNTED BY DRAWING rather than by division: a quotient cannot see that a 30 ft stretch holds
    one car and wastes 8 ft. Here a stall exists when there is room to draw it, and the number
    returned is the number of boxes on the page.

    Marks run from the kerb inward by `depth_ft`, following the kerb rather than standing off the
    narrowest point - a parked car sits where the kerb is.
    """
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT
    from src.geometry.treatments.parking import PARKING_STALL_DEPTH_DEFAULT_FT

    depth_ft = PARKING_STALL_DEPTH_DEFAULT_FT if depth_ft is None else depth_ft
    stall_ft = PARKING_STALL_LENGTH_DEFAULT_FT if stall_ft is None else stall_ft
    sign = 1.0 if side == "left" else -1.0
    marks, stalls = [], 0
    for lo, _hi, whole in stalls_per_span(spans, stall_ft):
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
