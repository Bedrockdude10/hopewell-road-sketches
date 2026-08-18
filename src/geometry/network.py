"""A ROAD: one street, with continuous stationing - through one junction, or through a borough.

STEPS 1 AND 2 OF docs/network-model.md. This module builds Roads ALONGSIDE the existing legs and
renders nothing from them. Its first job is the checkpoint question that document asks: does the
traced kerb, read as one continuous road, reproduce the widths the per-leg model already produces?

Its second job (Corridor, below) is the one step 2 of that document asks for: put the CORRIDOR
questions on a road, because they are currently unanswerable inside the pipeline.

WHY A ROAD AND NOT TWO LEGS. A leg starts at the junction and runs outward, so a street through a
junction is two legs pointing away from each other, each with its own station 0 and its own frame.
A marking measured from one leg cannot be continued onto the other. A corridor question has no
object to ask. And a surveyed crossing at a junction this site does not model cannot be drawn at
all, because a crossing needs a station, an orientation and a reach to both kerbs, and all three
come from a leg.

A Road here is the two through legs' centrelines joined head-to-head. They are ALREADY joined
tangentially at the node by intersection/fitting.py:_join_through_legs, so this is a re-reading of
geometry the model has, not a second construction of it - which is what makes the comparison
meaningful rather than circular.

A `Corridor` is a chain of those Roads, bridged along the NJDOT alignment each end's leg was cut
from, with the traced kerb read side by side the whole way. It carries no station-ranged facts yet;
that is step 3.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from src.geometry.model import (Alignment, STRIP_SAMPLE_FT, curb_edge_by_station,
                                curb_offsets_at_stations, curb_station_span,
                                frame_at, is_through_street, leg_bearing_deg, line_direction,
                                place_in_measured_frame, station_offset_many, vertex_tangents)


@dataclass(frozen=True)
class Road:
    """One street through a junction: a continuous centreline and a kerb line per side.

    `near`/`far` name the two legs it was built from - `near` runs backwards along this road
    (station 0 is at its far end), `far` runs forwards. `node_ft` is the station of the junction.

    Sides are the ROAD's, not either leg's, and that is the point of the object: leg A's left kerb
    and leg B's right kerb are one physical kerb (see model.through_street_sides), so on the road
    they are one line with one name.
    """
    name: str
    centerline: LineString
    node_ft: float
    near_leg: str
    far_leg: str
    #: The frame's own attribute names (src/geometry/model/leg_frame.py:Alignment), not
    #: `left_kerb`/`right_kerb`. A Road IS an alignment - one centreline, one traced kerb per
    #: side - so under these names every frame function takes a Road unmodified, and moving the
    #: datum off the leg becomes a change of caller rather than a rewrite of the frame.
    left_curb: LineString | None = None
    right_curb: LineString | None = None

    @property
    def length_ft(self) -> float:
        return self.centerline.length

    def width_at_ft(self, station_ft: float) -> float | None:
        """Kerb to kerb at one station, or None where either side is untraced there.

        The road's own answer to the question `Leg.curb_to_curb_ft` answers per leg - and the
        comparison between the two is what tests/test_network.py checks.
        """
        offsets = [_kerb_offset_at(self.centerline, self.left_curb, "left", station_ft),
                   _kerb_offset_at(self.centerline, self.right_curb, "right", station_ft)]
        return None if any(o is None for o in offsets) else sum(offsets)


def _kerb_offset_at(centerline: LineString, kerb: LineString | None, side: str,
                     station_ft: float) -> float | None:
    """How far out one kerb sits at one station, unsigned, or None where it is not traced there.

    The single place a kerb is read in a road's frame. Refusing OUTSIDE THE TRACED SPAN is the
    load-bearing part: np.interp is happy to extend the first and last offset flat forever, so
    without the span test a corridor with 1,126 ft of untraced kerb would report a width across it.
    """
    if kerb is None:
        return None
    one = Alignment.one_sided(centerline, side, kerb)
    span = curb_station_span(one, side)
    if span is None or not (span[0] <= station_ft <= span[1]):
        return None
    at = curb_offsets_at_stations(one, side, np.array([station_ft]))
    if at is None or at[0] is None or not np.isfinite(at[0]):
        return None
    return abs(float(at[0]))


def _joined_centerline(near, far) -> LineString:
    """`near`'s centreline reversed, then `far`'s - one line running through the junction.

    Both legs start AT the node and run outward, so the near one is reversed to run inward. The
    shared node appears once: _join_through_legs has already given both legs the same first point,
    so the duplicate is dropped rather than left as a zero-length segment for `project` to trip on.

    EXCEPT WHERE IT HAS NOT. _blend_onto applies the shared junction point as a lateral offset
    profile, taking station_offset_many's offset and discarding its station, so it can slide a
    leg's end sideways onto the joint but never along the street. At W Broad & Louellen, 2.74 ft
    of the gap between the two NJDOT alignments is longitudinal and survives. Closing it here
    would be a second opinion about where the street is; one road built once has no joint to
    disagree about, which is the actual fix (task: retire _join_through_legs).
    """
    back = list(near.centerline.coords)[::-1]
    ahead = list(far.centerline.coords)
    if back and ahead and np.allclose(back[-1], ahead[0]):
        ahead = ahead[1:]
    return LineString(back + ahead)


def _joined_kerb(near, near_side: str, far, far_side: str) -> LineString | None:
    """One physical kerb, from the two legs' halves of it.

    near_side/far_side are opposite by construction - leg A's left is leg B's right - which is
    model.through_street_sides' pairing and the reason those two are one unbroken kerb with no
    corner in it.
    """
    a = getattr(near, f"{near_side}_curb", None)
    b = getattr(far, f"{far_side}_curb", None)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return LineString(list(a.coords)[::-1])
    coords = list(a.coords)[::-1] + list(b.coords)
    return LineString(coords)


def roads_from_model(model) -> list[Road]:
    """Every street that runs THROUGH this junction, as a Road.

    Only through pairs: a stem (Louellen at W Broad, Princeton at E Broad) is one leg and already
    has a leg frame that covers it. Reuses the same pairing
    intersection/fitting.py:_through_leg_pairs found when it joined the centrelines, rather than
    re-deriving it.
    """
    from src.geometry.intersection.fitting import _through_leg_pairs

    roads = []
    for name_a, name_b in _through_leg_pairs(model.legs):
        leg_a, leg_b = model.legs[name_a], model.legs[name_b]
        if not is_through_street(leg_a, leg_b):
            continue
        # `near` is whichever runs backwards along the finished road, chosen by bearing so the
        # station axis is stable.
        if leg_bearing_deg(leg_a) <= leg_bearing_deg(leg_b):
            near, near_name, far, far_name = leg_a, name_a, leg_b, name_b
        else:
            near, near_name, far, far_name = leg_b, name_b, leg_a, name_a
        street = (model.config["legs"].get(far_name, {}).get("street_name")
                  or model.config["legs"].get(near_name, {}).get("street_name")
                  or f"{near_name}/{far_name}")
        roads.append(Road(
            name=street,
            centerline=_joined_centerline(near, far),
            # The NEAR leg's own station 0, which is exact for it and out by the whole unclosed
            # gap for the far one - see _joined_centerline (task: retire _join_through_legs).
            node_ft=near.centerline.length,
            near_leg=near_name,
            far_leg=far_name,
            # The road's LEFT is the near leg's RIGHT: reversing the near leg's direction swaps
            # its sides.
            left_curb=_joined_kerb(near, "right", far, "left"),
            right_curb=_joined_kerb(near, "left", far, "right"),
        ))
    return roads


@dataclass(frozen=True)
class Approach:
    """One direction of one road, at one node. What a Leg NAMED, holding nothing a Leg HELD.

    docs/network-model.md says "there is no Leg. Not 'a leg becomes a view' - the object goes
    away". What goes away is a leg OWNING geometry. This owns none: a name, a road, a node and a
    direction, and every line, kerb, station and width it can be asked for is derived from the road
    on the spot.

    `forward` is whether this approach runs the way the road's stations increase. A road is built
    head-to-head from two legs, so exactly one of its two approaches runs against it, and that
    one's left kerb is the road's right (see roads_from_model, which pairs the sides that way).
    """
    name: str
    road: Road
    node_ft: float
    forward: bool

    def station_of(self, approach_ft: float) -> float:
        """Where a distance measured OUTWARD from the node falls on the road's own axis.

        The translation the migration turns on: everything that says "42 ft along broad_st_east"
        has to keep meaning the same place once the datum is the road.
        """
        return self.node_ft + (approach_ft if self.forward else -approach_ft)

    def outward_ft(self, road_station_ft: float) -> float:
        """The inverse: how far out from the node a road station is, along this approach.

        Negative behind the node, which is a real place - the far side of the junction - and not
        an error.
        """
        return (road_station_ft - self.node_ft) * (1.0 if self.forward else -1.0)

    def side_on_road(self, side: str) -> str:
        """This approach's left/right, named as the ROAD's left/right."""
        if self.forward:
            return str(side)
        return "right" if str(side) == "left" else "left"

    @property
    def span_ft(self) -> tuple[float, float]:
        """The road stations this approach covers, low to high."""
        return (self.node_ft, self.road.length_ft) if self.forward else (0.0, self.node_ft)

    @property
    def length_ft(self) -> float:
        lo, hi = self.span_ft
        return hi - lo

    @property
    def centerline(self) -> LineString:
        """The road's own centreline over this approach's span, running OUTWARD from the node.

        A view, cut on demand. Nothing caches it, and nothing may edit it: the moment an approach
        keeps its own copy of the line, the copy can disagree with the road.
        """
        lo, hi = self.span_ft
        cut = substring(self.road.centerline, lo, hi)
        return cut if self.forward else LineString(list(cut.coords)[::-1])

    def curb(self, side: str) -> LineString | None:
        """This approach's kerb on one side - the ROAD's kerb, cut at the node.

        Through curb_edge_by_station, so the kerb's own traced vertices are what comes back and
        only the two ends are interpolated. Cutting it any other way would resample the
        surveyor's line onto a grid and hand back this project's redrawing of it.
        """
        lo, hi = self.span_ft
        edge = curb_edge_by_station(self.road, self.side_on_road(side), lo, hi)
        if edge is None or len(edge) < 2:
            return None
        line = LineString(edge)
        return line if self.forward else LineString(list(line.coords)[::-1])

    @property
    def alignment(self) -> Alignment:
        """This approach as the frame reads it: a centreline and a kerb per side, outward."""
        return Alignment(self.centerline, left_curb=self.curb("left"),
                         right_curb=self.curb("right"))


def approaches_of(road: Road) -> tuple[Approach, ...]:
    """A road's two approaches at its node, named by the legs it was built from."""
    return (Approach(road.far_leg, road, road.node_ft, forward=True),
            Approach(road.near_leg, road, road.node_ft, forward=False))


def road_station_of_leg_station(road: Road, leg_name: str, leg_station_ft: float) -> float:
    """Where a station measured along one LEG falls on the road's own axis.

    Kept as the name 20 call sites will reach for during the migration; the arithmetic lives on
    Approach.
    """
    for approach in approaches_of(road):
        if approach.name == leg_name:
            return approach.station_of(leg_station_ft)
    raise KeyError(f"{leg_name} is not one of {road.name}'s two legs "
                   f"({road.near_leg}, {road.far_leg})")


# ---------------------------------------------------------------------------
# STEP 2 of docs/network-model.md: the CORRIDOR - one road across every junction on it.
# ---------------------------------------------------------------------------
#
# A per-junction Road stops at its two legs' far ends (260-300 ft). Every corridor question is
# about the 2,400 ft between Louellen St and Princeton Ave, and none of it can be asked of a 300 ft
# object.
#
# HOW THE ROAD IS EXTENDED. Inside each modelled junction the centreline is THE JUNCTION ROAD'S
# OWN, vertex for vertex: the checkpoint above compares the road's width reading against the leg's,
# and the two only agree if they share a frame. Beyond the legs there is no fitted centreline to
# inherit, so the road follows NJDOT's SRI alignment, eased laterally onto each modelled junction's
# centre at the seam over fitting.py's THROUGH_JOIN_BLEND_FT.
#
# WHAT IS NOT INVENTED. The kerb is the traced kerb and nothing else: where the tracing stops,
# `width_at_ft` returns None rather than interpolating across the gap (see _kerb_offset_at).

# How far out traced kerb is collected for a corridor. NOT the junction fetch's 120 m: Greenwood
# Ave to Louellen St is 413 m, so 120 m circles leave 173 m never fetched. 400 m is the widest
# radius whose window still fits every site's snapshot area (500 m fails at W Broad & Louellen).
# at any member junction covers the whole block to the next one.
CORRIDOR_KERB_RADIUS_M = 400

# How far past the outermost modelled leg the road is carried along NJDOT's alignment. The reach
# is TRIMMED to where the tracing stops; the cap is a second bound on top of that.
#
# IT IS THE FETCH RADIUS, not a round number. A flat 500 ft cap silently dropped 21 of 22 kerb
# ways northeast of Princeton Ave. CORRIDOR_KERB_RADIUS_M is the honest bound: past it no kerb
# was fetched, and the trim cannot catch "no kerb fetched" vs "no kerb traced".
CORRIDOR_EXTENSION_FT = CORRIDOR_KERB_RADIUS_M * 3.28084

# Two traced kerb ways whose station ranges come this close are one unbroken kerb. OSM splits a
# kerb wherever a tag changes - at every dropped kerb across a driveway - so adjacent ways share
# an endpoint and must not be read as two runs with a hole between them.
KERB_RUN_JOIN_FT = 2.0

# Two kerb samples closer together than this in station are the same place on the kerb; keeping
# both lets the offset table double back, which folds the edge over itself. Same figure and same
# reason as src/geometry/model/traced_kerbs.py:curb_line_from_points.
KERB_SAMPLE_MIN_GAP_FT = 0.25

#: Where a stretch of kerb came from. A modelled junction's kerb has been through the width fit
#: and carries this project's extrapolation to the leg's working length; a traced run is the
#: surveyor's own line and nothing else. Every coverage figure is counted over the traced runs
#: only, which is what stops a corridor being reported as better surveyed than it is.
KERB_FROM_JUNCTION = "modelled junction"
KERB_FROM_TRACING = "OSM tracing"


@dataclass(frozen=True)
class KerbRun:
    """One unbroken stretch of ONE side's kerb, in corridor stations.

    A LIST of these per side rather than one line per side. One LineString per side has to bridge
    every hole - a side street's mouth, a block nobody traced - and once bridged the hole is
    invisible: np.interp reads a straight chord across 1,126 ft of unmapped street as a kerb.
    Runs make the hole a hole.
    """
    side: str
    line: LineString
    start_ft: float
    end_ft: float
    source: str
    way_ids: tuple = ()

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)

    @property
    def is_traced(self) -> bool:
        return self.source == KERB_FROM_TRACING


@dataclass(frozen=True)
class JunctionOnRoad:
    """One modelled junction's stretch of a corridor, and the two legs it contributed.

    `legs` carries a SIGN per leg because a leg's own station 0 is at the node and it runs
    outward, so one of the two runs backwards along the corridor. That sign is the whole of the
    leg-to-corridor station translation, and it is exact: the corridor's centreline contains this
    junction Road's vertices verbatim, so arc length along one is arc length along the other.
    """
    site: str
    node_ft: float
    start_ft: float
    end_ft: float
    legs: tuple[tuple[str, float], ...]

    def station_of(self, leg_name: str, leg_station_ft: float) -> float:
        for name, sign in self.legs:
            if name == leg_name:
                return self.node_ft + sign * leg_station_ft
        raise KeyError(f"{leg_name} is not one of {self.site}'s legs on this road "
                       f"({', '.join(name for name, _ in self.legs)})")


@dataclass(frozen=True)
class Corridor:
    """One street across every junction this project models on it, with one station axis.

    The object every corridor question is asked of. It differs from `Road` in exactly two ways
    that matter: it spans several junctions (so a fact between them has somewhere to live), and
    its kerb is a list of RUNS rather than a line per side (so a stretch nobody traced is
    reported as untraced instead of being interpolated across).
    """
    name: str
    centerline: LineString
    junctions: tuple[JunctionOnRoad, ...]
    kerb_runs: tuple[KerbRun, ...]
    #: (start_ft, end_ft, SRI) - which NJDOT route carries which stretch. Not decoration: CR 518
    #: turns west onto Louellen St, so Broad Street carries two SRIs and any report that says
    #: "SRI 00000518__" of the whole thing is wrong about the western third of it.
    sri_spans: tuple[tuple[float, float, str], ...] = ()
    #: Every OTHER street's centre station along this road, modelled or not. Carried because a
    #: hole in the kerb AT one of these is an intersection mouth - there is no kerb across a
    #: street - and a hole anywhere else is tracing nobody has done. Reporting both as "untraced"
    #: sends a surveyor to re-trace something already correct, and understates the road's coverage.
    cross_street_ft: tuple[float, ...] = ()
    #: (station, lateral gap in ft) at each seam between a modelled junction and NJDOT's
    #: alignment. Reported rather than smoothed away silently - see _eased_alignment.
    seams: tuple[tuple[float, float], ...] = ()

    @property
    def length_ft(self) -> float:
        return self.centerline.length

    @property
    def sites(self) -> tuple[str, ...]:
        return tuple(j.site for j in self.junctions)

    @property
    def leg_names(self) -> tuple[str, ...]:
        return tuple(name for j in self.junctions for name, _sign in j.legs)

    def station_of(self, leg_name: str, leg_station_ft: float) -> float:
        """Where a station measured along one modelled LEG falls on the corridor's axis.

        The corridor's `road_station_of_leg_station`, and the same reason for existing: a
        measurement recorded as "42 ft along broad_st_east" has to keep meaning the same place
        when the datum becomes a road 3,500 ft long.
        """
        for junction in self.junctions:
            if any(name == leg_name for name, _sign in junction.legs):
                return junction.station_of(leg_name, leg_station_ft)
        raise KeyError(f"{leg_name} is not a leg on {self.name} ({', '.join(self.leg_names)})")

    def kerb_run_at(self, side: str, station_ft: float) -> KerbRun | None:
        """The run governing one side at one station, preferring a modelled junction's kerb.

        The junction's kerb wins where both cover a station because it is the assignment the
        width fit settled (intersection/fitting.py:_fit_legs_to_traced_kerbs) - deciding again,
        in the corridor frame, would be the second definition docs/network-renderer-plan.md
        forbids, and it is what shifted the throat readings by up to 2.8 ft.
        """
        best = None
        for run in self.kerb_runs:
            if run.side != side or not (run.start_ft <= station_ft <= run.end_ft):
                continue
            if best is None or (best.is_traced and not run.is_traced):
                best = run
        return best

    def width_at_ft(self, station_ft: float) -> float | None:
        """Kerb to kerb at one station, or None where either side has no kerb line to read.

        `Road.width_at_ft` for a road with holes in it. Answering None rather than a plausible
        number is the point: see unmeasurable_gaps_ft for where that happens, and untraced_gaps_ft
        for the wider set of stations where a width is reported but is not a survey.
        """
        offsets = []
        for side in ("left", "right"):
            run = self.kerb_run_at(side, station_ft)
            if run is None:
                return None
            offset = _kerb_offset_at(self.centerline, run.line, side, station_ft)
            if offset is None:
                return None
            offsets.append(offset)
        return sum(offsets)

    def traced_spans(self, side: str) -> tuple[tuple[float, float], ...]:
        """The stations where a surveyor's kerb is traced on one side, as disjoint spans.

        Counted over the TRACED runs only, so a modelled junction's extrapolation to its leg's
        working length is not reported as coverage. This is the denominator every count in
        scripts/corridor_report.py is printed beside.
        """
        return _merged_spans([(run.start_ft, run.end_ft) for run in self.kerb_runs
                              if run.side == side and run.is_traced])

    def both_traced_spans(self) -> tuple[tuple[float, float], ...]:
        """Where BOTH kerbs are traced - the only stations at which a width is a measurement."""
        return _intersect_spans(self.traced_spans("left"), self.traced_spans("right"))

    def traced_ft(self, side: str) -> float:
        return sum(hi - lo for lo, hi in self.traced_spans(side))

    @property
    def both_traced_ft(self) -> float:
        return sum(hi - lo for lo, hi in self.both_traced_spans())

    def untraced_gaps_ft(self, min_ft: float = 0.0) -> tuple[tuple[float, float], ...]:
        """The holes in `both_traced_spans` - the stretches with no SURVEYED width.

        Named `gaps` and returned rather than papered over because the honest answer on Broad St
        is that there is a 1,126 ft one.

        Not the same as `unmeasurable_gaps_ft`: inside a modelled junction the leg's own kerb line
        reaches its working length whether or not the tracing does, so a width IS reported there.
        That is the per-leg model's answer; it is not a survey, so it is not counted as coverage.
        """
        return tuple((lo, hi) for lo, hi in _complement_spans(self.both_traced_spans(),
                                                              0.0, self.length_ft)
                     if hi - lo >= min_ft)

    def measurable_spans(self) -> tuple[tuple[float, float], ...]:
        """Where both sides have SOME kerb to read - traced, or a modelled junction's own line."""
        return _intersect_spans(
            *[_merged_spans([(run.start_ft, run.end_ft) for run in self.kerb_runs
                             if run.side == side]) for side in ("left", "right")])

    def unmeasurable_gaps_ft(self, min_ft: float = 0.0) -> tuple[tuple[float, float], ...]:
        """The stretches where `width_at_ft` answers None because there is nothing to read."""
        return tuple((lo, hi) for lo, hi in _complement_spans(self.measurable_spans(),
                                                              0.0, self.length_ft)
                     if hi - lo >= min_ft)

    def narrowest_width_ft(self, sample_ft: float = STRIP_SAMPLE_FT) -> tuple[float, float] | None:
        """(width, station) at the tightest measurable cross-section, or None if there is none.

        Only where both kerbs are traced, because anywhere else the number would be an
        extrapolation dressed as a measurement.
        """
        best = None
        for lo, hi in self.both_traced_spans():
            n = max(int(np.ceil((hi - lo) / sample_ft)) + 1, 2)
            for station in np.linspace(lo, hi, n):
                width = self.width_at_ft(float(station))
                if width is not None and (best is None or width < best[0]):
                    best = (width, float(station))
        return best


def _merged_spans(spans) -> tuple[tuple[float, float], ...]:
    """Overlapping or touching (lo, hi) pairs collapsed into disjoint ones, in order."""
    out: list[list[float]] = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1] + KERB_RUN_JOIN_FT:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((lo, hi) for lo, hi in out)


def _intersect_spans(a, b) -> tuple[tuple[float, float], ...]:
    """The stretches both span lists cover. Sorted, so a figure derived from it is reproducible.

    Assumes each input is already disjoint (which _merged_spans guarantees), so the result is
    disjoint too and its total length can be summed without double-counting.
    """
    return tuple(sorted((lo, hi) for lo, hi in
                        ((max(x[0], y[0]), min(x[1], y[1])) for x in a for y in b) if hi > lo))


def _complement_spans(spans, lo: float, hi: float) -> tuple[tuple[float, float], ...]:
    out, cursor = [], lo
    for start, end in _merged_spans(spans):
        if start > cursor:
            out.append((cursor, min(start, hi)))
        cursor = max(cursor, end)
    if cursor < hi:
        out.append((cursor, hi))
    return tuple((a, b) for a, b in out if b > a)


# Compass words a street name may start with. Stripped when a corridor is named, because the whole
# point of the object is that "East Broad Street" and "West Broad Street" are one street: NJDOT
# splits CR 518 at the Route 569 signal, OSM splits it at Greenwood Ave, and neither split is a
# fact about the road that a corridor report should inherit.
_COMPASS_WORDS = frozenset({"north", "south", "east", "west", "n", "s", "e", "w",
                            "northeast", "northwest", "southeast", "southwest",
                            "ne", "nw", "se", "sw"})


def _street_name(raw: str) -> str:
    """"West Broad Street (west of Greenwood Ave) - CR 518" -> "Broad Street"."""
    head = raw.split("(")[0].split(" - ")[0].strip().rstrip(",")
    words = head.split()
    while words and words[0].lower().strip(".") in _COMPASS_WORDS:
        words = words[1:]
    return " ".join(words) or head


def _corridor_name(raw_names) -> str:
    """The street name most of a corridor's legs agree on, compass halves collapsed.

    Not the longest common prefix, which is empty for {"East Broad Street", "West Broad Street"}
    and would have named this corridor after nothing.
    """
    from collections import Counter

    counted = Counter(_street_name(name) for name in raw_names if name)
    return counted.most_common(1)[0][0] if counted else "unnamed street"


def _junction_road_ends(models) -> list[dict]:
    """Every per-junction Road, listed once per end, with what an end needs to be linked up.

    An "end" is a road plus one of its two legs. Two ends are the same street continuing if they
    are on the same SRI, point back at each other, and each junction lies ahead of the other's leg.
    """
    out = []
    for site, model in models.items():
        for road in roads_from_model(model):
            key = (site, road.near_leg, road.far_leg)
            for which in ("near", "far"):
                leg_name = getattr(road, f"{which}_leg")
                out.append({"key": key, "site": site, "road": road, "leg": leg_name,
                            "which": which, "leg_obj": model.legs[leg_name],
                            "sri": model.config["legs"][leg_name].get("sri"),
                            "outward": line_direction(model.legs[leg_name].centerline),
                            "centre": model.center_ft})
    return out


def _linked_ends(ends: list[dict]) -> list[tuple[dict, dict]]:
    """Which junction-road ends face each other across a block, NEAREST FIRST.

    Nearest first and one link per end, which is what stops a corridor skipping a junction.
    """
    candidates = []
    for i, a in enumerate(ends):
        for b in ends[i + 1:]:
            if a["site"] == b["site"] or a["sri"] is None or a["sri"] != b["sri"]:
                continue
            # The same test that decides a through street at a junction, asked between two
            # junctions: two legs more than THROUGH_STREET_ANGLE_DEG apart are one street.
            if not is_through_street(a["leg_obj"], b["leg_obj"]):
                continue
            gap = np.array([b["centre"].x - a["centre"].x, b["centre"].y - a["centre"].y])
            if np.dot(gap, a["outward"]) <= 0 or np.dot(-gap, b["outward"]) <= 0:
                continue
            candidates.append((float(np.hypot(*gap)), a, b))
    candidates.sort(key=lambda candidate: candidate[0])
    taken, links = set(), []
    for _distance, a, b in candidates:
        here, there = (a["key"], a["leg"]), (b["key"], b["leg"])
        if here in taken or there in taken:
            continue
        taken |= {here, there}
        links.append((a, b))
    return links


def _chains(ends: list[dict], links: list[tuple[dict, dict]]) -> list[list[tuple]]:
    """[(road key, first leg, second leg)] per street, ordered along it.

    A junction road with no link is its own chain of one - and still becomes a Corridor, because
    extending a single junction along NJDOT's alignment to where the tracing stops is exactly as
    useful there as it is on a three-junction street.
    """
    roads = {}
    for end in ends:
        roads.setdefault(end["key"], {})[end["which"]] = end
    linked: dict[tuple, dict[str, tuple]] = {}
    for a, b in links:
        linked.setdefault(a["key"], {})[a["leg"]] = (b["key"], b["leg"])
        linked.setdefault(b["key"], {})[b["leg"]] = (a["key"], a["leg"])

    walks, seen = [], set()
    # Chain ENDS first, so a walk never starts in the middle and produces two half-chains.
    for start in sorted(roads, key=lambda key: len(linked.get(key, {}))):
        if start in seen:
            continue
        walk = [start]
        seen.add(start)
        for _direction in range(2):
            while True:
                nxt = next((key for _leg, (key, _far) in linked.get(walk[-1], {}).items()
                            if key not in seen), None)
                if nxt is None:
                    break
                seen.add(nxt)
                walk.append(nxt)
            walk.reverse()
        walks.append(walk)

    chains = []
    for walk in walks:
        items = []
        for i, key in enumerate(walk):
            entry = next((leg for leg, (other, _l) in linked.get(key, {}).items()
                          if i and other == walk[i - 1]), None)
            exit_ = next((leg for leg, (other, _l) in linked.get(key, {}).items()
                          if i + 1 < len(walk) and other == walk[i + 1]), None)
            near, far = key[1], key[2]
            first = entry or (near if exit_ != near else far)
            items.append((key, first, far if first == near else near))
        chains.append(_oriented_chain(items, roads))
    return chains


def _oriented_chain(items: list[tuple], roads: dict) -> list[tuple]:
    """The chain reversed if it runs east to west, so every corridor is stationed west to east.

    An arbitrary but FIXED choice: the station axis is what every corridor figure is reported
    against. West to east is also NJDOT's own direction on these routes.
    """
    def easting(item, end):
        key, first, second = item
        leg = first if end == 0 else second
        return roads[key][next(w for w in ("near", "far")
                               if roads[key][w]["leg"] == leg)]["leg_obj"].centerline.coords[-1][0]

    if easting(items[0], 0) <= easting(items[-1], 1):
        return items
    return [(key, second, first) for key, first, second in reversed(items)]


def _sri_alignment(models, sri: str):
    """NJDOT's own centreline for one SRI, in state-plane feet, or None if it is not in the layer.

    ONE LineString for the whole route: SRI 00000518__ comes back as a single 108,645 ft feature,
    so the corridor between two junctions is a substring of a line that already exists.
    """
    from src.geometry.intersection.junction import ROOT_DIR
    from src.geometry.model import buffer_point_wgs84, reproject_to_state_plane
    from src.sources.data_loader import load_road_network

    first = next(iter(models.values()))
    path = ROOT_DIR / first.config["data_sources"]["road_network"]
    boxes = [buffer_point_wgs84(model.center_wgs84, _ALIGNMENT_BBOX_MARGIN_M)
             for model in models.values()]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    rows = load_road_network(bbox=bbox, path=path)
    rows = rows[rows["SRI"] == sri]
    if rows.empty:
        return None
    return reproject_to_state_plane(rows).iloc[0].geometry


# How far around the member junctions the road-network read is bounded. Only a bbox FILTER - an
# NJDOT feature comes back whole, so this decides which routes are seen, not how much of one.
_ALIGNMENT_BBOX_MARGIN_M = 400


def _ease(t: np.ndarray) -> np.ndarray:
    """1 at t=0 falling to 0 at t=1, flat at both ends. Hermite, as in fitting.py:_blend_onto."""
    return 2 * t ** 3 - 3 * t ** 2 + 1


def _alignment_stations(align: LineString, lo: float, hi: float) -> np.ndarray:
    """The station grid a stretch of NJDOT alignment is re-laid on.

    Carries the alignment's OWN vertices, so its shape survives the lateral correction - the same
    reason fitting.py:_traced_centre_profile returns its correction on a grid that includes them,
    and the same thinning afterwards, because two vertices an inch apart turn a hundredth of a
    foot of correction into a 34 degree turn.
    """
    from src.geometry.intersection import CENTRE_SAMPLE_FT, MIN_CENTRE_VERTEX_GAP_FT

    own, _offsets = station_offset_many(align, np.asarray(align.coords, dtype=float))
    grid = np.unique(np.concatenate([
        np.arange(lo, hi, CENTRE_SAMPLE_FT), [hi],
        # Two extra stations a vertex-gap in from each seam. Without them the first bridge vertex
        # sits a full sample out, where the eased correction has already given up 7% of the seam
        # gap - a 3.4 degree kink in the frame right where a junction's kerb readings end.
        [lo + MIN_CENTRE_VERTEX_GAP_FT, hi - MIN_CENTRE_VERTEX_GAP_FT],
        own[(own > lo) & (own < hi)]]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    kept = [float(grid[0])]
    for station in grid[1:-1]:
        if station - kept[-1] >= MIN_CENTRE_VERTEX_GAP_FT:
            kept.append(float(station))
    kept.append(float(grid[-1]))
    return np.asarray(kept)


def _eased_alignment(align: LineString, lo: float, hi: float, off_lo: float, off_hi: float,
                     blend_ft: float) -> np.ndarray:
    """A stretch of NJDOT alignment, moved sideways at its ends to meet what it joins.

    THE ONE PLACE THIS MODULE MOVES SURVEYED GEOMETRY, and it moves NJDOT's alignment rather than
    anybody's tracing. An SRI line is a linear-referencing reference, not a carriageway centre, and
    the modelled junctions' fitted centres sit 4.6-10.9 ft off it here. Butt-jointed, the road
    would jog sideways at every leg end.

    The correction is the MEASURED seam gap, eased to zero over blend_ft. Same mechanism as
    fitting.py:_join_through_legs; the gaps are reported on Corridor.seams rather than absorbed.
    """
    stations = _alignment_stations(align, lo, hi)
    blend = min(blend_ft, (hi - lo) / 2)
    if blend <= 0:
        return np.asarray(align.coords, dtype=float)[:0]
    delta = (off_lo * _ease(np.clip((stations - lo) / blend, 0.0, 1.0))
             + off_hi * _ease(np.clip((hi - stations) / blend, 0.0, 1.0)))
    return np.asarray(place_in_measured_frame(align, stations, delta), dtype=float)


def _seam(align: LineString, point) -> tuple[float, float]:
    """(station, signed offset) of a piece's end on the alignment it is about to be joined to."""
    stations, offsets = station_offset_many(align, np.asarray([point], dtype=float))
    return float(stations[0]), float(offsets[0])


def _tracing_reach_ft(align: LineString, seam_ft: float, forward: bool, kerb_ways,
                      max_ft: float) -> float:
    """How far past a seam the traced kerb continues along the alignment, capped at max_ft.

    Measured against NJDOT's alignment rather than the finished corridor, because the corridor
    cannot be built until its length is known. The reach only ever shortens the road, so an error
    here cannot invent street.
    """
    from src.geometry.intersection import KERB_PLAUSIBLE_HALF_WIDTH_FT

    near, far = KERB_PLAUSIBLE_HALF_WIDTH_FT
    lo = seam_ft if forward else max(seam_ft - max_ft, 0.0)
    hi = min(seam_ft + max_ft, align.length) if forward else seam_ft
    reach = 0.0
    for line, _tags in kerb_ways.values():
        stations, offsets = station_offset_many(align, _dense_kerb_points(line))
        beside = (np.abs(offsets) >= near) & (np.abs(offsets) <= far)
        inside = beside & (stations >= lo) & (stations <= hi)
        if inside.any():
            beyond = stations[inside] - seam_ft if forward else seam_ft - stations[inside]
            reach = max(reach, float(beyond.max()))
    return min(reach, max_ft)


def _dense_kerb_points(line: LineString) -> np.ndarray:
    """A traced kerb resampled ALONG itself - see context_roads.py:kerb_points for why.

    A straight run of kerb is mapped with two vertices, so reading vertices alone would miss kerb
    at most stations. Same spacing constant, imported rather than repeated.
    """
    from src.geometry.context_roads import kerb_points

    return kerb_points([line])


def _oriented_piece(road: Road, first_leg: str) -> dict:
    """One junction Road turned to run the corridor's way, with its sides renamed to match.

    Reversing a road swaps its left and right - the same trap roads_from_model's own side pairing
    warns about.
    """
    if road.near_leg == first_leg:
        return {"centerline": road.centerline, "left": road.left_curb, "right": road.right_curb,
                "node_from_start_ft": road.node_ft,
                "legs": ((road.near_leg, -1.0), (road.far_leg, 1.0))}
    def flip(line):
        return None if line is None else LineString(list(line.coords)[::-1])

    return {"centerline": flip(road.centerline), "left": flip(road.right_curb),
            "right": flip(road.left_curb),
            "node_from_start_ft": road.length_ft - road.node_ft,
            "legs": ((road.far_leg, -1.0), (road.near_leg, 1.0))}


def _cumulative_ft(coords) -> np.ndarray:
    steps = np.hypot(*np.diff(np.asarray(coords, dtype=float), axis=0).T)
    return np.concatenate(([0.0], np.cumsum(steps)))


def corridors_from_models(models) -> list[Corridor]:
    """Every street these junction models share, as one road each.

    `models` is {site: IntersectionModel} - the shape tests/conftest.py's site_models fixture
    already has. Broad St, E Broad St and W Broad St come back as ONE Corridor spanning all three
    junctions on it; Greenwood Ave, Columbia Ave and Princeton Ave each come back as a corridor of
    one junction, extended along their own alignments, because this project models only one
    junction on each of them.
    """
    ends = _junction_road_ends(models)
    roads_by_key = {end["key"]: end["road"] for end in ends}
    corridors = [_build_corridor(models, chain, roads_by_key)
                 for chain in _chains(ends, _linked_ends(ends))]
    return [corridor for corridor in corridors if corridor is not None]


def _build_corridor(models, chain: list[tuple], roads_by_key: dict) -> Corridor | None:
    """Assemble one chain into a Corridor: pieces, bridges, extensions, kerb runs.

    The order here is the order the geometry depends on. The centreline has to exist before a kerb
    can be placed in its frame, and the extensions' length depends on where the tracing stops - so
    the reach is measured against NJDOT's alignment first (see _tracing_reach_ft) and the road is
    built once, at its final length, rather than built long and trimmed.
    """
    from src.geometry.intersection import THROUGH_JOIN_BLEND_FT

    pieces, sris = [], []
    for key, first, second in chain:
        site = key[0]
        model = models[site]
        pieces.append({"site": site, **_oriented_piece(roads_by_key[key], first)})
        sris.append((model.config["legs"][first].get("sri"),
                     model.config["legs"][second].get("sri")))
    kerb_ways = _corridor_kerb_ways(models)

    coords: list[tuple] = []
    seam_marks: list[tuple[int, float]] = []      # (index into coords, lateral gap in ft)

    first = np.asarray(pieces[0]["centerline"].coords[:2], dtype=float)
    head, head_gap = _extension(models, kerb_ways, first[0], first[0] - first[1], sris[0][0],
                                THROUGH_JOIN_BLEND_FT)
    if len(head):
        coords.extend(tuple(point) for point in head[::-1][:-1])

    for i, piece in enumerate(pieces):
        start = len(coords)
        # Only the HEAD seam is recorded here. Every other piece is preceded by a bridge, which
        # records both of its own seams below; recording one again from this side put a spurious
        # zero-gap entry beside each real one and made a 5-seam road report 7.
        if i == 0 and start:
            seam_marks.append((start, head_gap))
        coords.extend(piece["centerline"].coords)
        piece["index"] = (start, len(coords) - 1)
        if i + 1 >= len(pieces):
            continue
        here = np.asarray(piece["centerline"].coords[-1], dtype=float)
        there = np.asarray(pieces[i + 1]["centerline"].coords[0], dtype=float)
        align = _sri_alignment(models, sris[i][1])
        if align is None:
            return None
        (station_a, offset_a), (station_b, offset_b) = _seam(align, here), _seam(align, there)
        forward = station_a < station_b
        lo, hi = sorted((station_a, station_b))
        bridge = _eased_alignment(align, lo, hi,
                                  offset_a if forward else offset_b,
                                  offset_b if forward else offset_a, THROUGH_JOIN_BLEND_FT)
        if not forward:
            bridge = bridge[::-1]
        seam_marks.append((len(coords) - 1, abs(offset_a)))
        coords.extend(tuple(point) for point in bridge[1:-1])
        seam_marks.append((len(coords), abs(offset_b)))

    last = np.asarray(pieces[-1]["centerline"].coords[-2:], dtype=float)
    tail, tail_gap = _extension(models, kerb_ways, last[1], last[1] - last[0], sris[-1][1],
                                THROUGH_JOIN_BLEND_FT)
    if len(tail):
        seam_marks.append((len(coords) - 1, tail_gap))
        coords.extend(tuple(point) for point in tail[1:])

    if len(coords) < 2:
        return None
    centerline = LineString(coords)
    stations = _cumulative_ft(coords)
    junctions = tuple(JunctionOnRoad(
        site=piece["site"],
        node_ft=float(stations[piece["index"][0]]) + piece["node_from_start_ft"],
        start_ft=float(stations[piece["index"][0]]),
        end_ft=float(stations[piece["index"][1]]),
        legs=piece["legs"]) for piece in pieces)
    junction_ft = tuple(j.node_ft for j in junctions)
    junction_runs = _junction_kerb_runs(pieces, stations)

    def corridor_with(runs, cross_street_ft=()) -> Corridor:
        return Corridor(
            name=_corridor_name([models[piece["site"]].config["legs"][leg].get("street_name")
                                 for piece in pieces for leg, _sign in piece["legs"]]),
            centerline=centerline, junctions=junctions, kerb_runs=tuple(runs),
            sri_spans=_sri_spans(junctions, sris, centerline.length),
            cross_street_ft=tuple(cross_street_ft),
            seams=tuple((float(stations[index]), gap) for index, gap in seam_marks))

    # TWICE, DELIBERATELY, and the second pass is the whole point of the first.
    #
    # _kerb_samples_on suspends the heading test near a node (corner returns sweep 90 degrees) but
    # only near a MODELLED junction - and a corridor crosses far more streets than this project
    # models. At the other 8 on Broad St, traced corner returns were discarded as "too skewed",
    # producing 34-48 ft untraced gaps 0-27 ft from a cross street.
    #
    # Cross streets are resolved FROM a corridor, so the first pass builds one good enough to
    # locate the mouths, and the second rebuilds the traced runs with every junction on the road
    # counted as a node.
    provisional = corridor_with(junction_runs + _traced_kerb_runs(centerline, kerb_ways,
                                                                  junction_ft))
    crossing_ft = tuple(sorted({round(cross.station_ft, 1)
                                for cross in _cross_streets_on(provisional, models)}))
    return corridor_with(
        junction_runs + _traced_kerb_runs(centerline, kerb_ways, junction_ft + crossing_ft),
        cross_street_ft=crossing_ft)


def _extension(models, kerb_ways, seam_point, away, sri: str,
               blend_ft: float) -> tuple[np.ndarray, float]:
    """NJDOT's alignment carried AWAY from the end of a chain, as far as the tracing continues.

    `away` is the direction the road is leaving in, and it is needed because which way that is
    along NJDOT's line depends on how the route was digitised - CR 518 is "West to East", CR 654
    need not be. Returns (points running away from the seam, the lateral seam gap).
    """
    align = _sri_alignment(models, sri)
    if align is None:
        return np.empty((0, 2)), 0.0
    seam_ft, offset_ft = _seam(align, seam_point)
    _origin, tangent = frame_at(align, seam_ft)
    forward = bool(np.dot(tangent, np.asarray(away, dtype=float)) > 0)
    reach = _tracing_reach_ft(align, seam_ft, forward, kerb_ways, CORRIDOR_EXTENSION_FT)
    lo = max(seam_ft, 0.0) if forward else max(seam_ft - reach, 0.0)
    hi = min(seam_ft + reach, align.length) if forward else min(seam_ft, align.length)
    if hi - lo <= blend_ft:
        # Shorter than the blend is not an extension, it is a stub of eased correction with no
        # NJDOT alignment left in the middle of it. Better to end the road at the modelled leg.
        return np.empty((0, 2)), abs(offset_ft)
    points = _eased_alignment(align, lo, hi, offset_ft if forward else 0.0,
                              0.0 if forward else offset_ft, blend_ft)
    if len(points) < 2:
        return np.empty((0, 2)), abs(offset_ft)
    seam = np.asarray(seam_point, dtype=float)
    if np.hypot(*(points[0] - seam)) > np.hypot(*(points[-1] - seam)):
        points = points[::-1]
    return points, abs(offset_ft)


def _sri_spans(junctions, sris, length_ft: float) -> tuple[tuple[float, float, str], ...]:
    """Which NJDOT route carries which stretch, split at every node where the SRI changes.

    Broad Street is the case this exists for: CR 518 turns west onto Louellen St, so W Broad
    southwest of Louellen is CR 654.
    """
    spans, cursor = [], 0.0
    for junction, (sri_in, _sri_out) in zip(junctions, sris):
        spans.append((cursor, junction.node_ft, sri_in))
        cursor = junction.node_ft
    spans.append((cursor, length_ft, sris[-1][1]))
    merged = []
    for lo, hi, sri in spans:
        if merged and merged[-1][2] == sri:
            merged[-1] = (merged[-1][0], hi, sri)
        else:
            merged.append((lo, hi, sri))
    return tuple((lo, hi, sri) for lo, hi, sri in merged if hi > lo)


def _corridor_kerb_ways(models) -> dict:
    """{OSM way id: (LineString in feet, tags)} for every traced kerb near any member junction.

    Keyed by way id and unioned across the members, so a kerb traced as one way down a whole block
    is read once. Fetched at CORRIDOR_KERB_RADIUS_M rather than the junction radius - see the
    constant.
    """
    from src.geometry.intersection import to_state_plane
    from src.sources.osm_context import fetch_kerbs

    ways = {}
    for model in models.values():
        for kerb in fetch_kerbs(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            coords = kerb.get("coords_wgs84")
            if not coords or len(coords) < 2:
                continue          # a lone barrier=kerb NODE has no line to read
            ways[kerb["id"]] = (LineString(to_state_plane(coords)), kerb.get("tags", {}))
    return ways


def _junction_kerb_runs(pieces: list[dict], stations: np.ndarray) -> list[KerbRun]:
    """Each modelled junction's own kerb, clipped to its own stretch of the corridor.

    MEASURED IN THE JUNCTION ROAD'S FRAME, then read back in the corridor's. A traced kerb runs
    PAST the leg it belongs to - e_broad_st_east's last vertex is 141 ft out on a 130 ft leg -
    and a point beyond the end of a centreline is stationed by extrapolating that centreline's last
    segment. So the same surveyed vertex reads as station 141.3 offset 17.97 to the leg and 143.7
    offset 17.07 to the corridor. Clipping in the road's own frame removes it.

    CLIPPED WITH INTERPOLATED ENDS, not filtered by vertex: dropping the vertex beyond the stretch
    takes away the anchor for everything between the last vertex inside it and the boundary.
    """
    runs = []
    for piece in pieces:
        base = float(stations[piece["index"][0]])
        for side in ("left", "right"):
            kerb = piece[side]
            if kerb is None:
                continue
            one = Alignment.one_sided(piece["centerline"], side, kerb)
            span = curb_station_span(one, side)
            if span is None or span[1] - span[0] < STRIP_SAMPLE_FT:
                continue
            edge = curb_edge_by_station(one, side, span[0], span[1])
            if edge is None or len(edge) < 2:
                continue
            runs.append(KerbRun(side=side, line=LineString(edge), start_ft=base + span[0],
                                end_ft=base + span[1], source=KERB_FROM_JUNCTION))
    return runs


def _kerb_samples_on(centerline: LineString, node_stations, line: LineString) -> tuple:
    """(stations, offsets, points, keep) for one traced way read as THIS road's kerb.

    THREE tests decide whether a sample belongs to this road, each the one the per-leg fit already
    uses:

      * IN THE ROAD - station inside [0, length].
      * BESIDE IT - |offset| inside KERB_PLAUSIBLE_HALF_WIDTH_FT (8-45 ft).
      * ALONG IT - the kerb's own heading within CURB_POINT_MAX_SKEW_DEG of the road's, SUSPENDED
        within CURB_POINT_CORNER_ZONE_FT of a modelled node (corner returns sweep 90 degrees).
    """
    from src.geometry.intersection import KERB_PLAUSIBLE_HALF_WIDTH_FT
    from src.geometry.model import CURB_POINT_CORNER_ZONE_FT, CURB_POINT_MAX_SKEW_DEG

    near, far = KERB_PLAUSIBLE_HALF_WIDTH_FT
    points = _dense_kerb_points(line)
    if len(points) < 2:
        return np.empty(0), np.empty(0), points, np.zeros(len(points), bool)
    stations, offsets = station_offset_many(centerline, points)
    tangents = vertex_tangents(LineString(points))
    road_dirs = np.asarray([frame_at(centerline, float(station))[1] for station in stations])
    along = (np.abs(np.einsum("ij,ij->i", tangents, road_dirs))
             >= np.cos(np.radians(CURB_POINT_MAX_SKEW_DEG)))
    nodes = np.asarray(node_stations, dtype=float)
    in_corner = (np.abs(stations[:, None] - nodes[None, :]).min(axis=1)
                 <= CURB_POINT_CORNER_ZONE_FT) if len(nodes) else np.zeros(len(stations), bool)
    keep = ((stations >= 0.0) & (stations <= centerline.length)
            & (np.abs(offsets) >= near) & (np.abs(offsets) <= far) & (along | in_corner))
    return stations, offsets, points, keep


def _traced_kerb_runs(centerline: LineString, kerb_ways: dict,
                      node_stations: tuple) -> list[KerbRun]:
    """The surveyor's kerb along the whole corridor, as one run per unbroken stretch.

    A way with samples on both sides contributes to both, per sample: OSM ways are drawn to suit
    the mapper and one of them can cover both kerbs of a street.
    """
    stretches: dict[str, list[dict]] = {"left": [], "right": []}
    for way_id, (line, _tags) in sorted(kerb_ways.items()):
        stations, offsets, points, keep = _kerb_samples_on(centerline, node_stations, line)
        if keep.sum() < 2:
            continue
        for side, on_side in (("left", offsets > 0), ("right", offsets <= 0)):
            mine = keep & on_side
            if mine.sum() < 2:
                continue
            stretches[side].append({"way_id": way_id, "stations": stations[mine],
                                    "points": points[mine]})

    runs = []
    for side, found in stretches.items():
        for group in _grouped_stretches(found):
            samples = sorted(zip(np.concatenate([s["stations"] for s in group]),
                                 (tuple(p) for s in group for p in s["points"])))
            kept = [samples[0]]
            for station, point in samples[1:]:
                if station - kept[-1][0] > KERB_SAMPLE_MIN_GAP_FT:
                    kept.append((station, point))
            if len(kept) < 2:
                continue
            runs.append(KerbRun(side=side, line=LineString([point for _s, point in kept]),
                                start_ft=float(kept[0][0]), end_ft=float(kept[-1][0]),
                                source=KERB_FROM_TRACING,
                                way_ids=tuple(s["way_id"] for s in group)))
    return runs


def _grouped_stretches(found: list[dict]) -> list[list[dict]]:
    """Traced stretches gathered into unbroken runs, by whether their station ranges meet.

    Grouped by WAY SPAN rather than by the distance between successive samples: a way's own
    samples are contiguous by construction, but the gap between two of them says nothing - a
    straight kerb is two vertices a hundred feet apart.
    """
    groups: list[list[dict]] = []
    for stretch in sorted(found, key=lambda s: float(s["stations"].min())):
        lo = float(stretch["stations"].min())
        if groups and lo <= max(float(s["stations"].max()) for s in groups[-1]) + KERB_RUN_JOIN_FT:
            groups[-1].append(stretch)
        else:
            groups.append([stretch])
    return groups


# ---------------------------------------------------------------------------
# The corridor QUESTIONS. Step 2 of docs/network-model.md: these move onto the road first,
# because they have no goldens and are currently wrong in scratch scripts.
#
# Every rule below is the rule the per-leg code already applies, with its constants IMPORTED
# rather than retyped, asked of a road instead of a leg.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorridorFacts:
    """What is positioned along one corridor: openings, crossings, and where parking is legal.

    Resolved ONCE against one road and handed round: two consumers each assembling the same
    geometry are free to diverge the moment one of their tolerances is touched.
    """
    #: (side, KerbOpening) - every place a vehicle crosses one of the two kerbs.
    openings: tuple = ()
    #: CrossStreet - every other street that meets this one, at any of its junctions.
    crossings: tuple = ()
    #: (station_ft, crossing:markings or None, way id) per surveyed pedestrian crossing.
    marked_crossings: tuple = ()
    #: {side: (NoParkingZone, ...)} - R.S. 39:4-138 and what OSM records, stretch by stretch.
    no_parking: tuple = ()
    #: {side: ((lo, hi), ...)} - where a stall may legally be marked.
    parkable: tuple = ()

    def by_side(self, field: str, side: str) -> tuple:
        return next((values for name, values in getattr(self, field) if name == side), ())


def corridor_facts(corridor: Corridor, models) -> CorridorFacts:
    """Answer every corridor question about one road, in one pass over the OSM snapshot."""
    kerb_ways = _corridor_kerb_ways(models)
    crossings = _cross_streets_on(corridor, models)
    openings = _openings_on(corridor, models, kerb_ways)
    no_parking = tuple((side, _no_parking_zones_on(corridor, side, models, crossings))
                       for side in ("left", "right"))
    parkable = tuple((side, _complement_spans([(zone.start_ft, zone.end_ft)
                                               for zone in zones], 0.0, corridor.length_ft))
                     for side, zones in no_parking)
    return CorridorFacts(openings=openings, crossings=crossings,
                         marked_crossings=_marked_crossings_on(corridor, models),
                         no_parking=no_parking, parkable=parkable)


def _placed_on_corridor(corridor: Corridor, line: LineString) -> tuple | None:
    """(side, first station, last station) for a traced line lying along one of the road's kerbs.

    THE WAY IS PLACED AS A WHOLE and then its whole extent measured: a dropped kerb across a
    driveway mouth is often drawn ACROSS the opening rather than along the street, so taking only
    the samples that individually sit near the kerb collapsed four of six real openings to zero
    length.
    """
    stations, offsets, _points, keep = _kerb_samples_on(
        corridor.centerline, [junction.node_ft for junction in corridor.junctions], line)
    if keep.sum() < 2:
        return None
    side = "left" if float(np.median(offsets[keep])) > 0 else "right"
    return side, float(stations[keep].min()), float(stations[keep].max())


def _openings_on(corridor: Corridor, models, kerb_ways: dict) -> tuple:
    """((side, KerbOpening), ...) for every place a vehicle crosses one of this road's kerbs.

    BOTH SIGNALS, for the reason src/geometry/kerbs.py's module docstring gives: a dropped kerb
    carries a surveyed extent, a mapped driveway carries none but is tagged in places no dropped
    kerb is. The surveyed extent wins where the two overlap.
    """
    from src.geometry.kerbs import (DRIVEWAY_WIDTH_FT, MIN_OPENING_LENGTH_FT, KerbOpening,
                                    KerbType, OpeningSource, opens_the_kerb)
    from src.sources.osm_context import fetch_driveways

    found = []
    for way_id, (line, tags) in sorted(kerb_ways.items()):
        if not opens_the_kerb(tags):
            continue
        placed = _placed_on_corridor(corridor, line)
        if placed is None:
            continue
        side, start, end = placed
        if end - start < MIN_OPENING_LENGTH_FT:
            continue
        found.append((side, KerbOpening(start_ft=start, end_ft=end,
                                        source=OpeningSource.DROPPED_KERB,
                                        kerb=KerbType.from_tags(tags), way_id=way_id)))

    seen = set()
    for model in models.values():
        for drive in fetch_driveways(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if drive["id"] in seen:
                continue
            seen.add(drive["id"])
            meeting = _driveway_meeting(corridor, drive)
            if meeting is None:
                continue
            side, station = meeting
            if any(other_side == side and opening.start_ft <= station <= opening.end_ft
                   for other_side, opening in found):
                continue
            found.append((side, KerbOpening(start_ft=max(station - DRIVEWAY_WIDTH_FT / 2, 0.0),
                                            end_ft=station + DRIVEWAY_WIDTH_FT / 2,
                                            source=OpeningSource.DRIVEWAY, way_id=drive["id"])))
    return tuple(sorted(found, key=lambda pair: (pair[0], pair[1].start_ft)))


def _driveway_meeting(corridor: Corridor, drive: dict) -> tuple | None:
    """(side, station) where a mapped driveway reaches one of this road's kerb runs, or None.

    Which kerb it meets is decided by distance to the kerb LINE, and the station is taken from the
    point on the KERB nearest it - both for the reasons kerbs.py:_driveway_meetings gives.
    """
    from shapely.ops import nearest_points

    from src.geometry.intersection import to_state_plane
    from src.geometry.kerbs import DRIVEWAY_REACH_FT

    coords = drive.get("coords_wgs84") or []
    if len(coords) < 2:
        return None
    line = LineString(to_state_plane(coords))
    best = None
    for run in corridor.kerb_runs:
        gap = line.distance(run.line)
        if gap > DRIVEWAY_REACH_FT or (best is not None and gap >= best[0]):
            continue
        _on_drive, on_kerb = nearest_points(line, run.line)
        station = float(corridor.centerline.project(Point(on_kerb)))
        best = (gap, run.side, station)
    return None if best is None else (best[1], best[2])


def _corridor_ways(models, predicate) -> dict:
    """{way id: (LineString in feet, tags)} for OSM ways near any member junction that match."""
    from src.geometry.intersection import to_state_plane
    from src.sources.osm_context import fetch_roads

    ways = {}
    for model in models.values():
        for way in fetch_roads(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if not predicate(way.get("tags", {})) or len(way.get("coords_wgs84") or []) < 2:
                continue
            ways[way["id"]] = (LineString(to_state_plane(way["coords_wgs84"])), way["tags"])
    return ways


def _half_width_at(corridor: Corridor, station_ft: float) -> float:
    """Half the traced width at a station, falling back to the corridor's typical half-width.

    Needed wherever a test is "does this reach the carriageway" at a station that may sit in an
    untraced stretch. The fallback is this road's OWN median, not a class assumption.
    """
    width = corridor.width_at_ft(station_ft)
    if width is not None:
        return width / 2
    sampled = [corridor.width_at_ft(float(station))
               for lo, hi in corridor.both_traced_spans()
               for station in np.linspace(lo, hi, 5)]
    real = [w for w in sampled if w is not None]
    from src.geometry.context_roads import ROADWAY_DEFAULT_WIDTH_FT

    return float(np.median(real)) / 2 if real else ROADWAY_DEFAULT_WIDTH_FT / 2


def _cross_streets_on(corridor: Corridor, models) -> tuple:
    """Every other street that meets this one, anywhere along it.

    The corridor form of src/geometry/cross_streets.py. R.S. 39:4-138(e) applies at EVERY
    intersection, and this is the object that can finally say where they all are: on a corridor
    every junction is somebody's own.
    """
    from shapely.ops import nearest_points

    from src.geometry.context_roads import MAX_HALF_WIDTH_FT, assumed_width_ft, is_carriageway
    from src.geometry.cross_streets import (JOIN_TOLERANCE_FT, MIN_CROSS_ANGLE_DEG, CrossStreet,
                                            _crossing_angle_deg, _sides_of)

    out = []
    for way_id, (line, tags) in sorted(_corridor_ways(models, is_carriageway).items()):
        # NOT a geometric intersection, for the reason cross_streets_ft gives: a side street's OSM
        # way stops on OSM's centreline for the main road while this road is NJDOT's alignment, so
        # `intersects` is False for every real side street on the block. The test is APPROACH.
        if corridor.centerline.distance(line) > MAX_HALF_WIDTH_FT + JOIN_TOLERANCE_FT:
            continue
        on_road, _on_way = nearest_points(corridor.centerline, line)
        station = float(corridor.centerline.project(on_road))
        if on_road.distance(line) > _half_width_at(corridor, station) + JOIN_TOLERANCE_FT:
            continue
        if _crossing_angle_deg(corridor.centerline, line, on_road) < MIN_CROSS_ANGLE_DEG:
            continue
        sides = _sides_of(corridor.centerline, line, on_road)
        if not sides:
            continue
        out.append(CrossStreet(leg=corridor.name, station_ft=station,
                               half_width_ft=assumed_width_ft(tags) / 2, sides=sides,
                               name=tags.get("name"), way_id=way_id))
    return tuple(sorted(out, key=lambda cross: cross.station_ft))


def _marked_crossings_on(corridor: Corridor, models) -> tuple:
    """((station, crossing:markings or None), ...) for the surveyed pedestrian crossings on it.

    Counted, not drawn - drawing them as traced is stream A of docs/network-renderer-plan.md. What
    this is for is the corridor figure: how many crossings a walk down this street actually has,
    and how many of those the surveyor recorded MARKINGS for.
    """
    from src.geometry.cross_streets import MIN_CROSS_ANGLE_DEG, _crossing_angle_deg
    from src.geometry.intersection import to_state_plane
    from src.sources.osm_context import fetch_crossings

    found = {}
    for model in models.values():
        for crossing in fetch_crossings(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            coords = crossing.get("coords_wgs84") or []
            if len(coords) < 2:
                continue
            line = LineString(to_state_plane(coords))
            middle = line.interpolate(0.5, normalized=True)
            station = float(corridor.centerline.project(middle))
            if middle.distance(corridor.centerline) > _half_width_at(corridor, station):
                continue
            if _crossing_angle_deg(corridor.centerline, line, middle) < MIN_CROSS_ANGLE_DEG:
                continue
            key = tuple(crossing.get("node_ids") or [tuple(coords[0])])[:1]
            found[key] = (station, crossing["tags"].get("crossing:markings"))
    return tuple(sorted(found.values()))


def _no_parking_zones_on(corridor: Corridor, side: str, models, crossings: tuple) -> tuple:
    """Every stretch of one kerb where parking is forbidden, with the clause that forbids it.

    THE SAME FOUR RULES src/geometry/daylighting.py applies to a leg, with its statutory distances
    imported rather than retyped. What is different is only that they are applied along a ROAD:

      * (e) 25 ft from the side line of EVERY intersecting street, which is what the statute says.
      * (h) 50 ft from a stop sign and (i) 10 ft from a hydrant, as radii round a point.
      * what OSM RECORDS about this kerb, per stretch of it.

    The junction end of a leg needed a corner fillet's tangent point to find the side line
    (daylighting.sideline_station_ft); on a corridor the member junctions arrive through the same
    cross-street list as every other side street.
    """
    from src.geometry.daylighting import (FIRE_HYDRANT_SETBACK_FT, FOOTWAY_REACH_FT,
                                          SIDELINE_SETBACK_FT, STOP_SIGN_SETBACK_FT, NoParkingZone)
    from src.geometry.intersection import (PARKING_RESTRICTION_KEYS, parking_is_restricted,
                                           parking_restriction_by_side)
    from src.sources.osm_context import fetch_street_furniture, fetch_traffic_control

    zones = []
    for cross in crossings:
        if side not in cross.sides:
            continue
        near_ft, far_ft = cross.mouth_ft
        zones.append(NoParkingZone(
            near_ft - SIDELINE_SETBACK_FT, far_ft + SIDELINE_SETBACK_FT,
            f"R.S. 39:4-138(e), {SIDELINE_SETBACK_FT:.0f} ft from the side line of "
            f"{cross.citation}"))

    for radius_ft, citation, nodes in (
            (STOP_SIGN_SETBACK_FT, "R.S. 39:4-138(h), 50 ft from a stop sign",
             _corridor_nodes(models, fetch_traffic_control, lambda t: t.get("highway") == "stop")),
            (FIRE_HYDRANT_SETBACK_FT, "R.S. 39:4-138(i), 10 ft from a fire hydrant",
             _corridor_nodes(models, fetch_street_furniture,
                             lambda t: t.get("emergency") == "fire_hydrant"))):
        for point in nodes:
            stations, offsets = station_offset_many(corridor.centerline,
                                                    np.asarray([point], dtype=float))
            station, offset = float(stations[0]), float(offsets[0])
            if (offset > 0) != (side == "left"):
                continue        # a hydrant on the north kerb says nothing about the south one
            if abs(offset) > _half_width_at(corridor, station) + FOOTWAY_REACH_FT:
                continue        # further out than a footway reaches: somebody else's kerb
            if not -radius_ft <= station <= corridor.length_ft + radius_ft:
                continue
            zones.append(NoParkingZone(station - radius_ft, station + radius_ft, citation))

    for way_id, span, tags, aligned in _road_spans_on(corridor, models):
        restriction = parking_restriction_by_side(tags, aligned)[side]
        if not parking_is_restricted(restriction):
            continue
        key = PARKING_RESTRICTION_KEYS["both" if tags.get(PARKING_RESTRICTION_KEYS["both"])
                                       else side]
        zones.append(NoParkingZone(span[0], span[1],
                                   f"OSM {key}={restriction} on way {way_id}"))
    return tuple(sorted(zones, key=lambda zone: zone.start_ft))


def _corridor_nodes(models, fetch, predicate) -> list[tuple]:
    """Every OSM node near any member junction that matches, in state-plane feet, once each."""
    from src.geometry.intersection import to_state_plane

    found = {}
    for model in models.values():
        for node in fetch(model.center_wgs84, radius_m=CORRIDOR_KERB_RADIUS_M):
            if predicate(node.get("tags", {})):
                found[(round(node["lon"], 7), round(node["lat"], 7))] = None
    return to_state_plane(list(found)) if found else []


def _road_spans_on(corridor: Corridor, models) -> list[tuple]:
    """(way id, (start_ft, end_ft), tags, aligned) for every OSM way lying ALONG this road.

    A LIST and not one way, because what a way says varies along a street and OSM says so by
    SPLITTING THE WAY.

    Matched on lying along the road over a real stretch of it, with the same two thresholds
    _match_legs_to_osm_roads uses, and restricted to CARRIAGEWAY classes for the reason recorded
    there.
    """
    from src.geometry.intersection import (MIN_ROAD_SPAN_FT, ROAD_MATCH_HIGHWAY_CLASSES,
                                           ROAD_MATCH_MAX_ANGLE_DEG, ROAD_MATCH_MAX_OFFSET_FT)

    spans = []
    ways = _corridor_ways(models, lambda t: t.get("highway") in ROAD_MATCH_HIGHWAY_CLASSES)
    road_dir = line_direction(corridor.centerline)
    for way_id, (line, tags) in sorted(ways.items()):
        along = float(np.dot(line_direction(line), road_dir))
        stations, offsets = station_offset_many(corridor.centerline,
                                                np.asarray(line.coords, dtype=float))
        lo = max(float(stations.min()), 0.0)
        hi = min(float(stations.max()), corridor.length_ft)
        if hi - lo < MIN_ROAD_SPAN_FT:
            continue
        covering = (stations >= -MIN_ROAD_SPAN_FT) & (stations <= corridor.length_ft
                                                      + MIN_ROAD_SPAN_FT)
        if not covering.any() or float(np.abs(offsets[covering]).min()) > ROAD_MATCH_MAX_OFFSET_FT:
            continue
        # Per-station rather than on the chord: a 2,849 ft road bends, so its end-to-end direction
        # is a poor description of it anywhere in particular. Taken where the way actually overlaps.
        local = np.asarray([frame_at(corridor.centerline, float(s))[1]
                            for s in stations[covering]])
        way_dir = line_direction(line)
        angle = np.degrees(np.arccos(np.clip(np.abs(local @ way_dir).max(), -1.0, 1.0)))
        if angle > ROAD_MATCH_MAX_ANGLE_DEG:
            continue
        spans.append((way_id, (lo, hi), tags, bool(along >= 0)))
    return spans


def marked_parking_capacity(corridor: Corridor, facts: CorridorFacts, side: str,
                            within: tuple = ()) -> tuple[int, float]:
    """(stalls, ft of kerb they were counted over) where parking is legal on one side.

    `within` narrows the count to a set of spans - which is how the report prints a figure and its
    own coverage from ONE function rather than from two that could drift apart. Returns the length
    as well as the count because "486 stalls" over a corridor that is 43% traced is not a corridor
    figure.

    Stall length is PARKING_STALL_LENGTH_DEFAULT_FT, imported - the same figure the per-leg
    marking uses.
    """
    from src.geometry.model import parking_stall_count_ft
    from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT

    runs = facts.by_side("parkable", side)
    if within:
        runs = _intersect_spans(runs, within)
    stalls, measured_ft = 0, 0.0
    for lo, hi in runs:
        measured_ft += hi - lo
        # Through the same counter parking_stall_lines_ft uses, so the reported number is the
        # number that would be drawn. Given both bounds it reads nothing off its first argument
        # but `.centerline`, which a Corridor has.
        stalls += parking_stall_count_ft(corridor, PARKING_STALL_LENGTH_DEFAULT_FT, lo, hi)
    return stalls, measured_ft


# How finely the OSM fetch window is walked along a road. Only used to say how much of the road
# the fetch actually covered, so a couple of feet of edge either way does not matter.
_WINDOW_SAMPLE_FT = 10.0


def osm_window_spans(corridor: Corridor, models) -> tuple[tuple[float, float], ...]:
    """The stretches of a road that fall inside the OSM fetch window round its member junctions.

    THE DENOMINATOR FOR ANYTHING COUNTED OUT OF OSM, and it exists because "nothing fetched" and
    "nothing mapped" arrive identically. At the junction radius of 120 m the three circles on Broad
    St leave 173 m outside every window; at CORRIDOR_KERB_RADIUS_M the same road is fully covered.
    Reported rather than believed.
    """
    reach_ft = CORRIDOR_KERB_RADIUS_M / 0.3048
    centres = [model.center_ft for model in models.values()]
    n = max(int(np.ceil(corridor.length_ft / _WINDOW_SAMPLE_FT)) + 1, 2)
    inside = []
    for station in np.linspace(0.0, corridor.length_ft, n):
        point = corridor.centerline.interpolate(float(station))
        if any(point.distance(centre) <= reach_ft for centre in centres):
            inside.append((float(station) - _WINDOW_SAMPLE_FT / 2,
                           float(station) + _WINDOW_SAMPLE_FT / 2))
    return _intersect_spans(_merged_spans(inside), ((0.0, corridor.length_ft),))
