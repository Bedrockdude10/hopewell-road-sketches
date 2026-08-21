"""A CORRIDOR: a chain of roads, bridged along the NJDOT alignment each end's leg was cut from.

STEP 2 OF docs/network-model.md, and the object that makes corridor questions askable at all: the
traced kerb is read side by side the whole way, so a fact can be stationed once and stay stationed
across every junction on the street.

BRIDGING IS THE HARD PART. Two modelled junctions on one street leave a gap between them that no
leg covers, and the two legs facing each other across it were cut from the same NJDOT alignment.
So the corridor follows that alignment through the gap and EASES onto it - `_eased_alignment` -
because the alignment and the fitted leg disagree by a few feet and a hard switch at the seam puts
a kink in every marking drawn over it.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from src.geometry.model import (STRIP_SAMPLE_FT, frame_at, is_through_street, line_direction,
                                place_in_measured_frame, station_offset_many)
from src.geometry.network.kerb import (CORRIDOR_KERB_RADIUS_M, KerbRun, _complement_spans,
                                       _corridor_kerb_ways, _intersect_spans, _junction_kerb_runs,
                                       _merged_spans, _traced_kerb_runs, _tracing_reach_ft,
                                       junction_corner_reach_ft)
from src.geometry.network.road import (Road, _kerb_offset_at, roads_from_model)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: this type is layered above this package, so importing it
    # for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel


# How far past the outermost modelled leg the road is carried along NJDOT's alignment. The reach
# is TRIMMED to where the tracing stops; the cap is a second bound on top of that.
#
# IT IS THE FETCH RADIUS, not a round number. A flat 500 ft cap silently dropped 21 of 22 kerb
# ways northeast of Princeton Ave. CORRIDOR_KERB_RADIUS_M is the honest bound: past it no kerb
# was fetched, and the trim cannot catch "no kerb fetched" vs "no kerb traced".
CORRIDOR_EXTENSION_FT = CORRIDOR_KERB_RADIUS_M * 3.28084


@dataclass(frozen=True)
class JunctionOnRoad:
    """One modelled junction's stretch of a corridor, and the two legs it contributed.

    `legs` carries a SIGN per leg because a leg's own station 0 is at the node and it runs
    outward, so one of the two runs backwards along the corridor. Arc length along the corridor is
    arc length along this junction Road, whose vertices it contains verbatim.

    THE SIGN IS NOT QUITE THE WHOLE TRANSLATION. Both legs start at the node in the model, but
    they come from two NJDOT alignments that do not meet, and the joined centreline carries what
    the blend could not close as real length (Road.far_node_ft). `leg_joint_ft` is that offset per
    leg - zero for the leg the road's stations are measured from, and the open joint for the other.
    Left out, W Broad & Louellen's far leg was translated 2.79 ft up the street and the corridor
    read 2.9 ft wider there than the leg it was built from.
    """
    site: str
    node_ft: float
    start_ft: float
    end_ft: float
    legs: tuple[tuple[str, float], ...]
    leg_joint_ft: tuple[tuple[str, float], ...] = ()

    def station_of(self, leg_name: str, leg_station_ft: float) -> float:
        joints = dict(self.leg_joint_ft)
        for name, sign in self.legs:
            if name == leg_name:
                return self.node_ft + sign * leg_station_ft + sign * joints.get(name, 0.0)
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


def _junction_road_ends(models: dict[str, "IntersectionModel"]) -> list[dict]:
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


def _sri_alignment(models: dict[str, "IntersectionModel"], sri: str):
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


def _road_joint_ft(road: Road) -> float:
    """How far past the node the FAR leg's own station 0 sits - see Road.far_node_ft."""
    return 0.0 if road.far_node_ft is None else road.far_node_ft - road.node_ft


def _oriented_piece(road: Road, first_leg: str) -> dict:
    """One junction Road turned to run the corridor's way, with its sides renamed to match.

    Reversing a road swaps its left and right - the same trap roads_from_model's own side pairing
    warns about.
    """
    if road.near_leg == first_leg:
        return {"centerline": road.centerline, "left": road.left_curb, "right": road.right_curb,
                "node_from_start_ft": road.node_ft,
                "legs": ((road.near_leg, -1.0), (road.far_leg, 1.0)),
                "leg_joint_ft": ((road.far_leg, _road_joint_ft(road)),)}
    def flip(line: LineString):
        return None if line is None else LineString(list(line.coords)[::-1])

    return {"centerline": flip(road.centerline), "left": flip(road.right_curb),
            "right": flip(road.left_curb),
            "node_from_start_ft": road.length_ft - road.node_ft,
            "legs": ((road.far_leg, -1.0), (road.near_leg, 1.0)),
            "leg_joint_ft": ((road.far_leg, _road_joint_ft(road)),)}


def _cumulative_ft(coords) -> np.ndarray:
    steps = np.hypot(*np.diff(np.asarray(coords, dtype=float), axis=0).T)
    return np.concatenate(([0.0], np.cumsum(steps)))


def corridors_from_models(models: dict[str, "IntersectionModel"]) -> list[Corridor]:
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


def _build_corridor(models: dict[str, "IntersectionModel"], chain: list[tuple],
                    roads_by_key: dict) -> Corridor | None:
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
        legs=piece["legs"], leg_joint_ft=piece.get("leg_joint_ft", ()))
        for piece in pieces)
    # Each modelled junction carries its OWN corner reach; an unmodelled cross street keeps the
    # square-corner default, because nothing here knows its angle.
    junction_ft = tuple((junction.node_ft, junction_corner_reach_ft(models[piece["site"]]))
                        for piece, junction in zip(pieces, junctions))
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
    # That two-pass shape is also the ONE place this package's layering runs backwards: facts sit
    # above corridor everywhere else, so the import is function-level to keep the module graph a
    # DAG. Moving it to the header closes a cycle at import time.
    from src.geometry.network.facts import _cross_streets_on

    provisional = corridor_with(junction_runs + _traced_kerb_runs(centerline, kerb_ways,
                                                                  junction_ft))
    crossing_ft = tuple(sorted({round(cross.station_ft, 1)
                                for cross in _cross_streets_on(provisional, models)}))
    return corridor_with(
        junction_runs + _traced_kerb_runs(centerline, kerb_ways, junction_ft + crossing_ft),
        cross_street_ft=crossing_ft)


def _extension(models: dict[str, "IntersectionModel"], kerb_ways, seam_point, away, sri: str,
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
    # The nearest modelled junction is the one whose fetch reaches out here, so its centre is the
    # circle the cap belongs to - see _tracing_reach_ft.
    centre = min((model.center_ft for model in models.values()),
                 key=lambda point: point.distance(Point(seam_point)), default=None)
    reach = _tracing_reach_ft(align, seam_ft, forward, kerb_ways, CORRIDOR_EXTENSION_FT,
                              centre_xy=None if centre is None else (centre.x, centre.y))
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
