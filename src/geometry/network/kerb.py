"""WHERE THE KERB IS TRACED along a road, and where it is a junction corner instead.

A corridor's kerb arrives in pieces: OSM splits a kerb way wherever a tag changes, the tracing
stops short of every corner, and the corner returns themselves come from the junction model rather
than from OSM. This module turns all of that into `KerbRun`s - station ranges on one side of a
corridor, each labelled with where it came from - so a caller can ask "is the kerb known here?"
without knowing which source answered.

THE SPAN ARITHMETIC LIVES HERE, not in a neutral module of its own, because `_merged_spans` bakes
in `KERB_RUN_JOIN_FT`: two ranges closer than one kerb-run join are one range, whatever they are
ranges OF. A file called `spans.py` would read as pure interval arithmetic and would be wrong.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from src.geometry.model import (Alignment, STRIP_SAMPLE_FT, curb_edge_by_station, curb_station_span, frame_at,
                                station_offset_many, vertex_tangents)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: this type is layered above this package, so importing it
    # for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel



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


def _traced_end_ft(align: LineString, node_ft: float, forward: bool, kerb_ways,
                   max_ft: float, centre_xy=None) -> float:
    """The ABSOLUTE station on the alignment where the traced kerb stops, out from a junction node.

    Measured against NJDOT's alignment rather than the finished corridor, because the corridor
    cannot be built until its length is known. The answer only ever shortens the road, so an error
    here cannot invent street.

    ANCHORED ON THE NODE, NOT ON THE SEAM, AND THAT IS THE WHOLE POINT. This used to take the end
    of the junction piece and return a reach measured FROM it. Both the search window and the cap
    were then relative to a point that moves with HOPEWELL_FRAME_SCALE, because the piece is a
    frame-cut leg - so a wider sheet slid the window outward and the corridor discovered street
    that a narrower sheet had not looked for. Greenwood Ave came out 1,695 ft at 1x and 1,891 ft at
    2.5x, and a facility's governing cross-section is taken over the span, so the sheet was voting
    on the design. A junction centre is a surveyed fact and does not move, so anchoring here makes
    the answer a property of the survey and leaves the frame to crop.

    THE CAP IS A RADIUS, SO IT IS APPLIED AS ONE. `max_ft` is CORRIDOR_KERB_RADIUS_M - the distance
    past which no kerb was FETCHED, and the fetch is a circle round the junction. Capping the
    arc length instead let the road run past its own fetch window wherever the street bends, since
    an arc is always longer than its chord: Broad St came out 4,655 ft with only 4,531 inside a
    window, and an opening count over the last 124 ft would have been counting where nothing was
    looked for. So a kerb point earns reach only if it is inside the circle as well as along the
    road.
    """
    from src.geometry.intersection import KERB_PLAUSIBLE_HALF_WIDTH_FT

    near, far = KERB_PLAUSIBLE_HALF_WIDTH_FT
    lo = node_ft if forward else max(node_ft - max_ft, 0.0)
    hi = min(node_ft + max_ft, align.length) if forward else node_ft
    centre = (np.asarray(align.interpolate(node_ft).coords[0]) if centre_xy is None
              else np.asarray(centre_xy, dtype=float))
    end = node_ft
    for line, _tags in kerb_ways.values():
        points = _dense_kerb_points(line)
        stations, offsets = station_offset_many(align, points)
        beside = (np.abs(offsets) >= near) & (np.abs(offsets) <= far)
        within_radius = np.hypot(*(points - centre).T) <= max_ft
        inside = beside & within_radius & (stations >= lo) & (stations <= hi)
        if inside.any():
            here = stations[inside]
            end = max(end, float(here.max())) if forward else min(end, float(here.min()))
    return end


def _dense_kerb_points(line: LineString) -> np.ndarray:
    """A traced kerb resampled ALONG itself - see context_roads.py:kerb_points for why.

    A straight run of kerb is mapped with two vertices, so reading vertices alone would miss kerb
    at most stations. Same spacing constant, imported rather than repeated.
    """
    from src.geometry.context_roads import kerb_points

    return kerb_points([line])


def _corridor_kerb_ways(models: dict[str, "IntersectionModel"]) -> dict:
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


def junction_corner_reach_ft(model: "IntersectionModel") -> float:
    """How far this junction's corner returns sweep along the roads meeting in it.

    One number per junction - the most generous of its legs - because the corridor reads kerb
    along a ROAD and cannot ask which leg a sample belongs to. See model.corner_return_scale for
    why a flat distance is wrong at an acute junction.
    """
    from src.geometry.model import CURB_POINT_CORNER_ZONE_FT, corner_return_scale

    return CURB_POINT_CORNER_ZONE_FT * max(
        (corner_return_scale(leg, model.legs) for leg in model.legs.values()), default=1.0)


def _kerb_samples_on(centerline: LineString, node_stations, line: LineString) -> tuple:
    """(stations, offsets, points, keep) for one traced way read as THIS road's kerb.

    THREE tests decide whether a sample belongs to this road, each the one the per-leg fit already
    uses:

      * IN THE ROAD - station inside [0, length].
      * BESIDE IT - |offset| inside KERB_PLAUSIBLE_HALF_WIDTH_FT (8-45 ft).
      * ALONG IT - the kerb's own heading within CURB_POINT_MAX_SKEW_DEG of the road's, SUSPENDED
        near a node (corner returns sweep 90 degrees). Each entry in `node_stations` may carry its
        own reach as (station, reach_ft); a bare station takes the square-corner default. A flat
        distance here against a scaled one in the per-leg fit is the two paths reading different
        kerb, which test_the_road_reproduces_each_legs_measured_width catches.
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
    pairs = [entry if isinstance(entry, tuple) else (entry, CURB_POINT_CORNER_ZONE_FT)
             for entry in node_stations]
    if pairs:
        nodes = np.asarray([station for station, _reach in pairs], dtype=float)
        reaches = np.asarray([reach for _station, reach in pairs], dtype=float)
        in_corner = (np.abs(stations[:, None] - nodes[None, :]) <= reaches[None, :]).any(axis=1)
    else:
        in_corner = np.zeros(len(stations), bool)
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
