"""Getting the TRACED KERB out of OSM and into state-plane feet.

One cache, one radius, one projection - and the radius is the DRAWING's, not the fit's, which is a
distinction this project has got wrong more than once: what a render contains is a question about
the render, while how far out to trust a kerb for a corner-radius fit is a question about the
fit."""


import numpy as np
from shapely.geometry import LineString, Point

from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_kerbs
from src.geometry.model import (
    CURB_POINT_BEHIND_TOLERANCE_FT,
    CURB_POINT_MAX_WIDTH_RATIO,
    station_offset_many,
)
from src.geometry.intersection.junction import OSMDataUnavailableError

KERB_CONTEXT_RADIUS_M = 120  # fetch radius, metres - generous enough to catch a whole return
KERB_NEAR_JUNCTION_FT = 80   # but a return belonging to THIS junction is within this of centre
# How far outside a leg's plausible half-width band a traced vertex may sit and still count as
# that leg's kerb, for deciding whether a whole kerb WAY is relevant to this junction.
KERB_ALONG_LEG_TOLERANCE_FT = 8.0


def _runs_along_a_leg(line: LineString, legs: dict) -> bool:
    """Whether any vertex of `line` sits where one of these legs' kerbs would be.

    The test for "is this OUR kerb", as opposed to "is this near the middle of the junction".
    KERB_NEAR_JUNCTION_FT is the latter, and it is right for fitting a corner radius - a
    return belonging to this junction is within 80 ft of its centre, and at 120 m the fetch
    otherwise drags in neighbouring junctions' returns, which produced a nonsense 7.9-30.2 ft
    radius spread at Columbia & Princeton.

    It is wrong for building the CURB LINES, which want kerb anywhere along a 130 ft leg. On a
    130 ft leg a kerb at station 100 is 100 ft from the junction centre and was being thrown
    away: 14 traced ways across the four junctions, at stations 76-127 and plausible kerb
    offsets, including both sides of greenwood_ave_south from 87 ft out. Their absence was
    invisible because curb_line_from_points EXTRAPOLATES to the working length, so the outer
    half of those legs was drawn from a bearing instead of from the tracing that existed.
    """
    for leg in legs.values():
        if leg.curb_to_curb_ft is None:
            continue
        stations, offsets = station_offset_many(leg.centerline,
                                                np.asarray(line.coords, dtype=float))
        half_ft = leg.curb_to_curb_ft / 2
        along = ((stations > -CURB_POINT_BEHIND_TOLERANCE_FT)
                 & (stations < leg.centerline.length))
        beside = np.abs(offsets) < half_ft * CURB_POINT_MAX_WIDTH_RATIO + KERB_ALONG_LEG_TOLERANCE_FT
        if (along & beside).any():
            return True
    return False


def drawn_kerb_radius_ft() -> float:
    """How far out a kerb still counts as part of the PICTURE, in feet.

    Every kerb that was fetched, which is the honest answer: the fetch radius already scales
    with the frame, so this is "all the traced kerb around this junction" and not a second
    independent idea of how much is relevant.

    Both renderers take this one number, so the set they draw is literally the same set. The
    plan view is a square window on it and matplotlib crops what falls outside the axes; the
    3D camera is a tilted perspective and necessarily sees further. That asymmetry is the one
    src/render/frame.py already describes - the views share the subject and the radius, not the
    outline of the visible region - and it is not the same thing as the two disagreeing about
    which kerbs exist, which is what the frame radius was quietly doing when the roads ran to
    938 ft and the kerbs stopped at 379.
    """
    from src.render.frame import context_radius_m

    return context_radius_m(KERB_CONTEXT_RADIUS_M) / 0.3048


def kerb_lines_with_tags_ft(center_wgs84: Point, center_ft: Point, legs: dict | None = None,
                             radius_ft: float | None = None) -> list:
    """[(LineString, tags)] for traced kerbs near the junction - geometry plus what OSM
    says about each (kerb=lowered, tactile_paving=yes, wheelchair=yes).

    THREE relevance tests, because "is this kerb ours" has three different answers and they are
    not interchangeable:

      * `radius_ft` - everything within that distance of the centre. The DRAWING test, and the
        only one of the three that is about the picture rather than about the junction. Use it
        for anything being rendered.
      * `legs` - _runs_along_a_leg: kerb anywhere along a leg, however far out, which is what a
        curb LINE wants. 14 traced ways across the four junctions sit at stations 76-127 with
        plausible kerb offsets and fail the near test - both sides of greenwood_ave_south from
        ~90 ft out among them.
      * neither - the NEAR set, within KERB_NEAR_JUNCTION_FT of the junction CENTRE. The right
        test for fitting a corner RADIUS and for measuring a width: a return belonging to this
        junction is close to it, and at the fetch radius anything looser drags in the
        neighbouring junctions' returns.

    The near set was the DEFAULT, and both renderers took the default, which is how a wide render
    came to show a cross of asphalt floating on grass: 8,938 ft of kerb is traced within 600 m of
    Broad & Greenwood - both sides of the corridor, Louellen to Princeton - and an 80 ft filter
    meant for the radius fit threw all but the corner returns away. A test written to keep a
    neighbouring junction out of a CIRCLE FIT was silently deciding what a drawing contains.

    The wide set is deliberately NOT fed to the fit. Admitting those ways shifts
    w_broad_st_southwest's measured width, that reshuffles the vertex contest at the one
    junction with an acute Y and partial tracing, and louellen_st_west drops from two traced
    kerbs to one - more data in, less data used. So the fit runs on the near set and
    _extend_curbs_with_far_tracing rebuilds the curb lines from the wide set afterwards, once
    the widths are settled and the extra ways can only lengthen a curb, never redefine one.
    See tests/test_leg_frame.py.
    """
    def relevant(line):
        if radius_ft is not None:
            return line.distance(center_ft) <= radius_ft
        if legs:
            return _runs_along_a_leg(line, legs)
        return line.distance(center_ft) <= KERB_NEAR_JUNCTION_FT

    return [(line, tags, way_id) for line, tags, way_id in _projected_kerbs(center_wgs84)
            if relevant(line)]


# Every traced kerb way at one junction, projected into state-plane feet once. The projection
# is a fact about the fetched ways, and the ways are read four times over per model load (the
# radius fit, the width fit, the far-tracing pass, the plan view) plus once per scenario. Held
# alongside the fetched list and served only while that is still the list fetch_kerbs returns,
# the same identity rule src/sources/osm_context.py:_LAYER_VIEWS uses - so a re-pulled snapshot
# cannot be served a projection of the old one.
_PROJECTED_KERBS: dict[tuple, tuple] = {}


def _projected_kerbs(center_wgs84: Point) -> list[tuple]:
    """[(LineString in feet, tags, way id)] for every traced kerb WAY near the junction.

    The id is carried because a marking this project BREAKS for a kerb has to be traceable to
    the kerb that broke it - see src/geometry/kerbs.py:KerbOpening.citation. It was dropped here
    before, and nothing read the tags either, so a caller wanting to know which way a line came
    from had to re-fetch and re-project to find out.

    Lone `barrier=kerb` NODES are dropped: they carry no arc to fit and no line to draw.
    """
    from src.render.frame import context_radius_m

    # Scaled with the frame (see _paved_surfaces_ft for why the import is lazy). Widening this
    # cannot disturb the fits: the near set filters to 80 ft and the wide set to a 130-170 ft
    # leg, both far inside the unscaled 120 m, so the extra ways are candidates that every
    # existing test then rejects. It is the DRAWING that needed them.
    try:
        kerbs = fetch_kerbs(center_wgs84, radius_m=context_radius_m(KERB_CONTEXT_RADIUS_M))
    except RuntimeError as e:
        # An outage must not look like "nothing is mapped here". Returning [] silently
        # would drop the traced kerbs, and with them the measured widths, the per-corner
        # radii and every tactile pad - producing a confident-looking render built on
        # absent data. Overpass mirrors are flaky enough that this happens in practice.
        raise OSMDataUnavailableError(
            f"could not fetch OSM kerbs for this junction ({e}). Refusing to build geometry: "
            "without them the widths, corner radii and tactile paving would silently fall back "
            "to placeholders, and the render would look finished while being wrong. Retry when "
            "Overpass is reachable."
        ) from e

    key = (round(center_wgs84.x, 7), round(center_wgs84.y, 7))
    cached = _PROJECTED_KERBS.get(key)
    if cached is not None and cached[0] is kerbs:
        return cached[1]
    projected = []
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords:
            continue
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        projected.append((LineString(zip(xs, ys)), kerb.get("tags", {}), kerb.get("id")))
    _PROJECTED_KERBS[key] = (kerbs, projected)
    return projected


def _kerb_lines_ft(center_wgs84: Point, center_ft: Point) -> list[LineString]:
    """Traced OSM kerb ways near this junction, in state-plane feet.

    Only kerbs within KERB_NEAR_JUNCTION_FT of the centre are kept. The fetch radius has
    to be generous enough to catch a whole corner return, but at 120 m it also pulls in
    kerbs belonging to NEIGHBOURING junctions - and those were being assigned to this
    junction's corners, producing a nonsense 7.9-30.2 ft spread at Columbia & Princeton.
    A corner return sits within a few tens of feet of the junction it belongs to.

    The near set of kerb_lines_with_tags_ft without the tags. It used to be a second copy of
    that function's fetch-and-project loop, which meant an Overpass outage reached this one
    first and came out as a bare RuntimeError rather than the OSMDataUnavailableError written
    to explain it.
    """
    return [line for line, *_ in kerb_lines_with_tags_ft(center_wgs84, center_ft)]
