"""Every piece of curbside paint a DesignState calls for, built once.

This exists because the 2D plan view and the 3D export were each assembling the same
markings from the same primitives, independently - roughly sixty lines apiece of "work out
the anchor stations, build the strip, build the taper, clip it". They drifted, exactly as
you would expect: at the time this module was written the two disagreed about where a
parking buffer's taper starts (the plan view used the corner clearance, the export used the
point where the stalls begin) and about whether taper fill gets clipped around a crossing at
all (only the export did it). Two pictures of the same junction that don't match is the one
thing this project cannot ship, since the whole premise of the plan view is that it shows
what the render will show.

So the geometry is decided here and both renderers draw what they are handed. Nothing in
this module knows about matplotlib, meters, or Blender - it returns shapely in state-plane
feet, and each renderer converts.

It is also what src/checks.py inspects. A check that rebuilds the paint itself would just be
a third copy free to drift from the other two; checking THIS list is checking what is drawn.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.geometry.model import (_point_at, bollard_points_ft, clip_paint_clear_of,
                                corner_apron_annulus, corner_overlay_polygon,
                                curb_offsets_at_stations, curbside_strip_polygon,
                                inset_line_ft, lane_narrowing_edge_lines_ft,
                                lane_narrowing_polygons_ft, lane_narrowing_taper_ft,
                                lane_narrowing_taper_polygons_ft, leg_clearance_ft,
                                parking_lane_edge_line_ft, parking_stall_lines_ft,
                                points_at_offset_ft,
                                station_offset_many, through_street_sides)
from src.geometry.markings import (APRON, BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE, BOLLARD,
                                   BUFFER_EDGE_LINE, BUFFER_FILL,
                                   CROSSING_RIM_LINE, DAYLIGHT_EDGE_LINE, DAYLIGHT_FILL,
                                   LANE_EDGE_LINE, LANE_NARROWING_FILL, PARKING_EDGE_LINE,
                                   PaintKind, STALL_DIVIDER, TAPER_FILL, TAPER_LINE,
                                   ZONE_END_LINE)
from src.geometry.treatments import PARKING_STALL_LENGTH_DEFAULT_FT, TARGET_LANE_WIDTH_FT
from src.geometry.daylighting import (merged_no_parking_spans_ft, no_parking_zones_ft,
                                      parkable_runs_ft)
from src.render.coords import FT_TO_M
from src.render.crosswalks import (CROSSWALK_CLEARANCE_FT, CROSSWALK_DEPTH_FT,
                                   crosswalk_reach_on_leg_side_ft)


@dataclass(frozen=True)
class PaintPiece:
    """One painted marking. `kind` is what it is, `leg`/`side` where it belongs.

    leg/side are None for the corner treatments, which sit at a corner between two legs and
    so belong to neither - they are the reason the curb check below skips pieces without a
    side rather than assuming one.

    `kind` is a src/geometry/markings.py:PaintKind rather than a string, so a piece carries its
    own answer to "how is this drawn, and where does it travel to the 3D render" instead of
    every consumer looking that up in a table of its own. It used to be a string.
    """
    kind: PaintKind
    geometry: LineString | Polygon
    leg: str | None = None
    side: str | None = None

    @property
    def is_fill(self) -> bool:
        """Hatched paint - asked of the MARKING, not of its geometry.

        These two answers differ, and the difference caused real bugs: a bollard is stored as a
        degenerate polygon standing in for a point, so a geometry test called it a fill and
        every invariant that asked about polygons had to carry `and p.kind != "bollard"`. See
        markings.Role.
        """
        return self.kind.is_fill

    @property
    def covers_area(self) -> bool:
        """Occupies ground rather than tracing a line: a hatched zone or a built surface."""
        return self.kind.covers_area


@dataclass(frozen=True)
class LegAnchors:
    """The two stations every curbside treatment on ONE SIDE of a leg is measured from.

    target_ft - where a taper meets the real curb, and so the closest to the junction any of
                this paint gets: CROSSWALK_CLEARANCE_FT beyond where the crossing's paint
                actually reaches on this side.
    anchor_ft - where a paint-only buffer's straight run begins. Past the corner return AND
                past the crossing; taking the corner clearance alone let paint run to within
                3.9 ft of Princeton Ave north's crossing.

    Per SIDE, not per leg, because a skewed crossing reaches further along one kerb than the
    other - 9.4 ft further at broad_st_west. A single per-leg target either overlapped the
    crossing on one side or left a gap on the other; it overlapped, and the overlap was
    chopped off square by clip_paint_clear_of.

    Where marked STALLS may begin is also not a property of the leg - the side line, the
    stop signs and the hydrants all differ by side. See src/geometry/daylighting.py.
    """
    anchor_ft: float
    target_ft: float
    crossing_ft: float = 0.0     # where the crossing's paint actually reaches on this side
    clearance_ft: float = 0.0    # past THIS SIDE's corner return, if it has one


def leg_anchors(state, leg_name: str, side: str, crosswalk_offsets: dict,
                 keep_clear=None, inner_offset_ft: float = 0.0,
                 crosswalk_is_marked: bool = True) -> LegAnchors:
    """inner_offset_ft is how far from the centerline this treatment's paint starts - the
    lane edge. Only the crossing inside that strip can get in its way.

    Clearance is asked PER SIDE. This paint belongs to one kerb, and a corner return belongs
    to one side of each leg it touches, so a per-leg maximum holds the paint back for a curve
    that may be on the opposite kerb. See leg_clearance_ft.

    With no painted crossing on this leg there is nothing to keep clear OF, so the only limit
    is that same corner return. The nominal crossing station an unmarked leg still carries is
    the geometric estimate - itself the per-leg corner clearance - and reserving room around
    it held the north side of E Broad's kerbside paint 37 ft out for a crossing that is not
    painted, on a kerb with no corner. Same mistake centerline_start_ft was making.
    """
    clearance_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets, side=side)
    reach_ft = crosswalk_reach_on_leg_side_ft(state.legs[leg_name], side, keep_clear,
                                               inner_offset_ft)
    if not crosswalk_is_marked:
        return LegAnchors(anchor_ft=clearance_ft, target_ft=clearance_ft,
                           crossing_ft=reach_ft or 0.0, clearance_ft=clearance_ft)
    if not reach_ft:
        # Marked, but no band geometry to measure against - fall back to this leg's crossing
        # centre offset. Half the crossing depth is inside CROSSWALK_CLEARANCE_FT, so this is
        # the old behaviour, and it is right for a square crossing.
        reach_ft = crosswalk_offsets[leg_name].offset_ft
    target_ft = reach_ft + CROSSWALK_CLEARANCE_FT
    return LegAnchors(anchor_ft=max(clearance_ft, target_ft), target_ft=target_ft,
                       crossing_ft=reach_ft, clearance_ft=clearance_ft)


# A run of kerb shorter than one stall cannot hold a parked car, so marking it would be
# claiming a space that isn't there.
MIN_PARKING_RUN_FT = 22.0

# How steep a taper may be before it stops reading as a taper. A taper is a TRANSITION - it
# guides a driver across a change in lane width - and only says that when it is gentle.
# Measured at Broad & Greenwood: the lane-narrowing buffers on Greenwood run 1.5 ft of depth
# across 8-11 ft of station (0.14-0.19), and read well. The parking buffers on Broad St have
# to swing 13-17 ft of depth across 0-5.6 ft (2.97 and up, one of them infinite) - that is a
# hairpin, and it is what looked wrong. 1.0 sits an order of magnitude clear of the good
# cases and well below every bad one.
MAX_TAPER_DEPTH_PER_RUN = 1.0

# How far paint keeps off a painted crossing. Small on purpose: where a crossing exists, the
# hatching is meant to run right up to it and be cut by it, which is what gives the zone its
# clean diagonal end. This is the striper's gap, not a design setback.
PAINT_TO_CROSSWALK_GAP_FT = 1.0

# How close a piece of a fill's boundary has to lie to the crossing to BE the cut edge. The
# clip puts it exactly on the buffered band, so this only absorbs float noise.
RIM_SNAP_FT = 0.05
# Below this a rim is a clipping artifact at a corner, not a painted line.
MIN_RIM_LENGTH_FT = 1.0

# The painted width of a lane-edge line, matching scripts/blender/blender_scene.py's
# add_paint_polyline(..., 0.25, ...). Paint has width, and where it goes decides whether the
# lane behind it is really the width it claims: an edge line CENTRED on the 11 ft mark puts
# half its own body inside the lane, leaving 10.59 ft. So the line is placed outside the mark
# - its inner edge lands on 11 ft - and the hatching starts outside the line rather than
# running under it. The width comes out of the treatment, which is where the spare asphalt
# is, not out of the travel lane.
LANE_EDGE_LINE_WIDTH_FT = 0.25 / FT_TO_M


def lane_edge_stripes(depth_ft: float) -> tuple[float, float]:
    """(depth for the edge LINE, depth for the FILL) given a treatment `depth_ft` deep.

    Both are measured the way lane_narrowing_polygons_ft measures a stripe width: inward from
    the kerb-to-kerb half. Shrinking them moves the treatment's lane-side boundary outward.
    """
    return (max(depth_ft - LANE_EDGE_LINE_WIDTH_FT / 2, 0.0),
            max(depth_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0))


def tapers_cleanly(depth_ft: float, at: LegAnchors) -> bool:
    """Whether a curved taper into the corner would read as one.

    Only consulted where there is NO crossing for the paint to end against. Measured at
    Broad & Greenwood: Greenwood's lane-narrowing buffers run 1.5 ft of depth across 8-11 ft
    of station (0.14-0.19) and read well; Broad St's parking buffers had to swing 13-17 ft
    across 0-5.6 ft (2.97 and up, one of them infinite), which is a hairpin.
    """
    run_ft = at.anchor_ft - at.target_ft
    return run_ft > 0 and depth_ft / run_ft <= MAX_TAPER_DEPTH_PER_RUN


def end_against_crossing(at: LegAnchors, zone_start_ft: float = 0.0) -> tuple[float, float]:
    """(start station, station below which a surviving offcut is discarded) for paint that
    should run INTO its leg's crossing and be cut by it.

    Where a crossing exists, that is what the paint should end against - it runs up to the
    crossing and the crossing trims it, which leaves the end cut along the crossing's own
    edge. On a skewed crossing that edge is a diagonal, and the diagonal meeting the straight
    lane-edge line is the right-angled corner you see on a real street. A curved taper into
    the corner is for the other case: no crossing to end against, so the paint has to resolve
    itself back to the kerb.

    Deliberately starting inside the crossing is what makes the trim do the work. It leaves
    an offcut on the junction side, hence the second return value.

    TRIED AND REVERTED: reaching the paint back to this side's own corner clearance instead,
    and discarding against that, so a side with no corner return keeps the offcut. The north
    side of E Broad at Princeton is one unbroken kerb under one continuous no-stopping
    restriction, and ~20 ft of it between the crossing and the junction node is bare. It did
    not fix that - the shared kerb way covering that stretch has two vertices, one claimed by
    each of the two collinear legs, so neither has the two points curb_line_from_points needs -
    and it put paint over a kerb and through a crossing at W Broad & Louellen, whose acute Y
    and partial tracing make the reach-back land outside the roadway. The real fix is to let
    the two legs SHARE that endpoint vertex, which is what a shared OSM node is; that means
    relaxing assign_curb_points_to_legs' one-vertex-one-leg rule, deliberately rather than in
    passing.
    """
    return max(zone_start_ft, at.crossing_ft - CROSSWALK_DEPTH_FT), at.crossing_ft


def zone_end_line_ft(leg, side: str, start_ft: float, inner_offset_ft: float):
    """The transverse line closing off the junction end of a hatched zone, or None.

    Three ways a zone can end, and until this existed only two of them were drawn. Into a
    crossing: the crossing cuts it and `rim` outlines the cut. Resolving back to the kerb:
    the taper carries the outline round. Square, against nothing: the outline simply stopped
    and the hatch strokes ended in mid-air.

    That third case is not rare, it is every leg with no painted crossing - e_broad_st_east
    among them. Such a leg has nothing to end against, and it cannot taper either, because
    leg_anchors puts anchor_ft AT target_ft where the crossing is only nominal, leaving a
    taper no run to happen over. So the zone gets a square end whether that reads well or
    not, and a square end wants a line across it.

    Returns None where the kerb has come inside the zone's own lane edge, which leaves
    nothing to draw a line across.
    """
    sign = 1 if side == "left" else -1
    curb = curb_offsets_at_stations(leg, side, np.asarray([start_ft], dtype=float))
    outer_ft = float(curb[0]) if curb is not None else sign * leg.curb_to_curb_ft / 2
    inner_ft = sign * inner_offset_ft
    if abs(outer_ft) - abs(inner_ft) < MIN_RIM_LENGTH_FT:
        return None
    return LineString([_point_at(leg.centerline, start_ft, inner_ft),
                       _point_at(leg.centerline, start_ft, outer_ft)])


def _station_of(leg, geometry) -> float:
    """A piece's mean station along its leg - enough to tell which side of a crossing it
    fell on after being cut."""
    coords = (geometry.exterior.coords if geometry.geom_type == "Polygon" else geometry.coords)
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(coords, dtype=float))
    return float(stations.mean())


def parking_runs(state, leg_name: str, side: str, crosswalk_offsets: dict,
                  props: list[dict] | None = None) -> list[tuple[float, float]]:
    """The station spans of this kerb where stalls may legally be marked."""
    return parkable_runs_ft(
        state, leg_name, side, crosswalk_offsets, props,
        physical_clearance_ft=leg_clearance_ft(leg_name, state.legs, state.corner_fillets),
        min_run_ft=MIN_PARKING_RUN_FT)


@dataclass
class PaintContext:
    """The machinery every treatment paints through, and the pieces it has painted so far.

    curbside_paint_ft used to be one function holding this as local state and a closure per
    helper, with a block per treatment reading DesignState's dicts. That is why forgetting one
    line was invisible: the bike lane's kerb hatching was `add(...)` where every other hatched
    zone was `rim(add(...))`, so in the 3D render its strokes stopped in mid-air at the crossing
    while the plan view - which outlines a fill polygon for free - looked finished.

    With the machinery in an object, a treatment can own its own markings (Treatment.paint), and
    what is shared stays shared: the crossing bands everything is cut around, the apron surfaces
    everything stops at, and the running list of pieces.
    """
    state: object
    crosswalk_offsets: dict
    center_ft: object
    keep_clear: object = None          # union of the painted crossings, buffered by the gap
    marked: set = field(default_factory=set)
    straight_through: set = field(default_factory=set)
    props: list | None = None
    surfaces: object = None            # the mountable aprons: paint stops at them
    pieces: list = field(default_factory=list)
    # Zones already placed on a kerb that runs straight through - see add(shares_a_kerb=True).
    through_painted: list = field(default_factory=list)

    def emit(self, piece: PaintPiece) -> PaintPiece:
        """Keep a piece as-is, without clipping. For the things that are not paint: an apron is
        a surface the paint stops at, and a bollard is a point standing in the road."""
        self.pieces.append(piece)
        return piece

    def add(self, kind, geometry, leg=None, side=None, beyond_ft=None, shares_a_kerb=False):
        """Clip `geometry` clear of the crossings, keep what survives, return those pieces.

        beyond_ft drops any surviving piece that fell on the JUNCTION side of the crossing.
        A zone drawn deliberately through a crossing (so the crossing cuts its end into a
        clean diagonal) leaves an offcut back at the corner, and that offcut is not paint.

        shares_a_kerb dedupes against the other zones on the same through-running kerb.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        if shares_a_kerb and self.through_painted:
            geometry = geometry.difference(unary_union(self.through_painted))
            if geometry.is_empty:
                return added
        # Cut clear of the mountable surfaces, then of the crossings - both may fragment a
        # piece, so each stage runs over everything the previous one left.
        surviving = [part for whole in clip_paint_clear_of(geometry, self.surfaces)
                     for part in clip_paint_clear_of(whole, self.keep_clear)]
        for part in surviving:
            if beyond_ft is not None and _station_of(self.state.legs[leg], part) < beyond_ft:
                continue
            piece = PaintPiece(kind, part, leg, side)
            self.pieces.append(piece)
            added.append(piece)
        if shares_a_kerb:
            self.through_painted.extend(p.geometry for p in added)
        return added

    def rim(self, fills) -> None:
        """The line along a fill's cut end, where it meets the crossing.

        A hatched zone is outlined, and that outline carries on around the end where the
        crossing cuts it - which is the diagonal you see finishing the zone off against the
        crossing on a real street. The lane edge already gets a line of its own; without this
        one the zone just stopped, with hatch strokes ending in mid-air.
        """
        if self.keep_clear is None:
            return
        for piece in fills:
            edge = piece.geometry.exterior.intersection(self.keep_clear.buffer(RIM_SNAP_FT))
            for part in getattr(edge, "geoms", [edge]):
                if part.geom_type == "LineString" and part.length >= MIN_RIM_LENGTH_FT:
                    self.pieces.append(PaintPiece(CROSSING_RIM_LINE, part, piece.leg, piece.side))

    def anchors(self, leg_name: str, side: str, inner_offset_ft: float = 0.0):
        """This leg-side's measuring stations, with the shared crossing geometry filled in."""
        return leg_anchors(self.state, leg_name, side, self.crosswalk_offsets, self.keep_clear,
                            inner_offset_ft=inner_offset_ft,
                            crosswalk_is_marked=leg_name in self.marked)


def curbside_paint_ft(state, crosswalk_offsets: dict, center_ft,
                       crosswalk_bands: dict | None = None,
                       props: list[dict] | None = None,
                       marked_crosswalks: set | None = None) -> list[PaintPiece]:
    """Every painted marking this design puts on the roadway, in state-plane feet.

    props supplies the stop signs and fire hydrants that carry statutory parking setbacks of
    their own (see src/geometry/daylighting.py). Without them the daylight zone is computed
    from the crossing and the side line only, which is right for those two rules and short
    of the law wherever a sign or hydrant governs instead.

    crosswalk_bands does two jobs. Each treatment's taper is aimed to stop just short of
    where that leg's crossing really reaches on that side (see leg_anchors), and the union
    of the bands is then subtracted from everything as a backstop, because markings are
    layered by priority and a crossing outranks a buffer or a parking lane. Aiming correctly
    is what makes the paint LOOK right - the subtraction alone leaves a taper chopped off
    square where it ran into the crossing.
    """
    # Only crossings that are actually PAINTED get out of the way of anything. Every leg
    # gets a resolved offset, including ones with no marking today - cutting paint around
    # those was reserving room for a crossing that isn't there.
    marked = set(marked_crosswalks) if marked_crosswalks is not None else set(state.legs)
    # Kerbs with no corner return at the junction: the kerb runs straight through, so the
    # crossing cuts the zone in two and BOTH halves are paint. Everywhere else the piece on
    # the junction side of a crossing sits in the corner throat and is discarded.
    straight_through = through_street_sides(state.legs)
    bands = {name: band for name, band in (crosswalk_bands or {}).items()
             if band is not None and not band.is_empty and name in marked}
    keep_clear = (unary_union(list(bands.values())).buffer(PAINT_TO_CROSSWALK_GAP_FT)
                   if bands else None)
    pieces: list[PaintPiece] = []

    # The mountable aprons, built FIRST because paint has to stop at them. An apron is a
    # SURFACE - flush pavers or textured concrete, part of the corner rather than part of the
    # carriageway - so the ground it occupies is not roadway to be hatched. Left unsubtracted,
    # a curb extension's daylight zone and its own swept-path apron overlapped by 2-4 sq ft at
    # three of the four corners at Broad & Greenwood, which markings_collide reported as the
    # same ground painted twice. It is the same layering the crossings already get, one rung
    # further up: a surface outranks a marking the way a marking outranks a buffer.
    #
    # No gap buffer, unlike a crossing: paint runs up to an apron's edge and stops there. The
    # striper's gap around a crossing (PAINT_TO_CROSSWALK_GAP_FT) exists because both are paint.
    aprons = []
    for corner, apron in sorted(state.corner_aprons.items()):
        if "error" in state.corner_fillets[corner]:
            continue
        polygon = _apron_polygon(state, corner, apron, center_ft)
        if polygon is not None:
            aprons.append(polygon)
            pieces.append(PaintPiece(APRON, polygon))
    surfaces = unary_union(aprons) if aprons else None

    # Zones already placed on a kerb that runs straight through. The two legs sharing such a
    # kerb both paint up to the junction node, and where they meet at an angle their strips
    # overlap in the wedge between the two frames - 5.6 sq ft at W Broad & Louellen, whose
    # legs are 17.3 deg off collinear. The same ground painted twice is a markings_collide
    # violation and, on asphalt, doubled ink. Whichever zone is built first keeps the wedge
    # and the second takes the remainder, so they butt instead of overlapping and no gap
    # opens between them.
    through_painted: list = []

    def add(kind, geometry, leg=None, side=None, beyond_ft=None, shares_a_kerb=False):
        """Clip `geometry` clear of the crossings, keep what survives, return those pieces.

        beyond_ft drops any surviving piece that fell on the JUNCTION side of the crossing.
        A zone drawn deliberately through a crossing (so the crossing cuts its end into a
        clean diagonal) leaves an offcut back at the corner, and that offcut is not paint.

        shares_a_kerb dedupes against the other zones on the same through-running kerb.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        if shares_a_kerb and through_painted:
            geometry = geometry.difference(unary_union(through_painted))
            if geometry.is_empty:
                return added
        # Cut clear of the mountable surfaces, then of the crossings - both may fragment a
        # piece, so each stage runs over everything the previous one left.
        surviving = [part for whole in clip_paint_clear_of(geometry, surfaces)
                     for part in clip_paint_clear_of(whole, keep_clear)]
        for part in surviving:
            if beyond_ft is not None and _station_of(state.legs[leg], part) < beyond_ft:
                continue
            piece = PaintPiece(kind, part, leg, side)
            pieces.append(piece)
            added.append(piece)
        if shares_a_kerb:
            through_painted.extend(p.geometry for p in added)
        return added

    def rim(fills):
        """The line along a fill's cut end, where it meets the crossing.

        A hatched zone is outlined, and that outline carries on around the end where the
        crossing cuts it - which is the diagonal you see finishing the zone off against the
        crossing on a real street. The lane edge already gets a line of its own; without this
        one the zone just stopped, with hatch strokes ending in mid-air.
        """
        if keep_clear is None:
            return
        for piece in fills:
            edge = piece.geometry.exterior.intersection(keep_clear.buffer(RIM_SNAP_FT))
            for part in getattr(edge, "geoms", [edge]):
                if part.geom_type == "LineString" and part.length >= MIN_RIM_LENGTH_FT:
                    pieces.append(PaintPiece(CROSSING_RIM_LINE, part, piece.leg, piece.side))

    # --- paint-only lane narrowing: an edge line, a hatched buffer, and a taper back to
    # the curb. line_only legs get the boundary lines without the fill.
    for leg_name, stripe_width_ft in sorted(state.lane_narrowing.items()):
        leg = state.legs[leg_name]
        sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
        fill = leg_name not in state.lane_narrowing_line_only

        for side in sides:
            at = leg_anchors(state, leg_name, side, crosswalk_offsets, keep_clear,
                              inner_offset_ft=leg.curb_to_curb_ft / 2 - stripe_width_ft,
                              crosswalk_is_marked=leg_name in marked)
            # A crossing is something to end against: run into it and let it cut the end.
            # Only where there is none does the paint have to resolve itself back to the
            # kerb, and only then is a taper the right way to do it.
            if (leg_name, side) in straight_through:
                # One unbroken kerb under one restriction, with no corner return at either
                # end of it: run from the junction NODE and let any crossing cut it, keeping
                # both halves. Tested before the marked/unmarked split because it applies to
                # both - the two E Broad legs' north kerbs are one kerb, and the zones on
                # them have to meet at the node rather than each stopping a few feet short of
                # it. Discarding the junction-side half left ~20 ft of a no-stopping kerb
                # bare between the crossing and the node, with no corner there to justify it.
                start_ft, beyond_ft, curved = 0.0, None, False
            elif leg_name in marked:
                start_ft, beyond_ft = end_against_crossing(at)
                curved = False
            else:
                curved = tapers_cleanly(stripe_width_ft, at)
                start_ft, beyond_ft = (at.anchor_ft if curved else at.target_ft), None
            line_ft, fill_ft = lane_edge_stripes(stripe_width_ft)
            add(LANE_EDGE_LINE, _one(lane_narrowing_edge_lines_ft(
                leg, line_ft, start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,),
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)), leg_name, side, beyond_ft)
            if curved:
                add(TAPER_LINE, _one(lane_narrowing_taper_ft(
                    leg, line_ft, at.anchor_ft, at.target_ft, sides=(side,))), leg_name, side)
            if fill:
                rim(add(LANE_NARROWING_FILL, _one(lane_narrowing_polygons_ft(
                    leg, fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                    sides=(side,))), leg_name, side, beyond_ft,
                    shares_a_kerb=(leg_name, side) in straight_through))
                if curved:
                    add(TAPER_FILL, _one(lane_narrowing_taper_polygons_ft(
                        leg, fill_ft, at.anchor_ft, at.target_ft, sides=(side,))),
                        leg_name, side)
                elif leg_name not in marked and (leg_name, side) not in straight_through:
                    # Not on a kerb that runs straight through: the zone does not END at the
                    # junction node, it continues into the adjoining leg's zone on the same
                    # unbroken kerb. Closing it off drew a line across the hatching in the
                    # middle of the intersection.
                    add(ZONE_END_LINE, zone_end_line_ft(
                        leg, side, start_ft, leg.curb_to_curb_ft / 2 - fill_ft),
                        leg_name, side)

        if leg_name in state.bollard_lines:
            for point in bollard_points_ft(leg, stripe_width_ft,
                                            leg_clearance_ft(leg_name, state.legs, state.corner_fillets),
                                            state.bollard_lines[leg_name], sides=sides):
                pieces.append(PaintPiece(BOLLARD, _dot(point), leg_name, None))

    # --- marked curbside parking: stalls, plus the hatched buffer between them and the curb
    for (leg_name, side), zone in sorted(state.parking_zones.items()):
        leg = state.legs[leg_name]
        depth_ft, stall_length_ft = zone["depth_ft"], zone["stall_length_ft"]
        curb_offset_ft = zone["curb_offset_ft"]
        at = leg_anchors(state, leg_name, side, crosswalk_offsets, keep_clear,
                          inner_offset_ft=leg.curb_to_curb_ft / 2 - depth_ft - curb_offset_ft,
                          crosswalk_is_marked=leg_name in marked)
        runs = parking_runs(state, leg_name, side, crosswalk_offsets, props)

        # DAYLIGHTING. Every stretch where R.S. 39:4-138 forbids parking is hatched across
        # the FULL depth of the parking lane, not just the buffer strip beside it. Those
        # stretches were already no-parking in law - the treatment is MARKING them, because
        # an unmarked setback is one people park in, and an unmarked setback next to a
        # marked stall reads as more stall. This is the part of the proposal that actually
        # daylights the crossing. Zones are clipped to the leg and to the point where the
        # corner return leaves room to paint at all.
        #
        # The zone runs INTO the crossing and the crossing cuts its end, leaving it rimmed
        # along the crossing's own edge - a diagonal where the crossing is skewed, meeting
        # the straight lane-edge line at a right angle. It used to end in the same curved
        # taper a lane-narrowing buffer gets, and on a wide leg that curve is a hairpin: at
        # Broad St it had to swing the full 13-17 ft depth of the parking lane across 0-5.6 ft
        # of station. Where a leg has no marked crossing there is nothing to end against, so
        # it falls back to a taper if a gentle one exists and a square cut otherwise.
        daylight_line_ft, daylight_fill_ft = lane_edge_stripes(depth_ft + curb_offset_ft)
        lane_edge_offset_ft = leg.curb_to_curb_ft / 2 - daylight_line_ft
        for zone_start_ft, zone_end_ft in merged_no_parking_spans_ft(
                no_parking_zones_ft(state, leg_name, side, crosswalk_offsets, props)):
            if leg_name in marked and (leg_name, side) in straight_through:
                start_ft, beyond_ft = zone_start_ft, None
            elif leg_name in marked:
                start_ft, beyond_ft = end_against_crossing(at, zone_start_ft)
            else:
                start_ft, beyond_ft = max(zone_start_ft, at.target_ft), None
            rim(add(DAYLIGHT_FILL, _one(lane_narrowing_polygons_ft(
                leg, daylight_fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=zone_end_ft)), leg_name, side, beyond_ft,
                shares_a_kerb=(leg_name, side) in straight_through))
            # A solid line wherever hatching meets the travel lane, so the lane reads as a
            # lane. The buffer beside the stalls already has one; the daylight zone runs the
            # full depth of the parking lane, so ITS inner edge is the lane edge, and without
            # this the hatching just faded into the carriageway.
            add(DAYLIGHT_EDGE_LINE,
                inset_line_ft(leg, side, lane_edge_offset_ft, start_ft, zone_end_ft,
                               keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                leg_name, side, beyond_ft)
            # Nothing to end against and no taper available: close the square end. See
            # zone_end_line_ft. Not where the kerb runs straight through - the zone carries
            # on into the next leg there.
            if leg_name not in marked and (leg_name, side) not in straight_through:
                add(ZONE_END_LINE, zone_end_line_ft(
                    leg, side, start_ft, leg.curb_to_curb_ft / 2 - daylight_fill_ft),
                    leg_name, side)

        for start_ft, end_ft in runs:
            # ORDER ACROSS THE ROAD, and what gives when the road's width changes:
            #
            #   travel lane   0 -> TARGET             fixed
            #   lane edge line                        its own width, out of the treatment
            #   parking       -> TARGET + depth_ft    fixed, held against the LANE
            #   HATCHING      -> the traced kerb      absorbs ALL of the variation
            #
            # Everything is measured from the centerline, so the only thing that touches the
            # traced kerb is the hatching - which is just paint filling whatever asphalt is
            # left over. The lane holds its width, which is the entire point of the markings:
            # a lane that widens is a lane people speed in. The stall holds its width too, so
            # the leftover cannot end up inside it.
            #
            # (Anchoring the stalls to the KERB instead was tried and is wrong here: it makes
            # the parking position depend on the noisiest input in the model, and puts the
            # variable-width hatching between the travel lane and the parked cars.)
            edge = parking_lane_edge_line_ft(
                leg, side, depth_ft, start_ft, end_ft,
                curb_offset_ft=curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT / 2)
            if edge is None:
                continue  # the corner return consumes the whole leg - see plan_view's note
            add(PARKING_EDGE_LINE, edge, leg_name, side)
            for divider in parking_stall_lines_ft(
                    leg, side, depth_ft, stall_length_ft, start_ft, end_ft,
                    curb_offset_ft=curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT):
                add(STALL_DIVIDER, divider, leg_name, side)

            if not curb_offset_ft:
                continue
            buffer_ft = max(curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0)
            add(BUFFER_EDGE_LINE, inset_line_ft(
                leg, side, leg.curb_to_curb_ft / 2 - buffer_ft, start_ft, end_ft,
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2), leg_name, side)
            add(BUFFER_FILL, _one(lane_narrowing_polygons_ft(
                leg, buffer_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=end_ft)), leg_name, side)
            if (leg_name, side) in state.parking_buffer_bollards:
                for point in bollard_points_ft(leg, curb_offset_ft, start_ft,
                                                state.parking_buffer_bollards[(leg_name, side)],
                                                sides=(side,)):
                    pieces.append(PaintPiece(BOLLARD, _dot(point), leg_name, side))

    # --- exclusive bike lanes. An edge line each side of the lane, so it reads as a lane
    # rather than as the spare asphalt a lane-narrowing buffer marks; the buffer beside it, and
    # the parking outside it, are hatched and ticked with the machinery already here.
    for (leg_name, side), lane in sorted(state.bike_lanes.items()):
        leg = state.legs[leg_name]
        at = leg_anchors(state, leg_name, side, crosswalk_offsets, keep_clear,
                          inner_offset_ft=leg.curb_to_curb_ft / 2 - lane.total_ft + TARGET_LANE_WIDTH_FT,
                          crosswalk_is_marked=leg_name in marked)
        # A bike lane RUNS INTO its crossing and is cut by it, like every other kerbside zone
        # here - a real one carries on to the crossing and often across it. Stopping it at the
        # corner clearance instead left the buffer 5.5 ft short of the crossing, which
        # test_curbside_paint_ends_against_its_crossing reads as hatching that gave up early.
        if (leg_name, side) in straight_through:
            start_ft, beyond_ft = 0.0, None
        elif leg_name in marked:
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        bounds = lane.offsets_from_centerline_ft()
        # Every stripe at its own CENTRE, which BikeLane has already offset half a stripe out
        # from the face it marks - so the travel lane keeps its 11 ft and the bike lane keeps
        # its own width, and the paint comes out of the buffer between them. Getting this wrong
        # is not subtle: an edge line centred on the mark leaves a 10.59 ft lane, which
        # check_paint_clear_of_the_travel_lane reports on every vertex.
        for key in ("inner_line_ft", "buffer_outer_line_ft", "outer_line_ft"):
            if bounds[key] is None:
                continue
            add(BIKE_LANE_EDGE_LINE,
                inset_line_ft(leg, side, bounds[key], start_ft,
                               keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                leg_name, side, beyond_ft)
        if lane.buffer_ft:
            # The hatched buffer, between the two lines that bound it rather than under them.
            # lane_narrowing_polygons_ft measures its stripe inward from the kerb-to-kerb half,
            # so the depth is the distance from the kerb to the buffer's inner FACE, and the
            # zone is then cut back to the buffer's outer face.
            inner_face_ft = bounds["travel_lane_edge_ft"] + LANE_EDGE_LINE_WIDTH_FT
            fill = _one(lane_narrowing_polygons_ft(
                leg, leg.curb_to_curb_ft / 2 - inner_face_ft,
                start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,)))
            outer_face_ft = bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT
            beyond = curbside_strip_polygon(leg, side, outer_face_ft, start_ft)
            if fill is not None and beyond is not None:
                fill = fill.difference(beyond)
            rim(add(BIKE_BUFFER_FILL, fill, leg_name, side, beyond_ft))
        if lane.parking_ft:
            # Parking-protected: the stalls sit OUTSIDE the bike lane, between it and the kerb,
            # which is what shields the lane. Ticked at the standard stall length over the runs
            # where parking is legal, exactly as a kerbside parking lane would be.
            for run_start_ft, run_end_ft in parking_runs(state, leg_name, side,
                                                          crosswalk_offsets, props):
                for divider in parking_stall_lines_ft(
                        leg, side, lane.parking_ft, PARKING_STALL_LENGTH_DEFAULT_FT,
                        max(run_start_ft, start_ft), run_end_ft,
                        curb_offset_ft=lane.shy_ft):
                    add(STALL_DIVIDER, divider, leg_name, side)
        else:
            # The leftover between the lane's outer stripe and the kerb, hatched. A bike lane is
            # a standard width and the street's spare asphalt is not part of it - the same
            # accounting an 8 ft parking stall gets, where the remainder becomes the kerb buffer
            # rather than a wider stall. Without this the lane read as reaching the kerb, which
            # is what made the drawn lanes look far wider than they are.
            #
            # Rimmed, like every other hatched zone here. The plan view outlines a fill polygon
            # for free, so this zone read as finished in 2D while the 3D render - which gets
            # only the hatch strokes and the lines actually painted - had its strokes stopping
            # in mid-air where the crossing cut them. See rim().
            rim(add(BUFFER_FILL, _one(lane_narrowing_polygons_ft(
                leg, leg.curb_to_curb_ft / 2 - bounds["outer_ft"],
                start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,))),
                leg_name, side, beyond_ft))
        if (leg_name, side) in state.bike_lane_bollards:
            # Down the middle of the buffer, on the TRAFFIC side of the lane - the side a rider
            # needs protecting from. add_bike_lane_bollards refuses a lane with no buffer, so
            # there is always a strip to centre them in here.
            #
            # Started at target_ft, not at the zone's own start_ft. A marked leg's paint
            # deliberately begins INSIDE the crossing so the crossing cuts its end
            # (end_against_crossing), and a post is not paint: it cannot be trimmed by a
            # crossing, it would simply be standing in one. target_ft is the first station clear
            # of where the crossing actually reaches on this side.
            centre_ft = (bounds["travel_lane_edge_ft"] + bounds["bike_inner_ft"]) / 2
            for point in points_at_offset_ft(
                    leg, side, centre_ft, max(start_ft, at.target_ft),
                    spacing_ft=state.bike_lane_bollards[(leg_name, side)]):
                pieces.append(PaintPiece(BOLLARD, _dot(point), leg_name, side))

    # --- and whatever the treatments paint for themselves. Each one that owns its markings
    # (Treatment.paint) is dispatched here in painting order, so its geometry lives with its
    # validation and its provenance instead of in a block of this function keyed off a dict.
    # The blocks above are the ones not yet moved; see the README.
    ctx = PaintContext(state=state, crosswalk_offsets=crosswalk_offsets, center_ft=center_ft,
                        keep_clear=keep_clear, marked=marked,
                        straight_through=straight_through, props=props, surfaces=surfaces,
                        pieces=pieces, through_painted=through_painted)
    for treatment in sorted(getattr(state, "treatments", []),
                             key=lambda t: (t.paint_group, str(t.target), t.paint_rank)):
        treatment.paint(ctx)
    return ctx.pieces      # the aprons were built first - see `surfaces` above


def _apron_polygon(state, corner: tuple[str, str], apron, center_ft):
    """The ground one CornerApron covers - a fixed-depth kite, or the swept-path annulus.

    Two shapes because there are two reasons for an apron; see
    src/geometry/treatments.py:CornerApron. The annulus is the one a curb extension lays, and it
    is built from the corner's two real curb lines rather than offset off the drawn arc, so the
    outer edge genuinely reaches the radius a bus needs.
    """
    if apron.swept_radius_ft is None:
        return corner_overlay_polygon(state.corner_fillets[corner], center_ft, apron.depth_ft)
    leg_a, leg_b = corner
    return corner_apron_annulus(state.legs[leg_a].left_curb, state.legs[leg_b].right_curb,
                                 apron.face_radius_ft, apron.swept_radius_ft)


def of_kind(pieces: list[PaintPiece], *kinds: PaintKind) -> list[PaintPiece]:
    return [p for p in pieces if p.kind in kinds]


def in_channel(pieces: list[PaintPiece], channel) -> list[PaintPiece]:
    """Every piece the 3D render will find in one of its JSON lists (markings.Channel).

    The export used to name the kinds per list itself, in a table beside the one in
    src/geometry/markings.py - so a marking could be declared and still reach no renderer.
    """
    return [p for p in pieces if p.kind.channel is channel]


def _one(geometries):
    """These builders take a `sides` tuple and return a list; called per side they return at
    most one. Unpacking here keeps the caller from pretending otherwise."""
    return geometries[0] if geometries else None


def _dot(point) -> Polygon:
    """A bollard is a point, but PaintPiece holds geometry so the curb check can treat every
    piece the same way. A degenerate square is the cheapest honest polygon for one."""
    x, y = point
    return Polygon([(x, y), (x + 1e-6, y), (x + 1e-6, y + 1e-6), (x, y + 1e-6)])
