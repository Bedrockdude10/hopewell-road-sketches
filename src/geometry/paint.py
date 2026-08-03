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
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.geometry.model import (bollard_points_ft, clip_paint_clear_of, corner_overlay_polygon,
                                inset_line_ft, lane_narrowing_edge_lines_ft,
                                lane_narrowing_polygons_ft, lane_narrowing_taper_ft,
                                lane_narrowing_taper_polygons_ft, leg_clearance_ft,
                                parking_lane_edge_line_ft, parking_stall_lines_ft)
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
    """
    kind: str
    geometry: LineString | Polygon
    leg: str | None = None
    side: str | None = None

    @property
    def is_fill(self) -> bool:
        """A polygon to be hatched (3D) or filled with a hatch pattern (2D), vs. a line."""
        return self.geometry.geom_type == "Polygon"


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


def leg_anchors(state, leg_name: str, side: str, crosswalk_offsets: dict,
                 keep_clear=None, inner_offset_ft: float = 0.0) -> LegAnchors:
    """inner_offset_ft is how far from the centerline this treatment's paint starts - the
    lane edge. Only the crossing inside that strip can get in its way."""
    clearance_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
    reach_ft = crosswalk_reach_on_leg_side_ft(state.legs[leg_name], side, keep_clear,
                                               inner_offset_ft)
    if not reach_ft:
        # No crossing geometry to measure against - fall back to this leg's crossing centre
        # offset. Half the crossing depth is inside CROSSWALK_CLEARANCE_FT, so this is the
        # old behaviour, and it is right for a square crossing.
        reach_ft = crosswalk_offsets[leg_name][0]
    target_ft = reach_ft + CROSSWALK_CLEARANCE_FT
    return LegAnchors(anchor_ft=max(clearance_ft, target_ft), target_ft=target_ft,
                       crossing_ft=reach_ft)


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
    """
    return max(zone_start_ft, at.crossing_ft - CROSSWALK_DEPTH_FT), at.crossing_ft


def _station_of(leg, geometry) -> float:
    """A piece's mean station along its leg - enough to tell which side of a crossing it
    fell on after being cut."""
    import numpy as np

    from src.geometry.model import station_offset_many

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
    bands = {name: band for name, band in (crosswalk_bands or {}).items()
             if band is not None and not band.is_empty and name in marked}
    keep_clear = (unary_union(list(bands.values())).buffer(PAINT_TO_CROSSWALK_GAP_FT)
                   if bands else None)
    pieces: list[PaintPiece] = []

    def add(kind, geometry, leg=None, side=None, beyond_ft=None):
        """Clip `geometry` clear of the crossings, keep what survives, return those pieces.

        beyond_ft drops any surviving piece that fell on the JUNCTION side of the crossing.
        A zone drawn deliberately through a crossing (so the crossing cuts its end into a
        clean diagonal) leaves an offcut back at the corner, and that offcut is not paint.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        for part in clip_paint_clear_of(geometry, keep_clear):
            if beyond_ft is not None and _station_of(state.legs[leg], part) < beyond_ft:
                continue
            piece = PaintPiece(kind, part, leg, side)
            pieces.append(piece)
            added.append(piece)
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
                    pieces.append(PaintPiece("crossing_rim_line", part, piece.leg, piece.side))

    # --- paint-only lane narrowing: an edge line, a hatched buffer, and a taper back to
    # the curb. line_only legs get the boundary lines without the fill.
    for leg_name, stripe_width_ft in sorted(state.lane_narrowing.items()):
        leg = state.legs[leg_name]
        sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
        fill = leg_name not in state.lane_narrowing_line_only

        for side in sides:
            at = leg_anchors(state, leg_name, side, crosswalk_offsets, keep_clear,
                              inner_offset_ft=leg.curb_to_curb_ft / 2 - stripe_width_ft)
            # A crossing is something to end against: run into it and let it cut the end.
            # Only where there is none does the paint have to resolve itself back to the
            # kerb, and only then is a taper the right way to do it.
            if leg_name in marked:
                start_ft, beyond_ft = end_against_crossing(at)
                curved = False
            else:
                curved = tapers_cleanly(stripe_width_ft, at)
                start_ft, beyond_ft = (at.anchor_ft if curved else at.target_ft), None
            line_ft, fill_ft = lane_edge_stripes(stripe_width_ft)
            add("lane_edge_line", _one(lane_narrowing_edge_lines_ft(
                leg, line_ft, start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,),
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)), leg_name, side, beyond_ft)
            if curved:
                add("taper_line", _one(lane_narrowing_taper_ft(
                    leg, line_ft, at.anchor_ft, at.target_ft, sides=(side,))), leg_name, side)
            if fill:
                rim(add("lane_narrowing_fill", _one(lane_narrowing_polygons_ft(
                    leg, fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                    sides=(side,))), leg_name, side, beyond_ft))
                if curved:
                    add("taper_fill", _one(lane_narrowing_taper_polygons_ft(
                        leg, fill_ft, at.anchor_ft, at.target_ft, sides=(side,))),
                        leg_name, side)

        if leg_name in state.bollard_lines:
            for point in bollard_points_ft(leg, stripe_width_ft,
                                            leg_clearance_ft(leg_name, state.legs, state.corner_fillets),
                                            state.bollard_lines[leg_name], sides=sides):
                pieces.append(PaintPiece("bollard", _dot(point), leg_name, None))

    # --- marked curbside parking: stalls, plus the hatched buffer between them and the curb
    for (leg_name, side), zone in sorted(state.parking_zones.items()):
        leg = state.legs[leg_name]
        depth_ft, stall_length_ft = zone["depth_ft"], zone["stall_length_ft"]
        curb_offset_ft = zone["curb_offset_ft"]
        at = leg_anchors(state, leg_name, side, crosswalk_offsets, keep_clear,
                          inner_offset_ft=leg.curb_to_curb_ft / 2 - depth_ft - curb_offset_ft)
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
            if leg_name in marked:
                start_ft, beyond_ft = end_against_crossing(at, zone_start_ft)
            else:
                start_ft, beyond_ft = max(zone_start_ft, at.target_ft), None
            rim(add("daylight_fill", _one(lane_narrowing_polygons_ft(
                leg, daylight_fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=zone_end_ft)), leg_name, side, beyond_ft))
            # A solid line wherever hatching meets the travel lane, so the lane reads as a
            # lane. The buffer beside the stalls already has one; the daylight zone runs the
            # full depth of the parking lane, so ITS inner edge is the lane edge, and without
            # this the hatching just faded into the carriageway.
            add("daylight_edge_line",
                inset_line_ft(leg, side, lane_edge_offset_ft, start_ft, zone_end_ft,
                               keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                leg_name, side, beyond_ft)

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
            add("parking_edge_line", edge, leg_name, side)
            for divider in parking_stall_lines_ft(
                    leg, side, depth_ft, stall_length_ft, start_ft, end_ft,
                    curb_offset_ft=curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT):
                add("stall_divider", divider, leg_name, side)

            if not curb_offset_ft:
                continue
            buffer_ft = max(curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0)
            add("buffer_edge_line", inset_line_ft(
                leg, side, leg.curb_to_curb_ft / 2 - buffer_ft, start_ft, end_ft,
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2), leg_name, side)
            add("buffer_fill", _one(lane_narrowing_polygons_ft(
                leg, buffer_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=end_ft)), leg_name, side)
            if (leg_name, side) in state.parking_buffer_bollards:
                for point in bollard_points_ft(leg, curb_offset_ft, start_ft,
                                                state.parking_buffer_bollards[(leg_name, side)],
                                                sides=(side,)):
                    pieces.append(PaintPiece("bollard", _dot(point), leg_name, side))

    # --- corner treatments. No leg/side: they span the corner between two legs.
    for corner, depth_ft in sorted(state.corner_hatching.items()):
        if "error" in state.corner_fillets[corner]:
            continue
        add("corner_hatch_fill", corner_overlay_polygon(state.corner_fillets[corner], center_ft, depth_ft))
    for corner, extent_ft in sorted(state.corner_aprons.items()):
        if "error" in state.corner_fillets[corner]:
            continue
        pieces.append(PaintPiece("apron", corner_overlay_polygon(state.corner_fillets[corner],
                                                                  center_ft, extent_ft)))
    return pieces


def of_kind(pieces: list[PaintPiece], *kinds: str) -> list[PaintPiece]:
    return [p for p in pieces if p.kind in kinds]


def _one(geometries):
    """These builders take a `sides` tuple and return a list; called per side they return at
    most one. Unpacking here keeps the caller from pretending otherwise."""
    return geometries[0] if geometries else None


def _dot(point) -> Polygon:
    """A bollard is a point, but PaintPiece holds geometry so the curb check can treat every
    piece the same way. A degenerate square is the cheapest honest polygon for one."""
    x, y = point
    return Polygon([(x, y), (x + 1e-6, y), (x + 1e-6, y + 1e-6), (x, y + 1e-6)])
