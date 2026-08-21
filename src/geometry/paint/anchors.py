"""WHERE ON A LEG-SIDE A TREATMENT MAY PAINT AT ALL, and what is too small to draw.

Every curbside treatment on one side of one leg is bounded by the same two stations, so they are
resolved ONCE into `LegAnchors` rather than each treatment re-deriving them - which is how one leg
came to be held back 21.5 ft by a limiter another treatment had already cleared.

THE MINIMUMS ARE HERE TOO, and they are refusals rather than tolerances: a zone below
MIN_ZONE_AREA_SQ_FT or a line below MIN_LINE_LENGTH_FT is not drawn small, it is not drawn. A
sliver that survives a clip is a rendering artefact, and it reads on the sheet as real paint.
"""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import LineString
from src.geometry.model import curb_offsets_at_stations, leg_clearance_ft, point_at, station_offset_many
from src.geometry.daylighting import parkable_runs_ft
from src.render.crosswalks import CROSSWALK_CLEARANCE_FT, CROSSWALK_DEPTH_FT, crosswalk_reach_on_leg_side_ft
from src.geometry.paint.pieces import LANE_EDGE_LINE_WIDTH_FT
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.treatments.state import DesignState

@dataclass(frozen=True)
class LegAnchors:
    """The two stations every curbside treatment on ONE SIDE of a leg is measured from.

    target_ft - where a taper meets the real curb, and so the closest to the junction any of
                this paint gets: CROSSWALK_CLEARANCE_FT beyond where the crossing's paint
                actually reaches on this side.
    anchor_ft - where a paint-only buffer's straight run begins. Past the corner return AND
                past the crossing; the corner clearance alone is not enough.

    Per SIDE, not per leg, because a skewed crossing reaches further along one kerb than the
    other - 9.4 ft further at broad_st_west, so a single per-leg target either overlaps the
    crossing on one side or leaves a gap on the other.

    Where marked STALLS may begin is also not a property of the leg - the side line, the
    stop signs and the hydrants all differ by side. See src/geometry/daylighting.py.
    """
    anchor_ft: float
    target_ft: float
    crossing_ft: float = 0.0     # where the crossing's paint actually reaches on this side
    clearance_ft: float = 0.0    # past THIS SIDE's corner return, if it has one


def leg_anchors(state: "DesignState", leg_name: str, side: str, crosswalk_offsets: dict,
                 keep_clear=None, inner_offset_ft: float = 0.0,
                 crosswalk_is_marked: bool = True,
                 mouth_end_ft: float | None = None) -> LegAnchors:
    """This leg-side's LegAnchors.

    inner_offset_ft is how far from the centerline this treatment's paint starts - the lane
    edge. Only the crossing inside that strip can get in its way.

    Clearance is asked PER SIDE. This paint belongs to one kerb, and a corner return belongs
    to one side of each leg it touches, so a per-leg maximum holds the paint back for a curve
    that may be on the opposite kerb. See leg_clearance_ft.

    With no painted crossing on this leg there is nothing to keep clear OF, so no striper's gap
    is reserved - `mouth_end_ft`, where this junction's mouth ends on this kerb, IS the target.
    The zone begins where the intersection stops, which is the same rule the painted case gets;
    what it does NOT get is CROSSWALK_CLEARANCE_FT, because holding paint off a crossing that is
    not painted leaves a bare stretch for nothing.

    WHY NOT THE CORNER RETURN, WHICH IS WHAT THIS USED TO ANSWER. It is the same holdback the
    mouth already applies, said a second time and less well: the tangent point scales as
    1/tan(theta/2), so on an acute Y it runs far past the crossing and the treatment starved the
    corner it exists to protect - 31.7 ft of bare kerb inside a statutory no-parking zone on
    W Broad & Louellen's northwest kerb, 14.6 ft at Princeton & E Prospect, 5.7 ft at E Broad.
    Every one of those legs had `paint@ == clearance_ft` exactly. junction_mouths_ft is where
    "where does the intersection end on this kerb" lives now, for the cut AND for the aim; the
    clearance stays the fallback for a caller with no scene behind it, and stays on the returned
    LegAnchors because a treatment may still want to know.
    """
    clearance_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets, side=side)
    reach_ft = crosswalk_reach_on_leg_side_ft(state.legs[leg_name], side, keep_clear,
                                               inner_offset_ft)
    if not crosswalk_is_marked:
        start_ft = clearance_ft if mouth_end_ft is None else mouth_end_ft
        return LegAnchors(anchor_ft=start_ft, target_ft=start_ft,
                           crossing_ft=reach_ft or start_ft, clearance_ft=clearance_ft)
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

# How steep a taper may be before it stops reading as a taper (depth per run, dimensionless).
# A taper is a TRANSITION and only says that when it is gentle. Measured at Broad & Greenwood:
# Greenwood's lane-narrowing buffers run 0.14-0.19 and read well; Broad St's parking buffers had
# to swing 2.97 and up, which is a hairpin. 1.0 sits clear of both.
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

# Below this a zone is a HAIRLINE LEFT BY A CLIP, not a marking: differencing polygons that share
# an edge leaves slivers along it, and a zone with no area is not paint. A real hatched zone here
# is tens to hundreds of square feet, so this cannot reach one.
MIN_ZONE_AREA_SQ_FT = 1.0

# And the same thing for a LINE: a clip landing on a vertex leaves a LineString of near-zero
# length, drawn as a stray tick with nothing attached to it. Well under a stall divider (the
# shortest real line here, a few feet), so it cannot reach a marking anyone meant to draw.
MIN_LINE_LENGTH_FT = 0.25


def lane_edge_stripes(depth_ft: float) -> tuple[float, float]:
    """(depth for the edge LINE, depth for the FILL) given a treatment `depth_ft` deep.

    Both are measured the way lane_narrowing_polygons_ft measures a stripe width: inward from
    the kerb-to-kerb half. Shrinking them moves the treatment's lane-side boundary outward.
    """
    return (max(depth_ft - LANE_EDGE_LINE_WIDTH_FT / 2, 0.0),
            max(depth_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0))


def tapers_cleanly(depth_ft: float, at: LegAnchors) -> bool:
    """Whether a curved taper into the corner would read as one.

    Only consulted where there is NO crossing for the paint to end against. The threshold and
    the measurements behind it are MAX_TAPER_DEPTH_PER_RUN's.
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

    TRIED AND REVERTED: reaching the paint back to this side's own corner clearance instead. It
    puts paint over a kerb and through a crossing at W Broad & Louellen, whose acute Y and
    partial tracing make the reach-back land outside the roadway. The bare ~20 ft it was meant to
    fill on E Broad north needs the two collinear legs to SHARE their endpoint vertex, which means
    relaxing assign_curb_points_to_legs' one-vertex-one-leg rule.
    """
    return max(zone_start_ft, at.crossing_ft - CROSSWALK_DEPTH_FT), at.crossing_ft


def zone_end_line_ft(leg, side: str, start_ft: float, inner_offset_ft: float):
    """The transverse line closing off the junction end of a hatched zone, or None.

    Three ways a zone can end. Into a crossing: the crossing cuts it and `rim` outlines the cut.
    Resolving back to the kerb: the taper carries the outline round. Square, against nothing -
    which is every leg with no painted crossing, and such a leg cannot taper either, because
    leg_anchors puts anchor_ft AT target_ft where the crossing is only nominal. A square end wants
    a line across it, or the hatch strokes end in mid-air.

    Returns None where the kerb has come inside the zone's own lane edge, which leaves
    nothing to draw a line across.
    """
    sign = 1 if side == "left" else -1
    curb = curb_offsets_at_stations(leg, side, np.asarray([start_ft], dtype=float))
    outer_ft = float(curb[0]) if curb is not None else sign * leg.curb_to_curb_ft / 2
    inner_ft = sign * inner_offset_ft
    if abs(outer_ft) - abs(inner_ft) < MIN_RIM_LENGTH_FT:
        return None
    return LineString([point_at(leg.centerline, start_ft, inner_ft),
                       point_at(leg.centerline, start_ft, outer_ft)])


def _lies_wholly_behind(leg, geometry, station_ft: float) -> bool:
    """Whether EVERY vertex of a piece falls short of `station_ft` - so it is an offcut.

    A MEAN STATION IS NOT A SIDE. A piece cut off a zone by a skewed crossing is a long diagonal
    sliver: at W Broad & Louellen the crossing is surveyed 43.7 deg off square, so an offcut
    running from station 26 to 47 has its mean at 34.4, PAST the crossing's own 32.0, and 164 sq
    ft of hatching stays in the intersection. Every vertex, or it is not behind.

    The same shape of test as checks.NoPaintInsideTheJunction's, deliberately: "wholly behind the
    crossing" and "wholly inside the mouth" are the same question asked of the two things that
    end a kerbside zone.
    """
    coords = (geometry.exterior.coords if geometry.geom_type == "Polygon" else geometry.coords)
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(coords, dtype=float))
    return float(stations.max()) <= station_ft


def parking_runs(state: "DesignState", leg_name: str, side: str, crosswalk_offsets: dict,
                  props: list[dict] | None = None) -> list[tuple[float, float]]:
    """The station spans of this kerb where stalls may legally be marked."""
    return parkable_runs_ft(
        state, leg_name, side, crosswalk_offsets, props,
        physical_clearance_ft=leg_clearance_ft(leg_name, state.legs, state.corner_fillets),
        min_run_ft=MIN_PARKING_RUN_FT)
