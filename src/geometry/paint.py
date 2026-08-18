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
import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.geometry.model import (point_at, clip_paint_clear_of, corner_apron_annulus,
                                curbside_strip_polygon,
                                corner_overlay_polygon, curb_offsets_at_stations,
                                leg_clearance_ft, station_offset_many, through_street_sides)
from src.geometry.markings import PaintKind
from src.geometry.daylighting import parkable_runs_ft
from src.render.coords import FT_TO_M
from src.render.crosswalks import (CROSSWALK_CLEARANCE_FT, CROSSWALK_DEPTH_FT,
                                   crosswalk_reach_on_leg_side_ft)


class RimCause(StrEnum):
    """What cut a zone short, for the line that closes it there.

    CROSSING a painted crossing, which the hatching runs into and is cut by - the clean diagonal
             end you see on a real street.
    OPENING  a gap in the kerb. At a DRIVEWAY that is the apron's fillet, which the hatching
             stops short of, because that arc's chord is at the hatch angle and a stroke laid
             beside it reads as a fork. At an INTERSECTING APPROACH there is no fillet - a street
             mouth has no apron (see kerb_opening_bands) - so the rim is the square end instead.
    """
    CROSSING = "crossing"
    OPENING = "opening"


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
    # What cut this piece, if it is the line along a zone's CUT END rather than along its length.
    # A rim carries the same `kind` as the zone's edge line, deliberately (see PaintContext.rim), so
    # this is the only thing that distinguishes the two - and the cause matters as well as the fact,
    # because the two ends do not want the same treatment: the hatching keeps half a spacing off an
    # OPENING's fillet, whose chord runs at the hatch angle and so reads as a stroke, and runs
    # straight into a CROSSING's diagonal, which is what gives a zone its clean end there.
    rim: "RimCause | None" = None

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

# How far an opening's ends are trimmed back, with a rounded corner, past the dropped kerb's own
# extent. A driveway apron flares at the kerb in reality and a car turning in cuts the corner, so
# a square-ended gap the exact width of the kerb tag both looks punched-out and gives a turning
# vehicle nothing. Kept to a foot and a half on purpose: pedestrians and cyclists have priority
# here, and every foot of trim is a foot of bike lane or hatched buffer given up. Not a swept-path
# figure - see kerb_opening_bands.
OPENING_TRIM_FT = 1.5

# A HATCHED zone ends at an opening on an arc that LEAVES ITS OWN EDGE LINE TANGENTIALLY and
# curves out to the kerb - a fillet, not a chamfer and not a bulge. That tangency is the whole
# difference between a line and a cut: on a real street the white line beside the travel lane
# runs straight, peels away in one sweep around the driveway apron, and comes back - one
# continuous stroke, no corner anywhere in it.
#
# The first version had the arc the wrong way round. It was tangent to the TRANSVERSE direction at
# the lane edge instead, i.e. flat where the eye follows the edge line and curved only in the last
# foot or two at the kerb, which is precisely the blunt end it was meant to fix - 2.5 ft of sweep
# across 14 ft of depth at Broad St, invisible at any drawing scale.
#
# The radius is the depth of the strip being closed, so the arc uses the whole cross-section and
# arrives at the kerb exactly as the mouth's own trim: one arc, no straight portion, nothing to
# tune per site. It is bounded by that depth rather than by a constant because the run and the
# depth are the same measurement seen twice - a shallow strip needs a short sweep to look right and
# a deep one needs a long one. It costs HATCHING and nothing else: the lane's lines carry a dotted
# extension across (PaintContext.dashes_through_openings) and its green stops at the trim, so what
# is given up is the paint that says "nothing belongs here", next to a mouth a car swings through.
OPENING_FILLET_PER_DEPTH = 1.0

# The dotted extension a lane line becomes where it crosses an opening. MUTCD's dotted lane
# extension is a 2 ft segment with a 2-6 ft gap; the tight end of that range is used because a
# driveway mouth is short - E Broad's openings run 4-37 ft, and a 2+6 pattern would put a single
# dash in a 10 ft one, which reads as a stray mark rather than as a line continuing.
DOTTED_MARK_FT = 2.0
DOTTED_GAP_FT = 2.0

# The painting order reserved for built ground - an apron. Everything else is cut around it, so
# it has to be laid before anything else is painted; see PaintContext.seal_surfaces.
SURFACE_PAINT_GROUP = 0

# How close a piece of a fill's boundary has to lie to the crossing to BE the cut edge. The
# clip puts it exactly on the buffered band, so this only absorbs float noise.
RIM_SNAP_FT = 0.05
# Below this a rim is a clipping artifact at a corner, not a painted line.
MIN_RIM_LENGTH_FT = 1.0

# Below this a zone is a HAIRLINE LEFT BY A CLIP, not a marking. Differencing polygons that share
# an edge leaves slivers along it, and one had been surviving all along: a lane-narrowing fill of
# **0.0 sq ft with a 12.0 ft perimeter** on broad_st_east's left kerb, drawn in the plan view as an
# outline with nothing inside it. Harmless until the openings were rimmed too, at which point that
# sliver's own boundary produced two rim segments 1.7 ft of which lay on top of each other, which
# MarkingsDoNotCollide reported. The check found a real defect one layer below the change - a zone
# with no area is not paint, whatever it is drawn around. A hatched zone here is tens to hundreds
# of square feet, so this cannot reach a real one.
MIN_ZONE_AREA_SQ_FT = 1.0

# And the same thing for a LINE. A clip that lands on a vertex leaves a LineString of zero or
# near-zero length: three of them survived at E Broad & Princeton, including one at exactly the
# station of a cross street's mouth, each drawn as a stray tick of lane edge line with nothing
# attached to it. MIN_RIM_LENGTH_FT already discards a rim this short - a rim is a sweep and a
# 1 ft sweep is nothing - but an ordinary line went through unfiltered, so the guard existed for
# fills and for rims and not for the case in between. Well under a stall divider (the shortest
# real line here, a few feet), so it cannot reach a marking anyone meant to draw.
MIN_LINE_LENGTH_FT = 0.25

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
    return LineString([point_at(leg.centerline, start_ft, inner_ft),
                       point_at(leg.centerline, start_ft, outer_ft)])


def _lies_wholly_behind(leg, geometry, station_ft: float) -> bool:
    """Whether EVERY vertex of a piece falls short of `station_ft` - so it is an offcut.

    A MEAN STATION IS NOT A SIDE, and this used to take one. A piece cut off a zone by a skewed
    crossing is a long diagonal sliver: at W Broad & Louellen the crossing is surveyed 43.7 deg
    off square, so the offcut behind it ran from station 26 to 47 and its mean landed at 34.4,
    PAST the crossing's own 32.0. The test passed and 164 sq ft of hatching stayed in the
    intersection. Every vertex, or it is not behind.

    The same shape of test as checks.NoPaintInsideTheJunction's, deliberately: "wholly behind the
    crossing" and "wholly inside the mouth" are the same question asked of the two things that
    end a kerbside zone, and asking them the same way is what stops one of them being sound and
    the other a heuristic.
    """
    coords = (geometry.exterior.coords if geometry.geom_type == "Polygon" else geometry.coords)
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(coords, dtype=float))
    return float(stations.max()) <= station_ft


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
    keep_clear: object = None          # EVERY painted crossing in the frame, buffered by the gap
    # This junction's own crossings, same buffer. The two used to be one field doing two jobs,
    # which curbside_paint_ft's docstring already described as two jobs - and they stopped being
    # the same set the moment crossings at UNMODELLED junctions had to be kept clear of too. See
    # anchors() for why aiming at those would be a disaster and cutting against them is required.
    junction_crossings: object = None
    marked: set = field(default_factory=set)
    straight_through: set = field(default_factory=set)
    props: list | None = None
    openings: object = None            # dropped kerbs a vehicle crosses: paint breaks over them
    surfaces: object = None            # the mountable aprons: paint stops at them
    surface_polygons: list = field(default_factory=list)
    pieces: list = field(default_factory=list)
    # Zones already placed on a kerb that runs straight through - see add(shares_a_kerb=True).
    through_painted: list = field(default_factory=list)
    # (leg, side) -> the footprint this kerb's dotted extensions take their phase from. See
    # dash_phase: one shape per kerb, so everything crossing an opening breaks at the same
    # stations rather than each marking dashing along its own length.
    dash_phases: dict = field(default_factory=dict)

    def add_surface(self, kind, polygon) -> None:
        """Ground that is BUILT rather than painted, which every marking then stops at.

        An apron is flush pavers or textured concrete - part of the corner rather than part of
        the carriageway - so the ground it occupies is not roadway to be hatched. Left
        unsubtracted, a curb extension's daylight zone and its own swept-path apron overlapped by
        2-4 sq ft at three of the four corners at Broad & Greenwood, which MarkingsDoNotCollide
        reported as the same ground painted twice. It is the same layering the crossings already
        get, one rung further up: a surface outranks a marking the way a marking outranks a
        buffer.

        Collected rather than unioned here, because the union has to be complete before any
        marking is cut against it - see seal_surfaces.
        """
        if polygon is None or polygon.is_empty:
            return
        self.surface_polygons.append(polygon)
        self.pieces.append(PaintPiece(kind, polygon))

    def seal_surfaces(self) -> None:
        """Close the surface pass: from here on, every marking is cut around all of them.

        No gap buffer, unlike a crossing: paint runs up to an apron's edge and stops there. The
        striper's gap around a crossing (PAINT_TO_CROSSWALK_GAP_FT) exists because both are paint.
        """
        self.surfaces = unary_union(self.surface_polygons) if self.surface_polygons else None

    def emit(self, piece: PaintPiece) -> PaintPiece:
        """Keep a piece as-is, without clipping. For the things that are not paint: an apron is
        a surface the paint stops at, and a bollard is a point standing in the road.

        WITH TWO CLIPS, because neither a kerb opening nor a crossing is paint. A flex post
        cannot be trimmed the way a stripe can - it is either standing in the way or it is not -
        so a post that lands in one is dropped rather than shortened.

        AN OPENING: without this the paint broke over a driveway and the bollards marched
        straight across it, which is worse than not breaking the paint at all - it reads as a
        protected lane whose protection you are expected to drive through.

        A CROSSING: a post planted in a marked crosswalk is the same statement about a person
        walking, and it is the worse of the two. This never arose at THIS junction's crossings,
        because a bike lane's post row starts past where the crossing reaches
        (props.bollard_props_from_paint takes its stations from the painted row for exactly that
        reason) - so the rule was satisfied by an accident of ordering rather than stated. At a
        crossing belonging to another junction in the frame there is no such ordering, and two
        posts stood in Blackwell Avenue's zebras on the corridor proposal.

        `keep_clear` and not `junction_crossings`: every crosswalk in the picture, for the same
        reason `add` cuts against all of them.
        """
        if piece.kind.is_object and (stands_in_an_opening(self.openings, piece.geometry)
                                      or _stands_in_a_crossing(self.keep_clear, piece.geometry)):
            return piece
        self.pieces.append(piece)
        return piece

    def add(self, kind, geometry, leg=None, side=None, beyond_ft=None, shares_a_kerb=False):
        """Clip `geometry` clear of the crossings, keep what survives, return those pieces.

        beyond_ft drops any surviving piece that fell WHOLLY on the JUNCTION side of the
        crossing. A zone drawn deliberately through a crossing (so the crossing cuts its end into
        a clean diagonal) leaves an offcut back at the corner, and that offcut is not paint.

        NARROWER THAN IT WAS, because the junction now cuts its own corners: an offcut inside the
        corner return is removed by the junction's mouth (kerbs.OpeningSource.JUNCTION), so what
        is left for this to catch is only the strip between the mouth and the crossing, on the
        legs where the crossing sits outside the corner. Measured across all four sites and every
        scenario that is one 1.7 sq ft sliver, on broad_st_east - which is worth knowing, because
        a filter doing almost nothing is a filter whose failure would be invisible. It is kept and
        made sound (see _lies_wholly_behind) rather than kept and trusted.

        shares_a_kerb dedupes against the other zones on the same through-running kerb.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        if shares_a_kerb and self.through_painted:
            geometry = geometry.difference(unary_union(self.through_painted))
            if geometry.is_empty:
                return added
        # Cut clear of the mountable surfaces, then of the crossings, then of the kerb
        # openings - each may fragment a piece, so every stage runs over whatever the last left.
        # WHICH openings, and how much of each, is markings.AT_AN_OPENING's answer and no longer
        # this function's: see KerbOpenings.against.
        opening = self.openings.against(kind, leg, side) if self.openings else None
        surviving = [cut for whole in clip_paint_clear_of(geometry, self.surfaces)
                     for part in clip_paint_clear_of(whole, self.keep_clear)
                     for cut in clip_paint_clear_of(part, opening)]
        for part in surviving:
            if beyond_ft is not None and _lies_wholly_behind(self.state.legs[leg], part,
                                                              beyond_ft):
                continue
            if kind.covers_area and part.area < MIN_ZONE_AREA_SQ_FT:
                continue
            if kind.is_line and not kind.is_object and part.length < MIN_LINE_LENGTH_FT:
                continue
            piece = PaintPiece(kind, part, leg, side)
            self.pieces.append(piece)
            added.append(piece)
        if shares_a_kerb:
            self.through_painted.extend(p.geometry for p in added)
        # ...AND THE DASHES BACK ACROSS, for a marking whose row says DOTTED. The complement of
        # the cut just made, so a treatment cannot paint the solid part and forget the broken one
        # - which is what "whether a marking carried a dotted extension was decided by whether its
        # treatment remembered to call emit_across_opening" meant in practice, and only the
        # bikeways ever remembered.
        added.extend(self._dashes_across_openings(kind, geometry, leg, side))
        return added

    def dash_phase(self, leg: str, side: str, geometry) -> None:
        """Register the footprint a kerb's dotted extensions take their phase from.

        ASKED ONCE PER KERB, OFF ONE SHAPE, and handed to everything that crosses - so a lane's
        two edge lines and the green between them break at the same stations instead of each
        being dashed along its own length and drifting out of phase. Which shape is canonical
        matters and it is the SURFACE: the lines are its edges.

        A treatment that paints a DOTTED marking has to call this first. It is an error not to
        rather than a silently un-dashed lane, because the whole failure this table replaces is
        an omission that looks like nothing at all.
        """
        if geometry is not None and not geometry.is_empty:
            self.dash_phases[(leg, side)] = geometry

    def _dashes_across_openings(self, kind, geometry, leg, side) -> list:
        """The marks of `kind`'s dotted extension, laid IN the openings its row says it crosses.

        THE PARENT MARKING, CLIPPED - not rebuilt over the dash stations. The dashes have to lie
        exactly on the line they continue, and a second construction of the same locus is a
        second chance to disagree with the first; this project has paid for that distinction
        several times over (see the module docstring). It also means a marking gets its extension
        from its row alone, with nothing for a treatment to supply but the phase.

        Confined to the opening's own ground, which is what "laid IN an opening" means. The dash
        spans are STATION bands, and a station band is a band right across the marking: where an
        entrance meets the street at a skew the span reaches further along the kerb than the
        entrance's own polygon does, by more the wider the marking is. The clip against `dotted`
        is what keeps the two exact complements - 10 sq ft of doubled green on e_broad_st_west's
        12 ft two-way lane, which markings_collide reported, was that difference.
        """
        from src.geometry.markings import opening_rule

        rule = opening_rule(kind)
        dotted = self.openings.dotted(kind, leg, side) if self.openings else None
        if rule.dotted_as is None or dotted is None or dotted.is_empty:
            return []
        inside = geometry.intersection(dotted)
        if inside.is_empty:
            return []
        phase = self.dash_phases.get((leg, side))
        if phase is None:
            raise KeyError(
                f"{kind} goes dotted across an opening on {leg} {side}, but nothing registered "
                f"the phase its marks are laid on. The treatment that paints it must call "
                f"PaintContext.dash_phase(leg, side, surface) with the lane's own footprint "
                f"first - every marking on one kerb has to break at the SAME stations or they "
                f"drift out of phase with each other.")
        added = []
        for start_ft, end_ft in self._dash_spans_along(phase, dotted, leg):
            band = _station_band(self.state.legs[leg], start_ft, end_ft)
            mark = inside.intersection(band) if band is not None else None
            if mark is None or mark.is_empty:
                continue
            for clear in clip_paint_clear_of(mark, self.surfaces):
                for part in clip_paint_clear_of(clear, self.keep_clear):
                    if rule.dotted_as.covers_area and part.area < MIN_ZONE_AREA_SQ_FT:
                        continue
                    # A PART OF A MARK IS NOT A MARK. The span is a station band right across the
                    # road and the marking inside it may be shorter than the band; a 0.12 ft stub
                    # of a 2 ft dotted mark reads as a striping error, which is what it would be.
                    if rule.dotted_as.is_line and part.length < DOTTED_MARK_FT / 2:
                        continue
                    piece = PaintPiece(rule.dotted_as, part, leg, side)
                    self.pieces.append(piece)
                    added.append(piece)
        return added

    def _dash_spans_along(self, phase, dotted, leg_name: str) -> list[tuple[float, float]]:
        """The station spans the marks fall in, one run of them per opening the phase CROSSES.

        STATIONS, not distance along one line, because everything carried across an opening has
        to break at the SAME places: dashing each marking along its own arc length puts them out
        of phase - by little on a straight leg and visibly on a curved one, where the inner and
        outer stripes have different lengths through the same mouth.

        AND ONLY WHERE THE MARKING GOES ON AFTERWARDS. A dotted extension carries a lane ACROSS
        an entrance - "the lane still runs here, and here is where it is crossed" - so a lane that
        simply ENDS at an opening has nothing to extend. The two cases look identical to a clip
        and are opposite in meaning, and the difference only became visible once the junction's
        own mouth was an opening: at Broad & Greenwood the bike lane already stops at its crossing,
        19 ft short of the mouth's far end, so the intersection with the mouth was a 1.7 ft sliver
        of the lane's own tail and the "extension" across a 25 ft junction came out as one 2 ft
        mark and three stubs of 0.1-1.1 ft.
        """
        centerline = self.state.legs[leg_name].centerline

        def stations_of(geometry):
            coords = (geometry.exterior.coords if geometry.geom_type == "Polygon"
                       else geometry.coords)
            values, _offsets = station_offset_many(centerline,
                                                    np.asarray(coords, dtype=float))
            return float(values.min()), float(values.max())

        phase_lo, phase_hi = stations_of(phase) if not phase.is_empty else (0.0, 0.0)
        spans = []
        inside = phase.intersection(dotted)
        for part in getattr(inside, "geoms", [inside]):
            if part.is_empty:
                continue
            lo, hi = stations_of(part)
            if not (phase_lo < lo - DASH_CROSSING_SLACK_FT
                    and phase_hi > hi + DASH_CROSSING_SLACK_FT):
                continue        # the marking ends here rather than crossing - nothing to extend
            spans.extend(_dash_spans(lo, hi))
        return spans

    def rim(self, fills, kind) -> None:
        """The line along a fill's cut end - at a crossing, AND around a kerb opening.

        A hatched zone is outlined, and that outline carries on around the end where something
        cuts it: the diagonal that finishes a zone off against a crossing, and the fillet that
        sweeps it around a driveway mouth. The lane edge already gets a line of its own; without
        this one the zone just stops, with hatch strokes ending in mid-air. It was doing exactly
        that at every opening, because only the crossings were rimmed.

        `kind` is the zone's OWN edge line, passed by the caller, so the rim is the same paint
        continued rather than a line of its own colour - which is the point of it. On a real
        street the white line beside the lane peels away around the apron and comes back as one
        continuous stroke; drawing that sweep in a different colour from the line it continues is
        what makes a drawing look assembled out of pieces. This replaced a dedicated
        `crossing_rim_line` kind, drawn orangered whatever it closed.

        A rim is only the part of the cut that is NOT ALREADY PAINTED. Where the fillet meets the
        zone's inner edge the two run together, and emitting the whole intersection laid 1.8 ft of
        a second lane edge line on top of the first at Broad St and E Broad - which
        MarkingsDoNotCollide reported, correctly: it is a joint, not a stroke, and the line through
        it is already there.
        """
        # The tolerance the collision check uses, imported rather than restated: "already painted"
        # has to mean here what it means there. Locally imported, like everything else in this
        # module that would otherwise be a cycle.
        from src.checks import COLLINEAR_PAINT_TOLERANCE_FT

        for piece in fills:
            # Seeded with EVERY line already on this kerb, not only this kind's. The zone's own
            # edge line is the one this rim continues, and now that the rim is drawn on that
            # line's locus rather than half a stripe off it, the two are collinear where the
            # fillet leaves the lane edge - a joint, not a second stroke. But a kerbside zone's
            # inner edge is also some other marking's outer edge: beside a bike lane it is the
            # lane's own outer stripe, so the rim ran 1.8 ft along the dotted extension crossing
            # the mouth. Whatever kind painted it, it is painted.
            painted = [p.geometry for p in self.pieces
                       if p.kind.is_line and p.leg == piece.leg and p.side == piece.side]
            for cutter, cause in (
                    (self.keep_clear, RimCause.CROSSING),
                    (self.openings.against(piece.kind, piece.leg, piece.side)
                     if self.openings else None, RimCause.OPENING)):
                if cutter is None or cutter.is_empty:
                    continue
                # HALF A STRIPE OUTSIDE THE FILL, because that is where the line it continues
                # runs. lane_edge_stripes puts the edge line's centre half its own width outside
                # the hatching, so the stripe's body fills the space between them - and a rim
                # traced on the fill's boundary instead sits 0.41 ft to the side of the line it is
                # supposed to be part of. Near the fillet's tangent point, where the arc runs
                # almost along the road, those 0.41 ft of offset stretch into a 1.78 ft break in
                # the line: the seam where the sweep begins.
                # ROUND joins, not mitre. A mitre corner extends to half a stripe / cos(t/2),
                # so where the zone's inner edge turns to sweep around an opening the join
                # spikes past the offset it is supposed to hold - 0.16 ft into the travel lane
                # at a right-angled corner, which is what it produced on e_broad_st_west and
                # w_broad_st_northeast once their alignments were centred on the carriageway
                # and the corner came out square. A rim is a line held half a stripe off the
                # fill; a spike is not part of that line.
                grown = piece.geometry.buffer(LANE_EDGE_LINE_WIDTH_FT / 2, join_style=1)
                edge = grown.exterior.intersection(
                    cutter.buffer(RIM_SNAP_FT + LANE_EDGE_LINE_WIDTH_FT / 2))
                # The buffer grows the fill in EVERY direction, the kerb included, and
                # PaintInsideTheCurb duly reported the rim 0.4 ft over the traced kerb on all four
                # of Columbia & Princeton's kerbs. Held back inside the kerb here: a marking may
                # meet it, never cross it, and half a stripe is exactly the amount by which an
                # unclipped grown boundary would.
                if piece.leg and piece.side:
                    inside = curbside_strip_polygon(self.state.legs[piece.leg], piece.side,
                                                     0.0, 0.0)
                    if inside is not None and not inside.is_empty:
                        edge = edge.intersection(inside)
                for part in getattr(edge, "geoms", [edge]):
                    if part.geom_type != "LineString":
                        continue
                    # A zone can be cut by a crossing AND by a driveway at the same corner, and
                    # where the two cuts converge their rims run together - 1.8 ft of doubled lane
                    # edge line at Broad St and E Broad, which MarkingsDoNotCollide reported. The
                    # sweep is one stroke however many things cut it.
                    if painted:
                        part = part.difference(
                            unary_union(painted).buffer(COLLINEAR_PAINT_TOLERANCE_FT))
                    for got in getattr(part, "geoms", [part]):
                        if got.geom_type == "LineString" and got.length >= MIN_RIM_LENGTH_FT:
                            self.pieces.append(PaintPiece(kind, got, piece.leg, piece.side,
                                                          rim=cause))
                            painted.append(got)

    def anchors(self, leg_name: str, side: str, inner_offset_ft: float = 0.0):
        """This leg-side's measuring stations, with the shared crossing geometry filled in.

        THIS JUNCTION'S CROSSINGS ONLY, and this is the one place the distinction bites.
        leg_anchors asks crosswalk_reach_on_leg_side_ft how far along the kerb a crossing
        reaches, takes the FURTHEST, and starts the kerbside treatment beyond it - which is right
        for a crossing at the corner the paint is backing away from, and catastrophic for one
        220 ft down the block. Handed the full set, broad_st_east's taper would aim at station
        322 (Blackwell's far crossing) instead of 26, and the leg's entire kerbside treatment
        would vanish behind the crossing it was supposed to start after.

        Everything else - `add`, its dotted extensions and `rim` - reads `keep_clear`, which is the
        full set, because "do not paint over a crosswalk" is true of every crosswalk in the frame.
        """
        return leg_anchors(self.state, leg_name, side, self.crosswalk_offsets,
                            self.junction_crossings, inner_offset_ft=inner_offset_ft,
                            crosswalk_is_marked=leg_name in self.marked)


def curbside_paint_ft(state, crosswalk_offsets: dict, center_ft,
                       crosswalk_bands: dict | None = None,
                       props: list[dict] | None = None,
                       marked_crosswalks: set | None = None,
                       crossings_elsewhere=None) -> list[PaintPiece]:
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

    `crossings_elsewhere` is the SECOND of those jobs done for the crossings this junction does
    not own: the marked ones at the other junctions inside the frame (src/geometry/surveyed.py),
    which have no leg here and therefore no entry in `crosswalk_bands`. They are subtracted and
    never aimed at - see PaintContext.anchors for why the second half of that is not an oversight.

    THEY WERE GETTING NEITHER. Broad St's frame contains Blackwell Avenue with three crossings
    traced across it, two of them a zebra; src/geometry/surveyed.py drew them and nothing got out
    of their way, so the corridor proposal painted 120 sq ft of green bike lane, 36 ft of lane
    edge line and its yellow contraflow stripe straight over a marked crosswalk. Measured across
    the four sites and every scenario, up to 164 sq ft of area paint and 48 ft of line. The
    drawing was making a claim about the street that the same drawing contradicted 6 ft away.
    """
    # Only crossings that are actually PAINTED get out of the way of anything. Every leg
    # gets a resolved offset, including ones with no marking today - cutting paint around
    # those was reserving room for a crossing that isn't there.
    marked = set(marked_crosswalks) if marked_crosswalks is not None else set(state.legs)
    # Kerbs with no corner return at the junction: the kerb runs straight through, so the
    # crossing cuts the zone in two and BOTH halves are paint. Everywhere else the piece on
    # the junction side of a crossing sits in the corner throat and is discarded.
    #
    # THE SAME KERBS model.junction_mouth_ft RETURNS None FOR, and not by coincidence - both
    # answers come out of through_street_sides, which is what makes "this junction opens this
    # kerb" and "this kerb has a corner in it" one fact rather than two. A zone on a through kerb
    # is therefore never cut at the junction by the OPENING either; it simply runs on into the
    # adjoining leg's zone, which is what this set has always been arranging by hand.
    straight_through = through_street_sides(state.legs)
    bands = {name: band for name, band in (crosswalk_bands or {}).items()
             if band is not None and not band.is_empty and name in marked}
    junction_crossings = (unary_union(list(bands.values())).buffer(PAINT_TO_CROSSWALK_GAP_FT)
                           if bands else None)
    elsewhere = [band for band in (crossings_elsewhere or ())
                 if band is not None and not band.is_empty]
    all_bands = list(bands.values()) + elsewhere
    keep_clear = (unary_union(all_bands).buffer(PAINT_TO_CROSSWALK_GAP_FT) if all_bands else None)
    openings = kerb_opening_bands(state)
    # --- and now the treatments paint themselves. Each one that has markings owns them
    # (Treatment.paint), so a marking's geometry lives beside the validation and the provenance
    # of the thing that calls for it, rather than in a block of this function keyed off one of
    # DesignState's dicts. That separation is what let the bike lane's kerb hatching be added
    # without the rim() every other hatched zone got.
    #
    # Dispatched in painting order, and deduplicated by (type, target) keeping the LAST applied:
    # a design's dicts are last-write-wins, so two MarkedParking treatments on one kerb are one
    # marked lane and not two painted on top of each other.
    ctx = PaintContext(state=state, crosswalk_offsets=crosswalk_offsets, center_ft=center_ft,
                        keep_clear=keep_clear, junction_crossings=junction_crossings,
                        marked=marked, openings=openings,
                        straight_through=straight_through, props=props)
    current = {}
    for treatment in getattr(state, "treatments", []):
        current[(type(treatment), str(treatment.target))] = treatment
    def order_of(treatment):
        # In the surface pass, by the CORNER the ground lands at: a curb extension is aimed at a
        # leg-side and lays its apron at the corner that kerb feeds, so ordering those by target
        # would interleave two corners' aprons. Everywhere else, by the target itself.
        if treatment.paint_group == SURFACE_PAINT_GROUP:
            corner = treatment.apron_corner(state)
            return (treatment.paint_group, str(corner), treatment.paint_rank)
        return (treatment.paint_group, str(treatment.target), treatment.paint_rank)

    ordered = sorted(current.values(), key=order_of)
    # The SURFACE pass runs first and is then sealed, because paint stops at built ground and
    # every marking after this is cut around all of it - a surface added later would be a surface
    # the earlier markings were never cut against. See PaintContext.add_surface.
    surface_pass = [t for t in ordered if t.paint_group == SURFACE_PAINT_GROUP]
    for treatment in surface_pass:
        treatment.paint(ctx)
    ctx.seal_surfaces()
    for treatment in ordered[len(surface_pass):]:
        treatment.paint(ctx)
    return ctx.pieces


# How far past the nominal half-width a dash's station band reaches before the opening's own
# polygon bounds it laterally. Generous on purpose - the band is a STATION filter, and one that
# stopped at the nominal kerb would clip the outer end off every mark on a leg whose traced kerb
# flares, which approaching a corner is every one of them.
DASH_BAND_REACH = 3.0
DASH_BAND_MARGIN_FT = 20.0


def _station_band(leg, start_ft: float, end_ft: float):
    """The band right across a leg between two stations - one mark's worth of ground.

    Deliberately NOT offset_band_polygon, which clamps its offsets to the traced kerb: this is a
    station filter and not a lateral one. It exists to cut a marking down to the run of stations
    one dash covers, and how far across the road that dash reaches is the marking's own business.
    The clip against the opening's polygon is what bounds it laterally.

    Sampled along the centreline rather than taken as one rectangle between the two ends, so a
    dash laid on a leg that bends (louellen_st_west turns 29.4 deg in its first 15 ft) sits on
    the road rather than cutting the corner of it.
    """
    length_ft = leg.centerline.length
    lo, hi = max(min(start_ft, length_ft), 0.0), max(min(end_ft, length_ft), 0.0)
    if hi - lo < 1e-6:
        return None
    reach_ft = abs(leg.curb_to_curb_ft or 0.0) * DASH_BAND_REACH + DASH_BAND_MARGIN_FT
    stations = np.linspace(lo, hi, max(int((hi - lo) / DASH_BAND_STEP_FT) + 2, 2))
    left = [point_at(leg.centerline, float(s), reach_ft) for s in stations]
    right = [point_at(leg.centerline, float(s), -reach_ft) for s in stations]
    band = Polygon([*left, *reversed(right)])
    if not band.is_valid:
        band = band.buffer(0)
    return None if band.is_empty else band


# How finely the station band is sampled along the centreline. Well under a dash's own length, so
# a bend inside one mark is followed rather than chorded.
DASH_BAND_STEP_FT = 1.0

# How far past an opening a marking has to go on before it counts as CROSSING the opening rather
# than ending at it - see PaintContext._dash_spans_along. A dotted extension is only for the
# first. Half a mark: below that there is nothing on the far side to continue into.
DASH_CROSSING_SLACK_FT = 1.0


def _dash_spans(lo_ft: float, hi_ft: float) -> list[tuple[float, float]]:
    """`lo_ft`..`hi_ft` cut into MUTCD dotted-extension marks, centred in it.

    STATIONS, not distance along one line, because everything the dashes carry across an opening
    has to break at the SAME places: the lane's two edge lines and the green between them are one
    marking seen three ways, and dashing each along its own arc length puts them out of phase - by
    little on a straight leg and visibly on a curved one, where the inner and outer stripes have
    different lengths through the same mouth.

    Centred rather than started at one end so the pattern reads as deliberate: an opening is only a
    few marks long, and one clipped to a stub at the far end looks like a striping error. The count
    comes out of the length, so a wide entrance gets more marks rather than longer ones.
    """
    length_ft = hi_ft - lo_ft
    if length_ft < DOTTED_MARK_FT:
        return []
    period = DOTTED_MARK_FT + DOTTED_GAP_FT
    n = max(1, int(round((length_ft + DOTTED_GAP_FT) / period)))
    span = n * DOTTED_MARK_FT + (n - 1) * DOTTED_GAP_FT
    while n > 1 and span > length_ft:
        n -= 1
        span = n * DOTTED_MARK_FT + (n - 1) * DOTTED_GAP_FT
    start_ft = lo_ft + max((length_ft - span) / 2, 0.0)
    return [(start_ft + i * period, start_ft + i * period + DOTTED_MARK_FT) for i in range(n)]


def _union(shapes) -> object:
    """The union of whatever is not None, or None. Used to compose an opening's shapes per rule."""
    parts = [s for s in shapes if s is not None and not s.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


@dataclass(frozen=True)
class KerbSideOpenings:
    """Where ONE KERB opens for a vehicle, kept apart BY WHAT KIND OF OPENING IT IS.

    Three shapes, and the split is MUTCD 1C.02's, not this project's: an intersecting approach and
    a driveway are different things and the markings do different things at them. What each
    marking does is declared once in markings.AT_AN_OPENING; this class holds the ground, and
    `against` composes the two.

      * `driveway_mouths` - the entrances that are NOT intersections, trimmed back and rounded
        because a driveway apron flares at the kerb and a vehicle turning in cuts the corner.
      * `driveway_tapered` - the same, plus the rounded run-out at the travel lane's edge that a
        HATCHED zone ends on. A no-travel zone that stops square reads as a rectangle punched
        through the hatching; the same zone at a crossing ends on the crossing's own diagonal,
        which is what makes it look painted rather than deleted.
      * `intersection_mouths` - the approaches that ARE intersections: a side street, and this
        junction's own mouth. No trim and no run-out, and both omissions are the same point -
        they model a driveway APRON, which is a thing a street mouth does not have. Its flare is
        its corner return, and that is already in the geometry. So there is no separate
        `intersection_tapered`: at an intersection, FILLETED and STOPPED are the same shape, which
        is why markings.AtAnOpening carries one value for the pair.

    This was three fields answering "which of these do I subtract" with an if-chain reading two
    frozensets. The fields are now the ground and the table is the rule, so adding a marking
    cannot silently inherit whichever branch it happens to fall through - it has no row and
    markings.require_every_kind refuses to import.
    """
    driveway_mouths: object = None
    driveway_tapered: object = None
    intersection_mouths: object = None

    @property
    def driven(self) -> object:
        """Every entrance, of both kinds, at its real width. What an OBJECT is kept out of and
        what a dotted extension is laid inside - neither question cares which kind it is."""
        return _union((self.driveway_mouths, self.intersection_mouths))

    @property
    def tapered(self) -> object:
        """`driven` plus every driveway run-out - the widest of the three, and the ground a
        hatched zone gives up. Equal in area to `driven` on a kerb whose only openings are
        intersections, which is the shape of "a street mouth has no apron"."""
        return _union((self.driveway_tapered, self.driveway_mouths, self.intersection_mouths))

    def against(self, kind) -> object:
        """The ground `kind` is cut out of, composed from its row in markings.AT_AN_OPENING.

        One rule per column: CARRIED subtracts nothing, FILLETED subtracts the run-out (and at an
        intersection there is none, so it subtracts the mouth), DOTTED and STOPPED subtract the
        mouth. What differs between the two columns is which SHAPES they apply to, which is the
        whole reason the shapes are kept apart above.

        3B.11(07)'s exception - solid edge lines MAY continue "through that part of an
        intersection with no intersecting approach (such as at the far side of a T-intersection)"
        - needs no code here, and that is worth stating because it looks like it should. An
        opening is only ever made on the kerb the approach actually leaves on: cross_streets.py
        reads that off the street's own vertices, and model.junction_mouth_ft returns None where
        the kerb runs straight through. A T's far kerb never enters `intersection_mouths` in the
        first place and its line is never cut. A crossroads opens both.
        """
        from src.geometry.markings import AtAnOpening, opening_rule

        rule = opening_rule(kind)
        driveway = {AtAnOpening.CARRIED: None,
                    AtAnOpening.FILLETED: self.driveway_tapered}.get(rule.at_a_driveway,
                                                                      self.driveway_mouths)
        intersection = (None if rule.at_an_intersection is AtAnOpening.CARRIED
                         else self.intersection_mouths)
        return _union((driveway, intersection))

    def dotted(self, kind) -> object:
        """The ground `kind` lays a dotted extension across, or None. The complement of `against`
        restricted to the columns whose rule is DOTTED, so the two never overlap: `add` keeps
        what is outside and this is where the dashes go back."""
        from src.geometry.markings import AtAnOpening, opening_rule

        rule = opening_rule(kind)
        return _union((
            self.driveway_mouths if rule.at_a_driveway is AtAnOpening.DOTTED else None,
            self.intersection_mouths if rule.at_an_intersection is AtAnOpening.DOTTED else None))

    def __bool__(self) -> bool:
        driven = self.driven
        return driven is not None and not driven.is_empty


@dataclass(frozen=True)
class KerbOpenings:
    """Every kerb's openings, KEPT PER KERB, plus the union for the things that stand in one.

    AN OPENING CUTS THE KERB IT OPENS, and it used to cut everything. That was harmless while a
    driveway was the only kind: a driveway on E Broad's north kerb is nowhere near Greenwood's, so
    a single union gave the same answer as a per-kerb one. It stopped being harmless the moment
    this junction's own mouth became an opening, because every leg's mouth reaches into the SAME
    throat - so at W Broad & Louellen, whose two streets meet at 43.6 deg, Louellen's south mouth
    swallowed part of W Broad's two-way bike lane and broke the corridor at the junction it was
    built to run through. That is not a rule about the bike lane; it is an opening being asked
    about a kerb that is not the one it opens.

    So `against` and `dotted` take the marking's own (leg, side) - which PaintPiece has carried all
    along - and fall back to the union only where a marking belongs to no single kerb, which is the
    corner treatments.

    THE OBJECTS STILL READ THE UNION, deliberately: `driven` is ground a vehicle crosses, and a
    flex post standing on it is in the way whichever kerb's entrance put it there. That is how two
    posts standing in W Broad & Louellen's junction throat came out of the drawing.
    """
    by_kerb: dict = field(default_factory=dict)     # (leg, side) -> KerbSideOpenings

    @property
    def everywhere(self) -> KerbSideOpenings:
        """Every kerb's openings unioned - the answer for a marking that belongs to no one kerb."""
        return KerbSideOpenings(
            driveway_mouths=_union([o.driveway_mouths for o in self.by_kerb.values()]),
            driveway_tapered=_union([o.driveway_tapered for o in self.by_kerb.values()]),
            intersection_mouths=_union([o.intersection_mouths for o in self.by_kerb.values()]))

    def on(self, leg, side) -> KerbSideOpenings:
        if leg is None or side is None:
            return self.everywhere
        return self.by_kerb.get((leg, str(side)), KerbSideOpenings())

    @property
    def driven(self) -> object:
        """Every entrance on every kerb. What an OBJECT is kept out of - see the class docstring
        for why that one question is not asked per kerb."""
        return self.everywhere.driven

    @property
    def tapered(self) -> object:
        return self.everywhere.tapered

    def against(self, kind, leg=None, side=None) -> object:
        return self.on(leg, side).against(kind)

    def dotted(self, kind, leg=None, side=None) -> object:
        return self.on(leg, side).dotted(kind)

    def __bool__(self) -> bool:
        driven = self.driven
        return driven is not None and not driven.is_empty


def _stands_in_a_crossing(keep_clear, geometry) -> bool:
    """Whether an OBJECT is standing on a painted crossing, so it must not be placed.

    Measured against `keep_clear`, which is the crossings already buffered by
    PAINT_TO_CROSSWALK_GAP_FT. The gap is deliberately included: a post a foot from a crosswalk
    is a post in the crosswalk as far as anyone walking into it is concerned, and the same
    striper's gap that keeps paint off it keeps a bollard off it.
    """
    return (keep_clear is not None and not keep_clear.is_empty
            and keep_clear.intersects(geometry))


def stands_in_an_opening(openings, geometry) -> bool:
    """Whether an OBJECT belongs to ground a vehicle drives over, so it must not be placed.

    Shared by PaintContext.emit and the prop builders in src/render/props.py, which compute their
    own post positions and would otherwise disagree with the paint about where a post stands -
    the 2D/3D split this project keeps finding. Takes the openings rather than the state so a
    caller that already has them does not rebuild them per post.

    Measured against `driven`, not against the taper: a post beside a driveway is in the way only
    if it stands in the entrance. The extra few feet the hatching gives up is paint ending
    gracefully, not roadway a car uses.
    """
    if openings is None or geometry is None:
        return False
    driven = openings.driven if isinstance(openings, KerbOpenings) else openings
    return driven is not None and driven.intersects(geometry)


# How many points the fillet's arc is sampled at. A curve, not a staircase - see _opening_run_out
# for why it stopped being one.
FILLET_ARC_POINTS = 28


def _opening_run_out(leg, side, inner_ft, outer_ft, start_ft, end_ft):
    """The fillet a hatched zone ends on at an opening: one polygon per end, or [].

    An arc of radius = the strip's own depth, TANGENT TO THE ZONE'S EDGE LINE at the travel lane
    and arriving at the mouth at the kerb. So the zone's outline runs straight beside the lane,
    peels away in one sweep, and meets the entrance - which is what the white line does around a
    driveway apron on a real street, and the reason this is a fillet rather than a chamfer or a
    bulge. `run(u) = R - sqrt(R^2 - (R-u)^2)` for a strip depth R, u measured out from the lane
    edge: R at the lane edge, 0 at the kerb, vertical tangent at u=0.

    SAMPLED AS THE ARC ITSELF, in the leg's own frame. It was 32 nested offset_band_polygon
    slices, whose staircase was only smooth because OPENING_TRIM_FT was buffered over the top of
    it afterwards - and that buffer is what put a BULGE where the sweep begins, since a round
    buffer grows the fillet by 1.5 ft in every direction including along its own tangent, so the
    curve left the edge line 1.5 ft wide instead of at a point. The trim belongs to the mouth,
    where a turning vehicle needs the room; the fillet is exact and unbuffered, and joins the
    trimmed mouth at both of its ends because it is built from the trimmed stations.

    `outer_ft` HAS TO BE THE REAL KERB, measured off the band, not the nominal width the band was
    asked for. The band is deliberately requested wider than the road so offset_band_polygon
    clamps it to the traced kerb - and profiling across the 25.9 ft that WAS asked for on a strip
    only 7.6 ft deep put every step within 3% of the full run, i.e. a square-ended gap 4 ft wider
    at both ends with no taper in it at all. The result is intersected with the kerbside strip by
    the caller, which is what holds the arc to the traced kerb rather than to this one number.
    """
    from src.geometry.model import inset_point_at_station
    from src.geometry.targets import Side

    depth_ft = outer_ft - inner_ft
    if depth_ft <= 0:
        return []
    radius_ft = depth_ft * OPENING_FILLET_PER_DEPTH
    sign = Side(side).sign

    def at(station_ft, offset_ft):
        return tuple(inset_point_at_station(leg, station_ft, sign * offset_ft))

    out = []
    for mouth_ft, direction in ((start_ft, -1), (end_ft, +1)):
        ring = []
        for i in range(FILLET_ARC_POINTS + 1):
            u_ft = depth_ft * i / FILLET_ARC_POINTS
            run_ft = radius_ft - math.sqrt(max(0.0, radius_ft ** 2 - (radius_ft - u_ft) ** 2))
            ring.append(at(max(mouth_ft + direction * run_ft, 0.0), inner_ft + u_ft))
        ring.append(at(mouth_ft, inner_ft))     # back along the mouth, then the lane edge closes it
        fillet = Polygon(ring)
        if not fillet.is_valid:
            fillet = fillet.buffer(0)
        if not fillet.is_empty:
            out.append(fillet)
    return out


def kerb_opening_bands(state) -> KerbOpenings:
    """Where the kerbside markings open for a vehicle, in the two shapes KerbOpenings holds.

    WHERE A VEHICLE CROSSES THE KERB, the markings it drives over open for it. A driveway is not
    a place to paint a bike lane's green surface, a parking stall or a hatched buffer across:
    those markings describe how the kerbside is used, and at a driveway it is used as an
    entrance. The spans come from the traced kerbs' own kerb=lowered / kerb=flush tags - see
    src/geometry/kerbs.py for why a dropped kerb rather than a driveway way is the signal.

    HOW DEEP. From the travel lane's edge out to the real kerb, and no further in. A driveway
    breaks what a car drives over on its way in; it does not break the line that marks the edge
    of the running lane, which carries straight past. TARGET_LANE_WIDTH_FT is the inner bound
    because that is the lane every treatment here holds - TravelLanesKeepTheirWidth is the
    invariant that makes it true, so no kerbside marking on a passing leg starts inside it.

    THE ENDS ARE TRIMMED BACK AND ROUNDED by OPENING_TRIM_FT, so a vehicle turning in or out has
    a little room and the gap reads as an entrance rather than as a rectangle punched through the
    markings. Deliberately small: this is cohesion, not a swept-path design, and every foot of it
    is a foot of bike lane or hatching given up. The trim is clipped back inside the kerbside
    strip so it can never reach into the travel lane, whose edge line runs straight past.

    AND A HATCHED ZONE GETS MORE THAN A TRIM. The trim alone is a foot and a half at 2D scale -
    correct for an entrance, but it left every no-travel zone ending on a blunt transverse edge,
    which is not how one ends anywhere else in this project: at a crossing it ends on the
    crossing's own diagonal. So the fills are cut against `tapered`, the same band plus
    _opening_run_out, and the lines and the green against `driven`.

    NEITHER THE TRIM NOR THE FILLET IS APPLIED AT AN INTERSECTING APPROACH, and both omissions
    are the same point: they model a DRIVEWAY APRON, which is a thing a street mouth does not
    have. A driveway's apron flares at the kerb and a car turning in cuts the corner, so the gap
    is widened and rounded and a hatched zone sweeps away from it. A street mouth's flare is its
    CORNER RETURN, and it is already in the geometry - kerbs._mouth_from_the_tracing takes the
    mouth from where the traced kerb really stops, which is that return's tangent point - so
    adding an apron's 1.5 ft to each end counts the same flare twice, and sweeping a fillet onto
    it draws a driveway apron across the mouth of Blackwell Avenue. Measured at Broad &
    Greenwood, the trim alone was 3.0 ft of kerb per intersection given up to a feature that is
    not there.

    A zone that ends at a street therefore ends SQUARE, which is not a shrug - it is the same end
    zone_end_line_ft already draws for a zone with nothing to end against, and past it the
    statutory setback (R.S. 39:4-138(e), src/geometry/daylighting.py) has usually stopped the
    parking well before the mouth anyway.
    """
    from src.geometry.model import offset_band_polygon
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT, divider_shift_toward_ft

    by_kerb: dict = {}
    for (leg_name, side), openings in getattr(state, "kerb_openings", {}).items():
        leg = state.legs.get(leg_name)
        if leg is None or leg.curb_to_curb_ft is None:
            continue
        # The whole kerbside strip on this side, as the bound the trim is clipped to. The outer
        # offset is deliberately past the nominal half-width: offset_band_polygon clamps it to
        # the traced kerb, so asking for more than the road has means "out to the kerb, wherever
        # it really is" rather than to a mid-block cross-section.
        # WHERE THE KERBSIDE ZONE BEGINS, which is where the travel lane ENDS on this side - not
        # a fixed TARGET_LANE_WIDTH_FT from the alignment. Those coincide only while the two travel
        # lanes straddle the alignment. Under a two-way bike lane the section starts far closer in
        # (4.22 ft from the alignment on e_broad_st_east against 11), so a region beginning at 11
        # covered only the OUTER part of the lane: the driveway break was drawn across some of the
        # bike lane and not the rest of it, which is visible in the render as striping that stops
        # part way across. Same signed definition every check and both renderers use.
        inner_ft = divider_shift_toward_ft(state, leg_name, side) + TARGET_LANE_WIDTH_FT
        # BEYOND THE TRACING, because an opening is a FACT about the street and not a marking
        # proposed on it - the same short list model.paint_stations lets past that bound, and for
        # the same reason a daylight zone is on it. Where a vehicle crosses the kerb does not stop
        # being true where nobody traced the kerb, and a cut built only over the traced stretch is
        # narrower than the markings it has to cut. W Broad & Louellen's south kerb is traced from
        # station 60.3 against a junction mouth of 0-68.0, so without this the mouth came out 7.7 ft
        # long and left the daylight hatching it exists to remove standing in the intersection in
        # pieces - MORE pieces than before it was cut, which is the worst of both.
        kerbside = offset_band_polygon(leg, side, inner_ft, leg.curb_to_curb_ft, 0.0, None,
                                        beyond_the_tracing=True)
        for opening in openings:
            band = offset_band_polygon(leg, side, inner_ft, leg.curb_to_curb_ft,
                                        opening.start_ft, opening.end_ft,
                                        beyond_the_tracing=True)
            if band is None or band.is_empty:
                continue
            # The kerb as traced HERE, off the band the clamping already produced - see
            # _opening_run_out for the flat 4 ft gap that using the requested width gave instead.
            _stations, offsets = station_offset_many(
                leg.centerline, np.asarray(band.exterior.coords, dtype=float))
            # THE TRIM IS THE MOUTH'S, and only the mouth's. JOIN_STYLE 1 is round, so the corners
            # where the entrance meets the travel lane edge and the kerb come off as arcs rather
            # than right angles. Buffering the fillet along with it - which is what this did - grew
            # the sweep by 1.5 ft in every direction including along its own tangent, so the curve
            # left the edge line 1.5 ft wide: the bulge where the sweep begins.
            #
            # AND ONLY AN APRON HAS ONE. An intersecting approach keeps the mouth the tracing
            # gave it, square - see this function's docstring for why the trim and the fillet are
            # one decision and both belong to a driveway.
            if opening.is_an_intersection:
                mouth, run_out = band, []
            else:
                mouth = band.buffer(OPENING_TRIM_FT, join_style=1, cap_style=1)
                # Grown from the TRIMMED mouth, so the arc's square end lands exactly on the
                # entrance's edge and the two join without a step.
                run_out = _opening_run_out(leg, side, inner_ft,
                                           float(np.abs(offsets).max()),
                                           opening.start_ft - OPENING_TRIM_FT,
                                           opening.end_ft + OPENING_TRIM_FT)
            shapes = by_kerb.setdefault((leg_name, side), {"driveway_mouths": [],
                                                            "driveway_tapered": [],
                                                            "intersection_mouths": []})
            if opening.is_an_intersection:
                targets = [(mouth, "intersection_mouths")]
            else:
                targets = [(mouth, "driveway_mouths"),
                           (unary_union([mouth, *run_out]), "driveway_tapered")]
            for shape, target in targets:
                if kerbside is not None and not kerbside.is_empty:
                    shape = shape.intersection(kerbside)
                if not shape.is_empty:
                    shapes[target].append(shape)
    return KerbOpenings(by_kerb={
        key: KerbSideOpenings(**{name: (unary_union(parts) if parts else None)
                                  for name, parts in shapes.items()})
        for key, shapes in by_kerb.items()})


def apron_polygon(state, corner: tuple[str, str], apron, center_ft):
    """The ground one CornerApron covers - a fixed-depth kite, or the swept-path annulus.

    Two shapes because there are two reasons for an apron; see
    src/geometry/treatments/base.py:CornerApron. The annulus is the one a curb extension lays, and it
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
