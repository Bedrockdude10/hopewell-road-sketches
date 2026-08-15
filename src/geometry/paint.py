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

from src.geometry.model import (_point_at, clip_paint_clear_of, corner_apron_annulus,
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
    OPENING  a driveway's fillet, which the hatching stops short of, because that arc's chord is
             at the hatch angle and a stroke laid beside it reads as a fork.
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
    openings: object = None            # dropped kerbs a vehicle crosses: paint breaks over them
    surfaces: object = None            # the mountable aprons: paint stops at them
    surface_polygons: list = field(default_factory=list)
    pieces: list = field(default_factory=list)
    # Zones already placed on a kerb that runs straight through - see add(shares_a_kerb=True).
    through_painted: list = field(default_factory=list)

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

        WITH ONE CLIP, because a kerb opening is not paint either. A flex post cannot be trimmed
        by a driveway the way a stripe can - it is either standing in the entrance or it is not -
        so a post that lands inside an opening is dropped rather than shortened. Without this the
        paint broke over a driveway and the bollards marched straight across it, which is worse
        than not breaking the paint at all: it reads as a protected lane whose protection you are
        expected to drive through.
        """
        if piece.kind.is_object and stands_in_an_opening(self.openings, piece.geometry):
            return piece
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
        # Cut clear of the mountable surfaces, then of the crossings, then of the kerb
        # openings - each may fragment a piece, so every stage runs over whatever the last left.
        # A hatched zone is cut against the openings' rounded RUN-OUT and everything else against
        # the entrance itself, which is what makes a no-travel zone taper off where a lane line
        # simply stops - see KerbOpenings.
        opening = self.openings.against(kind) if self.openings else None
        surviving = [cut for whole in clip_paint_clear_of(geometry, self.surfaces)
                     for part in clip_paint_clear_of(whole, self.keep_clear)
                     for cut in clip_paint_clear_of(part, opening)]
        for part in surviving:
            if beyond_ft is not None and _station_of(self.state.legs[leg], part) < beyond_ft:
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
        return added

    def opening_dash_spans(self, geometry, leg_name: str) -> list[tuple[float, float]]:
        """The station spans a marking is broken into where `geometry` crosses an opening.

        A LANE DOES NOT STOP AT A DRIVEWAY, it goes dotted - lines and green alike. That is what a
        striper paints and what a rider needs to see: the lane still runs here, and here is where
        it is crossed. This module used to argue the opposite, that a plain gap was the honest
        version of "the paint does not continue", which was really an argument that the marking had
        not been built yet.

        Asked once per lane, off the LANE'S OWN FOOTPRINT, and handed to everything that crosses -
        so the two edge lines and the green between them break at the same stations instead of each
        being dashed along its own length and drifting out of phase. Which shape is canonical
        matters and it is the surface: the lines are its edges.
        """
        driven = self.openings.driven if self.openings else None
        if geometry is None or geometry.is_empty or driven is None:
            return []
        centerline = self.state.legs[leg_name].centerline
        spans = []
        inside = geometry.intersection(driven)
        for part in getattr(inside, "geoms", [inside]):
            if part.is_empty:
                continue
            coords = (part.exterior.coords if part.geom_type == "Polygon" else part.coords)
            stations, _offsets = station_offset_many(centerline, np.asarray(coords, dtype=float))
            spans.extend(_dash_spans(float(stations.min()), float(stations.max())))
        return spans

    def emit_across_opening(self, kind, geometry, leg=None, side=None):
        """One mark of a dotted extension, laid IN an opening rather than clipped out of it.

        Cut clear of the surfaces and the crossings like any other paint - an opening that overlaps
        a crossing band gets no marks across the crossing - but not against the opening itself,
        which is the whole point: `add` would remove exactly what this is placing.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        # CONFINED TO THE OPENING'S OWN GROUND, which is what "laid IN an opening" means. The
        # caller builds this from the STATION SPAN of the opening (opening_dash_spans), and a
        # station span is a band right across the marking: where a driveway meets the street at a
        # skew, the span reaches further along the kerb than the driveway's own polygon does, by
        # more the wider the marking is. `add` removed the polygon, so the difference between the
        # two is ground painted twice - 10 sq ft of it on e_broad_st_west's 12 ft two-way lane,
        # which markings_collide reported. A 5 ft one-way lane skews little enough to stay inside
        # the tolerance, so this was latent rather than absent.
        #
        # `driven` rather than against(kind): the entrance itself is the definition of where an
        # extension may lie. Where the complementary cut used the wider rounded run-out, not
        # filling that run-out is correct - a taper is not something you paint dashes across.
        driven = self.openings.driven if self.openings else None
        if driven is None:
            # NO OPENINGS ON THIS KERB, so there is no opening to extend across and this call has
            # nothing to place. Returning the geometry unclipped instead emitted the whole mark a
            # SECOND time - `add` had already laid the part outside the openings, which on a kerb
            # with none is all of it - so every dash was painted twice down the same stretch.
            # Invisible until a kerb had zero driveways: w_broad_st_northeast is the first, and
            # markings_collide reported the contraflow stripe overlapping itself for 3.0 ft.
            return added
        geometry = geometry.intersection(driven)
        if geometry.is_empty:
            return added
        for clear in clip_paint_clear_of(geometry, self.surfaces):
            for part in clip_paint_clear_of(clear, self.keep_clear):
                if kind.covers_area and part.area < MIN_ZONE_AREA_SQ_FT:
                    continue
                added.append(PaintPiece(kind, part, leg, side))
        self.pieces.extend(added)
        return added

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
                    (self.openings.against(piece.kind) if self.openings else None,
                     RimCause.OPENING)):
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
                        keep_clear=keep_clear, marked=marked, openings=openings,
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


@dataclass(frozen=True)
class KerbOpenings:
    """Where the kerbside markings open for a vehicle - in the TWO shapes that needs.

    A marking does not stop at a driveway the same way whatever it is. The ground a car drives
    over is one shape, and how each kind of paint ends against it is a different question:

      * `driven` is that ground itself: the dropped kerb's own extent, trimmed back and rounded.
        A line stops here (and a lane line then carries a dotted extension across it, see
        PaintContext.dashes_through_openings), the green stops here, a post is dropped if it
        stands here, and this is the entrance's real width.
      * `tapered` is the same thing with a rounded run-out at the travel lane's edge, and it is
        what a HATCHED zone ends against. A no-travel zone that stops square reads as a rectangle
        punched through the hatching; the same zone at a crossing ends on the crossing's own
        diagonal, which is what makes it look painted rather than deleted.

    One shape for everything is what produced the blunt ends: the two questions had the same
    answer because nothing had asked them separately.
    """
    driven: object = None
    tapered: object = None

    def against(self, kind) -> object:
        """The shape `kind` is cut against - the fillet for a hatched zone AND THE LINES THAT BOUND
        IT, the entrance itself for everything else, and NOTHING for the edge of the travelled way.

        The edge line has to go with its zone. Cut at the mouth while the hatching swept away on
        its fillet, it ran on with nothing behind it and the fillet's rim cut across it at an angle
        - a hook and a Y in the render, at every driveway. See markings.ZONE_BOUNDARY_LINES for why
        that set is declared rather than derived from the role.

        And the line marking the edge of the running lane is cut by nothing here, which is what
        this docstring and kerb_opening_bands both always said should happen ("it does not break
        the line that marks the edge of the running lane, which carries straight past") and what
        the code did not do: a parking edge line was cut against `driven` like a stall, so at the
        driveway 178-204 ft along broad_st_east's south kerb the stalls stopped, both lines
        stopped, and 26 ft of kerb was left with nothing drawn on it at all. See
        markings.LINES_UNBROKEN_BY_A_DRIVEWAY for the standard.
        """
        from src.geometry.markings import LINES_UNBROKEN_BY_A_DRIVEWAY, ZONE_BOUNDARY_LINES

        if kind in LINES_UNBROKEN_BY_A_DRIVEWAY:
            return None
        return (self.tapered if kind.is_fill or kind in ZONE_BOUNDARY_LINES else self.driven)

    def __bool__(self) -> bool:
        return self.driven is not None and not self.driven.is_empty


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
    """
    from src.geometry.model import offset_band_polygon
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT, divider_shift_toward_ft

    driven, tapered = [], []
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
        kerbside = offset_band_polygon(leg, side, inner_ft, leg.curb_to_curb_ft, 0.0, None)
        for opening in openings:
            band = offset_band_polygon(leg, side, inner_ft, leg.curb_to_curb_ft,
                                        opening.start_ft, opening.end_ft)
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
            mouth = band.buffer(OPENING_TRIM_FT, join_style=1, cap_style=1)
            # Grown from the TRIMMED mouth, so the arc's square end lands exactly on the entrance's
            # edge and the two join without a step.
            run_out = _opening_run_out(leg, side, inner_ft,
                                       float(np.abs(offsets).max()),
                                       opening.start_ft - OPENING_TRIM_FT,
                                       opening.end_ft + OPENING_TRIM_FT)
            for shape, target in ((mouth, driven), (unary_union([mouth, *run_out]), tapered)):
                if kerbside is not None and not kerbside.is_empty:
                    shape = shape.intersection(kerbside)
                if not shape.is_empty:
                    target.append(shape)
    return KerbOpenings(driven=unary_union(driven) if driven else None,
                        tapered=unary_union(tapered) if tapered else None)


def apron_polygon(state, corner: tuple[str, str], apron, center_ft):
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
