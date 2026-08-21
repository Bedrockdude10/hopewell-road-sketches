"""THE MACHINERY EVERY TREATMENT PAINTS THROUGH, and the one function that runs it.

`PaintContext.emit` is the single door paint comes out of: it clips against the crossings, holds
the piece inside the traced kerb, drops what is too small to draw, and records what cut the zone
short. A treatment that builds its own polygon and appends it is a second construction of the same
locus, which is a second chance to disagree with the first.

THE GEOMETRY IS DECIDED HERE AND BOTH RENDERERS DRAW WHAT THEY ARE HANDED. src/checks.py inspects
these pieces rather than rebuilding them, for the same reason.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from shapely.ops import unary_union
from src.geometry.model import clip_paint_clear_of, station_offset_many, through_street_sides
from src.geometry.markings import ZONE_END_LINE, lies_legitimately_on, yields_the_ground_to
from src.geometry.paint.pieces import LANE_EDGE_LINE_WIDTH_FT, PaintPiece, RimCause, SURFACE_PAINT_GROUP
from src.geometry.paint.anchors import (MIN_LINE_LENGTH_FT, MIN_RIM_LENGTH_FT, MIN_ZONE_AREA_SQ_FT,
                                        PAINT_TO_CROSSWALK_GAP_FT, RIM_SNAP_FT, ZONE_END_REACH_FT,
                                        _lies_wholly_behind, leg_anchors)
from src.geometry.paint.openings import (DASH_CROSSING_SLACK_FT, DOTTED_MARK_FT, _dash_spans,
                                         _held_inside_the_kerb, _inside_the_traced_kerb,
                                         _stands_in_a_crossing, _station_band, junction_mouths_ft,
                                         kerb_opening_bands, stands_in_an_opening)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from shapely.geometry import Point
    from src.geometry.treatments.state import DesignState

@dataclass
class PaintContext:
    """The machinery every treatment paints through, and the pieces it has painted so far.

    A treatment owns its own markings (Treatment.paint); what is shared stays shared here - the
    crossing bands everything is cut around, the apron surfaces everything stops at, and the
    running list of pieces.
    """
    state: object
    crosswalk_offsets: dict
    center_ft: object
    keep_clear: object = None          # EVERY painted crossing in the frame, buffered by the gap
    # This junction's own crossings, same buffer. Distinct from keep_clear, which also holds
    # crossings at UNMODELLED junctions in the frame. See anchors() for why paint is cut against
    # those but never aimed at them.
    junction_crossings: object = None
    marked: set = field(default_factory=set)
    straight_through: set = field(default_factory=set)
    props: list | None = None
    openings: object = None            # dropped kerbs a vehicle crosses: paint breaks over them
    # (leg, side) -> (0.0, end_ft): where THIS junction's own mouth ends on each kerb, as
    # junction_mouths_ft resolved it for `openings`. Held as the numbers and not only as the
    # opening polygon because a treatment has to AIM at the mouth's end as well as be cut by it -
    # see anchors(). One resolution, two uses; recomputing it here is how the aim and the cut
    # drifted apart in the first place.
    junction_mouths: dict = field(default_factory=dict)
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
        the carriageway - so the ground it occupies is not roadway to be hatched. Same layering
        the crossings get, one rung further up: a surface outranks a marking the way a marking
        outranks a buffer.

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
        so a post that lands in one is DROPPED rather than shortened.

        AN OPENING: bollards marching across a driveway read as a protected lane whose protection
        you are expected to drive through. A CROSSING: a post planted in a marked crosswalk is the
        same statement about a person walking. `keep_clear` and not `junction_crossings` - every
        crosswalk in the picture, for the same reason `add` cuts against all of them.
        """
        if piece.kind.is_object and (stands_in_an_opening(self.openings, piece.geometry)
                                      or _stands_in_a_crossing(self.keep_clear, piece.geometry)):
            return piece
        self.pieces.append(piece)
        return piece

    def add(self, kind, geometry, leg=None, side: str | None = None, beyond_ft=None,
            shares_a_kerb=False):
        """Clip `geometry` clear of the crossings, keep what survives, return those pieces.

        beyond_ft drops any surviving piece that fell WHOLLY on the JUNCTION side of the
        crossing. A zone drawn deliberately through a crossing (so the crossing cuts its end into
        a clean diagonal) leaves an offcut back at the corner, and that offcut is not paint.

        Narrow: an offcut inside the corner return is already removed by the junction's mouth
        (kerbs.OpeningSource.JUNCTION), so what is left for this to catch is only the strip
        between the mouth and the crossing, on the legs where the crossing sits outside the
        corner. Across all four sites and every scenario that is one 1.7 sq ft sliver, on
        broad_st_east - worth knowing, because a filter doing almost nothing is a filter whose
        failure would be invisible. Kept and made sound (see _lies_wholly_behind), not trusted.

        shares_a_kerb dedupes against the other zones on the same through-running kerb.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        if shares_a_kerb and self.through_painted:
            geometry = geometry.difference(unary_union(self.through_painted))
            if geometry.is_empty:
                return added
        geometry = self._clear_of_the_paint_already_down(kind, geometry)
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
        # the cut just made, so a treatment cannot paint the solid part and forget the broken one.
        added.extend(self._dashes_across_openings(kind, geometry, leg, side))
        return added

    def _clear_of_the_paint_already_down(self, kind, geometry):
        """Hold a new area fill out of ground an earlier one already covers.

        checks.MarkingsDoNotCollide is the invariant this serves, and it is not a rendering
        nicety: real paint is opaque and laid once, so two zones over one patch assert two
        different things about it. `shares_a_kerb` above is the same rule for the narrow case it
        was written for - two zones on one through-running kerb - and the case it misses is the
        CORNER, where the two zones belong to DIFFERENT leg-sides and meet only because a kerb
        ends at the junction mouth and the hatching carries on over the open throat (see
        leg_frame.paint_stations). At W Broad & Louellen that is W Broad's lane-narrowing buffer
        against Louellen's daylight zone, 1.0 sq ft, and only on a wide sheet - a corner sliver
        whose size is a function of the frame, which is exactly the class of defect that should
        not be left to whoever notices it in a render.

        Whoever is down first keeps the ground. Which zone that is, is insertion order and so
        arbitrary; it is also immaterial at a sliver, and any overlap big enough for the choice to
        matter is a design fault that MIN_ZONE_AREA_SQ_FT and the invariants should surface rather
        than this quietly resolving. Layers are exempt in either direction - markings.MAY_LIE_ON,
        the same predicate the check consults, so the two cannot drift apart.
        """
        if not kind.covers_area:
            return geometry
        yielding = []
        cut = {}
        for i, piece in enumerate(self.pieces):
            if not piece.covers_area or lies_legitimately_on(kind, piece.kind):
                continue
            if not piece.geometry.intersects(geometry):
                continue
            if yields_the_ground_to(piece.kind, kind):
                # The zone already down is the one that gives way, so cut IT and leave the
                # incoming geometry whole. Rewritten in place: this runs while the list is being
                # built, and a piece nobody has read yet is still a decision, not a drawing.
                cut[i] = piece.geometry.difference(geometry)
            else:
                yielding.append(piece.geometry)
        if cut:
            # ONE POLYGON PER PIECE, because that is what add() guarantees everywhere else and
            # what the hatchers and the digest read. A cut can sever a zone in two, and two
            # separate zones are two pieces - a MultiPolygon smuggled into one would hatch as a
            # single run across the gap. Offcuts below MIN_ZONE_AREA_SQ_FT go the same way they
            # would have if the zone had arrived at that size.
            rebuilt = []
            for i, piece in enumerate(self.pieces):
                if i not in cut:
                    rebuilt.append(piece)
                    continue
                for part in getattr(cut[i], "geoms", [cut[i]]):
                    if part.geom_type == "Polygon" and part.area >= MIN_ZONE_AREA_SQ_FT:
                        rebuilt.append(replace(piece, geometry=part))
            self.pieces[:] = rebuilt
        return geometry.difference(unary_union(yielding)) if yielding else geometry

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

    def _dashes_across_openings(self, kind, geometry, leg, side: str) -> list:
        """The marks of `kind`'s dotted extension, laid IN the openings its row says it crosses.

        THE PARENT MARKING, CLIPPED - not rebuilt over the dash stations. The dashes have to lie
        exactly on the line they continue, and a second construction of the same locus is a second
        chance to disagree with it (see the module docstring). It also means a marking gets its
        extension from its row alone, with nothing for a treatment to supply but the phase.

        Confined to the opening's own ground, which is what "laid IN an opening" means. The dash
        spans are STATION bands, and a station band is a band right across the marking: where an
        entrance meets the street at a skew the span reaches further along the kerb than the
        entrance's own polygon does, by more the wider the marking is. The clip against `dotted`
        is what keeps the two exact complements.
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

    def add_across_the_junction(self, kind, geometry, min_area_sq_ft: float | None = None,
                                 min_length_ft: float | None = None) -> list:
        """A marking laid in the junction BOX, belonging to no single leg.

        `add` is the wrong door for this and the reason is its opening clip. A marking's row in
        AT_AN_OPENING says what it does at an entrance, and BIKE_LANE_SURFACE's says DOTTED - so
        `add` subtracts the intersection mouths, which is exactly the ground a lane extension
        exists to occupy. Sent through there, every mark of a crossbike would be cut away by the
        rule that calls for it.

        The other half of BIKE_LANE_DOTTED_EXTENSION's CARRIED/CARRIED row: the marking laid
        inside an opening is never cut against the opening it exists to cross. This is that door
        opened to a caller that builds the ground itself rather than clipping a parent marking to
        it - which a lane extension has to, there being no parent marking spanning the box.

        STILL CUT AGAINST THE TWO THINGS THAT ARE NOT OPENINGS. A crossing outranks this like it
        outranks everything else on the kerb (`keep_clear`, every crossing in the frame, not just
        this junction's), and an apron is built ground that paint stops at. So a crossbike gives
        way to the zebras it runs beside and stops at a mountable corner, which is the layering
        every other marking here already gets.

        leg/side are None on the pieces, and truthfully: the geometry spans two frames and sits
        in neither. Every invariant in src/checks.py already skips a piece without a leg - that
        is how the corner treatments' paint is handled - so this inherits the right answer rather
        than needing an exemption written for it.

        `min_area_sq_ft` IS "A PART OF A MARK IS NOT A MARK" for an area. MIN_ZONE_AREA_SQ_FT is
        NOT that rule - it is a hairline floor for clip slivers, and at 1 sq ft it passes a wedge
        that is plainly an artifact. A caller laying discrete marks knows what a WHOLE one is and
        can say so; left None for a caller laying a continuous zone, where there is no whole mark
        to be a fraction of. `min_length_ft` is the same rule for a LINE mark (the value
        _dashes_across_openings uses is DOTTED_MARK_FT / 2). Two arguments because a dash and a
        patch of colour are not measured in the same units, not because they are different rules.
        """
        added = []
        if geometry is None or geometry.is_empty:
            return added
        floor_sq_ft = MIN_ZONE_AREA_SQ_FT if min_area_sq_ft is None else min_area_sq_ft
        floor_ft = MIN_LINE_LENGTH_FT if min_length_ft is None else min_length_ft
        for whole in clip_paint_clear_of(geometry, self.surfaces):
            for part in clip_paint_clear_of(whole, self.keep_clear):
                if kind.covers_area and part.area < floor_sq_ft:
                    continue
                if kind.is_line and not kind.is_object and part.length < floor_ft:
                    continue
                piece = PaintPiece(kind, part)
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
        an entrance, so a lane that simply ENDS at an opening has nothing to extend. The two cases
        look identical to a clip and are opposite in meaning: a lane stopping 19 ft short of a
        mouth's far end otherwise yields an "extension" of one whole mark and three 0.1-1.1 ft
        stubs.

        UNLESS THE OPENING RUNS OFF THE END OF THE DRAWING, in which case there IS no afterwards to
        look for and the absence of one says nothing. E Broad's east approach ends inside a 57.9 ft
        dropped kerb and its west approach inside a side-street mouth; read as lanes that end,
        those two forfeited the last 59.4 ft and 17.3 ft of their kerb to no marking at all. The
        street does not stop at the edge of the sheet and neither does the entrance, so the lane is
        carried to the edge dotted - which is also what it would be doing at the station after.
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
            if phase_lo >= lo - DASH_CROSSING_SLACK_FT:
                continue        # the marking starts here - it is not coming into this opening
            if (phase_hi <= hi + DASH_CROSSING_SLACK_FT
                    and hi < centerline.length - DASH_CROSSING_SLACK_FT):
                continue        # the marking ends here rather than crossing - nothing to extend
            spans.extend(_dash_spans(lo, hi))
        return spans

    def rim(self, fills, kind) -> None:
        """The line along a fill's cut end - at a crossing, AND around a kerb opening.

        A hatched zone is outlined, and that outline carries on around the end where something
        cuts it: the diagonal that finishes a zone off against a crossing, and the fillet that
        sweeps it around a driveway mouth. Without it the zone just stops, with hatch strokes
        ending in mid-air.

        `kind` is the zone's OWN edge line, passed by the caller, so the rim is the same paint
        continued rather than a line of its own colour. On a real street the white line beside the
        lane peels away around the apron and comes back as one continuous stroke.

        A rim is only the part of the cut that is NOT ALREADY PAINTED. Where the fillet meets the
        zone's inner edge the two run together, and emitting the whole intersection lays a second
        lane edge line on top of the first: it is a joint, not a stroke.
        """
        # The tolerance the collision check uses, imported rather than restated: "already painted"
        # has to mean here what it means there. Local import to break a cycle.
        from src.checks import COLLINEAR_PAINT_TOLERANCE_FT

        for piece in fills:
            # Seeded with EVERY line already on this kerb, not only this kind's: a kerbside
            # zone's inner edge is also some other marking's outer edge - beside a bike lane it is
            # the lane's own outer stripe. Whatever kind painted it, it is painted.
            painted = [p.geometry for p in self.pieces
                       if p.kind.is_line and p.leg == piece.leg and p.side == piece.side]
            for cutter, cause in (
                    (self.keep_clear, RimCause.CROSSING),
                    (self.openings.against(piece.kind, piece.leg, piece.side)
                     if self.openings else None, RimCause.OPENING)):
                if cutter is None or cutter.is_empty:
                    continue
                # HALF A STRIPE OUTSIDE THE FILL, because that is where the line it continues
                # runs (lane_edge_stripes puts the edge line's centre half its own width outside
                # the hatching). A rim traced on the fill's boundary instead sits 0.41 ft to the
                # side of the line it is part of, which near the fillet's tangent point stretches
                # into a 1.78 ft break in the line.
                # ROUND joins, not mitre. A mitre corner extends to half a stripe / cos(t/2), so
                # where the zone's inner edge turns to sweep around an opening the join spikes
                # 0.16 ft into the travel lane at a right angle. A spike is not part of the line.
                grown = piece.geometry.buffer(LANE_EDGE_LINE_WIDTH_FT / 2, join_style=1)
                edge = grown.exterior.intersection(
                    cutter.buffer(RIM_SNAP_FT + LANE_EDGE_LINE_WIDTH_FT / 2))
                # The buffer grows the fill in EVERY direction, the kerb included, so the rim
                # would sit half a stripe over the TRACED kerb. Held back inside it here: a
                # marking may meet the kerb, never cross it.
                if piece.leg and piece.side:
                    inside = _inside_the_traced_kerb(self.state.legs[piece.leg], piece.side, edge)
                    if inside is not None and not inside.is_empty:
                        edge = edge.intersection(inside)
                for part in getattr(edge, "geoms", [edge]):
                    if part.geom_type != "LineString":
                        continue
                    # A zone can be cut by a crossing AND by a driveway at the same corner, and
                    # where the two cuts converge their rims run together. The sweep is one stroke
                    # however many things cut it.
                    if painted:
                        part = part.difference(
                            unary_union(painted).buffer(COLLINEAR_PAINT_TOLERANCE_FT))
                    for got in getattr(part, "geoms", [part]):
                        got = _held_inside_the_kerb(self.state.legs.get(piece.leg), piece.side, got)
                        if got.geom_type == "LineString" and got.length >= MIN_RIM_LENGTH_FT:
                            self.pieces.append(PaintPiece(kind, got, piece.leg, piece.side,
                                                          rim=cause))
                            painted.append(got)

    def anchors(self, leg_name: str, side: str, inner_offset_ft: float = 0.0):
        """This leg-side's measuring stations, with the shared crossing geometry filled in.

        THIS JUNCTION'S CROSSINGS ONLY, and this is the one place the distinction bites.
        leg_anchors takes the FURTHEST crossing reach and starts the kerbside treatment beyond it
        - right for a crossing at the corner the paint is backing away from, catastrophic for one
        220 ft down the block. Handed the full set, broad_st_east's taper would aim at station 322
        (Blackwell's far crossing) instead of 26, and the leg's whole treatment would vanish.
        """
        mouth = self.junction_mouths.get((leg_name, side))
        return leg_anchors(self.state, leg_name, side, self.crosswalk_offsets,
                            self.junction_crossings, inner_offset_ft=inner_offset_ft,
                            crosswalk_is_marked=leg_name in self.marked,
                            mouth_end_ft=None if mouth is None else mouth[1])


def curbside_paint_ft(state: "DesignState", crosswalk_offsets: dict, center_ft: "Point",
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
    Omit them and this design's paint runs straight over a crossing another junction owns.
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
    # EVERY leg's band, not just the painted ones. What paint has to keep clear of is a fact
    # about paint, so `bands` above is filtered to the marked legs; where the JUNCTION ends is a
    # fact about the street, and a leg without a painted crossing still has one - see
    # junction_mouths_ft.
    mouths = junction_mouths_ft(state, crosswalk_bands)
    openings = kerb_opening_bands(state, mouths)
    # --- and now the treatments paint themselves. Each one that has markings owns them
    # (Treatment.paint), so a marking's geometry lives beside the validation and the provenance
    # of the thing that calls for it, rather than in a block of this function keyed off one of
    # DesignState's dicts.
    #
    # Dispatched in painting order, and deduplicated by (type, target) keeping the LAST applied:
    # a design's dicts are last-write-wins, so two MarkedParking treatments on one kerb are one
    # marked lane and not two painted on top of each other.
    ctx = PaintContext(state=state, crosswalk_offsets=crosswalk_offsets, center_ft=center_ft,
                        keep_clear=keep_clear, junction_crossings=junction_crossings,
                        marked=marked, openings=openings, junction_mouths=mouths,
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
    return without_stranded_end_lines(ctx.pieces)


def without_stranded_end_lines(pieces: list[PaintPiece]) -> list[PaintPiece]:
    """Drop every square end that no longer has hatching to close. Runs LAST, and has to.

    A zone's end line is the transverse stripe across the front of a hatched zone, placed by the
    treatment that placed the hatching, at the station that treatment ASKED the zone to start at.
    Whether the zone still starts there is not knowable then: a later treatment's fill outranks it
    and takes the ground (`_clear_of_the_paint_already_down` rewrites pieces already placed), and a
    crossing or an opening can take the front off as well. So the line is placed hopefully and the
    question is settled here, once, against the geometry that is actually going out.

    THE CALLER CANNOT DO THIS AND NEITHER CAN THE CALLER'S OWN PIECE LIST. Both were tried. The
    pieces `ctx.add` hands back are REPLACED, not mutated (`replace(piece, geometry=part)`), so a
    list captured at add time still holds the uncut polygons and reports a zone reaching back to a
    station it no longer reaches. On w_broad_st_southwest's right kerb `daylight_fill` cut
    `lane_narrowing_fill` from 2855.7 to 2761.0 sq ft and was then dropped below
    MIN_ZONE_AREA_SQ_FT itself, which left the end line standing 9.3 ft clear of any hatching and
    3.2 ft inside the junction mouth - fatal to NoPaintInsideTheJunction. Same defect, three ways
    of arriving at it; only this one sees the answer.

    IT WAS ALSO SURVIVING ON FLOAT LUCK BEFORE THAT. Drawn at its requested station the line lies
    exactly along the transverse edge of the junction mouth's cut, and whether GEOS's difference
    keeps a line lying on a polygon's boundary depends on where the overlay nodes it. Widening the
    opening cut by a foot for unrelated reasons moved the coincidence and the line appeared. A
    drawing decision resting on that is a defect whichever way the luck falls, so the rule is
    stated instead of left to the overlay.

    Touching is measured with ZONE_END_REACH_FT, one edge-line width: closer than the width of the
    line itself is not a gap a reader can see, and a real cut here moves the zone by feet.
    """
    hatching: dict[tuple, list] = {}
    for piece in pieces:
        if piece.kind.is_fill:
            hatching.setdefault((piece.leg, str(piece.side)), []).append(piece.geometry)
    return [piece for piece in pieces
            if piece.kind is not ZONE_END_LINE
            or any(piece.geometry.distance(fill) <= ZONE_END_REACH_FT
                   for fill in hatching.get((piece.leg, str(piece.side)), ()))]
