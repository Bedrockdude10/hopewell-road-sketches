"""THE TREATMENTS THAT PLACE A BIKEWAY, and all the paint one puts down.

`AddTwoWayBikeLane` subclasses `AddBikeLane` and they are one file for that reason: painting the
section is the same problem at different offsets, and the base class asks `isinstance(self,
AddTwoWayBikeLane)` where the two differ. Splitting an inheritance pair across modules buys a
smaller file and a lazy import, which is a worse trade than reading 500 lines.

THE PAINT COMES OUT THROUGH `PaintContext.emit`, never appended directly, so every piece is
clipped against the crossings and held inside the traced kerb by one code path.
"""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import shapely.ops
from src.geometry.targets import Side
from src.geometry.model import narrowest_half_width_ft
from src.geometry.treatments.base import (LANE_WIDTH_SLACK_FT, PARKING_STALL_LENGTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT, Treatment)
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.sections import (BikeLane, CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                                       MIN_TWO_WAY_BIKE_LANE_FT, NJDOT_TWO_WAY_OBJECTION,
                                                       TWO_WAY_BIKE_LANE_WIDTH_FT, TwoWayBikeLane, _feet)
from src.geometry.treatments.bikeways.fit import far_kerb_surplus_ft, travel_lane_divider_shift_ft
from src.geometry.treatments.bikeways.symbols import (CONTRAFLOW_DASH_FT, CONTRAFLOW_GAP_FT,
                                                      SYMBOL_CLEAR_OF_DIVIDER_FT, SYMBOL_LENGTH_FT,
                                                      SYMBOL_WIDTH_FT,
                                                      bike_symbol_polygon, bike_symbol_stations_ft)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# How far behind the junction node a THROUGH-RUNNING kerb's paint starts, so the two legs' halves
# overlap and fuse instead of each stopping at its own station 0. Enough to cover the 1.28 ft seam
# at W Broad & Louellen with margin, and bounded by the tracing either way, so a kerb traced right
# up to the node overlaps by this much and one traced only from the node out overlaps not at all
# and is no worse off than before.
THROUGH_JUNCTION_OVERLAP_FT = 3.0


@dataclass(frozen=True)
class AddBikeLane(Treatment):
    """Mark an exclusive bike lane along one side of a leg. Paint only - no kerb moves.

    LaneNarrowing cannot express this. It paints a BUFFER: a hatched strip of spare
    asphalt between the travel lane and the kerb, saying "nothing belongs here". A bike lane
    says the opposite about the same ground - that a specific vehicle belongs in it - so it
    needs its own edge line on both sides and its own reserved width, and where it is
    parking-protected it also needs the parking lane to sit OUTSIDE it rather than against the
    kerb in the ordinary way.

    Refused rather than shrunk when the leg cannot hold the cross-section asked for. The point
    of the exercise is to find out which legs can take a bike lane, and a lane quietly narrowed
    to fit answers a different question - see AASHTO_MIN_BIKE_LANE_FT.

    Measured against the NARROWEST point of the traced kerb, not the nominal half-width, because
    a bike lane is a promise about a whole leg and the two figures differ by feet. broad_st_east
    is 52.0 ft nominal - 26.0 per side - and its kerbs come within 22.8 ft of the alignment
    somewhere along the traced run; a cross-section sized off the nominal number would be drawn
    over the kerb there. This is what turns "verify before promising it corridor-wide" from a
    caveat into a refusal.

    The cross-section itself (BikeLane) validates its own widths, and this validates the fit
    against the street - which needs the design, so it happens in apply_to rather than in
    __post_init__. Both refusals are ValueErrors carrying the measurement that caused them.
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 30
    paint_rank: ClassVar[int] = 0
    width_ft: float = 0.0
    buffer_ft: float = 0.0
    parking_ft: float = 0.0
    shy_ft: float = 0.0

    @property
    def lane(self) -> BikeLane:
        """The cross-section this treatment marks - validated on construction, and askable
        without a design, which is how every width in it is tested."""
        return BikeLane(width_ft=self.width_ft, buffer_ft=self.buffer_ft,
                         parking_ft=self.parking_ft, shy_ft=self.shy_ft)

    def section(self, state: "DesignState") -> BikeLane:
        """The cross-section to paint, given the design.

        A hook, because a two-way lane's section depends on BOTH of the leg's half-widths and so
        cannot be known without the state, while a one-way lane's is fixed at construction. Both
        views and every check read the section through here, so a subclass cannot end up
        validated against one cross-section and drawn at another.
        """
        return self.lane

    def __post_init__(self):
        self.lane   # noqa: B018 - evaluated for its exception: raises for a lane under AASHTO's minimum

    def describe(self) -> str:
        # leg, side rather than str(target): a note is meant to read as the constructor call
        # that produced it, so someone reading a render's provenance can paste it back.
        #
        # A DECIMAL WHERE THERE IS ONE. Rounded to whole feet this reported E Broad's narrowed
        # protected lane as "4 ft lane" when it is 4.49 - understating a width by half a foot in
        # the one line a reader would check it against.
        return (f"AddBikeLane({self.target.leg}, {self.target.side}): {_feet(self.width_ft)} ft lane"
                + (f", {_feet(self.buffer_ft)} ft buffer" if self.buffer_ft else "")
                + (f", parking-protected behind {self.parking_ft:.0f} ft of marked parking"
                   if self.parking_ft
                   else f", {self.shy_ft:.1f} ft shy of the kerb" if self.shy_ft else ""))

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        leg = state.legs[self.target.leg]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {self.target.leg!r} has no width - nothing to fit a bike lane into.")
        lane = self.lane
        available_ft = narrowest_half_width_ft(leg, str(self.target.side))
        if lane.total_ft > available_ft + LANE_WIDTH_SLACK_FT:
            raise ValueError(
                f"{self.target.leg} {self.target.side} comes within {available_ft:.2f} ft of the "
                f"centerline at its narrowest traced point ({leg.curb_to_curb_ft / 2:.2f} ft "
                f"nominal), and this cross-section needs {lane.total_ft:.2f} ft "
                f"({TARGET_LANE_WIDTH_FT:.0f} travel + {self.buffer_ft:.1f} buffer + "
                f"{self.width_ft:.1f} bike + {self.parking_ft:.1f} parking + {self.shy_ft:.1f} "
                f"shy). Short by {lane.total_ft - available_ft:.2f} ft.")
        spare_ft = available_ft - lane.total_ft
        return (f". Uses {lane.total_ft:.1f} of the {available_ft:.1f} ft this leg has at its "
                f"narrowest" + (f", {spare_ft:.1f} ft spare." if spare_ft > 0.05 else "."))


    def paint(self, ctx) -> None:
        """An edge line each side of the lane, so it reads as a lane rather than as the spare
        asphalt a lane-narrowing buffer marks; the buffer beside it, and the parking outside it,
        hatched and ticked with the machinery already here."""
        from src.geometry.markings import (BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE,
                                           BIKE_LANE_SURFACE, BIKE_LANE_SYMBOL, BUFFER_EDGE_LINE,
                                           BUFFER_FILL, STALL_DIVIDER)
        from src.geometry.model import (band_from_offsets, curbside_strip_polygon, inset_line_ft,
                                        kerb_inset_offsets, kerb_parallel_line_ft,
                                        kerb_referenced_band_polygon, lane_narrowing_polygons_ft,
                                        offset_band_polygon, paint_stations,
                                        parking_stall_lines_ft, stall_lane_runs_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, MIN_LINE_LENGTH_FT, _one,
                                        end_against_crossing, parking_runs)

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        lane = self.section(ctx.state)
        at = ctx.anchors(leg_name, side, inner_offset_ft=(
            leg.curb_to_curb_ft / 2 - lane.total_ft + TARGET_LANE_WIDTH_FT))
        # A bike lane RUNS INTO its crossing and is cut by it, like every other kerbside zone
        # here - a real one carries on to the crossing and often across it. Stopping it at the
        # corner clearance instead left the buffer 5.5 ft short of the crossing, which
        # test_curbside_paint_ends_against_its_crossing reads as hatching that gave up early.
        through = (leg_name, side) in ctx.straight_through
        if through:
            # BEHIND THE NODE, not up to it. Each leg's paint is built in its own frame, so two
            # halves that both stop at their own station 0 stop just shy of each other - 1.28 ft
            # of hole at W Broad & Louellen, in the middle of a lane whose whole point is running
            # continuously through the junction. Starting behind it makes the two overlap, and the
            # overlap is deduped by shares_a_kerb below, which is the same mechanism that keeps two
            # zones on one through kerb from double-painting. Honoured only as far as the kerb is
            # really traced there - see model.paint_stations.
            start_ft, beyond_ft = -THROUGH_JUNCTION_OVERLAP_FT, None
        elif leg_name in ctx.marked:
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        bounds = lane.offsets_from_centerline_ft()
        kerb = lane.offsets_from_kerb_ft()
        # THE FLOOR IS CAPPED AT THE ROOM THERE IS, and this is the ONLY thing the kerb is
        # allowed to do to the section. floor_ft says "never come inward of the design", which is
        # sound only while the design fits: the test that granted this section measures the TRAVEL
        # WAY, so a section can be granted whose outer stripe sits outside the near kerb - 24.16 ft
        # of section against a kerb 20.32 ft out on W Broad's southwest approach. Held at that
        # floor the lane is drawn over the kerb; capped here it follows the kerb inward instead.
        #
        # WHAT IT MAY NOT DO IS END THE FACILITY. There was a second guard here that stopped every
        # band at the first station where the section no longer fit, on the reasoning that a
        # section is only known to fit over the span it was sized on. It amputated two legs:
        # broad_st_east carried green for 180 ft of a 425 ft leg, under 42 flex posts and a centre
        # stripe that both ran the full length, because the sizing span was 1x and the drawing was
        # 2.5x. Sizing over the whole drawn leg is what actually fixes that (see
        # narrowest_half_width_ft) - a facility that fits the street it is drawn on needs no stop
        # station, and one that does not fit is a design decision for the rung ladder to take, not
        # a length for this method to trim. BikewayReachesTheEndOfItsKerb is the check that says so.
        room_ft = narrowest_half_width_ft(leg, side, max(start_ft, 0.0))
        floor = {key: None if offset_ft is None else min(offset_ft, room_ft)
                 for key, offset_ft in bounds.items()}
        # Every stripe at its own CENTRE, which BikeLane has already offset half a stripe out
        # from the face it marks - so the travel lane keeps its 11 ft and the bike lane keeps
        # its own width, and the paint comes out of the buffer between them. Getting this wrong
        # is not subtle: an edge line centred on the mark leaves a 10.59 ft lane, which
        # PaintClearOfTheTravelLane reports on every vertex.
        #
        # WHICH DATUM EACH BOUNDARY IS MEASURED FROM. The travel lane's edge always comes off the
        # alignment, so the lane holds TARGET_LANE_WIDTH_FT whatever the kerb does. The lane's own
        # two edges come off the KERB where this section hugs it (see BikeLane.hugs_kerb) and off
        # the alignment where it does not - one branch, so a section cannot end up with its green
        # on one datum and its stripes on the other.
        def lane_edge_line(key, from_ft, to_ft=None):
            if lane.hugs_kerb:
                # floor_ft is this stripe's own designed offset, so it follows the kerb OUTWARD
                # and never comes in tighter than the section - see kerb_inset_offsets.
                return (kerb_parallel_line_ft(leg, side, kerb[key], from_ft, to_ft,
                                               floor_ft=floor[key])
                        if kerb[key] is not None else None)
            return (inset_line_ft(leg, side, bounds[key], from_ft, to_ft,
                                   keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)
                    if bounds[key] is not None else None)

        def lane_surface(from_ft, to_ft=None):
            if lane.hugs_kerb:
                return kerb_referenced_band_polygon(leg, side, kerb["bike_outer_ft"],
                                                     lane.width_ft, from_ft, to_ft,
                                                     floor_ft=floor["bike_outer_ft"])
            return offset_band_polygon(leg, side, bounds["bike_inner_ft"], bounds["bike_outer_ft"],
                                        from_ft, to_ft,
                                        keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)

        # THE LANE'S OWN FOOTPRINT, REGISTERED BEFORE ANYTHING ON THIS KERB IS PAINTED. Every
        # marking that crosses an entrance breaks at the stations this shape gives, so the green
        # marks land between the white ones instead of each being dashed along its own length and
        # drifting out of phase. The surface is canonical because the lines are its edges.
        #
        # What then happens at each entrance is markings.AT_AN_OPENING's answer, not this
        # treatment's: the lane's lines and its green go dotted across, the hatched buffer beside
        # it sweeps away from the mouth instead. This method used to re-lay each of those marks
        # itself, in a loop that only the bikeways had - which is why a lane's markings were
        # carried across a driveway and nothing else's ever could be.
        surface = lane_surface(start_ft)
        ctx.dash_phase(leg_name, side, surface)
        ctx.add(BIKE_LANE_EDGE_LINE,
                 inset_line_ft(leg, side, bounds["inner_line_ft"], start_ft,
                                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                 leg_name, side, beyond_ft, shares_a_kerb=through)
        for key in ("buffer_outer_line_ft", "outer_line_ft"):
            ctx.add(BIKE_LANE_EDGE_LINE, lane_edge_line(key, start_ft), leg_name, side, beyond_ft,
                     shares_a_kerb=through)
        # THE LANE'S OWN ASPHALT, PAINTED GREEN - between the two edge stripes, i.e. exactly the
        # width a rider gets. Bounded by the stripes' faces rather than their centres, so the
        # green stops where the white starts instead of running under it; MarkingsDoNotCollide
        # would report the overlap if it did, since a colour covers ground like a hatch does.
        #
        # offset_band_polygon, because the lane's own two offsets are what define it. Built as a
        # difference of two kerb-referenced strips instead, the green ran 6.6 ft past its outer
        # stripe wherever the kerb is unmapped - see that function.
        #
        # Through ctx.add like every other marking, NOT ctx.add_surface: a surface is built
        # ground that everything else is cut around (seal_surfaces), and colouring the lane must
        # not cut the lane's own edge lines - or the buffer hatching beside it - back out.
        ctx.add(BIKE_LANE_SURFACE, surface, leg_name, side, beyond_ft, shares_a_kerb=through)
        # THE BIKE LANE SYMBOL (MUTCD Fig 9E-1). NACTO asks for one after every driveway and
        # intersection and at least every 500 ft; both halves of that rule live in
        # bike_symbol_stations_ft, so this leg, the corridor strip and the 3D export all call for
        # the same symbols. The stations come from state.kerb_openings, which is where a mouth's
        # start and end actually are - the PaintContext knows openings as GROUND, which is the
        # right shape for cutting paint and the wrong one for measuring 15 ft past a mouth.
        mouths = tuple((o.start_ft, o.end_ft)
                       for o in ctx.state.kerb_openings.get((leg_name, side), ()))
        # THE SAME DATUM THE LANE ITSELF IS ON, and getting this wrong put symbols in the buffer.
        # A two-way lane hugs the kerb (BikeLane.hugs_kerb), so its edges are insets FROM THE KERB
        # and its position at a station is wherever the kerb is there. Measuring the symbol's
        # centre off the alignment instead places it where the lane would be if it did not hug -
        # which at Broad & Greenwood was 5 sq ft inside bike_buffer_fill, and the collision check
        # said so. Centreline-referenced only for the sections that are.
        def lane_centre_at(station_ft: float) -> float | None:
            centre_ft = (bounds["bike_inner_ft"] + bounds["bike_outer_ft"]) / 2
            if not lane.hugs_kerb:
                return centre_ft
            # THROUGH kerb_inset_offsets, which is the one home for "this many feet in from the
            # traced kerb" - and the same call the contraflow divider's axis is built from, with
            # the same floor. Rebuilt here as abs(raw kerb) - half instead, it read the RAW
            # tracing where the divider reads the TAPERED one, and the two disagreed by 0.87 ft
            # on broad_st_east: the divider ran 0.20 sq ft through the corner of a stencil that
            # was supposed to sit clear of it. Two derivations of the lane's centre, in agreement
            # with each other nowhere.
            at = kerb_inset_offsets(
                leg, side, np.array([station_ft]),
                (kerb["bike_inner_ft"] + kerb["bike_outer_ft"]) / 2, floor_ft=centre_ft)
            if at is None or not np.isfinite(at[0]):
                return None
            return abs(float(at[0]))
        # WHERE THE LANE IS ACTUALLY DRAWN, and that is not start_ft. start_ft is what this
        # treatment ASKED for; paint_stations then bounds the ask by the stations where the kerb is
        # traced, and on W Broad's southwest approach the two are 22 ft apart - the ask is station
        # 32, the tracing starts at 54.4. Everything between is the crossbike box, which has its
        # own dotted edges and its own carried centre stripe on the extension's straight axis. Run
        # from the ask and two stencils were placed in there off THIS leg's kerb-following centre,
        # 0.8 ft from the stripe actually drawn there, and 0.08 sq ft of one went under it.
        #
        # NOT beyond_ft for the far end either. beyond_ft is a clipping THRESHOLD - the station
        # past which a piece is discarded for lying behind a crossing - and reading it as the run's
        # end gave 6 ft "runs" at Broad & Greenwood, inside which no symbol interval could ever
        # land. Three different quantities that are all stations.
        drawn = paint_stations(leg, side, start_ft)
        for station_ft in (() if drawn is None else
                           bike_symbol_stations_ft(float(drawn[0]), float(drawn[-1]), mouths)):
            centre_ft = lane_centre_at(station_ft)
            if centre_ft is None:
                continue
            # THE LANE MOVES UNDER THE STENCIL, so a pair is spread from the extremes of its centre
            # over the footprint the symbol covers, and not from its value at the middle station. A
            # stencil is rigid - one offset for all of its vertices - while the divider it has to
            # clear follows the traced kerb continuously, and on W Broad's approach that kerb swings
            # 0.70 ft across the 5.5 ft a symbol is long. Read once at the centre, the stencil was
            # placed on a stripe that had moved out from under it by more than the whole clearance.
            over_footprint = [across for across in
                              (lane_centre_at(station_ft - SYMBOL_LENGTH_FT / 2), centre_ft,
                               lane_centre_at(station_ft + SYMBOL_LENGTH_FT / 2))
                              if across is not None]
            inner_ft, outer_ft = min(over_footprint), max(over_footprint)
            # A two-way lane's two halves face opposite ways, so each gets its own symbol: that is
            # what tells a driver at a mouth which direction the rider bearing down on them is
            # coming from, and it is the reason the symbol is worth drawing rather than decorative.
            faces = (True, False) if isinstance(self, AddTwoWayBikeLane) else (True,)
            for index, forward in enumerate(faces):
                # Half a symbol plus the divider's own clearance either side of centre. Bounded
                # by the lane's own sixth so a narrow lane does not push them into the edge
                # stripes, and floored at half a symbol plus that clearance so a wide one does not
                # let the pair touch - which is what a plain sixth did on a 10 ft lane, overlapping
                # them by 2 sq ft.
                room_ft = max(lane.width_ft / 2 - SYMBOL_WIDTH_FT / 2, 0.0)
                spread_ft = max(SYMBOL_WIDTH_FT / 2 + SYMBOL_CLEAR_OF_DIVIDER_FT,
                                lane.width_ft / 6)
                spread_ft = min(spread_ft, room_ft)
                if len(faces) == 1:
                    across_ft = centre_ft
                else:
                    # Clear of the stripe wherever it runs under this stencil, but never further
                    # out than the lane's own edge over that same footprint: a symbol shoved off
                    # the lane to dodge its centre stripe has traded one collision for another.
                    # Where the kerb runs straight the two terms are equal and this is the old
                    # centre +/- spread exactly, which is why a straight leg's symbols do not move.
                    across_ft = (min(outer_ft + spread_ft, inner_ft + room_ft) if index
                                 else max(inner_ft - spread_ft, outer_ft - room_ft))
                # NOT shares_a_kerb, unlike the lane it sits on. That flag subtracts ground the
                # adjoining leg has already painted, which is right for a zone running through the
                # node and fatal for a symbol: the lane's own green is registered first, so a
                # symbol lying inside it was subtracted to nothing and 0 of them reached either
                # renderer. A symbol is a discrete mark at a station, not a run that two legs
                # could each paint half of.
                ctx.add(BIKE_LANE_SYMBOL,
                        bike_symbol_polygon(leg, side, station_ft, across_ft, forward),
                        leg_name, side, beyond_ft)
        if lane.buffer_ft:
            # The hatched buffer, between the two lines that bound it rather than under them.
            # lane_narrowing_polygons_ft measures its stripe inward from the kerb-to-kerb half,
            # so the depth is the distance from the kerb to the buffer's inner FACE, and the
            # zone is then cut back to the buffer's outer face.
            # THE BUFFER IS THE MIXED BAND, and it is the piece that makes the whole split work:
            # its inner face is the travel lane's edge, measured from the alignment so the lane
            # holds its target, and its outer face is the bike lane's inner face, measured from the
            # kerb so the lane hugs it. Everything the street does between those two - the 8 ft
            # convergence at W Broad's junction throat - ends up here, widening the separation
            # exactly where the turning conflicts are, which is where a designer would want it.
            inner_face_ft = bounds["travel_lane_edge_ft"] + LANE_EDGE_LINE_WIDTH_FT
            fill = None
            if lane.hugs_kerb:
                stations = paint_stations(leg, side, start_ft)
                if stations is not None:
                    outer = kerb_inset_offsets(
                        leg, side, stations, kerb["bike_inner_ft"] + LANE_EDGE_LINE_WIDTH_FT,
                        floor_ft=bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT)
                    if outer is not None:
                        # Never inside the travel lane's edge: where the kerb comes in far enough
                        # that the buffer would have negative width it pinches to nothing, rather
                        # than reaching back across the stripe that bounds it.
                        fill = band_from_offsets(leg, side, stations,
                                                  np.full(stations.shape, inner_face_ft),
                                                  np.maximum(outer, inner_face_ft))
            else:
                fill = _one(lane_narrowing_polygons_ft(
                    leg, leg.curb_to_curb_ft / 2 - inner_face_ft,
                    start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,)))
                beyond = curbside_strip_polygon(
                    leg, side, bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT, start_ft)
                if fill is not None and beyond is not None:
                    fill = fill.difference(beyond)
            # Deduped against the other half of the same kerb like the lane itself, or the two
            # legs' buffers overlap through the node now that both reach behind it - 18 sq ft of
            # it at Louellen, which markings_collide reported.
            ctx.rim(ctx.add(BIKE_BUFFER_FILL, fill, leg_name, side, beyond_ft,
                             shares_a_kerb=through), BIKE_LANE_EDGE_LINE)
        if lane.parking_ft:
            # Parking-protected: the stalls sit OUTSIDE the bike lane, between it and the kerb,
            # which is what shields the lane. Ticked at the standard stall length over the runs
            # where parking is legal, exactly as a kerbside parking lane would be.
            #
            # NOT over the legal span directly - the same fix MarkedParking.paint applies and
            # for the same reason (see there): STALL_DIVIDER is STOPPED at every opening, so a
            # grid laid over ground the treatment has not yet been cut against draws a stall
            # with an open-ended tick at a driveway or a crossing, or one straddling it outright.
            # ctx.open_runs asks the real cut ahead of laying anything.
            #
            # NOT beyond_the_tracing: a stall proposes paint on the physical kerb, so it may
            # reach no further than the kerb is actually surveyed - see the note in
            # MarkedParking.paint for the disagreement asking past that bound produces.
            half = leg.curb_to_curb_ft / 2
            outer_off = max(half - lane.shy_ft, 0.5)
            inner_off = max(half - lane.shy_ft - lane.parking_ft, 0.5)
            for run_start_ft, run_end_ft in parking_runs(ctx.state, leg_name, side,
                                                          ctx.crosswalk_offsets, ctx.props):
                band = offset_band_polygon(leg, side, inner_off, outer_off,
                                           max(run_start_ft, start_ft), run_end_ft)
                open_runs = ctx.open_runs(leg_name, side, STALL_DIVIDER, band) if band else []
                for lo, hi in stall_lane_runs_ft(open_runs, PARKING_STALL_LENGTH_DEFAULT_FT,
                                                  keep_inside_ft=MIN_LINE_LENGTH_FT):
                    for divider in parking_stall_lines_ft(
                            leg, side, lane.parking_ft, PARKING_STALL_LENGTH_DEFAULT_FT, lo, hi,
                            curb_offset_ft=lane.shy_ft):
                        ctx.add(STALL_DIVIDER, divider, leg_name, side)
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
            # in mid-air where the crossing cut them. See PaintContext.rim.
            # Now a CONSTANT shy_ft against the kerb rather than whatever the street had spare,
            # because the lane's outer edge follows the kerb instead of standing off the narrowest
            # point. That is the whole visible fix: this zone used to be the wedge, 0.87 ft of
            # hatching at one end of W Broad's lane and 8.68 ft at the other. Where shy_ft is 0
            # there is nothing to hatch and the lane meets its own edge stripe at the kerb.
            # Where the lane hugs the kerb this is a CONSTANT shy_ft against it rather than
            # whatever the street had spare, which is the whole visible fix: it used to be the
            # wedge - 0.87 ft of hatching at one end of W Broad's lane and 8.68 ft at the other.
            # With shy_ft at 0 there is nothing to hatch and the lane meets its own edge stripe.
            hatch = (kerb_referenced_band_polygon(leg, side, 0.0, lane.shy_ft, start_ft)
                     if lane.shy_ft else None) if lane.hugs_kerb else _one(
                lane_narrowing_polygons_ft(leg, leg.curb_to_curb_ft / 2 - bounds["outer_ft"],
                                            start_left_ft=start_ft, start_right_ft=start_ft,
                                            sides=(side,)))
            # WHICH BUFFER IT IS. On a kerb with nothing outside the lane this leftover is the
            # bikeway's own separation from the kerb, so it is drawn in the BIKE buffer's channel
            # and reads as one protected corridor - lane with separation either side - instead of
            # a lane that stops short of the kerb beside an unexplained hatch. Where the section
            # puts parking out there, or the leg carries MarkedParking on this kerb, the leftover
            # belongs to THAT marking and keeps the parking buffer's channel; routing it to the
            # bike buffer painted 100 sq ft of broad_st_west's parking buffer twice, which
            # markings_collide reported.
            from src.geometry.targets import LegSide
            from src.geometry.treatments.parking import MarkedParking

            kerbside_is_the_bikeway_s = (
                not lane.parking_ft
                and ctx.state.treatment_for(MarkedParking, LegSide(leg_name, side)) is None)
            kind, edge = ((BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE) if kerbside_is_the_bikeway_s
                          else (BUFFER_FILL, BUFFER_EDGE_LINE))
            ctx.rim(ctx.add(kind, hatch, leg_name, side, beyond_ft,
                             shares_a_kerb=through), edge)


@dataclass(frozen=True)
class AddTwoWayBikeLane(AddBikeLane):
    """A bidirectional bike lane along ONE side of a leg, with the travel lanes shifted off it.

    Subclasses AddBikeLane because everything about painting the section is the same problem -
    two edge stripes, a hatched buffer, the green surface, the dotted extension across every
    driveway, all cut around the crossing bands. Only two things differ, and both are additions
    rather than changes: the section starts further out (TwoWayBikeLane resolves that), and the
    lane carries a yellow centre stripe because it holds opposing riders.

    THE SIDE IS A CORRIDOR DECISION, NOT A PER-JUNCTION ONE. A two-way lane that changes sides
    mid-corridor makes riders cross the street to stay on it, so the side is chosen once for the
    whole route from how many streets cut each kerb - see sites/*/scenarios.py for Broad St's.
    """
    paint_group: ClassVar[int] = 30
    paint_rank: ClassVar[int] = 0
    constrained: bool = False

    def __post_init__(self):
        """The width floor, checked without a street.

        NOT by constructing a TwoWayBikeLane - that also checks the fit against two half-widths
        this treatment does not carry, and feeding it invented ones to get at the width check
        would be a validation passing on made-up geometry. A 6 ft two-way lane is wrong before
        any street is consulted, so that part is checked here and the fit is checked in apply_to
        where the real half-widths exist.
        """
        floor = (CONSTRAINED_TWO_WAY_BIKE_LANE_FT if self.constrained
                 else MIN_TWO_WAY_BIKE_LANE_FT)
        if self.width_ft < floor:
            raise ValueError(
                f"A {self.width_ft:.2f} ft two-way bike lane is under NACTO's {floor:.0f} ft "
                f"{'constrained-conditions' if self.constrained else 'minimum'} width "
                f"({TWO_WAY_BIKE_LANE_WIDTH_FT:.0f} ft is the width to design to).")

    def section(self, state: "DesignState") -> TwoWayBikeLane:
        """The section as this leg's own kerbs make it - measured at the NARROWEST traced point on
        each side, which is where a promise about the whole kerb has to hold.

        Raises through TwoWayBikeLane when the leg cannot hold two travel lanes beside it.
        """
        leg = state.legs[self.target.leg]
        side = Side(str(self.target.side))
        return TwoWayBikeLane(
            width_ft=self.width_ft, buffer_ft=self.buffer_ft, constrained=self.constrained,
            near_half_ft=narrowest_half_width_ft(leg, str(side)),
            far_half_ft=narrowest_half_width_ft(leg, str(side.other)))

    def describe(self) -> str:
        return (f"AddTwoWayBikeLane({self.target.leg}, {self.target.side}): "
                f"{_feet(self.width_ft)} ft two-way lane"
                + (f", {_feet(self.buffer_ft)} ft buffer" if self.buffer_ft else ""))

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        leg = state.legs[self.target.leg]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {self.target.leg!r} has no width - nothing to fit a lane into.")
        # Building the section IS the fit check: TwoWayBikeLane refuses one that leaves less than
        # two travel lanes, and it does so carrying the measurement. Reraised untouched.
        section = self.section(state)
        shift_ft = travel_lane_divider_shift_ft(section)
        surplus_ft = far_kerb_surplus_ft(section)
        other = Side(str(self.target.side)).other
        # The ACTUAL lane width, not half the travel way. Those are the same number only on a leg
        # too narrow to hold the target, and reporting the equal-split figure on a leg that holds
        # 11 ft lanes plus a stall's worth of surplus described a design nobody drew.
        lane_ft = (TARGET_LANE_WIDTH_FT if surplus_ft >= 0
                   else (section.near_half_ft + section.far_half_ft - section.section_ft) / 2)
        note = (f". Spends {section.section_ft:.2f} ft of this leg's "
                f"{section.near_half_ft + section.far_half_ft:.2f} ft between kerbs, leaving two "
                f"{lane_ft:.2f} ft travel lanes with the centreline shifted {shift_ft:.2f} ft "
                f"toward the {other} kerb")
        if surplus_ft >= 0:
            note += f", and {surplus_ft:.2f} ft spare against that kerb"
        else:
            note += (f" - under the {TARGET_LANE_WIDTH_FT:.0f} ft target by "
                     f"{TARGET_LANE_WIDTH_FT - lane_ft:.2f} ft, which is this leg's width rather "
                     f"than a choice, so the travel way is split equally")
        return (note + ". The NJDOT alignment does not move; every station and crossing frame is "
                "measured from it as before. " + NJDOT_TWO_WAY_OBJECTION)

    def paint(self, ctx) -> None:
        """The one-way section's markings, plus the yellow stripe down the middle of the lane."""
        from src.geometry.markings import BIKE_CONTRAFLOW_DIVIDER
        from src.geometry.model import inset_line_ft, kerb_parallel_line_ft

        # Everything AddBikeLane paints, at this section's own offsets. Reached through the
        # resolved lane, so the stripes land where the shifted section actually is.
        section = self.section(ctx.state)
        super().paint(ctx)

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        bounds = section.offsets_from_centerline_ft()
        centre_ft = (bounds["bike_inner_ft"] + bounds["bike_outer_ft"]) / 2
        # Measured from the KERB, like the lane's own two edges - see BikeLane.offsets_from_kerb_ft.
        # Left on the alignment it stayed put while the lane moved onto the kerb, and on
        # broad_st_east's right kerb the divider ended up running along the lane's edge stripe for
        # 1.2 ft, which MarkingsDoNotCollide reported and was right to: a lane's centre stripe that
        # is not down the lane's centre is not a centre stripe.
        from_kerb = section.offsets_from_kerb_ft()
        centre_from_kerb_ft = (from_kerb["bike_inner_ft"] + from_kerb["bike_outer_ft"]) / 2
        at = ctx.anchors(leg_name, side, inner_offset_ft=centre_ft)
        through = (leg_name, side) in ctx.straight_through
        if through:
            # BEHIND THE NODE, not up to it. Each leg's paint is built in its own frame, so two
            # halves that both stop at their own station 0 stop just shy of each other - 1.28 ft
            # of hole at W Broad & Louellen, in the middle of a lane whose whole point is running
            # continuously through the junction. Starting behind it makes the two overlap, and the
            # overlap is deduped by shares_a_kerb below, which is the same mechanism that keeps two
            # zones on one through kerb from double-painting. Honoured only as far as the kerb is
            # really traced there - see model.paint_stations.
            start_ft, beyond_ft = -THROUGH_JUNCTION_OVERLAP_FT, None
        elif leg_name in ctx.marked:
            from src.geometry.paint import end_against_crossing
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        # BROKEN, not continuous - passing is permitted in a two-way bikeway where sight
        # distance allows, and MUTCD's yellow-broken is what says so. Cut into dashes here
        # rather than left to a line style, for the reason every other dashed marking in this
        # project is: a style is a 2D property and the 3D render gets geometry, so a continuous
        # line with a dashed style renders solid.
        axis = (kerb_parallel_line_ft(leg, side, centre_from_kerb_ft, start_ft,
                                       floor_ft=centre_ft)
                 if section.hugs_kerb else inset_line_ft(leg, side, centre_ft, start_ft))
        if axis is None or axis.is_empty:
            return
        # AND IT CARRIES THROUGH EVERY DRIVEWAY, like the lane's other markings.
        #
        # MUTCD 11th ed. §9E.04 Option 02 permits a bicycle lane to be continued through a
        # driveway with solid or dotted longitudinal lines, and §9E.06 Guidance 15 says lane
        # extension markings SHOULD be used to extend a buffer-separated bicycle lane across
        # intersections and driveways. NACTO's Urban Bikeway Design Guide is more specific for
        # this facility: contraflow and bidirectional protected lanes must continue through
        # intersections and driveways, with a DOTTED YELLOW CENTRELINE along the lane and through
        # the crossings. See STANDARDS.md §4.
        #
        # This stripe used to simply stop at each driveway - 22 dashes on a kerb with two of them
        # against 30 on a kerb with none - while the edge lines continued dotted and the green
        # carried across. Three answers to one conflict point, and this was the one that belonged
        # to nobody: it was an omission, not a design.
        #
        # Its row in markings.AT_AN_OPENING says CARRIED, so `add` does not cut it at an entrance
        # at all and the cadence cannot break phase across one. Being already a broken line, it
        # needs no separate dotted pattern - the standard's "dotted extension" is what it already
        # looks like. It used to be cut and the part inside re-laid as an exact complement, which
        # is the same statement made twice and in two places.
        period_ft = CONTRAFLOW_DASH_FT + CONTRAFLOW_GAP_FT
        at_ft = 0.0
        while at_ft + CONTRAFLOW_DASH_FT <= axis.length:
            dash = shapely.ops.substring(axis, at_ft, at_ft + CONTRAFLOW_DASH_FT)
            if dash.geom_type == "LineString" and dash.length > 0:
                ctx.add(BIKE_CONTRAFLOW_DIVIDER, dash, leg_name, side, beyond_ft)
            at_ft += period_ft
