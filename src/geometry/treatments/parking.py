"""Kerbside PARKING: marked stalls, the buffer that protects them, and what the borough's
own restrictions do to both.

apply_osm_parking lives here rather than in a source module on purpose - it is a treatment
builder, not a reader: it turns OSM's parking:*  tags into MarkedParking and LaneNarrowing."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from src.geometry.targets import LegSide, LegTarget, Side
from src.geometry.model import half_width_profile, narrowest_half_width_ft
from src.geometry.treatments.base import (BOLLARD_DEFAULT_SPACING_FT,
                                          LANE_NARROWING_DEFAULT_STRIPE_FT,
                                          LANE_WIDTH_SLACK_FT, MIN_MARKED_PARKING_DEPTH_FT,
                                          PARKING_STALL_DEPTH_DEFAULT_FT,
                                          PARKING_STALL_LENGTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT, Treatment,
                                          kerbside_allowance_ft)
from src.geometry.treatments.bikeways import AddBikeLane, divider_shift_toward_ft
from src.geometry.treatments.lanes import LaneNarrowing
from src.geometry.treatments.state import DesignState, FacilityRefusal
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

                                  # an intersection - a real legal minimum, not a rendering choice. Marked parking
                                  # (src/render/export.py/plan_view.py) starts at max(this distance past the real
                                  # crosswalk, leg_clearance_ft's physical past-the-corner-curve point) - whichever
                                  # is farther from the intersection - so it never starts somewhere a car legally
                                  # couldn't park even if the curb geometry alone would allow it.


@dataclass(frozen=True)
class MarkedParking(Treatment):
    """Marked curbside parallel parking along one side of a leg: a lane-edge
    line depth_ft in from the curb, plus perpendicular divider ticks every
    stall_length_ft (src/geometry/model/stripes.py:parking_lane_edge_line_ft /
    parking_stall_lines_ft) - paint-only, zero curb/pavement change, same
    convention as LaneNarrowing/CornerHatching in that regard.
    Independent of LaneNarrowing - a leg can have marked parking with or
    without a separate travel-lane-narrowing buffer on the same or other
    side; nothing here assumes the two are combined, though a scenario is
    free to apply both (e.g. narrow the near lane while marking parking in
    what the SLD calls the far side's shoulder zone).

    curb_offset_ft > 0 pulls the parking lane in from the curb by that much,
    leaving a striped no-parking buffer between the curb and the parking
    lane itself (so parking sits directly against the active travel lane
    instead of against the curb) - src/geometry/paint/ paints that buffer with the
    same chevron treatment as a lane narrowing. 0 (the default) means the
    parking lane starts right at the curb, no buffer.
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 20
    paint_rank: ClassVar[int] = 0
    depth_ft: float = PARKING_STALL_DEPTH_DEFAULT_FT
    stall_length_ft: float = PARKING_STALL_LENGTH_DEFAULT_FT
    curb_offset_ft: float = 0.0
    # Where the room this depth was sized on runs out - see LaneNarrowing.end_ft, the same idea
    # for the other kerbside treatment. None draws every run/zone to wherever it would otherwise
    # end (a crossing, the leg's own end); a station caps every run AND every daylight zone at
    # that point, because both are stalls/hatching sized off ONE depth_ft and neither may run
    # past the stretch of kerb that depth was measured to fit.
    end_ft: float | None = None

    def __post_init__(self):
        # None of this was checked before. A zero-depth lane marked an edge line on top of the
        # kerb, and a stall shorter than a car claimed spaces that cannot exist.
        if self.depth_ft <= 0:
            raise ValueError(f"A parking lane needs a depth; got depth_ft={self.depth_ft}.")
        if self.stall_length_ft <= 0:
            raise ValueError(f"A stall needs a length; got stall_length_ft={self.stall_length_ft}.")
        if self.curb_offset_ft < 0:
            raise ValueError(f"A kerb buffer cannot be negative; got {self.curb_offset_ft}.")

    def describe(self) -> str:
        end = f", end_ft={self.end_ft:.1f}" if self.end_ft is not None else ""
        return (f"MarkedParking({self.target.leg}, side={str(self.target.side)!r}, "
                f"depth_ft={self.depth_ft}, stall_length_ft={self.stall_length_ft}, "
                f"curb_offset_ft={self.curb_offset_ft}{end})")

    def paint(self, ctx) -> None:
        """The stalls, the hatched buffer between them and the kerb, and the daylight zones
        where the law forbids parking at all."""
        from src.geometry.daylighting import merged_no_parking_spans_ft, no_parking_zones_ft
        from src.geometry.markings import (BUFFER_EDGE_LINE, BUFFER_FILL, DAYLIGHT_EDGE_LINE,
                                           DAYLIGHT_FILL, PARKING_EDGE_LINE, STALL_DIVIDER,
                                           ZONE_END_LINE)
        from src.geometry.model import (inset_line_ft, lane_narrowing_polygons_ft,
                                        offset_band_polygon, parking_lane_edge_line_ft,
                                        parking_stall_lines_ft, stall_lane_runs_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, MIN_LINE_LENGTH_FT, _one,
                                        end_against_crossing, lane_edge_stripes, parking_runs,
                                        zone_end_line_ft)

        leg_name, side = self.target.leg, str(self.target.side)
        state = ctx.state
        leg = state.legs[leg_name]
        depth_ft, stall_length_ft = self.depth_ft, self.stall_length_ft
        curb_offset_ft = self.curb_offset_ft
        at = ctx.anchors(leg_name, side,
                          inner_offset_ft=leg.curb_to_curb_ft / 2 - depth_ft - curb_offset_ft)
        runs = parking_runs(state, leg_name, side, ctx.crosswalk_offsets, ctx.props)
        if self.end_ft is not None:
            runs = [(s, min(e, self.end_ft)) for s, e in runs if s < self.end_ft]

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
        daylight_spans = merged_no_parking_spans_ft(
            no_parking_zones_ft(state, leg_name, side, ctx.crosswalk_offsets, ctx.props))
        for zone_start_ft, zone_end_ft in daylight_spans:
            # Capped, not filtered out early with the runs above: a daylight zone is the statute
            # restated in paint (see the comment on beyond_the_tracing below) and still applies
            # up to wherever this depth_ft was actually sized to reach, even though the zone
            # itself may run further in law.
            capped = self.end_ft is not None and zone_end_ft > self.end_ft
            if self.end_ft is not None:
                if zone_start_ft >= self.end_ft:
                    continue
                zone_end_ft = min(zone_end_ft, self.end_ft)
            if leg_name in ctx.marked and (leg_name, side) in ctx.straight_through:
                start_ft, beyond_ft = zone_start_ft, None
            elif leg_name in ctx.marked:
                start_ft, beyond_ft = end_against_crossing(at, zone_start_ft)
            else:
                start_ft, beyond_ft = max(zone_start_ft, at.target_ft), None
            # A solid line wherever hatching meets the travel lane, so the lane reads as a
            # lane. The buffer beside the stalls already has one; the daylight zone runs the
            # full depth of the parking lane, so ITS inner edge is the lane edge, and without
            # this the hatching just faded into the carriageway.
            #
            # BEFORE the fill, so that the rim - which is this same line continued around the
            # zone's cut end - can be trimmed against it. Painted after, the two overlapped by
            # 3.3 ft where the fillet leaves the lane edge tangentially, which is exactly where
            # they are supposed to meet; MarkingsDoNotCollide reported it.
            # PAST THE END OF THE TRACING, which nothing else here may do. A daylight zone is
            # not a design choice about this kerb, it is R.S. 39:4-138 restated in paint, and the
            # statute does not stop where OSM's kerb tracing does. W Broad & Louellen's south
            # kerb is traced only from station 60.3 against a statutory zone of 0-93.3, so the
            # hatching was drawn over the last third of the zone and stopped 7.5 ft short of the
            # crosswalk it exists to daylight - and hatching that stops short of a crossing
            # undoes the treatment, because the bare stretch beside the crossing is exactly where
            # a car parks and blocks the sight line. See leg_frame.paint_stations for what is
            # assumed outside the tracing (the kerb held at its first traced offset) and why the
            # stalls and buffers below deliberately do NOT ask for the same.
            ctx.add(DAYLIGHT_EDGE_LINE,
                     inset_line_ft(leg, side, lane_edge_offset_ft, start_ft, zone_end_ft,
                                    keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2,
                                    beyond_the_tracing=True),
                     leg_name, side, beyond_ft)
            ctx.rim(ctx.add(DAYLIGHT_FILL, _one(lane_narrowing_polygons_ft(
                leg, daylight_fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=zone_end_ft, beyond_the_tracing=True)),
                leg_name, side, beyond_ft,
                shares_a_kerb=(leg_name, side) in ctx.straight_through), DAYLIGHT_EDGE_LINE)
            # Nothing to end against and no taper available: close the square end. See
            # zone_end_line_ft. Not where the kerb runs straight through - the zone carries
            # on into the next leg there.
            if leg_name not in ctx.marked and (leg_name, side) not in ctx.straight_through:
                ctx.add(ZONE_END_LINE, zone_end_line_ft(
                    leg, side, start_ft, leg.curb_to_curb_ft / 2 - daylight_fill_ft),
                    leg_name, side)
            if capped:
                # The far end above was cut mid-zone by self.end_ft, which the crossing/opening
                # cuts ctx.rim knows about is not one of - so without this the hatch just stops,
                # no line, same failure LaneNarrowing.end_ft's own closing line exists to avoid.
                ctx.add(ZONE_END_LINE, zone_end_line_ft(
                    leg, side, zone_end_ft, leg.curb_to_curb_ft / 2 - daylight_fill_ft),
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
            ctx.add(PARKING_EDGE_LINE, edge, leg_name, side)

            # A STALL, UNLIKE THE EDGE LINE ABOVE, MUST NOT BE DRAWN WHERE IT WILL BE CUT.
            # PARKING_EDGE_LINE is CARRIED across a driveway (real curbside parking keeps its
            # edge line straight through one), but STALL_DIVIDER is STOPPED at every opening -
            # a car cannot be told to park across a driveway mouth. Laying the divider grid over
            # the LEGAL run and letting ctx.add cut it afterwards is what left a stall's far tick
            # open-ended at the edge line, a driveway's dropped tick fusing its neighbour into a
            # 44 ft "stall", and a stall straddling a driveway outright - see
            # model.stall_lane_runs_ft. So the grid is built over the ground STALL_DIVIDER will
            # actually keep: the same footprint the dividers themselves occupy, cut by the same
            # openings/crossings/surfaces ctx.add cuts against, trimmed back to whole stalls.
            #
            # NOT beyond_the_tracing: a stall proposes paint on the physical kerb, not a
            # statement of law like the daylight zone above, so it may reach no further than
            # the kerb is actually surveyed - paint_stations' own distinction (leg_frame.py).
            # parking_stall_lines_ft enforces the same curb_station_span bound when it places
            # each tick's offset; asking ctx.open_runs to look past it here just meant this
            # run and that clamp disagreed about where the ground ends, which is the second
            # source of truth that left the run's closing tick clipped off again.
            stall_curb_offset_ft = curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT
            half = leg.curb_to_curb_ft / 2
            outer_off = max(half - stall_curb_offset_ft, 0.5)
            inner_off = max(half - stall_curb_offset_ft - depth_ft, 0.5)
            band = offset_band_polygon(leg, side, inner_off, outer_off, start_ft, end_ft)
            open_runs = ctx.open_runs(leg_name, side, STALL_DIVIDER, band) if band else []
            for lo, hi in stall_lane_runs_ft(open_runs, stall_length_ft,
                                              keep_inside_ft=MIN_LINE_LENGTH_FT):
                for divider in parking_stall_lines_ft(
                        leg, side, depth_ft, stall_length_ft, lo, hi,
                        curb_offset_ft=stall_curb_offset_ft):
                    ctx.add(STALL_DIVIDER, divider, leg_name, side)

            if not curb_offset_ft:
                continue
            buffer_ft = max(curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0)
            ctx.add(BUFFER_EDGE_LINE, inset_line_ft(
                leg, side, leg.curb_to_curb_ft / 2 - buffer_ft, start_ft, end_ft,
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2), leg_name, side)
            ctx.add(BUFFER_FILL, _one(lane_narrowing_polygons_ft(
                leg, buffer_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=end_ft)), leg_name, side)


@dataclass(frozen=True)
class ParkingBufferBollards(Treatment):
    """Plastic bollards (flex-post delineators) centered in the striped
    no-parking buffer between a marked-parking lane and the curb - i.e. on
    the OUTSIDE of the parking lane (the curb side), protecting/delineating
    parked cars from that buffer, the mirror image of LaneNarrowingBollards (which
    centers bollards in a lane-narrowing buffer on the travel-lane side).
    Requires MarkedParking to already be applied to this (leg, side) with
    curb_offset_ft > 0 - there's no buffer to put bollards in otherwise."""
    paint_group: ClassVar[int] = 20
    paint_rank: ClassVar[int] = 1
    spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT

    def __post_init__(self):
        if self.spacing_ft <= 0:
            raise ValueError(f"Posts need a spacing; got spacing_ft={self.spacing_ft}.")

    def describe(self) -> str:
        return (f"ParkingBufferBollards({self.target.leg}, "
                f"side={str(self.target.side)!r}, spacing_ft={self.spacing_ft})")

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> None:
        parking = state.treatment_for(MarkedParking, self.target)
        if parking is None:
            raise KeyError(f"{self.target} has no marked parking - apply MarkedParking first.")
        if not parking.curb_offset_ft:
            raise ValueError(f"{self.target}'s marked parking has curb_offset_ft=0 - no curb "
                              f"buffer to put bollards in.")

    def paint(self, ctx) -> None:
        """Down the buffer between the stalls and the kerb, over the runs where stalls are marked.

        The buffer's width belongs to the MarkedParking treatment underneath, so it is read from
        the design rather than restated here - the same reason this treatment refuses a parking
        lane with no buffer.
        """
        from src.geometry.markings import BOLLARD
        from src.geometry.model import bollard_points_ft
        from src.geometry.paint import PaintPiece, _dot, parking_runs

        leg_name, side = self.target.leg, str(self.target.side)
        curb_offset_ft = ctx.state.treatment_for(MarkedParking, self.target).curb_offset_ft
        leg = ctx.state.legs[leg_name]
        for start_ft, _end_ft in parking_runs(ctx.state, leg_name, side, ctx.crosswalk_offsets,
                                               ctx.props):
            for point in bollard_points_ft(leg, curb_offset_ft, start_ft, self.spacing_ft,
                                            sides=(side,)):
                ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, side))


def _kerb_already_treated(state: DesignState, leg_name: str, side: str) -> bool:
    """Has a scenario already decided what happens on this kerb?

    Asked of the treatments rather than of the dicts they write, so it is a question about
    decisions someone made: apply_osm_parking fills in what OSM says about kerbs a proposal
    has not spoken for, and it must not paint over one that it has.

    Takes the state explicitly rather than closing over the caller's loop variable. The state
    is rebound on every iteration there, so a closure would read whatever it happened to be at
    call time - correct today only because the call is in the same iteration, and silently
    wrong the moment it isn't.
    """
    if state.treatment_for(MarkedParking, LegSide(leg_name, side)) is not None:
        return True
    narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
    return narrowing is not None and Side(side) in narrowing.sides


def apply_osm_parking(state: DesignState, model: "IntersectionModel", depth_ft: float = PARKING_STALL_DEPTH_DEFAULT_FT,
                       stripe_width_ft: float = LANE_NARROWING_DEFAULT_STRIPE_FT,
                       legs: tuple | None = None) -> DesignState:
    """Paint each kerb according to what OSM says about parking there.

    `legs` limits it to the legs named, leaving the rest of the junction bare. Not a
    rendering convenience - a scenario that treats two legs of a crossroads and not the
    other two is a real proposal, and Columbia Ave is one (see
    sites/columbia_princeton/scenarios.py).

    Restricted (parking:*:restriction = no_parking / no_standing / no_stopping) gets crossed
    hatching - that kerb cannot hold parked cars, and a proposal that drew stalls there
    would be proposing something illegal. Everything else gets marked stalls: both an
    explicit restriction=none, which is a positive statement that parking is allowed, and an
    untagged side, which is the ordinary residential-street default.

    A RESTRICTION OVER PART OF A KERB gets both. OSM records a restriction that changes part way
    along a street by splitting the way, which is how "no parking for the first 100 ft from the
    junction" is expressed - so a kerb can be restricted near the corner and open beyond it. Such
    a kerb is marked for parking here, and the restricted stretch is carved back out of it by
    src/geometry/daylighting.py, which treats a mapped prohibition as a no-parking zone exactly
    like a statutory one: the stretch gets hatched and no stall is marked inside it. Only a kerb
    restricted along its WHOLE length is hatched end to end.

    That distinction is the reason this reads state.parking_restrictions rather than one way's
    tags. It used to take the tags of the single way nearest the leg's midpoint, so at Broad &
    Greenwood a no_parking restriction covering East Broad's first 79.5 ft was dropped in favour
    of the unrestricted way beyond it, and the render marked stalls where the mapper had just
    said there is none.

    "Unless otherwise specified": a side the scenario has ALREADY treated is left alone, so
    this can be applied as a baseline and then overridden per side.

    Side mapping goes through parking_restriction_by_side per span, which flips OSM's left/right
    for ways that run against the leg - without which half these kerbs would have the restriction
    painted on the wrong side.
    """
    new_state = state
    for leg_name in sorted(state.legs):
        if legs is not None and leg_name not in legs:
            continue
        leg = state.legs[leg_name]
        leg_length_ft = leg.centerline.length
        sides = {side: restriction_summary(state, leg_name, side, leg_length_ft)
                 for side in ("left", "right")}

        untouched = [s for s in ("left", "right")
                     if not _kerb_already_treated(new_state, leg_name, s)]

        # Two questions here, and one number was answering both.
        #
        # WHERE THE PAINT GOES is an offset from the nominal half-width, because that is the
        # datum MarkedParking and LaneNarrowing express themselves in: each subtracts its own
        # widths from `curb_to_curb_ft / 2`, so both land their inner edge on
        # TARGET_LANE_WIDTH_FT whatever the kerb does, and their outer edge is the traced kerb
        # itself (curbside_strip_polygon). That is a coordinate, not a measurement, and it is
        # named for what it is rather than borrowing the word "allowance".
        #
        # WHETHER THERE IS ROOM is a measurement of the kerb, per side - kerbside_allowance_ft.
        # Answering it with the nominal figure is what marked 8 ft stalls on a kerb with 5 ft
        # behind the lane edge and drew them clipped to 4.6 ft.
        half_ft = leg.curb_to_curb_ft / 2
        lane_edge_from_nominal_ft = half_ft - TARGET_LANE_WIDTH_FT
        room_ft = {side: kerbside_allowance_ft(leg, side) for side in ("left", "right")}
        if not untouched or max(room_ft[s] for s in untouched) <= 0:
            if untouched:
                print(f"  NOTE: {leg_name} is {leg.curb_to_curb_ft:.1f} ft curb to curb - too narrow "
                      f"for two {TARGET_LANE_WIDTH_FT:.0f} ft lanes, so no kerbside paint is marked "
                      f"here. Its lanes are {half_ft:.1f} ft as they stand.")
            continue

        # Hatched end to end only where the restriction covers the whole kerb. A kerb restricted
        # over PART of its length is marked for parking, and daylighting carves the restricted
        # stretch back out - see the docstring.
        restricted = [s for s in untouched
                      if sides[s].restricted_throughout
                      or (sides[s].restricted_in_part and sides[s].holds_no_stall)]
        # A standard stall or nothing: an unrestricted kerb with less than one stall's worth
        # of room gets its spare width HATCHED, not left bare. Leaving it bare was keeping
        # the lane at 18 ft on E Broad, which defeats the whole point of the target - and
        # hatching beside a travel lane reads as a buffer/shoulder, the same thing the strip
        # between a parking lane and the kerb already is, not as a parking prohibition.
        parkable = [s for s in untouched
                    if s not in restricted and room_ft[s] >= MIN_MARKED_PARKING_DEPTH_FT]
        hatched = [s for s in untouched if s not in restricted and s not in parkable]
        for side in hatched:
            print(f"  NOTE: {leg_name} {side} is unrestricted, but only {room_ft[side]:.1f} ft is "
                  f"spare beside a {TARGET_LANE_WIDTH_FT:.0f} ft lane at the kerb's narrowest - "
                  f"under one {MIN_MARKED_PARKING_DEPTH_FT:.0f} ft stall, so it is hatched as "
                  f"buffer rather than marked for parking.")
        restricted = restricted + hatched

        if restricted:
            new_state = new_state.apply(LaneNarrowing(LegTarget(leg_name),
                                                       stripe_width_ft=lane_edge_from_nominal_ft,
                                                       sides=tuple(restricted)))
            for side in restricted:
                why = (sides[side].describe() if sides[side].prohibited_ft
                       else "too narrow for a stall")
                new_state.notes.append(f"apply_osm_parking({leg_name}, {side}): "
                                        f"{room_ft[side]:.1f} ft hatched - {why}")
        for side in parkable:
            # The stall is a fixed standard depth and the leftover between it and the kerb is
            # hatched (add_marked_parking's curb_offset_ft draws that buffer with the same
            # geometry a lane-narrowing buffer uses). Handing the whole allowance to depth_ft
            # instead produced 10-12 ft "parking spaces", which is a stall plus a strip of
            # unmarked asphalt drawn as though you could park on it.
            buffer_ft = lane_edge_from_nominal_ft - PARKING_STALL_DEPTH_DEFAULT_FT
            new_state = new_state.apply(
                MarkedParking(LegSide(leg_name, side),
                               depth_ft=PARKING_STALL_DEPTH_DEFAULT_FT,
                               curb_offset_ft=buffer_ft))
            # What is REALLY hatched between the stall and the kerb, which is the nominal
            # buffer only where the nominal width is the real one.
            hatched_ft = room_ft[side] - PARKING_STALL_DEPTH_DEFAULT_FT
            extra = (f" + {hatched_ft:.1f} ft hatched to the kerb" if hatched_ft > 0.05 else "")
            new_state.notes.append(f"apply_osm_parking({leg_name}, {side}): "
                                    f"{PARKING_STALL_DEPTH_DEFAULT_FT:.0f} ft stalls{extra} - "
                                    f"{sides[side].describe()}")
    return new_state


@dataclass(frozen=True)
class RestrictionSummary:
    """What OSM says about one kerb, reduced to what a treatment and a label need.

    A kerb can now be restricted over part of its length, so "is this side restricted" is no
    longer a yes/no. Three cases have to be told apart, because they lead to three different
    markings and three different sentences on the drawing:

      restricted THROUGHOUT   hatch it end to end; no stall anywhere
      restricted IN PART      mark parking, and let daylighting carve the restricted stretch out
      not restricted          mark parking (whether tagged "none" or not tagged at all)
    """
    prohibited_ft: float           # how much of the kerb OSM forbids parking on
    kerb_length_ft: float
    worst_value: str | None        # the prohibition itself, e.g. "no_parking"
    stated_ft: float               # how much of the kerb OSM says ANYTHING about
    spans: tuple = ()              # the ParkingRestrictions behind it, in station order

    @property
    def open_ft(self) -> float:
        """Kerb OSM does not forbid parking on. Untagged counts as open, which is the same
        ordinary-street default apply_osm_parking has always applied to an untagged side."""
        return max(self.kerb_length_ft - self.prohibited_ft, 0.0)

    @property
    def restricted_throughout(self) -> bool:
        return self.prohibited_ft >= self.kerb_length_ft - RESTRICTION_COVERAGE_SLACK_FT

    @property
    def restricted_in_part(self) -> bool:
        return not self.restricted_throughout and self.prohibited_ft > RESTRICTION_COVERAGE_SLACK_FT

    @property
    def holds_no_stall(self) -> bool:
        """Whether what OSM leaves open is too short to park one car in.

        The same "a standard stall or nothing" rule MIN_MARKED_PARKING_DEPTH_FT applies across the
        road, applied along it. e_broad_st_west is tagged no_stopping over its first 114.5 ft and
        open for the last 15.5, and 15.5 ft is not a parking space - so marking that kerb for
        parking would claim a stall that cannot exist, and it is hatched end to end instead.
        """
        return self.open_ft < PARKING_STALL_LENGTH_DEFAULT_FT

    def describe(self) -> str:
        """One clause naming what OSM says, for a note or a plan-view label."""
        if self.restricted_throughout:
            return f"OSM says {self.worst_value!r} for the whole kerb"
        if self.restricted_in_part and self.holds_no_stall:
            return (f"OSM says {self.worst_value!r} for all but {self.open_ft:.0f} ft, under one "
                    f"{PARKING_STALL_LENGTH_DEFAULT_FT:.0f} ft stall")
        if self.restricted_in_part:
            stretches = ", ".join(f"{r.start_ft:.0f}-{r.end_ft:.0f} ft" for r in self.spans
                                  if r.prohibits)
            return f"OSM says {self.worst_value!r} over {stretches}"
        if self.stated_ft <= RESTRICTION_COVERAGE_SLACK_FT:
            return "no restriction tagged"
        return "restriction=none"


# How much of a kerb may be untagged before "restricted throughout" stops being true. A way's
# ends land a foot or two off the leg's own, and OSM splits are not surveyed to the inch.
RESTRICTION_COVERAGE_SLACK_FT = 2.0


def restriction_summary(state: DesignState, leg_name: str, side: str,
                          kerb_length_ft: float) -> RestrictionSummary:
    """Reduce this kerb's ParkingRestriction spans to a RestrictionSummary.

    Spans are clipped to the leg and merged, so a way that runs 900 ft down the block counts
    only for the part of it that is on this leg, and two ways meeting at a split do not
    double-count the foot they share.
    """
    spans = tuple(state.parking_restrictions.get((leg_name, side), []))
    prohibited, stated = [], []
    worst = None
    for restriction in spans:
        lo = max(restriction.start_ft, 0.0)
        hi = min(restriction.end_ft, kerb_length_ft)
        if hi <= lo:
            continue
        if restriction.value is not None:
            stated.append((lo, hi))
        if restriction.prohibits:
            prohibited.append((lo, hi))
            worst = worst or restriction.value
    return RestrictionSummary(prohibited_ft=_merged_length_ft(prohibited),
                               kerb_length_ft=kerb_length_ft, worst_value=worst,
                               stated_ft=_merged_length_ft(stated), spans=spans)


def _merged_length_ft(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by possibly-overlapping (start, end) intervals."""
    total, reach = 0.0, None
    for lo, hi in sorted(intervals):
        if reach is None or lo > reach:
            total += hi - lo
            reach = hi
        elif hi > reach:
            total += hi - reach
            reach = hi
    return total


# The narrowest parallel stall worth marking. Below 7 ft a car cannot sit clear of the travel
# lane, so it is not a stall; MIN_MARKED_PARKING_DEPTH_FT (8 ft) is the width to mark when the
# street can spare it, not the floor for whether parking exists at all.
MIN_USABLE_STALL_FT = 7.0
#: The narrowest hatched zone worth painting - and, more to the point, the narrowest one that can
#: be painted WITHOUT TAKING FROM THE TRAVEL LANE. Both callers below used to floor the stripe at
#: this figure (`max(spare, 0.5)`), which is the opposite rule: where the street spared 0.37 ft it
#: painted 0.5 and the missing 0.13 came out of the lane. At E Broad & Princeton that turned a
#: two-way bikeway that FIT - 11.00 ft lanes, 0.14 ft spare - into three invariant failures and a
#: refused 3D export, over a sliver of asphalt nobody would stripe.
#:
#: So it is a FLOOR ON WHETHER, not on how wide: below this the spare is left unpainted, which is
#: what it is on the ground.
MIN_HATCHED_ZONE_FT = 0.5


def lane_surplus_that_cannot_be_striped_ft() -> float:
    """How far over TARGET_LANE_WIDTH_FT a travel lane may be left, and why it is not a tolerance.

    A surplus is taken off a lane by painting a zone beside it, so a surplus narrower than the
    narrowest paintable zone cannot be taken off at all - see MIN_HATCHED_ZONE_FT. That is a fact
    about paint, not slack for float noise, and it is bigger than the noise allowance by an order
    of magnitude.

    ONE HOME because it was briefly two: checks.TravelLanesHoldTheTarget and
    tests/test_two_way_bike_lane.py each wrote `max(<their own tolerance>, MIN_HATCHED_ZONE_FT)`,
    and the test's copy hardcoded 0.05 rather than importing it. Two copies of one rule is the
    defect this repo has the most history with.
    """
    from src.geometry.treatments.base import LANE_WIDTH_SLACK_FT

    return max(LANE_WIDTH_SLACK_FT, MIN_HATCHED_ZONE_FT)


def osm_derived_baseline(state: DesignState, model: "IntersectionModel", legs: tuple | None = None) -> DesignState:
    """Paint every kerb the way OSM says it is used, complete the centrelines, upgrade the
    crossings - the design that proposes nothing.

    NOT A PROPOSAL, which is the point of it having one home. Every mark it makes is derived
    from surveyed data: hatching where OSM records a parking restriction, stalls where it does
    not, a centreline on a leg that has none today, continental paint on the crossings that
    already exist. There is no design choice in it to make per site, and it was the
    `build_demo_scenario` of four sites - byte-identical in three of them, and differing in the
    fourth only by which legs it touches.

    `legs` restricts the parking pass to some of them, which IS a per-site fact (Columbia &
    Princeton calms only its two Princeton Ave legs) and so stays a parameter rather than
    becoming a second copy of the function.

    Local import because crossings.py is a sibling and the package's __init__ imports both;
    reaching for it at module scope would order the two by accident.
    """
    from src.geometry.treatments.crossings import all_crosswalks_continental, complete_centerlines

    if model is None:
        return state
    state = apply_osm_parking(state, model, legs=legs) if legs else apply_osm_parking(state, model)
    state = complete_centerlines(state)
    return all_crosswalks_continental(state)


def narrow_lanes_and_recover_parking(state: DesignState) -> DesignState:
    """Narrow EVERY leg to TARGET_LANE_WIDTH_FT and put the recovered width to work as parking.

    The two treatments have to be sized together, not stacked: an 11 ft lane plus an 8 ft stall
    needs 19 ft per side, and a small-borough leg is commonly 33-39 ft curb-to-curb (16.5-19.5
    per side). So the recovered width - half the roadway minus the target lane - is what is
    available, and each leg gets whichever of these fits:

      * >= PARKING_STALL_DEPTH_DEFAULT_FT: a full 8 ft stall, remainder as a striped buffer
        between the stall and the kerb (MarkedParking's curb_offset_ft);
      * >= MIN_USABLE_STALL_FT: a single stall taking the whole recovered width;
      * less than that: no parking - paint-only narrowing, since a 6 ft stall is not a stall.
        The leg still gets its target lanes.

    Printed per leg so the trade-off is visible rather than buried in geometry.

    ONE DEFINITION, for the reason hold_travel_lane_at_target below gives at length and this
    function proves again: it existed as a byte-identical private `_parking_and_narrowing` in
    THREE site scenario files, each with its own local `PARKING_DEPTH_FT = 8.0` and
    `MIN_PARKING_DEPTH_FT = 7.0` shadowing the standards in this module. Three copies of a rule
    is three chances for one of them to drift, and a site is exactly where nobody looks for a
    standard. See tests/test_sites.py:test_no_site_redeclares_what_src_already_defines.

    THE SISTER RULE IS hold_travel_lane_at_target, and they are deliberately not merged: this
    one narrows a WHOLE JUNCTION symmetrically and is what a plain road-diet proposal wants;
    that one holds ONE KERB at the target and is what a bikeway proposal needs for the far side.
    """
    for leg_name, leg in state.legs.items():
        recovered_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        if recovered_ft < MIN_USABLE_STALL_FT:
            if recovered_ft < MIN_HATCHED_ZONE_FT:
                print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) recovers only "
                      f"{recovered_ft:.1f} ft per side at {TARGET_LANE_WIDTH_FT:.0f} ft lanes - "
                      f"under {MIN_HATCHED_ZONE_FT} ft, so nothing is painted here rather than a "
                      f"stripe the lane would have to pay for.")
                continue
            state = state.apply(LaneNarrowing(LegTarget(leg_name), recovered_ft))
            print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) recovers only "
                  f"{recovered_ft:.1f} ft per side at {TARGET_LANE_WIDTH_FT:.0f} ft lanes - too "
                  f"narrow for a stall, so paint-only narrowing here, no parking.")
            continue
        depth_ft = min(recovered_ft, PARKING_STALL_DEPTH_DEFAULT_FT)
        buffer_ft = max(recovered_ft - depth_ft, 0.0)
        for side in ("left", "right"):
            state = state.apply(MarkedParking(LegSide(leg_name, side), depth_ft=depth_ft,
                                               curb_offset_ft=buffer_ft))
        print(f"  NOTE: {leg_name} ({leg.curb_to_curb_ft:.0f} ft) -> "
              f"{TARGET_LANE_WIDTH_FT:.0f} ft lanes + {depth_ft:.1f} ft parking both sides"
              + (f" + {buffer_ft:.1f} ft striped buffer" if buffer_ft > 0.1 else "") + ".")
    return state


def _lane_target_reach_ft(leg, side: str, lane_edge_ft: float
                           ) -> tuple[float | None, float | None] | None:
    """How far up this kerb, CONTINUOUSLY FROM STATION 0, a TARGET_LANE_WIDTH_FT lane (plus this
    leg's own divider shift) actually fits - and the narrowest this kerb gets beyond that point.

    PER STATION, through half_width_profile, which is what a single whole-leg minimum used to
    replace here: on W Broad's southwest approach the kerb holds this lane over 336 of 390 ft and
    only pinches inside it over the last 52, but narrowest_half_width_ft's own reduction is the
    LEAST half-width anywhere on the leg, so that one pinch refused a lane over the 336 ft that
    fit it fine and nothing at all was drawn - see half_width_profile's own docstring for the
    identical bug on a two-way section, and corridor.py's AddTwoWayBikeLane._reach_on for the
    same fix already made there for that treatment.

    Returns None where nothing fits even at the start of the traced kerb - the caller's existing
    "nothing to spend" answer. Otherwise returns (reach_ft, narrowest_ft): reach_ft is None where
    every traced station holds the lane (whole leg, exactly as before this existed) or a station
    short of the leg's end where a tail too narrow to hold it begins; narrowest_ft is the least
    half-width found beyond that station, for the refusal recorded there, and is None alongside a
    None reach_ft.
    """
    profile = half_width_profile(leg, side)
    if profile is None:
        # Nothing traced to split on - narrowest_half_width_ft already falls back to the nominal
        # half-width in this case, so there is no per-station reach to compute either.
        if narrowest_half_width_ft(leg, side) - lane_edge_ft <= LANE_WIDTH_SLACK_FT:
            return None
        return None, None
    stations, half_ft = profile
    # Accumulated, not the local value, so "the first station that fails is where the lane
    # stops" is exact: a run that has already failed cannot start fitting again by getting
    # narrower still, and a lane held from station 0 cannot skip over a pinch to a wider run
    # beyond it without putting the reader in a lane that is not actually there.
    run_min = np.minimum.accumulate(half_ft)
    fits = (run_min - lane_edge_ft) > LANE_WIDTH_SLACK_FT
    if not fits[0]:
        return None                     # no room even where this kerb starts being traced
    if fits.all():
        return None, None               # holds all the way - whole leg, exactly as before
    broke = int(np.argmin(fits))
    return float(stations[broke - 1]), float(half_ft[broke:].min())


def hold_travel_lane_at_target(state: DesignState, leg_name: str, side: str) -> DesignState:
    """Bring ONE kerb's travel lane down to TARGET_LANE_WIDTH_FT, spending the surplus.

    An 11 ft lane is the point of the exercise, not a detail of one proposal: a wide lane invites
    the speed every treatment here exists to reduce, and it is the cheapest intervention
    available - paint. So wherever a design has decided to restripe a leg, both of its lanes get
    held at the target and the leftover becomes parking (where the kerb may legally hold it and
    there is room for a usable stall) or hatching (where it may not) - OVER WHATEVER STRETCH OF
    THE KERB ACTUALLY HOLDS THE LANE, per _lane_target_reach_ft, with the rest carried as a
    FacilityRefusal rather than silently dropped or let veto the whole leg.

    ONE DEFINITION, because this was written twice. sites/broad_st_greenwood/scenarios.py grew it
    inline for the two-way corridor and sites/ebroad_princeton/scenarios.py did not, so the same
    corridor treatment left E Broad with 11.68 ft and 13.21 ft lanes while Broad & Greenwood held
    11.00 - the "two records of one decision" failure, in the layer that is supposed to be one
    decision. TravelLanesHoldTheTarget now fails the build for it.

    THE TWO DATUMS ARE BOTH HERE, and confusing them draws a 20 ft hatch. WHETHER there is room
    is a measurement of the traced kerb; WHERE the paint goes is an offset from the NOMINAL
    half-width, because that is the datum MarkedParking and LaneNarrowing subtract from. On
    broad_st_east those differ by 25 ft.
    """
    leg = state.legs[leg_name]
    if leg.curb_to_curb_ft is None:
        return state
    if state.treatment_for(AddBikeLane, LegSide(leg_name, side)) is not None:
        return state          # a bike lane already defines this side's edge
    divider_ft = divider_shift_toward_ft(state, leg_name, side)
    lane_edge_ft = divider_ft + TARGET_LANE_WIDTH_FT
    zone_from_nominal_ft = leg.curb_to_curb_ft / 2 - lane_edge_ft
    state.record_target_lane_room(leg_name, side, zone_from_nominal_ft)
    reach = _lane_target_reach_ft(leg, side, lane_edge_ft)
    if reach is None:
        return state          # the street has nothing spare; the lane is already at or under target
    reach_ft, narrowest_ft = reach
    # Room beyond the travel lane, measured over the stretch that is actually reached rather
    # than the whole leg's minimum - see _lane_target_reach_ft.
    surplus_ft = narrowest_half_width_ft(leg, side, to_ft=reach_ft) - lane_edge_ft
    if zone_from_nominal_ft <= 0:
        return state
    end_ft = None
    if reach_ft is not None:
        end_ft = reach_ft
        state.refuse(leg_name, side, FacilityRefusal(
            reach_ft, leg.centerline.length,
            f"the kerb narrows to {narrowest_ft:.2f} ft off the {leg_name} {side} centreline "
            f"past station {reach_ft:.1f} ft, under the {lane_edge_ft:.2f} ft a "
            f"{TARGET_LANE_WIDTH_FT:.0f} ft travel lane (plus this leg's own divider shift) "
            f"needs there - so nothing is marked on this kerb beyond that station rather than "
            f"paint drawn past the room the kerb actually gives.",
            narrowest_ft))
    if kerb_may_hold_parking(state, leg_name, side) and surplus_ft >= MIN_USABLE_STALL_FT:
        depth_ft = min(surplus_ft, PARKING_STALL_DEPTH_DEFAULT_FT)
        return state.apply(MarkedParking(LegSide(leg_name, side), depth_ft=depth_ft,
                                          curb_offset_ft=max(zone_from_nominal_ft - depth_ft, 0.0),
                                          end_ft=end_ft))
    if zone_from_nominal_ft < MIN_HATCHED_ZONE_FT:
        # Sized from what the road can spare, which is checks.PaintClearOfTheTravelLane's own
        # wording. See MIN_HATCHED_ZONE_FT.
        return state
    return state.apply(LaneNarrowing(LegTarget(leg_name),
                                      stripe_width_ft=zone_from_nominal_ft,
                                      sides=(side,), end_ft=end_ft))


def kerb_may_hold_parking(state: DesignState, leg_name: str, side: str) -> bool:
    """Whether stalls may be marked on this kerb at all, by OSM's own restrictions.

    THE SAME THREE-OUTCOME RULE apply_osm_parking applies, in one place so a scenario cannot
    invent a fourth. A restriction over PART of a kerb does not close it: OSM expresses "no
    parking for the first 100 ft from the junction" by splitting the way, so a kerb is commonly
    restricted at the corner and open beyond it, and daylighting carves the restricted stretch
    back out of the stalls by itself.

    Getting this wrong is not a small error. A scenario here treated any prohibiting span as
    closing the whole kerb, and so hatched 90.4 ft of explicitly `restriction=none` kerb on
    broad_st_east while reporting it as "OSM tags it no_parking" - when OSM tags 79.6 of that
    leg's 170 ft that way and positively permits the rest. On a corridor whose viability depends
    on keeping parking, that is the difference between a plan and a non-starter.
    """
    summary = restriction_summary(state, leg_name, side, state.legs[leg_name].centerline.length)
    return not (summary.restricted_throughout
                or (summary.restricted_in_part and summary.holds_no_stall))
