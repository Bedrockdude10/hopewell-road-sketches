"""BIKEWAYS: the cross-sections (BikeLane, TwoWayBikeLane), the treatments that place them,
and the arithmetic that decides whether one fits.

The cross-section is separated from the treatment deliberately. A section can be asked whether it
fits a given kerb-to-kerb width without anything being applied, which is what lets a scenario try
the standard section, fall back to the constrained one, and report which it got."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import shapely.ops

from src.geometry.targets import AcrossTheJunction, LegSide, Side
from src.geometry.model import narrowest_half_width_ft
from src.geometry.treatments.base import (BOLLARD_DEFAULT_SPACING_FT, LANE_WIDTH_SLACK_FT,
                                          PARKING_STALL_LENGTH_DEFAULT_FT,
                                          TARGET_LANE_WIDTH_FT, Treatment)
from src.geometry.treatments.state import DesignState
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel



# AASHTO gives two figures for an exclusive on-street bike lane and this project needs both, so
# they are two constants rather than one that quietly changes meaning:
#
#   5 ft  the width to design to, and what AASHTO asks for where the lane runs against a curb and
#         gutter or a parking lane - the gutter pan is not ridable, so a 5 ft lane there is about
#         4 ft of usable surface.
#   4 ft  the hard floor, AASHTO's figure where there is no curb face taking part of the lane.
#         Below this it is not a bike lane, and drawing one would propose something that fails the
#         standard it is meant to meet.
#
# The floor is what rules Greenwood Ave and Princeton Ave out entirely (1.0-1.7 ft of lane would
# be left on those kerbs), and it is what lets E Broad's east kerb keep its protection at 4.49 ft
# instead of losing the buffer to hold a nominal 5 - see widest_protected_lane_ft.
MIN_BIKE_LANE_FT = 4.0
AASHTO_MIN_BIKE_LANE_FT = 5.0
# THE BIKE LANE THIS PROJECT PROPOSES: a 5 ft lane with a 2 ft painted buffer. In src rather
# than in each site's scenarios.py for the reason TARGET_LANE_WIDTH_FT gives - it is a standard
# section, not a per-site choice, and two sites each holding their own copy is how one leg gets
# narrowed to one number and checked against another. Broad & Greenwood was 6 ft + 3 ft and
# E Broad derived its buffer from whatever the kerb could spare; both are this now.
#
# The lane width IS AASHTO's minimum, which is worth saying out loud rather than leaving to be
# noticed: this proposes the narrowest lane the standard permits, and the buffer is where the
# rest of the protection comes from. A 5 ft lane plus a 2 ft buffer beats a 6 ft lane with no
# buffer for the same asphalt, because the buffer is what a flex post stands in.
#
# AT 2 FT THE BUFFER IS ESSENTIALLY ITS OWN TWO STRIPES, and that is a real consequence of this
# figure rather than a drawing artifact. Every width here is between paint FACES and the stripes
# come out of the buffer (see BikeLane), and a stripe here is 0.82 ft - 10 in, chosen in
# src/geometry/paint/ to read at the render's scale, against MUTCD's 4-6 in for a lane line. Two
# of them leave 0.36 ft of asphalt showing, against 1.36 ft at the 3 ft buffer this replaced, so
# the buffer's diagonal hatching disappears from the 3D render: there is no longer a strip wide
# enough to draw a stroke across. A post still fits, which is what the buffer is for.
#
# Three ways out if that reads too thin, none of them taken here because 5 + 2 is what was asked
# for: widen the buffer, narrow LANE_EDGE_LINE_WIDTH_FT toward the real 6 in, or accept that a
# 2 ft buffer is two lines and stop hatching it.
BIKE_LANE_WIDTH_FT = AASHTO_MIN_BIKE_LANE_FT
BIKE_LANE_BUFFER_FT = 2.0

# A bike lane hard against the kerb loses its outer foot or so to the gutter pan and to riders
# keeping clear of the kerb. Holding the lane off the kerb by a shy distance instead buys back
# usable width without claiming a wider lane than exists. Used on E Broad, a truck route, where
# 5 ft of lane plus 2 ft of shy reads better than 6 ft of lane against the kerb.
BIKE_LANE_DEFAULT_SHY_FT = 2.0

# THE TWO-WAY (BIDIRECTIONAL) SECTION. A lane carrying riders in both directions on one side
# of the street, which is a different object from two one-way lanes and not just a wider one:
# it needs a centre stripe of its own, and it puts contraflow riders at every junction and
# driveway on that kerb arriving from the direction a turning driver does not check. That is
# why the corridor's side is chosen on how many streets cut the kerb rather than on width.
#
# NACTO's Urban Bikeway Design Guide: 12 ft desirable, 10 ft minimum, 8 ft in constrained
# conditions. The 8 ft case is not offered here - at that width two riders cannot pass an
# oncoming pair, which on a corridor route is the condition rather than the exception.
TWO_WAY_BIKE_LANE_WIDTH_FT = 12.0
MIN_TWO_WAY_BIKE_LANE_FT = 10.0
# NACTO's CONSTRAINED-CONDITIONS width, and it is opt-in rather than a floor the section slides
# down to on its own. At 8 ft two riders cannot pass an oncoming pair, so it is not a width to
# design a route at - which is why an earlier version of this file refused to offer it at all.
#
# What changed is the question. That refusal assumed 8 ft would be the CORRIDOR width; here it is
# a short constrained run through ONE junction whose alternative is no facility at all. W Broad &
# Louellen has 32.10 ft between its traced kerbs, and the standard 10 + 3 section leaves 9.14 ft
# travel lanes - under NJDOT's 10 ft traffic-calming floor. An 8 + 3 section leaves 10.14 ft, so
# the pinch keeps its full buffer, keeps its posts, and the corridor stays continuous.
#
# Requires `constrained=True` on the section, so a scenario cannot reach this width by accident:
# it has to say that it is accepting NACTO's constrained case and why.
CONSTRAINED_TWO_WAY_BIKE_LANE_FT = 8.0
# With vertical elements (flex posts) NACTO asks 3 ft where a two-way lane runs beside moving
# traffic - more than the 2 ft a one-way lane gets, because a head-on error here is a closing
# speed, not an overtaking one.
TWO_WAY_BIKE_LANE_BUFFER_FT = 3.0
# The pitch of the flex posts down that buffer. Close enough to read as a continuous delineator
# rather than a row of dots, which is the whole difference between a painted lane and one a
# driver perceives as protected. Here rather than in a scenario file because it was in TWO of
# them under the same name and the same value, and a corridor whose posts are 8 ft apart on one
# leg and 12 on the next is one facility drawn as two.
BIKE_LANE_BOLLARD_SPACING_FT = 8.0
# WHICH KERB THE BOROUGH'S TWO-WAY LANE RUNS ALONG. A ROUTE decision, not a junction one - a
# lane that changes sides mid-corridor makes riders cross the street to stay on it - and that is
# exactly why it does not belong in a site file. It was written out in THREE of them; had one
# been edited, the corridor would have switched sides at that junction and every drawing would
# still have looked locally correct.
#
# SOUTH, decided by counting the whole borough length from OSM (2026-08-13):
#
#   * side streets cutting the kerb   north 10, SOUTH 7. Five crossings cut both kerbs whichever
#     side is chosen; the difference is the one-sided T-junctions - Windsor Way, Louellen,
#     Mercer, Blackwell and Hamilton on the north against Seminary and Princeton on the south.
#   * parking capacity lost           north 246 stalls, SOUTH 241 - a 2% difference derived from
#     geometry rather than counted, so a tie, not a finding.
#   * mapped driveways                north 20, south 21, and NOT usable either way: OSM has a
#     driveway for 29% of the parcels fronting Broad St, so both are threefold undercounts.
#
# The crossings decided it, because that is the count OSM records completely, and junctions are
# the hazard this treatment specifically creates: a two-way lane puts contraflow riders at every
# one of them, arriving from the direction a turning driver does not check.
#
# CHANGED TO NORTH, 2026-08-18, by Danny, to protect rider safety. The side-street count above was
# never the right denominator: a rider meets every DRIVEWAY mouth on their own kerb too, and
# driveways outnumber side streets roughly 2:1 here. Counted over the whole corridor on OSM:
#
#     lane on   breaks in the lane   mouths on it   parking left (other kerb)
#     north            29                19                26 stalls
#     south            36                26                39 stalls
#
# It is a genuine trade and the two criteria disagree - parking lands on whichever kerb the lane
# does NOT take, so the lane's quiet kerb is the parking's broken one. North costs 13 of the 39
# stalls the street can hold and removes 7 conflict points. The decision is that a conflict point
# is a crash type concentrated on a few riders while lost parking is an inconvenience spread over
# many, and NACTO warns that other street users do not expect contraflow bike traffic
# (STANDARDS.md, verified 2026-08-18). Driveway conflicts also respond to design - corner islands,
# turn wedges, visibility zones - and lost parking responds to nothing.
#
# Translated per leg by side_facing() - a leg's left/right is in its own frame, so the same real
# kerb is "left" on one approach and "right" on the next.
CORRIDOR_SIDE = "north"
# The contraflow stripe's cadence. Shorter than the roadway's dashed centreline, because it is
# read at bicycle speed over a 12 ft lane rather than at 25 mph over a 40 ft one, and a stripe
# scaled to the road reads as two or three marks over a whole block.
# WHERE THE BIKE LANE SYMBOL GOES. NACTO asks for it after every driveway and intersection AND at
# least every 500 ft along the lane - both rules, not either, because on a corridor with 19
# junctions the interval alone leaves long unmarked stretches while the mouths alone cluster them
# where nobody needs reminding. MUTCD Fig 9E-1 is the marking. STANDARDS.md, verified 2026-08-18.
SYMBOL_INTERVAL_FT = 500.0
# How far past a mouth the reminder sits: clear of the opening itself, near enough to read as
# belonging to it. It is what keeps the symbol out of any opening, which is why BIKE_LANE_SYMBOL's
# AT_AN_OPENING row can be CARRIED - there is never one inside a mouth to cut.
SYMBOL_CLEAR_OF_OPENING_FT = 15.0
# The painted footprint. A schematic arrow rather than a drawn bicycle - see markings.py:
# BIKE_LANE_SYMBOL_POLYGONS for why, and the legend says which marking it represents.
SYMBOL_LENGTH_FT = 5.5
SYMBOL_WIDTH_FT = 2.4

CONTRAFLOW_DASH_FT = 3.0
CONTRAFLOW_GAP_FT = 5.0
# Below this the two travel lanes are no longer lanes. NACTO's urban minimum is 10 ft, and
# TARGET_LANE_WIDTH_FT (11) is what this project designs to; a corridor that cannot hold two
# 10 ft lanes beside the section is reported rather than drawn.
MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT = 10.0
# How far behind the junction node a THROUGH-RUNNING kerb's paint starts, so the two legs' halves
# overlap and fuse instead of each stopping at its own station 0. Enough to cover the 1.28 ft seam
# at W Broad & Louellen with margin, and bounded by the tracing either way, so a kerb traced right
# up to the node overlaps by this much and one traced only from the node out overlaps not at all
# and is no worse off than before.
THROUGH_JUNCTION_OVERLAP_FT = 3.0


def _feet(value: float) -> str:
    """A width for a note: a decimal only where the number has one. 5.0 -> "5", 4.4947 -> "4.5"."""
    return f"{value:.1f}".removesuffix(".0")


@dataclass(frozen=True)
class BikeLane:
    """One exclusive bike lane, described from the centerline outward.

    Across the road on this side: TARGET_LANE_WIDTH_FT of travel lane, then `buffer_ft` of
    painted buffer (or just the lane line where there is no buffer), then `width_ft` of bike
    lane, then `parking_ft` of marked parking, then `shy_ft` of spare asphalt to the kerb. Any
    of the last three may be zero.

    EVERY WIDTH HERE IS BETWEEN PAINT FACES, not between stripe centrelines, and the stripes'
    own bodies come out of the buffer rather than out of either lane. A 0.82 ft edge line
    centred on the 11 ft mark leaves a 10.59 ft travel lane, which
    check_paint_clear_of_the_travel_lane reports and was right to: it is the same accounting
    lane_edge_stripes already does for a lane-narrowing buffer.

    `parking_ft` > 0 is the parking-protected form: the parked cars sit OUTSIDE the bike lane,
    between it and the kerb, so the lane is shielded from moving traffic by the parking rather
    than only by paint. That ordering is the whole point of it and is why the parking lane's
    position is part of this record rather than a separate add_marked_parking call.
    """
    width_ft: float
    buffer_ft: float = 0.0
    parking_ft: float = 0.0
    shy_ft: float = 0.0
    # Where this side's section BEGINS, as a distance from the alignment. Normally the travel
    # lane's own width, because the two travel lanes straddle the alignment symmetrically. A
    # two-way lane on one side does not leave them straddling it - see TwoWayBikeLane - so the
    # inner edge becomes a property of the section rather than a constant. Defaulted, so every
    # existing one-way scenario is unchanged.
    travel_edge_ft: float | None = None

    def __post_init__(self):
        if self.width_ft < MIN_BIKE_LANE_FT:
            raise ValueError(
                f"A {self.width_ft:.2f} ft bike lane is under the {MIN_BIKE_LANE_FT:.0f} ft floor "
                f"(AASHTO's minimum where no curb face eats into the lane; {AASHTO_MIN_BIKE_LANE_FT:.0f} ft "
                f"is the width to design to). Draw no lane rather than one that fails the standard "
                f"it is meant to meet.")
        if self.buffer_ft and self.buffer_ft < min_bike_lane_buffer_ft():
            raise ValueError(
                f"A {self.buffer_ft:.2f} ft buffer cannot hold the two {_lane_line_ft():.2f} ft "
                f"lines that bound it. Use no buffer - the lane then takes a single line against "
                f"the travel lane, which is what a conventional bike lane is.")

    @property
    def has_outer_line(self) -> bool:
        """A bike lane's outer edge is always painted, because it is never the kerb.

        This returned False without parking outside, on the reasoning that a conventional bike
        lane against a kerb is bounded by the kerb. It is not bounded by the kerb here: a lane
        is a STANDARD width and the asphalt left over between it and the kerb is hatched, the
        same way an 8 ft parking stall is a standard width with its leftover hatched
        (add_marked_parking's curb_offset_ft). Without the outer stripe the lane read as running
        all the way to the kerb - which is why the drawn lanes looked far wider than the 6 ft
        they were specified at.
        """
        return True

    def kerb_hatch_ft(self, available_ft: float) -> float:
        """Leftover asphalt between the lane's outer stripe and the kerb, to be hatched.

        The variable part of the cross-section, exactly as it is for a parking lane: the travel
        lane holds its width, the bike lane holds its width, and the hatching absorbs everything
        the street happens to have. `available_ft` is the room to the kerb at the station being
        drawn, so this pinches to nothing where a leg narrows rather than pushing paint over the
        kerb.
        """
        return max(available_ft - self.offsets_from_centerline_ft()["outer_ft"], 0.0)

    @property
    def total_ft(self) -> float:
        """Everything this side needs, travel lane and stripes included."""
        return self.offsets_from_centerline_ft()["outer_ft"] + self.shy_ft

    def offsets_from_centerline_ft(self) -> dict[str, float]:
        """Where each boundary sits, as a distance from the centerline.

        One place, so the plan view, the 3D export and the checks cannot disagree about which
        stripe is which - the ordering across the road IS the design. Each `*_line_ft` is the
        stripe's CENTRE, offset half a stripe outward from the face it marks so the protected
        width behind it stays whole.
        """
        line_ft = _lane_line_ft()
        travel_edge = TARGET_LANE_WIDTH_FT if self.travel_edge_ft is None else self.travel_edge_ft
        # With a buffer the two stripes bounding it come out of the buffer's own width; without
        # one there is a single stripe and it comes out of nothing but itself.
        bike_inner = travel_edge + (self.buffer_ft if self.buffer_ft else line_ft)
        bike_outer = bike_inner + self.width_ft
        parking_inner = bike_outer + (line_ft if self.has_outer_line else 0.0)
        return {"travel_lane_edge_ft": travel_edge,
                "inner_line_ft": travel_edge + line_ft / 2,
                "buffer_outer_line_ft": bike_inner - line_ft / 2 if self.buffer_ft else None,
                "bike_inner_ft": bike_inner,
                "bike_outer_ft": bike_outer,
                "outer_line_ft": bike_outer + line_ft / 2 if self.has_outer_line else None,
                "parking_outer_ft": parking_inner + self.parking_ft,
                "outer_ft": parking_inner + self.parking_ft}

    @property
    def hugs_kerb(self) -> bool:
        """Whether this section's outer edge FOLLOWS the traced kerb rather than standing off the
        leg's narrowest point.

        False here, and that is a measured decision rather than caution. broad_st_east's left kerb
        is traced between 18 and 32 ft from the centreline over one leg - a 14 ft range, because
        the tracing takes in a bay and the corner flare as well as the straight run. A one-way lane
        that followed it would swing 14 ft and read as snaking; drawn against the narrowest point it
        stays straight and the leftover is hatched, which is what a striper would do beside a bay.

        TwoWayBikeLane also returns False: the lane stays straight, like the car lane and like a
        one-way bike lane. The kerb provides physical protection (the vertical element), but the
        paint does not need to follow every wiggle. The buffer between the lane and the kerb absorbs
        the variation, just like hatching does for a one-way lane.
        """
        return False

    def offsets_from_kerb_ft(self) -> dict[str, float]:
        """The same section read from the KERB INWARD, as insets from the traced kerb.

        WHY BOTH DIRECTIONS EXIST, and which boundary belongs to which. The offsets above are
        measured from the alignment at the leg's NARROWEST traced point, because that is where a
        section has to fit. But the kerb is not at the narrowest point everywhere: on
        w_broad_st_northeast's south-east side it runs 17.24 ft out at station 46.5 and 25.13 ft
        out at the junction throat, a real mapped convergence where two streets of different
        widths meet. Drawn from the alignment, the lane held straight through it and left a
        hatched wedge against the kerb widening to 8.68 ft - a protected lane visibly wandering
        away from the thing protecting it.

        So the boundaries are split by WHAT THEY BELONG TO, not by convenience:

            the lane's outer edge, its asphalt, its inner edge   -> the KERB (here)
            the travel lane's edge stripe                        -> the ALIGNMENT (above)

        and the buffer between them absorbs the difference, which is where a designer puts it.
        Measuring the travel lane's edge from the kerb instead would hand the kerb's convergence
        to the travel lane, and it would stop holding TARGET_LANE_WIDTH_FT - which is the one
        thing every check here is about.

        Insets are positive INWARD from the kerb, and each `*_line_ft` is again the stripe's
        CENTRE, half a stripe outward from the face it marks - the same convention, mirrored.
        """
        line_ft = _lane_line_ft()
        bike_outer = self.shy_ft + self.parking_ft + (line_ft if self.has_outer_line else 0.0)
        bike_inner = bike_outer + self.width_ft
        return {"kerb_hatch_ft": self.shy_ft,
                "parking_outer_ft": self.shy_ft,
                "outer_line_ft": (self.shy_ft + self.parking_ft + line_ft / 2
                                   if self.has_outer_line else None),
                "bike_outer_ft": bike_outer,
                "bike_inner_ft": bike_inner,
                "buffer_outer_line_ft": bike_inner + line_ft / 2 if self.buffer_ft else None}


def _lane_line_ft() -> float:
    """The painted width of one edge line. Local import: src/geometry/paint/ imports this
    module, and the figure is single-sourced there against what the 3D renderer actually lays."""
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT

    return LANE_EDGE_LINE_WIDTH_FT


def min_bike_lane_buffer_ft() -> float:
    """The narrowest strip that is a buffer at all: one wide enough to hold its own two lines.

    A PHYSICAL floor, not a design minimum like AASHTO_MIN_BIKE_LANE_FT. A buffer is bounded by
    a stripe on each side, and two 0.82 ft stripes are 1.64 ft of paint - below that there is no
    buffer, only a double line. It is also the figure that decides whether a lane can be
    PROTECTED, since a flex post has to stand inside the buffer and not in either lane, and it
    is what rules out both kerbs of e_broad_st_east (0.80 and 1.49 ft spare) while permitting
    e_broad_st_west (2.01 and 2.14).

    A function rather than a constant for the same reason _lane_line_ft is one: the stripe width
    lives in src/geometry/paint/, which imports this module.
    """
    return 2 * _lane_line_ft()


# THE STATE'S OWN GUIDANCE RULES THIS FACILITY OUT, and every design that uses it carries this
# sentence in its provenance note as a result. NJDOT's Bicycle Compatible Roadways and Bikeways
# (1996): "Two-way bicycle lanes on one side of the roadway are unacceptable because they promote
# riding against the flow of motor vehicle traffic."
#
# Not a caveat to bury. It is the published guidance for the state this project is in, a county
# engineer may cite it first, and a render that omits it is claiming more consensus than exists.
# The counter-argument - 1996 predates separated bikeways entirely, and MUTCD 11th ed. §9E.06 now
# provides markings for exactly this facility - belongs in the submission, not in the drawing's
# silence. STANDARDS.md §4 carries the full text and the case either way.
NJDOT_TWO_WAY_OBJECTION = (
    "NOTE: NJDOT's Bicycle Compatible Roadways and Bikeways (1996) calls a two-way bike lane on "
    "one side of the roadway 'unacceptable'; that guidance predates separated bikeways and MUTCD "
    "11th ed. 9E.06 now marks them, but the departure has to be argued, not assumed - see "
    "STANDARDS.md 4."
)


@dataclass(frozen=True)
class TwoWayBikeLane(BikeLane):
    """A bidirectional bike lane on ONE side of a leg, and the shifted section it implies.

    THE ALIGNMENT DOES NOT MOVE. That is the whole trick, and it is why this is drawable at
    all. The main README recorded parking-protected lanes as undrawable here because "fitting
    it would mean shifting the travel lanes off the NJDOT alignment - a real design, but not
    one this pipeline can draw, since the alignment is the datum every offset, stop bar and
    crossing frame is measured from". The datum genuinely cannot move. But it never had to be
    the middle of the travel lanes: it is the line stations are measured along, and a
    cross-section is free to be asymmetric about it. So every station, every crossing frame and
    every stop bar stays exactly where it was, and what changes is that this side's section
    starts further out and the painted divider between the travel lanes sits off the alignment
    by travel_lane_divider_shift_ft.

    Across the road, from the FAR kerb: travel lane, the divider, travel lane, buffer, two-way
    bike lane, whatever is left hatched to the near kerb.

    `near_half_ft` is the distance from the alignment to the kerb this lane is on, and
    `far_half_ft` to the opposite kerb - both measured at the leg's NARROWEST traced point, for
    the reason AddBikeLane gives: a treatment applied to a kerb is a promise about the whole of
    it. They are not interchangeable and the asymmetry is the point, so they are separate
    fields rather than one width.
    """
    near_half_ft: float = 0.0
    far_half_ft: float = 0.0
    # Accepting NACTO's constrained-conditions width. Declared per section rather than inferred
    # from how narrow the street is, so the drawing records a DECISION and not an accommodation.
    constrained: bool = False

    def __post_init__(self):
        floor = (CONSTRAINED_TWO_WAY_BIKE_LANE_FT if self.constrained
                 else MIN_TWO_WAY_BIKE_LANE_FT)
        if self.width_ft < floor:
            raise ValueError(
                f"A {self.width_ft:.2f} ft two-way bike lane is under NACTO's {floor:.0f} ft "
                f"{'constrained-conditions' if self.constrained else 'minimum'} width "
                f"({TWO_WAY_BIKE_LANE_WIDTH_FT:.0f} ft is the width to design to). Two riders "
                f"meeting head-on need the width of two riders; a one-way lane's "
                f"{AASHTO_MIN_BIKE_LANE_FT:.0f} ft floor does not apply to a lane carrying both "
                f"directions.")
        super().__post_init__()
        travel_way_ft = self.near_half_ft + self.far_half_ft - self.section_ft
        if travel_way_ft / 2 < MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT:
            raise ValueError(
                f"A {self.width_ft:.1f} ft lane and a {self.buffer_ft:.1f} ft buffer spend "
                f"{self.section_ft:.2f} ft of the {self.near_half_ft + self.far_half_ft:.2f} ft "
                f"this leg has between its kerbs at its narrowest, leaving {travel_way_ft:.2f} ft "
                f"for traffic - {travel_way_ft / 2:.2f} ft per travel lane, under the "
                f"{MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT:.0f} ft floor. Narrow the lane, drop the "
                f"buffer, or put a conventional pair of one-way lanes on this leg instead.")

    @property
    def hugs_kerb(self) -> bool:
        """True: a lane that exists to be shielded by a kerb is measured from that kerb.

        Held straight instead, it falls away from its own protection. On W Broad the kerb walks
        outward the whole length of both legs, so a lane pinned to the narrowest traced point sat
        4.8 ft off the kerb at station 67 and 8.4 ft off at station 222 - a few feet of bare
        pavement between the bikeway and the kerb protecting it, for 270 ft, on every leg.

        The reason this used to be False was the other failure: broad_st_east's kerb moves 20.4 ft,
        because the tracing takes in a corner flare, and a lane following THAT reads as snaking.
        Both are real, and neither is a reason to choose per leg - the two differ by how sharply
        the kerb moves, not by how far, so the limit belongs on the rate and applies everywhere.
        tapered_curb_offsets is where that lives, and MAX_KERB_FOLLOW_TAPER is the rate; a drift
        gentler than 1:10 is followed to within a few inches and a 1:2 kink is refused by up to
        12 ft, at any frame scale.
        """
        return True

    @property
    def section_ft(self) -> float:
        """What this side's section spends: the lane, its buffer and the stripes bounding them.

        NOT `total_ft`, which is measured from the alignment and therefore already contains the
        travel lane. This is the width taken OUT of the roadway, which is what the two travel
        lanes have to be fitted around.
        """
        line_ft = _lane_line_ft()
        return self.width_ft + (self.buffer_ft if self.buffer_ft else line_ft) + line_ft

    def offsets_from_centerline_ft(self) -> dict[str, float]:
        """The one-way section's own arithmetic, re-anchored to where this section starts.

        BikeLane already lays out every stripe from `travel_edge_ft` outward, so the two-way
        case is that same layout with a different starting offset - not a second copy of the
        ordering. Getting a second copy is exactly what offsets_from_centerline_ft exists to
        prevent: the ordering across the road IS the design, and two of them can disagree.
        """
        return BikeLane(width_ft=self.width_ft, buffer_ft=self.buffer_ft,
                         parking_ft=self.parking_ft, shy_ft=self.shy_ft,
                         travel_edge_ft=self.near_half_ft - self.section_ft
                         ).offsets_from_centerline_ft()


def travel_lane_divider_shift_ft(section: TwoWayBikeLane) -> float:
    """How far the painted divider between the travel lanes sits off the alignment.

    Positive TOWARD THE FAR KERB - away from the side carrying the lane, which is the direction
    traffic is pushed by taking width out of one kerbside.

    THE TRAVEL LANES HOLD TARGET_LANE_WIDTH_FT AND THE FAR KERB KEEPS THE SURPLUS. Placing the
    divider mid-way through whatever the section leaves is the obvious rule and it is the wrong
    one on a wide street: Broad St's west leg has 52.5 ft between kerbs, so an equal split gave
    two 18.35 ft travel lanes, and an 18 ft lane invites exactly the speed this whole project
    exists to reduce. Spare width beside a travel lane is not the travel lane's - it is parking,
    or it is hatched - which is the same accounting a bike lane and an 8 ft stall already get.

    Where the leg cannot hold two target-width lanes beside the section, it falls back to an
    equal split, because then the shortfall is the street's and there is nothing to allocate.
    E Broad's east leg is that case at 10.04 ft a lane.
    """
    inner_edge_ft = section.near_half_ft - section.section_ft
    travel_way_ft = section.near_half_ft + section.far_half_ft - section.section_ft
    if travel_way_ft < 2 * TARGET_LANE_WIDTH_FT:
        return section.far_half_ft - travel_way_ft / 2
    return TARGET_LANE_WIDTH_FT - inner_edge_ft


def far_kerb_surplus_ft(section: TwoWayBikeLane) -> float:
    """Width left against the FAR kerb once the section and two target-width lanes are placed.

    What a two-way lane on one side frees up on the other, and the reason the pair belongs in one
    proposal: the kerb losing its parking to the bike lane is not the kerb that gains this. Zero
    or negative where the leg has nothing spare.
    """
    inner_edge_ft = section.near_half_ft - section.section_ft
    return section.far_half_ft + inner_edge_ft - 2 * TARGET_LANE_WIDTH_FT


def bike_lane_spare_ft(state: DesignState, leg_name: str, side: str, width_ft: float,
                        buffer_ft: float = 0.0, parking_ft: float = 0.0) -> float:
    """Room left over on this kerb after a bike lane cross-section, at its narrowest point.

    What a caller sizing a shy distance needs, and it goes through BikeLane's own accounting
    rather than being re-derived: a caller subtracting the travel lane and the lane width by
    hand misses the lane LINE, which is 0.82 ft and the difference between a section that fits
    e_broad_st_east and one that is refused for being 0.70 ft too wide.
    """
    lane = BikeLane(width_ft=width_ft, buffer_ft=buffer_ft, parking_ft=parking_ft)
    return narrowest_half_width_ft(state.legs[leg_name], side) - lane.total_ft


def widest_protected_lane_ft(state: DesignState, leg_name: str, side: str) -> float | None:
    """The widest PROTECTED bike lane this kerb can hold, or None if that is under the floor.

    THE BUFFER IS KEPT AND THE LANE GIVES, which is the opposite of what this project did first.
    The earlier rule held the lane at a nominal 5 ft and dropped the 2 ft buffer whenever the last
    few inches did not fit, so a kerb 0.51 ft short lost its flex posts entirely and got a
    conventional lane instead - trading all of the protection for 6 in of paint. A rider is better
    served by a 4.49 ft lane with a post beside it than by a 5 ft lane with a moving truck beside
    it, and 4 ft is a width AASHTO recognises (MIN_BIKE_LANE_FT).

    Ordered outward from the centerline, which is the order the widths are given up in: the 11 ft
    travel lane is fixed (TravelLanesKeepTheirWidth), the 2 ft buffer is fixed because it is what a
    post stands in, and the bike lane takes what is left - capped at the 5 ft design width, since
    spare beyond that is hatched rather than spent on a lane wider than the standard.

    Measured, this is the difference between one protected kerb and two on E Broad's east leg:
    +0.01 and +0.14 ft spare on the west leg (5 ft either side), -0.51 on the east right (4.49 ft,
    protected) and -1.20 on the east left (3.80 ft, under the floor - see the caller for what
    happens then).
    """
    spare_ft = bike_lane_spare_ft(state, leg_name, side, width_ft=BIKE_LANE_WIDTH_FT,
                                   buffer_ft=BIKE_LANE_BUFFER_FT)
    fitted_ft = min(BIKE_LANE_WIDTH_FT, BIKE_LANE_WIDTH_FT + spare_ft)
    return fitted_ft if fitted_ft >= MIN_BIKE_LANE_FT - LANE_WIDTH_SLACK_FT else None


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
                                        offset_band_polygon, paint_stations, parking_stall_lines_ft,
                                        section_holds_to_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, _one, end_against_crossing,
                                        parking_runs)

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
        # AND WHERE THE STREET RUNS OUT. The section is sized at the narrowest point of the span it
        # was measured over (Leg.design_length_ft), so that is the span it is known to fit; past it
        # a kerb that comes in tighter leaves every kerb-referenced stripe held at its floor_ft, on
        # the far side of the kerb. Drawn to here instead - see model.section_holds_to_ft. None at
        # 1x on every leg of every site, where the whole leg is inside the measured span.
        outermost_ft = max(o for k, o in bounds.items()
                            if o is not None and k != "travel_lane_edge_ft")
        # Silent, unlike a rung refusal: which section an APPROACH takes is a design decision and
        # gets a note, while how far down the block the drawing carries it is a fact about the
        # frame. It is in the geometry either way, and PaintInsideTheCurb is what would catch its
        # absence.
        stop_ft = section_holds_to_ft(leg, side, outermost_ft, start_ft)
        # ...AND THE FLOOR ITSELF IS CAPPED AT THE ROOM THERE IS. floor_ft says "never come inward
        # of the design", which is sound only while the design fits: the test that granted this
        # section measures the TRAVEL WAY, so a section can be granted whose outer stripe sits
        # outside the near kerb - 24.16 ft of section against a kerb 20.32 ft out on W Broad's
        # southwest approach. Held at that floor the lane is drawn over the kerb; capped here it
        # follows the kerb inward instead, which is the graduated width this facility is meant to
        # have - full where the street allows it, narrower where it does not.
        room_ft = narrowest_half_width_ft(leg, side, max(start_ft, 0.0),
                                          leg.centerline.length if stop_ft is None else stop_ft)
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
            to_ft = stop_ft if to_ft is None else to_ft
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
            to_ft = stop_ft if to_ft is None else to_ft
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
                 inset_line_ft(leg, side, bounds["inner_line_ft"], start_ft, stop_ft,
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
        # The leg's own drawn length, NOT beyond_ft. beyond_ft is a clipping THRESHOLD - the
        # station past which a piece is discarded for lying behind a crossing - and reading it as
        # the run's end gave 6 ft "runs" at Broad & Greenwood, inside which no symbol interval
        # could ever land. Two different quantities that are both stations.
        run_end_ft = leg.centerline.length
        for station_ft in bike_symbol_stations_ft(max(start_ft, 0.0), run_end_ft, mouths):
            centre_ft = lane_centre_at(station_ft)
            if centre_ft is None:
                continue
            # A two-way lane's two halves face opposite ways, so each gets its own symbol: that is
            # what tells a driver at a mouth which direction the rider bearing down on them is
            # coming from, and it is the reason the symbol is worth drawing rather than decorative.
            faces = (True, False) if isinstance(self, AddTwoWayBikeLane) else (True,)
            for index, forward in enumerate(faces):
                # Half a symbol plus clearance either side of centre. Bounded by the lane's own
                # sixth so a narrow lane does not push them into the edge stripes, and floored at
                # half a symbol plus a margin so a wide one does not let the pair touch - which is
                # what a plain sixth did on a 10 ft lane, overlapping them by 2 sq ft.
                spread_ft = max(SYMBOL_WIDTH_FT / 2 + 0.4, lane.width_ft / 6)
                spread_ft = min(spread_ft, max(lane.width_ft / 2 - SYMBOL_WIDTH_FT / 2, 0.0))
                across_ft = centre_ft + (0.0 if len(faces) == 1
                                         else spread_ft * (1 if index else -1))
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
                stations = paint_stations(leg, side, start_ft, stop_ft)
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
            for run_start_ft, run_end_ft in parking_runs(ctx.state, leg_name, side,
                                                          ctx.crosswalk_offsets, ctx.props):
                for divider in parking_stall_lines_ft(
                        leg, side, lane.parking_ft, PARKING_STALL_LENGTH_DEFAULT_FT,
                        max(run_start_ft, start_ft), run_end_ft,
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


@dataclass(frozen=True)
class AddBikeLaneBollards(Treatment):
    """Flex-post delineators down the buffer between a bike lane and the travel lane.

    This is what turns a painted bike lane into a protected one, and the position is the whole
    point: the posts go on the TRAFFIC side of the lane, in the buffer, because that is the side
    a rider needs protecting from. Posts in the kerb-side hatching would protect nothing.

    Requires a buffer to stand them in, and refuses rather than improvising when there is none -
    a lane with no buffer has no room for a post that is not either in the travel lane or in the
    bike lane. That is a real constraint and not a formality: E Broad St has 17.6 ft from the
    alignment to its nearest kerb, and an 11 ft lane plus a 5 ft lane plus their two edge stripes
    already account for 17.6 of it.

    The precondition is on another TREATMENT rather than on the street, which is why it is
    checked in apply_to: nothing about a spacing is wrong on its own, and what makes this
    unbuildable is the absence of a buffered lane under it. A treatment that depends on another
    is the case a self-validating constructor cannot cover by itself, and the reason apply_to
    gets the design.
    """
    paint_group: ClassVar[int] = 30
    paint_rank: ClassVar[int] = 1
    spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT

    def __post_init__(self):
        if self.spacing_ft <= 0:
            raise ValueError(f"Posts need a spacing; got spacing_ft={self.spacing_ft}.")

    def describe(self) -> str:
        return f"AddBikeLaneBollards({self.target.leg}, {self.target.side}): "

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        bike_lane = state.treatment_for(AddBikeLane, self.target)
        if bike_lane is None:
            raise KeyError(f"{self.target} has no bike lane - apply AddBikeLane first.")
        lane = bike_lane.section(state)   # resolved, not declared - see this class's paint()
        if not lane.buffer_ft:
            raise ValueError(
                f"{self.target}'s bike lane has no buffer, so there is nowhere to stand a "
                f"delineator that is not in a travel lane or in the bike lane itself. A protected "
                f"lane needs a buffer; give it one, or leave the lane conventional and say so.")
        return (f"flex-post delineators at {self.spacing_ft:.0f} ft in the {lane.buffer_ft:.0f} ft "
                f"buffer between the travel lane and the bike lane - the traffic side, which is "
                f"the side that needs protecting.")


    def paint(self, ctx) -> None:
        """Down the middle of the buffer, on the TRAFFIC side of the lane - the side a rider
        needs protecting from. This treatment refuses a lane with no buffer, so there is always
        a strip to centre them in here.

        Started at target_ft, not at the zone's own start_ft. A marked leg's paint deliberately
        begins INSIDE the crossing so the crossing cuts its end (end_against_crossing), and a
        post is not paint: it cannot be trimmed by a crossing, it would simply be standing in
        one. target_ft is the first station clear of where the crossing actually reaches on this
        side.

        AND THE GAP HAS TO BE ADDED BACK ON A LEG WITH NO PAINTED CROSSING. There, target_ft is
        the junction mouth's own end with no striper's gap in it, which is right for paint - the
        mouth CUTS paint, so starting level with it leaves no bare stretch - and wrong here for
        the same reason the paragraph above gives. Left level, the first post of the row landed
        exactly on the mouth's lip (station 26.404 against a mouth ending at 26.404 on E Broad's
        south kerb) and whether PaintContext.emit dropped it came down to float noise in the
        band's own vertices. A post either stands in the intersection or it does not.

        The lane's cross-section belongs to the AddBikeLane underneath, so it is read from the
        design rather than restated - the same reason this treatment requires one.
        """
        from src.geometry.markings import BOLLARD
        from src.geometry.model import points_at_offset_ft
        from src.geometry.paint import PaintPiece, _dot, end_against_crossing
        # From its home rather than through paint, which only ever passed it along.
        from src.render.crosswalks import CROSSWALK_CLEARANCE_FT

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        # section(state), NOT .lane. The declared cross-section starts at TARGET_LANE_WIDTH_FT by
        # definition, and for a TWO-WAY lane that is not where the section is - it starts wherever
        # the shifted travel lanes end. Reading the declared one put this row of posts 12.5 ft from
        # the alignment on broad_st_east, INSIDE a lane spanning 8.85-20.85 ft: flex posts standing
        # in the bike lane they are supposed to protect. BollardsStandInTheirBuffer now fails the
        # build for it, and src/render/props.py had the same mistake, which is why the 2D and 3D
        # views agreed and post_not_in_the_render stayed green.
        lane = ctx.state.treatment_for(AddBikeLane, self.target).section(ctx.state)
        bounds = lane.offsets_from_centerline_ft()
        at = ctx.anchors(leg_name, side, inner_offset_ft=(
            leg.curb_to_curb_ft / 2 - lane.total_ft + TARGET_LANE_WIDTH_FT))
        if (leg_name, side) in ctx.straight_through:
            start_ft = clear_ft = 0.0
        elif leg_name in ctx.marked:
            start_ft, _beyond_ft = end_against_crossing(at)
            clear_ft = at.target_ft
        else:
            start_ft = clear_ft = at.target_ft + CROSSWALK_CLEARANCE_FT
        centre_ft = (bounds["travel_lane_edge_ft"] + bounds["bike_inner_ft"]) / 2
        for point in points_at_offset_ft(leg, side, centre_ft, max(start_ft, clear_ft),
                                          spacing_ft=self.spacing_ft):
            ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, side))


def divider_shift_toward_ft(state: DesignState, leg_name: str, side: str) -> float:
    """How far the travel-lane divider sits off the alignment, measured TOWARD `side`.

    Zero on every leg whose travel lanes straddle the alignment, which is all of them until a
    two-way bike lane takes width out of one kerbside. Signed, because the two sides of a leg see
    the same shift in opposite directions, and anything that ignores the sign is wrong on exactly
    one of them.

    ONE DEFINITION, because four things need it and they must agree: the two travel-lane checks in
    src/checks.py, the plan view's lane dimension label, and the centreline paint both views draw.
    The label was the one that got it wrong - it measured the lane from the ALIGNMENT and printed
    "lane 9.6 ft" beside a lane the geometry had built at 11.00 ft. A wrong number on a correct
    drawing is worse than a wrong drawing, because it is the number a reviewer takes away, and an
    11 ft lane is not negotiable with a county engineer.
    """
    for treatment in state.treatments_of(AddTwoWayBikeLane):
        if treatment.target.leg != leg_name:
            continue
        shift_ft = travel_lane_divider_shift_ft(treatment.section(state))
        # The shift is defined as positive AWAY from the side carrying the lane.
        return -shift_ft if str(treatment.target.side) == str(side) else shift_ft
    return 0.0


def travel_lane_width_ft(state: DesignState, leg_name: str, side: str, painted_ft: float) -> float:
    """The real width of the travel lane on this side, given how much kerbside paint it has.

    From the DIVIDER to the paint, not from the alignment to the paint - those are the same thing
    only while the two lanes straddle the alignment. Everything that reports or checks a lane
    width goes through here.
    """
    half_ft = state.legs[leg_name].curb_to_curb_ft / 2
    return half_ft - painted_ft - divider_shift_toward_ft(state, leg_name, side)


def bike_symbol_stations_ft(start_ft: float, end_ft: float, openings=()) -> list[float]:
    """Stations along one run of lane where a BIKE LANE symbol belongs.

    Both of NACTO's rules at once: one after every opening the lane crosses, and one at least
    every SYMBOL_INTERVAL_FT regardless. `openings` are (lo, hi) station pairs on this kerb.

    ONE PLACE, so the plan view, the 3D export and the corridor strip cannot disagree about how
    many symbols a design calls for - the same reason the section's own offsets live on BikeLane.
    """
    at = [start_ft + SYMBOL_CLEAR_OF_OPENING_FT]
    station = start_ft + SYMBOL_INTERVAL_FT
    while station < end_ft:
        at.append(station)
        station += SYMBOL_INTERVAL_FT
    for _lo, hi in openings:
        if start_ft < hi < end_ft:
            at.append(hi + SYMBOL_CLEAR_OF_OPENING_FT)
    # THINNED, because the two rules can land on top of each other: a mouth 15 ft before an
    # interval station puts two symbols in the same 5.5 ft of road, which the collision check
    # reads - correctly - as ground painted twice. One symbol per place; whichever rule asked
    # for it, the rider only needs telling once.
    kept: list[float] = []
    for station in sorted(s for s in at if start_ft <= s <= end_ft):
        if not kept or station - kept[-1] >= SYMBOL_LENGTH_FT * 1.5:
            kept.append(station)
    return kept


def bike_symbol_polygon(on, side: str, station_ft: float, centre_offset_ft: float,
                        forward: bool = True):
    """The symbol's painted footprint at one station, centred in the lane.

    An arrowhead on a shaft, pointing the way that half of the lane runs. `forward` is what makes
    a bidirectional lane's two halves face opposite ways, which is the whole reason a symbol earns
    its place here rather than being decoration: it tells a driver at a mouth which direction the
    rider bearing down on them is coming from.
    """
    import numpy as np
    from shapely.geometry import Polygon

    from src.geometry.model import place_in_measured_frame

    sign = 1.0 if side == "left" else -1.0
    nose = SYMBOL_LENGTH_FT / 2 * (1.0 if forward else -1.0)
    tail = -nose
    half = SYMBOL_WIDTH_FT / 2
    shaft = SYMBOL_WIDTH_FT / 6
    # (along, across) in the lane's own terms, then placed in the road frame once.
    outline = [(nose, 0.0), (nose - nose / 2, half), (nose - nose / 2, shaft),
               (tail, shaft), (tail, -shaft), (nose - nose / 2, -shaft),
               (nose - nose / 2, -half)]
    stations = np.array([station_ft + along for along, _across in outline])
    offsets = np.array([sign * (centre_offset_ft + across) for _along, across in outline])
    placed = place_in_measured_frame(on.centerline, stations, offsets)
    return Polygon([tuple(point) for point in placed])


# How far past the green's last station the cross-section is sampled to read the lane's WIDTH
# there. The end face is not always square: a lane cut by a skewed crossing ends on that
# crossing's diagonal, so a single station gives one vertex rather than a face. Two feet is
# comfortably wider than any such diagonal is deep here and far shorter than the run over which
# a kerb-hugging lane's width changes, so the min/max it takes are the lane's two real edges.
LANE_END_FACE_SAMPLE_FT = 2.0

# Below this there is no gap worth crossing. A junction whose two lane ends are already within a
# mark's length of each other does not need an extension - it needs nothing, and drawing one
# would put a stray dash in the join.
MIN_EXTENSION_GAP_FT = 6.0

# How much of a dotted mark has to survive the clips for it to still BE that mark. The same rule
# PaintContext._dashes_across_openings applies to a LINE ("a part of a mark is not a mark", at half
# a mark) written for an area, because the green is an area and nothing had ever stated it there.
# A half is the line rule's own fraction, so the two do not disagree about what a partial dash is.
MIN_MARK_FRACTION = 0.5


def lane_end_face(ctx, leg_name: str, side: str):
    """Where this kerb's green STOPS, as (station_ft, inner_offset_ft, outer_offset_ft).

    READ OFF THE PAINT, NOT REBUILT FROM THE SECTION, and that is the whole reason this is a
    function rather than four lines in the caller. The lane's edges are on one of two datums
    depending on whether the section hugs the kerb (see AddBikeLane.paint's lane_edge_line), and
    where the green actually ends depends on the crossing that cut it - so a second construction
    of either would be a second chance to disagree with the first, which is the failure this
    module has paid for repeatedly. The pieces are already in ctx by the time an extension is
    painted (paint_group orders it after), so the lane's own answer is simply there to be asked.

    Offsets come back SIGNED, in the leg's own frame - the convention model.point_at and
    band_from_offsets share - so the caller places points with them directly instead of
    recovering a side from their magnitude.

    None where this kerb has no green at all: a leg the corridor could not fit a section on. A
    facility that breaks at a junction must not have an extension drawn across it, because the
    drawing would then assert a continuity the design does not have.
    """
    import numpy as np

    from src.geometry.markings import BIKE_LANE_SURFACE
    from src.geometry.model import station_offset_many

    centerline = ctx.state.legs[leg_name].centerline
    stations, offsets = [], []
    for piece in ctx.pieces:
        if piece.kind is not BIKE_LANE_SURFACE or piece.leg != leg_name:
            continue
        if str(piece.side) != side or piece.geometry.geom_type != "Polygon":
            continue
        station, offset = station_offset_many(
            centerline, np.asarray(piece.geometry.exterior.coords, dtype=float))
        stations.append(station)
        offsets.append(offset)
    if not stations:
        return None
    stations, offsets = np.concatenate(stations), np.concatenate(offsets)
    end_ft = float(stations.min())
    near = offsets[stations <= end_ft + LANE_END_FACE_SAMPLE_FT]
    if near.size < 2:
        return None
    # By MAGNITUDE, then carrying the sign: "inner" is the edge nearer the alignment and "outer"
    # the one against the kerb, and on a right-hand kerb both offsets are negative - so a plain
    # min/max would name them the wrong way round on exactly half the legs.
    inner_ft = float(near[np.argmin(np.abs(near))])
    outer_ft = float(near[np.argmax(np.abs(near))])
    return end_ft, inner_ft, outer_ft


@dataclass(frozen=True)
class ExtendBikeLaneThroughJunction(Treatment):
    """The lane's dotted green extension ACROSS the junction box - NACTO's crossbike.

    THE ONE OPENING THE TABLE COULD NOT REACH. markings.AT_AN_OPENING gives BIKE_LANE_SURFACE
    `DOTTED / DOTTED`, and PaintContext._dashes_across_openings has always laid the green back
    across a driveway and a side street. It could never lay it across THIS junction, for a
    geometric reason rather than a standards one, and STANDARDS.md said so: a lane is built leg
    by leg, so at the junction each leg's green simply ends at its own corner return and there is
    no single marking spanning the mouth for an extension to be the continuation OF. The dash
    machinery is explicit about refusing that case - see _dash_spans_along, "a lane that simply
    ENDS at an opening has nothing to extend", which is the guard that stopped the stubs. That
    guard is right and is untouched here. What was missing was the marking it should have been
    extending: this builds it.

    MEASURED, at the two junctions that need it. Broad & Greenwood's north kerb: the east leg's
    green stops at station 26.83 and the west leg's at 22.21, so 49 ft of a continuous corridor
    facility was drawn as nothing. W Broad & Louellen: 12.70 and 68.30, and 81 ft - the longer
    because that junction is a Y whose crossing is surveyed 43.7 deg off square. At E Broad &
    Princeton the corridor kerb has no mouth at all (Princeton Ave is a stem on the far side), so
    the two legs' green already runs on through the node and this correctly draws nothing.

    WHY STRAIGHT, over a span that long. The extension is RULED between the two end
    cross-sections - each edge a straight line from one leg's lane edge to the other's - rather
    than curved to follow the kerb between them. That is not a simplification: at a junction
    there IS no kerb between them, which is what the mouth means, and a crossing is drawn
    straight across the ground it crosses. The corridor legs at Louellen are 170.9 deg apart, so
    the chord runs about 3 ft off where a curve through the node would, well inside the lane's
    own 12 ft - and the alternative, a marking that curves through a junction because the street
    either side of it does, is exactly the wobble a striper does not paint.

    GREEN, THE TWO EDGE LINES AND THE CONTRAFLOW STRIPE - the whole crossbike NACTO asks for.
    The edge lines are the same construction as the green against a different pair of offsets:
    half a stripe outside each face, which is where BikeLane puts every other `*_line_ft` and
    the one thing that has to be got right here, because a line ruled along the face it marks
    paints half its body over the colour it bounds. See `rules` in paint.
    """
    paint_group: ClassVar[int] = 40
    target: AcrossTheJunction

    def describe(self) -> str:
        return (f"ExtendBikeLaneThroughJunction({self.target}): the bike lane's green carried "
                f"across the junction as a dotted lane extension")

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str:
        from src.geometry.model.corners import through_street_sides

        through = through_street_sides(state.legs)
        for leg_name, side in self.target.ends:
            if state.treatment_for(AddBikeLane, LegSide(leg_name, side)) is None:
                raise KeyError(
                    f"{leg_name} {side} has no bike lane, so there is nothing to extend across "
                    f"the junction from it. An extension is the CONTINUATION of a facility; "
                    f"drawn without one at both ends it would assert a corridor that breaks "
                    f"here (see CorridorFacility, which is where that refusal is reported).")
            if (leg_name, side) in through:
                # REFUSED RATHER THAN DRAWN EMPTY, and the difference is what state.notes claims.
                # A kerb running straight through has no mouth (model.junction_mouth_ft returns
                # None for exactly this set), so the two legs' green is already laid behind the
                # node and overlapped - THROUGH_JUNCTION_OVERLAP_FT - and the lane is continuous
                # without any extension. Applying one anyway painted nothing at E Broad &
                # Princeton while recording in the provenance that a crossbike had been added.
                # A note nobody can see contradicted is the failure mode this project keeps
                # finding; the refusal is caught and reported by CorridorFacility.
                raise ValueError(
                    f"{leg_name} {side} runs STRAIGHT THROUGH this junction - MUTCD 11th ed. "
                    f"3B.11(07)'s T-intersection case - so its kerb is never opened and the "
                    f"lane already carries across unbroken. There is no gap here to extend over.")
        return (" - MUTCD 11th ed. 9E.03(07) Standard (extensions through intersections shall be "
                "dotted) and 9E.06(15) Guidance (extension markings should cross intersections "
                "AND driveways for a buffer-separated lane); NACTO Urban Bikeway Design Guide, "
                "bidirectional lanes continue through intersections as a crossbike.")

    def paint(self, ctx) -> None:
        """One quad per dotted mark, ruled between the two lane ends and cut at the crossings."""
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union

        from src.geometry.markings import BIKE_LANE_DOTTED_EXTENSION, BIKE_LANE_SURFACE
        from src.geometry.model import point_at
        from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT, _dash_spans

        ends, rules = [], []
        for leg_name, side in self.target.ends:
            face = lane_end_face(ctx, leg_name, side)
            if face is None:
                return          # no lane on one end - see apply_to, and CorridorFacility's note
            station_ft, inner_ft, outer_ft = face
            centerline = ctx.state.legs[leg_name].centerline
            ends.append((point_at(centerline, station_ft, inner_ft),
                          point_at(centerline, station_ft, outer_ft)))
            # HALF A STRIPE OUTSIDE THE GREEN, WHICH IS WHERE AN EDGE LINE GOES. lane_end_face
            # reports the lane's two FACES - the green's own edges - and a line ruled along a
            # face lies half its body on the green: 0.41 ft of white over colour on every mark
            # of every crossbike, 38.0 sq ft at W Broad & Louellen. It is invisible in the plan
            # view, which strokes a line at a cosmetic 1.6 pt from its axis and has no width to
            # overlap with, and MarkingsDoNotCollide could not see it either until it learned to
            # stroke a line (see checks.py). BikeLane states the convention both its datums
            # encode - "each `*_line_ft` is the stripe's CENTRE, offset half a stripe outward
            # from the face it marks" - and it is exactly half a stripe in both, measured, on
            # every leg-side of every site. So this is that same step, not a second derivation
            # of it: the extension's rules land on the per-leg lines they continue instead of
            # stepping 0.41 ft in from them.
            #
            # SIGNED, and the sign is the SIDE's. Offsets come out of lane_end_face in the leg's
            # own frame, so on a right-hand kerb both are negative and "outward" is more
            # negative; taking the step off the magnitude would move the outer rule inward on
            # half the legs. outer_ft carries the side because it is the larger magnitude.
            outward_ft = float(np.copysign(LANE_EDGE_LINE_WIDTH_FT / 2, outer_ft))
            rules.append((point_at(centerline, station_ft, inner_ft - outward_ft),
                           point_at(centerline, station_ft, outer_ft + outward_ft)))
        (a_inner, a_outer), (b_inner, b_outer) = ends
        (a_rule_inner, a_rule_outer), (b_rule_inner, b_rule_outer) = rules
        # INNER TO INNER. Both ends are the same physical kerb seen from two approaches, so the
        # edge against the kerb on one leg is the edge against the kerb on the other; pairing
        # inner to OUTER would draw the extension as a bow tie crossing itself in the middle.
        length_ft = float(np.hypot(a_inner[0] - b_inner[0], a_inner[1] - b_inner[1]))
        if length_ft < MIN_EXTENSION_GAP_FT:
            return
        def _lerp(p, q, fraction: float):
            return (p[0] + (q[0] - p[0]) * fraction, p[1] + (q[1] - p[1]) * fraction)
        def across(fraction: float):
            return _lerp(a_inner, b_inner, fraction), _lerp(a_outer, b_outer, fraction)
        def rules_across(fraction: float):
            """The same parameter, on the two RULE lines rather than the two faces.

            Interpolated between the two ends' own rule points rather than stepped outward from
            the interpolated face, so each end lands exactly on the per-leg line it continues
            and the join has no step in it. The cost is a slight tilt: the two outward steps are
            the same length but not the same direction - each is normal to its own leg, and the
            legs at Louellen are 170.9 deg apart - so the rule is not quite parallel to the face
            chord and the clearance dips from 0.4101 ft to 0.4031 in the middle. That is 0.007 ft
            of stroke over the green, 0.08 in, against 0.410 ft before the rules existed.

            Stepping normal to the CHORD instead would hold the clearance exactly and put the
            same 0.007 ft into a kink at each join instead. Continuity wins: continuing the
            facility is what this marking is FOR, and checks.STROKE_ON_COLOUR_FRACTION is set
            knowing this 0.008 of a stripe is the largest such residual the design contains.
            """
            return (_lerp(a_rule_inner, b_rule_inner, fraction),
                     _lerp(a_rule_outer, b_rule_outer, fraction))
        # The lane's own green, so a mark meeting the end face is trimmed to butt against it
        # rather than lie over it. The face is where the paint stops, and where the paint stops is
        # a diagonal wherever a skewed crossing cut it - so the two touch along a slanted edge
        # that neither side can compute from the other's stations. Subtracting is the only join
        # that cannot leave a sliver of green painted twice, which MarkingsDoNotCollide reads as
        # the design asserting two things about one patch of ground.
        painted = unary_union([p.geometry for p in ctx.pieces
                                if p.kind is BIKE_LANE_SURFACE and p.geometry.geom_type == "Polygon"])
        # ONE SET OF SPANS FOR ALL THREE, which is the whole point of PaintContext.dash_phase on a
        # leg: "a lane's two edge lines and the green between them break at the same stations
        # instead of each being dashed along its own length and drifting out of phase". Here the
        # spans are simply shared directly - there is one parameter along the extension and every
        # marking on it is cut at the same two fractions, so they cannot drift.
        for start_ft, end_ft in _dash_spans(0.0, length_ft):
            near_inner, near_outer = across(start_ft / length_ft)
            far_inner, far_outer = across(end_ft / length_ft)
            mark = Polygon([near_inner, near_outer, far_outer, far_inner])
            if not mark.is_valid:
                mark = mark.buffer(0)
            # Half the mark it was meant to be, or it is not that mark. The two things that trim
            # one here - the crosswalk it gives way to, and the lane's own end face it butts
            # against - both cut on a diagonal, so what they leave is a wedge rather than a
            # shorter dash. See PaintContext.add_across_the_junction.
            ctx.add_across_the_junction(BIKE_LANE_SURFACE, mark.difference(painted),
                                         min_area_sq_ft=mark.area * MIN_MARK_FRACTION)
            # THE LANE'S TWO EDGES, on those same spans. Laid as BIKE_LANE_DOTTED_EXTENSION and
            # not as BIKE_LANE_EDGE_LINE, because that is the kind the edge line's own row names
            # for its dashes (AT_AN_OPENING: BIKE_LANE_EDGE_LINE is DOTTED, dotted_as=this) - "a
            # broken line is a different instruction from the solid one it continues". So the
            # marks here are the same kind the per-leg paint already lays across every driveway,
            # and both renderers draw them without being told anything new.
            #
            # ON THE RULES, NOT THE FACES - see where `rules` is built. A line drawn along the
            # face it marks lies half on the green.
            near_rule_inner, near_rule_outer = rules_across(start_ft / length_ft)
            far_rule_inner, far_rule_outer = rules_across(end_ft / length_ft)
            for near, far in ((near_rule_inner, far_rule_inner),
                               (near_rule_outer, far_rule_outer)):
                ctx.add_across_the_junction(
                    BIKE_LANE_DOTTED_EXTENSION, LineString([near, far]),
                    min_length_ft=(end_ft - start_ft) * MIN_MARK_FRACTION)
        self._carry_the_contraflow_stripe(ctx, a_inner, a_outer, b_inner, b_outer, length_ft)

    def _carry_the_contraflow_stripe(self, ctx, a_inner, a_outer, b_inner, b_outer,
                                      length_ft: float) -> None:
        """The yellow centre stripe, down the middle of the extension.

        NACTO asks for a dotted yellow centreline along a bidirectional lane AND through the
        crossbikes (STANDARDS.md), and the stripe's row in AT_AN_OPENING already reads CARRIED at
        an intersection - it just had nothing to be carried along here, for the same reason the
        green did not. Without it the crossbike says a lane runs through the junction and says
        nothing about it having two directions in it, which is the one fact a driver turning
        across it most needs: the rider bearing down on them may be coming from the direction
        they did not check.

        AT THE LEG'S OWN CADENCE, not re-dashed. CONTRAFLOW_DASH_FT / CONTRAFLOW_GAP_FT is the
        pattern either side, and the whole reason this stripe is CARRIED rather than DOTTED is
        that its own cadence IS the broken line the standard asks for - laying a second, finer
        rhythm inside the box would be the mistake that row was written to stop.
        """
        from shapely.geometry import LineString

        from src.geometry.markings import BIKE_CONTRAFLOW_DIVIDER

        def middle(inner, outer):
            return ((inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2)

        axis = LineString([middle(a_inner, a_outer), middle(b_inner, b_outer)])
        period_ft = CONTRAFLOW_DASH_FT + CONTRAFLOW_GAP_FT
        at_ft = 0.0
        while at_ft + CONTRAFLOW_DASH_FT <= axis.length:
            dash = shapely.ops.substring(axis, at_ft, at_ft + CONTRAFLOW_DASH_FT)
            ctx.add_across_the_junction(BIKE_CONTRAFLOW_DIVIDER, dash,
                                         min_length_ft=CONTRAFLOW_DASH_FT * MIN_MARK_FRACTION)
            at_ft += period_ft
