"""WHAT A BIKEWAY IS IN CROSS-SECTION, and every standards figure that sizes one.

A section can be asked whether it fits a kerb without anything being applied, which is what lets a
scenario try the standard section, fall back to the constrained one, and report which it got. So
the section is separated from the treatment that places it, and it is the leaf of this package.

THE FIGURES LIVE HERE AND NOWHERE ELSE. AASHTO gives two widths for an exclusive lane and this
file keeps both, because a caller that collapses them to one silently changes which streets are
buildable - see MIN_BIKE_LANE_FT. Anything published from here belongs in STANDARDS.md too.
"""
from dataclasses import dataclass
from src.geometry.treatments.base import LANE_WIDTH_SLACK_FT, TARGET_LANE_WIDTH_FT

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
# Below this the two travel lanes are no longer lanes. NACTO's urban minimum is 10 ft, and
# TARGET_LANE_WIDTH_FT (11) is what this project designs to; a corridor that cannot hold two
# 10 ft lanes beside the section is reported rather than drawn.
MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT = 10.0


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
        # LANE_WIDTH_SLACK_FT, like every other width-against-room comparison in this package
        # (AddBikeLane.apply_to, CurbExtension, the parking surplus). Its absence here was the
        # anomaly, and it is not a free concession on the floor: 0.05 ft is 0.6 inch, against two
        # half-widths read off a kerb traced from aerial imagery and interpolated on a 2 ft grid.
        # W Broad's southwest approach on a 3x sheet came to 9.9964 ft per travel lane at one
        # station of 169 - short of the floor by four THOUSANDTHS of a foot - and that refusal
        # denied a protected bikeway over the whole 335 ft approach. Refusing at that margin is
        # false precision about a measurement, not fidelity to NACTO.
        if travel_way_ft / 2 < MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT - LANE_WIDTH_SLACK_FT:
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
