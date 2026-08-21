"""CARRYING A LANE ACROSS THE JUNCTION: the crossbike and what closes the gap.

A bikeway that stops at the junction mouth and resumes on the far side is two facilities on the
sheet, so this extends one into the other. THE EXTENSION IS SURFACE, NOT JUST DOTS: measured, two
legs' green once sat 80.3 ft apart with 0.0 sq ft between them and dotted lines only, which reads
as a gap to every reader and to `--continuity`.

Above `place` because it needs the lanes it is joining to already exist on the state.
"""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import shapely.ops
from src.geometry.targets import AcrossTheJunction, LegSide
from src.geometry.treatments.base import Treatment
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.symbols import CONTRAFLOW_DASH_FT, CONTRAFLOW_GAP_FT
from src.geometry.treatments.bikeways.place import AddBikeLane
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

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
