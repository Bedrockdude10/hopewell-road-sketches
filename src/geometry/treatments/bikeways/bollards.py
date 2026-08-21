"""WHAT STANDS IN THE BUFFER: the flex posts that make a painted lane a protected one.

A post's offset is read from the `AddBikeLane` underneath rather than re-derived, because the two
derivations disagreed once and 30 posts were drawn inside the bike lane. The lane's cross-section
is the single source for where the buffer is.
"""
from dataclasses import dataclass
from typing import ClassVar
from src.geometry.treatments.base import BOLLARD_DEFAULT_SPACING_FT, TARGET_LANE_WIDTH_FT, Treatment
from src.geometry.treatments.state import DesignState
from src.geometry.treatments.bikeways.place import AddBikeLane
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel

# The pitch of the flex posts down that buffer. Close enough to read as a continuous delineator
# rather than a row of dots, which is the whole difference between a painted lane and one a
# driver perceives as protected. Here rather than in a scenario file because it was in TWO of
# them under the same name and the same value, and a corridor whose posts are 8 ft apart on one
# leg and 12 on the next is one facility drawn as two.
BIKE_LANE_BOLLARD_SPACING_FT = 8.0


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
