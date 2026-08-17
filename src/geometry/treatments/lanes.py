"""Lane narrowing, and the flex posts that hold it.

Narrowing is the one treatment applied at every site, on every leg, in every scenario - it is
the traffic-calming baseline the rest is layered onto."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar


from src.geometry.targets import BOTH_SIDES, Side

if TYPE_CHECKING:                       # DesignState is layered above this module;
    from src.geometry.treatments.state import DesignState   # the annotation is a string
from src.geometry.treatments.base import (BOLLARD_DEFAULT_SPACING_FT,
                                          LANE_NARROWING_DEFAULT_STRIPE_FT, Treatment)



@dataclass(frozen=True)
class LaneNarrowing(Treatment):
    """Paint-only visual lane narrowing: a striped buffer/shoulder painted along
    one or both curbs of a leg (sides - see below). Zero curb/pavement
    geometry change - the lowest-cost alternative to a real curb
    extension, achieving the same 'narrower-looking travel way' cue with
    paint instead of concrete.

    line_only=True skips the diagonal chevron fill entirely - just the solid
    line (straight run + corner taper) delineating the outside of the real
    travel lane, nothing painted in the buffer itself. Useful as a debugging/
    comparison scenario (bare minimum lane-width marking, easy to check by eye
    or by measurement against the plan view without chevron hatch density
    affecting the read) as well as a real low-cost treatment option in its
    own right.

    sides restricts which side(s) of the leg get narrowed - defaults to both
    (the usual case: a real two-lane road narrowed symmetrically). Pass a
    single side (e.g. (Side.LEFT,)) when the OTHER side's edge is already owned
    by a different treatment - e.g. a marked-parking lane (MarkedParking)
    already delineates its own side; this just adds the matching plain
    delineating line on the opposite (entering-traffic) side, matching real
    curb-to-curb width there but with no buffer painted for it.

    The width bounds are the first validation this treatment has ever had. As a function it
    checked only that the leg existed, so a zero or negative stripe was a buffer with no
    width - it produced a degenerate polygon that the paint builder then had to guard against
    (see src/geometry/model.py:lane_narrowing_polygons_ft's 0.5 ft floor).
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 10
    paint_rank: ClassVar[int] = 0
    stripe_width_ft: float = LANE_NARROWING_DEFAULT_STRIPE_FT
    line_only: bool = False
    sides: tuple = BOTH_SIDES

    def __post_init__(self):
        if self.stripe_width_ft <= 0:
            raise ValueError(f"A lane-narrowing buffer needs a width; got "
                             f"stripe_width_ft={self.stripe_width_ft}.")
        object.__setattr__(self, "sides", tuple(Side(side) for side in self.sides))
        if not self.sides:
            raise ValueError("A lane narrowing with no sides paints nothing - pass at least one.")

    def describe(self) -> str:
        return (f"LaneNarrowing({self.target}, stripe_width_ft={self.stripe_width_ft}, "
                f"line_only={self.line_only}, sides={tuple(str(s) for s in self.sides)})")

    def paint(self, ctx) -> None:
        """An edge line, a hatched buffer, and a taper back to the kerb. line_only legs get the
        boundary lines without the fill."""
        from src.geometry.markings import (LANE_EDGE_LINE, LANE_NARROWING_FILL, TAPER_FILL,
                                           TAPER_LINE, ZONE_END_LINE)
        from src.geometry.model import (lane_narrowing_edge_lines_ft, lane_narrowing_polygons_ft,
                                        lane_narrowing_taper_ft, lane_narrowing_taper_polygons_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, _one, end_against_crossing,
                                        lane_edge_stripes, tapers_cleanly, zone_end_line_ft)

        leg_name = self.target.leg
        leg = ctx.state.legs[leg_name]
        stripe_width_ft = self.stripe_width_ft
        fill = not self.line_only
        for side in (str(s) for s in self.sides):
            at = ctx.anchors(leg_name, side,
                              inner_offset_ft=leg.curb_to_curb_ft / 2 - stripe_width_ft)
            # A crossing is something to end against: run into it and let it cut the end.
            # Only where there is none does the paint have to resolve itself back to the
            # kerb, and only then is a taper the right way to do it.
            if (leg_name, side) in ctx.straight_through:
                # One unbroken kerb under one restriction, with no corner return at either
                # end of it: run from the junction NODE and let any crossing cut it, keeping
                # both halves. Tested before the marked/unmarked split because it applies to
                # both - the two E Broad legs' north kerbs are one kerb, and the zones on
                # them have to meet at the node rather than each stopping a few feet short of
                # it. Discarding the junction-side half left ~20 ft of a no-stopping kerb
                # bare between the crossing and the node, with no corner there to justify it.
                start_ft, beyond_ft, curved = 0.0, None, False
            elif leg_name in ctx.marked:
                start_ft, beyond_ft = end_against_crossing(at)
                curved = False
            else:
                curved = tapers_cleanly(stripe_width_ft, at)
                start_ft, beyond_ft = (at.anchor_ft if curved else at.target_ft), None
            line_ft, fill_ft = lane_edge_stripes(stripe_width_ft)
            ctx.add(LANE_EDGE_LINE, _one(lane_narrowing_edge_lines_ft(
                leg, line_ft, start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,),
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2)), leg_name, side, beyond_ft)
            if curved:
                ctx.add(TAPER_LINE, _one(lane_narrowing_taper_ft(
                    leg, line_ft, at.anchor_ft, at.target_ft, sides=(side,))), leg_name, side)
            if fill:
                ctx.rim(ctx.add(LANE_NARROWING_FILL, _one(lane_narrowing_polygons_ft(
                    leg, fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                    sides=(side,))), leg_name, side, beyond_ft,
                    shares_a_kerb=(leg_name, side) in ctx.straight_through), LANE_EDGE_LINE)
                if curved:
                    ctx.add(TAPER_FILL, _one(lane_narrowing_taper_polygons_ft(
                        leg, fill_ft, at.anchor_ft, at.target_ft, sides=(side,))),
                        leg_name, side)
                elif leg_name not in ctx.marked and (leg_name, side) not in ctx.straight_through:
                    # Not on a kerb that runs straight through: the zone does not END at the
                    # junction node, it continues into the adjoining leg's zone on the same
                    # unbroken kerb. Closing it off drew a line across the hatching in the
                    # middle of the intersection.
                    ctx.add(ZONE_END_LINE, zone_end_line_ft(
                        leg, side, start_ft, leg.curb_to_curb_ft / 2 - fill_ft),
                        leg_name, side)


@dataclass(frozen=True)
class LaneNarrowingBollards(Treatment):
    """Plastic bollards (flex-post delineators) down the center of a leg's
    painted lane-narrowing buffer (LaneNarrowing) - a firmer, but still
    fully paint-plus-delineator (no curb/pavement change) escalation of that
    same treatment. Requires LaneNarrowing to already be applied to this
    leg - a bollard line only makes sense inside a buffer that exists, and its
    lateral placement (centered in that buffer) is derived from the buffer's
    own stripe_width_ft, not a separately-specified position."""
    paint_group: ClassVar[int] = 10
    paint_rank: ClassVar[int] = 1
    spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT

    def __post_init__(self):
        if self.spacing_ft <= 0:
            raise ValueError(f"Posts need a spacing; got spacing_ft={self.spacing_ft}.")

    def describe(self) -> str:
        return f"LaneNarrowingBollards({self.target}, spacing_ft={self.spacing_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        if state.treatment_for(LaneNarrowing, self.target) is None:
            raise KeyError(f"{self.target} has no lane-narrowing buffer - apply LaneNarrowing "
                            f"first. A row of posts is placed inside a buffer, so its lateral "
                            f"position comes from that buffer's own width.")

    def paint(self, ctx) -> None:
        """Down the centre of the buffer LaneNarrowing paints, on both sides it narrowed.

        The offset comes from that buffer's own stripe_width_ft rather than being specified
        again here, which is the whole reason this treatment requires one: a post placed off a
        separately-guessed offset is a post standing somewhere the buffer is not.
        """
        from src.geometry.markings import BOLLARD
        from src.geometry.model import bollard_points_ft, leg_clearance_ft
        from src.geometry.paint import PaintPiece, _dot

        leg_name = self.target.leg
        leg = ctx.state.legs[leg_name]
        narrowing = ctx.state.treatment_for(LaneNarrowing, self.target)
        stripe_width_ft = narrowing.stripe_width_ft
        sides = tuple(str(s) for s in narrowing.sides)
        for point in bollard_points_ft(
                leg, stripe_width_ft,
                leg_clearance_ft(leg_name, ctx.state.legs, ctx.state.corner_fillets),
                self.spacing_ft, sides=sides):
            ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, None))
