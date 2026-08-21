"""A FACILITY THAT RUNS ALONG A ROUTE, declared once and applied at every junction on it.

A Treatment targets one kerb of one approach, which is the right scope for a cross-section. A
facility is not: "a two-way protected bike lane along Broad Street's south kerb, the length of
the borough" is ONE decision about one route, and the junctions it passes through are not three
studies that each happen to reach the same conclusion. Declaring it per site duplicated the leg
loop, the kerb-to-side translation and the not-fitting policy, which is how one junction got
NACTO's constrained fallback and the others did not.

WHAT STAYS PER SITE: which of a junction's other legs get their OSM parking, the surrounding
scenario, and the notes about that junction. A facility states the route; everything a junction
knows and the route does not stays where it is.
"""
from dataclasses import dataclass

from src.geometry.model import side_facing
from src.geometry.network import _street_name
from src.geometry.targets import AcrossTheJunction, LegSide, Side
from src.geometry.treatments.bikeways import (BIKE_LANE_BOLLARD_SPACING_FT,
                                              CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                              CORRIDOR_SIDE, MIN_TWO_WAY_BIKE_LANE_FT,
                                              TWO_WAY_BIKE_LANE_BUFFER_FT, AddBikeLane,
                                              AddBikeLaneBollards,
                                              AddTwoWayBikeLane,
                                              ExtendBikeLaneThroughJunction)
from src.geometry.treatments.parking import hold_travel_lane_at_target
from src.geometry.treatments.state import DesignState
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel


@dataclass(frozen=True)
class Section:
    """One cross-section the facility will accept, and whether accepting it costs something.

    A LADDER of these rather than one section: a route crosses streets of different widths and
    the interesting output is WHICH RUNG each junction lands on. `constrained` marks a rung as a
    concession - not a geometry flag, AddTwoWayBikeLane already carries that - so landing on it
    prints what was given up rather than passing silently as if the standard section had fitted.
    """
    width_ft: float
    buffer_ft: float
    constrained: bool = False


@dataclass(frozen=True)
class CorridorFacility:
    """A route-level decision: this section, along this kerb of this street, wherever it fits.

    `road` is a street name as network._street_name normalises it. BY NAME, NOT BY SRI: Broad St
    is CR 518 through Greenwood and CR 654 southwest of Louellen, so a route keyed on the road
    number would stop at the junction where the number changes - the junction the corridor most
    needs to carry.

    `side` is a COMPASS kerb, translated per approach by side_facing, because a leg's left/right
    is in its own frame: the same physical kerb is "left" on one approach and "right" on the next.
    """
    road: str
    side: str
    sections: tuple[Section, ...]
    bollard_spacing_ft: float = BIKE_LANE_BOLLARD_SPACING_FT

    def legs_on(self, model: "IntersectionModel") -> list[str]:
        """This junction's approaches that lie on this route, in a stable order.

        Read off each leg's configured `street_name`, never a per-site list of leg names: that
        would be a second record of a fact the config already states, free to disagree with it
        the moment a leg is renamed.
        """
        legs_cfg = model.config.get("legs", {})
        return sorted(name for name, cfg in legs_cfg.items()
                      if name in model.legs
                      and _street_name(cfg.get("street_name", "")) == self.road)

    def apply_to(self, state: DesignState, model: "IntersectionModel", quiet: bool = False) -> DesignState:
        """Place the facility on every approach of this junction that is on the route.

        Each approach takes the first section that fits. WHERE IT LANDS ON THE LADDER, AND WHERE
        IT FALLS OFF, IS THE OUTPUT - a corridor plan that quietly stops at its hardest point is
        the plan nobody costed - so every rung refused and every concession accepted is reported
        against the measurement that decided it.
        """
        carrying = []
        for leg_name in self.legs_on(model):
            try:
                side = side_facing(state.legs[leg_name], self.side)
            except ValueError:
                # A leg that faces neither way - a stem meeting the route square on. It is on the
                # street by name and has no kerb on this side to carry the facility.
                continue
            state = self._place_on(state, leg_name, side, quiet)
            # ASKED OF THE DESIGN, not inferred from a treatment count: _place_on applies
            # several things on success and none on failure, so a count would stop meaning "this
            # approach carries the facility" the moment any of them moved. isinstance matching
            # makes AddTwoWayBikeLane answer to AddBikeLane, which is the question being asked.
            if state.treatment_for(AddBikeLane, LegSide(leg_name, side)) is not None:
                carrying.append((leg_name, side))
        return self._carry_through_the_junction(state, carrying, quiet)

    def _carry_through_the_junction(self, state: DesignState, carrying: list, quiet: bool
                                     ) -> DesignState:
        """Join the approaches' lanes across the junction box - NACTO's crossbike.

        HERE AND NOT IN _place_on: the extension lies between two legs, in neither leg's frame,
        so it belongs to no approach.

        ONLY BETWEEN LANES THAT WERE ACTUALLY PLACED. `carrying` is the approaches that took a
        section, not the ones the route passes through, and the difference is the junction where
        the corridor breaks - extending from a leg that refused every rung would paint a
        continuous facility across the one place it stops. Below two there is nothing to join.
        """
        if len(carrying) < 2:
            return state
        if len(carrying) > 2 and not quiet:
            # A route crossing its own junction has two approaches. Three would mean a street
            # meeting itself, and pairing them by hand is a guess about which two are opposite.
            print(f"  NOTE: {self.road} has {len(carrying)} approaches carrying the facility at "
                  f"this junction, and a lane extension joins a PAIR. Drawn between "
                  f"{carrying[0][0]} and {carrying[1][0]}; check the others by eye.")
        (leg_a, side_a), (leg_b, side_b) = carrying[0], carrying[1]
        try:
            return state.apply(ExtendBikeLaneThroughJunction(
                AcrossTheJunction(leg_a, side_a, leg_b, side_b)))
        except ValueError as no_gap:
            # The facility's kerb is never opened here - the stem is on the far side, so the lane
            # runs through unbroken and there is nothing to extend. Reported, not swallowed: "no
            # crossbike here" is a fact about the junction.
            if not quiet:
                print(f"  NOTE: no lane extension across this junction - {no_gap}")
            return state

    def _place_on(self, state: DesignState, leg_name: str, side: str, quiet: bool) -> DesignState:
        for section in self.sections:
            try:
                state = state.apply(AddTwoWayBikeLane(
                    LegSide(leg_name, side), width_ft=section.width_ft,
                    buffer_ft=section.buffer_ft, constrained=section.constrained))
            except ValueError as too_narrow:
                if not quiet:
                    print(f"  NOTE: {leg_name} {side} ({self.side} kerb) cannot take a "
                          f"{section.width_ft:.0f} ft lane with a {section.buffer_ft:.0f} ft "
                          f"buffer - {too_narrow}")
                continue
            if section.constrained and not quiet:
                print(f"  NOTE: {leg_name} {side} carries NACTO's CONSTRAINED "
                      f"{section.width_ft:.0f} ft two-way width, not the "
                      f"{self.sections[0].width_ft:.0f} ft minimum. At {section.width_ft:.0f} ft "
                      f"two riders cannot pass an oncoming pair - a real cost, accepted because "
                      f"the alternative is a gap in the route. The full {section.buffer_ft:.0f} ft "
                      f"buffer is kept, so it stays a protected lane.")
            # POSTS NEED A BUFFER TO STAND IN. On an unbuffered rung there is nowhere to put one
            # that is not in the bike lane or the travel lane, so the lane is painted rather than
            # protected here - said out loud, because "protected bikeway" is the claim the whole
            # facility makes and this is the one place it does not hold.
            from src.geometry.treatments.bikeways import min_bike_lane_buffer_ft

            if section.buffer_ft >= min_bike_lane_buffer_ft():
                state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                        spacing_ft=self.bollard_spacing_ft))
            elif not quiet:
                print(f"  NOTE: {leg_name} {side} takes the full {section.width_ft:.0f} ft lane "
                      f"with a {section.buffer_ft:.1f} ft buffer - too narrow for a flex post, so "
                      f"there are NO posts here and the lane is PAINTED, not protected. The kerb "
                      f"is still alongside it. Width was kept over the buffer deliberately - see "
                      f"BROAD_ST_TWO_WAY_BIKEWAY.")
            # THE FAR KERB GETS THE SURPLUS: the kerb that loses its parking to the bike lane is
            # not the kerb that gains this. Single home: hold_travel_lane_at_target in parking.py.
            return hold_travel_lane_at_target(state, leg_name, str(Side(side).other))

        if not quiet:
            print(f"  NOTE: {leg_name} {side} carries NO two-way lane. THIS IS WHERE THE "
                  f"BOROUGH-LENGTH CORRIDOR BREAKS - riders would rejoin the carriageway through "
                  f"this junction. The refusals above say which limit stopped it here, which is "
                  f"not the same limit everywhere.")
        return state


#: THE BOROUGH'S TWO-WAY PROTECTED BIKEWAY, declared once; no site file restates any part of it.
#:
#: THE BUFFER NEVER GIVES, AND THE WIDTH DOES. "Protected" is the claim the whole facility makes,
#: so a rung that drops the buffer is not a cheaper version of this design - it is a different
#: design, paint beside 30 mph traffic, on the corridor's hardest junction where riders need the
#: separation most. The width is the thing with a graduated answer: 5 ft per direction where the
#: street allows it, 4 ft where it does not, and the concession stated on the drawing either way.
#:
#: This reverses a width-before-buffer ordering that put an unbuffered 10 ft rung ahead of the
#: constrained 8 ft one, on the argument that 5 ft per direction with the kerb alongside beats 4 ft
#: per direction with posts. What that bought at W Broad & Louellen was 5 ft per direction and NO
#: PROTECTION AT ALL, on both W Broad approaches - the junction the corridor most needs to carry.
#: The constrained rung fits there: 60 posts, travel lanes still over the 10 ft floor.
#:
#: NO INTERMEDIATE PART-BUFFER RUNG. Nothing on this route has room for one: a 10 ft lane with a
#: 2 ft buffer leaves 9.58 ft travel lanes on Louellen's northeast approach, still under the floor,
#: so the rung would refuse everywhere the full one does and add a width with no standing.
#:
#: TEN FEET ON THE FIRST RUNG, AND PARKING IS WHY. Hopewell Borough is car-dependent, so a plan
#: that removes a kerb of parking and returns none is not viable here. Over its first 170 ft
#: broad_st_east has 43.26 ft between traced kerbs: a 12 ft lane plus 3 ft buffer plus two 11 ft
#: travel lanes leaves 5.44 ft at the far kerb - under MIN_USABLE_STALL_FT, so the leg came out
#: with no parking at all. At 10 ft it leaves 7.44 ft, a usable stall.
#:
#: THAT PAIR OF FIGURES IS A 170 FT MEASUREMENT, AND IT DOES NOT SURVIVE THE WHOLE LEG. Measured
#: over all 374 ft the sheet draws, broad_st_east is 39.95 ft between kerbs and the 10 ft rung
#: leaves 4.13 ft - under a stall, so this leg is hatched on the drawing whichever rung it takes,
#: and the 12 ft one would leave 2.13 ft. So the argument above is the reason the FIRST rung is 10
#: and not 12, and it is no longer a stall this corridor actually gets: the parking it does keep is
#: broad_st_west's, 8 ft deep, which the section never threatened. Worth knowing before the width
#: is defended on parking grounds again - and worth reopening 12 ft on, which is 1 ft off NACTO's
#: ask, if nothing else claims that 2 ft. Note also WHY the long span is tighter: the section is
#: sized on the narrowest pinch anywhere along the leg, so one tight spot 300 ft out sets the
#: width for all 374 ft. A section that varied by station would keep both; none of this does.
#:
#: THE 10 FT RUNG IS NOT A NACTO WIDTH: NACTO asks at least 13 ft and gives 8 ft as the absolute
#: minimum, so this rung has no standing in the guide and the 8 ft rung is the absolute floor being
#: spent (STANDARDS.md 4). The alternative on the table was narrowing the travel lanes to 10 ft to
#: keep 12 ft of bikeway; it was not taken - see TARGET_LANE_WIDTH_FT.
#:
#: WHICH RUNG A LEG LANDS ON MOVES WITH THE SHEET, and that is the design working rather than
#: leaking. A sheet that shows more street asks the question of more street: W Broad's southwest
#: approach measures 20.32 ft over 130 ft and 16.58 ft over 325 ft, off a real pinch 318 ft out, and
#: takes the constrained rung on the sheet that shows it. The alternative was tried - freeze the
#: measurement at a configured span so the rung cannot move - and it bought frame-invariance with a
#: worse lie, a section sized over 130 ft and DRAWN over 325 ft whose paint was then trimmed off at
#: the first station it stopped fitting. broad_st_east carried green over 180 ft of a 425 ft leg.
#: A treatment applies to the street in the drawing; extent is the load-bearing claim, and the rung
#: is what gives to keep it. What may NOT move with the sheet is whether an approach is PROTECTED -
#: every rung here keeps the full buffer, which is what makes that safe.
BROAD_ST_TWO_WAY_BIKEWAY = CorridorFacility(
    road="Broad Street",
    side=CORRIDOR_SIDE,
    # Every rung keeps the full buffer; only the width steps down. See above.
    sections=(Section(MIN_TWO_WAY_BIKE_LANE_FT, TWO_WAY_BIKE_LANE_BUFFER_FT),
              Section(CONSTRAINED_TWO_WAY_BIKE_LANE_FT, TWO_WAY_BIKE_LANE_BUFFER_FT,
                      constrained=True)),
)
