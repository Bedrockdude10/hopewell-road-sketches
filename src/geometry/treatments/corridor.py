"""A FACILITY THAT RUNS ALONG A ROUTE, declared once and applied at every junction on it.

A treatment targets one kerb of one approach, which is right: that is the ground a design has
measurements for. A FACILITY IS NOT LIKE THAT. "A 12 ft two-way protected bike lane along Broad
Street's south kerb, the length of the borough" is one decision about one route, and the three
junctions this project models are places it passes through, not three studies that each happen to
reach the same conclusion.

It was written as three studies. `build_proposal_two_way_bike_lane` existed in
sites/broad_st_greenwood, sites/ebroad_princeton and sites/wbroad_louellen, each with its own leg
loop, its own translation of the kerb to a side, its own bollard call and its own policy about what
to do when the section does not fit. `CORRIDOR_LANE_WIDTH_FT` was defined verbatim in two of them.
Only Louellen had NACTO's constrained fallback, so the corridor was allowed to narrow at one
junction and required to break at the others - a rule nobody chose, arrived at by which file got
edited.

THIS IS THE THIRD TIME THAT SHAPE OF BUG HAS BEEN PAID FOR HERE, and the first two are already
written down a few lines apart: CORRIDOR_SIDE moved into bikeways.py because it was spelled out in
three site files and editing one would have switched the route to the other kerb at one junction
with every drawing still looking locally right; parking.py:hold_travel_lane_at_target moved into
src because broad_st_greenwood grew it inline and ebroad_princeton did not, which left E Broad with
11.68 ft and 13.21 ft travel lanes. Each fix pulled out the constant or the rule and left the LOOP
duplicated. This pulls out the loop.

WHAT IS STILL PER SITE, and rightly: which of a junction's other legs get their OSM parking, what
the surrounding scenario says, and the notes about that particular junction. A facility is a
statement about the route. Everything a junction knows and the route does not stays where it is.
"""
from dataclasses import dataclass

from src.geometry.model import side_facing
from src.geometry.network import _street_name
from src.geometry.targets import LegSide, Side
from src.geometry.treatments.bikeways import (BIKE_LANE_BOLLARD_SPACING_FT,
                                              CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                              CORRIDOR_SIDE, MIN_TWO_WAY_BIKE_LANE_FT,
                                              TWO_WAY_BIKE_LANE_BUFFER_FT, AddBikeLaneBollards,
                                              AddTwoWayBikeLane)
from src.geometry.treatments.parking import hold_travel_lane_at_target
from src.geometry.treatments.state import DesignState


@dataclass(frozen=True)
class Section:
    """One cross-section the facility will accept, and whether accepting it costs something.

    A LADDER of these rather than one section, because a route crosses streets of different
    widths and the interesting output is WHICH RUNG each junction lands on. `constrained` is not
    a flag on the geometry - AddTwoWayBikeLane already carries that - it is the marker that this
    rung is a concession, so landing on it prints what was given up rather than passing silently
    as if the standard section had fitted.
    """
    width_ft: float
    buffer_ft: float
    constrained: bool = False


@dataclass(frozen=True)
class CorridorFacility:
    """A route-level decision: this section, along this kerb of this street, wherever it fits.

    `road` is a street name as network._street_name normalises it - "West Broad Street (southwest
    of Louellen St) - CR 654" and "East Broad Street (east of Greenwood Ave)" both reduce to "Broad
    Street", which is what makes one declaration reach six approaches across three junctions and
    two SRIs. Selecting by NAME rather than by SRI is deliberate: Broad St is CR 518 through
    Greenwood and CR 654 southwest of Louellen, so a route identified by its road number would
    stop at the junction where the number changes, which is exactly the junction the corridor most
    needs to carry.

    `side` is a COMPASS kerb, translated per approach by side_facing. A leg's left/right is in its
    own frame, so the same physical kerb is "left" on one approach and "right" on the next, and
    doing that translation by hand is how a facility ends up on the north kerb of one leg and the
    south kerb of the one after it.
    """
    road: str
    side: str
    sections: tuple[Section, ...]
    bollard_spacing_ft: float = BIKE_LANE_BOLLARD_SPACING_FT

    def legs_on(self, model) -> list[str]:
        """This junction's approaches that lie on this route, in a stable order.

        Read off each leg's configured `street_name`, not off a per-site list of leg names. The
        three site files each kept their own list - a tuple, a different tuple, and a substring
        test on the leg name - and a list of names is a second record of a fact the config already
        states, free to disagree with it the moment a leg is renamed.
        """
        legs_cfg = model.config.get("legs", {})
        return sorted(name for name, cfg in legs_cfg.items()
                      if name in model.legs
                      and _street_name(cfg.get("street_name", "")) == self.road)

    def apply_to(self, state: DesignState, model, quiet: bool = False) -> DesignState:
        """Place the facility on every approach of this junction that is on the route.

        Each approach takes the first section that fits. WHERE IT LANDS ON THE LADDER, AND WHERE
        IT FALLS OFF, IS THE OUTPUT - a corridor plan that quietly stops at its hardest point is
        the plan nobody costed - so every rung refused and every concession accepted is reported
        against the measurement that decided it.
        """
        for leg_name in self.legs_on(model):
            try:
                side = side_facing(state.legs[leg_name], self.side)
            except ValueError:
                # A leg that faces neither way - a stem meeting the route square on. It is on the
                # street by name and has no kerb on this side to carry the facility.
                continue
            state = self._place_on(state, leg_name, side, quiet)
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
            state = state.apply(AddBikeLaneBollards(LegSide(leg_name, side),
                                                    spacing_ft=self.bollard_spacing_ft))
            # THE FAR KERB GETS THE SURPLUS, and the two belong together: the kerb that loses its
            # parking to the bike lane is not the kerb that gains this. One definition, in
            # parking.py - see its docstring for what having two cost.
            return hold_travel_lane_at_target(state, leg_name, str(Side(side).other))

        if not quiet:
            print(f"  NOTE: {leg_name} {side} carries NO two-way lane. THIS IS WHERE THE "
                  f"BOROUGH-LENGTH CORRIDOR BREAKS - riders would rejoin the carriageway through "
                  f"this junction. The refusals above say which limit stopped it here, which is "
                  f"not the same limit everywhere.")
        return state


#: THE BOROUGH'S TWO-WAY PROTECTED BIKEWAY, declared once. Every junction on Broad Street reads
#: this; no site file restates any part of it.
#:
#: The ladder is NACTO's standard section, then NACTO's constrained one, and the BUFFER NEVER
#: GIVES. Narrowing the buffer instead of the lane spends the protection to buy nothing: at W
#: Broad & Louellen, 32.10 ft between traced kerbs, the 10 + 3 section leaves 9.14 ft travel lanes
#: (under NJDOT's 10 ft traffic-calming floor) and 10 + 2 leaves 9.64 - still short. The
#: constrained 8 ft lane with the full 3 ft buffer leaves 10.14 ft: the pinch keeps its posts and
#: the corridor stays continuous.
#:
#: TEN FEET ON THE FIRST RUNG, NOT THE 12 FT DESIGN WIDTH, AND PARKING IS WHY. Hopewell Borough is
#: car-dependent; a corridor plan that removes a kerb of parking and returns none is not viable
#: here whatever it does for riders. broad_st_east has 43.26 ft between its traced kerbs, and 12 ft
#: of lane plus a 3 ft buffer plus two 11 ft travel lanes leaves 5.44 ft against the far kerb -
#: under a stall, so the whole leg came out with no parking at all. At 10 ft the section leaves
#: 7.44 ft, which is a usable stall. 10 ft is NACTO's MINIMUM (12 ft desirable): two riders can
#: pass, but an oncoming pair is tight. That is a real cost, paid deliberately to keep the parking;
#: the alternative on the table was narrowing the travel lanes to 10 ft to keep a 12 ft lane, and
#: it was not taken, so the travel lanes hold 11 ft.
#:
#: An unbuffered rung was tried and removed. It fits the kerb and leaves the opposing lane 13.02 ft
#: with no room to narrow (TravelLanesHoldTheTarget fails the build), and an unbuffered two-way
#: lane is paint beside 25 mph traffic rather than protection.
BROAD_ST_TWO_WAY_BIKEWAY = CorridorFacility(
    road="Broad Street",
    side=CORRIDOR_SIDE,
    sections=(Section(MIN_TWO_WAY_BIKE_LANE_FT, TWO_WAY_BIKE_LANE_BUFFER_FT),
              Section(CONSTRAINED_TWO_WAY_BIKE_LANE_FT, TWO_WAY_BIKE_LANE_BUFFER_FT,
                      constrained=True)),
)
