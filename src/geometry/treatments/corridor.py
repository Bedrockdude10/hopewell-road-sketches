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

TWO KINDS OF ROUTE DECISION LIVE HERE, because a street that cannot hold a new facility still
gets treated along its whole length: CorridorFacility places a new cross-section wherever it
fits, and CorridorCalming names the legs a route's existing cross-section is narrowed on. Both
answer "which approaches of this junction are on this street" the same way, off the configured
street name - see legs_on_road, which is the one place that question is asked.
"""
from dataclasses import dataclass

import numpy as np

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
from src.geometry.treatments.parking import (hold_travel_lane_at_target,
                                             osm_derived_baseline)
from src.geometry.treatments.state import DesignState
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel


def legs_on_road(model: "IntersectionModel", road: str) -> list[str]:
    """One junction's approaches that lie on a named route, in a stable order.

    Read off each leg's configured `street_name`, never a per-site list of leg names: that would
    be a second record of a fact the config already states, free to disagree with it the moment a
    leg is renamed. `road` is normalised as network._street_name normalises it, so the compass
    halves of one street ("East Broad Street", "West Broad Street") answer to one route.

    Two route-level decisions ask this - a facility and a calming - so it is a function rather
    than a method on either.
    """
    legs_cfg = model.config.get("legs", {})
    return sorted(name for name, cfg in legs_cfg.items()
                  if name in model.legs
                  and _street_name(cfg.get("street_name", "")) == road)


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
        """This junction's approaches that lie on this route, in a stable order."""
        return legs_on_road(model, self.road)

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

    def _reach_on(self, state: DesignState, leg_name: str, side: str
                   ) -> tuple[float | None, str | None]:
        """How far up this kerb the facility runs CONTINUOUSLY FROM THE JUNCTION, and why not further.

        Returns (to_ft, refused). `to_ft` is None for "the whole traced kerb", which is the answer
        wherever every station fits and is what keeps an approach the street can carry byte-for-byte
        as it was. A non-None `refused` means this approach carries nothing at all and says what
        stopped it.

        PER STATION, THROUGH section_at, WHICH IS THE RENDERER'S OWN PREDICATE. The whole-leg
        minimum this replaces let one station veto an entire approach: at W Broad & Louellen on a
        3x sheet, 168 of 169 stations on the southwest approach fit and station 363.6 - where the
        travel way measures 31.813 ft against the 31.820 the constrained rung needs, short by seven
        THOUSANDTHS of a foot - refused all 335 ft, including the 307 ft where the FULL rung fits.

        CONTINUITY FROM THE JUNCTION IS THE PROPERTY, which is why this reports a reach and not a
        set of runs. A corridor exists to be ridden through the junction: a stretch of room 200 ft
        out that cannot be reached without rejoining traffic is not coverage, and the crossbike
        (_carry_through_the_junction) has to start from a lane that is actually at the node.

        SO A SECOND FITTING RUN BEYOND THE BREAK IS REFUSED, NOT DRAWN, and that is a real
        limitation rather than a judgement about the street. Drawing it needs a SECOND section on
        one approach, and the divider shift (DesignState.travel_lane_divider_shift), the post row
        and the far kerb's surplus are each keyed one-per-approach today - two rungs on one leg
        would give the reader one centre stripe for two different sections. corridor_paint._collect
        already draws every fitting run of the between-junction strip; bringing that here is the
        next step, and the refusal below records the span so it is a span to pick up rather than
        one to rediscover.
        """
        from src.geometry.model import curb_station_span
        from src.geometry.treatments.bikeways import (MIN_FACILITY_RUN_FT, section_at,
                                                      travel_way_profile)
        from src.geometry.treatments.state import FacilityRefusal

        leg = state.legs[leg_name]
        profile = travel_way_profile(leg, side)
        if profile is None:
            # Nothing traced on one of the two sides, so there is no station-by-station
            # measurement to split on and the nominal width is all there is. Whole leg, as before.
            return None, None
        stations, near_ft, far_ft = profile
        # ON THE RUN'S OWN TWO MINIMA, NOT ON THE LOCAL CROSS-SECTION. section_at(near(s), far(s))
        # asks whether the STREET holds a rung at that station; what gets drawn is one
        # constant-width section placed off the near kerb's minimum over the whole run, so a
        # station holds the facility only if the run REACHING it does. Asked per station the two
        # arithmetics disagreed - _place_on then sized the rung through governing_half_widths_ft
        # over [0, reach], which could refuse a section every station had individually approved,
        # and the approach went bare with no refusal recorded to say why.
        #
        # Accumulating the near minimum also makes `fits` monotone, so "the first station that
        # fails is where the ride stops" is exact rather than nearly true: a run that has failed
        # cannot start fitting again by getting longer.
        #
        # THE FAR SIDE IS THE WHOLE LEG'S MINIMUM AND NOT THE RUN'S, because the divider this
        # section implies is drawn to the end of the leg even where the green stops - see
        # governing_half_widths_ft, which this has to agree with station for station or _place_on
        # refuses a rung this approved. It makes the predicate STRICTER as the tail gets narrower,
        # which is the point: a reach is only worth having if the centre stripe it puts on the
        # street holds for its whole length.
        run_near = np.minimum.accumulate(near_ft)
        run_far = np.full_like(far_ft, far_ft.min())
        fits = np.array([section_at(self, float(near), float(far))[0] is not None
                         for near, far in zip(run_near, run_far)])
        if fits.all():
            return None, None

        # The first station that fails is where the ride stops; the reach is the one before it.
        broke = int(np.argmin(fits))
        reach_ft = float(stations[broke - 1]) if broke else float(stations[0])
        # Quoted off the same accumulated pair the predicate refused, so the measurement in the
        # message is the binding one - see SKILLS 0a.
        _section, why = section_at(self, float(run_near[broke]), float(run_far[broke]))
        # THE NARROWEST STATION IN THE TAIL, not at the break: the break is where continuity ends
        # and the pinch that a wider design would have to solve can be anywhere beyond it. Both
        # figures go in, because a refusal whose measurement is not the binding one is the mistake
        # SKILLS 0a is about.
        tail = slice(broke, None)
        tail_total = near_ft[tail] + far_ft[tail]
        narrowest_ft = float(tail_total.min())
        span = curb_station_span(leg, side)
        tail_end_ft = float(stations[-1]) if span is None else max(float(stations[-1]), span[1])
        reason = (f"the facility cannot continue past station {stations[broke]:.1f} ft - {why} "
                  f"Beyond it this kerb runs to {tail_end_ft:.1f} ft and the street is "
                  f"{narrowest_ft:.2f} ft between kerbs at its narrowest there; any of it that "
                  f"does hold a rung cannot be reached from this junction without putting riders "
                  f"back in traffic, so it is left bare deliberately.")
        state.refuse(leg_name, side, FacilityRefusal(
            float(stations[broke]), tail_end_ft, reason, narrowest_ft))

        covered_ft = reach_ft - float(stations[0])
        if covered_ft < MIN_FACILITY_RUN_FT:
            # The head is bare too, and for a DIFFERENT reason - it fits, there is just not enough
            # of it. Recorded separately rather than folded into the tail's refusal: one span says
            # the street is too narrow and the other says the stretch is too short, and a reader
            # given one reason for both would go looking for a pinch that is not there.
            too_short = (f"{covered_ft:.0f} ft of continuous room from the junction, under the "
                         f"{MIN_FACILITY_RUN_FT:.0f} ft a usable facility needs - a rider cannot "
                         f"use it and drawing it would invite the reader to count it as coverage")
            state.refuse(leg_name, side, FacilityRefusal(
                float(stations[0]), float(stations[broke]), too_short))
            return None, f"{too_short}. Beyond that, {reason}"
        return reach_ft, None

    def _place_on(self, state: DesignState, leg_name: str, side: str, quiet: bool) -> DesignState:
        to_ft, refused = self._reach_on(state, leg_name, side)
        if refused is not None:
            # _reach_on has already recorded the span; nothing else is claimed on this kerb.
            if not quiet:
                print(f"  NOTE: {leg_name} {side} ({self.side} kerb) carries NO two-way lane - "
                      f"{refused}")
            return state
        if to_ft is not None and not quiet:
            print(f"  NOTE: {leg_name} {side} carries the facility to station {to_ft:.0f} ft and "
                  f"no further. The rung below is sized over THAT stretch on THIS kerb and over "
                  f"the WHOLE approach on the far one, because the centre stripe it shifts runs "
                  f"the full leg while the green, the posts and the buffer end together at "
                  f"{to_ft:.0f} ft - see the refusal recorded on this kerb for what stopped it.")
        for section in self.sections:
            try:
                state = state.apply(AddTwoWayBikeLane(
                    LegSide(leg_name, side), width_ft=section.width_ft,
                    buffer_ft=section.buffer_ft, constrained=section.constrained, to_ft=to_ft))
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

        # RECORDED, NOT ONLY PRINTED, and over the whole kerb: every rung was refused, so there
        # is no span of this approach the design is claiming. A check reading stdout is not a
        # check - see DesignState.facility_refusals.
        from src.geometry.model import curb_station_span
        from src.geometry.treatments.state import FacilityRefusal

        span = curb_station_span(state.legs[leg_name], side)
        if span is not None:
            state.refuse(leg_name, side, FacilityRefusal(
                span[0], span[1], "every rung of the ladder was refused on this approach - see "
                                  "the widths printed above for which limit stopped each"))
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


@dataclass(frozen=True)
class CorridorCalming:
    """A route-level decision for a street with no room for a new facility: WHICH LEGS ARE CALMED.

    It carries no section and no side because it places nothing new. It names the route, and every
    approach on it gets the design that proposes nothing - travel lanes held at
    TARGET_LANE_WIDTH_FT with the recovered width painted as OSM says that kerb is used, hatched
    where parking is restricted and marked where it is not. See osm_derived_baseline, which is
    that design; this decides only what it is applied to.

    WHICH LEGS IS THE WHOLE DECISION, and it was a hand-written tuple in two site files: the same
    two leg names at both junctions, each read off by eye from the street name the config already
    states. That is the duplication CorridorFacility exists to remove, here in a form the
    duplicate-rule test cannot see - tests/test_sites.py exempts a one-line scenario body, which
    is what both of these were.

    THE REST OF THE JUNCTION IS STILL UPGRADED, because osm_derived_baseline also completes the
    centrelines and makes the crossings continental, and those are junction-wide rather than
    per-street: a crossing of the cross street is still a crossing at this junction. The route
    decides the CROSS-SECTION pass and nothing else.
    """
    road: str

    def legs_on(self, model: "IntersectionModel") -> list[str]:
        """This junction's approaches that lie on this route, in a stable order."""
        return legs_on_road(model, self.road)

    def apply_to(self, state: DesignState, model: "IntersectionModel" = None,
                 quiet: bool = False) -> DesignState:
        """Calm this junction's approaches that are on the route, and only those.

        `model` is optional and None returns the state untouched, because that is the older
        single-argument convention every site's scenario builder still answers to.
        """
        if model is None:
            return state
        legs = tuple(self.legs_on(model))
        if not legs:
            # NOT osm_derived_baseline(legs=()), WHICH WOULD CALM THE WHOLE JUNCTION: its `legs`
            # is falsy-tested, so an empty tuple falls through to every kerb. A route that does
            # not reach this junction has to leave its cross-sections alone, and a route named
            # wrongly - a street name that matches nothing here - must say so rather than quietly
            # treating four legs nobody asked about.
            if not quiet:
                print(f"  NOTE: no leg of this junction is on {self.road}, so nothing is calmed "
                      f"here. Nothing about the cross-sections has changed.")
            return state
        return osm_derived_baseline(state, model, legs=legs)


#: PRINCETON AVE (CR 569), CALMED END TO END - 1,565 ft over two modelled junctions.
#:
#: WHY CALMING AND NOT A BIKEWAY, which is the question this declaration exists to answer, because
#: the corridor's other route decision is BROAD_ST_TWO_WAY_BIKEWAY and the obvious next move is to
#: run it down this street too. It refuses every foot it can test. Reproduce it with
#: `scripts/corridor_render.py --road "Princeton Avenue"`: over the 349 ft of the route where both
#: kerbs are traced, the constrained rung's 8 ft lane and 3 ft buffer spend 11.82 ft of the 30.13 ft
#: between the kerbs, leaving 9.15 ft per travel lane - under the 10 ft floor - and the route's
#: narrowest traced width is 30.0 ft. A conventional one-way pair is out on the same measurement
#: from the other end: 4.1-4.6 ft spare per side beside a TARGET_LANE_WIDTH_FT lane, under
#: AASHTO_MIN_BIKE_LANE_FT, so what would be drawn is a stripe that reads as a bike lane and is not
#: one. See each site's scenarios.py for its own junction's figures.
#:
#: SO BOTH OF THIS PROJECT'S USUAL LEVERS ARE UNAVAILABLE HERE, and that is a fact about the street
#: rather than about the section: Princeton Ave is a designated truck route (hgv=designated) past an
#: elementary school, so the travel lanes cannot go under the floor, and it is no_parking for its
#: whole length, so there is no parking lane to reclaim and the recovered width hatches rather than
#: becoming stalls. What is left is the width beside an 11 ft lane, which is what this places.
#:
#: NOT APPLIED AT ebroad_princeton, the route's third junction, where Princeton Ave arrives as a
#: stem: that site's scenarios paint EVERY kerb, E Broad's included, so the stem is already calmed
#: by that pass. Applying this on top would put a second LaneNarrowing and MarkedParking on the
#: same kerb - DesignState.apply has no duplicate guard.
PRINCETON_AVE_CALMING = CorridorCalming(road="Princeton Avenue")


#: EVERY ROUTE-LEVEL DECISION THIS PROJECT HAS MADE, so "what is proposed along this street" is a
#: lookup and not an import. scripts/corridor_render.py hardcoded BROAD_ST_TWO_WAY_BIKEWAY, so
#: `--road "Princeton Avenue"` drew BROAD ST'S BIKEWAY refusing down a street whose route decision
#: is a calming: a strip plan headed "0 ft placed of 1,565 ft" for a design nobody proposed there,
#: with a which-kerb-carries-the-lane table under it for a lane that does not exist. A drawing may
#: not quietly answer a question about a different street.
ROUTE_DECISIONS: tuple = (BROAD_ST_TWO_WAY_BIKEWAY, PRINCETON_AVE_CALMING)


def route_decision_for(road: str):
    """What this project proposes along a named street, or None if it has decided nothing.

    Matched through the same _street_name normalisation legs_on_road uses, so a corridor named
    off its legs ("East Broad Street", "W Broad St") finds the one Broad St decision. A street
    with no row here is not an error - it is a street this project has not yet decided about, and
    the caller is expected to say so rather than to borrow another street's proposal.
    """
    want = _street_name(road)
    for decision in ROUTE_DECISIONS:
        if _street_name(decision.road) == want:
            return decision
    return None
