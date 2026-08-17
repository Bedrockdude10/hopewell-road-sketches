"""DesignState - the thing every treatment transforms.

A treatment returns a NEW state rather than mutating one, so scenarios stack without disturbing
the existing-conditions baseline. Kept in its own module because everything else in this package
imports it and it imports (almost) nothing back: the few concrete treatments it has to reach for
are imported inside the methods that use them, which is what keeps this module at the bottom of
the graph next to base."""
from copy import deepcopy
from dataclasses import dataclass, field


from src.geometry.cross_streets import cross_streets_from_model
from src.geometry.kerbs import kerb_openings_from_model
from src.geometry.targets import LegTarget, Side
from src.geometry.treatments.base import (DEFAULT_CENTERLINE_STYLE, Treatment,
                                          _parking_restrictions_from_model)



@dataclass
class DesignState:
    """A mutable-by-copy snapshot of intersection geometry. Treatments clone the
    state, apply one change, and return the clone - so `state = bump_out(state, ...)`
    chains cleanly and the original scenario is never touched."""
    legs: dict
    corner_fillets: dict
    # leg name -> what is painted down that leg's middle TODAY, one of VALID_CENTERLINE_STYLES.
    # An OBSERVED FACT and not a treatment's parameter, which is why it survived the collapse
    # alongside parking_restrictions: from_model seeds it from config.yaml (street-view
    # confirmed) or from OSM's overtaking=no, exactly as the parking restrictions come from the
    # OSM tags. What a PROPOSAL paints instead is a SetCenterlineStyle treatment; ask
    # centerline_style() for the resolved answer rather than reading this, or a proposal's
    # change is invisible.
    existing_centerline_styles: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [KerbOpening]. Where OSM says the kerb is DROPPED for a
    # vehicle to cross - a driveway or a yard entrance. Seeded from the traced kerbs' own
    # kerb=lowered / kerb=flush tags in from_model, the third observed fact on this design
    # alongside the two below, and read by src/geometry/paint.py to break the kerbside markings
    # over it. See src/geometry/kerbs.py.
    kerb_openings: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [ParkingRestriction]. What OSM says about this kerb, per
    # STRETCH of it - seeded from the model in from_model. Read by src/geometry/daylighting.py,
    # which turns a prohibition into a no-parking zone like any statutory one.
    parking_restrictions: dict = field(default_factory=dict)
    #: {leg name: [CrossStreet]} - every OTHER street a leg runs across. Seeded here with the
    #: two above because it is the same kind of thing: an observed fact about the street that no
    #: treatment chose. R.S. 39:4-138(e) applies at every intersection, not only the one the
    #: drawing is about, and a leg drawn 374 ft crosses several - see src/geometry/cross_streets.
    cross_streets: dict = field(default_factory=dict)
    # Every Treatment applied to this design, in order (see apply) - the design as a list of
    # decisions, which is what a scenario actually is, what every renderer reads its parameters
    # from (treatment_for / treatments_of / every_treatment) and what provenance is written from.
    treatments: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @classmethod
    def from_model(cls, model) -> "DesignState":
        # A double yellow centerline IS the no-passing marking, so OSM's overtaking=no is a
        # direct statement about what is painted on the road - better evidence than this
        # project's dashed-line default, which was only ever a placeholder. Five ways in
        # Hopewell carry it, covering both Broad Streets, both Greenwood Avenues and
        # Princeton Avenue.
        #
        # An explicit centerline_style in config.yaml still wins: that is direct observation
        # (src/provenance.py - if OSM disagrees with something we looked at, OSM is wrong).
        # Precedence is by PROVENANCE, not by which file the value came from. A config entry
        # that merely repeats DEFAULT_CENTERLINE_STYLE is not an observation - it is this
        # repo's own placeholder written down, and every config that has one says so in its
        # comment ("NOT confirmed - repo default, retained"). Letting that outrank a surveyed
        # OSM tag is the project's core principle exactly backwards: it is the generic guess
        # beating the real sourced data. So a default-valued entry defers to OSM, while any
        # other value (double_yellow, none) is a positive statement and wins.
        osm_tags = getattr(model, "leg_osm_tags", {})
        centerline_styles = {}
        for name, leg_cfg in model.config["legs"].items():
            configured = leg_cfg.get("centerline_style")
            no_passing = osm_tags.get(name, {}).get("overtaking") == "no"
            if configured is not None and configured != DEFAULT_CENTERLINE_STYLE:
                centerline_styles[name] = configured
            elif no_passing:
                centerline_styles[name] = "double_yellow"
                if configured is not None:
                    print(f"  NOTE: {name} is tagged overtaking=no in OSM - drawing a double "
                          f"yellow centerline, over the retained repo default in config.yaml. "
                          f"Set a non-default centerline_style there if you've observed "
                          f"otherwise.")
            else:
                centerline_styles[name] = DEFAULT_CENTERLINE_STYLE
        return cls(legs=deepcopy(model.legs), corner_fillets=deepcopy(model.corner_fillets),
                   existing_centerline_styles=centerline_styles,
                   kerb_openings=kerb_openings_from_model(model),
                   parking_restrictions=_parking_restrictions_from_model(model),
                   cross_streets=cross_streets_from_model(model))

    def clone(self) -> "DesignState":
        return deepcopy(self)

    def centerline_style(self, leg_name: str) -> str:
        """What this design paints down `leg_name`'s middle: a proposal's choice, else what is
        there today.

        Two sources, and the order is the design's: a SetCenterlineStyle is a decision this
        proposal made and outranks the observed fact from_model seeded (see
        existing_centerline_styles). Both renderers go through here so they cannot disagree
        about which one won - the 3D render reads it into the geometry JSON and the plan view
        draws it, and this view exists to show what that render will show.
        """
        # Imported here, not at module scope: crossings sits ABOVE this module in the package's
        # layering (see __init__.py) and importing it up here would close the cycle.
        from src.geometry.treatments.crossings import SetCenterlineStyle

        treatment = self.treatment_for(SetCenterlineStyle, LegTarget(leg_name))
        if treatment is not None:
            return treatment.style
        return self.existing_centerline_styles.get(leg_name, DEFAULT_CENTERLINE_STYLE)

    def travel_lane_divider_shift(self, leg_name: str) -> tuple[float, str] | None:
        """How far off the alignment this leg's centreline paint sits, and toward which side.

        None where nothing moved it, which is every leg in every other scenario - the alignment
        IS the divider unless a two-way bike lane on one side has pushed the travel lanes over.
        Returns the distance and the side it moves toward, because a distance with no side is
        half a fact and the sign convention has bitten this project before (see Side.sign).

        Read by BOTH the plan view and the 3D export, which is the point of it being here rather
        than computed in either: a shift one view honours and the other does not is a render
        whose two travel lanes are different widths, and the seam where that would hide is
        exactly the one src/render/crosswalks.py:centerline_paint_ft was written to close.
        """
        # Same reason as centerline_style above - bikeways is layered above state.
        from src.geometry.treatments.bikeways import AddTwoWayBikeLane, divider_shift_toward_ft

        for treatment in self.treatments_of(AddTwoWayBikeLane):
            if treatment.target.leg != leg_name:
                continue
            # CANONICAL FORM: a NON-NEGATIVE distance and the side it is actually on. The sign is
            # resolved here, once, rather than travelling alongside a side that may contradict it.
            #
            # It used to return the raw signed shift paired with "the side away from the lane",
            # and those two disagree whenever the shift is negative - which happens on a street
            # wide enough that the near travel lane still does not reach the alignment.
            # broad_st_west is exactly that: -1.42 paired with "right", where the divider is
            # really 1.42 ft to the LEFT. centerline_paint_ft took abs() of the distance and drew
            # the double yellow 1.42 ft to the right - 2.84 ft from the divider, and from the stop
            # bar resting against it. Its travel lanes came out 13.84 and 8.16 ft while every
            # check, measuring the intention rather than the drawing, reported 11.00.
            #
            # Note the divider is NOT always on the far side. It is wherever a target-width lane
            # from the section's inner edge lands, and on a wide leg that is still short of the
            # alignment.
            toward_left_ft = divider_shift_toward_ft(self, leg_name, Side.LEFT)
            if toward_left_ft >= 0:
                return toward_left_ft, str(Side.LEFT)
            return -toward_left_ft, str(Side.RIGHT)
        return None

    def treatment_for(self, kind, target):
        """The treatment of `kind` applied at `target`, or None if there is none.

        The last one applied wins, because a design is a sequence of decisions and the later one
        is the decision: two MarkedParking treatments on one kerb are one marked lane.

        This is how a treatment asks about ANOTHER treatment, and it replaces asking about the
        dict that treatment happens to write. A bollard row's precondition is "is there a
        buffered bike lane here", not "is there an entry under this key" - and the difference
        showed: a dict lookup answers about state that anything could have written, including a
        test poking it directly, while this answers about a decision someone actually made.
        """
        found = None
        for treatment in self.treatments:
            if isinstance(treatment, kind) and treatment.target == target:
                found = treatment
        return found

    def treatments_of(self, kind) -> list:
        """Every treatment of `kind`, ONE PER TARGET (the last applied), sorted by target.

        Last-applied-wins per target, for the same reason treatment_for is: a design is a
        sequence of decisions and the later one is the decision. Two MarkedParking treatments
        on one kerb are one marked lane, not two painted on top of each other - and painting
        both is what makes MarkingsDoNotCollide fire.

        SORTED BY TARGET rather than in application order, so what a consumer sees is a
        property of the design and not of the order the scenario builder's loops happened to
        run in. The props array is order-sensitive in the exported JSON, and BROAD_ST_LEGS is
        ("broad_st_west", "broad_st_east") - west first - so an application-ordered read makes
        the file depend on a tuple in a site's scenarios.py.

        Where a treatment ACCUMULATES rather than replacing - ShiftCrosswalk, ExtraProp - this
        is the wrong question and every_treatment is the right one.
        """
        by_target = {}
        for treatment in self.treatments:
            if isinstance(treatment, kind):
                by_target[treatment.target] = treatment
        return [by_target[target] for target in sorted(by_target, key=str)]

    def every_treatment(self, kind, target=None) -> list:
        """Every treatment of `kind`, in application order, WITHOUT collapsing per target.

        For the two treatments that add up instead of replacing: ShiftCrosswalk shifts a
        crossing by a delta, and ExtraProp puts one more sign on a leg. Both wrote something
        cumulative - a `+=` and a list append - where every other treatment wrote a key, so
        asking treatments_of for them would silently drop all but the last, which for a second
        RRFB on one leg is a prop that quietly stops being drawn.
        """
        return [t for t in self.treatments
                if isinstance(t, kind) and (target is None or t.target == target)]

    def apply(self, *treatments: Treatment, model=None) -> "DesignState":
        """Apply treatments to a COPY of this design and return it.

        The single way one treatment enters a design, and the reason it exists is that everything
        every treatment needs checked can be checked here, once:

          * the target exists at this junction - a leg name typo used to write a dict key that
            nothing read, so the treatment silently did nothing;
          * a treatment that needs the model got one - the alternative is the bug that shipped,
            where a dropped argument produced a scenario with no treatments that rendered
            plausibly;
          * the design records what was applied, in state.notes, without each treatment having to
            remember to append to it. Several did not, so the provenance printed with a render was
            missing the treatments that forgot.

        Chains, so a scenario reads `state.apply(a).apply(b)` or `state.apply(a, b)`. That also
        removes a real failure mode of the `state = add_x(state, ...)` form: dropping the
        assignment left the treatment applied to a discarded copy, and the render looked fine.
        """
        new_state = self.clone()
        for treatment in treatments:
            missing = treatment.target.missing_from(new_state)
            if missing:
                raise KeyError(f"{type(treatment).__name__} cannot be applied: {missing}")
            if treatment.needs_model and model is None:
                raise ValueError(
                    f"{type(treatment).__name__} needs the IntersectionModel - it reads geometry "
                    f"that the design does not carry. Pass model= (see src/site.py:run_scenario, "
                    f"which hands every scenario builder the model for exactly this).")
            measured = treatment.apply_to(new_state, model)
            new_state.treatments.append(treatment)
            new_state.notes.append(treatment.describe() + (measured or ""))
        return new_state
