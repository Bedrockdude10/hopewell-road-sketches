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
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel



@dataclass
class DesignState:
    """A mutable-by-copy snapshot of intersection geometry. Treatments clone the
    state, apply one change, and return the clone - so `state = bump_out(state, ...)`
    chains cleanly and the original scenario is never touched."""
    legs: dict
    corner_fillets: dict
    # leg name -> what is painted down that leg's middle TODAY, one of VALID_CENTERLINE_STYLES.
    # An OBSERVED FACT, not a treatment's parameter: seeded in from_model from config.yaml or
    # OSM's overtaking=no. What a PROPOSAL paints is a SetCenterlineStyle; ask
    # centerline_style() for the resolved answer or a proposal's change is invisible.
    existing_centerline_styles: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [KerbOpening]. Where OSM says the kerb is DROPPED for a
    # vehicle to cross - a driveway or yard entrance. Seeded in from_model from the traced kerbs'
    # kerb=lowered / kerb=flush tags; read by src/geometry/paint.py to break the kerbside
    # markings over it. See src/geometry/kerbs.py.
    kerb_openings: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> [ParkingRestriction]. What OSM says about this kerb, per
    # STRETCH of it. Read by src/geometry/daylighting.py, which turns a prohibition into a
    # no-parking zone like any statutory one.
    parking_restrictions: dict = field(default_factory=dict)
    #: {leg name: [CrossStreet]} - every OTHER street a leg runs across. An observed fact like
    #: the two above. R.S. 39:4-138(e) applies at every intersection, not only the one the
    #: drawing is about, and a leg drawn 374 ft crosses several - see src/geometry/cross_streets.
    cross_streets: dict = field(default_factory=dict)
    # Every Treatment applied to this design, in order (see apply) - the design as a list of
    # decisions. Every renderer reads its parameters from here, through treatment_for /
    # treatments_of / every_treatment, and provenance is written from it.
    treatments: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @classmethod
    def from_model(cls, model: "IntersectionModel") -> "DesignState":
        # PRECEDENCE IS BY PROVENANCE, not by which file the value came from (src/provenance.py).
        # A double yellow IS the no-passing marking, so OSM's overtaking=no is a direct statement
        # about the paint. An explicit config.yaml centerline_style is direct observation and
        # wins - but a config entry that merely repeats DEFAULT_CENTERLINE_STYLE is not an
        # observation, it is this repo's own placeholder, so it defers to the surveyed OSM tag.
        # Any other configured value (double_yellow, none) is a positive statement and wins.
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

        A SetCenterlineStyle outranks the observed fact from_model seeded. Both renderers go
        through here so they cannot disagree about which source won.
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

        None where nothing moved it: the alignment IS the divider unless a two-way bike lane on
        one side has pushed the travel lanes over. Returns distance AND side, because a distance
        with no side is half a fact (see Side.sign).

        Read by BOTH the plan view and the 3D export, which is why it lives here rather than in
        either: a shift one view honours and the other does not is a render whose two travel
        lanes are different widths.
        """
        # Same reason as centerline_style above - bikeways is layered above state.
        from src.geometry.treatments.bikeways import AddTwoWayBikeLane, divider_shift_toward_ft

        for treatment in self.treatments_of(AddTwoWayBikeLane):
            if treatment.target.leg != leg_name:
                continue
            # CANONICAL FORM: a NON-NEGATIVE distance paired with the side it is actually on.
            # The sign is resolved here, once, rather than travelling alongside a side that can
            # contradict it - a consumer taking abs() of a signed shift draws the paint on the
            # wrong side of the alignment.
            #
            # The divider is NOT always on the far side. It is wherever a target-width lane from
            # the section's inner edge lands, and on a wide leg (broad_st_west) that is still
            # short of the alignment, i.e. the shift is toward the lane's own side.
            toward_left_ft = divider_shift_toward_ft(self, leg_name, Side.LEFT)
            if toward_left_ft >= 0:
                return toward_left_ft, str(Side.LEFT)
            return -toward_left_ft, str(Side.RIGHT)
        return None

    def treatment_for(self, kind, target) -> Treatment | None:
        """The treatment of `kind` applied at `target`, or None if there is none.

        The last one applied wins: a design is a sequence of decisions and the later one is the
        decision, so two MarkedParking treatments on one kerb are one marked lane.

        This is how a treatment asks about ANOTHER treatment. A bollard row's precondition is "is
        there a buffered bike lane here", which is a question about a decision someone made - not
        about an entry under a key that anything, including a test, could have written.
        """
        found = None
        for treatment in self.treatments:
            if isinstance(treatment, kind) and treatment.target == target:
                found = treatment
        return found

    def treatments_of(self, kind) -> list[Treatment]:
        """Every treatment of `kind`, ONE PER TARGET (the last applied), sorted by target.

        Last-applied-wins per target, for the reason treatment_for gives. Painting both instead
        is what makes MarkingsDoNotCollide fire.

        SORTED BY TARGET rather than in application order, so what a consumer sees is a property
        of the design and not of the order a scenario builder's loops ran in - the props array is
        order-sensitive in the exported JSON.

        Where a treatment ACCUMULATES rather than replacing - ShiftCrosswalk, ExtraProp - this is
        the wrong question and every_treatment is the right one.
        """
        by_target = {}
        for treatment in self.treatments:
            if isinstance(treatment, kind):
                by_target[treatment.target] = treatment
        return [by_target[target] for target in sorted(by_target, key=str)]

    def every_treatment(self, kind, target=None) -> list[Treatment]:
        """Every treatment of `kind`, in application order, WITHOUT collapsing per target.

        For the two treatments that add up instead of replacing: ShiftCrosswalk shifts a crossing
        by a delta, ExtraProp puts one more sign on a leg. treatments_of would silently drop all
        but the last, which for a second RRFB on one leg is a prop that stops being drawn.
        """
        return [t for t in self.treatments
                if isinstance(t, kind) and (target is None or t.target == target)]

    def apply(self, *treatments: Treatment, model: "IntersectionModel" = None) -> "DesignState":
        """Apply treatments to a COPY of this design and return it.

        THE SINGLE WAY a treatment enters a design, so everything every treatment needs checked
        is checked here once: that the target exists at this junction (a leg-name typo must not
        silently do nothing), that a treatment declaring needs_model got one, and that
        state.notes records what was applied without each treatment remembering to append.

        Chains, so a scenario reads `state.apply(a).apply(b)` or `state.apply(a, b)`.
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
