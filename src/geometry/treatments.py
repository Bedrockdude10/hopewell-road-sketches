"""Parametric pedestrian-safety treatments: composable geometry transforms over a
DesignState. Each treatment returns a new DesignState so scenarios can be stacked
without mutating the baseline (existing-conditions) model."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from shapely.geometry import Polygon

from src.geometry.targets import BOTH_SIDES, LegSide, LegTarget, Side, Target
from src.geometry.model import (BULBOUT_TAPER_RATE, build_pavement_polygon, curb_extension_line,
                                fillet_curb_corner, leg_clearance_ft, narrowest_half_width_ft)


def _band_across_the_road(centerline, from_ft: float, to_ft: float, half_width_ft: float,
                           what: str) -> Polygon:
    """The rectangle spanning `half_width_ft` either side of a leg, between two stations.

    Shared by refuge_island and raise_crossing, which built the same shape from the same
    three interpolations and the same normal - and divided by the length of (p_far - p_near)
    without checking it. That vector collapses whenever both stations clamp to the same point,
    which is not hypothetical: leg_clearance_ft returns 133 ft on W Broad & Louellen's 130 ft
    southwest leg (its acute Y makes the corner return eat the whole leg), so raise_crossing
    there interpolated both ends to the leg's far end and divided by zero. A ZeroDivisionError
    out of a treatment function says nothing about the junction; the error below does.
    """
    near = centerline.interpolate(max(min(from_ft, centerline.length), 0.0))
    far = centerline.interpolate(max(min(to_ft, centerline.length), 0.0))
    dx, dy = far.x - near.x, far.y - near.y
    length = np.hypot(dx, dy)
    if length < 1e-9:
        raise ValueError(
            f"Can't place a {what} between {from_ft:.1f} ft and {to_ft:.1f} ft along a "
            f"{centerline.length:.1f} ft leg - both ends land on the same point, so the shape "
            f"has no extent along the road. The leg is too short for it (usually a corner "
            f"return consuming the whole leg - see leg_clearance_ft).")
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux            # unit normal, across the road
    return Polygon([
        (near.x + nx * half_width_ft, near.y + ny * half_width_ft),
        (far.x + nx * half_width_ft, far.y + ny * half_width_ft),
        (far.x - nx * half_width_ft, far.y - ny * half_width_ft),
        (near.x - nx * half_width_ft, near.y - ny * half_width_ft),
    ])

NACTO_MIN_REFUGE_ISLAND_WIDTH_FT = 6
LANE_NARROWING_DEFAULT_STRIPE_FT = 5.0  # common low-cost NACTO paint buffer/shoulder-stripe width

# The travel lane width every road diet here aims at: NACTO/AASHTO urban minimum. Defined
# once, in src, because it is a standard rather than a per-site choice - it was previously
# redeclared in each site's scenarios.py, which is how nothing ended up enforcing it.
TARGET_LANE_WIDTH_FT = 11.0
# A parking lane is a STANDARD width, not "whatever is left over". Anything wider than this
# isn't a wider parking space, it's a parking space plus unmarked asphalt - which is what
# gets hatched. Where the spare width can't even cover one standard stall, no parking is
# marked at all rather than a token strip painted.
MIN_MARKED_PARKING_DEPTH_FT = 8.0
CORNER_HATCHING_DEFAULT_DEPTH_FT = 6.0  # paint-only zone depth, comparable footprint to a modest real curb extension
CORNER_APRON_DEFAULT_EXTENT_FT = 5.0  # mountable-apron zone depth - same shape as hatching, different surface finish

# What's actually painted down the middle of a leg today: a single dashed
# yellow line (default - the ordinary two-way-undivided-road marking), a solid
# double yellow (no-passing zone), or none at all (some real local streets
# genuinely have no centerline paint). Unlike crosswalk style, there's no OSM
# tag for this - it's read directly from a site's config.yaml per leg (see
# sites/README.md), confirmed the same way as the `signals` block (street-view
# photo review, not a field survey).
DEFAULT_CENTERLINE_STYLE = "single_yellow_dashed"
VALID_CENTERLINE_STYLES = ("single_yellow_dashed", "double_yellow", "none")

# Float slack when comparing a requested width against the room a leg has. The widths
# themselves are specified to a tenth of a foot; this only absorbs the arithmetic.
LANE_WIDTH_SLACK_FT = 0.05


@dataclass(frozen=True)
class ParkingRestriction:
    """What OSM says about parking on ONE KERB over ONE STRETCH of it, in the leg's frame.

    A stretch and not a whole side, because a restriction that changes part way along a street
    is recorded in OSM by splitting the way - which is how "no parking for the first 100 ft from
    the junction" is expressed, and it is what this project used to discard. See
    src/geometry/intersection.py:RoadSpan.

    `value` is the raw OSM value: no_parking / no_standing / no_stopping, or "none" for an
    explicit statement that parking IS allowed, or None where that way says nothing at all. The
    last two are different facts and must not be collapsed.
    """
    start_ft: float
    end_ft: float
    value: str | None
    way_id: int | None = None

    @property
    def prohibits(self) -> bool:
        """True where OSM forbids parking. Absent or "none" is not a prohibition."""
        return self.value is not None and self.value != "none"

    @property
    def citation(self) -> str:
        return (f"OSM parking restriction {self.value!r}"
                + (f" on way {self.way_id}" if self.way_id is not None else ""))


def _parking_restrictions_from_model(model) -> dict:
    """{(leg, side): [ParkingRestriction]} from every OSM way lying along each leg.

    Seeded onto the state the way centerline_styles is, so treatments, both renderers and the
    invariants all read one resolved answer rather than each reaching back into the model - and
    so a scenario can be handed a state without a model behind it.
    """
    out: dict[tuple[str, str], list[ParkingRestriction]] = {}
    spans = getattr(model, "parking_restriction_spans", None)
    if spans is None:
        return out
    for leg_name in getattr(model, "leg_road_spans", {}):
        for start_ft, end_ft, sides, way_id in spans(leg_name):
            for side, value in sides.items():
                out.setdefault((leg_name, side), []).append(
                    ParkingRestriction(start_ft=start_ft, end_ft=end_ft, value=value,
                                        way_id=way_id))
    for key in out:
        out[key].sort(key=lambda r: r.start_ft)
    return out


@dataclass(frozen=True)
class CurbExtension:
    """One kerb moved into the roadway, and the numbers that put it there.

    Recorded per (leg, side) because that is the degree of freedom the street gives: how deep
    an extension a leg can take is set by its own spare width, and at Broad & Greenwood the two
    Broad legs can give 15.0 and 16.8 ft per side while the two Greenwood legs can give 2.3 and
    4.6. A corner between one of each cannot be extended symmetrically.
    """
    extension_ft: float       # how far the kerb moved, measured from the NOMINAL half-width
    full_ft: float            # station the straight face runs to
    taper_ft: float           # length of the return to the real kerb
    face_radius_ft: float     # the corner a passenger car sees
    swept_radius_ft: float | None = None   # the corner a bus still gets, via the apron

    @property
    def footprint_ft(self) -> float:
        """How much kerb the extension occupies end to end - what has to fit inside the length
        the parking ordinance already prohibits, if it is to cost no spaces."""
        return self.full_ft + self.taper_ft


@dataclass(frozen=True)
class CornerApron:
    """A flush, drivable corner surface. Two shapes, because there are two reasons for one.

    `depth_ft` is a fixed reach inward from the corner arc: the standalone treatment
    (add_mountable_apron), for a corner where a hard bulb-out is not an option.

    `swept_radius_ft` is the radius a large vehicle needs, and the apron is then the ANNULUS
    between that and `face_radius_ft` (src/geometry/model.py:corner_apron_annulus). This is the
    one a curb extension lays: the claim it supports is that the swept path survives the
    tightened corner, and a fixed depth cannot support that claim because nothing ties it to
    the radius the vehicle needs.
    """
    depth_ft: float | None = None
    swept_radius_ft: float | None = None
    face_radius_ft: float | None = None

    def __post_init__(self):
        if (self.depth_ft is None) == (self.swept_radius_ft is None):
            raise ValueError(
                "A CornerApron is either a fixed depth inward from the arc (depth_ft) or the "
                "annulus out to a swept radius (swept_radius_ft) - exactly one, since they are "
                f"different shapes for different reasons. Got depth_ft={self.depth_ft}, "
                f"swept_radius_ft={self.swept_radius_ft}.")
        if self.swept_radius_ft is not None and self.face_radius_ft is None:
            raise ValueError("An annulus apron needs the face_radius_ft it is measured from.")


@dataclass(frozen=True)
class Treatment(ABC):
    """One change a proposal makes to a design.

    Every treatment is a frozen dataclass validating itself in `__post_init__`, and that is the
    point of the base class. Before it, a treatment was a function that wrote a dict, and
    validation was a convention: `add_bike_lane` refused a lane under AASHTO's 5 ft minimum and
    `add_bike_lane_bollards` refused a lane with no buffer, while `add_lane_narrowing` and
    `add_corner_hatching` checked nothing at all. A convention is exactly what this codebase
    kept discovering it had not followed. An object cannot exist unvalidated.

    Three things every treatment declares:

      * `target` - a src/geometry/targets.py:Target, checked to exist in the design by
        DesignState.apply before anything is written. A treatment aimed at a leg the junction
        does not have used to write a key nothing ever read.
      * `describe()` - one line for state.notes, so provenance is recorded by applying a
        treatment rather than by each function remembering to append to a list. Several did not.
      * `apply_to(state, model)` - the change itself, on an already-cloned state.

    Subclasses that need the IntersectionModel (a kerb rebuild reads the traced kerbs) declare
    `needs_model = True` and receive it; asking for one that was not supplied is an error rather
    than a silently skipped treatment, which is a bug that shipped here - phase4 dropped the
    model argument and produced a proposal with no treatments in it that rendered fine.
    """
    #: Set by a subclass that cannot be applied without the model's traced geometry.
    needs_model: ClassVar[bool] = False

    #: Where this treatment goes. First field of every treatment, so the constructor reads
    #: `LaneNarrowing(LegTarget("broad_st_east"), ...)` - the thing being changed, then how.
    target: Target

    @abstractmethod
    def describe(self) -> str:
        ...

    #: Where this treatment's markings fall in the painting order. Groups run in ascending
    #: order and are painted target by target within a group; rank breaks a tie between two
    #: treatments on the same target (a bollard row is painted after the buffer it stands in).
    #: See src/geometry/paint.py:curbside_paint_ft.
    paint_group: ClassVar[int] = 50
    paint_rank: ClassVar[int] = 0

    def paint(self, ctx) -> None:
        """Put this treatment's markings on the roadway, through ctx (paint.PaintContext).

        Nothing by default: several treatments change geometry rather than markings - a curb
        extension moves the kerb and everything downstream re-measures against it - and a
        crosswalk restyle is drawn by the crossing renderer, not from the paint list.
        """

    @abstractmethod
    def apply_to(self, state: "DesignState", model=None) -> str | None:
        """Make the change. May return a suffix for the note, for a treatment whose
        provenance includes something only measurable against the design - a bike lane records
        how much of the width the leg actually has it used."""
        ...


@dataclass
class DesignState:
    """A mutable-by-copy snapshot of intersection geometry. Treatments clone the
    state, apply one change, and return the clone - so `state = bump_out(state, ...)`
    chains cleanly and the original scenario is never touched."""
    legs: dict
    corner_fillets: dict
    refuge_islands: dict = field(default_factory=dict)   # name -> {"polygon": Polygon, "width_ft": float}
    raised_crossings: dict = field(default_factory=dict)  # leg name -> Polygon
    crosswalk_styles: dict = field(default_factory=dict)  # leg name -> "lines" | "continental" | "ladder"
    centerline_styles: dict = field(default_factory=dict)  # leg name -> one of VALID_CENTERLINE_STYLES - seeded
                                                             # from config.yaml in from_model(), see set_centerline_style
    lane_narrowing: dict = field(default_factory=dict)  # leg name -> stripe_width_ft (paint-only, no curb change)
    lane_narrowing_line_only: set = field(default_factory=set)  # leg names (subset of lane_narrowing) that get
                                                                  # ONLY the solid edge/taper line delineating the
                                                                  # lane, no diagonal chevron fill - see add_lane_narrowing's
                                                                  # line_only param
    lane_narrowing_sides: dict = field(default_factory=dict)  # leg name -> ("left",) | ("right",) | ("left","right") -
                                                                 # which side(s) of the leg add_lane_narrowing actually
                                                                 # narrowed (defaults to both if a leg is absent here -
                                                                 # see add_lane_narrowing's sides param). Only ever a
                                                                 # strict subset when something else (e.g. add_marked_parking)
                                                                 # already owns the other side's edge.
    bollard_lines: dict = field(default_factory=dict)  # leg name -> spacing_ft (see add_bollards - requires
                                                         # lane_narrowing on the same leg, sits inside that buffer)
    parking_zones: dict = field(default_factory=dict)  # (leg name, "left"|"right") -> {"depth_ft", "stall_length_ft",
                                                         # "curb_offset_ft"} - marked curbside parking (paint-only,
                                                         # no curb change) - see add_marked_parking
    parking_buffer_bollards: dict = field(default_factory=dict)  # (leg name, "left"|"right") -> spacing_ft - flex-post
                                                                   # bollards centered in that zone's curb_offset_ft
                                                                   # buffer (requires curb_offset_ft > 0) - see
                                                                   # add_parking_buffer_bollards
    daylight_devices: dict = field(default_factory=dict)  # (leg name, "left"|"right") -> {"kind", "spacing_ft"} -
                                                            # physical objects standing in the daylight zone. See
                                                            # protect_daylight_zone. A kind listed in
                                                            # CURB_EXTENSION_DEVICES also cuts the setback under
                                                            # R.S. 39:4-138(e) from 25 ft to 10 ft.
    # (leg name, "left"|"right") -> [ParkingRestriction]. What OSM says about this kerb, per
    # STRETCH of it - seeded from the model in from_model. Read by src/geometry/daylighting.py,
    # which turns a prohibition into a no-parking zone like any statutory one.
    parking_restrictions: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> CurbExtension. The ONE treatment here that moves a kerb
    # rather than painting on it, so the Leg's own curb line changes too and every measurement
    # downstream follows - see add_curb_extension.
    curb_extensions: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> BikeLane. Paint-only: an exclusive lane with its own edge
    # line, a buffer, and optionally a parking lane outside it - see add_bike_lane.
    bike_lanes: dict = field(default_factory=dict)
    # (leg name, "left"|"right") -> spacing_ft. Flex posts in the buffer on the TRAFFIC side of
    # a bike lane, which is what makes it protected - see add_bike_lane_bollards.
    bike_lane_bollards: dict = field(default_factory=dict)
    corner_hatching: dict = field(default_factory=dict)  # corner tuple -> depth_ft (paint-only, no curb change)
    corner_aprons: dict = field(default_factory=dict)  # corner tuple -> CornerApron (mountable, no curb change)
    crosswalk_offset_overrides: dict = field(default_factory=dict)  # leg name -> +/- delta_ft on top of the
                                                                     # normally-resolved offset (see shift_crosswalk)
    extra_props: list = field(default_factory=list)  # [{"leg","type","offset_ft","side","note"}] - see add_extra_prop
    # Every Treatment applied to this design, in order (see apply). The dicts above are what the
    # paint and prop builders read today; this is the design as a list of decisions, which is
    # what a scenario actually is and what provenance is written from.
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
                   centerline_styles=centerline_styles,
                   parking_restrictions=_parking_restrictions_from_model(model))

    def clone(self) -> "DesignState":
        return deepcopy(self)

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


def find_corner(state: DesignState, leg_a: str, leg_b: str) -> tuple[str, str]:
    """Look up the (name_a, name_b) key in state.corner_fillets for the corner
    where leg_a and leg_b meet, regardless of which order build_corner_fillets
    happened to store it in (it sorts by compass bearing, not by call-site
    convenience) - corners are identified by which two legs meet there, not by
    tuple order."""
    wanted = {leg_a, leg_b}
    for corner in state.corner_fillets:
        if set(corner) == wanted:
            return corner
    raise KeyError(f"No corner between {leg_a!r} and {leg_b!r} in this state.")


@dataclass(frozen=True)
class SetCornerRadius(Treatment):
    """Re-cut one corner's fillet at a different radius. Does NOT shorten a crossing here.

    This was called `bump_out` and its docstring claimed "the curb physically extends into the
    corner". It does not. It solves a new arc between two curb lines and leaves both of them
    where they were, so all it moves is the corner itself. Measured on
    broad_st_east x greenwood_ave_north, 29.2 -> 15.0 ft:

        arc length     19.48 -> 3.51 ft      the arc is genuinely re-cut
        trimmed_a     156.19 -> 164.19 ft    the curb only runs on to the new tangent point
        pavement area  23,989.7 -> 23,989.5 sq ft         0.2 sq ft of 24,000
        crossing spans unchanged to 0.00 ft on all four legs

    Nothing was wrong with the arithmetic; the claim was wrong. The crossings at these
    junctions sit 21-42 ft out, past the corner, so a radius change never reaches them. What
    DOES shorten a crossing is AddCurbExtension, which moves the kerb line laterally.

    Still a real operation, and the one a curb extension needs: the tightened face a curb
    extension presents to a passenger car IS a corner radius. `source` is recorded so the plan
    view can say whether a corner's radius was traced or chosen.
    """
    radius_ft: float = 0.0
    source: str = "designed"

    def __post_init__(self):
        if self.radius_ft <= 0:
            raise ValueError(f"A corner radius has to be positive; got radius_ft={self.radius_ft}.")

    def describe(self) -> str:
        return f"SetCornerRadius({self.target.key}, radius_ft={self.radius_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        _rebuild_corner(state, self.target.key, self.radius_ft, self.source)


def _rebuild_corner(state: DesignState, corner: tuple[str, str], radius_ft: float,
                     source: str) -> None:
    """Solve `corner`'s fillet afresh off whatever its two curb lines currently are.

    Mutates `state` - callers have already cloned. Separate from set_corner_radius because
    add_curb_extension needs it too: an extension moves a kerb, and the corner that kerb feeds
    has to be re-cut against the moved line or the pavement ring still follows the old one.
    """
    leg_a, leg_b = corner
    if leg_a not in state.legs or leg_b not in state.legs:
        raise KeyError(f"Corner {corner} references a leg not present in this state.")
    trimmed_a, arc, trimmed_b = fillet_curb_corner(
        state.legs[leg_a].left_curb, state.legs[leg_b].right_curb, radius_ft)
    state.corner_fillets[corner] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b,
                                     "radius_ft": radius_ft, "source": source}


@dataclass(frozen=True)
class RefugeIsland(Treatment):
    """A raised pedestrian refuge island splitting a leg's roadway, centered `offset_ft` from the
    intersection along the centerline.

    width_ft is the island's extent in the direction pedestrians cross (i.e.
    perpendicular to the road) - NACTO's minimum is 6 ft so a person/wheelchair
    can wait clear of both travel directions. along_road_ft is the island's
    length parallel to the road (how much of the crosswalk it shelters).
    """
    offset_ft: float = 0.0
    width_ft: float = NACTO_MIN_REFUGE_ISLAND_WIDTH_FT
    along_road_ft: float = 20
    name: str | None = None

    def __post_init__(self):
        if self.width_ft < NACTO_MIN_REFUGE_ISLAND_WIDTH_FT:
            raise ValueError(
                f"Refuge island width {self.width_ft} ft is below the NACTO minimum of "
                f"{NACTO_MIN_REFUGE_ISLAND_WIDTH_FT} ft.")
        # No along_road_ft check here: _band_across_the_road already refuses a zero-length span,
        # and it says which treatment asked for it and that the two stations resolved to the same
        # point - a better message than anything this constructor knows enough to write.

    def describe(self) -> str:
        return (f"RefugeIsland({self.target}, offset_ft={self.offset_ft}, "
                f"width_ft={self.width_ft})")

    def apply_to(self, state: "DesignState", model=None) -> None:
        leg = state.legs[self.target.leg]
        polygon = _band_across_the_road(leg.centerline, self.offset_ft - self.along_road_ft / 2,
                                         self.offset_ft + self.along_road_ft / 2,
                                         self.width_ft / 2,
                                         f"{self.width_ft:.0f} ft refuge island")
        island_name = self.name or f"{self.target.leg}_refuge_{int(self.offset_ft)}ft"
        state.refuge_islands[island_name] = {"polygon": polygon, "width_ft": self.width_ft}


@dataclass(frozen=True)
class RaiseCrossing(Treatment):
    """Mark the crosswalk over a leg's roadway (right at the intersection) as a raised crossing
    (speed table to sidewalk grade). In plan view this is just the crosswalk footprint, rendered
    distinctly; Phase 4 gives it height."""
    crossing_width_ft: float = 10

    def __post_init__(self):
        if self.crossing_width_ft <= 0:
            raise ValueError(f"A raised crossing needs a width; got {self.crossing_width_ft}.")

    def describe(self) -> str:
        return f"RaiseCrossing({self.target}, crossing_width_ft={self.crossing_width_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        leg = state.legs[self.target.leg]
        if leg.left_curb is None or leg.right_curb is None:
            raise ValueError(f"Leg {self.target.leg!r} has no curb lines (width unknown) - "
                              f"can't place a crossing on it.")
        # Start beyond the curve of this leg's corner fillets, not at the
        # intersection point itself - a crossing placed right at the corner point
        # lands inside the curb-return curve rather than on the straight section
        # of roadway where a real crosswalk would sit.
        start = leg_clearance_ft(self.target.leg, state.legs, state.corner_fillets)
        state.raised_crossings[self.target.leg] = _band_across_the_road(
            leg.centerline, start, start + self.crossing_width_ft, leg.curb_to_curb_ft / 2,
            f"{self.crossing_width_ft:.0f} ft raised crossing on {self.target.leg!r}")


VALID_CROSSWALK_STYLES = ("lines", "continental", "ladder")


@dataclass(frozen=True)
class UpgradeCrosswalkMarkings(Treatment):
    """Repaint a leg's crosswalk to a more visible marking style. FHWA/NACTO both
    rank visibility roughly lines < continental < ladder - "lines" (two thin
    transverse boundary lines) is what most of this intersection has today;
    upgrading to continental or ladder is a real, low-cost pedestrian-safety
    treatment on its own, independent of any geometry change."""
    style: str = "continental"

    def __post_init__(self):
        if self.style not in VALID_CROSSWALK_STYLES:
            raise ValueError(f"Unknown crosswalk style {self.style!r} - expected one of "
                              f"{VALID_CROSSWALK_STYLES}")

    def describe(self) -> str:
        return f"UpgradeCrosswalkMarkings({self.target}, style={self.style!r})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.crosswalk_styles[self.target.leg] = self.style


@dataclass(frozen=True)
class SetCenterlineStyle(Treatment):
    """Change what's painted down the middle of a leg: 'single_yellow_dashed'
    (ordinary two-way marking), 'double_yellow' (solid no-passing zone), or
    'none' (some real local streets have no centerline paint at all). Unlike
    UpgradeCrosswalkMarkings, this isn't a visibility ranking - it's just
    what's actually there, or a proposal's choice to change it - so any value
    is a valid target, not just an "upgrade."
    """
    style: str = DEFAULT_CENTERLINE_STYLE

    def __post_init__(self):
        if self.style not in VALID_CENTERLINE_STYLES:
            raise ValueError(f"Unknown centerline style {self.style!r} - expected one of "
                              f"{VALID_CENTERLINE_STYLES}")

    def describe(self) -> str:
        return f"SetCenterlineStyle({self.target}, style={self.style!r})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.centerline_styles[self.target.leg] = self.style


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

    def apply_to(self, state: "DesignState", model=None) -> None:
        leg_name = self.target.leg
        state.lane_narrowing[leg_name] = self.stripe_width_ft
        state.lane_narrowing_sides[leg_name] = tuple(str(side) for side in self.sides)
        if self.line_only:
            state.lane_narrowing_line_only.add(leg_name)
        else:
            state.lane_narrowing_line_only.discard(leg_name)


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
                    shares_a_kerb=(leg_name, side) in ctx.straight_through))
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


PARKING_STALL_DEPTH_DEFAULT_FT = 8.0  # AASHTO/NACTO typical parallel-parking lane depth (curb to travel-lane edge)
PARKING_STALL_LENGTH_DEFAULT_FT = 22.0  # AASHTO/NACTO typical parallel-parking stall length
LEGAL_PARKING_SETBACK_FT = 25.0  # NJSA 39:4-138: no stopping/standing/parking within 25 ft of a marked crosswalk at
                                  # an intersection - a real legal minimum, not a rendering choice. Marked parking
                                  # (src/render/export.py/plan_view.py) starts at max(this distance past the real
                                  # crosswalk, leg_clearance_ft's physical past-the-corner-curve point) - whichever
                                  # is farther from the intersection - so it never starts somewhere a car legally
                                  # couldn't park even if the curb geometry alone would allow it.


@dataclass(frozen=True)
class MarkedParking(Treatment):
    """Marked curbside parallel parking along one side of a leg: a lane-edge
    line depth_ft in from the curb, plus perpendicular divider ticks every
    stall_length_ft (src/geometry/model.py:parking_lane_edge_line_ft /
    parking_stall_lines_ft) - paint-only, zero curb/pavement change, same
    convention as LaneNarrowing/CornerHatching in that regard.
    Independent of LaneNarrowing - a leg can have marked parking with or
    without a separate travel-lane-narrowing buffer on the same or other
    side; nothing here assumes the two are combined, though a scenario is
    free to apply both (e.g. narrow the near lane while marking parking in
    what the SLD calls the far side's shoulder zone).

    curb_offset_ft > 0 pulls the parking lane in from the curb by that much,
    leaving a striped no-parking buffer between the curb and the parking
    lane itself (so parking sits directly against the active travel lane
    instead of against the curb) - src/geometry/paint.py paints that buffer with the
    same chevron treatment as a lane narrowing. 0 (the default) means the
    parking lane starts right at the curb, no buffer.
    """
    # Painted in the order the markings are layered: the kerbside zones first, and a
    # row of posts after the buffer it stands in - see paint.curbside_paint_ft.
    paint_group: ClassVar[int] = 20
    paint_rank: ClassVar[int] = 0
    depth_ft: float = PARKING_STALL_DEPTH_DEFAULT_FT
    stall_length_ft: float = PARKING_STALL_LENGTH_DEFAULT_FT
    curb_offset_ft: float = 0.0

    def __post_init__(self):
        # None of this was checked before. A zero-depth lane marked an edge line on top of the
        # kerb, and a stall shorter than a car claimed spaces that cannot exist.
        if self.depth_ft <= 0:
            raise ValueError(f"A parking lane needs a depth; got depth_ft={self.depth_ft}.")
        if self.stall_length_ft <= 0:
            raise ValueError(f"A stall needs a length; got stall_length_ft={self.stall_length_ft}.")
        if self.curb_offset_ft < 0:
            raise ValueError(f"A kerb buffer cannot be negative; got {self.curb_offset_ft}.")

    def describe(self) -> str:
        return (f"MarkedParking({self.target.leg}, side={str(self.target.side)!r}, "
                f"depth_ft={self.depth_ft}, stall_length_ft={self.stall_length_ft}, "
                f"curb_offset_ft={self.curb_offset_ft})")

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.parking_zones[self.target.key] = {
            "depth_ft": self.depth_ft, "stall_length_ft": self.stall_length_ft,
            "curb_offset_ft": self.curb_offset_ft,
        }


    def paint(self, ctx) -> None:
        """The stalls, the hatched buffer between them and the kerb, and the daylight zones
        where the law forbids parking at all."""
        from src.geometry.daylighting import merged_no_parking_spans_ft, no_parking_zones_ft
        from src.geometry.markings import (BUFFER_EDGE_LINE, BUFFER_FILL, DAYLIGHT_EDGE_LINE,
                                           DAYLIGHT_FILL, PARKING_EDGE_LINE, STALL_DIVIDER,
                                           ZONE_END_LINE)
        from src.geometry.model import (inset_line_ft, lane_narrowing_polygons_ft,
                                        parking_lane_edge_line_ft, parking_stall_lines_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, _one, end_against_crossing,
                                        lane_edge_stripes, parking_runs, zone_end_line_ft)

        leg_name, side = self.target.leg, str(self.target.side)
        state = ctx.state
        leg = state.legs[leg_name]
        depth_ft, stall_length_ft = self.depth_ft, self.stall_length_ft
        curb_offset_ft = self.curb_offset_ft
        at = ctx.anchors(leg_name, side,
                          inner_offset_ft=leg.curb_to_curb_ft / 2 - depth_ft - curb_offset_ft)
        runs = parking_runs(state, leg_name, side, ctx.crosswalk_offsets, ctx.props)

        # DAYLIGHTING. Every stretch where R.S. 39:4-138 forbids parking is hatched across
        # the FULL depth of the parking lane, not just the buffer strip beside it. Those
        # stretches were already no-parking in law - the treatment is MARKING them, because
        # an unmarked setback is one people park in, and an unmarked setback next to a
        # marked stall reads as more stall. This is the part of the proposal that actually
        # daylights the crossing. Zones are clipped to the leg and to the point where the
        # corner return leaves room to paint at all.
        #
        # The zone runs INTO the crossing and the crossing cuts its end, leaving it rimmed
        # along the crossing's own edge - a diagonal where the crossing is skewed, meeting
        # the straight lane-edge line at a right angle. It used to end in the same curved
        # taper a lane-narrowing buffer gets, and on a wide leg that curve is a hairpin: at
        # Broad St it had to swing the full 13-17 ft depth of the parking lane across 0-5.6 ft
        # of station. Where a leg has no marked crossing there is nothing to end against, so
        # it falls back to a taper if a gentle one exists and a square cut otherwise.
        daylight_line_ft, daylight_fill_ft = lane_edge_stripes(depth_ft + curb_offset_ft)
        lane_edge_offset_ft = leg.curb_to_curb_ft / 2 - daylight_line_ft
        for zone_start_ft, zone_end_ft in merged_no_parking_spans_ft(
                no_parking_zones_ft(state, leg_name, side, ctx.crosswalk_offsets, ctx.props)):
            if leg_name in ctx.marked and (leg_name, side) in ctx.straight_through:
                start_ft, beyond_ft = zone_start_ft, None
            elif leg_name in ctx.marked:
                start_ft, beyond_ft = end_against_crossing(at, zone_start_ft)
            else:
                start_ft, beyond_ft = max(zone_start_ft, at.target_ft), None
            ctx.rim(ctx.add(DAYLIGHT_FILL, _one(lane_narrowing_polygons_ft(
                leg, daylight_fill_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=zone_end_ft)), leg_name, side, beyond_ft,
                shares_a_kerb=(leg_name, side) in ctx.straight_through))
            # A solid line wherever hatching meets the travel lane, so the lane reads as a
            # lane. The buffer beside the stalls already has one; the daylight zone runs the
            # full depth of the parking lane, so ITS inner edge is the lane edge, and without
            # this the hatching just faded into the carriageway.
            ctx.add(DAYLIGHT_EDGE_LINE,
                     inset_line_ft(leg, side, lane_edge_offset_ft, start_ft, zone_end_ft,
                                    keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                     leg_name, side, beyond_ft)
            # Nothing to end against and no taper available: close the square end. See
            # zone_end_line_ft. Not where the kerb runs straight through - the zone carries
            # on into the next leg there.
            if leg_name not in ctx.marked and (leg_name, side) not in ctx.straight_through:
                ctx.add(ZONE_END_LINE, zone_end_line_ft(
                    leg, side, start_ft, leg.curb_to_curb_ft / 2 - daylight_fill_ft),
                    leg_name, side)

        for start_ft, end_ft in runs:
            # ORDER ACROSS THE ROAD, and what gives when the road's width changes:
            #
            #   travel lane   0 -> TARGET             fixed
            #   lane edge line                        its own width, out of the treatment
            #   parking       -> TARGET + depth_ft    fixed, held against the LANE
            #   HATCHING      -> the traced kerb      absorbs ALL of the variation
            #
            # Everything is measured from the centerline, so the only thing that touches the
            # traced kerb is the hatching - which is just paint filling whatever asphalt is
            # left over. The lane holds its width, which is the entire point of the markings:
            # a lane that widens is a lane people speed in. The stall holds its width too, so
            # the leftover cannot end up inside it.
            #
            # (Anchoring the stalls to the KERB instead was tried and is wrong here: it makes
            # the parking position depend on the noisiest input in the model, and puts the
            # variable-width hatching between the travel lane and the parked cars.)
            edge = parking_lane_edge_line_ft(
                leg, side, depth_ft, start_ft, end_ft,
                curb_offset_ft=curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT / 2)
            if edge is None:
                continue  # the corner return consumes the whole leg - see plan_view's note
            ctx.add(PARKING_EDGE_LINE, edge, leg_name, side)
            for divider in parking_stall_lines_ft(
                    leg, side, depth_ft, stall_length_ft, start_ft, end_ft,
                    curb_offset_ft=curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT):
                ctx.add(STALL_DIVIDER, divider, leg_name, side)

            if not curb_offset_ft:
                continue
            buffer_ft = max(curb_offset_ft - LANE_EDGE_LINE_WIDTH_FT, 0.0)
            ctx.add(BUFFER_EDGE_LINE, inset_line_ft(
                leg, side, leg.curb_to_curb_ft / 2 - buffer_ft, start_ft, end_ft,
                keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2), leg_name, side)
            ctx.add(BUFFER_FILL, _one(lane_narrowing_polygons_ft(
                leg, buffer_ft, start_left_ft=start_ft, start_right_ft=start_ft,
                sides=(side,), end_ft=end_ft)), leg_name, side)


BOLLARD_DEFAULT_SPACING_FT = 10.0  # typical flex-post delineator spacing for a channelized buffer


@dataclass(frozen=True)
class ParkingBufferBollards(Treatment):
    """Plastic bollards (flex-post delineators) centered in the striped
    no-parking buffer between a marked-parking lane and the curb - i.e. on
    the OUTSIDE of the parking lane (the curb side), protecting/delineating
    parked cars from that buffer, the mirror image of LaneNarrowingBollards (which
    centers bollards in a lane-narrowing buffer on the travel-lane side).
    Requires MarkedParking to already be applied to this (leg, side) with
    curb_offset_ft > 0 - there's no buffer to put bollards in otherwise."""
    paint_group: ClassVar[int] = 20
    paint_rank: ClassVar[int] = 1
    spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT

    def __post_init__(self):
        if self.spacing_ft <= 0:
            raise ValueError(f"Posts need a spacing; got spacing_ft={self.spacing_ft}.")

    def describe(self) -> str:
        return (f"ParkingBufferBollards({self.target.leg}, "
                f"side={str(self.target.side)!r}, spacing_ft={self.spacing_ft})")

    def apply_to(self, state: "DesignState", model=None) -> None:
        zone = state.parking_zones.get(self.target.key)
        if zone is None:
            raise KeyError(f"{self.target} has no marked parking - apply MarkedParking first.")
        if not zone["curb_offset_ft"]:
            raise ValueError(f"{self.target}'s marked parking has curb_offset_ft=0 - no curb "
                              f"buffer to put bollards in.")
        state.parking_buffer_bollards[self.target.key] = self.spacing_ft


    def paint(self, ctx) -> None:
        """Down the buffer between the stalls and the kerb, over the runs where stalls are marked.

        The buffer's width belongs to the MarkedParking treatment underneath, so it is read from
        the design rather than restated here - the same reason this treatment refuses a parking
        lane with no buffer.
        """
        from src.geometry.markings import BOLLARD
        from src.geometry.model import bollard_points_ft
        from src.geometry.paint import PaintPiece, _dot, parking_runs

        leg_name, side = self.target.leg, str(self.target.side)
        curb_offset_ft = ctx.state.parking_zones[self.target.key]["curb_offset_ft"]
        leg = ctx.state.legs[leg_name]
        for start_ft, _end_ft in parking_runs(ctx.state, leg_name, side, ctx.crosswalk_offsets,
                                               ctx.props):
            for point in bollard_points_ft(leg, curb_offset_ft, start_ft, self.spacing_ft,
                                            sides=(side,)):
                ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, side))


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
        if self.target.leg not in state.lane_narrowing:
            raise KeyError(f"Leg {self.target.leg!r} has no lane_narrowing buffer - apply "
                            f"LaneNarrowing first.")
        state.bollard_lines[self.target.leg] = self.spacing_ft


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
        stripe_width_ft = ctx.state.lane_narrowing[leg_name]
        sides = ctx.state.lane_narrowing_sides.get(leg_name, ("left", "right"))
        for point in bollard_points_ft(
                leg, stripe_width_ft,
                leg_clearance_ft(leg_name, ctx.state.legs, ctx.state.corner_fillets),
                self.spacing_ft, sides=sides):
            ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, None))


@dataclass(frozen=True)
class CornerHatching(Treatment):
    """Paint-only diagonal hatching in a corner's gutter zone: a visual
    narrowing cue with zero curb/fillet geometry change - the paint-only
    alternative to a real curb extension at the same corner."""
    # Last: a corner treatment is cut around every kerbside zone that reaches the corner.
    paint_group: ClassVar[int] = 90
    depth_ft: float = CORNER_HATCHING_DEFAULT_DEPTH_FT

    def __post_init__(self):
        if self.depth_ft <= 0:
            raise ValueError(f"Corner hatching needs a depth; got depth_ft={self.depth_ft}.")

    def describe(self) -> str:
        return f"CornerHatching({self.target.key}, depth_ft={self.depth_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.corner_hatching[self.target.key] = self.depth_ft

    def paint(self, ctx) -> None:
        from src.geometry.model import corner_overlay_polygon
        from src.geometry.markings import CORNER_HATCH_FILL

        fillet = ctx.state.corner_fillets[self.target.key]
        if "error" in fillet:
            return          # a corner whose fillet failed has no gutter zone to hatch
        # No leg or side: a corner treatment spans the corner between two legs and belongs to
        # neither, which is why the kerb checks skip a piece with no side rather than guessing.
        ctx.add(CORNER_HATCH_FILL, corner_overlay_polygon(fillet, ctx.center_ft, self.depth_ft))


@dataclass(frozen=True)
class MountableApron(Treatment):
    """Mountable apron: a textured (not painted-line) surface treatment at a
    corner, flush with the existing pavement grade - visually/optically
    narrows the corner for pedestrians while remaining fully drivable (e.g. by
    a fire apparatus's rear wheels during a wide turn) since no curb or
    elevation change is introduced. Same footprint as CornerHatching, a
    different real-world treatment for corners where a hard bump-out isn't an
    option (see fire_apparatus_constraint in a proposal's spec).

    A FIXED DEPTH inward from the corner arc. Where the apron exists to preserve a large
    vehicle's swept path around a tightened corner, its depth is not free - it has to reach the
    radius that vehicle needs - so AddCurbExtension records CornerApron(swept_radius_ft=...)
    instead and the annulus is built from the two radii. See CornerApron.
    """
    paint_group: ClassVar[int] = 0
    extent_ft: float = CORNER_APRON_DEFAULT_EXTENT_FT

    def describe(self) -> str:
        return f"MountableApron({self.target.key}, extent_ft={self.extent_ft})"

    @property
    def apron(self) -> CornerApron:
        """CornerApron validates the depth/radius exclusivity itself."""
        return CornerApron(depth_ft=self.extent_ft)

    def apron_corner(self, state) -> tuple[str, str] | None:
        return self.target.key

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.corner_aprons[self.target.key] = self.apron

    def paint(self, ctx) -> None:
        """The apron surface, laid in the SURFACE pass so every marking is cut around it.

        Its own apron, from its own fields, rather than whatever is in state.corner_aprons for
        this corner: the dict holds one entry per corner, so reading from it would let two
        treatments that each asked for an apron there paint the same one twice. A corner with two
        aprons specified is a design error, and painting both is what makes
        MarkingsDoNotCollide say so.
        """
        from src.geometry.markings import APRON
        from src.geometry.paint import apron_polygon

        corner = self.apron_corner(ctx.state)
        if corner is None or "error" in ctx.state.corner_fillets[corner]:
            return
        ctx.add_surface(APRON, apron_polygon(ctx.state, corner, self.apron, ctx.center_ft))


# How wide a face a tightened corner presents to a passenger car. The design figure for the
# bulb-outs at Broad & Greenwood: a 15 ft radius is a corner a car has to slow for, and the
# apron behind it (see CornerApron) hands the larger radius back to a bus or a truck. A design
# choice, not a measurement - which is exactly why the apron's own radius is measured.
CURB_EXTENSION_FACE_RADIUS_FT = 15.0


@dataclass(frozen=True)
class AddCurbExtension(Treatment):
    """A real curb extension: move this kerb `extension_ft` into the roadway and taper it back.

    This is the treatment SetCornerRadius was mistaken for. It changes the KERB LINE, so
    everything downstream that measures against the kerb follows it without being told:
    the crossing gets shorter (src/render/crosswalks.py:crosswalk_reach_to_curbs_ft walks out
    to the real kerb), the pavement polygon loses the corner, the kerbside paint rebuilds
    against the new edge, and the invariants check the geometry that results.

    HOW LONG. The face runs from the junction to `crossing_ft` plus half a crossing plus the
    10 ft R.S. 39:4-138(e) setback that the extension itself buys - i.e. it covers exactly the
    kerb where parking is prohibited once this is built - then tapers back over
    `extension_ft * BULBOUT_TAPER_RATE`. Nothing about the length is chosen to look right; it
    is the statutory zone plus a stated taper rate.

    HOW FAR. Bounded by the travel lane that has to survive: an extension deeper than the
    leg's spare width beside a TARGET_LANE_WIDTH_FT lane is refused rather than clamped,
    because silently building a shallower bulb-out than the caller asked for is how a drawing
    stops matching its own description. At Broad & Greenwood that permits the 8 ft asked for on
    both Broad legs (15.0 and 16.8 ft spare per side) and refuses it on Greenwood (2.3 and
    4.6 ft) - which is the finding, not an obstacle: Greenwood cannot hold a bulb-out and two
    11 ft lanes at once.

    WHAT IT COSTS IN PARKING. Nothing, at Broad & Greenwood. Schedule I of the borough code
    prohibits parking 100 ft each way on both sides of both Broad legs, and the whole footprint
    - face plus taper - fits inside that, so the extension occupies kerb that is already
    legally not-parking. A curb extension normally trades spaces for safety; here it does not,
    and that is the strongest thing that can be said for it.

    `swept_radius_ft` is the corner's OWN measured radius, and passing it lays a mountable
    apron over the annulus between that and `face_radius_ft` so a bus keeps the path it has
    today. On CR 518, a rural arterial carrying buses and trucks, that is not optional.
    """
    # Laid in the surface pass: built ground, and every marking is cut around it.
    paint_group: ClassVar[int] = 0
    extension_ft: float = 0.0
    crossing_ft: float = 0.0
    swept_radius_ft: float | None = None
    face_radius_ft: float = CURB_EXTENSION_FACE_RADIUS_FT
    taper_ft: float | None = None

    def __post_init__(self):
        if self.extension_ft <= 0:
            raise ValueError(f"An extension has to move the kerb; got "
                             f"extension_ft={self.extension_ft}.")
        if self.face_radius_ft <= 0:
            raise ValueError(f"The face is a corner radius; got {self.face_radius_ft}.")
        if self.swept_radius_ft is not None and self.swept_radius_ft <= self.face_radius_ft:
            raise ValueError(
                f"The swept radius ({self.swept_radius_ft} ft) is the corner a bus keeps via the "
                f"apron, so it has to be LARGER than the {self.face_radius_ft} ft face a car "
                f"sees - an annulus between them is what the apron is (see CornerApron).")

    @property
    def apron(self) -> CornerApron | None:
        """The annulus a bus keeps, or None where no swept radius was measured.

        A fixed depth cannot support the claim this apron exists to make - that the swept path
        survives the tightened corner - because nothing ties a depth to the radius a vehicle
        needs. See CornerApron.
        """
        if self.swept_radius_ft is None:
            return None
        return CornerApron(swept_radius_ft=self.swept_radius_ft,
                            face_radius_ft=self.face_radius_ft)

    def apron_corner(self, state) -> tuple[str, str] | None:
        """The corner this moved kerb feeds - not this treatment's own target.

        build_corner_fillets pairs leg A's LEFT curb with leg B's RIGHT, so which corner a kerb
        belongs to depends on the side. This is why the apron pass is ordered by corner rather
        than by target: a curb extension is aimed at a leg-side and lays ground at a corner.
        """
        return _corner_fed_by(state, self.target.leg, str(self.target.side))

    def paint(self, ctx) -> None:
        """The swept-path apron, in the SURFACE pass so every marking is cut around it.

        Its own apron rather than whatever state.corner_aprons holds for this corner: the dict has
        one entry per corner, so reading it would let two treatments that each asked for an apron
        there paint the same ground twice. A corner with two aprons specified is a design error,
        and painting both is what makes MarkingsDoNotCollide say so.
        """
        from src.geometry.markings import APRON
        from src.geometry.paint import apron_polygon

        apron = self.apron
        corner = self.apron_corner(ctx.state)
        if apron is None or corner is None or "error" in ctx.state.corner_fillets[corner]:
            return
        ctx.add_surface(APRON, apron_polygon(ctx.state, corner, apron, ctx.center_ft))

    def describe(self) -> str:
        return f"AddCurbExtension({self.target.leg}, {self.target.side}): "

    def apply_to(self, state: "DesignState", model=None) -> str:
        # Local: src/render/crosswalks.py imports DesignState from here, and
        # src/geometry/daylighting.py reads CURB_EXTENSION_DEVICES from here - both cycles back.
        # The statutory figure and the crossing depth are single-sourced in those modules and must
        # not be copied, since the whole length of the face is measured off them.
        from src.geometry.daylighting import CROSSWALK_SETBACK_WITH_BULBOUT_FT
        from src.render.crosswalks import CROSSWALK_DEPTH_FT

        leg_name, side = self.target.leg, str(self.target.side)
        leg = state.legs[leg_name]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {leg_name!r} has no width - nothing to measure an extension from.")

        spare_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        if self.extension_ft > spare_ft + LANE_WIDTH_SLACK_FT:
            raise ValueError(
                f"A {self.extension_ft:.1f} ft curb extension on {leg_name} {side} would leave a "
                f"{leg.curb_to_curb_ft / 2 - self.extension_ft:.1f} ft travel lane, under the "
                f"{TARGET_LANE_WIDTH_FT:.0f} ft target. That leg is {leg.curb_to_curb_ft:.1f} ft "
                f"curb to curb, so it has {spare_ft:.1f} ft per side to give.")

        taper_ft = (self.extension_ft * BULBOUT_TAPER_RATE if self.taper_ft is None
                    else self.taper_ft)
        full_ft = self.crossing_ft + CROSSWALK_DEPTH_FT / 2 + CROSSWALK_SETBACK_WITH_BULBOUT_FT
        built = curb_extension_line(leg, side, self.extension_ft, full_ft, taper_ft)
        if built is None:
            raise ValueError(
                f"{leg_name} {side} has no traced kerb to extend - a curb extension is measured "
                f"from the kerb that is there, and nothing is mapped on that side.")

        setattr(state.legs[leg_name], f"{side}_curb", built)
        state.curb_extensions[self.target.key] = CurbExtension(
            extension_ft=self.extension_ft, full_ft=full_ft, taper_ft=taper_ft,
            face_radius_ft=self.face_radius_ft, swept_radius_ft=self.swept_radius_ft)
        # The corner this kerb feeds has to be re-cut against the line that moved, or the pavement
        # ring keeps following the kerb that is no longer there. build_corner_fillets pairs leg A's
        # LEFT curb with leg B's RIGHT, so which corner that is depends on the side.
        corner = self.apron_corner(state)
        if corner is not None:
            _rebuild_corner(state, corner, self.face_radius_ft, "curb_extension")
            if self.apron is not None:
                state.corner_aprons[corner] = self.apron
        return (f"kerb moved {self.extension_ft:.1f} ft into the roadway to station "
                f"{full_ft:.0f} ft, tapering back over {taper_ft:.0f} ft; "
                f"{self.face_radius_ft:.0f} ft face"
                + (f" with a mountable apron out to the corner's measured "
                   f"{self.swept_radius_ft:.1f} ft" if self.swept_radius_ft is not None else "")
                + f". Leaves a {leg.curb_to_curb_ft / 2 - self.extension_ft:.1f} ft travel lane.")


def _corner_fed_by(state: DesignState, leg_name: str, side: str) -> tuple[str, str] | None:
    """The corner whose fillet is built from this (leg, side)'s curb line, or None.

    build_corner_fillets' contract: a corner keyed (A, B) is bounded by A's LEFT curb and B's
    RIGHT curb. So a left side feeds the corner it is first in, a right side the corner it is
    second in - and each side feeds exactly one.
    """
    wanted = 0 if side == "left" else 1
    for corner in state.corner_fillets:
        if corner[wanted] == leg_name:
            return corner
    return None


# AASHTO's minimum width for an exclusive on-street bike lane. A hard floor, not a target: a
# lane narrower than this is not a bike lane, and proposing one would be proposing something
# that fails its own standard. It is what rules Greenwood Ave (2.3 and 4.6 ft spare per side)
# and Princeton Ave (4.1) out of a bike lane entirely - see build_proposal_bike_lanes.
AASHTO_MIN_BIKE_LANE_FT = 5.0
# A bike lane hard against the kerb loses its outer foot or so to the gutter pan and to riders
# keeping clear of the kerb. Holding the lane off the kerb by a shy distance instead buys back
# usable width without claiming a wider lane than exists. Used on E Broad, a truck route, where
# 5 ft of lane plus 2 ft of shy reads better than 6 ft of lane against the kerb.
BIKE_LANE_DEFAULT_SHY_FT = 2.0


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

    def __post_init__(self):
        if self.width_ft < AASHTO_MIN_BIKE_LANE_FT:
            raise ValueError(
                f"A {self.width_ft:.1f} ft bike lane is under AASHTO's {AASHTO_MIN_BIKE_LANE_FT:.0f} ft "
                f"minimum for an exclusive lane. Draw no lane rather than one that fails the "
                f"standard it is meant to meet.")
        if self.buffer_ft and self.buffer_ft < 2 * _lane_line_ft():
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

    def offsets_from_centerline_ft(self) -> dict:
        """Where each boundary sits, as a distance from the centerline.

        One place, so the plan view, the 3D export and the checks cannot disagree about which
        stripe is which - the ordering across the road IS the design. Each `*_line_ft` is the
        stripe's CENTRE, offset half a stripe outward from the face it marks so the protected
        width behind it stays whole.
        """
        line_ft = _lane_line_ft()
        travel_edge = TARGET_LANE_WIDTH_FT
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


def _lane_line_ft() -> float:
    """The painted width of one edge line. Local import: src/geometry/paint.py imports this
    module, and the figure is single-sourced there against what the 3D renderer actually lays."""
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT

    return LANE_EDGE_LINE_WIDTH_FT


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

    def __post_init__(self):
        self.lane        # raises for a lane under AASHTO's minimum, before anything is applied

    def describe(self) -> str:
        # leg, side rather than str(target): a note is meant to read as the constructor call
        # that produced it, so someone reading a render's provenance can paste it back.
        return (f"AddBikeLane({self.target.leg}, {self.target.side}): {self.width_ft:.0f} ft lane"
                + (f", {self.buffer_ft:.0f} ft buffer" if self.buffer_ft else "")
                + (f", parking-protected behind {self.parking_ft:.0f} ft of marked parking"
                   if self.parking_ft
                   else f", {self.shy_ft:.1f} ft shy of the kerb" if self.shy_ft else ""))

    def apply_to(self, state: "DesignState", model=None) -> None:
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
        state.bike_lanes[self.target.key] = lane
        spare_ft = available_ft - lane.total_ft
        return (f". Uses {lane.total_ft:.1f} of the {available_ft:.1f} ft this leg has at its "
                f"narrowest" + (f", {spare_ft:.1f} ft spare." if spare_ft > 0.05 else "."))


    def paint(self, ctx) -> None:
        """An edge line each side of the lane, so it reads as a lane rather than as the spare
        asphalt a lane-narrowing buffer marks; the buffer beside it, and the parking outside it,
        hatched and ticked with the machinery already here."""
        from src.geometry.markings import (BIKE_BUFFER_FILL, BIKE_LANE_EDGE_LINE, BUFFER_FILL,
                                           STALL_DIVIDER)
        from src.geometry.model import (curbside_strip_polygon, inset_line_ft,
                                        lane_narrowing_polygons_ft, parking_stall_lines_ft)
        from src.geometry.paint import (LANE_EDGE_LINE_WIDTH_FT, _one, end_against_crossing,
                                        parking_runs)

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        lane = self.lane
        at = ctx.anchors(leg_name, side, inner_offset_ft=(
            leg.curb_to_curb_ft / 2 - lane.total_ft + TARGET_LANE_WIDTH_FT))
        # A bike lane RUNS INTO its crossing and is cut by it, like every other kerbside zone
        # here - a real one carries on to the crossing and often across it. Stopping it at the
        # corner clearance instead left the buffer 5.5 ft short of the crossing, which
        # test_curbside_paint_ends_against_its_crossing reads as hatching that gave up early.
        if (leg_name, side) in ctx.straight_through:
            start_ft, beyond_ft = 0.0, None
        elif leg_name in ctx.marked:
            start_ft, beyond_ft = end_against_crossing(at)
        else:
            start_ft, beyond_ft = at.target_ft, None
        bounds = lane.offsets_from_centerline_ft()
        # Every stripe at its own CENTRE, which BikeLane has already offset half a stripe out
        # from the face it marks - so the travel lane keeps its 11 ft and the bike lane keeps
        # its own width, and the paint comes out of the buffer between them. Getting this wrong
        # is not subtle: an edge line centred on the mark leaves a 10.59 ft lane, which
        # PaintClearOfTheTravelLane reports on every vertex.
        for key in ("inner_line_ft", "buffer_outer_line_ft", "outer_line_ft"):
            if bounds[key] is None:
                continue
            ctx.add(BIKE_LANE_EDGE_LINE,
                     inset_line_ft(leg, side, bounds[key], start_ft,
                                    keep_inside_ft=LANE_EDGE_LINE_WIDTH_FT / 2),
                     leg_name, side, beyond_ft)
        if lane.buffer_ft:
            # The hatched buffer, between the two lines that bound it rather than under them.
            # lane_narrowing_polygons_ft measures its stripe inward from the kerb-to-kerb half,
            # so the depth is the distance from the kerb to the buffer's inner FACE, and the
            # zone is then cut back to the buffer's outer face.
            inner_face_ft = bounds["travel_lane_edge_ft"] + LANE_EDGE_LINE_WIDTH_FT
            fill = _one(lane_narrowing_polygons_ft(
                leg, leg.curb_to_curb_ft / 2 - inner_face_ft,
                start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,)))
            outer_face_ft = bounds["bike_inner_ft"] - LANE_EDGE_LINE_WIDTH_FT
            beyond = curbside_strip_polygon(leg, side, outer_face_ft, start_ft)
            if fill is not None and beyond is not None:
                fill = fill.difference(beyond)
            ctx.rim(ctx.add(BIKE_BUFFER_FILL, fill, leg_name, side, beyond_ft))
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
            ctx.rim(ctx.add(BUFFER_FILL, _one(lane_narrowing_polygons_ft(
                leg, leg.curb_to_curb_ft / 2 - bounds["outer_ft"],
                start_left_ft=start_ft, start_right_ft=start_ft, sides=(side,))),
                leg_name, side, beyond_ft))


def resolved_crossing_stations(model, state: DesignState) -> dict:
    """{leg name: the station its crossing is resolved to}, for treatments measured off it.

    A curb extension has to cover its leg's crossing, so it needs to know where that crossing
    is - and the answer is resolved data, not a parameter: a real OSM-surveyed position where
    one was matched, else the geometric estimate. Reaching for it here rather than making every
    scenario re-derive it is what keeps a bulb-out's length tied to the same crossing the
    renderers draw.

    Local imports for the usual cycle: src/render/crosswalks.py imports DesignState from here.
    """
    from src.render.crosswalks import resolve_crosswalk_offsets
    from src.sources.osm_context import fetch_crossings

    crossings = fetch_crossings(model.center_wgs84, radius_m=CROSSING_CONTEXT_RADIUS_M)
    return {name: offset.offset_ft
            for name, offset in resolve_crosswalk_offsets(state, crossings).items()}


# Matches src/render/export.py and src/render/plan_view.py, so a crossing resolved for a
# treatment is the same crossing the renderers resolve rather than one from a different radius.
CROSSING_CONTEXT_RADIUS_M = 130


def bulb_out_corner_pair(state: DesignState, leg_name: str, extension_ft: float,
                          crossing_ft: float, sides: tuple = ("left", "right")) -> DesignState:
    """Curb extensions on both kerbs of one leg, each corner's apron out to its OWN traced radius.

    The apron radius is READ FROM THE BASELINE FILLET rather than passed in, which is the point:
    the four corners at Broad & Greenwood are traced at 29.2, 24.6, 29.0 and 22.9 ft, and a
    scenario repeating those as literals would keep whatever they were the day it was written.
    Re-tracing a kerb in OSM now flows through to the apron by itself.
    """
    for side in sides:
        corner = _corner_fed_by(state, leg_name, side)
        swept_radius_ft = None if corner is None else state.corner_fillets[corner].get("radius_ft")
        state = state.apply(
            AddCurbExtension(LegSide(leg_name, side), extension_ft=extension_ft,
                              crossing_ft=crossing_ft, swept_radius_ft=swept_radius_ft),
            ProtectDaylightZone(LegSide(leg_name, side), kind="curb_extension"))
    return state


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

    def apply_to(self, state: "DesignState", model=None) -> str:
        lane = state.bike_lanes.get(self.target.key)
        if lane is None:
            raise KeyError(f"{self.target} has no bike lane - apply AddBikeLane first.")
        if not lane.buffer_ft:
            raise ValueError(
                f"{self.target}'s bike lane has no buffer, so there is nowhere to stand a "
                f"delineator that is not in a travel lane or in the bike lane itself. A protected "
                f"lane needs a buffer; give it one, or leave the lane conventional and say so.")
        state.bike_lane_bollards[self.target.key] = self.spacing_ft
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

        The lane's cross-section belongs to the AddBikeLane underneath, so it is read from the
        design rather than restated - the same reason this treatment requires one.
        """
        from src.geometry.markings import BOLLARD
        from src.geometry.model import points_at_offset_ft
        from src.geometry.paint import PaintPiece, _dot, end_against_crossing

        leg_name, side = self.target.leg, str(self.target.side)
        leg = ctx.state.legs[leg_name]
        lane = ctx.state.bike_lanes[self.target.key]
        bounds = lane.offsets_from_centerline_ft()
        at = ctx.anchors(leg_name, side, inner_offset_ft=(
            leg.curb_to_curb_ft / 2 - lane.total_ft + TARGET_LANE_WIDTH_FT))
        if (leg_name, side) in ctx.straight_through:
            start_ft = 0.0
        elif leg_name in ctx.marked:
            start_ft, _beyond_ft = end_against_crossing(at)
        else:
            start_ft = at.target_ft
        centre_ft = (bounds["travel_lane_edge_ft"] + bounds["bike_inner_ft"]) / 2
        for point in points_at_offset_ft(leg, side, centre_ft, max(start_ft, at.target_ft),
                                          spacing_ft=self.spacing_ft):
            ctx.emit(PaintPiece(BOLLARD, _dot(point), leg_name, side))


@dataclass(frozen=True)
class ShiftCrosswalk(Treatment):
    """Shift a leg's crosswalk further from (positive) or closer to (negative)
    the intersection, on top of whatever src/render/crosswalks.py:resolve_crosswalk_offsets
    would otherwise resolve (a real OSM-surveyed position or the geometric
    curve-clearance estimate) - e.g. to give a turning fire apparatus more room
    before it encounters the crosswalk mid-turn.

    Accumulates rather than replaces, so two shifts of the same leg add up - which is what a
    dict of overrides already did, and worth stating since it is the one treatment here that is
    not idempotent.
    """
    delta_ft: float = 0.0

    def describe(self) -> str:
        return f"ShiftCrosswalk({self.target}, delta_ft={self.delta_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.crosswalk_offset_overrides[self.target.leg] = (
            state.crosswalk_offset_overrides.get(self.target.leg, 0.0) + self.delta_ft)


@dataclass(frozen=True)
class ExtraProp(Treatment):
    """Add one scenario-specific street-furniture prop (e.g. an RRFB, a
    relocated school-zone sign) along a leg - the treatment-level equivalent of
    a site config's `props.extra` (see sites/README.md), for props that only
    belong to this particular proposal, not every scenario at this site.

    offset_ft defaults to None, meaning "place it at this leg's real resolved
    crosswalk offset" (src/render/props.py:_extra_props_from_state falls back to it,
    same as _extra_props_from_config does for site-config props) - an RRFB or
    a relocated crossing sign belongs AT the crossing, and a real OSM-surveyed
    crosswalk can sit much farther from the corner than a small guessed
    number (e.g. ~42 ft on greenwood_ave_south here) - a hardcoded offset_ft
    can easily land inside the curb-return curve, in the roadway, instead of
    on the sidewalk. Only pass an explicit offset_ft when the prop genuinely
    belongs somewhere other than the crosswalk.
    """
    prop_type: str = ""
    offset_ft: float | None = None
    side: Side = Side.RIGHT
    note: str = ""

    def __post_init__(self):
        if not self.prop_type:
            raise ValueError("An extra prop needs a type - it is what decides what is drawn.")
        object.__setattr__(self, "side", Side(self.side))

    def describe(self) -> str:
        return f"ExtraProp({self.target}, {self.prop_type!r}, offset_ft={self.offset_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        state.extra_props.append({"leg": self.target.leg, "type": self.prop_type,
                                   "offset_ft": self.offset_ft, "side": str(self.side),
                                   "note": self.note})


def build_sidewalk_pieces(state: DesignState, sidewalk_width_ft: float = 6) -> list[Polygon]:
    """A sidewalk band hugging the real kerb, all the way round the junction.

    Built by widening each piece of the ACTUAL pavement boundary - the same
    (trimmed_a, arc, trimmed_b) that build_pavement_polygon walks - and cutting the roadway
    back out. Every piece therefore follows the surveyor's traced kerb by construction,
    including round the corner returns.

    It used to re-derive the outer edge instead: same centerlines, widened by
    2 * sidewalk_width_ft, re-filleted. That was correct only while the curbs were symmetric
    offsets of the NJDOT centerline, and they stopped being that when they became traced
    kerbs. Measured against the real pavement, 11-19% of the kerb had no sidewalk against it
    at all - gaps up to 27 ft where grass ran straight up to the roadway - and at W Broad
    658 sq ft of "sidewalk" lay inside the carriageway.

    Emitted as separate pieces rather than one ring on purpose: the renderer draws a
    polygon from its exterior only (scripts/blender/blender_scene.py:extrude_polygon), so a
    ring-with-a-hole would come out as a slab over the whole intersection.
    """
    try:
        pavement = build_pavement_polygon(state.corner_fillets)
    except ValueError:
        return []   # no closed roadway to lay a sidewalk against

    pieces = []
    for corner, parts in state.corner_fillets.items():
        if "error" in parts:
            continue
        for key in ("trimmed_a", "arc", "trimmed_b"):
            edge = parts.get(key)
            if edge is None or edge.is_empty or edge.length <= 0:
                continue
            # Flat caps: a round cap would spill a half-disc past the end of each leg.
            band = edge.buffer(sidewalk_width_ft, cap_style=2).difference(pavement)
            if band.is_empty:
                continue
            pieces.extend(g for g in getattr(band, "geoms", [band])
                          if g.geom_type == "Polygon" and g.is_valid and not g.is_empty)
    return pieces


# Spacing by device. A flex-post line reads as a delineator at 8 ft. A curb extension has no
# spacing - it is one continuous kerb, not a row of objects.
DAYLIGHT_DEVICE_SPACING_FT = {"bollards": 8.0, "curb_extension": 0.0}
# Devices drawn as a row of physical objects standing in the zone. A curb extension is not one:
# it is built ground, already drawn as the kerb itself (add_curb_extension moves the curb line),
# so src/render/props.py must not also stand posts along it.
DAYLIGHT_DEVICES_AS_POSTS = frozenset({"bollards"})
# Which devices the statute's "curb extension or bulbout has been constructed" clause covers,
# cutting the setback in R.S. 39:4-138(e) from 25 ft to 10 ft.
#
# `curb_extension` is in it because add_curb_extension builds the thing the clause names: the
# kerb line moves and the pavement polygon loses the corner, so the parking lane is physically
# out of the sight line rather than painted out of it. A flex-post delineator is NOT - it bends
# flat under a tyre - and planters were listed here once and are not any more, because the
# argument that a row of them occupies the corner the way a built bulbout does was never the
# Borough's to concede.
CURB_EXTENSION_DEVICES: frozenset = frozenset({"curb_extension"})
VALID_DAYLIGHT_DEVICES = ("bollards", "curb_extension")


@dataclass(frozen=True)
class ProtectDaylightZone(Treatment):
    """Stand physical objects in the daylight zone so it is not merely painted.

    An unmarked statutory setback gets parked in; a painted one gets parked in less. Objects
    in it get parked in not at all, and that is the difference between a drawing of the law
    and a street that enforces it.

    `kind` can matter legally, not just visually: R.S. 39:4-138(e) cuts the 25 ft setback to
    10 ft "if a curb extension or bulbout has been constructed", so a device that counts as one
    buys back kerb for parking. `curb_extension` does and `bollards` does not - a flex-post
    bends flat under a tyre. See src/geometry/daylighting.py for where that is applied, and
    CURB_EXTENSION_DEVICES for why the set has one member and not two.

    Declaring `curb_extension` here is what makes the statutory reduction apply; it does not
    BUILD anything. AddCurbExtension moves the kerb. The two go together, and its caller is
    expected to declare the device as well - which is why the note below says which of the two
    setbacks now governs.
    """
    kind: str = "bollards"
    spacing_ft: float | None = None

    def __post_init__(self):
        if self.kind not in VALID_DAYLIGHT_DEVICES:
            raise ValueError(f"kind must be one of {VALID_DAYLIGHT_DEVICES}, got {self.kind!r}")
        if self.spacing_ft is not None and self.spacing_ft <= 0:
            raise ValueError(f"Devices need a spacing; got spacing_ft={self.spacing_ft}.")

    @property
    def resolved_spacing_ft(self) -> float:
        return (DAYLIGHT_DEVICE_SPACING_FT[self.kind] if self.spacing_ft is None
                else self.spacing_ft)

    def describe(self) -> str:
        return f"ProtectDaylightZone({self.target.leg}, {self.target.side}): "

    def apply_to(self, state: "DesignState", model=None) -> str:
        # The one check that is about the LAW rather than about the street, and it depends on
        # another treatment having been applied - so it belongs here, where the design is
        # visible, not in the constructor.
        if self.kind in CURB_EXTENSION_DEVICES and self.target.key not in state.curb_extensions:
            raise ValueError(
                f"{self.target} is declared as a {self.kind!r} daylight device, which cuts the "
                f"R.S. 39:4-138(e) setback from 25 ft to 10 ft - but no curb extension has been "
                f"built there. Apply AddCurbExtension first; the statute's reduction is for an "
                f"extension that EXISTS, and claiming it without one would let a proposal mark "
                f"parking 15 ft closer to a crossing than the law allows.")
        spacing_ft = self.resolved_spacing_ft
        state.daylight_devices[self.target.key] = {"kind": self.kind, "spacing_ft": spacing_ft}
        return (f"{self.kind} at {spacing_ft:.0f} ft spacing"
                + (" - counts as a curb extension, so R.S. 39:4-138(e) allows parking from 10 ft "
                   "rather than 25 ft" if self.kind in CURB_EXTENSION_DEVICES else ""))


def apply_osm_parking(state: DesignState, model, depth_ft: float = PARKING_STALL_DEPTH_DEFAULT_FT,
                       stripe_width_ft: float = LANE_NARROWING_DEFAULT_STRIPE_FT,
                       legs: tuple | None = None) -> DesignState:
    """Paint each kerb according to what OSM says about parking there.

    `legs` limits it to the legs named, leaving the rest of the junction bare. Not a
    rendering convenience - a scenario that treats two legs of a crossroads and not the
    other two is a real proposal, and Columbia Ave is one (see
    sites/columbia_princeton/scenarios.py).

    Restricted (parking:*:restriction = no_parking / no_standing / no_stopping) gets crossed
    hatching - that kerb cannot hold parked cars, and a proposal that drew stalls there
    would be proposing something illegal. Everything else gets marked stalls: both an
    explicit restriction=none, which is a positive statement that parking is allowed, and an
    untagged side, which is the ordinary residential-street default.

    A RESTRICTION OVER PART OF A KERB gets both. OSM records a restriction that changes part way
    along a street by splitting the way, which is how "no parking for the first 100 ft from the
    junction" is expressed - so a kerb can be restricted near the corner and open beyond it. Such
    a kerb is marked for parking here, and the restricted stretch is carved back out of it by
    src/geometry/daylighting.py, which treats a mapped prohibition as a no-parking zone exactly
    like a statutory one: the stretch gets hatched and no stall is marked inside it. Only a kerb
    restricted along its WHOLE length is hatched end to end.

    That distinction is the reason this reads state.parking_restrictions rather than one way's
    tags. It used to take the tags of the single way nearest the leg's midpoint, so at Broad &
    Greenwood a no_parking restriction covering East Broad's first 79.5 ft was dropped in favour
    of the unrestricted way beyond it, and the render marked stalls where the mapper had just
    said there is none.

    "Unless otherwise specified": a side the scenario has ALREADY treated is left alone, so
    this can be applied as a baseline and then overridden per side.

    Side mapping goes through parking_restriction_by_side per span, which flips OSM's left/right
    for ways that run against the leg - without which half these kerbs would have the restriction
    painted on the wrong side.
    """
    new_state = state
    for leg_name in sorted(state.legs):
        if legs is not None and leg_name not in legs:
            continue
        leg = state.legs[leg_name]
        leg_length_ft = leg.centerline.length
        sides = {side: _restriction_summary(state, leg_name, side, leg_length_ft)
                 for side in ("left", "right")}

        def already_treated(side):
            return ((leg_name, side) in new_state.parking_zones
                    or (leg_name in new_state.lane_narrowing
                        and side in new_state.lane_narrowing_sides.get(leg_name, ("left", "right"))))

        untouched = [s for s in ("left", "right") if not already_treated(s)]

        # The allowance is PER SIDE, and it is half the road minus one target lane - not the
        # spare width divided by however many sides happen to be untreated. Those coincide
        # at two sides and diverge at one: handing a lone side the whole spare would shrink
        # its own lane to 2*target - half, well under the target it is supposed to protect.
        half_ft = leg.curb_to_curb_ft / 2
        allowance_ft = half_ft - TARGET_LANE_WIDTH_FT
        if allowance_ft <= 0 or not untouched:
            if untouched:
                print(f"  NOTE: {leg_name} is {leg.curb_to_curb_ft:.1f} ft curb to curb - too narrow "
                      f"for two {TARGET_LANE_WIDTH_FT:.0f} ft lanes, so no kerbside paint is marked "
                      f"here. Its lanes are {half_ft:.1f} ft as they stand.")
            continue

        # Hatched end to end only where the restriction covers the whole kerb. A kerb restricted
        # over PART of its length is marked for parking, and daylighting carves the restricted
        # stretch back out - see the docstring.
        restricted = [s for s in untouched
                      if sides[s].restricted_throughout
                      or (sides[s].restricted_in_part and sides[s].holds_no_stall)]
        # A standard stall or nothing: an unrestricted kerb with less than one stall's worth
        # of room gets its spare width HATCHED, not left bare. Leaving it bare was keeping
        # the lane at 18 ft on E Broad, which defeats the whole point of the target - and
        # hatching beside a travel lane reads as a buffer/shoulder, the same thing the strip
        # between a parking lane and the kerb already is, not as a parking prohibition.
        parkable = [s for s in untouched
                    if s not in restricted and allowance_ft >= MIN_MARKED_PARKING_DEPTH_FT]
        hatched = [s for s in untouched if s not in restricted and s not in parkable]
        for side in hatched:
            print(f"  NOTE: {leg_name} {side} is unrestricted, but only {allowance_ft:.1f} ft is "
                  f"spare beside a {TARGET_LANE_WIDTH_FT:.0f} ft lane - under one "
                  f"{MIN_MARKED_PARKING_DEPTH_FT:.0f} ft stall, so it is hatched as buffer "
                  f"rather than marked for parking.")
        restricted = restricted + hatched

        if restricted:
            new_state = new_state.apply(LaneNarrowing(LegTarget(leg_name),
                                                       stripe_width_ft=allowance_ft,
                                                       sides=tuple(restricted)))
            for side in restricted:
                why = (sides[side].describe() if sides[side].prohibited_ft
                       else "too narrow for a stall")
                new_state.notes.append(f"apply_osm_parking({leg_name}, {side}): {allowance_ft:.1f} ft "
                                        f"hatched - {why}")
        for side in parkable:
            # The stall is a fixed standard depth and the leftover between it and the kerb is
            # hatched (add_marked_parking's curb_offset_ft draws that buffer with the same
            # geometry a lane-narrowing buffer uses). Handing the whole allowance to depth_ft
            # instead produced 10-12 ft "parking spaces", which is a stall plus a strip of
            # unmarked asphalt drawn as though you could park on it.
            buffer_ft = allowance_ft - PARKING_STALL_DEPTH_DEFAULT_FT
            new_state = new_state.apply(
                MarkedParking(LegSide(leg_name, side),
                               depth_ft=PARKING_STALL_DEPTH_DEFAULT_FT,
                               curb_offset_ft=buffer_ft))
            extra = (f" + {buffer_ft:.1f} ft hatched to the kerb" if buffer_ft > 0.05 else "")
            new_state.notes.append(f"apply_osm_parking({leg_name}, {side}): "
                                    f"{PARKING_STALL_DEPTH_DEFAULT_FT:.0f} ft stalls{extra} - "
                                    f"{sides[side].describe()}")
    return new_state


@dataclass(frozen=True)
class RestrictionSummary:
    """What OSM says about one kerb, reduced to what a treatment and a label need.

    A kerb can now be restricted over part of its length, so "is this side restricted" is no
    longer a yes/no. Three cases have to be told apart, because they lead to three different
    markings and three different sentences on the drawing:

      restricted THROUGHOUT   hatch it end to end; no stall anywhere
      restricted IN PART      mark parking, and let daylighting carve the restricted stretch out
      not restricted          mark parking (whether tagged "none" or not tagged at all)
    """
    prohibited_ft: float           # how much of the kerb OSM forbids parking on
    kerb_length_ft: float
    worst_value: str | None        # the prohibition itself, e.g. "no_parking"
    stated_ft: float               # how much of the kerb OSM says ANYTHING about
    spans: tuple = ()              # the ParkingRestrictions behind it, in station order

    @property
    def open_ft(self) -> float:
        """Kerb OSM does not forbid parking on. Untagged counts as open, which is the same
        ordinary-street default apply_osm_parking has always applied to an untagged side."""
        return max(self.kerb_length_ft - self.prohibited_ft, 0.0)

    @property
    def restricted_throughout(self) -> bool:
        return self.prohibited_ft >= self.kerb_length_ft - RESTRICTION_COVERAGE_SLACK_FT

    @property
    def restricted_in_part(self) -> bool:
        return not self.restricted_throughout and self.prohibited_ft > RESTRICTION_COVERAGE_SLACK_FT

    @property
    def holds_no_stall(self) -> bool:
        """Whether what OSM leaves open is too short to park one car in.

        The same "a standard stall or nothing" rule MIN_MARKED_PARKING_DEPTH_FT applies across the
        road, applied along it. e_broad_st_west is tagged no_stopping over its first 114.5 ft and
        open for the last 15.5, and 15.5 ft is not a parking space - so marking that kerb for
        parking would claim a stall that cannot exist, and it is hatched end to end instead.
        """
        return self.open_ft < PARKING_STALL_LENGTH_DEFAULT_FT

    def describe(self) -> str:
        """One clause naming what OSM says, for a note or a plan-view label."""
        if self.restricted_throughout:
            return f"OSM says {self.worst_value!r} for the whole kerb"
        if self.restricted_in_part and self.holds_no_stall:
            return (f"OSM says {self.worst_value!r} for all but {self.open_ft:.0f} ft, under one "
                    f"{PARKING_STALL_LENGTH_DEFAULT_FT:.0f} ft stall")
        if self.restricted_in_part:
            stretches = ", ".join(f"{r.start_ft:.0f}-{r.end_ft:.0f} ft" for r in self.spans
                                  if r.prohibits)
            return f"OSM says {self.worst_value!r} over {stretches}"
        if self.stated_ft <= RESTRICTION_COVERAGE_SLACK_FT:
            return "no restriction tagged"
        return "restriction=none"


# How much of a kerb may be untagged before "restricted throughout" stops being true. A way's
# ends land a foot or two off the leg's own, and OSM splits are not surveyed to the inch.
RESTRICTION_COVERAGE_SLACK_FT = 2.0


def _restriction_summary(state: DesignState, leg_name: str, side: str,
                          kerb_length_ft: float) -> RestrictionSummary:
    """Reduce this kerb's ParkingRestriction spans to a RestrictionSummary.

    Spans are clipped to the leg and merged, so a way that runs 900 ft down the block counts
    only for the part of it that is on this leg, and two ways meeting at a split do not
    double-count the foot they share.
    """
    spans = tuple(state.parking_restrictions.get((leg_name, side), []))
    prohibited, stated = [], []
    worst = None
    for restriction in spans:
        lo = max(restriction.start_ft, 0.0)
        hi = min(restriction.end_ft, kerb_length_ft)
        if hi <= lo:
            continue
        if restriction.value is not None:
            stated.append((lo, hi))
        if restriction.prohibits:
            prohibited.append((lo, hi))
            worst = worst or restriction.value
    return RestrictionSummary(prohibited_ft=_merged_length_ft(prohibited),
                               kerb_length_ft=kerb_length_ft, worst_value=worst,
                               stated_ft=_merged_length_ft(stated), spans=spans)


def _merged_length_ft(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by possibly-overlapping (start, end) intervals."""
    total, reach = 0.0, None
    for lo, hi in sorted(intervals):
        if reach is None or lo > reach:
            total += hi - lo
            reach = hi
        elif hi > reach:
            total += hi - reach
            reach = hi
    return total


def complete_centerlines(state: DesignState, style: str = "double_yellow") -> DesignState:
    """Give every leg that has no centerline paint one.

    An unmarked centerline is a real gap in the street's markings, not a design preference -
    Greenwood Ave south of Broad has none today, so nothing tells a driver where their half
    of the road ends. Adding it is part of completing the markings, and it is exactly the
    kind of thing a proposal should carry.

    Only legs recorded as having NOTHING are changed. A leg already marked - dashed or
    double - is left alone: whether to upgrade a dashed line to a no-passing double is a
    traffic-engineering judgement about sight lines, not a gap to be filled in.
    """
    new_state = state
    for leg_name, current in sorted(state.centerline_styles.items()):
        if current != "none":
            continue
        # Through the treatment rather than writing the dict, so the design records the change
        # as a treatment like any other - a policy that edits state directly leaves
        # state.treatments an incomplete account of what was applied. The explanation is the
        # POLICY's, not the treatment's, so it is a note of its own.
        new_state = new_state.apply(SetCenterlineStyle(LegTarget(leg_name), style))
        new_state.notes.append(
            f"complete_centerlines({leg_name}): {style} added - the leg has no centerline "
            f"paint today, so nothing marks the middle of the road")
    return new_state


def all_crosswalks_continental(state: DesignState) -> DesignState:
    """Repaint every marked crossing at this junction to continental.

    FHWA and NACTO both rank crosswalk visibility roughly lines < continental < ladder, and
    continental is the low-cost repaint that every proposal here starts from - so it applies
    to all legs rather than being chosen one at a time. Existing conditions keep whatever
    OSM's crossing:markings records; this only changes the proposal.
    """
    new_state = state
    for leg_name in sorted(state.legs):
        new_state = new_state.apply(UpgradeCrosswalkMarkings(LegTarget(leg_name), "continental"))
    return new_state
