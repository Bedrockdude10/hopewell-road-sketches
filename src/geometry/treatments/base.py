"""What every treatment is made of: the ABC, the shared value objects, and the constants
more than one family of treatments reads.

Nothing here knows about a DesignState or about any particular treatment, which is what makes it
the bottom of this package - see __init__.py for the layering and why it is shaped this way."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from shapely.geometry import Polygon

from src.geometry.targets import Target
from src.geometry.model import (narrowest_half_width_ft)

if TYPE_CHECKING:                       # DesignState is layered above this module;
    from src.geometry.treatments.state import DesignState   # the annotation is a string



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

# The travel lane width every road diet here aims at. Defined once, in src, because it is a
# standard rather than a per-site choice - it was previously redeclared in each site's
# scenarios.py, which is how nothing ended up enforcing it.
#
# ELEVEN FEET IS TWO NUMBERS, AND THAT IS THE POINT (Danny, 2026-08-14). It is the 10 ft
# NACTO/AASHTO urban minimum PLUS the 1 ft NJDOT asks for where trucks exceed 15% of the traffic
# mix - and they do here: Broad St is CR 518 and E Broad and NJ 31 both carry hgv=designated,
# with NJ 31 on the state truck network. So the truck allowance is not an outstanding item to
# add on top of 11 ft; it is already inside it, and narrowing to 10 ft anywhere would be
# spending it.
#
# Worth stating because the arithmetic is invisible in the number. Reading "11 ft urban minimum"
# and then reading NJDOT's "+1 ft on truck routes" leads straight to proposing 12 ft lanes on
# this corridor, which is a wider road drawn in the name of a standard already satisfied.
TARGET_LANE_WIDTH_FT = 11.0
# A parking lane is a STANDARD width, not "whatever is left over". Anything wider than this
# isn't a wider parking space, it's a parking space plus unmarked asphalt - which is what
# gets hatched. Where the spare width can't even cover one standard stall, no parking is
# marked at all rather than a token strip painted.
MIN_MARKED_PARKING_DEPTH_FT = 8.0
CORNER_HATCHING_DEFAULT_DEPTH_FT = 6.0  # paint-only zone depth, comparable footprint to a modest real curb extension
CORNER_APRON_DEFAULT_EXTENT_FT = 5.0  # mountable-apron zone depth - same shape as hatching, different surface finish


def kerbside_allowance_ft(leg, side: str) -> float:
    """How much room this kerb has beside a target-width travel lane. ONE definition.

    There used to be two numbers for this, and they disagreed. The DECISION - is there room
    for a stall here? - halved `leg.curb_to_curb_ft`, a single figure field-measured at the
    intersection. The DRAWING measured the traced kerb, station by station, because that is
    where the paint actually has to fit. On broad_st_east the two read 15.0 ft and 5.0 ft at
    the same place: an 8 ft stall was committed to off the nominal figure and then clipped to
    4.6 ft against a kerb 16.0 ft out, which is what "the parking spaces look unusable" was.

    So this is the measured one, per side, and everything that asks the question asks it here
    - apply_osm_parking, the plan view's kerb labels, and TravelLanesKeepTheirWidth. The
    nominal width keeps its own job: reporting the approach, and standing in for legs with no
    tracing at all (narrowest_half_width_ft falls back to it).

    Narrowest rather than typical, for the reason AddBikeLane gives: a treatment applied to a
    kerb is a promise about the whole of it, and a promise sized off the average is broken
    wherever the street pinches.
    """
    if leg.curb_to_curb_ft is None:
        return 0.0
    return narrowest_half_width_ft(leg, side) - TARGET_LANE_WIDTH_FT

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
class CornerApron:
    """A flush, drivable corner surface. Two shapes, because there are two reasons for one.

    `depth_ft` is a fixed reach inward from the corner arc: the standalone treatment
    (add_mountable_apron), for a corner where a hard bulb-out is not an option.

    `swept_radius_ft` is the radius a large vehicle needs, and the apron is then the ANNULUS
    between that and `face_radius_ft` (src/geometry/model/corners.py:corner_apron_annulus). This is the
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

    def apply_to(self, state: "DesignState", model=None) -> str | None:
        """Check this treatment against the design, and change the MODELLED STREET if it moves it.

        Nothing by default, and that default is now the common case. Every treatment used to
        write a dict here and every renderer read those dicts back; now a treatment IS the
        record, so being applied is the whole of the change for most of them - a marked parking
        lane's depth is a fact about the MarkedParking in state.treatments, not about an entry
        under a key that anything could have written.

        What is left for a subclass to do here is one of two things, and both are about the
        design rather than about storage:

          * REFUSE. A precondition that needs the design rather than just the arguments -
            AddBikeLane's cross-section against the leg's narrowest traced width,
            AddBikeLaneBollards' requirement of a buffered lane, ProtectDaylightZone's
            requirement that a `curb_extension` device have an extension under it. A
            constructor cannot check any of those, which is why this receives the state.
          * MOVE THE KERB. AddCurbExtension and SetCornerRadius change `legs` and
            `corner_fillets` - the modelled street itself - and that is not a parameter a
            renderer could read off the treatment instead.

        May return a suffix for the note, for a treatment whose provenance includes something
        only measurable against the design: a bike lane records how much of the width the leg
        actually has it used.
        """
        return None


VALID_CROSSWALK_STYLES = ("lines", "continental", "ladder")


PARKING_STALL_DEPTH_DEFAULT_FT = 8.0  # AASHTO/NACTO typical parallel-parking lane depth (curb to travel-lane edge)
PARKING_STALL_LENGTH_DEFAULT_FT = 22.0  # AASHTO/NACTO typical parallel-parking stall length
LEGAL_PARKING_SETBACK_FT = 25.0  # NJSA 39:4-138: no stopping/standing/parking within 25 ft of a marked crosswalk at


BOLLARD_DEFAULT_SPACING_FT = 10.0  # typical flex-post delineator spacing for a channelized buffer
