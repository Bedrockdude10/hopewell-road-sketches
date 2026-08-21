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
    from src.geometry.intersection.junction import IntersectionModel



def _band_across_the_road(centerline, from_ft: float, to_ft: float, half_width_ft: float,
                           what: str) -> Polygon:
    """The rectangle spanning `half_width_ft` either side of a leg, between two stations.

    Shared by RefugeIsland and RaiseCrossing. Both stations can clamp to the same point on a leg
    whose corner return consumes its whole length, so the zero-length case is refused with a
    message naming the leg rather than raising ZeroDivisionError.
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

# The travel lane width every road diet here aims at. Single home: a standard, not a per-site
# choice.
#
# ELEVEN FEET IS TWO NUMBERS: the 10 ft NACTO/AASHTO urban minimum PLUS the 1 ft NJDOT asks for
# where trucks exceed 15% of the mix, which they do here (Broad St is CR 518; E Broad and NJ 31
# carry hgv=designated). The truck allowance is already INSIDE the 11 ft, so neither narrowing
# to 10 ft nor widening to 12 ft is available - both spend an allowance already made.
TARGET_LANE_WIDTH_FT = 11.0
# A parking lane is a STANDARD width, not "whatever is left over": surplus beyond this is
# hatched, and a kerb with less than this spare is marked for no parking at all.
MIN_MARKED_PARKING_DEPTH_FT = 8.0
CORNER_HATCHING_DEFAULT_DEPTH_FT = 6.0  # paint-only zone depth, comparable footprint to a modest real curb extension
CORNER_APRON_DEFAULT_EXTENT_FT = 5.0  # mountable-apron zone depth - same shape as hatching, different surface finish


def kerbside_allowance_ft(leg, side: str) -> float:
    """How much room this kerb has beside a target-width travel lane. ONE definition.

    THE DATUM IS THE TRACED KERB, not the nominal `leg.curb_to_curb_ft / 2`. The two are
    different measurements and they disagree - 15.0 ft vs 5.0 ft at one place on broad_st_east -
    so every "is there room here?" question goes through this one function: apply_osm_parking,
    the plan view's kerb labels, TravelLanesKeepTheirWidth. The nominal width keeps its own job:
    reporting the approach, and standing in for legs with no tracing (narrowest_half_width_ft
    falls back to it).

    Narrowest rather than typical: a treatment on a kerb is a promise about the whole of it, and
    one sized off the average is broken wherever the street pinches.
    """
    if leg.curb_to_curb_ft is None:
        return 0.0
    return narrowest_half_width_ft(leg, side) - TARGET_LANE_WIDTH_FT

# What is painted down the middle of a leg TODAY: a dashed yellow line (the ordinary two-way
# marking), a solid double yellow (no-passing), or none at all. Read from a site's config.yaml
# per leg (see sites/README.md), street-view confirmed like the `signals` block.
DEFAULT_CENTERLINE_STYLE = "single_yellow_dashed"
VALID_CENTERLINE_STYLES = ("single_yellow_dashed", "double_yellow", "none")

# Float slack when comparing a requested width against the room a leg has. The widths
# themselves are specified to a tenth of a foot; this only absorbs the arithmetic.
LANE_WIDTH_SLACK_FT = 0.05


@dataclass(frozen=True)
class ParkingRestriction:
    """What OSM says about parking on ONE KERB over ONE STRETCH of it, in the leg's frame.

    A stretch and not a whole side: OSM records a restriction that changes part way along a
    street by splitting the way, which is how "no parking for the first 100 ft from the junction"
    is expressed. See src/geometry/intersection/junction.py:RoadSpan.

    `value` is the raw OSM value: no_parking / no_standing / no_stopping, or "none" for an
    explicit statement that parking IS allowed, or None where the way says nothing at all. The
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


def _parking_restrictions_from_model(model: "IntersectionModel") -> dict:
    """{(leg, side): [ParkingRestriction]} from every OSM way lying along each leg.

    Seeded onto the state like centerline_styles, so treatments, renderers and invariants read
    one resolved answer, and a scenario can be handed a state with no model behind it.
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

    `depth_ft` is a fixed reach inward from the corner arc: the standalone MountableApron, for a
    corner where a hard bulb-out is not an option.

    `swept_radius_ft` is the radius a large vehicle needs, and the apron is then the ANNULUS
    between that and `face_radius_ft` (src/geometry/model/corners.py:corner_apron_annulus). A
    curb extension lays this one, because its claim is that the swept path survives the tightened
    corner and a fixed depth is tied to no vehicle's radius.
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

    INVARIANT: a treatment is a frozen dataclass that validates itself in `__post_init__`, so an
    unvalidated one cannot exist. Validation as a per-function convention is what this replaced.

    Three things every treatment declares:

      * `target` - a src/geometry/targets.py:Target, checked to exist in the design by
        DesignState.apply before anything is written.
      * `describe()` - one line for state.notes, so provenance follows from applying a treatment
        rather than from each function remembering to append.
      * `apply_to(state, model)` - the change itself, on an already-cloned state.

    Subclasses needing the IntersectionModel (a kerb rebuild reads the traced kerbs) declare
    `needs_model = True`; asking for one that was not supplied is an error rather than a silently
    skipped treatment, which once produced a proposal with no treatments that rendered fine.
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
    #: See src/geometry/paint/context.py:curbside_paint_ft.
    paint_group: ClassVar[int] = 50
    paint_rank: ClassVar[int] = 0

    def paint(self, ctx) -> None:
        """Put this treatment's markings on the roadway, through ctx (paint.PaintContext).

        Nothing by default: several treatments change geometry rather than markings, and a
        crosswalk restyle is drawn by the crossing renderer rather than from the paint list.
        """

    def apply_to(self, state: "DesignState", model: "IntersectionModel" = None) -> str | None:
        """Check this treatment against the design, and change the MODELLED STREET if it moves it.

        Nothing by default: a treatment IS the record, so being applied is the whole of the
        change for most of them. A subclass overrides for one of two reasons, both about the
        design rather than about storage:

          * REFUSE, on a precondition that needs the design and not just the arguments -
            AddBikeLane's cross-section against the leg's narrowest traced width,
            AddBikeLaneBollards' requirement of a buffered lane, ProtectDaylightZone's
            requirement of an extension under a `curb_extension` device. A constructor cannot
            check any of those, which is why this receives the state.
          * MOVE THE KERB. AddCurbExtension and SetCornerRadius change `legs` and
            `corner_fillets` - the modelled street itself.

        May return a suffix for the note, where provenance includes something only measurable
        against the design (a bike lane records how much of the leg's width it used).
        """
        return None


VALID_CROSSWALK_STYLES = ("lines", "continental", "ladder")


PARKING_STALL_DEPTH_DEFAULT_FT = 8.0  # AASHTO/NACTO typical parallel-parking lane depth (curb to travel-lane edge)
PARKING_STALL_LENGTH_DEFAULT_FT = 22.0  # AASHTO/NACTO typical parallel-parking stall length
# NJSA 39:4-138: no stopping/standing/parking within 25 ft of a marked crosswalk at an
# intersection. A legal minimum, not a rendering choice - marked parking starts at whichever of
# this and leg_clearance_ft's past-the-corner-curve point is farther from the intersection.
LEGAL_PARKING_SETBACK_FT = 25.0


BOLLARD_DEFAULT_SPACING_FT = 10.0  # typical flex-post delineator spacing for a channelized buffer
