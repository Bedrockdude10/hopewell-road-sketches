"""Treatments at the CROSSING: refuge islands, raised crossings, the crosswalk's markings,
where it sits, and the centreline that has to stop short of it."""
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.geometry.targets import Everywhere, LegTarget
from src.geometry.model import (leg_clearance_ft)
from src.geometry.treatments.base import (DEFAULT_CENTERLINE_STYLE,
                                          NACTO_MIN_REFUGE_ISLAND_WIDTH_FT,
                                          VALID_CENTERLINE_STYLES, VALID_CROSSWALK_STYLES,
                                          Treatment, _band_across_the_road)
from src.geometry.treatments.state import DesignState



@dataclass(frozen=True)
class RefugeIsland(Treatment):
    """A raised pedestrian refuge island splitting a leg's roadway, centered `offset_ft` from the
    intersection along the centerline.

    width_ft is the island's extent ACROSS the road, in the direction pedestrians cross -
    NACTO's minimum is 6 ft, so a person or wheelchair can wait clear of both travel directions.
    along_road_ft is its length ALONG the road: how much of the crosswalk it shelters.
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
        # No along_road_ft check: _band_across_the_road already refuses a zero-length span, and
        # its message names the treatment and the leg, which this constructor cannot.

    def describe(self) -> str:
        return (f"RefugeIsland({self.target}, offset_ft={self.offset_ft}, "
                f"width_ft={self.width_ft})")

    @property
    def island_name(self) -> str:
        """What this island is called in the exported geometry. Defaults to its leg and station,
        so two islands on one leg are distinguishable without either being named by hand."""
        return self.name or f"{self.target.leg}_refuge_{int(self.offset_ft)}ft"

    def polygon(self, state: "DesignState") -> Polygon:
        """The ground this island occupies, measured against the design it is asked about.

        Resolved here rather than frozen in at apply time: a design is a set of decisions, not a
        sequence of snapshots, so the shape must not depend on WHEN in a scenario the treatment
        ran. Every marking in this project is resolved against the final street for that reason.
        """
        leg = state.legs[self.target.leg]
        return _band_across_the_road(leg.centerline, self.offset_ft - self.along_road_ft / 2,
                                      self.offset_ft + self.along_road_ft / 2,
                                      self.width_ft / 2,
                                      f"{self.width_ft:.0f} ft refuge island")

    def apply_to(self, state: "DesignState", model=None) -> None:
        # Built and discarded, for the refusal only: a leg too short to hold the band says so
        # here, where the scenario that asked for it is on the stack, rather than in a renderer.
        self.polygon(state)


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

    def polygon(self, state: "DesignState") -> Polygon:
        """The speed table's footprint, measured against the design it is asked about.

        Resolved here rather than frozen in at apply time, and here that is load-bearing: the
        start station comes from leg_clearance_ft, which reads the corner fillets, and
        AddCurbExtension re-cuts them. Frozen, a table applied before an extension on the same
        leg keeps the corner it was measured against while every other marking follows the kerb.
        """
        leg = state.legs[self.target.leg]
        if leg.left_curb is None or leg.right_curb is None:
            raise ValueError(f"Leg {self.target.leg!r} has no curb lines (width unknown) - "
                              f"can't place a crossing on it.")
        # Start beyond the corner fillets, not at the intersection point: a crossing placed at
        # the corner point lands inside the curb-return curve rather than on the straight
        # roadway where a real crosswalk sits.
        start = leg_clearance_ft(self.target.leg, state.legs, state.corner_fillets)
        return _band_across_the_road(
            leg.centerline, start, start + self.crossing_width_ft, leg.curb_to_curb_ft / 2,
            f"{self.crossing_width_ft:.0f} ft raised crossing on {self.target.leg!r}")

    def apply_to(self, state: "DesignState", model=None) -> None:
        # Built and discarded, for the refusals only - a leg with no traced kerbs, or one whose
        # corner return consumes its whole length. Both are things the scenario author needs told.
        self.polygon(state)


@dataclass(frozen=True)
class UpgradeCrosswalkMarkings(Treatment):
    """Repaint a leg's crosswalk to a more visible marking style. FHWA and NACTO both rank
    visibility roughly lines < continental < ladder; "lines" is what most of this intersection
    has today, and repainting is a low-cost treatment independent of any geometry change."""
    style: str = "continental"

    def __post_init__(self):
        if self.style not in VALID_CROSSWALK_STYLES:
            raise ValueError(f"Unknown crosswalk style {self.style!r} - expected one of "
                              f"{VALID_CROSSWALK_STYLES}")

    def describe(self) -> str:
        return f"UpgradeCrosswalkMarkings({self.target}, style={self.style!r})"


@dataclass(frozen=True)
class SetCenterlineStyle(Treatment):
    """Change what is painted down the middle of a leg: 'single_yellow_dashed' (ordinary two-way
    marking), 'double_yellow' (solid no-passing zone), or 'none'. Unlike UpgradeCrosswalkMarkings
    this is not a visibility ranking, so any value is a valid target rather than only an
    "upgrade"."""
    style: str = DEFAULT_CENTERLINE_STYLE

    def __post_init__(self):
        if self.style not in VALID_CENTERLINE_STYLES:
            raise ValueError(f"Unknown centerline style {self.style!r} - expected one of "
                              f"{VALID_CENTERLINE_STYLES}")

    def describe(self) -> str:
        return f"SetCenterlineStyle({self.target}, style={self.style!r})"


def resolved_crossing_stations(model, state: DesignState) -> dict:
    """{leg name: the station its crossing is resolved to}, for treatments measured off it.

    RESOLVED DATA, not a parameter: a real OSM-surveyed position where one was matched, else the
    geometric estimate. One home for it keeps a bulb-out's length tied to the same crossing the
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


@dataclass(frozen=True)
class ShiftCrosswalk(Treatment):
    """Shift a leg's crosswalk further from (positive) or closer to (negative) the intersection,
    on top of whatever src/render/crosswalks.py:resolve_crosswalk_offsets resolves - e.g. to give
    a turning fire apparatus room before it meets the crosswalk mid-turn.

    ACCUMULATES rather than replacing, so two shifts of one leg add up. Worth stating: it is the
    one treatment here that is not idempotent.
    """
    delta_ft: float = 0.0

    def describe(self) -> str:
        return f"ShiftCrosswalk({self.target}, delta_ft={self.delta_ft})"


def complete_centerlines(state: DesignState, style: str = "double_yellow") -> DesignState:
    """Give every leg that has no centerline paint one.

    An unmarked centerline is a gap in the street's markings, not a design preference - Greenwood
    Ave south of Broad has none today, so nothing tells a driver where their half of the road
    ends.

    ONLY legs recorded as having NOTHING. A leg already marked, dashed or double, is left alone:
    upgrading a dashed line to a no-passing double is a traffic-engineering judgement about sight
    lines, not a gap to be filled in.
    """
    new_state = state
    for leg_name in sorted(state.legs):
        if state.centerline_style(leg_name) != "none":
            continue
        # Through the treatment rather than writing state directly, or state.treatments is an
        # incomplete account of what was applied. The explanation belongs to the POLICY, not to
        # the treatment, so it is a note of its own.
        new_state = new_state.apply(SetCenterlineStyle(LegTarget(leg_name), style))
        new_state.notes.append(
            f"complete_centerlines({leg_name}): {style} added - the leg has no centerline "
            f"paint today, so nothing marks the middle of the road")
    return new_state


def all_crosswalks_continental(state: DesignState) -> DesignState:
    """Repaint every marked crossing IN THE FRAME to continental.

    The low-cost repaint every proposal here starts from, so it applies to all of them rather
    than being chosen one at a time. Existing conditions keep whatever OSM's crossing:markings
    records; this only changes the proposal.

    EVERY CROSSING, NOT EVERY LEG - one treatment against Everywhere(), not one per leg. A frame
    drawn at 2.5x holds crossings belonging to streets with no leg at this junction, and looping
    over `state.legs` left those drawn from their own OSM tag whatever the proposal said. See
    src/geometry/targets.py:Everywhere.

    A PER-LEG UpgradeCrosswalkMarkings STILL WINS, which makes this a default rather than an
    override - resolve_crosswalk_style checks the leg first.
    """
    return state.apply(UpgradeCrosswalkMarkings(Everywhere(), "continental"))
