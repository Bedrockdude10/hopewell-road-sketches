"""Treatments at the CROSSING: refuge islands, raised crossings, the crosswalk's markings,
where it sits, and the centreline that has to stop short of it."""
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.geometry.targets import LegTarget
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

    @property
    def island_name(self) -> str:
        """What this island is called in the exported geometry. Defaults to its leg and station,
        so two islands on one leg are distinguishable without either being named by hand."""
        return self.name or f"{self.target.leg}_refuge_{int(self.offset_ft)}ft"

    def polygon(self, state: "DesignState") -> Polygon:
        """The ground this island occupies, measured against the design it is asked about.

        Resolved here rather than frozen into state.refuge_islands when the treatment was
        applied. Both raise_crossing and this one used to build their polygon in apply_to, which
        made the shape depend on WHEN in a scenario the treatment ran - a design is a set of
        decisions, not a sequence of snapshots, and every marking in this project is already
        resolved against the final street for exactly that reason.
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

        Resolved here rather than frozen into state.raised_crossings at apply time, and for this
        treatment that is not merely tidier: the start station comes from leg_clearance_ft, which
        reads the corner fillets, and AddCurbExtension re-cuts them. Applied before an extension
        on the same leg, this used to keep the corner it happened to be measured against while
        every other marking followed the kerb that moved.
        """
        leg = state.legs[self.target.leg]
        if leg.left_curb is None or leg.right_curb is None:
            raise ValueError(f"Leg {self.target.leg!r} has no curb lines (width unknown) - "
                              f"can't place a crossing on it.")
        # Start beyond the curve of this leg's corner fillets, not at the
        # intersection point itself - a crossing placed right at the corner point
        # lands inside the curb-return curve rather than on the straight section
        # of roadway where a real crosswalk would sit.
        start = leg_clearance_ft(self.target.leg, state.legs, state.corner_fillets)
        return _band_across_the_road(
            leg.centerline, start, start + self.crossing_width_ft, leg.curb_to_curb_ft / 2,
            f"{self.crossing_width_ft:.0f} ft raised crossing on {self.target.leg!r}")

    def apply_to(self, state: "DesignState", model=None) -> None:
        # Built and discarded, for the refusals only - a leg with no traced kerbs, or one whose
        # corner return consumes its whole length (W Broad & Louellen's southwest leg: 133 ft of
        # clearance on 130 ft of leg). Both are things the scenario author needs told.
        self.polygon(state)


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
    for leg_name in sorted(state.legs):
        if state.centerline_style(leg_name) != "none":
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
