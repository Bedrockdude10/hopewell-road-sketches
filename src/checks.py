"""Scene invariants: the things that must be true of every render, checked every time.

The failure mode guarded against is not a crash but a finished-looking render that asserts
something false about the street. Each invariant is checked on BOTH paths - the 2D plan view
and the 3D export - because the premise of the 2D reconstruction is that it shows what the 3D
render will show, and a check that only guards the export lets the two drift.

Two design choices worth stating:

  * ALL violations are collected before anything is raised, so a bad junction costs one
    edit-run cycle rather than one per violation.
  * A violation carries its coordinates. The plan view draws them, so the error message and
    the picture agree about where to look.

`check_scene` reports; `assert_scene_valid` raises. Phase scripts save the plot first and
assert after, so a failure always comes with a picture of itself.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point

from src.geometry.markings import (PARKING_EDGE_LINE, STALL_DIVIDER,
                                   lies_legitimately_on)
from src.geometry.targets import Side
from src.geometry.model import curb_offsets_at_stations, station_offset_many

if TYPE_CHECKING:                      # the runtime import is in _empty_state, below
    from src.geometry.treatments import DesignState

# ---------------------------------------------------------------------------
# Tolerances. Each is a real physical claim, not a fudge factor.
# ---------------------------------------------------------------------------

# A pad polygon may graze the kerb line by a hair from polygon tolerance; beyond this it is
# genuinely sitting in the carriageway.
MAX_PAD_ROADWAY_OVERLAP = 0.02
# A detectable warning surface marks a kerb ramp, so it belongs against a curb. Well beyond
# this and it is floating somewhere that isn't a ramp.
PAD_MAX_DISTANCE_FROM_CURB_FT = 12.0
# A curb line may start a shade behind the junction node (the cross-section is not exactly
# at the node), but not run back up the opposite leg.
CURB_BEHIND_JUNCTION_TOLERANCE_FT = 6.0
# A painted crosswalk lies in the roadway it crosses, full stop. Tight because the reach is
# bounded by the roadway itself (crosswalks.crosswalk_reach_to_curbs_ft); a loose bound here
# hides the failure it is named for - end bars painted up the corner onto the sidewalk.
MIN_CROSSWALK_IN_PAVEMENT = 0.99
# A stop bar covers the entering half only, and it RESTS AGAINST the centreline rather than
# crossing it - so this is float noise and the width of a polygon vertex, not a design margin.
# The bar is built to start exactly at the line (crosswalks.stop_bar_band_geometry_ft), so
# anything past it is a fault rather than a tolerance being used up.
STOP_BAR_PAST_CENTERLINE_TOLERANCE_FT = 0.1
# Paint is specified to a tenth of a foot; this absorbs float noise, nothing more.
LANE_WIDTH_TOLERANCE_FT = 0.05
# A marking may touch the kerb - that is what a curbside marking does - and may sit a hair
# past it where the strip is sampled across a curve between two traced vertices. Beyond this
# it is painted on the footway.
PAINT_PAST_CURB_TOLERANCE_FT = 0.25
# Statutory setbacks are whole feet; this absorbs the float noise of resolving a station,
# nothing more. It is deliberately far below a foot - the distances themselves are the law.
PARKING_SETBACK_TOLERANCE_FT = 0.1
# Two markings may share an edge, and clipping one against another leaves slivers along that
# shared edge. Beyond this they genuinely cover the same ground.
MARKING_OVERLAP_TOLERANCE_SQ_FT = 1.0
# How close two lines have to be to count as the same line. Well under a paint stripe's own
# width, so it only catches lines genuinely drawn on top of each other.
COLLINEAR_PAINT_TOLERANCE_FT = 0.1
# And how far they must run together before it is a collision rather than a shared endpoint
# or a crossing - a stall divider meets the lane edge at right angles by design.
MIN_COLLINEAR_OVERLAP_FT = 1.0

# A post is one object, and the paint dot and the prop are that one object placed once - the
# prop is read off the paint. This absorbs float noise, nothing more.
POST_PROP_TOLERANCE_FT = 0.1

# Props that belong on the footway. Anything not listed is assumed to belong there too -
# a new prop type is checked by default, and the exceptions have to be declared. Bollards
# and delineators are the deliberate exception: they are placed IN the carriageway.
ROADWAY_PROP_TYPES = frozenset({"bollard", "delineator", "flexible_delineator"})


class SceneInvariantError(AssertionError):
    """One or more scene invariants failed. Message lists every violation found."""


# Kept as aliases: these checks started life as separate one-off assertions.
PedestrianFurnitureInRoadwayError = SceneInvariantError
TactilePadInRoadwayError = SceneInvariantError


@dataclass(frozen=True)
class Violation:
    check: str                        # short machine-ish name, e.g. "furniture_in_roadway"
    detail: str                       # one line, readable, says what and why it's wrong
    where: tuple[float, float] | None = None   # state-plane feet, for the plot marker
    # False for a disagreement between two SOURCES rather than a bug in our placement - an
    # OSM node surveyed at a position that falls inside our modelled roadway. One of the two
    # is wrong and it's worth saying so every run, but no amount of editing this repo fixes
    # it, so it must not block the site from ever rendering again.
    fatal: bool = True

    def __str__(self) -> str:
        at = f" at ({self.where[0]:.1f}, {self.where[1]:.1f})" if self.where else ""
        return f"[{self.check}]{at} {self.detail}"


def _empty_state():
    """A DesignState with no legs and no treatments - the default scene.

    Imported here rather than at module scope only to keep this module importable on its own;
    src.geometry.treatments does not import this one, so there is no cycle either way.
    """
    from src.geometry.treatments import DesignState

    return DesignState(legs={}, corner_fillets={})


@dataclass(frozen=True)
class SceneContext:
    """Everything an invariant may ask about one scene, resolved once and handed to all of them.

    One object rather than a per-check argument list: two checks handed hand-picked subsets of
    the same scene can silently validate geometry built two different ways, one of it geometry
    no renderer drew. A check cannot now be handed a different scene from its neighbour.

    Everything defaults, and `state` defaults to a real empty DesignState rather than None, so a
    test can describe only the parts of a scene its check reads and every other check still runs
    over it and finds nothing. A scene with nothing in it is vacuously valid - pinned by
    test_a_check_reading_a_field_the_caller_left_out_gets_a_default.
    """
    model: object = None
    state: "DesignState" = field(default_factory=lambda: _empty_state())
    pavement: object = None
    props: tuple = ()
    paint: tuple = ()
    crosswalk_bands: dict = field(default_factory=dict)
    # Which of those bands are actually PAINTED. Every leg gets a resolved band, including legs
    # with no marking today, and the difference decides where the intersection ends for marking
    # purposes - see NoPaintInsideTheJunction. Carried on the context rather than re-derived,
    # because it is the same set curbside_paint_ft cut the paint against.
    marked_crosswalks: frozenset = frozenset()
    crosswalk_offsets: dict = field(default_factory=dict)
    stop_bars: dict = field(default_factory=dict)
    # The MARKED crossings at junctions inside the frame that this site does not model - the ones
    # with no leg here and therefore no entry in crosswalk_bands. Separate rather than merged into
    # that dict on purpose: crosswalk_bands is keyed by leg and several checks iterate it as "this
    # junction's four crossings", so a Blackwell Avenue entry keyed by nothing would quietly change
    # what those checks mean. See CrossingsAreNotPaintedOver.
    unmodelled_crossing_bands: tuple = ()

    @property
    def legs(self) -> dict:
        return self.state.legs or {}

    @property
    def corner_fillets(self) -> dict:
        return self.state.corner_fillets or {}


# Every invariant, in declaration order. Populated by SceneCheck.__init_subclass__ - defining a
# check is what registers it, so the list cannot fall behind the file.
CHECKS: list["SceneCheck"] = []


class SceneCheck:
    """One invariant. Subclassing runs it.

    The registry is not written by hand: a check defined but never added to a hand-maintained
    chain is dead code that looks live. Now the chain IS the file.

    A check reads what it needs off the SceneContext, returns every violation it finds, and never
    raises. See assert_scene_valid for the raising wrapper.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CHECKS.append(cls())

    def run(self, scene: SceneContext) -> list[Violation]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# The invariants. Each returns a list of Violations and never raises.
# ---------------------------------------------------------------------------

class FurnitureOffRoadway(SceneCheck):
    """Nothing that belongs on the footway may sit in the carriageway.

    Signs, signal poles, pushbuttons, beacons, streetlights, hydrants and tactile pads are
    all footway furniture. A pad drawn in the road is the worst case - the render asserting
    something false about an accessibility feature - but a stop sign in the middle of the
    street is just as wrong.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        props, pavement = scene.props, scene.pavement
        from src.render.props import pad_polygon  # local: props imports geometry, avoid a cycle

        if pavement is None or pavement.is_empty:
            return []
        violations = []
        for prop in props:
            kind = prop.get("type")
            if kind in ROADWAY_PROP_TYPES:
                continue
            position = prop.get("position_ft")
            if position is None:
                continue
            if kind == "tactile_paving_pad":
                pad = pad_polygon(*position, prop.get("heading_deg", 0.0))
                if pad.is_empty or pad.area <= 0:
                    continue
                overlap = pad.intersection(pavement).area / pad.area
                if overlap > MAX_PAD_ROADWAY_OVERLAP:
                    violations.append(Violation(
                        "furniture_in_roadway",
                        f"tactile paving pad has {overlap * 100:.0f}% of its area in the roadway - a "
                        f"detectable warning surface is on the footway at a kerb ramp, by definition",
                        position))
            elif pavement.contains(Point(*position)):
                if prop.get("surveyed_position"):
                    violations.append(Violation(
                        "surveyed_furniture_in_roadway",
                        f"{kind} is drawn at its surveyed OSM position and that position falls inside our "
                        f"modelled roadway - so either the OSM node is misplaced or this junction's modelled "
                        f"pavement is too wide. Not something placement code can fix; check the two sources",
                        position, fatal=False))
                else:
                    violations.append(Violation(
                        "furniture_in_roadway",
                        f"{kind} stands in the roadway - it belongs on the footway. Either its placement is "
                        f"wrong (src/render/props.py) or this junction's modelled pavement is too wide",
                        position))
        return violations


class PadsAgainstACurb(SceneCheck):
    """A tactile pad marks a kerb ramp, so it has to be at a kerb.

    Off the roadway is necessary but not sufficient: a pad nudged clear of a too-wide
    pavement can end up out in a front garden, which reads as fine in plan and absurd in 3D.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        props, legs, corner_fillets = scene.props, scene.legs, scene.corner_fillets
        curbs = _all_curb_lines(legs, corner_fillets)
        if not curbs:
            return []
        violations = []
        for prop in props:
            if prop.get("type") != "tactile_paving_pad":
                continue
            point = Point(*prop["position_ft"])
            distance = min(curb.distance(point) for curb in curbs)
            if distance > PAD_MAX_DISTANCE_FROM_CURB_FT:
                violations.append(Violation(
                    "pad_off_the_kerb",
                    f"tactile paving pad sits {distance:.1f} ft from the nearest curb line (limit "
                    f"{PAD_MAX_DISTANCE_FROM_CURB_FT:.0f} ft) - it marks a ramp, so it belongs against one",
                    prop["position_ft"]))
        return violations


class CurbsClearOfJunction(SceneCheck):
    """No leg's curb may run back through the intersection.

    A leg's curb line starts at that leg's cross-section and goes outward. One that runs
    backwards past the junction marks a kerb where there is open roadway, and crosses the
    opposite leg's curb, which is what makes the pavement ring self-intersect. Measured in the
    leg's own frame, so it is the same signed station the curb was built from.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        legs = scene.legs
        violations = []
        for name, leg in legs.items():
            for side in ("left", "right"):
                curb = getattr(leg, f"{side}_curb")
                if curb is None:
                    continue
                stations, _offsets = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
                worst = float(stations.min())
                if worst < -CURB_BEHIND_JUNCTION_TOLERANCE_FT:
                    index = int(np.argmin(stations))
                    violations.append(Violation(
                        "curb_through_junction",
                        f"{name}'s {side} curb runs {abs(worst):.1f} ft back past the junction, drawing curb "
                        f"across the middle of the intersection (tolerance "
                        f"{CURB_BEHIND_JUNCTION_TOLERANCE_FT:.0f} ft)",
                        tuple(curb.coords[index])))
        return violations


class CurbsDoNotCross(SceneCheck):
    """A leg's two curb lines are the two sides of one street: they never meet."""

    def run(self, scene: SceneContext) -> list[Violation]:
        legs = scene.legs
        violations = []
        for name, leg in legs.items():
            left, right = leg.left_curb, leg.right_curb
            if left is None or right is None or not left.intersects(right):
                continue
            crossing = left.intersection(right)
            point = crossing.centroid if not crossing.is_empty else None
            violations.append(Violation(
                "curbs_cross",
                f"{name}'s left and right curb lines cross - the modelled roadway closes to zero width "
                f"and reopens inverted. Usually an extrapolated curb taking its bearing from a corner "
                f"return rather than from the street",
                (point.x, point.y) if point is not None else None))
        return violations


class TravelLanesKeepTheirWidth(SceneCheck):
    """Kerbside paint must never squeeze a travel lane below the target width.

    Only fires where THIS design painted something. A leg that is naturally narrower than
    the target is a fact about the street, not an error we introduced - Louellen Street is
    19.3 ft curb to curb and no amount of checking widens it. What must not happen is a
    treatment taking a road that could hold two target-width lanes and marking parking or
    hatching that leaves less, which is what fixed paint widths applied without reference to
    the road left over produce.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        state = scene.state
        from src.geometry.targets import BOTH_SIDES, LegSide, LegTarget
        from src.geometry.treatments import (TARGET_LANE_WIDTH_FT, LaneNarrowing,
                                              MarkedParking, travel_lane_width_ft)

        violations = []
        for leg_name, leg in state.legs.items():
            if leg.curb_to_curb_ft is None:
                continue
            # travel_lane_width_ft works in the NOMINAL frame on purpose, and it must stay that
            # way: `painted_ft` below is read straight off the treatments, which express their
            # widths as offsets from that same datum (see apply_osm_parking's
            # lane_edge_from_nominal_ft). Both sides of the subtraction are in one frame, so it
            # measures what it says it measures. Swapping in the measured kerb -
            # kerbside_allowance_ft - would compare a traced offset against a nominal one and
            # report a lane width that is neither. Whether the paint fits the real kerb is a
            # different question, asked by check_paint_inside_the_curb.
            narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
            for side in BOTH_SIDES:
                painted_ft = 0.0
                if narrowing is not None and side in narrowing.sides:
                    painted_ft = narrowing.stripe_width_ft
                # Marked parking is asked second and REPLACES rather than maxes, which is what
                # the two dicts did in this order and is left as it was: a kerb with both
                # treatments is measured by the stalls plus their kerb buffer. Nothing in the
                # four sites applies both to one kerb, so this has never had to arbitrate.
                parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
                if parking is not None:
                    painted_ft = parking.depth_ft + parking.curb_offset_ft
                if painted_ft <= 0:
                    continue
                # The travel lane runs from the DIVIDER to the paint, not from the alignment to
                # the paint. Those are the same thing only while the two lanes straddle the
                # alignment; where a two-way bike lane has shifted them, ignoring it reported
                # broad_st_west's correctly-sized 11.00 ft lane as 9.58 ft. Signed, so the shift
                # is subtracted on the side it moved away from and added on the other.
                lane_ft = travel_lane_width_ft(state, leg_name, side, painted_ft)
                if lane_ft < TARGET_LANE_WIDTH_FT - LANE_WIDTH_TOLERANCE_FT:
                    violations.append(Violation(
                        "travel_lane_too_narrow",
                        f"{leg_name} {side} is painted {painted_ft:.1f} ft wide, leaving a "
                        f"{lane_ft:.1f} ft travel lane - under the {TARGET_LANE_WIDTH_FT:.0f} ft "
                        f"target. The paint has to be sized from what the road can spare",
                        tuple(leg.centerline.interpolate(leg.centerline.length / 2).coords[0])))
        return violations


class PaintInsideTheCurb(SceneCheck):
    """Road markings are painted on the road. None may cross its own side's curb.

    Touching the kerb is the point of a curbside marking, so this is not a clearance check -
    it is the difference between a line that ENDS at the kerb and one that carries on over
    it onto the footway.

    Measured against the real traced kerb at each vertex's own station, not against the
    nominal half-width and not against the pavement polygon. Both of those hide the failure:
    the nominal half-width IS the wrong number (broad_st_east's left kerb is traced at 22.7
    ft against a nominal 24.2 ft, so paint sized off the nominal figure sat 1.5 ft over it),
    and the pavement polygon is built from the same over-wide assumption, so paint outside
    the kerb still tested as inside the roadway.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        state, paint = scene.state, scene.paint
        violations = []
        for piece in paint:
            if piece.side is None or piece.leg is None:
                continue        # a corner treatment spans two legs - no single side to measure from
            leg = state.legs.get(piece.leg)
            if leg is None:
                continue
            coords = (piece.geometry.exterior.coords if piece.geometry.geom_type == "Polygon"
                      else piece.geometry.coords)
            points = np.asarray(coords, dtype=float)
            stations, offsets = station_offset_many(leg.centerline, points)
            curb_offsets = curb_offsets_at_stations(leg, piece.side, stations)
            if curb_offsets is None:
                continue        # no traced kerb on this side - nothing to be outside of
            over_ft = np.abs(offsets) - np.abs(curb_offsets)
            worst = float(over_ft.max())
            if worst > PAINT_PAST_CURB_TOLERANCE_FT:
                index = int(np.argmax(over_ft))
                violations.append(Violation(
                    "paint_over_the_curb",
                    f"{piece.leg} {piece.side}: {piece.kind} is painted {worst:.1f} ft past the traced "
                    f"kerb (tolerance {PAINT_PAST_CURB_TOLERANCE_FT} ft) - a marking may meet the kerb, "
                    f"never cross it. Usually paint sized off the nominal half-width instead of the "
                    f"kerb that was actually traced there",
                    tuple(points[index])))
        return violations


class ParkingIsLegal(SceneCheck):
    """No marked stall may sit inside a statutory no-parking setback.

    A proposal that paints a stall within 25 ft of a crossing, 50 ft of a stop sign or 10 ft
    of a hydrant is proposing something illegal under R.S. 39:4-138 - which is worse than a
    cosmetic error, because the drawing is what someone would build from. See
    src/geometry/daylighting.py for each distance and its citation.

    Measured on the stall dividers actually drawn, not on the intended start station: the
    two agree only if every builder downstream honoured it, and this is exactly the class of
    thing that silently stops being true.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        state, paint, crosswalk_offsets, props = (
            scene.state, scene.paint, scene.crosswalk_offsets, scene.props)
        from src.geometry.daylighting import no_parking_zones_ft

        violations = []
        zones_by_side = {}
        for piece in paint:
            if piece.kind not in (STALL_DIVIDER, PARKING_EDGE_LINE) or piece.leg is None:
                continue
            leg = state.legs.get(piece.leg)
            if leg is None:
                continue
            key = (piece.leg, piece.side)
            if key not in zones_by_side:
                zones_by_side[key] = no_parking_zones_ft(state, piece.leg, piece.side,
                                                          crosswalk_offsets, props)
            points = np.asarray(piece.geometry.coords, dtype=float)
            stations, _offsets = station_offset_many(leg.centerline, points)
            for zone in zones_by_side[key]:
                # Any part of the marking inside a prohibited interval, not just its near end -
                # a stall run that starts legally can still cross a hydrant further along.
                inside = ((stations > zone.start_ft + PARKING_SETBACK_TOLERANCE_FT)
                          & (stations < zone.end_ft - PARKING_SETBACK_TOLERANCE_FT))
                if not inside.any():
                    continue
                index = int(np.argmax(inside))
                violations.append(Violation(
                    "parking_inside_a_legal_setback",
                    f"{piece.leg} {piece.side}: marked parking reaches station "
                    f"{float(stations[inside].min()):.1f} ft, inside the no-parking zone from "
                    f"{zone.start_ft:.1f} to {zone.end_ft:.1f} ft - {zone.reason}",
                    tuple(points[index])))
        return violations


class MarkingsDoNotCollide(SceneCheck):
    """Two painted markings may not occupy the same asphalt.

    Real paint is opaque and applied once. Two hatch zones over the same ground get their
    strokes drawn twice - z-fighting in the 3D render, double ink on the plan - and it means
    the design is asserting two different things about one patch of road.

    Every other invariant here checks paint against the STREET - the kerb, the roadway, the
    crosswalk. This is the one that checks paint against other paint, which is how two
    overlapping no-parking zones came to be hatched twice with nothing complaining.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        paint = scene.paint
        violations = []
        # covers_area, not "is a Polygon": a bollard's geometry is a degenerate 1e-6 ft square
        # standing in for a point (src/geometry/paint.py:_dot), so it is a Polygon by type with no
        # area to collide. The marking knows what it is - see src/geometry/markings.py:Role.
        fills = [p for p in paint if p.covers_area]
        # Bounding boxes first: two markings can only share ground if their extents do, and an
        # envelope test is arithmetic against a GEOS overlay.
        fill_bounds = [p.geometry.bounds for p in fills]
        for i, a in enumerate(fills):
            for j in range(i + 1, len(fills)):
                if _boxes_apart(fill_bounds[i], fill_bounds[j]):
                    continue
                if lies_legitimately_on(a.kind, fills[j].kind):
                    continue        # a layer, not a collision - see markings.MAY_LIE_ON
                shared = a.geometry.intersection(fills[j].geometry)
                if shared.area <= MARKING_OVERLAP_TOLERANCE_SQ_FT:
                    continue
                where = shared.centroid
                violations.append(Violation(
                    "markings_collide",
                    f"{a.kind} and {fills[j].kind} overlap by {shared.area:.0f} sq ft"
                    + (f" on {a.leg} {a.side}" if a.leg else "")
                    + " - that ground would be painted twice",
                    (where.x, where.y)))

        # Lines too, and only where they run ALONG each other. Two lines that touch or cross are
        # ordinary - a stall divider meets the lane edge at right angles by design, and a hatch
        # stroke ends exactly on the edge line that bounds its zone. What is wrong is two lines
        # painted down the same stretch of road: the daylight zone's edge line and the parking
        # lane's sit at the same offset and are kept apart only by their station ranges.
        lines = [p for p in paint if p.kind.is_line]
        # Buffered once each, not once per comparison: buffering is the expensive half of this test.
        fattened = [p.geometry.buffer(COLLINEAR_PAINT_TOLERANCE_FT) for p in lines]
        line_bounds = [g.bounds for g in fattened]
        for i, a in enumerate(lines):
            for j in range(i + 1, len(lines)):
                if _boxes_apart(line_bounds[i], line_bounds[j]):
                    continue
                shared = fattened[i].intersection(lines[j].geometry)
                if shared.length <= MIN_COLLINEAR_OVERLAP_FT:
                    continue
                violations.append(Violation(
                    "markings_collide",
                    f"{a.kind} and {lines[j].kind} run along each other for {shared.length:.1f} ft"
                    + (f" on {a.leg} {a.side}" if a.leg else "")
                    + " - two lines painted down the same stretch of road",
                    (shared.centroid.x, shared.centroid.y)))
        return violations


def _boxes_apart(a: tuple, b: tuple) -> bool:
    """True when two (minx, miny, maxx, maxy) extents cannot possibly overlap."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _divider_shift_toward_ft(state, leg_name: str, side: str) -> float:
    """Signed divider offset toward `side`. Delegates to src/geometry/treatments/, which is
    where the one definition lives - see divider_shift_toward_ft for why there is only one."""
    from src.geometry.treatments import divider_shift_toward_ft

    return divider_shift_toward_ft(state, leg_name, side)


def _travel_lane_target_ft(state, leg_name: str, side: str) -> float:
    """How far from the alignment this kerb's travel lane reaches, for the paint checks.

    TARGET_LANE_WIDTH_FT everywhere the travel lanes straddle the alignment, which is every leg
    of every scenario but one. Where a two-way bike lane is involved the travel way has been
    shifted off the alignment, and the two sides of the leg are different questions:

      * the side CARRYING the lane - the travel way stops at the section's own inner edge, and
        that is what the lane's paint has to stay outside of.
      * the side OPPOSITE it - the travel lane still holds its target width, but it is measured
        from the shifted divider rather than from the alignment, so the offset paint must clear
        is the target plus however far the divider moved this way. Missing this second case
        reports a correctly-sized 11 ft lane as a violation.
    """
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT, AddTwoWayBikeLane

    for treatment in state.treatments_of(AddTwoWayBikeLane):
        if treatment.target.leg == leg_name and str(treatment.target.side) == str(side):
            return treatment.section(state).offsets_from_centerline_ft()["travel_lane_edge_ft"]
    return TARGET_LANE_WIDTH_FT + _divider_shift_toward_ft(state, leg_name, side)


class PaintClearOfTheTravelLane(SceneCheck):
    """The travel lane is clear asphalt, all the way to the target width.

    Distinct from check_travel_lanes, which checks the DESIGN arithmetic (does the paint the
    design calls for leave a target-width lane?). This checks the drawn geometry, and
    crucially it accounts for the fact that PAINT HAS WIDTH: an edge line centred on the 11 ft
    mark and painted 0.82 ft wide puts half its body inside the lane, so the arithmetic says
    11.0 ft where the asphalt is 10.59.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        state, paint = scene.state, scene.paint
        from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT

        violations = []
        for piece in paint:
            if piece.leg is None or piece.side is None or piece.kind.is_object:
                continue
            leg = state.legs.get(piece.leg)
            if leg is None or leg.curb_to_curb_ft is None:
                continue
            coords = (piece.geometry.exterior.coords if piece.geometry.geom_type == "Polygon"
                      else piece.geometry.coords)
            points = np.asarray(coords, dtype=float)
            stations, offsets = station_offset_many(leg.centerline, points)
            curb_offsets = curb_offsets_at_stations(leg, piece.side, stations)
            if curb_offsets is None:
                continue
            # What the lane is entitled to AT EACH STATION: the target, or the TRACED kerb where
            # the kerb is closer than that. A road narrower than the target is a fact about the
            # street, not something this design introduced - W Broad's north-east approach has
            # the NJDOT alignment 7.2 ft from its right kerb, so there is no 11 ft lane to
            # protect there and the paint correctly clamps to the kerb. Comparing against the
            # NOMINAL half-width instead would call that a violation on every vertex.
            #
            # AND THE TARGET ITSELF MOVES WHERE THE TRAVEL LANES DO. Measuring from the alignment
            # is right only while the two travel lanes straddle it; a two-way bike lane on one
            # side shifts them off it (see TwoWayBikeLane), so on that kerb the travel lane
            # begins at the section's inner edge, not at 11 ft from the alignment.
            #
            # Re-expressed rather than skipped: the property is unchanged - no paint inside the
            # travel lane - and exempting these legs would drop the check on precisely the
            # design most likely to get the arithmetic wrong.
            target_ft = _travel_lane_target_ft(state, piece.leg, piece.side)
            entitled = np.minimum(target_ft,
                                   np.abs(curb_offsets) - LANE_EDGE_LINE_WIDTH_FT)
            # The painted body reaches half a stripe width inside its own centreline.
            shortfall = entitled - (np.abs(offsets) - LANE_EDGE_LINE_WIDTH_FT / 2)
            worst = float(shortfall.max())
            if worst > LANE_WIDTH_TOLERANCE_FT:
                index = int(np.argmax(shortfall))
                inner_ft = float(np.abs(offsets[index])) - LANE_EDGE_LINE_WIDTH_FT / 2
                violations.append(Violation(
                    "paint_in_the_travel_lane",
                    f"{piece.leg} {piece.side}: {piece.kind} is painted to {inner_ft:.2f} ft from "
                    f"the centerline, leaving a {inner_ft:.2f} ft travel lane where "
                    f"{float(entitled[index]):.2f} ft was available - the stripe's own width has "
                    f"to come out of the treatment, not out of the lane",
                    tuple(points[index])))
        return violations


class TravelLanesHoldTheTarget(SceneCheck):
    """A leg the design has RESTRIPED must not leave a travel lane wider than the target.

    An 11 ft lane is the cheapest safety intervention this project has - paint - and a wide one
    invites exactly the speed every other treatment here is trying to reduce. So it is not a
    per-proposal nicety: wherever a design has already decided to restripe a leg, leaving the
    other side over-wide is an omission rather than a choice.

    ONLY LEGS THAT CARRY PAINT. A leg nobody has touched is the street as it is, and existing
    conditions must be drawable however wide the road happens to be - failing that would be
    reporting reality as a defect. What this catches is a design that narrowed one kerb and
    forgot the other.

    Distinct from TravelLanesKeepTheirWidth, which is the same measurement in the other
    direction - that one catches paint that leaves TOO LITTLE lane. Together they pin the lane
    to the target from both sides.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        from src.geometry.targets import LegSide, LegTarget
        from src.geometry.model import narrowest_half_width_ft
        from src.geometry.treatments import (TARGET_LANE_WIDTH_FT, AddBikeLane, LaneNarrowing,
                                              MarkedParking, travel_lane_width_ft)

        state = scene.state
        violations = []
        for leg_name, leg in state.legs.items():
            if leg.curb_to_curb_ft is None:
                continue
            narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
            sides = ("left", "right")

            def painted_ft(side, narrowing=narrowing, leg_name=leg_name):
                parking = state.treatment_for(MarkedParking, LegSide(leg_name, side))
                if parking is not None:
                    return parking.depth_ft + parking.curb_offset_ft
                if narrowing is not None and side in narrowing.sides:
                    return narrowing.stripe_width_ft
                return 0.0

            restriped = any(painted_ft(side) > 0
                            or state.treatment_for(AddBikeLane, LegSide(leg_name, side)) is not None
                            for side in sides)
            if not restriped:
                continue        # untouched leg - the street as it is, not a design
            for side in sides:
                if state.treatment_for(AddBikeLane, LegSide(leg_name, side)) is not None:
                    continue    # the lane's own cross-section defines this edge
                # WHERE THERE IS NO PAINT THE KERB IS THE BOUND, and the traced kerb is not the
                # nominal one. travel_lane_width_ft works in the NOMINAL frame, which is right
                # when it is subtracting nominal-referenced paint - but on an unpainted side the
                # lane really ends at the traced kerb, and on w_broad_st_northeast that is 1.66 ft
                # closer than nominal (12.93 ft nominal against the 10.14 ft the section leaves).
                lane_ft = travel_lane_width_ft(state, leg_name, side, painted_ft(side))
                if painted_ft(side) <= 0:
                    traced_ft = narrowest_half_width_ft(leg, side)
                    lane_ft = min(lane_ft, traced_ft - _divider_shift_toward_ft(state, leg_name,
                                                                                side))
                if lane_ft > TARGET_LANE_WIDTH_FT + LANE_WIDTH_TOLERANCE_FT:
                    violations.append(Violation(
                        "travel_lane_over_target",
                        f"{leg_name} {side} is left {lane_ft:.2f} ft wide on a leg this design has "
                        f"already restriped - {lane_ft - TARGET_LANE_WIDTH_FT:.2f} ft over the "
                        f"{TARGET_LANE_WIDTH_FT:.0f} ft target. Spare width beside a travel lane "
                        f"is parking or hatching, never lane; call "
                        f"treatments.hold_travel_lane_at_target for this kerb",
                        tuple(leg.centerline.interpolate(leg.centerline.length / 2).coords[0])))
        return violations


class BollardsStandInTheirBuffer(SceneCheck):
    """A flex post protecting a bike lane must stand in the BUFFER, not in the lane.

    A post inside the lane is worse than no post: it removes ridable width, it is an obstacle
    exactly where a rider is meant to be, and the drawing still reads as a protected lane.

    Nothing else catches it. post_not_in_the_render compares the paint against the props, and a
    wrong cross-section produces both, so they agree; PaintClearOfTheTravelLane looks at the
    other edge; a golden only catches a CHANGE, and a new scenario has none. Two consistent
    views of a wrong design is the failure mode this module exists for, so the guard compares a
    post against the lane it protects rather than against another derivation of itself.

    Checked against the painted lane SURFACE rather than recomputed offsets, deliberately: the
    surface is what a rider rides on and what the render draws, so a post outside it is genuinely
    clear of the lane however the arithmetic got there.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        from shapely.ops import unary_union

        from src.geometry.markings import BIKE_LANE_SURFACE

        paint = scene.paint
        violations = []
        # Per kerb, so a post is only ever tested against the lane on its own side of its own leg.
        surfaces: dict[tuple, list] = {}
        for piece in paint:
            if piece.kind is BIKE_LANE_SURFACE and piece.leg is not None and piece.side is not None:
                surfaces.setdefault((piece.leg, piece.side), []).append(piece.geometry)
        if not surfaces:
            return violations
        merged = {kerb: unary_union(polys) for kerb, polys in surfaces.items()}
        for piece in paint:
            if not piece.kind.is_object or piece.leg is None or piece.side is None:
                continue
            lane = merged.get((piece.leg, piece.side))
            if lane is None:
                continue        # no bike lane on this kerb - a daylight or parking-buffer post
            post = piece.geometry.centroid
            if not lane.contains(post):
                continue
            # How far in, so the report says whether this is a rounding error or a row of posts
            # down the middle of the lane.
            depth_ft = post.distance(lane.exterior)
            violations.append(Violation(
                "bollard_in_the_bike_lane",
                f"{piece.leg} {piece.side}: a flex post stands {depth_ft:.2f} ft INSIDE the bike "
                f"lane surface, not in the buffer beside it - it removes ridable width and puts an "
                f"obstacle where a rider belongs, while the drawing still reads as protected. A "
                f"protecting post belongs in the buffer on the traffic side; see "
                f"AddBikeLaneBollards.paint, which must read section(state) rather than the "
                f"declared cross-section",
                (post.x, post.y)))
        return violations


class BollardsAreProps(SceneCheck):
    """A bollard the treatment layer paints must also exist as a prop, or the 3D render
    has no post there.

    The two renderers get posts from different places. The plan view draws them straight off
    the paint (src/geometry/paint.py emits a dot per post); the 3D render builds objects, and
    it only ever builds objects from props - it never turns a marking into one. So a post that
    exists only as a PaintPiece is a post that is in the 2D picture and absent from the render,
    with neither view internally wrong about anything.

    Deliberately one-directional. A daylight zone's posts are props ONLY - nothing paints
    them, and the plan view draws them from the props - so a prop with no paint behind it is
    correct and common. See src/render/props.py:bollard_props_from_paint.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        paint, props = scene.paint, scene.props
        placed = np.array([p["position_ft"] for p in props if p["type"] == "bollard"], dtype=float)
        violations = []
        for piece in paint:
            if not piece.kind.is_object:
                continue
            point = piece.geometry.centroid
            if len(placed) and np.hypot(placed[:, 0] - point.x,
                                        placed[:, 1] - point.y).min() <= POST_PROP_TOLERANCE_FT:
                continue
            violations.append(Violation(
                "post_not_in_the_render",
                f"{piece.leg} {piece.side}: a bollard is drawn in the plan view with no prop at "
                f"that position, so the 3D render builds no post there - see "
                f"src/render/props.py:bollard_props_from_paint",
                (point.x, point.y)))
        return violations


class PavementRingCloses(SceneCheck):
    """The pavement must be one simple polygon - no bowties, no pinches."""

    def run(self, scene: SceneContext) -> list[Violation]:
        pavement = scene.pavement
        if pavement is None or pavement.is_empty:
            return [Violation("pavement_ring", "no pavement polygon was built for this junction")]
        if not pavement.is_valid:
            from shapely.validation import explain_validity
            return [Violation("pavement_ring", f"pavement polygon is invalid: {explain_validity(pavement)}")]
        return []


class CrosswalksCrossTheRoadway(SceneCheck):
    """A painted crosswalk lies across the roadway, touching the curb at both ends.

    Catches the two failures seen here: a band drawn out in the middle of the carriageway
    parallel to traffic (it was inheriting a leg offset from the wrong frame), and a band
    sitting almost entirely outside the pavement.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        bands, pavement = scene.crosswalk_bands, scene.pavement
        if pavement is None or pavement.is_empty:
            return []
        violations = []
        for leg_name, band in bands.items():
            if band is None or band.is_empty or band.area <= 0:
                continue
            inside = band.intersection(pavement).area / band.area
            if inside < MIN_CROSSWALK_IN_PAVEMENT:
                violations.append(Violation(
                    "crosswalk_off_the_roadway",
                    f"{leg_name}'s crosswalk is only {inside * 100:.0f}% inside the roadway it crosses "
                    f"(expected at least {MIN_CROSSWALK_IN_PAVEMENT * 100:.0f}%)",
                    (band.centroid.x, band.centroid.y)))
        return violations


class CrossingsAreNotPaintedOver(SceneCheck):
    """No marking is painted over a crossing that belongs to another junction in the frame.

    MarkingsDoNotCollide already stops two markings sharing ground, and this junction's own four
    crossings are cut out of everything by curbside_paint_ft. Neither covered the case this
    exists for: a crosswalk at a junction this site does not model. Broad St's frame contains
    Blackwell Avenue, whose three traced crossings src/geometry/surveyed.py draws and whose
    ground nothing was cut against, so kerbside paint was laid straight over a marked zebra.
    The crossings reached the drawing through one path and the paint through another, and no
    invariant looked across the two. The fix (curbside_paint_ft's `crossings_elsewhere`) is a
    subtraction, and a subtraction that is quietly dropped looks like nothing at all - which is
    what this makes visible.

    Not restricted to fills. A line painted down a crosswalk is the same false statement about
    the street as a hatched zone over one, and on the corridor proposal the lines were most of it.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        from shapely.ops import unary_union

        bands = [b for b in scene.unmodelled_crossing_bands if b is not None and not b.is_empty]
        if not bands:
            return []
        crossings = unary_union(bands)
        extent = crossings.bounds
        violations = []
        for piece in scene.paint:
            # An OBJECT is a point standing in for a post and has no area to overlap; it is kept
            # out of a driveway by paint.stands_in_an_opening and out of a crossing by the same
            # keep_clear this checks, so testing its 1e-6 ft square here would only ever misreport.
            if piece.kind.is_object or _boxes_apart(extent, piece.geometry.bounds):
                continue
            shared = crossings.intersection(piece.geometry)
            if shared.is_empty:
                continue
            if piece.covers_area:
                if shared.area <= MARKING_OVERLAP_TOLERANCE_SQ_FT:
                    continue
                how_much = f"{shared.area:.0f} sq ft of"
            else:
                if shared.length <= MIN_COLLINEAR_OVERLAP_FT:
                    continue
                how_much = f"{shared.length:.1f} ft of"
            violations.append(Violation(
                "paint_over_a_crossing",
                f"{piece.kind} is painted across {how_much} a surveyed crosswalk at another "
                f"junction in this frame"
                + (f" ({piece.leg} {piece.side})" if piece.leg else "")
                + " - a crossing outranks kerbside paint wherever it is, not only at the "
                  "junction the drawing is centred on",
                (shared.centroid.x, shared.centroid.y)))
        return violations


class NoPaintInsideTheJunction(SceneCheck):
    """No kerbside marking may be drawn entirely inside the junction's own mouth.

    THE ONE INTERSECTION THAT HAD NO RULE. Every other street these legs cross becomes a
    src/geometry/kerbs.py:KerbOpening and one table decides what each marking does at it; the
    junction the drawing is CENTRED on was handled by hand, with a mean-station test as the only
    thing keeping paint out of the corner. A mean station is not a side - on a crossing surveyed
    43.7 deg off square the offcut behind it runs diagonally and its mean lands past the
    crossing - so hatching was drawn inside the intersection. CrossingsAreNotPaintedOver does not
    cover this: this junction's CROSSINGS are cut out of everything by curbside_paint_ft, but the
    OFFCUT is not.

    THE MOUTH, NOT THE CROSSING, is what this measures against - see
    src/geometry/model/corners.py:junction_mouth_ft. The corner return's tangent point is where
    this kerb starts existing, so a marking drawn beside it before that point is drawn beside a
    corner. And it is ZERO on a kerb that runs straight through, which is why this check does not
    have to carve out MUTCD 11th ed. 3B.11(07)'s T-intersection exception: a through-running kerb
    has no mouth, so nothing on it can be inside one.

    ENTIRELY INSIDE, deliberately, so this has no false positives to argue about. A zone cut by a
    skewed crossing has a diagonal end whose stations reach back into the mouth while the zone
    itself sits outside; reporting the overlap would fire on every one of those. A piece whose
    every vertex is inside the mouth is not a zone with a long end - it is a marking in the
    intersection. That is the claim the picture makes, and it is the one worth failing on.

    RIMS ARE NOT MARKINGS IN THE INTERSECTION, they are the line that closes one AT its edge, so
    they are skipped rather than reported. A rim runs ACROSS the kerbside strip at the mouth's
    far end, and the mouth's far end is a station - so every vertex of it is inside by
    construction, always, however correct it is. What must not be inside the junction is the zone
    the rim closes, and that is what the loop below tests. This exemption is the reason the
    junction's square end can be one mechanism (PaintContext.rim) rather than a second
    ZONE_END_LINE emitted beside it.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        from src.geometry.markings import carries_across_an_intersection
        from src.geometry.paint import junction_mouths_ft

        # THE SAME RESOLUTION THE PAINT WAS CUT AGAINST - the mouth ends at the leg's crosswalk
        # where one is painted, and at the corner return only where none is. Asked of the shared
        # resolver rather than recomputed from the corner alone, because a check measuring against
        # a different boundary from the cut is a check that passes on paint nobody drew.
        legs = scene.legs
        mouths = junction_mouths_ft(scene.state, scene.crosswalk_bands,
                                     scene.marked_crosswalks)
        violations = []
        for piece in scene.paint:
            if not piece.leg or not piece.side or piece.rim is not None:
                continue
            if carries_across_an_intersection(piece.kind):
                continue
            leg = legs.get(piece.leg)
            if leg is None:
                continue
            mouth = mouths.get((piece.leg, str(piece.side)))
            if mouth is None:
                continue
            geometry = piece.geometry
            coords = (geometry.exterior.coords if geometry.geom_type == "Polygon"
                       else getattr(geometry, "coords", None))
            if coords is None:
                continue
            stations, _offsets = station_offset_many(leg.centerline,
                                                      np.asarray(coords, dtype=float))
            if float(stations.max()) > mouth[1]:
                continue
            how_much = (f"{geometry.area:.0f} sq ft of" if piece.kind.covers_area
                         else f"{geometry.length:.1f} ft of")
            violations.append(Violation(
                "paint_inside_the_junction",
                f"{how_much} {piece.kind} is drawn inside the junction's own mouth on "
                f"{piece.leg} {piece.side} - every vertex of it lies between the junction node "
                f"and the corner return's tangent point at {mouth[1]:.1f} ft, where there is no "
                f"kerb for it to run beside. This junction opens the kerb exactly the way a "
                f"cross street does (src/geometry/kerbs.py:OpeningSource.is_an_intersection); "
                f"a marking that survives it was never cut against it",
                (geometry.centroid.x, geometry.centroid.y)))
        return violations


class StopBarsOnEnteringHalf(SceneCheck):
    """A driver stops in their own lanes, never across the opposing ones.

    The bar must stay on one side of its leg's centerline, never full width across both
    directions of travel.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        bars, legs = scene.stop_bars, scene.legs
        violations = []
        for leg_name, bar in bars.items():
            leg = legs.get(leg_name)
            if leg is None or bar is None or bar.is_empty or bar.area <= 0:
                continue
            _stations, offsets = station_offset_many(
                leg.centerline, np.asarray(bar.exterior.coords, dtype=float))
            # AGAINST THE PAINTED LINE, NOT THE ALIGNMENT. They coincide until a two-way bike
            # lane shifts the travel lanes off the alignment, and then the line a driver actually
            # sees is the divider. Measured from the alignment, a bar resting correctly against
            # that divider looks like it crosses, and a bar genuinely painted 3.15 ft across it
            # looks fine - which is what shipped on broad_st_east.
            divider_ft = _divider_shift_toward_ft(scene.state, leg_name, Side.LEFT)
            # Positive is the entering (LEFT) side, so anything below the divider is over the line.
            past_ft = float(divider_ft - offsets.min())
            if past_ft > STOP_BAR_PAST_CENTERLINE_TOLERANCE_FT:
                violations.append(Violation(
                    "stop_bar_crosses_centerline",
                    f"{leg_name}'s stop bar is painted {past_ft:.2f} ft past the centreline into "
                    f"the opposing lanes - it must rest AGAINST that line, not cross it. A stop "
                    f"bar covers the entering half only (MUTCD), and where a two-way bike lane "
                    f"has shifted the travel lanes the line to rest against is the divider, not "
                    f"the NJDOT alignment",
                    (bar.centroid.x, bar.centroid.y)))
        return violations


# ---------------------------------------------------------------------------
# Running them together
# ---------------------------------------------------------------------------

def check_scene(scene: SceneContext) -> list[Violation]:
    """Every registered invariant, all violations, no raising.

    A loop over CHECKS, not a hand-written list of calls: that lets a check be written and never
    run, and lets two checks be handed different versions of the same geometry. See SceneCheck
    and SceneContext for both halves of that.
    """
    return [violation for check in CHECKS for violation in check.run(scene)]


def assert_scene_valid(scene: SceneContext, scenario: str = "") -> None:
    """Raise SceneInvariantError listing EVERY violation in this scene, or return quietly."""
    model = scene.model
    violations = check_scene(scene)
    for violation in (v for v in violations if not v.fatal):
        print(f"  SOURCE CONFLICT: {violation}")

    fatal = [v for v in violations if v.fatal]
    if not fatal:
        return
    where = f" ({scenario})" if scenario else ""
    listing = "\n  ".join(str(v) for v in fatal)
    raise SceneInvariantError(
        f"{len(fatal)} scene invariant(s) failed for "
        f"{model.config.get('intersection', {}).get('name', 'this junction')}{where}:\n  {listing}\n"
        "These are geometry errors, not rendering preferences - the render would be asserting "
        "something false about the street. See src/checks.py for what each one means."
    )


def _all_curb_lines(legs: dict, corner_fillets: dict) -> list:
    """Every line a kerb is actually drawn along: both sides of every leg, plus the corner
    arcs. A pad at a corner ramp is nearest the ARC, not either leg's straight run."""
    lines = [getattr(leg, f"{side}_curb") for leg in legs.values() for side in ("left", "right")]
    lines += [pieces["arc"] for pieces in corner_fillets.values() if "arc" in pieces]
    return [line for line in lines if line is not None and not line.is_empty]
