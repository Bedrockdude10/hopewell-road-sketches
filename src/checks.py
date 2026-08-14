"""Scene invariants: the things that must be true of every render, checked every time.

This project's recurring failure mode is not a crash. It is a render that looks finished
and asserts something false about the street - a tactile pad in the carriageway, a curb
line drawn straight across the middle of the intersection, a crosswalk floating in the
roadway. Every one of those shipped, was spotted by eye in a picture, and cost a round
trip to diagnose. They are all cheap to detect in geometry.

So each one is an invariant here, and each is checked on BOTH paths - the 2D plan view and
the 3D export - because the whole premise of the 2D reconstruction is that it shows what
the 3D render will show. A check that only guards the export lets the two drift.

Two design choices worth stating:

  * ALL violations are collected before anything is raised. Failing on the first one turns
    a single bad junction into one edit-run cycle per violation, which is exactly the slow
    iteration this module exists to stop.
  * A violation carries its coordinates. The plan view draws them, so the error message and
    the picture agree about where to look.

`check_scene` reports; `assert_scene_valid` raises. Phase scripts save the plot first and
assert after, so a failure always comes with a picture of itself.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point

from src.geometry.markings import PARKING_EDGE_LINE, STALL_DIVIDER
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
# A painted crosswalk lies in the roadway it crosses, full stop. This was 0.55 - i.e. 45% of
# a crossing was allowed to sit on the footway - back when the span was half the leg's
# NOMINAL width either side of the centerline and routinely overshot the traced kerb. Now the
# reach is bounded by the roadway itself (crosswalks.crosswalk_reach_to_curbs_ft), every band
# at every site measures 99.96% inside or better, and the loose bound was hiding exactly the
# failure it was named for: end bars painted up the corner onto the sidewalk.
MIN_CROSSWALK_IN_PAVEMENT = 0.99
# A stop bar covers the entering half only. This much of it may cross the centerline before
# it is genuinely painted across opposing lanes.
MAX_STOP_BAR_OPPOSING_FRACTION = 0.15
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

    One object rather than a per-check argument list, because the argument list was the bug.
    check_scene called thirteen functions with thirteen hand-picked subsets of the same scene,
    and getting a subset wrong was invisible: one check was handed the crossing bands built WITH
    the two-pass mutual-exclusion reaches and another the bands built without them, so at W Broad
    & Louellen the two were validating geometry 15 sq ft apart, and one of them geometry that no
    renderer drew. A check cannot now be handed a different scene from its neighbour.

    Everything defaults, and `state` defaults to a real empty DesignState rather than None, so a
    test can describe the two parts of a scene its check reads and every other check still runs
    over it and finds nothing. A scene with nothing in it is vacuously valid, which is what
    test_a_check_reading_a_field_the_caller_left_out_gets_a_default pins - it caught this class
    reaching `scene.state.legs` through a None.
    """
    model: object = None
    state: "DesignState" = field(default_factory=lambda: _empty_state())
    pavement: object = None
    props: tuple = ()
    paint: tuple = ()
    crosswalk_bands: dict = field(default_factory=dict)
    crosswalk_offsets: dict = field(default_factory=dict)
    stop_bars: dict = field(default_factory=dict)

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

    The point of the base class is that the registry is not written by hand. check_scene used to
    be a `+` chain of thirteen calls, and a check that was defined and never added to that chain
    was dead code that looked live - the same shape of mistake as a paint kind declared and
    routed nowhere (see src/geometry/markings.py). Now the chain IS the file.

    A check reads what it needs off the SceneContext, returns every violation it finds, and never
    raises: collecting all of them means one edit-run cycle for a bad junction instead of one per
    violation. See assert_scene_valid for the raising wrapper.
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
    all footway furniture. A pad drawn in the road is the worst case - it is the render
    asserting something false about an accessibility feature - but a stop sign in the
    middle of the street is just as wrong and had no check at all before.
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

    A leg's curb line starts at that leg's cross-section and goes outward. When one runs
    backwards past the junction it draws curb straight across the middle of the
    intersection - marking a kerb where there is open roadway - and it crosses the opposite
    leg's curb, which is what makes the pavement ring self-intersect. Measured in the leg's
    own frame, so it is the same signed station the curb was built from.
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
    """A leg's two curb lines are the two sides of one street: they never meet.

    They crossed when a curb was extrapolated out of a corner return's flare, which closed
    the roadway to zero width and then opened it inside out.
    """

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
    hatching that leaves less: fixed 5 ft and 8 ft paint widths, applied without reference
    to how much road was left, once produced 1.7 ft lanes there.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        state = scene.state
        from src.geometry.targets import BOTH_SIDES, LegSide, LegTarget
        from src.geometry.treatments import LaneNarrowing, MarkedParking, TARGET_LANE_WIDTH_FT

        violations = []
        for leg_name, leg in state.legs.items():
            if leg.curb_to_curb_ft is None:
                continue
            # NOMINAL half-width on purpose, and it must stay that way: `painted_ft` below is
            # read straight off the treatments, which express their widths as offsets from
            # this same datum (see apply_osm_parking's lane_edge_from_nominal_ft). Both sides
            # of the subtraction are in one frame, so it measures what it says it measures.
            # Swapping in the measured kerb here - kerbside_allowance_ft - would compare a
            # traced offset against a nominal one and report a lane width that is neither.
            # Whether the paint fits the real kerb is a different question, asked by
            # check_paint_inside_the_curb.
            half_ft = leg.curb_to_curb_ft / 2
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
                lane_ft = half_ft - painted_ft - _divider_shift_toward_ft(state, leg_name, side)
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

    The bug this was written for: every curbside strip was built by pairing
    `substring(curb, start, curb.length)` with `substring(inner, start, inner.length)`.
    Those are arc lengths along two different lines, so the two boundaries were cut at
    unrelated stations - up to 49 ft apart at the far end, where the traced kerb ran on past
    the leg. The result was a wedge with long diagonal ends rather than a strip, which both
    fragmented the hatching and pushed paint outside the kerb.
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

    Written after the daylighting work put a hydrant's no-parking zone (18.9-38.9 ft on
    broad_st_west) entirely inside the junction's (0-45.7 ft) and hatched both. Nothing
    caught it: every other invariant here checks paint against the STREET - the kerb, the
    roadway, the crosswalk - and none checked paint against other paint.
    """

    def run(self, scene: SceneContext) -> list[Violation]:
        paint = scene.paint
        violations = []
        # covers_area, not "is a Polygon": a bollard's geometry is a degenerate 1e-6 ft square
        # standing in for a point (src/geometry/paint.py:_dot), so it is a Polygon by type with no
        # area to collide, and every test here used to carry `and p.kind != "bollard"` to say so.
        # The marking knows what it is - see src/geometry/markings.py:Role.
        fills = [p for p in paint if p.covers_area]
        # Bounding boxes first: two markings can only share ground if their extents do, and an
        # envelope test is arithmetic against a GEOS overlay. This is O(n^2) either way, and a
        # proposal carries a few hundred pieces, so the pairs that reach GEOS should be the pairs
        # that might actually overlap.
        fill_bounds = [p.geometry.bounds for p in fills]
        for i, a in enumerate(fills):
            for j in range(i + 1, len(fills)):
                if _boxes_apart(fill_bounds[i], fill_bounds[j]):
                    continue
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
        # Buffered once each, not once per comparison: buffering is the expensive half of this
        # test and it was inside the inner loop, so each line was re-buffered for every line
        # after it.
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
    """How far the travel-lane divider sits off the alignment, measured TOWARD `side`.

    Zero on every leg whose travel lanes straddle the alignment, which is all of them until a
    two-way bike lane takes width out of one kerbside. Signed, because the two sides of a leg
    see the same shift in opposite directions and a check that ignores the sign is wrong on
    exactly one of them.

    ONE DEFINITION, read off the design, because two checks and two renderers all need it and a
    check carrying its own copy of the arithmetic is the divergence this module exists to catch.
    """
    from src.geometry.treatments import AddTwoWayBikeLane, travel_lane_divider_shift_ft

    for treatment in state.treatments_of(AddTwoWayBikeLane):
        if treatment.target.leg != leg_name:
            continue
        shift_ft = travel_lane_divider_shift_ft(treatment.section(state))
        # The treatment's own side is the side the lane is on; the shift is defined as positive
        # AWAY from it.
        return -shift_ft if str(treatment.target.side) == str(side) else shift_ft
    return 0.0


def _travel_lane_target_ft(state, leg_name: str, side: str) -> float:
    """How far from the alignment this kerb's travel lane reaches, for the paint checks.

    TARGET_LANE_WIDTH_FT everywhere the travel lanes straddle the alignment, which is every leg
    of every scenario but one. Where a two-way bike lane is involved the travel way has been
    shifted off the alignment, and the two sides of the leg are different questions:

      * the side CARRYING the lane - the travel way stops at the section's own inner edge, and
        that is what the lane's paint has to stay outside of.
      * the side OPPOSITE it - the travel lane still holds its target width, but it is measured
        from the shifted divider rather than from the alignment, so the offset paint must clear
        is the target plus however far the divider moved this way.

    Missing that second case reported broad_st_west's own correctly-sized 11.00 ft lane as a
    9.58 ft violation, because the divider there sits 1.42 ft on the far side of the alignment.
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
    crucially it accounts for the fact that PAINT HAS WIDTH. Every edge line was centred on
    the 11 ft mark and painted 0.82 ft wide, so half its body lay inside the lane and every
    approach at every site was really 10.59 ft. The arithmetic said 11.0 and the check that
    only looked at the arithmetic agreed with it.
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
            # What the lane is entitled to AT EACH STATION: the target, or the kerb where the
            # kerb is closer than that. A road narrower than the target is a fact about the
            # street, not something this design introduced - and it is not hypothetical.
            # W Broad's north-east approach has the NJDOT alignment 7.2 ft from its right kerb
            # and 25-31 ft from its left, so on that side there is no 11 ft lane to protect and
            # the paint correctly clamps to the kerb. Comparing against the NOMINAL half-width
            # instead would call that a violation on every vertex.
            #
            # AND THE TARGET ITSELF MOVES WHERE THE TRAVEL LANES DO. This check measures the lane
            # from the alignment, which is right only while the two travel lanes straddle it. A
            # two-way bike lane on one side shifts them off it (see TwoWayBikeLane), so on that
            # kerb the lane a rider is protected from does not begin at 11 ft from the alignment
            # - it begins at the section's inner edge, and paint outside that is in the lane the
            # design actually drew rather than in a travel lane.
            #
            # Re-expressed rather than skipped. The property being checked is unchanged - no
            # paint inside the travel lane - and dropping the check on these legs would have
            # dropped it on precisely the design most likely to get the arithmetic wrong.
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


class BollardsStandInTheirBuffer(SceneCheck):
    """A flex post protecting a bike lane must stand in the BUFFER, not in the lane.

    A post inside the lane is worse than no post: it removes ridable width, it is an obstacle
    exactly where a rider is meant to be, and the drawing still reads as a protected lane. This
    is the invariant that was missing when a two-way lane's posts were drawn 12.5 ft from the
    alignment on broad_st_east, inside a lane spanning 8.85-20.85 ft - 30 posts down the middle
    of the ridable surface, in both views.

    NOTHING CAUGHT IT, and the reason is worth stating. post_not_in_the_render compares the paint
    against the props, and both were derived from the same wrong cross-section, so they agreed.
    PaintClearOfTheTravelLane looks at the other edge. The geometry goldens would have caught a
    CHANGE, but this scenario was new, so there was no golden to differ from. Two consistent views
    of a wrong design is precisely the failure mode this module exists for, and the guard has to
    compare a post against the lane it protects rather than against another derivation of itself.

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
    exists only as a PaintPiece is a post that is in the 2D picture and absent from the
    render. That shipped: Broad St's bike lanes were drawn with 61 protecting flex posts and
    exported with none, and neither view was internally wrong about anything.

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


class StopBarsOnEnteringHalf(SceneCheck):
    """A driver stops in their own lanes, never across the opposing ones.

    The bar must stay on one side of its leg's centerline. It was previously drawn full
    width, across both directions of travel.
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
            spans_both = offsets.min() < 0 < offsets.max()
            if not spans_both:
                continue
            minority = min(abs(offsets.min()), abs(offsets.max())) / (offsets.max() - offsets.min())
            if minority > MAX_STOP_BAR_OPPOSING_FRACTION:
                violations.append(Violation(
                    "stop_bar_crosses_centerline",
                    f"{leg_name}'s stop bar reaches {minority * 100:.0f}% of its width across the "
                    f"centerline into opposing lanes - a stop bar covers the entering half only",
                    (bar.centroid.x, bar.centroid.y)))
        return violations


# ---------------------------------------------------------------------------
# Running them together
# ---------------------------------------------------------------------------

def check_scene(scene: SceneContext) -> list[Violation]:
    """Every registered invariant, all violations, no raising.

    A loop over CHECKS, not a list of calls: this used to name each check and pick its arguments
    here, so a check could be written and never run, and two checks could be handed different
    versions of the same geometry. See SceneCheck and SceneContext for both halves of that.
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
