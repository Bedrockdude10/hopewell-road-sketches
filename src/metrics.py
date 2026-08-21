"""What a design ACHIEVES, measured off the geometry it actually drew.

Every dimension this project puts on a drawing is an INPUT — the 55.5 ft street, the 8 ft
stall, R=20 at the corner. The outcome numbers live here, computed from the same resolved
scene both renderers draw from (src/render/scene.py:SceneGeometry), never from the config.

A crossing's length is NOT the leg's configured width: crosswalk_reach_to_curbs_ft measures
to the traced kerbs, asymmetrically (12 ft one way, 20 the other on a 30 ft street). Parking
stalls are counted off the PARKING_EDGE_LINE pieces the paint builder emitted, because a
hydrant or driveway splits a kerb into two runs and the paint says where the runs are.

Nothing here decides anything. Every value comes from geometry some other module resolved;
this module only measures it and says what changed.
"""
import itertools
from dataclasses import dataclass
from math import sqrt

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from src.geometry.markings import PARKING_EDGE_LINE
from src.geometry.model import station_offset_many
from src.geometry.targets import Corner, LegSide
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.treatments.state import DesignState

# MUTCD's normal walking speed for timing a pedestrian clearance interval. Stated on the
# panel beside every time it produces, because a time in seconds is an assumption about who
# is crossing, not a measurement - and the slower walker is usually the person the treatment
# is for, which is what SLOW_WALKING_SPEED_FT_S is for.
MUTCD_WALKING_SPEED_FT_S = 3.5
SLOW_WALKING_SPEED_FT_S = 3.0

# AASHTO's low-speed side friction factor, and a flat (uncrowned) turn. Together these give
# the classic minimum-radius relation R = V^2 / (15 * (e + f)), solved for V. It is a comfort
# model of a vehicle tracking the curb face, not a measured speed and not a design speed -
# which is why turn speed is labelled as modelled wherever it is shown.
TURN_SIDE_FRICTION = 0.30
TURN_SUPERELEVATION = 0.0

# A crossing axis clipped by an island can leave a hairline piece where the two touch. Below
# this it is not a stage anyone walks, it is a tangent.
MIN_STAGE_FT = 0.1


def leg_label(leg_name: str) -> str:
    """A leg's name as a street, not as a dict key: broad_st_east -> "Broad St East".

    The internal name is the key everything is looked up by and has to stay a key. A reader
    orients by the street name, so that is what a public-facing number is labelled with.
    """
    return leg_name.replace("_", " ").title()


def stalls_in_run(length_ft: float, stall_length_ft: float) -> int:
    """How many whole stalls fit in one painted run of parking.

    Here rather than at the two call sites (this module's total, and the per-run label in
    src/render/plan_view.py:_label_paint) so the number beside a run on the drawing and the
    number in the summary panel cannot be two different rules. They were computed in one
    place and about to be computed in a second, which is how the twenty dicts started.
    """
    if stall_length_ft <= 0:
        raise ValueError(f"A stall needs a length; got stall_length_ft={stall_length_ft}.")
    return max(int(length_ft // stall_length_ft), 0)


def turn_speed_mph(radius_ft: float) -> float:
    """The speed a vehicle can hold around a corner of this radius, modelled (see the constants).

    R=20 ft and R=15 ft are the difference an argument about a curb extension turns on, and
    neither number means anything to a reader. About 9.5 mph against 8.2 does.
    """
    return sqrt(15 * radius_ft * (TURN_SUPERELEVATION + TURN_SIDE_FRICTION))


def motor_lane_reach_ft(state: "DesignState", leg_name: str, reach: tuple) -> tuple[float, float]:
    """How far either side of the centerline a person is in front of MOTOR traffic.

    The kerb, unless a treatment put something else against it — a bike lane, its buffer, a
    parking lane, a hatched buffer. All of it is roadway a person walks across and none of it
    is ground a car drives on, so counting it as exposure would wrongly credit a proposal
    with nothing when it narrowed the part that matters.

    Asked of src/render/crosswalks.py:travel_lane_edge_ft (the same rule the STOP BAR
    already stops at), but NOT measured off the paint at the cross-section. Treatments are
    held back from the crossing by the daylight setback and CROSSWALK_CLEARANCE_FT, so the
    cross-section a person walks through is the leg's allocation, which is what treatments
    say.
    """
    from src.render.crosswalks import travel_lane_edge_ft

    out = []
    for side, reach_ft in (("left", reach[0]), ("right", reach[1])):
        try:
            edge_ft = travel_lane_edge_ft(state, leg_name, side)
        except Exception:
            edge_ft = None
        out.append(reach_ft if edge_ft is None else min(reach_ft, max(edge_ft, 0.0)))
    return tuple(out)


def crossing_stages_ft(leg, offset_ft: float, skew_deg: float, reach: tuple,
                        islands: list) -> tuple[float, ...]:
    """The unprotected walks a crossing is made of, in order across the road.

    One stage on an ordinary street; two where a refuge island splits it. Measured by cutting
    islands out of the crossing's own axis rather than subtracting the island's width from the
    total — an island 60 ft down the leg cuts nothing.
    """
    from src.render.crosswalks import crosswalk_axes

    (cx, cy), _u, (nx, ny), _cos = crosswalk_axes(leg, offset_ft, skew_deg)
    left_ft, right_ft = reach
    axis = LineString([(cx + nx * left_ft, cy + ny * left_ft),
                       (cx - nx * right_ft, cy - ny * right_ft)])
    shelter = [poly for poly in islands if poly is not None and not poly.is_empty]
    if not shelter:
        return (axis.length,)
    remaining = axis.difference(unary_union(shelter))
    if remaining.is_empty:
        return ()
    parts = [remaining] if isinstance(remaining, LineString) else list(remaining.geoms)
    ordered = sorted(parts, key=lambda part: axis.project(part.centroid))
    return tuple(part.length for part in ordered if part.length >= MIN_STAGE_FT)


@dataclass(frozen=True)
class Crossing:
    """One leg's crossing, as a person walks it."""
    leg: str
    stages_ft: tuple[float, ...]
    #: The CrosswalkOffset source string ("osm_survey" or "geometric_estimate").
    source: str
    #: The stages cut at the travel lane edge rather than at the kerb (motor_lane_reach_ft).
    motor_stages_ft: tuple[float, ...] = ()

    @property
    def is_surveyed(self) -> bool:
        return self.source.startswith("osm_survey")

    @property
    def is_staged(self) -> bool:
        return len(self.stages_ft) > 1

    @property
    def distance_ft(self) -> float:
        """Roadway crossed, excluding any refuge stood on."""
        return sum(self.stages_ft)

    @property
    def longest_stage_ft(self) -> float:
        return max(self.stages_ft, default=0.0)

    @property
    def motor_distance_ft(self) -> float:
        """Roadway in front of motor traffic - the whole crossing less the kerbside bands."""
        return sum(self.motor_stages_ft) if self.motor_stages_ft else self.distance_ft

    @property
    def longest_motor_stage_ft(self) -> float:
        return max(self.motor_stages_ft, default=self.longest_stage_ft)

    def exposure_s(self, speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> float:
        """Time in front of MOTOR traffic, worst stage.

        A refuge island is somewhere to stand, so the worst stage — not the sum — is the
        honest exposure for a multi-stage crossing. Measured across TRAVEL LANES, not curb
        to curb: a person in a bike lane or parking lane is not in front of a car. Bicycle
        traffic is a real conflict and is not counted here (hence MOTOR, not traffic).
        Hatching is paint, not a kerb — this is the exposure the design intends rather than
        one anything physically enforces.
        """
        return self.longest_motor_stage_ft / speed_ft_s

    def crossing_time_s(self, speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> float:
        """Time actually walking, all stages - not counting the wait on the refuge."""
        return self.distance_ft / speed_ft_s


def split_at_surveyed_end(geometry: LineString, leg, surveyed_length_ft: float | None) -> tuple:
    """(measured_ft, projected_ft) for a run, split where the SURVEYED leg ends.

    The frame scale carries legs out so a treatment runs the length of the drawn street — a
    drawing decision that must not become an arithmetic one: the part of a run past the
    length the site configured is measured separately and reported as projected.

    Split by STATION along the leg rather than by distance from the junction, because a kerb
    run is offset from the centerline and on a curved leg the two diverge.
    """
    if surveyed_length_ft is None or leg is None:
        return geometry.length, 0.0
    coords = list(geometry.coords)
    if len(coords) < 2:
        return geometry.length, 0.0
    mids = np.asarray([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in itertools.pairwise(coords)])
    stations, _offsets = station_offset_many(leg.centerline, mids)
    measured = projected = 0.0
    for (a, b), station in zip(itertools.pairwise(coords), stations):
        seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if station <= surveyed_length_ft:
            measured += seg
        else:
            projected += seg
    return measured, projected


@dataclass(frozen=True)
class ParkingRun:
    """One continuous run of marked stalls, as painted."""
    leg: str
    side: str
    stalls: int
    length_ft: float
    #: How much of this run lies past the length the site configured for its leg - drawn because
    #: the frame was widened, not because anybody surveyed that far. 0.0 at an unscaled frame.
    projected_ft: float = 0.0
    #: The stall length the treatment marks at. Carried rather than recovered as length/stalls,
    #: which is the AVERAGE and rounds a run's measured share to the wrong whole stall.
    stall_length_ft: float = 0.0

    @property
    def measured_ft(self) -> float:
        return self.length_ft - self.projected_ft

    @property
    def is_projected(self) -> bool:
        return self.projected_ft > 0.0


@dataclass(frozen=True)
class CornerTurn:
    corner: Corner
    radius_ft: float

    @property
    def turn_speed_mph(self) -> float:
        return turn_speed_mph(self.radius_ft)


@dataclass(frozen=True)
class SceneMetrics:
    """What one scenario achieves. Built from a resolved scene, never from the config."""
    crossings: tuple[Crossing, ...]
    parking: tuple[ParkingRun, ...]
    corners: tuple[CornerTurn, ...]

    @classmethod
    def of(cls, state: "DesignState", reaches: dict, offsets: dict, skews: dict, paint: list,
           marked=None, surveyed_leg_lengths: dict | None = None) -> "SceneMetrics":
        """Measure a design. Arguments are SceneGeometry's own fields - see its `metrics`.

        `marked` is the set of legs carrying a crossing; None measures every leg with a
        resolved offset — but every leg has one, so measuring by offset alone would report a
        crossing distance for a leg the drawing shows as unmarked.
        """
        from src.geometry.treatments import MarkedParking, RefugeIsland

        islands_by_leg: dict[str, list] = {}
        for island in state.treatments_of(RefugeIsland):
            islands_by_leg.setdefault(island.target.leg, []).append(island.polygon(state))

        crossings = []
        for leg_name, leg in state.legs.items():
            if leg_name not in reaches or leg_name not in offsets:
                continue
            if marked is not None and leg_name not in marked:
                continue
            offset = offsets[leg_name]
            skew = skews.get(leg_name, 0.0)
            islands = islands_by_leg.get(leg_name, [])
            crossings.append(Crossing(
                leg=leg_name, source=offset.source,
                stages_ft=crossing_stages_ft(leg, offset.offset_ft, skew,
                                              reaches[leg_name], islands),
                motor_stages_ft=crossing_stages_ft(
                    leg, offset.offset_ft, skew,
                    motor_lane_reach_ft(state, leg_name, reaches[leg_name]), islands)))

        runs = []
        for piece in paint:
            if piece.kind is not PARKING_EDGE_LINE:
                continue
            parking = state.treatment_for(MarkedParking, LegSide(piece.leg, piece.side))
            if parking is None:
                continue
            length_ft = piece.geometry.length
            _measured, projected = split_at_surveyed_end(
                piece.geometry, state.legs.get(piece.leg),
                (surveyed_leg_lengths or {}).get(piece.leg))
            runs.append(ParkingRun(leg=piece.leg, side=piece.side, length_ft=length_ft,
                                    projected_ft=projected,
                                    stall_length_ft=parking.stall_length_ft,
                                    stalls=stalls_in_run(length_ft, parking.stall_length_ft)))

        corners = []
        for key in sorted(state.corner_fillets):
            pieces = state.corner_fillets[key]
            # A corner that failed to solve records an "error"; a leg pair that is not a corner
            # at all (one street running through the junction) has no radius. Neither has a turn
            # to report, and reaching for a radius on either is what the plan view already
            # guards against when it draws the arcs.
            if "error" in pieces or pieces.get("radius_ft") is None:
                continue
            corners.append(CornerTurn(corner=Corner(*key), radius_ft=pieces["radius_ft"]))

        return cls(crossings=tuple(crossings), parking=tuple(runs), corners=tuple(corners))

    def crossing(self, leg: str) -> Crossing | None:
        return next((c for c in self.crossings if c.leg == leg), None)

    @property
    def total_stalls(self) -> int:
        return sum(run.stalls for run in self.parking)

    @property
    def measured_stalls(self) -> int:
        """Stalls on the length of leg the SITE configured - the number that does not move.

        Counted from each run's measured length rather than by scaling the total, because a run
        crossing the surveyed end is one run and the stalls in it are whole.
        """
        return sum(stalls_in_run(run.measured_ft, run.stall_length_ft)
                   for run in self.parking if run.stall_length_ft > 0)

    @property
    def projected_stalls(self) -> int:
        return self.total_stalls - self.measured_stalls


@dataclass(frozen=True)
class CrossingChange:
    """One leg's crossing, before against after."""
    leg: str
    before: Crossing | None
    after: Crossing | None

    @property
    def before_ft(self) -> float | None:
        return None if self.before is None else self.before.distance_ft

    @property
    def after_ft(self) -> float | None:
        return None if self.after is None else self.after.distance_ft

    @property
    def is_new(self) -> bool:
        """Marked by the proposal, absent today. "0 ft saved" would be false either way."""
        return self.before is None and self.after is not None

    @property
    def is_removed(self) -> bool:
        return self.after is None and self.before is not None

    @property
    def saved_ft(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.before.distance_ft - self.after.distance_ft

    def saved_s(self, speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.before.exposure_s(speed_ft_s) - self.after.exposure_s(speed_ft_s)

    @property
    def shown(self) -> Crossing:
        """Whichever side of the comparison exists, for labelling."""
        return self.after or self.before


@dataclass(frozen=True)
class Comparison:
    """A before/after pair, which is what the figure is about. The panel is deliberately short —
    a reader looking at two drawings needs to know what moved, not a reconstruction of every line."""
    before: SceneMetrics
    after: SceneMetrics
    speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S

    @classmethod
    def of(cls, before: SceneMetrics, after: SceneMetrics,
           speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> "Comparison":
        return cls(before=before, after=after, speed_ft_s=speed_ft_s)

    @property
    def crossings(self) -> tuple[CrossingChange, ...]:
        legs = [c.leg for c in self.before.crossings]
        legs += [c.leg for c in self.after.crossings if c.leg not in legs]
        return tuple(CrossingChange(leg=leg, before=self.before.crossing(leg),
                                     after=self.after.crossing(leg))
                     for leg in legs)

    def crossing(self, leg: str) -> CrossingChange | None:
        return next((c for c in self.crossings if c.leg == leg), None)

    @property
    def stalls_before(self) -> int:
        return self.before.total_stalls

    @property
    def stalls_after(self) -> int:
        return self.after.total_stalls

    @property
    def stalls_delta(self) -> int:
        return self.stalls_after - self.stalls_before

    def panel_text(self) -> str:
        """The summary block, as the plan view draws it — built here so what it says is
        testable without a figure and printable by a phase script."""
        lines = [f"WHAT CHANGES   (walking speed {self.speed_ft_s:.1f} ft/s)", ""]

        if self.crossings:
            lines.append("CROSSING DISTANCE, curb to curb")
        for change in self.crossings:
            label = leg_label(change.leg)
            note = "" if change.shown.is_surveyed else "  est. position"
            if change.is_new:
                lines.append(f"  {label:<22}     new  {change.after_ft:5.1f} ft{note}")
                continue
            if change.is_removed:
                lines.append(f"  {label:<22}  {change.before_ft:5.1f} -> removed{note}")
                continue
            staged = "  staged (refuge)" if change.after.is_staged else ""
            lines.append(f"  {label:<22}  {change.before_ft:5.1f} -> {change.after_ft:5.1f} ft"
                          f"  {change.after_ft - change.before_ft:+6.1f} ft{staged}{note}")

        measurable = [c for c in self.crossings if c.saved_ft is not None]
        if measurable:
            lines += ["", "TIME EXPOSED TO MOTOR TRAFFIC, worst stage"]
        for change in measurable:
            before_s = change.before.exposure_s(self.speed_ft_s)
            after_s = change.after.exposure_s(self.speed_ft_s)
            lines.append(f"  {leg_label(change.leg):<22}  {before_s:5.1f} -> {after_s:5.1f} s "
                          f"  {after_s - before_s:+6.1f} s")

        lines += ["", f"MARKED PARKING            {self.stalls_before} -> {self.stalls_after} "
                       f"stalls   {self.stalls_delta:+d}"]
        # A projected stall is one the frame scale drew, past the length of leg the site
        # configured. Said out loud rather than folded into the total, because otherwise the
        # same proposal reports 2 stalls at 1x and 8 at 2.5x and nothing on the drawing says
        # why - a number that moves with a camera setting, presented as a measurement. It is
        # not even the same kerb: at 1x those 2 are on broad_st_east inside its surveyed
        # length, and at 2.5x that leg's far kerb pinches too narrow to park, so the 8 are
        # broad_st_west's and every one of them is past the surveyed end.
        if self.after.projected_stalls or self.before.projected_stalls:
            lines.append(f"  of which surveyed       {self.before.measured_stalls} -> "
                          f"{self.after.measured_stalls}   the rest is projected past the "
                          f"surveyed leg")

        turns = {turn.corner: turn for turn in self.before.corners}
        changed = [turn for turn in self.after.corners
                   if turn.corner in turns and turn.radius_ft != turns[turn.corner].radius_ft]
        if changed:
            lines += ["", "TURN SPEED AT THE CORNER  (modelled, not measured)"]
        for turn in changed:
            was = turns[turn.corner]
            corner = f"{leg_label(turn.corner.leg_a)} x {leg_label(turn.corner.leg_b)}"
            lines.append(f"  {corner}")
            lines.append(f"  {'':<22}  {was.turn_speed_mph:5.1f} -> "
                          f"{turn.turn_speed_mph:5.1f} mph  "
                          f"   (R {was.radius_ft:.0f} -> {turn.radius_ft:.0f} ft)")
        return "\n".join(lines)
