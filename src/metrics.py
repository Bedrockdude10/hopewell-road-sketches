"""What a design ACHIEVES, measured off the geometry it actually drew.

Every dimension this project puts on a drawing is an INPUT - the 55.5 ft street, the 8 ft
stall, R=20 at the corner. Those say what was built. None of them says what it accomplished,
and "the crossing is 14 ft shorter and a person is in the road for 4 fewer seconds" is the
sentence a proposal is argued over. So the outcome numbers live here, and they are computed
from the same resolved scene both renderers draw from (src/render/scene.py:SceneGeometry) -
never from the config that scene was built out of.

That distinction is the whole reason this is a module rather than three f-strings in the plan
view. A crossing's length is NOT the leg's configured width: crosswalk_reach_to_curbs_ft
measures out to the traced kerbs, two passes, with adjoining crossings cut out of the roadway
each one is allowed to occupy, and the answer is asymmetric (12 ft one way, 20 the other on a
30 ft street). A curb extension changes it. Re-deriving "crossing distance" from
`leg.curb_to_curb_ft` would produce a number that agrees with the drawing on a straight
symmetric leg, disagrees quietly everywhere else, and keeps reporting the old width after a
treatment moves the kerb - the exact failure this codebase keeps designing against.

The same applies to parking. Stalls are counted off the PARKING_EDGE_LINE pieces the paint
builder emitted, one run at a time, because that is what is marked on the road: a hydrant or
a driveway splits a kerb into two runs, and the daylight zone at a corner shortens the run
rather than the leg. Counting stalls from the leg's length would count spaces that the paint
itself says are not there.

Nothing here decides anything. Every value comes from geometry some other module resolved;
this module only measures it and says what changed.
"""
from dataclasses import dataclass
from math import sqrt

from shapely.geometry import LineString
from shapely.ops import unary_union

from src.geometry.markings import PARKING_EDGE_LINE
from src.geometry.targets import Corner, LegSide

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


def crossing_stages_ft(leg, offset_ft: float, skew_deg: float, reach: tuple,
                        islands: list) -> tuple[float, ...]:
    """The unprotected walks a crossing is made of, in order across the road.

    One stage on an ordinary street; two where a refuge island splits it. Measured by cutting
    the islands out of the crossing's own axis - the same axis crosswalk_band_ft builds the
    painted band on (src/render/crosswalks.py:crosswalk_axes) - rather than by subtracting the
    island's width from the total. The difference is that an island 60 ft down the leg cuts
    nothing, which is what stops a refuge anywhere on a leg being credited to every crossing
    on it.
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
    #: The CrosswalkOffset source string - "osm_survey" or "geometric_estimate", possibly with
    #: a scenario-shift suffix. Carried so a reported distance always says whether the position
    #: it was measured at is surveyed, on the same terms the drawing says it.
    source: str

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

    def exposure_s(self, speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> float:
        """Time in front of moving traffic, worst stage.

        The longest stage rather than the sum: a refuge island is somewhere to stand, so a
        staged crossing exposes a person one stage at a time and the honest number for a
        two-stage crossing is the worse of the two. Summing them would credit the island with
        nothing at all, which is the opposite of what it does.
        """
        return self.longest_stage_ft / speed_ft_s

    def crossing_time_s(self, speed_ft_s: float = MUTCD_WALKING_SPEED_FT_S) -> float:
        """Time actually walking, all stages - not counting the wait on the refuge."""
        return self.distance_ft / speed_ft_s


@dataclass(frozen=True)
class ParkingRun:
    """One continuous run of marked stalls, as painted."""
    leg: str
    side: str
    stalls: int
    length_ft: float


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
    def of(cls, state, reaches: dict, offsets: dict, skews: dict, paint: list,
           marked=None) -> "SceneMetrics":
        """Measure a design. Arguments are SceneGeometry's own fields - see its `metrics`.

        `marked` is the set of legs carrying a crossing; None measures every leg with a
        resolved offset. Every leg has an offset (a proposal may mark a leg that has nothing
        today), so measuring by offset alone would report a crossing distance for a leg the
        drawing shows as an unmarked outline.
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
            crossings.append(Crossing(
                leg=leg_name, source=offset.source,
                stages_ft=crossing_stages_ft(leg, offset.offset_ft, skews.get(leg_name, 0.0),
                                              reaches[leg_name], islands_by_leg.get(leg_name, []))))

        runs = []
        for piece in paint:
            if piece.kind is not PARKING_EDGE_LINE:
                continue
            parking = state.treatment_for(MarkedParking, LegSide(piece.leg, piece.side))
            if parking is None:
                continue
            length_ft = piece.geometry.length
            runs.append(ParkingRun(leg=piece.leg, side=piece.side, length_ft=length_ft,
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
    """A before/after pair, which is what the figure is about.

    The panel this produces is deliberately short. The plan view's own legend runs to forty
    entries because it is a reconstruction anyone should be able to check line by line; a
    reader looking at two drawings needs to know what moved.
    """
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
        """The summary block, as the plan view draws it.

        Built as text here rather than laid out in the renderer so what it says is testable
        without a figure - and so the same summary can be printed by a phase script.
        """
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
            lines += ["", "TIME EXPOSED TO TRAFFIC, worst stage"]
        for change in measurable:
            before_s = change.before.exposure_s(self.speed_ft_s)
            after_s = change.after.exposure_s(self.speed_ft_s)
            lines.append(f"  {leg_label(change.leg):<22}  {before_s:5.1f} -> {after_s:5.1f} s "
                          f"  {after_s - before_s:+6.1f} s")

        lines += ["", f"MARKED PARKING            {self.stalls_before} -> {self.stalls_after} "
                       f"stalls   {self.stalls_delta:+d}"]

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
