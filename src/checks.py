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
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from src.geometry.model import station_offset_many

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
# A painted crosswalk should lie in the roadway it crosses. Some overhang past the curb is
# normal where it ties into the ramp.
MIN_CROSSWALK_IN_PAVEMENT = 0.55
# A stop bar covers the entering half only. This much of it may cross the centerline before
# it is genuinely painted across opposing lanes.
MAX_STOP_BAR_OPPOSING_FRACTION = 0.15

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


# ---------------------------------------------------------------------------
# Individual checks. Each returns a list of Violations and never raises.
# ---------------------------------------------------------------------------

def check_furniture_off_roadway(props: list[dict], pavement) -> list[Violation]:
    """Nothing that belongs on the footway may sit in the carriageway.

    Signs, signal poles, pushbuttons, beacons, streetlights, hydrants and tactile pads are
    all footway furniture. A pad drawn in the road is the worst case - it is the render
    asserting something false about an accessibility feature - but a stop sign in the
    middle of the street is just as wrong and had no check at all before.
    """
    from src.render.props import _pad_polygon  # local: props imports geometry, avoid a cycle

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
            pad = _pad_polygon(*position, prop.get("heading_deg", 0.0))
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


def check_pads_against_a_curb(props: list[dict], legs: dict, corner_fillets: dict) -> list[Violation]:
    """A tactile pad marks a kerb ramp, so it has to be at a kerb.

    Off the roadway is necessary but not sufficient: a pad nudged clear of a too-wide
    pavement can end up out in a front garden, which reads as fine in plan and absurd in 3D.
    """
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


def check_curbs_clear_of_junction(legs: dict) -> list[Violation]:
    """No leg's curb may run back through the intersection.

    A leg's curb line starts at that leg's cross-section and goes outward. When one runs
    backwards past the junction it draws curb straight across the middle of the
    intersection - marking a kerb where there is open roadway - and it crosses the opposite
    leg's curb, which is what makes the pavement ring self-intersect. Measured in the leg's
    own frame, so it is the same signed station the curb was built from.
    """
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


def check_curbs_do_not_cross(legs: dict) -> list[Violation]:
    """A leg's two curb lines are the two sides of one street: they never meet.

    They crossed when a curb was extrapolated out of a corner return's flare, which closed
    the roadway to zero width and then opened it inside out.
    """
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


def check_pavement_ring(pavement) -> list[Violation]:
    """The pavement must be one simple polygon - no bowties, no pinches."""
    if pavement is None or pavement.is_empty:
        return [Violation("pavement_ring", "no pavement polygon was built for this junction")]
    if not pavement.is_valid:
        from shapely.validation import explain_validity
        return [Violation("pavement_ring", f"pavement polygon is invalid: {explain_validity(pavement)}")]
    return []


def check_crosswalks_cross_the_roadway(bands: dict, pavement) -> list[Violation]:
    """A painted crosswalk lies across the roadway, touching the curb at both ends.

    Catches the two failures seen here: a band drawn out in the middle of the carriageway
    parallel to traffic (it was inheriting a leg offset from the wrong frame), and a band
    sitting almost entirely outside the pavement.
    """
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


def check_stop_bars_on_entering_half(bars: dict, legs: dict) -> list[Violation]:
    """A driver stops in their own lanes, never across the opposing ones.

    The bar must stay on one side of its leg's centerline. It was previously drawn full
    width, across both directions of travel.
    """
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

def check_scene(model, state, props: list[dict], pavement, crosswalk_bands: dict | None = None,
                 stop_bars: dict | None = None) -> list[Violation]:
    """Every invariant, all violations, no raising. See assert_scene_valid to fail on them."""
    return (
        check_furniture_off_roadway(props, pavement)
        + check_pads_against_a_curb(props, state.legs, state.corner_fillets)
        + check_curbs_clear_of_junction(state.legs)
        + check_curbs_do_not_cross(state.legs)
        + check_pavement_ring(pavement)
        + check_crosswalks_cross_the_roadway(crosswalk_bands or {}, pavement)
        + check_stop_bars_on_entering_half(stop_bars or {}, state.legs)
    )


def assert_scene_valid(model, state, props: list[dict], pavement, crosswalk_bands: dict | None = None,
                        stop_bars: dict | None = None, scenario: str = "") -> None:
    """Raise SceneInvariantError listing EVERY violation in this scene, or return quietly."""
    violations = check_scene(model, state, props, pavement, crosswalk_bands, stop_bars)
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
