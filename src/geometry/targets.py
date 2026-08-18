"""What a treatment is applied TO: a leg, one kerb of a leg, or a corner between two legs.

These were bare strings and bare tuples keyed into twenty-three dicts on DesignState, and
nothing distinguished them. A target is a value with a type: it knows how to check that it
exists in a design, and DesignState.apply asks it to before applying anything. The three shapes
are genuinely different (a bike lane belongs to one kerb, a lane narrowing to a whole leg, an
apron to the corner between two legs), which is why this is three classes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    """Which kerb of a leg, in the leg's own frame: LEFT is the +offset side.

    A StrEnum, so `Side.LEFT == "left"` and it hashes like the string - state dicts, OSM tag
    keys and traced kerb attributes all work unchanged. What it adds is that `Side("north")`
    raises, and the sign convention has one home.
    """
    LEFT = "left"
    RIGHT = "right"

    @property
    def sign(self) -> float:
        """+1 on the left, -1 on the right - the sign of a lateral offset on this side."""
        return 1.0 if self is Side.LEFT else -1.0

    @property
    def other(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @property
    def curb_attr(self) -> str:
        """The Leg attribute holding this side's traced kerb."""
        return f"{self.value}_curb"


BOTH_SIDES: tuple[Side, Side] = (Side.LEFT, Side.RIGHT)


class Target(ABC):
    """Somewhere in a design a treatment can be applied.

    `missing_from` returns why this target does not exist in a state, or None if it does.
    A reason rather than a bool so DesignState.apply can say what is wrong.
    """

    @abstractmethod
    def missing_from(self, state) -> str | None:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...


@dataclass(frozen=True, order=True)
class Everywhere(Target):
    """THE WHOLE DRAWN FRAME, including the junctions this site does not model.

    A MARKING POLICY IS NOT like a kerbside treatment. "Repaint every crosswalk continental" is
    a statement about the picture, and the picture contains cross streets with surveyed crossings.
    Applied per leg it reached four crossings out of ten, rendering two of them as the parallel
    lines they are today - the same "the statute is about AN intersection, not THIS one" mistake
    src/geometry/cross_streets.py exists for.

    Vacuously present: a frame always exists, so `missing_from` is always None. It is still a
    Target so a frame-wide decision is recorded as a treatment, and `state.treatments` stays a
    complete account.

    WHAT IT DOES NOT DO is invent paint. A crossing nobody has marked stays unmarked - see
    src/geometry/surveyed.py:crossing_style_in, which only ever RESTYLES a crossing that already
    carries markings.
    """

    def missing_from(self, state) -> str | None:
        return None

    def __str__(self) -> str:
        return "everywhere"


@dataclass(frozen=True, order=True)
class LegTarget(Target):
    """A whole leg: both kerbs, or the roadway between them."""
    leg: str

    def missing_from(self, state) -> str | None:
        if self.leg not in state.legs:
            return f"no leg {self.leg!r} at this junction - it has {sorted(state.legs)}"
        return None

    def __str__(self) -> str:
        return self.leg


@dataclass(frozen=True, order=True)
class LegSide(Target):
    """One kerb of one leg - the target of every kerbside treatment."""
    leg: str
    side: Side

    def __post_init__(self):
        # Coerces as well as validates: "left" gets the enum, and anything that is not a side
        # is refused here rather than becoming a dict key that never matches.
        object.__setattr__(self, "side", Side(self.side))

    def missing_from(self, state) -> str | None:
        return LegTarget(self.leg).missing_from(state)

    @property
    def leg_target(self) -> LegTarget:
        return LegTarget(self.leg)

    @property
    def key(self) -> tuple[str, str]:
        """The (leg, side) tuple the state dicts are keyed by."""
        return (self.leg, str(self.side))

    def __str__(self) -> str:
        return f"{self.leg} {self.side}"


@dataclass(frozen=True, order=True)
class Corner(Target):
    """The corner between two legs, as build_corner_fillets keys it.

    Order is load-bearing and not symmetric: a corner is always (leg_a's LEFT kerb, leg_b's
    RIGHT kerb), so Corner("a", "b") and Corner("b", "a") are different corners.
    """
    leg_a: str
    leg_b: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.leg_a, self.leg_b)

    def missing_from(self, state) -> str | None:
        if self.key not in state.corner_fillets:
            return (f"no corner {self.key} at this junction - it has "
                    f"{sorted(state.corner_fillets)}. A corner is (leg_a's left kerb, leg_b's "
                    f"right kerb), so the order matters; try find_corner()")
        return None

    def __str__(self) -> str:
        return f"{self.leg_a} x {self.leg_b}"


@dataclass(frozen=True, order=True)
class AcrossTheJunction(Target):
    """The GAP between one facility's two kerbs, on either side of the junction box.

    Every other target names ground inside one leg's frame. A LANE EXTENSION is the piece of
    a continuous facility that lies where neither leg reaches - across the mouth of the street
    between them - so it belongs to the pair or to nothing.

    BOTH SIDES ARE SPELT OUT rather than implied. The two kerbs are the SAME physical kerb
    seen from two approaches; which of left/right that is on each comes out of model.side_facing
    per leg.
    """
    leg_a: str
    side_a: Side
    leg_b: str
    side_b: Side

    def __post_init__(self):
        object.__setattr__(self, "side_a", Side(self.side_a))
        object.__setattr__(self, "side_b", Side(self.side_b))
        if (self.leg_a, self.side_a) == (self.leg_b, self.side_b):
            raise ValueError(
                f"An extension spans the gap BETWEEN two kerbs, and {self.leg_a} "
                f"{self.side_a} is one kerb given twice - there is no gap for it to cross.")

    @property
    def ends(self) -> tuple[tuple[str, str], tuple[str, str]]:
        """The two (leg, side) keys, in the order the extension runs."""
        return ((self.leg_a, str(self.side_a)), (self.leg_b, str(self.side_b)))

    def missing_from(self, state) -> str | None:
        for leg in (self.leg_a, self.leg_b):
            missing = LegTarget(leg).missing_from(state)
            if missing:
                return missing
        return None

    def __str__(self) -> str:
        return f"{self.leg_a} {self.side_a} -> {self.leg_b} {self.side_b}"
