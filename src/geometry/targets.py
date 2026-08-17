"""What a treatment is applied TO: a leg, one kerb of a leg, or a corner between two legs.

These were bare strings and bare tuples - `"east"`, `("east", "left")`, `("broad_st_east",
"greenwood_ave_north")` - keyed into twenty-three dicts on DesignState, and nothing distinguished
them. `state.bike_lanes[("east", "north")]` was a perfectly good expression that simply never
matched anything, and a treatment aimed at a leg the junction does not have wrote a key nobody
read: no error, no paint, no marking in either view.

So a target is a value with a type, it knows how to check that it exists in a design, and
DesignState.apply asks it to before applying anything. The three shapes are genuinely different -
a bike lane belongs to one kerb, a lane narrowing to a whole leg, an apron to the corner between
two legs - which is why this is three classes rather than one with optional fields.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    """Which kerb of a leg, in the leg's own frame: LEFT is the +offset side.

    A StrEnum, so `Side.LEFT == "left"` and it hashes like the string - the state dicts, the
    OSM tag keys (`parking:left`) and the traced kerb attributes (`leg.left_curb`) all keep
    working unchanged. What it adds is that `Side("north")` raises, and that the sign
    convention has one home: `1 if side == "left" else -1` was written out in ten places, and
    an invariant that forgot the sign passed anything on the right-hand side of a leg.
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

    `missing_from` returns why this target does not exist in a state, or None if it does. A
    reason rather than a bool so DesignState.apply can say what is wrong - "no leg
    'broad_st_norht'" with the available names beside it is one round trip; a silent no-op is
    several.
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

    The other three targets name something in `state.legs`, which is this junction's own four
    approaches - and for a treatment that moves a kerb or paints one, that is the right scope,
    because it is the only ground the design has measurements for.

    A MARKING POLICY IS NOT LIKE THAT. "Repaint every crosswalk continental" is a statement
    about the picture, and the picture contains Blackwell Avenue, Model Avenue and Seminary
    Avenue with six surveyed crossings between them. Applied per leg it reached four crossings
    out of ten, so a proposal captioned "all crosswalks continental" rendered two of them as the
    two parallel lines they are today - in the same frame, 260 ft apart, with nothing to say why.
    That is the same "the statute is about AN intersection, not THIS one" mistake
    src/geometry/cross_streets.py exists for, in the marking layer.

    Vacuously present: a frame always exists, so `missing_from` is always None. It is still a
    Target rather than a bare flag on DesignState so a frame-wide decision is recorded as a
    treatment like any other, and `state.treatments` stays a complete account of what was applied.

    WHAT IT DOES NOT DO is invent paint. A crossing nobody has marked stays unmarked whatever a
    policy says - see src/geometry/surveyed.py:crossing_style_in, which only ever RESTYLES a
    crossing that already carries markings. Painting a crosswalk where there is none is a new
    crossing, which MUTCD 3C.02(04) wants an engineering study for (STANDARDS.md section 2).
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
        # Coerces as well as validates: a scenario written with "left" gets the enum, and
        # anything that is not a side of a leg is refused here rather than becoming a dict key
        # that never matches.
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
    RIGHT kerb), so Corner("a", "b") and Corner("b", "a") are different corners of the
    junction. See src/geometry/model/corners.py:fillet_curb_corner.
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
