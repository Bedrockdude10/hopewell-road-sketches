"""Treatments that reshape a CORNER: its radius, the curb extension built into it, the
hatching and mountable apron that go with one, and the daylighting that protects the sight line.

Grouped because they all read and rewrite the same corner fillet - AddCurbExtension and
ProtectDaylightZone in particular have to agree about it, and did not when they were 800 lines
apart."""
from dataclasses import dataclass
from typing import ClassVar


from src.geometry.targets import LegSide
from src.geometry.model import (BULBOUT_TAPER_RATE, curb_extension_line,
                                fillet_curb_corner)
from src.geometry.treatments.base import (CORNER_APRON_DEFAULT_EXTENT_FT,
                                          CORNER_HATCHING_DEFAULT_DEPTH_FT,
                                          LANE_WIDTH_SLACK_FT, TARGET_LANE_WIDTH_FT,
                                          CornerApron, Treatment)
from src.geometry.treatments.state import DesignState



def find_corner(state: DesignState, leg_a: str, leg_b: str) -> tuple[str, str]:
    """Look up the (name_a, name_b) key in state.corner_fillets for the corner
    where leg_a and leg_b meet, regardless of which order build_corner_fillets
    happened to store it in (it sorts by compass bearing, not by call-site
    convenience) - corners are identified by which two legs meet there, not by
    tuple order."""
    wanted = {leg_a, leg_b}
    for corner in state.corner_fillets:
        if set(corner) == wanted:
            return corner
    raise KeyError(f"No corner between {leg_a!r} and {leg_b!r} in this state.")


@dataclass(frozen=True)
class SetCornerRadius(Treatment):
    """Re-cut one corner's fillet at a different radius. Does NOT shorten a crossing here.

    This was called `bump_out` and its docstring claimed "the curb physically extends into the
    corner". It does not. It solves a new arc between two curb lines and leaves both of them
    where they were, so all it moves is the corner itself. Measured on
    broad_st_east x greenwood_ave_north, 29.2 -> 15.0 ft:

        arc length     19.48 -> 3.51 ft      the arc is genuinely re-cut
        trimmed_a     156.19 -> 164.19 ft    the curb only runs on to the new tangent point
        pavement area  23,989.7 -> 23,989.5 sq ft         0.2 sq ft of 24,000
        crossing spans unchanged to 0.00 ft on all four legs

    Nothing was wrong with the arithmetic; the claim was wrong. The crossings at these
    junctions sit 21-42 ft out, past the corner, so a radius change never reaches them. What
    DOES shorten a crossing is AddCurbExtension, which moves the kerb line laterally.

    Still a real operation, and the one a curb extension needs: the tightened face a curb
    extension presents to a passenger car IS a corner radius. `source` is recorded so the plan
    view can say whether a corner's radius was traced or chosen.
    """
    radius_ft: float = 0.0
    source: str = "designed"

    def __post_init__(self):
        if self.radius_ft <= 0:
            raise ValueError(f"A corner radius has to be positive; got radius_ft={self.radius_ft}.")

    def describe(self) -> str:
        return f"SetCornerRadius({self.target.key}, radius_ft={self.radius_ft})"

    def apply_to(self, state: "DesignState", model=None) -> None:
        _rebuild_corner(state, self.target.key, self.radius_ft, self.source)


def _rebuild_corner(state: DesignState, corner: tuple[str, str], radius_ft: float,
                     source: str) -> None:
    """Solve `corner`'s fillet afresh off whatever its two curb lines currently are.

    Mutates `state` - callers have already cloned. Separate from set_corner_radius because
    add_curb_extension needs it too: an extension moves a kerb, and the corner that kerb feeds
    has to be re-cut against the moved line or the pavement ring still follows the old one.
    """
    leg_a, leg_b = corner
    if leg_a not in state.legs or leg_b not in state.legs:
        raise KeyError(f"Corner {corner} references a leg not present in this state.")
    trimmed_a, arc, trimmed_b = fillet_curb_corner(
        state.legs[leg_a].left_curb, state.legs[leg_b].right_curb, radius_ft)
    state.corner_fillets[corner] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b,
                                     "radius_ft": radius_ft, "source": source}


@dataclass(frozen=True)
class CornerHatching(Treatment):
    """Paint-only diagonal hatching in a corner's gutter zone: a visual
    narrowing cue with zero curb/fillet geometry change - the paint-only
    alternative to a real curb extension at the same corner."""
    # Last: a corner treatment is cut around every kerbside zone that reaches the corner.
    paint_group: ClassVar[int] = 90
    depth_ft: float = CORNER_HATCHING_DEFAULT_DEPTH_FT

    def __post_init__(self):
        if self.depth_ft <= 0:
            raise ValueError(f"Corner hatching needs a depth; got depth_ft={self.depth_ft}.")

    def describe(self) -> str:
        return f"CornerHatching({self.target.key}, depth_ft={self.depth_ft})"

    def paint(self, ctx) -> None:
        from src.geometry.model import corner_overlay_polygon
        from src.geometry.markings import CORNER_HATCH_FILL

        fillet = ctx.state.corner_fillets[self.target.key]
        if "error" in fillet:
            return          # a corner whose fillet failed has no gutter zone to hatch
        # No leg or side: a corner treatment spans the corner between two legs and belongs to
        # neither, which is why the kerb checks skip a piece with no side rather than guessing.
        ctx.add(CORNER_HATCH_FILL, corner_overlay_polygon(fillet, ctx.center_ft, self.depth_ft))


@dataclass(frozen=True)
class MountableApron(Treatment):
    """Mountable apron: a textured (not painted-line) surface treatment at a
    corner, flush with the existing pavement grade - visually/optically
    narrows the corner for pedestrians while remaining fully drivable (e.g. by
    a fire apparatus's rear wheels during a wide turn) since no curb or
    elevation change is introduced. Same footprint as CornerHatching, a
    different real-world treatment for corners where a hard bump-out isn't an
    option (see fire_apparatus_constraint in a proposal's spec).

    A FIXED DEPTH inward from the corner arc. Where the apron exists to preserve a large
    vehicle's swept path around a tightened corner, its depth is not free - it has to reach the
    radius that vehicle needs - so AddCurbExtension records CornerApron(swept_radius_ft=...)
    instead and the annulus is built from the two radii. See CornerApron.
    """
    paint_group: ClassVar[int] = 0
    extent_ft: float = CORNER_APRON_DEFAULT_EXTENT_FT

    def describe(self) -> str:
        return f"MountableApron({self.target.key}, extent_ft={self.extent_ft})"

    @property
    def apron(self) -> CornerApron:
        """CornerApron validates the depth/radius exclusivity itself."""
        return CornerApron(depth_ft=self.extent_ft)

    def apron_corner(self, state) -> tuple[str, str] | None:
        return self.target.key

    def paint(self, ctx) -> None:
        """The apron surface, laid in the SURFACE pass so every marking is cut around it.

        Its own apron, from its own fields. There was a state.corner_aprons holding one entry per
        corner, and reading from that would have let two treatments which each asked for an apron
        there paint one apron between them - a corner with two aprons specified is a design error,
        and painting both is what makes MarkingsDoNotCollide say so.
        """
        from src.geometry.markings import APRON
        from src.geometry.paint import apron_polygon

        corner = self.apron_corner(ctx.state)
        if corner is None or "error" in ctx.state.corner_fillets[corner]:
            return
        ctx.add_surface(APRON, apron_polygon(ctx.state, corner, self.apron, ctx.center_ft))


# How wide a face a tightened corner presents to a passenger car. The design figure for the
# bulb-outs at Broad & Greenwood: a 15 ft radius is a corner a car has to slow for, and the
# apron behind it (see CornerApron) hands the larger radius back to a bus or a truck. A design
# choice, not a measurement - which is exactly why the apron's own radius is measured.
CURB_EXTENSION_FACE_RADIUS_FT = 15.0


@dataclass(frozen=True)
class AddCurbExtension(Treatment):
    """A real curb extension: move this kerb `extension_ft` into the roadway and taper it back.

    This is the treatment SetCornerRadius was mistaken for. It changes the KERB LINE, so
    everything downstream that measures against the kerb follows it without being told:
    the crossing gets shorter (src/render/crosswalks.py:crosswalk_reach_to_curbs_ft walks out
    to the real kerb), the pavement polygon loses the corner, the kerbside paint rebuilds
    against the new edge, and the invariants check the geometry that results.

    HOW LONG. The face runs from the junction to `crossing_ft` plus half a crossing plus the
    10 ft R.S. 39:4-138(e) setback that the extension itself buys - i.e. it covers exactly the
    kerb where parking is prohibited once this is built - then tapers back over
    `extension_ft * BULBOUT_TAPER_RATE`. Nothing about the length is chosen to look right; it
    is the statutory zone plus a stated taper rate.

    HOW FAR. Bounded by the travel lane that has to survive: an extension deeper than the
    leg's spare width beside a TARGET_LANE_WIDTH_FT lane is refused rather than clamped,
    because silently building a shallower bulb-out than the caller asked for is how a drawing
    stops matching its own description. At Broad & Greenwood that permits the 8 ft asked for on
    both Broad legs (15.0 and 16.8 ft spare per side) and refuses it on Greenwood (2.3 and
    4.6 ft) - which is the finding, not an obstacle: Greenwood cannot hold a bulb-out and two
    11 ft lanes at once.

    WHAT IT COSTS IN PARKING. Nothing, at Broad & Greenwood. Schedule I of the borough code
    prohibits parking 100 ft each way on both sides of both Broad legs, and the whole footprint
    - face plus taper - fits inside that, so the extension occupies kerb that is already
    legally not-parking. A curb extension normally trades spaces for safety; here it does not,
    and that is the strongest thing that can be said for it.

    `swept_radius_ft` is the corner's OWN measured radius, and passing it lays a mountable
    apron over the annulus between that and `face_radius_ft` so a bus keeps the path it has
    today. On CR 518, a rural arterial carrying buses and trucks, that is not optional.
    """
    # Laid in the surface pass: built ground, and every marking is cut around it.
    paint_group: ClassVar[int] = 0
    extension_ft: float = 0.0
    crossing_ft: float = 0.0
    swept_radius_ft: float | None = None
    face_radius_ft: float = CURB_EXTENSION_FACE_RADIUS_FT
    taper_ft: float | None = None

    def __post_init__(self):
        if self.extension_ft <= 0:
            raise ValueError(f"An extension has to move the kerb; got "
                             f"extension_ft={self.extension_ft}.")
        if self.face_radius_ft <= 0:
            raise ValueError(f"The face is a corner radius; got {self.face_radius_ft}.")
        if self.swept_radius_ft is not None and self.swept_radius_ft <= self.face_radius_ft:
            raise ValueError(
                f"The swept radius ({self.swept_radius_ft} ft) is the corner a bus keeps via the "
                f"apron, so it has to be LARGER than the {self.face_radius_ft} ft face a car "
                f"sees - an annulus between them is what the apron is (see CornerApron).")

    @property
    def resolved_taper_ft(self) -> float:
        """How long the return to the real kerb is: the stated rate, unless one was given."""
        return self.extension_ft * BULBOUT_TAPER_RATE if self.taper_ft is None else self.taper_ft

    @property
    def full_ft(self) -> float:
        """The station the straight face runs to.

        Nothing about the length is chosen to look right: it is the crossing, plus half a
        crossing's depth, plus the 10 ft R.S. 39:4-138(e) setback the extension itself buys - so
        the face covers exactly the kerb where parking is prohibited once this is built.

        Local imports for the usual cycles (src/render/crosswalks.py imports DesignState from
        here, src/geometry/daylighting.py reads CURB_EXTENSION_DEVICES from here). Both figures
        are single-sourced there and must not be copied, since the whole length is measured off
        them.
        """
        from src.geometry.daylighting import CROSSWALK_SETBACK_WITH_BULBOUT_FT
        from src.render.crosswalks import CROSSWALK_DEPTH_FT

        return self.crossing_ft + CROSSWALK_DEPTH_FT / 2 + CROSSWALK_SETBACK_WITH_BULBOUT_FT

    @property
    def footprint_ft(self) -> float:
        """How much kerb the extension occupies end to end - what has to fit inside the length
        the parking ordinance already prohibits, if it is to cost no spaces. At Broad & Greenwood
        that is 74 ft against the 100 ft Schedule I already bans, which is the whole argument
        that this bulb-out removes no parking space; tests/test_curb_extensions.py pins it."""
        return self.full_ft + self.resolved_taper_ft

    @property
    def apron(self) -> CornerApron | None:
        """The annulus a bus keeps, or None where no swept radius was measured.

        A fixed depth cannot support the claim this apron exists to make - that the swept path
        survives the tightened corner - because nothing ties a depth to the radius a vehicle
        needs. See CornerApron.
        """
        if self.swept_radius_ft is None:
            return None
        return CornerApron(swept_radius_ft=self.swept_radius_ft,
                            face_radius_ft=self.face_radius_ft)

    def apron_corner(self, state) -> tuple[str, str] | None:
        """The corner this moved kerb feeds - not this treatment's own target.

        build_corner_fillets pairs leg A's LEFT curb with leg B's RIGHT, so which corner a kerb
        belongs to depends on the side. This is why the apron pass is ordered by corner rather
        than by target: a curb extension is aimed at a leg-side and lays ground at a corner.
        """
        return _corner_fed_by(state, self.target.leg, str(self.target.side))

    def paint(self, ctx) -> None:
        """The swept-path apron, in the SURFACE pass so every marking is cut around it.

        Its own apron rather than a per-corner lookup, for the reason MountableApron.paint
        gives: one entry per corner collapses two treatments that each asked for an apron there,
        and a corner with two aprons specified is a design error the collision invariant reports.
        """
        from src.geometry.markings import APRON
        from src.geometry.paint import apron_polygon

        apron = self.apron
        corner = self.apron_corner(ctx.state)
        if apron is None or corner is None or "error" in ctx.state.corner_fillets[corner]:
            return
        ctx.add_surface(APRON, apron_polygon(ctx.state, corner, apron, ctx.center_ft))

    def describe(self) -> str:
        return f"AddCurbExtension({self.target.leg}, {self.target.side}): "

    def apply_to(self, state: "DesignState", model=None) -> str:
        leg_name, side = self.target.leg, str(self.target.side)
        leg = state.legs[leg_name]
        if leg.curb_to_curb_ft is None:
            raise ValueError(f"Leg {leg_name!r} has no width - nothing to measure an extension from.")

        spare_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        if self.extension_ft > spare_ft + LANE_WIDTH_SLACK_FT:
            raise ValueError(
                f"A {self.extension_ft:.1f} ft curb extension on {leg_name} {side} would leave a "
                f"{leg.curb_to_curb_ft / 2 - self.extension_ft:.1f} ft travel lane, under the "
                f"{TARGET_LANE_WIDTH_FT:.0f} ft target. That leg is {leg.curb_to_curb_ft:.1f} ft "
                f"curb to curb, so it has {spare_ft:.1f} ft per side to give.")

        taper_ft, full_ft = self.resolved_taper_ft, self.full_ft
        built = curb_extension_line(leg, side, self.extension_ft, full_ft, taper_ft)
        if built is None:
            raise ValueError(
                f"{leg_name} {side} has no traced kerb to extend - a curb extension is measured "
                f"from the kerb that is there, and nothing is mapped on that side.")

        # THE one thing this treatment writes onto the design, and the reason it is the one
        # treatment that still has a body here: it moves a kerb. Everything downstream that
        # measures against the kerb then follows without being told.
        setattr(state.legs[leg_name], f"{side}_curb", built)
        # The corner this kerb feeds has to be re-cut against the line that moved, or the pavement
        # ring keeps following the kerb that is no longer there. build_corner_fillets pairs leg A's
        # LEFT curb with leg B's RIGHT, so which corner that is depends on the side.
        corner = self.apron_corner(state)
        if corner is not None:
            _rebuild_corner(state, corner, self.face_radius_ft, "curb_extension")
        return (f"kerb moved {self.extension_ft:.1f} ft into the roadway to station "
                f"{full_ft:.0f} ft, tapering back over {taper_ft:.0f} ft; "
                f"{self.face_radius_ft:.0f} ft face"
                + (f" with a mountable apron out to the corner's measured "
                   f"{self.swept_radius_ft:.1f} ft" if self.swept_radius_ft is not None else "")
                + f". Leaves a {leg.curb_to_curb_ft / 2 - self.extension_ft:.1f} ft travel lane.")


def _corner_fed_by(state: DesignState, leg_name: str, side: str) -> tuple[str, str] | None:
    """The corner whose fillet is built from this (leg, side)'s curb line, or None.

    build_corner_fillets' contract: a corner keyed (A, B) is bounded by A's LEFT curb and B's
    RIGHT curb. So a left side feeds the corner it is first in, a right side the corner it is
    second in - and each side feeds exactly one.
    """
    wanted = 0 if side == "left" else 1
    for corner in state.corner_fillets:
        if corner[wanted] == leg_name:
            return corner
    return None


def bulb_out_corner_pair(state: DesignState, leg_name: str, extension_ft: float,
                          crossing_ft: float, sides: tuple = ("left", "right")) -> DesignState:
    """Curb extensions on both kerbs of one leg, each corner's apron out to its OWN traced radius.

    The apron radius is READ FROM THE BASELINE FILLET rather than passed in, which is the point:
    the four corners at Broad & Greenwood are traced at 29.2, 24.6, 29.0 and 22.9 ft, and a
    scenario repeating those as literals would keep whatever they were the day it was written.
    Re-tracing a kerb in OSM now flows through to the apron by itself.
    """
    for side in sides:
        corner = _corner_fed_by(state, leg_name, side)
        swept_radius_ft = None if corner is None else state.corner_fillets[corner].get("radius_ft")
        state = state.apply(
            AddCurbExtension(LegSide(leg_name, side), extension_ft=extension_ft,
                              crossing_ft=crossing_ft, swept_radius_ft=swept_radius_ft),
            ProtectDaylightZone(LegSide(leg_name, side), kind="curb_extension"))
    return state


# Spacing by device. A flex-post line reads as a delineator at 8 ft. A curb extension has no
# spacing - it is one continuous kerb, not a row of objects.
DAYLIGHT_DEVICE_SPACING_FT = {"bollards": 8.0, "curb_extension": 0.0}
# Devices drawn as a row of physical objects standing in the zone. A curb extension is not one:
# it is built ground, already drawn as the kerb itself (add_curb_extension moves the curb line),
# so src/render/props.py must not also stand posts along it.
DAYLIGHT_DEVICES_AS_POSTS = frozenset({"bollards"})
# Which devices the statute's "curb extension or bulbout has been constructed" clause covers,
# cutting the setback in R.S. 39:4-138(e) from 25 ft to 10 ft.
#
# `curb_extension` is in it because add_curb_extension builds the thing the clause names: the
# kerb line moves and the pavement polygon loses the corner, so the parking lane is physically
# out of the sight line rather than painted out of it. A flex-post delineator is NOT - it bends
# flat under a tyre - and planters were listed here once and are not any more, because the
# argument that a row of them occupies the corner the way a built bulbout does was never the
# Borough's to concede.
CURB_EXTENSION_DEVICES: frozenset = frozenset({"curb_extension"})
VALID_DAYLIGHT_DEVICES = ("bollards", "curb_extension")


@dataclass(frozen=True)
class ProtectDaylightZone(Treatment):
    """Stand physical objects in the daylight zone so it is not merely painted.

    An unmarked statutory setback gets parked in; a painted one gets parked in less. Objects
    in it get parked in not at all, and that is the difference between a drawing of the law
    and a street that enforces it.

    `kind` can matter legally, not just visually: R.S. 39:4-138(e) cuts the 25 ft setback to
    10 ft "if a curb extension or bulbout has been constructed", so a device that counts as one
    buys back kerb for parking. `curb_extension` does and `bollards` does not - a flex-post
    bends flat under a tyre. See src/geometry/daylighting.py for where that is applied, and
    CURB_EXTENSION_DEVICES for why the set has one member and not two.

    Declaring `curb_extension` here is what makes the statutory reduction apply; it does not
    BUILD anything. AddCurbExtension moves the kerb. The two go together, and its caller is
    expected to declare the device as well - which is why the note below says which of the two
    setbacks now governs.
    """
    kind: str = "bollards"
    spacing_ft: float | None = None

    def __post_init__(self):
        if self.kind not in VALID_DAYLIGHT_DEVICES:
            raise ValueError(f"kind must be one of {VALID_DAYLIGHT_DEVICES}, got {self.kind!r}")
        if self.spacing_ft is not None and self.spacing_ft <= 0:
            raise ValueError(f"Devices need a spacing; got spacing_ft={self.spacing_ft}.")

    @property
    def resolved_spacing_ft(self) -> float:
        return (DAYLIGHT_DEVICE_SPACING_FT[self.kind] if self.spacing_ft is None
                else self.spacing_ft)

    def describe(self) -> str:
        return f"ProtectDaylightZone({self.target.leg}, {self.target.side}): "

    def apply_to(self, state: "DesignState", model=None) -> str:
        # The one check that is about the LAW rather than about the street, and it depends on
        # another treatment having been applied - so it belongs here, where the design is
        # visible, not in the constructor.
        if (self.kind in CURB_EXTENSION_DEVICES
                and state.treatment_for(AddCurbExtension, self.target) is None):
            raise ValueError(
                f"{self.target} is declared as a {self.kind!r} daylight device, which cuts the "
                f"R.S. 39:4-138(e) setback from 25 ft to 10 ft - but no curb extension has been "
                f"built there. Apply AddCurbExtension first; the statute's reduction is for an "
                f"extension that EXISTS, and claiming it without one would let a proposal mark "
                f"parking 15 ft closer to a crossing than the law allows.")
        spacing_ft = self.resolved_spacing_ft
        return (f"{self.kind} at {spacing_ft:.0f} ft spacing"
                + (" - counts as a curb extension, so R.S. 39:4-138(e) allows parking from 10 ft "
                   "rather than 25 ft" if self.kind in CURB_EXTENSION_DEVICES else ""))
